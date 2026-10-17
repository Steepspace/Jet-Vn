#!/usr/bin/env python3
"""
Compares jet energy scale (JES) calibrations on the same jets, against the legacy (eta-binned
TF1) JES that the trees were produced with:
  legacy -- pt_calib_<tag>, eta-binned JES_Calib_Default (reference)
  emfrac -- pt_calib_emfrac_<tag>, EMfrac JES (trees produced with --do-emfrac-calib 1)
  z0     -- legacy JES without the eta bins: the eta-averaged JES_Calib_Func_R0X_Z0 applied
            offline to the raw pT (what set_ApplyEtaDependentCalib(false) uses for |z| < 30 cm)

The trees keep a jet if its legacy (or EMfrac) calibrated pT >= 10 GeV, so a z0 sample is only
complete above the z0 pT of the highest raw-pT threshold among the legacy eta bins (about 10.2 GeV
for R = 0.2). The z0 cut is raised to that value automatically.

Event selection follows Jet-Anav3 (|z| < 10 cm, centrality <= 59.5%, no iter flow failure,
|calo v2| < 0.48, jet E > 0) except for the Calo-MBD consistency cut, which needs an external
profile and does not matter for a calibration comparison.

Note: the tree branch psi2_NS holds 2*Psi2 (see GlobalQA.cc), so 2*dphi = 2*phi - psi2_NS.
"""

import argparse
import concurrent.futures
from pathlib import Path
import numpy as np
import uproot
import awkward as ak
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mplhep as hep

from plot_jet_bkgsub_compare import save_fig, calc_ratio_errors

METHODS = {
    "iter": "Iterative",
    "iternf": "Iterative No Flow",
    "mult": "Multiplicity",
    "multnf": "Multiplicity No Flow",
    "unsub": "Unsubtracted",
}
ETA_MAX = {"r02": 0.9, "r03": 0.8}

# branch None: computed offline from the raw pT
CALIBS = {
    "legacy": {"label": r"Legacy JES ($\eta$-binned)", "short": "Legacy", "color": "#984ea3", "branch": "pt_calib"},
    "emfrac": {"label": "EMfrac JES", "short": "EMfrac", "color": "#1b9e77", "branch": "pt_calib_emfrac"},
    "z0": {"label": r"Legacy JES $z$-only (Z0)", "short": "Z0", "color": "#ff7f00", "branch": None},
}
REFERENCE = "legacy"

# CDB payload the trees were calibrated with (newcdbtag, run 68144), see the job logs
LEGACY_JES_FILE = "/cvmfs/sphenix.sdcc.bnl.gov/calibrations/sphnxpro/cdb/JES_Calib_Default/0b/08/0b0822d756de4bfc5c3793e6db0716c4_JES_Calib_Default_47289-53864_v6.root"
TREE_PT_MIN = 10.0  # JetValidationv3 m_jet_pt_min_cut

CENT_CLASSES = [(0, 10), (10, 30), (30, 60)]
CENT_COLORS = ["#e41a1c", "#377eb8", "#4daf4a"]

# binning (eta matches h2JetEtav3 in Jet-Anav3: 88 bins over [-1.1, 1.1])
ETA_EDGES = np.linspace(-1.1, 1.1, 89)
DPHI_EDGES = np.linspace(0, np.pi / 2, 13)
CENT_EDGES = np.linspace(0, 60, 13)
EMFRAC_EDGES = np.linspace(-0.5, 1.5, 81)
EMFRAC_PROF_EDGES = np.linspace(-0.5, 1.5, 41)
PROFILE_VARS = {"eta": ETA_EDGES, "cent": CENT_EDGES, "dphi": DPHI_EDGES, "emfrac": EMFRAC_PROF_EDGES}

TEXT_INFO = (
    r"$|z| < 10$ cm, Centrality: 0–60%" "\n"
    r"No Flow Failure, $|\mathrm{Calo}\ v_{2}| < 0.48$" "\n"
    "Energy > 0"
)


def legacy_eta_edges(r_tag):
    """Edges of JetCalib's legacy eta bins for |z| < 30 cm: 4 equal bins on [-1.2+R, 1.2-R]."""
    radius = int(r_tag[1:]) / 10
    lo, hi = -1.2 + radius, 1.2 - radius
    return [lo + (hi - lo) * k / 4 for k in (1, 2, 3)]


def load_z0(radii, jes_file):
    """Tabulates JES_Calib_Func_R0X_Z0 on a raw-pT grid and finds its minimum complete pT cut.

    Returns (grid, {r_tag: table}, {r_tag: safe cut}). The safe cut is the z0 pT of the highest raw pT
    the trees keep (legacy pT = TREE_PT_MIN) among the legacy eta bins.
    """
    import ROOT
    fin = ROOT.TFile.Open(jes_file)
    if not fin or fin.IsZombie():
        raise RuntimeError(f"cannot open {jes_file}")
    grid = np.arange(3.0, 150.0, 0.001)
    tables, safe = {}, {}
    for r_tag in radii:
        name = f"JES_Calib_Func_R{r_tag[1:]}_Z0"
        z0 = fin.Get(name)
        tables[r_tag] = np.array([z0.Eval(x) for x in grid])
        raw_at_cut = [fin.Get(f"{name}_Eta{k}").GetX(TREE_PT_MIN, 1, 50) for k in range(4)]
        safe[r_tag] = max(z0.Eval(x) for x in raw_at_cut)
    fin.Close()
    return grid, tables, safe


def find_trees(inputs):
    """Accepts tree files, directories (searched for tree-*.root) or .list files."""
    files = []
    for item in inputs:
        path = Path(item)
        if path.is_dir():
            files += sorted(path.rglob("tree-*.root"))
        elif path.suffix == ".list":
            files += [Path(l.strip()) for l in path.read_text().splitlines() if l.strip() and not l.startswith("#")]
        else:
            files.append(path)
    return files


def new_profile(n):
    return {"sum": np.zeros(n), "sum2": np.zeros(n), "n": np.zeros(n)}


def fill_profile(prof, x, y, edges):
    idx = np.digitize(x, edges) - 1
    ok = (idx >= 0) & (idx < len(edges) - 1) & np.isfinite(y)
    prof["sum"] += np.bincount(idx[ok], weights=y[ok], minlength=len(edges) - 1)
    prof["sum2"] += np.bincount(idx[ok], weights=y[ok] ** 2, minlength=len(edges) - 1)
    prof["n"] += np.bincount(idx[ok], minlength=len(edges) - 1)


def new_accumulator(calibs, with_emfrac):
    acc = {"n_events": 0, "prof": {}}
    for calib in calibs:
        acc[f"eta_{calib}"] = np.zeros(len(ETA_EDGES) - 1)
        acc[f"dphi_{calib}"] = np.zeros(len(DPHI_EDGES) - 1)
        acc[f"cos2_{calib}"] = np.zeros(3)  # sum cos, sum cos^2, n
        if calib != REFERENCE:
            acc["prof"][calib] = {var: new_profile(len(edges) - 1) for var, edges in PROFILE_VARS.items() if with_emfrac or var != "emfrac"}
    if with_emfrac:
        for icent in range(len(CENT_CLASSES)):
            acc[f"emfrac_c{icent}"] = np.zeros(len(EMFRAC_EDGES) - 1)
            acc[f"emfrac_range_c{icent}"] = np.zeros(4)  # < 0, > 1, nan, total
    return acc


def merge(total, part):
    for key, val in part.items():
        if isinstance(val, dict):
            merge(total[key], val)
        else:
            total[key] += val


def process_file(path, collections, calibs, pt_cuts, pt_max, calo_v2_max, z0):
    """Returns {collection: accumulator} for one tree file, or None if unreadable."""
    with_emfrac = "emfrac" in calibs
    event_branches = ["zvtx", "centrality", "calo_v2_iter", "is_flow_failure_iter", "psi2_NS"]
    jet_fields = ["pt", "pt_calib", "eta", "phi", "e"] + (["pt_calib_emfrac", "emcal_frac"] if with_emfrac else [])
    branches = event_branches + [f"{f}_{c}" for c in collections for f in jet_fields]

    try:
        tree = uproot.open(f"{path}:T")
        missing = [b for b in branches if b not in tree]
        if missing:
            hint = " (not an EMfrac tree? drop emfrac from --calibs)" if with_emfrac else ""
            print(f"Skipping {path}: missing {missing[:3]}{'...' if len(missing) > 3 else ''}{hint}")
            return None
        arr = tree.arrays(branches)
    except Exception as exc:  # partially written files from running jobs
        print(f"Skipping {path}: {exc}")
        return None

    evt = ((np.abs(arr["zvtx"]) < 10) & (arr["centrality"] <= 59.5)
           & ~arr["is_flow_failure_iter"] & (np.abs(arr["calo_v2_iter"]) < calo_v2_max))
    arr = arr[evt]

    out = {}
    for coll in collections:
        acc = new_accumulator(calibs, with_emfrac)
        acc["n_events"] = len(arr)
        r_tag = coll[-3:]

        get = lambda field: ak.flatten(arr[f"{field}_{coll}"]).to_numpy()
        pt_raw, eta, phi, energy = get("pt"), get("eta"), get("phi"), get("e")
        pts = {}
        for calib in calibs:
            branch = CALIBS[calib]["branch"]
            pts[calib] = get(branch) if branch else np.interp(pt_raw, z0[0], z0[1][r_tag])
        cent = ak.flatten(ak.broadcast_arrays(arr["centrality"], arr[f"eta_{coll}"])[0]).to_numpy()
        psi2x2 = ak.flatten(ak.broadcast_arrays(arr["psi2_NS"], arr[f"eta_{coll}"])[0]).to_numpy()

        # 2*dphi wrapped to [-pi, pi], folded dphi in [0, pi/2]
        two_dphi = np.angle(np.exp(1j * (2 * phi - psi2x2)))
        dphi = np.abs(two_dphi) / 2
        cos2 = np.cos(two_dphi)

        base = (energy > 0) & (np.abs(eta) < ETA_MAX[r_tag])
        for calib, pt in pts.items():
            sel = base & (pt >= pt_cuts[calib][r_tag]) & (pt < pt_max)
            acc[f"eta_{calib}"] += np.histogram(eta[sel], ETA_EDGES)[0]
            acc[f"dphi_{calib}"] += np.histogram(dphi[sel], DPHI_EDGES)[0]
            acc[f"cos2_{calib}"] += [cos2[sel].sum(), (cos2[sel] ** 2).sum(), sel.sum()]

        # emfrac and per-jet calibration ratios for the reference-selected jets
        sel = base & (pts[REFERENCE] >= pt_cuts[REFERENCE][r_tag]) & (pts[REFERENCE] < pt_max)
        xs = {"eta": eta[sel], "cent": cent[sel], "dphi": dphi[sel]}
        if with_emfrac:
            emfrac = get("emcal_frac")
            xs["emfrac"] = emfrac[sel]
            for icent, (lo, hi) in enumerate(CENT_CLASSES):
                s = sel & (cent >= lo) & (cent < hi)
                ef = emfrac[s]
                acc[f"emfrac_c{icent}"] += np.histogram(ef, EMFRAC_EDGES)[0]
                acc[f"emfrac_range_c{icent}"] += [(ef < 0).sum(), (ef > 1).sum(), np.isnan(ef).sum(), len(ef)]

        for calib, profs in acc["prof"].items():
            ratio = pts[calib][sel] / pts[REFERENCE][sel]
            for var, prof in profs.items():
                fill_profile(prof, xs[var], ratio, PROFILE_VARS[var])
        out[coll] = acc
    return out


def profile_mean(prof):
    n = np.where(prof["n"] > 0, prof["n"], np.nan)
    mean = prof["sum"] / n
    var = prof["sum2"] / n - mean ** 2
    return mean, np.sqrt(np.clip(var, 0, None) / n)


def centers(edges):
    return 0.5 * (edges[1:] + edges[:-1])


def add_text(ax, run, coll, n_files, extra=None):
    method, r_tag = coll[:-4], coll[-3:]
    head = rf"Run {run}, $R = {int(r_tag[1:]) / 10:g}$, {n_files} segments"
    ax.text(1.0, 1.01, head, transform=ax.transAxes, ha="right", va="bottom", fontsize=13)
    text = f"{METHODS[method]}\n{TEXT_INFO}" + (f"\n{extra}" if extra else "")
    ax.text(0.97, 0.96, text, transform=ax.transAxes, ha="right", va="top", fontsize=12)


def plot_shape_compare(acc, key, edges, calibs, labels, xlabel, ylabel, coll, run, n_files, out_path, save_pdf, vlines=()):
    """Normalized distribution for each calibration with a ratio-to-legacy panel."""
    fig, (ax, ax_r) = plt.subplots(2, 1, figsize=(10, 10), sharex=True, gridspec_kw={"height_ratios": [3, 1], "hspace": 0.05})
    widths = np.diff(edges)
    shapes = {}
    for calib in calibs:
        counts = acc[f"{key}_{calib}"]
        norm = counts.sum() if counts.sum() > 0 else 1.0
        shapes[calib] = (counts / norm / widths, np.sqrt(counts) / norm / widths)
        ax.stairs(shapes[calib][0], edges, color=CALIBS[calib]["color"], linewidth=2, label=labels[calib])

    # the samples share most jets, so independent errors overestimate the ratio uncertainty
    den, den_err = shapes[REFERENCE]
    finite = []
    for calib in calibs:
        if calib == REFERENCE:
            continue
        num, num_err = shapes[calib]
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = num / den
            ratio_err = calc_ratio_errors(ratio, num_err, den_err, den)
        ax_r.errorbar(centers(edges), ratio, yerr=ratio_err, fmt="o", color=CALIBS[calib]["color"], markersize=4)
        finite.append(ratio[np.isfinite(ratio)])
    ax_r.axhline(1.0, color="gray", linestyle="--", linewidth=1)

    for x in vlines:
        for a in (ax, ax_r):
            a.axvline(x, color="gray", linestyle=":", linewidth=1)

    finite = np.concatenate(finite) if finite else np.array([])
    if finite.size:
        spread = max(np.max(np.abs(finite - 1)) * 1.3, 0.02)
        ax_r.set_ylim(1 - spread, 1 + spread)
    ax.set_ylim(0, np.nanmax([s[0].max() for s in shapes.values()]) * 1.6)
    ax.set_ylabel(ylabel)
    ax.legend(loc="upper left", fontsize=12, frameon=False)
    ax_r.set_xlabel(xlabel)
    ax_r.set_ylabel(f"X / {CALIBS[REFERENCE]['short']}", fontsize=13)
    add_text(ax, run, coll, n_files)
    save_fig(fig, out_path, save_pdf)
    plt.close(fig)


def plot_emfrac(acc, coll, run, n_files, out_path, save_pdf):
    fig, ax = plt.subplots(figsize=(10, 8))
    widths = np.diff(EMFRAC_EDGES)
    for icent, (lo, hi) in enumerate(CENT_CLASSES):
        counts = acc[f"emfrac_c{icent}"]
        below, above, nan, total = acc[f"emfrac_range_c{icent}"]
        if total == 0:
            continue
        label = f"{lo}–{hi}%: <0 {below / total:.1%}, >1 {above / total:.1%}"
        if nan:
            label += f", NaN {nan / total:.1%}"
        ax.stairs(counts / total / widths, EMFRAC_EDGES, color=CENT_COLORS[icent], linewidth=2, label=label)
    for x in (0, 1):
        ax.axvline(x, color="gray", linestyle=":", linewidth=1)
    ax.set_yscale("log")
    ax.set_xlabel("Jet EMCal energy fraction")
    ax.set_ylabel(r"$\frac{1}{N_{\mathrm{jets}}}\frac{dN}{d f_{\mathrm{EM}}}$")
    ax.legend(loc="lower center", fontsize=12, frameon=False)
    add_text(ax, run, coll, n_files, extra=r"Legacy $p_{T}$ selection")
    save_fig(fig, out_path, save_pdf)
    plt.close(fig)


def plot_ratio_profiles(acc, coll, run, n_files, out_path, save_pdf):
    r_tag = coll[-3:]
    panels = [
        ("eta", r"$\eta$", legacy_eta_edges(r_tag), (-ETA_MAX[r_tag], ETA_MAX[r_tag])),
        ("cent", "Centrality [%]", (), (CENT_EDGES[0], CENT_EDGES[-1])),
        ("dphi", r"$|\phi - \Psi_{2}|$", (), (DPHI_EDGES[0], DPHI_EDGES[-1])),
        ("emfrac", "Jet EMCal energy fraction", (0, 1), (EMFRAC_PROF_EDGES[0], EMFRAC_PROF_EDGES[-1])),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    for ax, (var, xlabel, vlines, xlim) in zip(axes.flat, panels):
        drawn = False
        for calib, profs in acc["prof"].items():
            if var not in profs:
                continue
            mean, err = profile_mean(profs[var])
            ax.errorbar(centers(PROFILE_VARS[var]), mean, yerr=err, fmt="o", color=CALIBS[calib]["color"], markersize=4,
                        label=f"{CALIBS[calib]['short']} / {CALIBS[REFERENCE]['short']}")
            drawn = True
        if not drawn:
            ax.set_visible(False)
            continue
        ax.axhline(1.0, color="gray", linestyle="--", linewidth=1)
        for x in vlines:
            ax.axvline(x, color="gray", linestyle=":", linewidth=1)
        ax.set_xlim(*xlim)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(r"$\langle p_{T}^{X} / p_{T}^{\mathrm{Legacy}} \rangle$")
    axes[0, 0].legend(loc="lower center", fontsize=12, frameon=False)
    add_text(axes[0, 1], run, coll, n_files, extra=r"Legacy $p_{T}$ selection")
    fig.tight_layout()
    save_fig(fig, out_path, save_pdf)
    plt.close(fig)


def step_ratio(counts, edge, width=0.05):
    """N(edge <= |eta| < edge+width) / N(edge-width <= |eta| < edge), both sides combined."""
    c = centers(ETA_EDGES)
    a = np.abs(c)
    outside = counts[(a >= edge) & (a < edge + width)].sum()
    inside = counts[(a >= edge - width) & (a < edge)].sum()
    return outside / inside if inside else np.nan


def v2_obs(cos2):
    s, s2, n = cos2
    if n == 0:
        return np.nan, np.nan
    mean = s / n
    return mean, np.sqrt(max(s2 / n - mean ** 2, 0) / n)


def main():
    parser = argparse.ArgumentParser(description="JES calibration comparison (legacy vs EMfrac / z-only) from Fun4All_BkgSub trees.")
    parser.add_argument("-i", "--input", nargs="+", required=True, help="Tree files, directories (searched for tree-*.root) or .list files.")
    parser.add_argument("-o", "--output", default="plots/Jet-Ana/emfrac_compare", help="Output directory.")
    parser.add_argument("--run", default="68144", help="Run label for the plots.")
    parser.add_argument("--methods", nargs="+", default=["iter", "iternf", "unsub"], choices=list(METHODS))
    parser.add_argument("--radii", nargs="+", default=["r02", "r03"], choices=["r02", "r03"])
    parser.add_argument("--calibs", nargs="+", default=["emfrac", "z0"], choices=[c for c in CALIBS if c != REFERENCE],
                        help="Calibrations to compare with the legacy one (emfrac needs --do-emfrac-calib trees).")
    parser.add_argument("--jes-file", default=LEGACY_JES_FILE, help="Legacy JES payload for the z0 calibration.")
    parser.add_argument("--pt-min", type=float, default=10.0, help="Calibrated jet pT min [GeV] (>= 10, the tree cut; z0 is raised to its complete cut).")
    parser.add_argument("--pt-max", type=float, default=80.0, help="Calibrated jet pT max [GeV].")
    parser.add_argument("--calo-v2-max", type=float, default=0.48)
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--pdf", action="store_true", help="Also save PDFs.")
    args = parser.parse_args()

    if args.pt_min < TREE_PT_MIN:
        parser.error(f"--pt-min below {TREE_PT_MIN:g} GeV is incomplete: the tree only stores jets with calibrated pT >= {TREE_PT_MIN:g} GeV")

    calibs = [REFERENCE] + args.calibs
    pt_cuts = {calib: {r: args.pt_min for r in args.radii} for calib in calibs}
    z0 = None
    if "z0" in calibs:
        grid, tables, safe = load_z0(args.radii, args.jes_file)
        z0 = (grid, tables)
        pt_cuts["z0"] = {r: max(args.pt_min, np.ceil(safe[r] * 100) / 100) for r in args.radii}
        print("z0 pT cuts (complete given the tree cut): " + ", ".join(f"{r}: {pt_cuts['z0'][r]:.2f} GeV" for r in args.radii))

    files = find_trees(args.input)[: args.max_files]
    if not files:
        parser.error("no tree files found")
    collections = [f"{m}_{r}" for m in args.methods for r in args.radii]
    print(f"Processing {len(files)} files for {', '.join(collections)}; calibrations: {', '.join(calibs)}")

    with_emfrac = "emfrac" in calibs
    totals = {c: new_accumulator(calibs, with_emfrac) for c in collections}
    n_good = 0
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(process_file, f, collections, calibs, pt_cuts, args.pt_max, args.calo_v2_max, z0) for f in files]
        for fut in concurrent.futures.as_completed(futures):
            result = fut.result()
            if result is None:
                continue
            n_good += 1
            for coll, acc in result.items():
                merge(totals[coll], acc)
    if n_good == 0:
        print("No readable trees.")
        return
    print(f"Read {n_good} / {len(files)} files")

    hep.style.use("ATLAS")
    out_dir = Path(args.output)
    for coll in collections:
        acc = totals[coll]
        r_tag = coll[-3:]
        edges = legacy_eta_edges(r_tag)
        edge = abs(edges[0])

        v2 = {calib: v2_obs(acc[f"cos2_{calib}"]) for calib in calibs}
        cut_labels = {c: CALIBS[c]["label"] + (rf" ($p_{{T}} \geq {pt_cuts[c][r_tag]:.2f}$)" if pt_cuts[c][r_tag] != args.pt_min else "") for c in calibs}
        plot_shape_compare(acc, "eta", ETA_EDGES, calibs, cut_labels, r"$\eta$", r"$\frac{1}{N_{\mathrm{jets}}}\frac{dN}{d\eta}$",
                           coll, args.run, n_good, out_dir / "eta" / f"run_{args.run}_eta_{coll}.png", args.pdf, vlines=edges)
        v2_labels = {c: cut_labels[c] + "\n" + rf"$\langle\cos 2\Delta\phi\rangle = {v2[c][0]:.4f} \pm {v2[c][1]:.4f}$" for c in calibs}
        plot_shape_compare(acc, "dphi", DPHI_EDGES, calibs, v2_labels, r"$|\phi - \Psi_{2}|$", r"$\frac{1}{N_{\mathrm{jets}}}\frac{dN}{d\Delta\phi}$",
                           coll, args.run, n_good, out_dir / "dphi" / f"run_{args.run}_dphi_{coll}.png", args.pdf)
        if with_emfrac:
            plot_emfrac(acc, coll, args.run, n_good, out_dir / "emfrac" / f"run_{args.run}_emfrac_{coll}.png", args.pdf)
        plot_ratio_profiles(acc, coll, args.run, n_good, out_dir / "ratio" / f"run_{args.run}_ratio_{coll}.png", args.pdf)

        print(f"\n=== {coll}: {acc['n_events']:.0f} events ===")
        for calib in calibs:
            n_jets = acc[f"eta_{calib}"].sum()
            print(f"  {calib:7s} pT >= {pt_cuts[calib][r_tag]:5.2f} N jets = {n_jets:10.0f}"
                  f" | step at |eta|={edge:.2f}: {step_ratio(acc[f'eta_{calib}'], edge):.3f}"
                  f" (control |eta|=0.30: {step_ratio(acc[f'eta_{calib}'], 0.30):.3f})"
                  f" | <cos 2dphi> = {v2[calib][0]:.4f} +- {v2[calib][1]:.4f}")
        if with_emfrac:
            for icent, (lo, hi) in enumerate(CENT_CLASSES):
                below, above, nan, total = acc[f"emfrac_range_c{icent}"]
                if total:
                    print(f"  emfrac {lo:2d}-{hi:2d}%: N = {total:9.0f}  <0: {below / total:6.2%}  >1: {above / total:6.2%}  NaN: {nan / total:6.2%}")


if __name__ == "__main__":
    main()
