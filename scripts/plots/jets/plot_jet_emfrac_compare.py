#!/usr/bin/env python3
"""
Compares the legacy (eta-binned TF1) JES with the EMfrac JES on the same jets.

Reads the Fun4All_BkgSub trees produced with --do-emfrac-calib 1, which store for every jet
pt_calib_<tag> (legacy), pt_calib_emfrac_<tag> and the per-layer calo fractions. A jet is stored
if either calibration passes the 10 GeV tree cut, so each calibration's sample is complete for
--pt-min >= 10.

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

CALIBS = {
    "legacy": {"label": r"Legacy JES ($\eta$-binned)", "color": "#984ea3", "branch": "pt_calib"},
    "emfrac": {"label": "EMfrac JES", "color": "#1b9e77", "branch": "pt_calib_emfrac"},
}

CENT_CLASSES = [(0, 10), (10, 30), (30, 60)]
CENT_COLORS = ["#e41a1c", "#377eb8", "#4daf4a"]

# binning (eta matches h2JetEtav3 in Jet-Anav3: 88 bins over [-1.1, 1.1])
ETA_EDGES = np.linspace(-1.1, 1.1, 89)
DPHI_EDGES = np.linspace(0, np.pi / 2, 13)
CENT_EDGES = np.linspace(0, 60, 13)
EMFRAC_EDGES = np.linspace(-0.5, 1.5, 81)
EMFRAC_PROF_EDGES = np.linspace(-0.5, 1.5, 41)

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


def new_accumulator():
    acc = {"n_events": 0}
    for calib in CALIBS:
        acc[f"eta_{calib}"] = np.zeros(len(ETA_EDGES) - 1)
        acc[f"dphi_{calib}"] = np.zeros(len(DPHI_EDGES) - 1)
        acc[f"cos2_{calib}"] = np.zeros(3)  # sum cos, sum cos^2, n
    for icent in range(len(CENT_CLASSES)):
        acc[f"emfrac_c{icent}"] = np.zeros(len(EMFRAC_EDGES) - 1)
        acc[f"emfrac_range_c{icent}"] = np.zeros(4)  # < 0, > 1, nan, total
    acc["prof_eta"] = new_profile(len(ETA_EDGES) - 1)
    acc["prof_cent"] = new_profile(len(CENT_EDGES) - 1)
    acc["prof_dphi"] = new_profile(len(DPHI_EDGES) - 1)
    acc["prof_emfrac"] = new_profile(len(EMFRAC_PROF_EDGES) - 1)
    return acc


def merge(total, part):
    for key, val in part.items():
        if isinstance(val, dict):
            for sub in val:
                total[key][sub] += val[sub]
        else:
            total[key] += val


def process_file(path, collections, pt_min, pt_max, calo_v2_max):
    """Returns {collection: accumulator} for one tree file, or None if unreadable."""
    event_branches = ["zvtx", "centrality", "calo_v2_iter", "is_flow_failure_iter", "psi2_NS"]
    jet_fields = ["pt_calib", "pt_calib_emfrac", "eta", "phi", "e", "emcal_frac"]
    branches = event_branches + [f"{f}_{c}" for c in collections for f in jet_fields]

    try:
        tree = uproot.open(f"{path}:T")
        missing = [b for b in branches if b not in tree]
        if missing:
            print(f"Skipping {path}: missing {missing[:3]}{'...' if len(missing) > 3 else ''} (not an EMfrac tree?)")
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
        acc = new_accumulator()
        acc["n_events"] = len(arr)
        r_tag = coll[-3:]

        get = lambda field: ak.flatten(arr[f"{field}_{coll}"]).to_numpy()
        eta, phi, energy, emfrac = get("eta"), get("phi"), get("e"), get("emcal_frac")
        pts = {calib: get(meta["branch"]) for calib, meta in CALIBS.items()}
        cent = ak.flatten(ak.broadcast_arrays(arr["centrality"], arr[f"eta_{coll}"])[0]).to_numpy()
        psi2x2 = ak.flatten(ak.broadcast_arrays(arr["psi2_NS"], arr[f"eta_{coll}"])[0]).to_numpy()

        # 2*dphi wrapped to [-pi, pi], folded dphi in [0, pi/2]
        two_dphi = np.angle(np.exp(1j * (2 * phi - psi2x2)))
        dphi = np.abs(two_dphi) / 2
        cos2 = np.cos(two_dphi)

        base = (energy > 0) & (np.abs(eta) < ETA_MAX[r_tag])
        for calib, pt in pts.items():
            sel = base & (pt >= pt_min) & (pt < pt_max)
            acc[f"eta_{calib}"] += np.histogram(eta[sel], ETA_EDGES)[0]
            acc[f"dphi_{calib}"] += np.histogram(dphi[sel], DPHI_EDGES)[0]
            acc[f"cos2_{calib}"] += [cos2[sel].sum(), (cos2[sel] ** 2).sum(), sel.sum()]

        # emfrac and per-jet calibration ratio for the legacy-selected jets
        sel = base & (pts["legacy"] >= pt_min) & (pts["legacy"] < pt_max)
        for icent, (lo, hi) in enumerate(CENT_CLASSES):
            s = sel & (cent >= lo) & (cent < hi)
            ef = emfrac[s]
            acc[f"emfrac_c{icent}"] += np.histogram(ef, EMFRAC_EDGES)[0]
            acc[f"emfrac_range_c{icent}"] += [(ef < 0).sum(), (ef > 1).sum(), np.isnan(ef).sum(), len(ef)]

        ratio = pts["emfrac"][sel] / pts["legacy"][sel]
        fill_profile(acc["prof_eta"], eta[sel], ratio, ETA_EDGES)
        fill_profile(acc["prof_cent"], cent[sel], ratio, CENT_EDGES)
        fill_profile(acc["prof_dphi"], dphi[sel], ratio, DPHI_EDGES)
        fill_profile(acc["prof_emfrac"], emfrac[sel], ratio, EMFRAC_PROF_EDGES)
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


def plot_shape_compare(acc, key, edges, xlabel, ylabel, coll, run, n_files, out_path, save_pdf, vlines=(), extra_label=None):
    """Normalized distribution for both calibrations with an EMfrac / Legacy ratio panel."""
    fig, (ax, ax_r) = plt.subplots(2, 1, figsize=(10, 10), sharex=True, gridspec_kw={"height_ratios": [3, 1], "hspace": 0.05})
    widths = np.diff(edges)
    shapes = {}
    for calib, meta in CALIBS.items():
        counts = acc[f"{key}_{calib}"]
        norm = counts.sum() if counts.sum() > 0 else 1.0
        shapes[calib] = (counts / norm / widths, np.sqrt(counts) / norm / widths)
        label = meta["label"]
        if extra_label:
            label += extra_label(calib)
        ax.stairs(shapes[calib][0], edges, color=meta["color"], linewidth=2, label=label)

    # the two samples share most jets, so independent errors overestimate the ratio uncertainty
    num, num_err = shapes["emfrac"]
    den, den_err = shapes["legacy"]
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = num / den
        ratio_err = calc_ratio_errors(ratio, num_err, den_err, den)
    ax_r.errorbar(centers(edges), ratio, yerr=ratio_err, fmt="o", color=CALIBS["emfrac"]["color"], markersize=4)
    ax_r.axhline(1.0, color="gray", linestyle="--", linewidth=1)

    for x in vlines:
        for a in (ax, ax_r):
            a.axvline(x, color="gray", linestyle=":", linewidth=1)

    finite = ratio[np.isfinite(ratio)]
    if finite.size:
        spread = max(np.max(np.abs(finite - 1)) * 1.3, 0.02)
        ax_r.set_ylim(1 - spread, 1 + spread)
    ax.set_ylim(0, np.nanmax([s[0].max() for s in shapes.values()]) * 1.6)
    ax.set_ylabel(ylabel)
    ax.legend(loc="upper left", fontsize=13, frameon=False)
    ax_r.set_xlabel(xlabel)
    ax_r.set_ylabel("EMfrac / Legacy", fontsize=13)
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
        ("prof_eta", ETA_EDGES, r"$\eta$", legacy_eta_edges(r_tag), (-ETA_MAX[r_tag], ETA_MAX[r_tag])),
        ("prof_cent", CENT_EDGES, "Centrality [%]", (), (CENT_EDGES[0], CENT_EDGES[-1])),
        ("prof_dphi", DPHI_EDGES, r"$|\phi - \Psi_{2}|$", (), (DPHI_EDGES[0], DPHI_EDGES[-1])),
        ("prof_emfrac", EMFRAC_PROF_EDGES, "Jet EMCal energy fraction", (0, 1), (EMFRAC_PROF_EDGES[0], EMFRAC_PROF_EDGES[-1])),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    for ax, (key, edges, xlabel, vlines, xlim) in zip(axes.flat, panels):
        mean, err = profile_mean(acc[key])
        ax.errorbar(centers(edges), mean, yerr=err, fmt="o", color=CALIBS["emfrac"]["color"], markersize=4)
        ax.axhline(1.0, color="gray", linestyle="--", linewidth=1)
        for x in vlines:
            ax.axvline(x, color="gray", linestyle=":", linewidth=1)
        if xlim:
            ax.set_xlim(*xlim)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(r"$\langle p_{T}^{\mathrm{EMfrac}} / p_{T}^{\mathrm{Legacy}} \rangle$")
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
    parser = argparse.ArgumentParser(description="Legacy vs EMfrac JES comparison from Fun4All_BkgSub trees.")
    parser.add_argument("-i", "--input", nargs="+", required=True, help="Tree files, directories (searched for tree-*.root) or .list files.")
    parser.add_argument("-o", "--output", default="plots/Jet-Ana/emfrac_compare", help="Output directory.")
    parser.add_argument("--run", default="68144", help="Run label for the plots.")
    parser.add_argument("--methods", nargs="+", default=["iter", "iternf", "unsub"], choices=list(METHODS))
    parser.add_argument("--radii", nargs="+", default=["r02", "r03"], choices=["r02", "r03"])
    parser.add_argument("--pt-min", type=float, default=10.0, help="Calibrated jet pT min [GeV] (>= 10, the tree cut).")
    parser.add_argument("--pt-max", type=float, default=80.0, help="Calibrated jet pT max [GeV].")
    parser.add_argument("--calo-v2-max", type=float, default=0.48)
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--pdf", action="store_true", help="Also save PDFs.")
    args = parser.parse_args()

    if args.pt_min < 10:
        parser.error("--pt-min below 10 GeV is incomplete: the tree only stores jets with either calibrated pT >= 10 GeV")

    files = find_trees(args.input)[: args.max_files]
    if not files:
        parser.error("no tree files found")
    collections = [f"{m}_{r}" for m in args.methods for r in args.radii]
    print(f"Processing {len(files)} files for {', '.join(collections)}")

    totals = {c: new_accumulator() for c in collections}
    n_good = 0
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(process_file, f, collections, args.pt_min, args.pt_max, args.calo_v2_max) for f in files]
        for fut in concurrent.futures.as_completed(futures):
            result = fut.result()
            if result is None:
                continue
            n_good += 1
            for coll, acc in result.items():
                merge(totals[coll], acc)
    if n_good == 0:
        print("No readable EMfrac trees.")
        return
    print(f"Read {n_good} / {len(files)} files")

    hep.style.use("ATLAS")
    out_dir = Path(args.output)
    for coll in collections:
        acc = totals[coll]
        r_tag = coll[-3:]
        edges = legacy_eta_edges(r_tag)
        edge = abs(edges[0])

        v2 = {calib: v2_obs(acc[f"cos2_{calib}"]) for calib in CALIBS}
        plot_shape_compare(acc, "eta", ETA_EDGES, r"$\eta$", r"$\frac{1}{N_{\mathrm{jets}}}\frac{dN}{d\eta}$",
                           coll, args.run, n_good, out_dir / "eta" / f"run_{args.run}_eta_{coll}.png", args.pdf,
                           vlines=edges)
        plot_shape_compare(acc, "dphi", DPHI_EDGES, r"$|\phi - \Psi_{2}|$", r"$\frac{1}{N_{\mathrm{jets}}}\frac{dN}{d\Delta\phi}$",
                           coll, args.run, n_good, out_dir / "dphi" / f"run_{args.run}_dphi_{coll}.png", args.pdf,
                           extra_label=lambda c: rf", $\langle\cos 2\Delta\phi\rangle = {v2[c][0]:.4f} \pm {v2[c][1]:.4f}$")
        plot_emfrac(acc, coll, args.run, n_good, out_dir / "emfrac" / f"run_{args.run}_emfrac_{coll}.png", args.pdf)
        plot_ratio_profiles(acc, coll, args.run, n_good, out_dir / "ratio" / f"run_{args.run}_ratio_{coll}.png", args.pdf)

        print(f"\n=== {coll}: {acc['n_events']:.0f} events ===")
        for calib in CALIBS:
            n_jets = acc[f"eta_{calib}"].sum()
            print(f"  {calib:7s} N jets = {n_jets:10.0f} | step at |eta|={edge:.2f}: {step_ratio(acc[f'eta_{calib}'], edge):.3f}"
                  f" (control |eta|=0.30: {step_ratio(acc[f'eta_{calib}'], 0.30):.3f})"
                  f" | <cos 2dphi> = {v2[calib][0]:.4f} +- {v2[calib][1]:.4f}")
        for icent, (lo, hi) in enumerate(CENT_CLASSES):
            below, above, nan, total = acc[f"emfrac_range_c{icent}"]
            if total:
                print(f"  emfrac {lo:2d}-{hi:2d}%: N = {total:9.0f}  <0: {below / total:6.2%}  >1: {above / total:6.2%}  NaN: {nan / total:6.2%}")


if __name__ == "__main__":
    main()
