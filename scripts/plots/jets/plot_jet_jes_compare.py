#!/usr/bin/env python3
"""
Compares the jet energy scale (JES) and resolution (JER) of three samples, one set of plots per jet radius:
  overlay: data + PYTHIA8 overlay made with --do-flow 0 (run 72955, Jet10 / Jet20, 0-60%)
  hijing:  PYTHIA8 embedded in HIJING (run 31, 0-60%)
  pp:      pure PYTHIA8 p+p MC (run 28)
All with the iterative no-flow subtraction (iternf; --ue).

It reads the jes_table.csv written by plot_jet_jes_overlay.py for each sample (--mode overlay / hijing / pp), so it
uses the same fits: JES = Gaussian mean of pT_reco / pT_truth, JER = sigma / mean, in truth-pT bins, raw pT and the stored
pt_calib (eta-binned legacy JES), mid-rapidity, forward (legacy JES eta bins in truth eta) and eta inclusive (incl: mid + forward,
the most statistics). The ratio panels are each sample over the reference sample (--ref, default the p+p MC).

Per R (in <output>/r02 and <output>/r03):
  jes_<calib>_<region>.png, jer_<calib>_<region>.png: the three samples, with the ratio to the reference (region: mid, fwd, incl)
  jes_<calib>_fwd_over_mid.png: forward / mid JES of each sample, with the ratio to the reference
  jes_<calib>_<region>_cent.png: overlay vs HIJING in each centrality class, with p+p (all events) in every panel
and a summary of the mean JES over --avg-range (inverse-variance weighted), printed and saved to jes_compare_summary.csv.
The ratios to the reference in the summary (mid_over_ref, fwd_over_ref) and fwd/mid are means of the per-bin ratios: comparing
two means instead is biased when the samples' errors spread differently over pT (the response rises with pT).
"""

import argparse
import csv
from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mplhep as hep

from plot_jet_bkgsub_compare import save_fig, setup_ratio_panel
from plot_jet_noflow_compare import STYLES_WITH_UNSUB as METHODS
from plot_jet_jes_z0_compare import eta_regions
from plot_jet_jes_overlay import (MODES, RADII, CALIBS, REGIONS, INCL_REGION, PT_BINS, JES_YLABEL, JER_YLABEL, TRUTH_PT_LABEL, RECO_JET_TEXT,
                                  jer, field, draw_points, draw_ratio, set_jes_ylim, shade_low_pt, legend_after_band)

# Samples: the plot_jet_jes_overlay.py mode whose output they are, and their style
SAMPLES = {
    "overlay": {"mode": "overlay", "label": "Data + PYTHIA8 overlay", "color": "black", "marker": "o"},
    "hijing": {"mode": "hijing", "label": "PYTHIA8 + HIJING", "color": "#d95f02", "marker": "^"},
    "pp": {"mode": "pp", "label": "PYTHIA8 p+p", "color": "#1f77b4", "marker": "s"},
}
OFFSETS = {"overlay": 0.0, "hijing": -0.35, "pp": 0.35}  # x offsets [GeV] so the error bars don't overlap
FLOAT_COLUMNS = ["pt_lo", "pt_hi", "mu", "mu_err", "sigma", "sigma_err", "mean", "mean_err", "n_eff"]
KINDS = {"jes": JES_YLABEL, "jer": JER_YLABEL}
# mid, fwd and the eta-inclusive incl (mid + fwd) rows of jes_table.csv
ALL_REGIONS = {**REGIONS, **INCL_REGION}


def sample_label(sample, with_run=True):
    meta, mode = SAMPLES[sample], MODES[SAMPLES[sample]["mode"]]
    extra = [f"run {mode['run']}"] if with_run else []
    if mode["cent_label"]:
        extra.append(mode["cent_label"])
    return f"{meta['label']} ({', '.join(extra)})" if extra else meta["label"]


def load_table(path):
    """jes_table.csv rows grouped by (R, ue, calib, region, cent), sorted in pT, with the numbers as floats."""
    table = {}
    with path.open() as f:
        for row in csv.DictReader(f):
            for key in FLOAT_COLUMNS:
                row[key] = float(row[key])
            table.setdefault((float(row["R"]), row["ue"], row["calib"], row["region"], row["cent"]), []).append(row)
    for key, rows in table.items():
        rows.sort(key=lambda r: r["pt_lo"])
        edges = [r["pt_lo"] for r in rows] + [rows[-1]["pt_hi"]]
        if not np.array_equal(edges, PT_BINS):
            print(f"Error: {path} {key} has truth-pT bins {edges}, expected {list(PT_BINS)}")
            sys.exit(1)
    return table


def points(rows, kind):
    """Per pT bin {"y", "yerr"}: JES (Gaussian mu) or JER (sigma / mu)."""
    if kind == "jes":
        return [{"y": r["mu"], "yerr": r["mu_err"]} for r in rows]
    return [dict(zip(("y", "yerr"), jer(r))) for r in rows]


def over(num, den):
    """num / den per pT bin, with independent errors."""
    out = []
    for n, d in zip(num, den):
        y = n["y"] / d["y"]
        out.append({"y": y, "yerr": abs(y) * np.hypot(n["yerr"] / n["y"], d["yerr"] / d["y"])})
    return out


def weighted_mean(pts, lo, hi):
    """Inverse-variance weighted mean of the finite points with lo <= pT bin < hi."""
    sel = (PT_BINS[:-1] >= lo) & (PT_BINS[1:] <= hi)
    y, e = field(pts, "y")[sel], field(pts, "yerr")[sel]
    ok = np.isfinite(y) & np.isfinite(e) & (e > 0)
    if not ok.any():
        return np.nan, np.nan
    w = 1.0 / e[ok] ** 2
    return float(np.sum(w * y[ok]) / np.sum(w)), float(1.0 / np.sqrt(np.sum(w)))


def header(ax, r_jet, ue):
    ax.text(1.0, 1.01, rf"$R = {r_jet:g}$, {METHODS[ue]['label']}", transform=ax.transAxes, ha="right", va="bottom", fontsize=14)


def plot_samples(series, ref, r_jet, ue, info, ylabel, ratio_label, with_one, pt_trust_min, out_path, save_pdf):
    """One quantity vs truth pT for each sample (series: {sample: points}), with each sample / ref."""
    hep.style.use("ATLAS")
    fig, (ax, ax_ratio) = plt.subplots(2, 1, figsize=(8, 8), gridspec_kw={"height_ratios": [3, 1], "hspace": 0.08}, sharex=True)
    trusted = PT_BINS[:-1] >= pt_trust_min
    ratios = []
    for sample, pts in series.items():
        meta = SAMPLES[sample]
        draw_points(ax, pts, "y", "yerr", meta["color"], meta["marker"], sample_label(sample), offset=OFFSETS[sample])
        if sample != ref:
            draw_ratio(ax_ratio, pts, series[ref], "y", "yerr", meta["color"], meta["marker"])
            # ratio panel limits from the well-measured ratios above the truncated low pT region
            rat = over(pts, series[ref])
            y, e = field(rat, "y"), field(rat, "yerr")
            ratios += list(y[trusted & np.isfinite(y) & (e < 0.05)])
    shade_low_pt(ax, pt_trust_min)
    shade_low_pt(ax_ratio, pt_trust_min, label=False)
    if with_one is not None:
        ax.axhline(1.0, color="gray", linestyle="--", linewidth=1)
    set_jes_ylim(ax, [field(pts, "y")[trusted] for pts in series.values()], with_one=bool(with_one), below=0.3, above=1.4)
    ax.set_xlim(PT_BINS[0], PT_BINS[-1])
    ax.set_ylabel(ylabel)
    header(ax, r_jet, ue)
    # the selection as the legend title, so the long sample labels get the full width
    legend_after_band(ax, pt_trust_min, fontsize=11, title="\n".join(info), title_fontsize=12, alignment="left")
    setup_ratio_panel(ax_ratio, xlabel=TRUTH_PT_LABEL, ylabel=ratio_label, all_ratios=ratios, min_margin=0.03)
    plt.subplots_adjust(left=0.15, right=0.96, bottom=0.10, top=0.94)
    save_fig(fig, out_path, save_pdf)
    plt.close(fig)


def plot_cent_grid(tables, r_jet, ue, calib, region, pt_trust_min, out_path, save_pdf):
    """JES vs truth pT of the overlay and HIJING in each centrality class, with p+p (all events) as the reference in every panel."""
    hep.style.use("ATLAS")
    cent_classes = MODES[SAMPLES["overlay"]["mode"]]["cent_classes"]
    if cent_classes != MODES[SAMPLES["hijing"]["mode"]]["cent_classes"]:
        print("Warning: overlay and HIJING centrality classes differ, skipping the centrality grid")
        return
    pp = points(tables["pp"][(r_jet, ue, calib, region, MODES[SAMPLES["pp"]["mode"]]["cent_all"])], "jes")
    fig, axes = plt.subplots(2, 3, figsize=(18, 11), sharex=True, sharey=True)
    trusted = PT_BINS[:-1] >= pt_trust_min
    y_all = [field(pp, "y")[trusted]]
    for ax, (lo, hi) in zip(axes.flat, cent_classes):
        for sample in ("overlay", "hijing"):
            meta = SAMPLES[sample]
            pts = points(tables[sample][(r_jet, ue, calib, region, f"{lo}_{hi}")], "jes")
            draw_points(ax, pts, "y", "yerr", meta["color"], meta["marker"], f"{meta['label']} (run {MODES[meta['mode']]['run']})",
                        offset=OFFSETS[sample])
            y_all.append(field(pts, "y")[trusted])
        draw_points(ax, pp, "y", "yerr", SAMPLES["pp"]["color"], SAMPLES["pp"]["marker"], sample_label("pp"), offset=OFFSETS["pp"])
        shade_low_pt(ax, pt_trust_min, label=False)
        ax.axhline(1.0, color="gray", linestyle="--", linewidth=1)
        ax.text(0.95, 0.95, f"{lo}–{hi}%", transform=ax.transAxes, ha="right", va="top", fontsize=16)
    set_jes_ylim(axes[0, 0], y_all, with_one=(calib == "calib"), below=0.6, above=0.8)
    axes[0, 0].set_xlim(PT_BINS[0], PT_BINS[-1])
    axes[0, 0].set_xticks(np.arange(20, PT_BINS[-1], 10))  # inner ticks only, so the labels of neighbouring panels don't collide
    for ax in axes[1]:
        ax.set_xlabel(TRUTH_PT_LABEL)
    for ax in axes[:, 0]:
        ax.set_ylabel(JES_YLABEL, fontsize=14)
    axes[0, 0].legend(loc="lower right", frameon=False, fontsize=11)
    labels, _ = eta_regions(r_jet)
    axes[0, 0].text(0.0, 1.01, f"{CALIBS[calib]['label']}, {labels[ALL_REGIONS[region]['key']]}", transform=axes[0, 0].transAxes,
                    ha="left", va="bottom", fontsize=14)
    header(axes[0, 2], r_jet, ue)
    plt.subplots_adjust(left=0.07, right=0.98, bottom=0.08, top=0.95, wspace=0.05, hspace=0.05)
    save_fig(fig, out_path, save_pdf)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="JES / JER of the data+MC overlay, PYTHIA8+HIJING and p+p MC compared, one set of plots per R.")
    for sample, meta in SAMPLES.items():
        default = Path(MODES[meta["mode"]]["output_dir"]) / "jes_table.csv"
        parser.add_argument(f"--{sample}", type=Path, default=default,
                            help=f"jes_table.csv of plot_jet_jes_overlay.py --mode {meta['mode']} (default {default})")
    parser.add_argument("--ref", choices=list(SAMPLES), default="pp", help="Denominator of the ratio panels (default: pp)")
    parser.add_argument("--ue", default="iternf", choices=["unsub", "iter", "iternf"], help="Subtraction method, in all three tables (default: iternf)")
    parser.add_argument("--pt-trust-min", type=float, default=20.0, help="Truth pT below which the response is truncated by the reco tree cut, shaded (default: 20 GeV)")
    parser.add_argument("--avg-range", type=float, nargs=2, default=[24, 52], metavar=("LO", "HI"),
                        help="Truth-pT range of the summary means, on the bin edges (default: 24 52 GeV)")
    parser.add_argument("-o", "--output-dir", type=Path, default=Path("/sphenix/user/anarde/sEPD-Study/plots/Jet-Ana/jes_compare"),
                        help="Directory to save the plots (default: /sphenix/user/anarde/sEPD-Study/plots/Jet-Ana/jes_compare)")
    parser.add_argument("--save-pdf", action="store_true", help="Save PDF version of plots in addition to PNG")
    args = parser.parse_args()

    tables = {}
    for sample in SAMPLES:
        path = getattr(args, sample)
        if not path.is_file():
            print(f"Error: {path} not found (run plot_jet_jes_overlay.py --mode {SAMPLES[sample]['mode']} first)")
            sys.exit(1)
        tables[sample] = load_table(path)
        print(f"{sample}: {path}")
    cents = {sample: MODES[meta["mode"]]["cent_all"] for sample, meta in SAMPLES.items()}
    for sample in SAMPLES:
        if not any(key[1] == args.ue for key in tables[sample]):
            print(f"Error: no {args.ue} rows in {getattr(args, sample)}")
            sys.exit(1)

    # ref first, so it is drawn below the others and listed first
    order = [args.ref] + [s for s in SAMPLES if s != args.ref]
    ref_label = SAMPLES[args.ref]["label"].split(" (")[0]
    short = {"overlay": "Overlay", "hijing": "HIJING", "pp": "p+p"}
    lo, hi = args.avg_range
    summary = []
    n_plots = 0
    for r_tag, r_jet in RADII.items():
        out_dir = args.output_dir / r_tag
        labels, _ = eta_regions(r_jet)
        for calib in CALIBS:
            region_pts = {}
            for region, meta in ALL_REGIONS.items():
                rows = {s: tables[s][(r_jet, args.ue, calib, region, cents[s])] for s in order}
                info = [f"{CALIBS[calib]['label']}, {RECO_JET_TEXT}", rf"$\eta^{{\mathrm{{truth}}}}$: {labels[meta['key']]}"]
                for kind, ylabel in KINDS.items():
                    series = {s: points(rows[s], kind) for s in order}
                    with_one = (calib == "calib") if kind == "jes" else None
                    plot_samples(series, args.ref, r_jet, args.ue, info, ylabel, f"Sample / {short[args.ref]}", with_one, args.pt_trust_min,
                                 out_dir / f"{kind}_{calib}_{region}.png", args.save_pdf)
                    n_plots += 1
                    if kind == "jes":
                        region_pts[region] = series
                plot_cent_grid(tables, r_jet, args.ue, calib, region, args.pt_trust_min, out_dir / f"jes_{calib}_{region}_cent.png", args.save_pdf)
                n_plots += 1

            fwd_mid = {s: over(region_pts["fwd"][s], region_pts["mid"][s]) for s in order}
            info = [f"{CALIBS[calib]['label']}, {RECO_JET_TEXT}", f"Fwd: {labels['outside']}, Mid: {labels['inside']}"]
            plot_samples(fwd_mid, args.ref, r_jet, args.ue, info, "JES Fwd / Mid", f"Sample / {short[args.ref]}", True, args.pt_trust_min,
                         out_dir / f"jes_{calib}_fwd_over_mid.png", args.save_pdf)
            n_plots += 1

            for s in order:
                row = {"R": r_jet, "ue": args.ue, "calib": calib, "sample": s, "cent": cents[s]}
                for name, pts in (("incl", region_pts["incl"][s]), ("mid", region_pts["mid"][s]), ("fwd", region_pts["fwd"][s]),
                                  ("fwd_over_mid", fwd_mid[s])):
                    row[name], row[f"{name}_err"] = weighted_mean(pts, lo, hi)
                # ratio to the reference: mean of the per-bin ratios (the ratio panels), not the ratio of the two means, which
                # depends on how each sample's errors spread over pT (the response rises with pT)
                for region in ALL_REGIONS:
                    if s == args.ref:
                        row[f"{region}_over_ref"], row[f"{region}_over_ref_err"] = 1.0, 0.0
                    else:
                        row[f"{region}_over_ref"], row[f"{region}_over_ref_err"] = weighted_mean(
                            over(region_pts[region][s], region_pts[region][args.ref]), lo, hi)
                summary.append(row)

    print(f"\nMean JES ({args.ue}, truth pT {lo:g}-{hi:g} GeV, inverse-variance weighted); ratios to {ref_label} (mean of the per-bin ratios)")
    print(f"{'R':<5}{'calib':<7}{'sample':<9}" + "".join(f"{k:>18}" for k in ("incl", "mid", "fwd", "fwd/mid", "incl/ref", "mid/ref", "fwd/ref")))
    for r in summary:
        cells = [f"{r[k]:.4f} +- {r[f'{k}_err']:.4f}" for k in ("incl", "mid", "fwd", "fwd_over_mid", "incl_over_ref", "mid_over_ref", "fwd_over_ref")]
        print(f"{r['R']:<5g}{r['calib']:<7}{r['sample']:<9}" + "".join(f"{c:>18}" for c in cells))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    table_path = args.output_dir / "jes_compare_summary.csv"
    with table_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    print(f"\nSaved: {table_path}")
    print(f"Generated {n_plots} plots in {args.output_dir}")


if __name__ == "__main__":
    main()
