#!/usr/bin/env python3
"""
Reco vs truth check of the run-31 HIJING-embedded PYTHIA8 Jet20 sample (tasks/jet-jes-comparison.md,
"Reco vs truth in the bad HIJING Jet20 events").

About 25% of the HIJING Jet20 (and Jet30) events have a leading truth R=0.4 jet below the 20 GeV generator filter,
against ~6% in run-28 p+p Jet20. This tells apart the two explanations:
  - the events are really soft: reco and truth agree, as in the control;
  - the truth record is incomplete: the calorimeter still sees the hard jet, so there is a reco jet > 20 GeV and the
    reco / truth response is far above the control at the same truth pT.
Control: HIJING Jet10 at the same leading truth R=0.4 pT. Segment N of each sample holds the same HIJING events, so the
background is identical. p+p Jet20 (HardQCD) and Jet12 (minimum bias) are the references without background, and
also show whether the generator processes change the response.

It reads the TreeFiller trees (Fun4All_HIJING_JES.C / Fun4All_PP_JES.C): reco iter jets (pt_calib >= 10 GeV), truth jets
(>= 5 GeV, |eta| < 1.5), max_pt_truth_r04. Centrality < 60% for HIJING (p+p trees store 9999).

Printed:
  1. per leading truth R=0.4 pT bin: fraction of events with a reco jet > 20 GeV in |eta| < 0.7, fraction with such a jet
     further than dR = 0.4 from every truth jet, and the median response of the leading |eta| < 0.7 truth jet
     (closest reco jet within dR < 0.2);
  2. in-window bias: events with leading truth R=0.4 in [21, 31) (the Jet20 window), response of the leading truth jet in
     bins of its pT. HIJING Jet10 is outside its own window there, used only as the unaffected control.
"""

import argparse
import concurrent.futures
import os
from pathlib import Path
import warnings
import numpy as np
import uproot
import awkward as ak
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mplhep as hep

from plot_jet_bkgsub_compare import save_fig, setup_ratio_panel

ETA_FID = 0.7  # reco and leading truth jet fiducial |eta|
MATCH_DR = 0.2  # leading truth jet to reco jet
UNMATCHED_DR = 0.4  # hard reco jet with no truth jet within this
HARD_PT = 20.0
LEAD_BINS = [(8, 12), (12, 16), (16, 20), (21, 31)]
WINDOW = (21, 31)  # HIJING Jet20 window
TRUTH_PT_BINS = [(16, 20), (20, 24), (24, 28), (28, 32)]
CORE = (0.5, 1.7)  # response range of the core mean

WINDOW_SAMPLES = ["HIJING Jet20", "HIJING Jet10", "p+p Jet20", "p+p Jet12"]  # samples of the reco-vs-truth and in-window figures
FILTER_PAIRS = {20: ("HIJING Jet20", "p+p Jet20"), 30: ("HIJING Jet30", "p+p Jet30")}  # generator filter [GeV]: same-filter samples
STYLE = {
    "HIJING Jet20": {"color": "#d95f02", "marker": "o", "label": "HIJING Jet20 (run 31, 2025-08)"},
    "HIJING Jet10": {"color": "#66a61e", "marker": "s", "label": "HIJING Jet10 (run 31, 2026-03): same background"},
    "HIJING Jet30": {"color": "#7570b3", "marker": "o", "label": "HIJING Jet30 (run 31, 2025-08)"},
    "p+p Jet20": {"color": "black", "marker": "^", "label": "p+p Jet20 (run 28)"},
    "p+p Jet12": {"color": "#1f77b4", "marker": "v", "label": "p+p Jet12 (run 28)"},
    "p+p Jet30": {"color": "black", "marker": "^", "label": "p+p Jet30 (run 28)"},
}
LEAD_EDGES = np.array([8, 10, 12, 14, 16, 18, 20, 21, 24, 27, 31])  # leading truth R=0.4 bins of the reco-vs-truth figure
RESP_EDGES = np.array([16, 18, 20, 22, 24, 26, 28, 32])  # leading truth jet pT bins of the in-window figure

HIJING_LIST ="/sphenix/user/anarde/sEPD-Study/f4a_bkgsub_mc/11-19-26-hijing/tree.list"
PP_LIST = "/sphenix/user/anarde/sEPD-Study/f4a_bkgsub_mc/11-19-26-pythia/tree.list"


def files(lst, sample, n):
    """First n trees of a sample (jetNN directory in the path) in a tree list."""
    return [p for p in Path(lst).read_text().split() if f"/{sample}/" in p][:n]


def dphi(a, b):
    return (a - b + np.pi) % (2 * np.pi) - np.pi


def process(task):
    path, r_tag = task
    branches = ["max_pt_truth_r04", "centrality"] + [f"{v}_truth_{r_tag}" for v in ("pt", "eta", "phi")] + \
               [f"{v}_iter_{r_tag}" for v in ("pt_calib", "eta", "phi")]
    a = uproot.open(path)["T"].arrays(branches, library="ak")
    a = a[(a.centrality < 60) | (a.centrality >= 9999)]  # p+p trees: 9999
    tpt, teta, tphi = a[f"pt_truth_{r_tag}"], a[f"eta_truth_{r_tag}"], a[f"phi_truth_{r_tag}"]
    rpt, reta, rphi = a[f"pt_calib_iter_{r_tag}"], a[f"eta_iter_{r_tag}"], a[f"phi_iter_{r_tag}"]
    rsel = abs(reta) < ETA_FID
    rpt, reta, rphi = rpt[rsel], reta[rsel], rphi[rsel]
    lead_r = ak.fill_none(ak.max(rpt, axis=1), 0.0)  # 0 if no reco jet passes the tree cut
    # leading truth jet in |eta| < ETA_FID
    tf = abs(teta) < ETA_FID
    idx = ak.argmax(ak.where(tf, tpt, -1.0), axis=1, keepdims=True)
    lt_pt = ak.fill_none(ak.firsts(tpt[idx]), 0.0)
    lt_ok = ak.fill_none(ak.firsts(tf[idx]), False)
    lt_eta, lt_phi = ak.firsts(teta[idx]), ak.firsts(tphi[idx])
    # hard reco jets far from every truth jet
    pairs = ak.cartesian([ak.zip({"eta": reta, "phi": rphi}), ak.zip({"eta": teta, "phi": tphi})], nested=True)
    dr = np.sqrt((pairs["0"].eta - pairs["1"].eta) ** 2 + dphi(pairs["0"].phi, pairs["1"].phi) ** 2)
    dr_min = ak.fill_none(ak.min(dr, axis=2), 9.0)
    unmatched = ak.any((rpt > HARD_PT) & (dr_min > UNMATCHED_DR), axis=1)
    # response of the leading truth jet: closest reco jet within MATCH_DR
    d_lt = np.sqrt((reta - lt_eta) ** 2 + dphi(rphi, lt_phi) ** 2)
    best = ak.argmin(d_lt, axis=1, keepdims=True)
    m_dr = ak.fill_none(ak.firsts(d_lt[best]), 9.0)
    m_pt = ak.fill_none(ak.firsts(rpt[best]), 0.0)
    resp = ak.where(lt_ok & (m_dr < MATCH_DR), m_pt / lt_pt, np.nan)
    out = dict(lead4=a.max_pt_truth_r04, lead_r=lead_r, lt_pt=lt_pt, unmatched=unmatched, resp=resp)
    return {k: ak.to_numpy(v).astype(float) for k, v in out.items()}


def load(paths, r_tag, jobs):
    with concurrent.futures.ProcessPoolExecutor(max_workers=jobs) as ex:
        outs = list(ex.map(process, [(p, r_tag) for p in paths]))
    return {k: np.concatenate([o[k] for o in outs]) for k in outs[0]}


def finite(x):
    return x[np.isfinite(x)]


def median_err(r):
    """Median and its approximate error (1.2533 sigma / sqrt(N), sigma from the interquartile range)."""
    if len(r) < 30:
        return np.nan, np.nan
    q1, q3 = np.percentile(r, [25, 75])
    return np.median(r), 1.2533 * (q3 - q1) / 1.349 / np.sqrt(len(r))


def binned(d, edges, select):
    """Per bin of the leading truth R=0.4 pT: (fraction, error) of events with a reco jet > HARD_PT, and median response."""
    frac, frac_err, med, med_err = [], [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        s = select & (d["lead4"] >= lo) & (d["lead4"] < hi)
        n = s.sum()
        f = np.mean(d["lead_r"][s] > HARD_PT) if n else np.nan
        frac.append(f)
        frac_err.append(np.sqrt(f * (1 - f) / n) if n else np.nan)
        m, e = median_err(finite(d["resp"][s]))
        med.append(m)
        med_err.append(e)
    return [np.array(x) for x in (frac, frac_err, med, med_err)]


def plot_lead_truth(data, hij, pp, thr, out_path, save_pdf):
    """Leading truth R=0.4 jet pT per event, HIJING-embedded vs p+p with the same generator filter, with HIJING / p+p."""
    hep.style.use("ATLAS")
    fig, (ax, ax_ratio) = plt.subplots(2, 1, figsize=(8, 8), gridspec_kw={"height_ratios": [3, 1], "hspace": 0.08}, sharex=True)
    edges = np.arange(0, 61, 1.0)
    vals = {}
    for name in (hij, pp):
        lead = data[name]["lead4"]
        h, _ = np.histogram(lead, bins=edges)
        vals[name] = (h / len(lead), np.sqrt(h) / len(lead))
        below = np.mean(lead < thr)
        hep.histplot(vals[name][0], edges, ax=ax, histtype="step", linewidth=2, color=STYLE[name]["color"],
                     label=f"{STYLE[name]['label']}: {100 * below:.0f}% below {thr} GeV")
    ax.axvline(thr, ymax=0.74, color="gray", linestyle="--", linewidth=1.5)  # below the legend
    ax.text(thr + 0.6, 0.03, f"Generator filter:\nR = 0.4 jet ≥ {thr} GeV", transform=ax.get_xaxis_transform(),
            fontsize=12, color="gray", va="bottom")
    ax.set_yscale("log")
    ax.set_ylim(1e-5, 50)
    ax.set_xlim(edges[0], edges[-1])
    ax.set_ylabel("Fraction of events / GeV")
    ax.legend(loc="upper right", frameon=False, fontsize=12)
    ax.text(0.97, 0.84, "Leading truth $R = 0.4$ jet, all events in the trees\n(HIJING: centrality < 60%)", transform=ax.transAxes,
            fontsize=12, va="top", ha="right")
    (hv, he), (pv, pe) = vals[hij], vals[pp]
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = hv / pv
        err = ratio * np.sqrt((he / hv) ** 2 + (pe / pv) ** 2)
    centers = 0.5 * (edges[1:] + edges[:-1])
    ok = np.isfinite(ratio) & (pv > 0) & (hv > 0)
    ax_ratio.errorbar(centers[ok], ratio[ok], yerr=err[ok], fmt="o", color=STYLE[hij]["color"], markersize=3, capsize=1)
    ax_ratio.axvline(thr, color="gray", linestyle="--", linewidth=1.5)
    setup_ratio_panel(ax_ratio, xlabel=r"Leading truth $R = 0.4$ jet $p_{T}$ [GeV]", ylabel="HIJING / p+p", ratio_ylim=(0, 2))
    plt.subplots_adjust(left=0.15, right=0.96, bottom=0.10, top=0.96)
    save_fig(fig, out_path, save_pdf)
    plt.close(fig)


def plot_reco_vs_truth(data, r_tag, out_path, save_pdf):
    """Per leading truth R=0.4 pT bin: fraction of events with a hard reco jet, and median response of the leading truth jet."""
    hep.style.use("ATLAS")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 9), sharex=True, gridspec_kw={"hspace": 0.06})
    centers = 0.5 * (LEAD_EDGES[1:] + LEAD_EDGES[:-1])
    xerr = 0.5 * np.diff(LEAD_EDGES)
    for i, name in enumerate(WINDOW_SAMPLES):
        d = data[name]
        frac, frac_err, med, med_err = binned(d, LEAD_EDGES, np.ones(len(d["lead4"]), dtype=bool))
        st = STYLE[name]
        off = (i - 1.5) * 0.15
        ok = np.isfinite(frac)
        ax1.errorbar(centers[ok] + off, frac[ok], yerr=frac_err[ok], xerr=xerr[ok], fmt=st["marker"], color=st["color"], markersize=6,
                     capsize=2, label=st["label"])
        ok = np.isfinite(med)
        ax2.errorbar(centers[ok] + off, med[ok], yerr=med_err[ok], xerr=xerr[ok], fmt=st["marker"], color=st["color"], markersize=6, capsize=2)
    r_jet = int(r_tag[1:]) / 10
    for ax in (ax1, ax2):
        ax.axvspan(LEAD_EDGES[0], 20, color="#d95f02", alpha=0.07)
    ax1.text(0.03, 0.62, "Below the Jet20\ngenerator filter", transform=ax1.transAxes, fontsize=12, color="#d95f02")
    ax1.set_ylabel(rf"Events with a reco jet $> {HARD_PT:g}$ GeV", fontsize=14)
    ax1.set_ylim(0, 0.9)
    ax1.legend(loc="upper left", frameon=False, fontsize=11)
    ax1.text(1.0, 1.01, rf"Reco $R = {r_jet:g}$ iter $p_{{T}}^{{\mathrm{{calib}}}}$, $|\eta| < {ETA_FID:g}$", transform=ax1.transAxes,
             ha="right", va="bottom", fontsize=13)
    ax2.axhline(1.0, color="gray", linestyle="--", linewidth=1)
    ax2.set_ylabel(rf"Median response, leading truth $R = {r_jet:g}$ jet", fontsize=14)
    ax2.set_ylim(0.8, 2.4)
    ax2.set_xlabel(r"Leading truth $R = 0.4$ jet $p_{T}$ [GeV]")
    ax2.set_xlim(LEAD_EDGES[0], LEAD_EDGES[-1])
    plt.subplots_adjust(left=0.15, right=0.96, bottom=0.09, top=0.95)
    save_fig(fig, out_path, save_pdf)
    plt.close(fig)


def plot_in_window(data, r_tag, out_path, save_pdf):
    """Events in the Jet20 window: median response vs leading truth jet pT, with HIJING Jet20 / Jet10 and p+p Jet20 / Jet12."""
    hep.style.use("ATLAS")
    fig, (ax, ax_ratio) = plt.subplots(2, 1, figsize=(8, 8), gridspec_kw={"height_ratios": [3, 1], "hspace": 0.08}, sharex=True)
    centers = 0.5 * (RESP_EDGES[1:] + RESP_EDGES[:-1])
    xerr = 0.5 * np.diff(RESP_EDGES)
    meds = {}
    for i, name in enumerate(WINDOW_SAMPLES):
        d = data[name]
        sel = (d["lead4"] >= WINDOW[0]) & (d["lead4"] < WINDOW[1])
        m, e = [], []
        for lo, hi in zip(RESP_EDGES[:-1], RESP_EDGES[1:]):
            mm, ee = median_err(finite(d["resp"][sel & (d["lt_pt"] >= lo) & (d["lt_pt"] < hi)]))
            m.append(mm)
            e.append(ee)
        meds[name] = (np.array(m), np.array(e))
        st = STYLE[name]
        ok = np.isfinite(meds[name][0])
        ax.errorbar(centers[ok] + (i - 1.5) * 0.15, meds[name][0][ok], yerr=meds[name][1][ok], xerr=xerr[ok], fmt=st["marker"],
                    color=st["color"], markersize=6, capsize=2, label=st["label"])
    ratios = []
    for num, den in (("HIJING Jet20", "HIJING Jet10"), ("p+p Jet20", "p+p Jet12")):
        (n, ne), (dd, de) = meds[num], meds[den]
        r = n / dd
        re = r * np.sqrt((ne / n) ** 2 + (de / dd) ** 2)
        ok = np.isfinite(r)
        ax_ratio.errorbar(centers[ok], r[ok], yerr=re[ok], xerr=xerr[ok], fmt=STYLE[num]["marker"], color=STYLE[num]["color"], markersize=5,
                          capsize=2, label=f"{num.split()[0]} Jet20 / {den.split()[1]}")
        ratios += list(r[ok])
    r_jet = int(r_tag[1:]) / 10
    ax.set_ylabel(rf"Median response, leading truth $R = {r_jet:g}$ jet", fontsize=14)
    ax.set_xlim(RESP_EDGES[0], RESP_EDGES[-1])
    lo = min(np.nanmin(m) for m, _ in meds.values())
    hi = max(np.nanmax(m) for m, _ in meds.values())
    ax.set_ylim(lo - 0.3 * (hi - lo), hi + 1.0 * (hi - lo))
    ax.legend(loc="upper right", frameon=False, fontsize=11)
    ax.text(0.03, 0.04, f"Events with the leading truth $R = 0.4$ jet in [{WINDOW[0]}, {WINDOW[1]}) GeV\n(the HIJING Jet20 window); "
            rf"reco iter $p_{{T}}^{{\mathrm{{calib}}}}$, $|\eta| < {ETA_FID:g}$", transform=ax.transAxes, fontsize=12, va="bottom")
    ax_ratio.legend(loc="upper right", frameon=False, fontsize=10, ncol=2)
    setup_ratio_panel(ax_ratio, xlabel=rf"Leading truth $R = {r_jet:g}$ jet $p_{{T}}$ [GeV]", ylabel="Jet20 / control", ratio_ylim=(0.96, 1.06))
    plt.subplots_adjust(left=0.15, right=0.96, bottom=0.10, top=0.96)
    save_fig(fig, out_path, save_pdf)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Reco vs truth in HIJING Jet20 events with the leading truth jet below the generator filter.")
    parser.add_argument("-r", "--radius", choices=["r02", "r03"], default="r03", help="Jet radius (default: r03)")
    parser.add_argument("--hijing-list", default=HIJING_LIST, help=f"HIJING tree list (default {HIJING_LIST})")
    parser.add_argument("--pp-list", default=PP_LIST, help=f"p+p tree list (default {PP_LIST})")
    parser.add_argument("--n-files", type=int, nargs=6, default=[300, 1500, 300, 3000, 300, 300], metavar=("H20", "H10", "PP20", "PP12", "H30", "PP30"),
                        help="Trees per sample: HIJING Jet20, HIJING Jet10, p+p Jet20, p+p Jet12, HIJING Jet30, p+p Jet30 "
                             "(default: 300 1500 300 3000 300 300; Jet10 and Jet12 need many for the in-window control)")
    parser.add_argument("-j", "--jobs", type=int, default=min(16, os.cpu_count() or 1), help="Parallel worker processes")
    parser.add_argument("--plot-dir", type=Path, help="Save the figures here (leading truth jet distributions, reco vs truth, in-window response)")
    parser.add_argument("--save-pdf", action="store_true", help="Also save the figures as PDF")
    parser.add_argument("--save-npz", type=Path, help="Save the per-event arrays to this .npz file")
    args = parser.parse_args()
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    n_h20, n_h10, n_p20, n_p12, n_h30, n_p30 = args.n_files
    samples = {"HIJING Jet20": files(args.hijing_list, "jet20", n_h20), "HIJING Jet10": files(args.hijing_list, "jet10", n_h10),
               "p+p Jet20": files(args.pp_list, "jet20", n_p20), "p+p Jet12": files(args.pp_list, "jet12", n_p12),
               "HIJING Jet30": files(args.hijing_list, "jet30", n_h30), "p+p Jet30": files(args.pp_list, "jet30", n_p30)}
    data = {}
    for name, paths in samples.items():
        if not paths:
            print(f"Warning: no trees for {name}, skipped")
            continue
        data[name] = load(paths, args.radius, args.jobs)
        d = data[name]
        print(f"{name}: {len(paths)} trees, {len(d['lead4'])} events, leading truth R=0.4 < 20 GeV: {np.mean(d['lead4'] < 20):.3f}")

    print(f"\nR = {args.radius}: reco iter pt_calib >= 10 GeV in |eta| < {ETA_FID}; truth jets >= 5 GeV in |eta| < 1.5")
    print(f"1. Per leading truth R=0.4 pT bin: reco jet > {HARD_PT:g} GeV, the same with no truth jet within dR {UNMATCHED_DR:g}, "
          f"median response of the leading truth jet")
    for name, d in data.items():
        for lo, hi in LEAD_BINS:
            s = (d["lead4"] >= lo) & (d["lead4"] < hi)
            if s.sum() < 50:
                continue
            resp = finite(d["resp"][s])
            print(f"  {name:<13} [{lo:>2},{hi:>2}): N {s.sum():>8}  reco > {HARD_PT:g}: {np.mean(d['lead_r'][s] > HARD_PT):.3f}"
                  f"  unmatched: {np.mean(d['unmatched'][s]):.4f}  response median {np.median(resp) if len(resp) else np.nan:.3f} (N {len(resp)})")
        print()

    print(f"2. In-window: leading truth R=0.4 in [{WINDOW[0]}, {WINDOW[1]}), response of the leading truth jet vs its pT: "
          f"median (core mean in {CORE[0]:g}-{CORE[1]:g}), fraction > 1.4")
    for lo, hi in TRUTH_PT_BINS:
        cells = []
        for name, d in data.items():
            if name not in WINDOW_SAMPLES:
                continue
            s = (d["lead4"] >= WINDOW[0]) & (d["lead4"] < WINDOW[1]) & (d["lt_pt"] >= lo) & (d["lt_pt"] < hi)
            r = finite(d["resp"][s])
            if len(r) < 30:
                cells.append(f"{name}: N {len(r)}")
                continue
            core = r[(r > CORE[0]) & (r < CORE[1])]
            cells.append(f"{name}: N {len(r)} {np.median(r):.3f} ({core.mean():.3f}) {np.mean(r > 1.4):.3f}")
        print(f"  truth {args.radius} [{lo},{hi}): " + " | ".join(cells))

    if args.plot_dir:
        for thr, (hij, pp) in FILTER_PAIRS.items():
            if hij in data and pp in data:
                plot_lead_truth(data, hij, pp, thr, args.plot_dir / f"lead_truth_r04_jet{thr}.png", args.save_pdf)
        if all(name in data for name in WINDOW_SAMPLES):
            plot_reco_vs_truth(data, args.radius, args.plot_dir / f"reco_vs_truth_{args.radius}.png", args.save_pdf)
            plot_in_window(data, args.radius, args.plot_dir / f"in_window_response_{args.radius}.png", args.save_pdf)

    if args.save_npz:
        np.savez(args.save_npz, **{f"{n.replace(' ', '_').replace('+', '')}__{k}": v for n, d in data.items() for k, v in d.items()})
        print(f"\nSaved: {args.save_npz}")


if __name__ == "__main__":
    main()
