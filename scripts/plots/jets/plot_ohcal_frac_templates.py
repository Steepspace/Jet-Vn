#!/usr/bin/env python3
"""Supporting plots of the OHCal fraction templates (tasks/jet-ohcal-frac-templates.md): how the minimum-bias HIJING
(fake-jet model) and PYTHIA8 p+p (real-jet model) inputs of C5 (plot_jet_dijet_timing.py) were obtained.

Inputs: the merged Fun4All MB HIJING hist file (EventQA hEvent), the merged Jet-JES mode 3 output of MB HIJING, the
merged Jet-JES p+p outputs JetNN.root with the x templates, the merged Fun4All p+p hist files (h_gen_integral, for the
wiki weights) and the C5 CSVs.

This script makes, in --output-dir:
- M1_mb_events: MB HIJING event selection (EventQA cut flow, then the Jet-JES 0-60% class) and the centrality of the
  selected events (0-60%, 1% bins).
- M2_mb_jet_pt_<r>: raw pT spectrum of the MB HIJING reco jets (every jet with E > 0, |eta| < eta_max, x finite; the
  jets of the x templates) per 10% centrality class, per selected event of the class, with the C4 raw pT classes marked
  and the jets per class (0-60%) listed.
- P1_pp_contrib_<r>: raw pT spectrum of the matched p+p reco jets (iternf), wiki-weighted, per pT-hat sample (each in its
  leading truth R=0.4 window) and summed, with the fraction from each sample.
- P2_pp_unmatched_<r>: fraction of the p+p reco jets (iternf and unsub) not matched to a truth jet vs raw pT
  (wiki-weighted sum of the samples).
- S1_mean_x: <x> per C4 raw pT class of the four C5 shapes (data inside and outside the window from the C5 CSV, the MB
  HIJING and p+p templates from the hists, with the error RMS / sqrt(N_eff)), both radii, every jet.
"""

import argparse
import csv
from pathlib import Path
import numpy as np
import uproot
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mplhep as hep

from plot_jet_bkgsub_compare import save_fig
from plot_jet_jes_overlay import parse_sample_list, load_weights, XSEC_WIKI_RUN28
from plot_jet_dijet_timing import (MB_FILE, PP_LIST, PP_GEN_LIST, MB_CENTS, CUT_PT_CLASSES, C_IN, C_OUT, C_FAKE, C_REAL,
                                   load_templates, template_class)

MB_F4A_FILE = Path("/gpfs/mnt/gpfs02/sphenix/user/anarde/sEPD-Study/f4a_bkgsub_mc/11-23-26-hijing-mb/merged/output/MB.root")
C5_DIR = Path("/sphenix/user/anarde/sEPD-Study/plots/Jet-Ana/dijet_timing/timing/cut")
OUT_DIR = Path("/sphenix/user/anarde/sEPD-Study/plots/Jet-Ana/ohcal_frac_templates")
RADII = {"r02": {"r": 0.2, "eta": 0.9}, "r03": {"r": 0.3, "eta": 0.8}}
C4_EDGES = sorted({e for c in CUT_PT_CLASSES for e in c})
UNMATCHED_EDGES = np.array([7, 8, 9, 10, 12, 14, 16, 18, 20, 23, 26, 30, 35, 40, 50, 60])
SAMPLE_COLORS = {"Jet12": "#1b9e77", "Jet20": "#d95f02", "Jet30": "#7570b3", "Jet40": "#e7298a"}
CENT_COLORS = plt.get_cmap("viridis")(np.linspace(0.0, 0.85, len(MB_CENTS)))


def label(ax, lines, x=0.04, y=0.96, ha="left", fs=14):
    ax.text(x, y, "\n".join(lines), transform=ax.transAxes, ha=ha, va="top", fontsize=fs, linespacing=1.3)


def mark_classes(ax, lo=None, hi=None):
    for e in C4_EDGES:
        if (lo is None or e >= lo) and (hi is None or e <= hi):
            ax.axvline(e, color="grey", linestyle=":", linewidth=1.0, zorder=0)


def h2(rf, name):
    h = rf[name]
    return h.values(flow=True).astype(np.float64), h.variances(flow=True).astype(np.float64), np.asarray(h.axis(1).edges())


def pt_spectrum(v, var):
    """Jets per raw pT bin (pT axis without flow) of an x vs pT hist with flow: summed over x, flow included."""
    return v.sum(axis=0)[1:-1], var.sum(axis=0)[1:-1]


def merge_pt(n, var, pe, edges):
    """1 GeV pT bins merged to edges (each an edge of pe)."""
    idx = np.searchsorted(pe, edges - 1e-9)
    return np.add.reduceat(n, idx[:-1]), np.add.reduceat(var, idx[:-1])


# M1 / M2: MB HIJING
# ---------------------------------------------------------------------------------------------------------------------

def plot_m1(f4a_file, jes_file, out_dir, save_pdf):
    """EventQA cut flow of the MB HIJING production and the Jet-JES 0-60% class; centrality of the selected events."""
    with uproot.open(f4a_file) as rf:
        h = rf["hEvent"]
        flow = [(l, v) for l, v in zip(h.axis().labels(), h.values()) if l and l != "MB Trig"]   # no trigger in simulation
    with uproot.open(jes_file) as rf:
        jl = dict(zip(rf["hEvent"].axis().labels(), rf["hEvent"].values()))
        hc = rf["hCentrality"]
        cvals, cedges = hc.values(), np.asarray(hc.axis().edges())
    names = {"All": "All", "Has Z": "MBD vertex", "|z| < 150 cm": r"$|z| < 150$ cm", "|z| < 10 cm": r"$|z| < 10$ cm",
             "MB": "Minimum bias (simulation)", "Cent": "Centrality < 80%"}
    steps = [(names.get(l, l), v) for l, v in flow] + [("Centrality 0–60%", jl["Centrality 0-60%"])]
    hep.style.use("ATLAS")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8), gridspec_kw={"width_ratios": [1.1, 1]})
    y = np.arange(len(steps))[::-1]
    ax1.barh(y, [v for _, v in steps], color=C_FAKE, alpha=0.8)
    for yi, (n, v) in zip(y, steps):
        ax1.text(v * 1.01, yi, f"{v / 1e6:.3f} M ({v / steps[0][1]:.1%})", va="center", fontsize=20)
    ax1.set_yticks(y, [n for n, _ in steps], fontsize=20)
    ax1.set_xlim(0, steps[0][1] * 1.65)
    ax1.set_ylim(-2.6, len(steps) - 0.4)                                 # empty space under the bars for the text
    ax1.set_xlabel("Events")
    label(ax1, [r"$\bf{MB\ HIJING}$ (run 31, 0–20 fm, no pileup)", "14988 segments × 200 events",
                "Last step: analysis centrality range 0–60%"], x=0.97, y=0.19, ha="right", fs=19)
    centers = 0.5 * (cedges[1:] + cedges[:-1])
    ax2.stairs(cvals, cedges, color=C_FAKE, linewidth=2)
    ax2.axhline(cvals[3:].mean(), color="grey", linestyle="--", linewidth=1.2, label=f"mean of 3–59%: {cvals[3:].mean() / 1e3:.1f} k")
    for c in (10, 20, 30, 40, 50):
        ax2.axvline(c - 0.5, color="grey", linestyle=":", linewidth=1.0)
    ax2.set_xlim(cedges[0], cedges[-1])
    ax2.set_ylim(0, cvals.max() * 1.3)
    ax2.set_xlabel("Centrality [%]")
    ax2.set_ylabel("Events per 1%")
    ax2.legend(loc="upper right", bbox_to_anchor=(0.99, 0.72), fontsize=18, frameon=False)   # under the text
    label(ax2, ["Selected events, 0–60%", f"{cvals.sum() / 1e3:.0f} k events",
                f"most central bins: {cvals[1] / 1e3:.1f} k, {cvals[2] / 1e3:.1f} k"], x=0.25, y=0.96, fs=18)
    print(f"M1: " + ", ".join(f"{n} {v:.0f}" for n, v in steps) + f"; centrality bins 0-3: {cvals[:4]}")
    fig.subplots_adjust(left=0.17, right=0.98, bottom=0.12, top=0.95, wspace=0.18)
    save_fig(fig, out_dir / "M1_mb_events.png", save_pdf)
    plt.close(fig)
    return centers, cvals, cedges


def plot_m2(jes_file, cent_vals, cent_edges, r_tag, out_dir, save_pdf, pt_max=40):
    """Raw pT spectrum of the MB HIJING x-template jets per 10% class, per selected event of the class."""
    hep.style.use("ATLAS")
    fig, ax = plt.subplots(figsize=(11, 8.5))
    counts = {}
    with uproot.open(jes_file) as rf:
        for ic, (lo, hi) in enumerate(MB_CENTS):
            v, var, pe = h2(rf, f"h2OHCalFracPt_{r_tag}_iternf_all_incl_cent_{lo}_{hi}")
            n, nv = pt_spectrum(v, var)
            sel = (cent_edges[:-1] >= lo - 0.5 - 1e-9) & (cent_edges[1:] <= hi - 0.5 + 1e-9)
            n_ev = cent_vals[sel].sum()
            ok = (pe[:-1] < pt_max) & (n > 0)
            xc = 0.5 * (pe[1:] + pe[:-1])
            ax.errorbar(xc[ok], n[ok] / n_ev, yerr=np.sqrt(nv[ok]) / n_ev, xerr=0.5, fmt="o", markersize=4,
                        color=CENT_COLORS[ic], label=f"{lo}–{hi}% ({n_ev / 1e3:.0f} k events)")
            for a, b in CUT_PT_CLASSES:
                s = (pe[:-1] >= a - 1e-9) & (pe[1:] <= b + 1e-9)
                counts[(a, b)] = counts.get((a, b), 0) + n[s].sum()
    mark_classes(ax, hi=pt_max)
    ax.set_yscale("log")
    ax.set_xlim(0, pt_max)
    ax.set_ylim(top=ax.get_ylim()[1] * 3)
    ax.set_xlabel(r"Reco jet raw $p_{T}$ [GeV]")
    ax.set_ylabel(r"Jets per event per GeV")
    ax.legend(loc="upper right", fontsize=16, frameon=False, ncol=2, columnspacing=1.0)
    lines = [rf"$\bf{{MB\ HIJING}}$, R = {RADII[r_tag]['r']}, Iterative No Flow",
             rf"$E > 0$, $|\eta| < {RADII[r_tag]['eta']}$, $x$ finite, stored if $p_{{T}}^{{calib}} \geq 10$ GeV",
             r"Jets per raw $p_{T}$ class (0–60%):"] + [f"  {a}–{b} GeV: {counts[(a, b)]:.0f}" for a, b in CUT_PT_CLASSES if a < pt_max]
    label(ax, lines, x=0.96, y=0.80, ha="right", fs=18)                 # right, under the legend (the spectra fall to the right)
    print(f"M2 {r_tag}: " + ", ".join(f"{a}-{b}: {c:.0f}" for (a, b), c in counts.items()))
    fig.subplots_adjust(left=0.13, right=0.96, bottom=0.11, top=0.96)
    save_fig(fig, out_dir / f"M2_mb_jet_pt_{r_tag}.png", save_pdf)
    plt.close(fig)


# P1 / P2: PYTHIA8 p+p
# ---------------------------------------------------------------------------------------------------------------------

def load_pp(pp_list, pp_gen_list):
    samples = parse_sample_list(pp_list)
    weights = load_weights(parse_sample_list(pp_gen_list), samples, "wiki", XSEC_WIKI_RUN28, {})
    windows = {}
    for s, p in samples.items():
        with uproot.open(p) as rf:
            windows[s] = rf["hEvent"].axis().labels()[-1]
    return samples, weights, windows


def plot_p1(samples, weights, windows, r_tag, out_dir, save_pdf, pt_max=80):
    """Matched p+p reco jets (iternf, every jet) per raw pT GeV, weighted, per sample and summed; fraction per sample."""
    hep.style.use("ATLAS")
    fig, (ax, axr) = plt.subplots(2, 1, figsize=(11, 10), sharex=True, gridspec_kw={"height_ratios": [2.2, 1]})
    parts = {}
    for s, p in samples.items():
        with uproot.open(p) as rf:
            v, var, pe = h2(rf, f"h2OHCalFracPt_{r_tag}_iternf_all_matched_cent_0_100")
        n, nv = pt_spectrum(v, var)
        parts[s] = (n * weights[s], nv * weights[s] ** 2)
    tot = sum(n for n, _ in parts.values())
    tot_v = sum(nv for _, nv in parts.values())
    xc = 0.5 * (pe[1:] + pe[:-1])
    ok = (pe[:-1] < pt_max) & (tot > 0)
    ax.errorbar(xc[ok], tot[ok], yerr=np.sqrt(tot_v[ok]), xerr=0.5, fmt="o", color="black", markersize=4, label="sum")
    for s, (n, nv) in parts.items():
        win = windows[s].split("Window")[-1].strip().replace("#infty", r"$\infty$")
        m = ok & (n > 0)
        ax.stairs(np.where(ok, n, np.nan), pe, color=SAMPLE_COLORS.get(s), linewidth=2, label=f"{s}, lead truth R=0.4 {win}")
        with np.errstate(invalid="ignore", divide="ignore"):
            axr.stairs(np.where(ok, n / tot, np.nan), pe, color=SAMPLE_COLORS.get(s), linewidth=2)
    for a in (ax, axr):
        mark_classes(a, hi=pt_max)
    ax.set_yscale("log")
    ax.set_ylim(tot[ok].min() * 0.3, tot[ok].max() * 10)
    ax.set_ylabel(r"$d\sigma/dp_{T}$ (matched jets) [pb/GeV]")
    ax.legend(loc="upper right", fontsize=17, frameon=False)
    label(ax, [rf"$\bf{{PYTHIA8\ p+p}}$, R = {RADII[r_tag]['r']}, Iterative No Flow", "Reco jets matched to truth",
               r"weight = $\sigma$ / $N_{\mathrm{events}}$ per sample"], x=0.96, y=0.62, ha="right", fs=18)   # under the legend
    axr.set_ylim(0, 1.05)
    axr.set_xlim(0, pt_max)
    axr.set_ylabel("Fraction")
    axr.set_xlabel(r"Reco jet raw $p_{T}$ [GeV]")
    fig.subplots_adjust(left=0.13, right=0.96, bottom=0.09, top=0.97, hspace=0.05)
    save_fig(fig, out_dir / f"P1_pp_contrib_{r_tag}.png", save_pdf)
    plt.close(fig)
    for a, b in CUT_PT_CLASSES[:6]:
        s = (pe[:-1] >= a - 1e-9) & (pe[1:] <= b + 1e-9)
        print(f"P1 {r_tag} {a}-{b} GeV: " + ", ".join(f"{k} {parts[k][0][s].sum() / tot[s].sum():.3f}" for k in parts))


def plot_p2(samples, weights, r_tag, out_dir, save_pdf):
    """Unmatched fraction of the p+p reco jets vs raw pT (weighted sum of the samples), iternf and unsub."""
    hep.style.use("ATLAS")
    fig, ax = plt.subplots(figsize=(11, 8))
    for ue, col, mk, ue_label in (("iternf", C_REAL, "o", "Iterative No Flow"), ("unsub", "#984ea3", "s", "No subtraction")):
        acc = {}
        for s, p in samples.items():
            with uproot.open(p) as rf:
                for m in ("matched", "unmatched"):
                    v, var, pe = h2(rf, f"h2OHCalFracPt_{r_tag}_{ue}_all_{m}_cent_0_100")
                    n, nv = pt_spectrum(v, var)
                    n, nv = merge_pt(n, nv, pe, UNMATCHED_EDGES.astype(float))
                    a = acc.get(m, (0, 0))
                    acc[m] = (a[0] + n * weights[s], a[1] + nv * weights[s] ** 2)
        (mn, mv), (un, uv) = acc["matched"], acc["unmatched"]
        tot = mn + un
        frac = un / tot
        err = np.sqrt((mn / tot ** 2) ** 2 * uv + (un / tot ** 2) ** 2 * mv)
        xc = 0.5 * (UNMATCHED_EDGES[1:] + UNMATCHED_EDGES[:-1])
        ax.errorbar(xc, frac, yerr=err, xerr=0.5 * np.diff(UNMATCHED_EDGES), fmt=mk, color=col, markersize=7, label=ue_label)
        print(f"P2 {r_tag} {ue}: " + ", ".join(f"{a:g}-{b:g} {f:.3f}" for a, b, f in zip(UNMATCHED_EDGES[:-1], UNMATCHED_EDGES[1:], frac)))
    mark_classes(ax, hi=UNMATCHED_EDGES[-1])
    ax.set_xlim(UNMATCHED_EDGES[0], UNMATCHED_EDGES[-1])
    ax.set_ylim(0, ax.get_ylim()[1] * 1.5)
    ax.set_xlabel(r"Reco jet raw $p_{T}$ [GeV]")
    ax.set_ylabel("Unmatched fraction")
    ax.legend(loc="upper right", fontsize=20, frameon=False)
    label(ax, [rf"$\bf{{PYTHIA8\ p+p}}$, R = {RADII[r_tag]['r']}, every reco jet",
               rf"$E > 0$, $|\eta| < {RADII[r_tag]['eta']}$, $x$ finite",
               r"unmatched: no truth jet within $\Delta R < 0.75\,R$", "(one-to-one, truth jets by decreasing $p_T$)",
               r"Jet12–40, weight = $\sigma$ / $N_{\mathrm{events}}$ per sample"], fs=18)
    fig.subplots_adjust(left=0.13, right=0.96, bottom=0.11, top=0.96)
    save_fig(fig, out_dir / f"P2_pp_unmatched_{r_tag}.png", save_pdf)
    plt.close(fig)


# S1: <x> per class
# ---------------------------------------------------------------------------------------------------------------------

def mean_err(n, var, xe):
    """<x>, RMS / sqrt(N_eff) over the x axis (flow left out)."""
    n, var = n[1:-1], var[1:-1]
    xc = 0.5 * (xe[1:] + xe[:-1])
    tot = n.sum()
    if tot <= 0:
        return np.nan, np.nan, 0.0
    m = (n * xc).sum() / tot
    rms = np.sqrt(max((n * xc ** 2).sum() / tot - m ** 2, 0))
    neff = tot ** 2 / var.sum()
    return m, rms / np.sqrt(neff), neff


def plot_s1(templates, c5_dir, out_dir, save_pdf):
    hep.style.use("ATLAS")
    fig, axs = plt.subplots(1, 2, figsize=(20, 8), sharey=True)
    for ax, r_tag in zip(axs, RADII):
        rows = list(csv.DictReader(open(c5_dir / f"C5_jetx_3sigma_{r_tag}.csv")))
        lo = np.array([float(r["pt_lo"]) for r in rows])
        hi = np.array([float(r["pt_hi"]) for r in rows])
        xc = np.sqrt(lo * hi)
        xerr = [xc - lo, hi - xc]
        for key, col, mk, lab, dx in (("in", C_IN, "o", "data inside", 0.97), ("out", C_OUT, "s", "data outside", 1.03)):
            ax.errorbar(xc * dx, [float(r[f"mean_x_{key}"]) for r in rows], xerr=[xerr[0] * dx, xerr[1] * dx], fmt=mk,
                        color=col, markersize=8, label=lab)
        for key, col, mk, lab, dx in (("fake", C_FAKE, "^", "MB HIJING", 0.94), ("real", C_REAL, "D", "PYTHIA8 p+p matched", 1.06)):
            t = templates[(r_tag, "all")][key]
            vals = [mean_err(*template_class(t, a, b), t[2]) for a, b in zip(lo, hi)]
            m = np.array([v[0] for v in vals])
            e = np.array([v[1] for v in vals])
            neff = np.array([v[2] for v in vals])
            good = neff >= 100
            ax.errorbar(xc[good] * dx, m[good], yerr=e[good], fmt=mk, color=col, markersize=8, label=lab)
            if (~good & (neff > 0)).any():
                bad = ~good & (neff > 0)
                ax.errorbar(xc[bad] * dx, m[bad], yerr=e[bad], fmt=mk, color=col, markersize=8, markerfacecolor="white",
                            label=f"{lab} (< 100 jets)")
        ax.set_xscale("log")
        ax.set_xticks([7, 10, 15, 20, 30, 40, 60, 150], ["7", "10", "15", "20", "30", "40", "60", "150"])
        ax.set_xlim(6, 170)
        ax.set_ylim(0, 1.05)
        ax.set_xlabel(r"Reco jet raw $p_{T}$ class [GeV]")
        label(ax, [rf"$\bf{{R = {RADII[r_tag]['r']}}}$, Iterative No Flow, every jet, 0–60%"], y=0.09, fs=21)   # bottom left: empty there
        ax.legend(loc="upper right", fontsize=16, frameon=False, ncol=2, columnspacing=1.0)
    axs[0].set_ylabel(r"$\langle x \rangle$ (OHCal energy fraction)")
    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.12, top=0.96, wspace=0.05)
    save_fig(fig, out_dir / "S1_mean_x.png", save_pdf)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Supporting plots of the OHCal fraction templates (MB HIJING, PYTHIA8 p+p).")
    parser.add_argument("--mb-f4a-file", type=Path, default=MB_F4A_FILE, help="Merged Fun4All MB HIJING hist file (default: %(default)s)")
    parser.add_argument("--mb-file", type=Path, default=MB_FILE, help="Merged Jet-JES mode 3 MB HIJING output (default: %(default)s)")
    parser.add_argument("--pp-list", type=Path, default=PP_LIST, help="Merged Jet-JES p+p outputs (default: %(default)s)")
    parser.add_argument("--pp-gen-list", type=Path, default=PP_GEN_LIST, help="Merged Fun4All p+p hist files (default: %(default)s)")
    parser.add_argument("--c5-dir", type=Path, default=C5_DIR, help="Directory of the C5 CSVs (default: %(default)s)")
    parser.add_argument("-o", "--output-dir", type=Path, default=OUT_DIR, help="Output directory (default: %(default)s)")
    parser.add_argument("--save-pdf", action="store_true", help="Save PDF versions in addition to PNG")
    args = parser.parse_args()

    out = args.output_dir
    _, cvals, cedges = plot_m1(args.mb_f4a_file, args.mb_file, out, args.save_pdf)
    for r_tag in RADII:
        plot_m2(args.mb_file, cvals, cedges, r_tag, out, args.save_pdf)
    samples, weights, windows = load_pp(args.pp_list, args.pp_gen_list)
    for r_tag in RADII:
        plot_p1(samples, weights, windows, r_tag, out, args.save_pdf)
        plot_p2(samples, weights, r_tag, out, args.save_pdf)
    templates, _ = load_templates(args.mb_file, args.pp_list, args.pp_gen_list, list(RADII))
    plot_s1(templates, args.c5_dir, out, args.save_pdf)


if __name__ == "__main__":
    main()
