#!/usr/bin/env python3
"""Leading-jet checks in the clean dijet sample (tasks/jet-dijet-timing.md), from the Jet-Dijet production with the
dijet timing hists (first in 11-10-26) and the leading-jet pT per selection (from 11-11-26 on), the production 11-11-26.

Jet-Dijet.C applies one pT_sub cut per radius (the 0-20% cut of the purity scan, R = 0.2: 14.25 GeV, R = 0.3: 21.5 GeV,
raw, delta phi > 7pi/8) to every centrality class and fills, per radius:
- h2LeadTimeOHCalFrac_<r>_<dijet|incl>_cent_<c>_ptlead_<a>_<b>: leading jet t - MBD t0 vs OHCal fraction x, for the
  dijet sample and for the inclusive comparison (leading jet raw pT >= the cut, no dijet requirement), on a grid of
  centrality classes (10% up to production 11-12-26, 20% after) x raw pT_lead bins;
- h2JetPtCent_<r>_<dijet|all>: raw pT x centrality of every jet passing the jet cuts, in dijet events and in all events.

This script makes:
1. pt/P1_leadpt_<r>: the leading-jet raw pT of the dijet sample against the inclusive one (h2LeadPtCent_<r>_<sel>, from
   production 11-11-26 on), 0-60%, and their ratio: the dijet leading jets are a subset of the inclusive ones, so the
   ratio is the fraction of inclusive leading jets with a back-to-back partner. A check that the high-pT tail of the
   inclusive spectrum (jets without a balancing partner) is not in the dijet sample. leadpt_tail_<r>.csv: leading jets
   above 40 ... 100 GeV in each selection.
2. ohcal_frac/: the OHCal fraction x of the leading jet (the x projection of the timing hists, every filled jet), as
   unit-area shapes, all pT_lead:
   - X1_cent20_<r>: inclusive (left) and dijet (right), the three 20% classes, ratio to 0-60%;
   - X2_runs_<r>: inclusive (left) and dijet (right), 0-60%, the run groups (light: early, dark: late), ratio to all runs;
   - X3_runs_cent20_<r>_<incl|dijet>: one selection, one panel per 20% class, the run groups as in X2, ratio to all runs
     of the class;
   - x_moments_<r>.csv: N, <x> and its error, RMS for every (selection, run group or all, grid class / 20% class or 0-60%,
     pT_lead bin or all).
3. timing/T1_time_vs_x_<r>_dijet: leading jet t - MBD t0 vs x of the dijet sample, 0-60% and all pT_lead, with the
   profile <t - t0> per x bin, its standard deviation and a quartic fitted to the profile;
   timing/T2_profiles_cent20_<r>_dijet: the profiles of the three 20% classes with the T1 fit, and profile - fit [ns].
   timing/T3_closure_<r>_dijet: uncorrected (left) and corrected (right) t vs x of the dijet sample, with their profiles
   (the _corr hists of Jet-Dijet.C with a timing correction file, production 11-12-26 on).
   timing/T4_leadpt_<r>_<dijet|lead>[_zoom60], T5_jetpt_<r>_<dijet|all>[_zoom60]: the corrected t - t0 - f(x) vs raw pT
   (h2LeadTimeCorrPt / h2JetTimeCorrPt, production 11-12-26-v2 on), summed over the 20% classes, with the profile per
   pT bin; lead / all shown to 150 GeV, and zoomed to 60 GeV.
   timing/projections/lead_dijet/T4proj_<r>_dijet*: the corrected time of the dijet leading jets (the sample the
   correction was fitted to), each with a Gaussian fitted iteratively within +-2 sigma of its peak and its mu +- 3 sigma
   marked: full pT range (log over +-75 ns and linear over +-10 ns), per pT window and per 20% class (single plots and
   panels), mu and sigma vs pT and centrality (_fits_vs_pt, _fits_vs_cent), and a CSV of every fit (_fits.csv).
   timing/projections/all_jets/T5proj_<r>_all[_leaddijet3sigma][_zoom10]: every jet of every selected event projected
   on t - t0 - f(x) (every pT), log and +-10 ns; _leaddijet3sigma with the dijet leading-jet full-range mu +- 3 sigma
   marked and the jets outside counted.
   timing/cut/C1_jetpt_3sigma_<r>: the raw pT spectrum of every jet of every selected event before and after a single
   timing window (the dijet leading-jet full-range fit's mu +- 3 sigma), with the fraction kept per pT bin;
   timing/cut/C2_jett_3sigma_<r>[_panels][_zoom10]: the corrected time of those jets per raw pT class (counts per ns,
   overlaid, or one class per panel) with the same window and the fraction of each class outside it.
   With --write-calib PATH, the T1 fits are written as the timing correction file of Jet-Dijet.C
   (files/calib/timing_dijet_fit_iternf.csv).
"""

import argparse
import bisect
import concurrent.futures
import csv
import hashlib
import os
from pathlib import Path
import pickle
import re
import sys
import numpy as np
import uproot
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.ticker import LogLocator, NullFormatter
import mplhep as hep
from scipy.optimize import curve_fit
from scipy.special import erf

from plot_jet_bkgsub_compare import save_fig
from plot_jet_noflow_compare import parse_list_file, load_luminosity
from plot_jet_iternf_timing import StageTimer, add_header
from plot_jet_dijet_control import RADII, JET_PT_MIN, cent_label

CACHE_VERSION = 6                                          # 2: leading-jet raw pT per selection (11-11-26 on); 3: t vs x per (class, pT_lead) bin; 4: corrected dijet t (11-12-26 on); 5: grid classes from the hist names; 6: corrected t vs pT (11-12-26-v2 on)

SELS = ("dijet", "incl")                                   # timing hists: dijet sample, inclusive comparison
SEL_LABEL = {"dijet": "Dijet", "incl": "Inclusive"}
CLASSES20 = ((0, 20), (20, 40), (40, 60))
# First runs of the blocks in b_run of the timing study (tasks/jet-iternf-timing-ohcal.md, X_RUN_BLOCKS of
# plot_jet_iternf_timing.py): the run groups
RUN_BLOCKS = (67701, 68335, 69129, 73816, 75256, 77680)
# Leading-jet raw pT bins of P1 [GeV]: 1 GeV to 30, 2 GeV to 40, then wider so the tail stays visible
PT_PLOT_EDGES = (list(range(0, 30)) + list(range(30, 40, 2))
                 + [40, 45, 50, 55, 60, 70, 80, 90, 100, 120, 140, 160, 200, 250, 300])
TAIL_THRESHOLDS = (40, 50, 60, 80, 100)
X_LABEL = r"Leading jet OHCal energy fraction $x$"
# Corrected time vs raw pT hists of Jet-Dijet.C (with a timing correction file, 11-12-26-v2 on), per 20% class:
# (kind, name prefix, sample tags)
TPT_HISTS = (("lead", "h2LeadTimeCorrPt", ("dijet", "lead")), ("jet", "h2JetTimeCorrPt", ("dijet", "all")))
TPT_PREFIX = {kind: pre for kind, pre, _ in TPT_HISTS}
TPT_MAX = 150                                              # [GeV] pT axis of the lead / all plots (jets above are counted in the text)
TPT_ZOOM = 60                                              # [GeV] pT axis of their zoomed versions
NAME_RE = re.compile(r"h2LeadTimeOHCalFrac_(r0\d)_(dijet|incl)_cent_(\d+)_(\d+)_ptlead_([\dp]+)_([\dp]+)$")


def edge_tag(x):
    """A bin edge as in the hist names of Jet-Dijet.C: 14.25 -> "14p25", 20 -> "20"."""
    return f"{x:g}".replace(".", "p")


def edge_val(tag):
    return float(tag.replace("p", "."))


# Input
# ---------------------------------------------------------------------------------------------------------------------

def read_layout(path):
    """Per radius: the pT_lead bin edges of the timing grid; the centrality classes of the grid (10% up to production
    11-12-26, 20% after); both from the hist names; and the x and jet pT axes."""
    lead = {r: set() for r in RADII}
    classes = set()
    with uproot.open(path) as f:
        names = f.keys(cycle=False)
        for k in names:
            m = NAME_RE.match(k)
            if m:
                lead[m.group(1)].update((edge_val(m.group(5)), edge_val(m.group(6))))
                classes.add((int(m.group(3)), int(m.group(4))))
        first = next(k for k in names if NAME_RE.match(k))
        h = f[first]
        hp = f["h2JetPtCent_r02_all"]
        layout = {"lead_edges": {r: sorted(v) for r, v in lead.items()},
                  "cent_classes": sorted(classes),
                  "x_edges": h.axis(0).edges(),
                  "t_edges": h.axis(1).edges(),
                  "pt_edges": hp.axis(0).edges(),
                  "cent_edges": hp.axis(1).edges(),
                  "tpt_edges": {}}
        for kind, pre, tags in TPT_HISTS:                               # from production 11-12-26-v2 on
            for r in RADII:
                for tag in tags:
                    name = f"{pre}_{r}_{tag}_cent_{CLASSES20[0][0]}_{CLASSES20[0][1]}"
                    if name in f:
                        layout["tpt_edges"][(kind, r, tag)] = (f[name].axis(0).edges(), f[name].axis(1).edges())
    return layout


def grid_name(r, sel, c, lo, hi):
    tag = lambda x: f"{x:g}".replace(".", "p")
    return f"h2LeadTimeOHCalFrac_{r}_{sel}_cent_{c[0]}_{c[1]}_ptlead_{tag(lo)}_{tag(hi)}"


def load_run(task):
    """Per run: the x projection (all t, flow included) of every timing hist as (cent, pT_lead, x) per (radius,
    selection), the x under/overflow per (radius, selection), the jet pT x centrality hists and the status hists."""
    run, path, layout = task
    d = {"grid": {}, "xflow": {}, "jetpt": {}, "status": {}, "leadpt": {}, "tx": {}, "tx_corr": {}, "tpt": {}}
    try:
        with uproot.open(path) as f:
            for r in RADII:
                le = layout["lead_edges"][r]
                for sel in SELS:
                    classes = layout["cent_classes"]
                    g = np.zeros((len(classes), len(le) - 1, len(layout["x_edges"]) - 1), dtype=np.float32)
                    fl = 0.0
                    tx = np.zeros((len(classes), len(le) - 1, len(layout["x_edges"]) - 1, len(layout["t_edges"]) - 1),
                                  dtype=np.float32)
                    txc = np.zeros_like(tx)
                    has_corr = True
                    for ic, c in enumerate(classes):
                        for il in range(len(le) - 1):
                            v = f[grid_name(r, sel, c, le[il], le[il + 1])].values(flow=True)   # (x+2, t+2)
                            px = v.sum(axis=1)
                            g[ic, il] = px[1:-1]
                            tx[ic, il] = v[1:-1, 1:-1]
                            cname = grid_name(r, sel, c, le[il], le[il + 1]) + "_corr"      # dijet only, from 11-12-26 on
                            if has_corr and cname in f:
                                txc[ic, il] = f[cname].values(flow=True)[1:-1, 1:-1]
                            else:
                                has_corr = False
                            fl += px[0] + px[-1]
                    d["grid"][(r, sel)] = g
                    d["xflow"][(r, sel)] = fl
                    d["tx"][(r, sel)] = tx
                    if has_corr:
                        d["tx_corr"][(r, sel)] = txc
                    d["status"][(r, sel)] = f[f"hLeadTimingStatus_{r}_{sel}"].values().astype(np.float64)
                    lname = f"h2LeadPtCent_{r}_{sel}"                       # from production 11-11-26 on
                    if lname in f:
                        d["leadpt"][(r, sel)] = f[lname].values(flow=True).astype(np.float64)
                for s in ("dijet", "all"):
                    d["jetpt"][(r, s)] = f[f"h2JetPtCent_{r}_{s}"].values(flow=True).astype(np.float64)
            # corrected time vs raw pT per 20% class, with flow: (class, pT + 2, t + 2)
            for key in layout.get("tpt_edges", {}):
                kind, r, tag = key
                names = [f"{TPT_PREFIX[kind]}_{r}_{tag}_cent_{lo}_{hi}" for lo, hi in CLASSES20]
                if all(n in f for n in names):
                    d["tpt"][key] = np.stack([f[n].values(flow=True) for n in names]).astype(np.float32)
    except (KeyError, OSError) as e:
        print(f"Warning: run {run}: {e}, skipping")
        return run, None
    return run, d


def cache_key(tasks, lumi_paths):
    h = hashlib.sha1(f"v{CACHE_VERSION}".encode())
    for run, path in tasks:
        st = path.stat()
        h.update(f"{run}|{path}|{st.st_size}|{st.st_mtime_ns}\n".encode())
    for path in lumi_paths:
        st = path.stat()
        h.update(f"L|{path}|{st.st_size}|{st.st_mtime_ns}\n".encode())
    return h.hexdigest()[:16]


def load_inputs(tasks, lumi_files, n_jobs):
    layout = read_layout(tasks[0][1])
    runs, grids, xflow, jetpt, status, leadpt, tx, tx_corr = [], {}, {}, {}, {}, {}, {}, {}
    ex = concurrent.futures.ProcessPoolExecutor(max_workers=n_jobs) if n_jobs > 1 else None
    n_corr = 0                                                          # runs with the corrected dijet timing hists
    tpt, n_tpt = {}, {}                                                 # corrected t vs pT, and the runs that have it
    jobs = [(r, p, layout) for r, p in tasks]
    for run, d in (ex.map(load_run, jobs, chunksize=8) if ex else map(load_run, jobs)):
        if d is None:
            continue
        runs.append(run)
        for k, v in d["grid"].items():
            grids.setdefault(k, []).append(v)
        for k, v in d["xflow"].items():
            xflow[k] = xflow.get(k, 0) + v
        for k, v in d["jetpt"].items():
            jetpt[k] = jetpt.get(k, 0) + v
        for k, v in d["status"].items():
            status[k] = status.get(k, 0) + v
        for k, v in d["leadpt"].items():
            leadpt[k] = leadpt.get(k, 0) + v
        for k, v in d["tx"].items():                               # summed over runs: (cent, pT_lead, x, t)
            tx[k] = tx.get(k, 0) + v.astype(np.float64)
        for k, v in d["tx_corr"].items():
            tx_corr[k] = tx_corr.get(k, 0) + v.astype(np.float64)
        n_corr = n_corr + bool(d["tx_corr"])
        for k, v in d["tpt"].items():
            tpt[k] = tpt.get(k, 0) + v.astype(np.float64)
            n_tpt[k] = n_tpt.get(k, 0) + 1
    if ex:
        ex.shutdown()
    order = np.argsort(runs)
    runs = [runs[i] for i in order]
    grids = {k: np.stack(v)[order] for k, v in grids.items()}          # (run, cent, pT_lead, x)

    lumi = None
    missing = [r for r in runs if r not in lumi_files or not lumi_files[r].exists()]
    if missing:
        print(f"Warning: no luminosity file for {len(missing)} run(s) (e.g. {missing[:5]}), omitting the luminosity label")
    else:
        lumis = [load_luminosity(lumi_files[r]) for r in runs]
        lumi = None if any(l is None for l in lumis) else float(sum(lumis))
    return {"runs": runs, "grids": grids, "xflow": xflow, "jetpt": jetpt, "status": status, "leadpt": leadpt,
            "tx": tx, "tx_corr": tx_corr if n_corr == len(runs) else {},
            "tpt": {k: v for k, v in tpt.items() if n_tpt[k] == len(runs)}, "layout": layout, "lumi": lumi}


# Helpers
# ---------------------------------------------------------------------------------------------------------------------

def run_groups(runs, blocks):
    """Index of the run group of every run, and the group labels (first-last run)."""
    edges = sorted(blocks)
    gi = np.array([bisect.bisect_right(edges, r) for r in runs])
    labels = []
    for g in range(len(edges) + 1):
        rr = [r for r, k in zip(runs, gi) if k == g]
        labels.append(f"{min(rr)}–{max(rr)}" if rr else "")
    return gi, labels


def moments(n, xc):
    """N, <x>, error of <x> (RMS / sqrt N) and RMS of a binned distribution."""
    tot = n.sum()
    if tot <= 0:
        return 0.0, np.nan, np.nan, np.nan
    m = (n * xc).sum() / tot
    rms = np.sqrt(max((n * xc ** 2).sum() / tot - m ** 2, 0))
    return tot, m, rms / np.sqrt(tot), rms


def rebin(a, k):
    """Sum groups of k bins along the last axis."""
    return a.reshape(*a.shape[:-1], a.shape[-1] // k, k).sum(axis=-1)


def density(n, edges):
    tot = n.sum()
    if tot <= 0:
        return np.full(len(n), np.nan), np.full(len(n), np.nan)
    w = np.diff(edges)
    return n / tot / w, np.sqrt(n) / tot / w


def ratio(a, ea, b, eb):
    with np.errstate(invalid="ignore", divide="ignore"):
        r = a / b
        e = np.abs(r) * np.sqrt((ea / a) ** 2 + (eb / b) ** 2)
    return r, e


def gap_steps(ax, y, edges, **kw):
    """Step line that leaves empty bins blank."""
    y = np.where(np.isfinite(y) & (y > 0), y, np.nan)
    ax.stairs(y, edges, **kw)


def header(ax, runs, r_tag, lumi, extra=""):
    add_header(ax, runs, RADII[r_tag]["r"], lumi, extra=extra)


# P1: leading-jet pT, inclusive vs dijet
# ---------------------------------------------------------------------------------------------------------------------

def plot_p1(inputs, r_tag, cut, runs, lumi, out_dir, save_pdf=False):
    """Leading-jet raw pT, inclusive against dijet, 0-60%, and the dijet fraction of the inclusive leading jets."""
    if (r_tag, "dijet") not in inputs.get("leadpt", {}):
        print(f"Warning: no h2LeadPtCent_{r_tag}_<sel> in the production (needs 11-11-26 or later), skipping P1")
        return []
    hep.style.use("ATLAS")
    pe = inputs["layout"]["pt_edges"]
    edges = np.array(PT_PLOT_EDGES, dtype=float)
    idx = np.searchsorted(pe, edges[:-1])
    w = np.diff(edges)
    cen = 0.5 * (edges[1:] + edges[:-1])
    n1 = {s: inputs["leadpt"][(r_tag, s)][1:-1, 1:-1].sum(axis=1) for s in SELS}       # 1 GeV bins, 0-60%
    over = {s: inputs["leadpt"][(r_tag, s)][-1, :].sum() for s in SELS}               # above 300 GeV
    n = {s: np.add.reduceat(n1[s], idx) for s in SELS}

    fig = plt.figure(figsize=(11, 10))
    gs = fig.add_gridspec(2, 1, height_ratios=(3, 1), hspace=0.05)
    ax, axr = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
    for s, col, lab in (("incl", "black", "Inclusive"), ("dijet", "#e41a1c", "Dijet")):
        gap_steps(ax, n[s] / w, edges, color=col, linewidth=2.5, label=f"{lab}: {fmt_n(n1[s].sum() + over[s])} leading jets")
    # dijet / inclusive = the fraction of inclusive leading jets with a back-to-back partner; binomial errors. Bins with
    # no dijet leading jets are left empty, so the panel ends where the dijet sample does
    with np.errstate(invalid="ignore", divide="ignore"):
        frac = n["dijet"] / n["incl"]
        efrac = np.sqrt(frac * (1 - frac) / n["incl"])
    ok = n["incl"] > 0
    has = ok & (n["dijet"] > 0)
    axr.errorbar(cen[has], frac[has], yerr=efrac[has], xerr=w[has] / 2, fmt="o", color="#e41a1c", markersize=4)

    rows, txt = [], []
    for t in TAIL_THRESHOLDS:
        ni, nd = n1["incl"][int(t):].sum() + over["incl"], n1["dijet"][int(t):].sum() + over["dijet"]
        rows.append({"radius": r_tag, "pt_lead_min": t, "n_incl": f"{ni:.0f}", "n_dijet": f"{nd:.0f}",
                     "dijet_fraction": f"{nd / ni:.4g}" if ni else ""})
        txt.append(rf"$\geq {t}$ GeV: {nd:.0f} / {ni:.0f}")
    hi = {s: pe[np.nonzero(n1[s])[0][-1] + 1] if n1[s].sum() else np.nan for s in SELS}
    rows.append({"radius": r_tag, "pt_lead_min": "highest", "n_incl": f"{hi['incl']:.0f}", "n_dijet": f"{hi['dijet']:.0f}", "dijet_fraction": ""})

    ax.set_yscale("log")
    ymax = max(np.nanmax(n[s] / w) for s in SELS)
    ymin = 0.5 * min(np.nanmin(np.where(n[s] > 0, n[s] / w, np.nan)) for s in SELS)
    ax.set_ylim(ymin, ymax * 3e3)
    ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=20))
    ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10), numticks=50))
    ax.yaxis.set_minor_formatter(NullFormatter())
    xlim = (10, edges[-1])
    for a in (ax, axr):
        a.set_xlim(*xlim)
        a.axvline(cut, color="gray", linestyle=":")
    ax.tick_params(labelbottom=False)
    ax.set_ylabel(r"Leading jets / GeV")
    lines = [r"$\bf{Leading\ jet,\ 0\text{–}60\%}$".replace(r"\text{–}", "-"),
             rf"Inclusive: raw $p_{{T,\mathrm{{lead}}}} \geq {cut:g}$ GeV",
             rf"Dijet: $\Delta\phi > 7\pi/8$, raw $p_{{T,\mathrm{{sub}}}} \geq {cut:g}$ GeV",
             r"$|z| < 10$ cm & MB, Good Calo-Cent, Iterative No Flow",
             rf"Jets: $|\eta| < {RADII[r_tag]['eta_max']:g}$, Energy > 0, raw $p_{{T}} \geq {JET_PT_MIN}$ GeV"]
    t = ax.text(0.97, 0.96, "\n".join(lines), transform=ax.transAxes, ha="right", va="top", fontsize=18, linespacing=1.3)
    leg = ax.legend(loc="upper right", bbox_to_anchor=(0.985, axes_bottom(fig, ax, t) - 0.01), fontsize=18, frameon=False)
    ax.text(0.97, axes_bottom(fig, ax, leg) - 0.02,
            "Dijet / inclusive leading jets\n" + "\n".join(txt) + f"\nhighest: {hi['dijet']:.0f} / {hi['incl']:.0f} GeV",
            transform=ax.transAxes, ha="right", va="top", fontsize=17, linespacing=1.3, color="0.25")
    header(ax, runs, r_tag, lumi)
    top = np.nanmax(np.where(has, frac + efrac, np.nan)) if has.any() else 1
    axr.set_ylim(0, min(1.05, 1.25 * top))
    axr.set_xlabel(r"Leading jet raw $p_{T}$ [GeV]")
    axr.set_ylabel("Dijet / Inclusive", fontsize=15)
    fig.subplots_adjust(left=0.12, right=0.96, bottom=0.09, top=0.95)
    save_fig(fig, out_dir / f"P1_leadpt_{r_tag}.png", save_pdf)
    plt.close(fig)
    return rows


# T1: leading-jet time vs OHCal fraction
# ---------------------------------------------------------------------------------------------------------------------

T_LABEL = r"Leading jet $t - t_{0}^{\mathrm{MBD}}$ [ns]"
PROFILE_MIN_N = 10                                         # x bins with fewer jets are left out of the profile
T1_YLIM = (-12, 12)                                        # [ns]; jets outside are counted on the plot
T1_FIT_DEG = 4                                             # quartic fitted to the profile (pol2 leaves an S-shaped residual)


T_DISPLAY_BIN = 0.5                                        # [ns] t bins of the 2D plots (the hists have 0.1 ns bins within +-5 ns from 11-12-26-v2 on)


def merge_t(h, te, width=T_DISPLAY_BIN):
    """h (..., t) on the edges te, merged to uniform bins of width (every new edge must be an edge of te; the fine core
    bins of the variable time axis are, so this gives back the 0.5 ns binning). Returns (merged h, new edges)."""
    new = np.arange(te[0], te[-1] + width / 2, width)
    idx = np.searchsorted(te, new - 1e-9)
    if len(idx) < 2 or not np.allclose(te[np.minimum(idx, len(te) - 1)], new, atol=1e-6):
        raise ValueError(f"the time edges cannot be merged to {width} ns bins")
    return np.add.reduceat(h, idx[:-1], axis=-1), new


def time_profile(inputs, r_tag, sel, x_rebin, cents=None, key="tx"):
    """t - t0 vs x summed over runs, pT_lead and the grid classes cents (all if None), with x rebinned, and its profile:
    <t - t0> per x bin from the bin centres of the t axis, the standard deviation, and the mask of bins with N >= PROFILE_MIN_N."""
    xe = inputs["layout"]["x_edges"][::x_rebin]
    te = inputs["layout"]["t_edges"]
    tc = 0.5 * (te[1:] + te[:-1])
    g = inputs[key][(r_tag, sel)]                                        # (cent, pT_lead, x, t), fine x bins
    h = (g if cents is None else g[list(cents)]).sum(axis=(0, 1))
    h = h.reshape(len(xe) - 1, x_rebin, len(tc)).sum(axis=1)
    n = h.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = (h * tc).sum(axis=1) / n
        rms = np.sqrt(np.maximum((h * tc ** 2).sum(axis=1) / n - mean ** 2, 0))
    return {"xe": xe, "te": te, "tc": tc, "xc": 0.5 * (xe[1:] + xe[:-1]), "h": h, "n": n, "mean": mean, "rms": rms,
            "err": rms / np.sqrt(n), "ok": n >= PROFILE_MIN_N}


def fit_profile(P):
    """Quartic fitted to the profile points (weights 1 / error^2): coefficients (highest power first), chi2, ndf and the
    fitted x range (first and last bin centre); outside it the correction uses f at the nearest edge."""
    ok = P["ok"]
    xf, yf, ef = P["xc"][ok], P["mean"][ok], P["err"][ok]
    c = np.polyfit(xf, yf, T1_FIT_DEG, w=1 / ef)
    chi2, ndf = (((yf - np.polyval(c, xf)) / ef) ** 2).sum(), len(xf) - T1_FIT_DEG - 1
    return {"c": c, "chi2": chi2, "ndf": ndf, "x_lo": xf[0], "x_hi": xf[-1]}


def fit_eval(fit, x):
    return np.polyval(fit["c"], np.clip(x, fit["x_lo"], fit["x_hi"]))


def sel_lines(r_tag, cut, sel):
    return [rf"$\Delta\phi > 7\pi/8$, raw $p_{{T,\mathrm{{sub}}}} \geq {cut:g}$ GeV" if sel == "dijet"
            else rf"Leading jet raw $p_{{T}} \geq {cut:g}$ GeV",
            r"$|z| < 10$ cm & MB, Good Calo-Cent, Iterative No Flow",
            rf"Jets: $|\eta| < {RADII[r_tag]['eta_max']:g}$, Energy > 0, raw $p_{{T}} \geq {JET_PT_MIN}$ GeV"]


def plot_t1(inputs, r_tag, cut, x_rebin, runs, lumi, out_dir, sel="dijet", save_pdf=False):
    """t - t0 vs x of the leading jet, 0-60% and all pT_lead, with the profile (<t - t0> per x bin, from the bin centres
    of the t axis; error sigma / sqrt N) and the quartic fitted to it. Returns the fit."""
    hep.style.use("ATLAS")
    P = time_profile(inputs, r_tag, sel, x_rebin)
    xe, te, tc, xc, h, n, mean, rms, ok = (P[k] for k in ("xe", "te", "tc", "xc", "h", "n", "mean", "rms", "ok"))

    fig, ax = plt.subplots(figsize=(10.5, 8.5))
    hd, ted = merge_t(h, te)                                             # 2D shown in 0.5 ns bins; the profile uses the hist bins
    z = np.where(hd > 0, hd, np.nan)
    pcm = ax.pcolormesh(xe, ted, z.T, cmap="viridis", norm=LogNorm(vmin=1, vmax=np.nanmax(z)), rasterized=True)
    fig.colorbar(pcm, ax=ax, pad=0.02).set_label("Leading jets")
    ax.errorbar(xc[ok], mean[ok], yerr=rms[ok] / np.sqrt(n[ok]), fmt="o", color="black", markersize=5, capsize=0,
                linewidth=1.4, label=r"Profile: $\langle t - t_{0} \rangle$", zorder=3)
    ax.errorbar(xc[ok], mean[ok], yerr=rms[ok], fmt="none", color="black", alpha=0.35, linewidth=4, capsize=0,
                label=r"$\sigma$ (std. dev.)", zorder=2)
    # quartic fitted to the profile points, drawn over the fitted x range
    fit = fit_profile(P)
    print(f"  T1 {r_tag} {sel} quartic: chi2/ndf = {fit['chi2']:.1f}/{fit['ndf']}, x in [{fit['x_lo']:.3f}, {fit['x_hi']:.3f}], "
          f"coefficients (highest first) {np.round(fit['c'], 4)}")
    xs = np.linspace(fit["x_lo"], fit["x_hi"], 300)
    ax.plot(xs, np.polyval(fit["c"], xs), "-", color="#e41a1c", linewidth=2.2, zorder=4,
            label=rf"Quartic fit: $\chi^2$/ndf = {fit['chi2']:.0f}/{fit['ndf']}")
    N = h.sum()
    n_out = h[:, (tc < T1_YLIM[0]) | (tc > T1_YLIM[1])].sum()
    print(f"  T1 {r_tag} {sel} profile <t - t0> [ns]: " + ", ".join(f"x={x:.3f}: {m:.2f}" for x, m in zip(xc[ok], mean[ok])))
    lines = ([rf"$\bf{{{SEL_LABEL[sel]},\ 0\text{{–}}60\%}}$".replace(r"\text{–}", "-")] + sel_lines(r_tag, cut, sel) +
             [             rf"$N$ = {fmt_n(N)} leading jets" + (f", {n_out:.0f} outside the y range" if n_out else "")])
    box = dict(facecolor="white", alpha=0.8, edgecolor="none", pad=3)
    ax.text(0.97, 0.97, "\n".join(lines), transform=ax.transAxes, ha="right", va="top", fontsize=13, linespacing=1.3,
            bbox=box)
    leg = ax.legend(loc="lower left", fontsize=13, frameon=True, framealpha=0.8, facecolor="0.75", edgecolor="none")
    ax.set_xlim(xe[0], xe[-1])
    ax.set_ylim(*T1_YLIM)
    ax.set_xlabel(X_LABEL)
    ax.set_ylabel(T_LABEL)
    header(ax, runs, r_tag, lumi)
    fig.subplots_adjust(left=0.12, right=0.96, bottom=0.11, top=0.94)
    save_fig(fig, out_dir / f"T1_time_vs_x_{r_tag}_{sel}.png", save_pdf)
    plt.close(fig)
    return fit


T2_COLORS = ("#1b9e77", "#d95f02", "#7570b3")             # 0-20, 20-40, 40-60 %
T2_ERR_RANGE = 0.25                                        # [ns] profile points with larger errors do not set the lower panel range


def plot_t2(inputs, r_tag, cut, x_rebin, fit, runs, lumi, out_dir, sel="dijet", save_pdf=False):
    """The profiles <t - t0> of the three 20% classes (all pT_lead) with the 0-60% quartic of T1, and below them each
    profile minus the fit in ns (f clamped to the fitted x range, as the correction will use it)."""
    hep.style.use("ATLAS")
    fig = plt.figure(figsize=(10.5, 9.5))
    gs = fig.add_gridspec(2, 1, height_ratios=(3, 1), hspace=0.05)
    ax, axr = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
    xs = np.linspace(fit["x_lo"], fit["x_hi"], 300)
    dmax = 0.0                                                          # largest |profile - fit| + error, for the lower panel
    ax.plot(xs, np.polyval(fit["c"], xs), "-", color="black", linewidth=2, zorder=2,
            label=rf"Quartic fit, 0–60%: $\chi^2$/ndf = {fit['chi2']:.0f}/{fit['ndf']}")
    for k, c in enumerate(CLASSES20):
        P = time_profile(inputs, r_tag, sel, x_rebin, cents=cent_idx(c, inputs["layout"]["cent_classes"]))
        ok = P["ok"]
        x, m, e = P["xc"][ok], P["mean"][ok], P["err"][ok]
        f = fit_eval(fit, x)
        chi2 = (((m - f) / e) ** 2).sum()
        dx = (k - 1) * 0.008                                              # offset so the points do not overlap
        ax.errorbar(x + dx, m, yerr=e, fmt="o", color=T2_COLORS[k], markersize=5, capsize=0, linewidth=1.3, zorder=3,
                    label=rf"{cent_label(c)}: $N$ = {fmt_n(P['n'].sum())}, $\chi^2$/$n$ to fit = {chi2:.0f}/{ok.sum()}")
        axr.errorbar(x + dx, m - f, yerr=e, fmt="o", color=T2_COLORS[k], markersize=4, capsize=0, linewidth=1.2)
        well = e < T2_ERR_RANGE                                         # sparse points do not set the range
        if well.any():
            dmax = max(dmax, np.max(np.abs(m - f)[well] + e[well]))
        print(f"  T2 {r_tag} {sel} {cent_label(c)}: profile - fit [ns] = " + ", ".join(f"x={a:.3f}: {b:+.3f}" for a, b in zip(x, m - f)))
    t = ax.text(0.97, 0.96, "\n".join([rf"$\bf{{{SEL_LABEL[sel]}}}$"] + sel_lines(r_tag, cut, sel)), transform=ax.transAxes,
                ha="right", va="top", fontsize=17, linespacing=1.3)
    ax.legend(loc="lower left", fontsize=17, frameon=False)
    ax.set_xlim(fit["x_lo"] - 0.1, fit["x_hi"] + 0.1)
    lo = np.polyval(fit["c"], fit["x_hi"])
    ax.set_ylim(lo - 1.6, 1.2)                                           # room for the legend below, the text above
    ax.tick_params(labelbottom=False)
    ax.set_ylabel(r"$\langle t - t_{0}^{\mathrm{MBD}} \rangle$ [ns]")
    axr.axhline(0, color="gray", linestyle="--")
    axr.set_xlim(*ax.get_xlim())
    lim = np.ceil(1.15 * max(dmax, 0.2) * 10) / 10                      # symmetric, rounded up to 0.1 ns
    axr.set_ylim(-lim, lim)
    axr.set_xlabel(X_LABEL)
    axr.set_ylabel("Profile − fit [ns]", fontsize=15)
    header(ax, runs, r_tag, lumi)
    fig.subplots_adjust(left=0.12, right=0.96, bottom=0.09, top=0.95)
    save_fig(fig, out_dir / f"T2_profiles_cent20_{r_tag}_{sel}.png", save_pdf)
    plt.close(fig)


def plot_t3(inputs, r_tag, cut, x_rebin, fit, runs, lumi, out_dir, sel="dijet", save_pdf=False):
    """Closure of the timing correction, 1 x 2: t - t0 vs x (left, with the quartic of T1) and the corrected
    t - t0 - f(x) (right, with 0), dijet, 0-60%, all pT_lead, each with its profile and standard deviation; one colour
    scale for both."""
    if (r_tag, sel) not in inputs.get("tx_corr", {}):
        print(f"Warning: no corrected {sel} timing hists of {r_tag} in every run (needs a production with a timing correction "
              "file, 11-12-26 on), skipping T3")
        return
    hep.style.use("ATLAS")
    P = {k: time_profile(inputs, r_tag, sel, x_rebin, key=k) for k in ("tx", "tx_corr")}
    D = {k: merge_t(P[k]["h"], P[k]["te"]) for k in P}                  # 2D shown in 0.5 ns bins
    vmax = max(np.nanmax(D[k][0]) for k in P)
    fig = plt.figure(figsize=(19, 8.5))
    # columns: left panel, a spacer for the right panel's y title, right panel, colour bar (close to the right panel)
    gs = fig.add_gridspec(1, 4, width_ratios=(1, 0.17, 1, 0.035), wspace=0.025)
    axs = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 2])]
    cax = fig.add_subplot(gs[0, 3])
    box = dict(facecolor="white", alpha=0.8, edgecolor="none", pad=3)
    for ax, k in zip(axs, ("tx", "tx_corr")):
        Q = P[k]
        hd, ted = D[k]
        z = np.where(hd > 0, hd, np.nan)
        pcm = ax.pcolormesh(Q["xe"], ted, z.T, cmap="viridis", norm=LogNorm(vmin=1, vmax=vmax), rasterized=True)
        ok = Q["ok"]
        ax.errorbar(Q["xc"][ok], Q["mean"][ok], yerr=Q["rms"][ok], fmt="none", color="black", alpha=0.35, linewidth=4,
                    capsize=0, label=r"$\sigma$ (std. dev.)", zorder=2)
        ax.errorbar(Q["xc"][ok], Q["mean"][ok], yerr=Q["err"][ok], fmt="o", color="black", markersize=5, capsize=0,
                    linewidth=1.4, label="Profile", zorder=3)
        if k == "tx":
            xs = np.linspace(fit["x_lo"], fit["x_hi"], 300)
            ax.plot(xs, np.polyval(fit["c"], xs), "-", color="#e41a1c", linewidth=2.2, zorder=4, label=r"Quartic fit $f(x)$")
            title = r"$\bf{Uncorrected}$: $t - t_{0}$"
        else:
            ax.axhline(0, color="#e41a1c", linewidth=2.2, zorder=4, label="0")
            title = r"$\bf{Corrected}$: $t - t_{0} - f(x)$"
            print(f"  T3 {r_tag} {sel} corrected profile [ns]: " + ", ".join(
                f"x={a:.3f}: {b:+.3f}" for a, b in zip(Q["xc"][ok], Q["mean"][ok])))
        N, n_out = Q["h"].sum(), Q["h"][:, (Q["tc"] < T1_YLIM[0]) | (Q["tc"] > T1_YLIM[1])].sum()
        lines = [title, rf"$N$ = {fmt_n(N)} leading jets" + (f", {n_out:.0f} outside the y range" if n_out else "")]
        ax.text(0.97, 0.97, "\n".join(lines), transform=ax.transAxes, ha="right", va="top", fontsize=16, linespacing=1.3,
                bbox=box)
        ax.legend(loc="lower left", fontsize=15, frameon=True, framealpha=0.8, facecolor="0.75", edgecolor="none")
        ax.set_xlim(Q["xe"][0], Q["xe"][-1])
        ax.set_ylim(*T1_YLIM)
        ax.set_xlabel(X_LABEL)
    axs[0].set_ylabel(T_LABEL)
    axs[1].set_ylabel(r"Leading jet $t - t_{0}^{\mathrm{MBD}} - f(x)$ [ns]")
    # the selection, bottom right of the corrected panel (empty: the corrected band is around 0)
    axs[1].text(0.97, 0.03, "\n".join([rf"$\bf{{{SEL_LABEL[sel]},\ 0\text{{–}}60\%}}$".replace(r"\text{–}", "-")]
                                      + sel_lines(r_tag, cut, sel)), transform=axs[1].transAxes, ha="right", va="bottom",
                fontsize=13, linespacing=1.3, color="0.2")
    fig.colorbar(pcm, cax=cax).set_label("Leading jets")
    header(axs[1], runs, r_tag, lumi)
    fig.subplots_adjust(left=0.06, right=0.94, bottom=0.11, top=0.93)
    save_fig(fig, out_dir / f"T3_closure_{r_tag}_{sel}.png", save_pdf)
    plt.close(fig)


def plot_t45(inputs, r_tag, cut, kind, tag, runs, lumi, out_dir, x_max=None, suffix="", save_pdf=False):
    """Corrected t - t0 - f(x) vs raw pT, 0-60% (the three 20% classes summed), with the profile per pT bin (N >= 10)
    and its standard deviation. kind "lead": the leading jet (T4; tag dijet or lead = every selected event's leading
    jet); kind "jet": every jet passing the cuts (T5; tag dijet = in dijet events, all = in all selected events).
    x_max: upper end of the pT axis (no rebinning; default the hist's top edge); suffix: added to the file name."""
    key = (kind, r_tag, tag)
    if key not in inputs.get("tpt", {}):
        print(f"Warning: no {TPT_PREFIX[kind]}_{r_tag}_{tag} hists in every run (production 11-12-26-v2 on), skipping")
        return
    hep.style.use("ATLAS")
    xe, te = inputs["layout"]["tpt_edges"][key]
    hf = inputs["tpt"][key].sum(axis=0)                                  # (pT + 2, t + 2), with flow
    h = hf[1:-1, 1:-1]
    tc = 0.5 * (te[1:] + te[:-1])
    xc = 0.5 * (xe[1:] + xe[:-1])
    n = h.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = (h * tc).sum(axis=1) / n
        rms = np.sqrt(np.maximum((h * tc ** 2).sum(axis=1) / n - mean ** 2, 0))
    ok = n >= PROFILE_MIN_N
    hi = xe[-1] if x_max is None else x_max
    n_tot = hf.sum()
    n_t_out = hf[:, 0].sum() + hf[:, -1].sum()                          # outside +-75 ns (every pT)
    n_above = hf[-1, :].sum() + (hf[1:-1][xe[:-1] >= hi].sum() if x_max is not None else 0)

    fig, ax = plt.subplots(figsize=(10.5, 8.5))
    hd, ted = merge_t(h, te)                                             # 2D shown in 0.5 ns bins; the profile uses the hist bins
    z = np.where(hd > 0, hd, np.nan)
    pcm = ax.pcolormesh(xe, ted, z.T, cmap="viridis", norm=LogNorm(vmin=1, vmax=np.nanmax(z)), rasterized=True)
    fig.colorbar(pcm, ax=ax, pad=0.02).set_label("Leading jets" if kind == "lead" else "Jets")
    ax.axhline(0, color="#e41a1c", linewidth=1.8, zorder=2, label="0")
    ax.errorbar(xc[ok], mean[ok], yerr=rms[ok], fmt="none", color="black", alpha=0.35, linewidth=3, capsize=0,
                label=r"$\sigma$ (std. dev.)", zorder=3)
    ax.errorbar(xc[ok], mean[ok], yerr=rms[ok] / np.sqrt(n[ok]), fmt="o", color="black", markersize=3, capsize=0,
                linewidth=1.2, label=rf"Profile ($N \geq {PROFILE_MIN_N}$ per bin)", zorder=4)
    title = {("lead", "dijet"): "Dijet, leading jet", ("lead", "lead"): "Leading jet, every selected event",
             ("jet", "dijet"): "Every jet in dijet events", ("jet", "all"): "Every jet, every selected event"}[(kind, tag)]
    common = sel_lines(r_tag, cut, "dijet")
    bold = title.replace(" ", r"\ ") + r",\ 0-60\%"
    lines = ([rf"$\bf{{{bold}}}$"]
             + (common if tag == "dijet" else common[1:])
             + [rf"$N$ = {fmt_n(n_tot)}" + (rf", {n_t_out:.0f} outside $\pm${te[-1]:g} ns" if n_t_out else "")
                + (f", {n_above:.0f} above {hi:g} GeV" if n_above else "")])
    ax.text(0.97, 0.97, "\n".join(lines), transform=ax.transAxes, ha="right", va="top", fontsize=13, linespacing=1.3,
            bbox=dict(facecolor="white", alpha=0.8, edgecolor="none", pad=3))
    ax.legend(loc="lower right", fontsize=13, frameon=True, framealpha=0.8, facecolor="0.75", edgecolor="none")
    ax.set_xlim(xe[0], hi)
    ax.set_ylim(te[0], te[-1])
    ax.set_xlabel(("Leading jet" if kind == "lead" else "Jet") + r" raw $p_{T}$ [GeV]")
    ax.set_ylabel(r"Jet $t - t_{0}^{\mathrm{MBD}} - f(x)$ [ns]")
    header(ax, runs, r_tag, lumi)
    fig.subplots_adjust(left=0.12, right=0.96, bottom=0.11, top=0.94)
    stem = f"{'T4_leadpt' if kind == 'lead' else 'T5_jetpt'}_{r_tag}_{tag}{suffix}"
    save_fig(fig, out_dir / f"{stem}.png", save_pdf)
    plt.close(fig)
    print(f"  {stem}: N {n_tot:.0f}, outside +-{te[-1]:g} ns {n_t_out:.0f}, above {hi:g} GeV {n_above:.0f}")


PROJ_ZOOM = 10                                             # [ns] range of the linear (zoomed) projections
PROJ_FIT_NSIG = 2                                          # Gaussian fit window: +-2 sigma around the fitted mean
PROJ_FAR = 10                                              # [ns] |t_corr| above this counts as far out of time


def fit_gauss_core(n, edges, nsig=PROJ_FIT_NSIG, max_iter=20):
    """Gaussian fitted to the core of a binned distribution: start from the mean and RMS, fit the bins whose centres are
    within +-nsig sigma of the mean, move the window to the new mean and sigma, until both change by less than 1e-4 ns.
    The model is the Gaussian integrated over each bin (the bins are not small against sigma); errors sqrt(N) (empty
    bins: 1). Returns the parameters, errors, chi2, ndf, window and the iterations used, or None."""
    c = 0.5 * (edges[1:] + edges[:-1])
    tot = n.sum()
    if tot <= 0:
        return None
    mu = (n * c).sum() / tot
    sig = np.sqrt(max((n * c ** 2).sum() / tot - mu ** 2, 1e-6))

    def model(lo_hi, a, m, s):
        lo, hi = lo_hi
        return a * 0.5 * (erf((hi - m) / (np.sqrt(2) * s)) - erf((lo - m) / (np.sqrt(2) * s)))

    res = None
    for it in range(1, max_iter + 1):
        sel = np.abs(c - mu) <= nsig * sig
        if sel.sum() < 4:
            return None
        x = (edges[:-1][sel], edges[1:][sel])
        y = n[sel]
        e = np.sqrt(np.maximum(y, 1))
        try:
            p, cov = curve_fit(model, x, y, p0=(y.sum(), mu, sig), sigma=e, absolute_sigma=True)
        except RuntimeError:
            return None
        a, m, s = p[0], p[1], abs(p[2])
        chi2 = (((y - model(x, a, m, s)) / e) ** 2).sum()
        res = {"a": a, "mu": m, "sigma": s, "err": np.sqrt(np.diag(cov)), "chi2": chi2, "ndf": int(sel.sum()) - 3,
               "window": (mu - nsig * sig, mu + nsig * sig), "iter": it}
        done = abs(m - mu) < 1e-4 and abs(s - sig) < 1e-4
        mu, sig = m, s
        if done:
            break
    return res


def plot_t5_proj(inputs, r_tag, cut, runs, lumi, out_dir, zoom=None, ref_fit=None, save_pdf=False):
    """The T5 hist of every jet in every selected event (h2JetTimeCorrPt_<r>_all) projected on t - t0 - f(x), every pT,
    0-60%. zoom None: log scale over the whole axis; zoom z: linear over +-z ns. ref_fit: the dijet leading-jet
    full-range fit (plot_lead_dijet_proj), whose mu +- 3 sigma is marked and counted (file suffix _leaddijet3sigma)."""
    key = ("jet", r_tag, "all")
    if key not in inputs.get("tpt", {}):
        return
    hep.style.use("ATLAS")
    xe, te = inputs["layout"]["tpt_edges"][key]
    hf = inputs["tpt"][key].sum(axis=0).sum(axis=0)                     # t with flow, every pT (flow included)
    n = hf[1:-1]
    tc = 0.5 * (te[1:] + te[:-1])
    dens = n / np.diff(te)                                               # jets per ns (the bins are not all equal)
    tot, n_out = hf.sum(), hf[0] + hf[-1]
    mean = (n * tc).sum() / n.sum()
    rms = np.sqrt((n * tc ** 2).sum() / n.sum() - mean ** 2)
    far = n[np.abs(tc) > PROJ_FAR].sum() + n_out

    fig, ax = plt.subplots(figsize=(10.5, 8))
    ax.stairs(dens, te, color="black", linewidth=1.6, label="Jets")
    lines = [rf"$N$ = {fmt_n(tot)}" + (rf", {n_out:.0f} outside $\pm${te[-1]:g} ns" if n_out else ""),
             rf"Mean {mean:+.3f} ns, RMS {rms:.3f} ns (within $\pm${te[-1]:g} ns)",
             rf"$|t_{{\mathrm{{corr}}}}| > {PROJ_FAR}$ ns: {fmt_n(far)} ({far / tot:.2e})"]
    if ref_fit is not None:
        lo3, hi3 = ref_fit["mu"] - 3 * ref_fit["sigma"], ref_fit["mu"] + 3 * ref_fit["sigma"]
        for x3 in (lo3, hi3):
            ax.axvline(x3, color="#377eb8", linestyle=":", linewidth=2,
                       label=r"Dijet leading-jet $\mu \pm 3\sigma$" if x3 == lo3 else None)
        # jets outside the window: bins whose centre is outside, and the flow (the window edges fall inside 0.1 ns bins)
        out3 = n[(tc < lo3) | (tc > hi3)].sum() + n_out
        lines += [rf"Dijet leading-jet fit $\mu \pm 3\sigma$ (full $p_{{T}}$ range): [{lo3:+.2f}, {hi3:+.2f}] ns",
                  rf"Outside: {fmt_n(out3)} ({out3 / tot:.2e}); below {fmt_n(n[tc < lo3].sum() + hf[0])}, "
                  rf"above {fmt_n(n[tc > hi3].sum() + hf[-1])}"]
        if zoom is None:
            print(f"  T5proj {r_tag} all: outside the dijet leading-jet mu +- 3 sigma [{lo3:+.3f}, {hi3:+.3f}]: "
                  f"{out3:.0f} / {tot:.0f} ({out3 / tot:.4f})")
    head = [r"$\bf{Every\ jet,\ every\ selected\ event,\ 0-60\%}$"] + sel_lines(r_tag, cut, "dijet")[1:]
    ax.text(0.97, 0.97, "\n".join(head + lines), transform=ax.transAxes, ha="right", va="top", fontsize=12,
            linespacing=1.3, bbox=dict(facecolor="white", alpha=0.8, edgecolor="none", pad=3))
    ax.legend(loc="upper left", fontsize=13, frameon=False)
    if zoom is None:
        ax.set_yscale("log")
        ax.set_xlim(te[0], te[-1])
        ax.set_ylim(0.5, dens.max() * 3e4)                              # peak below the text
    else:
        ax.set_xlim(-zoom, zoom)
        ax.set_ylim(0, dens.max() * 1.6)
    ax.set_xlabel(r"Jet $t - t_{0}^{\mathrm{MBD}} - f(x)$ [ns]")
    ax.set_ylabel("Jets / ns")
    header(ax, runs, r_tag, lumi)
    fig.subplots_adjust(left=0.12, right=0.96, bottom=0.11, top=0.94)
    suffix = ("_leaddijet3sigma" if ref_fit is not None else "") + ("" if zoom is None else f"_zoom{zoom:g}")
    save_fig(fig, out_dir / f"T5proj_{r_tag}_all{suffix}.png", save_pdf)
    plt.close(fig)


# Projections of the dijet leading jets (the sample the correction was fitted to): full pT range, pT windows, 20% classes
LEAD_PT_WINDOWS = {"r02": ((14.25, 16), (16, 18), (18, 20), (20, 25), (25, 30), (30, 60)),
                   "r03": ((21.5, 24), (24, 27), (27, 30), (30, 35), (35, 60))}


def proj_stats(nf, te):
    """Counts, mean, RMS, |t| > PROJ_FAR and the Gaussian core fit of a projection nf (t with flow)."""
    n = nf[1:-1]
    tc = 0.5 * (te[1:] + te[:-1])
    st = {"n_flow": nf, "n": n, "tc": tc, "tot": nf.sum(), "out": nf[0] + nf[-1]}
    st["mean"] = (n * tc).sum() / n.sum() if n.sum() else np.nan
    st["rms"] = np.sqrt((n * tc ** 2).sum() / n.sum() - st["mean"] ** 2) if n.sum() else np.nan
    st["far"] = n[np.abs(tc) > PROJ_FAR].sum() + st["out"]
    st["fit"] = fit_gauss_core(n, te)
    if st["fit"] is not None:
        f = st["fit"]
        st["out3"] = n[np.abs(tc - f["mu"]) > 3 * f["sigma"]].sum() + st["out"]
    return st


def draw_proj(st, te, head, path, runs, r_tag, lumi, zoom=None, save_pdf=False):
    """One projection with its Gaussian core fit and mu +- 3 sigma: log over the whole axis (zoom None) or linear over
    +-zoom ns."""
    hep.style.use("ATLAS")
    n, tc, tot = st["n"], st["tc"], st["tot"]
    dens = n / np.diff(te)
    fig, ax = plt.subplots(figsize=(10.5, 8))
    ax.stairs(dens, te, color="black", linewidth=1.6, label="Leading jets")
    lines = [rf"$N$ = {fmt_n(tot)}" + (rf", {st['out']:.0f} outside $\pm${te[-1]:g} ns" if st["out"] else ""),
             rf"Mean {st['mean']:+.3f} ns, RMS {st['rms']:.3f} ns (within $\pm${te[-1]:g} ns)",
             rf"$|t_{{\mathrm{{corr}}}}| > {PROJ_FAR}$ ns: {st['far']:.0f} ({st['far'] / tot:.2e})"]
    f = st["fit"]
    if f is not None:
        lo, hi = f["window"]
        xs = np.linspace(-PROJ_ZOOM * 2, PROJ_ZOOM * 2, 2000)
        g = f["a"] / (np.sqrt(2 * np.pi) * f["sigma"]) * np.exp(-0.5 * ((xs - f["mu"]) / f["sigma"]) ** 2)
        inwin = (xs >= lo) & (xs <= hi)
        ax.plot(xs[inwin], g[inwin], "-", color="#e41a1c", linewidth=2.2, label=rf"Gaussian fit ($\pm{PROJ_FIT_NSIG}\sigma$)")
        ax.plot(xs, np.where(inwin, np.nan, g), "--", color="#e41a1c", linewidth=1.4, label="Fit, extrapolated")
        for sgn in (-1, 1):
            ax.axvline(f["mu"] + sgn * 3 * f["sigma"], color="#377eb8", linestyle=":", linewidth=2,
                       label=r"$\mu \pm 3\sigma$" if sgn < 0 else None)
        lines += [rf"Fit: $\mu$ = {f['mu']:+.3f} $\pm$ {f['err'][1]:.3f} ns, $\sigma$ = {f['sigma']:.3f} $\pm$ {f['err'][2]:.3f} ns",
                  rf"$\chi^2$/ndf = {f['chi2']:.1f}/{f['ndf']}, window [{lo:+.2f}, {hi:+.2f}] ns",
                  rf"Outside $\mu \pm 3\sigma$ [{f['mu'] - 3 * f['sigma']:+.2f}, {f['mu'] + 3 * f['sigma']:+.2f}] ns: "
                  rf"{st['out3']:.0f} ({st['out3'] / tot:.2e})"]
    else:
        lines.append("Gaussian fit failed")
    ax.text(0.97, 0.97, "\n".join(head + lines), transform=ax.transAxes, ha="right", va="top", fontsize=12,
            linespacing=1.3, bbox=dict(facecolor="white", alpha=0.8, edgecolor="none", pad=3), zorder=5)
    ax.legend(loc="upper left", fontsize=13, frameon=False)
    if zoom is None:
        ax.set_yscale("log")
        ax.set_xlim(te[0], te[-1])
        ax.set_ylim(0.5, dens.max() * 30)
    else:
        ax.set_xlim(-zoom, zoom)
        ax.set_ylim(0, dens.max() * 1.6)
    ax.set_xlabel(r"Leading jet $t - t_{0}^{\mathrm{MBD}} - f(x)$ [ns]")
    ax.set_ylabel("Leading jets / ns")
    header(ax, runs, r_tag, lumi)
    fig.subplots_adjust(left=0.12, right=0.96, bottom=0.11, top=0.94)
    save_fig(fig, path, save_pdf)
    plt.close(fig)


def draw_proj_panels(stats, labels, te, head, path, runs, r_tag, lumi, zoom=PROJ_ZOOM, save_pdf=False):
    """The projections of draw_proj (linear, +-zoom ns) one per panel, in 1 row (up to 3 slices) or 2, plus a slot with
    the selection text and the legend. Each panel: the slice, N, the Gaussian core fit and the fraction outside its
    mu +- 3 sigma."""
    hep.style.use("ATLAS")
    n_sl = len(stats)
    nrows = 1 if n_sl <= 3 else 2
    ncols = int(np.ceil((n_sl + 1) / nrows))
    fig, axs = plt.subplots(nrows, ncols, figsize=(5.6 * ncols, 5.6 * nrows + 0.8), squeeze=False)
    handles = None
    for k, ax in enumerate(axs.flat):
        if k >= n_sl:
            ax.axis("off")
            continue
        st = stats[k]
        dens = st["n"] / np.diff(te)
        ax.stairs(dens, te, color="black", linewidth=1.4, label="Leading jets")
        lines = [labels[k], rf"$N$ = {fmt_n(st['tot'])}"]
        f = st["fit"]
        if f is not None:
            lo, hi = f["window"]
            xs = np.linspace(-zoom, zoom, 2000)
            g = f["a"] / (np.sqrt(2 * np.pi) * f["sigma"]) * np.exp(-0.5 * ((xs - f["mu"]) / f["sigma"]) ** 2)
            inwin = (xs >= lo) & (xs <= hi)
            ax.plot(xs[inwin], g[inwin], "-", color="#e41a1c", linewidth=2, label=rf"Gaussian fit ($\pm{PROJ_FIT_NSIG}\sigma$)")
            ax.plot(xs, np.where(inwin, np.nan, g), "--", color="#e41a1c", linewidth=1.3, label="Fit, extrapolated")
            for sgn in (-1, 1):
                ax.axvline(f["mu"] + sgn * 3 * f["sigma"], color="#377eb8", linestyle=":", linewidth=2,
                           label=r"$\mu \pm 3\sigma$" if sgn < 0 else None)
            lines += [rf"$\mu$ = {f['mu']:+.3f} $\pm$ {f['err'][1]:.3f} ns",
                      rf"$\sigma$ = {f['sigma']:.3f} $\pm$ {f['err'][2]:.3f} ns",
                      rf"$\chi^2$/ndf = {f['chi2']:.1f}/{f['ndf']}",
                      rf"Outside $\mu \pm 3\sigma$: {st['out3'] / st['tot']:.3f}"]
            if handles is None:
                handles = ax.get_legend_handles_labels()
        else:
            lines.append("Gaussian fit failed")
        ax.text(0.04, 0.96, "\n".join(lines), transform=ax.transAxes, ha="left", va="top", fontsize=13, linespacing=1.3,
                bbox=dict(facecolor="white", alpha=0.8, edgecolor="none", pad=2), zorder=5)
        ax.set_xlim(-zoom, zoom)
        ax.set_ylim(0, dens.max() * 1.75)
        ax.ticklabel_format(axis="y", style="plain")                     # no x10^n label under the header
        if k % ncols == 0:
            ax.set_ylabel("Leading jets / ns")
        ax.set_xlabel(r"Leading jet $t - t_{0}^{\mathrm{MBD}} - f(x)$ [ns]", fontsize=15)
    header(axs[0, -1] if n_sl >= ncols else axs[0, n_sl - 1], runs, r_tag, lumi)
    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.1 if nrows == 2 else 0.17, top=0.93 if nrows == 2 else 0.88,
                        wspace=0.28, hspace=0.32)
    slot = axs.flat[n_sl]
    if handles is not None:
        slot.legend(*handles, loc="lower left", fontsize=15, frameon=False, borderaxespad=0)
    fit_text_box(fig, slot, head, fs_max=20, top=0.95, bottom=0.42 if handles is not None else 0.0)
    save_fig(fig, path, save_pdf)
    plt.close(fig)


def plot_fit_summary(rows, ref, xlabel, path, runs, r_tag, lumi, head, save_pdf=False):
    """mu and sigma of the Gaussian core fits vs the slice variable (rows: dicts with lo, hi, fit), with the full-range
    fit (ref) as a line and band; below, the mu +- 3 sigma window of each slice (a box over the slice, marker at mu) against
    the full-range window (dashed lines)."""
    hep.style.use("ATLAS")
    fig = plt.figure(figsize=(10.5, 11))
    gs = fig.add_gridspec(3, 1, height_ratios=(1, 1, 1), hspace=0.06)
    axs = [fig.add_subplot(gs[i]) for i in range(3)]
    ok = [r for r in rows if r["fit"] is not None]
    xc = np.array([0.5 * (r["lo"] + r["hi"]) for r in ok])
    xw = np.array([0.5 * (r["hi"] - r["lo"]) for r in ok])
    rf = ref["fit"]
    lo3_ref, hi3_ref = rf["mu"] - 3 * rf["sigma"], rf["mu"] + 3 * rf["sigma"]
    for ax, key, ie, lab in ((axs[0], "mu", 1, r"$\mu$ [ns]"), (axs[1], "sigma", 2, r"$\sigma$ [ns]")):
        y = np.array([r["fit"][key] for r in ok]); e = np.array([r["fit"]["err"][ie] for r in ok])
        ax.errorbar(xc, y, xerr=xw, yerr=e, fmt="o", color="black", markersize=6, capsize=0, label="Slice fit")
        ax.axhline(rf[key], color="#e41a1c", linewidth=1.8, label="Full range")
        ax.axhspan(rf[key] - rf["err"][ie], rf[key] + rf["err"][ie], color="#e41a1c", alpha=0.2)
        ax.set_ylabel(lab)
        pad = max(0.05, 0.25 * (np.max(y + e) - np.min(y - e)))
        ax.set_ylim(min(np.min(y - e), rf[key]) - pad, max(np.max(y + e), rf[key]) + 3 * pad)
    # the mu +- 3 sigma window of each slice as a box over the slice, mu as a marker; the full-range window as lines
    for k, r in enumerate(ok):
        f = r["fit"]
        axs[2].fill_between([r["lo"], r["hi"]], f["mu"] - 3 * f["sigma"], f["mu"] + 3 * f["sigma"], color="#377eb8",
                            alpha=0.35, linewidth=0, label=r"Slice $\mu \pm 3\sigma$" if k == 0 else None)
        axs[2].plot([r["lo"], r["hi"]], [f["mu"] - 3 * f["sigma"]] * 2, color="#377eb8", linewidth=1.5)
        axs[2].plot([r["lo"], r["hi"]], [f["mu"] + 3 * f["sigma"]] * 2, color="#377eb8", linewidth=1.5)
    axs[2].plot(xc, [r["fit"]["mu"] for r in ok], "o", color="black", markersize=5, label=r"Slice $\mu$")
    axs[2].axhline(rf["mu"], color="#e41a1c", linewidth=1.5)
    for k, y in enumerate((lo3_ref, hi3_ref)):
        axs[2].axhline(y, color="#e41a1c", linestyle="--", linewidth=1.8, label=r"Full range $\mu \pm 3\sigma$" if k == 0 else None)
    edges = np.r_[[r["fit"]["mu"] - 3 * r["fit"]["sigma"] for r in ok], [r["fit"]["mu"] + 3 * r["fit"]["sigma"] for r in ok],
                  lo3_ref, hi3_ref]
    span = edges.max() - edges.min()
    axs[2].set_ylim(edges.min() - 0.15 * span, edges.max() + 0.45 * span)
    axs[2].set_ylabel(r"$\mu \pm 3\sigma$ [ns]")
    axs[2].legend(loc="upper right", fontsize=15, ncol=3, frameon=False)
    for ax in axs:
        ax.set_xlim(min(r["lo"] for r in rows), max(r["hi"] for r in rows))
    for ax in axs[:2]:
        ax.tick_params(labelbottom=False)
    axs[2].set_xlabel(xlabel)
    axs[0].legend(loc="upper right", fontsize=15, frameon=False)
    axs[0].text(0.03, 0.95, "\n".join(head + [rf"Full range: $\mu \pm 3\sigma$ = [{lo3_ref:+.2f}, {hi3_ref:+.2f}] ns"]),
                transform=axs[0].transAxes, ha="left", va="top", fontsize=15, linespacing=1.25)
    header(axs[0], runs, r_tag, lumi)
    fig.subplots_adjust(left=0.13, right=0.96, bottom=0.07, top=0.95)
    save_fig(fig, path, save_pdf)
    plt.close(fig)


def plot_lead_dijet_proj(inputs, r_tag, cut, runs, lumi, out_dir, save_pdf=False):
    """Projections of the dijet leading jets' corrected time (h2LeadTimeCorrPt_<r>_dijet): the full pT range (log and
    +-10 ns), each pT window of LEAD_PT_WINDOWS and each 20% class (+-10 ns), each with a Gaussian core fit; summaries of
    mu and sigma vs pT and vs centrality; panel versions of the slices; and T4proj_<r>_dijet_fits.csv, all in out_dir.
    Returns the full-range fit."""
    key = ("lead", r_tag, "dijet")
    if key not in inputs.get("tpt", {}):
        return None
    xe, te = inputs["layout"]["tpt_edges"][key]
    h = inputs["tpt"][key]                                               # (class, pT + 2, t + 2)
    common = sel_lines(r_tag, cut, "dijet")
    title = lambda extra: [rf"$\bf{{Dijet,\ leading\ jet,\ {extra}}}$"] + common

    full = proj_stats(h.sum(axis=(0, 1)), te)
    for zoom in (None, PROJ_ZOOM):
        draw_proj(full, te, title(r"0-60\%"), out_dir / f"T4proj_{r_tag}_dijet{'' if zoom is None else f'_zoom{zoom:g}'}.png",
                  runs, r_tag, lumi, zoom=zoom, save_pdf=save_pdf)
    rows = [{"slice": "all", "lo": "", "hi": "", **full}]

    pt_rows = []
    for lo, hi in LEAD_PT_WINDOWS[r_tag]:
        sel = (xe[:-1] >= lo - 1e-9) & (xe[1:] <= hi + 1e-9)            # pT bins inside the window (edges coincide)
        st = proj_stats(h.sum(axis=0)[1:-1][sel].sum(axis=0), te)
        pt_rows.append({"slice": "pt", "lo": lo, "hi": hi, **st})
        txt = rf"0-60\%,\ {lo:g} \leq p_{{T}} < {hi:g}\ GeV"
        draw_proj(st, te, title(txt), out_dir / f"T4proj_{r_tag}_dijet_pt_{edge_tag(lo)}_{edge_tag(hi)}.png",
                  runs, r_tag, lumi, zoom=PROJ_ZOOM, save_pdf=save_pdf)
    cent_rows = []
    for k, (lo, hi) in enumerate(CLASSES20):
        st = proj_stats(h[k].sum(axis=0), te)
        cent_rows.append({"slice": "cent", "lo": lo, "hi": hi, **st})
        draw_proj(st, te, title(rf"{lo}-{hi}\%"), out_dir / f"T4proj_{r_tag}_dijet_cent_{lo}_{hi}.png",
                  runs, r_tag, lumi, zoom=PROJ_ZOOM, save_pdf=save_pdf)
    rows += pt_rows + cent_rows
    panel_head = [r"$\bf{Dijet,\ leading\ jet}$"] + common
    draw_proj_panels(pt_rows, [rf"${lo:g} \leq p_{{T}} < {hi:g}$ GeV" for lo, hi in LEAD_PT_WINDOWS[r_tag]], te,
                     panel_head + ["0-60%"], out_dir / f"T4proj_{r_tag}_dijet_pt_panels.png", runs, r_tag, lumi,
                     save_pdf=save_pdf)
    draw_proj_panels(cent_rows, [f"{lo}-{hi}%" for lo, hi in CLASSES20], te, panel_head + [r"All leading-jet $p_{T}$"],
                     out_dir / f"T4proj_{r_tag}_dijet_cent_panels.png", runs, r_tag, lumi, save_pdf=save_pdf)

    if full["fit"] is not None:
        head = [r"$\bf{Dijet,\ leading\ jet}$", common[0]]
        plot_fit_summary(pt_rows, full, r"Leading jet raw $p_{T}$ [GeV]", out_dir / f"T4proj_{r_tag}_dijet_fits_vs_pt.png",
                         runs, r_tag, lumi, head + [r"0-60%"], save_pdf)
        plot_fit_summary(cent_rows, full, "Centrality [%]", out_dir / f"T4proj_{r_tag}_dijet_fits_vs_cent.png",
                         runs, r_tag, lumi, head + [r"All leading-jet $p_{T}$"], save_pdf)

    # table: every slice, with the mu +- 3 sigma edges and their shift against the full range
    rf = full["fit"]
    out = []
    for r in rows:
        f = r["fit"]
        row = {"radius": r_tag, "slice": r["slice"], "lo": f"{r['lo']:g}" if r["lo"] != "" else "", "hi": f"{r['hi']:g}" if r["hi"] != "" else "",
               "n": f"{r['tot']:.0f}", "mean": f"{r['mean']:.4f}", "rms": f"{r['rms']:.4f}"}
        if f is not None:
            lo3, hi3 = f["mu"] - 3 * f["sigma"], f["mu"] + 3 * f["sigma"]
            row.update({"mu": f"{f['mu']:.4f}", "mu_err": f"{f['err'][1]:.4f}", "sigma": f"{f['sigma']:.4f}",
                        "sigma_err": f"{f['err'][2]:.4f}", "chi2": f"{f['chi2']:.1f}", "ndf": f["ndf"],
                        "mu_minus_3sigma": f"{lo3:.4f}", "mu_plus_3sigma": f"{hi3:.4f}",
                        "shift_lo_vs_all": f"{lo3 - (rf['mu'] - 3 * rf['sigma']):.4f}" if rf else "",
                        "shift_hi_vs_all": f"{hi3 - (rf['mu'] + 3 * rf['sigma']):.4f}" if rf else "",
                        "frac_outside_3sigma": f"{r['out3'] / r['tot']:.5f}"})
        out.append(row)
        if f is not None:
            print(f"  lead dijet {r_tag} {r['slice']:4s} {str(r['lo']):>5s}-{str(r['hi']):<5s} N {r['tot']:7.0f}: mu {f['mu']:+.3f}+-{f['err'][1]:.3f} "
                  f"sigma {f['sigma']:.3f}+-{f['err'][2]:.3f} chi2/ndf {f['chi2']:.0f}/{f['ndf']} 3sigma [{f['mu'] - 3 * f['sigma']:+.3f}, {f['mu'] + 3 * f['sigma']:+.3f}]")
        else:
            print(f"  lead dijet {r_tag} {r['slice']} {r['lo']}-{r['hi']}: fit failed (N {r['tot']:.0f})")
    keys = list(dict.fromkeys(k for row in out for k in row))
    for row in out:
        for k in keys:
            row.setdefault(k, "")
    write_csv([{k: row[k] for k in keys} for row in out], out_dir / f"T4proj_{r_tag}_dijet_fits.csv")
    return full["fit"]


CUT_PT_MAX = 150                                           # [GeV] pT axis of the cut-effect plot (jets above counted in the text)


def plot_cut_effect(inputs, r_tag, fit, runs, lumi, out_dir, nsig=3, save_pdf=False):
    """Effect of a single timing window, the dijet leading-jet full-range fit's mu +- nsig sigma, on every jet of every
    selected event (h2JetTimeCorrPt_<r>_all, 0-60%): the raw pT spectrum before and after the cut, and the fraction
    kept per pT bin (binomial errors). Jets count as inside when their t bin centre is inside the window (the window
    edges fall inside 0.1 ns bins, so the edges are good to +-0.05 ns). pT bins: PT_PLOT_EDGES from 7 GeV."""
    key = ("jet", r_tag, "all")
    if key not in inputs.get("tpt", {}):
        return
    hep.style.use("ATLAS")
    xe, te = inputs["layout"]["tpt_edges"][key]
    hf = inputs["tpt"][key].sum(axis=0)                                  # (pT + 2, t + 2)
    tc = 0.5 * (te[1:] + te[:-1])
    lo_t, hi_t = fit["mu"] - nsig * fit["sigma"], fit["mu"] + nsig * fit["sigma"]
    inside = np.r_[False, (tc >= lo_t) & (tc <= hi_t), False]             # t flow bins are outside
    n_all1 = hf[1:-1].sum(axis=1)                                        # per 1 GeV pT bin, every t (flow included)
    n_in1 = hf[1:-1][:, inside].sum(axis=1)
    edges = np.array([e for e in PT_PLOT_EDGES if e >= xe[0]], dtype=float)
    idx = np.searchsorted(xe, edges[:-1] - 1e-9)
    n_all = np.add.reduceat(n_all1, idx)
    n_in = np.add.reduceat(n_in1, idx)
    w = np.diff(edges)
    cen = 0.5 * (edges[1:] + edges[:-1])
    tot_all, tot_in = hf.sum(), hf[:, inside].sum()
    above = n_all1[xe[:-1] >= CUT_PT_MAX].sum() + hf[-1].sum()
    with np.errstate(invalid="ignore", divide="ignore"):
        frac = n_in / n_all
        efrac = np.sqrt(frac * (1 - frac) / n_all)

    fig = plt.figure(figsize=(11, 10))
    gs = fig.add_gridspec(2, 1, height_ratios=(3, 1.3), hspace=0.05)
    ax, axr = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
    gap_steps(ax, n_all / w, edges, color="black", linewidth=2.2, label=f"Every jet: {fmt_n(tot_all)}")
    gap_steps(ax, n_in / w, edges, color="#e41a1c", linewidth=2.2,
              label=rf"$|t_{{\mathrm{{corr}}}} - \mu| \leq {nsig}\sigma$: {fmt_n(tot_in)} ({tot_in / tot_all:.3f})")
    ok = n_all > 0
    axr.errorbar(cen[ok], frac[ok], xerr=w[ok] / 2, yerr=efrac[ok], fmt="o", color="#e41a1c", markersize=3.5, capsize=0)
    axr.axhline(1, color="gray", linestyle="--")
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=20))
    ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10), numticks=50))
    ax.yaxis.set_minor_formatter(NullFormatter())
    pos = (n_all > 0) & (cen < CUT_PT_MAX)
    ax.set_ylim(0.3 * np.min(n_in[pos & (n_in > 0)] / w[pos & (n_in > 0)]), 100 * np.max(n_all / w))
    for a in (ax, axr):
        a.set_xlim(edges[0], CUT_PT_MAX)
    ax.tick_params(labelbottom=False)
    ax.set_ylabel(r"Jets / GeV")
    lines = [r"$\bf{Every\ jet,\ every\ selected\ event,\ 0-60\%}$"] + sel_lines(r_tag, 0, "all")[1:] + [
        rf"Timing window: dijet leading-jet fit $\mu \pm {nsig}\sigma$ = [{lo_t:+.2f}, {hi_t:+.2f}] ns",
        rf"($\mu$ = {fit['mu']:+.3f} ns, $\sigma$ = {fit['sigma']:.3f} ns, full $p_{{T}}$ range, 0-60%)"] + (
        [f"{above:.0f} jets above {CUT_PT_MAX} GeV, not shown"] if above else [])
    t = ax.text(0.97, 0.97, "\n".join(lines), transform=ax.transAxes, ha="right", va="top", fontsize=14, linespacing=1.3)
    ax.legend(loc="upper right", bbox_to_anchor=(0.985, axes_bottom(fig, ax, t) - 0.01), fontsize=15, frameon=False)
    header(ax, runs, r_tag, lumi)
    axr.set_ylim(0, 1.1)
    axr.set_xlabel(r"Jet raw $p_{T}$ [GeV]")
    axr.set_ylabel("Kept fraction", fontsize=16)
    fig.subplots_adjust(left=0.12, right=0.96, bottom=0.09, top=0.95)
    save_fig(fig, out_dir / f"C1_jetpt_{nsig}sigma_{r_tag}.png", save_pdf)
    plt.close(fig)
    show = [(7, 10), (10, 15), (15, 20), (20, 30), (30, 40), (40, 60), (60, 150)]
    print(f"  C1 {r_tag}: window [{lo_t:+.3f}, {hi_t:+.3f}] ns, kept {tot_in:.0f} / {tot_all:.0f} = {tot_in / tot_all:.4f}; kept fraction: "
          + ", ".join(f"{a}-{b} GeV {n_in1[(xe[:-1] >= a) & (xe[:-1] < b)].sum() / max(n_all1[(xe[:-1] >= a) & (xe[:-1] < b)].sum(), 1):.3f}"
                      for a, b in show))


CUT_LOG_BIN = 2.5                                          # [ns] t bins of the log (whole-axis) C2 plot
CUT_PT_CLASSES = ((7, 10), (10, 15), (15, 20), (20, 30), (30, 40), (40, 60), (60, 150))   # [GeV]


def cut_tdist_classes(inputs, r_tag, fit, nsig, zoom):
    """Per CUT_PT_CLASSES class of h2JetTimeCorrPt_<r>_all (0-60%): (lo, hi, N with flow, fraction outside mu +- nsig
    sigma, counts per ns, t edges). zoom None: CUT_LOG_BIN wide t bins (the sparse high-pT classes would be noise in the
    hist bins); otherwise the hist bins. None if the hist is missing."""
    key = ("jet", r_tag, "all")
    if key not in inputs.get("tpt", {}):
        return None
    xe, te = inputs["layout"]["tpt_edges"][key]
    hf = inputs["tpt"][key].sum(axis=0)                                  # (pT + 2, t + 2)
    tc = 0.5 * (te[1:] + te[:-1])
    lo_t, hi_t = fit["mu"] - nsig * fit["sigma"], fit["mu"] + nsig * fit["sigma"]
    inside = np.r_[False, (tc >= lo_t) & (tc <= hi_t), False]
    rows = []
    for lo, hi in CUT_PT_CLASSES:
        sel = (xe[:-1] >= lo - 1e-9) & (xe[1:] <= hi + 1e-9)
        nf = hf[1:-1][sel].sum(axis=0)                                   # t with flow
        tot = nf.sum()
        if tot <= 0:
            continue
        nd, ted = merge_t(nf[1:-1], te, CUT_LOG_BIN) if zoom is None else (nf[1:-1], te)
        rows.append((lo, hi, tot, 1 - nf[inside].sum() / tot, nd / np.diff(ted), ted))
    return rows, (lo_t, hi_t), te


def log_ylim(d, frac):
    """(lo, hi) of a log y axis with the lowest nonzero point near the bottom and the highest at frac of the height."""
    lo_y = 10 ** np.floor(np.log10(d[d > 0].min())) * 0.5
    return lo_y, lo_y * (d.max() / lo_y) ** (1 / frac)


def cut_tdist_text(r_tag, nsig):
    return [r"$\bf{Every\ jet,\ every\ selected\ event,\ 0-60\%}$"] + sel_lines(r_tag, 0, "all")[1:] + [
        rf"Window: dijet leading-jet fit $\mu \pm {nsig}\sigma$ (full $p_{{T}}$ range)",
        "Counts per ns; outside = fraction beyond the window"]


T_LABEL = r"Jet $t - t_{0}^{\mathrm{MBD}} - f(x)$ [ns]"


def plot_cut_tdist(inputs, r_tag, fit, runs, lumi, out_dir, nsig=3, zoom=None, save_pdf=False):
    """The corrected time of every jet of every selected event (h2JetTimeCorrPt_<r>_all, 0-60%) in the raw pT classes
    CUT_PT_CLASSES as counts per ns (dN/dt, not normalized to the class total), overlaid (light: low pT, dark: high pT),
    with the single timing window of C1 (the dijet leading-jet full-range fit's mu +- nsig sigma) and the fraction of each
    class outside it. Log y, since the classes differ by many decades. zoom None: the whole axis in CUT_LOG_BIN wide t
    bins; zoom z: +-z ns in the hist bins. The legend is upper left, the text upper right."""
    got = cut_tdist_classes(inputs, r_tag, fit, nsig, zoom)
    if got is None:
        return
    rows, (lo_t, hi_t), te = got
    hep.style.use("ATLAS")
    cols = plt.get_cmap("viridis")(np.linspace(0.85, 0.0, len(CUT_PT_CLASSES)))
    fig, ax = plt.subplots(figsize=(11, 8.5))
    for k, (lo, hi, tot, out_frac, d, ted) in enumerate(rows):
        ax.stairs(np.where(d > 0, d, np.nan), ted, color=cols[k], linewidth=1.8,
                  label=rf"${lo} \leq p_{{T}} < {hi}$ GeV: $N$ = {fmt_n(tot)}, outside {out_frac:.3f}")
    for k, x3 in enumerate((lo_t, hi_t)):
        ax.axvline(x3, color="#e41a1c", linestyle="--", linewidth=1.8,
                   label=rf"$\mu \pm {nsig}\sigma$ = [{lo_t:+.2f}, {hi_t:+.2f}] ns" if k == 0 else None)
    t = ax.text(0.97, 0.97, "\n".join(cut_tdist_text(r_tag, nsig)), transform=ax.transAxes, ha="right", va="top",
                fontsize=12, linespacing=1.3, bbox=dict(facecolor="white", alpha=0.8, edgecolor="none", pad=3), zorder=5)
    leg = ax.legend(loc="upper left", fontsize=11, frameon=True, facecolor="white", framealpha=0.8, edgecolor="none")
    leg.set_zorder(5)
    ax.set_yscale("log")
    ax.set_xlim(*((te[0], te[-1]) if zoom is None else (-zoom, zoom)))
    ax.set_ylim(*log_ylim(np.concatenate([r[4] for r in rows]), axes_bottom(fig, ax, t) - 0.03))   # peak just below the text
    ax.set_xlabel(T_LABEL)
    ax.set_ylabel(r"$dN/dt$ [ns$^{-1}$]")
    header(ax, runs, r_tag, lumi)
    fig.subplots_adjust(left=0.12, right=0.96, bottom=0.11, top=0.94)
    save_fig(fig, out_dir / f"C2_jett_{nsig}sigma_{r_tag}{'' if zoom is None else f'_zoom{zoom:g}'}.png", save_pdf)
    plt.close(fig)
    if zoom is None:
        print(f"  C2 {r_tag}: fraction outside [{lo_t:+.3f}, {hi_t:+.3f}] ns: " + ", ".join(f"{lo}-{hi} GeV {f:.3f} (N {n:.0f})" for lo, hi, n, f, *_ in rows))


def split_commas(line):
    """Pieces of line split at ", " outside $...$ math; each but the last keeps its comma."""
    parts, cur, in_math = [], "", False
    i = 0
    while i < len(line):
        c = line[i]
        if c == "$":
            in_math = not in_math
        if not in_math and line.startswith(", ", i):
            parts.append(cur + ",")
            cur, i = "", i + 2
            continue
        cur += c
        i += 1
    return parts + [cur]


def fit_text_box(fig, ax, lines, fs_max=24, fs_min=10, top=0.98, bottom=0.0, linespacing=1.35):
    """Write lines into the (hidden) axes ax at the largest font size that fits its width and height, wrapping long lines
    at commas (continuations indented), between top and bottom (axes fractions). Call after the figure layout is
    final."""
    ax.axis("off")
    rend = fig.canvas.get_renderer()
    box = ax.get_window_extent(rend)

    def width(txt, fs):
        t = ax.text(0, 0, txt, fontsize=fs, transform=ax.transAxes)
        w = t.get_window_extent(rend).width
        t.remove()
        return w

    for fs in range(fs_max, fs_min - 1, -1):
        out = []
        for line in lines:
            parts = split_commas(line)
            cur = parts[0]
            for p_ in parts[1:]:
                if width(cur + " " + p_, fs) <= box.width:
                    cur = cur + " " + p_
                else:
                    out.append(cur)
                    cur = "    " + p_
            out.append(cur)
        if max(width(o, fs) for o in out) > box.width:
            continue
        t = ax.text(0.0, top, "\n".join(out), transform=ax.transAxes, ha="left", va="top", fontsize=fs,
                    linespacing=linespacing)
        bb = t.get_window_extent(rend)
        if bb.height <= (top - bottom) * box.height or fs == fs_min:
            return t
        t.remove()


def plot_cut_tdist_panels(inputs, r_tag, fit, runs, lumi, out_dir, nsig=3, zoom=None, save_pdf=False):
    """C2 with one panel per pT class: 2 rows (4 + 3), each panel with its own log y range, the window, N and the
    fraction outside; the last slot holds the selection text. Same binning as plot_cut_tdist."""
    got = cut_tdist_classes(inputs, r_tag, fit, nsig, zoom)
    if got is None:
        return
    rows, (lo_t, hi_t), te = got
    hep.style.use("ATLAS")
    cols = plt.get_cmap("viridis")(np.linspace(0.85, 0.0, len(CUT_PT_CLASSES)))
    fig, axs = plt.subplots(2, 4, figsize=(22, 10.5), sharex=True)
    for k, ax in enumerate(axs.flat):
        if k >= len(rows):
            ax.axis("off")
            continue
        lo, hi, tot, out_frac, d, ted = rows[k]
        ax.stairs(np.where(d > 0, d, np.nan), ted, color=cols[k], linewidth=1.8)
        for x3 in (lo_t, hi_t):
            ax.axvline(x3, color="#e41a1c", linestyle="--", linewidth=1.5)
        ax.set_yscale("log")
        ax.set_xlim(*((te[0], te[-1]) if zoom is None else (-zoom, zoom)))
        vis = d if zoom is None else d[(ted[1:] > -zoom) & (ted[:-1] < zoom)]   # y range from the visible bins
        ax.set_ylim(*log_ylim(vis, 0.72))
        ax.text(0.04, 0.96, rf"${lo} \leq p_{{T}} < {hi}$ GeV" + f"\n$N$ = {fmt_n(tot)}\noutside {out_frac:.3f}",
                transform=ax.transAxes, ha="left", va="top", fontsize=14, linespacing=1.3)
        if k % 4 == 0:
            ax.set_ylabel(r"$dN/dt$ [ns$^{-1}$]")
        if k >= len(rows) - 4:                                           # bottom of each column
            ax.set_xlabel(T_LABEL, fontsize=16)
            ax.xaxis.set_tick_params(labelbottom=True)
    header(axs[0, -1], runs, r_tag, lumi)
    fig.subplots_adjust(left=0.06, right=0.98, bottom=0.09, top=0.95, wspace=0.22, hspace=0.12)
    if len(rows) < axs.size:                                             # selection text filling the empty slot
        fit_text_box(fig, axs.flat[len(rows)], [
            r"$\bf{Every\ jet,\ every\ selected\ event}$", r"$\bf{0-60\%}$"] + sel_lines(r_tag, 0, "all")[1:3] + [
            "Window (dashed): dijet leading-jet fit",
            rf"$\mu \pm {nsig}\sigma$ = [{lo_t:+.2f}," + "\u00a0" + f"{hi_t:+.2f}] ns",          # no-break space: no wrap there
            rf"(full $p_{{T}}$ range)",
            "Counts per ns",
            "Outside = fraction beyond the window"], top=0.88)
    save_fig(fig, out_dir / f"C2_jett_{nsig}sigma_{r_tag}_panels{'' if zoom is None else f'_zoom{zoom:g}'}.png", save_pdf)
    plt.close(fig)


# P2: OHCal fraction
# ---------------------------------------------------------------------------------------------------------------------

class XGrid:
    """x distributions of the leading jet per (selection, run, centrality class, pT_lead bin), rebinned in x and with the
    pT_lead bins from merge_above merged."""

    def __init__(self, inputs, r_tag, x_rebin, merge_above):
        self.r_tag = r_tag
        le = inputs["layout"]["lead_edges"][r_tag]
        xe = inputs["layout"]["x_edges"]
        self.xe = xe[::x_rebin]
        self.xc = 0.5 * (self.xe[1:] + self.xe[:-1])
        # merged pT_lead bins: index lists into the original bins
        self.lead_bins, self.lead_idx = [], []
        cur = []
        for i in range(len(le) - 1):
            if le[i] >= merge_above:
                cur.append(i)
            else:
                self.lead_bins.append((le[i], le[i + 1]))
                self.lead_idx.append([i])
        if cur:
            self.lead_bins.append((le[cur[0]], le[cur[-1] + 1]))
            self.lead_idx.append(cur)
        self.cut = le[0]
        self.classes = inputs["layout"]["cent_classes"]                 # centrality classes of the grid
        self.g = {sel: rebin(inputs["grids"][(r_tag, sel)].astype(np.float64), x_rebin) for sel in SELS}   # (run, cent, lead, x)

    def get(self, sel, runs_mask=None, cents=None, leads=None):
        """Summed x counts over the runs in runs_mask, the grid classes in cents and the merged pT_lead bins in leads."""
        g = self.g[sel] if runs_mask is None else self.g[sel][runs_mask]
        g = g.sum(axis=0)                                                   # (cent, lead, x)
        ci = range(len(self.classes)) if cents is None else cents
        li = [i for k in (range(len(self.lead_bins)) if leads is None else leads) for i in self.lead_idx[k]]
        return g[np.ix_(list(ci), li)].sum(axis=(0, 1))


def cent_idx(c, classes):
    """Indices of the grid classes inside the class c; they must cover it exactly (a 10% class cannot be taken from a
    20% grid)."""
    idx = [i for i, (lo, hi) in enumerate(classes) if lo >= c[0] and hi <= c[1]]
    covered = sum(classes[i][1] - classes[i][0] for i in idx)
    if covered != c[1] - c[0]:
        raise ValueError(f"centrality class {c[0]}-{c[1]}% is not a union of the grid classes {classes}")
    return idx


X1_COLORS = ("#000000", "#e41a1c", "#377eb8")             # 0-20, 20-40, 40-60 %: black, red, blue


def fmt_mean(m, e):
    """<x> with 3 decimals; errors below 0.0005 are shown as < 0.001."""
    return f"{m:.3f} ± {e:.3f}" if e >= 0.0005 else f"{m:.3f} ± <0.001"


def fmt_n(n):
    """Counts: every digit below 1000, otherwise 3 significant figures with k, M or B (5578 -> 5.58 k,
    2873552053 -> 2.87 B)."""
    n = float(n)
    if round(n) < 1000:
        return f"{n:.0f}"
    n = float(f"{n:.3g}")                                               # round first, so 999.6 k becomes 1.00 M
    for div, unit in ((1e9, "B"), (1e6, "M"), (1e3, "k")):
        if n >= div:
            v = n / div
            return f"{v:.{max(0, 2 - int(np.floor(np.log10(v))))}f} {unit}"


def axes_bottom(fig, ax, artist):
    """Bottom edge of a drawn text or legend, in the axes coordinates of ax."""
    bb = artist.get_window_extent(fig.canvas.get_renderer())
    return ax.transAxes.inverted().transform((bb.x0, bb.y0))[1]


def plot_x1(X, runs, lumi, out_dir, save_pdf=False):
    """1 x 2: inclusive (left) and dijet (right), the three 20% classes, all pT_lead; ratio to 0-60% of the selection.
    The selection and cuts are written inside the panels."""
    hep.style.use("ATLAS")
    fig = plt.figure(figsize=(17, 9.5))
    gs = fig.add_gridspec(2, 2, height_ratios=(3, 1), hspace=0.05, wspace=0.18)
    sel_text, common = x_sel_text(X)
    for col, sel in enumerate(("incl", "dijet")):
        ax, axr = fig.add_subplot(gs[0, col]), fig.add_subplot(gs[1, col])
        ref = X.get(sel)
        rd, red = density(ref, X.xe)
        ymax = 0
        dens = []
        for k, c in enumerate(CLASSES20):
            n = X.get(sel, cents=cent_idx(c, X.classes))
            d, ed = density(n, X.xe)
            N, m, em, _ = moments(n, X.xc)
            ymax = max(ymax, np.nanmax(d))
            dens.append(d)
            gap_steps(ax, d, X.xe, color=X1_COLORS[k], linewidth=2.5,
                      label=rf"{cent_label(c)}: $\langle x\rangle$ = {fmt_mean(m, em)}, $N$ = {fmt_n(N)}")
            r_, e_ = ratio(d, ed, rd, red)
            ok = np.isfinite(r_) & (n > 0)
            axr.errorbar(X.xc[ok], r_[ok], yerr=e_[ok], fmt="o", color=X1_COLORS[k], markersize=4, linewidth=1.2)
        # inside the panel, top right: the selection, the legend below it and, in the right panel, the event and jet
        # cuts below that (each placed under the measured bottom of the one above)
        ax.set_ylim(0, ymax * 1.12)
        t = ax.text(0.97, 0.96, "\n".join([rf"$\bf{{{sel_text[sel][0]}}}$"] + sel_text[sel][1:]), transform=ax.transAxes,
                    ha="right", va="top", fontsize=15, linespacing=1.3)
        leg = ax.legend(loc="upper right", bbox_to_anchor=(0.985, axes_bottom(fig, ax, t) - 0.01), fontsize=14, frameon=False)
        boxes = [t, leg]
        if col == 1:
            boxes.append(ax.text(0.97, axes_bottom(fig, ax, leg) - 0.02, "\n".join(common), transform=ax.transAxes, ha="right",
                                 va="top", fontsize=14, linespacing=1.3, color="0.25"))
        fit_ylim(fig, ax, X, dens, boxes)
        ax.set_xlim(X.xe[0], X.xe[-1])
        ax.tick_params(labelbottom=False)
        axr.axhline(1, color="gray", linestyle="--")
        axr.set_ylim(0, 2)
        axr.set_xlim(X.xe[0], X.xe[-1])
        axr.set_xlabel(X_LABEL)
        axr.set_ylabel("Class / 0–60%", fontsize=15)
        if col == 0:
            ax.set_ylabel(r"$\frac{1}{N}\frac{dN}{dx}$")
        else:
            header(ax, runs, X.r_tag, lumi)
    fig.subplots_adjust(left=0.07, right=0.965, bottom=0.09, top=0.95)
    save_fig(fig, out_dir / f"X1_cent20_{X.r_tag}.png", save_pdf)
    plt.close(fig)


def fit_ylim(fig, ax, X, dens, boxes):
    """Raise the y range until no curve reaches into a text block or the legend (they sit at fixed axes positions)."""
    top = ax.get_ylim()[1]
    dmax = np.nanmax(np.vstack(dens), axis=0)
    for b in boxes:
        bb = b.get_window_extent(fig.canvas.get_renderer()).transformed(ax.transAxes.inverted())
        x0, x1 = (X.xe[0] + f * (X.xe[-1] - X.xe[0]) for f in (bb.x0, bb.x1))
        under = (X.xe[1:] > x0) & (X.xe[:-1] < x1)
        if under.any() and np.isfinite(dmax[under]).any():
            top = max(top, np.nanmax(dmax[under]) / max(bb.y0 - 0.02, 0.05))
    ax.set_ylim(0, top)


def x_sel_text(X):
    """Selection lines of the inclusive and dijet panels, and the event and jet cuts (X1, X2)."""
    r_info = RADII[X.r_tag]
    sel_text = {
        "incl": ["Inclusive", rf"Leading jet raw $p_{{T}} \geq {X.cut:g}$ GeV", "No dijet requirement"],
        "dijet": ["Dijet", rf"$\Delta\phi > 7\pi/8$", rf"Raw $p_{{T,\mathrm{{sub}}}} \geq {X.cut:g}$ GeV"],
    }
    common = [r"$|z| < 10$ cm & MB", "Good Calo-Cent, Iterative No Flow",
              rf"Jets: $|\eta| < {r_info['eta_max']:g}$, Energy > 0", rf"raw $p_{{T}} \geq {JET_PT_MIN}$ GeV"]
    return sel_text, common


def group_colors(n):
    """Sequential colours for n run groups, light (early runs) to dark (late runs)."""
    return [plt.get_cmap("viridis")(v) for v in np.linspace(0.85, 0.0, n)]


def draw_group_panel(fig, ax, axr, X, sel, gi, glabels, title_lines, cents=None, left_lines=None, legend_fs=11.5,
                     text_fs=15):
    """One run-group panel (X2, X3): the run groups of sel in the classes cents (light: early, dark: late) and all runs
    (grey dashed) as unit-area shapes, ratio of each group to all runs. title_lines top right (first line bold), the
    legend under them, left_lines (grey) top left."""
    groups = [g for g in range(len(glabels)) if glabels[g]]
    cols = group_colors(len(groups))
    ref = X.get(sel, cents=cents)
    rd, red = density(ref, X.xe)
    N, m, em, _ = moments(ref, X.xc)
    gap_steps(ax, rd, X.xe, color="0.5", linewidth=1.5, linestyle="--",
              label=rf"All runs: $\langle x\rangle$ = {fmt_mean(m, em)}, $N$ = {fmt_n(N)}")
    dens = [rd]
    for k, g in enumerate(groups):
        n = X.get(sel, runs_mask=(gi == g), cents=cents)
        d, ed = density(n, X.xe)
        N, m, em, _ = moments(n, X.xc)
        dens.append(d)
        gap_steps(ax, d, X.xe, color=cols[k], linewidth=2,
                  label=rf"{glabels[g]}: $\langle x\rangle$ = {fmt_mean(m, em)}, $N$ = {fmt_n(N)}")
        r_, e_ = ratio(d, ed, rd, red)
        ok = np.isfinite(r_) & (n > 0)
        axr.errorbar(X.xc[ok] + (k - (len(groups) - 1) / 2) * 0.003, r_[ok], yerr=e_[ok], fmt="o", color=cols[k],
                     markersize=3.5, linewidth=1)
    ax.set_ylim(0, np.nanmax(np.vstack(dens)) * 1.12)
    t = ax.text(0.97, 0.96, "\n".join([rf"$\bf{{{title_lines[0]}}}$"] + list(title_lines[1:])),
                transform=ax.transAxes, ha="right", va="top", fontsize=text_fs, linespacing=1.3)
    leg = ax.legend(loc="upper right", bbox_to_anchor=(0.985, axes_bottom(fig, ax, t) - 0.005), fontsize=legend_fs,
                    frameon=False, labelspacing=0.25, handlelength=1.6)
    boxes = [t, leg]
    if left_lines:
        boxes.append(ax.text(0.03, 0.96, "\n".join(left_lines), transform=ax.transAxes, ha="left", va="top",
                             fontsize=text_fs - 1, linespacing=1.3, color="0.25"))
    fit_ylim(fig, ax, X, dens, boxes)
    ax.set_xlim(X.xe[0], X.xe[-1])
    ax.tick_params(labelbottom=False)
    axr.axhline(1, color="gray", linestyle="--")
    axr.set_ylim(0.5, 1.5)
    axr.set_xlim(X.xe[0], X.xe[-1])
    axr.set_xlabel(X_LABEL)
    axr.set_ylabel("Group / all runs", fontsize=15)


def plot_x2(X, gi, glabels, runs, lumi, out_dir, save_pdf=False):
    """1 x 2: inclusive (left) and dijet (right), 0-60%, all pT_lead, one curve per run group (light: early, dark: late);
    ratio to all runs of the selection."""
    hep.style.use("ATLAS")
    fig = plt.figure(figsize=(17, 9.5))
    gs = fig.add_gridspec(2, 2, height_ratios=(3, 1), hspace=0.05, wspace=0.18)
    sel_text, common = x_sel_text(X)
    for col, sel in enumerate(("incl", "dijet")):
        ax, axr = fig.add_subplot(gs[0, col]), fig.add_subplot(gs[1, col])
        # event and jet cuts top left of the dijet panel, beside the selection
        draw_group_panel(fig, ax, axr, X, sel, gi, glabels, sel_text[sel] + ["0–60%"],
                         left_lines=common if col == 1 else None)
        if col == 0:
            ax.set_ylabel(r"$\frac{1}{N}\frac{dN}{dx}$")
        else:
            header(ax, runs, X.r_tag, lumi)
    fig.subplots_adjust(left=0.07, right=0.965, bottom=0.09, top=0.95)
    save_fig(fig, out_dir / f"X2_runs_{X.r_tag}.png", save_pdf)
    plt.close(fig)


def plot_x3(X, sel, gi, glabels, runs, lumi, out_dir, save_pdf=False):
    """1 x 3: one selection, one panel per 20% class, all pT_lead, one curve per run group (as X2); ratio to all runs of
    the class."""
    hep.style.use("ATLAS")
    fig = plt.figure(figsize=(25, 9.5))
    gs = fig.add_gridspec(2, 3, height_ratios=(3, 1), hspace=0.05, wspace=0.16)
    sel_text, common = x_sel_text(X)
    for col, c in enumerate(CLASSES20):
        ax, axr = fig.add_subplot(gs[0, col]), fig.add_subplot(gs[1, col])
        # selection on the first panel, event and jet cuts on the second, the class on each
        left = {0: sel_text[sel][1:], 1: common}.get(col)
        draw_group_panel(fig, ax, axr, X, sel, gi, glabels, [rf"{SEL_LABEL[sel]},\ {c[0]}–{c[1]}\%"],
                         cents=cent_idx(c, X.classes), left_lines=left, legend_fs=11)
        if col == 0:
            ax.set_ylabel(r"$\frac{1}{N}\frac{dN}{dx}$")
        if col == 2:
            header(ax, runs, X.r_tag, lumi)
    fig.subplots_adjust(left=0.05, right=0.975, bottom=0.09, top=0.95)
    save_fig(fig, out_dir / f"X3_runs_cent20_{X.r_tag}_{sel}.png", save_pdf)
    plt.close(fig)


def x_moment_rows(X, gi, glabels):
    rows = []
    groups = [("all", None)] + [(glabels[g], gi == g) for g in range(len(glabels)) if glabels[g]]
    for sel in SELS:
        for gname, mask in groups:
            for c in dict.fromkeys(tuple(X.classes) + CLASSES20 + ((0, 60),)):   # grid classes, 20%, 0-60%, no repeats
                for j, (lo, hi) in [(-1, (X.lead_bins[0][0], X.lead_bins[-1][1]))] + list(enumerate(X.lead_bins)):
                    n = X.get(sel, runs_mask=mask, cents=cent_idx(c, X.classes), leads=None if j < 0 else [j])
                    N, m, em, rms = moments(n, X.xc)
                    rows.append({"radius": X.r_tag, "selection": sel, "run_group": gname, "cent_lo": c[0], "cent_hi": c[1],
                                 "ptlead_lo": f"{lo:g}", "ptlead_hi": f"{hi:g}", "n": f"{N:.0f}", "mean_x": f"{m:.5f}",
                                 "mean_x_err": f"{em:.5f}", "rms_x": f"{rms:.5f}"})
    return rows


def write_timing_calib(fits, source, path):
    """The Jet-Dijet timing correction (one 0-60% row per radius), in the columns of timing_correction_iternf.csv:
    t_corr = t - t0 - f(clamp(x, x_lo, x_hi)), f = sum_i c_i x^i (coefficients lowest power first)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        f.write(f"# t_corr = t - t0 - f(clamp(x, x_lo, x_hi)), f = sum_i c_i x^i; quartic fitted to the dijet leading-jet "
                f"profile <t - t0> vs x (T1 of plot_jet_dijet_timing.py), 0-60%, all pT_lead; input {source}\n")
        f.write("radius,cent_lo,cent_hi,x_lo,x_hi," + ",".join(f"c{i}" for i in range(T1_FIT_DEG + 1)) + "\n")
        for r_tag, fit in sorted(fits.items()):
            c = fit["c"][::-1]                                           # lowest power first
            f.write(f"{r_tag},0,60,{fit['x_lo']:.6g},{fit['x_hi']:.6g}," + ",".join(f"{v:.10g}" for v in c) + "\n")
    print(f"Saved: {path}")


def write_csv(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"Saved: {path}")


# Main
# ---------------------------------------------------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Leading-jet checks in the clean dijet sample: jet pT tail and OHCal "
                                                 "fraction (Jet-Dijet output with the dijet timing hists).")
    parser.add_argument("-f", "--file", type=Path, default=Path("/sphenix/user/anarde/sEPD-Study/jetDijet/11-12-26-v3/test.list"),
                        help="List of Jet-Dijet merged ROOT files named <run>.root (default: %(default)s)")
    parser.add_argument("-l", "--lumi-list", type=Path, default=Path("/sphenix/user/anarde/sEPD-Study/f4a_bkgsub/10-30-26/test.list"),
                        help="List of Event QA ROOT files named <run>.root with hLuminosity (default: %(default)s)")
    parser.add_argument("-o", "--output-dir", type=Path, default=Path("/sphenix/user/anarde/sEPD-Study/plots/Jet-Ana/dijet_timing"),
                        help="Output directory (default: %(default)s)")
    parser.add_argument("--runs", nargs="*", type=int, help="Specific run number(s) to combine (default: all runs in the list)")
    parser.add_argument("--radii", nargs="+", choices=list(RADII), default=list(RADII), help="Jet radii (default: both)")
    parser.add_argument("--x-rebin", type=int, default=2, help="Merge this many OHCal fraction bins (60 bins of 0.025; default: %(default)s)")
    parser.add_argument("--merge-lead-above", type=float, default=30,
                        help="Merge the pT_lead bins from this edge up into one, for the pT_lead bins of x_moments_<r>.csv (default: %(default)s GeV)")
    parser.add_argument("--run-blocks", nargs="*", type=int, default=list(RUN_BLOCKS),
                        help="First runs of the run groups after the first (default: the b_run blocks %(default)s)")
    parser.add_argument("-j", "--jobs", type=int, default=min(8, os.cpu_count() or 1), help="Parallel workers (default: %(default)s)")
    parser.add_argument("--save-pdf", action="store_true", help="Save PDF versions in addition to PNG")
    parser.add_argument("--cache-dir", type=Path, default=None, help="Input cache directory (default: <output-dir>/cache)")
    parser.add_argument("--no-cache", action="store_true", help="Neither read nor write the input cache")
    parser.add_argument("--refresh-cache", action="store_true", help="Reread the ROOT files and overwrite the cache")
    parser.add_argument("--write-calib", type=Path, default=None,
                        help="Write the T1 quartic of every radius as the Jet-Dijet timing correction file, e.g. "
                             "files/calib/timing_dijet_fit_iternf.csv (needs both radii; default: not written)")
    args = parser.parse_args()
    if args.write_calib is not None and set(args.radii) != set(RADII):
        sys.exit("--write-calib needs both radii (the macro requires a row for each)")
    tick = StageTimer()
    if 60 % args.x_rebin:
        sys.exit("--x-rebin must divide 60")

    run_files = parse_list_file(args.file)
    if not run_files:
        sys.exit(f"Error: No ROOT files found in {args.file}")
    tasks = [(r, run_files[r]) for r in (sorted(args.runs) if args.runs else sorted(run_files)) if r in run_files and run_files[r].exists()]
    lumi_files = parse_list_file(args.lumi_list) if args.lumi_list.exists() else {}
    lumi_paths = [lumi_files[r] for r, _ in tasks if r in lumi_files and lumi_files[r].exists()]

    cache_path = None
    if not args.no_cache:
        cache_dir = args.cache_dir if args.cache_dir is not None else args.output_dir / "cache"
        cache_path = cache_dir / f"inputs_{cache_key(tasks, lumi_paths)}.pkl"
    if cache_path is not None and cache_path.exists() and not args.refresh_cache:
        print(f"Reading the inputs from the cache {cache_path}")
        with cache_path.open("rb") as f:
            inputs = pickle.load(f)
    else:
        n_jobs = max(1, min(args.jobs, len(tasks)))
        print(f"Reading {len(tasks)} runs with {n_jobs} worker(s)")
        inputs = load_inputs(tasks, lumi_files, n_jobs)
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with cache_path.open("wb") as f:
                pickle.dump(inputs, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"Saved the input cache {cache_path}")
    tick("inputs")
    runs, lumi = inputs["runs"], inputs["lumi"]
    gi, glabels = run_groups(runs, args.run_blocks)
    print(f"{len(runs)} runs; run groups: " + ", ".join(f"{l} ({(gi == g).sum()} runs)" for g, l in enumerate(glabels) if l))

    out = args.output_dir
    fits = {}
    for r_tag in args.radii:
        cut = inputs["layout"]["lead_edges"][r_tag][0]
        st = {s: inputs["status"][(r_tag, s)] for s in SELS}
        print(f"\n{r_tag}: cut {cut:g} GeV; leading jets selected / filled: dijet {st['dijet'][0]:.0f} / {st['dijet'][1]:.0f}, "
              f"inclusive {st['incl'][0]:.0f} / {st['incl'][1]:.0f}; x outside [-0.25, 1.25]: dijet {inputs['xflow'][(r_tag, 'dijet')]:.0f}, "
              f"inclusive {inputs['xflow'][(r_tag, 'incl')]:.0f}")
        rows = plot_p1(inputs, r_tag, cut, runs, lumi, out / "pt", args.save_pdf)
        if rows:
            write_csv(rows, out / "pt" / f"leadpt_tail_{r_tag}.csv")
            for row in rows:
                print(f"  leading jets >= {row['pt_lead_min']} GeV: inclusive {row['n_incl']}, dijet {row['n_dijet']}")

        fit = plot_t1(inputs, r_tag, cut, args.x_rebin, runs, lumi, out / "timing", save_pdf=args.save_pdf)
        fits[r_tag] = fit
        plot_t2(inputs, r_tag, cut, args.x_rebin, fit, runs, lumi, out / "timing", save_pdf=args.save_pdf)
        plot_t3(inputs, r_tag, cut, args.x_rebin, fit, runs, lumi, out / "timing", save_pdf=args.save_pdf)
        for kind, _, tags in TPT_HISTS:
            for tag in tags:
                if tag == "dijet":                                       # up to 60 GeV: the whole hist
                    plot_t45(inputs, r_tag, cut, kind, tag, runs, lumi, out / "timing", save_pdf=args.save_pdf)
                else:                                                    # lead / all: to 150 GeV, and zoomed to 60 GeV
                    plot_t45(inputs, r_tag, cut, kind, tag, runs, lumi, out / "timing", x_max=TPT_MAX, save_pdf=args.save_pdf)
                    plot_t45(inputs, r_tag, cut, kind, tag, runs, lumi, out / "timing", x_max=TPT_ZOOM,
                             suffix=f"_zoom{TPT_ZOOM:g}", save_pdf=args.save_pdf)
        pdir = out / "timing" / "projections"
        lead_fit = plot_lead_dijet_proj(inputs, r_tag, cut, runs, lumi, pdir / "lead_dijet", save_pdf=args.save_pdf)
        if lead_fit is not None:
            plot_cut_effect(inputs, r_tag, lead_fit, runs, lumi, out / "timing" / "cut", save_pdf=args.save_pdf)
            for zoom in (None, PROJ_ZOOM):
                plot_cut_tdist(inputs, r_tag, lead_fit, runs, lumi, out / "timing" / "cut", zoom=zoom, save_pdf=args.save_pdf)
                plot_cut_tdist_panels(inputs, r_tag, lead_fit, runs, lumi, out / "timing" / "cut", zoom=zoom, save_pdf=args.save_pdf)
        for zoom in (None, PROJ_ZOOM):                                   # every jet; with the dijet leading-jet window
            plot_t5_proj(inputs, r_tag, cut, runs, lumi, pdir / "all_jets", zoom=zoom, save_pdf=args.save_pdf)
            if lead_fit is not None:
                plot_t5_proj(inputs, r_tag, cut, runs, lumi, pdir / "all_jets", zoom=zoom, ref_fit=lead_fit,
                             save_pdf=args.save_pdf)
        X = XGrid(inputs, r_tag, args.x_rebin, args.merge_lead_above)
        xd = out / "ohcal_frac"
        plot_x1(X, runs, lumi, xd, args.save_pdf)
        plot_x2(X, gi, glabels, runs, lumi, xd, args.save_pdf)
        for sel in ("incl", "dijet"):
            plot_x3(X, sel, gi, glabels, runs, lumi, xd, args.save_pdf)
        write_csv(x_moment_rows(X, gi, glabels), xd / f"x_moments_{r_tag}.csv")
        tick(r_tag)

    if args.write_calib is not None:
        write_timing_calib(fits, args.file, args.write_calib)


if __name__ == "__main__":
    main()
