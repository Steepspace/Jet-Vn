#!/usr/bin/env python3
"""Leading-jet checks in the clean dijet sample (tasks/jet-dijet-timing.md), from the Jet-Dijet production with the
dijet timing hists (first in 11-10-26) and the leading-jet pT per selection (from 11-11-26 on), the production 11-11-26.

Jet-Dijet.C applies one pT_sub cut per radius (the 0-20% cut of the purity scan, R = 0.2: 14.25 GeV, R = 0.3: 21.5 GeV,
raw, delta phi > 7pi/8) to every centrality class and fills, per radius:
- h2LeadTimeOHCalFrac_<r>_<dijet|incl>_cent_<c>_ptlead_<a>_<b>: leading jet t - MBD t0 vs OHCal fraction x, for the
  dijet sample and for the inclusive comparison (leading jet raw pT >= the cut, no dijet requirement), on a grid of 10%
  centrality classes x raw pT_lead bins;
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
   - X3_runs_cent20_<r>: inclusive only, one panel per 20% class, the run groups as in X2, ratio to all runs of the class;
   - x_moments_<r>.csv: N, <x> and its error, RMS for every (selection, run group or all, 10% / 20% class or 0-60%,
     pT_lead bin or all).
3. timing/T1_time_vs_x_<r>_dijet: leading jet t - MBD t0 vs x of the dijet sample, 0-60% and all pT_lead, with the
   profile <t - t0> per x bin, its standard deviation and a quartic fitted to the profile;
   timing/T2_profiles_cent20_<r>_dijet: the profiles of the three 20% classes with the T1 fit, and profile - fit [ns].
   timing/T3_closure_<r>_dijet: uncorrected (left) and corrected (right) t vs x of the dijet sample, with their profiles
   (the _corr hists of Jet-Dijet.C with a timing correction file, production 11-12-26 on).
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

from plot_jet_bkgsub_compare import save_fig
from plot_jet_noflow_compare import parse_list_file, load_luminosity
from plot_jet_iternf_timing import StageTimer, add_header
from plot_jet_dijet_control import RADII, CENT10, EVENT_INFO, JET_PT_MIN, cent_label, one_line

CACHE_VERSION = 4                                          # 2: leading-jet raw pT per selection (11-11-26 on); 3: t vs x per (class, pT_lead) bin; 4: corrected dijet t (11-12-26 on)

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
NAME_RE = re.compile(r"h2LeadTimeOHCalFrac_(r0\d)_(dijet|incl)_cent_(\d+)_(\d+)_ptlead_([\dp]+)_([\dp]+)$")


def edge_val(tag):
    return float(tag.replace("p", "."))


# Input
# ---------------------------------------------------------------------------------------------------------------------

def read_layout(path):
    """Per radius: the pT_lead bin edges of the timing grid (from the hist names), and the x and jet pT axes."""
    lead = {r: set() for r in RADII}
    with uproot.open(path) as f:
        names = f.keys(cycle=False)
        for k in names:
            m = NAME_RE.match(k)
            if m:
                lead[m.group(1)].update((edge_val(m.group(5)), edge_val(m.group(6))))
        first = next(k for k in names if NAME_RE.match(k))
        h = f[first]
        hp = f["h2JetPtCent_r02_all"]
        layout = {"lead_edges": {r: sorted(v) for r, v in lead.items()},
                  "x_edges": h.axis(0).edges(),
                  "t_edges": h.axis(1).edges(),
                  "pt_edges": hp.axis(0).edges(),
                  "cent_edges": hp.axis(1).edges()}
    return layout


def grid_name(r, sel, c, lo, hi):
    tag = lambda x: f"{x:g}".replace(".", "p")
    return f"h2LeadTimeOHCalFrac_{r}_{sel}_cent_{c[0]}_{c[1]}_ptlead_{tag(lo)}_{tag(hi)}"


def load_run(task):
    """Per run: the x projection (all t, flow included) of every timing hist as (cent, pT_lead, x) per (radius,
    selection), the x under/overflow per (radius, selection), the jet pT x centrality hists and the status hists."""
    run, path, layout = task
    d = {"grid": {}, "xflow": {}, "jetpt": {}, "status": {}, "leadpt": {}, "tx": {}, "tx_corr": {}}
    try:
        with uproot.open(path) as f:
            for r in RADII:
                le = layout["lead_edges"][r]
                for sel in SELS:
                    g = np.zeros((len(CENT10), len(le) - 1, len(layout["x_edges"]) - 1), dtype=np.float32)
                    fl = 0.0
                    tx = np.zeros((len(CENT10), len(le) - 1, len(layout["x_edges"]) - 1, len(layout["t_edges"]) - 1),
                                  dtype=np.float32)
                    txc = np.zeros_like(tx)
                    has_corr = True
                    for ic, c in enumerate(CENT10):
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
            "tx": tx, "tx_corr": tx_corr if n_corr == len(runs) else {}, "layout": layout, "lumi": lumi}


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


def time_profile(inputs, r_tag, sel, x_rebin, cents=None, key="tx"):
    """t - t0 vs x summed over runs, pT_lead and the 10% classes cents (all if None), with x rebinned, and its profile:
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
    z = np.where(h > 0, h, np.nan)
    pcm = ax.pcolormesh(xe, te, z.T, cmap="viridis", norm=LogNorm(vmin=1, vmax=np.nanmax(z)), rasterized=True)
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
        P = time_profile(inputs, r_tag, sel, x_rebin, cents=cent_idx(c))
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
    vmax = max(np.nanmax(P[k]["h"]) for k in P)
    fig = plt.figure(figsize=(19, 8.5))
    # columns: left panel, a spacer for the right panel's y title, right panel, colour bar (close to the right panel)
    gs = fig.add_gridspec(1, 4, width_ratios=(1, 0.17, 1, 0.035), wspace=0.025)
    axs = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 2])]
    cax = fig.add_subplot(gs[0, 3])
    box = dict(facecolor="white", alpha=0.8, edgecolor="none", pad=3)
    for ax, k in zip(axs, ("tx", "tx_corr")):
        Q = P[k]
        z = np.where(Q["h"] > 0, Q["h"], np.nan)
        pcm = ax.pcolormesh(Q["xe"], Q["te"], z.T, cmap="viridis", norm=LogNorm(vmin=1, vmax=vmax), rasterized=True)
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
        self.g = {sel: rebin(inputs["grids"][(r_tag, sel)].astype(np.float64), x_rebin) for sel in SELS}   # (run, cent, lead, x)

    def get(self, sel, runs_mask=None, cents=None, leads=None):
        """Summed x counts over the runs in runs_mask, the 10% classes in cents and the merged pT_lead bins in leads."""
        g = self.g[sel] if runs_mask is None else self.g[sel][runs_mask]
        g = g.sum(axis=0)                                                   # (cent, lead, x)
        ci = range(len(CENT10)) if cents is None else cents
        li = [i for k in (range(len(self.lead_bins)) if leads is None else leads) for i in self.lead_idx[k]]
        return g[np.ix_(list(ci), li)].sum(axis=(0, 1))


def cent_idx(c):
    return [i for i, (lo, hi) in enumerate(CENT10) if lo >= c[0] and hi <= c[1]]


X1_COLORS = ("#000000", "#e41a1c", "#377eb8")             # 0-20, 20-40, 40-60 %: black, red, blue


def fmt_mean(m, e):
    """<x> with 3 decimals; errors below 0.0005 are shown as < 0.001."""
    return f"{m:.3f} ± {e:.3f}" if e >= 0.0005 else f"{m:.3f} ± <0.001"


def fmt_n(n):
    return f"{n / 1e6:.2f} M" if n >= 1e6 else (f"{n / 1e3:.1f} k" if n >= 1e3 else f"{n:.0f}")


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
            n = X.get(sel, cents=cent_idx(c))
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


def plot_x3(X, gi, glabels, runs, lumi, out_dir, save_pdf=False):
    """1 x 3: inclusive, one panel per 20% class, all pT_lead, one curve per run group (as X2); ratio to all runs of the
    class."""
    hep.style.use("ATLAS")
    fig = plt.figure(figsize=(25, 9.5))
    gs = fig.add_gridspec(2, 3, height_ratios=(3, 1), hspace=0.05, wspace=0.16)
    sel_text, common = x_sel_text(X)
    for col, c in enumerate(CLASSES20):
        ax, axr = fig.add_subplot(gs[0, col]), fig.add_subplot(gs[1, col])
        # selection on the first panel, event and jet cuts on the second, the class on each
        left = {0: sel_text["incl"][1:], 1: common}.get(col)
        draw_group_panel(fig, ax, axr, X, "incl", gi, glabels, [rf"Inclusive,\ {c[0]}–{c[1]}\%"],
                         cents=cent_idx(c), left_lines=left, legend_fs=11)
        if col == 0:
            ax.set_ylabel(r"$\frac{1}{N}\frac{dN}{dx}$")
        if col == 2:
            header(ax, runs, X.r_tag, lumi)
    fig.subplots_adjust(left=0.05, right=0.975, bottom=0.09, top=0.95)
    save_fig(fig, out_dir / f"X3_runs_cent20_{X.r_tag}.png", save_pdf)
    plt.close(fig)


def x_moment_rows(X, gi, glabels):
    rows = []
    groups = [("all", None)] + [(glabels[g], gi == g) for g in range(len(glabels)) if glabels[g]]
    for sel in SELS:
        for gname, mask in groups:
            for c in CENT10 + CLASSES20 + ((0, 60),):
                for j, (lo, hi) in [(-1, (X.lead_bins[0][0], X.lead_bins[-1][1]))] + list(enumerate(X.lead_bins)):
                    n = X.get(sel, runs_mask=mask, cents=cent_idx(c), leads=None if j < 0 else [j])
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
    parser.add_argument("-f", "--file", type=Path, default=Path("/sphenix/user/anarde/sEPD-Study/jetDijet/11-12-26/test.list"),
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
        X = XGrid(inputs, r_tag, args.x_rebin, args.merge_lead_above)
        xd = out / "ohcal_frac"
        plot_x1(X, runs, lumi, xd, args.save_pdf)
        plot_x2(X, gi, glabels, runs, lumi, xd, args.save_pdf)
        plot_x3(X, gi, glabels, runs, lumi, xd, args.save_pdf)
        write_csv(x_moment_rows(X, gi, glabels), xd / f"x_moments_{r_tag}.csv")
        tick(r_tag)

    if args.write_calib is not None:
        write_timing_calib(fits, args.file, args.write_calib)


if __name__ == "__main__":
    main()
