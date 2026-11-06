#!/usr/bin/env python3
"""Run-combined iternf jet plots and the leading-jet timing correction from Jet-Anav3 output (iter_is_noflow mode).

1. Raw jet pT (hJetPtv3_raw_<r>_iternf) summed over runs.
2. The leading-jet timing model, built step by step on the run-combined raw pT 15-40 GeV profiles of
   p2LeadJetTimeOHCalFracRawPt_<r>_iternf_cent_<lo>_<hi> (TProfile2D per run and centrality class):

       t(x, c, run) = f_c(x) + a_run + b_run * x

   with t = t_jet - t0(MBD), x the OHCal energy fraction and c the centrality class.
   - f(x): one quartic fitted to the run-combined profile (all classes).
   - a_run + b_run x: a line per run, fitted to the run's profile with f fixed, on one radius (default R = 0.3) and
     applied to every radius. Centered: the jet-weighted means of a and b over runs are 0.
   - f_c(x): a quartic per centrality class, fitted to the class profile of t - a_run - b_run x (runs pooled, lines
     fixed). This is the model; each radius has its own f and f_c.
   The model has no pT dependence: its residuals in raw pT bins (15-20, 20-30, 30-40 GeV) are plotted as its known
   limitation. Each step has a plot of what it misses and of its closure (timing/baseline/).
3. The OHCal fraction distributions of the same leading jets (timing/ohcal_frac/), from the jet counts of the same
   profiles (fBinEntries): per centrality class and raw pT bin, per run (<x> and RMS next to a_run, b_run), and per
   block of runs. They show whether the x the timing depends on changes with run, pT or centrality.
4. When the production has the jet-time-alone hists (*_tjet, the same jets as t - t0) and t vs t0 (timing/tjet_check/):
   satellite collisions in t0 and the slope of t vs t0, the per-run means of t - t0, t and t0, the per-run lines
   refitted on t (their t0 part), and the per-class mean t0 vs x. They show which part of the run and centrality
   dependence of t - t0 comes from the MBD t0.

A bin's error in a run is sigma_x / sqrt(n_run,x), with sigma_x the time RMS of the bin pooled over all runs.
"""

import argparse
import textwrap
import concurrent.futures
import csv
import hashlib
import json
import os
from pathlib import Path
import pickle
import sys
import time
import warnings
import numpy as np
import uproot
from scipy.optimize import OptimizeWarning, curve_fit
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.ticker import LogLocator, MaxNLocator
import mplhep as hep

from plot_jet_bkgsub_compare import save_fig
from plot_jet_noflow_compare import parse_list_file, load_luminosity

# (tag, R, |eta| acceptance used in Jet-Anav3.C: m_jet_eta_max_r02 / m_jet_eta_max_r03)
RADII = {
    "r02": {"r": 0.2, "eta_max": 0.9},
    "r03": {"r": 0.3, "eta_max": 0.8},
}

TIMING_PT = (15, 40)                              # raw pT range of the model [GeV], and of h2LeadJetTimeOHCalFrac
CENT_CLASSES = ((0, 20), (20, 40), (40, 60))      # m_timing_cent_ranges of Jet-Anav3.C
CENT_COLORS = ("#1b9e77", "#d95f02", "#7570b3")
JET_PT_MIN = 10                                   # calibrated jet pT cut of the production (jet_pt_min job arg)
FIT_DEG = 4                                       # degree of f(x) and f_c(x)
MIN_RUN_BIN_JETS = 20                             # min jets in a run's x bin to enter its line fit and be drawn
# Example runs for the run dependence (R = 0.3 lines): a = +0.52, b = +0.53 ns; a = -0.56, b = +0.45 ns
# (same slope, opposite offsets); a = +0.05, b = -0.67 ns (no offset, opposite slope). All have > 0.6 M jets.
EXAMPLE_RUNS = (75143, 73841, 78470)
EXAMPLE_COLORS = ("#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00", "#a65628")
PT_SPLIT = ((15, 20), (20, 30), (30, 40))         # raw pT bins of the pT residual plots [GeV]; they partition TIMING_PT
PT_SPLIT_COLORS = ("#1b7837", "#e08214", "#762a83")
# OHCal fraction distributions (timing/ohcal_frac/): raw pT bins of the profiles [GeV], on their bin edges
X_PT_BINS = ((10, 15),) + PT_SPLIT + ((40, 60),)
X_PT_COLORS = ("#999999",) + PT_SPLIT_COLORS + ("#2166ac",)
# First runs of the blocks in b_run (baseline_lines_r03_iternf): the x distributions are compared between the run
# ranges they bound (--x-run-blocks)
X_RUN_BLOCKS = (67701, 68335, 69129, 73816, 75256, 77680)
X_MIN_BIN_JETS = 20                               # min jets in an x bin for its ratio to be drawn
X_MIN_RUN_JETS = 1000                             # min jets of a run at 15-40 GeV for the <x>-line correlations

# Event selection shared by all histograms. In iter_is_noflow mode calo v2 = 0 and there is no flow failure,
# so the |Calo v2| and flow failure cuts of hJetPtv3 are not applied.
EVENT_INFO = (
    r"$|z| < 10$ cm & MB" "\n"
    "Good Calo-Cent\n"
    r"Centrality: 0–60%" "\n"
    "Iterative No Flow"
)

TIME_LABEL = r"Jet $t - t_{0}^{\mathrm{MBD}}$ [ns]"
CORR_LABEL = r"$t - t_{0} - a_{\mathrm{run}} - b_{\mathrm{run}}\,x$ [ns]"
FRAC_LABEL = r"OHCal Energy Fraction $x$"
POINTS_NOTE = r"points: $t - t_{0} - a_{\mathrm{run}} - b_{\mathrm{run}}\,x$, all runs combined"


def pt_hist_name(r_tag):
    return f"hJetPtv3_raw_{r_tag}_iternf"


def timing_hist_name(r_tag, pt=TIMING_PT):
    return f"h2LeadJetTimeOHCalFrac_{r_tag}_iternf_pt{pt[0]}_{pt[1]}"


def timing_status_name(r_tag):
    return f"hLeadJetTimingStatus_{r_tag}_iternf"


def profile_name(r_tag, cent):
    return f"p2LeadJetTimeOHCalFracRawPt_{r_tag}_iternf_cent_{cent[0]}_{cent[1]}"


def t0_hist_name(r_tag):
    return f"h2LeadJetTimeT0_{r_tag}_iternf_pt{TIMING_PT[0]}_{TIMING_PT[1]}"


def centers(e):
    return 0.5 * (e[:-1] + e[1:])


# ---------------------------------------------------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------------------------------------------------

def th2_column_moments(h):
    """Per x bin of a TH2: [sum w, sum w t, sum w t^2, sum w^2] with t the y bin center. Shape (4, n_x)."""
    v = h.values().astype(np.float64)
    w2 = h.variances().astype(np.float64)
    ye = h.axes[1].edges()
    yc = 0.5 * (ye[:-1] + ye[1:])
    return np.stack([v.sum(1), (v * yc).sum(1), (v * yc ** 2).sum(1), w2.sum(1)])


def profile2d_moments(p):
    """Raw sums of a TProfile2D per (y = raw pT, x) cell, without flow bins: shape (4, n_y, n_x).

    [sum w (fBinEntries), sum w t (the TArrayD of the TH2D base), sum w t^2 (fSumw2), sum w^2 (fBinSumw2)].
    ROOT stores cells as global bin = ix + (nx + 2) * iy.
    """
    nx = p.member("fXaxis").member("fNbins")
    ny = p.member("fYaxis").member("fNbins")
    swt = next(np.asarray(b) for b in p.bases[0].bases if "TArray" in type(b).__name__)
    shape = (ny + 2, nx + 2)
    arrays = [np.asarray(p.member("fBinEntries")), swt, np.asarray(p.member("fSumw2")), np.asarray(p.member("fBinSumw2"))]
    return np.stack([np.asarray(a, dtype=np.float64).reshape(shape)[1:-1, 1:-1] for a in arrays])


CACHE_VERSION = 4   # bump when the cached inputs change, so old caches are not reused (2: *_corr, 3: *_tjet, 4: t0_qa)


def cache_key(tasks, pt_radii, t_radii, lumi_paths):
    """Hash of what the inputs depend on: the run files (path, size, mtime), the lumi files, the radii read."""
    h = hashlib.sha1(f"v{CACHE_VERSION}|{pt_radii}|{t_radii}".encode())
    for run, path, *_ in tasks:
        st = path.stat()
        h.update(f"{run}|{path}|{st.st_size}|{st.st_mtime_ns}\n".encode())
    for path in lumi_paths:
        st = path.stat()
        h.update(f"L|{path}|{st.st_size}|{st.st_mtime_ns}\n".encode())
    return h.hexdigest()[:16]


class StageTimer:
    """Prints the wall time of each stage."""

    def __init__(self):
        self.t0 = self.last = time.time()

    def __call__(self, label):
        now = time.time()
        print(f"[time] {label}: {now - self.last:.0f} s (total {now - self.t0:.0f} s)")
        self.last = now


def load_run(task):
    """Reads the per-run histograms. Returns (run, data) where data is None if a histogram is missing.

    t_mom, t_filled and t_flow are not used any more. When the file has the timing-corrected hists (Jet-Anav3 run with
    the correction), also: corr_run (2, 4) the raw pT 15-40 GeV moments of t - t0 and of the corrected time (profiles,
    classes summed), corr_status, rawpt_edges, and under "sums" the hists that main sums over runs while streaming.
    When it has the jet-time-alone hists (*_tjet, and t vs t0): tjet_mom (4, n_cent, n_x) the raw pT 15-40 GeV
    moments of t, t0_mom (3, n_t0) jets, sum t, sum t^2 per t0 bin, t0_edges, and under "sums_tjet" the pooled TH2s.
    """
    run, file_path, pt_radii, t_radii = task
    data = {}
    try:
        with uproot.open(file_path) as rf:
            keys = set(rf.keys(cycle=False))
            data["n_events"] = float(rf["hCentrality"].values().sum() - rf["hCentralityCaloFail"].values().sum())
            for r_tag in pt_radii:
                h_pt = rf[pt_hist_name(r_tag)]
                # float64: the TH1F contents are float32, and the sums over thousands of runs exceed its precision
                data[r_tag] = {
                    "pt_values": h_pt.values().astype(np.float64),
                    "pt_variances": h_pt.variances().astype(np.float64),
                    "pt_edges": h_pt.axis().edges(),
                }
            for r_tag in t_radii:
                h_t = rf[timing_hist_name(r_tag)]
                t_flow = h_t.values(flow=True)
                profiles = [rf[profile_name(r_tag, c)] for c in CENT_CLASSES]
                d = data.setdefault(r_tag, {})
                d.update({
                    "t_values": h_t.values().astype(np.float64),   # full 2D, only summed for the combined plot
                    "t_mom": th2_column_moments(h_t),
                    "t_xedges": h_t.axes[0].edges(),
                    "t_yedges": h_t.axes[1].edges(),
                    "t_filled": float(rf[timing_status_name(r_tag)].values()[1]),
                    # Leading jets outside the plotted range: OHCal fraction below / above the axis, |t - t0| > 25 ns
                    "t_flow": np.array([t_flow[0].sum(), t_flow[-1].sum(),
                                        t_flow[1:-1, 0].sum() + t_flow[1:-1, -1].sum()]),
                    "p2_mom": np.stack([profile2d_moments(p) for p in profiles], axis=1),   # (4, n_cent, n_pt, n_x)
                    "p2_xedges": profiles[0].member("fXaxis").edges(),
                    "p2_ptedges": profiles[0].member("fYaxis").edges(),
                })
                # the timing-corrected hists, when present
                c_name = timing_hist_name(r_tag) + "_corr"
                if c_name in keys:
                    p2c = [profile2d_moments(rf[profile_name(r_tag, c) + "_corr"]) for c in CENT_CLASSES]
                    pt_e = d["p2_ptedges"]
                    sel = (pt_e[:-1] >= TIMING_PT[0]) & (pt_e[1:] <= TIMING_PT[1])
                    run_mom = lambda moms: np.sum([m[:, sel].sum(axis=(1, 2)) for m in moms], axis=0)
                    h_rp = rf[f"h2LeadJetTimeRawPt_{r_tag}_iternf"]
                    d.update({
                        "corr_run": np.stack([run_mom([d["p2_mom"][:, k] for k in range(len(CENT_CLASSES))]), run_mom(p2c)]),
                        "corr_status": rf[f"hLeadJetTimingCorrStatus_{r_tag}_iternf"].values().astype(np.float64),
                        "rawpt_edges": h_rp.axes[0].edges(),
                        "sums": {"th2_corr": rf[c_name].values().astype(np.float64),
                                 "rawpt": h_rp.values().astype(np.float64),
                                 "rawpt_corr": rf[f"h2LeadJetTimeRawPt_{r_tag}_iternf_corr"].values().astype(np.float64),
                                 "p2_corr": np.stack(p2c, axis=1)},               # (4, n_cent, n_pt, n_x)
                    })
                # the jet time alone (same jets as t - t0) and t vs t0, when present
                j_name = timing_hist_name(r_tag) + "_tjet"
                if j_name in keys:
                    pt_e = d["p2_ptedges"]
                    sel = (pt_e[:-1] >= TIMING_PT[0]) & (pt_e[1:] <= TIMING_PT[1])
                    p2j = np.stack([profile2d_moments(rf[profile_name(r_tag, c) + "_tjet"]) for c in CENT_CLASSES], axis=1)
                    h_tt = rf[t0_hist_name(r_tag)]
                    v = h_tt.values().astype(np.float64)                  # (n_t0, n_t), within the axes
                    tc = centers(h_tt.axes[1].edges())
                    d.update({
                        "tjet_mom": p2j[:, :, sel].sum(axis=2),          # (4, n_cent, n_x)
                        "t0_mom": np.stack([v.sum(1), v @ tc, v @ tc ** 2]),
                        "t0_edges": h_tt.axes[0].edges(),
                        "tt0_tedges": h_tt.axes[1].edges(),
                        "sums_tjet": {"th2_tjet": rf[j_name].values().astype(np.float64), "t0_2d": v},
                    })
    except (KeyError, OSError, StopIteration) as e:
        print(f"Warning: run {run}: {e}, skipping")
        return run, None
    return run, data


def load_t0_qa(task):
    """The MB MBD t0 distribution of a run (Event QA hT0Corrected): (run, (values, edges)), or (run, None)."""
    run, path = task
    try:
        with uproot.open(path) as rf:
            h = rf["hT0Corrected"]
            return run, (h.values().astype(np.float64), h.axis().edges())
    except (KeyError, OSError) as e:
        print(f"Warning: run {run}: no hT0Corrected in {path} ({e})")
        return run, None


# ---------------------------------------------------------------------------------------------------------------------
# Profiles and fits
# ---------------------------------------------------------------------------------------------------------------------

def profile_from_moments(mom):
    """Profile from moments (4, ..., n_x): mean, error on the mean, N_eff."""
    s0, s1, s2, sw2 = mom
    with np.errstate(divide="ignore", invalid="ignore"):
        mean = s1 / s0
        neff = np.where(sw2 > 0, s0 ** 2 / sw2, 0.0)
        err = np.sqrt(np.clip(s2 / s0 - mean ** 2, 0, None) / neff)
    return mean, err, neff


def pt_bins_in(ptedges, pt):
    """Mask of the profile's raw pT bins inside [pt0, pt1); warns if the range does not fall on bin edges."""
    sel = (ptedges[:-1] >= pt[0]) & (ptedges[1:] <= pt[1])
    lo, hi = ptedges[:-1][sel].min(), ptedges[1:][sel].max()
    if not (np.isclose(lo, pt[0]) and np.isclose(hi, pt[1])):
        print(f"Warning: raw pT {pt} is not on the profile bin edges, using {lo:g}-{hi:g} GeV")
    return sel


def fit_poly(xc, mean, err, ok):
    """Weighted fit of a degree-FIT_DEG polynomial over the bins in ok (np.polyfit order: highest power first)."""
    coef, cov = np.polyfit(xc[ok], mean[ok], FIT_DEG, w=1.0 / err[ok], cov="unscaled")
    chi2 = float((((mean[ok] - np.polyval(coef, xc[ok])) / err[ok]) ** 2).sum())
    hw = 0.5 * (xc[1] - xc[0])
    return {"coef": coef, "cov": cov, "chi2": chi2, "ndf": int(ok.sum()) - FIT_DEG - 1,
            "x_range": (float(xc[ok].min() - hw), float(xc[ok].max() + hw))}


def clamped(fit):
    """The fitted curve with x clamped to its fitted range: outside it the polynomial is unconstrained."""
    return lambda x: np.polyval(fit["coef"], np.clip(x, *fit["x_range"]))


def fit_run_lines(mom_r, xc, f_x, sigma, in_range):
    """Per run, the line a + b x fitted to (run profile - f(x)) with f fixed, by weighted least squares.

    mom_r: (n_runs, 4, n_x) profile moments. A bin's error is sigma_x / sqrt(n_run,x); bins inside in_range with
    >= MIN_RUN_BIN_JETS jets of the run are used. a, b are nan for runs with < 3 such bins.
    Returns dict a, b, a_err, b_err, cov_ab, chi2, n_bins (n_runs,) and ok (n_runs, n_x).
    """
    s0, s1, sw2 = mom_r[:, 0], mom_r[:, 1], mom_r[:, 3]
    with np.errstate(divide="ignore", invalid="ignore"):
        y = s1 / s0
        n = np.where(sw2 > 0, s0 ** 2 / sw2, 0.0)
        w = n / sigma[None, :] ** 2
    ok = in_range[None, :] & (n >= MIN_RUN_BIN_JETS) & np.isfinite(y) & np.isfinite(w) & (w > 0)
    w = np.where(ok, w, 0.0)
    d = np.where(ok, y - f_x[None, :], 0.0)
    S0, S1, S2 = w.sum(1), (w * xc).sum(1), (w * xc ** 2).sum(1)
    T0, T1 = (w * d).sum(1), (w * xc * d).sum(1)
    det = S0 * S2 - S1 ** 2
    good = (ok.sum(1) >= 3) & (det > 0)
    det = np.where(good, det, 1.0)
    a = np.where(good, (S2 * T0 - S1 * T1) / det, np.nan)
    b = np.where(good, (S0 * T1 - S1 * T0) / det, np.nan)
    chi2 = (w * (d - np.nan_to_num(a)[:, None] - np.nan_to_num(b)[:, None] * xc) ** 2).sum(1)
    nan = lambda v: np.where(good, v, np.nan)
    return {"a": a, "b": b, "a_err": np.sqrt(nan(S2 / det)), "b_err": np.sqrt(nan(S0 / det)), "cov_ab": nan(-S1 / det),
            "chi2": nan(chi2), "n_bins": ok.sum(1), "ok": ok}


def corrected_moments(mom, a, b, xc):
    """Moments of t - a_run - b_run x summed over the runs with a line: mom (n_runs, 4, ..., n_x) -> (4, ..., n_x)."""
    gd = np.isfinite(a)
    m = mom[gd]
    L = (a[gd][:, None] + b[gd][:, None] * xc[None, :]).reshape((int(gd.sum()),) + (1,) * (m.ndim - 3) + (len(xc),))
    s0, s1, s2, sw2 = m[:, 0], m[:, 1], m[:, 2], m[:, 3]
    return np.stack([s0.sum(0), (s1 - L * s0).sum(0), (s2 - 2 * L * s1 + L ** 2 * s0).sum(0), sw2.sum(0)])


def chi2_label(d, err, n_par=0):
    """chi2/ndf when n_par parameters were fitted to these points, else chi2/N."""
    chi2 = float(((d / err) ** 2).sum())
    if n_par:
        return rf"$\chi^{{2}}/\mathrm{{ndf}}$ = {chi2:.0f}/{len(d) - n_par}"
    return rf"$\chi^{{2}}/N$ = {chi2:.0f}/{len(d)}"


# ---------------------------------------------------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------------------------------------------------

def add_header(ax, runs, r_jet, lumi, x=1.0, extra=""):
    text = rf"{len(runs)} runs ({min(runs)}–{max(runs)})" + (rf", $R = {r_jet:g}$" if r_jet is not None else "")
    if lumi is not None:
        text += rf", $\mathcal{{L}} = {lumi / 1e3:.2f}\ \mathrm{{nb}}^{{-1}}$"   # lumi is in ub^-1
    ax.text(x, 1.01, text + extra, transform=ax.transAxes, ha="right", va="bottom", fontsize=13)


def pt_info(eta_max):
    return (f"{EVENT_INFO}\n"
            rf"$|\eta| < {eta_max:g}$, Energy > 0" "\n"
            rf"Calib. $p_{{T}} \geq {JET_PT_MIN}$ GeV")


def timing_info(eta_max, pt=TIMING_PT):
    return (f"{EVENT_INFO}\n"
            rf"Leading jet: $|\eta| < {eta_max:g}$, Energy > 0" "\n"
            rf"Raw ${pt[0]} \leq p_{{T}} < {pt[1]}$ GeV")


def sci(x):
    """Mathtext scientific notation, e.g. 3.18 \\times 10^{8}."""
    exp = int(np.floor(np.log10(x))) if x > 0 else 0
    return rf"{x / 10 ** exp:.2f} \times 10^{{{exp}}}"


def cent_label(c):
    return f"{CENT_CLASSES[c][0]}–{CENT_CLASSES[c][1]}%"


def pt_label(pt):
    return rf"${pt[0]} \leq p_{{T}} < {pt[1]}$ GeV"


def lines_from_text(r_tag, src):
    return "fitted per run" if r_tag == src else rf"from $R = {RADII[src]['r']:g}$"


def setup_run_axis(ax, runs):
    # Label every run when few, otherwise ~40 evenly spaced runs (x is the run index, so gaps between runs are not stretched)
    step = max(1, int(np.ceil(len(runs) / 40)))
    ticks = np.arange(0, len(runs), step)
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(runs[i]) for i in ticks], rotation=90, fontsize=7)
    ax.set_xlim(-1, len(runs))
    ax.set_xlabel("Run", labelpad=6)
    ax.tick_params(axis="x", which="minor", bottom=False, top=False)


def write_csv(rows, keys, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved: {path}")


# ---------------------------------------------------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------------------------------------------------

def plot_raw_pt(values, variances, edges, n_events, runs, r_tag, lumi, per_event, output_path, pt_max=None, save_pdf=False):
    hep.style.use("ATLAS")
    meta = RADII[r_tag]

    scale = 1.0 / (n_events * np.diff(edges)) if per_event else 1.0
    vals = values * scale
    errs = np.sqrt(variances) * scale

    nz = np.where(values > 0)[0]
    max_x_val = edges[nz[-1] + 1] if len(nz) else 60

    fig, ax = plt.subplots(figsize=(8, 7))
    hep.histplot((vals, edges), yerr=errs, ax=ax, histtype="step", color="#e41a1c", linewidth=2, label="Iterative No Flow")

    positive = vals[vals > 0]
    y_max = positive.max() if len(positive) else 1.0
    y_min = positive.min() * 0.5 if len(positive) else 0.5
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=20))
    ax.set_ylim(bottom=y_min, top=y_max * 50)
    ax.set_xlim(0, pt_max if pt_max is not None else max_x_val * 1.04)
    ax.set_xlabel(r"Jet raw $p_{T}$ [GeV]")
    ax.set_ylabel(r"$\frac{1}{N_{\mathrm{evt}}}\frac{dN}{dp_{T}}$ [GeV$^{-1}$]" if per_event else "Counts")

    add_header(ax, runs, meta["r"], lumi)
    ax.text(0.95, 0.95, pt_info(meta["eta_max"]), transform=ax.transAxes, ha="right", va="top", fontsize=14)
    ax.text(0.05, 0.05, rf"$N_{{\mathrm{{evt}}}} = {sci(n_events)}$, $N_{{\mathrm{{jets}}}} = {sci(values.sum())}$",
            transform=ax.transAxes, ha="left", va="bottom", fontsize=13)
    plt.subplots_adjust(left=0.15, right=0.96, bottom=0.12, top=0.94)
    save_fig(fig, output_path, save_pdf)
    plt.close(fig)

    if pt_max is None and max_x_val > 60:
        plot_raw_pt(values, variances, edges, n_events, runs, r_tag, lumi, per_event,
                    output_path.with_name(f"{output_path.stem}_pt60{output_path.suffix}"), pt_max=60, save_pdf=save_pdf)


def plot_baseline(values, xedges, yedges, xc, prof, fit, runs, r_tag, lumi, output_path, time_ylim=None, save_pdf=False):
    """The 15-40 GeV TH2 with the run-combined profile and f(x)."""
    hep.style.use("ATLAS")
    meta = RADII[r_tag]
    fig, ax = plt.subplots(figsize=(9, 7))
    z = np.where(values > 0, values, np.nan)
    pcm = ax.pcolormesh(xedges, yedges, z.T, cmap="viridis", norm=LogNorm(vmin=1, vmax=np.nanmax(z)), rasterized=True)
    cbar = fig.colorbar(pcm, ax=ax, pad=0.02)
    cbar.set_label("Counts")

    mean, err = prof
    ok = np.isfinite(mean)
    ax.errorbar(xc[ok], mean[ok], yerr=np.nan_to_num(err[ok]), fmt="o", color="black", markersize=4, capsize=0,
                linewidth=1.2, label="Profile", zorder=3)
    xs = np.linspace(*fit["x_range"], 300)
    ax.plot(xs, np.polyval(fit["coef"], xs), "-", color="#e41a1c", linewidth=2, zorder=4,
            label=rf"$f(x)$: polynomial of degree {FIT_DEG} fitted to the profile")

    ax.set_xlim(xedges[0], xedges[-1])
    ax.set_ylim(*(time_ylim if time_ylim is not None else (yedges[0], yedges[-1])))
    ax.set_xlabel(FRAC_LABEL)
    ax.set_ylabel(TIME_LABEL)
    box = dict(facecolor="white", alpha=0.8, edgecolor="none", pad=3)
    add_header(ax, runs, meta["r"], lumi)
    ax.text(0.97, 0.97, timing_info(meta["eta_max"]), transform=ax.transAxes, ha="right", va="top", fontsize=12, bbox=box)
    ax.legend(loc="upper left", fontsize=11, frameon=True, framealpha=0.8, edgecolor="none")
    plt.subplots_adjust(left=0.13, right=0.97, bottom=0.12, top=0.94)
    save_fig(fig, output_path, save_pdf)
    plt.close(fig)


def plot_profiles(xc, fit, entries, x_show, r_tag, ylim, rlim, output_path, mode="f", header=None,
                  colors=EXAMPLE_COLORS, ylabel=TIME_LABEL, res_label=r"Run $-\ f(x)$ [ns]", note=None, save_pdf=False):
    """Profiles of a few entries (runs, centrality classes or pT bins) with the run-combined f(x); data - model below.

    mode: 'f' every entry against f(x); 'line' against f(x) + a + b x, the entry's run line; 'own' against the
    entry's own curve 'fn' (drawn over its points); 'model' against a value per x bin 'model'.
    Each entry: name, mean, err, ok, n_par (parameters fitted to its points, 0: chi2/N), and a, b / fn / model.
    header: (runs, lumi) of the header line, default the entries' runs. x_show: the x range drawn.
    """
    hep.style.use("ATLAS")
    meta = RADII[r_tag]
    fig, (ax, axr) = plt.subplots(2, 1, figsize=(9, 8.5), sharex=True,
                                  gridspec_kw={"height_ratios": [2, 1], "hspace": 0.05})
    xs = np.linspace(*x_show, 300)
    f_xs = np.polyval(fit["coef"], xs)
    plain = mode == "f"
    ax.plot(xs, f_xs, "-" if plain else "--", color="black" if plain else "gray", linewidth=2 if plain else 1.5,
            label=rf"$f(x)$, fitted to all {fit['n_runs']} runs combined", zorder=2)
    axr.axhline(0.0, color="black", linewidth=1.5)
    hw = 0.5 * (xc[1] - xc[0])
    n = len(entries)
    for i, (e, color) in enumerate(zip(entries, colors)):
        ok = e["ok"]
        x = xc[ok] + 0.004 * (i - (n - 1) / 2)              # small offsets so the entries do not hide each other
        if mode == "f":
            model = np.polyval(fit["coef"], xc)
        elif mode == "line":
            model = np.polyval(fit["coef"], xc) + e["a"] + e["b"] * xc
            ax.plot(xs, f_xs + e["a"] + e["b"] * xs, "-", color=color, linewidth=1.8, zorder=2)
        elif mode == "own":
            model = e["fn"](xc)
            xo = np.linspace(xc[ok].min() - hw, xc[ok].max() + hw, 300)
            ax.plot(xo, e["fn"](xo), "-", color=color, linewidth=1.8, zorder=2)
        else:
            model = e["model"]
            ax.plot(xc[ok], model[ok], "-", color=color, linewidth=1.8, zorder=2)
        d = e["mean"][ok] - model[ok]
        ab = rf"$a$ = {e['a']:+.2f}, $b$ = {e['b']:+.2f} ns, " if mode == "line" else ""
        ax.errorbar(x, e["mean"][ok], yerr=e["err"][ok], fmt="o", color=color, markersize=4, capsize=0, linewidth=1.0,
                    label=f"{e['name']}: {ab}{chi2_label(d, e['err'][ok], e.get('n_par', 0))}", zorder=3)
        axr.errorbar(x, d, yerr=e["err"][ok], fmt="o", color=color, markersize=4, capsize=0, linewidth=1.0, zorder=3)
    ax.set_xlim(*x_show)
    ax.set_ylim(*ylim)
    axr.set_ylim(-rlim, rlim)
    ax.set_ylabel(ylabel)
    if note is not None:
        ax.text(0.03, 0.30 + 0.045 * max(0, n - 3), note, transform=ax.transAxes, ha="left", va="bottom", fontsize=11)
    axr.set_ylabel(res_label)
    axr.set_xlabel(FRAC_LABEL)
    ax.legend(loc="lower left", fontsize=11, frameon=False)
    ax.text(0.97, 0.97, timing_info(meta["eta_max"]), transform=ax.transAxes, ha="right", va="top", fontsize=12)
    h_runs, h_lumi = header if header is not None else ([e["run"] for e in entries], None)
    add_header(ax, h_runs, meta["r"], h_lumi)
    plt.subplots_adjust(left=0.13, right=0.97, bottom=0.10, top=0.95)
    save_fig(fig, output_path, save_pdf)
    plt.close(fig)


def plot_grid(xc, panels, panel_names, entry_names, colors, x_show, r_tag, rlim, runs, lumi, output_path,
              entry_fn=None, panel_fn=None, panel_curve_names=None, res_label=r"Data $-$ model [ns]", lines_txt="",
              save_pdf=False):
    """A row of panels (centrality classes or pT bins), each with several profiles of t - a_run - b_run x (runs pooled)
    and their model; data - model below.

    panels[i][j]: dict mean, err, ok. The model of entry (i, j) is entry_fn[i][j], drawn in the entry's colour over its
    points, or panel_fn[i], one black curve per panel (label panel_curve_names[i]).
    """
    hep.style.use("ATLAS")
    meta = RADII[r_tag]
    nP = len(panels)
    fig, axes = plt.subplots(2, nP, figsize=(18, 8.5), sharex=True, sharey="row",
                             gridspec_kw={"height_ratios": [2, 1], "hspace": 0.05, "wspace": 0.04})
    hw = 0.5 * (xc[1] - xc[0])
    lows, highs = [], []
    for i in range(nP):
        ax, axr = axes[0, i], axes[1, i]
        axr.axhline(0.0, color="black", linewidth=1.5)
        n = len(panels[i])
        for j, (e, color, name) in enumerate(zip(panels[i], colors, entry_names)):
            ok = e["ok"]
            if not ok.any():
                continue
            x = xc[ok] + 0.004 * (j - (n - 1) / 2)              # small offsets so the entries do not hide each other
            good = e["err"][ok] < 0.3
            if good.any():
                lows.append(e["mean"][ok][good].min())
                highs.append(e["mean"][ok][good].max())
            fn = entry_fn[i][j] if entry_fn is not None else panel_fn[i]
            if entry_fn is not None:
                xo = np.linspace(xc[ok].min() - hw, xc[ok].max() + hw, 300)
                ax.plot(xo, fn(xo), "-", color=color, linewidth=1.8, zorder=2)
            d = e["mean"][ok] - fn(xc[ok])
            ax.errorbar(x, e["mean"][ok], yerr=e["err"][ok], fmt="o", color=color, markersize=3.5, capsize=0,
                        linewidth=1.0, zorder=3, label=f"{name}: {chi2_label(d, e['err'][ok])}")
            axr.errorbar(x, d, yerr=e["err"][ok], fmt="o", color=color, markersize=3.5, capsize=0, linewidth=1.0,
                         zorder=3)
        if panel_fn is not None:                           # one curve per panel, over the panel's points
            shown = np.any([e["ok"] for e in panels[i]], axis=0)
            xo = np.linspace(xc[shown].min() - hw, xc[shown].max() + hw, 300)
            ax.plot(xo, panel_fn[i](xo), "-", color="black", linewidth=1.8, zorder=2, label=panel_curve_names[i])
        ax.legend(loc="lower left", fontsize=10, frameon=False)
        ax.text(0.05, 0.95, panel_names[i], transform=ax.transAxes, ha="left", va="top", fontsize=16)
        axr.set_xlabel(FRAC_LABEL, fontsize=16)
    lo, hi = min(lows), max(highs)
    axes[0, 0].set_ylim(lo - 0.55 * (hi - lo), hi + 0.45 * (hi - lo))
    axes[1, 0].set_ylim(-rlim, rlim)
    axes[0, 0].set_xlim(*x_show)
    axes[0, 0].set_ylabel(CORR_LABEL)
    axes[1, 0].set_ylabel(res_label)
    axes[1, 0].xaxis.set_major_locator(MaxNLocator(nbins=6, prune="both"))   # shared x: no clash at the panel edges
    axes[0, 1].text(0.05, 0.26 if panel_fn is not None else 0.22, POINTS_NOTE + "\n" + lines_txt,
                    transform=axes[0, 1].transAxes, ha="left", va="bottom", fontsize=11)
    axes[0, -1].text(0.96, 0.95, timing_info(meta["eta_max"]), transform=axes[0, -1].transAxes, ha="right", va="top",
                     fontsize=10)
    add_header(axes[0, -1], runs, meta["r"], lumi)
    plt.subplots_adjust(left=0.07, right=0.98, bottom=0.10, top=0.95)
    save_fig(fig, output_path, save_pdf)
    plt.close(fig)


def plot_lines_runs(rows, r_tag, lumi, output_path, note, save_pdf=False, run_range=None):
    """The per-run lines a and b vs run (the correction table). run_range zooms and uses the run number as x."""
    hep.style.use("ATLAS")
    if run_range is not None:
        rows = [r for r in rows if run_range[0] <= r["run"] <= run_range[1]]
        if len(rows) < 2:
            print(f"Warning: fewer than 2 runs in {run_range}, skipping zoom")
            return
    runs = [r["run"] for r in rows]
    x = np.array(runs, float) if run_range is not None else np.arange(len(rows))
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={"hspace": 0.06})
    ms = 4 if len(rows) < 500 else 2
    for ax, key, label in ((axes[0], "a", r"$a_{\mathrm{run}}$ [ns]"), (axes[1], "b", r"$b_{\mathrm{run}}$ [ns]")):
        y = np.array([r[key] for r in rows])
        e = np.array([r[f"{key}_err"] for r in rows])
        ax.axhline(0.0, color="gray", linestyle="--", linewidth=1)
        ax.errorbar(x, y, yerr=e, fmt="o", color="black", markersize=ms, capsize=0, linewidth=0.8)
        ax.set_ylabel(label)
        lo, hi = np.nanpercentile(y - e, 0.5), np.nanpercentile(y + e, 99.5)
        pad = 0.25 * (hi - lo)
        ax.set_ylim(lo - pad, hi + pad)
        ax.text(0.01, 0.95, f"RMS over runs = {np.nanstd(y):.3f} ns, median error = {np.nanmedian(e):.4f} ns",
                transform=ax.transAxes, ha="left", va="top", fontsize=12)
    axes[0].text(0.99, 0.95, note, transform=axes[0].transAxes, ha="right", va="top", fontsize=12)
    if run_range is None:
        setup_run_axis(axes[1], runs)
    else:
        axes[1].set_xlabel("Run")
        axes[1].ticklabel_format(axis="x", useOffset=False, style="plain")
    add_header(axes[0], runs, RADII[r_tag]["r"], lumi)
    plt.subplots_adjust(left=0.09, right=0.98, bottom=0.15 if run_range is None else 0.09, top=0.95)
    save_fig(fig, output_path, save_pdf)
    plt.close(fig)


# ---------------------------------------------------------------------------------------------------------------------
# The model, step by step
# ---------------------------------------------------------------------------------------------------------------------

def prepare_radius(loaded, runs, r_tag, args):
    """Profile moments of one radius at raw pT 15-40 GeV and the run-combined f(x).

    Returns dict: xc, mom_split (n_runs, 4, n_cent, n_pt_split, n_x), momk (n_runs, 4, n_cent, n_x), mom_r
    (n_runs, 4, n_x), the pooled profile (mean, err), fit (f(x)), sigma (pooled time RMS per x bin), in_range, f_x.
    """
    ref = loaded[runs[0]][r_tag]
    xc = 0.5 * (ref["p2_xedges"][:-1] + ref["p2_xedges"][1:])
    edges = ref["p2_ptedges"]
    split = [pt_bins_in(edges, pt) for pt in PT_SPLIT]
    if not np.array_equal(np.any(split, axis=0), pt_bins_in(edges, TIMING_PT)):
        sys.exit(f"Error: PT_SPLIT {PT_SPLIT} does not partition {TIMING_PT}")
    mom_split = np.stack([np.stack([loaded[r][r_tag]["p2_mom"][:, :, m].sum(axis=2) for m in split], axis=2)
                          for r in runs])
    momk = mom_split.sum(axis=3)
    mom_r = momk.sum(axis=2)
    pooled = mom_r.sum(axis=0)
    mean, err, neff = profile_from_moments(pooled)
    fit = fit_poly(xc, mean, err, np.isfinite(mean) & np.isfinite(err) & (err > 0) & (neff >= args.min_cell_entries))
    fit["n_runs"] = len(runs)
    print(f"{r_tag}: f(x) (degree {FIT_DEG}) on the combined {TIMING_PT[0]}-{TIMING_PT[1]} GeV profile: "
          f"chi2/ndf = {fit['chi2']:.0f}/{fit['ndf']}, x range {fit['x_range'][0]:.3f}-{fit['x_range'][1]:.3f}")
    with np.errstate(divide="ignore", invalid="ignore"):
        sigma = np.sqrt(np.clip(pooled[2] / pooled[0] - mean ** 2, 0, None))
    in_range = (xc >= fit["x_range"][0]) & (xc <= fit["x_range"][1])
    return {"xc": xc, "mom_split": mom_split, "momk": momk, "mom_r": mom_r, "prof": (mean, err), "fit": fit,
            "sigma": sigma, "in_range": in_range, "f_x": np.polyval(fit["coef"], xc), "ref": ref}


def run_lines(P, runs):
    """The per-run lines of one radius (f fixed), centered: jet-weighted means of a and b over runs are 0."""
    lines = fit_run_lines(P["mom_r"], P["xc"], P["f_x"], P["sigma"], P["in_range"])
    good = np.isfinite(lines["a"])
    jets = P["mom_r"][:, 0].sum(1)
    lines["jets"] = jets
    means = [np.average(lines[k][good], weights=jets[good]) for k in ("a", "b")]
    lines["a"] = lines["a"] - means[0]
    lines["b"] = lines["b"] - means[1]
    print(f"lines (f fixed) for {good.sum()}/{len(runs)} runs ({jets[good].sum() / jets.sum():.4f} of the jets); "
          f"means removed: a {means[0]:+.4f}, b {means[1]:+.4f} ns")
    return lines


def apply_model(P, lines, runs, r_tag, src, ex_runs, args):
    """With the lines: the example runs (B2/B3), the class profiles and f_c (B4/B5), the pT residuals (B6).

    Returns dict per_run, per_cent (classes + all), fc (fit per class), per_pt (pooled entries), by_class
    ([class][pT bin] profiles), pt_bias, x ranges.
    """
    xc, fit, sigma, in_range = P["xc"], P["fit"], P["sigma"], P["in_range"]
    a, b = lines["a"], lines["b"]
    nK = len(CENT_CLASSES)
    hw = 0.5 * (xc[1] - xc[0])
    span = lambda m: (float(xc[m].min() - hw), float(xc[m].max() + hw))

    # example runs, against f and against f + their line
    per_run = []
    for run in ex_runs:
        i = runs.index(run)
        m = P["mom_r"][i]
        with np.errstate(divide="ignore", invalid="ignore"):
            rm = m[1] / m[0]
            n = np.where(m[3] > 0, m[0] ** 2 / m[3], 0.0)
            re_ = sigma / np.sqrt(n)
        ok = in_range & (n >= MIN_RUN_BIN_JETS) & np.isfinite(rm) & np.isfinite(re_) & (re_ > 0)
        per_run.append({"run": run, "name": f"Run {run}", "mean": rm, "err": re_, "ok": ok, "a": float(a[i]),
                        "b": float(b[i]), "n_par": 2 if r_tag == src else 0})
        d = rm[ok] - P["f_x"][ok]
        d_l = d - a[i] - b[i] * xc[ok]
        print(f"{r_tag}: run {run}: {m[0].sum():.0f} jets; vs f(x) chi2 = {((d / re_[ok]) ** 2).sum():.0f}, "
              f"with its line (a {a[i]:+.3f}, b {b[i]:+.3f} ns) chi2 = {((d_l / re_[ok]) ** 2).sum():.1f}, {ok.sum()} bins")

    # centrality classes: profiles of t - a - b x, against f and against f_c (the model)
    corr = corrected_moments(P["momk"], a, b, xc)                              # (4, n_cent, n_x)
    per_cent, fc = [], []
    for k in range(nK + 1):
        cmom = corr[:, k] if k < nK else corr.sum(axis=1)
        cm, ce, _ = profile_from_moments(cmom)
        ok = in_range & (cmom[0] >= args.min_cell_entries) & np.isfinite(cm) & np.isfinite(ce) & (ce > 0)
        e = {"name": cent_label(k) if k < nK else "0–60% (all classes)", "mean": cm, "err": ce, "ok": ok}
        d = cm[ok] - P["f_x"][ok]
        msg = f"{r_tag}: {e['name']}: {cmom[0].sum():.0f} jets; vs f(x) mean {np.average(d, weights=cmom[0][ok]):+.3f} ns"
        if k < nK:
            fk = fit_poly(xc, cm, ce, ok)
            fc.append(fk)
            e.update({"fn": clamped(fk), "n_par": FIT_DEG + 1})
            msg += f"; f_c(x) chi2/ndf = {fk['chi2']:.0f}/{fk['ndf']}, x range {fk['x_range'][0]:.3f}-{fk['x_range'][1]:.3f}"
        per_cent.append(e)
        print(msg)

    # raw pT bins: the model's residuals (no pT dependence), classes pooled and per class
    corr_s = corrected_moments(P["mom_split"], a, b, xc)                      # (4, n_cent, n_pt_split, n_x)
    fc_x = np.stack([clamped(f)(xc) for f in fc])                             # (n_cent, n_x)
    per_pt, by_class = [], [[None] * len(PT_SPLIT) for _ in range(nK)]
    pt_bias = {}
    for s, pt in enumerate(PT_SPLIT):
        A = corr_s[:, :, s].sum(axis=1)
        pm, pe, _ = profile_from_moments(A)
        ok = in_range & (A[0] >= args.min_cell_entries) & np.isfinite(pm) & np.isfinite(pe) & (pe > 0)
        n_k = corr_s[0, :, s]
        with np.errstate(divide="ignore", invalid="ignore"):
            model = (n_k * fc_x).sum(0) / n_k.sum(0)                           # f_c averaged over the jets
        per_pt.append({"name": pt_label(pt), "mean": pm, "err": pe, "ok": ok, "model": model})
        d = pm[ok] - model[ok]
        key = f"{pt[0]}-{pt[1]}"
        pt_bias[key] = {"all": float(np.average(d, weights=A[0][ok]))}
        for k in range(nK):
            km, ke, _ = profile_from_moments(corr_s[:, k, s])
            kok = in_range & (corr_s[0, k, s] >= args.min_cell_entries) & np.isfinite(km) & np.isfinite(ke) & (ke > 0)
            by_class[k][s] = {"mean": km, "err": ke, "ok": kok}
            if kok.any():
                pt_bias[key][CENT_CLASSES[k]] = float(np.average(km[kok] - fc_x[k][kok], weights=corr_s[0, k, s][kok]))
        print(f"{r_tag}: raw pT {key} GeV ({A[0].sum():.0f} jets): data - <f_c(x)> jet-weighted mean "
              + ", ".join(f"{('all' if c == 'all' else cent_label(CENT_CLASSES.index(c)))} {v:+.3f}"
                          for c, v in pt_bias[key].items()) + " ns")

    return {"per_run": per_run, "x_run": span(np.any([e["ok"] for e in per_run], axis=0)) if per_run else fit["x_range"],
            "per_cent": per_cent, "x_cent": span(np.any([e["ok"] for e in per_cent], axis=0)), "fc": fc,
            "per_pt": per_pt, "x_pt": span(np.any([e["ok"] for e in per_pt], axis=0)), "by_class": by_class,
            "pt_bias": pt_bias}


def ranges(sets):
    """Shared y ranges (top, residual) from entries against f(x): sets of (xc, fit, entries, x range). The residual
    range uses the points with errors < 0.1 ns, so sparse edges do not set it."""
    curves = [np.polyval(fit["coef"], np.linspace(*xs, 300)) for _, fit, _, xs in sets]
    res = [np.abs(e["mean"][e["ok"]] - np.polyval(fit["coef"], xc[e["ok"]]))[e["err"][e["ok"]] < 0.1]
           for xc, fit, es, _ in sets for e in es]
    rl = max(0.5, 1.25 * max((r.max() for r in res if len(r)), default=0.5))
    return (min(c.min() for c in curves) - rl - 1.8, max(c.max() for c in curves) + rl + 2.2), rl


def write_model(path, r_tag, src, M, P, args):
    """The correction for one radius: f_c coefficients per class (ascending powers), their covariance, the ranges to
    apply it in, and the known bias from leaving out pT."""
    with path.open("w") as f:
        json.dump({
            "profiles": [profile_name(r_tag, c) for c in CENT_CLASSES],
            "model": "t = f_c(x) + a_run + b_run * x; t = leading jet t - MBD t0 [ns], x = OHCal energy fraction",
            "f_c": f"sum_i coefficients[c][i] * x^i, i = 0..{FIT_DEG}, one per centrality class",
            "centrality_classes": [list(c) for c in CENT_CLASSES],
            "coefficients": [fk["coef"][::-1].tolist() for fk in M["fc"]],
            "covariance": [fk["cov"][::-1, ::-1].tolist() for fk in M["fc"]],
            "chi2_ndf": [[fk["chi2"], fk["ndf"]] for fk in M["fc"]],
            "apply_within": {
                "raw_pt": list(TIMING_PT),
                "x_range": [list(fk["x_range"]) for fk in M["fc"]],
                "note": "clamp x to the class's x_range (the polynomial is unconstrained outside); validated only for "
                        "raw pT inside raw_pt, see known_pt_bias_ns"},
            "known_pt_bias_ns": {
                "description": "jet-weighted mean of data - model per raw pT bin [GeV], all classes and per class "
                               "(the model has no pT dependence)",
                **{k: {(c if c == "all" else f"{c[0]}-{c[1]}"): v for c, v in d.items()} for k, d in M["pt_bias"].items()}},
            "line_source": src, "lines_file": f"per_run_lines_{src}_iternf.csv",
            "convention": f"jet-weighted means of a_run and b_run over runs are 0 ({src}, raw pT "
                          f"{TIMING_PT[0]}-{TIMING_PT[1]} GeV); f_c carries the average level and slope",
            "f_all_classes": {"coefficients": P["fit"]["coef"][::-1].tolist(), "x_range": list(P["fit"]["x_range"]),
                              "note": "the run-combined f(x) the lines were fitted with"},
            "settings": {"degree": FIT_DEG, "min_cell_entries": args.min_cell_entries,
                         "min_run_bin_jets": MIN_RUN_BIN_JETS, "n_runs": P["fit"]["n_runs"]},
        }, f, indent=2)
    print(f"Saved: {path}")


def write_correction_csv(path, M, src):
    """The correction in the form Jet-Anav3.C reads (--timing-fc-file): one row per (radius, centrality class) with
    the x clamp range and the f_c coefficients c0..c4 (ascending powers of x). The lines are in the per-run CSV."""
    rows = []
    for r_tag, Mr in M.items():
        for (lo, hi), fk in zip(CENT_CLASSES, Mr["fc"]):
            rows.append({"radius": r_tag, "cent_lo": lo, "cent_hi": hi, "x_lo": fk["x_range"][0], "x_hi": fk["x_range"][1],
                         **{f"c{i}": float(c) for i, c in enumerate(fk["coef"][::-1])}})
    keys = ["radius", "cent_lo", "cent_hi", "x_lo", "x_hi"] + [f"c{i}" for i in range(FIT_DEG + 1)]
    with path.open("w", newline="") as f:
        f.write(f"# t_corr = t - t0 - [f_c(clamp(x, x_lo, x_hi)) + a_run + b_run * x], f_c = sum_i c_i x^i; "
                f"raw pT {TIMING_PT[0]}-{TIMING_PT[1]} GeV; lines: per_run_lines_{src}_iternf.csv\n")
        writer = csv.DictWriter(f, fieldnames=keys, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: (f"{v:.10g}" if isinstance(v, float) else v) for k, v in row.items()})
    print(f"Saved: {path}")


# ---------------------------------------------------------------------------------------------------------------------
# OHCal fraction distributions: the jet counts (fBinEntries) of the per-class timing profiles
# ---------------------------------------------------------------------------------------------------------------------

def x_info(r_tag):
    return (f"{EVENT_INFO}\n"
            rf"Leading jet: $|\eta| < {RADII[r_tag]['eta_max']:g}$, Energy > 0" "\n"
            r"$t$, $t_{0}$ finite, $|t - t_{0}| < 25$ ns")


def x_moments(n, xc):
    """From jets per x bin n (..., n_x): jets, mean x, RMS of x and the error on the mean (bin centres)."""
    N = n.sum(-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        m = (n * xc).sum(-1) / N
        rms = np.sqrt(np.clip((n * xc ** 2).sum(-1) / N - m ** 2, 0, None))
        return N, m, rms, rms / np.sqrt(N)


def density(n, xe):
    """Unit-area density per unit x and its statistical error, from jets per x bin n (unit weights)."""
    N = n.sum()
    if N <= 0:
        return np.zeros_like(n), np.zeros_like(n)
    s = N * np.diff(xe)
    return n / s, np.sqrt(n) / s


def gap_steps(ax, d, xe, **kw):
    """A step line of d over the bins xe with gaps at empty bins (no drops to the bottom of a log axis)."""
    ax.plot(np.repeat(xe, 2)[1:-1], np.repeat(np.where(d > 0, d, np.nan), 2), **kw)


def x_text(n, xc):
    N, m, _, _ = x_moments(n, xc)
    return rf"$\langle x \rangle$ = {m:.3f}, ${sci(N)}$ jets"


def plot_x_dists(xe, panels, runs, r_tag, lumi, output_path, ratio_label, save_pdf=False):
    """Unit-area OHCal fraction distributions, one panel each; below, each entry over the panel's reference.

    panels[i]: dict title, ref (label, jets per x bin) and entries [(label, jets per x bin, color)]. Ratios are drawn
    in bins with >= X_MIN_BIN_JETS jets; their errors are the entry's statistical error only.
    """
    hep.style.use("ATLAS")
    nP = len(panels)
    fig, axes = plt.subplots(2, nP, figsize=(5.2 * nP + 1.0, 9), sharex=True, sharey="row", squeeze=False,
                             gridspec_kw={"height_ratios": [2, 1], "hspace": 0.05, "wspace": 0.04})
    xc = 0.5 * (xe[:-1] + xe[1:])
    tops, devs = [], []
    for i, p in enumerate(panels):
        ax, axr = axes[0, i], axes[1, i]
        ref_lab, ref_n = p["ref"]
        rd, _ = density(ref_n, xe)
        gap_steps(ax, rd, xe, color="black", linewidth=1.5, linestyle="--", label=f"{ref_lab}: {x_text(ref_n, xc)}", zorder=3)
        axr.axhline(1.0, color="black", linewidth=1.5, linestyle="--")
        tops.append(rd.max())
        for lab, n, color in p["entries"]:
            if n.sum() <= 0:
                continue
            d, e = density(n, xe)
            gap_steps(ax, d, xe, color=color, linewidth=1.6, label=f"{lab}: {x_text(n, xc)}", zorder=2)
            ok = (n >= X_MIN_BIN_JETS) & (rd > 0)
            r, re_ = d[ok] / rd[ok], e[ok] / rd[ok]
            axr.errorbar(xc[ok], r, yerr=re_, fmt="o", color=color, markersize=3, capsize=0, linewidth=1.0)
            devs.append(np.abs(r - 1)[re_ < 0.05])
            tops.append(d.max())
        ax.legend(loc="lower center", fontsize=10, frameon=False)
        ax.text(0.04, 0.96, p["title"], transform=ax.transAxes, ha="left", va="top", fontsize=16)
        axr.set_xlabel(FRAC_LABEL, fontsize=16)
    top = max(tops)
    axes[0, 0].set_yscale("log")
    axes[0, 0].set_ylim(top * 1e-8, top * 300)      # room below the distributions for the legends, above for the text
    rl = min(1.0, max(0.1, 1.3 * max((d.max() for d in devs if len(d)), default=0.1)))
    axes[1, 0].set_ylim(1 - rl, 1 + rl)
    axes[0, 0].set_xlim(xe[0], xe[-1])
    axes[1, 0].xaxis.set_major_locator(MaxNLocator(nbins=6, prune="both"))
    axes[0, 0].set_ylabel(r"$\frac{1}{N}\frac{dN}{dx}$")
    axes[1, 0].set_ylabel(ratio_label)
    axes[0, -1].text(0.96, 0.96, x_info(r_tag), transform=axes[0, -1].transAxes, ha="right", va="top", fontsize=10)
    add_header(axes[0, -1], runs, RADII[r_tag]["r"], lumi)
    plt.subplots_adjust(left=0.07 if nP > 2 else 0.10, right=0.98, bottom=0.09, top=0.95)
    save_fig(fig, output_path, save_pdf)
    plt.close(fig)


def plot_x_runs(runs, mk, ek, rk, lines, rho, r_tag, src, lumi, output_path, save_pdf=False):
    """X3: per run at raw pT 15-40 GeV, <x> and RMS(x) per centrality class, and the run lines a_run, b_run."""
    hep.style.use("ATLAS")
    idx = np.arange(len(runs))
    fig, axes = plt.subplots(4, 1, figsize=(13, 13), sharex=True, gridspec_kw={"hspace": 0.06})
    ms = 2 if len(runs) >= 500 else 4

    def robust_ylim(ax, y, pad=0.25):
        lo, hi = np.nanpercentile(y, 0.5), np.nanpercentile(y, 99.5)
        ax.set_ylim(lo - pad * (hi - lo), hi + pad * (hi - lo))

    for k in range(len(CENT_CLASSES)):
        axes[0].errorbar(idx, mk[:, k], yerr=ek[:, k], fmt="o", color=CENT_COLORS[k], markersize=ms, capsize=0,
                         linewidth=0.6, label=cent_label(k))
        axes[1].plot(idx, rk[:, k], "o", color=CENT_COLORS[k], markersize=ms)
    robust_ylim(axes[0], mk)
    robust_ylim(axes[1], rk)
    axes[0].set_ylabel(r"$\langle x \rangle$")
    axes[1].set_ylabel(r"RMS of $x$")
    axes[0].legend(loc="upper left", ncol=3, frameon=False, fontsize=13, markerscale=2.5)
    axes[0].text(0.99, 0.95, rf"Leading jet, raw ${TIMING_PT[0]} \leq p_{{T}} < {TIMING_PT[1]}$ GeV, "
                 r"$|t - t_{0}| < 25$ ns", transform=axes[0].transAxes, ha="right", va="top", fontsize=12)
    for ax, key, label in ((axes[2], "a", r"$a_{\mathrm{run}}$ [ns]"), (axes[3], "b", r"$b_{\mathrm{run}}$ [ns]")):
        y, e = lines[key], lines[f"{key}_err"]
        ax.axhline(0.0, color="gray", linestyle="--", linewidth=1)
        ax.errorbar(idx, y, yerr=e, fmt="o", color="black", markersize=ms, capsize=0, linewidth=0.6)
        robust_ylim(ax, y)
        ax.set_ylabel(label)
        ax.text(0.01, 0.95, rf"correlation with $\langle x \rangle$ (all classes): $\rho$ = {rho[key]:+.2f}",
                transform=ax.transAxes, ha="left", va="top", fontsize=12)
    axes[2].text(0.99, 0.95, r"$a_{\mathrm{run}}, b_{\mathrm{run}}$ " + lines_from_text(r_tag, src),
                 transform=axes[2].transAxes, ha="right", va="top", fontsize=12)
    setup_run_axis(axes[3], runs)
    add_header(axes[0], runs, RADII[r_tag]["r"], lumi)
    plt.subplots_adjust(left=0.09, right=0.98, bottom=0.11, top=0.96)
    save_fig(fig, output_path, save_pdf)
    plt.close(fig)


def ohcal_frac_checks(loaded, runs, r_tag, lines, src, lumi, xdir, ex_runs, args):
    """The OHCal fraction x of the leading jets of the timing profiles of one radius, from their jet counts: by raw pT
    bin and centrality class (X1, X2), per run next to the run lines (X3), and per block of runs and example run (X4).
    Also writes per_run_x_<r>_iternf.csv."""
    ref = loaded[runs[0]][r_tag]
    xe, pte = ref["p2_xedges"], ref["p2_ptedges"]
    xc = 0.5 * (xe[:-1] + xe[1:])
    nK = len(CENT_CLASSES)
    name = f"{r_tag}_iternf"
    jets = np.stack([loaded[r][r_tag]["p2_mom"][0] for r in runs])                 # (n_runs, n_cent, n_pt, n_x)
    by = {pt: jets[:, :, pt_bins_in(pte, pt)].sum(axis=2) for pt in X_PT_BINS + (TIMING_PT,)}   # (n_runs, n_cent, n_x)
    tot = {pt: v.sum(0) for pt, v in by.items()}                                     # (n_cent, n_x)

    # jets outside the x axis (the 15-40 GeV TH2, all t), which the profiles do not keep
    flow = sum(loaded[r][r_tag]["t_flow"] for r in runs)
    filled = sum(loaded[r][r_tag]["t_filled"] for r in runs)
    print(f"{r_tag}: x distributions: of {filled:.0f} leading jets at raw pT {TIMING_PT[0]}-{TIMING_PT[1]} GeV, "
          f"{flow[0] / filled:.2e} have x < {xe[0]:g} and {flow[1] / filled:.2e} have x > {xe[-1]:g} (not shown)")
    for pt in X_PT_BINS:
        N, m, rms, _ = x_moments(tot[pt], xc)
        print(f"{r_tag}: raw pT {pt[0]}-{pt[1]} GeV: " + ", ".join(
            f"{cent_label(k)} {N[k]:.0f} jets <x> {m[k]:.3f} RMS {rms[k]:.3f}" for k in range(nK)))

    # X1: per class, the pT bins over the class's 15-40 GeV; X2: per pT bin, the classes over all classes
    panels = [{"title": cent_label(k), "ref": (pt_label(TIMING_PT), tot[TIMING_PT][k]),
               "entries": [(pt_label(pt), tot[pt][k], c) for pt, c in zip(X_PT_BINS, X_PT_COLORS)]} for k in range(nK)]
    plot_x_dists(xe, panels, runs, r_tag, lumi, xdir / f"X1_by_pt_{name}.png",
                 rf"Bin / {TIMING_PT[0]}–{TIMING_PT[1]} GeV", args.save_pdf)
    panels = [{"title": pt_label(pt), "ref": ("0–60%", tot[pt].sum(0)),
               "entries": [(cent_label(k), tot[pt][k], CENT_COLORS[k]) for k in range(nK)]} for pt in X_PT_BINS]
    plot_x_dists(xe, panels, runs, r_tag, lumi, xdir / f"X2_by_cent_{name}.png", "Class / 0–60%", args.save_pdf)

    # X3: per run at 15-40 GeV, next to the run lines
    n15 = by[TIMING_PT]
    _, mk, rk, ek = x_moments(n15, xc)                                               # (n_runs, n_cent)
    Na, ma, ra, ea = x_moments(n15.sum(axis=1), xc)                                  # (n_runs,)
    good = np.isfinite(lines["a"]) & (Na >= X_MIN_RUN_JETS)
    rho = {k: float(np.corrcoef(ma[good], lines[k][good])[0, 1]) for k in ("a", "b")}
    print(f"{r_tag}: <x> per run at {TIMING_PT[0]}-{TIMING_PT[1]} GeV ({good.sum()} runs with a line and >= "
          f"{X_MIN_RUN_JETS} jets): RMS over runs {np.std(ma[good]):.4f} (median error {np.median(ea[good]):.4f}); "
          f"correlation with a_run {rho['a']:+.2f}, with b_run {rho['b']:+.2f}")
    plot_x_runs(runs, mk, ek, rk, lines, rho, r_tag, src, lumi, xdir / f"X3_runs_{name}.png", args.save_pdf)
    write_csv([{"run": r, "jets": float(Na[i]), "mean_x": float(ma[i]), "mean_x_err": float(ea[i]),
                "rms_x": float(ra[i]), **{f"mean_x_{lo}_{hi}": float(mk[i, k]) for k, (lo, hi) in enumerate(CENT_CLASSES)},
                "a": float(lines["a"][i]), "b": float(lines["b"][i])} for i, r in enumerate(runs)],
              ["run", "jets", "mean_x", "mean_x_err", "rms_x"] + [f"mean_x_{lo}_{hi}" for lo, hi in CENT_CLASSES]
              + ["a", "b"], xdir / f"per_run_x_{name}.csv")

    # X4: blocks of runs (bounded by the b_run blocks) and the example runs, over all runs (15-40 GeV, all classes)
    ra_ = np.array(runs)
    bounds = [ra_[0]] + sorted(b for b in args.x_run_blocks if ra_[0] < b <= ra_[-1]) + [ra_[-1] + 1]
    groups = [(lo, hi) for lo, hi in zip(bounds[:-1], bounds[1:]) if np.any((ra_ >= lo) & (ra_ < hi))]
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(groups)))
    all15 = n15.sum(axis=(0, 1))
    entries = []
    for (lo, hi), c in zip(groups, colors):
        sel = (ra_ >= lo) & (ra_ < hi)
        entries.append((f"{ra_[sel].min()}–{ra_[sel].max()} ({sel.sum()})", n15[sel].sum(axis=(0, 1)), c))
        N, m, rms, _ = x_moments(entries[-1][1], xc)
        print(f"{r_tag}: runs {ra_[sel].min()}-{ra_[sel].max()} ({sel.sum()} runs): {N:.0f} jets, <x> {m:.3f}, RMS {rms:.3f}")
    panels = [{"title": "Blocks of runs (runs)", "ref": ("All runs", all15), "entries": entries}]
    if ex_runs:
        panels.append({"title": "Example runs", "ref": ("All runs", all15),
                       "entries": [(f"Run {r}", n15[runs.index(r)].sum(0), c) for r, c in zip(ex_runs, EXAMPLE_COLORS)]})
    plot_x_dists(xe, panels, runs, r_tag, lumi, xdir / f"X4_run_blocks_{name}.png", "Runs / all runs", args.save_pdf)


# ---------------------------------------------------------------------------------------------------------------------
# Checks of the corrected hists (Jet-Anav3 run with the correction: t - t0 - model in the *_corr hists)
# ---------------------------------------------------------------------------------------------------------------------

CORR_LABEL_SHORT = r"$t - t_{0}$ $-$ model"
UNCORR_COLOR, CORR_COLOR = "#377eb8", "#e41a1c"


def moments_mean_sd(m):
    """Mean, standard deviation and error on the mean from moments (4, ...): sum w, sum w t, sum w t^2, sum w^2."""
    with np.errstate(divide="ignore", invalid="ignore"):
        mean = m[1] / m[0]
        sd = np.sqrt(np.clip(m[2] / m[0] - mean ** 2, 0, None))
        err = sd / np.sqrt(np.where(m[3] > 0, m[0] ** 2 / m[3], 0.0))
    return mean, sd, err


def plot_corr_th2(hu, hc, xe, te, xc, prof_u, prof_c, runs, r_tag, lumi, output_path, ylim, save_pdf=False,
                  labels=(r"$t - t_{0}$", CORR_LABEL_SHORT), ylabel=TIME_LABEL):
    """C1: the raw pT 15-40 GeV TH2 of t - t0 and of the corrected time, with their profiles (T2: of t - t0 and t)."""
    hep.style.use("ATLAS")
    meta = RADII[r_tag]
    fig, (a0, a1, cax) = plt.subplots(1, 3, figsize=(17, 7), gridspec_kw={"wspace": 0.04, "width_ratios": [1, 1, 0.04]})
    a1.sharey(a0)
    a1.tick_params(labelleft=False)
    axes = np.array([a0, a1])
    vmax = max(hu.max(), hc.max())
    box = dict(facecolor="white", alpha=0.8, edgecolor="none", pad=3)
    for ax, h, (m, e), lab in ((axes[0], hu, prof_u, labels[0]), (axes[1], hc, prof_c, labels[1])):
        z = np.where(h > 0, h, np.nan)
        pcm = ax.pcolormesh(xe, te, z.T, cmap="viridis", norm=LogNorm(vmin=1, vmax=vmax), rasterized=True)
        ok = np.isfinite(m)
        ax.errorbar(xc[ok], m[ok], yerr=np.nan_to_num(e[ok]), fmt="o", color="black", markersize=4, capsize=0,
                    linewidth=1.2, label="Profile", zorder=3)
        ax.axhline(0.0, color="white", linestyle="--", linewidth=1)
        ax.set_xlim(xe[0], xe[-1])
        ax.set_ylim(*ylim)
        ax.set_xlabel(FRAC_LABEL)
        ax.text(0.03, 0.97, lab, transform=ax.transAxes, ha="left", va="top", fontsize=15, bbox=box)
    axes[0].set_ylabel(ylabel)
    axes[0].legend(loc="lower left", fontsize=12, frameon=True, framealpha=0.8, edgecolor="none")
    axes[1].text(0.97, 0.97, timing_info(meta["eta_max"]), transform=axes[1].transAxes, ha="right", va="top",
                 fontsize=12, bbox=box)
    add_header(axes[1], runs, meta["r"], lumi)
    fig.colorbar(pcm, cax=cax).set_label("Counts")
    plt.subplots_adjust(left=0.07, right=0.94, bottom=0.12, top=0.93)
    save_fig(fig, output_path, save_pdf)
    plt.close(fig)


def plot_corr_runs(runs, mu, eu, mc, ec, r_tag, lumi, output_path, save_pdf=False):
    """C2: the mean of t - t0 and of the corrected time per run (raw pT 15-40 GeV) vs run."""
    hep.style.use("ATLAS")
    meta = RADII[r_tag]
    fig, ax = plt.subplots(figsize=(13, 6))
    idx = np.arange(len(runs))
    for m, e, color, lab in ((mu, eu, UNCORR_COLOR, r"$t - t_{0}$"), (mc, ec, CORR_COLOR, CORR_LABEL_SHORT)):
        ax.errorbar(idx, m, yerr=e, fmt="o", color=color, markersize=2, capsize=0, linewidth=0.6,
                    label=rf"{lab}: RMS over runs {np.nanstd(m):.3f} ns")
    ax.axhline(0.0, color="gray", linestyle="--", linewidth=1)
    allv = np.concatenate([mu, mc])
    lo, hi = np.nanmin(allv), np.nanmax(allv)
    ax.set_ylim(lo - 0.1 * (hi - lo), hi + 0.45 * (hi - lo))
    ax.set_ylabel(r"Mean of jet $t - t_{0}^{\mathrm{MBD}}$ per run [ns]")
    setup_run_axis(ax, runs)
    ax.legend(loc="upper left", frameon=False, fontsize=16)
    ax.text(0.99, 0.97, rf"Leading jet, raw ${TIMING_PT[0]} \leq p_{{T}} < {TIMING_PT[1]}$ GeV, $|t - t_{{0}}| < 25$ ns",
            transform=ax.transAxes, ha="right", va="top", fontsize=12)
    add_header(ax, runs, meta["r"], lumi)
    plt.subplots_adjust(left=0.08, right=0.98, bottom=0.20, top=0.93)
    save_fig(fig, output_path, save_pdf)
    plt.close(fig)


def plot_corr_rawpt(edges, U, C, yc, runs, r_tag, lumi, output_path, save_pdf=False, min_jets=100):
    """C3: mean and standard deviation of t - t0 and of the corrected time per 1 GeV raw pT bin (10-100 GeV), from
    the h2LeadJetTimeRawPt TH2s (|t - t0| < 25 ns, 0.5 ns bins)."""
    hep.style.use("ATLAS")
    meta = RADII[r_tag]
    pc, hw = 0.5 * (edges[:-1] + edges[1:]), 0.5 * np.diff(edges)
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(10, 9), sharex=True, gridspec_kw={"height_ratios": [1.3, 1], "hspace": 0.05})
    x_hi = pc[0]
    for H, color, lab in ((U, UNCORR_COLOR, r"$t - t_{0}$"), (C, CORR_COLOR, CORR_LABEL_SHORT)):
        n = H.sum(1)
        ok = n >= min_jets
        m = (H * yc).sum(1)[ok] / n[ok]
        sd = np.sqrt(np.clip((H * yc ** 2).sum(1)[ok] / n[ok] - m ** 2, 0, None))
        a1.errorbar(pc[ok], m, yerr=sd / np.sqrt(n[ok]), xerr=hw[ok], fmt="o", color=color, markersize=4, capsize=0,
                    label=lab)
        a2.plot(pc[ok], sd, "o", color=color, markersize=4)
        x_hi = max(x_hi, edges[1:][ok].max())
    for a in (a1, a2):
        a.axvspan(*TIMING_PT, color="gray", alpha=0.12, linewidth=0, label="fit range of the model" if a is a1 else None)
    a1.axhline(0.0, color="gray", linestyle="--", linewidth=1)
    a1.set_xlim(edges[0], x_hi + 1)
    a1.set_ylabel("Mean [ns]")
    a2.set_ylabel("Std. dev. [ns]")
    a2.set_xlabel(r"Leading jet raw $p_{T}$ [GeV]")
    a1.legend(loc="lower left", frameon=False, fontsize=16)
    a1.text(0.98, 0.95, rf"Leading jet, $|\eta| < {meta['eta_max']:g}$, Energy > 0" "\n"
            rf"1 GeV bins with $\geq {min_jets}$ jets", transform=a1.transAxes, ha="right", va="top", fontsize=12,
            bbox=dict(facecolor="white", alpha=0.8, edgecolor="none", pad=3), zorder=5)
    a2.text(0.02, 0.95, r"standard deviation of the jet time in the bin ($|t - t_{0}| < 25$ ns)",
            transform=a2.transAxes, ha="left", va="top", fontsize=11)
    add_header(a1, runs, meta["r"], lumi)
    plt.subplots_adjust(left=0.12, right=0.97, bottom=0.09, top=0.95)
    save_fig(fig, output_path, save_pdf)
    plt.close(fig)


def plot_corr_profiles(Pc, xc, pt_c, runs, r_tag, lumi, output_path, min_jets=50, save_pdf=False):
    """C4: mean of the corrected time vs x per centrality class, in raw pT bins (10-15 GeV is outside the fit range)."""
    hep.style.use("ATLAS")
    meta = RADII[r_tag]
    bins = ((10, 15),) + PT_SPLIT
    colors = ("#999999",) + PT_SPLIT_COLORS
    fig, axes = plt.subplots(1, len(CENT_CLASSES), figsize=(18, 6.5), sharex=True, sharey=True, gridspec_kw={"wspace": 0.04})
    for k, ax in enumerate(axes):
        for (lo, hi), color in zip(bins, colors):
            m = Pc[:, k][:, (pt_c >= lo) & (pt_c < hi)].sum(axis=1)
            mean, _, err = moments_mean_sd(m)
            ok = (m[0] >= min_jets) & np.isfinite(mean)
            ax.errorbar(xc[ok], mean[ok], yerr=err[ok], fmt="o", color=color, markersize=3.5, capsize=0, linewidth=1.0,
                        label=pt_label((lo, hi)))
        ax.axhline(0.0, color="black", linewidth=1)
        ax.set_xlabel(FRAC_LABEL, fontsize=16)
        ax.text(0.05, 0.95, cent_label(k), transform=ax.transAxes, ha="left", va="top", fontsize=16)
    axes[0].set_ylim(-2.5, 1.5)
    axes[0].set_xlim(-0.3, 1.3)
    axes[0].xaxis.set_major_locator(MaxNLocator(nbins=6, prune="both"))
    axes[0].set_ylabel(r"Mean of $t - t_{0}$ $-$ model [ns]")
    axes[0].legend(loc="lower left", frameon=False, fontsize=11, title=r"raw $p_{T}$")
    axes[1].text(0.05, 0.85, rf"runs pooled, $|t - t_{{0}}| < 25$ ns, x bins with $\geq {min_jets}$ jets" "\n"
                 rf"model fitted on {TIMING_PT[0]}–{TIMING_PT[1]} GeV (no $p_{{T}}$ dependence)",
                 transform=axes[1].transAxes, ha="left", va="top", fontsize=11)
    add_header(axes[-1], runs, meta["r"], lumi)
    plt.subplots_adjust(left=0.06, right=0.98, bottom=0.13, top=0.93)
    save_fig(fig, output_path, save_pdf)
    plt.close(fig)


def corr_checks(loaded, cs, th2_uncorr, runs, r_tag, lumi, Mr, cdir, args):
    """Checks of the timing-corrected hists of one radius: the correction status, the pooled and per-run means and
    standard deviations, the residual per class and raw pT bin (vs the model's known pT bias), and plots C1-C4.

    cs: the corrected hists summed over runs; th2_uncorr: the uncorrected 15-40 GeV TH2 summed over runs."""
    ref = loaded[runs[0]][r_tag]
    te = ref["t_yedges"]
    yc = 0.5 * (te[:-1] + te[1:])
    xc = 0.5 * (ref["p2_xedges"][:-1] + ref["p2_xedges"][1:])
    pt_c = 0.5 * (ref["p2_ptedges"][:-1] + ref["p2_ptedges"][1:])
    name = timing_hist_name(r_tag)

    st = sum(loaded[r][r_tag]["corr_status"] for r in runs)
    no_line = [r for r in runs if loaded[r][r_tag]["corr_status"][2] > 0]
    print(f"{r_tag}: correction status (raw pT 10-100 GeV): t, t0 finite {st[0]:.0f}, corrected {st[1]:.0f}, "
          f"no line for the run {st[2]:.0f} ({len(no_line)} runs{': ' + str(no_line[:10]) if no_line else ''}), "
          f"OHCal frac NaN or no class {st[3]:.0f}")

    # pooled, from the exact profile moments (|t - t0| < 25 ns)
    Pu, Pc = sum(loaded[r][r_tag]["p2_mom"] for r in runs), cs["p2_corr"]          # (4, n_cent, n_pt, n_x)
    sel = (pt_c >= TIMING_PT[0]) & (pt_c < TIMING_PT[1])
    for lab, sl in [("all", slice(None))] + [(cent_label(k), k) for k in range(len(CENT_CLASSES))]:
        mu, su, _ = moments_mean_sd(Pu[:, sl][..., sel, :].reshape(4, -1).sum(1))
        mc, sc, _ = moments_mean_sd(Pc[:, sl][..., sel, :].reshape(4, -1).sum(1))
        print(f"{r_tag}: raw pT {TIMING_PT[0]}-{TIMING_PT[1]} GeV, {lab}: mean {mu:+.3f} -> {mc:+.4f} ns, "
              f"std. dev. {su:.3f} -> {sc:.3f} ns")
    for lo, hi in ((10, 15),) + PT_SPLIT + ((40, 60),):
        s = (pt_c >= lo) & (pt_c < hi)
        mc = moments_mean_sd(Pc[:, :, s].reshape(4, -1).sum(1))[0]
        per = [moments_mean_sd(Pc[:, k][:, s].reshape(4, -1).sum(1))[0] for k in range(len(CENT_CLASSES))]
        known = Mr["pt_bias"].get(f"{lo}-{hi}")
        print(f"{r_tag}: raw pT {lo}-{hi} GeV corrected mean: all {mc:+.3f}"
              + (f" (model's known bias {known['all']:+.3f})" if known else " (outside the fit range)")
              + "; " + ", ".join(f"{cent_label(k)} {v:+.3f}" for k, v in enumerate(per)) + " ns")

    # per run
    cr = np.stack([loaded[r][r_tag]["corr_run"] for r in runs])                     # (n_runs, 2, 4)
    mu, _, eu = moments_mean_sd(cr[:, 0].T)
    mc, _, ec = moments_mean_sd(cr[:, 1].T)
    n = cr[:, 0, 0]
    w = n > 0
    print(f"{r_tag}: mean per run at raw pT {TIMING_PT[0]}-{TIMING_PT[1]} GeV: RMS over runs {np.nanstd(mu):.3f} -> "
          f"{np.nanstd(mc):.3f} ns; median error {np.nanmedian(ec):.3f} ns; |corrected| > 0.2 ns in "
          f"{int(np.sum(np.abs(np.nan_to_num(mc)) > 0.2))} runs; largest: "
          + ", ".join(f"{runs[i]} ({mc[i]:+.2f} ns, {n[i]:.0f} jets)" for i in np.argsort(-np.abs(np.nan_to_num(mc)))[:5]))
    write_csv([{"run": r, "jets": float(n[i]), "mean": float(mu[i]), "mean_err": float(eu[i]), "mean_corr": float(mc[i]),
                "mean_corr_err": float(ec[i])} for i, r in enumerate(runs)],
              ["run", "jets", "mean", "mean_err", "mean_corr", "mean_corr_err"], cdir / f"per_run_means_{r_tag}_iternf.csv")

    # plots
    pu, pc_ = profile_from_moments(Pu[:, :, sel].sum(axis=(1, 2))), profile_from_moments(Pc[:, :, sel].sum(axis=(1, 2)))
    plot_corr_th2(th2_uncorr, cs["th2_corr"], ref["t_xedges"], te, xc, pu[:2], pc_[:2], runs, r_tag, lumi,
                  cdir / f"C1_{name}.png", args.time_ylim or (-10, 10), args.save_pdf)
    plot_corr_runs(runs, mu, eu, mc, ec, r_tag, lumi, cdir / f"C2_run_means_{r_tag}_iternf.png", args.save_pdf)
    plot_corr_rawpt(ref["rawpt_edges"], cs["rawpt"], cs["rawpt_corr"], yc, runs, r_tag, lumi,
                    cdir / f"C3_vs_rawpt_{r_tag}_iternf.png", args.save_pdf)
    plot_corr_profiles(Pc, xc, pt_c, runs, r_tag, lumi, cdir / f"C4_corr_profile_{r_tag}_iternf.png", save_pdf=args.save_pdf)


# ---------------------------------------------------------------------------------------------------------------------
# The jet time alone (Jet-Anav3 *_tjet hists: t without the MBD t0, same jets as t - t0) and t vs t0
# ---------------------------------------------------------------------------------------------------------------------

T0_MAIN = 2.5                                     # |t0| < T0_MAIN [ns]: main-bunch collisions; outside: satellites
T0_WINDOWS = (("early", rf"$t_{{0}} < -{T0_MAIN:g}$ ns", -np.inf, -T0_MAIN, "#2166ac"),
              ("main", rf"$|t_{{0}}| < {T0_MAIN:g}$ ns", -T0_MAIN, T0_MAIN, "black"),
              ("late", rf"$t_{{0}} > {T0_MAIN:g}$ ns", T0_MAIN, np.inf, "#b2182b"))
T0_REBIN = 2                                      # t0 bins merged for the mean t per t0 bin (0.2 ns -> 0.4 ns)
T0_MIN_BIN_JETS = 100                             # min jets in a merged t0 bin for its mean t to be drawn and fitted
T0_FIT_RANGE = 8                                  # |t0| < T0_FIT_RANGE [ns]: range of the t vs t0 slope fits
DT_COLOR, T_COLOR, T0_COLOR = "#377eb8", "#4daf4a", "#984ea3"
TJET_LABEL = r"$t$ (no $t_{0}$)"


def t_given_t0(n, st, st2, t0c, rebin=1, min_jets=0):
    """The mean jet t per t0 bin, merged by rebin: (jet-weighted t0 of the bin, mean t, error, ok), from the jets,
    sum t and sum t^2 per t0 bin."""
    k = len(t0c) // rebin * rebin
    r = lambda a: a[:k].reshape(-1, rebin).sum(1)
    n_, st_, st2_ = r(n), r(st), r(st2)
    with np.errstate(divide="ignore", invalid="ignore"):
        c = r(t0c * n) / n_
        mean = st_ / n_
        err = np.sqrt(np.clip(st2_ / n_ - mean ** 2, 0, None) / n_)
    return c, mean, err, (n_ >= min_jets) & np.isfinite(mean) & (err > 0)


def hist_mean_rms(h, e):
    c = centers(e)
    n = h.sum()
    m = (h * c).sum() / n
    return m, np.sqrt(max((h * c ** 2).sum() / n - m ** 2, 0.0))


def plot_t_vs_t0(t0e, te, H, wins, fit, runs, r_tag, lumi, output_path, save_pdf=False):
    """T1: jet t vs MBD t0 at raw pT 15-40 GeV: the t0 distribution with the main-bunch and satellite windows, the
    TH2 with the mean t per t0 bin and its line fit, and the mean t - t0 per t0 bin."""
    hep.style.use("ATLAS")
    meta = RADII[r_tag]
    t0c, tc = centers(t0e), centers(te)
    n0 = H.sum(1)
    fig, axes = plt.subplots(1, 3, figsize=(21, 7), gridspec_kw={"wspace": 0.3})
    xl = (-12, 12)

    ax = axes[0]
    gap_steps(ax, n0 / (n0.sum() * np.diff(t0e)), t0e, color="black", linewidth=1.6)
    for w in wins:
        if w["key"] != "main":
            ax.axvspan(max(xl[0], t0e[0]) if w["key"] == "early" else T0_MAIN,
                       -T0_MAIN if w["key"] == "early" else min(xl[1], t0e[-1]), color=w["color"], alpha=0.1, linewidth=0)
    ax.set_yscale("log")
    ax.set_ylim(top=ax.get_ylim()[1] * 30)
    ax.set_xlim(*xl)
    ax.set_xlabel(r"MBD $t_{0}$ [ns]")
    ax.set_ylabel(r"$\frac{1}{N}\frac{dN}{dt_{0}}$ [ns$^{-1}$]")
    ax.text(0.03, 0.97, "\n".join(rf"{w['label']}: {100 * w['frac']:.1f}%, $\langle t_{{0}} \rangle$ = {w['t0']:+.2f} ns"
                                  for w in wins), transform=ax.transAxes, ha="left", va="top", fontsize=11)

    ax = axes[1]
    z = np.where(H > 0, H, np.nan)
    pcm = ax.pcolormesh(t0e, te, z.T, cmap="viridis", norm=LogNorm(vmin=1, vmax=np.nanmax(z)), rasterized=True)
    fig.colorbar(pcm, ax=ax, pad=0.02).set_label("Counts")
    c, mean, err, ok = fit["points"]
    ax.errorbar(c[ok], mean[ok], yerr=err[ok], fmt="o", color="black", markersize=3, capsize=0, label=r"Mean $t$")
    xs = np.linspace(-T0_FIT_RANGE, T0_FIT_RANGE, 2)
    ax.plot(xs, fit["c0"] + fit["slope"] * xs, "-", color="#e41a1c", linewidth=2,
            label=rf"Fit: slope {fit['slope']:.3f} $\pm$ {fit['slope_err']:.3f}")
    m = wins[1]
    ax.plot(xs, m["t"] + (xs - m["t0"]), "--", color="#e7298a", linewidth=1.5, label="Slope 1")
    ax.set_xlim(*xl)
    ax.set_ylim(-12, 12)
    ax.set_xlabel(r"MBD $t_{0}$ [ns]")
    ax.set_ylabel(r"Jet $t$ [ns]")
    ax.legend(loc="upper left", fontsize=11, frameon=True, framealpha=0.8, edgecolor="none")

    ax = axes[2]
    ax.errorbar(c[ok], mean[ok] - c[ok], yerr=err[ok], fmt="o", color="black", markersize=3, capsize=0,
                label=r"Mean $t - t_{0}$")
    ax.plot(xs, fit["c0"] + (fit["slope"] - 1) * xs, "-", color="#e41a1c", linewidth=2, label="From the fit")
    ax.set_xlim(*xl)
    ax.set_xlabel(r"MBD $t_{0}$ [ns]")
    ax.set_ylabel(TIME_LABEL)
    ax.legend(loc="upper right", fontsize=11, frameon=False)
    ax.text(0.03, 0.03, timing_info(meta["eta_max"]), transform=ax.transAxes, ha="left", va="bottom", fontsize=10)
    add_header(ax, runs, meta["r"], lumi)
    plt.subplots_adjust(left=0.06, right=0.98, bottom=0.12, top=0.93)
    save_fig(fig, output_path, save_pdf)
    plt.close(fig)


def plot_time_dists(te, h_dt, h_t, t0e, H, wins, runs, r_tag, lumi, output_path, save_pdf=False):
    """T3: unit-area distributions at raw pT 15-40 GeV. Left: t - t0, t and t0 of all jets. Right: per t0 window, t
    and t - t0 (from the bin centres of the t vs t0 TH2)."""
    hep.style.use("ATLAS")
    meta = RADII[r_tag]
    t0c, tc = centers(t0e), centers(te)
    fig, axes = plt.subplots(1, 2, figsize=(17, 7), sharey=True, gridspec_kw={"wspace": 0.04})
    dens = lambda h, e: h / (h.sum() * np.diff(e))
    stat = lambda h, e: r"$\mu$ = {:+.2f}, RMS = {:.2f} ns".format(*hist_mean_rms(h, e))
    ax = axes[0]
    for h, e, color, lab in ((h_dt, te, DT_COLOR, r"$t - t_{0}$"), (h_t, te, T_COLOR, TJET_LABEL),
                             (H.sum(1), t0e, T0_COLOR, r"MBD $t_{0}$")):
        gap_steps(ax, dens(h, e), e, color=color, linewidth=1.8, label=f"{lab}: {stat(h, e)}")
    ax.text(0.03, 0.97, "All jets", transform=ax.transAxes, ha="left", va="top", fontsize=16)
    ax = axes[1]
    for w in wins:
        hw = H[w["sel"]]
        h_tw = hw.sum(0)
        h_dw = np.histogram((tc[None, :] - t0c[w["sel"], None]).ravel(), bins=te, weights=hw.ravel())[0]
        gap_steps(ax, dens(h_tw, te), te, color=w["color"], linewidth=1.8, label=f"{w['label']}, $t$: {stat(h_tw, te)}")
        gap_steps(ax, dens(h_dw, te), te, color=w["color"], linewidth=1.5, linestyle="--",
                  label=f"{w['label']}, $t - t_{{0}}$: {stat(h_dw, te)}")
    ax.text(0.03, 0.97, r"Per $t_{0}$ window ($t - t_{0}$ from bin centres)", transform=ax.transAxes, ha="left",
            va="top", fontsize=14)
    for ax in axes:
        ax.set_yscale("log")
        ax.set_xlim(-15, 15)
        ax.set_xticks([-10, -5, 0, 5, 10])                       # no clash of the edge labels between the panels
        ax.set_xlabel("Time [ns]")
        ax.legend(loc="lower center", fontsize=10, frameon=False)
    axes[0].set_ylim(1e-7, 30)
    axes[0].set_ylabel(r"$\frac{1}{N}\frac{dN}{dt}$ [ns$^{-1}$]")
    axes[1].text(0.97, 0.88, timing_info(meta["eta_max"]), transform=axes[1].transAxes, ha="right", va="top", fontsize=10)
    add_header(axes[1], runs, meta["r"], lumi)
    plt.subplots_adjust(left=0.08, right=0.98, bottom=0.12, top=0.93)
    save_fig(fig, output_path, save_pdf)
    plt.close(fig)


def plot_run_rows(runs, rows, r_tag, lumi, output_path, note=None, save_pdf=False):
    """Rows of per-run values vs run (the run index), sharing the run axis. rows[i]: dict ylabel, series
    [(y, err or None, color, label or None)], text (top left), zero (draw y = 0, default True), hlines (extra red dashed lines)."""
    hep.style.use("ATLAS")
    n = len(rows)
    height = 2.6 * n + 1.6
    fig, axes = plt.subplots(n, 1, figsize=(13, height), sharex=True, squeeze=False, gridspec_kw={"hspace": 0.06})
    axes = axes[:, 0]
    idx = np.arange(len(runs))
    ms = 2 if len(runs) >= 500 else 4
    for ax, row in zip(axes, rows):
        for y, e, color, label in row["series"]:
            if e is None:
                ax.plot(idx, y, "o", color=color, markersize=ms, label=label)
            else:
                ax.errorbar(idx, y, yerr=e, fmt="o", color=color, markersize=ms, capsize=0, linewidth=0.6, label=label)
        if row.get("zero", True):
            ax.axhline(0.0, color="gray", linestyle="--", linewidth=1)
        for yl in row.get("hlines", []):
            ax.axhline(yl, color="#e41a1c", linestyle="--", linewidth=1.3)
        if "ylim" in row:
            ax.set_ylim(*row["ylim"])
        else:
            allv = np.concatenate([s[0] for s in row["series"]])
            lo, hi = np.nanpercentile(allv, 0.5), np.nanpercentile(allv, 99.5)
            pad = 0.3 * (hi - lo) if hi > lo else 0.1
            ax.set_ylim(lo - pad, hi + 2 * pad)              # room at the top for the text
        ax.set_ylabel(row["ylabel"], fontsize=16)
        if row.get("text"):
            ax.text(0.01, 0.95, row["text"], transform=ax.transAxes, ha="left", va="top", fontsize=11)
        if any(s[3] for s in row["series"]):
            ax.legend(loc="upper right", ncol=len(row["series"]), frameon=False, fontsize=11, markerscale=2.5)
    if note:
        axes[0].text(0.99, 0.95, note, transform=axes[0].transAxes, ha="right", va="top", fontsize=11)
    setup_run_axis(axes[-1], runs)
    add_header(axes[0], runs, RADII[r_tag]["r"] if r_tag is not None else None, lumi)
    plt.subplots_adjust(left=0.10, right=0.98, bottom=1.1 / height, top=1 - 0.45 / height)
    save_fig(fig, output_path, save_pdf)
    plt.close(fig)


def plot_cent_tjet(xc, Pdt, Pt, rms_t0, runs, r_tag, lumi, output_path, min_jets, save_pdf=False):
    """T6: per centrality class at raw pT 15-40 GeV (runs pooled, no lines removed), the mean of t - t0, of t and of
    t0 = <t> - <t - t0> vs x. The error of t0 is its pooled RMS / sqrt(n)."""
    hep.style.use("ATLAS")
    meta = RADII[r_tag]
    fig, axes = plt.subplots(1, 3, figsize=(20, 7), gridspec_kw={"wspace": 0.25})
    for k in range(len(CENT_CLASSES)):
        md, _, ed = moments_mean_sd(Pdt[:, k])
        mt, _, et = moments_mean_sd(Pt[:, k])
        n = Pdt[0, k]
        ok = (n >= min_jets) & np.isfinite(md) & np.isfinite(mt)
        tot_d, tot_t = moments_mean_sd(Pdt[:, k].sum(-1))[0], moments_mean_sd(Pt[:, k].sum(-1))[0]
        with np.errstate(divide="ignore"):
            e0 = rms_t0 / np.sqrt(n)
        for ax, m, e, tot in zip(axes, (md, mt, mt - md), (ed, et, e0), (tot_d, tot_t, tot_t - tot_d)):
            ax.errorbar(xc[ok], m[ok], yerr=e[ok], fmt="o", color=CENT_COLORS[k], markersize=3.5, capsize=0,
                        linewidth=1.0, label=rf"{cent_label(k)}: mean {tot:+.2f} ns")
    for ax, lab in zip(axes, (r"$t - t_{0}$", TJET_LABEL, r"$t_{0} = \langle t \rangle - \langle t - t_{0} \rangle$")):
        ax.text(0.04, 0.96, lab, transform=ax.transAxes, ha="left", va="top", fontsize=16)
        ax.set_xlim(-0.3, 1.3)
        ax.set_xlabel(FRAC_LABEL)
        ax.set_ylabel("Mean [ns]")
        lo, hi = ax.get_ylim()
        ax.set_ylim(lo - 0.35 * (hi - lo), hi + 0.25 * (hi - lo))
        ax.legend(loc="lower left", fontsize=11, frameon=False)
    axes[2].text(0.97, 0.96, rf"runs pooled, x bins with $\geq {min_jets:g}$ jets" "\n"
                 rf"raw ${TIMING_PT[0]} \leq p_{{T}} < {TIMING_PT[1]}$ GeV, $|t - t_{{0}}| < 25$ ns",
                 transform=axes[2].transAxes, ha="right", va="top", fontsize=11)
    add_header(axes[2], runs, meta["r"], lumi)
    plt.subplots_adjust(left=0.06, right=0.98, bottom=0.12, top=0.93)
    save_fig(fig, output_path, save_pdf)
    plt.close(fig)


def tjet_checks(loaded, js, th2_dt, runs, r_tag, src, lumi, Pr, lines, jdir, args):
    """Is the run dependence of t - t0 from the jet or from the MBD t0, and how do satellite collisions enter?

    From the jet-time-alone hists (same jets as t - t0) and t vs t0 of one radius, at raw pT 15-40 GeV:
    T1 t vs t0 (satellite windows, slope), T2 the TH2s of t - t0 and t vs x, T3 the time distributions, T4 per run the
    means of t - t0, t and t0 and the satellite fractions, T5 (line-source radius only) the per-run lines refitted on t
    and their t0 part, T6 per class the mean t - t0, t and t0 vs x. Writes per_run_tjet_<r>_iternf.csv.

    js: the jet-time-alone hists summed over runs; th2_dt: the t - t0 TH2 at 15-40 GeV summed over runs;
    Pr: prepare_radius of this radius; lines: the per-run lines of the line source (fitted on t - t0)."""
    ref = loaded[runs[0]][r_tag]
    xc = centers(ref["p2_xedges"])
    t0e, tte = ref["t0_edges"], ref["tt0_tedges"]
    t0c, ttc = centers(t0e), centers(tte)
    name = f"{r_tag}_iternf"
    nK = len(CENT_CLASSES)

    Mdt = Pr["momk"]                                                   # (n_runs, 4, n_cent, n_x): t - t0
    Mt = np.stack([loaded[r][r_tag]["tjet_mom"] for r in runs])       # (n_runs, 4, n_cent, n_x): t, same jets
    T0 = np.stack([loaded[r][r_tag]["t0_mom"] for r in runs])         # (n_runs, 3, n_t0): jets, sum t, sum t^2
    if not np.array_equal(Mdt[:, 0], Mt[:, 0]):
        print(f"Warning: {r_tag}: the t and t - t0 profiles do not have the same jets per cell")

    # ---- pooled t vs t0: the satellite windows and the slope
    H = js["t0_2d"]
    n0, st, st2 = H.sum(1), H @ ttc, H @ ttc ** 2
    wins = []
    for key, lab, lo, hi, color in T0_WINDOWS:
        s = (t0c >= lo) & (t0c < hi)
        wins.append({"key": key, "label": lab, "sel": s, "color": color, "frac": n0[s].sum() / n0.sum(),
                     "t0": n0[s] @ t0c[s] / n0[s].sum(), "t": st[s].sum() / n0[s].sum()})
    c, mean, err, ok = t_given_t0(n0, st, st2, t0c, T0_REBIN, T0_MIN_BIN_JETS)
    sel = ok & (np.abs(c) < T0_FIT_RANGE)
    coef, cov = np.polyfit(c[sel], mean[sel], 1, w=1 / err[sel], cov="unscaled")
    chi2 = float((((mean[sel] - np.polyval(coef, c[sel])) / err[sel]) ** 2).sum())
    sm = ok & (np.abs(c) < T0_MAIN)
    fit = {"slope": coef[0], "c0": coef[1], "slope_err": np.sqrt(cov[0, 0]), "points": (c, mean, err, ok)}
    e_, m_, l_ = wins
    print(f"{r_tag}: t vs t0 at raw pT {TIMING_PT[0]}-{TIMING_PT[1]} GeV ({n0.sum():.0f} jets): "
          + "; ".join(f"{w['key']} {100 * w['frac']:.1f}%, <t0> {w['t0']:+.2f}, <t> {w['t']:+.2f}, "
                      f"<t - t0> {w['t'] - w['t0']:+.2f} ns" for w in wins))
    print(f"{r_tag}: slope of t vs t0: between the satellite peaks {(l_['t'] - e_['t']) / (l_['t0'] - e_['t0']):.3f}; "
          f"fit |t0| < {T0_FIT_RANGE} ns {fit['slope']:.3f} +- {fit['slope_err']:.3f} (chi2/ndf {chi2:.0f}/{sel.sum() - 2}); "
          f"main bunch only {np.polyfit(c[sm], mean[sm], 1, w=1 / err[sm])[0]:.3f}")

    # ---- per run: the means of t - t0 and t (profiles, exact), t0 (their difference), satellite fractions, slope
    md, _, ed = moments_mean_sd(Mdt.sum(axis=(2, 3)).T)
    mt, _, et = moments_mean_sd(Mt.sum(axis=(2, 3)).T)
    m0 = mt - md
    N = Mdt[:, 0].sum(axis=(1, 2))
    nr = T0[:, 0]
    with np.errstate(divide="ignore", invalid="ignore"):
        frac = {w["key"]: nr[:, w["sel"]].sum(1) / nr.sum(1) for w in wins}
        ms = wins[1]["sel"]
        t0_main = nr[:, ms] @ t0c[ms] / nr[:, ms].sum(1)
        rms_t0_run = np.sqrt(np.clip(nr @ t0c ** 2 / nr.sum(1) - (nr @ t0c / nr.sum(1)) ** 2, 0, None))
        e0 = rms_t0_run / np.sqrt(N)
        fs = np.abs(t0c) < T0_FIT_RANGE
        x, nn, sy = t0c[fs], nr[:, fs], T0[:, 1][:, fs]
        S0, Sx, Sxx, Sy, Sxy = nn.sum(1), nn @ x, nn @ x ** 2, sy.sum(1), sy @ x
        slope_run = (S0 * Sxy - Sx * Sy) / (S0 * Sxx - Sx ** 2)
    a, b = lines["a"], lines["b"]
    good = np.isfinite(a) & (N >= X_MIN_RUN_JETS)
    rho = lambda u, v: float(np.corrcoef(u[good], v[good])[0, 1])
    wmean = lambda v: float(np.average(v[good], weights=N[good]))
    asym = frac["late"] - frac["early"]
    print(f"{r_tag}: per run ({good.sum()} runs with a line and >= {X_MIN_RUN_JETS} jets), RMS over runs and "
          f"correlation with a_run, b_run ({src} lines, fitted on t - t0):")
    for lab, v in (("<t - t0>", md), ("<t>", mt), ("<t0>", m0), ("<t0> main bunch", t0_main),
                   ("early fraction", frac["early"]), ("late fraction", frac["late"]), ("late - early", asym),
                   ("slope t vs t0", slope_run)):
        print(f"    {lab:16s} RMS {np.std(v[good]):.4f}, rho(a) {rho(v, a):+.2f}, rho(b) {rho(v, b):+.2f}")
    print(f"{r_tag}: per run, <t0> vs late - early satellite fraction: rho {rho(m0, asym):+.2f}, "
          f"slope {np.polyfit(asym[good], m0[good], 1)[0]:+.2f} ns")

    rows_csv = [{"run": r, "jets": float(N[i]), "mean_dt": float(md[i]), "mean_dt_err": float(ed[i]),
                 "mean_t": float(mt[i]), "mean_t_err": float(et[i]), "mean_t0": float(m0[i]), "mean_t0_err": float(e0[i]),
                 "t0_main": float(t0_main[i]), "f_early": float(frac["early"][i]), "f_main": float(frac["main"][i]),
                 "f_late": float(frac["late"][i]), "slope_t_t0": float(slope_run[i]), "a": float(a[i]), "b": float(b[i])}
                for i, r in enumerate(runs)]

    # ---- the per-run lines refitted on t alone (line source only); their t0 part is lines(t) - lines(t - t0)
    lt = None
    res = {"m0": m0}
    if r_tag == src:
        mom_t = Mt.sum(axis=2)                                           # (n_runs, 4, n_x)
        pooled = mom_t.sum(0)
        pm, pe, neff = profile_from_moments(pooled)
        okf = Pr["in_range"] & np.isfinite(pm) & np.isfinite(pe) & (pe > 0) & (neff >= args.min_cell_entries)
        fit_t = fit_poly(xc, pm, pe, okf)
        print(f"{r_tag}: t alone: f(x) on the combined profile chi2/ndf = {fit_t['chi2']:.0f}/{fit_t['ndf']} "
              f"(t - t0: {Pr['fit']['chi2']:.0f}/{Pr['fit']['ndf']})")
        lt = run_lines({"mom_r": mom_t, "xc": xc, "f_x": np.polyval(fit_t["coef"], xc), "in_range": Pr["in_range"],
                        "sigma": np.sqrt(np.clip(pooled[2] / pooled[0] - pm ** 2, 0, None))}, runs)
        a0, b0 = lt["a"] - a, lt["b"] - b
        res["a0"] = a0
        for k, (vd, vt, v0) in (("a", (a, lt["a"], a0)), ("b", (b, lt["b"], b0))):
            print(f"{r_tag}: {k}_run RMS over runs: from t - t0 {np.std(vd[good]):.3f}, from t {np.nanstd(vt[good]):.3f}, "
                  f"t0 part {np.nanstd(v0[good]):.3f} ns; rho(t - t0, t) {rho(vd, vt):+.2f}, rho(t - t0, t0 part) "
                  f"{rho(vd, v0):+.2f}")
        print(f"{r_tag}: t0 part of a_run vs <t0>: rho {rho(a0, m0):+.2f}, vs late - early fraction {rho(a0, asym):+.2f}; "
              f"of b_run vs <t0>: rho {rho(b0, m0):+.2f}")
        for row, i in zip(rows_csv, range(len(runs))):
            row.update({"a_t": float(lt["a"][i]), "b_t": float(lt["b"][i])})
    keys = list(rows_csv[0].keys())
    write_csv(rows_csv, keys, jdir / f"per_run_tjet_{name}.csv")

    # ---- plots
    plot_t_vs_t0(t0e, tte, H, wins, fit, runs, r_tag, lumi, jdir / f"T1_t_vs_t0_{name}.png", args.save_pdf)
    pdt, pt_ = Mdt.sum(axis=(0, 2)), Mt.sum(axis=(0, 2))                # (4, n_x)
    plot_corr_th2(th2_dt, js["th2_tjet"], ref["t_xedges"], ref["t_yedges"], xc, profile_from_moments(pdt)[:2],
                  profile_from_moments(pt_)[:2], runs, r_tag, lumi, jdir / f"T2_{timing_hist_name(r_tag)}_tjet.png",
                  args.time_ylim or (-10, 10), args.save_pdf, labels=(r"$t - t_{0}$", TJET_LABEL), ylabel="Jet time [ns]")
    plot_time_dists(ref["t_yedges"], th2_dt.sum(0), js["th2_tjet"].sum(0), t0e, H, wins, runs, r_tag, lumi,
                    jdir / f"T3_time_dists_{name}.png", args.save_pdf)
    dev = lambda v: v - wmean(v)
    rt = lambda v: rf"RMS over runs {np.std(v[good]):.3f} ns; correlation with $a_{{\mathrm{{run}}}}$ {rho(v, a):+.2f}, " \
                   rf"with $b_{{\mathrm{{run}}}}$ {rho(v, b):+.2f}"
    rows = [{"ylabel": r"$\Delta\langle t - t_{0} \rangle$ [ns]", "series": [(dev(md), ed, DT_COLOR, None)], "text": rt(md)},
            {"ylabel": r"$\Delta\langle t \rangle$ [ns]", "series": [(dev(mt), et, T_COLOR, None)], "text": rt(mt)},
            {"ylabel": r"$\Delta\langle t_{0} \rangle$ [ns]", "series": [(dev(m0), e0, T0_COLOR, None)], "text": rt(m0)},
            {"ylabel": "Satellite fraction", "zero": False,
             "series": [(frac["early"], None, wins[0]["color"], wins[0]["label"]),
                        (frac["late"], None, wins[2]["color"], wins[2]["label"])],
             "text": rf"late $-$ early: correlation with $\langle t_{{0}} \rangle$ {rho(asym, m0):+.2f}, "
                     rf"with $a_{{\mathrm{{run}}}}$ {rho(asym, a):+.2f}"},
            {"ylabel": r"Slope of $t$ vs $t_{0}$", "series": [(slope_run, None, "black", None)], "zero": False,
             "text": rf"$|t_{{0}}| < {T0_FIT_RANGE}$ ns; median {np.nanmedian(slope_run[good]):.3f}, "
                     rf"RMS over runs {np.nanstd(slope_run[good]):.3f}"}]
    note = (rf"Leading jet, raw ${TIMING_PT[0]} \leq p_{{T}} < {TIMING_PT[1]}$ GeV; $\Delta$: minus the mean over runs; "
            rf"$a$, $b$ {lines_from_text(r_tag, src)}")
    plot_run_rows(runs, rows, r_tag, lumi, jdir / f"T4_run_means_{name}.png", note, args.save_pdf)
    if lt is not None:
        rl = lambda vd, v: rf"RMS over runs {np.nanstd(v[good]):.3f} ns; correlation with the $t - t_{{0}}$ line {rho(vd, v):+.2f}"
        rows = [{"ylabel": r"$a_{\mathrm{run}}$, $t - t_{0}$ [ns]", "series": [(a, lines["a_err"], DT_COLOR, None)],
                 "text": rf"RMS over runs {np.std(a[good]):.3f} ns (the correction)"},
                {"ylabel": r"$a_{\mathrm{run}}$, $t$ [ns]", "series": [(lt["a"], lt["a_err"], T_COLOR, None)],
                 "text": rl(a, lt["a"])},
                {"ylabel": r"$a_{\mathrm{run}}$, $t_{0}$ part [ns]", "series": [(a0, None, T0_COLOR, None)],
                 "text": rl(a, a0)},
                {"ylabel": r"$b_{\mathrm{run}}$, $t - t_{0}$ [ns]", "series": [(b, lines["b_err"], DT_COLOR, None)],
                 "text": rf"RMS over runs {np.std(b[good]):.3f} ns (the correction)"},
                {"ylabel": r"$b_{\mathrm{run}}$, $t$ [ns]", "series": [(lt["b"], lt["b_err"], T_COLOR, None)],
                 "text": rl(b, lt["b"])},
                {"ylabel": r"$b_{\mathrm{run}}$, $t_{0}$ part [ns]", "series": [(b0, None, T0_COLOR, None)],
                 "text": rl(b, b0)}]
        plot_run_rows(runs, rows, r_tag, lumi, jdir / f"T5_lines_t_{name}.png",
                      r"$a_{\mathrm{run}} + b_{\mathrm{run}}\,x$ fitted per run on $t - t_{0}$ and on $t$ (own $f(x)$); "
                      r"$t_{0}$ part = $t$ line $-$ $t - t_{0}$ line", args.save_pdf)
    pooled_rms_t0 = hist_mean_rms(n0, t0e)[1]
    plot_cent_tjet(xc, Mdt.sum(0), Mt.sum(0), pooled_rms_t0, runs, r_tag, lumi, jdir / f"T6_cent_profiles_{name}.png",
                   args.min_cell_entries, args.save_pdf)
    for k in range(nK):
        tot_d = moments_mean_sd(Mdt[:, :, k].sum(axis=(0, 2)))[0]
        tot_t = moments_mean_sd(Mt[:, :, k].sum(axis=(0, 2)))[0]
        print(f"{r_tag}: {cent_label(k)}: <t - t0> {tot_d:+.3f}, <t> {tot_t:+.3f}, <t0> {tot_t - tot_d:+.3f} ns")
    return res


# ---------------------------------------------------------------------------------------------------------------------
# The MBD t0 per run (timing/mbd_t0/): main-bunch and satellite peaks of the MB events (Event QA hT0Corrected),
# and the runs whose main peak is off zero (outliers). Not about jets, apart from the a_run correlations printed.
# ---------------------------------------------------------------------------------------------------------------------

T0_PEAK_HALF = 0.8          # ns: each peak is fitted with a Gaussian within this distance of its highest bin
T0_PEAK_MIN = 1000          # min entries in a t0 window for its peak to be fitted
# Min fraction of a satellite for its fitted position to be trusted: below it, the tail of the main peak pulls the
# fit toward the main peak (the apparent shift grows as the fraction falls; a joint 3-Gaussian fit does not fix it,
# the peaks are not Gaussian). Smaller satellites are drawn in gray and left out of the stability numbers.
T0_PEAK_MIN_FRAC = 0.16
# Good MBD t0 in a run: the fitted main-bunch peak within +-T0_MAIN_CUT of zero. The runs form one peak from -0.08 to
# +0.17 ns (median +0.035, robust RMS 0.012) and the mis-calibrated runs sit at -0.36 ns and below, so the cut falls
# in the gap. The satellites and the peak width are fitted and plotted, but not cut on.
T0_MAIN_CUT = 0.25
MB_T0_INFO = r"MB, $|z| < 10$ cm, 0–60%"     # the hT0Corrected selection of the Event QA


def fit_t0_peaks(h, e):
    """Per t0 window (T0_WINDOWS): the fraction of entries and a Gaussian fit of its peak (amplitude per bin, mu,
    error of mu, sigma), within T0_PEAK_HALF of the highest bin. A peak whose highest bin is closer than T0_PEAK_HALF
    to the window edge is not fitted (nan): it is the tail of the main peak, not a satellite peak."""
    c = centers(e)
    tot = h.sum()
    out = {}
    for key, _, lo, hi, _ in T0_WINDOWS:
        w = (c >= lo) & (c < hi)
        out.update({f"{key}_frac": h[w].sum() / tot if tot > 0 else np.nan, f"{key}_amp": np.nan,
                    f"{key}_mu": np.nan, f"{key}_mu_err": np.nan, f"{key}_sigma": np.nan})
        if h[w].sum() < T0_PEAK_MIN:
            continue
        mode = c[w][np.argmax(h[w])]
        if mode - lo < T0_PEAK_HALF or hi - mode < T0_PEAK_HALF:
            continue
        s = (np.abs(c - mode) < T0_PEAK_HALF) & (h > 0)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", OptimizeWarning)
                p, cov = curve_fit(lambda x, n, mu, sg: n * np.exp(-0.5 * ((x - mu) / sg) ** 2), c[s], h[s],
                                   p0=(h[s].max(), mode, 0.7), sigma=np.sqrt(h[s]), absolute_sigma=True)
        except (RuntimeError, ValueError):
            continue
        out.update({f"{key}_amp": float(p[0]), f"{key}_mu": float(p[1]), f"{key}_mu_err": float(np.sqrt(cov[1, 1])),
                    f"{key}_sigma": float(abs(p[2]))})
    return out


def run_ranges(runs, sel, max_gap=3):
    """The selected runs as ranges of neighbouring runs in the list (gaps of up to max_gap runs joined)."""
    idx = np.flatnonzero(sel)
    if not len(idx):
        return "none"
    groups, start = [], idx[0]
    for i, j in zip(idx[:-1], idx[1:]):
        if j - i > max_gap:
            groups.append((start, i))
            start = j
    groups.append((start, idx[-1]))
    return ", ".join(f"{runs[a]}" if a == b else f"{runs[a]}-{runs[b]} ({int(sel[a:b + 1].sum())})" for a, b in groups)


def plot_t0_dists(dists, runs, info, output_path, save_pdf=False):
    """M3: pooled MBD t0 distributions (unit area): dists [(label, values, edges, color)], info the selection text."""
    hep.style.use("ATLAS")
    fig, ax = plt.subplots(figsize=(10, 7))
    for lab, h, e, color in dists:
        c = centers(e)
        m, _ = hist_mean_rms(h, e)
        fr = "/".join(f"{100 * h[(c >= lo) & (c < hi)].sum() / h.sum():.1f}" for _, _, lo, hi, _ in T0_WINDOWS)
        gap_steps(ax, h / (h.sum() * np.diff(e)), e, color=color, linewidth=1.8,
                  label=rf"{lab}: early/main/late {fr}%, $\mu$ = {m:+.2f} ns")
    for x in (-T0_MAIN, T0_MAIN):
        ax.axvline(x, color="gray", linestyle=":", linewidth=1)
    ax.set_yscale("log")
    ax.set_ylim(1e-4, 30)
    ax.set_xlim(-10, 10)
    ax.set_xlabel(r"MBD $t_{0}$ [ns]")
    ax.set_ylabel(r"$\frac{1}{N}\frac{dN}{dt_{0}}$ [ns$^{-1}$]")
    ax.legend(loc="upper left", fontsize=15, frameon=True, facecolor="white", framealpha=0.9, edgecolor="none")
    ax.text(0.97, 0.76, f"{info}\n" + rf"main bunch: $|t_{{0}}| < {T0_MAIN:g}$ ns",
            transform=ax.transAxes, ha="right", va="top", fontsize=15)
    ax.text(0.0, 1.01, rf"{len(runs)} runs ({min(runs)}–{max(runs)})", transform=ax.transAxes, ha="left", va="bottom",
            fontsize=13)
    plt.subplots_adjust(left=0.13, right=0.97, bottom=0.12, top=0.94)
    save_fig(fig, output_path, save_pdf)
    plt.close(fig)


def plot_t0_outliers(title, panels, output_path, save_pdf=False):
    """M4: outlier runs of the MBD t0 main peak. Per panel, one run's t0 distribution (unit area) with its fitted peaks
    (dotted lines at the fitted positions, colored Gaussians) and a gray line at t0 = 0 to see the shift.
    panels: [(run, values, edges, fit dict, text)]."""
    hep.style.use("ATLAS")
    nc = len(panels)
    fig, axes = plt.subplots(1, nc, figsize=(6.4 * nc, 6), sharey=True, squeeze=False, gridspec_kw={"wspace": 0.05})
    for ax, (run, h, e, fit, text) in zip(axes[0], panels):
        N, bw = h.sum(), np.diff(e)
        gap_steps(ax, h / (N * bw), e, color="black", linewidth=1.6, label=f"Run {run}")
        ax.axvline(0, color="gray", linestyle="--", linewidth=1.3)
        for key, _, _, _, color in T0_WINDOWS:
            if np.isfinite(fit[f"{key}_mu"]):
                mu, sg = fit[f"{key}_mu"], fit[f"{key}_sigma"]
                ax.axvline(mu, color="black", linestyle=":", linewidth=1.2)
                xs = np.linspace(mu - 2 * sg, mu + 2 * sg, 100)
                ax.plot(xs, fit[f"{key}_amp"] * np.exp(-0.5 * ((xs - mu) / sg) ** 2) / (N * bw[0]), "-",
                        color=color if color != "black" else "#ff7f00", linewidth=1.8)
        ax.text(0.03, 0.97, text, transform=ax.transAxes, ha="left", va="top", fontsize=14,
                zorder=5, bbox=dict(facecolor="white", alpha=0.85, edgecolor="none", pad=2))
        ax.legend(loc="lower center", fontsize=14, frameon=False)
        ax.set_yscale("log")
        ax.set_ylim(1e-6, 3000)                                   # room below the distributions for the legend
        ax.set_xlim(-10, 10)
        ax.set_xticks([-7.5, -5, -2.5, 0, 2.5, 5, 7.5])            # no clash of the edge labels between the panels
        ax.set_xlabel(r"MBD $t_{0}$ [ns]")
    axes[0, 0].set_ylabel(r"$\frac{1}{N}\frac{dN}{dt_{0}}$ [ns$^{-1}$]")
    axes[0, 0].text(0.0, 1.02, f"{title}\n{MB_T0_INFO}; " + rf"Gaussian fits within $\pm${T0_PEAK_HALF:g} ns of each peak "
                    "(dashed gray: $t_{0}$ = 0); $\Delta\mu$: satellite $-$ main-peak $\mu$", transform=axes[0, 0].transAxes, ha="left", va="bottom", fontsize=15)
    plt.subplots_adjust(left=0.07 if nc > 1 else 0.14, right=0.98, bottom=0.13, top=0.88)
    save_fig(fig, output_path, save_pdf)
    plt.close(fig)


def plot_t0_main(v, cut, notes, runs, lumi, output_path, save_pdf=False):
    """M5: the per-run distribution of the fitted main-bunch t0 peak position (not median subtracted), with the cut at
    +-cut ns and the outlier runs it flags filled red."""
    hep.style.use("ATLAS")
    fig, ax = plt.subplots(figsize=(10, 7))
    bw = 0.01
    lo = min(np.min(v), -2 * cut) - 5 * bw
    hi = max(np.max(v), 2 * cut) + 5 * bw
    edges = np.arange(np.floor(lo / bw) * bw, hi + bw, bw)
    h, _ = np.histogram(v, edges)
    inside = np.abs(v) <= cut
    ax.stairs(h, edges, color="black", linewidth=1.4, label=f"{len(v)} runs")
    hx, _ = np.histogram(v[~inside], edges)
    if hx.any():
        ax.stairs(hx, edges, color="#e41a1c", fill=True, alpha=0.5, label=f"outliers: {(~inside).sum()} runs")
    for x in (-cut, cut):
        ax.axvline(x, color="#e41a1c", linestyle="--", linewidth=1.5)
    ax.axvline(0, color="gray", linestyle=":", linewidth=1)
    ax.set_yscale("log")
    ax.set_ylim(0.5, h.max() * 40)
    ax.set_xlim(edges[0], edges[-1])
    ax.set_xlabel(r"Main-bunch MBD $t_{0}$ peak $\mu$ [ns]")
    ax.set_ylabel(f"Runs / {bw * 1000:g} ps")
    rob = 1.4826 * np.median(np.abs(v - np.median(v)))
    ax.legend(loc="upper right", fontsize=15, frameon=False)
    ax.text(0.03, 0.97, rf"Outlier: $|\mu| > {cut:g}$ ns; {(~inside).sum()} of {len(v)} runs" + "\n"
            + rf"median {np.median(v):+.3f} ns, robust RMS {rob:.3f} ns" + "\n"
            + rf"good runs: [{np.min(v[inside]):+.3f}, {np.max(v[inside]):+.3f}] ns"
            + ("".join("\n" + n for n in notes)) + "\n"
            + rf"{MB_T0_INFO};" "\n" rf"Gaussian fit within $\pm${T0_PEAK_HALF:g} ns of the peak",
            transform=ax.transAxes, ha="left", va="top", fontsize=14, zorder=5,
            bbox=dict(facecolor="white", alpha=0.85, edgecolor="none", pad=3))
    add_header(ax, runs, None, lumi)
    plt.subplots_adjust(left=0.12, right=0.97, bottom=0.12, top=0.94)
    save_fig(fig, output_path, save_pdf)
    plt.close(fig)


def mbd_t0_checks(t0_qa, loaded, runs, src, lumi, lines, tj, mdir, args):
    """Is the main MBD t0 peak at zero, are the satellite peaks stable, and which runs have the main peak off zero?

    Per run, Gaussian fits of the main-bunch and satellite peaks (fit_t0_peaks) of the MB t0 (Event QA hT0Corrected,
    same production as the trees). Prints their stability and correlation with a_run (and its t0 part, tj). A run is
    good when its main peak is within +-T0_MAIN_CUT of zero, else an outlier; nothing else is cut on. Writes
    per_run_mbd_t0.csv, the good and outlier run lists and plots M1-M5. The leading-jet
    events' main peak (h2LeadJetTimeT0 of the line source) is only printed against the MB one."""
    runs_qa = [r for r in runs if r in t0_qa]
    if not runs_qa:
        print("No hT0Corrected in the Event QA files: skipping the MBD t0 checks")
        return
    if len(runs_qa) < len(runs):
        print(f"Warning: hT0Corrected for only {len(runs_qa)}/{len(runs)} runs; the MBD t0 checks use those")
    runs = runs_qa
    fits = {r: fit_t0_peaks(*t0_qa[r]) for r in runs}
    arr = lambda k, f=fits: np.array([f[r][k] for r in runs], float)
    robust = lambda v: 1.4826 * np.nanmedian(np.abs(v - np.nanmedian(v)))
    pct = lambda v, q: np.nanpercentile(v, q)
    li = {r: i for i, r in enumerate(sorted(loaded))}
    a = np.array([lines["a"][li[r]] for r in runs])
    jets = np.array([lines["jets"][li[r]] for r in runs])
    a0 = np.array([tj["a0"][li[r]] for r in runs]) if tj and "a0" in tj else None
    corr = lambda u, v: float(np.corrcoef(u[np.isfinite(u) & np.isfinite(v)], v[np.isfinite(u) & np.isfinite(v)])[0, 1])
    events = np.array([t0_qa[r][0].sum() for r in runs])

    # ---- the peaks
    m, sm = arr("main_mu"), arr("main_sigma")
    print(f"MBD t0, MB events (hT0Corrected), {len(runs)} runs: main peak mu median {np.nanmedian(m):+.3f} ns, robust RMS "
          f"over runs {robust(m):.3f}, 1-99% [{pct(m, 1):+.3f}, {pct(m, 99):+.3f}], median fit error "
          f"{np.nanmedian(arr('main_mu_err')):.4f}; sigma median {np.nanmedian(sm):.3f} ns, 1-99% [{pct(sm, 1):.3f}, "
          f"{pct(sm, 99):.3f}]; corr with a_run {corr(m, a):+.2f}" + (f", with its t0 part {corr(m, a0):+.2f}" if a0 is not None else ""))
    spacing, meas, dev = {}, {}, {}
    for k in ("early", "late"):
        dv, fr = arr(f"{k}_mu") - m, arr(f"{k}_frac")
        meas[k] = np.isfinite(dv) & (fr > T0_PEAK_MIN_FRAC)
        spacing[k] = float(np.median(dv[meas[k]]))
        dev[k] = dv - spacing[k]
        small = np.isfinite(dv) & ~meas[k]
        print(f"    {k} - main, {meas[k].sum()} runs with fraction > {T0_PEAK_MIN_FRAC}: median {spacing[k]:+.3f} ns, "
              f"robust RMS {robust(dv[meas[k]]):.3f}, 5-95% [{pct(dv[meas[k]], 5):+.3f}, {pct(dv[meas[k]], 95):+.3f}]; "
              f"fraction median {np.nanmedian(fr):.3f} [1-99%: {pct(fr, 1):.3f}, {pct(fr, 99):.3f}]; {small.sum()} runs "
              f"at <= {T0_PEAK_MIN_FRAC} (position not trusted, median {k} - main there "
              f"{np.median(dv[small]) if small.any() else np.nan:+.3f} ns)")
    asym = arr("late_frac") - arr("early_frac")
    if a0 is not None:
        print(f"    late - early fraction: corr with a_run {corr(asym, a):+.2f}, with its t0 part {corr(asym, a0):+.2f}")
    if all("t0_mom" in loaded[r][src] for r in runs):
        jm = np.array([fit_t0_peaks(loaded[r][src]["t0_mom"][0], loaded[r][src]["t0_edges"])["main_mu"] for r in runs])
        print(f"    main peak of the leading-jet events ({src}, raw pT {TIMING_PT[0]}-{TIMING_PT[1]} GeV) - MB: median "
              f"{np.nanmedian(jm - m):+.3f} ns, robust RMS over runs {robust(jm - m):.3f}, corr {corr(jm, m):+.2f}")

    # ---- good MBD t0: the main-bunch peak within +-T0_MAIN_CUT of zero; everything else is only shown
    good = np.isfinite(m) & (np.abs(m) <= T0_MAIN_CUT)
    out = ~good
    print(f"MBD t0 main peak within +-{T0_MAIN_CUT} ns of zero: {good.sum()}/{len(runs)} runs; good runs span "
          f"[{np.min(m[good]):+.3f}, {np.max(m[good]):+.3f}] ns"
          + (f", nearest outlier {m[out][np.argmin(np.abs(m[out]))]:+.3f} ns" if out.any() else ""))
    print(f"    outliers: {out.sum()} runs: {run_ranges(runs, out)}")
    tot_j = np.nansum(jets)
    print(f"    good: {np.nansum(jets[good]) / tot_j:.3f} of the 15-40 GeV leading jets, {events[good].sum() / events.sum():.3f} "
          f"of the MB events; a_run RMS over the good runs {np.nanstd(a[good]):.3f} ns (all runs {np.nanstd(a):.3f})"
          + (f", its t0 part {np.nanstd(a0[good]):.3f} ({np.nanstd(a0):.3f})" if a0 is not None else ""))
    if out.any():
        print("    outliers: main mu, a_run" + (", a_run from t alone (a_run + t0 part)" if a0 is not None else "") + ": "
              + ", ".join(f"{runs[i]} {m[i]:+.2f}/{a[i]:+.2f}" + (f"/{a[i] + a0[i]:+.2f}" if a0 is not None else "")
                          for i in np.flatnonzero(out)))

    # ---- outputs
    rows_csv = [{"run": r, "events": float(events[i]),
                 **{f"{k}_{q}": fits[r][f"{k}_{q}"] for k, *_ in T0_WINDOWS for q in ("frac", "mu", "mu_err", "sigma")},
                 "early_spacing": float(arr("early_mu")[i] - m[i]), "late_spacing": float(arr("late_mu")[i] - m[i]),
                 "early_trusted": int(meas["early"][i]), "late_trusted": int(meas["late"][i]),
                 "good": int(good[i])} for i, r in enumerate(runs)]
    write_csv(rows_csv, list(rows_csv[0].keys()), mdir / "per_run_mbd_t0.csv")
    for name, s in (("mbd_t0_good_runs.txt", good), ("mbd_t0_outlier_runs.txt", out)):
        (mdir / name).write_text("".join(f"{r}\n" for r, keep in zip(runs, s) if keep))
        print(f"Saved: {mdir / name} ({s.sum()} runs)")

    # M1 (all rows) and M2 (the peak positions only), per run
    note = rf"{MB_T0_INFO}; Gaussian fits within $\pm${T0_PEAK_HALF:g} ns of each peak"

    def position_rows(sel, good_only):
        """The peak-position rows (main, early - main, late - main) for the runs in sel."""
        mm = m[sel]
        main = [(np.where(good[sel], mm, np.nan), arr("main_mu_err")[sel], "black", None)]
        if not good_only:
            main.append((np.where(good[sel], np.nan, mm), None, "#e41a1c", rf"outlier, $|\mu| > {T0_MAIN_CUT:g}$ ns"))
        what = "good runs only" if good_only else "all runs"
        out_ = [{"ylabel": r"Main peak $\mu$ [ns]", "series": main,
                 "ylim": ((-0.3, 0.4) if good_only else (min(-0.5, np.nanmin(mm) - 0.15), max(0.6, np.nanmax(mm) + 0.35))),
                 "text": note + "\n" + rf"{what}: median {np.nanmedian(mm):+.3f}, robust RMS {robust(mm):.3f} ns"}]
        for k, color in (("early", T0_WINDOWS[0][4]), ("late", T0_WINDOWS[2][4])):
            dv, ms_ = (arr(f"{k}_mu") - m)[sel], meas[k][sel]
            out_.append({"ylabel": rf"{k.capitalize()} $-$ main $\mu$ [ns]", "zero": False,
                         "series": [(np.where(ms_, dv, np.nan), None, color, rf"fraction $>$ {T0_PEAK_MIN_FRAC:g}"),
                                    (np.where(ms_, np.nan, dv), None, "gray", rf"fraction $\leq$ {T0_PEAK_MIN_FRAC:g}")],
                         "ylim": ((spacing[k] - 0.3, spacing[k] + 0.3) if good_only else (spacing[k] - 0.6, spacing[k] + 0.6)),
                         "text": rf"fraction $>$ {T0_PEAK_MIN_FRAC:g}: median {np.median(dv[ms_]):+.3f}, robust RMS "
                                 rf"{robust(dv[ms_]):.3f} ns; gray: small satellite, fit pulled by the main peak"})
        return out_

    every = np.ones(len(runs), bool)
    rows = position_rows(every, False)
    top = dict(rows[0], hlines=[-T0_MAIN_CUT, T0_MAIN_CUT],
               text=note + "\n" + rf"red dashed: outlier cut $\pm${T0_MAIN_CUT:g} ns; {out.sum()} of {len(runs)} runs outside")
    plot_run_rows(runs, [top], None, lumi, mdir / "M2_t0_main_peak_runs.png", None, args.save_pdf)
    plot_run_rows(runs, rows, None, lumi, mdir / "M2_t0_peak_positions_runs.png", None, args.save_pdf)
    plot_run_rows([r for r, s_ in zip(runs, good) if s_], position_rows(good, True), None, lumi,
                  mdir / "M2_t0_peak_positions_runs_good.png", None, args.save_pdf)
    rows.append({"ylabel": "Satellite fraction", "zero": False,
                 "series": [(arr(f"{k}_frac"), None, w[4], w[1]) for k, w in zip(("early", "late"), T0_WINDOWS[::2])]})
    rows.append({"ylabel": r"Peak $\sigma$ [ns]", "zero": False,
                 "series": [(arr(f"{w[0]}_sigma"), None, w[4], w[0]) for w in T0_WINDOWS]})
    plot_run_rows(runs, rows, None, lumi, mdir / "M1_t0_peaks_runs.png", None, args.save_pdf)

    # M3: pooled, all runs and the good runs
    e = t0_qa[runs[0]][1]
    h_all = sum(t0_qa[r][0] for r in runs)
    h_gd = sum(t0_qa[r][0] for r, s in zip(runs, good) if s)
    for fname, lab, hh, col in (("all", f"All {len(runs)} runs", h_all, "black"),
                                ("good", f"{good.sum()} good runs", h_gd, "#2166ac")):
        plot_t0_dists([(lab, hh, e, col)], runs, MB_T0_INFO, mdir / f"M3_t0_dist_{fname}.png", args.save_pdf)

    # M4: the three outlier runs with the largest main-peak shift
    if out.any():
        order = np.flatnonzero(out)[np.argsort(-np.abs(m[out]))][:3]
        panels = []
        for i in order:
            fit = fits[runs[i]]
            sp = lambda k: (rf"{fit[f'{k}_mu'] - fit['main_mu']:+.2f} ns" if np.isfinite(fit[f"{k}_mu"]) else "not fitted")
            text = (rf"Run {runs[i]}, ${sci(events[i])}$ events" "\n"
                    rf"main: $\mu$ = {fit['main_mu']:+.3f} ns, $\sigma$ = {fit['main_sigma']:.2f} ns" "\n"
                    rf"early: $\Delta\mu$ = {sp('early')} ({100 * fit['early_frac']:.0f}% of events)" "\n"
                    rf"late: $\Delta\mu$ = {sp('late')} ({100 * fit['late_frac']:.0f}% of events)")
            panels.append((runs[i], t0_qa[runs[i]][0], t0_qa[runs[i]][1], fit, text))
        print("MBD t0: M4 shows the worst outliers " + ", ".join(f"{runs[i]} ({m[i]:+.3f} ns)" for i in order))
        plot_t0_outliers(rf"Largest main-peak shifts of the {out.sum()} outlier runs ($|\mu| > {T0_MAIN_CUT:g}$ ns)", panels,
                         mdir / "M4_t0_outliers.png", args.save_pdf)

    # M5: the main-peak position per run, with the cut and the outliers it flags
    plot_t0_main(m[np.isfinite(m)], T0_MAIN_CUT, [textwrap.fill(f"outliers: {run_ranges(runs, out)}", 44)] if out.any() else [],
                 runs, lumi, mdir / "M5_t0_main_peak.png", args.save_pdf)


# ---------------------------------------------------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run-combined iternf raw jet pT, and the leading-jet timing model t = f_c(x) + a_run + b_run x "
                    "(quartic per centrality class, per-run line) at raw pT 15-40 GeV, built step by step."
    )
    parser.add_argument("-f", "--file", type=Path,
                        default=Path("/sphenix/user/anarde/sEPD-Study/jetAnav3/11-05-26/test.list"),
                        help="List of Jet-Anav3 merged ROOT files named <run>.root (default: %(default)s)")
    parser.add_argument("-l", "--lumi-list", type=Path,
                        default=Path("/sphenix/user/anarde/sEPD-Study/f4a_bkgsub/10-30-26/test.list"),
                        help="List of Event QA ROOT files named <run>.root with hLuminosity; the label is omitted "
                             "unless every run has one (default: %(default)s)")
    parser.add_argument("-o", "--output-dir", type=Path,
                        default=Path("/sphenix/user/anarde/sEPD-Study/plots/Jet-Ana/iternf_timing"),
                        help="Output directory (default: %(default)s)")
    parser.add_argument("--runs", nargs="*", type=int, help="Specific run number(s) to combine (default: all runs in the list)")
    parser.add_argument("--radii", nargs="+", choices=list(RADII), default=list(RADII), help="Jet radii to plot (default: both)")
    parser.add_argument("--line-source", choices=list(RADII), default="r03",
                        help="Radius the per-run lines a, b are measured from and applied to all radii (default: %(default)s)")
    parser.add_argument("--min-cell-entries", type=float, default=50,
                        help="Minimum jets in an x bin of a run-combined profile for it to be fitted and drawn "
                             "(default: %(default)s)")
    parser.add_argument("--example-runs", nargs="+", type=int, default=list(EXAMPLE_RUNS),
                        help="Runs whose profiles show the run dependence (default: %(default)s)")
    parser.add_argument("--per-event", action="store_true", help="Normalize the raw pT spectrum per event and per GeV")
    parser.add_argument("--pt-max", type=float, default=None,
                        help="x-axis max for the raw pT plot (default: auto, plus a _pt60 version if auto > 60)")
    parser.add_argument("--time-ylim", nargs=2, type=float, default=None, metavar=("TMIN", "TMAX"),
                        help="y-axis range of the 2D timing plot (default: full histogram range)")
    parser.add_argument("--zoom-runs", nargs=2, type=int, action="append", default=[], metavar=("RUNMIN", "RUNMAX"),
                        help="Also make the per-run line plot zoomed to this run range, vs run number (repeatable)")
    parser.add_argument("--no-pt", action="store_true", help="Skip the raw pT plot")
    parser.add_argument("--no-timing", action="store_true", help="Skip the timing model")
    parser.add_argument("--no-x-dist", action="store_true", help="Skip the OHCal fraction distributions (timing/ohcal_frac/)")
    parser.add_argument("--x-run-blocks", nargs="*", type=int, default=list(X_RUN_BLOCKS),
                        help="First runs of the blocks of runs whose x distributions are compared (default: %(default)s)")
    parser.add_argument("-j", "--jobs", type=int, default=min(8, os.cpu_count() or 1),
                        help="Parallel workers for reading the runs (default: %(default)s)")
    parser.add_argument("--save-pdf", action="store_true", help="Save PDF versions in addition to PNG")
    parser.add_argument("--cache-dir", type=Path, default=None,
                        help="Where the input cache is kept (default: <output-dir>/cache). The cache holds the per-run "
                             "inputs read from the ROOT files and the luminosity; it is reused only if every input file "
                             "is unchanged (path, size, mtime) and the same radii are read")
    parser.add_argument("--no-cache", action="store_true", help="Neither read nor write the input cache")
    parser.add_argument("--refresh-cache", action="store_true", help="Reread the ROOT files and overwrite the cache")
    args = parser.parse_args()
    tick = StageTimer()

    run_files = parse_list_file(args.file)
    if not run_files:
        print(f"Error: No ROOT files found in {args.file}")
        sys.exit(1)
    runs_req = sorted(args.runs) if args.runs else sorted(run_files)
    pt_radii = () if args.no_pt else tuple(args.radii)
    t_radii = () if args.no_timing else tuple(dict.fromkeys([args.line_source, *args.radii]))
    tasks = []
    for run in runs_req:
        path = run_files.get(run)
        if path is None or not path.exists():
            print(f"Warning: No file for run {run}, skipping")
            continue
        tasks.append((run, path, pt_radii, t_radii))

    lumi_files = parse_list_file(args.lumi_list) if args.lumi_list.exists() else {}
    lumi_paths = [lumi_files[r] for r, *_ in tasks if r in lumi_files and lumi_files[r].exists()]

    cache_path = None
    if not args.no_cache:
        cache_dir = args.cache_dir if args.cache_dir is not None else args.output_dir / "cache"
        cache_path = cache_dir / f"inputs_{cache_key(tasks, pt_radii, t_radii, lumi_paths)}.pkl"
    if cache_path is not None and cache_path.exists() and not args.refresh_cache:
        print(f"Reading the inputs from the cache {cache_path}")
        with cache_path.open("rb") as f:
            cached = pickle.load(f)
        loaded, t_sum, lumi, c_sum = cached["loaded"], cached["t_sum"], cached["lumi"], cached["c_sum"]
        j_sum, t0_qa = cached["j_sum"], cached["t0_qa"]
        runs = sorted(loaded)
    else:
        # Stream the results so only one full 2D per radius (the sum) is kept in memory
        n_jobs = max(1, min(args.jobs, len(tasks)))
        print(f"Reading {len(tasks)} runs with {n_jobs} worker(s)")
        loaded, t_sum, c_sum, j_sum = {}, {}, {}, {}
        ex = concurrent.futures.ProcessPoolExecutor(max_workers=n_jobs) if n_jobs > 1 else None
        for run, d in (ex.map(load_run, tasks, chunksize=8) if ex else map(load_run, tasks)):
            if d is None:
                continue
            for r_tag in t_radii:
                t_sum[r_tag] = t_sum.get(r_tag, 0) + d[r_tag].pop("t_values")
                cs = c_sum.setdefault(r_tag, {"n_runs": 0})
                if "sums" in d[r_tag]:                     # the timing-corrected hists, summed over runs
                    cs["n_runs"] += 1
                    for k, v in d[r_tag].pop("sums").items():
                        cs[k] = cs.get(k, 0) + v
                js = j_sum.setdefault(r_tag, {"n_runs": 0})
                if "sums_tjet" in d[r_tag]:                # the jet-time-alone hists, summed over runs
                    js["n_runs"] += 1
                    for k, v in d[r_tag].pop("sums_tjet").items():
                        js[k] = js.get(k, 0) + v
            loaded[run] = d
        if ex:
            ex.shutdown()
        if not loaded:
            print("Error: no runs loaded")
            sys.exit(1)
        runs = sorted(loaded)

        # Luminosity: only label it if every combined run has one, so the sum is not silently partial
        lumi = None
        if lumi_files:
            missing = [r for r in runs if r not in lumi_files or not lumi_files[r].exists()]
            if missing:
                print(f"Warning: no luminosity file for {len(missing)} run(s) (e.g. {missing[:5]}), omitting the luminosity label")
            else:
                lumis = [load_luminosity(lumi_files[r]) for r in runs]
                lumi = None if any(l is None for l in lumis) else float(sum(lumis))
        else:
            print(f"Warning: no luminosity list at {args.lumi_list}, omitting the luminosity label")

        # The MB MBD t0 distribution per run, from the same Event QA files (hT0Corrected)
        t0_qa = {}
        qa_tasks = [(r, lumi_files[r]) for r in runs if r in lumi_files and lumi_files[r].exists()] if t_radii else []
        if qa_tasks:
            with concurrent.futures.ProcessPoolExecutor(max_workers=n_jobs) as ex:
                t0_qa = {r: v for r, v in ex.map(load_t0_qa, qa_tasks, chunksize=16) if v is not None}
            print(f"Read hT0Corrected for {len(t0_qa)}/{len(runs)} runs")

        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = cache_path.with_suffix(".tmp")
            with tmp.open("wb") as f:
                pickle.dump({"loaded": loaded, "t_sum": t_sum, "lumi": lumi, "c_sum": c_sum, "j_sum": j_sum,
                             "t0_qa": t0_qa}, f,
                            protocol=pickle.HIGHEST_PROTOCOL)
            tmp.replace(cache_path)                       # atomic: an interrupted write never leaves a broken cache
            print(f"Saved the input cache: {cache_path} ({cache_path.stat().st_size / 1e6:.0f} MB)")
    tick(f"inputs ({len(runs)} runs)")

    n_events = sum(loaded[r]["n_events"] for r in runs)
    out = args.output_dir

    for r_tag in pt_radii:
        pt_vals = sum(loaded[r][r_tag]["pt_values"] for r in runs)
        pt_vars = sum(loaded[r][r_tag]["pt_variances"] for r in runs)
        print(f"{r_tag}: combined raw pT jets = {pt_vals.sum():.0f}, N_evt = {n_events:.0f}")
        plot_raw_pt(pt_vals, pt_vars, loaded[runs[0]][r_tag]["pt_edges"], n_events, runs, r_tag, lumi, args.per_event,
                    out / "pt" / f"combined_{pt_hist_name(r_tag)}.png", pt_max=args.pt_max, save_pdf=args.save_pdf)

    if args.no_timing:
        return

    # ---- The model
    src = args.line_source
    tdir, bdir = out / "timing", out / "timing" / "baseline"
    ex_runs = [r for r in args.example_runs if r in loaded]
    if len(ex_runs) < len(args.example_runs):
        print(f"Warning: example run(s) not loaded, skipped: {sorted(set(args.example_runs) - set(ex_runs))}")
    order = [src] + [r for r in args.radii if r != src]
    P = {r_tag: prepare_radius(loaded, runs, r_tag, args) for r_tag in order}
    lines = run_lines(P[src], runs)
    M = {r_tag: apply_model(P[r_tag], lines, runs, r_tag, src, ex_runs, args) for r_tag in args.radii}
    tick("model")

    # the per-run lines: table and plot
    good = np.isfinite(lines["a"])
    rows = [{"run": run, "status": "ok" if good[i] else "no_line", "jets": float(lines["jets"][i]),
             "n_bins": int(lines["n_bins"][i]), "a": float(lines["a"][i]), "b": float(lines["b"][i]),
             "a_err": float(lines["a_err"][i]), "b_err": float(lines["b_err"][i]),
             "cov_ab": float(lines["cov_ab"][i]), "chi2": float(lines["chi2"][i])} for i, run in enumerate(runs)]
    write_csv(rows, list(rows[0].keys()), tdir / f"per_run_lines_{src}_iternf.csv")
    rows_ok = [r for r in rows if r["status"] == "ok"]
    note_l = (rf"$a_{{\mathrm{{run}}}} + b_{{\mathrm{{run}}}}\,x$ with $f(x)$ fixed, raw $p_{{T}}$ "
              rf"{TIMING_PT[0]}–{TIMING_PT[1]} GeV")
    lumi_l = lumi if good.all() else None                  # the luminosity label only when every run has a line
    plot_lines_runs(rows_ok, src, lumi_l, bdir / f"baseline_lines_{src}_iternf.png", note_l, args.save_pdf)
    for lo, hi in args.zoom_runs:
        plot_lines_runs(rows_ok, src, lumi_l, bdir / f"baseline_lines_{src}_iternf_runs{lo}_{hi}.png", note_l,
                        args.save_pdf, run_range=(lo, hi))

    for r_tag in args.radii:
        write_model(tdir / f"model_{r_tag}_iternf.json", r_tag, src, M[r_tag], P[r_tag], args)
    write_correction_csv(tdir / "timing_correction_iternf.csv", M, src)

    # ---- The plots, with y ranges shared by the radii
    for r_tag in args.radii:
        Pr = P[r_tag]
        plot_baseline(t_sum[r_tag], Pr["ref"]["t_xedges"], Pr["ref"]["t_yedges"], Pr["xc"], Pr["prof"], Pr["fit"], runs,
                      r_tag, lumi, bdir / f"baseline_{timing_hist_name(r_tag)}.png", time_ylim=args.time_ylim,
                      save_pdf=args.save_pdf)
    rlim = 0.5
    if ex_runs:
        ylim, rlim = ranges([(P[r]["xc"], P[r]["fit"], M[r]["per_run"], M[r]["x_run"]) for r in args.radii])
        for r_tag in args.radii:
            Pr, Mr = P[r_tag], M[r_tag]
            name = timing_hist_name(r_tag)
            plot_profiles(Pr["xc"], Pr["fit"], Mr["per_run"], Mr["x_run"], r_tag, ylim, rlim,
                          bdir / f"baseline_runs_{name}.png", save_pdf=args.save_pdf)
            plot_profiles(Pr["xc"], Pr["fit"], Mr["per_run"], Mr["x_run"], r_tag, ylim, rlim,
                          bdir / f"baseline_runs_lines_{name}.png", mode="line", res_label=r"Run $-$ model [ns]",
                          note=r"lines: $f(x) + a_{\mathrm{run}} + b_{\mathrm{run}}\,x$, $a$ and $b$ "
                               + lines_from_text(r_tag, src), save_pdf=args.save_pdf)
    ylim_c, rlim_c = ranges([(P[r]["xc"], P[r]["fit"], M[r]["per_cent"], M[r]["x_cent"]) for r in args.radii])
    rlim_c = max(rlim_c, rlim)                               # at least the run plots' residual range
    for r_tag in args.radii:
        Pr, Mr = P[r_tag], M[r_tag]
        name = timing_hist_name(r_tag)
        lines_note = rf"$a_{{\mathrm{{run}}}}, b_{{\mathrm{{run}}}}$: fitted per run with $f(x)$ fixed" + (
            "" if r_tag == src else rf" on $R = {RADII[src]['r']:g}$")
        plot_profiles(Pr["xc"], Pr["fit"], Mr["per_cent"], Mr["x_cent"], r_tag, ylim_c, rlim_c,
                      bdir / f"baseline_cent_{name}.png", header=(runs, lumi), colors=CENT_COLORS + ("black",),
                      ylabel=CORR_LABEL, res_label=r"Class $-\ f(x)$ [ns]", note=POINTS_NOTE + "\n" + lines_note,
                      save_pdf=args.save_pdf)
        plot_profiles(Pr["xc"], Pr["fit"], Mr["per_cent"][:len(CENT_CLASSES)], Mr["x_cent"], r_tag, ylim_c, rlim_c,
                      bdir / f"baseline_cent_fc_{name}.png", mode="own", header=(runs, lumi), colors=CENT_COLORS,
                      ylabel=CORR_LABEL, res_label=r"Class $-\ f_{c}(x)$ [ns]",
                      note=POINTS_NOTE + "\n" + rf"lines: $f_{{c}}(x)$, degree {FIT_DEG}, fitted per class",
                      save_pdf=args.save_pdf)

    # pT residuals of the model: residual range from data - <f_c(x)>, at least the centrality plots' range
    res_p = [np.abs(e["mean"] - e["model"])[e["ok"] & (e["err"] < 0.1)] for r in args.radii for e in M[r]["per_pt"]]
    rlim_p = max(rlim_c, 1.25 * max((r.max() for r in res_p if len(r)), default=0.5))
    low_p = min(np.nanmin(e["mean"][e["ok"]]) for r in args.radii for e in M[r]["per_pt"])
    ylim_p = (min(ylim_c[0], low_p - 0.5), ylim_c[1])
    cent_names = [cent_label(k) for k in range(len(CENT_CLASSES))]
    pt_names = [pt_label(pt) for pt in PT_SPLIT]
    for r_tag in args.radii:
        Pr, Mr = P[r_tag], M[r_tag]
        name = timing_hist_name(r_tag)
        fc_fn = [clamped(fk) for fk in Mr["fc"]]
        plot_profiles(Pr["xc"], Pr["fit"], Mr["per_pt"], Mr["x_pt"], r_tag, ylim_p, rlim_p,
                      bdir / f"baseline_pt_{name}.png", mode="model", header=(runs, lumi), colors=PT_SPLIT_COLORS,
                      ylabel=CORR_LABEL, res_label=r"Data $-\ \langle f_{c}(x)\rangle$ [ns]",
                      note=r"points: $t - t_{0} - a_{\mathrm{run}} - b_{\mathrm{run}}\,x$, all runs and classes combined" "\n"
                           r"lines: $f_{c}(x)$ averaged over the jets of each $p_{T}$ bin", save_pdf=args.save_pdf)
        plot_grid(Pr["xc"], Mr["by_class"], cent_names, pt_names, PT_SPLIT_COLORS, Mr["x_pt"], r_tag, rlim_p, runs, lumi,
                  bdir / f"baseline_pt_fcx_{name}.png", panel_fn=fc_fn,
                  panel_curve_names=[rf"$f_{{c}}(x)$ of {c}, all $p_{{T}}$" for c in cent_names],
                  res_label=r"Data $-\ f_{c}(x)$ [ns]",
                  lines_txt=rf"line: $f_{{c}}(x)$, degree {FIT_DEG}, fitted per class on "
                            rf"{TIMING_PT[0]}–{TIMING_PT[1]} GeV", save_pdf=args.save_pdf)
        plot_grid(Pr["xc"], [list(col) for col in zip(*Mr["by_class"])], pt_names, cent_names, CENT_COLORS, Mr["x_pt"],
                  r_tag, rlim_p, runs, lumi, bdir / f"baseline_pt_fcx_bypt_{name}.png",
                  entry_fn=[fc_fn for _ in PT_SPLIT], res_label=r"Data $-\ f_{c}(x)$ [ns]",
                  lines_txt=rf"lines: $f_{{c}}(x)$ of each class, degree {FIT_DEG}, fitted on "
                            rf"{TIMING_PT[0]}–{TIMING_PT[1]} GeV", save_pdf=args.save_pdf)
    tick("plots")

    # ---- The OHCal fraction distributions of the same jets (the lines of the line source for every radius)
    if not args.no_x_dist:
        for r_tag in args.radii:
            ohcal_frac_checks(loaded, runs, r_tag, lines, src, lumi, out / "timing" / "ohcal_frac", ex_runs, args)
        tick("OHCal fraction distributions")

    # ---- The corrected hists of the production (Jet-Anav3 with the correction), when every run has them
    for r_tag in args.radii:
        n_corr = c_sum.get(r_tag, {}).get("n_runs", 0)
        if n_corr == 0:
            print(f"{r_tag}: no timing-corrected hists in the inputs, skipping the correction checks")
        elif n_corr < len(runs):
            print(f"Warning: {r_tag}: only {n_corr}/{len(runs)} runs have the timing-corrected hists, skipping the checks")
        else:
            corr_checks(loaded, c_sum[r_tag], t_sum[r_tag], runs, r_tag, lumi, M[r_tag], out / "timing" / "corr_check", args)
    tick("correction checks")

    # ---- The jet time alone (*_tjet) and t vs t0, when every run has them
    tj = {}
    for r_tag in args.radii:
        n_j = j_sum.get(r_tag, {}).get("n_runs", 0)
        if n_j == 0:
            print(f"{r_tag}: no jet-time-alone hists in the inputs, skipping the t0 checks")
        elif n_j < len(runs):
            print(f"Warning: {r_tag}: only {n_j}/{len(runs)} runs have the jet-time-alone hists, skipping the t0 checks")
        else:
                tj[r_tag] = tjet_checks(loaded, j_sum[r_tag], t_sum[r_tag], runs, r_tag, src, lumi, P[r_tag], lines,
                                    out / "timing" / "tjet_check", args)
    tick("t0 checks")

    # ---- The MBD t0 per run (MB events, Event QA hT0Corrected): peaks, stability, the outlier runs
    mbd_t0_checks(t0_qa, loaded, runs, src, lumi, lines, tj.get(src), out / "timing" / "mbd_t0", args)
    tick("MBD t0")


if __name__ == "__main__":
    main()
