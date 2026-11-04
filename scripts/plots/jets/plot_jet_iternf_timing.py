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

A bin's error in a run is sigma_x / sqrt(n_run,x), with sigma_x the time RMS of the bin pooled over all runs.
"""

import argparse
import concurrent.futures
import csv
import hashlib
import json
import os
from pathlib import Path
import pickle
import sys
import time
import numpy as np
import uproot
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
POINTS_NOTE = r"points: $t - a_{\mathrm{run}} - b_{\mathrm{run}}\,x$, all runs combined"


def pt_hist_name(r_tag):
    return f"hJetPtv3_raw_{r_tag}_iternf"


def timing_hist_name(r_tag, pt=TIMING_PT):
    return f"h2LeadJetTimeOHCalFrac_{r_tag}_iternf_pt{pt[0]}_{pt[1]}"


def timing_status_name(r_tag):
    return f"hLeadJetTimingStatus_{r_tag}_iternf"


def profile_name(r_tag, cent):
    return f"p2LeadJetTimeOHCalFracRawPt_{r_tag}_iternf_cent_{cent[0]}_{cent[1]}"


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


CACHE_VERSION = 1   # bump when load_run's output changes, so old caches are not reused


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

    t_mom, t_filled and t_flow are not used any more; they are kept so the input cache of this output stays valid.
    """
    run, file_path, pt_radii, t_radii = task
    data = {}
    try:
        with uproot.open(file_path) as rf:
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
    except (KeyError, OSError, StopIteration) as e:
        print(f"Warning: run {run}: {e}, skipping")
        return run, None
    return run, data


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
    text = rf"{len(runs)} runs ({min(runs)}–{max(runs)}), $R = {r_jet:g}$"
    if lumi is not None:
        text += rf", $\mathcal{{L}} = {lumi:.1f}\ \mu\mathrm{{b}}^{{-1}}$"
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
# Main
# ---------------------------------------------------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run-combined iternf raw jet pT, and the leading-jet timing model t = f_c(x) + a_run + b_run x "
                    "(quartic per centrality class, per-run line) at raw pT 15-40 GeV, built step by step."
    )
    parser.add_argument("-f", "--file", type=Path,
                        default=Path("/sphenix/user/anarde/sEPD-Study/jetAnav3/11-02-26-v3/test.list"),
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
        loaded, t_sum, lumi = cached["loaded"], cached["t_sum"], cached["lumi"]
        runs = sorted(loaded)
    else:
        # Stream the results so only one full 2D per radius (the sum) is kept in memory
        n_jobs = max(1, min(args.jobs, len(tasks)))
        print(f"Reading {len(tasks)} runs with {n_jobs} worker(s)")
        loaded, t_sum = {}, {}
        ex = concurrent.futures.ProcessPoolExecutor(max_workers=n_jobs) if n_jobs > 1 else None
        for run, d in (ex.map(load_run, tasks, chunksize=8) if ex else map(load_run, tasks)):
            if d is None:
                continue
            for r_tag in t_radii:
                t_sum[r_tag] = t_sum.get(r_tag, 0) + d[r_tag].pop("t_values")
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

        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = cache_path.with_suffix(".tmp")
            with tmp.open("wb") as f:
                pickle.dump({"loaded": loaded, "t_sum": t_sum, "lumi": lumi}, f, protocol=pickle.HIGHEST_PROTOCOL)
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
                      note=r"points: $t - a_{\mathrm{run}} - b_{\mathrm{run}}\,x$, all runs and classes combined" "\n"
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


if __name__ == "__main__":
    main()
