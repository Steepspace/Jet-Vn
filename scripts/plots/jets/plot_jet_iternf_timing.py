#!/usr/bin/env python3
"""Run-combined iternf jet plots and the leading-jet timing correction from Jet-Anav3 output (iter_is_noflow mode).

1. Raw jet pT (hJetPtv3_raw_<r>_iternf) summed over runs.
2. Leading jet (t - MBD t0) vs OHCal energy fraction x (h2LeadJetTimeOHCalFrac_<r>_iternf_pt15_40), modeled per run as

       t(x, run) = f(x) + a(run) + b(run) * x

   with f a polynomial (default degree 4) shared by all runs, and a per-run line a + b*x. The per-run lines are measured
   from one radius (default R = 0.3, which has ~30x the statistics) and applied to every radius; each radius gets its
   own f. Convention: the entries-weighted means of a and b over runs are 0, so f carries the average level and slope.
3. Tests of the model: degree scan, per-run chi2/ndf, pulls, residuals vs x, per-run lines from the other radius,
   raw pT sub-ranges (15-20, 20-30, 30-40 GeV), per-run and pooled timing RMS.

The fit uses the per-run profiles (mean t and its error per x bin). Everything is computed from per-run, per-x-bin
moments of the 2D histograms (sum w, sum w t, sum w t^2, sum w^2), so the corrected profiles and RMS are exact at the
level of the histogram's time bin centers.
"""

import argparse
import concurrent.futures
import csv
import json
import os
from pathlib import Path
import sys
import numpy as np
import uproot
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.ticker import LogLocator
import mplhep as hep

from plot_jet_bkgsub_compare import save_fig
from plot_jet_noflow_compare import parse_list_file, load_luminosity

# (tag, R, |eta| acceptance used in Jet-Anav3.C: m_jet_eta_max_r02 / m_jet_eta_max_r03)
RADII = {
    "r02": {"r": 0.2, "eta_max": 0.9},
    "r03": {"r": 0.3, "eta_max": 0.8},
}

TIMING_PT = (15, 40)                              # raw leading jet pT range of the main timing hist (m_timing_pt_ranges[0])
SUB_PT = ((15, 20), (20, 30), (30, 40))           # the other m_timing_pt_ranges, used for the pT dependence test
SUB_PT_COLORS = ("#1b9e77", "#d95f02", "#7570b3")
JET_PT_MIN = 10                                   # calibrated jet pT cut of the production (jet_pt_min job arg)
SCAN_DEGREES = (2, 3, 4, 5, 6)                    # shape degrees of the degree scan (plus a free value per x bin)

# Event selection shared by all histograms. In iter_is_noflow mode calo v2 = 0 and there is no flow failure,
# so the |Calo v2| and flow failure cuts of hJetPtv3 are not applied.
EVENT_INFO = (
    r"$|z| < 10$ cm & MB" "\n"
    "Good Calo-Cent\n"
    r"Centrality: 0–60%" "\n"
    "Iterative No Flow"
)

TIME_LABEL = r"Jet $t - t_{0}^{\mathrm{MBD}}$ [ns]"
FRAC_LABEL = "OHCal Energy Fraction"
MODEL_LABEL = r"$t = f(x) + a_{\mathrm{run}} + b_{\mathrm{run}}\,x$"


def pt_hist_name(r_tag):
    return f"hJetPtv3_raw_{r_tag}_iternf"


def timing_hist_name(r_tag, pt=TIMING_PT):
    return f"h2LeadJetTimeOHCalFrac_{r_tag}_iternf_pt{pt[0]}_{pt[1]}"


def timing_status_name(r_tag):
    return f"hLeadJetTimingStatus_{r_tag}_iternf"


def pt_tag(pt):
    return f"pt{pt[0]}_{pt[1]}"


# ---------------------------------------------------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------------------------------------------------

def column_moments(h):
    """Per x bin of a TH2: [sum w, sum w t, sum w t^2, sum w^2] with t the y bin center. Shape (4, n_x)."""
    v = h.values().astype(np.float64)
    w2 = h.variances().astype(np.float64)
    ye = h.axes[1].edges()
    yc = 0.5 * (ye[:-1] + ye[1:])
    return np.stack([v.sum(1), (v * yc).sum(1), (v * yc ** 2).sum(1), w2.sum(1)])


def load_run(task):
    """Reads the per-run histograms. Returns (run, data) where data is None if a histogram is missing."""
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
                d = data.setdefault(r_tag, {})
                d.update({
                    "t_values": h_t.values().astype(np.float64),   # full 2D, only summed for the combined plot
                    "t_mom": column_moments(h_t),
                    "t_xedges": h_t.axes[0].edges(),
                    "t_yedges": h_t.axes[1].edges(),
                    "t_filled": float(rf[timing_status_name(r_tag)].values()[1]),
                    # Leading jets outside the plotted range: OHCal fraction below / above the axis, |t - t0| > 25 ns
                    "t_flow": np.array([t_flow[0].sum(), t_flow[-1].sum(),
                                        t_flow[1:-1, 0].sum() + t_flow[1:-1, -1].sum()]),
                    "sub_mom": {pt: column_moments(rf[timing_hist_name(r_tag, pt)]) for pt in SUB_PT},
                })
    except (KeyError, OSError) as e:
        print(f"Warning: run {run}: {e}, skipping")
        return run, None
    return run, data


# ---------------------------------------------------------------------------------------------------------------------
# Profiles from moments
# ---------------------------------------------------------------------------------------------------------------------

def profile_from_moments(mom):
    """ProfileX from column moments (same as TH2::ProfileX, default error option). mom: (..., 4, n_x).

    Returns mean, error on the mean (RMS / sqrt(N_eff)), N_eff = (sum w)^2 / sum w^2.
    """
    s0, s1, s2, sw2 = mom[..., 0, :], mom[..., 1, :], mom[..., 2, :], mom[..., 3, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        mean = s1 / s0
        rms2 = s2 / s0 - mean ** 2
        neff = np.where(sw2 > 0, s0 ** 2 / sw2, 0.0)
        err = np.sqrt(np.clip(rms2, 0, None) / neff)
    return mean, err, neff


def shift_moments(mom, delta):
    """Moments of t - delta, with delta per x bin (n_x,) or per run and x bin (n_runs, n_x)."""
    s0, s1, s2, sw2 = mom[..., 0, :], mom[..., 1, :], mom[..., 2, :], mom[..., 3, :]
    return np.stack([s0, s1 - delta * s0, s2 - 2 * delta * s1 + delta ** 2 * s0, sw2], axis=-2)


def pooled_rms(mom, cols):
    """RMS of t over the x bins in cols, summing the moments of all runs given (mom: (n_runs, 4, n_x) or (4, n_x))."""
    m = mom[..., cols]
    s0, s1, s2 = m[..., 0, :].sum(), m[..., 1, :].sum(), m[..., 2, :].sum()
    mu = s1 / s0
    return float(np.sqrt(s2 / s0 - mu ** 2))


def per_run_rms(mom, cols):
    """RMS of t per run over the x bins in cols. mom: (n_runs, 4, n_x)."""
    m = mom[..., cols]
    s0, s1, s2 = m[:, 0].sum(1), m[:, 1].sum(1), m[:, 2].sum(1)
    with np.errstate(divide="ignore", invalid="ignore"):
        mu = s1 / s0
        return np.sqrt(np.clip(s2 / s0 - mu ** 2, 0, None))


# ---------------------------------------------------------------------------------------------------------------------
# The model
# ---------------------------------------------------------------------------------------------------------------------

def shape_design(xc, degree, cols=None, n_line=2):
    """Columns of the shape f not already free per run, or a free value per x bin (degree="bins").

    With a per-run line (n_line = 2) the constant and x are free per run, so f gets powers 2..degree; with a per-run
    offset only (n_line = 1) f gets powers 1..degree. For "bins", n_line used x bins are dropped for the same reason.
    """
    if degree == "bins":
        used = np.where(cols)[0]
        keep = cols.copy()                  # bins outside the used columns have no data, so no column
        keep[[used[0], used[-1]][:n_line]] = False
        return np.eye(len(xc))[:, keep]
    if degree < n_line:
        return np.zeros((len(xc), 0))
    return np.vstack([xc ** k for k in range(n_line, degree + 1)]).T


def joint_fit(y, e, m, xc, X, n_line=2):
    """Least-squares fit of y[run, bin] = X[bin] @ c + per-run line, exactly, by projecting the line out per run.

    y, e, m: (n_runs, n_x) profile means, errors and use mask. X: (n_x, K) shared shape columns.
    n_line = 2 -> per-run a + b*x, 1 -> per-run a only.
    Returns dict with c, cov_c, line params (n_runs, n_line), their covariances, chi2, ndf, pulls, fitted mask.
    """
    w = np.where(m, 1.0 / np.where(m, e, 1.0) ** 2, 0.0)
    L = np.vstack([np.ones_like(xc), xc][:n_line]).T                        # (n_x, n_line)
    G = np.einsum("rb,bi,bj->rij", w, L, L)                                 # (n_runs, n_line, n_line)
    ok = np.linalg.det(G) > 0 if n_line > 1 else G[:, 0, 0] > 0
    G[~ok] = np.eye(n_line)
    w[~ok] = 0.0
    Ginv = np.linalg.inv(G)
    y0 = np.where(m, y, 0.0)

    def project(z):  # z: (n_runs, n_x) or (n_runs, n_x, K) -> z minus its weighted per-run line
        if z.ndim == 2:
            coef = np.einsum("rij,rbj->ri", Ginv, np.einsum("rb,bj,rb->rbj", w, L, z))
            return z - coef @ L.T
        coef = np.einsum("rij,rjk->rik", Ginv, np.einsum("rb,bj,rbk->rjk", w, L, z))
        return z - np.einsum("bj,rjk->rbk", L, coef)

    py = project(y0)
    K = X.shape[1]
    if K > 0:
        PX = project(np.broadcast_to(X, (len(y),) + X.shape).copy())
        N = np.einsum("rb,rbk,rbl->kl", w, PX, PX)
        rhs = np.einsum("rb,rbk,rb->k", w, PX, py)
        cov_c = np.linalg.inv(N)
        c = cov_c @ rhs
        fit_shape = X @ c
    else:
        c, cov_c, fit_shape = np.zeros(0), np.zeros((0, 0)), np.zeros_like(xc)
    res0 = y0 - fit_shape[None, :]
    line = np.einsum("rij,rj->ri", Ginv, np.einsum("rb,bj,rb->rj", w, L, res0))
    model = fit_shape[None, :] + line @ L.T
    used = (w > 0)
    pulls = ((y0 - model) * np.sqrt(w))[used]
    n_runs = int(ok.sum())
    return {"c": c, "cov_c": cov_c, "line": np.where(ok[:, None], line, np.nan), "line_cov": Ginv, "ok": ok,
            "chi2": float(np.sum(pulls ** 2)), "ndf": int(used.sum() - n_line * n_runs - K), "pulls": pulls,
            "w": w, "model": model}


def fit_shape_fixed_lines(y, e, m, xc, degree, a, b):
    """Full polynomial f (powers 0..degree) fitted to y - (a + b x), with the per-run lines held fixed."""
    w = np.where(m, 1.0 / np.where(m, e, 1.0) ** 2, 0.0)
    w[~(np.isfinite(a) & np.isfinite(b))] = 0.0
    z = np.where(w > 0, y - np.nan_to_num(a)[:, None] - np.nan_to_num(b)[:, None] * xc[None, :], 0.0)
    X = np.vstack([xc ** k for k in range(degree + 1)]).T
    N = np.einsum("rb,bk,bl->kl", w, X, X)
    rhs = np.einsum("rb,bk,rb->k", w, X, z)
    cov = np.linalg.inv(N)
    coef = cov @ rhs
    pulls = ((z - X @ coef) * np.sqrt(w))[w > 0]
    return coef, cov, float(np.sum(pulls ** 2)), int((w > 0).sum() - degree - 1), pulls, w


def poly(coef, x):
    return np.polynomial.polynomial.polyval(x, coef)


def poly_err(cov, x):
    J = np.vstack([x ** k for k in range(cov.shape[0])])
    return np.sqrt(np.einsum("ik,ij,jk->k", J, cov, J))


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


def line_source_text(r_tag, src):
    return "own fit" if r_tag == src else rf"from $R = {RADII[src]['r']:g}$"


# ---------------------------------------------------------------------------------------------------------------------
# Plot 1: raw pT
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


# ---------------------------------------------------------------------------------------------------------------------
# Plot 2: combined timing vs OHCal fraction with the model
# ---------------------------------------------------------------------------------------------------------------------

def frac_outside_unit(values, xedges, flow):
    """Leading jets with OHCal fraction < 0 and > 1: bins of the histogram outside [0, 1] plus its x under/overflow.

    The 0 and 1 edges are bin edges in the current booking (-0.25 to 1.25 in 0.025 steps), so the split is exact.
    """
    xc = 0.5 * (xedges[:-1] + xedges[1:])
    return values[xc < 0].sum() + flow[0], values[xc > 1].sum() + flow[1]


def plot_timing(values, xedges, yedges, prof_raw, prof_corr, cols, shape, runs, r_tag, src, lumi, output_path, flow,
                time_ylim=None, save_pdf=False):
    hep.style.use("ATLAS")
    meta = RADII[r_tag]
    xc = 0.5 * (xedges[:-1] + xedges[1:])

    fig, ax = plt.subplots(figsize=(9, 7))
    z = np.where(values > 0, values, np.nan)
    pcm = ax.pcolormesh(xedges, yedges, z.T, cmap="viridis", norm=LogNorm(vmin=1, vmax=np.nanmax(z)), rasterized=True)
    cbar = fig.colorbar(pcm, ax=ax, pad=0.02)
    cbar.set_label("Counts")

    raw_mean, raw_err, _ = prof_raw
    ok = np.isfinite(raw_mean)
    ax.errorbar(xc[ok], raw_mean[ok], yerr=np.nan_to_num(raw_err[ok]), fmt="o", mfc="white", mec="gray", color="gray",
                markersize=4, capsize=0, linewidth=1.0, label="Profile", zorder=3)
    c_mean, c_err, _ = prof_corr
    ax.errorbar(xc[cols], c_mean[cols], yerr=c_err[cols], fmt="o", color="black", markersize=4, capsize=0,
                linewidth=1.2, label=r"Profile of $t - a_{\mathrm{run}} - b_{\mathrm{run}} x$", zorder=4)

    xf = np.linspace(xc[cols].min(), xc[cols].max(), 300)
    ax.plot(xf, poly(shape["coef"], xf), color="#e41a1c", linewidth=2, label=rf"$f(x)$, degree {shape['degree']}",
            zorder=5)

    ax.set_xlim(xedges[0], xedges[-1])
    ax.set_ylim(*(time_ylim if time_ylim is not None else (yedges[0], yedges[-1])))
    ax.set_xlabel(FRAC_LABEL + r" $x$")
    ax.set_ylabel(TIME_LABEL)

    box = dict(facecolor="white", alpha=0.8, edgecolor="none", pad=3)
    add_header(ax, runs, meta["r"], lumi)
    ax.text(0.97, 0.97, timing_info(meta["eta_max"]), transform=ax.transAxes, ha="right", va="top", fontsize=12, bbox=box)
    lines = [MODEL_LABEL, f"Per-run lines: {line_source_text(r_tag, src)}"]
    lines += [rf"$f_{{{k}}} = {p:+.4f}$" for k, p in enumerate(shape["coef"])]
    lines.append(rf"$\chi^{{2}}/\mathrm{{ndf}} = {shape['chi2']:.0f}/{shape['ndf']}$ ({shape['n_runs']} runs)")
    ax.text(0.03, 0.03, "\n".join(lines), transform=ax.transAxes, ha="left", va="bottom", fontsize=11, bbox=box)
    n_neg, n_over = frac_outside_unit(values, xedges, flow)
    total = values.sum() + flow.sum()
    ax.text(0.97, 0.03,
            f"Entries: {values.sum():.0f} in range / {total:.0f}\n"
            f"OHCal frac < 0: {n_neg:.0f} ({100 * n_neg / total:.1f}%)\n"
            f"OHCal frac > 1: {n_over:.0f} ({100 * n_over / total:.2f}%)\n"
            rf"$|t - t_{{0}}| > 25$ ns: {flow[2]:.0f}",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=11, bbox=box)
    ax.legend(loc="upper left", fontsize=11, frameon=True, framealpha=0.8, edgecolor="none")
    plt.subplots_adjust(left=0.13, right=0.97, bottom=0.12, top=0.94)
    save_fig(fig, output_path, save_pdf)
    plt.close(fig)


# ---------------------------------------------------------------------------------------------------------------------
# Plot 3: tests of the model
# ---------------------------------------------------------------------------------------------------------------------

def setup_run_axis(ax, runs):
    # Label every run when few, otherwise ~40 evenly spaced runs (x is the run index, so gaps between runs are not stretched)
    step = max(1, int(np.ceil(len(runs) / 40)))
    ticks = np.arange(0, len(runs), step)
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(runs[i]) for i in ticks], rotation=90, fontsize=7)
    ax.set_xlim(-1, len(runs))
    ax.set_xlabel("Run", labelpad=6)
    ax.tick_params(axis="x", which="minor", bottom=False, top=False)


def plot_chi2_runs(rows, r_tag, src, lumi, output_path, save_pdf):
    """D1: per-run chi2/ndf of the run profile against f only and against the full model."""
    hep.style.use("ATLAS")
    runs = [r["run"] for r in rows]
    idx = np.arange(len(rows))
    f_only = np.array([r["chi2ndf_f_only"] for r in rows])
    full = np.array([r["chi2ndf_model"] for r in rows])
    fig, ax = plt.subplots(figsize=(12, 6))
    ms = 5 if len(rows) < 500 else 2.5
    ax.plot(idx, f_only, "o", color="#377eb8", markersize=ms, label=r"$f(x)$ only")
    ax.plot(idx, full, "s", color="#e41a1c", markersize=ms,
            label=rf"$f(x) + a_{{\mathrm{{run}}}} + b_{{\mathrm{{run}}}} x$ ({line_source_text(r_tag, src)})")
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1)
    ax.set_yscale("log")
    vals = np.concatenate([f_only, full])
    vals = vals[np.isfinite(vals) & (vals > 0)]
    ax.set_ylim(min(0.3, vals.min() * 0.7), max(3.0, vals.max() * 5))
    ax.set_ylabel(r"$\chi^{2}/\mathrm{ndf}$ of run profile")
    ax.text(0.99, 0.95, f"Median: {np.nanmedian(f_only):.2f} (f only), {np.nanmedian(full):.2f} (model)",
            transform=ax.transAxes, ha="right", va="top", fontsize=13)
    setup_run_axis(ax, runs)
    add_header(ax, runs, RADII[r_tag]["r"], lumi)
    ax.legend(loc="upper left", frameon=False, fontsize=13)
    plt.subplots_adjust(left=0.08, right=0.98, bottom=0.20, top=0.93)
    save_fig(fig, output_path, save_pdf)
    plt.close(fig)


def plot_lines_runs(rows, r_tag, lumi, output_path, save_pdf, run_range=None):
    """D2: the per-run lines a and b vs run (the correction table). run_range zooms and uses the run number as x."""
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
        ax.text(0.01, 0.95, f"RMS over runs = {np.nanstd(y):.3f} ns, median error = {np.nanmedian(e):.3f} ns",
                transform=ax.transAxes, ha="left", va="top", fontsize=12)
    axes[0].text(0.99, 0.95, r"Correction: $a_{\mathrm{run}} + b_{\mathrm{run}}\,x$", transform=axes[0].transAxes,
                 ha="right", va="top", fontsize=13)
    if run_range is None:
        setup_run_axis(axes[1], runs)
    else:
        axes[1].set_xlabel("Run")
        axes[1].ticklabel_format(axis="x", useOffset=False, style="plain")
    add_header(axes[0], runs, RADII[r_tag]["r"], lumi)
    plt.subplots_adjust(left=0.09, right=0.98, bottom=0.15 if run_range is None else 0.09, top=0.95)
    save_fig(fig, output_path, save_pdf)
    plt.close(fig)


def plot_residuals_pulls(xc, resid, resid_err, pull_sets, runs, r_tag, lumi, output_path, save_pdf):
    """D3: (pooled corrected profile - f) vs x, and pulls of the run profiles for each model."""
    hep.style.use("ATLAS")
    fig, (ax, axp) = plt.subplots(1, 2, figsize=(15, 6.5), gridspec_kw={"width_ratios": [1.6, 1], "wspace": 0.22})
    ok = np.isfinite(resid)
    ax.axhline(0.0, color="gray", linestyle="--", linewidth=1)
    ax.errorbar(xc[ok], resid[ok], yerr=resid_err[ok], fmt="o", color="black", markersize=5, capsize=0, linewidth=1.3)
    ax.set_xlabel(FRAC_LABEL + r" $x$")
    ax.set_ylabel(r"Profile of $t - a_{\mathrm{run}} - b_{\mathrm{run}} x$, minus $f(x)$ [ns]", fontsize=14)
    chi2 = np.nansum((resid[ok] / resid_err[ok]) ** 2)
    ax.text(0.03, 0.95, f"All runs pooled\n" rf"$\chi^{{2}}$ vs 0: {chi2:.1f} / {ok.sum()} bins",
            transform=ax.transAxes, ha="left", va="top", fontsize=13)
    lim = max(0.02, 1.4 * np.nanmax(np.abs(resid[ok]) + resid_err[ok]))
    ax.set_ylim(-lim, lim)
    add_header(ax, runs, RADII[r_tag]["r"], lumi)

    allp = np.concatenate([p for _, p, _ in pull_sets])
    plim = max(5.0, np.ceil(np.percentile(np.abs(allp), 99.5)))
    bins = np.linspace(-plim, plim, 61)
    bw = bins[1] - bins[0]
    stats = []
    top = 1.0
    for label, p, color in pull_sets:
        counts, *_ = axp.hist(np.clip(p, bins[0] + 1e-9, bins[-1] - 1e-9), bins=bins, histtype="step", color=color,
                              linewidth=1.8, label=label)
        top = max(top, counts.max())
        stats.append(f"{label}: std {np.std(p):.2f}")
    n = len(pull_sets[-1][1])
    xs = np.linspace(-plim, plim, 400)
    axp.plot(xs, n * bw / np.sqrt(2 * np.pi) * np.exp(-0.5 * xs ** 2), "--", color="gray", linewidth=1.5,
             label=r"$\mathcal{N}(0, 1)$")
    axp.set_yscale("log")
    axp.set_ylim(0.5, top * 300)
    axp.set_xlim(-plim, plim)
    axp.set_xlabel(r"(Run profile $-$ model) / $\sigma$")
    axp.set_ylabel("Entries (run, x bin)")
    axp.text(0.03, 0.97, "\n".join(stats) + "\n(last bins incl. overflow)", transform=axp.transAxes, ha="left",
             va="top", fontsize=10)
    axp.legend(loc="upper right", frameon=False, fontsize=10)
    plt.subplots_adjust(left=0.07, right=0.98, bottom=0.13, top=0.93)
    save_fig(fig, output_path, save_pdf)
    plt.close(fig)


def plot_degree_scan(scan, runs, r_tag, lumi, output_path, save_pdf):
    """D4: chi2/ndf of the joint fit vs shape degree, for a per-run offset and a per-run line."""
    hep.style.use("ATLAS")
    labels = [str(d) for d in SCAN_DEGREES] + ["per bin"]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(9, 6.5))
    for n_line, color, marker, label in ((1, "#377eb8", "o", r"$f(x) + a_{\mathrm{run}}$"),
                                         (2, "#e41a1c", "s", r"$f(x) + a_{\mathrm{run}} + b_{\mathrm{run}} x$")):
        y = [scan[(n_line, d)]["chi2"] / scan[(n_line, d)]["ndf"] for d in list(SCAN_DEGREES) + ["bins"]]
        ax.plot(x, y, marker=marker, color=color, markersize=8, linewidth=1.5, label=label)
        for xi, yi in zip(x, y):
            ax.text(xi, yi * 1.08, f"{yi:.3f}", ha="center", va="bottom", fontsize=10, color=color)
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Shape $f(x)$: polynomial degree")
    ax.set_ylabel(r"$\chi^{2}/\mathrm{ndf}$ of the joint fit")
    ax.set_ylim(0.5, ax.get_ylim()[1] * 2)
    ax.legend(loc="upper right", frameon=False, fontsize=13)
    ax.text(0.03, 0.03, '"per bin": a free value in every x bin (best possible shape)', transform=ax.transAxes,
            ha="left", va="bottom", fontsize=11)
    add_header(ax, runs, RADII[r_tag]["r"], lumi)
    plt.subplots_adjust(left=0.13, right=0.97, bottom=0.12, top=0.93)
    save_fig(fig, output_path, save_pdf)
    plt.close(fig)


def plot_rms_runs(rows, pooled, r_tag, src, lumi, output_path, save_pdf):
    """D5: per-run timing RMS raw / after f / after the full model, with the all-run pooled RMS in the text."""
    hep.style.use("ATLAS")
    runs = [r["run"] for r in rows]
    idx = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(12, 6))
    ms = 5 if len(rows) < 500 else 2.5
    series = (("rms_raw", "#377eb8", "o", r"$t - t_{0}$"),
              ("rms_f", "#ff7f00", "^", r"$-\ f(x)$"),
              ("rms_model", "#e41a1c", "s", rf"$-\ f(x) - a_{{\mathrm{{run}}}} - b_{{\mathrm{{run}}}} x$ ({line_source_text(r_tag, src)})"))
    for key, color, marker, label in series:
        ax.plot(idx, [r[key] for r in rows], marker, color=color, markersize=ms, label=label)
    ax.set_ylabel("Timing RMS per run [ns]")
    vals = np.array([r[k] for r in rows for k, *_ in series])
    lo, hi = np.nanpercentile(vals, 0.5), np.nanpercentile(vals, 99.5)
    ax.set_ylim(lo - 0.3 * (hi - lo), hi + 0.9 * (hi - lo))
    ax.text(0.99, 0.95,
            "All runs pooled: "
            f"{pooled['raw']:.3f} → {pooled['f']:.3f} (f) → {pooled['model']:.3f} ns (model)\n"
            f"Median per run: {np.median([r['rms_raw'] for r in rows]):.3f} → "
            f"{np.median([r['rms_f'] for r in rows]):.3f} → {np.median([r['rms_model'] for r in rows]):.3f} ns",
            transform=ax.transAxes, ha="right", va="top", fontsize=12)
    setup_run_axis(ax, runs)
    add_header(ax, runs, RADII[r_tag]["r"], lumi)
    ax.legend(loc="upper left", frameon=False, fontsize=12, ncol=3)
    plt.subplots_adjust(left=0.08, right=0.98, bottom=0.20, top=0.93)
    save_fig(fig, output_path, save_pdf)
    plt.close(fig)


def plot_line_crosscheck(own, src_line, runs, r_tag, src, lumi, output_path, save_pdf):
    """D6: per-run lines fitted on this radius vs the lines from the source radius (a and b)."""
    hep.style.use("ATLAS")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5), gridspec_kw={"wspace": 0.25})
    for ax, k, name in ((axes[0], 0, "a"), (axes[1], 1, "b")):
        xo, xs = own[:, k], src_line[:, k]
        ok = np.isfinite(xo) & np.isfinite(xs)
        lo, hi = np.percentile(np.concatenate([xo[ok], xs[ok]]), [0.5, 99.5])
        ax.plot(xs[ok], xo[ok], "o", color="black", markersize=2, alpha=0.5)
        ax.plot([lo, hi], [lo, hi], "--", color="#e41a1c", linewidth=1.5, label="y = x")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        d = xo[ok] - xs[ok]
        ax.text(0.03, 0.97, f"corr = {np.corrcoef(xo[ok], xs[ok])[0, 1]:.3f}\nRMS(diff) = {d.std():.3f} ns\n"
                f"mean(diff) = {d.mean():+.3f} ns\n{ok.sum()} runs", transform=ax.transAxes, ha="left", va="top",
                fontsize=12)
        ax.set_xlabel(rf"${name}_{{\mathrm{{run}}}}$, $R = {RADII[src]['r']:g}$ [ns]")
        ax.set_ylabel(rf"${name}_{{\mathrm{{run}}}}$, $R = {RADII[r_tag]['r']:g}$ own fit [ns]")
        ax.legend(loc="lower right", frameon=False, fontsize=12)
    add_header(axes[1], runs, RADII[r_tag]["r"], lumi)
    plt.subplots_adjust(left=0.08, right=0.98, bottom=0.13, top=0.93)
    save_fig(fig, output_path, save_pdf)
    plt.close(fig)


def plot_pt_dependence(xc, sub, runs, r_tag, src, lumi, output_path, save_pdf):
    """D7: pooled corrected profile of each raw pT sub-range minus f of the full 15-40 GeV range."""
    hep.style.use("ATLAS")
    fig, ax = plt.subplots(figsize=(10, 6.5))
    ax.axhline(0.0, color="gray", linestyle="--", linewidth=1)
    lim = 0.05
    texts = []
    for (pt, (resid, err)), color, dx in zip(sub.items(), SUB_PT_COLORS, (-0.005, 0.0, 0.005)):
        ok = np.isfinite(resid) & np.isfinite(err) & (err > 0)
        ax.errorbar(xc[ok] + dx, resid[ok], yerr=err[ok], fmt="o", color=color, markersize=4, capsize=0, linewidth=1.0,
                    label=rf"Raw ${pt[0]} \leq p_{{T}} < {pt[1]}$ GeV")
        chi2 = np.sum((resid[ok] / err[ok]) ** 2)
        texts.append(rf"{pt[0]}–{pt[1]} GeV: $\chi^{{2}}$ vs 0 = {chi2:.0f}/{ok.sum()}")
        core = ok & (err < 0.05)
        if core.any():
            lim = max(lim, 1.3 * np.max(np.abs(resid[core]) + err[core]))
    ax.set_ylim(-lim, lim)
    ax.set_xlabel(FRAC_LABEL + r" $x$")
    ax.set_ylabel(r"Profile of $t - a_{\mathrm{run}} - b_{\mathrm{run}} x$, minus $f(x)$ [ns]", fontsize=14)
    ax.text(0.03, 0.03, rf"$f$ and lines from raw {TIMING_PT[0]}–{TIMING_PT[1]} GeV, lines {line_source_text(r_tag, src)}"
            + "\n" + "\n".join(texts), transform=ax.transAxes, ha="left", va="bottom", fontsize=11)
    ax.legend(loc="upper left", frameon=False, fontsize=12)
    add_header(ax, runs, RADII[r_tag]["r"], lumi)
    plt.subplots_adjust(left=0.12, right=0.97, bottom=0.12, top=0.93)
    save_fig(fig, output_path, save_pdf)
    plt.close(fig)


def write_csv(rows, keys, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved: {path}")


# ---------------------------------------------------------------------------------------------------------------------
# Timing analysis for one radius
# ---------------------------------------------------------------------------------------------------------------------

def timing_inputs(loaded, runs, r_tag, args):
    """Per-run profiles, the use mask and the columns (x bins) of the fit."""
    mom = np.stack([loaded[r][r_tag]["t_mom"] for r in runs])              # (n_runs, 4, n_x)
    xedges = loaded[runs[0]][r_tag]["t_xedges"]
    xc = 0.5 * (xedges[:-1] + xedges[1:])
    y, e, neff = profile_from_moments(mom)
    _, c_err, c_neff = profile_from_moments(mom.sum(0))
    cols = (c_neff >= args.min_bin_entries) & np.isfinite(c_err) & (c_err > 0)
    if args.fit_range is not None:
        cols &= (xc >= args.fit_range[0]) & (xc <= args.fit_range[1])
    m = cols[None, :] & (neff >= args.min_run_bin_entries) & np.isfinite(e) & (e > 0)
    entries = mom[:, 0][:, cols].sum(1)
    good_run = (entries >= args.min_run_entries) & (m.sum(1) >= args.min_run_points)
    m &= good_run[:, None]
    return {"mom": mom, "xedges": xedges, "xc": xc, "y": y, "e": e, "m": m, "cols": cols, "entries": entries,
            "good_run": good_run}


def finalize_lines(fit, entries):
    """Moves the entries-weighted mean a and b into f (the convention), returns (a, b, mean_a, mean_b)."""
    a, b = fit["line"][:, 0], fit["line"][:, 1]
    ok = fit["ok"]
    wn = np.where(ok, entries, 0.0)
    ma, mb = np.sum(wn * np.nan_to_num(a)) / wn.sum(), np.sum(wn * np.nan_to_num(b)) / wn.sum()
    return a - ma, b - mb, ma, mb


def main():
    parser = argparse.ArgumentParser(
        description="Run-combined iternf raw jet pT, and the leading-jet timing vs OHCal fraction model "
                    "t = f(x) + a_run + b_run x (shared polynomial f, per-run line), with tests of the model."
    )
    parser.add_argument("-f", "--file", type=Path,
                        default=Path("/sphenix/user/anarde/sEPD-Study/jetAnav3/11-02-26-v2/test.list"),
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
    parser.add_argument("--shape-degree", type=int, default=4, help="Polynomial degree of the shared shape f (default: %(default)s)")
    parser.add_argument("--per-event", action="store_true", help="Normalize the raw pT spectrum per event and per GeV")
    parser.add_argument("--pt-max", type=float, default=None,
                        help="x-axis max for the raw pT plot (default: auto, plus a _pt60 version if auto > 60)")
    parser.add_argument("--time-ylim", nargs=2, type=float, default=None, metavar=("TMIN", "TMAX"),
                        help="y-axis range of the combined timing plot (default: full histogram range)")
    parser.add_argument("--min-bin-entries", type=float, default=20,
                        help="Minimum effective entries in an x bin of the all-run profile for the bin to be used (default: %(default)s)")
    parser.add_argument("--fit-range", nargs=2, type=float, default=None, metavar=("XMIN", "XMAX"),
                        help="Restrict the model to this OHCal fraction range (applied on top of --min-bin-entries)")
    parser.add_argument("--min-run-entries", type=float, default=200,
                        help="Minimum timing entries in the used x bins for a run to get a line (default: %(default)s)")
    parser.add_argument("--min-run-points", type=int, default=5,
                        help="Minimum used x bins in a run profile for the run to get a line (default: %(default)s)")
    parser.add_argument("--min-run-bin-entries", type=float, default=10,
                        help="Minimum effective entries in an x bin of a run profile (default: %(default)s)")
    parser.add_argument("--zoom-runs", nargs=2, type=int, action="append", default=[], metavar=("RUNMIN", "RUNMAX"),
                        help="Also make the per-run line plot (D2) zoomed to this run range, vs run number (repeatable)")
    parser.add_argument("--no-scan", action="store_true", help="Skip the shape degree scan (D4)")
    parser.add_argument("--no-pt", action="store_true", help="Skip the raw pT plot")
    parser.add_argument("--no-timing", action="store_true", help="Skip the timing model and its tests")
    parser.add_argument("-j", "--jobs", type=int, default=min(8, os.cpu_count() or 1),
                        help="Parallel workers for reading the runs (default: %(default)s)")
    parser.add_argument("--save-pdf", action="store_true", help="Save PDF versions in addition to PNG")
    args = parser.parse_args()
    if args.shape_degree < 1:
        parser.error("--shape-degree must be >= 1")

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
    lumi_files = parse_list_file(args.lumi_list) if args.lumi_list.exists() else {}
    if lumi_files:
        missing = [r for r in runs if r not in lumi_files or not lumi_files[r].exists()]
        if missing:
            print(f"Warning: no luminosity file for {len(missing)} run(s) (e.g. {missing[:5]}), omitting the luminosity label")
        else:
            lumis = [load_luminosity(lumi_files[r]) for r in runs]
            lumi = None if any(l is None for l in lumis) else float(sum(lumis))
    else:
        print(f"Warning: no luminosity list at {args.lumi_list}, omitting the luminosity label")

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

    src = args.line_source
    deg = args.shape_degree
    suffix_of = lambda r: f"{r}_iternf_{pt_tag(TIMING_PT)}"
    tdir = out / "timing"
    ddir = tdir / "diagnostics"

    # ---- Per-run lines from the source radius: joint fit of f (powers >= 2) and a + b x per run
    S = timing_inputs(loaded, runs, src, args)
    fit_src = joint_fit(S["y"], S["e"], S["m"], S["xc"], shape_design(S["xc"], deg), n_line=2)
    a_src, b_src, ma, mb = finalize_lines(fit_src, S["entries"])
    print(f"{src}: joint fit (degree {deg}, per-run line): chi2/ndf = {fit_src['chi2']:.0f}/{fit_src['ndf']} = "
          f"{fit_src['chi2'] / fit_src['ndf']:.3f}, pull std {fit_src['pulls'].std():.3f}, "
          f"{int(fit_src['ok'].sum())}/{len(runs)} runs with a line")
    no_line = [(r, n) for r, n, ok in zip(runs, S["entries"], fit_src["ok"]) if not ok]
    if no_line:
        print(f"{src}: {len(no_line)} run(s) without a line (too few entries/points): "
              + ", ".join(f"{r} ({n:.0f})" for r, n in no_line))

    cov = fit_src["line_cov"]
    corr_rows = []
    for i, run in enumerate(runs):
        ok = bool(fit_src["ok"][i])
        corr_rows.append({
            "run": run, "status": "ok" if ok else "no_line", "entries": float(S["entries"][i]),
            "n_points": int(S["m"][i].sum()),
            "a": float(a_src[i]) if ok else np.nan, "b": float(b_src[i]) if ok else np.nan,
            "a_err": float(np.sqrt(cov[i, 0, 0])) if ok else np.nan, "b_err": float(np.sqrt(cov[i, 1, 1])) if ok else np.nan,
            "cov_ab": float(cov[i, 0, 1]) if ok else np.nan,
        })
    write_csv(corr_rows, ["run", "status", "entries", "n_points", "a", "b", "a_err", "b_err", "cov_ab"],
              tdir / f"per_run_lines_{suffix_of(src)}.csv")

    for r_tag in args.radii:
        T = S if r_tag == src else timing_inputs(loaded, runs, r_tag, args)
        xc, cols, mom = T["xc"], T["cols"], T["mom"]
        suffix = suffix_of(r_tag)

        # ---- Shape f of this radius with the source lines held fixed (for the source this equals the joint fit)
        coef, coef_cov, chi2, ndf, pulls_model, w_model = fit_shape_fixed_lines(T["y"], T["e"], T["m"], xc, deg, a_src, b_src)
        if r_tag == src:
            # same optimum as the joint fit, whose ndf also counts the 2 line parameters fitted per run
            chi2, ndf = fit_src["chi2"], fit_src["ndf"]
        has_line = np.isfinite(a_src)
        n_runs_fit = int((w_model.sum(1) > 0).sum())
        shape = {"degree": deg, "coef": coef, "cov": coef_cov, "chi2": chi2, "ndf": ndf, "n_runs": n_runs_fit}
        print(f"{r_tag}: f (degree {deg}) with lines from {src}: chi2/ndf = {chi2:.0f}/{ndf} = {chi2 / ndf:.3f}, "
              f"coef = {np.array2string(coef, precision=4)}")

        # ---- Combined plot: raw profile, profile of t - a - b x, and f
        delta = np.where(has_line[:, None], a_src[:, None] + b_src[:, None] * xc[None, :], 0.0)
        mom_corr = shift_moments(mom[has_line], delta[has_line])
        prof_corr = profile_from_moments(mom_corr.sum(0))
        prof_raw = profile_from_moments(mom.sum(0))
        t_flow = sum(loaded[r][r_tag]["t_flow"] for r in runs)
        filled = sum(loaded[r][r_tag]["t_filled"] for r in runs)
        n_neg, n_over = frac_outside_unit(t_sum[r_tag], T["xedges"], t_flow)
        print(f"{r_tag}: timing entries = {t_sum[r_tag].sum():.0f} in range + {t_flow.sum():.0f} flow "
              f"(OHCal frac <0: {n_neg:.0f}, >1: {n_over:.0f}, |t|>25: {t_flow[2]:.0f}); status 'Filled' = {filled:.0f}")
        plot_timing(t_sum[r_tag], T["xedges"], loaded[runs[0]][r_tag]["t_yedges"], prof_raw, prof_corr, cols, shape,
                    runs, r_tag, src, lumi, tdir / f"combined_{timing_hist_name(r_tag)}.png", t_flow,
                    time_ylim=args.time_ylim, save_pdf=args.save_pdf)

        # ---- Per-run tests
        f_x = poly(coef, xc)
        w = np.where(T["m"], 1.0 / np.where(T["m"], T["e"], 1.0) ** 2, 0.0)
        y0 = np.where(T["m"], T["y"], 0.0)
        res_f = (y0 - f_x[None, :]) * np.sqrt(w)
        res_model = (y0 - f_x[None, :] - delta) * np.sqrt(w)
        own = joint_fit(T["y"], T["e"], T["m"], xc, shape_design(xc, deg), n_line=2) if r_tag != src else fit_src
        own_a, own_b, _, _ = finalize_lines(own, T["entries"])
        run_used = T["good_run"] & has_line
        rms_raw = per_run_rms(mom, cols)
        rms_f = per_run_rms(shift_moments(mom, np.broadcast_to(f_x, mom[:, 0].shape)), cols)
        rms_model = per_run_rms(shift_moments(mom, f_x[None, :] + delta), cols)
        rows = []
        for i, run in enumerate(runs):
            if not run_used[i]:
                continue
            npts = int(T["m"][i].sum())
            rows.append({
                "run": run, "entries": float(T["entries"][i]), "n_points": npts,
                "chi2ndf_f_only": float(np.sum(res_f[i] ** 2) / npts),
                "chi2ndf_model": float(np.sum(res_model[i] ** 2) / (npts - 2 if r_tag == src else npts)),
                "a": float(a_src[i]), "b": float(b_src[i]),
                "a_err": float(np.sqrt(cov[i, 0, 0])), "b_err": float(np.sqrt(cov[i, 1, 1])),
                "a_own": float(own_a[i]), "b_own": float(own_b[i]),
                "rms_raw": float(rms_raw[i]), "rms_f": float(rms_f[i]), "rms_model": float(rms_model[i]),
            })
        skipped = len(runs) - len(rows)
        if skipped:
            print(f"{r_tag}: {skipped} run(s) not in the per-run tests (no line, or too few entries/points in {r_tag})")

        pooled = {"raw": pooled_rms(mom[has_line], cols),
                  "f": pooled_rms(shift_moments(mom[has_line], np.broadcast_to(f_x, mom[has_line][:, 0].shape)), cols),
                  "model": pooled_rms(shift_moments(mom[has_line], f_x[None, :] + delta[has_line]), cols)}
        print(f"{r_tag}: timing RMS, all runs pooled: {pooled['raw']:.3f} (raw) -> {pooled['f']:.3f} (f) -> "
              f"{pooled['model']:.3f} ns (f + per-run line); median per run {np.median([r['rms_raw'] for r in rows]):.3f} "
              f"-> {np.median([r['rms_f'] for r in rows]):.3f} -> {np.median([r['rms_model'] for r in rows]):.3f} ns")
        print(f"{r_tag}: median per-run chi2/ndf: {np.median([r['chi2ndf_f_only'] for r in rows]):.2f} (f only) -> "
              f"{np.median([r['chi2ndf_model'] for r in rows]):.2f} (model)")

        plot_chi2_runs(rows, r_tag, src, lumi, ddir / f"D1_chi2ndf_{suffix}.png", args.save_pdf)
        if r_tag == src:
            plot_lines_runs(rows, r_tag, lumi, ddir / f"D2_lines_{suffix}.png", args.save_pdf)
            for lo, hi in args.zoom_runs:
                plot_lines_runs(rows, r_tag, lumi, ddir / f"D2_lines_{suffix}_runs{lo}_{hi}.png", args.save_pdf,
                                run_range=(lo, hi))

        # D3: residual of the pooled corrected profile, and pulls for f only / f + offset / f + line
        resid = prof_corr[0] - f_x
        resid = np.where(cols, resid, np.nan)
        fit_off = joint_fit(T["y"], T["e"], T["m"], xc, shape_design(xc, deg, n_line=1), n_line=1)
        pull_sets = [(r"$f$ only", res_f[w > 0], "#377eb8"),
                     (r"$f + a_{\mathrm{run}}$ (own fit)", fit_off["pulls"], "#ff7f00")]
        if r_tag != src:
            pull_sets.append((r"$f + a_{\mathrm{run}} + b_{\mathrm{run}}x$ (own fit)", own["pulls"], "#984ea3"))
        pull_sets.append((rf"$f + a_{{\mathrm{{run}}}} + b_{{\mathrm{{run}}}}x$ ({line_source_text(r_tag, src)})",
                          pulls_model, "#e41a1c"))
        for label, p, _ in pull_sets:
            print(f"{r_tag}: pulls {label}: std {np.std(p):.3f}")
        plot_residuals_pulls(xc, resid, prof_corr[1], pull_sets, runs, r_tag, lumi,
                             ddir / f"D3_residuals_pulls_{suffix}.png", args.save_pdf)

        # D4: degree scan (own joint fits)
        if not args.no_scan:
            scan = {}
            for n_line in (1, 2):
                for d in list(SCAN_DEGREES) + ["bins"]:
                    # per-bin shape: only bins with at least one run point (some pass the all-run cut but no run cut)
                    ft = joint_fit(T["y"], T["e"], T["m"], xc, shape_design(xc, d, T["m"].any(0), n_line=n_line),
                                   n_line=n_line)
                    scan[(n_line, d)] = {"chi2": ft["chi2"], "ndf": ft["ndf"]}
            print(f"{r_tag}: degree scan chi2/ndf (offset | line): " + ", ".join(
                f"{d}: {scan[(1, d)]['chi2'] / scan[(1, d)]['ndf']:.3f} | {scan[(2, d)]['chi2'] / scan[(2, d)]['ndf']:.3f}"
                for d in list(SCAN_DEGREES) + ["bins"]))
            plot_degree_scan(scan, runs, r_tag, lumi, ddir / f"D4_degree_scan_{suffix}.png", args.save_pdf)

        plot_rms_runs(rows, pooled, r_tag, src, lumi, ddir / f"D5_rms_{suffix}.png", args.save_pdf)

        if r_tag != src:
            ok = run_used & own["ok"]
            plot_line_crosscheck(np.vstack([own_a, own_b]).T[ok], np.vstack([a_src, b_src]).T[ok], runs, r_tag, src,
                                 lumi, ddir / f"D6_lines_vs_{src}_{suffix}.png", args.save_pdf)

        # D7: pT sub-ranges with the 15-40 GeV f and lines
        sub = {}
        for pt in SUB_PT:
            smom = np.stack([loaded[r][r_tag]["sub_mom"][pt] for r in runs])[has_line]
            p = profile_from_moments(shift_moments(smom, delta[has_line]).sum(0))
            # same entry minimum as the main profile: a bin with all entries in one time bin has a ~0 error
            use = cols & (p[2] >= args.min_bin_entries) & np.isfinite(p[1]) & (p[1] > 0)
            sub[pt] = (np.where(use, p[0] - f_x, np.nan), np.where(use, p[1], np.nan))
            ok_b = use
            print(f"{r_tag}: raw pT {pt[0]}-{pt[1]}: corrected profile - f: chi2 vs 0 = "
                  f"{np.sum((sub[pt][0][ok_b] / p[1][ok_b]) ** 2):.0f}/{ok_b.sum()}, "
                  f"max |dev| (bins with err < 0.05 ns) = "
                  f"{np.nanmax(np.where(ok_b & (p[1] < 0.05), np.abs(sub[pt][0]), np.nan)):.3f} ns")
        plot_pt_dependence(xc, sub, runs, r_tag, src, lumi, ddir / f"D7_pt_dependence_{suffix}.png", args.save_pdf)

        write_csv(rows, ["run", "entries", "n_points", "chi2ndf_f_only", "chi2ndf_model", "a", "b", "a_err", "b_err",
                         "a_own", "b_own", "rms_raw", "rms_f", "rms_model"], ddir / f"per_run_{suffix}.csv")

        shape_json = tdir / f"shape_{suffix}.json"
        with shape_json.open("w") as f:
            json.dump({
                "hist": timing_hist_name(r_tag),
                "model": "t = f(x) + a_run + b_run*x, x = OHCal energy fraction, t = jet t - MBD t0 [ns]",
                "f": f"polynomial, coefficients ascending in x (f_0 + f_1 x + ... + f_{deg} x^{deg})",
                "degree": deg, "coefficients": coef.tolist(), "covariance": coef_cov.tolist(),
                "line_source": src, "lines_file": f"per_run_lines_{suffix_of(src)}.csv",
                "convention": "entries-weighted mean of a_run and b_run over runs is 0 for the source radius",
                "chi2": chi2, "ndf": ndf, "n_runs": n_runs_fit,
                "x_bins_used_lo_edges": T["xedges"][:-1][cols].tolist(),
                "x_range": [float(T["xedges"][:-1][cols].min()), float(T["xedges"][1:][cols].max())],
                "pooled_rms_ns": pooled,
                "settings": {k: getattr(args, k) for k in ("min_bin_entries", "fit_range", "min_run_entries",
                                                            "min_run_points", "min_run_bin_entries")},
            }, f, indent=2)
        print(f"Saved: {shape_json}")


if __name__ == "__main__":
    main()
