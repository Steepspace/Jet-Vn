#!/usr/bin/env python3
"""Run-combined iternf jet plots from Jet-Anav3 output (iter_is_noflow mode).

1. Raw jet pT (hJetPtv3_raw_<r>_iternf) summed over runs.
2. Leading jet (t - MBD t0) vs OHCal energy fraction (h2LeadJetTimeOHCalFrac_<r>_iternf_pt15_40) summed over runs,
   with its profileX and a quadratic fit to the profile.
3. Run-by-run diagnostics of how well the combined fit describes each run (fit held fixed).
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
from matplotlib.colors import LogNorm, Normalize
from matplotlib.ticker import FormatStrFormatter, LogLocator
import mplhep as hep
from scipy.optimize import curve_fit

from plot_jet_bkgsub_compare import save_fig
from plot_jet_noflow_compare import parse_list_file, load_luminosity

# (tag, R, |eta| acceptance used in Jet-Anav3.C: m_jet_eta_max_r02 / m_jet_eta_max_r03)
RADII = {
    "r02": {"r": 0.2, "eta_max": 0.9},
    "r03": {"r": 0.3, "eta_max": 0.8},
}

TIMING_PT = (15, 40)  # raw leading jet pT range of h2LeadJetTimeOHCalFrac (m_timing_pt_ranges[0])
JET_PT_MIN = 10       # calibrated jet pT cut of the 11-01-26 production (jet_pt_min job arg)

# Event selection shared by all histograms. In iter_is_noflow mode calo v2 = 0 and there is no flow failure,
# so the |Calo v2| and flow failure cuts of hJetPtv3 are not applied.
EVENT_INFO = (
    r"$|z| < 10$ cm & MB" "\n"
    "Good Calo-Cent\n"
    r"Centrality: 0–60%" "\n"
    "Iterative No Flow"
)

FIT_LABEL = r"$t = p_{0} + p_{1} f + p_{2} f^{2}$"
TIME_LABEL = r"Jet $t - t_{0}^{\mathrm{MBD}}$ [ns]"
FRAC_LABEL = "OHCal Energy Fraction"


def pt_hist_name(r_tag):
    return f"hJetPtv3_raw_{r_tag}_iternf"


def timing_hist_name(r_tag):
    return f"h2LeadJetTimeOHCalFrac_{r_tag}_iternf_pt{TIMING_PT[0]}_{TIMING_PT[1]}"


def timing_status_name(r_tag):
    return f"hLeadJetTimingStatus_{r_tag}_iternf"


# ---------------------------------------------------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------------------------------------------------

def load_run(task):
    """Reads the per-run histograms. Returns (run, data) where data is None if a histogram is missing."""
    run, file_path, r_tags = task
    data = {}
    try:
        with uproot.open(file_path) as rf:
            n_events = float(rf["hCentrality"].values().sum() - rf["hCentralityCaloFail"].values().sum())
            for r_tag in r_tags:
                h_pt = rf[pt_hist_name(r_tag)]
                h_t = rf[timing_hist_name(r_tag)]
                status = rf[timing_status_name(r_tag)].values()
                t_flow = h_t.values(flow=True)
                data[r_tag] = {
                    "pt_values": h_pt.values(),
                    "pt_variances": h_pt.variances(),
                    "pt_edges": h_pt.axis().edges(),
                    "t_values": h_t.values(),
                    "t_variances": h_t.variances(),
                    "t_xedges": h_t.axes[0].edges(),
                    "t_yedges": h_t.axes[1].edges(),
                    "t_filled": float(status[1]),
                    # Leading jets outside the plotted range: OHCal fraction < 0 / > 1, |t - t0| > 25 ns
                    "t_flow": np.array([t_flow[0].sum(), t_flow[-1].sum(),
                                        t_flow[1:-1, 0].sum() + t_flow[1:-1, -1].sum()]),
                }
            data["n_events"] = n_events
    except (KeyError, OSError) as e:
        print(f"Warning: run {run}: {e}, skipping")
        return run, None
    return run, data


# ---------------------------------------------------------------------------------------------------------------------
# Profile and fit
# ---------------------------------------------------------------------------------------------------------------------

def profile_x(values, variances, yedges):
    """ProfileX of a TH2 from its bins (same as TH2::ProfileX with the default error option).

    Returns per x bin: mean y, error on the mean (RMS / sqrt(N_eff)), and N_eff = (sum w)^2 / sum w^2.
    """
    yc = 0.5 * (yedges[:-1] + yedges[1:])
    sw = values.sum(axis=1)
    sw2 = variances.sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        mean = (values * yc).sum(axis=1) / sw
        rms2 = (values * yc ** 2).sum(axis=1) / sw - mean ** 2
        neff = np.where(sw2 > 0, sw ** 2 / sw2, 0.0)
        err = np.sqrt(np.clip(rms2, 0, None) / neff)
    return mean, err, neff


def quad(x, p0, p1, p2):
    return p0 + p1 * x + p2 * x ** 2


def quad_err(x, cov):
    """1 sigma uncertainty of the fitted curve at x from the parameter covariance."""
    jac = np.vstack([np.ones_like(x), x, x ** 2])
    return np.sqrt(np.einsum("ik,ij,jk->k", jac, cov, jac))


def fit_quad(x, y, err):
    """Weighted quadratic fit. Returns (params, cov, chi2, ndf)."""
    popt, pcov = curve_fit(quad, x, y, p0=np.polyfit(x, y, 2)[::-1], sigma=err, absolute_sigma=True)
    chi2 = float(np.sum(((y - quad(x, *popt)) / err) ** 2))
    return popt, pcov, chi2, len(x) - 3


def fit_mask(xc, err, neff, min_entries, fit_range):
    mask = (neff >= min_entries) & np.isfinite(err) & (err > 0)
    if fit_range is not None:
        mask &= (xc >= fit_range[0]) & (xc <= fit_range[1])
    return mask


# ---------------------------------------------------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------------------------------------------------

def add_header(ax, runs, r_jet, lumi, x=1.0):
    text = rf"{len(runs)} runs ({min(runs)}–{max(runs)}), $R = {r_jet:g}$"
    if lumi is not None:
        text += rf", $\mathcal{{L}} = {lumi:.1f}\ \mu\mathrm{{b}}^{{-1}}$"
    ax.text(x, 1.01, text, transform=ax.transAxes, ha="right", va="bottom", fontsize=13)


def pt_info(eta_max):
    return (f"{EVENT_INFO}\n"
            rf"$|\eta| < {eta_max:g}$, Energy > 0" "\n"
            rf"Calib. $p_{{T}} \geq {JET_PT_MIN}$ GeV")


def timing_info(eta_max):
    return (f"{EVENT_INFO}\n"
            rf"Leading jet: $|\eta| < {eta_max:g}$, Energy > 0" "\n"
            rf"Raw ${TIMING_PT[0]} \leq p_{{T}} < {TIMING_PT[1]}$ GeV")


def sci(x):
    """Mathtext scientific notation, e.g. 3.18 \\times 10^{8}."""
    exp = int(np.floor(np.log10(x))) if x > 0 else 0
    return rf"{x / 10 ** exp:.2f} \times 10^{{{exp}}}"


def fit_text(popt, pcov, chi2, ndf):
    perr = np.sqrt(np.diag(pcov))
    lines = [FIT_LABEL]
    for i, (p, e) in enumerate(zip(popt, perr)):
        lines.append(rf"$p_{{{i}}} = {p:.4f} \pm {e:.4f}$")
    lines.append(rf"$\chi^{{2}}/\mathrm{{ndf}} = {chi2:.1f}/{ndf}$")
    return "\n".join(lines)


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
# Plot 2: combined timing vs OHCal fraction
# ---------------------------------------------------------------------------------------------------------------------

def frac_outside_unit(values, xedges, flow):
    """Leading jets with OHCal fraction < 0 and > 1: bins of the histogram outside [0, 1] plus its x under/overflow.

    The 0 and 1 edges are bin edges in the current booking (-0.25 to 1.25 in 0.025 steps), so the split is exact.
    """
    xc = 0.5 * (xedges[:-1] + xedges[1:])
    return values[xc < 0].sum() + flow[0], values[xc > 1].sum() + flow[1]


def plot_timing(values, xedges, yedges, prof, mask, fit, runs, r_tag, lumi, output_path, flow, time_ylim=None, save_pdf=False):
    hep.style.use("ATLAS")
    meta = RADII[r_tag]
    mean, err, _ = prof
    popt, pcov, chi2, ndf = fit
    xc = 0.5 * (xedges[:-1] + xedges[1:])

    fig, ax = plt.subplots(figsize=(9, 7))
    z = np.where(values > 0, values, np.nan)
    pcm = ax.pcolormesh(xedges, yedges, z.T, cmap="viridis", norm=LogNorm(vmin=1, vmax=np.nanmax(z)), rasterized=True)
    cbar = fig.colorbar(pcm, ax=ax, pad=0.02)
    cbar.set_label("Counts")

    ax.errorbar(xc[mask], mean[mask], yerr=err[mask], fmt="o", color="black", markersize=4, capsize=0, linewidth=1.2,
                label="Profile", zorder=3)
    other = np.isfinite(mean) & ~mask
    if np.any(other):
        ax.errorbar(xc[other], mean[other], yerr=np.where(np.isfinite(err[other]), err[other], 0), fmt="o",
                    mfc="white", mec="gray", color="gray", markersize=4, capsize=0, linewidth=1.0,
                    label="Profile (not fitted)", zorder=3)

    xf = np.linspace(xc[mask].min(), xc[mask].max(), 200)
    yf = quad(xf, *popt)
    ef = quad_err(xf, pcov)
    ax.fill_between(xf, yf - ef, yf + ef, color="#e41a1c", alpha=0.35, linewidth=0, zorder=4)
    ax.plot(xf, yf, color="#e41a1c", linewidth=2, label="Quadratic fit", zorder=4)

    ax.set_xlim(xedges[0], xedges[-1])
    ax.set_ylim(*(time_ylim if time_ylim is not None else (yedges[0], yedges[-1])))
    ax.set_xlabel(FRAC_LABEL)
    ax.set_ylabel(TIME_LABEL)

    box = dict(facecolor="white", alpha=0.8, edgecolor="none", pad=3)
    add_header(ax, runs, meta["r"], lumi)
    ax.text(0.97, 0.97, timing_info(meta["eta_max"]), transform=ax.transAxes, ha="right", va="top", fontsize=12, bbox=box)
    ax.text(0.03, 0.03, fit_text(popt, pcov, chi2, ndf), transform=ax.transAxes, ha="left", va="bottom", fontsize=12, bbox=box)
    n_neg, n_over = frac_outside_unit(values, xedges, flow)
    total = values.sum() + flow.sum()
    ax.text(0.97, 0.03,
            f"Entries: {values.sum():.0f} in range / {total:.0f}\n"
            f"OHCal frac < 0: {n_neg:.0f} ({100 * n_neg / total:.1f}%)\n"
            f"OHCal frac > 1: {n_over:.0f} ({100 * n_over / total:.2f}%)\n"
            rf"$|t - t_{{0}}| > 25$ ns: {flow[2]:.0f}",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=11, bbox=box)
    ax.legend(loc="upper left", fontsize=12, frameon=True, framealpha=0.8, edgecolor="none")
    plt.subplots_adjust(left=0.13, right=0.97, bottom=0.12, top=0.94)
    save_fig(fig, output_path, save_pdf)
    plt.close(fig)


# ---------------------------------------------------------------------------------------------------------------------
# Plot 3: run-by-run diagnostics
# ---------------------------------------------------------------------------------------------------------------------

def run_diagnostics(per_run, xedges, yedges, mask, fit, min_run_entries, min_run_bin_entries):
    """Compares each run's profile with the fixed combined fit. Returns (rows, residual curves, skipped runs)."""
    popt, pcov, _, _ = fit
    xc = 0.5 * (xedges[:-1] + xedges[1:])
    yc = 0.5 * (yedges[:-1] + yedges[1:])
    f_x = quad(xc, *popt)
    f_err = quad_err(xc, pcov)
    cols = mask  # only the OHCal fraction columns the combined fit describes

    rows, curves, skipped = [], {}, []
    for run, (values, variances) in sorted(per_run.items()):
        entries = values[cols].sum()
        if entries < min_run_entries:
            skipped.append((run, entries))
            continue

        mean, err, neff = profile_x(values, variances, yedges)
        m = cols & (neff >= min_run_bin_entries) & np.isfinite(err) & (err > 0)
        if m.sum() < 4:
            skipped.append((run, entries))
            continue

        resid = mean[m] - f_x[m]
        sigma = np.sqrt(err[m] ** 2 + f_err[m] ** 2)
        pulls = resid / sigma
        w = 1.0 / sigma ** 2

        row = {
            "run": run,
            "entries": float(entries),
            "n_points": int(m.sum()),
            "chi2ndf_fixed": float(np.sum(pulls ** 2) / m.sum()),
            "mean_resid": float(np.sum(w * resid) / np.sum(w)),
            "mean_resid_err": float(1.0 / np.sqrt(np.sum(w))),
        }
        try:
            rp, rc, rchi2, rndf = fit_quad(xc[m], mean[m], err[m])
            rerr = np.sqrt(np.diag(rc))
            row["chi2ndf_refit"] = rchi2 / rndf if rndf > 0 else np.nan
            for i in range(3):
                row[f"p{i}"] = float(rp[i])
                row[f"p{i}_err"] = float(rerr[i])
        except (RuntimeError, ValueError):
            row["chi2ndf_refit"] = np.nan
            for i in range(3):
                row[f"p{i}"] = row[f"p{i}_err"] = np.nan

        # Timing RMS in the fitted OHCal fraction range, before and after t -> t - f(OHCal frac), from the bin centers
        v = values[cols]
        sw = v.sum()
        t_raw = np.broadcast_to(yc, v.shape)
        t_corr = t_raw - f_x[cols][:, None]
        for key, t in (("raw", t_raw), ("corr", t_corr)):
            mu = np.sum(v * t) / sw
            row[f"rms_{key}"] = float(np.sqrt(np.sum(v * (t - mu) ** 2) / sw))
            row[f"mean_{key}"] = float(mu)

        rows.append(row)
        curves[run] = (xc[m], resid, sigma, pulls)
    return rows, curves, skipped


def setup_run_axis(ax, runs):
    # Label every run when few, otherwise ~40 evenly spaced runs (x is the run index, so gaps between runs are not stretched)
    step = max(1, int(np.ceil(len(runs) / 40)))
    ticks = np.arange(0, len(runs), step)
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(runs[i]) for i in ticks], rotation=90, fontsize=7)
    ax.set_xlim(-1, len(runs))
    ax.set_xlabel("Run", labelpad=6)
    ax.tick_params(axis="x", which="minor", bottom=False, top=False)


def diag_header(ax, rows, r_tag, lumi, x=1.0):
    add_header(ax, [r["run"] for r in rows], RADII[r_tag]["r"], lumi, x=x)


def plot_diag_chi2(rows, r_tag, lumi, output_path, save_pdf):
    hep.style.use("ATLAS")
    runs = [r["run"] for r in rows]
    idx = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(idx, [r["chi2ndf_fixed"] for r in rows], "o", color="#e41a1c", markersize=6, label="Combined fit (fixed)")
    ax.plot(idx, [r["chi2ndf_refit"] for r in rows], "s", mfc="none", color="#377eb8", markersize=6, label="Per-run refit")
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1)
    ax.set_ylabel(r"$\chi^{2}/\mathrm{ndf}$ of run profile")
    ax.set_yscale("log")
    vals = np.array([r[k] for r in rows for k in ("chi2ndf_fixed", "chi2ndf_refit")], dtype=float)
    vals = vals[np.isfinite(vals) & (vals > 0)]
    ax.set_ylim(min(0.5, vals.min() * 0.7), max(3.0, vals.max() * 3))
    setup_run_axis(ax, runs)
    diag_header(ax, rows, r_tag, lumi)
    ax.legend(loc="upper left", frameon=False, fontsize=13)
    plt.subplots_adjust(left=0.08, right=0.98, bottom=0.20, top=0.93)
    save_fig(fig, output_path, save_pdf)
    plt.close(fig)


def plot_diag_offset(rows, r_tag, lumi, output_path, save_pdf):
    hep.style.use("ATLAS")
    runs = [r["run"] for r in rows]
    idx = np.arange(len(rows))
    y = np.array([r["mean_resid"] for r in rows])
    e = np.array([r["mean_resid_err"] for r in rows])
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axhline(0.0, color="gray", linestyle="--", linewidth=1)
    ax.errorbar(idx, y, yerr=e, fmt="o", color="black", markersize=5, capsize=0, linewidth=1.2)
    w = 1 / e ** 2
    mu = np.sum(w * y) / np.sum(w)
    chi2 = np.sum(((y - mu) / e) ** 2)
    ax.text(0.02, 0.95,
            f"Weighted mean = {mu:.3f} ns\nRMS over runs = {np.std(y):.3f} ns\n"
            rf"$\chi^{{2}}/\mathrm{{ndf}}$ (constant) = {chi2:.1f}/{len(y) - 1}",
            transform=ax.transAxes, ha="left", va="top", fontsize=13)
    ax.set_ylabel(r"$\langle$Run profile $-$ fit$\rangle$ [ns]")
    span = np.max(np.abs(y) + e)
    ax.set_ylim(-1.6 * span, 1.6 * span)
    setup_run_axis(ax, runs)
    diag_header(ax, rows, r_tag, lumi)
    plt.subplots_adjust(left=0.08, right=0.98, bottom=0.20, top=0.93)
    save_fig(fig, output_path, save_pdf)
    plt.close(fig)


def plot_diag_residuals(rows, curves, xedges, r_tag, lumi, output_path, save_pdf):
    hep.style.use("ATLAS")
    runs = [r["run"] for r in rows]
    fig, (ax, axp) = plt.subplots(1, 2, figsize=(15, 6.5), gridspec_kw={"width_ratios": [2.2, 1], "wspace": 0.25})

    cmap = plt.get_cmap("viridis")
    norm = Normalize(vmin=min(runs), vmax=max(runs))
    for run in runs:
        x, resid, _, _ = curves[run]
        ax.plot(x, resid, "-", color=cmap(norm(run)), linewidth=0.8, alpha=0.6)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = fig.colorbar(sm, ax=ax, pad=0.01, format=FormatStrFormatter("%d"))
    cbar.set_label("Run")

    # Error-weighted mean residual over runs per OHCal fraction bin
    xc = 0.5 * (xedges[:-1] + xedges[1:])
    sw = np.zeros_like(xc)
    swr = np.zeros_like(xc)
    for run in runs:
        x, resid, sigma, _ = curves[run]
        ib = np.searchsorted(xedges, x) - 1
        np.add.at(sw, ib, 1 / sigma ** 2)
        np.add.at(swr, ib, resid / sigma ** 2)
    has = sw > 0
    ax.axhline(0.0, color="gray", linestyle="--", linewidth=1)
    ax.errorbar(xc[has], swr[has] / sw[has], yerr=1 / np.sqrt(sw[has]), fmt="o", color="black", markersize=5,
                capsize=0, linewidth=1.5, label="Weighted mean over runs", zorder=5)
    ax.set_xlabel(FRAC_LABEL)
    ax.set_ylabel(r"Run profile $-$ combined fit [ns]")
    all_resid = np.concatenate([curves[r][1] for r in runs])
    lim = np.percentile(np.abs(all_resid), 98) * 1.3
    ax.set_ylim(-lim, lim)
    ax.legend(loc="upper left", frameon=False, fontsize=12)
    diag_header(ax, rows, r_tag, lumi)

    # Pulls against the fixed combined fit, and with each run's own constant offset removed (shape-only test)
    pulls = np.concatenate([curves[r][3] for r in runs])
    pulls_shape = np.concatenate([(curves[r][1] - row["mean_resid"]) / curves[r][2] for r, row in zip(runs, rows)])
    lim = max(5.0, np.ceil(np.percentile(np.abs(pulls), 99.5)))
    bins = np.linspace(-lim, lim, 51)
    bw = bins[1] - bins[0]
    xs = np.linspace(-lim, lim, 400)
    gauss = lambda x, m, s, n: n * bw / (np.sqrt(2 * np.pi) * s) * np.exp(-0.5 * ((x - m) / s) ** 2)
    stats = []
    for p, color, label in ((pulls, "black", "Combined fit"), (pulls_shape, "#377eb8", "Fit + per-run offset")):
        axp.hist(np.clip(p, bins[0] + 1e-9, bins[-1] - 1e-9), bins=bins, histtype="step", color=color, linewidth=1.8,
                 label=label)
        stats.append(f"{label}: mean {np.mean(p):.2f}, std {np.std(p):.2f}")
    axp.plot(xs, gauss(xs, 0, 1, len(pulls)), "--", color="gray", linewidth=1.5, label=r"$\mathcal{N}(0, 1)$")
    axp.text(0.03, 0.97, "\n".join(stats) + f"\nN = {len(pulls)} (last bins incl. overflow)", transform=axp.transAxes,
             ha="left", va="top", fontsize=10)
    axp.set_xlabel(r"(Run profile $-$ fit) / $\sigma$")
    axp.set_ylabel("Entries")
    axp.set_xlim(-lim, lim)
    axp.set_ylim(0, axp.get_ylim()[1] * 1.5)
    axp.legend(loc="upper right", bbox_to_anchor=(1.0, 0.82), frameon=False, fontsize=10)
    plt.subplots_adjust(left=0.07, right=0.98, bottom=0.13, top=0.93)
    save_fig(fig, output_path, save_pdf)
    plt.close(fig)


def plot_diag_params(rows, fit, r_tag, lumi, output_path, save_pdf):
    hep.style.use("ATLAS")
    popt, pcov, _, _ = fit
    perr = np.sqrt(np.diag(pcov))
    runs = [r["run"] for r in rows]
    idx = np.arange(len(rows))
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True, gridspec_kw={"hspace": 0.08})
    for i, ax in enumerate(axes):
        y = np.array([r[f"p{i}"] for r in rows])
        e = np.array([r[f"p{i}_err"] for r in rows])
        ax.axhspan(popt[i] - perr[i], popt[i] + perr[i], color="#e41a1c", alpha=0.3, linewidth=0)
        ax.axhline(popt[i], color="#e41a1c", linewidth=1.5, label="Combined fit" if i == 0 else None)
        ax.errorbar(idx, y, yerr=e, fmt="o", color="black", markersize=4, capsize=0, linewidth=1.0,
                    label="Per-run refit" if i == 0 else None)
        ok = np.isfinite(y) & np.isfinite(e)
        if np.any(ok):
            lo, hi = np.percentile(y[ok] - e[ok], 2), np.percentile(y[ok] + e[ok], 98)
            pad = 0.15 * (hi - lo)
            ax.set_ylim(min(lo, popt[i] - perr[i]) - pad, max(hi, popt[i] + perr[i]) + pad)
        ax.set_ylabel(rf"$p_{{{i}}}$")
    setup_run_axis(axes[-1], runs)
    axes[0].legend(loc="upper left", frameon=False, fontsize=12, ncol=2)
    axes[0].text(0.99, 0.95, FIT_LABEL, transform=axes[0].transAxes, ha="right", va="top", fontsize=13)
    diag_header(axes[0], rows, r_tag, lumi)
    plt.subplots_adjust(left=0.09, right=0.98, bottom=0.13, top=0.95)
    save_fig(fig, output_path, save_pdf)
    plt.close(fig)


def plot_diag_rms(rows, r_tag, lumi, output_path, save_pdf):
    hep.style.use("ATLAS")
    runs = [r["run"] for r in rows]
    idx = np.arange(len(rows))
    raw = np.array([r["rms_raw"] for r in rows])
    corr = np.array([r["rms_corr"] for r in rows])
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(idx, raw, "o", color="#377eb8", markersize=6, label=r"$t - t_{0}$")
    ax.plot(idx, corr, "s", color="#e41a1c", markersize=6, label=r"$t - t_{0} - f(\mathrm{OHCal\ frac})$ (combined fit)")
    ax.set_ylabel("Timing RMS [ns]")
    lo, hi = min(raw.min(), corr.min()), max(raw.max(), corr.max())
    ax.set_ylim(lo - 0.3 * (hi - lo), hi + 0.6 * (hi - lo))
    ax.text(0.99, 0.95, f"Median RMS: {np.median(raw):.2f} → {np.median(corr):.2f} ns\n"
            f"Runs with smaller RMS after: {np.sum(corr < raw)}/{len(rows)}",
            transform=ax.transAxes, ha="right", va="top", fontsize=13)
    setup_run_axis(ax, runs)
    diag_header(ax, rows, r_tag, lumi)
    ax.legend(loc="upper left", frameon=False, fontsize=13)
    plt.subplots_adjust(left=0.08, right=0.98, bottom=0.20, top=0.93)
    save_fig(fig, output_path, save_pdf)
    plt.close(fig)


def write_csv(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = ["run", "entries", "n_points", "chi2ndf_fixed", "chi2ndf_refit", "mean_resid", "mean_resid_err",
            "p0", "p0_err", "p1", "p1_err", "p2", "p2_err", "mean_raw", "rms_raw", "mean_corr", "rms_corr"]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved: {path}")


# ---------------------------------------------------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run-combined iternf raw jet pT and leading jet timing vs OHCal fraction (profile + quadratic fit), "
                    "with run-by-run diagnostics of the combined fit."
    )
    parser.add_argument("-f", "--file", type=Path,
                        default=Path("/sphenix/user/anarde/sEPD-Study/jetAnav3/11-01-26-v2/test.list"),
                        help="List of Jet-Anav3 merged ROOT files named <run>.root (default: %(default)s)")
    parser.add_argument("-l", "--lumi-list", type=Path,
                        default=Path("/sphenix/user/anarde/sEPD-Study/f4a_bkgsub/10-30-26/test.list"),
                        help="List of Event QA ROOT files named <run>.root with hLuminosity; the label is omitted "
                             "unless every run has one (default: %(default)s)")
    parser.add_argument("-o", "--output-dir", type=Path,
                        default=Path("/sphenix/user/anarde/sEPD-Study/plots/Jet-Ana/iternf_timing"),
                        help="Output directory (default: %(default)s)")
    parser.add_argument("--runs", nargs="*", type=int, help="Specific run number(s) to combine (default: all runs in the list)")
    parser.add_argument("--radii", nargs="+", choices=list(RADII), default=list(RADII), help="Jet radii (default: both)")
    parser.add_argument("--per-event", action="store_true", help="Normalize the raw pT spectrum per event and per GeV")
    parser.add_argument("--pt-max", type=float, default=None,
                        help="x-axis max for the raw pT plot (default: auto, plus a _pt60 version if auto > 60)")
    parser.add_argument("--time-ylim", nargs=2, type=float, default=None, metavar=("TMIN", "TMAX"),
                        help="y-axis range of the combined timing plot (default: full histogram range)")
    parser.add_argument("--min-bin-entries", type=float, default=20,
                        help="Minimum effective entries in an OHCal fraction bin of the combined profile to enter the fit (default: %(default)s)")
    parser.add_argument("--fit-range", nargs=2, type=float, default=None, metavar=("FMIN", "FMAX"),
                        help="Restrict the fit to this OHCal fraction range (applied on top of --min-bin-entries)")
    parser.add_argument("--min-run-entries", type=float, default=200,
                        help="Minimum timing entries in the fit range for a run to enter the diagnostics (default: %(default)s)")
    parser.add_argument("--min-run-bin-entries", type=float, default=10,
                        help="Minimum effective entries in an OHCal fraction bin of a run profile (default: %(default)s)")
    parser.add_argument("--no-pt", action="store_true", help="Skip the raw pT plot")
    parser.add_argument("--no-timing", action="store_true", help="Skip the timing plot, fit and diagnostics")
    parser.add_argument("--no-diag", action="store_true", help="Skip the run-by-run diagnostics")
    parser.add_argument("-j", "--jobs", type=int, default=min(8, os.cpu_count() or 1),
                        help="Parallel workers for reading the runs (default: %(default)s)")
    parser.add_argument("--save-pdf", action="store_true", help="Save PDF versions in addition to PNG")
    args = parser.parse_args()

    run_files = parse_list_file(args.file)
    if not run_files:
        print(f"Error: No ROOT files found in {args.file}")
        sys.exit(1)
    runs_req = sorted(args.runs) if args.runs else sorted(run_files)
    tasks = []
    for run in runs_req:
        path = run_files.get(run)
        if path is None or not path.exists():
            print(f"Warning: No file for run {run}, skipping")
            continue
        tasks.append((run, path, tuple(args.radii)))

    n_jobs = max(1, min(args.jobs, len(tasks)))
    print(f"Reading {len(tasks)} runs with {n_jobs} worker(s)")
    if n_jobs == 1:
        loaded = [load_run(t) for t in tasks]
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=n_jobs) as ex:
            loaded = list(ex.map(load_run, tasks))
    loaded = {run: d for run, d in loaded if d is not None}
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

    for r_tag in args.radii:
        ref = loaded[runs[0]][r_tag]

        if not args.no_pt:
            pt_vals = sum(loaded[r][r_tag]["pt_values"] for r in runs)
            pt_vars = sum(loaded[r][r_tag]["pt_variances"] for r in runs)
            print(f"{r_tag}: combined raw pT jets = {pt_vals.sum():.0f}, N_evt = {n_events:.0f}")
            plot_raw_pt(pt_vals, pt_vars, ref["pt_edges"], n_events, runs, r_tag, lumi, args.per_event,
                        out / "pt" / f"combined_{pt_hist_name(r_tag)}.png", pt_max=args.pt_max, save_pdf=args.save_pdf)

        if args.no_timing:
            continue

        xedges, yedges = ref["t_xedges"], ref["t_yedges"]
        xc = 0.5 * (xedges[:-1] + xedges[1:])
        t_vals = sum(loaded[r][r_tag]["t_values"] for r in runs)
        t_vars = sum(loaded[r][r_tag]["t_variances"] for r in runs)
        filled = sum(loaded[r][r_tag]["t_filled"] for r in runs)
        t_flow = sum(loaded[r][r_tag]["t_flow"] for r in runs)
        n_neg, n_over = frac_outside_unit(t_vals, xedges, t_flow)
        print(f"{r_tag}: combined timing entries = {t_vals.sum():.0f} in range + {t_flow.sum():.0f} flow "
              f"(OHCal frac <0: {n_neg:.0f}, >1: {n_over:.0f}, |t|>25: {t_flow[2]:.0f}; x range "
              f"[{xedges[0]:g}, {xedges[-1]:g}]); status 'Filled' = {filled:.0f}")

        prof = profile_x(t_vals, t_vars, yedges)
        mask = fit_mask(xc, prof[1], prof[2], args.min_bin_entries, args.fit_range)
        if mask.sum() < 4:
            print(f"Error: {r_tag}: only {mask.sum()} profile bins pass the fit selection, skipping timing")
            continue
        fit = fit_quad(xc[mask], prof[0][mask], prof[1][mask])
        popt, pcov, chi2, ndf = fit
        print(f"{r_tag}: fit p = {popt}, chi2/ndf = {chi2:.1f}/{ndf}, OHCal frac range [{xedges[:-1][mask].min():.2f}, {xedges[1:][mask].max():.2f}]")

        tname = timing_hist_name(r_tag)
        plot_timing(t_vals, xedges, yedges, prof, mask, fit, runs, r_tag, lumi,
                    out / "timing" / f"combined_{tname}.png", t_flow, time_ylim=args.time_ylim, save_pdf=args.save_pdf)

        fit_json = out / "timing" / f"fit_{r_tag}_iternf_pt{TIMING_PT[0]}_{TIMING_PT[1]}.json"
        fit_json.parent.mkdir(parents=True, exist_ok=True)
        with fit_json.open("w") as f:
            json.dump({
                "hist": tname,
                "function": "p0 + p1*f + p2*f^2, f = OHCal energy fraction, t in ns (jet t - MBD t0)",
                "params": popt.tolist(),
                "errors": np.sqrt(np.diag(pcov)).tolist(),
                "covariance": pcov.tolist(),
                "chi2": chi2,
                "ndf": ndf,
                "fit_bins_lo_edges": xedges[:-1][mask].tolist(),
                "fit_range": [float(xedges[:-1][mask].min()), float(xedges[1:][mask].max())],
                "min_bin_entries": args.min_bin_entries,
                "runs": runs,
            }, f, indent=2)
        print(f"Saved: {fit_json}")

        if args.no_diag:
            continue

        per_run = {r: (loaded[r][r_tag]["t_values"], loaded[r][r_tag]["t_variances"]) for r in runs}
        rows, curves, skipped = run_diagnostics(per_run, xedges, yedges, mask, fit, args.min_run_entries, args.min_run_bin_entries)
        if skipped:
            print(f"{r_tag}: skipped {len(skipped)} run(s) in diagnostics (too few entries): "
                  + ", ".join(f"{r} ({n:.0f})" for r, n in skipped))
        if len(rows) < 2:
            print(f"{r_tag}: fewer than 2 runs for diagnostics, skipping")
            continue

        ddir = out / "timing" / "diagnostics"
        suffix = f"{r_tag}_iternf_pt{TIMING_PT[0]}_{TIMING_PT[1]}"
        plot_diag_chi2(rows, r_tag, lumi, ddir / f"D1_chi2ndf_{suffix}.png", args.save_pdf)
        plot_diag_offset(rows, r_tag, lumi, ddir / f"D2_offset_{suffix}.png", args.save_pdf)
        plot_diag_residuals(rows, curves, xedges, r_tag, lumi, ddir / f"D3_residuals_pulls_{suffix}.png", args.save_pdf)
        plot_diag_params(rows, fit, r_tag, lumi, ddir / f"D4_params_{suffix}.png", args.save_pdf)
        plot_diag_rms(rows, r_tag, lumi, ddir / f"D5_rms_{suffix}.png", args.save_pdf)
        write_csv(rows, ddir / f"per_run_{suffix}.csv")


if __name__ == "__main__":
    main()
