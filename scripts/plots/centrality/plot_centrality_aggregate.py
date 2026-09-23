#!/usr/bin/env python3

import os
import sys
import re
import csv
import argparse
import functools
import traceback
from pathlib import Path
import concurrent.futures

import numpy as np
import uproot
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mplhep as hep
import tqdm

# Ensure local imports from the same directory work reliably
script_dir = Path(__file__).resolve().parent
if str(script_dir) not in sys.path:
    sys.path.insert(0, str(script_dir))

try:
    import plot_centrality_qa
except ImportError:
    plot_centrality_qa = None


def compute_centrality_average(values, edges, cent_min=10.0, cent_max=70.0):
    """
    Compute average y-value excluding outliers at the edges (e.g. low-centrality
    turn-on / uncalibrated bins and high-centrality trigger roll-off / zeros).
    """
    bin_centers = 0.5 * (edges[:-1] + edges[1:])
    mask = (bin_centers >= cent_min) & (bin_centers <= cent_max) & (values > 0)
    if np.sum(mask) >= 3:
        vals = values[mask]
        med = np.median(vals)
        good = np.abs(vals - med) < 0.3 * med
        if np.any(good):
            return float(np.mean(vals[good]))
        return float(np.mean(vals))
    pos_vals = values[values > 0]
    if len(pos_vals) > 0:
        return float(np.median(pos_vals))
    return 0.0


def extract_run_metrics(file_path, hist_name="hCentrality", cent_flat_min=10.0, cent_flat_max=70.0,
                        max_rms_pct=4.5, max_dev_pct_cut=8.0,
                        max_slope_per_10pct=2.5, min_central_ratio=0.80,
                        max_central_ratio=1.20, max_central_spike=1.30,
                        syst_floor=0.02):
    """
    Fast extraction of scalar flatness and QA metrics from a single ROOT file.
    Does not render any plots.
    """
    path = Path(file_path)
    if not path.exists():
        return None, f"File not found: {path}"

    try:
        try:
            run_number = int(path.name.split('.')[0])
        except ValueError:
            match = re.search(r'\d+', path.name)
            if match:
                run_number = int(match.group())
            else:
                return None, f"Could not parse run number from {path.name}"

        with uproot.open(path) as f:
            if hist_name not in f:
                return None, f"Histogram '{hist_name}' not found in {path.name}"

            hist = f[hist_name]
            values, edges = hist.to_numpy()

        total_events = float(np.sum(values))
        bin_centers = 0.5 * (edges[:-1] + edges[1:])

        # Plateau average
        plateau_avg = compute_centrality_average(values, edges, cent_min=cent_flat_min, cent_max=cent_flat_max)

        flags = []

        if plateau_avg <= 0 or total_events == 0:
            return {
                "run_number": run_number,
                "file_path": str(path),
                "total_events": total_events,
                "plateau_avg": 0.0,
                "rms_plat_pct": np.nan,
                "max_dev_pct": np.nan,
                "chi2_ndf": np.nan,
                "slope_per_10pct": np.nan,
                "ratio_1": np.nan,
                "ratio_1_5": np.nan,
                "ratio_80": np.nan,
                "values": values,
                "ratios": np.zeros(len(values)),
                "bin_centers": bin_centers,
                "edges": edges,
                "status": "FLAG_EMPTY_OR_ZERO",
            }, None

        # Normalized ratio array across all bins
        ratios = np.where(values > 0, values / plateau_avg, 0.0)

        # 1. Plateau Metrics (cent_flat_min to cent_flat_max)
        plateau_mask = (bin_centers >= cent_flat_min) & (bin_centers <= cent_flat_max)
        n_plateau_bins = int(np.sum(plateau_mask))

        if n_plateau_bins > 1:
            vals_plat = values[plateau_mask]
            ratios_plat = ratios[plateau_mask]
            dev_plat = ratios_plat - 1.0

            # Statistics-independent RMS flatness across plateau (%)
            rms_plat_pct = float(np.sqrt(np.mean(dev_plat ** 2))) * 100.0

            # Peak maximum relative deviation (%)
            max_dev_pct = float(np.max(np.abs(dev_plat))) * 100.0

            # Linear slope across plateau (% variation per 10% centrality)
            x_plat = bin_centers[plateau_mask]
            poly = np.polyfit(x_plat, ratios_plat, deg=1)
            raw_slope = float(poly[0])
            slope_per_10pct = raw_slope * 10.0 * 100.0

            # Statistics-aware chi2 with systematic floor (f_syst = 0.02)
            var_plat = np.where(vals_plat > 0, vals_plat, 1.0) + (syst_floor * plateau_avg) ** 2
            chi2 = float(np.sum(((vals_plat - plateau_avg) ** 2) / var_plat))
            chi2_ndf = chi2 / (n_plateau_bins - 1)
        else:
            rms_plat_pct = np.nan
            max_dev_pct = np.nan
            slope_per_10pct = np.nan
            chi2_ndf = np.nan

        # 2. 1% Centrality Bin (Bin 1: edges 0.5 to 1.5, center 1.0)
        # Note: Bin 0 [-0.5, 0.5] represents 0% centrality, but is currently empty in all runs
        if len(ratios) > 1 and values[1] > 0:
            ratio_1 = float(ratios[1])
        else:
            ratio_1 = 0.0

        # 3. 1-5% Centrality Bins (Bins 1 to 5: edges 0.5 to 5.5, centers 1.0 to 5.0)
        # Avoids the empty 0% centrality bin (Bin 0)
        cent15_mask = (bin_centers >= 0.5) & (bin_centers <= 5.5) & (values > 0)
        if np.any(cent15_mask):
            ratio_1_5 = float(np.mean(ratios[cent15_mask]))
        else:
            ratio_1_5 = 0.0

        # 4. Peripheral Ratio (near 80% Centrality: 75-85% window)
        cent80_mask = (bin_centers >= 75.0) & (bin_centers <= 85.0) & (values > 0)
        if np.any(cent80_mask):
            ratio_80 = float(np.mean(ratios[cent80_mask]))
        else:
            ratio_80 = 0.0

        # 5. Evaluate Outlier Rules
        if not np.isnan(rms_plat_pct) and rms_plat_pct > max_rms_pct:
            flags.append("FLAG_NON_FLAT")
        elif not np.isnan(max_dev_pct) and max_dev_pct > max_dev_pct_cut:
            flags.append("FLAG_NON_FLAT")

        if not np.isnan(slope_per_10pct) and abs(slope_per_10pct) > max_slope_per_10pct:
            flags.append("FLAG_SLOPE_DRIFT")

        # Central anomaly: either sharp spike at 1% or broader peak/drop in 1-5%
        if not np.isnan(ratio_1) and ratio_1 > max_central_spike:
            flags.append("FLAG_CENTRAL_SPIKE")
        elif not np.isnan(ratio_1_5) and ratio_1_5 > max_central_ratio:
            flags.append("FLAG_CENTRAL_SPIKE")
        elif not np.isnan(ratio_1_5) and ratio_1_5 < min_central_ratio:
            flags.append("FLAG_CENTRAL_DROP")

        status = ";".join(flags) if flags else "GOOD"

        metric_dict = {
            "run_number": run_number,
            "file_path": str(path),
            "total_events": total_events,
            "plateau_avg": plateau_avg,
            "rms_plat_pct": rms_plat_pct,
            "max_dev_pct": max_dev_pct,
            "chi2_ndf": chi2_ndf,
            "slope_per_10pct": slope_per_10pct,
            "ratio_1": ratio_1,
            "ratio_1_5": ratio_1_5,
            "ratio_80": ratio_80,
            "values": values,
            "ratios": ratios,
            "bin_centers": bin_centers,
            "edges": edges,
            "status": status,
        }
        return metric_dict, None

    except Exception as e:
        return None, f"Error processing {path.name}: {e}"


def plot_heatmap(data_list, output_path, test_runs=None):
    """
    Plot 1: 2D Centrality vs. Run Heatmap of Ratios to Plateau Average.
    Excludes empty 0% centrality Bin 0 [-0.5, 0.5] (empty across all runs) and displays bins 1-99 (1% to 99% centrality).
    """
    hep.style.use("ATLAS")
    fig, ax = plt.subplots(figsize=(12, 6.5))

    runs = [d["run_number"] for d in data_list]
    n_runs = len(runs)
    # Slicing out Bin 0 (0% centrality, empty across all runs) so columns represent 1% to 99% centrality
    ratio_matrix = np.array([d["ratios"][1:] for d in data_list])
    cent_edges = data_list[0]["edges"][1:]

    x_grid, y_grid = np.meshgrid(np.arange(n_runs + 1), cent_edges)

    c = ax.pcolormesh(
        x_grid,
        y_grid,
        ratio_matrix.T,
        cmap="coolwarm",
        vmin=0.8,
        vmax=1.2,
        shading="flat",
    )

    cbar = fig.colorbar(c, ax=ax, pad=0.015, aspect=25, fraction=0.046)
    cbar.set_label("Ratio to Plateau Average", fontsize=15)
    cbar.ax.tick_params(labelsize=13)

    ax.set_ylabel("Centrality [%]", fontsize=16)
    ax.set_xlabel("Run Index", fontsize=16)

    # Tight boundaries to eliminate any white space inside the plot
    ax.set_xlim(0, n_runs)
    ax.set_ylim(cent_edges[0], cent_edges[-1])

    # Display clean integer Run Index numbers on x-axis
    if n_runs >= 2000:
        step = 500
    elif n_runs >= 800:
        step = 200
    elif n_runs >= 400:
        step = 100
    elif n_runs >= 100:
        step = 25
    elif n_runs >= 20:
        step = 10
    else:
        step = max(1, n_runs // 5)

    tick_positions = np.arange(0, n_runs + 1, step)
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([str(p) for p in tick_positions], fontsize=13)

    # Highlight test runs (horizontal annotations with tight top margin)
    if test_runs:
        for tr in test_runs:
            if tr in runs:
                idx = runs.index(tr)
                ax.axvline(idx + 0.5, color="black", linestyle="--", linewidth=1.8, alpha=0.9)
                ha = "left" if idx < n_runs * 0.06 else ("right" if idx > n_runs * 0.94 else "center")
                ax.text(idx + 0.5, cent_edges[-1] + 0.6, f"Run {tr}", rotation=0, ha=ha, va="bottom",
                        fontsize=11, fontweight="bold", color="darkred", clip_on=False)

    fig.tight_layout(pad=0.3)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_trends(data_list, output_path, test_runs=None, max_rms_pct=4.5, max_slope_per_10pct=2.5,
                max_central_spike=1.30, min_central_ratio=0.80,
                save_individual=True):
    """
    Plot 2: Flatness & Quality Trends vs. Run Number (4-panel diagnostic scatter).
    Also generates individual single-plot figures for each metric if save_individual=True.
    """
    hep.style.use("ATLAS")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    runs = np.array([d["run_number"] for d in data_list])
    rms_vals = np.array([d["rms_plat_pct"] for d in data_list])
    slopes = np.array([d["slope_per_10pct"] for d in data_list])
    r1s = np.array([d["ratio_1"] for d in data_list])
    r15s = np.array([d["ratio_1_5"] for d in data_list])
    r80s = np.array([d["ratio_80"] for d in data_list])
    nevts = np.array([d["total_events"] for d in data_list])
    statuses = [d["status"] for d in data_list]

    is_good = np.array([s == "GOOD" for s in statuses])
    is_outlier = ~is_good

    test_run_set = set(test_runs or [])
    is_test = np.array([r in test_run_set for r in runs])

    # 4-panel combined figure
    fig, axes = plt.subplots(4, 1, figsize=(13, 16), sharex=True)

    # 1. Plateau RMS Non-Flatness (%)
    ax1 = axes[0]
    ax1.scatter(runs[is_good], rms_vals[is_good], color="forestgreen", s=30, alpha=0.7, label="Good Runs", edgecolors="none")
    ax1.scatter(runs[is_outlier], rms_vals[is_outlier], color="crimson", s=45, alpha=0.85, marker="^", label="Flagged Runs")
    if np.any(is_test):
        ax1.scatter(runs[is_test], rms_vals[is_test], facecolors="none", edgecolors="black", s=140, linewidths=2.0, marker="o", label="Test Runs")
        for r, val in zip(runs[is_test], rms_vals[is_test]):
            if not np.isnan(val):
                ax1.annotate(f"{r}", (r, val), textcoords="offset points", xytext=(0, 10), ha="center", fontsize=11, fontweight="bold")

    ax1.axhline(3.0, color="darkgreen", linestyle=":", linewidth=1.5, label="Nominal (3.0%)")
    ax1.axhline(max_rms_pct, color="crimson", linestyle="--", linewidth=1.8, label=rf"Cut ({max_rms_pct}%)")
    ax1.set_ylabel("Plateau RMS [%]\n(10-70% Flatness)", fontsize=13)
    ax1.set_ylim(bottom=0.0)
    ax1.legend(loc="upper right", ncol=3, fontsize=11, frameon=True, framealpha=0.9)
    ax1.grid(True, linestyle="--", alpha=0.4)

    # 2. Slope per 10% Centrality
    ax2 = axes[1]
    ax2.scatter(runs[is_good], slopes[is_good], color="forestgreen", s=30, alpha=0.7, edgecolors="none")
    ax2.scatter(runs[is_outlier], slopes[is_outlier], color="crimson", s=45, alpha=0.85, marker="^")
    if np.any(is_test):
        ax2.scatter(runs[is_test], slopes[is_test], facecolors="none", edgecolors="black", s=140, linewidths=2.0, marker="o")
    ax2.axhline(0.0, color="gray", linestyle="-", linewidth=1.2)
    ax2.axhspan(-max_slope_per_10pct, max_slope_per_10pct, color="forestgreen", alpha=0.12, label=rf"Tolerance $\pm${max_slope_per_10pct}%")
    ax2.axhline(max_slope_per_10pct, color="crimson", linestyle="--", linewidth=1.5)
    ax2.axhline(-max_slope_per_10pct, color="crimson", linestyle="--", linewidth=1.5)
    ax2.set_ylabel("Plateau Slope\n[% per 10% Cent]", fontsize=13)
    ax2.legend(loc="upper right", fontsize=11, frameon=True, framealpha=0.9)
    ax2.grid(True, linestyle="--", alpha=0.4)

    # 3. Central & Peripheral Ratios
    ax3 = axes[2]
    ax3.scatter(runs, r1s, color="crimson", s=35, alpha=0.65, marker="^", label=r"$R_{1\%}$ (1% Centrality)")
    ax3.scatter(runs, r15s, color="royalblue", s=30, alpha=0.6, marker="o", label=r"$R_{1-5\%}$ (1–5% Centrality)")
    ax3.scatter(runs, r80s, color="darkorange", s=30, alpha=0.6, marker="s", label=r"$R_{80\%}$ (Peripheral $\sim$80%)")
    ax3.axhline(1.0, color="gray", linestyle="-", linewidth=1.2)
    ax3.axhline(max_central_spike, color="crimson", linestyle="--", linewidth=1.5, label=rf"Spike Cut ({max_central_spike})")
    ax3.axhline(min_central_ratio, color="crimson", linestyle=":", linewidth=1.5, label=rf"Drop Cut ({min_central_ratio})")
    ax3.axhspan(min_central_ratio, 1.20, color="royalblue", alpha=0.08)
    ax3.set_ylabel("Ratio to Plateau", fontsize=14)
    ax3.set_ylim(0.3, 1.9)
    ax3.legend(loc="upper right", ncol=3, fontsize=11, frameon=True, framealpha=0.9)
    ax3.grid(True, linestyle="--", alpha=0.4)

    # 4. Total Events
    ax4 = axes[3]
    ax4.scatter(runs[is_good], nevts[is_good], color="forestgreen", s=30, alpha=0.7, edgecolors="none")
    ax4.scatter(runs[is_outlier], nevts[is_outlier], color="crimson", s=45, alpha=0.85, marker="^")
    if np.any(is_test):
        ax4.scatter(runs[is_test], nevts[is_test], facecolors="none", edgecolors="black", s=140, linewidths=2.0, marker="o")
    ax4.set_ylabel("Total Events", fontsize=14)
    ax4.set_xlabel("Run Number", loc="center", fontsize=15)
    ax4.set_yscale("log")
    ax4.grid(True, linestyle="--", alpha=0.4)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Individual single-plot images
    if save_individual:
        stem = output_path.stem
        # 1. Single RMS
        fig_rms, ax_rms = plt.subplots(figsize=(11, 6.5))
        ax_rms.scatter(runs[is_good], rms_vals[is_good], color="forestgreen", s=32, alpha=0.7, label="Good Runs", edgecolors="none")
        ax_rms.scatter(runs[is_outlier], rms_vals[is_outlier], color="crimson", s=48, alpha=0.85, marker="^", label="Flagged Runs")
        if np.any(is_test):
            ax_rms.scatter(runs[is_test], rms_vals[is_test], facecolors="none", edgecolors="black", s=140, linewidths=2.0, marker="o", label="Test Runs")
            for r, val in zip(runs[is_test], rms_vals[is_test]):
                if not np.isnan(val):
                    ax_rms.annotate(f"{r}", (r, val), textcoords="offset points", xytext=(0, 10), ha="center", fontsize=11, fontweight="bold")
        ax_rms.axhline(3.0, color="darkgreen", linestyle=":", linewidth=1.5, label="Nominal (3.0%)")
        ax_rms.axhline(max_rms_pct, color="crimson", linestyle="--", linewidth=1.8, label=rf"Cut ({max_rms_pct}%)")
        ax_rms.set_xlabel("Run Number", loc="center", fontsize=15)
        ax_rms.set_ylabel("Plateau RMS Non-Flatness [%] (10–70% Centrality)", fontsize=14)
        ax_rms.set_ylim(bottom=0.0)
        ax_rms.legend(loc="upper right", ncol=3, fontsize=11, frameon=True, framealpha=0.9)
        ax_rms.grid(True, linestyle="--", alpha=0.4)
        fig_rms.tight_layout()
        fig_rms.savefig(output_path.parent / f"{stem}_rms.png", dpi=300, bbox_inches="tight")
        plt.close(fig_rms)

        # 2. Single Slope
        fig_slope, ax_slope = plt.subplots(figsize=(11, 6.5))
        ax_slope.scatter(runs[is_good], slopes[is_good], color="forestgreen", s=32, alpha=0.7, label="Good Runs", edgecolors="none")
        ax_slope.scatter(runs[is_outlier], slopes[is_outlier], color="crimson", s=48, alpha=0.85, marker="^", label="Flagged Runs")
        if np.any(is_test):
            ax_slope.scatter(runs[is_test], slopes[is_test], facecolors="none", edgecolors="black", s=140, linewidths=2.0, marker="o", label="Test Runs")
            for r, val in zip(runs[is_test], slopes[is_test]):
                if not np.isnan(val):
                    ax_slope.annotate(f"{r}", (r, val), textcoords="offset points", xytext=(0, 10), ha="center", fontsize=11, fontweight="bold")
        ax_slope.axhline(0.0, color="gray", linestyle="-", linewidth=1.2)
        ax_slope.axhspan(-max_slope_per_10pct, max_slope_per_10pct, color="forestgreen", alpha=0.12, label=rf"Tolerance $\pm${max_slope_per_10pct}%")
        ax_slope.axhline(max_slope_per_10pct, color="crimson", linestyle="--", linewidth=1.5)
        ax_slope.axhline(-max_slope_per_10pct, color="crimson", linestyle="--", linewidth=1.5)
        ax_slope.set_xlabel("Run Number", loc="center", fontsize=15)
        ax_slope.set_ylabel("Plateau Slope [% per 10% Centrality]", fontsize=14)
        ax_slope.legend(loc="upper right", ncol=3, fontsize=11, frameon=True, framealpha=0.9)
        ax_slope.grid(True, linestyle="--", alpha=0.4)
        fig_slope.tight_layout()
        fig_slope.savefig(output_path.parent / f"{stem}_slope.png", dpi=300, bbox_inches="tight")
        plt.close(fig_slope)

        # 3. Single Ratios
        fig_rat, ax_rat = plt.subplots(figsize=(11, 6.5))
        ax_rat.scatter(runs, r1s, color="crimson", s=35, alpha=0.65, marker="^", label=r"$R_{1\%}$ (1% Centrality)")
        ax_rat.scatter(runs, r15s, color="royalblue", s=30, alpha=0.6, marker="o", label=r"$R_{1-5\%}$ (1–5% Centrality)")
        ax_rat.scatter(runs, r80s, color="darkorange", s=30, alpha=0.6, marker="s", label=r"$R_{80\%}$ (Peripheral $\sim$80%)")
        ax_rat.axhline(1.0, color="gray", linestyle="-", linewidth=1.2)
        ax_rat.axhline(max_central_spike, color="crimson", linestyle="--", linewidth=1.5, label=rf"Spike Cut ({max_central_spike})")
        ax_rat.axhline(min_central_ratio, color="crimson", linestyle=":", linewidth=1.5, label=rf"Drop Cut ({min_central_ratio})")
        ax_rat.axhspan(min_central_ratio, 1.20, color="royalblue", alpha=0.08)
        ax_rat.set_xlabel("Run Number", loc="center", fontsize=15)
        ax_rat.set_ylabel("Ratio to Plateau Average", fontsize=14)
        ax_rat.set_ylim(0.3, 1.9)
        ax_rat.legend(loc="upper right", ncol=3, fontsize=11, frameon=True, framealpha=0.9)
        ax_rat.grid(True, linestyle="--", alpha=0.4)
        fig_rat.tight_layout()
        fig_rat.savefig(output_path.parent / f"{stem}_ratios.png", dpi=300, bbox_inches="tight")
        plt.close(fig_rat)

        # 4. Single Events
        fig_ev, ax_ev = plt.subplots(figsize=(11, 6.5))
        ax_ev.scatter(runs[is_good], nevts[is_good], color="forestgreen", s=32, alpha=0.7, label="Good Runs", edgecolors="none")
        ax_ev.scatter(runs[is_outlier], nevts[is_outlier], color="crimson", s=48, alpha=0.85, marker="^", label="Flagged Runs")
        if np.any(is_test):
            ax_ev.scatter(runs[is_test], nevts[is_test], facecolors="none", edgecolors="black", s=140, linewidths=2.0, marker="o", label="Test Runs")
            for r, val in zip(runs[is_test], nevts[is_test]):
                if not np.isnan(val):
                    ax_ev.annotate(f"{r}", (r, val), textcoords="offset points", xytext=(0, 10), ha="center", fontsize=11, fontweight="bold")
        ax_ev.set_xlabel("Run Number", loc="center", fontsize=15)
        ax_ev.set_ylabel("Total Events", fontsize=14)
        ax_ev.set_yscale("log")
        ax_ev.legend(loc="lower right", fontsize=11, frameon=True, framealpha=0.9)
        ax_ev.grid(True, linestyle="--", alpha=0.4)
        fig_ev.tight_layout()
        fig_ev.savefig(output_path.parent / f"{stem}_events.png", dpi=300, bbox_inches="tight")
        plt.close(fig_ev)



def plot_metric_distributions(data_list, output_path, max_rms_pct=4.5, max_slope_per_10pct=2.5,
                              max_central_spike=1.30, min_central_ratio=0.80,
                              save_individual=True):
    """
    Plot 3: Metric Distributions (Population Summary Histograms).
    Uses step-style histograms and logarithmic y-axes so outlier runs are clearly visible.
    Also generates individual single-plot figures for each metric if save_individual=True.
    """
    hep.style.use("ATLAS")
    if hasattr(data_list, "to_dict"):
        data_list = data_list.to_dict("records")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rms_vals = [d["rms_plat_pct"] for d in data_list if not np.isnan(d["rms_plat_pct"])]
    slopes = [d["slope_per_10pct"] for d in data_list if not np.isnan(d["slope_per_10pct"])]
    r1s = [d["ratio_1"] for d in data_list if not np.isnan(d["ratio_1"])]
    r15s = [d["ratio_1_5"] for d in data_list if not np.isnan(d["ratio_1_5"])]
    nevts = [d["total_events"] for d in data_list if d["total_events"] > 0]

    # Pre-calculate counts and percentages
    n_fail_rms = int(np.sum(np.array(rms_vals) > max_rms_pct)) if len(rms_vals) > 0 else 0
    pct_fail_rms = 100.0 * n_fail_rms / len(rms_vals) if len(rms_vals) > 0 else 0.0
    n_pass_rms = len(rms_vals) - n_fail_rms
    pct_pass_rms = 100.0 * n_pass_rms / len(rms_vals) if len(rms_vals) > 0 else 0.0

    n_fail_slope = int(np.sum(np.abs(np.array(slopes)) > max_slope_per_10pct)) if len(slopes) > 0 else 0
    pct_fail_slope = 100.0 * n_fail_slope / len(slopes) if len(slopes) > 0 else 0.0
    n_pass_slope = len(slopes) - n_fail_slope
    pct_pass_slope = 100.0 * n_pass_slope / len(slopes) if len(slopes) > 0 else 0.0

    n_spike = int(np.sum(np.array(r1s) > max_central_spike)) if len(r1s) > 0 else 0
    pct_spike = 100.0 * n_spike / len(r1s) if len(r1s) > 0 else 0.0
    n_drop = int(np.sum(np.array(r15s) < min_central_ratio)) if len(r15s) > 0 else 0
    pct_drop = 100.0 * n_drop / len(r15s) if len(r15s) > 0 else 0.0

    log_bins = None
    evts_box_text = ""
    if len(nevts) > 0:
        log_bins = np.logspace(np.log10(max(1, min(nevts))), np.log10(max(nevts)), 40)
        median_evts = float(np.median(nevts))
        total_evts = float(np.sum(nevts))
        evts_box_text = (
            rf"Total Runs: {len(nevts):,}" + "\n" +
            rf"Median: {median_evts:.2e} evts/run" + "\n" +
            rf"Total: {total_evts:.2e} events"
        )

    # -------------------------------------------------------------
    # 1. Unzoomed 4-panel combined figure (Full Outlier Range)
    # -------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # RMS Flatness distribution (Unzoomed)
    ax1 = axes[0, 0]
    if len(rms_vals) > 0:
        x1_max = max(12.0, max(rms_vals) * 1.08)
        bins1 = np.linspace(0, x1_max, 50)
        ax1.hist(rms_vals, bins=bins1, histtype="step", color="steelblue", linewidth=2.0)
        ax1.axvline(max_rms_pct, color="crimson", linestyle="--", linewidth=2, label=rf"Cut ({max_rms_pct}%)")
        rms_box_text = (
            rf"Pass ($\leq {max_rms_pct}\%$): {n_pass_rms:,} ({pct_pass_rms:.1f}%)" + "\n" +
            rf"Fail (> {max_rms_pct}%): {n_fail_rms:,} ({pct_fail_rms:.1f}%)" + "\n" +
            rf"Max RMS: {max(rms_vals):.1f}%"
        )
        ax1.text(0.96, 0.78, rms_box_text, transform=ax1.transAxes, ha="right", va="top", fontsize=10.0,
                 bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="gray", alpha=0.92, zorder=4))
        ax1.set_xlim(0, x1_max)
    ax1.set_xlabel("Plateau RMS Non-Flatness [%]")
    ax1.set_ylabel("Runs")
    ax1.set_yscale("log")
    ax1.set_ylim(0.5, 3500)
    ax1.legend(loc="upper right", fontsize=10.5)
    ax1.grid(True, linestyle="--", alpha=0.3, which="both")

    # Slope distribution (Unzoomed)
    ax2 = axes[0, 1]
    if len(slopes) > 0:
        x2_min = min(-5.0, min(slopes) * 1.2)
        x2_max = max(6.0, max(slopes) * 1.15)
        bins2 = np.linspace(x2_min, x2_max, 55)
        ax2.hist(slopes, bins=bins2, histtype="step", color="mediumseagreen", linewidth=2.0)
        ax2.axvline(max_slope_per_10pct, color="crimson", linestyle="--", linewidth=2)
        ax2.axvline(-max_slope_per_10pct, color="crimson", linestyle="--", linewidth=2, label=rf"Tol ($\pm${max_slope_per_10pct}%)")
        slope_box_text = (
            rf"In Tol: {n_pass_slope:,} ({pct_pass_slope:.1f}%)" + "\n" +
            rf"Fail (> $\pm${max_slope_per_10pct}%): {n_fail_slope:,} ({pct_fail_slope:.1f}%)" + "\n" +
            rf"Max Slope: {max(slopes):+.1f}%"
        )
        ax2.text(0.04, 0.94, slope_box_text, transform=ax2.transAxes, ha="left", va="top", fontsize=10.0,
                 bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="gray", alpha=0.92, zorder=4))
        ax2.set_xlim(x2_min, x2_max)
    ax2.set_xlabel("Plateau Slope [% per 10% Cent]")
    ax2.set_ylabel("Runs")
    ax2.set_yscale("log")
    ax2.set_ylim(0.5, 3500)
    ax2.legend(loc="upper right", fontsize=10.5)
    ax2.grid(True, linestyle="--", alpha=0.3, which="both")

    # R1 and R1-5 distributions (Unzoomed)
    ax3 = axes[1, 0]
    if len(r1s) > 0 and len(r15s) > 0:
        x3_max = max(2.5, max(r1s) * 1.08)
        bins3 = np.linspace(0, x3_max, 55)
        ax3.hist(r1s, bins=bins3, histtype="step", color="crimson", linewidth=2.0, label=r"$R_{1\%}$ (1% Centrality)")
        ax3.hist(r15s, bins=bins3, histtype="step", color="royalblue", linewidth=2.0, label=r"$R_{1-5\%}$ (1–5% Centrality)")
        ax3.axvline(1.0, color="gray", linestyle="-", linewidth=1.5)
        ax3.axvline(max_central_spike, color="crimson", linestyle="--", linewidth=2, label=rf"Spike Cut ({max_central_spike})")
        ax3.axvline(min_central_ratio, color="darkorange", linestyle="--", linewidth=2, label=rf"Drop Cut ({min_central_ratio})")
        ratio_box_text = (
            rf"Spike Fail (> {max_central_spike}): {n_spike:,} ({pct_spike:.1f}%)" + "\n" +
            rf"Drop Fail (< {min_central_ratio}): {n_drop:,} ({pct_drop:.1f}%)" + "\n" +
            rf"Max $R_{{1\%}}$: {max(r1s):.2f}"
        )
        ax3.text(0.96, 0.94, ratio_box_text, transform=ax3.transAxes, ha="right", va="top", fontsize=9.8,
                 bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="gray", alpha=0.92, zorder=4))
        ax3.set_xlim(0, x3_max)
    ax3.set_xlabel("Central Ratio to Plateau")
    ax3.set_ylabel("Runs")
    ax3.set_yscale("log")
    ax3.set_ylim(0.5, 4500)
    ax3.legend(loc="upper left", fontsize=9.2, frameon=True, framealpha=0.9)
    ax3.grid(True, linestyle="--", alpha=0.3, which="both")

    # Event count distribution
    ax4 = axes[1, 1]
    if len(nevts) > 0 and log_bins is not None:
        ax4.hist(nevts, bins=log_bins, histtype="step", color="coral", linewidth=2.0)
        ax4.text(0.04, 0.94, evts_box_text, transform=ax4.transAxes, ha="left", va="top", fontsize=10.5,
                 bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="gray", alpha=0.92, zorder=4))
        ax4.set_xscale("log")
    ax4.set_xlabel("Total Events per Run")
    ax4.set_ylabel("Runs")
    ax4.set_yscale("log")
    ax4.set_ylim(0.5, 3500)
    ax4.grid(True, linestyle="--", alpha=0.3, which="both")

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    # -------------------------------------------------------------
    # 2. Zoomed 4-panel combined figure (Threshold Focus)
    # -------------------------------------------------------------
    zoomed_combined_path = output_path.parent / f"{output_path.stem}_zoomed.png"
    fig_z, axes_z = plt.subplots(2, 2, figsize=(12, 10))

    # Panel 1: RMS Zoomed [0, 10]
    ax1_z = axes_z[0, 0]
    if len(rms_vals) > 0:
        r_c = np.clip(rms_vals, 0, 10)
        ax1_z.hist(r_c, bins=40, histtype="step", color="steelblue", linewidth=2.0)
        ax1_z.axvline(max_rms_pct, color="crimson", linestyle="--", linewidth=2, label=rf"Cut ({max_rms_pct}%)")
        rms_box_text_z = (
            rf"Pass ($\leq {max_rms_pct}\%$): {n_pass_rms:,} ({pct_pass_rms:.1f}%)" + "\n" +
            rf"Fail (> {max_rms_pct}%): {n_fail_rms:,} ({pct_fail_rms:.1f}%)" + "\n" +
            rf"(Clipped at 10%)"
        )
        ax1_z.text(0.96, 0.78, rms_box_text_z, transform=ax1_z.transAxes, ha="right", va="top", fontsize=10.0,
                   bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="gray", alpha=0.92, zorder=4))
        ax1_z.set_xlim(0, 10)
    ax1_z.set_xlabel("Plateau RMS Non-Flatness [%]")
    ax1_z.set_ylabel("Runs")
    ax1_z.set_yscale("log")
    ax1_z.set_ylim(0.5, 3500)
    ax1_z.legend(loc="upper right", fontsize=10.5)
    ax1_z.grid(True, linestyle="--", alpha=0.3, which="both")

    # Panel 2: Slope Zoomed [-6, 6]
    ax2_z = axes_z[0, 1]
    if len(slopes) > 0:
        s_c = np.clip(slopes, -6, 6)
        ax2_z.hist(s_c, bins=40, histtype="step", color="mediumseagreen", linewidth=2.0)
        ax2_z.axvline(max_slope_per_10pct, color="crimson", linestyle="--", linewidth=2)
        ax2_z.axvline(-max_slope_per_10pct, color="crimson", linestyle="--", linewidth=2, label=rf"Tol ($\pm${max_slope_per_10pct}%)")
        slope_box_text_z = (
            rf"In Tol: {n_pass_slope:,} ({pct_pass_slope:.1f}%)" + "\n" +
            rf"Fail (> $\pm${max_slope_per_10pct}%): {n_fail_slope:,} ({pct_fail_slope:.1f}%)" + "\n" +
            rf"(Clipped at $\pm$6%)"
        )
        ax2_z.text(0.04, 0.94, slope_box_text_z, transform=ax2_z.transAxes, ha="left", va="top", fontsize=10.0,
                   bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="gray", alpha=0.92, zorder=4))
        ax2_z.set_xlim(-6, 6)
    ax2_z.set_xlabel("Plateau Slope [% per 10% Cent]")
    ax2_z.set_ylabel("Runs")
    ax2_z.set_yscale("log")
    ax2_z.set_ylim(0.5, 3500)
    ax2_z.legend(loc="upper right", fontsize=10.5)
    ax2_z.grid(True, linestyle="--", alpha=0.3, which="both")

    # Panel 3: Central Ratios Zoomed [0.4, 2.0]
    ax3_z = axes_z[1, 0]
    if len(r1s) > 0 and len(r15s) > 0:
        r1_c = np.clip(r1s, 0.4, 2.0)
        r15_c = np.clip(r15s, 0.4, 2.0)
        bins3_z = np.linspace(0.4, 2.0, 35)
        ax3_z.hist(r1_c, bins=bins3_z, histtype="step", color="crimson", linewidth=2.0, label=r"$R_{1\%}$ (1% Centrality)")
        ax3_z.hist(r15_c, bins=bins3_z, histtype="step", color="royalblue", linewidth=2.0, label=r"$R_{1-5\%}$ (1–5% Centrality)")
        ax3_z.axvline(1.0, color="gray", linestyle="-", linewidth=1.5)
        ax3_z.axvline(max_central_spike, color="crimson", linestyle="--", linewidth=2, label=rf"Spike Cut ({max_central_spike})")
        ax3_z.axvline(min_central_ratio, color="darkorange", linestyle="--", linewidth=2, label=rf"Drop Cut ({min_central_ratio})")
        ratio_box_text_z = (
            rf"Spike Fail (> {max_central_spike}): {n_spike:,} ({pct_spike:.1f}%)" + "\n" +
            rf"Drop Fail (< {min_central_ratio}): {n_drop:,} ({pct_drop:.1f}%)" + "\n" +
            rf"(Clipped at [0.4, 2.0])"
        )
        ax3_z.text(0.96, 0.94, ratio_box_text_z, transform=ax3_z.transAxes, ha="right", va="top", fontsize=9.8,
                   bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="gray", alpha=0.92, zorder=4))
        ax3_z.set_xlim(0.4, 2.0)
    ax3_z.set_xlabel("Central Ratio to Plateau")
    ax3_z.set_ylabel("Runs")
    ax3_z.set_yscale("log")
    ax3_z.set_ylim(0.5, 4500)
    ax3_z.legend(loc="upper left", fontsize=9.2, frameon=True, framealpha=0.9)
    ax3_z.grid(True, linestyle="--", alpha=0.3, which="both")

    # Panel 4: Events
    ax4_z = axes_z[1, 1]
    if len(nevts) > 0 and log_bins is not None:
        ax4_z.hist(nevts, bins=log_bins, histtype="step", color="coral", linewidth=2.0)
        ax4_z.text(0.04, 0.94, evts_box_text, transform=ax4_z.transAxes, ha="left", va="top", fontsize=10.5,
                   bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="gray", alpha=0.92, zorder=4))
        ax4_z.set_xscale("log")
    ax4_z.set_xlabel("Total Events per Run")
    ax4_z.set_ylabel("Runs")
    ax4_z.set_yscale("log")
    ax4_z.set_ylim(0.5, 3500)
    ax4_z.grid(True, linestyle="--", alpha=0.3, which="both")

    fig_z.tight_layout()
    fig_z.savefig(zoomed_combined_path, dpi=300, bbox_inches="tight")
    plt.close(fig_z)

    # -------------------------------------------------------------
    # 3. Individual single-plot images (both unzoomed & zoomed)
    # -------------------------------------------------------------
    if save_individual:
        stem = output_path.stem

        # 1a. Single RMS (Unzoomed)
        fig1, ax1_s = plt.subplots(figsize=(9, 6.5))
        if len(rms_vals) > 0:
            x1_max = max(12.0, max(rms_vals) * 1.08)
            bins1 = np.linspace(0, x1_max, 50)
            ax1_s.hist(rms_vals, bins=bins1, histtype="step", color="steelblue", linewidth=2.0)
            ax1_s.axvline(max_rms_pct, color="crimson", linestyle="--", linewidth=2, label=rf"Cut ({max_rms_pct}%)")
            rms_box_text = (
                rf"Pass ($\leq {max_rms_pct}\%$): {n_pass_rms:,} ({pct_pass_rms:.1f}%)" + "\n" +
                rf"Fail (> {max_rms_pct}%): {n_fail_rms:,} ({pct_fail_rms:.1f}%)" + "\n" +
                rf"Max RMS: {max(rms_vals):.1f}%"
            )
            ax1_s.text(0.96, 0.78, rms_box_text, transform=ax1_s.transAxes, ha="right", va="top", fontsize=11.5,
                       bbox=dict(boxstyle="round,pad=0.38", facecolor="white", edgecolor="gray", alpha=0.92, zorder=4))
            ax1_s.set_xlim(0, x1_max)
        ax1_s.set_xlabel("Plateau RMS Non-Flatness [%]", fontsize=15)
        ax1_s.set_ylabel("Runs", fontsize=15)
        ax1_s.set_yscale("log")
        ax1_s.set_ylim(0.5, 3500)
        ax1_s.legend(loc="upper right", fontsize=12)
        ax1_s.grid(True, linestyle="--", alpha=0.3, which="both")
        fig1.tight_layout()
        fig1.savefig(output_path.parent / f"{stem}_rms.png", dpi=300, bbox_inches="tight")
        plt.close(fig1)

        # 1b. Single RMS (Zoomed)
        fig1_z, ax1_sz = plt.subplots(figsize=(9, 6.5))
        if len(rms_vals) > 0:
            r_c = np.clip(rms_vals, 0, 10)
            ax1_sz.hist(r_c, bins=40, histtype="step", color="steelblue", linewidth=2.0)
            ax1_sz.axvline(max_rms_pct, color="crimson", linestyle="--", linewidth=2, label=rf"Cut ({max_rms_pct}%)")
            rms_box_text_z = (
                rf"Pass ($\leq {max_rms_pct}\%$): {n_pass_rms:,} ({pct_pass_rms:.1f}%)" + "\n" +
                rf"Fail (> {max_rms_pct}%): {n_fail_rms:,} ({pct_fail_rms:.1f}%)" + "\n" +
                rf"(Clipped at 10%)"
            )
            ax1_sz.text(0.96, 0.78, rms_box_text_z, transform=ax1_sz.transAxes, ha="right", va="top", fontsize=11.5,
                        bbox=dict(boxstyle="round,pad=0.38", facecolor="white", edgecolor="gray", alpha=0.92, zorder=4))
            ax1_sz.set_xlim(0, 10)
        ax1_sz.set_xlabel("Plateau RMS Non-Flatness [%]", fontsize=15)
        ax1_sz.set_ylabel("Runs", fontsize=15)
        ax1_sz.set_yscale("log")
        ax1_sz.set_ylim(0.5, 3500)
        ax1_sz.legend(loc="upper right", fontsize=12)
        ax1_sz.grid(True, linestyle="--", alpha=0.3, which="both")
        fig1_z.tight_layout()
        fig1_z.savefig(output_path.parent / f"{stem}_rms_zoomed.png", dpi=300, bbox_inches="tight")
        plt.close(fig1_z)

        # 2a. Single Slope (Unzoomed)
        fig2, ax2_s = plt.subplots(figsize=(9, 6.5))
        if len(slopes) > 0:
            x2_min = min(-5.0, min(slopes) * 1.2)
            x2_max = max(6.0, max(slopes) * 1.15)
            bins2 = np.linspace(x2_min, x2_max, 55)
            ax2_s.hist(slopes, bins=bins2, histtype="step", color="mediumseagreen", linewidth=2.0)
            ax2_s.axvline(max_slope_per_10pct, color="crimson", linestyle="--", linewidth=2)
            ax2_s.axvline(-max_slope_per_10pct, color="crimson", linestyle="--", linewidth=2, label=rf"Tol ($\pm${max_slope_per_10pct}%)")
            slope_box_text = (
                rf"In Tol: {n_pass_slope:,} ({pct_pass_slope:.1f}%)" + "\n" +
                rf"Fail (> $\pm${max_slope_per_10pct}%): {n_fail_slope:,} ({pct_fail_slope:.1f}%)" + "\n" +
                rf"Max Slope: {max(slopes):+.1f}%"
            )
            ax2_s.text(0.04, 0.94, slope_box_text, transform=ax2_s.transAxes, ha="left", va="top", fontsize=11.5,
                       bbox=dict(boxstyle="round,pad=0.38", facecolor="white", edgecolor="gray", alpha=0.92, zorder=4))
            ax2_s.set_xlim(x2_min, x2_max)
        ax2_s.set_xlabel("Plateau Slope [% per 10% Centrality]", fontsize=15)
        ax2_s.set_ylabel("Runs", fontsize=15)
        ax2_s.set_yscale("log")
        ax2_s.set_ylim(0.5, 3500)
        ax2_s.legend(loc="upper right", fontsize=12)
        ax2_s.grid(True, linestyle="--", alpha=0.3, which="both")
        fig2.tight_layout()
        fig2.savefig(output_path.parent / f"{stem}_slope.png", dpi=300, bbox_inches="tight")
        plt.close(fig2)

        # 2b. Single Slope (Zoomed)
        fig2_z, ax2_sz = plt.subplots(figsize=(9, 6.5))
        if len(slopes) > 0:
            s_c = np.clip(slopes, -6, 6)
            ax2_sz.hist(s_c, bins=40, histtype="step", color="mediumseagreen", linewidth=2.0)
            ax2_sz.axvline(max_slope_per_10pct, color="crimson", linestyle="--", linewidth=2)
            ax2_sz.axvline(-max_slope_per_10pct, color="crimson", linestyle="--", linewidth=2, label=rf"Tol ($\pm${max_slope_per_10pct}%)")
            slope_box_text_z = (
                rf"In Tol: {n_pass_slope:,} ({pct_pass_slope:.1f}%)" + "\n" +
                rf"Fail (> $\pm${max_slope_per_10pct}%): {n_fail_slope:,} ({pct_fail_slope:.1f}%)" + "\n" +
                rf"(Clipped at $\pm$6%)"
            )
            ax2_sz.text(0.04, 0.94, slope_box_text_z, transform=ax2_sz.transAxes, ha="left", va="top", fontsize=11.5,
                        bbox=dict(boxstyle="round,pad=0.38", facecolor="white", edgecolor="gray", alpha=0.92, zorder=4))
            ax2_sz.set_xlim(-6, 6)
        ax2_sz.set_xlabel("Plateau Slope [% per 10% Centrality]", fontsize=15)
        ax2_sz.set_ylabel("Runs", fontsize=15)
        ax2_sz.set_yscale("log")
        ax2_sz.set_ylim(0.5, 3500)
        ax2_sz.legend(loc="upper right", fontsize=12)
        ax2_sz.grid(True, linestyle="--", alpha=0.3, which="both")
        fig2_z.tight_layout()
        fig2_z.savefig(output_path.parent / f"{stem}_slope_zoomed.png", dpi=300, bbox_inches="tight")
        plt.close(fig2_z)

        # 3a. Single Ratios (Unzoomed)
        fig3, ax3_s = plt.subplots(figsize=(9, 6.5))
        if len(r1s) > 0 and len(r15s) > 0:
            x3_max = max(2.5, max(r1s) * 1.08)
            bins3 = np.linspace(0, x3_max, 55)
            ax3_s.hist(r1s, bins=bins3, histtype="step", color="crimson", linewidth=2.0, label=r"$R_{1\%}$ (1% Centrality)")
            ax3_s.hist(r15s, bins=bins3, histtype="step", color="royalblue", linewidth=2.0, label=r"$R_{1-5\%}$ (1–5% Centrality)")
            ax3_s.axvline(1.0, color="gray", linestyle="-", linewidth=1.5)
            ax3_s.axvline(max_central_spike, color="crimson", linestyle="--", linewidth=2, label=rf"Spike Cut ({max_central_spike})")
            ax3_s.axvline(min_central_ratio, color="darkorange", linestyle="--", linewidth=2, label=rf"Drop Cut ({min_central_ratio})")
            ratio_box_text = (
                rf"Spike Fail (> {max_central_spike}): {n_spike:,} ({pct_spike:.1f}%)" + "\n" +
                rf"Drop Fail (< {min_central_ratio}): {n_drop:,} ({pct_drop:.1f}%)" + "\n" +
                rf"Max $R_{{1\%}}$: {max(r1s):.2f}"
            )
            ax3_s.text(0.96, 0.94, ratio_box_text, transform=ax3_s.transAxes, ha="right", va="top", fontsize=11.0,
                       bbox=dict(boxstyle="round,pad=0.38", facecolor="white", edgecolor="gray", alpha=0.92, zorder=4))
            ax3_s.set_xlim(0, x3_max)
        ax3_s.set_xlabel("Central Ratio to Plateau", fontsize=15)
        ax3_s.set_ylabel("Runs", fontsize=15)
        ax3_s.set_yscale("log")
        ax3_s.set_ylim(0.5, 4500)
        ax3_s.legend(loc="upper left", fontsize=10.5, frameon=True, framealpha=0.9)
        ax3_s.grid(True, linestyle="--", alpha=0.3, which="both")
        fig3.tight_layout()
        fig3.savefig(output_path.parent / f"{stem}_ratios.png", dpi=300, bbox_inches="tight")
        plt.close(fig3)

        # 3b. Single Ratios (Zoomed)
        fig3_z, ax3_sz = plt.subplots(figsize=(9, 6.5))
        if len(r1s) > 0 and len(r15s) > 0:
            r1_c = np.clip(r1s, 0.4, 2.0)
            r15_c = np.clip(r15s, 0.4, 2.0)
            bins3_z = np.linspace(0.4, 2.0, 35)
            ax3_sz.hist(r1_c, bins=bins3_z, histtype="step", color="crimson", linewidth=2.0, label=r"$R_{1\%}$ (1% Centrality)")
            ax3_sz.hist(r15_c, bins=bins3_z, histtype="step", color="royalblue", linewidth=2.0, label=r"$R_{1-5\%}$ (1–5% Centrality)")
            ax3_sz.axvline(1.0, color="gray", linestyle="-", linewidth=1.5)
            ax3_sz.axvline(max_central_spike, color="crimson", linestyle="--", linewidth=2, label=rf"Spike Cut ({max_central_spike})")
            ax3_sz.axvline(min_central_ratio, color="darkorange", linestyle="--", linewidth=2, label=rf"Drop Cut ({min_central_ratio})")
            ratio_box_text_z = (
                rf"Spike Fail (> {max_central_spike}): {n_spike:,} ({pct_spike:.1f}%)" + "\n" +
                rf"Drop Fail (< {min_central_ratio}): {n_drop:,} ({pct_drop:.1f}%)" + "\n" +
                rf"(Clipped at [0.4, 2.0])"
            )
            ax3_sz.text(0.96, 0.94, ratio_box_text_z, transform=ax3_sz.transAxes, ha="right", va="top", fontsize=11.0,
                        bbox=dict(boxstyle="round,pad=0.38", facecolor="white", edgecolor="gray", alpha=0.92, zorder=4))
            ax3_sz.set_xlim(0.4, 2.0)
        ax3_sz.set_xlabel("Central Ratio to Plateau", fontsize=15)
        ax3_sz.set_ylabel("Runs", fontsize=15)
        ax3_sz.set_yscale("log")
        ax3_sz.set_ylim(0.5, 4500)
        ax3_sz.legend(loc="upper left", fontsize=10.5, frameon=True, framealpha=0.9)
        ax3_sz.grid(True, linestyle="--", alpha=0.3, which="both")
        fig3_z.tight_layout()
        fig3_z.savefig(output_path.parent / f"{stem}_ratios_zoomed.png", dpi=300, bbox_inches="tight")
        plt.close(fig3_z)

        # 4. Single Events (informational population statistics)
        fig4, ax4_s = plt.subplots(figsize=(9, 6.5))
        if len(nevts) > 0 and log_bins is not None:
            ax4_s.hist(nevts, bins=log_bins, histtype="step", color="coral", linewidth=2.0)
            ax4_s.text(0.04, 0.94, evts_box_text, transform=ax4_s.transAxes, ha="left", va="top", fontsize=11.5,
                       bbox=dict(boxstyle="round,pad=0.38", facecolor="white", edgecolor="gray", alpha=0.92, zorder=4))
            ax4_s.set_xscale("log")
        ax4_s.set_xlabel("Total Events per Run", fontsize=15)
        ax4_s.set_ylabel("Runs", fontsize=15)
        ax4_s.set_yscale("log")
        ax4_s.set_ylim(0.5, 3500)
        ax4_s.grid(True, linestyle="--", alpha=0.3, which="both")
        fig4.tight_layout()
        fig4.savefig(output_path.parent / f"{stem}_events.png", dpi=300, bbox_inches="tight")
        plt.close(fig4)



def plot_ensemble_profile(data_list, output_path, test_runs=None, subtitle=None):
    """
    Plot 4: Ensemble Median & Percentile Envelope Profile.
    Excludes empty 0% centrality Bin 0 [-0.5, 0.5] (empty across all runs) and displays bins 1-99 (1% to 99% centrality).
    """
    if hasattr(data_list, "to_dict"):
        data_list = data_list.to_dict("records")

    valid_data = [d for d in data_list if d.get("ratios") is not None and d.get("edges") is not None]
    if not valid_data:
        print(f"Warning: Cannot plot ensemble profile to {output_path}: no valid data found.")
        return

    hep.style.use("ATLAS")
    fig, ax = plt.subplots(figsize=(9.5, 7))

    # Exclude empty 0% centrality Bin 0 [-0.5, 0.5] (empty across all runs)
    ratio_matrix = np.array([d["ratios"][1:] for d in valid_data])
    cent_edges = valid_data[0]["edges"][1:]
    bin_centers = 0.5 * (cent_edges[:-1] + cent_edges[1:])

    # Compute percentiles across runs for each centrality bin
    p50 = np.median(ratio_matrix, axis=0)
    p16 = np.percentile(ratio_matrix, 16, axis=0)
    p84 = np.percentile(ratio_matrix, 84, axis=0)
    p02 = np.percentile(ratio_matrix, 2.5, axis=0)
    p97 = np.percentile(ratio_matrix, 97.5, axis=0)

    # Distinct high-contrast colors for population envelopes (HEP Brazilian convention)
    # 2-sigma (95% CL): Warm Gold
    ax.fill_between(
        bin_centers, p02, p97,
        color="#f9c74f", alpha=0.55,
        edgecolor="#d4ac0d", linewidth=0.8,
        label=r"95% ($2\sigma$) Population Envelope",
    )
    # 1-sigma (68% CL): Vibrant Green
    ax.fill_between(
        bin_centers, p16, p84,
        color="#2ca02c", alpha=0.65,
        edgecolor="#1e8449", linewidth=0.8,
        label=r"68% ($1\sigma$) Population Envelope",
    )
    # Ensemble Median: Solid Black
    ax.plot(bin_centers, p50, color="black", linewidth=2.5, label="Ensemble Median")

    # Ideal flat plateau baseline (1.0)
    ax.axhline(1.0, color="dimgray", linestyle="--", linewidth=1.5, alpha=0.85, label="Flat Baseline (1.0)")

    ax.set_xlim(0, 100)
    ax.set_ylim(0.0, 1.6)
    ax.set_xlabel("Centrality [%]", fontsize=15)
    ax.set_ylabel("Ratio to Plateau Average", fontsize=15)
    ax.grid(True, linestyle="--", alpha=0.35, which="both")

    # Reorder legend handles: Median, 1-sigma, 2-sigma, Baseline
    handles, labels = ax.get_legend_handles_labels()
    desired_order = [
        "Ensemble Median",
        r"68% ($1\sigma$) Population Envelope",
        r"95% ($2\sigma$) Population Envelope",
        "Flat Baseline (1.0)",
    ]
    handle_dict = dict(zip(labels, handles))
    ordered_handles = [handle_dict[lbl] for lbl in desired_order if lbl in handle_dict]
    ordered_labels = [lbl for lbl in desired_order if lbl in handle_dict]
    ax.legend(ordered_handles, ordered_labels, loc="upper right", fontsize=12, frameon=True, framealpha=0.92)

    if subtitle:
        ax.text(
            0.04, 0.94, subtitle,
            transform=ax.transAxes,
            fontsize=13,
            fontweight="bold",
            va="top",
            ha="left",
            bbox=dict(boxstyle="round,pad=0.45", facecolor="white", edgecolor="#aaaaaa", alpha=0.92),
        )

    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_summary_reports(data_list, output_dir):
    """
    Write CSV summary and plain text lists of good and flagged outlier runs.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "centrality_qa_summary.csv"
    flagged_list_path = output_dir / "flagged_outlier_runs.list"
    good_list_path = output_dir / "good_runs.list"

    fieldnames = [
        "run_number",
        "total_events",
        "plateau_avg",
        "rms_plat_pct",
        "max_dev_pct",
        "slope_per_10pct",
        "ratio_1",
        "ratio_1_5",
        "ratio_80",
        "chi2_ndf",
        "status",
        "file_path",
    ]

    with open(csv_path, "w", newline="") as f_csv, \
         open(flagged_list_path, "w") as f_bad, \
         open(good_list_path, "w") as f_good:

        writer = csv.DictWriter(f_csv, fieldnames=fieldnames)
        writer.writeheader()

        for d in data_list:
            row = {k: d[k] for k in fieldnames}
            # Format floats for clean CSV presentation
            for k in ["plateau_avg", "rms_plat_pct", "max_dev_pct", "slope_per_10pct", "ratio_1", "ratio_1_5", "ratio_80", "chi2_ndf"]:
                if isinstance(row[k], float) and not np.isnan(row[k]):
                    row[k] = f"{row[k]:.4f}"
            writer.writerow(row)

            if d["status"] == "GOOD":
                f_good.write(f"{d['run_number']}\n")
            else:
                f_bad.write(f"{d['run_number']}\t{d['status']}\n")

    print(f"Summary CSV written to: {csv_path}")
    print(f"Flagged runs list written to: {flagged_list_path}")
    print(f"Good runs list written to: {good_list_path}")


def plot_centrality_1d_diagnostic(
    metric,
    title_suffix,
    output_path,
    cent_flat_min=10.0,
    cent_flat_max=70.0,
    is_good=None,
    rank=None,
):
    """
    Generate a 2-panel 1D Centrality diagnostic plot (Events + Ratio to Plateau)
    for a representative run (failure mode example, top flat example, or user-specified run).
    """
    hep.style.use("ATLAS")

    run_number = metric.get("run_number", metric.get("run"))
    values = metric.get("values")
    edges = metric.get("edges")

    if (values is None or edges is None) and metric.get("file_path"):
        try:
            with uproot.open(metric["file_path"]) as f:
                hist_name = metric.get("hist_name", "hCentrality")
                if hist_name in f:
                    values, edges = f[hist_name].to_numpy()
                else:
                    for k in f.keys():
                        if "Centrality" in k:
                            values, edges = f[k].to_numpy()
                            break
        except Exception as e:
            print(f"Warning: Could not read ROOT file {metric.get('file_path')}: {e}")

    if values is None or edges is None:
        print(f"Warning: Cannot plot 1D diagnostic for run {run_number}: histogram data not found.")
        return

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(9.5, 8.5), sharex=True, gridspec_kw={"height_ratios": [2.5, 1.8]}
    )

    bin_centers = metric.get("bin_centers")
    if bin_centers is None:
        bin_centers = 0.5 * (edges[:-1] + edges[1:])

    plateau_avg = metric.get("plateau_avg")
    if plateau_avg is None:
        plateau_avg = compute_centrality_average(values, edges, cent_min=cent_flat_min, cent_max=cent_flat_max)

    ratios = metric.get("ratios")
    if ratios is None:
        ratios = np.where(values > 0, values / plateau_avg, 0.0) if plateau_avg > 0 else np.zeros(len(values))

    total_events = metric.get("total_events", float(np.sum(values)))

    # 1. Top Panel: Events vs Centrality [%]
    hep.histplot((values, edges), ax=ax1, histtype="step", color="black", linewidth=2.0, label="Centrality Distribution")
    if plateau_avg > 0:
        ax1.axhline(plateau_avg, color="crimson", linestyle="--", linewidth=1.8,
                    label=rf"Plateau Avg ({plateau_avg:.2e})")
        ax1.axvspan(cent_flat_min, cent_flat_max, color="forestgreen", alpha=0.08,
                    label=f"Plateau [{cent_flat_min:.0f}%, {cent_flat_max:.0f}%]")

    ax1.set_ylabel("Events", fontsize=16)
    ax1.set_xlim(0, 100)
    ax1.tick_params(axis="both", labelsize=13)

    plat_mask = (bin_centers >= cent_flat_min) & (bin_centers <= cent_flat_max)
    plat_max = float(np.max(values[plat_mask])) if np.any(plat_mask) else plateau_avg
    overall_max = float(np.max(values)) if len(values) > 0 else 0.0
    if overall_max > 0:
        # Tightly tuned y-max: enough room for info box, minimal dead space
        y1_max = max(overall_max * 1.15, plat_max * 1.40 if plat_max > 0 else 1.0)
        ax1.set_ylim(0, y1_max)
    else:
        ax1.set_ylim(0, 1.0)
    ax1.legend(loc="upper right", fontsize=11.5, frameon=True, framealpha=0.92)
    ax1.grid(True, linestyle="--", alpha=0.3)

    # Info banner in top panel - increased font size from 10 to 12.0
    if is_good is None:
        is_good = (metric.get("status") == "GOOD")

    if is_good:
        if rank is not None:
            rank_tag = f"Top Flat Example #{rank}"
        elif title_suffix:
            rank_tag = title_suffix
        else:
            rank_tag = "GOOD Run"
        info_text = (
            f"Run {run_number} | {rank_tag}\n"
            f"Total Events: {total_events:.2e}\n"
            f"Status: GOOD (Passed All QA)"
        )
        box_edge = "forestgreen"
    else:
        flags_list = [f for f in str(metric.get("status", "")).split(";") if f]
        if len(flags_list) > 2:
            flags_text = "Flags: " + "; ".join(flags_list[:2]) + ";\n       " + "; ".join(flags_list[2:])
        elif len(str(metric.get("status", ""))) > 32 and len(flags_list) > 1:
            flags_text = "Flags: " + flags_list[0] + ";\n       " + "; ".join(flags_list[1:])
        elif flags_list:
            flags_text = f"Flags: {metric.get('status')}"
        else:
            flags_text = "Flags: Flagged Outlier"

        tag = f"Example: {title_suffix}" if title_suffix else "Outlier Run"
        info_text = (
            f"Run {run_number} | {tag}\n"
            f"Total Events: {total_events:.2e}\n"
            f"{flags_text}"
        )
        box_edge = "crimson"

    ax1.text(0.12, 0.94, info_text, transform=ax1.transAxes, va="top", ha="left",
             fontsize=12.0, fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.45", facecolor="white", edgecolor=box_edge, linewidth=1.6, alpha=0.92))

    # 2. Bottom Panel: Ratio to Plateau Average (exclude empty bin 0)
    cent_edges = edges[1:]
    r_slice = ratios[1:]
    hep.histplot((r_slice, cent_edges), ax=ax2, histtype="step", color="royalblue", linewidth=2.0, label="Ratio to Plateau")
    ax2.axhline(1.0, color="gray", linestyle="-", linewidth=1.2)
    ax2.axvspan(cent_flat_min, cent_flat_max, color="forestgreen", alpha=0.08)

    pos_r = r_slice[r_slice > 0]
    if len(pos_r) > 0:
        r_min = float(np.min(pos_r))
        overall_r_max = float(np.max(pos_r))
        # Zoom in y-axis range to just include non-zero points with slight margin below r_min
        pad_bottom = max(0.04, 0.05 * (overall_r_max - r_min))
        y2_min = max(0.0, r_min - pad_bottom)
        if overall_r_max <= 1.15:
            if y2_min >= 0.40:
                # Flat run (like top flat examples, max ratio close to 1): tight zoom around 1.0
                y2_max = 1.25
            else:
                # Central drop run with y2_min near 0: headroom for text box above 1.0
                y2_max = 1.46
        elif overall_r_max <= 1.30:
            y2_max = max(1.35, overall_r_max * 1.18)
        else:
            # Spike or large excursion: scale with headroom
            y2_max = max(1.48, min(3.5, overall_r_max * 1.18))
        ax2.set_ylim(y2_min, y2_max)
    else:
        y2_min, y2_max = 0.0, 1.25
        ax2.set_ylim(y2_min, y2_max)

    if y2_max >= 1.28:
        ax2.axhline(1.30, color="crimson", linestyle=":", linewidth=1.3, label=r"Spike Cut ($R_{1\%} > 1.30$)")
    if y2_min <= 0.82:
        ax2.axhline(0.80, color="darkorange", linestyle=":", linewidth=1.3, label=r"Drop Cut ($R_{1-5\%} < 0.80$)")

    # Ratio metrics annotation - increased font size from 9.5 to 11.5
    rms_str = f"{metric.get('rms_plat_pct', np.nan):.2f}%" if not np.isnan(metric.get('rms_plat_pct', np.nan)) else "N/A"
    slope_str = f"{metric.get('slope_per_10pct', np.nan):+.2f}%" if not np.isnan(metric.get('slope_per_10pct', np.nan)) else "N/A"
    r1_str = f"{metric.get('ratio_1', np.nan):.3f}" if not np.isnan(metric.get('ratio_1', np.nan)) else "N/A"
    r15_str = f"{metric.get('ratio_1_5', np.nan):.3f}" if not np.isnan(metric.get('ratio_1_5', np.nan)) else "N/A"

    ratio_metrics_text = (
        rf"Plateau RMS: {rms_str} (Cut: 4.5%)" + "\n" +
        rf"Plateau Slope: {slope_str}/10% cent (Tol: $\pm 2.5\%$)" + "\n" +
        rf"$R_{{1\%}}$: {r1_str} | $R_{{1-5\%}}$: {r15_str}"
    )
    ax2.text(0.12, 0.94, ratio_metrics_text, transform=ax2.transAxes, va="top", ha="left",
             fontsize=10.5,
             bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor=box_edge if is_good else "gray", linewidth=1.4 if is_good else 1.0, alpha=0.92))

    ax2.set_xlabel("Centrality [%]", fontsize=16)
    ax2.set_ylabel("Ratio to Plateau", fontsize=16)
    ax2.set_xlim(0, 100)
    ax2.tick_params(axis="both", labelsize=13)

    ax2.legend(loc="upper right", fontsize=10.5, frameon=True, framealpha=0.92)
    ax2.grid(True, linestyle="--", alpha=0.3)

    # Reclaim top whitespace with no suptitle
    fig.subplots_adjust(top=0.96, bottom=0.10, left=0.12, right=0.96, hspace=0.08)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_failure_example(metric, failure_mode, output_path, cent_flat_min=10.0, cent_flat_max=70.0):
    """
    Generate a 2-panel 1D Centrality diagnostic plot (Events + Ratio to Plateau)
    for a representative run demonstrating a specific failure mode.
    """
    return plot_centrality_1d_diagnostic(
        metric=metric,
        title_suffix=failure_mode,
        output_path=output_path,
        cent_flat_min=cent_flat_min,
        cent_flat_max=cent_flat_max,
        is_good=False,
    )


def plot_top_flat_example(metric, rank, output_path, cent_flat_min=10.0, cent_flat_max=70.0):
    """
    Generate a 2-panel 1D Centrality diagnostic plot (Events + Ratio to Plateau)
    for a representative run demonstrating one of the best / flattest runs.
    """
    return plot_centrality_1d_diagnostic(
        metric=metric,
        title_suffix=f"Top Flat Example #{rank}",
        output_path=output_path,
        cent_flat_min=cent_flat_min,
        cent_flat_max=cent_flat_max,
        is_good=True,
        rank=rank,
    )


def plot_user_example(metric, output_path, cent_flat_min=10.0, cent_flat_max=70.0, title_suffix="User Specified"):
    """
    Generate a 2-panel 1D Centrality diagnostic plot (Events + Ratio to Plateau)
    for a user-specified run.
    """
    is_good = (metric.get("status") == "GOOD")
    return plot_centrality_1d_diagnostic(
        metric=metric,
        title_suffix=title_suffix,
        output_path=output_path,
        cent_flat_min=cent_flat_min,
        cent_flat_max=cent_flat_max,
        is_good=is_good,
    )




def plot_failure_mode_metric_distribution(metrics_list, failure_mode, example_runs, output_path,
                                          max_rms_pct=4.5, max_dev_pct=8.0,
                                          max_slope_per_10pct=2.5, max_central_spike=1.30,
                                          min_central_ratio=0.80):
    """
    Generate a version of the relevant centrality_metric_distributions plot for a specific
    failure mode, showing the overall run population distribution and vertical lines marking
    where the representative example runs fall to demonstrate how and by how much they failed.
    """
    if hasattr(metrics_list, "to_dict"):
        metrics_list = metrics_list.to_dict("records")
    if not example_runs:
        return

    run_dict = {d.get("run_number", d.get("run")): d for d in metrics_list}
    example_dicts = [run_dict[r] for r in example_runs if r in run_dict]
    if not example_dicts:
        return

    fig, ax = plt.subplots(figsize=(9, 6.5))

    # Distinct styles for the example run lines
    line_styles = [
        {"color": "#800080", "linestyle": "-.", "linewidth": 2.0},  # Purple
        {"color": "#d95f02", "linestyle": ":",  "linewidth": 2.4},  # Dark Orange
        {"color": "#1b9e77", "linestyle": "--", "linewidth": 2.0},  # Teal
        {"color": "#e7298a", "linestyle": "-",  "linewidth": 2.0},  # Pink fallback
    ]

    output_path = Path(output_path)
    if output_path.is_dir() or output_path.suffix == "":
        output_path = output_path / "centrality_metric_distributions.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    zoomed_path = output_path.parent / "centrality_metric_distributions_zoomed.png"

    fig_z = None  # Zoomed figure if needed

    if failure_mode == "FLAG_NON_FLAT":
        rms_vals = [float(d["rms_plat_pct"]) for d in metrics_list if not np.isnan(d.get("rms_plat_pct", np.nan))]
        run_vals = [float(d.get("rms_plat_pct", 0.0)) for d in example_dicts]

        # 1. Unzoomed (Full population outlier range)
        x_min = 0.0
        x_max = max(12.0, max(rms_vals) * 1.08 if rms_vals else 12.0)
        bins = np.linspace(x_min, x_max, 55)

        ax.hist(rms_vals, bins=bins, histtype="step", color="steelblue", linewidth=2.0, label="All Runs Distribution")
        ax.axvline(max_rms_pct, color="crimson", linestyle="--", linewidth=2.2, label=f"Flatness Cut ({max_rms_pct}%)")

        for idx, (d, val) in enumerate(zip(example_dicts, run_vals)):
            st = line_styles[idx % len(line_styles)]
            ax.axvline(val, color=st["color"], linestyle=st["linestyle"], linewidth=st["linewidth"],
                       label=f"Example Run {d.get('run_number', d.get('run'))} (RMS = {val:.2f}%)")

        ax.set_xlabel("Plateau RMS Non-Flatness [%]", fontsize=15)
        ax.set_ylabel("Runs", fontsize=15)
        ax.set_yscale("log")
        ax.set_ylim(0.5, 3500)
        ax.set_xlim(x_min, x_max)
        ax.legend(loc="upper right", fontsize=11, frameon=True, framealpha=0.92)
        ax.grid(True, linestyle="--", alpha=0.3, which="both")

        info_text = (
            r"$\bf{FLAG\_NON\_FLAT\ (Unzoomed)}$" + "\n" +
            f"Cut: RMS > {max_rms_pct}% (or MaxDev > {max_dev_pct}%)\n" +
            f"Population Max: {max(rms_vals):.1f}%\n" +
            f"Selected Runs: {', '.join(str(r) for r in example_runs)}"
        )
        ax.text(0.04, 0.94, info_text, transform=ax.transAxes, ha="left", va="top", fontsize=11.0,
                bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="gray", alpha=0.92, zorder=4))

        # 2. Zoomed (Detailed view around threshold [0, 10%])
        fig_z, ax_z = plt.subplots(figsize=(9, 6.5))
        z_min, z_max = 0.0, 10.0
        r_clipped = np.clip(rms_vals, z_min, z_max)
        ax_z.hist(r_clipped, bins=45, histtype="step", color="steelblue", linewidth=2.0, label="All Runs (Clipped at 10%)")
        ax_z.axvline(max_rms_pct, color="crimson", linestyle="--", linewidth=2.2, label=f"Flatness Cut ({max_rms_pct}%)")

        for idx, (d, val) in enumerate(zip(example_dicts, run_vals)):
            st = line_styles[idx % len(line_styles)]
            ax_z.axvline(val, color=st["color"], linestyle=st["linestyle"], linewidth=st["linewidth"],
                         label=f"Example Run {d.get('run_number', d.get('run'))} (RMS = {val:.2f}%)")

        ax_z.set_xlabel("Plateau RMS Non-Flatness [%]", fontsize=15)
        ax_z.set_ylabel("Runs", fontsize=15)
        ax_z.set_yscale("log")
        ax_z.set_ylim(0.5, 3500)
        ax_z.set_xlim(z_min, z_max)
        ax_z.legend(loc="upper right", fontsize=11, frameon=True, framealpha=0.92)
        ax_z.grid(True, linestyle="--", alpha=0.3, which="both")

        info_text_z = (
            r"$\bf{FLAG\_NON\_FLAT\ (Zoomed)}$" + "\n" +
            f"Cut: RMS > {max_rms_pct}%\n" +
            f"Selected Runs: {', '.join(str(r) for r in example_runs)}"
        )
        ax_z.text(0.04, 0.94, info_text_z, transform=ax_z.transAxes, ha="left", va="top", fontsize=11.0,
                  bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="gray", alpha=0.92, zorder=4))

    elif failure_mode == "FLAG_SLOPE_DRIFT":
        slopes = [float(d["slope_per_10pct"]) for d in metrics_list if not np.isnan(d.get("slope_per_10pct", np.nan))]
        run_vals = [float(d.get("slope_per_10pct", 0.0)) for d in example_dicts]

        # 1. Unzoomed (Full population outlier range)
        x_min = min(-5.0, min(slopes) * 1.2 if slopes else -5.0)
        x_max = max(6.0, max(slopes) * 1.15 if slopes else 6.0)
        bins = np.linspace(x_min, x_max, 60)

        ax.hist(slopes, bins=bins, histtype="step", color="mediumseagreen", linewidth=2.0, label="All Runs Distribution")
        ax.axvline(max_slope_per_10pct, color="crimson", linestyle="--", linewidth=2.2, label=rf"Tolerance Cut ($\pm${max_slope_per_10pct}%)")
        ax.axvline(-max_slope_per_10pct, color="crimson", linestyle="--", linewidth=2.2)

        for idx, (d, val) in enumerate(zip(example_dicts, run_vals)):
            st = line_styles[idx % len(line_styles)]
            ax.axvline(val, color=st["color"], linestyle=st["linestyle"], linewidth=st["linewidth"],
                       label=f"Example Run {d.get('run_number', d.get('run'))} (Slope = {val:+.2f}%)")

        ax.set_xlabel("Plateau Slope [% per 10% Centrality]", fontsize=15)
        ax.set_ylabel("Runs", fontsize=15)
        ax.set_yscale("log")
        ax.set_ylim(0.5, 3500)
        ax.set_xlim(x_min, x_max)
        ax.legend(loc="upper right", fontsize=11, frameon=True, framealpha=0.92)
        ax.grid(True, linestyle="--", alpha=0.3, which="both")

        info_text = (
            r"$\bf{FLAG\_SLOPE\_DRIFT\ (Unzoomed)}$" + "\n" +
            f"Tolerance: |Slope| > {max_slope_per_10pct}%\n" +
            f"Population Max: {max(slopes):+.1f}%\n" +
            f"Selected Runs: {', '.join(str(r) for r in example_runs)}"
        )
        ax.text(0.04, 0.94, info_text, transform=ax.transAxes, ha="left", va="top", fontsize=11.0,
                bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="gray", alpha=0.92, zorder=4))

        # 2. Zoomed (Detailed view around tolerance cut [-6%, +6%])
        fig_z, ax_z = plt.subplots(figsize=(9, 6.5))
        z_min, z_max = -6.0, 6.0
        s_clipped = np.clip(slopes, z_min, z_max)
        ax_z.hist(s_clipped, bins=40, histtype="step", color="mediumseagreen", linewidth=2.0, label=r"All Runs (Bulk in $\pm$6%)")
        ax_z.axvline(max_slope_per_10pct, color="crimson", linestyle="--", linewidth=2.2, label=rf"Tolerance Cut ($\pm${max_slope_per_10pct}%)")
        ax_z.axvline(-max_slope_per_10pct, color="crimson", linestyle="--", linewidth=2.2)

        ax_z.set_xlabel("Plateau Slope [% per 10% Centrality]", fontsize=15)
        ax_z.set_ylabel("Runs", fontsize=15)
        ax_z.set_yscale("log")
        ax_z.set_ylim(0.5, 3500)
        ax_z.set_xlim(z_min, z_max)
        ax_z.legend(loc="upper right", fontsize=11, frameon=True, framealpha=0.92)
        ax_z.grid(True, linestyle="--", alpha=0.3, which="both")

        info_text_z = (
            r"$\bf{FLAG\_SLOPE\_DRIFT\ (Zoomed\ Core)}$" + "\n" +
            f"Tolerance: |Slope| > {max_slope_per_10pct}%\n" +
            f"Example runs lie at +24% to +30%\n(visible on unzoomed plot)"
        )
        ax_z.text(0.04, 0.94, info_text_z, transform=ax_z.transAxes, ha="left", va="top", fontsize=11.0,
                  bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="gray", alpha=0.92, zorder=4))

    elif failure_mode == "FLAG_CENTRAL_DROP":
        r15s = [float(d["ratio_1_5"]) for d in metrics_list if not np.isnan(d.get("ratio_1_5", np.nan))]
        r1s = [float(d["ratio_1"]) for d in metrics_list if not np.isnan(d.get("ratio_1", np.nan))]
        run_vals = [float(d.get("ratio_1_5", 1.0)) for d in example_dicts]

        # 1. Unzoomed (Full population outlier range)
        x_min = 0.0
        x_max = max(2.5, max(r1s) * 1.08 if r1s else 2.5)
        bins = np.linspace(x_min, x_max, 55)

        ax.hist(r15s, bins=bins, histtype="step", color="royalblue", linewidth=2.0, label=r"$R_{1-5\%}$ Distribution")
        ax.hist(r1s, bins=bins, histtype="step", color="crimson", linewidth=1.4, alpha=0.5, label=r"$R_{1\%}$ Distribution")
        ax.axvline(1.0, color="gray", linestyle="-", linewidth=1.2)
        ax.axvline(min_central_ratio, color="darkorange", linestyle="--", linewidth=2.2, label=rf"Drop Cut ($R_{{1-5\%}} < {min_central_ratio}$)")

        for idx, (d, val) in enumerate(zip(example_dicts, run_vals)):
            st = line_styles[idx % len(line_styles)]
            ax.axvline(val, color=st["color"], linestyle=st["linestyle"], linewidth=st["linewidth"],
                       label=rf"Example Run {d.get('run_number', d.get('run'))} ($R_{{1-5\%}} = {val:.3f}$)")

        ax.set_xlabel("Central Ratio to Plateau", fontsize=15)
        ax.set_ylabel("Runs", fontsize=15)
        ax.set_yscale("log")
        ax.set_ylim(0.5, 4500)
        ax.set_xlim(x_min, x_max)
        ax.legend(loc="upper right", fontsize=10.5, frameon=True, framealpha=0.92)
        ax.grid(True, linestyle="--", alpha=0.3, which="both")

        info_text = (
            r"$\bf{FLAG\_CENTRAL\_DROP\ (Unzoomed)}$" + "\n" +
            rf"Drop Cut: $R_{{1-5\%}} < {min_central_ratio}$" + "\n" +
            f"Selected Runs: {', '.join(str(r) for r in example_runs)}"
        )
        ax.text(0.04, 0.94, info_text, transform=ax.transAxes, ha="left", va="top", fontsize=11.0,
                bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="gray", alpha=0.92, zorder=4))

        # 2. Zoomed (Detailed view around threshold [0, 2.2])
        fig_z, ax_z = plt.subplots(figsize=(9, 6.5))
        z_min, z_max = 0.0, 2.2
        r15_c = np.clip(r15s, z_min, z_max)
        r1_c = np.clip(r1s, z_min, z_max)
        bins_z = np.linspace(z_min, z_max, 45)

        ax_z.hist(r15_c, bins=bins_z, histtype="step", color="royalblue", linewidth=2.0, label=r"$R_{1-5\%}$ Distribution")
        ax_z.hist(r1_c, bins=bins_z, histtype="step", color="crimson", linewidth=1.4, alpha=0.5, label=r"$R_{1\%}$ Distribution")
        ax_z.axvline(1.0, color="gray", linestyle="-", linewidth=1.2)
        ax_z.axvline(min_central_ratio, color="darkorange", linestyle="--", linewidth=2.2, label=rf"Drop Cut ($R_{{1-5\%}} < {min_central_ratio}$)")

        for idx, (d, val) in enumerate(zip(example_dicts, run_vals)):
            st = line_styles[idx % len(line_styles)]
            ax_z.axvline(val, color=st["color"], linestyle=st["linestyle"], linewidth=st["linewidth"],
                         label=rf"Example Run {d.get('run_number', d.get('run'))} ($R_{{1-5\%}} = {val:.3f}$)")

        ax_z.set_xlabel("Central Ratio to Plateau", fontsize=15)
        ax_z.set_ylabel("Runs", fontsize=15)
        ax_z.set_yscale("log")
        ax_z.set_ylim(0.5, 4500)
        ax_z.set_xlim(z_min, z_max)
        ax_z.legend(loc="upper right", fontsize=10.5, frameon=True, framealpha=0.92)
        ax_z.grid(True, linestyle="--", alpha=0.3, which="both")

        info_text_z = (
            r"$\bf{FLAG\_CENTRAL\_DROP\ (Zoomed)}$" + "\n" +
            rf"Drop Cut: $R_{{1-5\%}} < {min_central_ratio}$" + "\n" +
            f"Selected Runs: {', '.join(str(r) for r in example_runs)}"
        )
        ax_z.text(0.04, 0.94, info_text_z, transform=ax_z.transAxes, ha="left", va="top", fontsize=11.0,
                  bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="gray", alpha=0.92, zorder=4))

    elif failure_mode == "FLAG_CENTRAL_SPIKE":
        r1s = [float(d["ratio_1"]) for d in metrics_list if not np.isnan(d.get("ratio_1", np.nan))]
        r15s = [float(d["ratio_1_5"]) for d in metrics_list if not np.isnan(d.get("ratio_1_5", np.nan))]
        run_vals = [float(d.get("ratio_1", 1.0)) for d in example_dicts]

        # 1. Unzoomed (Full population outlier range)
        x_min = 0.0
        x_max = max(2.5, max(r1s) * 1.08 if r1s else 2.5)
        bins = np.linspace(x_min, x_max, 55)

        ax.hist(r1s, bins=bins, histtype="step", color="crimson", linewidth=2.0, label=r"$R_{1\%}$ Distribution")
        ax.hist(r15s, bins=bins, histtype="step", color="royalblue", linewidth=1.4, alpha=0.5, label=r"$R_{1-5\%}$ Distribution")
        ax.axvline(1.0, color="gray", linestyle="-", linewidth=1.2)
        ax.axvline(max_central_spike, color="crimson", linestyle="--", linewidth=2.2, label=rf"Spike Cut ($R_{{1\%}} > {max_central_spike}$)")

        for idx, (d, val) in enumerate(zip(example_dicts, run_vals)):
            st = line_styles[idx % len(line_styles)]
            ax.axvline(val, color=st["color"], linestyle=st["linestyle"], linewidth=st["linewidth"],
                       label=rf"Example Run {d.get('run_number', d.get('run'))} ($R_{{1\%}} = {val:.3f}$)")

        ax.set_xlabel("Central Ratio to Plateau", fontsize=15)
        ax.set_ylabel("Runs", fontsize=15)
        ax.set_yscale("log")
        ax.set_ylim(0.5, 4500)
        ax.set_xlim(x_min, x_max)
        ax.legend(loc="upper right", fontsize=10.5, frameon=True, framealpha=0.92)
        ax.grid(True, linestyle="--", alpha=0.3, which="both")

        info_text = (
            r"$\bf{FLAG\_CENTRAL\_SPIKE\ (Unzoomed)}$" + "\n" +
            rf"Spike Cut: $R_{{1\%}} > {max_central_spike}$" + "\n" +
            f"Selected Runs: {', '.join(str(r) for r in example_runs)}"
        )
        ax.text(0.04, 0.94, info_text, transform=ax.transAxes, ha="left", va="top", fontsize=11.0,
                bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="gray", alpha=0.92, zorder=4))

        # 2. Zoomed (Detailed view around threshold [0, 2.5])
        fig_z, ax_z = plt.subplots(figsize=(9, 6.5))
        z_min, z_max = 0.0, 2.5
        r1_c = np.clip(r1s, z_min, z_max)
        r15_c = np.clip(r15s, z_min, z_max)
        bins_z = np.linspace(z_min, z_max, 45)

        ax_z.hist(r1_c, bins=bins_z, histtype="step", color="crimson", linewidth=2.0, label=r"$R_{1\%}$ Distribution")
        ax_z.hist(r15_c, bins=bins_z, histtype="step", color="royalblue", linewidth=1.4, alpha=0.5, label=r"$R_{1-5\%}$ Distribution")
        ax_z.axvline(1.0, color="gray", linestyle="-", linewidth=1.2)
        ax_z.axvline(max_central_spike, color="crimson", linestyle="--", linewidth=2.2, label=rf"Spike Cut ($R_{{1\%}} > {max_central_spike}$)")

        ax_z.set_xlabel("Central Ratio to Plateau", fontsize=15)
        ax_z.set_ylabel("Runs", fontsize=15)
        ax_z.set_yscale("log")
        ax_z.set_ylim(0.5, 4500)
        ax_z.set_xlim(z_min, z_max)
        ax_z.legend(loc="upper right", fontsize=10.5, frameon=True, framealpha=0.92)
        ax_z.grid(True, linestyle="--", alpha=0.3, which="both")

        info_text_z = (
            r"$\bf{FLAG\_CENTRAL\_SPIKE\ (Zoomed\ Core)}$" + "\n" +
            rf"Spike Cut: $R_{{1\%}} > {max_central_spike}$" + "\n" +
            f"Example runs lie at R_1% = 6.65 to 6.85\n(visible on unzoomed plot)"
        )
        ax_z.text(0.04, 0.94, info_text_z, transform=ax_z.transAxes, ha="left", va="top", fontsize=11.0,
                  bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="gray", alpha=0.92, zorder=4))

    elif failure_mode == "FLAG_EMPTY_OR_ZERO":
        nevts = [float(d["total_events"]) for d in metrics_list if not np.isnan(d.get("total_events", np.nan))]
        run_vals = [float(d.get("total_events", 0.0)) for d in example_dicts]

        log_bins = np.logspace(0, np.log10(max(nevts) if nevts else 1e7), 40)
        ax.hist(nevts, bins=log_bins, histtype="step", color="coral", linewidth=2.0, label="All Runs Distribution")
        for idx, (d, val) in enumerate(zip(example_dicts, run_vals)):
            st = line_styles[idx % len(line_styles)]
            ax.axvline(max(1.0, val), color=st["color"], linestyle=st["linestyle"], linewidth=st["linewidth"],
                       label=rf"Example Run {d.get('run_number', d.get('run'))} (Events = {val:.2e})")

        ax.set_xlabel("Total Events per Run", fontsize=15)
        ax.set_ylabel("Runs", fontsize=15)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylim(0.5, 3500)
        ax.legend(loc="upper left", fontsize=11, frameon=True, framealpha=0.92)
        ax.grid(True, linestyle="--", alpha=0.3, which="both")

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Save zoomed figure if generated
    if fig_z is not None:
        fig_z.tight_layout()
        fig_z.savefig(zoomed_path, dpi=300, bbox_inches="tight")
        plt.close(fig_z)


def generate_failure_mode_examples(metrics_list, output_dir, example_runs_per_mode=3,
                                   cent_flat_min=10.0, cent_flat_max=70.0,
                                   max_rms_pct=4.5, max_dev_pct=8.0,
                                   max_slope_per_10pct=2.5, max_central_spike=1.30,
                                   min_central_ratio=0.80):
    """
    For each failure mode, select up to `example_runs_per_mode` representative runs
    and generate their 1D centrality distribution + ratio diagnostic plots.
    Also generates a version of the relevant centrality metric distribution plot
    with vertical lines showing where the example runs fall.
    Saves plots into output_dir / "failure_examples" / <FAILURE_MODE> /
    Returns a dict mapping failure_mode -> list of example run numbers.
    """
    output_dir = Path(output_dir)
    examples_dir = output_dir / "failure_examples"

    # Configure failure modes with prioritization and domain-specific filtering:
    # 1. FLAG_SLOPE_DRIFT: extreme linear tilts across plateau
    # 2. FLAG_NON_FLAT: non-linear plateau structure (troughs, humps, ripples) excluding slope drift runs
    # 3. FLAG_CENTRAL_DROP: severe central drop (< 0.80)
    # 4. FLAG_CENTRAL_SPIKE: severe central spike (> 1.30)
    # 5. FLAG_EMPTY_OR_ZERO: zero events or zero plateau
    failure_modes_config = [
        {
            "mode": "FLAG_SLOPE_DRIFT",
            "description": "Plateau Slope Drift (|slope| > 2.5%/10% cent)",
            "filter_fn": lambda d: "FLAG_SLOPE_DRIFT" in d.get("status", "").split(";"),
            "sort_key": lambda d: abs(d.get("slope_per_10pct", 0.0)) if not np.isnan(d.get("slope_per_10pct", 0.0)) else 0.0,
            "reverse": True,
        },
        {
            "mode": "FLAG_NON_FLAT",
            "description": "Non-Flat Plateau (RMS > 4.5% or MaxDev > 8.0%)",
            # Prioritize pure non-flat runs not caught by any of the other three failure modes
            "filter_fn": lambda d: d.get("status") == "FLAG_NON_FLAT",
            "sort_key": lambda d: d.get("rms_plat_pct", 0.0) if not np.isnan(d.get("rms_plat_pct", 0.0)) else 0.0,
            "reverse": True,
        },
        {
            "mode": "FLAG_CENTRAL_DROP",
            "description": "Central Drop (R_1-5% < 0.80)",
            # Prioritize pure central drop runs not caught by any other failure mode
            "filter_fn": lambda d: d.get("status") == "FLAG_CENTRAL_DROP",
            "sort_key": lambda d: d.get("ratio_1_5", 1.0) if not np.isnan(d.get("ratio_1_5", 1.0)) else 1.0,
            "reverse": False,
        },
        {
            "mode": "FLAG_CENTRAL_SPIKE",
            "description": "Central Spike (R_1% > 1.30 or R_1-5% > 1.20)",
            # Prioritize pure central spike runs not caught by any other failure mode
            "filter_fn": lambda d: d.get("status") == "FLAG_CENTRAL_SPIKE",
            "sort_key": lambda d: max(
                d.get("ratio_1", 0.0) if not np.isnan(d.get("ratio_1", 0.0)) else 0.0,
                d.get("ratio_1_5", 0.0) if not np.isnan(d.get("ratio_1_5", 0.0)) else 0.0,
            ),
            "reverse": True,
        },
        {
            "mode": "FLAG_EMPTY_OR_ZERO",
            "description": "Empty Histogram or Zero Plateau",
            "filter_fn": lambda d: d.get("status") == "FLAG_EMPTY_OR_ZERO",
            "sort_key": lambda d: d.get("total_events", 0.0),
            "reverse": False,
        },
    ]

    examples_map = {}
    used_runs = set()

    for cfg in failure_modes_config:
        mode = cfg["mode"]
        filter_fn = cfg.get("filter_fn", lambda d: mode in d.get("status", "").split(";"))
        matching_runs = [d for d in metrics_list if filter_fn(d)]
        if not matching_runs:
            # Fallback if domain filter yielded no runs
            matching_runs = [d for d in metrics_list if mode in d.get("status", "").split(";")]

        if not matching_runs:
            examples_map[mode] = []
            continue

        # Prefer runs not already used as examples for another failure mode
        sorted_runs = sorted(
            matching_runs,
            key=lambda d: (d["run_number"] in used_runs, -cfg["sort_key"](d) if cfg["reverse"] else cfg["sort_key"](d))
        )
        selected = sorted_runs[:example_runs_per_mode]
        examples_map[mode] = [d["run_number"] for d in selected]
        for d in selected:
            used_runs.add(d["run_number"])

        mode_dir = examples_dir / mode
        mode_dir.mkdir(parents=True, exist_ok=True)

        for d in selected:
            plot_path = mode_dir / f"run_{d['run_number']}_centrality.png"
            plot_failure_example(
                metric=d,
                failure_mode=mode,
                output_path=plot_path,
                cent_flat_min=cent_flat_min,
                cent_flat_max=cent_flat_max,
            )

        # Generate the relevant metric distribution with vertical lines for the example runs
        dist_plot_path = mode_dir / "centrality_metric_distributions.png"
        plot_failure_mode_metric_distribution(
            metrics_list=metrics_list,
            failure_mode=mode,
            example_runs=[d["run_number"] for d in selected],
            output_path=dist_plot_path,
            max_rms_pct=max_rms_pct,
            max_dev_pct=max_dev_pct,
            max_slope_per_10pct=max_slope_per_10pct,
            max_central_spike=max_central_spike,
            min_central_ratio=min_central_ratio,
        )

    return examples_map


def generate_top_flat_examples(metrics_list, output_dir, n_examples=3,
                               cent_flat_min=10.0, cent_flat_max=70.0):
    """
    Select up to `n_examples` representative runs with the best / flattest centrality distributions
    (status == 'GOOD', ranked by lowest plateau RMS non-flatness) and generate their 1D
    centrality distribution + ratio diagnostic plots.
    Saves plots into output_dir / "top_flat_examples" /
    Returns a list of the selected run numbers.
    """
    output_dir = Path(output_dir)
    top_flat_dir = output_dir / "top_flat_examples"

    # Filter for GOOD runs first, or fallback to runs with valid plateau if none GOOD
    good_runs = [d for d in metrics_list if d.get("status") == "GOOD"]
    candidates = good_runs if good_runs else [
        d for d in metrics_list
        if d.get("plateau_avg", 0) > 0 and not np.isnan(d.get("rms_plat_pct", np.nan))
    ]

    if not candidates:
        print("Warning: No valid candidate runs found for top flat examples.")
        return []

    # Sort by lowest RMS non-flatness (flattest plateau)
    sorted_runs = sorted(
        candidates,
        key=lambda d: d.get("rms_plat_pct", 999.0) if not np.isnan(d.get("rms_plat_pct", 999.0)) else 999.0
    )
    selected = sorted_runs[:n_examples]
    top_flat_dir.mkdir(parents=True, exist_ok=True)

    selected_runs = []
    for rank, d in enumerate(selected, start=1):
        run_num = d["run_number"]
        selected_runs.append(run_num)
        plot_path = top_flat_dir / f"run_{run_num}_centrality.png"
        plot_top_flat_example(
            metric=d,
            rank=rank,
            output_path=plot_path,
            cent_flat_min=cent_flat_min,
            cent_flat_max=cent_flat_max,
        )

    return selected_runs


def generate_user_examples(metrics_list, output_dir, example_runs,
                           cent_flat_min=10.0, cent_flat_max=70.0):
    """
    Generate 2-panel 1D centrality diagnostic example plots (Events + Ratio to Plateau)
    for specific run numbers requested by the user.
    Saves plots into output_dir / "user_examples" / run_<run>_centrality.png
    Returns a list of run numbers that were successfully plotted.
    """
    if hasattr(metrics_list, "to_dict"):
        metrics_list = metrics_list.to_dict("records")

    if not example_runs:
        return []

    output_dir = Path(output_dir)
    user_examples_dir = output_dir / "user_examples"
    user_examples_dir.mkdir(parents=True, exist_ok=True)

    run_dict = {d.get("run_number", d.get("run")): d for d in metrics_list}
    plotted_runs = []

    for r in example_runs:
        if r not in run_dict:
            print(f"Warning: User-specified example run {r} not found in processed metrics.")
            continue
        d = run_dict[r]
        plot_path = user_examples_dir / f"run_{r}_centrality.png"
        plot_user_example(
            metric=d,
            output_path=plot_path,
            cent_flat_min=cent_flat_min,
            cent_flat_max=cent_flat_max,
            title_suffix="User Specified",
        )
        plotted_runs.append(r)

    if plotted_runs:
        print(f"User-specified example plots generated for {len(plotted_runs)} run(s) in {user_examples_dir.resolve()}")

    return plotted_runs


def print_failure_summary_table(metrics_list, output_dir=None, examples_map=None, top_flat_runs=None, user_example_runs=None):
    """
    Print an ASCII summary table of runs per failure mode to stdout
    and save it to centrality_qa_summary_table.txt in output_dir.
    """
    total_runs = len(metrics_list)
    good_runs = sum(1 for d in metrics_list if d.get("status") == "GOOD")
    flagged_runs = total_runs - good_runs

    failure_modes_info = [
        ("FLAG_CENTRAL_SPIKE", "Central Spike (R_1% > 1.30 / R_1-5% > 1.20)"),
        ("FLAG_CENTRAL_DROP",  "Central Drop (R_1-5% < 0.80)"),
        ("FLAG_NON_FLAT",      "Non-Flat Plateau (RMS > 4.5% / MaxDev > 8%)"),
        ("FLAG_SLOPE_DRIFT",   "Plateau Slope Drift (|slope| > 2.5%/10% cent)"),
        ("FLAG_EMPTY_OR_ZERO", "Empty Histogram or Zero Plateau Events"),
    ]

    mode_counts = {}
    mode_unique_counts = {}
    for mode, _ in failure_modes_info:
        cnt = sum(1 for d in metrics_list if mode in [f for f in str(d.get("status", "")).split(";") if f])
        uniq = sum(1 for d in metrics_list if [f for f in str(d.get("status", "")).split(";") if f] == [mode])
        mode_counts[mode] = cnt
        mode_unique_counts[mode] = uniq

    table_width = 98
    lines = []
    lines.append("=" * table_width)
    lines.append("CENTRALITY RUN AGGREGATE QA SUMMARY")
    lines.append("=" * table_width)
    lines.append(f"Total Runs Analyzed:     {total_runs:,}")
    pct_good = (good_runs / total_runs * 100.0) if total_runs > 0 else 0.0
    pct_flagged = (flagged_runs / total_runs * 100.0) if total_runs > 0 else 0.0
    lines.append(f"  Passed (GOOD):         {good_runs:,} ({pct_good:5.1f}%)")
    lines.append(f"  Flagged (Outliers):     {flagged_runs:,} ({pct_flagged:5.1f}%)")
    lines.append("-" * table_width)
    lines.append("FAILURE MODE BREAKDOWN:")
    lines.append(f"  {'Failure Mode':<22} {'Description':<46} {'Total':>7} {'% Runs':>8} {'Unique':>8} {'% Runs':>8}")
    lines.append("  " + "-" * (table_width - 4))

    for mode, desc in failure_modes_info:
        cnt = mode_counts[mode]
        uniq = mode_unique_counts[mode]
        pct = (cnt / total_runs * 100.0) if total_runs > 0 else 0.0
        pct_u = (uniq / total_runs * 100.0) if total_runs > 0 else 0.0
        lines.append(f"  {mode:<22} {desc:<46} {cnt:>7} {pct:>7.1f}% {uniq:>8} {pct_u:>7.1f}%")

    lines.append("-" * table_width)
    lines.append("(Note: 'Total' counts all runs triggering that flag; 'Unique' counts runs failing solely that mode)")

    if examples_map:
        lines.append("-" * table_width)
        lines.append("EXAMPLE RUNS GENERATED PER FAILURE MODE (in failure_examples/):")
        for mode, _ in failure_modes_info:
            ex_runs = examples_map.get(mode, [])
            if ex_runs:
                runs_str = ", ".join(str(r) for r in ex_runs)
                lines.append(f"  {mode:<22} -> Runs: [{runs_str}]  (dir: failure_examples/{mode}/)")
            else:
                lines.append(f"  {mode:<22} -> None flagged")

    if top_flat_runs:
        lines.append("-" * table_width)
        lines.append("TOP FLAT RUN EXAMPLES GENERATED (in top_flat_examples/):")
        run_dict = {d.get("run_number", d.get("run")): d for d in metrics_list}
        for rank, r in enumerate(top_flat_runs, start=1):
            d = run_dict.get(r, {})
            rms_val = f"{d.get('rms_plat_pct', np.nan):.2f}%" if not np.isnan(d.get('rms_plat_pct', np.nan)) else "N/A"
            slope_val = f"{d.get('slope_per_10pct', np.nan):+.2f}%" if not np.isnan(d.get('slope_per_10pct', np.nan)) else "N/A"
            evts_val = f"{float(d.get('total_events', 0)):.2e}" if d.get('total_events') is not None else "N/A"
            lines.append(f"  Rank #{rank}: Run {r:<8} (Plateau RMS: {rms_val}, Slope: {slope_val}, Evts: {evts_val})  -> top_flat_examples/run_{r}_centrality.png")

    if user_example_runs:
        lines.append("-" * table_width)
        lines.append("USER-SPECIFIED EXAMPLE RUNS GENERATED (in user_examples/):")
        run_dict = {d.get("run_number", d.get("run")): d for d in metrics_list}
        for r in user_example_runs:
            d = run_dict.get(r, {})
            status_val = d.get("status", "UNKNOWN")
            rms_val = f"{d.get('rms_plat_pct', np.nan):.2f}%" if not np.isnan(d.get('rms_plat_pct', np.nan)) else "N/A"
            slope_val = f"{d.get('slope_per_10pct', np.nan):+.2f}%" if not np.isnan(d.get('slope_per_10pct', np.nan)) else "N/A"
            evts_val = f"{float(d.get('total_events', 0)):.2e}" if d.get('total_events') is not None else "N/A"
            lines.append(f"  Run {r:<8} (Status: {status_val}, Plateau RMS: {rms_val}, Slope: {slope_val}, Evts: {evts_val})  -> user_examples/run_{r}_centrality.png")

    lines.append("=" * table_width)

    summary_text = "\n".join(lines)
    print("\n" + summary_text + "\n")

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        table_path = output_dir / "centrality_qa_summary_table.txt"
        with open(table_path, "w") as f:
            f.write(summary_text + "\n")
        print(f"Summary table text file saved to: {table_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Run Aggregate QA: Quantify Centrality Flatness, Generate Trend/Heatmap Plots, and Flag Outlier Runs."
    )
    parser.add_argument("-f", "--file", type=Path, help="Text file containing list of ROOT file paths (one per line).")
    parser.add_argument("-o", "--output-dir", type=Path, default=Path("centrality_aggregate_qa"),
                        help="Directory to save aggregate plots, reports, and detailed test QA (default: centrality_aggregate_qa).")
    parser.add_argument("--hist-name", type=str, default="hCentrality",
                        help="Name of centrality histogram to analyze (default: hCentrality).")
    parser.add_argument("--test-runs", nargs="*", type=int, default=[68144, 72020, 76020],
                        help="Run numbers to generate detailed per-run QA plots for (default: 68144 72020 76020).")
    parser.add_argument("--no-test-plots", action="store_true",
                        help="Disable generating detailed per-run plots for the test runs.")
    parser.add_argument("--plot-outliers", action="store_true",
                        help="Generate detailed per-run QA plots for all flagged outlier runs as well.")
    parser.add_argument("-j", "--workers", type=int, default=None,
                        help="Number of parallel worker processes (default: min(cpu_count(), 32)).")
    parser.add_argument("--cent-flat-min", type=float, default=10.0,
                        help="Minimum centrality [%%] for flat plateau (default: 10.0).")
    parser.add_argument("--cent-flat-max", type=float, default=70.0,
                        help="Maximum centrality [%%] for flat plateau (default: 70.0).")
    parser.add_argument("--max-rms", type=float, default=4.5,
                        help="Maximum plateau RMS non-flatness [%%] threshold before flagging NON_FLAT (default: 4.5).")
    parser.add_argument("--max-dev", type=float, default=8.0,
                        help="Maximum single-bin deviation [%%] threshold on plateau (default: 8.0).")
    parser.add_argument("--max-slope", type=float, default=2.5,
                        help="Maximum slope tolerance in %% per 10%% centrality (default: 2.5).")
    parser.add_argument("--max-central-spike", type=float, default=1.30,
                        help="Maximum allowed ratio in 1%% centrality bin (R_1%%) before flagging CENTRAL_SPIKE (default: 1.30).")
    parser.add_argument("--min-central-ratio", type=float, default=0.80,
                        help="Minimum allowed ratio in 1-5%% centrality (R_1-5%%) before flagging CENTRAL_DROP (default: 0.80).")
    parser.add_argument("--max-central-ratio", type=float, default=1.20,
                        help="Maximum allowed ratio in 1-5%% centrality (R_1-5%%) before flagging CENTRAL_SPIKE (default: 1.20).")
    parser.add_argument("--syst-floor", type=float, default=0.02,
                        help="Systematic fractional uncertainty floor for chi2 computation (default: 0.02).")
    parser.add_argument("--example-runs-per-mode", type=int, default=3,
                        help="Number of representative example run plots to generate for each failure mode (default: 3).")
    parser.add_argument("--no-failure-examples", action="store_true",
                        help="Disable generating failure mode example plots.")
    parser.add_argument("--top-flat-examples", type=int, default=3,
                        help="Number of representative example run plots to generate for top flattest runs (default: 3).")
    parser.add_argument("--no-top-flat-examples", action="store_true",
                        help="Disable generating top flat example plots.")
    parser.add_argument("--example-runs", "--user-example-runs", "--user-runs",
                        nargs="*", default=None,
                        help="Specific run number(s) specified by user to generate 2-panel 1D centrality diagnostic example plots for.")
    parser.add_argument("files", nargs="*", type=Path, help="Positional list of ROOT file paths.")
    args = parser.parse_args()

    file_list = []
    if args.files:
        file_list.extend(args.files)

    if args.file:
        try:
            with args.file.open("r") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#"):
                        file_list.append(Path(line))
        except Exception as e:
            print(f"Error reading file list {args.file}: {e}")
            sys.exit(1)

    if not file_list:
        print("Error: You must provide at least one ROOT file or a text file containing ROOT paths (-f / --file).")
        parser.print_help()
        sys.exit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    files_to_process = [Path(p) for p in file_list]
    max_workers = args.workers if args.workers is not None else min(os.cpu_count() or 4, 32)

    print(f"============================================================")
    print(f"Centrality Run Aggregate QA")
    print(f"Input files:        {len(files_to_process)}")
    print(f"Histogram:          {args.hist_name}")
    print(f"Plateau Range:      [{args.cent_flat_min}%, {args.cent_flat_max}%]")
    print(f"Max Plateau RMS:    {args.max_rms}%")
    print(f"Max Plateau Slope:  {args.max_slope}% per 10% cent")
    print(f"Central Spike Cut:  R(1%) > {args.max_central_spike}")
    print(f"Test runs:          {args.test_runs}")
    print(f"Output dir:         {args.output_dir.resolve()}")
    print(f"Worker count:       {max_workers}")
    print(f"============================================================")

    # 1. Parallel Extraction of Metrics
    extract_func = functools.partial(
        extract_run_metrics,
        hist_name=args.hist_name,
        cent_flat_min=args.cent_flat_min,
        cent_flat_max=args.cent_flat_max,
        max_rms_pct=args.max_rms,
        max_dev_pct_cut=args.max_dev,
        max_slope_per_10pct=args.max_slope,
        min_central_ratio=args.min_central_ratio,
        max_central_ratio=args.max_central_ratio,
        max_central_spike=args.max_central_spike,
        syst_floor=args.syst_floor,
    )

    print("Extracting flatness metrics across all runs...")
    metrics_list = []
    errors = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = list(tqdm.tqdm(executor.map(extract_func, files_to_process), total=len(files_to_process)))

    for m, err in results:
        if m is not None:
            metrics_list.append(m)
        if err:
            errors.append(err)

    if not metrics_list:
        print("Error: No valid histogram data could be extracted from any of the input files.")
        if errors:
            print("First 5 errors:")
            for err in errors[:5]:
                print(f"  {err}")
        sys.exit(1)

    # Sort runs chronologically by run number
    metrics_list.sort(key=lambda d: d["run_number"])
    n_total = len(metrics_list)
    n_good = sum(1 for d in metrics_list if d["status"] == "GOOD")
    n_flagged = n_total - n_good

    print(f"Extracted metrics for {n_total} runs ({n_good} GOOD, {n_flagged} FLAGGED).")
    if errors:
        print(f"Encountered {len(errors)} warnings/errors during extraction.")

    # 2. Generate Aggregate Plots
    print("Generating Run Aggregate plots...")
    plot_heatmap(
        metrics_list,
        args.output_dir / "centrality_ratio_heatmap_all_runs.png",
        test_runs=args.test_runs,
    )
    plot_trends(
        metrics_list,
        args.output_dir / "centrality_flatness_trends.png",
        test_runs=args.test_runs,
        max_rms_pct=args.max_rms,
        max_slope_per_10pct=args.max_slope,
        max_central_spike=args.max_central_spike,
        min_central_ratio=args.min_central_ratio,
    )
    plot_metric_distributions(
        metrics_list,
        args.output_dir / "centrality_metric_distributions.png",
        max_rms_pct=args.max_rms,
        max_slope_per_10pct=args.max_slope,
        max_central_spike=args.max_central_spike,
        min_central_ratio=args.min_central_ratio,
    )
    plot_ensemble_profile(
        metrics_list,
        args.output_dir / "centrality_ensemble_profile.png",
        subtitle=f"All Processed Runs (N = {n_total})",
    )
    good_metrics = [d for d in metrics_list if d.get("status") == "GOOD"]
    if good_metrics:
        plot_ensemble_profile(
            good_metrics,
            args.output_dir / "centrality_ensemble_profile_good_runs.png",
            subtitle=f"Good Runs (Passed All QA, N = {len(good_metrics)})",
        )
    else:
        print("Warning: No runs passed all QA checks. Skipping centrality_ensemble_profile_good_runs.png.")

    # 3. Write Reports
    write_summary_reports(metrics_list, args.output_dir)

    # 4. Generate Example 1D Centrality Plots for Each Failure Mode
    examples_map = None
    if not args.no_failure_examples:
        print(f"Generating example 1D centrality plots for each failure mode (up to {args.example_runs_per_mode} per mode)...")
        examples_map = generate_failure_mode_examples(
            metrics_list,
            args.output_dir,
            example_runs_per_mode=args.example_runs_per_mode,
            cent_flat_min=args.cent_flat_min,
            cent_flat_max=args.cent_flat_max,
            max_rms_pct=args.max_rms,
            max_dev_pct=args.max_dev,
            max_slope_per_10pct=args.max_slope,
            max_central_spike=args.max_central_spike,
            min_central_ratio=args.min_central_ratio,
        )

    # 4b. Generate Example 1D Centrality Plots for Top Flat Runs
    top_flat_runs = []
    if not args.no_top_flat_examples and args.top_flat_examples > 0:
        print(f"Generating example 1D centrality plots for top {args.top_flat_examples} flattest runs...")
        top_flat_runs = generate_top_flat_examples(
            metrics_list,
            args.output_dir,
            n_examples=args.top_flat_examples,
            cent_flat_min=args.cent_flat_min,
            cent_flat_max=args.cent_flat_max,
        )

    # 4c. Generate Example 1D Centrality Plots for User-Specified Runs
    user_example_runs = []
    if args.example_runs:
        user_runs_to_plot = []
        for item in args.example_runs:
            for part in str(item).replace(",", " ").split():
                try:
                    user_runs_to_plot.append(int(part))
                except ValueError:
                    print(f"Warning: Could not parse '{part}' as a run number.")
        if user_runs_to_plot:
            print(f"Generating example 1D centrality plots for {len(user_runs_to_plot)} user-specified run(s)...")
            user_example_runs = generate_user_examples(
                metrics_list,
                args.output_dir,
                example_runs=user_runs_to_plot,
                cent_flat_min=args.cent_flat_min,
                cent_flat_max=args.cent_flat_max,
            )

    # 5. Selective Detailed Plotting
    runs_to_detail = set()
    if not args.no_test_plots and args.test_runs:
        runs_to_detail.update(args.test_runs)

    if args.plot_outliers:
        flagged_runs = [d["run_number"] for d in metrics_list if d["status"] != "GOOD"]
        runs_to_detail.update(flagged_runs)

    if runs_to_detail and plot_centrality_qa is not None:
        # Find file paths corresponding to selected runs
        run_to_path = {d["run_number"]: Path(d["file_path"]) for d in metrics_list}
        target_files = [run_to_path[r] for r in runs_to_detail if r in run_to_path]

        if target_files:
            print(f"Generating detailed per-run QA plots for {len(target_files)} selected run(s)...")
            detail_output_dir = args.output_dir / "detailed_qa"
            detail_func = functools.partial(
                plot_centrality_qa.process_file,
                output_dir=detail_output_dir,
                logy=False,
                run_subdirs=True,
                cent_flat_min=args.cent_flat_min,
                cent_flat_max=args.cent_flat_max,
            )

            with concurrent.futures.ProcessPoolExecutor(max_workers=min(len(target_files), max_workers)) as executor:
                detail_results = list(tqdm.tqdm(executor.map(detail_func, target_files), total=len(target_files)))

            detail_errs = [e for e in detail_results if e is not None]
            if detail_errs:
                print(f"Detailed plotting had {len(detail_errs)} warning(s)/error(s).")
            print(f"Detailed QA plots saved in: {detail_output_dir.resolve()}")
    elif runs_to_detail and plot_centrality_qa is None:
        print("Warning: Could not import plot_centrality_qa.py; skipping detailed per-run plots.")

    # 6. Print and Save Summary Table
    print_failure_summary_table(
        metrics_list,
        output_dir=args.output_dir,
        examples_map=examples_map,
        top_flat_runs=top_flat_runs,
        user_example_runs=user_example_runs,
    )

    print(f"All Run Aggregate QA completed successfully. Results saved to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
