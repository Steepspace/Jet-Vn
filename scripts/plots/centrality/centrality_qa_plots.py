import csv
from pathlib import Path
import numpy as np
import uproot
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mplhep as hep
from centrality_qa_metrics import compute_centrality_average

def plot_heatmap(data_list, output_path, test_runs=None):
    hep.style.use("ATLAS")
    fig, ax = plt.subplots(figsize=(12, 6.5))

    runs = [d["run_number"] for d in data_list]
    n_runs = len(runs)
    ratio_matrix = np.array([d["ratios"][1:] for d in data_list])
    cent_edges = data_list[0]["edges"][1:]

    x_grid, y_grid = np.meshgrid(np.arange(n_runs + 1), cent_edges)

    c = ax.pcolormesh(x_grid, y_grid, ratio_matrix.T, cmap="coolwarm", vmin=0.8, vmax=1.2, shading="flat")

    cbar = fig.colorbar(c, ax=ax, pad=0.015, aspect=25, fraction=0.046)
    cbar.set_label("Ratio to Plateau Average", fontsize=15)
    cbar.ax.tick_params(labelsize=13)

    ax.set_ylabel("Centrality [%]", fontsize=16)
    ax.set_xlabel("Run Index", fontsize=16)

    ax.set_xlim(0, n_runs)
    ax.set_ylim(cent_edges[0], cent_edges[-1])

    if n_runs >= 2000: step = 500
    elif n_runs >= 800: step = 200
    elif n_runs >= 400: step = 100
    elif n_runs >= 100: step = 25
    elif n_runs >= 20: step = 10
    else: step = max(1, n_runs // 5)

    tick_positions = np.arange(0, n_runs + 1, step)
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([str(p) for p in tick_positions], fontsize=13)

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


def _draw_rms_scatter(ax, runs, rms_vals, is_good, is_outlier, is_test, max_rms_pct, s_good=30, s_bad=45):
    ax.scatter(runs[is_good], rms_vals[is_good], color="forestgreen", s=s_good, alpha=0.7, label="Good Runs", edgecolors="none")
    ax.scatter(runs[is_outlier], rms_vals[is_outlier], color="crimson", s=s_bad, alpha=0.85, marker="^", label="Flagged Runs")
    if np.any(is_test):
        ax.scatter(runs[is_test], rms_vals[is_test], facecolors="none", edgecolors="black", s=140, linewidths=2.0, marker="o", label="Test Runs")
        for r, val in zip(runs[is_test], rms_vals[is_test]):
            if not np.isnan(val):
                ax.annotate(f"{r}", (r, val), textcoords="offset points", xytext=(0, 10), ha="center", fontsize=11, fontweight="bold")
    ax.axhline(3.0, color="darkgreen", linestyle=":", linewidth=1.5, label="Nominal (3.0%)")
    ax.axhline(max_rms_pct, color="crimson", linestyle="--", linewidth=1.8, label=rf"Cut ({max_rms_pct}%)")
    ax.set_ylim(bottom=0.0)
    ax.grid(True, linestyle="--", alpha=0.4)

def _draw_slope_scatter(ax, runs, slopes, is_good, is_outlier, is_test, max_slope_per_10pct, s_good=30, s_bad=45):
    ax.scatter(runs[is_good], slopes[is_good], color="forestgreen", s=s_good, alpha=0.7, edgecolors="none")
    ax.scatter(runs[is_outlier], slopes[is_outlier], color="crimson", s=s_bad, alpha=0.85, marker="^")
    if np.any(is_test):
        ax.scatter(runs[is_test], slopes[is_test], facecolors="none", edgecolors="black", s=140, linewidths=2.0, marker="o")
        for r, val in zip(runs[is_test], slopes[is_test]):
            if not np.isnan(val):
                ax.annotate(f"{r}", (r, val), textcoords="offset points", xytext=(0, 10), ha="center", fontsize=11, fontweight="bold")
    ax.axhline(0.0, color="gray", linestyle="-", linewidth=1.2)
    ax.axhspan(-max_slope_per_10pct, max_slope_per_10pct, color="forestgreen", alpha=0.12, label=rf"Tolerance $\pm${max_slope_per_10pct}%")
    ax.axhline(max_slope_per_10pct, color="crimson", linestyle="--", linewidth=1.5)
    ax.axhline(-max_slope_per_10pct, color="crimson", linestyle="--", linewidth=1.5)
    ax.grid(True, linestyle="--", alpha=0.4)

def _draw_ratios_scatter(ax, runs, r1s, r15s, r80s, max_central_spike, min_central_ratio):
    ax.scatter(runs, r1s, color="crimson", s=35, alpha=0.65, marker="^", label=r"$R_{1\%}$ (1% Centrality)")
    ax.scatter(runs, r15s, color="royalblue", s=30, alpha=0.6, marker="o", label=r"$R_{1-5\%}$ (1–5% Centrality)")
    ax.scatter(runs, r80s, color="darkorange", s=30, alpha=0.6, marker="s", label=r"$R_{80\%}$ (Peripheral $\sim$80%)")
    ax.axhline(1.0, color="gray", linestyle="-", linewidth=1.2)
    ax.axhline(max_central_spike, color="crimson", linestyle="--", linewidth=1.5, label=rf"Spike Cut ({max_central_spike})")
    ax.axhline(min_central_ratio, color="crimson", linestyle=":", linewidth=1.5, label=rf"Drop Cut ({min_central_ratio})")
    ax.axhspan(min_central_ratio, 1.20, color="royalblue", alpha=0.08)
    ax.set_ylim(0.3, 1.9)
    ax.grid(True, linestyle="--", alpha=0.4)

def _draw_events_scatter(ax, runs, nevts, is_good, is_outlier, is_test, s_good=30, s_bad=45, label_runs=False):
    ax.scatter(runs[is_good], nevts[is_good], color="forestgreen", s=s_good, alpha=0.7, edgecolors="none", label="Good Runs" if label_runs else None)
    ax.scatter(runs[is_outlier], nevts[is_outlier], color="crimson", s=s_bad, alpha=0.85, marker="^", label="Flagged Runs" if label_runs else None)
    if np.any(is_test):
        ax.scatter(runs[is_test], nevts[is_test], facecolors="none", edgecolors="black", s=140, linewidths=2.0, marker="o", label="Test Runs" if label_runs else None)
        for r, val in zip(runs[is_test], nevts[is_test]):
            if not np.isnan(val):
                ax.annotate(f"{r}", (r, val), textcoords="offset points", xytext=(0, 10), ha="center", fontsize=11, fontweight="bold")
    ax.set_yscale("log")
    ax.grid(True, linestyle="--", alpha=0.4)


def plot_trends(data_list, output_path, test_runs=None, max_rms_pct=4.5, max_slope_per_10pct=2.5,
                max_central_spike=1.30, min_central_ratio=0.80, save_individual=True):
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

    fig, axes = plt.subplots(4, 1, figsize=(13, 16), sharex=True)

    _draw_rms_scatter(axes[0], runs, rms_vals, is_good, is_outlier, is_test, max_rms_pct)
    axes[0].set_ylabel("Plateau RMS [%]\n(10-70% Flatness)", fontsize=13)
    axes[0].legend(loc="upper right", ncol=3, fontsize=11, frameon=True, framealpha=0.9)

    _draw_slope_scatter(axes[1], runs, slopes, is_good, is_outlier, is_test, max_slope_per_10pct)
    axes[1].set_ylabel("Plateau Slope\n[% per 10% Cent]", fontsize=13)
    axes[1].legend(loc="upper right", fontsize=11, frameon=True, framealpha=0.9)

    _draw_ratios_scatter(axes[2], runs, r1s, r15s, r80s, max_central_spike, min_central_ratio)
    axes[2].set_ylabel("Ratio to Plateau", fontsize=14)
    axes[2].legend(loc="upper right", ncol=3, fontsize=11, frameon=True, framealpha=0.9)

    _draw_events_scatter(axes[3], runs, nevts, is_good, is_outlier, is_test)
    axes[3].set_ylabel("Total Events", fontsize=14)
    axes[3].set_xlabel("Run Number", loc="center", fontsize=15)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    if save_individual:
        stem = output_path.stem
        fig_rms, ax_rms = plt.subplots(figsize=(11, 6.5))
        _draw_rms_scatter(ax_rms, runs, rms_vals, is_good, is_outlier, is_test, max_rms_pct, s_good=32, s_bad=48)
        ax_rms.set_xlabel("Run Number", loc="center", fontsize=15)
        ax_rms.set_ylabel("Plateau RMS Non-Flatness [%] (10–70% Centrality)", fontsize=14)
        ax_rms.legend(loc="upper right", ncol=3, fontsize=11, frameon=True, framealpha=0.9)
        fig_rms.tight_layout()
        fig_rms.savefig(output_path.parent / f"{stem}_rms.png", dpi=300, bbox_inches="tight")
        plt.close(fig_rms)

        fig_slope, ax_slope = plt.subplots(figsize=(11, 6.5))
        _draw_slope_scatter(ax_slope, runs, slopes, is_good, is_outlier, is_test, max_slope_per_10pct, s_good=32, s_bad=48)
        ax_slope.set_xlabel("Run Number", loc="center", fontsize=15)
        ax_slope.set_ylabel("Plateau Slope [% per 10% Centrality]", fontsize=14)
        ax_slope.legend(loc="upper right", ncol=3, fontsize=11, frameon=True, framealpha=0.9)
        fig_slope.tight_layout()
        fig_slope.savefig(output_path.parent / f"{stem}_slope.png", dpi=300, bbox_inches="tight")
        plt.close(fig_slope)

        fig_rat, ax_rat = plt.subplots(figsize=(11, 6.5))
        _draw_ratios_scatter(ax_rat, runs, r1s, r15s, r80s, max_central_spike, min_central_ratio)
        ax_rat.set_xlabel("Run Number", loc="center", fontsize=15)
        ax_rat.set_ylabel("Ratio to Plateau Average", fontsize=14)
        ax_rat.legend(loc="upper right", ncol=3, fontsize=11, frameon=True, framealpha=0.9)
        fig_rat.tight_layout()
        fig_rat.savefig(output_path.parent / f"{stem}_ratios.png", dpi=300, bbox_inches="tight")
        plt.close(fig_rat)

        fig_ev, ax_ev = plt.subplots(figsize=(11, 6.5))
        _draw_events_scatter(ax_ev, runs, nevts, is_good, is_outlier, is_test, s_good=32, s_bad=48, label_runs=True)
        ax_ev.set_xlabel("Run Number", loc="center", fontsize=15)
        ax_ev.set_ylabel("Total Events", fontsize=14)
        ax_ev.legend(loc="lower right", fontsize=11, frameon=True, framealpha=0.9)
        fig_ev.tight_layout()
        fig_ev.savefig(output_path.parent / f"{stem}_events.png", dpi=300, bbox_inches="tight")
        plt.close(fig_ev)


def _draw_rms_hist(ax, rms_vals, max_rms_pct, n_pass_rms, pct_pass_rms, n_fail_rms, pct_fail_rms, zoomed=False, title_font=10.0, pad=0.35):
    if len(rms_vals) > 0:
        if zoomed:
            data = np.clip(rms_vals, 0, 10)
            bins = 40
            x_max = 10
            extra_txt = "\n(Clipped at 10%)"
        else:
            data = rms_vals
            x_max = max(12.0, max(rms_vals) * 1.08)
            bins = np.linspace(0, x_max, 50)
            extra_txt = f"\nMax RMS: {max(rms_vals):.1f}%"

        ax.hist(data, bins=bins, histtype="step", color="steelblue", linewidth=2.0)
        ax.axvline(max_rms_pct, color="crimson", linestyle="--", linewidth=2, label=rf"Cut ({max_rms_pct}%)")
        box_text = (
            rf"Pass ($\leq {max_rms_pct}\%$): {n_pass_rms:,} ({pct_pass_rms:.1f}%)" + "\n" +
            rf"Fail (> {max_rms_pct}%): {n_fail_rms:,} ({pct_fail_rms:.1f}%)" + extra_txt
        )
        ax.text(0.96, 0.78, box_text, transform=ax.transAxes, ha="right", va="top", fontsize=title_font,
                bbox=dict(boxstyle=f"round,pad={pad}", facecolor="white", edgecolor="gray", alpha=0.92, zorder=4))
        ax.set_xlim(0, x_max)
    ax.set_yscale("log")
    ax.set_ylim(0.5, 3500)
    ax.grid(True, linestyle="--", alpha=0.3, which="both")

def _draw_slope_hist(ax, slopes, max_slope, n_pass, pct_pass, n_fail, pct_fail, zoomed=False, title_font=10.0, pad=0.35):
    if len(slopes) > 0:
        if zoomed:
            data = np.clip(slopes, -6, 6)
            bins = 40
            x_min, x_max = -6, 6
            extra_txt = r"\n(Clipped at $\pm$6%)"
        else:
            data = slopes
            x_min = min(-5.0, min(slopes) * 1.2)
            x_max = max(6.0, max(slopes) * 1.15)
            bins = np.linspace(x_min, x_max, 55)
            extra_txt = f"\nMax Slope: {max(slopes):+.1f}%"

        ax.hist(data, bins=bins, histtype="step", color="mediumseagreen", linewidth=2.0)
        ax.axvline(max_slope, color="crimson", linestyle="--", linewidth=2)
        ax.axvline(-max_slope, color="crimson", linestyle="--", linewidth=2, label=rf"Tol ($\pm${max_slope}%)")
        box_text = (
            rf"In Tol: {n_pass:,} ({pct_pass:.1f}%)" + "\n" +
            rf"Fail (> $\pm${max_slope}%): {n_fail:,} ({pct_fail:.1f}%)" + extra_txt
        )
        ax.text(0.04, 0.94, box_text, transform=ax.transAxes, ha="left", va="top", fontsize=title_font,
                bbox=dict(boxstyle=f"round,pad={pad}", facecolor="white", edgecolor="gray", alpha=0.92, zorder=4))
        ax.set_xlim(x_min, x_max)
    ax.set_yscale("log")
    ax.set_ylim(0.5, 3500)
    ax.grid(True, linestyle="--", alpha=0.3, which="both")

def _draw_ratios_hist(ax, r1s, r15s, max_spike, min_drop, n_spike, pct_spike, n_drop, pct_drop, zoomed=False, title_font=9.8, pad=0.35):
    if len(r1s) > 0 and len(r15s) > 0:
        if zoomed:
            d1, d15 = np.clip(r1s, 0.4, 2.0), np.clip(r15s, 0.4, 2.0)
            bins = np.linspace(0.4, 2.0, 35)
            x_min, x_max = 0.4, 2.0
            extra_txt = "\n(Clipped at [0.4, 2.0])"
        else:
            d1, d15 = r1s, r15s
            x_min = 0
            x_max = max(2.5, max(r1s) * 1.08)
            bins = np.linspace(0, x_max, 55)
            extra_txt = rf"\nMax $R_{{1\%}}$: {max(r1s):.2f}"

        ax.hist(d1, bins=bins, histtype="step", color="crimson", linewidth=2.0, label=r"$R_{1\%}$ (1% Centrality)")
        ax.hist(d15, bins=bins, histtype="step", color="royalblue", linewidth=2.0, label=r"$R_{1-5\%}$ (1–5% Centrality)")
        ax.axvline(1.0, color="gray", linestyle="-", linewidth=1.5)
        ax.axvline(max_spike, color="crimson", linestyle="--", linewidth=2, label=rf"Spike Cut ({max_spike})")
        ax.axvline(min_drop, color="darkorange", linestyle="--", linewidth=2, label=rf"Drop Cut ({min_drop})")
        box_text = (
            rf"Spike Fail (> {max_spike}): {n_spike:,} ({pct_spike:.1f}%)" + "\n" +
            rf"Drop Fail (< {min_drop}): {n_drop:,} ({pct_drop:.1f}%)" + extra_txt
        )
        ax.text(0.96, 0.94, box_text, transform=ax.transAxes, ha="right", va="top", fontsize=title_font,
                bbox=dict(boxstyle=f"round,pad={pad}", facecolor="white", edgecolor="gray", alpha=0.92, zorder=4))
        ax.set_xlim(x_min, x_max)
    ax.set_yscale("log")
    ax.set_ylim(0.5, 4500)
    ax.grid(True, linestyle="--", alpha=0.3, which="both")

def _draw_events_hist(ax, nevts, log_bins, evts_box_text, title_font=10.5, pad=0.35):
    if len(nevts) > 0 and log_bins is not None:
        ax.hist(nevts, bins=log_bins, histtype="step", color="coral", linewidth=2.0)
        ax.text(0.04, 0.94, evts_box_text, transform=ax.transAxes, ha="left", va="top", fontsize=title_font,
                bbox=dict(boxstyle=f"round,pad={pad}", facecolor="white", edgecolor="gray", alpha=0.92, zorder=4))
        ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(0.5, 3500)
    ax.grid(True, linestyle="--", alpha=0.3, which="both")


def plot_metric_distributions(data_list, output_path, max_rms_pct=4.5, max_slope_per_10pct=2.5,
                              max_central_spike=1.30, min_central_ratio=0.80, save_individual=True):
    hep.style.use("ATLAS")
    if hasattr(data_list, "to_dict"): data_list = data_list.to_dict("records")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rms_vals = [d["rms_plat_pct"] for d in data_list if not np.isnan(d["rms_plat_pct"])]
    slopes = [d["slope_per_10pct"] for d in data_list if not np.isnan(d["slope_per_10pct"])]
    r1s = [d["ratio_1"] for d in data_list if not np.isnan(d["ratio_1"])]
    r15s = [d["ratio_1_5"] for d in data_list if not np.isnan(d["ratio_1_5"])]
    nevts = [d["total_events"] for d in data_list if d["total_events"] > 0]

    n_fail_rms = sum(1 for v in rms_vals if v > max_rms_pct)
    pct_fail_rms = 100.0 * n_fail_rms / len(rms_vals) if rms_vals else 0.0
    n_pass_rms = len(rms_vals) - n_fail_rms
    pct_pass_rms = 100.0 * n_pass_rms / len(rms_vals) if rms_vals else 0.0

    n_fail_slope = sum(1 for v in slopes if abs(v) > max_slope_per_10pct)
    pct_fail_slope = 100.0 * n_fail_slope / len(slopes) if slopes else 0.0
    n_pass_slope = len(slopes) - n_fail_slope
    pct_pass_slope = 100.0 * n_pass_slope / len(slopes) if slopes else 0.0

    n_spike = sum(1 for v in r1s if v > max_central_spike)
    pct_spike = 100.0 * n_spike / len(r1s) if r1s else 0.0
    n_drop = sum(1 for v in r15s if v < min_central_ratio)
    pct_drop = 100.0 * n_drop / len(r15s) if r15s else 0.0

    log_bins, evts_box_text = None, ""
    if nevts:
        log_bins = np.logspace(np.log10(max(1, min(nevts))), np.log10(max(nevts)), 40)
        evts_box_text = (
            rf"Total Runs: {len(nevts):,}" + "\n" +
            rf"Median: {float(np.median(nevts)):.2e} evts/run" + "\n" +
            rf"Total: {float(np.sum(nevts)):.2e} events"
        )

    # 1. Unzoomed 4-panel
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    _draw_rms_hist(axes[0, 0], rms_vals, max_rms_pct, n_pass_rms, pct_pass_rms, n_fail_rms, pct_fail_rms, zoomed=False)
    axes[0, 0].set_xlabel("Plateau RMS Non-Flatness [%]")
    axes[0, 0].set_ylabel("Runs")
    axes[0, 0].legend(loc="upper right", fontsize=10.5)

    _draw_slope_hist(axes[0, 1], slopes, max_slope_per_10pct, n_pass_slope, pct_pass_slope, n_fail_slope, pct_fail_slope, zoomed=False)
    axes[0, 1].set_xlabel("Plateau Slope [% per 10% Cent]")
    axes[0, 1].set_ylabel("Runs")
    axes[0, 1].legend(loc="upper right", fontsize=10.5)

    _draw_ratios_hist(axes[1, 0], r1s, r15s, max_central_spike, min_central_ratio, n_spike, pct_spike, n_drop, pct_drop, zoomed=False)
    axes[1, 0].set_xlabel("Central Ratio to Plateau")
    axes[1, 0].set_ylabel("Runs")
    axes[1, 0].legend(loc="upper left", fontsize=9.2, frameon=True, framealpha=0.9)

    _draw_events_hist(axes[1, 1], nevts, log_bins, evts_box_text)
    axes[1, 1].set_xlabel("Total Events per Run")
    axes[1, 1].set_ylabel("Runs")

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    # 2. Zoomed 4-panel
    fig_z, axes_z = plt.subplots(2, 2, figsize=(12, 10))
    _draw_rms_hist(axes_z[0, 0], rms_vals, max_rms_pct, n_pass_rms, pct_pass_rms, n_fail_rms, pct_fail_rms, zoomed=True)
    axes_z[0, 0].set_xlabel("Plateau RMS Non-Flatness [%]")
    axes_z[0, 0].set_ylabel("Runs")
    axes_z[0, 0].legend(loc="upper right", fontsize=10.5)

    _draw_slope_hist(axes_z[0, 1], slopes, max_slope_per_10pct, n_pass_slope, pct_pass_slope, n_fail_slope, pct_fail_slope, zoomed=True)
    axes_z[0, 1].set_xlabel("Plateau Slope [% per 10% Cent]")
    axes_z[0, 1].set_ylabel("Runs")
    axes_z[0, 1].legend(loc="upper right", fontsize=10.5)

    _draw_ratios_hist(axes_z[1, 0], r1s, r15s, max_central_spike, min_central_ratio, n_spike, pct_spike, n_drop, pct_drop, zoomed=True)
    axes_z[1, 0].set_xlabel("Central Ratio to Plateau")
    axes_z[1, 0].set_ylabel("Runs")
    axes_z[1, 0].legend(loc="upper left", fontsize=9.2, frameon=True, framealpha=0.9)

    _draw_events_hist(axes_z[1, 1], nevts, log_bins, evts_box_text)
    axes_z[1, 1].set_xlabel("Total Events per Run")
    axes_z[1, 1].set_ylabel("Runs")

    fig_z.tight_layout()
    fig_z.savefig(output_path.parent / f"{output_path.stem}_zoomed.png", dpi=300, bbox_inches="tight")
    plt.close(fig_z)

    # 3. Individual plots
    if save_individual:
        stem = output_path.stem
        def _save_single(draw_func, args, kwargs, name, xlabel, legend_loc="upper right", legend_size=12, yscale="log"):
            fig, ax = plt.subplots(figsize=(9, 6.5))
            draw_func(ax, *args, **kwargs)
            ax.set_xlabel(xlabel, fontsize=15)
            ax.set_ylabel("Runs", fontsize=15)
            if legend_loc: ax.legend(loc=legend_loc, fontsize=legend_size)
            fig.tight_layout()
            fig.savefig(output_path.parent / f"{stem}_{name}.png", dpi=300, bbox_inches="tight")
            plt.close(fig)

        _save_single(_draw_rms_hist, [rms_vals, max_rms_pct, n_pass_rms, pct_pass_rms, n_fail_rms, pct_fail_rms], {"zoomed": False, "title_font": 11.5, "pad": 0.38}, "rms", "Plateau RMS Non-Flatness [%]")
        _save_single(_draw_rms_hist, [rms_vals, max_rms_pct, n_pass_rms, pct_pass_rms, n_fail_rms, pct_fail_rms], {"zoomed": True, "title_font": 11.5, "pad": 0.38}, "rms_zoomed", "Plateau RMS Non-Flatness [%]")
        _save_single(_draw_slope_hist, [slopes, max_slope_per_10pct, n_pass_slope, pct_pass_slope, n_fail_slope, pct_fail_slope], {"zoomed": False, "title_font": 11.5, "pad": 0.38}, "slope", "Plateau Slope [% per 10% Centrality]")
        _save_single(_draw_slope_hist, [slopes, max_slope_per_10pct, n_pass_slope, pct_pass_slope, n_fail_slope, pct_fail_slope], {"zoomed": True, "title_font": 11.5, "pad": 0.38}, "slope_zoomed", "Plateau Slope [% per 10% Centrality]")
        _save_single(_draw_ratios_hist, [r1s, r15s, max_central_spike, min_central_ratio, n_spike, pct_spike, n_drop, pct_drop], {"zoomed": False, "title_font": 11.0, "pad": 0.38}, "ratios", "Central Ratio to Plateau", legend_loc="upper left", legend_size=10.5)
        _save_single(_draw_ratios_hist, [r1s, r15s, max_central_spike, min_central_ratio, n_spike, pct_spike, n_drop, pct_drop], {"zoomed": True, "title_font": 11.0, "pad": 0.38}, "ratios_zoomed", "Central Ratio to Plateau", legend_loc="upper left", legend_size=10.5)
        _save_single(_draw_events_hist, [nevts, log_bins, evts_box_text], {"title_font": 11.5, "pad": 0.38}, "events", "Total Events per Run", legend_loc=None)

def plot_ensemble_profile(data_list, output_path, test_runs=None, subtitle=None):
    if hasattr(data_list, "to_dict"): data_list = data_list.to_dict("records")
    valid_data = [d for d in data_list if d.get("ratios") is not None and d.get("edges") is not None]
    if not valid_data: return

    hep.style.use("ATLAS")
    fig, ax = plt.subplots(figsize=(9.5, 7))

    ratio_matrix = np.array([d["ratios"][1:] for d in valid_data])
    cent_edges = valid_data[0]["edges"][1:]
    bin_centers = 0.5 * (cent_edges[:-1] + cent_edges[1:])

    p50, p16, p84, p02, p97 = (np.percentile(ratio_matrix, p, axis=0) for p in [50, 16, 84, 2.5, 97.5])

    ax.fill_between(bin_centers, p02, p97, color="#f9c74f", alpha=0.55, edgecolor="#d4ac0d", linewidth=0.8, label=r"95% ($2\sigma$) Population Envelope")
    ax.fill_between(bin_centers, p16, p84, color="#2ca02c", alpha=0.65, edgecolor="#1e8449", linewidth=0.8, label=r"68% ($1\sigma$) Population Envelope")
    ax.plot(bin_centers, p50, color="black", linewidth=2.5, label="Ensemble Median")
    ax.axhline(1.0, color="dimgray", linestyle="--", linewidth=1.5, alpha=0.85, label="Flat Baseline (1.0)")

    ax.set_xlim(0, 100)
    ax.set_ylim(0.0, 1.6)
    ax.set_xlabel("Centrality [%]", fontsize=15)
    ax.set_ylabel("Ratio to Plateau Average", fontsize=15)
    ax.grid(True, linestyle="--", alpha=0.35, which="both")

    handles, labels = ax.get_legend_handles_labels()
    desired_order = ["Ensemble Median", r"68% ($1\sigma$) Population Envelope", r"95% ($2\sigma$) Population Envelope", "Flat Baseline (1.0)"]
    handle_dict = dict(zip(labels, handles))
    ax.legend([handle_dict[l] for l in desired_order if l in handle_dict], [l for l in desired_order if l in handle_dict], loc="upper right", fontsize=12, frameon=True, framealpha=0.92)

    if subtitle:
        ax.text(0.04, 0.94, subtitle, transform=ax.transAxes, fontsize=13, fontweight="bold", va="top", ha="left",
                bbox=dict(boxstyle="round,pad=0.45", facecolor="white", edgecolor="#aaaaaa", alpha=0.92))

    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_centrality_1d_diagnostic(metric, title_suffix, output_path, cent_flat_min=10.0, cent_flat_max=70.0, is_good=None, rank=None):
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
        except Exception:
            pass

    if values is None or edges is None: return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9.5, 8.5), sharex=True, gridspec_kw={"height_ratios": [2.5, 1.8]})

    bin_centers = metric.get("bin_centers", 0.5 * (edges[:-1] + edges[1:]))
    plateau_avg = metric.get("plateau_avg")
    if plateau_avg is None: plateau_avg = compute_centrality_average(values, edges, cent_min=cent_flat_min, cent_max=cent_flat_max)
    ratios = metric.get("ratios", np.where(values > 0, values / plateau_avg, 0.0) if plateau_avg > 0 else np.zeros(len(values)))
    total_events = metric.get("total_events", float(np.sum(values)))

    # Panel 1
    hep.histplot((values, edges), ax=ax1, histtype="step", color="black", linewidth=2.0, label="Centrality Distribution")
    if plateau_avg > 0:
        ax1.axhline(plateau_avg, color="crimson", linestyle="--", linewidth=1.8, label=rf"Plateau Avg ({plateau_avg:.2e})")
        ax1.axvspan(cent_flat_min, cent_flat_max, color="forestgreen", alpha=0.08, label=f"Plateau [{cent_flat_min:.0f}%, {cent_flat_max:.0f}%]")

    ax1.set_ylabel("Events", fontsize=16)
    ax1.set_xlim(0, 100)
    ax1.tick_params(axis="both", labelsize=13)

    plat_mask = (bin_centers >= cent_flat_min) & (bin_centers <= cent_flat_max)
    plat_max = float(np.max(values[plat_mask])) if np.any(plat_mask) else plateau_avg
    overall_max = float(np.max(values)) if len(values) > 0 else 0.0
    ax1.set_ylim(0, max(overall_max * 1.15, plat_max * 1.40 if plat_max > 0 else 1.0) if overall_max > 0 else 1.0)
    ax1.legend(loc="upper right", fontsize=11.5, frameon=True, framealpha=0.92)
    ax1.grid(True, linestyle="--", alpha=0.3)

    is_good = is_good if is_good is not None else (metric.get("status") == "GOOD")
    if is_good:
        rank_tag = f"Top Flat Example #{rank}" if rank is not None else (title_suffix or "GOOD Run")
        info_text = f"Run {run_number} | {rank_tag}\nTotal Events: {total_events:.2e}\nStatus: GOOD (Passed All QA)"
        box_edge = "forestgreen"
    else:
        flags_list = [f for f in str(metric.get("status", "")).split(";") if f]
        if len(flags_list) > 2: flags_text = "Flags: " + "; ".join(flags_list[:2]) + ";\n       " + "; ".join(flags_list[2:])
        elif len(str(metric.get("status", ""))) > 32 and len(flags_list) > 1: flags_text = "Flags: " + flags_list[0] + ";\n       " + "; ".join(flags_list[1:])
        else: flags_text = f"Flags: {metric.get('status')}" if flags_list else "Flags: Flagged Outlier"

        tag = f"Example: {title_suffix}" if title_suffix else "Outlier Run"
        info_text = f"Run {run_number} | {tag}\nTotal Events: {total_events:.2e}\n{flags_text}"
        box_edge = "crimson"

    ax1.text(0.12, 0.94, info_text, transform=ax1.transAxes, va="top", ha="left", fontsize=12.0, fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.45", facecolor="white", edgecolor=box_edge, linewidth=1.6, alpha=0.92))

    # Panel 2
    r_slice = ratios[1:]
    hep.histplot((r_slice, edges[1:]), ax=ax2, histtype="step", color="royalblue", linewidth=2.0, label="Ratio to Plateau")
    ax2.axhline(1.0, color="gray", linestyle="-", linewidth=1.2)
    ax2.axvspan(cent_flat_min, cent_flat_max, color="forestgreen", alpha=0.08)

    pos_r = r_slice[r_slice > 0]
    if len(pos_r) > 0:
        r_min, overall_r_max = float(np.min(pos_r)), float(np.max(pos_r))
        y2_min = max(0.0, r_min - max(0.04, 0.05 * (overall_r_max - r_min)))
        if overall_r_max <= 1.15: y2_max = 1.25 if y2_min >= 0.40 else 1.46
        elif overall_r_max <= 1.30: y2_max = max(1.35, overall_r_max * 1.18)
        else: y2_max = max(1.48, min(3.5, overall_r_max * 1.18))
    else: y2_min, y2_max = 0.0, 1.25
    ax2.set_ylim(y2_min, y2_max)

    if y2_max >= 1.28: ax2.axhline(1.30, color="crimson", linestyle=":", linewidth=1.3, label=r"Spike Cut ($R_{1\%} > 1.30$)")
    if y2_min <= 0.82: ax2.axhline(0.80, color="darkorange", linestyle=":", linewidth=1.3, label=r"Drop Cut ($R_{1-5\%} < 0.80$)")

    def ffmt(k, fmt=":.2f", sfx="%"):
        v = metric.get(k, np.nan)
        return (f"{{{fmt}}}").format(v)+sfx if not np.isnan(v) else "N/A"

    ratio_metrics_text = (
        rf"Plateau RMS: {ffmt('rms_plat_pct')} (Cut: 4.5%)" + "\n" +
        rf"Plateau Slope: {ffmt('slope_per_10pct', ':+.2f')}/10% cent (Tol: $\pm 2.5\%$)" + "\n" +
        rf"$R_{{1\%}}$: {ffmt('ratio_1', ':.3f', '')} | $R_{{1-5\%}}$: {ffmt('ratio_1_5', ':.3f', '')}"
    )
    ax2.text(0.12, 0.94, ratio_metrics_text, transform=ax2.transAxes, va="top", ha="left", fontsize=10.5,
             bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor=box_edge if is_good else "gray", linewidth=1.4 if is_good else 1.0, alpha=0.92))

    ax2.set_xlabel("Centrality [%]", fontsize=16)
    ax2.set_ylabel("Ratio to Plateau", fontsize=16)
    ax2.set_xlim(0, 100)
    ax2.tick_params(axis="both", labelsize=13)
    ax2.legend(loc="upper right", fontsize=10.5, frameon=True, framealpha=0.92)
    ax2.grid(True, linestyle="--", alpha=0.3)

    fig.subplots_adjust(top=0.96, bottom=0.10, left=0.12, right=0.96, hspace=0.08)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)

def plot_failure_example(metric, failure_mode, output_path, cent_flat_min=10.0, cent_flat_max=70.0):
    return plot_centrality_1d_diagnostic(metric, failure_mode, output_path, cent_flat_min, cent_flat_max, False)

def plot_top_flat_example(metric, rank, output_path, cent_flat_min=10.0, cent_flat_max=70.0):
    return plot_centrality_1d_diagnostic(metric, f"Top Flat Example #{rank}", output_path, cent_flat_min, cent_flat_max, True, rank)

def plot_user_example(metric, output_path, cent_flat_min=10.0, cent_flat_max=70.0, title_suffix="User Specified"):
    return plot_centrality_1d_diagnostic(metric, title_suffix, output_path, cent_flat_min, cent_flat_max, metric.get("status") == "GOOD")

def plot_failure_mode_metric_distribution(metrics_list, failure_mode, example_runs, output_path,
                                          max_rms_pct=4.5, max_dev_pct=8.0, max_slope_per_10pct=2.5,
                                          max_central_spike=1.30, min_central_ratio=0.80):
    if hasattr(metrics_list, "to_dict"): metrics_list = metrics_list.to_dict("records")
    if not example_runs: return
    run_dict = {d.get("run_number", d.get("run")): d for d in metrics_list}
    example_dicts = [run_dict[r] for r in example_runs if r in run_dict]
    if not example_dicts: return

    output_path = Path(output_path)
    if output_path.is_dir() or output_path.suffix == "": output_path = output_path / "centrality_metric_distributions.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    zoomed_path = output_path.parent / "centrality_metric_distributions_zoomed.png"

    line_styles = [{"color": "#800080", "ls": "-.", "lw": 2.0}, {"color": "#d95f02", "ls": ":", "lw": 2.4},
                   {"color": "#1b9e77", "ls": "--", "lw": 2.0}, {"color": "#e7298a", "ls": "-", "lw": 2.0}]

    def draw_lines(ax, keys, label_fmt, min_val=None):
        for idx, d in enumerate(example_dicts):
            st = line_styles[idx % len(line_styles)]
            val = max([d.get(k, 0.0) for k in keys])
            line_val = max(min_val, val) if min_val is not None else val
            ax.axvline(line_val, color=st["color"], linestyle=st["ls"], linewidth=st["lw"], label=label_fmt.format(d.get('run_number', d.get('run')), val))

    def make_plot(path, data1, data2, bins, c1, c2, l1, l2, cuts, xlabel, text, zoomed, clip_range=None, is_log_x=False):
        fig, ax = plt.subplots(figsize=(9, 6.5))
        if clip_range:
            data1 = np.clip(data1, *clip_range)
            if data2 is not None: data2 = np.clip(data2, *clip_range)
            if isinstance(bins, int):
                bins = np.linspace(clip_range[0], clip_range[1], bins)
        ax.hist(data1, bins=bins, histtype="step", color=c1, linewidth=2.0, label=l1)
        if data2 is not None: ax.hist(data2, bins=bins, histtype="step", color=c2, linewidth=1.4, alpha=0.5, label=l2)

        for cval, cc, cls, clw, clab in cuts: ax.axvline(cval, color=cc, linestyle=cls, linewidth=clw, label=clab)

        if failure_mode == "FLAG_NON_FLAT":
            draw_lines(ax, ["rms_plat_pct"], "Example Run {} (RMS = {:.2f}%)")
        elif failure_mode == "FLAG_SLOPE_DRIFT":
            if not zoomed:
                draw_lines(ax, ["slope_per_10pct"], "Example Run {} (Slope = {:+.2f}%)")
        elif failure_mode == "FLAG_CENTRAL_DROP":
            draw_lines(ax, ["ratio_1_5"], r"Example Run {} ($R_{{1-5\%}} = {:.3f}$)")
        elif failure_mode == "FLAG_CENTRAL_SPIKE":
            if not zoomed:
                draw_lines(ax, ["ratio_1"], r"Example Run {} ($R_{{1\%}} = {:.3f}$)")
        elif failure_mode == "FLAG_EMPTY_OR_ZERO":
            draw_lines(ax, ["total_events"], "Example Run {} (Events = {:.2e})", min_val=1.0)

        ax.set_xlabel(xlabel, fontsize=15)
        ax.set_ylabel("Runs", fontsize=15)
        if is_log_x: ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylim(0.5, 4500 if "CENTRAL" in failure_mode else 3500)
        if clip_range:
            ax.set_xlim(clip_range[0], clip_range[1])
        elif not is_log_x and hasattr(bins, "__getitem__"):
            ax.set_xlim(bins[0], bins[-1])
        ax.legend(loc="upper left" if is_log_x else "upper right", fontsize=11 if not "CENTRAL" in failure_mode else 10.5, frameon=True, framealpha=0.92)
        ax.grid(True, linestyle="--", alpha=0.3, which="both")
        if text:
            ax.text(0.04, 0.94, text, transform=ax.transAxes, ha="left", va="top", fontsize=11.0,
                    bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="gray", alpha=0.92, zorder=4))
        fig.tight_layout()
        fig.savefig(path, dpi=300, bbox_inches="tight")
        # Save both singular and plural forms for compatibility
        if "distributions" in path.name:
            alt_p = path.parent / path.name.replace("distributions", "distribution")
            fig.savefig(alt_p, dpi=300, bbox_inches="tight")
        elif "distribution" in path.name:
            alt_p = path.parent / path.name.replace("distribution", "distributions")
            fig.savefig(alt_p, dpi=300, bbox_inches="tight")
        plt.close(fig)

    sel_str = f"Selected Runs: {', '.join(map(str, example_runs))}"

    if failure_mode == "FLAG_NON_FLAT":
        rms = [float(d["rms_plat_pct"]) for d in metrics_list if not np.isnan(d.get("rms_plat_pct", np.nan))]
        c = [(max_rms_pct, "crimson", "--", 2.2, f"Flatness Cut ({max_rms_pct}%)")]
        txt = r"$\bf{FLAG\_NON\_FLAT\ (Unzoomed)}$" + f"\nCut: RMS > {max_rms_pct}% (or MaxDev > {max_dev_pct}%)\nPopulation Max: {max(rms):.1f}%\n" + sel_str if rms else sel_str
        make_plot(output_path, rms, None, np.linspace(0, max(12.0, max(rms)*1.08 if rms else 12.0), 55), "steelblue", None, "All Runs Distribution", None, c, "Plateau RMS Non-Flatness [%]", txt, False)
        txt_z = r"$\bf{FLAG\_NON\_FLAT\ (Zoomed)}$" + f"\nCut: RMS > {max_rms_pct}%\n" + sel_str
        make_plot(zoomed_path, rms, None, 45, "steelblue", None, "All Runs (Clipped at 10%)", None, c, "Plateau RMS Non-Flatness [%]", txt_z, True, [0, 10])

    elif failure_mode == "FLAG_SLOPE_DRIFT":
        slp = [float(d["slope_per_10pct"]) for d in metrics_list if not np.isnan(d.get("slope_per_10pct", np.nan))]
        c = [(max_slope_per_10pct, "crimson", "--", 2.2, rf"Tolerance Cut ($\pm${max_slope_per_10pct}%)"), (-max_slope_per_10pct, "crimson", "--", 2.2, None)]
        txt = r"$\bf{FLAG\_SLOPE\_DRIFT\ (Unzoomed)}$" + f"\nTolerance: |Slope| > {max_slope_per_10pct}%\nPopulation Max: {max(slp):+.1f}%\n" + sel_str if slp else sel_str
        make_plot(output_path, slp, None, np.linspace(min(-5.0, min(slp)*1.2 if slp else -5.0), max(6.0, max(slp)*1.15 if slp else 6.0), 60), "mediumseagreen", None, "All Runs Distribution", None, c, "Plateau Slope [% per 10% Centrality]", txt, False)
        txt_z = r"$\bf{FLAG\_SLOPE\_DRIFT\ (Zoomed\ Core)}$" + f"\nTolerance: |Slope| > {max_slope_per_10pct}%\nExample runs visible on unzoomed plot"
        make_plot(zoomed_path, slp, None, 40, "mediumseagreen", None, r"All Runs (Bulk in $\pm$6%)", None, c, "Plateau Slope [% per 10% Centrality]", txt_z, True, [-6, 6])

    elif failure_mode == "FLAG_CENTRAL_DROP":
        r15s = [float(d["ratio_1_5"]) for d in metrics_list if not np.isnan(d.get("ratio_1_5", np.nan))]
        r1s = [float(d["ratio_1"]) for d in metrics_list if not np.isnan(d.get("ratio_1", np.nan))]
        c = [(1.0, "gray", "-", 1.2, None), (min_central_ratio, "darkorange", "--", 2.2, rf"Drop Cut ($R_{{1-5\%}} < {min_central_ratio}$)") ]
        txt = r"$\bf{FLAG\_CENTRAL\_DROP\ (Unzoomed)}$" + rf"\nDrop Cut: $R_{{1-5\%}} < {min_central_ratio}$" + "\n" + sel_str
        make_plot(output_path, r15s, r1s, np.linspace(0, max(2.5, max(r1s)*1.08 if r1s else 2.5), 55), "royalblue", "crimson", r"$R_{1-5\%}$ Distribution", r"$R_{1\%}$ Distribution", c, "Central Ratio to Plateau", txt, False)
        txt_z = r"$\bf{FLAG\_CENTRAL\_DROP\ (Zoomed)}$" + rf"\nDrop Cut: $R_{{1-5\%}} < {min_central_ratio}$" + "\n" + sel_str
        make_plot(zoomed_path, r15s, r1s, np.linspace(0, 2.2, 45), "royalblue", "crimson", r"$R_{1-5\%}$ Distribution", r"$R_{1\%}$ Distribution", c, "Central Ratio to Plateau", txt_z, True, [0, 2.2])

    elif failure_mode == "FLAG_CENTRAL_SPIKE":
        r1s = [float(d["ratio_1"]) for d in metrics_list if not np.isnan(d.get("ratio_1", np.nan))]
        r15s = [float(d["ratio_1_5"]) for d in metrics_list if not np.isnan(d.get("ratio_1_5", np.nan))]
        c = [(1.0, "gray", "-", 1.2, None), (max_central_spike, "crimson", "--", 2.2, rf"Spike Cut ($R_{{1\%}} > {max_central_spike}$)") ]
        txt = r"$\bf{FLAG\_CENTRAL\_SPIKE\ (Unzoomed)}$" + rf"\nSpike Cut: $R_{{1\%}} > {max_central_spike}$" + "\n" + sel_str
        make_plot(output_path, r1s, r15s, np.linspace(0, max(2.5, max(r1s)*1.08 if r1s else 2.5), 55), "crimson", "royalblue", r"$R_{1\%}$ Distribution", r"$R_{1-5\%}$ Distribution", c, "Central Ratio to Plateau", txt, False)
        txt_z = r"$\bf{FLAG\_CENTRAL\_SPIKE\ (Zoomed\ Core)}$" + rf"\nSpike Cut: $R_{{1\%}} > {max_central_spike}$" + "\nExample runs visible on unzoomed plot"
        make_plot(zoomed_path, r1s, r15s, np.linspace(0, 2.5, 45), "crimson", "royalblue", r"$R_{1\%}$ Distribution", r"$R_{1-5\%}$ Distribution", c, "Central Ratio to Plateau", txt_z, True, [0, 2.5])

    elif failure_mode == "FLAG_EMPTY_OR_ZERO":
        nevts = [float(d["total_events"]) for d in metrics_list if not np.isnan(d.get("total_events", np.nan))]
        max_ev = max(nevts) if (nevts and max(nevts) > 1.0) else 1e7
        make_plot(output_path, nevts, None, np.logspace(0, np.log10(max_ev), 40), "coral", None, "All Runs Distribution", None, [], "Total Events per Run", None, False, is_log_x=True)
