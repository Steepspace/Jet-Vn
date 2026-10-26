from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mplhep as hep
from matplotlib.ticker import MaxNLocator, MultipleLocator
from centrality_qa_metrics import N_CENT_CLASSES, R1_CUTS, R1_5_CUTS, describe_selection


def _stamp_selection(fig, data, pad=8, grid=False):
    """
    Title `fig` with the event selection of `data` (a metric dict or a list of them), just above the
    top edge of the plot box. For a grid of panels it is centred over the whole grid instead of over
    the first panel.
    """
    d = data[0] if isinstance(data, list) and data else data
    hist_name = d.get("hist_name") if isinstance(d, dict) else None
    if not hist_name or not fig.axes:
        return
    text = describe_selection(hist_name)
    if grid:
        fig.suptitle(text, fontsize=17, color="dimgray", y=0.995)
    else:
        fig.axes[0].set_title(text, fontsize=17, pad=pad, color="dimgray")


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
    cbar.set_label("Ratio to Flat Expectation", fontsize=15)
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
    _stamp_selection(fig, data_list, pad=26 if test_runs else 8)  # lift it over the run labels
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _snap_axis(ax, lo, hi, n_ticks=8):
    """
    Set the x-axis to 'nice' major ticks covering [lo, hi] and return histogram bin edges with
    two bins per major tick, so every bin edge sits on a major or minor tick.
    """
    ticks = MaxNLocator(nbins=n_ticks, steps=[1, 2, 5, 10]).tick_values(lo, hi)
    step = float(ticks[1] - ticks[0])
    width = step / 2.0
    lo_s = np.floor(lo / step + 1e-9) * step
    hi_s = np.ceil(hi / step - 1e-9) * step
    ax.xaxis.set_major_locator(MultipleLocator(step))
    ax.xaxis.set_minor_locator(MultipleLocator(width))
    ax.set_xlim(lo_s, hi_s)
    return np.round(np.arange(lo_s, hi_s + width / 2.0, width), 10)


def _log_tick_bins(lo, hi):
    """Bin edges at 1-9 x 10^k, i.e. on the major and minor ticks of a log axis covering [lo, hi]."""
    k_lo, k_hi = int(np.floor(np.log10(max(lo, 1.0)))), int(np.ceil(np.log10(max(hi, 1.0))))
    edges = np.array([m * 10.0 ** k for k in range(k_lo, k_hi) for m in range(1, 10)] + [10.0 ** k_hi])
    return edges


def _draw_class_dev_scatter(ax, runs, devs, is_good, is_outlier, is_test, max_class_dev, s_good=30, s_bad=45):
    ax.scatter(runs[is_good], devs[is_good], color="forestgreen", s=s_good, alpha=0.7, label="Good Runs", edgecolors="none")
    ax.scatter(runs[is_outlier], devs[is_outlier], color="crimson", s=s_bad, alpha=0.85, marker="^", label="Flagged Runs")
    if np.any(is_test):
        ax.scatter(runs[is_test], devs[is_test], facecolors="none", edgecolors="black", s=140, linewidths=2.0, marker="o", label="Test Runs")
        for r, val in zip(runs[is_test], devs[is_test]):
            if not np.isnan(val):
                ax.annotate(f"{r}", (r, val), textcoords="offset points", xytext=(0, 10), ha="center", fontsize=11, fontweight="bold")
    ax.axhline(max_class_dev, color="crimson", linestyle="--", linewidth=1.8, label=rf"Cut ({max_class_dev:g}%)")
    ax.set_ylim(bottom=0.0)
    ax.grid(True, linestyle="--", alpha=0.4)

def _draw_ratios_scatter(ax, runs, r1s, r15s, r80s, r1_cuts, r1_5_cuts):
    # Each central ratio is drawn with its own allowed band in the matching colour
    ax.axhspan(*r1_cuts, color="crimson", alpha=0.06)
    ax.axhspan(*r1_5_cuts, color="royalblue", alpha=0.10)
    for c in r1_cuts: ax.axhline(c, color="crimson", linestyle="--", linewidth=1.4)
    for c in r1_5_cuts: ax.axhline(c, color="royalblue", linestyle=":", linewidth=1.6)
    ax.scatter(runs, r1s, color="crimson", s=35, alpha=0.65, marker="^", label=rf"$R_{{1\%}}$ (cuts {r1_cuts[0]:g}, {r1_cuts[1]:g})")
    ax.scatter(runs, r15s, color="royalblue", s=30, alpha=0.6, marker="o", label=rf"$R_{{1-5\%}}$ (cuts {r1_5_cuts[0]:g}, {r1_5_cuts[1]:g})")
    ax.scatter(runs, r80s, color="darkorange", s=30, alpha=0.6, marker="s", label=r"$R_{80\%}$ (Peripheral $\sim$80%)")
    ax.axhline(1.0, color="gray", linestyle="-", linewidth=1.2)
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


def plot_trends(data_list, output_path, test_runs=None, max_class_dev=3.5,
                r1_cuts=R1_CUTS, r1_5_cuts=R1_5_CUTS, flat_max=80.0, save_individual=True):
    hep.style.use("ATLAS")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    runs = np.array([d["run_number"] for d in data_list])
    devs = np.array([d["class_dev_max"] for d in data_list])
    r1s = np.array([d["ratio_1"] for d in data_list])
    r15s = np.array([d["ratio_1_5"] for d in data_list])
    r80s = np.array([d["ratio_80"] for d in data_list])
    nevts = np.array([d["total_events"] for d in data_list])
    statuses = [d["status"] for d in data_list]

    # Low-statistics runs skip the shape checks, so they appear only in the events panel
    low_stats = np.array(["FLAG_LOW_STATS" in s for s in statuses])
    for arr in (devs, r1s, r15s, r80s):
        arr[low_stats] = np.nan

    is_good = np.array([s == "GOOD" for s in statuses])
    is_outlier = ~is_good
    test_run_set = set(test_runs or [])
    is_test = np.array([r in test_run_set for r in runs])

    fig, axes = plt.subplots(3, 1, figsize=(13, 12.5), sharex=True)

    _draw_class_dev_scatter(axes[0], runs, devs, is_good, is_outlier, is_test, max_class_dev)
    axes[0].set_ylabel(f"Worst 20% Class\nDeviation [%]", fontsize=13)
    axes[0].legend(loc="upper right", ncol=4, fontsize=11, frameon=True, framealpha=0.9)

    _draw_ratios_scatter(axes[1], runs, r1s, r15s, r80s, r1_cuts, r1_5_cuts)
    axes[1].set_ylabel("Ratio to Flat", fontsize=14)
    axes[1].legend(loc="upper right", ncol=3, fontsize=11, frameon=True, framealpha=0.9)

    _draw_events_scatter(axes[2], runs, nevts, is_good, is_outlier, is_test)
    axes[2].set_ylabel("Total Events", fontsize=14)
    axes[2].set_xlabel("Run Number", loc="center", fontsize=15)

    fig.tight_layout()
    _stamp_selection(fig, data_list)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    if save_individual:
        stem = output_path.stem
        fig_dev, ax_dev = plt.subplots(figsize=(11, 6.5))
        _draw_class_dev_scatter(ax_dev, runs, devs, is_good, is_outlier, is_test, max_class_dev, s_good=32, s_bad=48)
        ax_dev.set_xlabel("Run Number", loc="center", fontsize=15)
        ax_dev.set_ylabel(f"Worst 20% Class Deviation from Flat [%] (1–{flat_max:.0f}%)", fontsize=14)
        ax_dev.legend(loc="upper right", ncol=4, fontsize=11, frameon=True, framealpha=0.9)
        fig_dev.tight_layout()
        _stamp_selection(fig_dev, data_list)
        fig_dev.savefig(output_path.parent / f"{stem}_class_dev.png", dpi=300, bbox_inches="tight")
        plt.close(fig_dev)

        fig_rat, ax_rat = plt.subplots(figsize=(11, 6.5))
        _draw_ratios_scatter(ax_rat, runs, r1s, r15s, r80s, r1_cuts, r1_5_cuts)
        ax_rat.set_xlabel("Run Number", loc="center", fontsize=15)
        ax_rat.set_ylabel("Ratio to Flat Expectation", fontsize=14)
        ax_rat.legend(loc="upper right", ncol=3, fontsize=11, frameon=True, framealpha=0.9)
        fig_rat.tight_layout()
        _stamp_selection(fig_rat, data_list)
        fig_rat.savefig(output_path.parent / f"{stem}_ratios.png", dpi=300, bbox_inches="tight")
        plt.close(fig_rat)

        fig_ev, ax_ev = plt.subplots(figsize=(11, 6.5))
        _draw_events_scatter(ax_ev, runs, nevts, is_good, is_outlier, is_test, s_good=32, s_bad=48, label_runs=True)
        ax_ev.set_xlabel("Run Number", loc="center", fontsize=15)
        ax_ev.set_ylabel("Total Events", fontsize=14)
        ax_ev.legend(loc="lower right", fontsize=11, frameon=True, framealpha=0.9)
        fig_ev.tight_layout()
        _stamp_selection(fig_ev, data_list)
        fig_ev.savefig(output_path.parent / f"{stem}_events.png", dpi=300, bbox_inches="tight")
        plt.close(fig_ev)


def plot_trigger_exclusive(data_list, output_path, z_max=60.0):
    """
    MB events with |z| < z_max summed over all runs, split into Trig 12 only, Trig 14 only, both
    and neither, each labelled with its share of all MB events.
    """
    hep.style.use("ATLAS")
    data_list = [d for d in data_list if not np.isnan(d.get("n_mb", np.nan))]
    if not data_list:
        print("Warning: No runs with hZVertex / hZVertex_*_MB histograms; skipping trigger exclusive plot.")
        return

    n12 = sum(d["n_trig12_only"] for d in data_list)
    n14 = sum(d["n_trig14_only"] for d in data_list)
    n_or = sum(d["n_trig_or"] for d in data_list)
    n_mb = sum(d["n_mb"] for d in data_list)
    counts = [n12, n14, n_or - n12 - n14, n_mb - n_or]
    labels = ["Trig 12 only", "Trig 14 only", "Trig 12 & Trig 14", "Neither"]
    colors = ["royalblue", "darkorange", "forestgreen", "gray"]

    fig, ax = plt.subplots(figsize=(11, 6.5))
    bars = ax.bar(labels, counts, color=colors, alpha=0.8, width=0.6)
    for bar, n in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, n, f"{n / n_mb:.1%}\n({n:.3e})",
                ha="center", va="bottom", fontsize=13)

    ax.set_xlim(-0.6, len(counts) - 0.4)
    ax.set_ylim(0, 1.2 * max(counts))
    ax.set_ylabel("MB Events", fontsize=15)
    ax.set_title(rf"MB, |z| < {z_max:g} cm, {len(data_list)} runs (Total = {n_mb:.3e})", fontsize=13)
    ax.tick_params(axis="x", which="both", top=False, bottom=False, labelsize=14)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)

    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _ratio_range(vals, cuts):
    """x-range for a central-ratio histogram: all values and both cuts, with a little margin."""
    lo, hi = min(min(vals), cuts[0]), max(max(vals), cuts[1])
    pad = 0.05 * (hi - lo)
    return lo - pad, hi + pad


def _draw_ratio_hist(ax, vals, cuts, color, symbol, title_font=10.5, pad=0.35):
    """One central ratio with its low and high cuts, and pass / low-fail / high-fail counts."""
    if len(vals) > 0:
        vals = np.asarray(vals)
        bins = _snap_axis(ax, *_ratio_range(vals, cuts), n_ticks=15)
        n_low, n_high = int(np.sum(vals < cuts[0])), int(np.sum(vals > cuts[1]))
        n_pass = len(vals) - n_low - n_high

        ax.hist(vals, bins=bins, histtype="step", color=color, linewidth=2.0, label=rf"${symbol}$")
        ax.axvline(1.0, color="gray", linestyle="-", linewidth=1.5)
        ax.axvline(cuts[0], color="crimson", linestyle="--", linewidth=2, label=f"Low Cut ({cuts[0]:g})")
        ax.axvline(cuts[1], color="darkorange", linestyle="--", linewidth=2, label=f"High Cut ({cuts[1]:g})")
        box_text = (
            rf"Pass: {n_pass:,} ({100.0 * n_pass / len(vals):.1f}%)" + "\n" +
            rf"Low (< {cuts[0]:g}): {n_low:,} ({100.0 * n_low / len(vals):.1f}%)" + "\n" +
            rf"High (> {cuts[1]:g}): {n_high:,} ({100.0 * n_high / len(vals):.1f}%)"
        )
        ax.text(0.96, 0.94, box_text, transform=ax.transAxes, ha="right", va="top", fontsize=title_font,
                bbox=dict(boxstyle=f"round,pad={pad}", facecolor="white", edgecolor="gray", alpha=0.92, zorder=4))
    ax.set_yscale("log")
    ax.set_ylim(0.5, 4500)
    ax.grid(True, linestyle="--", alpha=0.3, which="both")

def _draw_cut_hist(ax, vals, cut, color, x_max, unit, zoomed=False, title_font=10.0, pad=0.35, n_ticks=18):
    """Distribution of a one-sided QA metric with its flagging cut."""
    if len(vals) > 0:
        vals = np.asarray(vals)
        n_fail = int(np.sum(vals > cut))
        n_pass = len(vals) - n_fail
        if zoomed:
            extra_txt = f"\n(Clipped at {x_max:g}{unit})"
        else:
            x_max = max(x_max, float(np.max(vals)))
            extra_txt = f"\nMax: {np.max(vals):.2f}{unit}"
        bins = _snap_axis(ax, 0.0, x_max, n_ticks=n_ticks)
        data = np.clip(vals, 0, bins[-1] - 1e-9) if zoomed else vals

        ax.hist(data, bins=bins, histtype="step", color=color, linewidth=2.0)
        ax.axvline(cut, color="crimson", linestyle="--", linewidth=2, label=f"Cut ({cut:g}{unit})")
        box_text = (
            rf"Pass ($\leq {cut:g}${unit}): {n_pass:,} ({100.0 * n_pass / len(vals):.1f}%)" + "\n" +
            rf"Fail (> {cut:g}{unit}): {n_fail:,} ({100.0 * n_fail / len(vals):.1f}%)" + extra_txt
        )
        ax.text(0.96, 0.78, box_text, transform=ax.transAxes, ha="right", va="top", fontsize=title_font,
                bbox=dict(boxstyle=f"round,pad={pad}", facecolor="white", edgecolor="gray", alpha=0.92, zorder=4))
    ax.set_yscale("log")
    ax.set_ylim(0.5, 3500)
    ax.grid(True, linestyle="--", alpha=0.3, which="both")

def _draw_events_hist(ax, nevts, evts_box_text, title_font=10.5, pad=0.35):
    if len(nevts) > 0:
        ax.hist(nevts, bins=_log_tick_bins(min(nevts), max(nevts)), histtype="step", color="coral", linewidth=2.0)
        ax.text(0.04, 0.94, evts_box_text, transform=ax.transAxes, ha="left", va="top", fontsize=title_font,
                bbox=dict(boxstyle=f"round,pad={pad}", facecolor="white", edgecolor="gray", alpha=0.92, zorder=4))
        ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(0.5, 3500)
    ax.grid(True, linestyle="--", alpha=0.3, which="both")


def plot_metric_distributions(data_list, output_path, max_class_dev=3.5, r1_cuts=R1_CUTS, r1_5_cuts=R1_5_CUTS,
                              save_individual=True):
    hep.style.use("ATLAS")
    if hasattr(data_list, "to_dict"): data_list = data_list.to_dict("records")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Low-statistics runs skip the shape checks, so they are left out of the shape metrics to match the flag counts
    judged = [d for d in data_list if "FLAG_LOW_STATS" not in d["status"]]
    devs = [d["class_dev_max"] for d in judged if not np.isnan(d["class_dev_max"])]
    r1s = [d["ratio_1"] for d in judged if not np.isnan(d["ratio_1"])]
    r15s = [d["ratio_1_5"] for d in judged if not np.isnan(d["ratio_1_5"])]
    nevts = [d["total_events"] for d in data_list if d["total_events"] > 0]

    evts_box_text = ""
    if nevts:
        evts_box_text = (
            rf"Total Runs: {len(nevts):,}" + "\n" +
            rf"Median: {float(np.median(nevts)):.2e} evts/run" + "\n" +
            rf"Total: {float(np.sum(nevts)):.2e} events"
        )

    dev_label = "Worst 20% Class Deviation from Flat [%]"
    r1_label = r"Ratio to Flat in the 1% Bin, $R_{1\%}$"
    r15_label = r"Mean Ratio to Flat in the 1-5% Bins, $R_{1-5\%}$"

    # 1. Unzoomed / zoomed 4-panel (the zoomed version clips the class-deviation outliers)
    for zoomed, path in [(False, output_path), (True, output_path.parent / f"{output_path.stem}_zoomed.png")]:
        fig, axes = plt.subplots(2, 2, figsize=(15, 10.5))
        _draw_cut_hist(axes[0, 0], devs, max_class_dev, "darkviolet", 12.0, "%", zoomed=zoomed)
        axes[0, 0].set_xlabel(dev_label)
        axes[0, 0].legend(loc="upper right", fontsize=10.5)

        _draw_ratio_hist(axes[0, 1], r1s, r1_cuts, "crimson", r"R_{1\%}")
        axes[0, 1].set_xlabel(r1_label)
        axes[0, 1].legend(loc="upper left", fontsize=10, frameon=True, framealpha=0.9)

        _draw_ratio_hist(axes[1, 0], r15s, r1_5_cuts, "royalblue", r"R_{1-5\%}")
        axes[1, 0].set_xlabel(r15_label)
        axes[1, 0].legend(loc="upper left", fontsize=10, frameon=True, framealpha=0.9)

        _draw_events_hist(axes[1, 1], nevts, evts_box_text)
        axes[1, 1].set_xlabel("Total Events per Run")

        for ax in axes.flat: ax.set_ylabel("Runs")
        fig.tight_layout(rect=(0, 0, 1, 0.955))  # leave room for the selection title
        _stamp_selection(fig, data_list, grid=True)
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)

    # 2. Individual plots
    if save_individual:
        stem = output_path.stem
        def _save_single(draw_func, args, kwargs, name, xlabel, legend_loc="upper right", legend_size=12):
            fig, ax = plt.subplots(figsize=(9, 6.5))
            draw_func(ax, *args, **kwargs)
            ax.set_xlabel(xlabel, fontsize=15)
            ax.set_ylabel("Runs", fontsize=15)
            if legend_loc: ax.legend(loc=legend_loc, fontsize=legend_size)
            fig.tight_layout()
            _stamp_selection(fig, data_list)
            fig.savefig(output_path.parent / f"{stem}_{name}.png", dpi=300, bbox_inches="tight")
            plt.close(fig)

        _save_single(_draw_cut_hist, [devs, max_class_dev, "darkviolet", 12.0, "%"], {"zoomed": False, "title_font": 11.5, "pad": 0.38}, "class_dev", dev_label)
        _save_single(_draw_cut_hist, [devs, max_class_dev, "darkviolet", 12.0, "%"], {"zoomed": True, "title_font": 11.5, "pad": 0.38}, "class_dev_zoomed", dev_label)
        _save_single(_draw_ratio_hist, [r1s, r1_cuts, "crimson", r"R_{1\%}"], {"title_font": 11.5, "pad": 0.38}, "ratio_1", r1_label, legend_loc="upper left", legend_size=11)
        _save_single(_draw_ratio_hist, [r15s, r1_5_cuts, "royalblue", r"R_{1-5\%}"], {"title_font": 11.5, "pad": 0.38}, "ratio_1_5", r15_label, legend_loc="upper left", legend_size=11)
        _save_single(_draw_events_hist, [nevts, evts_box_text], {"title_font": 11.5, "pad": 0.38}, "events", "Total Events per Run", legend_loc=None)

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
    ax.set_ylabel("Ratio to Flat Expectation", fontsize=15)
    ax.grid(True, linestyle="--", alpha=0.35, which="both")

    handles, labels = ax.get_legend_handles_labels()
    desired_order = ["Ensemble Median", r"68% ($1\sigma$) Population Envelope", r"95% ($2\sigma$) Population Envelope", "Flat Baseline (1.0)"]
    handle_dict = dict(zip(labels, handles))
    ax.legend([handle_dict[l] for l in desired_order if l in handle_dict], [l for l in desired_order if l in handle_dict], loc="upper right", fontsize=12, frameon=True, framealpha=0.92)

    if subtitle:
        ax.text(0.04, 0.94, subtitle, transform=ax.transAxes, fontsize=13, fontweight="bold", va="top", ha="left",
                bbox=dict(boxstyle="round,pad=0.45", facecolor="white", edgecolor="#aaaaaa", alpha=0.92))

    fig.tight_layout()
    _stamp_selection(fig, valid_data)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_centrality_1d_diagnostic(metric, title_suffix, output_path, flat_max=80.0, max_class_dev=3.5, is_good=None, rank=None):
    hep.style.use("ATLAS")

    run_number = metric.get("run_number", metric.get("run"))
    values = metric.get("values")
    edges = metric.get("edges")
    if values is None or edges is None: return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 11), sharex=True, gridspec_kw={"height_ratios": [2.3, 2.1]})

    flat_ref = metric.get("flat_ref", 0.0)
    ratios = metric.get("ratios", np.zeros(len(values)))
    total_events = metric.get("total_events", float(np.sum(values)))

    # Panel 1: distribution and flat expectation
    hep.histplot((values, edges), ax=ax1, histtype="step", color="black", linewidth=2.0, label="Centrality Distribution")
    if flat_ref > 0:
        ax1.axhline(flat_ref, color="crimson", linestyle="--", linewidth=1.8, label=rf"Flat Expectation ({flat_ref:.2e})")
        ax1.axvspan(0, flat_max, color="forestgreen", alpha=0.08, label=f"Flatness Range [1%, {flat_max:.0f}%]")

    ax1.set_ylabel("Events", fontsize=17)
    ax1.set_xlim(0, 100)
    ax1.tick_params(axis="both", labelsize=14)
    overall_max = float(np.max(values)) if len(values) > 0 else 0.0
    # Headroom so the info box and legend sit above the distribution
    ax1.set_ylim(0, max(overall_max * 1.6, flat_ref * 1.6) if overall_max > 0 else 1.0)
    ax1.legend(loc="upper right", fontsize=13, frameon=True, framealpha=0.92)
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

    ax1.text(0.03, 0.95, info_text, transform=ax1.transAxes, va="top", ha="left", fontsize=15, fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.45", facecolor="white", edgecolor=box_edge, linewidth=1.6, alpha=0.92))

    # Panel 2: ratio to flat expectation, with the 20%-class averages the calibration check uses
    tol = max_class_dev / 100.0
    r_slice = ratios[1:]
    hep.histplot((r_slice, edges[1:]), ax=ax2, histtype="step", color="royalblue", linewidth=1.6, alpha=0.75, label="Ratio to Flat (1% bins)")
    ax2.axhline(1.0, color="gray", linestyle="-", linewidth=1.2)
    ax2.axhspan(1.0 - tol, 1.0 + tol, color="forestgreen", alpha=0.15, label=rf"Tolerance $\pm${max_class_dev:g}%")
    ax2.axvline(flat_max, color="gray", linestyle=":", linewidth=1.5)

    coarse_edges = metric.get("coarse_edges", np.array([]))
    coarse_ratios = metric.get("coarse_ratios", np.array([]))
    for i, (lo, hi) in enumerate(zip(coarse_edges[:-1], coarse_edges[1:])):
        r = coarse_ratios[i]
        dev = (r - 1.0) * 100.0
        color = "forestgreen" if abs(dev) <= max_class_dev else "crimson"
        ax2.hlines(r, lo, hi, color=color, linewidth=5, zorder=5,
                   label=f"{coarse_edges[1] - coarse_edges[0]:.0f}% Class Average" if i == 0 else None)

    # Fit the y-range to the populated bins (1-92%) and the class averages, leaving the top third for the info box
    r_centers = 0.5 * (edges[1:-1] + edges[2:])
    pos_r = r_slice[(r_slice > 0) & (r_centers < N_CENT_CLASSES + 0.5)]
    if len(pos_r) > 0:
        r_min = min(float(np.min(pos_r)), 1.0 - tol)
        r_max = max(float(np.max(pos_r)), 1.0 + tol)
        span = r_max - r_min
        y2_min, y2_max = max(0.0, r_min - 0.12 * span), r_max + 0.60 * span
    else: y2_min, y2_max, span = 0.0, 1.25, 0.25
    ax2.set_ylim(y2_min, y2_max)

    # Label each class average with its deviation, above the bar for an excess and below for a deficit
    for i, (lo, hi) in enumerate(zip(coarse_edges[:-1], coarse_edges[1:])):
        r = coarse_ratios[i]
        dev = (r - 1.0) * 100.0
        color = "forestgreen" if abs(dev) <= max_class_dev else "crimson"
        offset = 0.04 * span if dev >= 0 else -0.04 * span
        ax2.text(0.5 * (lo + hi), r + offset, f"{dev:+.1f}%", ha="center", va="bottom" if dev >= 0 else "top",
                 fontsize=14, fontweight="bold", color=color, zorder=6,
                 bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="none", alpha=0.8))

    def ffmt(k, fmt=":.2f", sfx="%"):
        v = metric.get(k, np.nan)
        return (f"{{{fmt}}}").format(v)+sfx if not np.isnan(v) else "N/A"

    worst = metric.get("worst_class", "")
    ratio_metrics_text = (
        rf"Worst 20% Class: {worst} at {ffmt('class_dev_signed', ':+.1f')} (cut $\pm${max_class_dev:g}%)" + "\n" +
        rf"$R_{{1\%}}$: {ffmt('ratio_1', ':.3f', '')} | $R_{{1-5\%}}$: {ffmt('ratio_1_5', ':.3f', '')}"
    )
    ax2.text(0.03, 0.95, ratio_metrics_text, transform=ax2.transAxes, va="top", ha="left", fontsize=14,
             bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor=box_edge, linewidth=1.4, alpha=0.92))

    ax2.set_xlabel("Centrality [%]", fontsize=17)
    ax2.set_ylabel("Ratio to Flat", fontsize=17)
    ax2.set_xlim(0, 100)
    ax2.tick_params(axis="both", labelsize=14)
    ax2.legend(loc="upper right", fontsize=12, frameon=True, framealpha=0.92)
    ax2.grid(True, linestyle="--", alpha=0.3)

    fig.subplots_adjust(top=0.945, bottom=0.08, left=0.12, right=0.96, hspace=0.08)
    _stamp_selection(fig, metric)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)

def plot_failure_example(metric, failure_mode, output_path, flat_max=80.0, max_class_dev=3.5):
    return plot_centrality_1d_diagnostic(metric, failure_mode, output_path, flat_max, max_class_dev)

def plot_top_flat_example(metric, rank, output_path, flat_max=80.0, max_class_dev=3.5):
    return plot_centrality_1d_diagnostic(metric, f"Top Flat Example #{rank}", output_path, flat_max, max_class_dev, True, rank)

def plot_user_example(metric, output_path, flat_max=80.0, max_class_dev=3.5, title_suffix="User Specified"):
    return plot_centrality_1d_diagnostic(metric, title_suffix, output_path, flat_max, max_class_dev, metric.get("status") == "GOOD")

def plot_failure_mode_metric_distribution(metrics_list, failure_mode, example_runs, output_path,
                                          max_class_dev=3.5, r1_cuts=R1_CUTS, r1_5_cuts=R1_5_CUTS,
                                          min_events=1e5, title=None):
    """
    Distribution of the metric behind `failure_mode` over all runs, with the cut(s) and a line per
    example run. `title` replaces the flag name in the text box (e.g. for the class-deviation peak sets).
    """
    if hasattr(metrics_list, "to_dict"): metrics_list = metrics_list.to_dict("records")
    if not example_runs: return
    run_dict = {d.get("run_number", d.get("run")): d for d in metrics_list}
    example_dicts = [run_dict[r] for r in example_runs if r in run_dict]
    if not example_dicts: return

    output_path = Path(output_path)
    if output_path.is_dir() or output_path.suffix == "": output_path = output_path / "centrality_metric_distributions.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    line_styles = [{"color": "#800080", "ls": "-.", "lw": 2.0}, {"color": "#d95f02", "ls": ":", "lw": 2.4},
                   {"color": "#1b9e77", "ls": "--", "lw": 2.0}, {"color": "#e7298a", "ls": "-", "lw": 2.0}]

    def collect(key, include_low_stats=False):
        # Low-statistics runs skip the shape checks, so by default they are left out to match the flag counts
        return [float(d[key]) for d in metrics_list
                if not np.isnan(d.get(key, np.nan)) and (include_low_stats or "FLAG_LOW_STATS" not in d.get("status", ""))]

    def make_plot(data, color, key, line_fmt, cuts, xlabel, header, x_range=None, n_ticks=18, is_log_x=False, min_val=None):
        fig, ax = plt.subplots(figsize=(9, 6.5))
        if is_log_x:
            bins = _log_tick_bins(max(1.0, min(data)), max(data))
            ax.set_xscale("log")
        else:
            bins = _snap_axis(ax, *x_range, n_ticks=n_ticks)
        ax.hist(data, bins=bins, histtype="step", color=color, linewidth=2.0, label="All Runs Distribution")
        for cval, cc, clab in cuts: ax.axvline(cval, color=cc, linestyle="--", linewidth=2.2, label=clab)
        for idx, d in enumerate(example_dicts):
            st = line_styles[idx % len(line_styles)]
            val = d.get(key, 0.0)
            line_val = max(min_val, val) if min_val is not None else val
            ax.axvline(line_val, color=st["color"], linestyle=st["ls"], linewidth=st["lw"],
                       label=line_fmt.format(d.get('run_number', d.get('run')), val))

        ax.set_xlabel(xlabel, fontsize=15)
        ax.set_ylabel("Runs", fontsize=15)
        ax.set_yscale("log")
        ax.set_ylim(0.5, 3500)
        ax.legend(loc="upper left" if is_log_x else "upper right", fontsize=11, frameon=True, framealpha=0.92)
        ax.grid(True, linestyle="--", alpha=0.3, which="both")
        if header:
            text = header + f"\nSelected Runs: {', '.join(map(str, example_runs))}"
            ax.text(0.04, 0.94, text, transform=ax.transAxes, ha="left", va="top", fontsize=11.0,
                    bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="gray", alpha=0.92, zorder=4))
        fig.tight_layout()
        _stamp_selection(fig, example_dicts)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

    def bold(s):
        return r"$\bf{" + (s or failure_mode).replace("_", r"\_").replace(" ", r"\ ") + "}$"

    if failure_mode == "FLAG_CALIB_SHIFT":
        dv = collect("class_dev_max")
        make_plot(dv, "darkviolet", "class_dev_max", "Example Run {} (Worst Class = {:.1f}%)",
                  [(max_class_dev, "crimson", f"Cut ({max_class_dev:g}%)")], "Worst 20% Class Deviation from Flat [%]",
                  bold(title) + f"\nCut: a 20% class off flat by > {max_class_dev:g}%", x_range=(0.0, max(dv)))

    elif failure_mode in ("FLAG_R1_LOW", "FLAG_R1_HIGH", "FLAG_R1_5_LOW", "FLAG_R1_5_HIGH"):
        is_r1 = failure_mode.startswith("FLAG_R1_") and not failure_mode.startswith("FLAG_R1_5")
        key, cuts, color, sym = ("ratio_1", r1_cuts, "crimson", r"R_{1\%}") if is_r1 else ("ratio_1_5", r1_5_cuts, "royalblue", r"R_{1-5\%}")
        vals = collect(key)
        sym_fmt = sym.replace("{", "{{").replace("}", "}}")  # literal braces for str.format
        make_plot(vals, color, key, "Example Run {} " + rf"(${sym_fmt}$" + " = {:.3f})",
                  [(cuts[0], "crimson", f"Low Cut ({cuts[0]:g})"), (cuts[1], "darkorange", f"High Cut ({cuts[1]:g})")],
                  rf"Ratio to Flat, ${sym}$", bold(title) + "\n" + rf"Allowed: {cuts[0]:g} $\leq {sym} \leq$ {cuts[1]:g}",
                  x_range=_ratio_range(vals, cuts), n_ticks=15)

    elif failure_mode == "FLAG_LOW_STATS":
        nevts = collect("total_events", include_low_stats=True)
        make_plot(nevts, "coral", "total_events", "Example Run {} (Events = {:.2e})",
                  [(min_events, "crimson", f"Min Events ({min_events:.0e})")], "Total Events per Run", None,
                  is_log_x=True, min_val=1.0)
