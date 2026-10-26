#!/usr/bin/env python3
"""
Compare the centrality QA of the same runs under different event selections.

Each selection is a --hist-name understood by centrality_qa_metrics.get_centrality_hist_data.
In each comparison the first selection is the reference; every other selection is compared
with it run by run, on the runs with enough events in all of them (no FLAG_LOW_STATS).
"""

import os
import sys
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplhep as hep

# Ensure local imports from the same directory work reliably
script_dir = Path(__file__).resolve().parent
if str(script_dir) not in sys.path:
    sys.path.insert(0, str(script_dir))

from centrality_qa_metrics import flat_expectation, R1_CUTS, R1_5_CUTS
from plot_centrality_aggregate import extract_metrics_cached, failure_mode_descriptions


# Comparison name -> [(label, hist_name), ...], reference first.
# "Trig N" means trigger N fired; the other trigger may also have fired. Every selection requires MB.
PRESETS = {
    # Offline z cut at fixed trigger; the 10 < |z| < 60 cm slice is disjoint from |z| < 10 cm
    "zvtx": [
        ("|z| < 60 cm, Trig 14", "hCentralityZ60_Trig14_MB"),
        ("|z| < 10 cm, Trig 14", "hCentrality_Trig14_MB"),
        ("10 < |z| < 60 cm, Trig 14", "hCentralityZOuter_Trig14_MB"),
    ],
    # Trigger at fixed z window. Trig 12's online vertex cut narrows its z distribution, so the
    # z-weighted Trig 14 sample shows how much of the Trig 12 vs Trig 14 difference is from z alone
    "trigger": [
        ("|z| < 60 cm, Trig 14", "hCentralityZ60_Trig14_MB"),
        ("|z| < 60 cm, Trig 12", "hCentralityZ60_Trig12_MB"),
        ("|z| < 60 cm, Trig 14, z-weighted to Trig 12", "hCentralityZ60_Trig14_MB_zw12"),
    ],
    # Current default selection vs the same triggers over the full MB z range; the |z| < 60 cm
    # sample also includes the Trig 14 and Trig 12 samples for reference
    "default_vs_z60": [
        ("|z| < 10 cm, Trig 12 | Trig 14 (default)", "hCentrality"),
        ("|z| < 60 cm, Trig 12 | Trig 14", "hCentralityZ60"),
        ("|z| < 60 cm, Trig 12", "hCentralityZ60_Trig12_MB"),
        ("|z| < 60 cm, Trig 14", "hCentralityZ60_Trig14_MB"),
        ("|z| < 10 cm, Trig 12", "hCentrality_Trig12_MB"),
    ],
    # The four selections of interest for the event-selection choice, each with all of its own runs
    "main_selections": [
        ("|z| < 10 cm, Trig 12 | Trig 14 (default)", "hCentrality"),
        ("|z| < 60 cm, Trig 12 | Trig 14", "hCentralityZ60"),
        ("|z| < 60 cm, Trig 14", "hCentralityZ60_Trig14_MB"),
        ("|z| < 10 cm, Trig 12", "hCentrality_Trig12_MB"),
    ],
    # Trigger in the narrow z window, where both triggers are efficient
    "trigger_z10": [
        ("|z| < 10 cm, Trig 14", "hCentrality_Trig14_MB"),
        ("|z| < 10 cm, Trig 12", "hCentrality_Trig12_MB"),
    ],
    # Events that fired exactly one trigger: disjoint samples
    "trigger_exclusive": [
        ("|z| < 60 cm, Trig 14 only", "hCentralityZ60_Trig14only_MB"),
        ("|z| < 60 cm, Trig 12 only", "hCentralityZ60_Trig12only_MB"),
    ],
}

# Run ranges with distinct calibration behaviour (tasks/centrality-qa.md); the last is open-ended
RUN_PERIODS = [(67597, 69499), (69500, 75599), (75600, 10**9)]

COLORS = ["black", "royalblue", "darkorange", "forestgreen", "crimson", "purple"]

# Colours of the class-deviation distribution plots: no green or red, which the legend uses for pass and fail
DIST_COLORS = ["black", "royalblue", "darkorange", "purple", "teal", "saddlebrown"]


def period_label(lo, hi):
    return f"{lo}+" if hi >= 10**9 else f"{lo}–{hi}"


def flags_of(d):
    return [f for f in d["status"].split(";") if f and f != "GOOD"]


def class_dev(d, i=0):
    """Signed deviation [%] of coarse class i (0 = 1-20%) from its flat share."""
    return (d["coarse_ratios"][i] - 1.0) * 100.0 if len(d["coarse_ratios"]) > i else np.nan


def paired_runs(sel_metrics):
    """Runs present in every selection with enough events to judge (no FLAG_LOW_STATS)."""
    sets = [{r for r, d in m.items() if "FLAG_LOW_STATS" not in d["status"]} for m in sel_metrics]
    return sorted(set.intersection(*sets))


def summed_shape(metrics, runs):
    """Centrality histogram summed over runs: (values, edges, ratio to flat, flat_ref)."""
    values = np.sum([metrics[r]["values"] for r in runs], axis=0)
    edges = metrics[runs[0]]["edges"]
    flat_ref, _, _ = flat_expectation(values, edges)
    return values, edges, values / flat_ref if flat_ref > 0 else np.zeros_like(values), flat_ref


def write_summary(name, sels, sel_metrics, runs, flags, out_path):
    labels = [lbl for lbl, _ in sels]
    width = max(len(lbl) for lbl in labels) + 2
    lines = ["=" * 110, f"CENTRALITY SELECTION COMPARISON: {name}", "=" * 110]
    for i, (lbl, hist) in enumerate(sels):
        lines.append(f"  [{i}] {lbl:<{width}} {hist}" + ("   (reference)" if i == 0 else ""))
    lines.append("-" * 110)

    # Per-selection flag table over every run in that selection
    short = [f.removeprefix("FLAG_") for f in flags]
    lines.append("FLAGS PER SELECTION (all runs; GOOD % is of runs with enough events)")
    lines.append(f"  {'Sel':<4}{'Runs':>6}{'GOOD':>7}{'GOOD%':>7}" + "".join(f"{s:>13}" for s in short)
                 + f"{'Med. events':>13}")
    for i, m in enumerate(sel_metrics):
        ds = list(m.values())
        judged = [d for d in ds if "FLAG_LOW_STATS" not in d["status"]]
        n_good = sum(d["status"] == "GOOD" for d in ds)
        pct = 100.0 * n_good / len(judged) if judged else 0.0
        counts = [sum(f in flags_of(d) for d in ds) for f in flags]
        med_evts = np.median([d["total_events"] for d in ds]) if ds else 0.0
        lines.append(f"  [{i}] {len(ds):>6}{n_good:>7}{pct:>6.1f}%" + "".join(f"{c:>13}" for c in counts)
                     + f"{med_evts:>13.3g}")

    lines.append("-" * 110)
    lines.append(f"PAIRED RUNS (enough events in every selection): {len(runs)}")
    if not runs:
        lines.append("=" * 110)
        text = "\n".join(lines)
        print("\n" + text + "\n")
        out_path.write_text(text + "\n")
        return

    ref = sel_metrics[0]
    lines.append("")
    lines.append("GOOD / FLAGGED vs reference (paired runs)")
    lines.append(f"  {'Sel':<4}{'both GOOD':>11}{'ref GOOD, sel FLAG':>20}{'ref FLAG, sel GOOD':>20}{'both FLAG':>11}")
    for i, m in enumerate(sel_metrics[1:], start=1):
        g_ref = np.array([ref[r]["status"] == "GOOD" for r in runs])
        g_sel = np.array([m[r]["status"] == "GOOD" for r in runs])
        lines.append(f"  [{i}] {np.sum(g_ref & g_sel):>11}{np.sum(g_ref & ~g_sel):>20}"
                     f"{np.sum(~g_ref & g_sel):>20}{np.sum(~g_ref & ~g_sel):>11}")

    lines.append("")
    lines.append("MEDIAN 20% CLASS DEVIATION FROM FLAT [%] (paired runs)")
    class_names = ["1-20%", "21-40%", "41-60%", "61-80%"]
    lines.append(f"  {'Sel':<4}{'Runs':<16}" + "".join(f"{c:>9}" for c in class_names))
    for lo, hi in [(0, 10**9)] + RUN_PERIODS:
        p_runs = [r for r in runs if lo <= r <= hi]
        tag = "all" if (lo, hi) == (0, 10**9) else period_label(lo, hi)
        for i, m in enumerate(sel_metrics):
            if not p_runs:
                continue
            meds = [np.median([class_dev(m[r], c) for r in p_runs]) for c in range(len(class_names))]
            lines.append(f"  [{i}] {tag:<16}" + "".join(f"{v:>+9.2f}" for v in meds))

    lines.append("")
    lines.append("RUN-BY-RUN DIFFERENCE sel - ref (paired runs): median [16%, 84%]")
    metrics = [
        ("1-20% class dev [%]", lambda d: class_dev(d, 0)),
        ("Worst class dev [%]", lambda d: d["class_dev_max"]),
        ("R_1%", lambda d: d["ratio_1"]),
        ("R_1-5%", lambda d: d["ratio_1_5"]),
    ]
    for i, m in enumerate(sel_metrics[1:], start=1):
        parts = []
        for key, fn in metrics:
            delta = np.array([fn(m[r]) - fn(ref[r]) for r in runs])
            q16, q50, q84 = np.percentile(delta, [16, 50, 84])
            parts.append(f"{key} {q50:+.3f} [{q16:+.3f}, {q84:+.3f}]")
        yields = np.array([m[r]["total_events"] / ref[r]["total_events"] for r in runs])
        lines.append(f"  [{i}] " + " | ".join(parts) + f" | N_sel/N_ref {np.median(yields):.3g}")

    # Per-run statistical precision, to tell real differences from noise
    lines.append("")
    lines.append("MEDIAN STATISTICAL UNCERTAINTY PER RUN (paired runs; approximate for z-weighted samples)")
    lines.append(f"  {'Sel':<4}{'20% class dev [%]':>19}{'R_1%':>9}{'R_1-5%':>9}")
    for i, m in enumerate(sel_metrics):
        n_flat = np.array([m[r]["flat_events"] for r in runs])
        flat_ref = np.array([m[r]["flat_ref"] for r in runs])
        # Multinomial share of one of four classes: sigma(dev) = 100 * sqrt(p (1 - p) / n) / p with p = 1/4
        lines.append(f"  [{i}] {np.median(100.0 * np.sqrt(3.0 / n_flat)):>19.3f}"
                     f"{np.median(1.0 / np.sqrt(flat_ref)):>9.4f}{np.median(1.0 / np.sqrt(5.0 * flat_ref)):>9.4f}")
    lines.append("(Overlapping selections share events, so their differences are smaller than these "
                 "uncertainties added in quadrature.)")
    lines.append("=" * 110)

    text = "\n".join(lines)
    print("\n" + text + "\n")
    out_path.write_text(text + "\n")
    print(f"Summary written to: {out_path}")


def plot_paired_scatter(sels, sel_metrics, runs, out_dir, max_class_dev):
    hep.style.use("ATLAS")
    panels = [
        ("1–20% class deviation [%]", lambda d: class_dev(d, 0), ()),
        ("Worst 20% class deviation [%]", lambda d: d["class_dev_max"], (max_class_dev,)),
        (r"$R_{1\%}$", lambda d: d["ratio_1"], R1_CUTS),
        (r"$R_{1-5\%}$", lambda d: d["ratio_1_5"], R1_5_CUTS),
    ]
    ref_lbl = sels[0][0]
    for i, (lbl, _) in enumerate(sels[1:], start=1):
        fig, axes = plt.subplots(2, 2, figsize=(13, 12))
        for ax, (title, fn, cuts) in zip(axes.flat, panels):
            x = np.array([fn(sel_metrics[0][r]) for r in runs])
            y = np.array([fn(sel_metrics[i][r]) for r in runs])
            lo, hi = np.nanpercentile(np.concatenate([x, y]), [0.5, 99.5])
            pad = 0.08 * (hi - lo) if hi > lo else 1.0
            lo, hi = lo - pad, hi + pad
            ax.scatter(x, y, s=10, alpha=0.5, color=COLORS[i])
            ax.plot([lo, hi], [lo, hi], color="dimgray", linestyle="--", linewidth=1.2)
            for c in cuts:
                ax.axvline(c, color="red", linestyle=":", linewidth=1.2)
                ax.axhline(c, color="red", linestyle=":", linewidth=1.2)
            ax.set_xlim(lo, hi)
            ax.set_ylim(lo, hi)
            ax.set_xlabel(f"{title}\n{ref_lbl}", fontsize=12, loc="center")
            ax.set_ylabel(f"{lbl}\n{title}", fontsize=12, loc="center")
            ax.grid(True, linestyle="--", alpha=0.35)
        fig.suptitle(f"{len(runs)} paired runs", fontsize=14)
        fig.tight_layout()
        fig.savefig(out_dir / f"paired_scatter_sel{i}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)


def plot_delta_trends(sels, sel_metrics, runs, out_path):
    hep.style.use("ATLAS")
    rows = [
        (r"$\Delta$ 1–20% class dev. [%]", lambda m, r: class_dev(m[r], 0) - class_dev(sel_metrics[0][r], 0), False),
        (r"$\Delta R_{1\%}$", lambda m, r: m[r]["ratio_1"] - sel_metrics[0][r]["ratio_1"], False),
        (r"$\Delta R_{1-5\%}$", lambda m, r: m[r]["ratio_1_5"] - sel_metrics[0][r]["ratio_1_5"], False),
        (r"$N_{sel} / N_{ref}$", lambda m, r: m[r]["total_events"] / sel_metrics[0][r]["total_events"], True),
    ]
    x = np.array(runs)
    fig, axes = plt.subplots(len(rows), 1, figsize=(13, 3.6 * len(rows)), sharex=True)
    for ax, (ylabel, fn, logy) in zip(axes, rows):
        for i, (lbl, _) in enumerate(sels[1:], start=1):
            y = np.array([fn(sel_metrics[i], r) for r in runs])
            ax.scatter(x, y, s=8, alpha=0.6, color=COLORS[i], label=f"[{i}] {lbl}")
        if not logy:
            ax.axhline(0.0, color="dimgray", linestyle="--", linewidth=1.2)
        else:
            ax.set_yscale("log")
        for lo, _ in RUN_PERIODS[1:]:
            ax.axvline(lo, color="gray", linestyle=":", linewidth=1.2)
        ax.set_ylabel(ylabel, fontsize=13)
        ax.grid(True, linestyle="--", alpha=0.35)
    axes[0].legend(loc="upper right", fontsize=10, frameon=True, framealpha=0.9)
    axes[0].set_title(f"sel − ref, ref = {sels[0][0]}", fontsize=12)
    axes[-1].set_xlabel("Run Number", fontsize=14, loc="center")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_metric_overlays(sels, sel_metrics, runs, out_path, max_class_dev):
    hep.style.use("ATLAS")
    panels = [
        ("1–20% class deviation [%]", lambda d: class_dev(d, 0), ()),
        ("Worst 20% class deviation [%]", lambda d: d["class_dev_max"], (max_class_dev,)),
        (r"$R_{1\%}$", lambda d: d["ratio_1"], R1_CUTS),
        (r"$R_{1-5\%}$", lambda d: d["ratio_1_5"], R1_5_CUTS),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for ax, (xlabel, fn, cuts) in zip(axes.flat, panels):
        vals = [np.array([fn(m[r]) for r in runs]) for m in sel_metrics]
        lo, hi = np.nanpercentile(np.concatenate(vals), [0.5, 99.5])
        bins = np.linspace(lo, hi, 60)
        for i, (v, (lbl, _)) in enumerate(zip(vals, sels)):
            ax.hist(v, bins=bins, histtype="step", linewidth=1.8, color=COLORS[i], label=f"[{i}] {lbl}")
        for c in cuts:
            ax.axvline(c, color="red", linestyle=":", linewidth=1.2)
        ax.set_xlabel(xlabel, fontsize=13, loc="center")
        ax.set_ylabel("Runs", fontsize=13)
        ax.grid(True, linestyle="--", alpha=0.35)
    axes[0, 0].legend(loc="upper right", fontsize=9, frameon=True, framealpha=0.9)
    fig.suptitle(f"{len(runs)} paired runs", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_summed_shapes(sels, sel_metrics, runs, out_path, subtitle, flat_max=80.0):
    """Centrality distributions summed over runs, as ratio to flat, and each selection over the reference."""
    if not runs:
        return
    hep.style.use("ATLAS")
    fig, (ax, ax_r) = plt.subplots(2, 1, figsize=(10, 9), sharex=True, gridspec_kw={"height_ratios": [2, 1]})

    shapes = [summed_shape(m, runs) for m in sel_metrics]
    edges = shapes[0][1]
    centers = 0.5 * (edges[:-1] + edges[1:])
    filled = (centers > 0.5) & (shapes[0][0] > 0)

    ref_vals, _, ref_ratio, _ = shapes[0]
    for i, ((vals, _, ratio, flat_ref), (lbl, _)) in enumerate(zip(shapes, sels)):
        err = np.sqrt(vals) / flat_ref if flat_ref > 0 else np.zeros_like(vals)
        ax.errorbar(centers[filled], ratio[filled], yerr=err[filled], fmt="o", ms=3, color=COLORS[i],
                    label=f"[{i}] {lbl} (N = {vals.sum():.3g})")
        if i == 0:
            continue
        ok = filled & (vals > 0) & (ref_vals > 0)
        rr = np.divide(ratio, ref_ratio, out=np.zeros_like(ratio), where=ref_ratio > 0)
        rr_err = rr * np.sqrt(np.divide(1.0, vals, out=np.zeros_like(vals), where=vals > 0)
                              + np.divide(1.0, ref_vals, out=np.zeros_like(ref_vals), where=ref_vals > 0))
        ax_r.errorbar(centers[ok], rr[ok], yerr=rr_err[ok], fmt="o", ms=3, color=COLORS[i])

    for a in (ax, ax_r):
        a.axhline(1.0, color="dimgray", linestyle="--", linewidth=1.2)
        a.axvline(flat_max + 0.5, color="gray", linestyle=":", linewidth=1.2)
        a.grid(True, linestyle="--", alpha=0.35)
    ax.set_ylabel("Ratio to Flat Expectation", fontsize=14)
    ax.legend(loc="lower left", fontsize=10, frameon=True, framealpha=0.9)
    ax.text(0.97, 0.95, subtitle, transform=ax.transAxes, fontsize=12, ha="right", va="top",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="#aaaaaa", alpha=0.92))
    ax_r.set_ylabel("Sel / Ref", fontsize=14)
    ax_r.set_xlabel("Centrality [%]", fontsize=14, loc="center")
    ax_r.set_xlim(0, 100)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _colored_legend(ax, sels, vals, signed, max_class_dev, fontsize=13):
    """
    Legend anchored at the upper right of `ax`: per selection its name in the histogram colour, the run
    count and median, then pass (green) and fail (red) counts against the cut on the plotted quantity.
    """
    from matplotlib.lines import Line2D
    from matplotlib.offsetbox import AnchoredOffsetbox, DrawingArea, HPacker, TextArea, VPacker

    def text(label, **kw):
        return TextArea(label, textprops=dict(fontsize=fontsize, **kw))

    def line_handle(color, linestyle="-"):
        da = DrawingArea(34, 12, 0, 0)
        da.add_artist(Line2D([0, 34], [6, 6], color=color, linestyle=linestyle, linewidth=3.2))
        return da

    rows = []
    for i, ((lbl, _), v) in enumerate(zip(sels, vals)):
        color = DIST_COLORS[i % len(DIST_COLORS)]
        entry = [HPacker(children=[line_handle(color), text(lbl, color=color, fontweight="bold")],
                         align="center", pad=0, sep=6)]
        if len(v):
            n_fail = int(np.sum(np.abs(v) > max_class_dev)) if signed else int(np.sum(v > max_class_dev))
            n_pass = len(v) - n_fail
            entry.append(text(f"{len(v)} runs, median {np.median(v):+.2f}%", color="black"))
            entry.append(HPacker(children=[text(f"Pass: {n_pass} ({n_pass / len(v):.1%})", color="forestgreen", fontweight="bold"),
                                           text(f"Fail: {n_fail} ({n_fail / len(v):.1%})", color="crimson", fontweight="bold")],
                                 align="baseline", pad=0, sep=14))
        rows.append(VPacker(children=entry, align="left", pad=0, sep=2))
    rows.append(HPacker(children=[line_handle("crimson", "--"),
                                  text(f"Cut ({'±' if signed else ''}{max_class_dev:g}%)", color="crimson")],
                        align="center", pad=0, sep=6))

    legend = AnchoredOffsetbox(loc="upper right", child=VPacker(children=rows, align="left", pad=0, sep=9),
                               pad=0.5, borderpad=0.5, frameon=True)
    legend.patch.set(facecolor="white", edgecolor="#cccccc", alpha=0.95)
    ax.add_artist(legend)
    return legend


def plot_class_dev_distributions(sels, sel_metrics, out_dir, max_class_dev, min_events):
    """
    One figure per class-deviation metric, with every selection overlaid. Each selection uses all of its
    own runs above the minimum-event threshold (no pairing between selections).
    """
    hep.style.use("ATLAS")
    metrics = [
        ("class_dev_1_20", "1–20% Class Deviation from Flat [%]", lambda d: class_dev(d, 0), True),
        ("worst_class_dev", "Worst 20% Class Deviation from Flat [%]", lambda d: d["class_dev_max"], False),
    ]
    judged = [[d for d in m.values() if "FLAG_LOW_STATS" not in d["status"]] for m in sel_metrics]

    for stem, xlabel, fn, signed in metrics:
        vals = [np.array([fn(d) for d in ds]) for ds in judged]
        vals = [v[~np.isnan(v)] for v in vals]
        if not any(len(v) for v in vals):
            continue
        all_vals = np.concatenate(vals)
        lo = np.floor(np.percentile(all_vals, 0.2) / 0.5) * 0.5 if signed else 0.0
        hi = np.ceil(np.percentile(all_vals, 99.8) / 0.5) * 0.5
        bins = np.arange(min(lo, -max_class_dev - 0.5) if signed else lo, max(hi, max_class_dev + 0.5) + 0.5, 0.5)

        fig, ax = plt.subplots(figsize=(10, 10))
        for i, v in enumerate(vals):
            ax.hist(np.clip(v, bins[0], bins[-1] - 1e-9), bins=bins, histtype="step", linewidth=2.2,
                    color=DIST_COLORS[i % len(DIST_COLORS)])
        cuts = (-max_class_dev, max_class_dev) if signed else (max_class_dev,)
        for c in cuts:
            ax.axvline(c, color="crimson", linestyle="--", linewidth=1.8)
        if signed:
            ax.axvline(0.0, color="dimgray", linewidth=1.2)

        ax.set_xlim(bins[0], bins[-1])
        ax.set_box_aspect(0.85)  # plot box height / width: a little wider than tall
        ax.set_xlabel(xlabel, fontsize=15)
        ax.set_ylabel("Runs", fontsize=15)
        ax.grid(True, linestyle="--", alpha=0.3, which="both")

        # Legend: selection name in its histogram colour, pass in green and fail in red
        legend = _colored_legend(ax, sels, vals, signed, max_class_dev)

        # Linear y-axis, tall enough for the legend to clear the bars beneath it and no taller
        fig.canvas.draw()
        box = legend.get_window_extent(fig.canvas.get_renderer()).transformed(ax.transAxes.inverted())
        x_left = bins[0] + box.x0 * (bins[-1] - bins[0])
        under = [np.histogram(np.clip(v, bins[0], bins[-1] - 1e-9), bins=bins)[0][bins[1:] > x_left] for v in vals if len(v)]
        peak_under = max((u.max() for u in under if len(u)), default=0)
        peak = max(np.histogram(np.clip(v, bins[0], bins[-1] - 1e-9), bins=bins)[0].max() for v in vals if len(v))
        ax.set_ylim(0, max(1.08 * peak, 1.04 * peak_under / max(1.0 - box.height - 0.03, 0.2)))
        ax.set_title(f"Runs with ≥ {min_events:.0e} events in 1–80%", fontsize=12, color="dimgray", pad=8)
        fig.tight_layout()
        fig.savefig(out_dir / f"{stem}_distribution.png", dpi=300, bbox_inches="tight")
        plt.close(fig)


def plot_selection_yields(sels, sel_metrics, out_path):
    """
    MB events of each selection summed over all runs, one bar per selection, most events first,
    each with its share of all MB events (hZVertex: every event passing the MB classifier, any trigger).
    """
    hep.style.use("ATLAS")
    counts = [sum(d["total_events"] for d in m.values()) for m in sel_metrics]
    n_mb = sum(d["n_mb"] for d in sel_metrics[0].values() if not np.isnan(d["n_mb"]))
    n_runs = [len(m) for m in sel_metrics]
    order = np.argsort(counts)[::-1]
    labels = [sels[i][0].replace(" (default)", "\n(default)").replace(", ", "\n", 1) for i in order]
    colors = [COLORS[i % len(COLORS)] for i in order]

    fig, ax = plt.subplots(figsize=(13, 7))
    bars = ax.bar(labels, [counts[i] for i in order], color=colors, alpha=0.8, width=0.6)
    for bar, i in zip(bars, order):
        ax.text(bar.get_x() + bar.get_width() / 2, counts[i], f"{counts[i]:.3e}\n({counts[i] / n_mb:.1%})",
                ha="center", va="bottom", fontsize=12)

    ax.set_xlim(-0.6, len(counts) - 0.4)
    ax.set_ylim(0, 1.2 * counts[order[0]])
    ax.set_ylabel("MB Events", fontsize=15)
    ax.set_title(f"MB events with valid centrality, {max(n_runs)} runs; "
                 f"percentages of all MB events (Total = {n_mb:.3e})", fontsize=13)
    ax.tick_params(axis="x", which="both", top=False, bottom=False, labelsize=13)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def run_comparison(name, sels, metrics_by_hist, out_dir, flags, args):
    out_dir.mkdir(parents=True, exist_ok=True)
    sel_metrics = [metrics_by_hist[hist] for _, hist in sels]
    runs = paired_runs(sel_metrics)

    write_summary(name, sels, sel_metrics, runs, flags, out_dir / "comparison_summary.txt")
    plot_selection_yields(sels, sel_metrics, out_dir / "selection_events.png")
    plot_class_dev_distributions(sels, sel_metrics, out_dir, args.max_class_dev, args.min_events)
    if not runs:
        print(f"Warning: no paired runs for comparison '{name}'; skipping plots.")
        return

    plot_paired_scatter(sels, sel_metrics, runs, out_dir, args.max_class_dev)
    plot_delta_trends(sels, sel_metrics, runs, out_dir / "delta_vs_run.png")
    plot_metric_overlays(sels, sel_metrics, runs, out_dir / "metric_distributions.png", args.max_class_dev)

    plot_summed_shapes(sels, sel_metrics, runs, out_dir / "summed_shape_all.png",
                       f"All paired runs (N = {len(runs)})", args.flat_max)
    good = [r for r in runs if all(m[r]["status"] == "GOOD" for m in sel_metrics)]
    plot_summed_shapes(sels, sel_metrics, good, out_dir / "summed_shape_good.png",
                       f"GOOD in every selection (N = {len(good)})", args.flat_max)
    for lo, hi in RUN_PERIODS:
        p_runs = [r for r in runs if lo <= r <= hi]
        plot_summed_shapes(sels, sel_metrics, p_runs, out_dir / f"summed_shape_runs_{lo}.png",
                           f"Runs {period_label(lo, hi)} (N = {len(p_runs)})", args.flat_max)
    print(f"Comparison '{name}' plots saved in: {out_dir.resolve()}")


def main():
    parser = argparse.ArgumentParser(description="Compare per-run centrality QA metrics between event selections.")
    parser.add_argument("-f", "--file", type=Path, help="Text file containing list of ROOT file paths (one per line).")
    parser.add_argument("-o", "--output-dir", type=Path, default=Path("centrality_selection_comparison"),
                        help="Output directory; each comparison gets a subdirectory (default: centrality_selection_comparison).")
    parser.add_argument("--preset", nargs="+", choices=list(PRESETS) + ["all"], default=None,
                        help="Predefined comparison(s) to run (default: all, unless --sel is given).")
    parser.add_argument("--sel", action="append", default=None, metavar="LABEL=HISTNAME",
                        help="Custom selection; repeat for 2 or more, the first is the reference.")
    parser.add_argument("--name", type=str, default="custom", help="Subdirectory name for the --sel comparison.")
    parser.add_argument("-j", "--workers", type=int, default=None,
                        help="Number of parallel worker processes (default: min(cpu_count(), 32)).")
    parser.add_argument("--flat-max", type=float, default=80.0,
                        help="Upper edge [%%] of the centrality range required to be flat (default: 80).")
    parser.add_argument("--max-class-dev", type=float, default=3.5,
                        help="Maximum 20%%-class deviation [%%] before flagging CALIB_SHIFT (default: 3.5).")
    parser.add_argument("--min-events", type=float, default=1e5,
                        help="Minimum events in the flatness range; runs below are flagged LOW_STATS (default: 1e5).")
    parser.add_argument("--no-cache", action="store_true",
                        help="Re-extract every run without reading or writing the metrics caches.")
    parser.add_argument("files", nargs="*", type=Path, help="Positional list of ROOT file paths.")
    args = parser.parse_args()

    file_list = list(args.files)
    if args.file:
        with args.file.open() as f:
            file_list.extend(Path(line.strip()) for line in f if line.strip() and not line.startswith("#"))
    if not file_list:
        parser.error("provide ROOT files or a file list (-f)")

    comparisons = {}
    if args.sel:
        if len(args.sel) < 2:
            parser.error("--sel needs at least two selections")
        sels = []
        for s in args.sel:
            if "=" not in s:
                parser.error(f"--sel '{s}' is not LABEL=HISTNAME")
            lbl, hist = s.rsplit("=", 1)
            sels.append((lbl.strip(), hist.strip()))
        comparisons[args.name] = sels
    presets = args.preset or ([] if args.sel else ["all"])
    for p in presets:
        for name in (PRESETS if p == "all" else [p]):
            comparisons[name] = PRESETS[name]

    max_workers = args.workers if args.workers is not None else min(os.cpu_count() or 4, 32)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    flags = [flag for flag, _ in failure_mode_descriptions(args)]

    # Extract each distinct selection once, with one metrics cache per histogram name
    metrics_by_hist = {}
    for hist in dict.fromkeys(h for sels in comparisons.values() for _, h in sels):
        print(f"Extracting metrics for {hist} ...")
        cache_path = None if args.no_cache else args.output_dir / "cache" / f"{hist}.pkl"
        metrics, errors = extract_metrics_cached(
            file_list,
            extract_kwargs=dict(hist_name=hist, flat_max=args.flat_max,
                                max_class_dev=args.max_class_dev, min_events=args.min_events),
            cache_path=cache_path,
            max_workers=max_workers,
        )
        if errors:
            print(f"  {len(errors)} file(s) failed for {hist}; first: {errors[0]}")
        metrics_by_hist[hist] = {d["run_number"]: d for d in metrics}

    for name, sels in comparisons.items():
        run_comparison(name, sels, metrics_by_hist, args.output_dir / name, flags, args)


if __name__ == "__main__":
    main()
