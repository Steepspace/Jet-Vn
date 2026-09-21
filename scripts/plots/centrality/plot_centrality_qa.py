#!/usr/bin/env python3

import uproot
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mplhep as hep
import os
import tqdm
import concurrent.futures
import argparse
from pathlib import Path
import numpy as np
import re
import functools
import sys
import traceback

from matplotlib.ticker import LogLocator, ScalarFormatter

def clean_root_latex(text):
    if not text:
        return ""
    text = re.sub(r'\s+', ' ', text).strip()
    # Replace ROOT TLatex # with \ for LaTeX math
    text = text.replace("#", "\\")
    if "$" not in text:
        # Wrap LaTeX command terms (like \chi, \eta, \Psi) or terms containing subscripts/superscripts in math mode $...$
        text = re.sub(
            r'(\\[a-zA-Z]+(?:\^\{[^}\s]+\}|_\{[^}\s]+\}|\^[a-zA-Z0-9]+|_[a-zA-Z0-9]+)*|[a-zA-Z0-9\\_*|()]*(?:\^\{[^}\s]+\}|_\{[^}\s]+\}|\^[a-zA-Z0-9]+|_[a-zA-Z0-9]+)+|\\[a-zA-Z]+)',
            r'$\1$',
            text
        )
    return text

def get_hist_axis_titles(hist1d, hist_name=""):
    raw_title = ""
    xlabel = ""
    ylabel = ""

    # 1. Try uproot all_members dict
    if hasattr(hist1d, "all_members"):
        members = hist1d.all_members
        raw_title = members.get("fTitle", "").strip()

        fXaxis = members.get("fXaxis")
        if fXaxis is not None:
            if hasattr(fXaxis, "all_members"):
                xlabel = fXaxis.all_members.get("fTitle", "").strip()
            elif hasattr(fXaxis, "member"):
                try:
                    xlabel = fXaxis.member("fTitle").strip()
                except Exception:
                    pass

        fYaxis = members.get("fYaxis")
        if fYaxis is not None:
            if hasattr(fYaxis, "all_members"):
                ylabel = fYaxis.all_members.get("fTitle", "").strip()
            elif hasattr(fYaxis, "member"):
                try:
                    ylabel = fYaxis.member("fTitle").strip()
                except Exception:
                    pass

    # 2. Try uproot axis high-level properties
    if not xlabel:
        for attr in ["label", "title"]:
            try:
                val = getattr(hist1d.axis(0), attr, None)
                if val:
                    xlabel = str(val).strip()
                    break
            except Exception:
                pass

    # 3. Check if raw_title is formatted as "Title;XTitle;YTitle"
    if not raw_title:
        raw_title = getattr(hist1d, "title", "") or ""

    if ";" in raw_title:
        parts = [p.strip() for p in raw_title.split(";")]
        if not xlabel and len(parts) > 1:
            xlabel = parts[1]
        if not ylabel and len(parts) > 2:
            ylabel = parts[2]

    xlabel = clean_root_latex(xlabel)
    ylabel = clean_root_latex(ylabel)

    return xlabel, ylabel

class EngScalarFormatter(ScalarFormatter):
    """
    Custom ScalarFormatter that constrains the order of magnitude
    to multiples of 3 (e.g. 10^3, 10^6), preventing labels like 1000 x 10^3
    and reducing them to 10^6 where appropriate.
    """
    def _set_order_of_magnitude(self):
        super()._set_order_of_magnitude()
        if self._orderOfMagnitude != 0:
            self._orderOfMagnitude = (self._orderOfMagnitude // 3) * 3

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

def get_ratio_ylim(ratio_vals, edges, xlim=None):
    """
    Determine tightly fitted y-axis limits for ratio plots:
    - Captures the lowest non-zero bin cleanly with a modest margin below it (without going down to 0).
    - Caps the top tightly above 1.0 (and above the max non-zero bin) to minimize excess white space.
    """
    if xlim is not None:
        mask = (edges[1:] > xlim[0]) & (edges[:-1] < xlim[1])
        visible_ratios = ratio_vals[mask]
    else:
        visible_ratios = ratio_vals

    pos_ratios = visible_ratios[visible_ratios > 0]
    if len(pos_ratios) == 0:
        return (0.8, 1.1)

    min_pos = float(np.min(pos_ratios))
    max_pos = float(np.max(pos_ratios))

    # Give a clear margin below the lowest non-zero bin (without extending to 0)
    margin_bottom = max((1.0 - min_pos) * 0.15, 0.04) if min_pos < 1.0 else 0.04
    y_bottom = max(0.0, min_pos - margin_bottom)

    # Tight margin above 1.0 / highest bin to avoid large white space above 1.0
    margin_top = max((max_pos - 1.0) * 1.5, 0.04) if max_pos > 1.0 else 0.04
    y_top = max(max_pos, 1.0) + margin_top

    return (y_bottom, y_top)

def make_1d_plot(hist_or_tuple, run_number, output_path, xlabel=None, ylabel=None, extra_labels=None, logy=False, xlim=None, ylim=None, hline=None):
    hep.style.use("ATLAS")
    fig, ax = plt.subplots(figsize=(8, 6))

    if hasattr(hist_or_tuple, "to_numpy"):
        values, edges = hist_or_tuple.to_numpy()
        h_xlabel, h_ylabel = get_hist_axis_titles(hist_or_tuple)
        if not xlabel:
            xlabel = h_xlabel
        if not ylabel:
            ylabel = h_ylabel
    else:
        values, edges = hist_or_tuple

    if not xlabel:
        xlabel = "Bin"
    if not ylabel:
        ylabel = "Events"

    if xlim is not None:
        mask = (edges[1:] > xlim[0]) & (edges[:-1] < xlim[1])
        visible_values = values[mask] if np.any(mask) else values
    else:
        visible_values = values

    has_positive = np.any(visible_values > 0)
    max_val = np.max(visible_values) if has_positive else 0
    top_ref = max(max_val, hline) if (hline is not None and hline > 0) else max_val

    hep.histplot((values, edges), ax=ax, histtype='step', color='blue', linewidth=3)

    if ylim is not None:
        ax.set_ylim(ylim)
    elif logy and has_positive:
        ax.set_yscale('log')
        ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=20))
        ax.set_ylim(bottom=0.5, top=top_ref * 10)
    else:
        ax.set_ylim(bottom=0, top=top_ref * 1.3 if has_positive else 10)
        if top_ref >= 1000:
            formatter_y = EngScalarFormatter(useMathText=True)
            formatter_y.set_powerlimits((0, 3))
            ax.yaxis.set_major_formatter(formatter_y)

    if hline is not None:
        ax.axhline(hline, color='red', linestyle='--', linewidth=2, alpha=0.85)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if xlim is not None:
        ax.set_xlim(xlim)
    else:
        ax.set_xlim(left=edges[0], right=edges[-1])

    ax.text(1.0, 1.01, rf"Run: {run_number}", transform=ax.transAxes, ha='right', va='bottom', fontsize=15)

    if extra_labels:
        ax.text(0.95, 0.95, "\n".join(extra_labels), transform=ax.transAxes, ha='right', va='top', fontsize=15)

    fig.tight_layout()
    plt.subplots_adjust(left=0.12, bottom=0.13, top=0.93)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)

def make_ratio_overlay_plot(hist_data_list, run_number, output_path, xlabel="Centrality [%]", ylabel="Ratio to Average", xlim=(-0.5, 10.5), hline=1.0):
    """
    Plots an overlay of multiple 1D histogram ratios on the same axes with proper legends.
    hist_data_list: list of tuples (ratio_vals, edges, label, color)
    """
    hep.style.use("ATLAS")
    fig, ax = plt.subplots(figsize=(8, 6))

    all_pos_ratios = []
    for ratio_vals, edges, label, color in hist_data_list:
        if xlim is not None:
            mask = (edges[1:] > xlim[0]) & (edges[:-1] < xlim[1])
            vis = ratio_vals[mask]
        else:
            vis = ratio_vals
        pos = vis[vis > 0]
        if len(pos) > 0:
            all_pos_ratios.extend(pos)

        hep.histplot((ratio_vals, edges), ax=ax, histtype='step', color=color, linewidth=2.5, label=label)

    if hline is not None:
        ax.axhline(hline, color='gray', linestyle='--', linewidth=1.5, alpha=0.8)

    if len(all_pos_ratios) > 0:
        min_pos = float(np.min(all_pos_ratios))
        max_pos = float(np.max(all_pos_ratios))
        margin_bottom = max((1.0 - min_pos) * 0.15, 0.04) if min_pos < 1.0 else 0.04
        y_bottom = max(0.0, min_pos - margin_bottom)
        margin_top = max((max_pos - 1.0) * 1.5, 0.04) if max_pos > 1.0 else 0.04
        y_top = max(max_pos, 1.0) + margin_top
        ax.set_ylim(y_bottom, y_top)

    if xlim is not None:
        ax.set_xlim(xlim)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    ax.legend(loc="lower right", fontsize=20, frameon=True, framealpha=0.9, edgecolor="none")
    ax.text(1.0, 1.01, rf"Run: {run_number}", transform=ax.transAxes, ha='right', va='bottom', fontsize=15)

    fig.tight_layout()
    plt.subplots_adjust(left=0.12, bottom=0.13, top=0.93)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)

def make_zvertex_cent_slices_plot(hist2d, run_number, output_path, slices=(1, 2, 3), logy=False):
    hep.style.use("ATLAS")
    fig, axes = plt.subplots(1, len(slices), figsize=(16, 5), sharey=True, gridspec_kw={'wspace': 0})

    values_2d, edges_x, edges_y = hist2d.to_numpy()

    # Find max y across the requested slices for consistent shared y scaling
    slice_data = []
    max_val = 0
    for c in slices:
        idx = int(round(c))
        if idx >= values_2d.shape[1]:
            continue
        v = values_2d[:, idx]
        m = np.max(v) if v.size > 0 else 0
        if m > max_val:
            max_val = m
        slice_data.append((c, idx, v))

    has_positive = max_val > 0

    for i, (c, idx, v) in enumerate(slice_data):
        ax = axes[i]
        hep.histplot((v, edges_x), ax=ax, histtype='step', color='blue', linewidth=2.5)
        ax.set_xlim(edges_x[0], edges_x[-1])
        ax.set_xlabel(r"$z_{\mathrm{vtx}}$ [cm]")

        if logy and has_positive:
            ax.set_yscale('log')
            ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=20))
            ax.set_ylim(bottom=0.5, top=max_val * 10)
        else:
            ax.set_ylim(bottom=0, top=max_val * 1.35 if has_positive else 10)

        # Inset label with Centrality %, MB, and Total events
        total_slice = np.sum(v)
        info_text = f"Centrality: {c}%\nMB\nTotal: {total_slice:.2e}"
        ax.text(0.92, 0.94, info_text, transform=ax.transAxes, ha='right', va='top', fontsize=13)

        if i == 0:
            ax.set_ylabel("Events")
            if not logy and max_val >= 1000:
                formatter_y = EngScalarFormatter(useMathText=True)
                formatter_y.set_powerlimits((0, 3))
                ax.yaxis.set_major_formatter(formatter_y)
        else:
            ax.tick_params(axis='y', which='both', left=True, labelleft=False)

    axes[-1].text(1.0, 1.02, rf"Run: {run_number}", transform=axes[-1].transAxes, ha='right', va='bottom', fontsize=15)

    fig.tight_layout()
    plt.subplots_adjust(left=0.08, right=0.97, bottom=0.15, top=0.92)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)

def process_file(path, output_dir=None, logy=False, run_subdirs=False, cent_flat_min=10.0, cent_flat_max=70.0):
    path = Path(path)
    if not path.exists():
        return f"File not found: {path}"

    try:
        try:
            run_number = int(path.name.split('.')[0])
        except ValueError:
            match = re.search(r'\d+', path.name)
            if match:
                run_number = int(match.group())
            else:
                return f"Could not parse run number from {path.name}"

        run_output_dir = output_dir if output_dir is not None else Path(".")
        if run_subdirs:
            run_output_dir = run_output_dir / f"{run_number}"

        dir_cent = run_output_dir / "centrality"
        dir_cent_zoom = run_output_dir / "centrality_zoom"
        dir_ratio = run_output_dir / "ratio"
        dir_ratio_zoom = run_output_dir / "ratio_zoom"
        dir_overlay = run_output_dir / "overlay"
        dir_zvtx = run_output_dir / "z_vertex"

        with uproot.open(path) as file:
            plots_made = 0
            cent_ratios = {}

            # 1. 1D Centrality plots (full 0-100% and zoomed first 11 bins: -0.5 to 10.5)
            cent_hist_names = [
                "hCentrality",
                "hCentrality_Trig12",
                "hCentrality_Trig12_MB",
                "hCentrality_Trig14",
                "hCentrality_Trig14_MB",
                "hCentralityZ150",
                "hCentralityZ150_Trig14",
                "hCentralityZOuter",
                "hCentralityZOuter_Trig14",
            ]
            for hist_name in cent_hist_names:
                if hist_name in file:
                    hist1d = file[hist_name]
                    values, edges = hist1d.to_numpy()
                    total_events = np.sum(values)

                    base_labels = []
                    cleaned_title = ""
                    title = hist1d.title
                    if title:
                        if ";" in title:
                            title = title.split(";")[0].strip()
                        cleaned_title = clean_root_latex(title)
                        if cleaned_title:
                            base_labels.append(cleaned_title)

                    # Compute flat average excluding edge outliers
                    avg_val = compute_centrality_average(
                        values, edges, cent_min=cent_flat_min, cent_max=cent_flat_max
                    )
                    avg_label = [f"Average: {avg_val:.2e}"] if avg_val > 0 else []

                    # Full 0-100% distribution with average line
                    extra_labels = base_labels + [f"Total: {total_events:.2e}"] + avg_label
                    output_path = dir_cent / f"run_{run_number}_{hist_name}.png"
                    make_1d_plot(
                        hist1d,
                        run_number,
                        output_path,
                        xlabel="Centrality [%]",
                        ylabel="Events",
                        extra_labels=extra_labels,
                        logy=logy,
                        xlim=(0, 100),
                        hline=avg_val if avg_val > 0 else None,
                    )
                    plots_made += 1

                    # Full 0-100% ratio to average
                    if avg_val > 0:
                        ratio_vals = np.where(values > 0, values / avg_val, 0.0)
                        cent_ratios[hist_name] = (ratio_vals, edges, cleaned_title)
                        ylim_full_ratio = get_ratio_ylim(ratio_vals, edges, xlim=(0, 100))
                        output_path_ratio = dir_ratio / f"run_{run_number}_{hist_name}_ratio.png"
                        make_1d_plot(
                            (ratio_vals, edges),
                            run_number,
                            output_path_ratio,
                            xlabel="Centrality [%]",
                            ylabel="Ratio to Average",
                            extra_labels=base_labels + avg_label,
                            logy=False,
                            xlim=(0, 100),
                            ylim=ylim_full_ratio,
                            hline=1.0,
                        )
                        plots_made += 1

                    # Zoomed 0-10% centrality (first 11 bins: -0.5 to 10.5) with average line
                    mask_zoom = (edges[1:] > -0.5) & (edges[:-1] < 10.5)
                    total_events_zoom = np.sum(values[mask_zoom])
                    extra_labels_zoom = base_labels + [f"Total: {total_events_zoom:.2e}"] + avg_label

                    output_path_zoom = dir_cent_zoom / f"run_{run_number}_{hist_name}_zoom.png"
                    make_1d_plot(
                        hist1d,
                        run_number,
                        output_path_zoom,
                        xlabel="Centrality [%]",
                        ylabel="Events",
                        extra_labels=extra_labels_zoom,
                        logy=logy,
                        xlim=(-0.5, 10.5),
                        hline=avg_val if avg_val > 0 else None,
                    )
                    plots_made += 1

                    # Zoomed 0-10% ratio to average
                    if avg_val > 0:
                        ylim_zoom_ratio = get_ratio_ylim(ratio_vals, edges, xlim=(-0.5, 10.5))
                        output_path_zoom_ratio = dir_ratio_zoom / f"run_{run_number}_{hist_name}_zoom_ratio.png"
                        make_1d_plot(
                            (ratio_vals, edges),
                            run_number,
                            output_path_zoom_ratio,
                            xlabel="Centrality [%]",
                            ylabel="Ratio to Average",
                            extra_labels=base_labels + avg_label,
                            logy=False,
                            xlim=(-0.5, 10.5),
                            ylim=ylim_zoom_ratio,
                            hline=1.0,
                        )
                        plots_made += 1
                else:
                    print(f"Warning: '{hist_name}' not found in {path}")

            # Overlay of zoomed ratio plots: hCentrality_zoom and hCentralityZ150_Trig14_zoom
            if "hCentrality" in cent_ratios and "hCentralityZ150_Trig14" in cent_ratios:
                r1, e1, t1 = cent_ratios["hCentrality"]
                r2, e2, t2 = cent_ratios["hCentralityZ150_Trig14"]

                label1 = t1 if t1 else "|z| < 10 cm and MB"
                label2 = t2 if t2 else "|z| < 150 cm and Trig 14"
                label2 = re.sub(r'MBD N&S\s*>=\s*2.*', 'Trig 14', label2).strip()

                output_path_overlay = dir_overlay / f"run_{run_number}_hCentrality_vs_hCentralityZ150_Trig14_zoom_ratio.png"
                make_ratio_overlay_plot(
                    [
                        (r1, e1, label1, "blue"),
                        (r2, e2, label2, "crimson"),
                    ],
                    run_number,
                    output_path_overlay,
                    xlabel="Centrality [%]",
                    ylabel="Ratio to Average",
                    xlim=(-0.5, 10.5),
                    hline=1.0,
                )
                plots_made += 1

            # Overlay of zoomed ratio plots: hCentrality, hCentrality_Trig12, and hCentrality_Trig14
            if "hCentrality" in cent_ratios and "hCentrality_Trig12" in cent_ratios and "hCentrality_Trig14" in cent_ratios:
                r_mb, e_mb, t_mb = cent_ratios["hCentrality"]
                r_12, e_12, t_12 = cent_ratios["hCentrality_Trig12"]
                r_14, e_14, t_14 = cent_ratios["hCentrality_Trig14"]

                label_mb = t_mb if t_mb else "|z| < 10 cm and MB"
                label_12 = t_12 if t_12 else "|z| < 10 cm and Trig 12"
                label_12 = re.sub(r'MBD N&S\s*>=\s*2.*', 'Trig 12', label_12).strip()
                label_14 = t_14 if t_14 else "|z| < 10 cm and Trig 14"
                label_14 = re.sub(r'MBD N&S\s*>=\s*2.*', 'Trig 14', label_14).strip()

                output_path_trig_overlay = dir_overlay / f"run_{run_number}_hCentrality_Trig12_Trig14_zoom_ratio.png"
                make_ratio_overlay_plot(
                    [
                        (r_mb, e_mb, label_mb, "blue"),
                        (r_12, e_12, label_12, "forestgreen"),
                        (r_14, e_14, label_14, "crimson"),
                    ],
                    run_number,
                    output_path_trig_overlay,
                    xlabel="Centrality [%]",
                    ylabel="Ratio to Average",
                    xlim=(-0.5, 10.5),
                    hline=1.0,
                )
                plots_made += 1

            # Overlay of zoomed ratio plots: hCentrality, hCentralityZ150, and hCentralityZOuter
            if "hCentrality" in cent_ratios and "hCentralityZ150" in cent_ratios and "hCentralityZOuter" in cent_ratios:
                r_mb, e_mb, t_mb = cent_ratios["hCentrality"]
                r_z150, e_z150, t_z150 = cent_ratios["hCentralityZ150"]
                r_zout, e_zout, t_zout = cent_ratios["hCentralityZOuter"]

                label_mb = t_mb if t_mb else "|z| < 10 cm and MB"
                label_z150 = t_z150 if t_z150 else "|z| < 150 cm and MB"
                label_zout = t_zout if t_zout else "10 cm < |z| < 150 cm and MB"

                output_path_z_overlay = dir_overlay / f"run_{run_number}_hCentrality_Z150_ZOuter_zoom_ratio.png"
                make_ratio_overlay_plot(
                    [
                        (r_mb, e_mb, label_mb, "blue"),
                        (r_z150, e_z150, label_z150, "crimson"),
                        (r_zout, e_zout, label_zout, "forestgreen"),
                    ],
                    run_number,
                    output_path_z_overlay,
                    xlabel="Centrality [%]",
                    ylabel="Ratio to Average",
                    xlim=(-0.5, 10.5),
                    hline=1.0,
                )
                plots_made += 1

            # Overlay of zoomed ratio plots: hCentrality_Trig14, hCentralityZ150_Trig14, and hCentralityZOuter_Trig14
            if "hCentrality_Trig14" in cent_ratios and "hCentralityZ150_Trig14" in cent_ratios and "hCentralityZOuter_Trig14" in cent_ratios:
                r_t14, e_t14, t_t14 = cent_ratios["hCentrality_Trig14"]
                r_z150_t14, e_z150_t14, t_z150_t14 = cent_ratios["hCentralityZ150_Trig14"]
                r_zout_t14, e_zout_t14, t_zout_t14 = cent_ratios["hCentralityZOuter_Trig14"]

                label_t14 = t_t14 if t_t14 else "|z| < 10 cm and Trig 14"
                label_t14 = re.sub(r'MBD N&S\s*>=\s*2.*', 'Trig 14', label_t14).strip()
                label_z150_t14 = t_z150_t14 if t_z150_t14 else "|z| < 150 cm and Trig 14"
                label_z150_t14 = re.sub(r'MBD N&S\s*>=\s*2.*', 'Trig 14', label_z150_t14).strip()
                label_zout_t14 = t_zout_t14 if t_zout_t14 else "10 cm < |z| < 150 cm and Trig 14"
                label_zout_t14 = re.sub(r'MBD N&S\s*>=\s*2.*', 'Trig 14', label_zout_t14).strip()

                output_path_z_trig14_overlay = dir_overlay / f"run_{run_number}_hCentrality_Trig14_Z150_ZOuter_zoom_ratio.png"
                make_ratio_overlay_plot(
                    [
                        (r_t14, e_t14, label_t14, "blue"),
                        (r_z150_t14, e_z150_t14, label_z150_t14, "crimson"),
                        (r_zout_t14, e_zout_t14, label_zout_t14, "forestgreen"),
                    ],
                    run_number,
                    output_path_z_trig14_overlay,
                    xlabel="Centrality [%]",
                    ylabel="Ratio to Average",
                    xlim=(-0.5, 10.5),
                    hline=1.0,
                )
                plots_made += 1

            # 3. 1D Z vertex plot (full X projection of h2ZVertexCentrality)
            if "h2ZVertexCentrality" in file:
                hist2d = file["h2ZVertexCentrality"]
                values_2d, edges_x, _ = hist2d.to_numpy()
                proj_x = np.sum(values_2d, axis=1)
                total_zvtx = np.sum(proj_x)

                extra_labels_zvtx = ["MB", f"Total: {total_zvtx:.2e}"]
                output_path_zvtx = dir_zvtx / f"run_{run_number}_z_vertex.png"
                make_1d_plot(
                    (proj_x, edges_x),
                    run_number,
                    output_path_zvtx,
                    xlabel=r"$z_{\mathrm{vtx}}$ [cm]",
                    ylabel="Events",
                    extra_labels=extra_labels_zvtx,
                    logy=logy,
                    xlim=(edges_x[0], edges_x[-1]),
                )

                # 4. 1x3 panel Z vertex 1D projections for centrality slices 1%, 2%, 3%
                output_path_slices = dir_zvtx / f"run_{run_number}_z_vertex_cent_slices.png"
                make_zvertex_cent_slices_plot(
                    hist2d,
                    run_number,
                    output_path_slices,
                    slices=(1, 2, 3),
                    logy=logy,
                )

                plots_made += 1
            else:
                print(f"Warning: 'h2ZVertexCentrality' not found in {path}")

            if plots_made == 0:
                return f"Warning: No valid QA histograms found in {path}"

        return None
    except Exception as e:
        traceback.print_exc()
        return f"Error processing {path}: {e}"

def main():
    parser = argparse.ArgumentParser(description="Plot Centrality QA (Centrality and Z-Vertex) 1D histograms per run in parallel.")
    parser.add_argument("-f", "--file", type=Path, help="Path to a text file containing ROOT file paths (one per line).")
    parser.add_argument("-o", "--output-dir", type=Path, default=Path("plots"), help="Directory to save the plots (default: plots).")
    parser.add_argument("--logy", action="store_true", help="Use log scale for y-axis.")
    parser.add_argument("--run-subdirs", action="store_true", help="Save plots in run-numbered subdirectories.")
    parser.add_argument("-j", "--workers", type=int, default=None, help="Number of parallel worker processes (default: min(os.cpu_count(), 32)).")
    parser.add_argument("--cent-flat-min", type=float, default=10.0, help="Minimum centrality [%] for computing flat average (default: 10.0).")
    parser.add_argument("--cent-flat-max", type=float, default=70.0, help="Maximum centrality [%] for computing flat average (default: 70.0).")
    parser.add_argument("files", nargs="*", type=Path, help="List of ROOT file paths.")
    args = parser.parse_args()

    file_list = []
    if args.files:
        file_list.extend(args.files)

    if args.file:
        try:
            with args.file.open('r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        file_list.append(Path(line))
        except Exception as e:
            print(f"Error reading file {args.file}: {e}")
            sys.exit(1)

    if not file_list:
        print("Error: You must provide at least one ROOT file or a text file containing ROOT file paths (-f / --file).")
        parser.print_help()
        sys.exit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Found {len(file_list)} input files. Generating Centrality QA plots...")
    print(f"Saving plots to: {args.output_dir.resolve()}")

    files_to_process = [Path(p) for p in file_list]
    process_func = functools.partial(
        process_file,
        output_dir=args.output_dir,
        logy=args.logy,
        run_subdirs=args.run_subdirs,
        cent_flat_min=args.cent_flat_min,
        cent_flat_max=args.cent_flat_max,
    )

    max_workers = args.workers if args.workers is not None else min(os.cpu_count() or 4, 32)
    print(f"Running with {max_workers} worker processes...")

    errors = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = list(tqdm.tqdm(executor.map(process_func, files_to_process), total=len(files_to_process)))

    for path, err in zip(files_to_process, results):
        if err:
            print(err)
            errors.append(err)

    if not errors:
        print(f"Successfully processed all {len(file_list)} files.")
        print(f"Plots saved to {args.output_dir.resolve()}")
    else:
        print(f"Completed with {len(errors)} error(s) out of {len(file_list)} files.")
        print(f"Plots saved to {args.output_dir.resolve()}")

if __name__ == "__main__":
    main()
