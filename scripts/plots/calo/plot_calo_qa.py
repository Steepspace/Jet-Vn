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

from matplotlib.ticker import LogLocator, ScalarFormatter, FormatStrFormatter
import matplotlib.ticker as ticker
from matplotlib.colors import LogNorm
from matplotlib.patches import Patch
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Add script directory to sys.path to allow importing tower_info_defs
_script_dir = Path(__file__).resolve().parent
if str(_script_dir) not in sys.path:
    sys.path.insert(0, str(_script_dir))

from tower_info_defs import (
    get_bad_tower_map,
    get_calo_tower_ieta_iphi,
    get_calo_tower_key,
    get_frac_bad_chi2_map,
)

def clean_root_latex(text):
    if not text:
        return ""
    text = text.strip()
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

def get_hist_axis_titles(hist2d, hist_name=""):
    raw_title = ""
    xlabel = ""
    ylabel = ""

    # 1. Try uproot all_members dict
    if hasattr(hist2d, "all_members"):
        members = hist2d.all_members
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
                val = getattr(hist2d.axis(0), attr, None)
                if val:
                    xlabel = str(val).strip()
                    break
            except Exception:
                pass

    if not ylabel:
        for attr in ["label", "title"]:
            try:
                val = getattr(hist2d.axis(1), attr, None)
                if val:
                    ylabel = str(val).strip()
                    break
            except Exception:
                pass

    # 3. Check if raw_title or hist2d.title is formatted as "Title;XTitle;YTitle"
    if not raw_title:
        raw_title = getattr(hist2d, "title", "") or ""

    if ";" in raw_title:
        parts = [p.strip() for p in raw_title.split(";")]
        if not xlabel and len(parts) > 1:
            xlabel = parts[1]
        if not ylabel and len(parts) > 2:
            ylabel = parts[2]

    xlabel = clean_root_latex(xlabel)
    ylabel = clean_root_latex(ylabel)

    return xlabel, ylabel

def make_2d_plot(hist2d, run_number, output_path, hist_name="", xlim_left=None, xlim_right=None, ylim_bottom=None, ylim_top=None, extra_label=None, logx=False, logy=False):
    hep.style.use("ATLAS")
    fig, ax = plt.subplots(figsize=(8, 6))

    values, xedges, yedges = hist2d.to_numpy()

    # For the linear (no log) plot of h2EMCalChi2Energy, use customized variable y-binning
    # so higher points have wider bin widths and are clearly visible.
    plot_yedges = yedges
    plot_values = values
    if not logx and not logy and hist_name == "h2EMCalChi2Energy":
        new_yedges = np.unique(np.concatenate([
            np.arange(0, 10000, 100),
            np.arange(10000, 30000, 500),
            np.arange(30000, 60000, 1500),
            np.arange(60000, 100000 + 4000, 4000)
        ]))
        y_centers = (yedges[:-1] + yedges[1:]) / 2.0
        bin_idx = np.digitize(y_centers, new_yedges) - 1
        n_new_y = len(new_yedges) - 1
        rebinned_values = np.zeros((values.shape[0], n_new_y), dtype=values.dtype)
        for k in range(n_new_y):
            mask = (bin_idx == k)
            if np.any(mask):
                rebinned_values[:, k] = np.sum(values[:, mask], axis=1)
        plot_yedges = new_yedges
        plot_values = rebinned_values

    xlabel, ylabel = get_hist_axis_titles(hist2d, hist_name)
    if not xlabel and hist_name == "h2EMCalChi2Energy":
        xlabel = r"Tower Energy [ADC]"
    if not ylabel and hist_name == "h2EMCalChi2Energy":
        ylabel = r"$\chi^{2}$"

    max_val = np.max(plot_values) if plot_values.size > 0 else 0
    if max_val <= 0:
        mesh = ax.pcolormesh(xedges, plot_yedges, plot_values.T, cmap='viridis', rasterized=True, zorder=1)
    else:
        values_masked = np.ma.masked_where(plot_values <= 0, plot_values)
        norm = LogNorm(vmin=1, vmax=max(max_val, 10))
        mesh = ax.pcolormesh(xedges, plot_yedges, values_masked.T, norm=norm, cmap='viridis', rasterized=True, zorder=1)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(mesh, cax=cax)
    cbar.set_label("Counts")

    if xlabel:
        ax.set_xlabel(xlabel, loc='center')
    if ylabel:
        ax.set_ylabel(ylabel, labelpad=2)

    if logx:
        ax.set_xscale('log')
        ax.xaxis.set_major_locator(LogLocator(base=10.0, numticks=20))
        if xlim_left is None or xlim_left <= 0:
            xlim_left = 1.0
        if xlim_right is None:
            xlim_right = np.max(xedges)

    if logy:
        ax.set_yscale('log')
        ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=20))
        if ylim_bottom is None or ylim_bottom <= 0:
            ylim_bottom = 1.0
        if ylim_top is None:
            ylim_top = np.max(yedges)

    if xlim_left is not None or xlim_right is not None:
        ax.set_xlim(left=xlim_left, right=xlim_right)
    if ylim_bottom is not None or ylim_top is not None:
        ax.set_ylim(bottom=ylim_bottom, top=ylim_top)

    if hist_name == "h2EMCalChi2Energy":
        x_min = max(xlim_left, 1.0) if (logx and xlim_left is not None) else (xlim_left if xlim_left is not None else np.min(xedges))
        x_max = xlim_right if xlim_right is not None else np.max(xedges)
        y_max = ylim_top if ylim_top is not None else np.max(yedges)

        x_vals = np.geomspace(x_min, x_max, 1000) if logx else np.linspace(x_min, x_max, 1000)
        badChi2_threshold_const = 1e4
        badChi2_threshold_quadratic = 0.01
        badChi2_threshold_max = 1e8
        y_cut = np.minimum(np.maximum(badChi2_threshold_const, (x_vals ** 2) * badChi2_threshold_quadratic), badChi2_threshold_max)

        ax.fill_between(x_vals, y_cut, y_max, where=(y_max >= y_cut), color='red', alpha=0.3, zorder=3)
        ax.plot(x_vals, y_cut, color='red', linestyle='--', linewidth=1.5, zorder=4)

        # Compute bad chi2 tower count and fraction over total
        x_centers = (xedges[:-1] + xedges[1:]) / 2.0
        y_centers = (yedges[:-1] + yedges[1:]) / 2.0
        X_grid, Y_grid = np.meshgrid(x_centers, y_centers, indexing='ij')
        threshold_grid = np.minimum(np.maximum(badChi2_threshold_const, (X_grid ** 2) * badChi2_threshold_quadratic), badChi2_threshold_max)
        bad_mask = Y_grid > threshold_grid
        bad_towers = np.sum(values[bad_mask])
        total_towers = np.sum(values)
        pct = (bad_towers / total_towers) * 100.0 if total_towers > 0 else 0.0

        patch = Patch(facecolor=(1, 0, 0, 0.3), edgecolor='red', linewidth=1.5, linestyle='--',
                      label=r"$\chi^{2} > \min(\max(10^{4}, 0.01 \cdot \mathrm{ADC}^{2}), 10^{8})$")
        if logx and logy:
            ax.legend(handles=[patch], loc='upper left', frameon=True, facecolor='white', edgecolor='none', framealpha=0.8, fontsize=12)
        else:
            ax.legend(handles=[patch], loc='lower right', bbox_to_anchor=(1.03, -0.02), frameon=True, facecolor='none', edgecolor='none', fontsize=12)

        if not logx:
            info_text = (
                f"Total Towers: {total_towers:.2e}\n"
                f"Bad $\\chi^{{2}}$ Towers: {bad_towers:.2e}\n"
                f"Bad $\\chi^{{2}}$ Percentage: {pct:.2f}%"
            )
            ax.text(0.98, 0.08, info_text, transform=ax.transAxes, ha='right', va='bottom', fontsize=13,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='none'))

    if not logx and np.max(xedges) >= 1000:
        formatter_x = ScalarFormatter(useMathText=True)
        formatter_x.set_powerlimits((3, 3))
        ax.xaxis.set_major_formatter(formatter_x)

    if not logy and np.max(yedges) >= 1000:
        formatter_y = ScalarFormatter(useMathText=True)
        formatter_y.set_powerlimits((3, 3))
        ax.yaxis.set_major_formatter(formatter_y)

    ax.text(1.0, 1.01, rf"Run: {run_number}", transform=ax.transAxes, ha='right', va='bottom', fontsize=15)

    if extra_label:
        ax.text(0.95, 0.95, extra_label, transform=ax.transAxes, ha='right', va='top', fontsize=15, color='white', bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.4, edgecolor='none'))

    fig.tight_layout()
    plt.subplots_adjust(left=0.12, bottom=0.13, top=0.93)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)

def make_1d_proj_plot(hist2d, run_number, output_path, hist_name="", logy=True, rebin_x=20):
    hep.style.use("ATLAS")
    fig, ax = plt.subplots(figsize=(8, 6))

    values, xedges, yedges = hist2d.to_numpy()
    proj_x = np.sum(values, axis=1)

    if rebin_x > 1 and len(proj_x) >= rebin_x:
        if len(proj_x) % rebin_x == 0:
            proj_x = proj_x.reshape(-1, rebin_x).sum(axis=1)
            xedges = xedges[::rebin_x]
        else:
            n_bins = (len(proj_x) // rebin_x) * rebin_x
            proj_x = proj_x[:n_bins].reshape(-1, rebin_x).sum(axis=1)
            xedges = xedges[:n_bins + 1:rebin_x]

    xlabel, _ = get_hist_axis_titles(hist2d, hist_name)
    if not xlabel:
        xlabel = r"Tower Energy [GeV]"

    hep.histplot((proj_x, xedges), ax=ax, histtype='step', color='navy', linewidth=2)

    if xlabel:
        ax.set_xlabel(xlabel, loc='center')
    ax.set_ylabel("Counts", loc='center')

    if logy:
        ax.set_yscale('log')
        ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=20))
        max_val = np.max(proj_x) if proj_x.size > 0 else 1
        ax.set_ylim(bottom=0.5, top=max(max_val * 5, 10))

    ax.set_xlim(left=np.min(xedges), right=np.max(xedges))

    ax.text(1.0, 1.01, rf"Run: {run_number}", transform=ax.transAxes, ha='right', va='bottom', fontsize=15)

    det = "OHCal" if "OHCal" in hist_name else ("EMCal" if "EMCal" in hist_name else "")
    if det:
        ax.text(0.05, 0.95, f"{det} ZS Towers", transform=ax.transAxes, ha='left', va='top', fontsize=15)

    fig.tight_layout()
    plt.subplots_adjust(left=0.12, bottom=0.13, top=0.93)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)

def make_1d_zs_ratio_plot(hist2d_zs, hist2d_total, run_number, output_path, hist_name="EMCal", use_cent_denom=False):
    hep.style.use("ATLAS")
    fig, ax = plt.subplots(figsize=(8, 6))

    zs_vals, zs_edges_e, _ = hist2d_zs.to_numpy()
    zs_e = np.sum(zs_vals, axis=1)

    if use_cent_denom:
        # hist2d_total is h2EMCalCent: axis 0 is energy (80 bins, 0.5 GeV width [-10, 30])
        tot_vals, tot_edges_e, _ = hist2d_total.to_numpy()
        tot_e = np.sum(tot_vals, axis=1)
        bin_width = 0.5
        rebin_factor = 50
        e_lows = np.arange(-10.0, 0.0, bin_width)
        valid_e = []
        valid_r = []
        valid_diff = []
        valid_tot = []
        valid_fail = []

        zs_e_rebin = zs_e.reshape(-1, rebin_factor).sum(axis=1)
        for i, e in enumerate(e_lows):
            c_tot = tot_e[i]
            c_zs = zs_e_rebin[i]
            if c_tot > 0:
                r = c_zs / c_tot
                valid_e.append(e)
                valid_r.append(r)
                valid_diff.append(r - 1.0)
                valid_tot.append(c_tot)
                valid_fail.append(max(0.0, c_tot - c_zs))
    else:
        # hist2d_total is h2EMCalEnergyTowerIndex: axis 1 is energy (350 bins, 1.0 GeV width [-150, 200])
        tot_vals, _, tot_edges_e = hist2d_total.to_numpy()
        tot_e = np.sum(tot_vals, axis=0)
        bin_width = 1.0
        e_lows = np.arange(-10.0, 0.0, bin_width)
        valid_e = []
        valid_r = []
        valid_diff = []
        valid_tot = []
        valid_fail = []

        for e in e_lows:
            i_tot_arr = np.where(np.isclose(tot_edges_e[:-1], e))[0]
            if len(i_tot_arr) == 0:
                continue
            i_tot = i_tot_arr[0]
            c_tot = tot_e[i_tot]

            i_zs_start_arr = np.where(np.isclose(zs_edges_e[:-1], e))[0]
            i_zs_end_arr = np.where(np.isclose(zs_edges_e, e + bin_width))[0]
            if len(i_zs_start_arr) == 0 or len(i_zs_end_arr) == 0:
                continue
            c_zs = np.sum(zs_e[i_zs_start_arr[0]:i_zs_end_arr[0]])

            if c_tot > 0:
                r = c_zs / c_tot
                valid_e.append(e)
                valid_r.append(r)
                valid_diff.append(r - 1.0)
                valid_tot.append(c_tot)
                valid_fail.append(max(0.0, c_tot - c_zs))

    valid_e = np.array(valid_e)
    valid_r = np.array(valid_r)
    valid_diff = np.array(valid_diff)
    valid_tot = np.array(valid_tot)
    valid_fail = np.array(valid_fail)

    # Dashed reference line at 1.0
    ax.axhline(1.0, color='crimson', linestyle='--', linewidth=1.5, zorder=2)

    # Step plot: draw horizontal segments and vertical transitions between adjacent valid bins
    for i in range(len(valid_e)):
        e = valid_e[i]
        r = valid_r[i]
        ax.plot([e, e + bin_width], [r, r], color='navy', linewidth=2.5, zorder=3)
        if i + 1 < len(valid_e) and np.isclose(valid_e[i+1], e + bin_width):
            ax.plot([e + bin_width, e + bin_width], [r, valid_r[i+1]], color='navy', linewidth=2.5, zorder=3)

    # Markers at bin centers
    ax.plot(valid_e + bin_width / 2.0, valid_r, 'o', color='navy', markersize=6, zorder=4)

    ax.set_xlabel('Tower Energy [GeV]', loc='center')
    ax.set_ylabel(r'$N_{\mathrm{ZS}} \,/\, N_{\mathrm{Total}}$', loc='center')
    ax.set_xlim(-10, 0)

    # Proper zoom logic
    max_dev = np.max(np.abs(valid_r - 1.0)) if len(valid_r) > 0 else 0.0
    if max_dev < 1e-12:
        ax.set_ylim(0.99, 1.01)
    else:
        pad = max(max_dev * 0.25, 1e-5)
        bottom = min(valid_r.min() - pad, 1.0 - 1.25 * max_dev)
        top = max(valid_r.max() + pad, 1.0 + 0.25 * max_dev)
        ax.set_ylim(bottom=bottom, top=top)

        if max_dev < 0.005:
            ax.yaxis.set_major_formatter(FormatStrFormatter('%.5f'))
        elif max_dev < 0.05:
            ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
        else:
            ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    ax.text(1.0, 1.01, rf'Run: {run_number}', transform=ax.transAxes, ha='right', va='bottom', fontsize=15)
    ax.text(0.04, 0.94, f'{hist_name}', transform=ax.transAxes, ha='left', va='top', fontsize=14, fontweight='bold')

    # Show summary of deviating bins placed in lower-left area
    dev_str_list = []
    for e, r, cf in zip(valid_e, valid_r, valid_fail):
        dev = 1.0 - r
        if abs(dev) > 1e-6:
            dev_str_list.append(f'[{e:g}, {e+bin_width:g}] GeV: {dev:.2e} (N_fail={cf:.2e})')

    if dev_str_list:
        dev_box_text = 'Deviation (1.0 - Ratio):\n' + '\n'.join(dev_str_list)
        ax.text(0.04, 0.05, dev_box_text, transform=ax.transAxes, ha='left', va='bottom', fontsize=16,
                bbox=dict(boxstyle='round,pad=0.4', facecolor='whitesmoke', alpha=0.9, edgecolor='darkgray'))

    fig.tight_layout()
    plt.subplots_adjust(left=0.18, bottom=0.13, top=0.93)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)

def make_1d_yproj_plot(hist2d, run_number, output_path, hist_name="", tower_index=None, exclude_towers=None, label_text=None, z_score=None, frac_bad_chi2=None, logy=True, auto_xlim=None, ref_tower_index=None, ref_z_score=None, ref_frac_bad_chi2=None):
    hep.style.use("ATLAS")
    fig, ax = plt.subplots(figsize=(8, 6))

    values, xedges, yedges = hist2d.to_numpy()

    tower_info_lines = []
    proj_y_ref = None
    if ref_tower_index is not None:
        if 0 <= ref_tower_index < values.shape[0]:
            proj_y_ref = values[ref_tower_index, :]
        else:
            print(f"Warning: Reference tower index {ref_tower_index} out of bounds (0, {values.shape[0]})")

    if tower_index is not None:
        if 0 <= tower_index < values.shape[0]:
            proj_y = values[tower_index, :]
        else:
            print(f"Warning: Tower index {tower_index} out of bounds (0, {values.shape[0]})")
            plt.close(fig)
            return
        if label_text is None:
            tower_info_lines.append(f"Tower Index: {tower_index}")
            ieta, iphi = get_calo_tower_ieta_iphi(tower_index, hist_name)
            if ieta is not None and iphi is not None:
                tower_info_lines.append(rf"$i\eta$: {ieta}, $i\phi$: {iphi}")
            if z_score is not None and not (isinstance(z_score, float) and np.isnan(z_score)):
                tower_info_lines.append(f"z-score: {z_score:+.2f}")
            if frac_bad_chi2 is not None and not (isinstance(frac_bad_chi2, float) and np.isnan(frac_bad_chi2)):
                if 0 < abs(frac_bad_chi2) < 0.01:
                    frac_str = f"{frac_bad_chi2:.2e}"
                else:
                    frac_str = f"{frac_bad_chi2:.2f}"
                tower_info_lines.append(f"frac badChi2: {frac_str}")
        else:
            tower_info_lines.append(label_text)
    elif exclude_towers is not None and len(exclude_towers) > 0:
        valid_excludes = [t for t in exclude_towers if 0 <= t < values.shape[0]]
        if len(valid_excludes) > 0:
            mask = np.ones(values.shape[0], dtype=bool)
            mask[valid_excludes] = False
            proj_y = np.sum(values[mask, :], axis=0)
            if label_text is None:
                if len(valid_excludes) == 1:
                    label_text = f"All Good Towers (Excl. Tower {valid_excludes[0]})"
                elif len(valid_excludes) <= 3:
                    label_text = f"All Good Towers (Excl. Towers {', '.join(map(str, valid_excludes))})"
                else:
                    label_text = f"All Good Towers (Excl. {len(valid_excludes)} Outliers)"
        else:
            proj_y = np.sum(values, axis=0)
            if label_text is None:
                label_text = "All Good Towers"
    else:
        proj_y = np.sum(values, axis=0)
        if label_text is None:
            label_text = "All Good Towers"

    _, ylabel = get_hist_axis_titles(hist2d, hist_name)
    if not ylabel:
        ylabel = r"Raw Tower Energy [ADC]" if "Raw" in hist_name else r"Tower Energy [GeV]"

    if proj_y_ref is not None:
        def format_overlay_legend_label(prefix, t_idx, z_val, f_val):
            ieta, iphi = get_calo_tower_ieta_iphi(t_idx, hist_name)
            line1 = f"{prefix}: Tower {t_idx}"
            if ieta is not None and iphi is not None:
                line1 += f" (ieta: {ieta}, iphi: {iphi})"
            line2_parts = []
            if z_val is not None and not (isinstance(z_val, float) and np.isnan(z_val)):
                line2_parts.append(f"z-score: {z_val:+.2f}")
            if f_val is not None and not (isinstance(f_val, float) and np.isnan(f_val)):
                c_str = f"{f_val:.2e}" if 0 < abs(f_val) < 0.01 else f"{f_val:.2f}"
                line2_parts.append(f"frac badChi2: {c_str}")
            if line2_parts:
                return line1 + "\n  " + ", ".join(line2_parts)
            return line1

        label_out = format_overlay_legend_label("Outlier", tower_index, z_score, frac_bad_chi2)
        label_ref = format_overlay_legend_label("Ref", ref_tower_index, ref_z_score, ref_frac_bad_chi2)

        hep.histplot((proj_y, yedges), ax=ax, histtype='step', color='crimson', linewidth=2, label=label_out)
        hep.histplot((proj_y_ref, yedges), ax=ax, histtype='step', color='navy', linewidth=2, linestyle='--', label=label_ref)
    else:
        hep.histplot((proj_y, yedges), ax=ax, histtype='step', color='navy', linewidth=2)

    ax.set_xlabel(ylabel, loc='center')
    ax.set_ylabel("Counts", loc='center')

    if logy:
        ax.set_yscale('log')
        ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=20))
        all_projs = [proj_y]
        if proj_y_ref is not None:
            all_projs.append(proj_y_ref)
        max_val = max(np.max(p) if p.size > 0 else 1 for p in all_projs)
        top_mult = 30 if proj_y_ref is not None else 5
        ax.set_ylim(bottom=0.5, top=max(max_val * top_mult, 10))

    if auto_xlim is None:
        auto_xlim = ("h2EMCalEnergyTowerIndex" in hist_name and "Zoom" not in hist_name)

    if auto_xlim:
        comb_y = proj_y + proj_y_ref if proj_y_ref is not None else proj_y
        nonzero = np.where(comb_y > 0)[0]
        if len(nonzero) > 0:
            xmin = float(yedges[nonzero[0]])
            xmax = float(yedges[nonzero[-1] + 1])
            span = xmax - xmin
            padding = max(span * 0.05, 1.0)
            left = max(xmin - padding, float(np.min(yedges)))
            right = min(xmax + padding, float(np.max(yedges)))
            ax.set_xlim(left=left, right=right)
        else:
            ax.set_xlim(left=np.min(yedges), right=np.max(yedges))
    else:
        ax.set_xlim(left=np.min(yedges), right=np.max(yedges))

    ax.text(1.0, 1.01, rf"Run: {run_number}", transform=ax.transAxes, ha='right', va='bottom', fontsize=15)

    if proj_y_ref is not None:
        # Check overlap for legend placement (upper right vs upper left)
        x_min, x_max = ax.get_xlim()
        y_min, y_max = ax.get_ylim()
        overlap_right = False
        overlap_left = False
        for y_arr in [proj_y, proj_y_ref]:
            for i in range(len(y_arr)):
                if y_arr[i] <= 0:
                    continue
                bx = (0.5 * (yedges[i] + yedges[i + 1]) - x_min) / (x_max - x_min) if (x_max - x_min) != 0 else 0
                if logy:
                    by = (np.log10(y_arr[i]) - np.log10(y_min)) / (np.log10(y_max) - np.log10(y_min)) if y_min > 0 and y_max > y_min else 0
                else:
                    by = (y_arr[i] - y_min) / (y_max - y_min) if (y_max - y_min) != 0 else 0
                if by >= 0.60:
                    if bx >= 0.50:
                        overlap_right = True
                    if bx <= 0.50:
                        overlap_left = True

        legend_loc = 'upper left' if (overlap_right and not overlap_left) else 'upper right'
        ax.legend(frameon=True, facecolor='white', edgecolor='lightgray', fontsize=11, loc=legend_loc)
    elif tower_index is not None and tower_info_lines:
        tower_info_text = "\n".join(tower_info_lines)
        text_obj = ax.text(0.95, 0.95, tower_info_text, transform=ax.transAxes, ha='right', va='top', fontsize=16, multialignment='left')

        # If text is overlapped by data in the top-right, move it to the top-left (inside plot)
        try:
            renderer = fig.canvas.get_renderer()
            bbox_axes = ax.transAxes.inverted().transform(text_obj.get_window_extent(renderer=renderer))
            x_min, x_max = ax.get_xlim()
            y_min, y_max = ax.get_ylim()
            tx0 = bbox_axes[0, 0] - 0.02
            tx1 = bbox_axes[1, 0] + 0.02
            ty0 = bbox_axes[0, 1] - 0.02

            overlap = False
            for i in range(len(proj_y)):
                if proj_y[i] <= 0:
                    continue
                bx0 = (yedges[i] - x_min) / (x_max - x_min) if (x_max - x_min) != 0 else 0
                bx1 = (yedges[i + 1] - x_min) / (x_max - x_min) if (x_max - x_min) != 0 else 0
                b_left = min(bx0, bx1)
                b_right = max(bx0, bx1)
                if logy:
                    if proj_y[i] > 0 and y_min > 0 and y_max > y_min:
                        by = (np.log10(proj_y[i]) - np.log10(y_min)) / (np.log10(y_max) - np.log10(y_min))
                    else:
                        by = 0
                else:
                    by = (proj_y[i] - y_min) / (y_max - y_min) if (y_max - y_min) != 0 else 0

                if b_right >= tx0 and b_left <= tx1 and by >= ty0:
                    overlap = True
                    break

            if overlap:
                text_obj.set_position((0.05, 0.95))
                text_obj.set_ha('left')
        except Exception:
            pass
    elif label_text:
        if len(label_text) > 42:
            fs = 13
        elif len(label_text) > 30:
            fs = 15
        else:
            fs = 18
        ax.text(0.03, 1.01, label_text, transform=ax.transAxes, ha='left', va='bottom', fontsize=fs)

    fig.tight_layout()
    plt.subplots_adjust(left=0.12, bottom=0.13, top=0.93)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)

class FracBadChi2ScalarFormatter(ScalarFormatter):
    """
    Custom ScalarFormatter for frac badChi2 that shifts order of magnitude from 10^-4 to 10^-5
    if 10^-4 happens to be chosen by default.
    """
    def _set_order_of_magnitude(self):
        super()._set_order_of_magnitude()
        if self._orderOfMagnitude == -4:
            self._orderOfMagnitude = -5

def make_1d_cdb_branch_plot(
    values,
    run_number,
    output_path,
    branch_name,
    xlabel=None,
    outlier_entries=None,
    bins=100,
    logy=True,
    x_max_cutoff=None,
):
    hep.style.use("ATLAS")
    fig, ax = plt.subplots(figsize=(8, 6))

    clean_values = np.asarray(values, dtype=float)
    clean_values = clean_values[~np.isnan(clean_values)]
    if len(clean_values) == 0:
        plt.close(fig)
        return

    if x_max_cutoff is not None:
        x_min = float(np.min(clean_values))
        x_max = float(x_max_cutoff)
        if outlier_entries:
            visible_outliers = []
            for entry in outlier_entries:
                v = entry[1]
                if x_min <= v <= x_max:
                    visible_outliers.append(entry)
            outlier_entries = visible_outliers
    else:
        x_min = float(np.min(clean_values))
        x_max = float(np.max(clean_values))
        if outlier_entries:
            for entry in outlier_entries:
                v = entry[1]
                x_min = min(x_min, float(v))
                x_max = max(x_max, float(v))

    if x_min == x_max:
        bin_edges = np.linspace(x_min - 1.0, x_max + 1.0, bins + 1)
    else:
        bin_edges = np.linspace(x_min, x_max, bins + 1)

    counts, _ = np.histogram(clean_values, bins=bin_edges)
    hep.histplot((counts, bin_edges), ax=ax, histtype='step', color='navy', linewidth=2, label="All Towers")

    if xlabel is None:
        if branch_name == "FCEMC_sigma":
            xlabel = "Z-score"
        elif branch_name == "Ffraction":
            xlabel = "frac badChi2"
        else:
            xlabel = branch_name

    ax.set_xlabel(xlabel, loc='center')
    ax.set_ylabel("Counts", loc='center')

    if logy:
        ax.set_yscale('log')
        ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=20))
        max_val = np.max(counts) if counts.size > 0 else 1
        ax.set_ylim(bottom=0.5, top=max(max_val * 5, 10))

    span = bin_edges[-1] - bin_edges[0]
    if span > 0:
        ax.set_xlim(left=bin_edges[0] - 0.03 * span, right=bin_edges[-1] + 0.03 * span)

    ax.text(1.0, 1.01, rf"Run: {run_number}", transform=ax.transAxes, ha='right', va='bottom', fontsize=15)

    # For Z-score plot, shade z-score < -5 in light blue and z-score > 5 in light red
    if branch_name == "FCEMC_sigma" or xlabel == "Z-score":
        x_left, x_right = ax.get_xlim()
        n_low = int(np.count_nonzero(clean_values < -5))
        n_high = int(np.count_nonzero(clean_values > 5))
        if x_left < -5:
            ax.axvspan(x_left, -5, color='dodgerblue', alpha=0.15, zorder=0, label=rf"$Z < -5$ (N={n_low})")
        if x_right > 5:
            ax.axvspan(5, x_right, color='red', alpha=0.15, zorder=0, label=rf"$Z > 5$ (N={n_high})")

    # For frac badChi2 plot, shade region > 0.01 in light red if x-axis goes above 0.01
    if branch_name == "Ffraction" or xlabel == "frac badChi2":
        x_left, x_right = ax.get_xlim()
        if x_right > 0.01:
            n_high = int(np.count_nonzero(clean_values > 0.01))
            ax.axvspan(max(0.01, x_left), x_right, color='red', alpha=0.15, zorder=0, label=rf"$> 0.01$ (N={n_high})")
        fmt = FracBadChi2ScalarFormatter(useMathText=True)
        ax.xaxis.set_major_formatter(fmt)

    # Vertical lines for outlier towers
    if outlier_entries:
        colors = ['red', 'darkorange', 'purple', 'magenta', 'cyan', 'green']
        if len(outlier_entries) <= 6:
            for idx, entry in enumerate(outlier_entries):
                color = colors[idx % len(colors)]
                if len(entry) >= 4:
                    t, val, ieta, iphi = entry[:4]
                else:
                    t, val = entry[0], entry[1]
                    ieta, iphi = get_calo_tower_ieta_iphi(t, "EMCal")

                if ieta is not None and iphi is not None:
                    tower_str = rf"$i\eta$: {ieta}, $i\phi$: {iphi}"
                else:
                    tower_str = f"Tower {t}"

                if (branch_name == "Ffraction" or xlabel == "frac badChi2") and 0 < abs(val) < 0.01:
                    v_str = f"{val:.2e}"
                elif branch_name == "Ffraction" or xlabel == "frac badChi2":
                    v_str = f"{val:.2f}"
                else:
                    v_str = f"{val:+.2f}"
                ax.axvline(x=val, color=color, linestyle='--', linewidth=1.8, label=rf"{tower_str} ({v_str})")
        else:
            first = True
            for entry in outlier_entries:
                val = entry[1]
                lbl = f"Outliers (N={len(outlier_entries)})" if first else None
                ax.axvline(x=val, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label=lbl)
                first = False

    ax.legend(frameon=True, fontsize=12, loc='best')

    fig.tight_layout()
    plt.subplots_adjust(left=0.12, bottom=0.13, top=0.93)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)

def find_high_outlier_zoom_max(values):
    """
    Detect if there is a clear outlier at very high x-value compared to the average / bulk
    of the main distribution. Returns the suggested maximum x-value for zooming in, or None
    if no zoom is needed.
    """
    clean_values = np.asarray(values, dtype=float)
    clean_values = clean_values[~np.isnan(clean_values)]
    if len(clean_values) < 20:
        return None

    s_vals = np.sort(clean_values)
    min_val = s_vals[0]
    max_val = s_vals[-1]
    total_span = max_val - min_val
    if total_span <= 0:
        return None

    p995 = np.percentile(clean_values, 99.5)
    pos_vals = clean_values[clean_values > 0]
    if len(pos_vals) > 0 and p995 == 0:
        ref_bulk = np.percentile(pos_vals, 95)
    else:
        ref_bulk = p995

    if ref_bulk <= 0:
        return None

    # Ratio check: max must be at least 10x higher than bulk reference
    if max_val < 10.0 * ref_bulk:
        return None

    # Search for the jump in the upper tail (top 2% or up to 200 towers)
    n_tail = max(10, min(int(0.02 * len(s_vals)), 200))
    tail_vals = s_vals[-n_tail:]
    diffs = np.diff(tail_vals)

    max_gap_idx = np.argmax(diffs)
    max_gap = diffs[max_gap_idx]

    val_before = tail_vals[max_gap_idx]

    bulk_span = max(val_before - min_val, ref_bulk - min_val)

    # Gap must be substantial: at least 30% of total span or >= 5x bulk span
    if max_gap >= 0.30 * total_span or (bulk_span > 0 and max_gap >= 5.0 * bulk_span):
        return val_before

    # If the ratio is extreme (>20x) even if gaps are somewhat spaced
    if max_val >= 20.0 * ref_bulk:
        return ref_bulk * 1.15

    return None

def find_outlier_towers(hist2d, threshold):
    """
    Find tower indices that have counts with energy below the given threshold.
    """
    values, xedges, yedges = hist2d.to_numpy()
    # Mask bins where the upper bin edge is <= threshold
    mask = yedges[1:] <= threshold
    if not np.any(mask):
        y_centers = 0.5 * (yedges[:-1] + yedges[1:])
        mask = y_centers < threshold
    if not np.any(mask):
        return np.array([], dtype=int)

    counts_below = np.sum(values[:, mask], axis=1)
    return np.where(counts_below > 0)[0]

def process_file(
    path,
    output_dir=None,
    do_nolog=True,
    do_logy=True,
    do_logxy=True,
    do_logx=False,
    energy_threshold=-10.0,
    max_outlier_towers=50,
    use_cdb=True,
    cdbtag="newcdbtag",
    ref_tower=None,
):
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

        run_output_dir = None
        if output_dir is not None:
            run_output_dir = output_dir / f"{run_number}"
            run_output_dir.mkdir(parents=True, exist_ok=True)

        with uproot.open(path) as file:
            # 1. 2D QA Histograms
            h2_names = [
                "h2EMCalChi2Energy",
            ]

            plot_modes = []
            if do_nolog:
                plot_modes.append(("", False, False))
            if do_logy:
                plot_modes.append(("_logy", False, True))
            if do_logxy:
                plot_modes.append(("_logxy", True, True))
            if do_logx:
                plot_modes.append(("_logx", True, False))

            for h2_name in h2_names:
                if h2_name not in file:
                    print(f"Warning: {h2_name} not found in {path}")
                    continue

                hist2d = file[h2_name]

                if run_output_dir is not None:
                    for suffix, lx, ly in plot_modes:
                        out_filename = f"run_{run_number}_{h2_name}{suffix}.png"
                        output_path = run_output_dir / out_filename
                        xlim_left = 1.0 if lx else 0
                        ylim_bottom = 1.0 if ly else 0
                        make_2d_plot(
                            hist2d,
                            run_number,
                            output_path,
                            hist_name=h2_name,
                            xlim_left=xlim_left,
                            ylim_bottom=ylim_bottom,
                            logx=lx,
                            logy=ly
                        )

            # 2. 1D X-Projection QA Histograms
            h1_proj_names = [
                "h2EMCalZSCent",
                "h2OHCalZSCent",
            ]

            for h2_name in h1_proj_names:
                if h2_name not in file:
                    print(f"Warning: {h2_name} not found in {path}")
                    continue

                hist2d = file[h2_name]

                if run_output_dir is not None:
                    out_filename = f"run_{run_number}_{h2_name}.png"
                    output_path = run_output_dir / out_filename
                    make_1d_proj_plot(
                        hist2d,
                        run_number,
                        output_path,
                        hist_name=h2_name,
                        logy=True
                    )

            # 2b. 1D ZS Fraction Ratio Plot in Negative Energy Range
            if run_output_dir is not None and "h2EMCalZSCent" in file and "h2EMCalEnergyTowerIndex" in file:
                out_filename_ratio = f"run_{run_number}_EMCal_ZS_ratio.png"
                output_path_ratio = run_output_dir / out_filename_ratio
                make_1d_zs_ratio_plot(
                    file["h2EMCalZSCent"],
                    file["h2EMCalEnergyTowerIndex"],
                    run_number,
                    output_path_ratio,
                    hist_name="EMCal"
                )

            # OHCal ZS Fraction Ratio Plot in Negative Energy Range
            if run_output_dir is not None and "h2OHCalZSCent" in file and "h2OHCalCent" in file:
                out_filename_ohcal_ratio = f"run_{run_number}_OHCal_ZS_ratio.png"
                output_path_ohcal_ratio = run_output_dir / out_filename_ohcal_ratio
                make_1d_zs_ratio_plot(
                    file["h2OHCalZSCent"],
                    file["h2OHCalCent"],
                    run_number,
                    output_path_ohcal_ratio,
                    hist_name="OHCal",
                    use_cent_denom=True
                )

            # IHCal ZS Fraction Ratio Plot in Negative Energy Range
            if run_output_dir is not None and "h2IHCalZSCent" in file and "h2IHCalCent" in file:
                out_filename_ihcal_ratio = f"run_{run_number}_IHCal_ZS_ratio.png"
                output_path_ihcal_ratio = run_output_dir / out_filename_ihcal_ratio
                make_1d_zs_ratio_plot(
                    file["h2IHCalZSCent"],
                    file["h2IHCalCent"],
                    run_number,
                    output_path_ihcal_ratio,
                    hist_name="IHCal",
                    use_cent_denom=True
                )

            # 3. 1D Y-Projection QA Histograms for Tower Energy vs Index
            h2_energy_index_names = [
                "h2EMCalEnergyTowerIndex",
                "h2EMCalEnergyTowerIndexZS",
                "h2EMCalEnergyTowerIndexZoom",
                "h2EMCalRawEnergyTowerIndex",
            ]

            # Find any outlier towers with energy below threshold (only calibrated energy in GeV, not raw ADC or zoom)
            outlier_towers_set = set()
            for h2_energy_name in h2_energy_index_names:
                if "Raw" in h2_energy_name or "Zoom" in h2_energy_name:
                    continue
                if h2_energy_name in file:
                    towers = find_outlier_towers(file[h2_energy_name], energy_threshold)
                    if len(towers) > 0:
                        outlier_towers_set.update(towers.tolist())

            outlier_towers = sorted(outlier_towers_set)
            bad_tower_map = {}
            frac_bad_chi2_map = {}
            if use_cdb:
                bad_tower_map = get_bad_tower_map(run_number, det="CEMC", dbtag=cdbtag)
                frac_bad_chi2_map = get_frac_bad_chi2_map(run_number, det="CEMC", dbtag=cdbtag)

            ref_z_score = None
            ref_frac_bad_chi2 = None
            if ref_tower is not None and use_cdb:
                ref_key = get_calo_tower_key(ref_tower, det="EMCal")
                ref_z_score = bad_tower_map.get(ref_key, {}).get("sigma") if bad_tower_map else None
                ref_frac_bad_chi2 = frac_bad_chi2_map.get(ref_key) if frac_bad_chi2_map else None

            if len(outlier_towers) > 0:
                tower_desc = []
                for t in outlier_towers[:max_outlier_towers]:
                    ieta, iphi = get_calo_tower_ieta_iphi(t, "EMCal")
                    t_key = get_calo_tower_key(t, det="EMCal")
                    z_val = bad_tower_map.get(t_key, {}).get("sigma") if bad_tower_map else None
                    z_txt = f", z-score={z_val:+.2f}" if z_val is not None else ""
                    chi2_val = frac_bad_chi2_map.get(t_key) if frac_bad_chi2_map else None
                    if chi2_val is not None:
                        c_str = f"{chi2_val:.2e}" if 0 < abs(chi2_val) < 0.01 else f"{chi2_val:.2f}"
                        chi2_txt = f", frac badChi2={c_str}"
                    else:
                        chi2_txt = ""
                    if ieta is not None and iphi is not None:
                        tower_desc.append(f"{t} (ieta={ieta}, iphi={iphi}{z_txt}{chi2_txt})")
                    else:
                        tower_desc.append(f"{t}{z_txt}{chi2_txt}")
                print(f"[{path.name}] Found {len(outlier_towers)} outlier tower(s) with energy < {energy_threshold} GeV: {tower_desc}")
                if max_outlier_towers is not None and max_outlier_towers > 0 and len(outlier_towers) > max_outlier_towers:
                    print(f"[{path.name}] Limiting outlier tower 1D plots to first {max_outlier_towers} towers.")
                    outlier_towers = outlier_towers[:max_outlier_towers]

            # Generate 1D CDB plots (Z-score and frac badChi2) for the run
            if run_output_dir is not None:
                # 1D plot of FCEMC_sigma branch (Z-score)
                if bad_tower_map:
                    all_sigmas = [v["sigma"] for v in bad_tower_map.values() if v.get("sigma") is not None and not np.isnan(v["sigma"])]
                    if len(all_sigmas) > 0:
                        outlier_sigmas = []
                        for t in outlier_towers:
                            t_key = get_calo_tower_key(t, det="EMCal")
                            s_val = bad_tower_map.get(t_key, {}).get("sigma")
                            if s_val is not None and not np.isnan(s_val):
                                ieta, iphi = get_calo_tower_ieta_iphi(t, "EMCal")
                                outlier_sigmas.append((t, s_val, ieta, iphi))

                        out_filename_sigma = f"run_{run_number}_FCEMC_sigma.png"
                        output_path_sigma = run_output_dir / out_filename_sigma
                        make_1d_cdb_branch_plot(
                            all_sigmas,
                            run_number,
                            output_path_sigma,
                            branch_name="FCEMC_sigma",
                            xlabel="Z-score",
                            outlier_entries=outlier_sigmas if len(outlier_sigmas) > 0 else None,
                        )

                # 1D plot of Ffraction branch (frac badChi2)
                if frac_bad_chi2_map:
                    all_fracs = [v for v in frac_bad_chi2_map.values() if v is not None and not np.isnan(v)]
                    if len(all_fracs) > 0:
                        outlier_fracs = []
                        for t in outlier_towers:
                            t_key = get_calo_tower_key(t, det="EMCal")
                            f_val = frac_bad_chi2_map.get(t_key)
                            if f_val is not None and not np.isnan(f_val):
                                ieta, iphi = get_calo_tower_ieta_iphi(t, "EMCal")
                                outlier_fracs.append((t, f_val, ieta, iphi))

                        out_filename_frac = f"run_{run_number}_Ffraction.png"
                        output_path_frac = run_output_dir / out_filename_frac
                        make_1d_cdb_branch_plot(
                            all_fracs,
                            run_number,
                            output_path_frac,
                            branch_name="Ffraction",
                            xlabel="frac badChi2",
                            outlier_entries=outlier_fracs if len(outlier_fracs) > 0 else None,
                        )

                        # Check if a zoomed version on the x-axis is needed for clear high outliers
                        zoom_max = find_high_outlier_zoom_max(all_fracs)
                        if zoom_max is not None:
                            out_filename_frac_zoom = f"run_{run_number}_Ffraction_zoom.png"
                            output_path_frac_zoom = run_output_dir / out_filename_frac_zoom
                            make_1d_cdb_branch_plot(
                                all_fracs,
                                run_number,
                                output_path_frac_zoom,
                                branch_name="Ffraction",
                                xlabel="frac badChi2",
                                outlier_entries=outlier_fracs if len(outlier_fracs) > 0 else None,
                                x_max_cutoff=zoom_max,
                            )

            for h2_energy_name in h2_energy_index_names:
                if h2_energy_name in file:
                    hist2d = file[h2_energy_name]
                    if run_output_dir is not None:
                        # Full y-projection (all towers)
                        out_filename_all = f"run_{run_number}_{h2_energy_name}.png"
                        output_path_all = run_output_dir / out_filename_all
                        make_1d_yproj_plot(
                            hist2d,
                            run_number,
                            output_path_all,
                            hist_name=h2_energy_name,
                            tower_index=None,
                            logy=True,
                        )

                        # Full y-projection excluding outlier towers below threshold
                        if len(outlier_towers) > 0:
                            out_filename_excl = f"run_{run_number}_{h2_energy_name}_excl_outliers.png"
                            output_path_excl = run_output_dir / out_filename_excl
                            make_1d_yproj_plot(
                                hist2d,
                                run_number,
                                output_path_excl,
                                hist_name=h2_energy_name,
                                tower_index=None,
                                exclude_towers=outlier_towers,
                                logy=True,
                            )

                        # Y-projection for outlier towers below threshold
                        for tower_idx in outlier_towers:
                            out_filename_tower = f"run_{run_number}_{h2_energy_name}_tower{tower_idx}.png"
                            output_path_tower = run_output_dir / out_filename_tower
                            tower_key = get_calo_tower_key(tower_idx, det=h2_energy_name)
                            z_score = bad_tower_map.get(tower_key, {}).get("sigma") if bad_tower_map else None
                            frac_bad_chi2 = frac_bad_chi2_map.get(tower_key) if frac_bad_chi2_map else None
                            make_1d_yproj_plot(
                                hist2d,
                                run_number,
                                output_path_tower,
                                hist_name=h2_energy_name,
                                tower_index=tower_idx,
                                z_score=z_score,
                                frac_bad_chi2=frac_bad_chi2,
                                logy=True,
                            )

                            if ref_tower is not None:
                                out_filename_overlay = f"run_{run_number}_{h2_energy_name}_tower{tower_idx}_ref{ref_tower}.png"
                                output_path_overlay = run_output_dir / out_filename_overlay
                                make_1d_yproj_plot(
                                    hist2d,
                                    run_number,
                                    output_path_overlay,
                                    hist_name=h2_energy_name,
                                    tower_index=tower_idx,
                                    z_score=z_score,
                                    frac_bad_chi2=frac_bad_chi2,
                                    ref_tower_index=ref_tower,
                                    ref_z_score=ref_z_score,
                                    ref_frac_bad_chi2=ref_frac_bad_chi2,
                                    logy=True,
                                )

            return None
    except Exception as e:
        traceback.print_exc()
        return f"Error processing {path}: {e}"

def main():
    parser = argparse.ArgumentParser(description="Plot Calorimeter QA for sPHENIX.")
    parser.add_argument("-f", "--file", type=Path, help="Path to a text file containing ROOT file paths (one per line).")
    parser.add_argument("-o", "--output-dir", type=Path, default=Path("."), help="Directory to save the plots (default: current directory).")
    parser.add_argument("--do-nolog", type=int, default=1, help="Generate linear/no-log scale plots (1=True, 0=False). Default: 1")
    parser.add_argument("--do-logy", type=int, default=1, help="Generate log-y scale plots (1=True, 0=False). Default: 1")
    parser.add_argument("--do-logxy", type=int, default=1, help="Generate log-xy scale plots (1=True, 0=False). Default: 1")
    parser.add_argument("--do-logx", type=int, default=0, help="Generate log-x scale plots (1=True, 0=False). Default: 0")
    parser.add_argument("--energy-threshold", type=float, default=-10.0, help="Energy threshold below which 1D tower energy plots are generated (default: -10.0 GeV).")
    parser.add_argument("--max-outlier-towers", type=int, default=50, help="Maximum number of outlier tower 1D plots to generate per run (default: 50).")
    parser.add_argument("--cdbtag", default="newcdbtag", help="CDB global tag to fetch BadTowerMap calibration (default: newcdbtag).")
    parser.add_argument("--no-cdb", action="store_true", help="Disable CDB BadTowerMap query for outlier tower z-scores.")
    parser.add_argument("--ref-tower", "--ref-tower-index", "--tower-index", type=int, default=None, dest="ref_tower", help="Reference tower index to overlay on outlier tower 1D plots.")
    parser.add_argument("files", nargs="*", type=Path, help="List of ROOT file paths")
    args = parser.parse_args()

    file_list = []
    if args.files:
        file_list.extend(args.files)

    if args.file:
        try:
            with open(args.file, "r") as f:
                for line in f:
                    stripped = line.strip()
                    if stripped and not stripped.startswith("#"):
                        file_list.append(Path(stripped))
        except Exception as e:
            print(f"Error reading file {args.file}: {e}")
            sys.exit(1)

    if not file_list:
        print("Error: You must provide at least one ROOT file or a text file containing ROOT file paths.")
        parser.print_help()
        sys.exit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Found {len(file_list)} input files. Starting processing...")

    files_to_process = [Path(p) for p in file_list]
    process_func = functools.partial(
        process_file,
        output_dir=args.output_dir,
        do_nolog=bool(args.do_nolog),
        do_logy=bool(args.do_logy),
        do_logxy=bool(args.do_logxy),
        do_logx=bool(args.do_logx),
        energy_threshold=args.energy_threshold,
        max_outlier_towers=args.max_outlier_towers,
        use_cdb=not args.no_cdb,
        cdbtag=args.cdbtag,
        ref_tower=args.ref_tower,
    )
    max_workers = min(os.cpu_count() or 4, 32)

    errors = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = list(tqdm.tqdm(executor.map(process_func, files_to_process), total=len(files_to_process)))

    for path, err in zip(files_to_process, results):
        if err:
            print(err)
            errors.append(err)

    if not errors:
        print(f"Successfully processed all {len(file_list)} files.")
        print(f"Plots saved to {args.output_dir}")
    else:
        print(f"Processed with {len(errors)} errors.")
        print(f"Plots saved to {args.output_dir}")

if __name__ == "__main__":
    main()
