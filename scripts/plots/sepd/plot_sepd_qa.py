#!/usr/bin/env python3

import uproot
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.ticker import ScalarFormatter
from matplotlib.patches import Patch
from mpl_toolkits.axes_grid1 import make_axes_locatable
import mplhep as hep
import os
import sys
import re
import tqdm
import argparse
import functools
import concurrent.futures
from pathlib import Path
from datetime import datetime
import numpy as np


SPHENIX_LABEL = "Internal"  # Change to "Performance", "Preliminary", etc. as needed

HIST_NAMES = [
    "h2sEPD_Centrality",
    "h2sEPD_MBD",
    "h2sEPD_CaloE",
    "h2CaloE_MBD",
    "h2sEPD_North_South",
    "h2sEPD_Centrality_cut",
    "h2sEPD_MBD_cut",
    "h2sEPD_CaloE_cut",
    "h2CaloE_MBD_cut",
    "h2sEPD_North_South_cut",
]

CENTRALITY_INTERVALS = [
    (0, 10),
    (10, 20),
    (20, 40),
    (40, 60),
    (60, 80),
    (80, 100),
]


def get_sphenix_label(status=SPHENIX_LABEL):
    if not status:
        return r"$\boldsymbol{sPHENIX}$"
    return rf"$\boldsymbol{{sPHENIX}}$ {status}"


def clean_root_latex(text):
    if not text:
        return ""
    text = text.strip()
    text = text.replace("#", "\\")
    if "$" not in text and any(kw in text for kw in ["\\", "_{", "^{"]):
        text = re.sub(r'([a-zA-Z0-9\\_*|()]+(?:_{[^}\s]+}|^{[^}\s]+}|_[a-zA-Z0-9]+|\^[a-zA-Z0-9]+)+)', r'$\1$', text)
    return text.strip()


def get_hist_axis_titles(hist2d, hist_name=""):
    raw_title = ""
    xlabel = ""
    ylabel = ""

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

    if not raw_title:
        raw_title = getattr(hist2d, "title", "") or ""

    if ";" in raw_title:
        parts = [p.strip() for p in raw_title.split(";")]
        if not xlabel and len(parts) > 1:
            xlabel = parts[1]
        if not ylabel and len(parts) > 2:
            ylabel = parts[2]

    return clean_root_latex(xlabel), clean_root_latex(ylabel)


def parse_run_number(path):
    path = Path(path)
    try:
        return int(path.name.split('.')[0])
    except ValueError:
        pass

    match = re.search(r'(?:run[_-]?|part[_-]?)?(\d{5,8})', str(path))
    if match:
        return int(match.group(1))

    match = re.search(r'\d+', path.name)
    if match:
        return int(match.group())

    return None


def get_best_label_corner(values, xedges, yedges, x_min, x_max, y_min, y_max):
    x_centers = (xedges[:-1] + xedges[1:]) / 2.0
    y_centers = (yedges[:-1] + yedges[1:]) / 2.0
    xc, yc = np.meshgrid(x_centers, y_centers, indexing='ij')

    in_view = (xc >= x_min) & (xc <= x_max) & (yc >= y_min) & (yc <= y_max)
    if not np.any(in_view):
        return 0.95, 0.95, 'right', 'top'

    x_mid = (x_min + x_max) / 2.0
    y_mid = (y_min + y_max) / 2.0

    corners = [
        (0.95, 0.95, 'right', 'top', in_view & (xc >= x_mid) & (yc >= y_mid)),
        (0.05, 0.95, 'left', 'top', in_view & (xc < x_mid) & (yc >= y_mid)),
        (0.95, 0.08, 'right', 'bottom', in_view & (xc >= x_mid) & (yc < y_mid)),
        (0.05, 0.08, 'left', 'bottom', in_view & (xc < x_mid) & (yc < y_mid)),
    ]

    best_score = float('inf')
    best_corner = corners[0][:4]

    for x_pos, y_pos, ha, va, mask in corners:
        score = np.sum(values[mask])
        if score < best_score:
            best_score = score
            best_corner = (x_pos, y_pos, ha, va)

    return best_corner


def compute_profile_x_spread(values, xedges, yedges):
    """
    Computes mean and standard deviation (spread) for each X bin,
    matching ROOT's ProfileX("...", 1, -1, "s").
    """
    y_centers = (yedges[:-1] + yedges[1:]) / 2.0
    x_centers = (xedges[:-1] + xedges[1:]) / 2.0

    weights_sum = np.sum(values, axis=1)
    valid = weights_sum > 0

    mean = np.zeros(len(x_centers))
    std = np.zeros(len(x_centers))

    mean[valid] = np.sum(values[valid] * y_centers, axis=1) / weights_sum[valid]
    mean_sq = np.zeros(len(x_centers))
    mean_sq[valid] = np.sum(values[valid] * (y_centers ** 2), axis=1) / weights_sum[valid]

    var = np.maximum(0.0, mean_sq - mean ** 2)
    std = np.sqrt(var)

    return x_centers, mean, std


def make_2d_plot(values, xedges, yedges, run_number, output_path, xlabel="", ylabel="", hist_name="", sphenix_label=SPHENIX_LABEL, date_str=None, save_pdf=False, calo_mbd_cut_profile=None, sigma_cut=3.5, custom_max_coord=None, custom_xlim=None, custom_ylim=None):
    if date_str is None:
        date_str = datetime.now().strftime("%m/%d/%Y")

    hep.style.use("ATLAS")
    fig, ax = plt.subplots(figsize=(8, 6))

    values_masked = np.ma.masked_where(values <= 0, values)

    if np.all(values <= 0):
        mesh = ax.pcolormesh(xedges, yedges, values.T, cmap='viridis', rasterized=True)
    else:
        mesh = ax.pcolormesh(xedges, yedges, values_masked.T, norm=LogNorm(), cmap='viridis', rasterized=True)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(mesh, cax=cax)
    cbar.set_label("Events")

    if xlabel:
        ax.set_xlabel(xlabel, loc='center')
    if ylabel:
        ax.set_ylabel(ylabel, loc='center', labelpad=10)

    # Set x-limits based on histogram type
    if custom_xlim is not None:
        ax.set_xlim(custom_xlim)
    elif hist_name in ["h2CaloE_MBD", "h2CaloE_MBD_cut"]:
        ax.set_xlim(left=0, right=2100)
    elif hist_name in ["h2sEPD_CaloE", "h2sEPD_CaloE_cut"]:
        nonzero_x, _ = np.where(values > 0)
        if len(nonzero_x) > 0:
            min_x = float(xedges[np.min(nonzero_x)])
            max_x = float(xedges[np.max(nonzero_x) + 1])
        else:
            min_x = float(xedges[0])
            max_x = float(xedges[-1])
        if min_x >= max_x:
            min_x = float(xedges[0])
            max_x = float(xedges[-1])
        ax.set_xlim(left=min_x, right=max_x)
    elif hist_name in ["h2sEPD_MBD", "h2sEPD_MBD_cut"]:
        nonzero_x, _ = np.where(values > 0)
        if len(nonzero_x) > 0:
            max_x = float(xedges[np.max(nonzero_x) + 1])
        else:
            max_x = float(xedges[-1])
        if max_x <= 0:
            max_x = float(xedges[-1])
        ax.set_xlim(left=0, right=max_x)
    elif hist_name in ["h2sEPD_Centrality", "h2sEPD_Centrality_cut"]:
        ax.set_xlim(left=0, right=100)
    elif hist_name in ["h2sEPD_North_South", "h2sEPD_North_South_cut"]:
        if custom_max_coord is not None:
            max_coord = float(custom_max_coord)
        else:
            nonzero_x, nonzero_y = np.where(values > 0)
            if len(nonzero_x) > 0:
                max_coord = float(max(xedges[np.max(nonzero_x) + 1], yedges[np.max(nonzero_y) + 1]))
            else:
                max_coord = float(min(xedges[-1], yedges[-1]))
            if max_coord <= 0:
                max_coord = float(xedges[-1])
        ax.set_xlim(left=0, right=max_coord)
    else:
        ax.set_xlim(left=xedges[0], right=xedges[-1])

    # Set y-limits based on histogram type
    if custom_ylim is not None:
        ax.set_ylim(custom_ylim)
    elif hist_name in ["h2CaloE_MBD", "h2CaloE_MBD_cut"]:
        ax.set_ylim(bottom=yedges[0], top=2100)
    elif hist_name in ["h2sEPD_North_South", "h2sEPD_North_South_cut"]:
        ax.set_ylim(bottom=0, top=max_coord)
    elif hist_name in ["h2sEPD_CaloE", "h2sEPD_CaloE_cut", "h2sEPD_Centrality", "h2sEPD_Centrality_cut", "h2sEPD_MBD", "h2sEPD_MBD_cut"]:
        _, nonzero_y = np.where(values > 0)
        if len(nonzero_y) > 0:
            max_y = float(yedges[np.max(nonzero_y) + 1])
        else:
            max_y = float(yedges[-1])
        if max_y <= 0:
            max_y = float(yedges[-1])
        ax.set_ylim(bottom=0, top=max_y)
    else:
        # sEPD total charge max is 20000 on y-axis
        ax.set_ylim(bottom=0, top=20000)

    cur_xlim = ax.get_xlim()
    cur_ylim = ax.get_ylim()

    if calo_mbd_cut_profile is not None:
        prof_x, prof_mean, prof_std = calo_mbd_cut_profile
        mask = (prof_std > 0.0) & (prof_x >= cur_xlim[0]) & (prof_x <= cur_xlim[1])
        if np.any(mask):
            x_vals = prof_x[mask]
            y_upper = prof_mean[mask] + sigma_cut * prof_std[mask]
            y_lower = prof_mean[mask] - sigma_cut * prof_std[mask]

            ax.plot(x_vals, y_upper, color='red', linewidth=2)
            ax.plot(x_vals, y_lower, color='red', linewidth=2)

            ax.fill_between(x_vals, np.clip(y_upper, cur_ylim[0], cur_ylim[1]), cur_ylim[1], color='red', alpha=0.15)
            ax.fill_between(x_vals, cur_ylim[0], np.clip(y_lower, cur_ylim[0], cur_ylim[1]), color='red', alpha=0.15)

            patch = Patch(facecolor=(1, 0, 0, 0.15), edgecolor='red', linewidth=2, label=rf"Excluded: $|E_{{\mathrm{{Calo}}}} - \mu| > {sigma_cut:g}\sigma$")
            ax.legend(handles=[patch], loc='lower right', frameon=False, fontsize=16, title=r"$|z| < 10$ cm & MB", title_fontsize=18, alignment='right')

    if np.max(np.abs(cur_xlim)) >= 1000:
        formatter_x = ScalarFormatter(useMathText=True)
        formatter_x.set_powerlimits((3, 3))
        ax.xaxis.set_major_formatter(formatter_x)

    has_y_offset = np.max(np.abs(cur_ylim)) >= 1000
    if has_y_offset:
        formatter_y = ScalarFormatter(useMathText=True)
        formatter_y.set_powerlimits((3, 3))
        ax.yaxis.set_major_formatter(formatter_y)

    # Top border labels: sPHENIX on left (shifted to avoid overlap with y-axis x10^3 multiplier), Run & Date on right
    sphenix_x = 0.14 if has_y_offset else 0.0
    ax.text(sphenix_x, 1.01, get_sphenix_label(sphenix_label), transform=ax.transAxes, ha='left', va='bottom', fontsize=16)

    right_text = f"Run: {run_number}, {date_str}" if run_number is not None else date_str
    ax.text(1.0, 1.01, right_text, transform=ax.transAxes, ha='right', va='bottom', fontsize=15)

    # Auto-place event selection label in the corner with the least data overlap (for non-cutline plots)
    if calo_mbd_cut_profile is None:
        lbl_x, lbl_y, lbl_ha, lbl_va = get_best_label_corner(values, xedges, yedges, cur_xlim[0], cur_xlim[1], cur_ylim[0], cur_ylim[1])
        label_text = r"$|z| < 10$ cm & MB"
        ax.text(lbl_x, lbl_y, label_text, transform=ax.transAxes, ha=lbl_ha, va=lbl_va, fontsize=18)

    fig.tight_layout()
    plt.subplots_adjust(left=0.12, bottom=0.13, top=0.93)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    if save_pdf:
        fig.savefig(output_path.with_suffix('.pdf'))
    plt.close(fig)


def make_centrality_slices_plot(values, xedges, yedges, run_number, output_path, xlabel="sEPD Total Charge", sphenix_label=SPHENIX_LABEL, date_str=None, save_pdf=False, custom_xmax=None):
    if date_str is None:
        date_str = datetime.now().strftime("%m/%d/%Y")

    hep.style.use("ATLAS")
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True, sharey='row', gridspec_kw={'hspace': 0, 'wspace': 0})

    bin_centers_x = (xedges[:-1] + xedges[1:]) / 2.0

    projections = []
    for cent_min, cent_max in CENTRALITY_INTERVALS:
        mask = (bin_centers_x >= cent_min) & (bin_centers_x < cent_max)
        if np.any(mask):
            proj_1d = np.sum(values[mask, :], axis=0)
        else:
            proj_1d = np.zeros(len(yedges) - 1)
        projections.append(proj_1d)

    # Calculate y-limits separately for top row (indices 0, 1, 2) and bottom row (indices 3, 4, 5)
    max_y_top = max(np.max(projections[i]) for i in range(3)) if len(projections) >= 3 else 1
    max_y_bottom = max(np.max(projections[i]) for i in range(3, 6)) if len(projections) >= 6 else 1

    if custom_xmax is not None:
        xmax = float(custom_xmax)
    else:
        nonzero_y = np.where(np.sum(values, axis=0) > 0)[0]
        if len(nonzero_y) > 0:
            xmax = float(yedges[np.max(nonzero_y) + 1])
        else:
            xmax = float(yedges[-1])
        if xmax <= 0:
            xmax = float(yedges[-1])

    for r in range(2):
        for c in range(3):
            idx = r * 3 + c
            ax = axes[r, c]
            proj_1d = projections[idx]
            cent_min, cent_max = CENTRALITY_INTERVALS[idx]

            hep.histplot((proj_1d, yedges), ax=ax, histtype='step', color='navy', linewidth=2)

            ax.set_xlim(left=0, right=xmax)
            if r == 0:
                ax.set_ylim(bottom=0, top=max_y_top * 1.2 if max_y_top > 0 else 1)
            else:
                ax.set_ylim(bottom=0, top=max_y_bottom * 1.2 if max_y_bottom > 0 else 1)

            # X-axis scalar formatter on bottom row
            if r == 1 and xmax >= 1000:
                formatter_x = ScalarFormatter(useMathText=True)
                formatter_x.set_powerlimits((3, 3))
                ax.xaxis.set_major_formatter(formatter_x)

            # Y-axis scalar formatter on leftmost column
            max_y_r = max_y_top if r == 0 else max_y_bottom
            if c == 0 and max_y_r >= 500:
                formatter_y = ScalarFormatter(useMathText=True)
                formatter_y.set_powerlimits((0, 2))
                ax.yaxis.set_major_formatter(formatter_y)

            # Centrality range and event selection label on top-right of each subplot
            cent_label = f"Centrality: {cent_min}–{cent_max}%\n" + r"$|z| < 10$ cm & MB"
            ax.text(0.92, 0.90, cent_label, transform=ax.transAxes, ha='right', va='top', fontsize=18)

            # Axis labels for outer edges (centered)
            if r == 1:
                ax.set_xlabel(xlabel if xlabel else "sEPD Total Charge", loc='center', fontsize=18)
            if c == 0:
                ax.set_ylabel("Events", loc='center', fontsize=18)

    # Top border labels on 2x3 figure: sPHENIX on left (shifted if top row has y-axis multiplier), Run & Date on right
    has_top_y_offset = max_y_top >= 500
    sphenix_slices_x = 0.14 if has_top_y_offset else 0.0
    axes[0, 0].text(sphenix_slices_x, 1.02, get_sphenix_label(sphenix_label), transform=axes[0, 0].transAxes, ha='left', va='bottom', fontsize=16)
    right_text = f"Run: {run_number}, {date_str}" if run_number is not None else date_str
    axes[0, 2].text(1.0, 1.02, right_text, transform=axes[0, 2].transAxes, ha='right', va='bottom', fontsize=15)

    plt.subplots_adjust(left=0.08, right=0.97, top=0.94, bottom=0.10, hspace=0, wspace=0)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    if save_pdf:
        fig.savefig(output_path.with_suffix('.pdf'))
    plt.close(fig)


def make_1d_plot(values, bin_edges, run_number, output_path, xlabel="", ylabel="Events", sphenix_label=SPHENIX_LABEL, date_str=None, save_pdf=False):
    if date_str is None:
        date_str = datetime.now().strftime("%m/%d/%Y")

    hep.style.use("ATLAS")
    fig, ax = plt.subplots(figsize=(8, 6))

    hep.histplot((values, bin_edges), ax=ax, histtype='step', color='navy', linewidth=2)

    ax.set_xlim(bin_edges[0], bin_edges[-1])
    max_y = np.max(values) if len(values) > 0 else 0
    ax.set_ylim(bottom=0, top=max_y * 1.15 if max_y > 0 else 1)

    if xlabel:
        ax.set_xlabel(xlabel, loc='center')
    if ylabel:
        ax.set_ylabel(ylabel, loc='center', labelpad=10)

    if np.max(np.abs(bin_edges)) >= 1000:
        formatter_x = ScalarFormatter(useMathText=True)
        formatter_x.set_powerlimits((3, 3))
        ax.xaxis.set_major_formatter(formatter_x)

    has_y_offset = max_y >= 500
    if has_y_offset:
        formatter_y = ScalarFormatter(useMathText=True)
        formatter_y.set_powerlimits((0, 2))
        ax.yaxis.set_major_formatter(formatter_y)

    sphenix_x = 0.14 if has_y_offset else 0.0
    ax.text(sphenix_x, 1.01, get_sphenix_label(sphenix_label), transform=ax.transAxes, ha='left', va='bottom', fontsize=16)

    right_text = f"Run: {run_number}, {date_str}" if run_number is not None else date_str
    ax.text(1.0, 1.01, right_text, transform=ax.transAxes, ha='right', va='bottom', fontsize=15)

    # Event selection label
    ax.text(0.95, 0.95, r"$|z| < 10$ cm & MB", transform=ax.transAxes, ha='right', va='top', fontsize=18)

    fig.tight_layout()
    plt.subplots_adjust(left=0.12, bottom=0.13, top=0.93)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    if save_pdf:
        fig.savefig(output_path.with_suffix('.pdf'))
    plt.close(fig)


def make_1d_slice_plot(values, yedges, cent_min, cent_max, run_number, output_path, xlabel="sEPD Total Charge", sphenix_label=SPHENIX_LABEL, date_str=None, save_pdf=False, custom_xmax=None):
    if date_str is None:
        date_str = datetime.now().strftime("%m/%d/%Y")

    hep.style.use("ATLAS")
    fig, ax = plt.subplots(figsize=(8, 6))

    hep.histplot((values, yedges), ax=ax, histtype='step', color='navy', linewidth=2)

    if custom_xmax is not None:
        xmax = float(custom_xmax)
    else:
        nonzero = np.where(values > 0)[0]
        if len(nonzero) > 0:
            xmax = float(yedges[np.max(nonzero) + 1])
        else:
            xmax = float(yedges[-1])
        if xmax <= 0:
            xmax = float(yedges[-1])

    ax.set_xlim(left=0, right=xmax)
    max_y = np.max(values) if len(values) > 0 else 0
    ax.set_ylim(bottom=0, top=max_y * 1.15 if max_y > 0 else 1)

    if xlabel:
        ax.set_xlabel(xlabel, loc='center')
    ax.set_ylabel("Events", loc='center', labelpad=10)

    if xmax >= 1000:
        formatter_x = ScalarFormatter(useMathText=True)
        formatter_x.set_powerlimits((3, 3))
        ax.xaxis.set_major_formatter(formatter_x)

    has_y_offset = max_y >= 500
    if has_y_offset:
        formatter_y = ScalarFormatter(useMathText=True)
        formatter_y.set_powerlimits((0, 2))
        ax.yaxis.set_major_formatter(formatter_y)

    sphenix_x = 0.14 if has_y_offset else 0.0
    ax.text(sphenix_x, 1.01, get_sphenix_label(sphenix_label), transform=ax.transAxes, ha='left', va='bottom', fontsize=16)

    right_text = f"Run: {run_number}, {date_str}" if run_number is not None else date_str
    ax.text(1.0, 1.01, right_text, transform=ax.transAxes, ha='right', va='bottom', fontsize=15)

    # Centrality range and event selection label
    cent_label = f"Centrality: {cent_min}–{cent_max}%\n" + r"$|z| < 10$ cm & MB"
    ax.text(0.95, 0.95, cent_label, transform=ax.transAxes, ha='right', va='top', fontsize=18)

    fig.tight_layout()
    plt.subplots_adjust(left=0.12, bottom=0.13, top=0.93)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    if save_pdf:
        fig.savefig(output_path.with_suffix('.pdf'))
    plt.close(fig)


def process_file(path, output_dir, runs_to_plot=None, sphenix_label=SPHENIX_LABEL, date_str=None, save_pdf=False, ref_profile=None, sigma_cut=3.5):
    path = Path(path)
    if not path.exists():
        return f"File not found: {path}"

    run_number = parse_run_number(path)
    if runs_to_plot is not None and run_number is not None and run_number not in runs_to_plot:
        return None

    try:
        with uproot.open(path) as file:
            plotted_any = False

            # If h2sEPD_North_South is present, compute its max_coord for matching bounds
            ns_uncut_max_coord = None
            if "h2sEPD_North_South" in file:
                ns_vals, ns_xe, ns_ye = file["h2sEPD_North_South"].to_numpy()
                nz_x, nz_y = np.where(ns_vals > 0)
                if len(nz_x) > 0:
                    ns_uncut_max_coord = float(max(ns_xe[np.max(nz_x) + 1], ns_ye[np.max(nz_y) + 1]))
                else:
                    ns_uncut_max_coord = float(min(ns_xe[-1], ns_ye[-1]))
                if ns_uncut_max_coord <= 0:
                    ns_uncut_max_coord = float(ns_xe[-1])

            # If h2sEPD_CaloE is present, compute its bounds for matching bounds
            caloe_uncut_xlim = None
            caloe_uncut_ylim = None
            if "h2sEPD_CaloE" in file:
                ce_vals, ce_xe, ce_ye = file["h2sEPD_CaloE"].to_numpy()
                nz_x, _ = np.where(ce_vals > 0)
                if len(nz_x) > 0:
                    ce_min_x = float(ce_xe[np.min(nz_x)])
                    ce_max_x = float(ce_xe[np.max(nz_x) + 1])
                else:
                    ce_min_x = float(ce_xe[0])
                    ce_max_x = float(ce_xe[-1])
                if ce_min_x >= ce_max_x:
                    ce_min_x = float(ce_xe[0])
                    ce_max_x = float(ce_xe[-1])
                caloe_uncut_xlim = (ce_min_x, ce_max_x)

                _, nz_y = np.where(ce_vals > 0)
                if len(nz_y) > 0:
                    ce_max_y = float(ce_ye[np.max(nz_y) + 1])
                else:
                    ce_max_y = float(ce_ye[-1])
                if ce_max_y <= 0:
                    ce_max_y = float(ce_ye[-1])
                caloe_uncut_ylim = (0.0, ce_max_y)

            # If h2sEPD_Centrality is present, compute its bounds for matching bounds
            cent_uncut_xlim = (0.0, 100.0)
            cent_uncut_ylim = None
            if "h2sEPD_Centrality" in file:
                cnt_vals, cnt_xe, cnt_ye = file["h2sEPD_Centrality"].to_numpy()
                _, nz_y = np.where(cnt_vals > 0)
                if len(nz_y) > 0:
                    cnt_max_y = float(cnt_ye[np.max(nz_y) + 1])
                else:
                    cnt_max_y = float(cnt_ye[-1])
                if cnt_max_y <= 0:
                    cnt_max_y = float(cnt_ye[-1])
                cent_uncut_ylim = (0.0, cnt_max_y)

            # If h2sEPD_MBD is present, compute its bounds for matching bounds
            mbd_uncut_xlim = None
            mbd_uncut_ylim = None
            if "h2sEPD_MBD" in file:
                mbd_vals, mbd_xe, mbd_ye = file["h2sEPD_MBD"].to_numpy()
                nz_x, _ = np.where(mbd_vals > 0)
                if len(nz_x) > 0:
                    mbd_max_x = float(mbd_xe[np.max(nz_x) + 1])
                else:
                    mbd_max_x = float(mbd_xe[-1])
                if mbd_max_x <= mbd_xe[0]:
                    mbd_max_x = float(mbd_xe[-1])
                mbd_uncut_xlim = (float(mbd_xe[0]), mbd_max_x)

                _, nz_y = np.where(mbd_vals > 0)
                if len(nz_y) > 0:
                    mbd_max_y = float(mbd_ye[np.max(nz_y) + 1])
                else:
                    mbd_max_y = float(mbd_ye[-1])
                if mbd_max_y <= 0:
                    mbd_max_y = float(mbd_ye[-1])
                mbd_uncut_ylim = (0.0, mbd_max_y)

            for name in HIST_NAMES:
                if name in file:
                    obj = file[name]
                    values, xedges, yedges = obj.to_numpy()
                    xlabel, ylabel = get_hist_axis_titles(obj, name)

                    prefix = f"run_{run_number}_" if run_number is not None else f"{path.stem}_"
                    out_path = output_dir / f"{prefix}{name}.png"
                    make_2d_plot(values, xedges, yedges, run_number, out_path, xlabel=xlabel, ylabel=ylabel, hist_name=name, sphenix_label=sphenix_label, date_str=date_str, save_pdf=save_pdf)

                    if name == "h2CaloE_MBD":
                        cut_out_path = output_dir / f"{prefix}{name}_cut_region.png"
                        prof = ref_profile if ref_profile is not None else compute_profile_x_spread(values, xedges, yedges)
                        make_2d_plot(values, xedges, yedges, run_number, cut_out_path, xlabel=xlabel, ylabel=ylabel, hist_name=name, sphenix_label=sphenix_label, date_str=date_str, save_pdf=save_pdf, calo_mbd_cut_profile=prof, sigma_cut=sigma_cut)

                    if name == "h2sEPD_North_South_cut" and ns_uncut_max_coord is not None:
                        matched_out_path = output_dir / f"{prefix}{name}_matched_bounds.png"
                        make_2d_plot(values, xedges, yedges, run_number, matched_out_path, xlabel=xlabel, ylabel=ylabel, hist_name=name, sphenix_label=sphenix_label, date_str=date_str, save_pdf=save_pdf, custom_max_coord=ns_uncut_max_coord)

                    if name == "h2sEPD_CaloE_cut" and caloe_uncut_xlim is not None:
                        matched_out_path = output_dir / f"{prefix}{name}_matched_bounds.png"
                        make_2d_plot(values, xedges, yedges, run_number, matched_out_path, xlabel=xlabel, ylabel=ylabel, hist_name=name, sphenix_label=sphenix_label, date_str=date_str, save_pdf=save_pdf, custom_xlim=caloe_uncut_xlim, custom_ylim=caloe_uncut_ylim)

                    if name == "h2sEPD_Centrality_cut" and cent_uncut_ylim is not None:
                        matched_out_path = output_dir / f"{prefix}{name}_matched_bounds.png"
                        make_2d_plot(values, xedges, yedges, run_number, matched_out_path, xlabel=xlabel, ylabel=ylabel, hist_name=name, sphenix_label=sphenix_label, date_str=date_str, save_pdf=save_pdf, custom_xlim=cent_uncut_xlim, custom_ylim=cent_uncut_ylim)

                    if name == "h2sEPD_MBD_cut" and mbd_uncut_xlim is not None and mbd_uncut_ylim is not None:
                        matched_out_path = output_dir / f"{prefix}{name}_matched_bounds.png"
                        make_2d_plot(values, xedges, yedges, run_number, matched_out_path, xlabel=xlabel, ylabel=ylabel, hist_name=name, sphenix_label=sphenix_label, date_str=date_str, save_pdf=save_pdf, custom_xlim=mbd_uncut_xlim, custom_ylim=mbd_uncut_ylim)

                    if name in ["h2sEPD_Centrality", "h2sEPD_Centrality_cut"]:
                        slices_out_path = output_dir / f"{prefix}{name}_slices.png"
                        make_centrality_slices_plot(values, xedges, yedges, run_number, slices_out_path, xlabel=ylabel, sphenix_label=sphenix_label, date_str=date_str, save_pdf=save_pdf)

                        # Individual 1D slice plots with optimized data ranges
                        bin_centers_x = (xedges[:-1] + xedges[1:]) / 2.0
                        for cent_min, cent_max in CENTRALITY_INTERVALS:
                            mask = (bin_centers_x >= cent_min) & (bin_centers_x < cent_max)
                            if np.any(mask):
                                proj_1d = np.sum(values[mask, :], axis=0)
                            else:
                                proj_1d = np.zeros(len(yedges) - 1)
                            slice_out_path = output_dir / f"{prefix}{name}_slice_{cent_min}_{cent_max}.png"
                            make_1d_slice_plot(proj_1d, yedges, cent_min, cent_max, run_number, slice_out_path, xlabel=ylabel if ylabel else "sEPD Total Charge", sphenix_label=sphenix_label, date_str=date_str, save_pdf=save_pdf)

                        if name == "h2sEPD_Centrality_cut" and cent_uncut_ylim is not None:
                            matched_slices_out_path = output_dir / f"{prefix}{name}_slices_matched_bounds.png"
                            make_centrality_slices_plot(values, xedges, yedges, run_number, matched_slices_out_path, xlabel=ylabel, sphenix_label=sphenix_label, date_str=date_str, save_pdf=save_pdf, custom_xmax=cent_uncut_ylim[1])

                            for cent_min, cent_max in CENTRALITY_INTERVALS:
                                mask = (bin_centers_x >= cent_min) & (bin_centers_x < cent_max)
                                if np.any(mask):
                                    proj_1d = np.sum(values[mask, :], axis=0)
                                else:
                                    proj_1d = np.zeros(len(yedges) - 1)
                                matched_slice_out_path = output_dir / f"{prefix}{name}_slice_{cent_min}_{cent_max}_matched_bounds.png"
                                make_1d_slice_plot(proj_1d, yedges, cent_min, cent_max, run_number, matched_slice_out_path, xlabel=ylabel if ylabel else "sEPD Total Charge", sphenix_label=sphenix_label, date_str=date_str, save_pdf=save_pdf, custom_xmax=cent_uncut_ylim[1])

                        cent_1d_out_path = output_dir / f"{prefix}{name}_1D.png"
                        proj_cent = np.sum(values, axis=1)
                        make_1d_plot(proj_cent, xedges, run_number, cent_1d_out_path, xlabel=xlabel if xlabel else "Centrality [%]", ylabel="Events", sphenix_label=sphenix_label, date_str=date_str, save_pdf=save_pdf)

                    plotted_any = True

            if not plotted_any:
                return f"None of target histograms found in {path.name}"

        return None
    except Exception as e:
        return f"Error processing {path}: {e}"


def main():
    parser = argparse.ArgumentParser(description="Plot sEPD QA histograms from ROOT files.")
    parser.add_argument("-f", "--file", type=Path, help="Path to a text/list file containing ROOT file paths (one per line).")
    parser.add_argument("-o", "--output-dir", type=Path, default=Path("plots/sepd_qa"), help="Directory to save the plots (default: plots/sepd_qa).")
    parser.add_argument("-c", "--calo-mbd-file", type=Path, default=None, help="Optional ROOT file containing reference h2CaloE_MBD for profile cut bounds.")
    parser.add_argument("--sigma-cut", type=float, default=3.5, help="Sigma cut threshold for Calo-MBD cut region (default: 3.5).")
    parser.add_argument("--label", "--sphenix-label", type=str, default=SPHENIX_LABEL, help=f"sPHENIX label status (e.g. Internal, Performance, Preliminary). Default: {SPHENIX_LABEL}")
    parser.add_argument("--date", type=str, default=datetime.now().strftime("%m/%d/%Y"), help=f"Date string in mm/dd/yyyy format. Default: today's date ({datetime.now().strftime('%m/%d/%Y')})")
    parser.add_argument("--save-pdf", action="store_true", help="Enable saving of plots in PDF format (in addition to PNG).")
    parser.add_argument("--runs", type=int, nargs="+", help="Specific run number(s) to plot.")
    parser.add_argument("-j", "--max-workers", type=int, default=None, help="Number of parallel workers (default: automatic).")
    parser.add_argument("files", nargs="*", type=Path, help="List of ROOT file paths")

    args = parser.parse_args()

    file_list = []
    if args.files:
        file_list.extend(args.files)

    if args.file:
        if not args.file.exists():
            print(f"Error: File list {args.file} does not exist.")
            sys.exit(1)
        with args.file.open('r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    file_list.append(Path(line))

    if not file_list:
        print("Error: You must provide at least one ROOT file or a text file with -f/--file.")
        parser.print_help()
        sys.exit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    runs_to_filter = set(args.runs) if args.runs else None

    ref_profile = None
    if args.calo_mbd_file:
        if not args.calo_mbd_file.exists():
            print(f"Warning: Calo-MBD file does not exist: {args.calo_mbd_file}. Will use each file's h2CaloE_MBD.")
        else:
            try:
                with uproot.open(args.calo_mbd_file) as rf:
                    if "h2CaloE_MBD" in rf:
                        ref_obj = rf["h2CaloE_MBD"]
                        ref_vals, ref_xe, ref_ye = ref_obj.to_numpy()
                        ref_profile = compute_profile_x_spread(ref_vals, ref_xe, ref_ye)
                    else:
                        print(f"Warning: h2CaloE_MBD not found in {args.calo_mbd_file}. Will use each file's h2CaloE_MBD.")
            except Exception as e:
                print(f"Warning: Could not read {args.calo_mbd_file}: {e}. Will use each file's h2CaloE_MBD.")

    print(f"Found {len(file_list)} input file(s). Starting plotting...")

    max_workers = args.max_workers or min(os.cpu_count() or 4, 32)
    process_func = functools.partial(
        process_file,
        output_dir=args.output_dir,
        runs_to_plot=runs_to_filter,
        sphenix_label=args.label,
        date_str=args.date,
        save_pdf=args.save_pdf,
        ref_profile=ref_profile,
        sigma_cut=args.sigma_cut,
    )

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        errors = list(tqdm.tqdm(executor.map(process_func, file_list), total=len(file_list)))

    for err in errors:
        if err:
            print(f"Warning: {err}")

    print(f"Plots saved to: {args.output_dir}")
    print("Done!")


if __name__ == "__main__":
    main()
