#!/usr/bin/env python3

import argparse
import concurrent.futures
import os
from pathlib import Path
import pickle
import re
import sys
import numpy as np
import pandas as pd
import tqdm
import uproot
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.ticker import MaxNLocator, MultipleLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.offsetbox import TextArea, HPacker, VPacker, AnnotationBbox
import mplhep as hep

# Repository imports
_calo_dir = Path(__file__).resolve().parent.parent / "calo"
if str(_calo_dir) not in sys.path:
    sys.path.insert(0, str(_calo_dir))

try:
    from tower_info_defs import get_cdb_calibration_url, decode_emcal, get_frac_bad_chi2_map
except ImportError:
    get_cdb_calibration_url = None
    decode_emcal = None
    get_frac_bad_chi2_map = None


def process_item(item_data):
    item, is_run, cdbtag = item_data
    if is_run:
        run_number = int(item)
        if get_cdb_calibration_url is None:
            return run_number, None, None, None, None, "tower_info_defs module could not be imported"
        try:
            url = get_cdb_calibration_url("CEMC_BadTowerMap", run_number, dbtag=cdbtag)
            if not url:
                return run_number, None, None, None, None, f"Could not find CDB calibration for run {run_number}"
            path = url
        except Exception as e:
            return run_number, None, None, None, None, f"Error getting CDB URL for run {run_number}: {e}"
    else:
        path = Path(item)
        if not path.exists():
            return None, None, None, None, None, f"File not found: {path}"
        try:
            # Try to extract run number from filename (e.g. EMCalHotMap_..._78348.root)
            match = re.search(r'_(\d+)\.root', path.name)
            if match:
                run_number = int(match.group(1))
            else:
                # Fallback
                match = re.search(r'\d+', path.name)
                if match:
                    run_number = int(match.group())
                else:
                    return None, None, None, None, None, f"Could not parse run number from {path.name}"
        except Exception as e:
            return None, None, None, None, None, f"Error parsing run number from {path}: {e}"

    try:
        with uproot.open(path) as file:
            if "h_hot" in file:
                hist = file["h_hot"]
                values = hist.values()
                bad_towers_count = np.count_nonzero(values != 0)
                dead_towers_count = np.count_nonzero(values == 1)
                hot_towers_count = np.count_nonzero(values == 2)

                eta_indices, phi_indices = np.nonzero(values != 0)
                bad_tower_keys = phi_indices + (eta_indices << 16)
                bad_tower_statuses = values[eta_indices, phi_indices].astype(int)
                tower_status_map = dict(zip(bad_tower_keys.astype(int), bad_tower_statuses))

                return run_number, bad_towers_count, dead_towers_count, hot_towers_count, tower_status_map, None
            elif "Multiple;1" in file or "Multiple" in file:
                tree = file["Multiple"]
                if "Istatus" in tree.keys():
                    data = tree.arrays(["IID", "Istatus"], library="np")
                    statuses = data["Istatus"]
                    iids = data["IID"]
                else:
                    return run_number, None, None, None, None, f"Istatus branch not found in {path}"
                bad_towers_count = np.count_nonzero(statuses != 0)
                dead_towers_count = np.count_nonzero(statuses == 1)
                hot_towers_count = np.count_nonzero(statuses == 2)

                bad_mask = statuses != 0
                tower_status_map = dict(zip(iids[bad_mask].astype(int), statuses[bad_mask].astype(int)))
            else:
                return run_number, None, None, None, None, f"Neither h_hot nor Multiple tree found in {path}"

        # Fetch and incorporate frac_bad_chi2 (second ROOT file per run)
        if run_number is not None and get_frac_bad_chi2_map is not None:
            try:
                dbtag = cdbtag if cdbtag else "newcdbtag"
                chi2_map = get_frac_bad_chi2_map(run_number, det="CEMC", dbtag=dbtag)
                for k, frac in chi2_map.items():
                    if frac > 0.01 and int(k) not in tower_status_map:
                        tower_status_map[int(k)] = 4
            except Exception as e:
                print(f"Warning: Could not load fracBadChi2 for run {run_number}: {e}")

        return run_number, bad_towers_count, dead_towers_count, hot_towers_count, tower_status_map, None
    except Exception as e:
        return run_number if 'run_number' in locals() else None, None, None, None, None, f"Error processing {path}: {e}"


def plot_towers(run_numbers, towers_count, output_dir, name, ylabel="Number of Bad Towers", suffix="", extra_text=None, ylim_bottom=None, legend_loc='lower left', legend_fontsize=18, sigma_threshold=None):
    hep.style.use("ATLAS")

    fig, ax = plt.subplots(figsize=(10, 6))

    runs = np.array(run_numbers)
    towers = np.array(towers_count)

    group_defs = [
        (runs < 73500, 'blue', r'$\leq$ 73500'),
        ((runs >= 73500) & (runs <= 78217), 'orange', '73500-78217'),
        (runs >= 78218, 'green', r'$\geq$ 78218'),
    ]

    is_outlier = np.zeros(len(runs), dtype=bool)

    total_runs = len(run_numbers)
    text_info = f"Runs = {total_runs}"

    avg_handles = []
    sigma_handles = []

    for mask, color, label_str in group_defs:
        if np.any(mask):
            group_towers = towers[mask]
            group_runs = runs[mask]
            avg = np.mean(group_towers)
            xmin, xmax = np.min(group_runs), np.max(group_runs)

            ax.hlines(avg, xmin, xmax, colors=color, linestyles='--', linewidth=1.5, zorder=4)
            avg_handles.append(Line2D([0], [0], color=color, linestyle='--', linewidth=1.5,
                                      label=rf'Avg ({label_str}) = {avg:.0f}'))

            if sigma_threshold is not None:
                std = np.std(group_towers)
                if std > 0:
                    thresh = avg + sigma_threshold * std
                    sig_str = f"{sigma_threshold:g}"
                    n_crossed = np.count_nonzero(group_towers > thresh)
                    is_outlier[mask & (towers > thresh)] = True
                    ax.hlines(thresh, xmin, xmax, colors=color, linestyles=':', linewidth=1.2, zorder=4)
                    sigma_handles.append(Line2D([0], [0], color=color, linestyle=':', linewidth=1.2,
                                                label=rf'+{sig_str}$\sigma$ = {thresh:.0f} (Runs={n_crossed})'))

    ax.plot(runs[~is_outlier], towers[~is_outlier], marker='o', markersize=4, linestyle='none', color='black', zorder=2)
    if np.any(is_outlier):
        ax.plot(runs[is_outlier], towers[is_outlier], marker='o', markersize=4, linestyle='none', color='red', zorder=5)

    if legend_loc == 'lower left':
        leg1 = ax.legend(handles=avg_handles, loc=legend_loc, bbox_to_anchor=(0.0, -0.03), frameon=False, fontsize=legend_fontsize)
    else:
        leg1 = ax.legend(handles=avg_handles, loc=legend_loc, frameon=False, fontsize=legend_fontsize)
    ax.add_artist(leg1)

    if sigma_threshold is not None and sigma_handles:
        leg2 = ax.legend(handles=sigma_handles, loc='lower left', bbox_to_anchor=(0.37, -0.03), frameon=False, fontsize=legend_fontsize)
        ax.add_artist(leg2)

    if sigma_threshold is not None and np.any(is_outlier):
        sig_str = f"{sigma_threshold:g}"
        outlier_handle = Line2D([0], [0], marker='o', color='white', markerfacecolor='red',
                                markeredgecolor='red', markersize=6, linestyle='none',
                                label=rf'Outliers ($>+{sig_str}\sigma$)')
        leg3 = ax.legend(handles=[outlier_handle], loc='upper right', frameon=False, fontsize=legend_fontsize)
        ax.add_artist(leg3)

    ax.set_xlabel("Run Number", labelpad=18)
    ax.set_ylabel(ylabel)
    ax.set_title("EMCal QA")
    if ylim_bottom is not None:
        ax.set_ylim(bottom=ylim_bottom)

    if extra_text:
        text_info += f"\n{extra_text}"

    ax.text(0.05, 0.95, text_info, transform=ax.transAxes, ha='left', va='top', fontsize=18)

    plt.tight_layout()
    plt.subplots_adjust(top=0.94, bottom=0.14)

    pdf_dir = output_dir / "pdf"
    image_dir = output_dir / "images"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)

    pdf_path = pdf_dir / f"{name}{suffix}.pdf"
    png_path = image_dir / f"{name}{suffix}.png"
    plt.savefig(pdf_path)
    plt.savefig(png_path, dpi=300)
    plt.close(fig)
    print(f"Saved plots as {pdf_path} and {png_path}")

# Maintain backward compatibility for plot_bad_towers function name if imported elsewhere
plot_bad_towers = plot_towers


def _get_target_sigmas_for_run(args):
    run_num, item_data, target_keys = args
    item, is_run, cdbtag = item_data
    path = None
    if is_run:
        if get_cdb_calibration_url is not None:
            try:
                url = get_cdb_calibration_url("CEMC_BadTowerMap", run_num, dbtag=cdbtag)
                if url: path = url
            except Exception:
                pass
    else:
        path = Path(item)

    if not path:
        return run_num, {tk: np.nan for tk in target_keys}

    sigmas = {tk: np.nan for tk in target_keys}
    try:
        with uproot.open(path) as file:
            if "Multiple;1" in file or "Multiple" in file:
                tree = file["Multiple"]
                sigma_branch = next((b for b in tree.keys() if b.endswith("_sigma")), None)
                if sigma_branch and "Istatus" in tree.keys():
                    data = tree.arrays(["IID", "Istatus", sigma_branch], library="np")
                    iids = data["IID"].astype(int)
                    statuses = data["Istatus"].astype(int)
                    all_sigmas = data[sigma_branch].astype(float)

                    for tk in target_keys:
                        idx = np.where(iids == tk)[0]
                        if len(idx) > 0:
                            idx = idx[0]
                            if statuses[idx] == 0:
                                sigmas[tk] = all_sigmas[idx]
    except Exception as e:
        print(f"Warning: error reading sigma for run {run_num}: {e}")

    return run_num, sigmas

def plot_frequently_hot_towers(tower_status_per_run, original_items_per_run, run_numbers, output_dir, name, total_runs=None, no_cache=False):
    """Analyze frequently hot towers across runs, save CSV, and produce 1D/2D plots."""
    if not tower_status_per_run:
        return None

    if total_runs is None or total_runs <= 0:
        total_runs = len(tower_status_per_run)

    valid_hot_keys = []
    for item in tower_status_per_run:
        if isinstance(item, dict):
            h_keys = [k for k, s in item.items() if s == 2]
            if h_keys:
                valid_hot_keys.append(np.array(h_keys, dtype=int))
        elif item is not None and len(item) > 0:
            valid_hot_keys.append(np.array(item, dtype=int))

    if not valid_hot_keys:
        return None

    all_keys = np.concatenate(valid_hot_keys)
    unique_keys, counts = np.unique(all_keys, return_counts=True)

    freq_data = []
    for k, count in zip(unique_keys, counts):
        ieta = k >> 16
        iphi = k & 0xFFFF
        try:
            tidx = decode_emcal(k) if decode_emcal else -1
        except Exception:
            tidx = -1
        freq_data.append({
            'TowerIndex': tidx,
            'ieta': ieta,
            'iphi': iphi,
            'TowerKey': k,
            'HotRunCount': count,
            'HotRunFraction': count / total_runs,
        })

    freq_df = pd.DataFrame(freq_data).sort_values(by="HotRunCount", ascending=False)
    freq_csv_path = output_dir / f"{name}_frequent_hot_towers.csv"
    freq_df.to_csv(freq_csv_path, index=False)
    print(f"Saved frequently hot towers info ({len(freq_df)} towers) to {freq_csv_path}")

    pdf_dir = output_dir / "pdf"
    image_dir = output_dir / "images"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)

    # Plot 1: 1D Tower Index vs HotRunFraction (y-axis starting at 0)
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(freq_df['TowerIndex'], freq_df['HotRunFraction'], marker='.', linestyle='none', color='red', alpha=0.6)
    ax.set_xlabel("Tower Index", loc='center', fontsize=18)
    ax.set_ylabel("Fraction of runs flagged as hot", loc='center', fontsize=18)
    ax.set_ylim(bottom=0, top=1)
    ax.set_title("Frequently Hot Towers", fontsize=18, pad=12)
    ax.tick_params(labelsize=18)
    plt.tight_layout()
    plt.savefig(pdf_dir / f"{name}_frequent_hot_1D.pdf", bbox_inches='tight')
    plt.savefig(image_dir / f"{name}_frequent_hot_1D.png", dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved frequently hot 1D plot to {image_dir / f'{name}_frequent_hot_1D.png'}")

    # Plot 2: 2D ieta vs iphi map (aspect ratio matching 256 phi x 96 eta bins, z-scale as fraction)
    fig, ax = plt.subplots(figsize=(8, 14))
    map_2d = np.zeros((256, 96))  # EMCal is 256 phi bins (y-axis) x 96 eta bins (x-axis)
    for _, row in freq_df.iterrows():
        if 0 <= row['iphi'] < 256 and 0 <= row['ieta'] < 96:
            map_2d[int(row['iphi']), int(row['ieta'])] = row['HotRunFraction']

    c = ax.imshow(map_2d, aspect='equal', origin='lower', cmap='inferno', extent=[-0.5, 95.5, -0.5, 255.5], vmin=0)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.2)
    cbar = fig.colorbar(c, cax=cax)
    cbar.set_label("Fraction of runs flagged as hot", loc='center', labelpad=15, fontsize=18)
    cbar.ax.tick_params(labelsize=18)

    ax.set_xlabel("ieta", fontsize=18)
    ax.set_ylabel("iphi", loc='center', fontsize=18)
    ax.set_title("Frequently Hot Towers Map", fontsize=18, pad=12)
    ax.tick_params(which='both', labelsize=18, color='white', labelcolor='black')

    plt.savefig(pdf_dir / f"{name}_frequent_hot_2D.pdf", bbox_inches='tight')
    plt.savefig(image_dir / f"{name}_frequent_hot_2D.png", dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved frequently hot 2D map to {image_dir / f'{name}_frequent_hot_2D.png'}")

    # 2D Map categorized by hot frequency classes: 0-10%, 10-30%, 30-50%, >50%
    fig, ax = plt.subplots(figsize=(8, 14))
    map_2d_classes = np.zeros((256, 96), dtype=int)
    for _, row in freq_df.iterrows():
        iphi = int(row['iphi'])
        ieta = int(row['ieta'])
        if 0 <= iphi < 256 and 0 <= ieta < 96:
            frac = row['HotRunFraction']
            if frac > 0.5:
                map_2d_classes[iphi, ieta] = 3
            elif frac > 0.1:
                map_2d_classes[iphi, ieta] = 2
            elif row['HotRunCount'] >= 1:
                map_2d_classes[iphi, ieta] = 1

    counts = [np.count_nonzero(map_2d_classes == i) for i in range(4)]
    def _fmt_cnt(cnt):
        return f"{cnt:,} tower" if cnt == 1 else f"{cnt:,} towers"

    class_colors = ['white', '#1f77b4', '#6a0dad', '#d62728']
    class_names = [
        f"0%\n({_fmt_cnt(counts[0])})",
        f">0% - 10%\n({_fmt_cnt(counts[1])})",
        f"10% - 50%\n({_fmt_cnt(counts[2])})",
        f">50%\n({_fmt_cnt(counts[3])})"
    ]
    cmap_classes = ListedColormap(class_colors)
    norm_classes = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap_classes.N)

    c = ax.imshow(map_2d_classes, aspect='equal', origin='lower', cmap=cmap_classes, norm=norm_classes,
                  extent=[-0.5, 95.5, -0.5, 255.5], interpolation='nearest')

    ax.set_xlim(-0.5, 95.5)
    ax.set_ylim(-0.5, 255.5)
    ax.set_xlabel("ieta", fontsize=18)
    ax.set_ylabel("iphi", loc='center', fontsize=18)
    ax.set_title("Hot Towers Frequency Map", fontsize=18, pad=12)
    ax.tick_params(which='both', labelsize=18, color='black', labelcolor='black')

    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.2)
    cbar = fig.colorbar(c, cax=cax, ticks=[0, 1, 2, 3])
    cbar.ax.set_yticklabels(class_names, fontsize=14)
    cbar.ax.tick_params(size=0)

    plt.tight_layout()
    plt.savefig(pdf_dir / f"{name}_frequent_hot_classes_2D.pdf", bbox_inches='tight')
    plt.savefig(image_dir / f"{name}_frequent_hot_classes_2D.png", dpi=300, bbox_inches='tight')
    plt.savefig(pdf_dir / f"{name}_frequent_hot_50pct_2D.pdf", bbox_inches='tight')
    plt.savefig(image_dir / f"{name}_frequent_hot_50pct_2D.png", dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved hot towers categorized 2D map to {image_dir / f'{name}_frequent_hot_classes_2D.png'}")

    # Plots for towers hot in >50% of the runs
    hot_50_df = freq_df[freq_df['HotRunFraction'] > 0.5]
    if len(hot_50_df) == 0:
        print("No towers were hot in >50% of the runs. Skipping 50% hot towers plots.")
    else:
        # Plot 3: 2D Run Index vs Tower for towers hot in >50% of the runs
        target_keys = hot_50_df['TowerKey'].values
        n_towers = len(target_keys)
        n_runs = len(tower_status_per_run)

        # Matrix: shape (n_towers, n_runs), default 0 (Good)
        matrix_t = np.zeros((n_towers, n_runs), dtype=int)
        for r_idx, run_status in enumerate(tower_status_per_run):
            if isinstance(run_status, dict):
                for t_idx, k in enumerate(target_keys):
                    matrix_t[t_idx, r_idx] = run_status.get(int(k), 0)
            elif run_status is not None and len(run_status) > 0:
                for t_idx, k in enumerate(target_keys):
                    if k in run_status:
                        matrix_t[t_idx, r_idx] = 2

        tower_labels = [
            f"({int(row['ieta'])}, {int(row['iphi'])})"
            for _, row in hot_50_df.iterrows()
        ]

        fig_width = max(12, min(24, n_runs * 0.15 + 4))
        fig_height = max(6, min(16, n_towers * 0.45 + 1))
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))

        status_names = ['Good', 'Dead', 'Hot', 'Cold', 'Bad Chi2']
        status_colors = ['#2ca02c', '#333333', '#d62728', '#1f77b4', '#6a0dad']
        cmap = ListedColormap(status_colors)
        norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5, 4.5], cmap.N)

        c = ax.imshow(matrix_t, aspect='auto', origin='lower', cmap=cmap, norm=norm,
                      extent=[-0.5, n_runs - 0.5, -0.5, n_towers - 0.5],
                      interpolation='nearest')
        ax.invert_yaxis()
        ax.set_xlim(0, n_runs - 1)

        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size=0.3, pad=0.25)
        cbar = fig.colorbar(c, cax=cax, ticks=[0, 1, 2, 3, 4])
        cbar.ax.set_yticklabels(status_names, fontsize=14)
        cbar.ax.tick_params(size=0)

        ax.set_yticks(range(n_towers))
        ax.set_yticklabels(tower_labels, fontsize=13)
        ax.set_ylabel("Tower (ieta, iphi) (Most to Least Frequent)", loc='center', fontsize=16, labelpad=10)
        ax.set_xlabel("Run Index", loc='center', fontsize=18)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_title("Hot Towers (>50% Runs) vs Run Index", fontsize=18, pad=12)
        ax.tick_params(axis='x', labelsize=16)

        for y in np.arange(0.5, n_towers - 0.5, 1.0):
            ax.axhline(y, color='gray', linewidth=0.8, alpha=0.5)

        plt.savefig(pdf_dir / f"{name}_frequent_hot_50pct_run_index.pdf", bbox_inches='tight')
        plt.savefig(image_dir / f"{name}_frequent_hot_50pct_run_index.png", dpi=800, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved hot towers (>50%) vs run index plot ({n_towers} towers) to {image_dir / f'{name}_frequent_hot_50pct_run_index.png'}")

        # Plot 3b: Annotated version with vertical lines for specific runs (68144, 72020, 76020)
        target_annotated_runs = [68144, 72020, 76020]
        found_annotations = []
        for ar in target_annotated_runs:
            indices = [idx for idx, r_val in enumerate(run_numbers) if r_val == ar]
            for idx in indices:
                found_annotations.append((ar, idx))

        if found_annotations:
            fig, ax = plt.subplots(figsize=(fig_width, fig_height))
            c = ax.imshow(matrix_t, aspect='auto', origin='lower', cmap=cmap, norm=norm,
                          extent=[-0.5, n_runs - 0.5, -0.5, n_towers - 0.5],
                          interpolation='nearest')
            ax.invert_yaxis()
            ax.set_xlim(0, n_runs - 1)

            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size=0.3, pad=0.25)
            cbar = fig.colorbar(c, cax=cax, ticks=[0, 1, 2, 3, 4])
            cbar.ax.set_yticklabels(status_names, fontsize=14)
            cbar.ax.tick_params(size=0)

            ax.set_yticks(range(n_towers))
            ax.set_yticklabels(tower_labels, fontsize=13)
            ax.set_ylabel("Tower (ieta, iphi) (Most to Least Frequent)", loc='center', fontsize=16, labelpad=10)
            ax.set_xlabel("Run Index", loc='center', fontsize=18)
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax.tick_params(axis='x', labelsize=16)

            for y in np.arange(0.5, n_towers - 0.5, 1.0):
                ax.axhline(y, color='gray', linewidth=0.8, alpha=0.5)

            # Annotated vertical lines, top tick labels, and good tower highlights
            top_tick_labels = []
            for ar, idx in found_annotations:
                ax.axvline(idx, color='white', linestyle='-', linewidth=2.5, zorder=5)
                ax.axvline(idx, color='black', linestyle='--', linewidth=1.8, zorder=6)

                good_t_indices = np.where(matrix_t[:, idx] == 0)[0]
                n_good = len(good_t_indices)
                top_tick_labels.append(f"Run {ar}\n({n_good}/{n_towers} Good)")

                if n_good > 0:
                    ax.scatter([idx] * n_good, good_t_indices, marker='s', s=100,
                               facecolors='none', edgecolors='#ffd700', linewidth=2.5, zorder=12)
                    good_labels = [tower_labels[t] for t in good_t_indices]
                    print(f"Run {ar} (run index {idx}): {n_good}/{n_towers} towers flagged Good: {', '.join(good_labels)}")
                else:
                    print(f"Run {ar} (run index {idx}): 0/{n_towers} towers flagged Good.")

            ax_top = ax.twiny()
            ax_top.set_axes_locator(ax.get_axes_locator())
            ax_top.set_xlim(ax.get_xlim())
            ax_top.set_xticks([idx for ar, idx in found_annotations])
            ax_top.set_xticklabels(top_tick_labels, fontsize=16, fontweight='bold')
            ax_top.tick_params(axis='x', pad=4, length=6, width=1.5)
            ax_top.set_title("Hot Towers (>50% Runs) vs Run Index  (Gold boxes = Good in Annotated Run)", fontsize=16, pad=10)

            plt.savefig(pdf_dir / f"{name}_frequent_hot_50pct_run_index_annotated.pdf", bbox_inches='tight')
            plt.savefig(image_dir / f"{name}_frequent_hot_50pct_run_index_annotated.png", dpi=800, bbox_inches='tight')
            plt.close(fig)
            print(f"Saved hot towers (>50%) vs run index plot with annotated runs to {image_dir / f'{name}_frequent_hot_50pct_run_index_annotated.png'}")

        # Plot 4: 1D Z-score distributions for towers hot in >50% runs
        sigmas_cache_path = output_dir / f".{name}_hot_sigmas_cache.pkl"
        run_sigmas_map = {}
        if not no_cache and sigmas_cache_path.exists():
            try:
                with sigmas_cache_path.open("rb") as f:
                    run_sigmas_map = pickle.load(f)
                if not isinstance(run_sigmas_map, dict):
                    run_sigmas_map = {}
            except Exception as e:
                print(f"Warning: Could not read sigmas cache {sigmas_cache_path}: {e}")
                run_sigmas_map = {}

        runs_to_process = []
        for r_idx, run_status in enumerate(tower_status_per_run):
            run_num = run_numbers[r_idx]
            item_data = original_items_per_run[r_idx]
            cached_sigmas = run_sigmas_map.get(run_num, {})

            missing_keys = []
            for tk in target_keys:
                if run_status.get(tk, 0) == 0 and tk not in cached_sigmas:
                    missing_keys.append(tk)

            if missing_keys:
                runs_to_process.append((run_num, item_data, missing_keys))

        if runs_to_process:
            print(f"Extracting z-scores for >50% hot towers from {len(runs_to_process)} runs ({len(run_numbers) - len(runs_to_process)} loaded from cache)...")
            max_workers = min(os.cpu_count() or 4, 32)
            with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
                for run_num, sigmas in tqdm.tqdm(executor.map(_get_target_sigmas_for_run, runs_to_process), total=len(runs_to_process)):
                    if run_num not in run_sigmas_map:
                        run_sigmas_map[run_num] = {}
                    run_sigmas_map[run_num].update(sigmas)

            if not no_cache:
                try:
                    with sigmas_cache_path.open("wb") as f:
                        pickle.dump(run_sigmas_map, f)
                    print(f"Saved z-scores cache to {sigmas_cache_path}")
                except Exception as e:
                    print(f"Warning: Could not save sigmas cache {sigmas_cache_path}: {e}")
        else:
            print(f"Loaded all z-scores for >50% hot towers from cache ({sigmas_cache_path}).")

        sigmas_to_plot = []
        for i, (tidx, row) in enumerate(hot_50_df.iterrows()):
            tkey = int(row['TowerKey'])
            ieta = int(row['ieta'])
            iphi = int(row['iphi'])

            sigmas = []
            for r_idx, run_status in enumerate(tower_status_per_run):
                run_num = run_numbers[r_idx]
                status = run_status.get(tkey, 0)
                # Only plot for runs where the tower is Good (status 0)
                if status == 0:
                    val = run_sigmas_map.get(run_num, {}).get(tkey, np.nan)
                    if not np.isnan(val):
                        sigmas.append(val)
            if sigmas:
                # Compute status percentages across all runs
                t_statuses = matrix_t[i, :]
                n_tot = len(t_statuses)
                status_labels = [
                    (0, 'good', '#2ca02c'),
                    (1, 'dead', '#333333'),
                    (2, 'hot', '#d62728'),
                    (3, 'cold', '#1f77b4'),
                    (4, 'bad chi2', '#6a0dad')
                ]
                type_pcts = []
                for scode, slabel, scolor in status_labels:
                    cnt = np.count_nonzero(t_statuses == scode)
                    if cnt > 0:
                        pct = (cnt / n_tot) * 100.0
                        pct_str = f"{pct:.1f}%" if pct >= 0.1 else f"{pct:.2f}%"
                        type_pcts.append((f"{slabel}: {pct_str}", scolor))
                sigmas_to_plot.append((row, ieta, iphi, sigmas, type_pcts))

        n_plots = len(sigmas_to_plot)
        if n_plots > 0:
            cols = 5
            # Force at least 5 rows if n_plots <= 25 to match 5x5 request
            rows = max(5, int(np.ceil(n_plots / cols))) if n_plots > 0 else 0

            if rows > 0:
                fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 4.5 * rows), sharex=True)
                if n_plots == 1:
                    axes = np.array([axes])
                axes_flat = axes.flatten()

                for i, (row, ieta, iphi, sigmas, type_pcts) in enumerate(sigmas_to_plot):
                    ax = axes_flat[i]
                    ax.hist(sigmas, bins=30, range=(-5, 5), histtype='step', color='#1f77b4', linewidth=2, log=True)
                    ax.set_ylim(bottom=0.5)
                    ax.set_xlim(-5, 5)

                    title_box = TextArea(f"Tower ({ieta}, {iphi})", textprops=dict(fontsize=12, color='black'))
                    vbox_children = [title_box]

                    if len(type_pcts) > 2:
                        mid = (len(type_pcts) + 1) // 2
                        lines = [type_pcts[:mid], type_pcts[mid:]]
                    else:
                        lines = [type_pcts]

                    for line_items in lines:
                        h_children = []
                        for idx_item, (item_text, item_color) in enumerate(line_items):
                            if idx_item > 0:
                                h_children.append(TextArea(", ", textprops=dict(fontsize=11, color='black')))
                            h_children.append(TextArea(item_text, textprops=dict(fontsize=11, color=item_color, fontweight='bold')))
                        h_line = HPacker(children=h_children, align='baseline', pad=0, sep=1)
                        vbox_children.append(h_line)

                    vbox = VPacker(children=vbox_children, align='center', pad=0, sep=2)
                    ann = AnnotationBbox(vbox, (0.5, 1.02), xycoords='axes fraction', box_alignment=(0.5, 0),
                                         pad=0, frameon=False)
                    ax.add_artist(ann)
                    ax.set_ylabel("Runs (where tower is good)", fontsize=14)

                for i in range(n_plots, len(axes_flat)):
                    fig.delaxes(axes_flat[i])

                # Ensure bottom-most active subplot in each column displays x-axis tick labels and xlabel
                for c in range(cols):
                    active_in_col = [r for r in range(rows) if (r * cols + c) < n_plots]
                    if active_in_col:
                        last_r = max(active_in_col)
                        bottom_ax = axes_flat[last_r * cols + c]
                        bottom_ax.tick_params(labelbottom=True)
                        bottom_ax.set_xlabel("z-score", fontsize=14)

                plt.tight_layout()
                plt.savefig(pdf_dir / f"{name}_frequent_hot_50pct_zscore.pdf", bbox_inches='tight')
                plt.savefig(image_dir / f"{name}_frequent_hot_50pct_zscore.png", dpi=400, bbox_inches='tight')
                plt.close(fig)
                print(f"Saved z-score distributions for {n_plots} towers to {image_dir / f'{name}_frequent_hot_50pct_zscore.png'}")

    return freq_df


def plot_status_run_fraction_distributions(tower_status_per_run, total_runs, output_dir, name):
    """
    Plot 1D distributions of the fraction of runs towers are flagged as Hot, Cold, or Bad Chi2.
    Produces separate plots for Hot (status 2), Cold (status 3), and Bad Chi2 (status 4).
    Uses unfilled step histogram style ('step') with a logarithmic y-axis scale.
    """
    if not tower_status_per_run:
        return

    if total_runs is None or total_runs <= 0:
        total_runs = len(tower_status_per_run)

    pdf_dir = output_dir / "pdf"
    image_dir = output_dir / "images"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)

    status_configs = [
        {
            'status_code': 2,
            'status_name': 'Hot',
            'color': '#d62728',
            'xlabel': 'Fraction of Runs Flagged as Hot',
            'title': 'Hot Towers vs Run Fraction',
            'base_filename': f"{name}_fraction_hot_towers",
            'csv_filename': f"{name}_frequent_hot_towers.csv",
        },
        {
            'status_code': 3,
            'status_name': 'Cold',
            'color': '#1f77b4',
            'xlabel': 'Fraction of Runs Flagged as Cold',
            'title': 'Cold Towers vs Run Fraction',
            'base_filename': f"{name}_fraction_cold_towers",
            'csv_filename': f"{name}_frequent_cold_towers.csv",
        },
        {
            'status_code': 4,
            'status_name': 'Bad Chi2',
            'color': '#6a0dad',
            'xlabel': r'Fraction of Runs Flagged as Bad $\chi^2$',
            'title': r'Bad $\chi^2$ Towers vs Run Fraction',
            'base_filename': f"{name}_fraction_bad_chi2_towers",
            'csv_filename': f"{name}_frequent_bad_chi2_towers.csv",
        },
    ]

    bins = np.linspace(0, 1, 51)  # 50 bins from 0 to 1 (bin width 0.02)

    for cfg in status_configs:
        code = cfg['status_code']
        tower_counts = {}
        for item in tower_status_per_run:
            if isinstance(item, dict):
                for k, s in item.items():
                    if s == code:
                        tower_counts[k] = tower_counts.get(k, 0) + 1
            elif isinstance(item, (list, np.ndarray)) and code == 2:
                for k in item:
                    tower_counts[int(k)] = tower_counts.get(int(k), 0) + 1

        fracs = np.array([c / total_runs for c in tower_counts.values()]) if tower_counts else np.array([])

        # Also save CSV for Cold and Bad Chi2 if not already generated
        if code in (3, 4) and tower_counts:
            freq_data = []
            for k, count in tower_counts.items():
                ieta = k >> 16
                iphi = k & 0xFFFF
                try:
                    tidx = decode_emcal(k) if decode_emcal else -1
                except Exception:
                    tidx = -1
                freq_data.append({
                    'TowerIndex': tidx,
                    'ieta': ieta,
                    'iphi': iphi,
                    'TowerKey': k,
                    f"{cfg['status_name'].replace(' ', '')}RunCount": count,
                    f"{cfg['status_name'].replace(' ', '')}RunFraction": count / total_runs,
                })
            df = pd.DataFrame(freq_data).sort_values(by=f"{cfg['status_name'].replace(' ', '')}RunCount", ascending=False)
            csv_path = output_dir / cfg['csv_filename']
            df.to_csv(csv_path, index=False)
            print(f"Saved frequently {cfg['status_name'].lower()} towers info ({len(df)} towers) to {csv_path}")

        max_frac_str = f"{np.max(fracs)*100:.1f}%" if len(fracs) > 0 else "N/A"
        stats_text = (
            f"Total Runs = {total_runs}\n"
            f"Flagged Towers = {len(fracs)}\n"
            f"Max Frac: {max_frac_str}\n"
            f"Frac > 10%: {np.sum(fracs > 0.10)}\n"
            f"Frac > 50%: {np.sum(fracs > 0.50)}"
        )

        # Logarithmic scale step plot
        fig, ax = plt.subplots(figsize=(10, 6))
        if len(fracs) > 0:
            n, _, _ = ax.hist(fracs, bins=bins, histtype='step', color=cfg['color'], linewidth=2.0)
            max_n = max(n) if len(n) > 0 and max(n) > 0 else 10
        else:
            max_n = 10
        ax.set_yscale('log')
        ax.set_xlabel(cfg['xlabel'], loc='center', fontsize=18)
        ax.set_ylabel("Number of Towers", loc='center', fontsize=18)
        ax.set_xlim(0, 1)
        ax.set_ylim(bottom=0.5, top=max_n * 3.0)
        ax.xaxis.set_major_locator(MultipleLocator(0.2))
        ax.xaxis.set_minor_locator(MultipleLocator(0.05))
        ax.tick_params(labelsize=16)
        ax.set_title(cfg['title'], fontsize=18, pad=12)
        ax.text(0.72, 0.95, stats_text, transform=ax.transAxes, fontsize=14,
                verticalalignment='top',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='gray'))
        plt.tight_layout()
        pdf_path = pdf_dir / f"{cfg['base_filename']}.pdf"
        png_path = image_dir / f"{cfg['base_filename']}.png"
        plt.savefig(pdf_path, bbox_inches='tight')
        plt.savefig(png_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved {cfg['status_name']} towers fraction 1D plot (log, step) to {png_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot Bad, Dead, and Hot Towers vs Run from a list of ROOT files.")
    parser.add_argument("-f", "--file", type=Path, help="Path to a text file containing ROOT file paths (one per line).")
    parser.add_argument("-o", "--output-dir", type=Path, default=Path("."), help="Directory to save the plots (default: current directory).")
    parser.add_argument("-n", "--name", "--plot-name", dest="name", type=str, default="bad_towers_per_run", help="Base filename for output plots.")
    parser.add_argument("-s", "--sigma-threshold", type=float, default=3, help="N-sigma threshold above average for identifying outlier runs (default: 3).")
    parser.add_argument("--cache-file", type=Path, default=None, help="Path to cache file for processed ROOT file data.")
    parser.add_argument("--no-cache", action="store_true", help="Disable caching and force re-processing of all ROOT files.")
    parser.add_argument("--runs", action="store_true", help="Treat inputs as run numbers and fetch EMCal Bad Tower Maps from CDB.")
    parser.add_argument("--cdbtag", type=str, default="newcdbtag", help="CDB global tag to use when fetching from CDB (default: newcdbtag).")
    parser.add_argument("files", nargs="*", type=str, help="List of ROOT file paths or run numbers")
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
                        file_list.append(line)
        except Exception as e:
            print(f"Error reading file {args.file}: {e}")
            sys.exit(1)

    if not file_list:
        print("Error: You must provide at least one ROOT file/run number or a text file containing them.")
        parser.print_help()
        sys.exit(1)

    print(f"Found {len(file_list)} input items.")

    cache = {}
    default_cache_name = ".bad_towers_cache.pkl"
    cache_path = args.cache_file if args.cache_file is not None else args.output_dir / default_cache_name

    if not args.no_cache and cache_path.exists():
        try:
            with cache_path.open("rb") as f:
                cache = pickle.load(f)
            print(f"Loaded {len(cache)} records from cache ({cache_path}).")
        except Exception as e:
            print(f"Warning: Could not read cache file {cache_path}: {e}")
            cache = {}

    files_to_process = []
    results = []

    for item in file_list:
        if args.runs:
            run_str = str(item)
            try:
                run_num = int(run_str)
            except ValueError:
                print(f"Warning: Expected run number, got {run_str}. Skipping.")
                continue

            cache_key = f"run_{run_num}_{args.cdbtag}"
            mtime = 0

            if not args.no_cache and cache_key in cache:
                entry = cache[cache_key]
                res = entry.get("result")
                if entry.get("version") == 2 and isinstance(res, (tuple, list)) and len(res) == 6 and isinstance(res[4], dict):
                    res_list = list(res)
                    res_list.append((run_num, True, args.cdbtag))
                    results.append(tuple(res_list))
                    continue
            files_to_process.append((run_num, True, args.cdbtag))
        else:
            path = Path(item)
            resolved_str = str(path.resolve())
            mtime = path.stat().st_mtime if path.exists() else 0

            if not args.no_cache and resolved_str in cache:
                entry = cache[resolved_str]
                res = entry.get("result")
                if entry.get("version") == 2 and entry.get("mtime") == mtime and isinstance(res, (tuple, list)) and len(res) == 6 and isinstance(res[4], dict):
                    res_list = list(res)
                    res_list.append((path, False, args.cdbtag))
                    results.append(tuple(res_list))
                    continue
            files_to_process.append((path, False, args.cdbtag))

    if files_to_process:
        print(f"Processing {len(files_to_process)} items ({len(results)} loaded from cache)...")
        max_workers = min(os.cpu_count() or 4, 32)
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            new_results = list(tqdm.tqdm(executor.map(process_item, files_to_process), total=len(files_to_process)))

        for item_data, res in zip(files_to_process, new_results):
            res_list = list(res)
            res_list.append(item_data)
            results.append(tuple(res_list))
            item, is_run, cdbtag = item_data
            if is_run:
                cache_key = f"run_{item}_{cdbtag}"
                mtime = 0
            else:
                path = Path(item)
                cache_key = str(path.resolve())
                mtime = path.stat().st_mtime if path.exists() else 0

            cache[cache_key] = {
                "version": 2,
                "mtime": mtime,
                "result": res
            }

        if not args.no_cache:
            try:
                args.output_dir.mkdir(parents=True, exist_ok=True)
                with cache_path.open("wb") as f:
                    pickle.dump(cache, f)
                print(f"Saved updated cache to {cache_path}.")
            except Exception as e:
                print(f"Warning: Could not write cache file {cache_path}: {e}")
    else:
        print(f"All {len(results)} files loaded from cache.")

    run_numbers = []
    bad_towers_list = []
    dead_towers_list = []
    hot_towers_list = []
    tower_status_list = []
    original_items = []

    for run_num, bad_count, dead_count, hot_count, tower_status, err, item_data in results:
        if err:
            print(err)
        elif run_num is not None:
            run_numbers.append(run_num)
            bad_towers_list.append(bad_count)
            dead_towers_list.append(dead_count)
            hot_towers_list.append(hot_count)
            if isinstance(tower_status, dict):
                tower_status_list.append(tower_status)
            elif isinstance(tower_status, (np.ndarray, list)):
                tower_status_list.append({int(k): 2 for k in tower_status})
            else:
                tower_status_list.append({})

            original_items.append(item_data)

    if not run_numbers:
        print("No valid data found to plot.")
        return

    # Sort by run number
    sorted_tuples = sorted(zip(run_numbers, bad_towers_list, dead_towers_list, hot_towers_list, tower_status_list, original_items), key=lambda x: x[0])
    run_numbers, bad_towers_list, dead_towers_list, hot_towers_list, tower_status_list, original_items = zip(*sorted_tuples)

    counts_data = [
        {'Run': r, 'BadTowers': b, 'DeadTowers': d, 'HotTowers': h}
        for r, b, d, h in zip(run_numbers, bad_towers_list, dead_towers_list, hot_towers_list)
    ]

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if counts_data:
        counts_df = pd.DataFrame(counts_data)
        csv_path = args.output_dir / f"{args.name}.csv"
        counts_df.to_csv(csv_path, index=False)
        print(f"Saved {len(counts_df)} runs to {csv_path}")

    # Outlier Detection for Bad Towers
    runs_arr = np.array(run_numbers)
    bad_arr = np.array(bad_towers_list)
    dead_arr = np.array(dead_towers_list)
    hot_arr = np.array(hot_towers_list)

    groups_for_outliers = [
        ("<= 73500", runs_arr < 73500),
        ("73500-78217", (runs_arr >= 73500) & (runs_arr <= 78217)),
        (">= 78218", runs_arr >= 78218),
    ]

    sigma_val = args.sigma_threshold
    sigma_str = f"{sigma_val:g}"
    sigma_tag = sigma_str.replace('.', 'p')

    outlier_rows = []
    for g_name, g_mask in groups_for_outliers:
        if not np.any(g_mask):
            continue
        g_bad = bad_arr[g_mask]
        g_runs = runs_arr[g_mask]
        g_dead = dead_arr[g_mask]
        g_hot = hot_arr[g_mask]

        g_mean = np.mean(g_bad)
        g_std = np.std(g_bad)
        thresh_sigma = g_mean + sigma_val * g_std

        outlier_mask = g_bad > thresh_sigma
        for r, b, d, h in zip(g_runs[outlier_mask], g_bad[outlier_mask], g_dead[outlier_mask], g_hot[outlier_mask]):
            sigma_dev = (b - g_mean) / g_std if g_std > 0 else 0
            outlier_rows.append({
                'Run': int(r),
                'Group': g_name,
                'BadTowers': int(b),
                'DeadTowers': int(d),
                'HotTowers': int(h),
                'GroupMean': round(g_mean, 1),
                'GroupStd': round(g_std, 1),
                f'Threshold{sigma_tag}Sigma': round(thresh_sigma, 1),
                'SigmaDeviation': round(sigma_dev, 2)
            })

    if outlier_rows:
        outliers_df = pd.DataFrame(outlier_rows)
        outliers_csv_path = args.output_dir / f"{args.name}_outliers_{sigma_tag}sigma.csv"
        outliers_df.to_csv(outliers_csv_path, index=False)
        print(f"Saved {len(outliers_df)} {sigma_str}-sigma outlier runs to {outliers_csv_path}")
    else:
        print(f"No {sigma_str}-sigma outlier runs detected.")

    # Plot Bad Towers with Sigma threshold lines
    plot_towers(
        run_numbers, bad_towers_list, args.output_dir, args.name,
        ylabel="Number of Bad Towers", suffix="", ylim_bottom=550,
        legend_fontsize=16, sigma_threshold=args.sigma_threshold
    )

    # Plot Dead Towers (value == 1)
    dead_name = "dead_towers_per_run" if args.name == "bad_towers_per_run" else f"{args.name}_dead"
    plot_towers(
        run_numbers, dead_towers_list, args.output_dir, dead_name,
        ylabel="Number of Dead Towers", suffix="", ylim_bottom=None
    )

    # Plot Hot Towers (value == 2)
    hot_name = "hot_towers_per_run" if args.name == "bad_towers_per_run" else f"{args.name}_hot"
    plot_towers(
        run_numbers, hot_towers_list, args.output_dir, hot_name,
        ylabel="Number of Hot Towers", suffix="", ylim_bottom=None, legend_loc='upper center', legend_fontsize=14
    )

    # Frequently Hot Towers Analysis
    plot_frequently_hot_towers(
        tower_status_list, original_items, run_numbers, args.output_dir, args.name,
        total_runs=len(run_numbers), no_cache=args.no_cache
    )

    # 1D Status Run Fraction Distributions (Hot, Cold, Bad Chi2)
    plot_status_run_fraction_distributions(
        tower_status_list, len(run_numbers), args.output_dir, args.name
    )

if __name__ == "__main__":
    main()
