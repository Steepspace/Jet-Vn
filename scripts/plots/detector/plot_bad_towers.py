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
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
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


def plot_frequently_hot_towers(tower_status_per_run, output_dir, name, total_runs=None):
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

    # Plot 3: 2D Run Index vs Tower for towers hot in >50% of the runs
    hot_50_df = freq_df[freq_df['HotRunFraction'] > 0.5]
    if len(hot_50_df) == 0:
        print("No towers were hot in >50% of the runs. Skipping 50% hot towers run index plot.")
    else:
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
        plt.savefig(image_dir / f"{name}_frequent_hot_50pct_run_index.png", dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved hot towers (>50%) vs run index plot ({n_towers} towers) to {image_dir / f'{name}_frequent_hot_50pct_run_index.png'}")

    return freq_df


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
                    results.append(res)
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
                    results.append(res)
                    continue
            files_to_process.append((path, False, args.cdbtag))

    if files_to_process:
        print(f"Processing {len(files_to_process)} items ({len(results)} loaded from cache)...")
        max_workers = min(os.cpu_count() or 4, 32)
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            new_results = list(tqdm.tqdm(executor.map(process_item, files_to_process), total=len(files_to_process)))

        for item_data, res in zip(files_to_process, new_results):
            results.append(res)
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

    for run_num, bad_count, dead_count, hot_count, tower_status, err in results:
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

    if not run_numbers:
        print("No valid data found to plot.")
        return

    # Sort by run number
    sorted_tuples = sorted(zip(run_numbers, bad_towers_list, dead_towers_list, hot_towers_list, tower_status_list), key=lambda x: x[0])
    run_numbers, bad_towers_list, dead_towers_list, hot_towers_list, tower_status_list = zip(*sorted_tuples)

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
    plot_frequently_hot_towers(tower_status_list, args.output_dir, args.name, total_runs=len(run_numbers))

if __name__ == "__main__":
    main()
