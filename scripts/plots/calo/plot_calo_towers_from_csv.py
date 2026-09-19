#!/usr/bin/env python3
"""
Plot 1D EMCal tower energy distributions (h2EMCalEnergyTowerIndex)
for towers specified in a CSV file (ieta, iphi, run).

Usage example:
    python plot_calo_towers_from_csv.py -c towers.csv -f file_list.txt -o ./output_plots
    python plot_calo_towers_from_csv.py -c towers.csv /path/to/68144.root -o ./output_plots
    python plot_calo_towers_from_csv.py -c towers.csv -f file_list.txt --ref-tower 5000 -o ./output_plots
"""

import argparse
import concurrent.futures
import csv
import functools
import os
from pathlib import Path
import re
import sys
import traceback

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, ScalarFormatter
import mplhep as hep
import numpy as np
import tqdm
import uproot

# Ensure tower_info_defs can be imported from current directory
sys.path.insert(0, str(Path(__file__).resolve().parent))
from tower_info_defs import (
    get_calo_tower_index,
    get_calo_tower_ieta_iphi,
    get_calo_tower_key,
    get_bad_tower_map,
    get_frac_bad_chi2_map,
    get_calib_adc_to_etower_map,
)


def clean_root_latex(text):
    """Clean ROOT-style LaTeX strings for matplotlib mathtext."""
    if not text:
        return text
    text = text.replace("#it{p}_{T}", r"$p_{T}$")
    text = text.replace("#it{p}_{T}^{leading}", r"$p_{T}^{\mathrm{leading}}$")
    text = text.replace("#it{E}_{T}", r"$E_{T}$")
    text = text.replace("#eta", r"$\eta$")
    text = text.replace("#phi", r"$\phi$")
    text = text.replace("#chi^{2}", r"$\chi^{2}$")
    text = text.replace("#Delta", r"$\Delta$")
    text = text.replace("#sigma", r"$\sigma$")
    return text


def get_hist_axis_titles(hist2d, hist_name=""):
    """Extract and clean x and y axis titles from a 2D histogram."""
    xlabel = ""
    ylabel = ""
    title = ""

    if hasattr(hist2d, "member"):
        try:
            xlabel = hist2d.member("fXaxis").member("fTitle")
            ylabel = hist2d.member("fYaxis").member("fTitle")
            title = hist2d.member("fTitle")
        except Exception:
            pass

    if hasattr(hist2d, "all_members"):
        try:
            members = hist2d.all_members
            if not xlabel and "fXaxis" in members:
                xlabel = members["fXaxis"].all_members.get("fTitle", "")
            if not ylabel and "fYaxis" in members:
                ylabel = members["fYaxis"].all_members.get("fTitle", "")
            if not title:
                title = members.get("fTitle", "")
        except Exception:
            pass

    raw_title = title if title else (getattr(hist2d, "title", "") or "")
    if raw_title and ";" in raw_title:
        parts = [p.strip() for p in raw_title.split(";")]
        if not xlabel and len(parts) > 1:
            xlabel = parts[1]
        if not ylabel and len(parts) > 2:
            ylabel = parts[2]

    xlabel = clean_root_latex(xlabel)
    ylabel = clean_root_latex(ylabel)

    return xlabel, ylabel


def format_run_header(run_number):
    """Format run number for the top-right plot header."""
    if str(run_number).lower() in ("combined", "run-combined") or "combined" in str(run_number).lower():
        return "Combined Runs"
    return rf"Run: {run_number}"


def load_tower_csv(csv_path):
    """
    Parse CSV containing tower coordinates (ieta, iphi) with optional run.

    Supports:
      - 3 columns: ieta, iphi, run (or tower_index, run) -> has_run=True
      - 2 columns: ieta, iphi (or tower_index, None)     -> has_run=False
      - Headers in any order (e.g. ieta, iphi, run or eta, phi, run or ieta, iphi)
      - Headerless files with 2 or 3 columns
      - Comma, tab, or whitespace delimited files
      - Comments starting with '#'

    Returns:
      towers_data: dict (run -> list of (ieta, iphi, tower_idx)) if has_run else list of (ieta, iphi, tower_idx)
      all_runs: set of all unique runs found in the CSV (empty if has_run is False)
      has_run: bool indicating whether run column was present
    """
    with open(csv_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip() and not line.strip().startswith("#")]

    if not lines:
        return {}, set(), False

    first_line = lines[0].lower()
    has_header = any(k in first_line for k in ["eta", "phi", "run", "tower", "index"])

    ieta_col = None
    iphi_col = None
    tower_col = None
    run_col = None
    start_idx = 0

    if has_header:
        start_idx = 1
        reader = csv.reader([lines[0]])
        header = [col.strip().lower() for col in next(reader)]
        for idx, col in enumerate(header):
            if col in ("ieta", "eta", "etabin", "i_eta") or ("eta" in col and "theta" not in col):
                ieta_col = idx
            elif col in ("iphi", "phi", "phibin", "i_phi") or "phi" in col:
                iphi_col = idx
            elif col in ("tower", "tower_index", "towerindex", "tower_id", "towerid", "index"):
                tower_col = idx
            elif col in ("run", "runnumber", "run_number", "run_id", "runid") or (col.startswith("run") and "count" not in col and "frac" not in col):
                run_col = idx

    has_run = (run_col is not None)

    if not has_header:
        first_data = lines[0]
        if "," in first_data:
            sample_parts = [p.strip() for p in first_data.split(",") if p.strip()]
        elif "\t" in first_data:
            sample_parts = [p.strip() for p in first_data.split("\t") if p.strip()]
        else:
            sample_parts = first_data.split()

        if len(sample_parts) >= 3:
            ieta_col = 0
            iphi_col = 1
            run_col = 2
            has_run = True
        elif len(sample_parts) == 2:
            ieta_col = 0
            iphi_col = 1
            run_col = None
            has_run = False
        elif len(sample_parts) == 1:
            tower_col = 0
            run_col = None
            has_run = False
    else:
        if ieta_col is None and tower_col is None:
            ieta_col = 0
        if iphi_col is None and tower_col is None:
            iphi_col = 1

    towers_by_run = {}
    tower_list = []
    all_runs = set()

    for line_num, line in enumerate(lines[start_idx:], start=start_idx + 1):
        if "," in line:
            parts = [p.strip() for p in line.split(",") if p.strip()]
        elif "\t" in line:
            parts = [p.strip() for p in line.split("\t")]
        else:
            parts = line.split()

        required_cols = [c for c in [ieta_col, iphi_col, tower_col] if c is not None]
        if has_run and run_col is not None:
            required_cols.append(run_col)

        if len(parts) < max(required_cols) + 1:
            print(f"Warning: Skipping malformed line {line_num} in {csv_path}: '{line}'")
            continue

        try:
            if tower_col is not None and (ieta_col is None or iphi_col is None):
                tower_idx = int(parts[tower_col])
                ieta, iphi = get_calo_tower_ieta_iphi(tower_idx, det="EMCal")
            else:
                ieta = int(parts[ieta_col])
                iphi = int(parts[iphi_col])
                tower_idx = get_calo_tower_index(ieta, iphi, det="EMCal")

            if tower_idx is None or tower_idx < 0 or tower_idx >= 24576:
                print(f"Warning: Invalid EMCal coordinates (ieta={ieta}, iphi={iphi}) on line {line_num}")
                continue

            entry = (ieta, iphi, tower_idx)

            if has_run:
                run_str = parts[run_col]
                try:
                    run_val = int(float(run_str))
                except ValueError:
                    run_val = run_str

                if run_val not in towers_by_run:
                    towers_by_run[run_val] = []
                if entry not in towers_by_run[run_val]:
                    towers_by_run[run_val].append(entry)
                all_runs.add(run_val)
            else:
                if entry not in tower_list:
                    tower_list.append(entry)
        except (ValueError, TypeError) as e:
            print(f"Warning: Could not parse numbers from line {line_num} ('{line}'): {e}")

    if has_run:
        return towers_by_run, all_runs, True
    else:
        return tower_list, set(), False


def make_1d_yproj_plot(
    hist2d,
    run_number,
    output_path,
    hist_name="h2EMCalEnergyTowerIndex",
    tower_index=None,
    ieta=None,
    iphi=None,
    label_text=None,
    z_score=None,
    frac_bad_chi2=None,
    calib=None,
    logy=True,
    auto_xlim=True,
    ref_tower_index=None,
    ref_z_score=None,
    ref_frac_bad_chi2=None,
    ref_calib=None,
):
    """Generate and save 1D y-projection plot for an EMCal tower, optionally overlaying a reference tower."""
    hep.style.use("ATLAS")
    fig, ax = plt.subplots(figsize=(8, 6))

    values, xedges, yedges = hist2d.to_numpy()

    tower_info_lines = []
    proj_y_ref = None
    if ref_tower_index is not None:
        if 0 <= ref_tower_index < values.shape[0]:
            ref_data = values[ref_tower_index, :]
            if np.any(ref_data > 0):
                proj_y_ref = ref_data
            else:
                print(f"Warning: Reference tower {ref_tower_index} has no positive values; omitting from overlay.")
        else:
            print(f"Warning: Reference tower index {ref_tower_index} out of bounds (0, {values.shape[0]})")

    if tower_index is not None:
        if 0 <= tower_index < values.shape[0]:
            proj_y = values[tower_index, :]
        else:
            msg = f"Tower index {tower_index} out of bounds (0, {values.shape[0]})"
            print(f"Warning: {msg}")
            plt.close(fig)
            return False, msg

        if not np.any(proj_y > 0):
            plt.close(fig)
            return False, "Data has no positive values (zero counts)"

        if label_text is None:
            tower_info_lines.append(f"Tower Index: {tower_index}")
            if ieta is None or iphi is None:
                ieta, iphi = get_calo_tower_ieta_iphi(tower_index, hist_name)
            if ieta is not None and iphi is not None:
                tower_info_lines.append(f"ieta: {ieta}, iphi: {iphi}")
            if z_score is not None and not (isinstance(z_score, float) and np.isnan(z_score)):
                tower_info_lines.append(f"z-score: {z_score:+.2f}")
            if frac_bad_chi2 is not None and not (isinstance(frac_bad_chi2, float) and np.isnan(frac_bad_chi2)):
                if 0 < abs(frac_bad_chi2) < 0.01:
                    frac_str = f"{frac_bad_chi2:.2e}"
                else:
                    frac_str = f"{frac_bad_chi2:.2f}"
                tower_info_lines.append(f"frac badChi2: {frac_str}")
            if calib is not None and not (isinstance(calib, float) and np.isnan(calib)):
                calib_mev = calib * 1000.0
                tower_info_lines.append(f"calib: {calib_mev:.2f} MeV/ADC")
        else:
            tower_info_lines.append(label_text)
    else:
        proj_y = np.sum(values, axis=0)

    _, ylabel = get_hist_axis_titles(hist2d, hist_name)
    if not ylabel:
        ylabel = r"Tower Energy [GeV]"

    if proj_y_ref is not None:
        def format_overlay_legend_label(prefix, t_idx, eta_val, phi_val, z_val, f_val, c_val=None):
            if eta_val is None or phi_val is None:
                eta_val, phi_val = get_calo_tower_ieta_iphi(t_idx, hist_name)
            line1 = f"{prefix}: Tower {t_idx}"
            if eta_val is not None and phi_val is not None:
                line1 += f" (ieta: {eta_val}, iphi: {phi_val})"
            line2_parts = []
            if z_val is not None and not (isinstance(z_val, float) and np.isnan(z_val)):
                line2_parts.append(f"z-score: {z_val:+.2f}")
            if f_val is not None and not (isinstance(f_val, float) and np.isnan(f_val)):
                c_str = f"{f_val:.2e}" if 0 < abs(f_val) < 0.01 else f"{f_val:.2f}"
                line2_parts.append(f"frac badChi2: {c_str}")
            if c_val is not None and not (isinstance(c_val, float) and np.isnan(c_val)):
                c_mev = c_val * 1000.0
                line2_parts.append(f"calib: {c_mev:.2f} MeV/ADC")
            if line2_parts:
                return line1 + "\n  " + ", ".join(line2_parts)
            return line1

        label_tower = format_overlay_legend_label("Tower", tower_index, ieta, iphi, z_score, frac_bad_chi2, calib)
        label_ref = format_overlay_legend_label("Ref", ref_tower_index, None, None, ref_z_score, ref_frac_bad_chi2, ref_calib)

        hep.histplot((proj_y, yedges), ax=ax, histtype='step', color='crimson', linewidth=2, label=label_tower)
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

    ax.text(1.0, 1.01, format_run_header(run_number), transform=ax.transAxes, ha='right', va='bottom', fontsize=15)

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

    fig.tight_layout()
    plt.subplots_adjust(left=0.12, bottom=0.13, top=0.93)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return True, None


def get_energy_mask(yedges, lower_threshold=None, upper_threshold=None):
    """
    Returns a boolean mask over the energy bins (length len(yedges)-1)
    satisfying lower_threshold and/or upper_threshold.
    """
    if lower_threshold is None and upper_threshold is None:
        return None

    y_low = yedges[:-1]
    y_high = yedges[1:]
    y_centers = 0.5 * (y_low + y_high)

    mask = np.ones(len(yedges) - 1, dtype=bool)

    if upper_threshold is not None:
        mask_upper = y_high <= upper_threshold
        if not np.any(mask_upper):
            mask_upper = y_centers <= upper_threshold
        mask &= mask_upper

    if lower_threshold is not None:
        mask_lower = y_low >= lower_threshold
        if not np.any(mask_lower):
            mask_lower = y_centers >= lower_threshold
        mask &= mask_lower

    return mask


def format_threshold_str(lower_threshold=None, upper_threshold=None):
    """Format human-readable description of the energy threshold constraint."""
    if lower_threshold is not None and upper_threshold is not None:
        return f"{lower_threshold} <= E <= {upper_threshold} GeV"
    elif lower_threshold is not None:
        return f"E >= {lower_threshold} GeV"
    elif upper_threshold is not None:
        return f"E <= {upper_threshold} GeV"
    return "any non-zero counts"


def process_file(
    path,
    towers_by_run,
    output_dir=None,
    hist_name="h2EMCalEnergyTowerIndex",
    lower_threshold=None,
    upper_threshold=None,
    use_cdb=True,
    cdbtag="newcdbtag",
    ref_tower=None,
    include_coords=False,
):
    """Process a single ROOT file and plot the requested towers for its corresponding run."""
    path = Path(path)
    if not path.exists():
        return f"File not found: {path}"

    try:
        stem = path.stem
        if "combined" in stem.lower():
            run_number = "combined"
        else:
            try:
                run_number = int(stem)
            except ValueError:
                match = re.search(r'\d+', stem)
                if match:
                    run_number = int(match.group())
                else:
                    run_number = stem

        is_combined = not isinstance(run_number, int) or "combined" in str(run_number).lower()

        # Match towers for this run
        towers_to_plot = []
        if run_number in towers_by_run:
            towers_to_plot.extend(towers_by_run[run_number])
        elif str(run_number) in towers_by_run:
            towers_to_plot.extend(towers_by_run[str(run_number)])
        elif is_combined:
            # If combined file, include all towers from all runs in the CSV
            all_unique = []
            for r_towers in towers_by_run.values():
                for t in r_towers:
                    if t not in all_unique:
                        all_unique.append(t)
            towers_to_plot = all_unique
        elif "all" in towers_by_run:
            towers_to_plot.extend(towers_by_run["all"])
        elif -1 in towers_by_run:
            towers_to_plot.extend(towers_by_run[-1])

        if not towers_to_plot:
            return None

        run_output_dir = output_dir if output_dir is not None else Path(".")
        run_output_dir.mkdir(parents=True, exist_ok=True)

        with uproot.open(path) as file:
            if hist_name not in file:
                return f"[{path.name}] Histogram '{hist_name}' not found."

            hist2d = file[hist_name]
            values, _, yedges = hist2d.to_numpy()

            mask = get_energy_mask(yedges, lower_threshold=lower_threshold, upper_threshold=upper_threshold)
            if mask is not None:
                towers_to_plot = [
                    t for t in towers_to_plot
                    if 0 <= t[2] < values.shape[0] and (np.any(mask) and np.any(values[t[2], mask] > 0))
                ]
                if not towers_to_plot:
                    thresh_str = format_threshold_str(lower_threshold, upper_threshold)
                    print(f"[{path.name}] No requested towers had counts satisfying threshold ({thresh_str}) for run {run_number}.")
                    return {
                        "path": str(path),
                        "run": run_number,
                        "requested": len(towers_by_run.get(run_number, [])),
                        "plotted": [],
                        "failed": [(t[0], t[1], t[2], f"No counts satisfying threshold ({thresh_str})") for t in towers_by_run.get(run_number, [])],
                        "error": None,
                    }

            # Query CDB BadTowerMap calibration if available
            bad_tower_map = {}
            frac_bad_chi2_map = {}
            calib_map = {}
            if use_cdb and not is_combined:
                bad_tower_map = get_bad_tower_map(run_number, det="CEMC", dbtag=cdbtag)
                frac_bad_chi2_map = get_frac_bad_chi2_map(run_number, det="CEMC", dbtag=cdbtag)
                calib_map = get_calib_adc_to_etower_map(run_number, det="CEMC", dbtag=cdbtag)

            ref_z_score = None
            ref_frac_bad_chi2 = None
            ref_calib = None
            if ref_tower is not None and use_cdb and not is_combined:
                ref_key = get_calo_tower_key(ref_tower, det="EMCal")
                ref_z_score = bad_tower_map.get(ref_key, {}).get("sigma") if bad_tower_map else None
                ref_frac_bad_chi2 = frac_bad_chi2_map.get(ref_key) if frac_bad_chi2_map else None
                ref_calib = calib_map.get(ref_key) if calib_map else None

            plotted_towers = []
            failed_towers = []

            for ieta, iphi, tower_idx in towers_to_plot:
                coords_str = f"_eta{ieta}_phi{iphi}" if include_coords else ""
                out_filename_tower = f"run_{run_number}_{hist_name}_tower{tower_idx}{coords_str}.png"
                output_path_tower = run_output_dir / out_filename_tower

                tower_key = get_calo_tower_key(tower_idx, det="EMCal")
                z_score = bad_tower_map.get(tower_key, {}).get("sigma") if bad_tower_map else None
                frac_bad_chi2 = frac_bad_chi2_map.get(tower_key) if frac_bad_chi2_map else None
                calib = calib_map.get(tower_key) if calib_map else None

                ok, reason = make_1d_yproj_plot(
                    hist2d,
                    run_number,
                    output_path_tower,
                    hist_name=hist_name,
                    tower_index=tower_idx,
                    ieta=ieta,
                    iphi=iphi,
                    z_score=z_score,
                    frac_bad_chi2=frac_bad_chi2,
                    calib=calib,
                    logy=True,
                    auto_xlim=True,
                )

                if not ok:
                    failed_towers.append((ieta, iphi, tower_idx, reason))
                    continue

                plotted_towers.append((ieta, iphi, tower_idx))

                if ref_tower is not None:
                    out_filename_overlay = f"run_{run_number}_{hist_name}_tower{tower_idx}_ref{ref_tower}{coords_str}.png"
                    output_path_overlay = run_output_dir / out_filename_overlay
                    make_1d_yproj_plot(
                        hist2d,
                        run_number,
                        output_path_overlay,
                        hist_name=hist_name,
                        tower_index=tower_idx,
                        ieta=ieta,
                        iphi=iphi,
                        z_score=z_score,
                        frac_bad_chi2=frac_bad_chi2,
                        calib=calib,
                        ref_tower_index=ref_tower,
                        ref_z_score=ref_z_score,
                        ref_frac_bad_chi2=ref_frac_bad_chi2,
                        ref_calib=ref_calib,
                        logy=True,
                        auto_xlim=True,
                    )

            if failed_towers:
                print(f"[{path.name}] Plotted {len(plotted_towers)} / {len(towers_to_plot)} tower(s) for run {run_number} ({len(failed_towers)} failed/skipped).")
            else:
                print(f"[{path.name}] Plotted {len(plotted_towers)} tower(s) for run {run_number}.")

            return {
                "path": str(path),
                "run": run_number,
                "requested": len(towers_to_plot),
                "plotted": plotted_towers,
                "failed": failed_towers,
                "error": None,
            }

    except Exception as e:
        traceback.print_exc()
        return {
            "path": str(path),
            "run": locals().get("run_number", getattr(path, "stem", str(path))),
            "requested": len(locals().get("towers_to_plot", [])),
            "plotted": locals().get("plotted_towers", []),
            "failed": locals().get("failed_towers", []),
            "error": f"Error processing {path}: {e}",
        }


def process_towers_without_runs(
    files,
    tower_list,
    output_dir,
    hist_name="h2EMCalEnergyTowerIndex",
    lower_threshold=None,
    upper_threshold=None,
    first_match=False,
    use_cdb=True,
    cdbtag="newcdbtag",
    ref_tower=None,
    include_coords=False,
):
    """
    For CSVs with only (ieta, iphi) [no run specified]:
    Plots each tower at most ONCE across the provided ROOT files list.

    Default behavior (first_match=False):
      - If threshold (lower, upper, or both) is provided: finds the run where each
        tower has the MOST counts satisfying the threshold condition.
      - If no threshold is provided: finds the run where each tower has the HIGHEST
        absolute value of energy (|E|).
      Then plots each tower exactly once from its winning run.

    If first_match=True:
      Searches runs sequentially and plots a tower from the first run satisfying the
      criteria (meeting threshold if specified, or having counts > 0).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    has_threshold = (lower_threshold is not None or upper_threshold is not None)
    thresh_str = format_threshold_str(lower_threshold, upper_threshold)

    # --- Fast Sequential Search Mode (first_match=True) ---
    if first_match:
        unplotted = {tower_idx: (ieta, iphi) for ieta, iphi, tower_idx in tower_list}
        plotted_towers = set()
        failed_towers = []

        for path in files:
            if not unplotted:
                print("All towers from CSV have been plotted. Stopping early.")
                break

            path = Path(path)
            if not path.exists():
                print(f"Warning: File not found: {path}")
                continue

            stem = path.stem
            if "combined" in stem.lower():
                run_number = "combined"
            else:
                try:
                    run_number = int(stem)
                except ValueError:
                    match = re.search(r'\d+', stem)
                    run_number = int(match.group()) if match else stem

            is_combined = not isinstance(run_number, int) or "combined" in str(run_number).lower()

            try:
                with uproot.open(path) as file:
                    if hist_name not in file:
                        print(f"[{path.name}] Histogram '{hist_name}' not found. Skipping.")
                        continue

                    hist2d = file[hist_name]
                    values, xedges, yedges = hist2d.to_numpy()

                    mask = get_energy_mask(yedges, lower_threshold=lower_threshold, upper_threshold=upper_threshold)

                    towers_to_plot_now = []
                    for tower_idx, (ieta, iphi) in list(unplotted.items()):
                        if 0 <= tower_idx < values.shape[0]:
                            proj_y = values[tower_idx, :]
                            if mask is not None:
                                has_match = np.any(mask) and np.any(proj_y[mask] > 0)
                            else:
                                has_match = np.any(proj_y > 0)

                            if has_match:
                                towers_to_plot_now.append((ieta, iphi, tower_idx))

                    if not towers_to_plot_now:
                        if mask is not None:
                            print(f"[{path.name}] No unplotted towers had counts satisfying threshold ({thresh_str}) in run {run_number}.")
                        else:
                            print(f"[{path.name}] No unplotted towers had non-zero counts in run {run_number}.")
                        continue

                    bad_tower_map = {}
                    frac_bad_chi2_map = {}
                    calib_map = {}
                    if use_cdb and not is_combined:
                        bad_tower_map = get_bad_tower_map(run_number, det="CEMC", dbtag=cdbtag)
                        frac_bad_chi2_map = get_frac_bad_chi2_map(run_number, det="CEMC", dbtag=cdbtag)
                        calib_map = get_calib_adc_to_etower_map(run_number, det="CEMC", dbtag=cdbtag)

                    ref_z_score = None
                    ref_frac_bad_chi2 = None
                    ref_calib = None
                    if ref_tower is not None and use_cdb and not is_combined:
                        ref_key = get_calo_tower_key(ref_tower, det="EMCal")
                        ref_z_score = bad_tower_map.get(ref_key, {}).get("sigma") if bad_tower_map else None
                        ref_frac_bad_chi2 = frac_bad_chi2_map.get(ref_key) if frac_bad_chi2_map else None
                        ref_calib = calib_map.get(ref_key) if calib_map else None

                    for ieta, iphi, tower_idx in towers_to_plot_now:
                        coords_str = f"_eta{ieta}_phi{iphi}" if include_coords else ""
                        out_filename_tower = f"run_{run_number}_{hist_name}_tower{tower_idx}{coords_str}.png"
                        output_path_tower = output_dir / out_filename_tower

                        tower_key = get_calo_tower_key(tower_idx, det="EMCal")
                        z_score = bad_tower_map.get(tower_key, {}).get("sigma") if bad_tower_map else None
                        frac_bad_chi2 = frac_bad_chi2_map.get(tower_key) if frac_bad_chi2_map else None
                        calib = calib_map.get(tower_key) if calib_map else None

                        ok, reason = make_1d_yproj_plot(
                            hist2d,
                            run_number,
                            output_path_tower,
                            hist_name=hist_name,
                            tower_index=tower_idx,
                            ieta=ieta,
                            iphi=iphi,
                            z_score=z_score,
                            frac_bad_chi2=frac_bad_chi2,
                            calib=calib,
                            logy=True,
                            auto_xlim=True,
                        )

                        if not ok:
                            failed_towers.append((run_number, ieta, iphi, tower_idx, reason))
                        else:
                            plotted_towers.add(tower_idx)
                            if ref_tower is not None:
                                out_filename_overlay = f"run_{run_number}_{hist_name}_tower{tower_idx}_ref{ref_tower}{coords_str}.png"
                                output_path_overlay = output_dir / out_filename_overlay
                                make_1d_yproj_plot(
                                    hist2d,
                                    run_number,
                                    output_path_overlay,
                                    hist_name=hist_name,
                                    tower_index=tower_idx,
                                    ieta=ieta,
                                    iphi=iphi,
                                    z_score=z_score,
                                    frac_bad_chi2=frac_bad_chi2,
                                    calib=calib,
                                    ref_tower_index=ref_tower,
                                    ref_z_score=ref_z_score,
                                    ref_frac_bad_chi2=ref_frac_bad_chi2,
                                    ref_calib=ref_calib,
                                    logy=True,
                                    auto_xlim=True,
                                )

                        unplotted.pop(tower_idx, None)

                    print(f"[{path.name}] Plotted {len(towers_to_plot_now)} tower(s) for run {run_number} ({len(unplotted)} remaining).")

            except Exception as e:
                traceback.print_exc()
                print(f"Error processing {path}: {e}")

        total_failed = len(unplotted) + len(failed_towers)
        print("\n" + "=" * 80)
        print("Tower Plotting Summary")
        print("=" * 80)
        print(f"Total towers requested: {len(tower_list)}")
        print(f"Successfully plotted:   {len(plotted_towers)}")
        print(f"Failed / Skipped:       {total_failed}")
        print(f"Plots saved to:         {output_dir}")

        if total_failed > 0:
            print("-" * 80)
            print(f"Failed / Skipped Towers Details ({total_failed}):")
            for t_idx, (ieta, iphi) in unplotted.items():
                reason = f"No counts satisfying threshold ({thresh_str})" if has_threshold else "Zero counts across all provided runs"
                print(f"  - Tower (ieta={ieta:2d}, iphi={iphi:3d}, idx={t_idx:5d}): {reason}")
            for run_val, ieta, iphi, t_idx, reason in failed_towers:
                print(f"  - Run {run_val}, Tower (ieta={ieta:2d}, iphi={iphi:3d}, idx={t_idx:5d}): {reason}")
        print("=" * 80)
        return

    # --- Best Run Search Mode (Default: Most counts if threshold, or highest |E| if no threshold) ---
    if has_threshold:
        print(f"Scanning {len(files)} file(s) to find runs with the MOST counts satisfying threshold ({thresh_str}) for {len(tower_list)} tower(s)...")
    else:
        print(f"Scanning {len(files)} file(s) to find runs with HIGHEST absolute energy for {len(tower_list)} tower(s)...")

    towers_dict = {t[2]: (t[0], t[1]) for t in tower_list}
    req_indices = np.array(list(towers_dict.keys()), dtype=int)
    best_tower_info = {}

    for path in tqdm.tqdm(files, desc="Scanning runs"):
        path = Path(path)
        if not path.exists():
            print(f"Warning: File not found: {path}")
            continue

        stem = path.stem
        if "combined" in stem.lower():
            run_number = "combined"
        else:
            try:
                run_number = int(stem)
            except ValueError:
                match = re.search(r'\d+', stem)
                run_number = int(match.group()) if match else stem

        try:
            with uproot.open(path) as file:
                if hist_name not in file:
                    continue

                hist2d = file[hist_name]
                values, xedges, yedges = hist2d.to_numpy()

                valid_mask = (req_indices >= 0) & (req_indices < values.shape[0])
                valid_indices = req_indices[valid_mask]
                if len(valid_indices) == 0:
                    continue

                sub_values = values[valid_indices, :]

                if has_threshold:
                    mask = get_energy_mask(yedges, lower_threshold=lower_threshold, upper_threshold=upper_threshold)
                    if mask is not None and np.any(mask):
                        counts_in_thresh = np.sum(sub_values[:, mask], axis=1)
                    else:
                        counts_in_thresh = np.zeros(len(valid_indices), dtype=np.float64)

                    for idx, tower_idx in enumerate(valid_indices):
                        c = float(counts_in_thresh[idx])
                        if c > 0:
                            prev = best_tower_info.get(tower_idx)
                            if prev is None or c > prev['score']:
                                ieta, iphi = towers_dict[tower_idx]
                                best_tower_info[tower_idx] = {
                                    'score': c,
                                    'path': path,
                                    'run_number': run_number,
                                    'ieta': ieta,
                                    'iphi': iphi,
                                }
                else:
                    y_centers = 0.5 * (yedges[:-1] + yedges[1:])
                    abs_y = np.abs(y_centers)
                    has_counts_mask = sub_values > 0

                    for idx, tower_idx in enumerate(valid_indices):
                        nz = has_counts_mask[idx]
                        if np.any(nz):
                            cur_max_abs_e = float(np.max(abs_y[nz]))
                            prev = best_tower_info.get(tower_idx)
                            if prev is None or cur_max_abs_e > prev['score']:
                                ieta, iphi = towers_dict[tower_idx]
                                best_tower_info[tower_idx] = {
                                    'score': cur_max_abs_e,
                                    'path': path,
                                    'run_number': run_number,
                                    'ieta': ieta,
                                    'iphi': iphi,
                                }

        except Exception as e:
            traceback.print_exc()
            print(f"Error scanning {path}: {e}")

    if not best_tower_info:
        if has_threshold:
            print(f"Notice: None of the requested towers had counts satisfying threshold ({thresh_str}) in any of the provided runs.")
        else:
            print("Notice: None of the requested towers had non-zero counts in any of the provided runs.")
        return

    unplotted = [t_idx for t_idx in towers_dict if t_idx not in best_tower_info]
    if unplotted:
        if has_threshold:
            print(f"Notice: {len(unplotted)} tower(s) did not have counts satisfying threshold ({thresh_str}) across any provided runs and will not be plotted.")
        else:
            print(f"Notice: {len(unplotted)} tower(s) had zero counts across all provided runs and will not be plotted.")

    towers_by_file = {}
    for tower_idx, info in best_tower_info.items():
        p = info['path']
        if p not in towers_by_file:
            towers_by_file[p] = (info['run_number'], [])
        towers_by_file[p][1].append((info['ieta'], info['iphi'], tower_idx, info['score']))

    print(f"Plotting {len(best_tower_info)} tower(s) across {len(towers_by_file)} unique run(s)...")

    plotted_count = 0
    failed_towers = []
    for path, (run_number, win_towers) in towers_by_file.items():
        is_combined = not isinstance(run_number, int) or "combined" in str(run_number).lower()

        try:
            with uproot.open(path) as file:
                if hist_name not in file:
                    continue
                hist2d = file[hist_name]

                bad_tower_map = {}
                frac_bad_chi2_map = {}
                calib_map = {}
                if use_cdb and not is_combined:
                    bad_tower_map = get_bad_tower_map(run_number, det="CEMC", dbtag=cdbtag)
                    frac_bad_chi2_map = get_frac_bad_chi2_map(run_number, det="CEMC", dbtag=cdbtag)
                    calib_map = get_calib_adc_to_etower_map(run_number, det="CEMC", dbtag=cdbtag)

                ref_z_score = None
                ref_frac_bad_chi2 = None
                ref_calib = None
                if ref_tower is not None and use_cdb and not is_combined:
                    ref_key = get_calo_tower_key(ref_tower, det="EMCal")
                    ref_z_score = bad_tower_map.get(ref_key, {}).get("sigma") if bad_tower_map else None
                    ref_frac_bad_chi2 = frac_bad_chi2_map.get(ref_key) if frac_bad_chi2_map else None
                    ref_calib = calib_map.get(ref_key) if calib_map else None

                for ieta, iphi, tower_idx, score in win_towers:
                    coords_str = f"_eta{ieta}_phi{iphi}" if include_coords else ""
                    out_filename_tower = f"run_{run_number}_{hist_name}_tower{tower_idx}{coords_str}.png"
                    output_path_tower = output_dir / out_filename_tower

                    tower_key = get_calo_tower_key(tower_idx, det="EMCal")
                    z_score = bad_tower_map.get(tower_key, {}).get("sigma") if bad_tower_map else None
                    frac_bad_chi2 = frac_bad_chi2_map.get(tower_key) if frac_bad_chi2_map else None
                    calib = calib_map.get(tower_key) if calib_map else None

                    ok, reason = make_1d_yproj_plot(
                        hist2d,
                        run_number,
                        output_path_tower,
                        hist_name=hist_name,
                        tower_index=tower_idx,
                        ieta=ieta,
                        iphi=iphi,
                        z_score=z_score,
                        frac_bad_chi2=frac_bad_chi2,
                        calib=calib,
                        logy=True,
                        auto_xlim=True,
                    )

                    if not ok:
                        failed_towers.append((run_number, ieta, iphi, tower_idx, reason))
                        continue

                    plotted_count += 1
                    if ref_tower is not None:
                        out_filename_overlay = f"run_{run_number}_{hist_name}_tower{tower_idx}_ref{ref_tower}{coords_str}.png"
                        output_path_overlay = output_dir / out_filename_overlay
                        make_1d_yproj_plot(
                            hist2d,
                            run_number,
                            output_path_overlay,
                            hist_name=hist_name,
                            tower_index=tower_idx,
                            ieta=ieta,
                            iphi=iphi,
                            z_score=z_score,
                            frac_bad_chi2=frac_bad_chi2,
                            calib=calib,
                            ref_tower_index=ref_tower,
                            ref_z_score=ref_z_score,
                            ref_frac_bad_chi2=ref_frac_bad_chi2,
                            ref_calib=ref_calib,
                            logy=True,
                            auto_xlim=True,
                        )

                if has_threshold:
                    print(f"[{path.name}] Plotted {len(win_towers)} tower(s) for run {run_number} (most counts satisfying {thresh_str}).")
                else:
                    print(f"[{path.name}] Plotted {len(win_towers)} tower(s) for run {run_number} (highest |E| versions).")

        except Exception as e:
            traceback.print_exc()
            print(f"Error plotting from {path}: {e}")

    total_failed = len(unplotted) + len(failed_towers)
    print("\n" + "=" * 80)
    print("Tower Plotting Summary")
    print("=" * 80)
    print(f"Total towers requested: {len(tower_list)}")
    print(f"Successfully plotted:   {plotted_count}")
    print(f"Failed / Skipped:       {total_failed}")
    print(f"Plots saved to:         {output_dir}")

    if total_failed > 0:
        print("-" * 80)
        print(f"Failed / Skipped Towers Details ({total_failed}):")
        for t_idx in unplotted:
            ieta, iphi = towers_dict[t_idx]
            reason = f"No counts satisfying threshold ({thresh_str})" if has_threshold else "Zero counts across all provided runs"
            print(f"  - Tower (ieta={ieta:2d}, iphi={iphi:3d}, idx={t_idx:5d}): {reason}")
        for run_val, ieta, iphi, t_idx, reason in failed_towers:
            print(f"  - Run {run_val}, Tower (ieta={ieta:2d}, iphi={iphi:3d}, idx={t_idx:5d}): {reason}")
    print("=" * 80)


def process_towers_max_energy(
    files,
    tower_list,
    output_dir,
    hist_name="h2EMCalEnergyTowerIndex",
    use_cdb=True,
    cdbtag="newcdbtag",
    ref_tower=None,
    include_coords=False,
):
    """Alias for process_towers_without_runs with no threshold (highest |E| mode)."""
    return process_towers_without_runs(
        files,
        tower_list,
        output_dir,
        hist_name=hist_name,
        lower_threshold=None,
        upper_threshold=None,
        first_match=False,
        use_cdb=use_cdb,
        cdbtag=cdbtag,
        ref_tower=ref_tower,
        include_coords=include_coords,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Plot 1D EMCal tower energy distribution (h2EMCalEnergyTowerIndex) for towers specified in a CSV file."
    )
    parser.add_argument("-c", "--csv", type=Path, required=True, help="Path to CSV file containing (ieta,iphi,run) or (ieta,iphi) format.")
    parser.add_argument("-f", "--file", type=Path, help="Path to a text file containing ROOT file paths (one per line).")
    parser.add_argument("-o", "--output-dir", type=Path, default=Path("."), help="Directory to save the plots (default: current directory).")
    parser.add_argument("--hist-name", default="h2EMCalEnergyTowerIndex", help="Name of the 2D energy vs tower index histogram (default: h2EMCalEnergyTowerIndex).")
    parser.add_argument("--ref-tower", "--ref-tower-index", "--tower-index", type=int, default=None, dest="ref_tower", help="Reference tower index to overlay on 1D tower energy plots.")
    parser.add_argument("--include-coords", action="store_true", help="Include _eta{ieta}_phi{iphi} in the output plot filename.")
    parser.add_argument("--cdbtag", default="newcdbtag", help="CDB global tag to fetch BadTowerMap calibration (default: newcdbtag).")
    parser.add_argument("--no-cdb", action="store_true", help="Disable CDB BadTowerMap query for tower z-scores and badChi2.")
    parser.add_argument("-t", "--threshold", "--energy-threshold", type=float, default=None, dest="threshold", help="Energy threshold [GeV]. Acts as upper threshold (counts <= threshold) by default, or lower threshold if --lower / --threshold-type lower is specified.")
    parser.add_argument("--upper-threshold", "--threshold-upper", "--max-threshold", type=float, default=None, dest="upper_threshold", help="Upper energy threshold [GeV]. Only plots towers if they have counts below this threshold (E <= upper_threshold).")
    parser.add_argument("--lower-threshold", "--threshold-lower", "--min-threshold", type=float, default=None, dest="lower_threshold", help="Lower energy threshold [GeV]. Only plots towers if they have counts above this threshold (E >= lower_threshold).")
    parser.add_argument("--threshold-type", choices=["upper", "lower"], default=None, help="Explicitly specify whether -t/--threshold is 'upper' (counts <= threshold) or 'lower' (counts >= threshold). Default: upper.")
    parser.add_argument("--lower", action="store_true", help="Treat -t/--threshold as a lower threshold (counts with E >= threshold).")
    parser.add_argument("--upper", action="store_true", help="Treat -t/--threshold as an upper threshold (counts with E <= threshold).")
    parser.add_argument("--first-match", action="store_true", help="When using an (ieta,iphi) CSV, plot tower from the first available run meeting criteria instead of scanning for the run with the most counts (if threshold given) or highest absolute energy.")
    parser.add_argument("--workers", type=int, default=None, help="Number of parallel worker processes when run numbers are specified in CSV (default: CPU count or up to 32).")
    parser.add_argument("files", nargs="*", type=Path, help="List of ROOT file paths")
    args = parser.parse_args()

    if not args.csv.exists():
        print(f"Error: CSV file not found: {args.csv}")
        sys.exit(1)

    upper_threshold = args.upper_threshold
    lower_threshold = args.lower_threshold

    if args.threshold is not None:
        if args.lower or (args.threshold_type == "lower"):
            if lower_threshold is None:
                lower_threshold = args.threshold
        else:
            if upper_threshold is None:
                upper_threshold = args.threshold

    if lower_threshold is not None or upper_threshold is not None:
        thresh_desc = format_threshold_str(lower_threshold, upper_threshold)
        print(f"Applying energy threshold filter: {thresh_desc}")

    towers_data, all_runs, has_run = load_tower_csv(args.csv)
    if has_run:
        total_entries = sum(len(v) for v in towers_data.values())
        print(f"Loaded {total_entries} tower entries across {len(all_runs)} unique run(s) from {args.csv}.")
    else:
        total_entries = len(towers_data)
        print(f"Loaded {total_entries} tower entries (no run specified) from {args.csv}.")

    if not towers_data:
        print("Warning: No valid tower entries found in the CSV file.")
        sys.exit(0)

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
    files_to_process = [Path(p) for p in file_list]
    print(f"Found {len(files_to_process)} input ROOT file(s). Starting processing...")

    if not has_run:
        # CSV has only ieta,iphi: process towers across runs (most counts if threshold, or highest |E|)
        process_towers_without_runs(
            files_to_process,
            towers_data,
            output_dir=args.output_dir,
            hist_name=args.hist_name,
            lower_threshold=lower_threshold,
            upper_threshold=upper_threshold,
            first_match=args.first_match,
            use_cdb=not args.no_cdb,
            cdbtag=args.cdbtag,
            ref_tower=args.ref_tower,
            include_coords=args.include_coords,
        )
        print(f"Plots saved to {args.output_dir}")
        return

    # CSV has run numbers specified: run per-file processing
    process_func = functools.partial(
        process_file,
        towers_by_run=towers_data,
        output_dir=args.output_dir,
        hist_name=args.hist_name,
        lower_threshold=lower_threshold,
        upper_threshold=upper_threshold,
        use_cdb=not args.no_cdb,
        cdbtag=args.cdbtag,
        ref_tower=args.ref_tower,
        include_coords=args.include_coords,
    )

    max_workers = args.workers if args.workers is not None else min(os.cpu_count() or 4, 32)
    results = []

    if len(files_to_process) == 1:
        res = process_func(files_to_process[0])
        results.append(res)
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            results = list(tqdm.tqdm(executor.map(process_func, files_to_process), total=len(files_to_process)))

    all_plotted = []
    all_failed = []
    errors = []

    for res in results:
        if isinstance(res, dict):
            run_val = res.get("run", "unknown")
            for t in res.get("plotted", []):
                all_plotted.append((run_val, *t))
            for f in res.get("failed", []):
                all_failed.append((run_val, *f))
            if res.get("error"):
                errors.append(res["error"])
                print(res["error"])
        elif isinstance(res, str):
            errors.append(res)
            print(res)

    total_requested = sum(len(t) for t in towers_data.values()) if isinstance(towers_data, dict) else len(towers_data)

    print("\n" + "=" * 80)
    print("Tower Plotting Summary")
    print("=" * 80)
    print(f"Total towers requested: {total_requested}")
    print(f"Successfully plotted:   {len(all_plotted)}")
    print(f"Failed / Skipped:       {len(all_failed)}")
    if errors:
        print(f"Files with errors:      {len(errors)}")
    print(f"Plots saved to:         {args.output_dir}")

    if all_failed:
        print("-" * 80)
        print(f"Failed / Skipped Towers Details ({len(all_failed)}):")
        for item in all_failed:
            run_val, ieta, iphi, t_idx, reason = item
            print(f"  - Run {run_val}, Tower (ieta={ieta:2d}, iphi={iphi:3d}, idx={t_idx:5d}): {reason}")

    if errors:
        print("-" * 80)
        print(f"File Processing Errors ({len(errors)}):")
        for err in errors:
            print(f"  - {err}")
    print("=" * 80)


if __name__ == "__main__":
    main()
