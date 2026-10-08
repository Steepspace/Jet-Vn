#!/usr/bin/env python3

import uproot
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import mplhep as hep
import os
import tqdm
import concurrent.futures
import argparse
from pathlib import Path
import numpy as np
import re
import pandas as pd
import functools
import pickle
import sys
import psycopg2

def process_file(path, pt_cut=30.0, neg_pt_cut=0.0, plot_run_dir=None, runs_to_plot=None, lumi_dict=None, duration_dict=None, r_jet=0.2):
    path = Path(path)
    if not path.exists():
        return None, None, None, None, None, None, None, None, None, f"File not found: {path}"

    try:
        run_number = int(path.name.split('.')[0])
    except ValueError:
        match = re.search(r'\d+', path.name)
        if match:
            run_number = int(match.group())
        else:
            return None, None, None, None, None, None, None, None, None, f"Could not parse run number from {path.name}"

    try:
        r_tag = f"{int(round(r_jet * 10)):02d}"
        jet_key = f"hJetPt_r{r_tag}_iter"
        jet_v2_key = f"hJetPtv2_r{r_tag}_iter"
        jet_v3_key = f"hJetPtv3_r{r_tag}_iter"

        with uproot.open(path) as file:
            if "hEvent" not in file:
                return run_number, None, None, None, None, None, None, None, None, f"Empty hist: hEvent not found in {path}"
            hist_event = file["hEvent"]
            values_event = hist_event.values()
            if len(values_event) == 0 or values_event[0] <= 0:
                return run_number, None, None, None, None, None, None, None, None, f"Empty hist: Invalid event count (0 events) in {path}"

            n_events = values_event[0]

            if jet_key not in file:
                return run_number, None, None, None, None, None, None, None, None, f"Empty hist: {jet_key} not found in {path}"

            hist_jet = file[jet_key]
            values_jet, edges_jet = hist_jet.to_numpy()
            if len(values_jet) == 0:
                return run_number, None, None, None, None, None, None, None, None, f"Empty hist: {jet_key} is empty in {path}"

            # Integrated counts above pt_cut GeV (assuming bin edges are in GeV)
            mask = edges_jet[:-1] >= pt_cut
            counts_above_cut = np.sum(values_jet[mask])

            neg_jets_counts = None
            values_jet_v2 = None
            if jet_v2_key in file:
                hist_jet_v2 = file[jet_v2_key]
                values_jet_v2 = hist_jet_v2.values()
                mask_neg = edges_jet[:-1] >= neg_pt_cut
                neg_jets_counts = np.sum(values_jet[mask_neg]) - np.sum(values_jet_v2[mask_neg])

            v3_jets_counts = None
            values_jet_v3 = None
            if jet_v3_key in file:
                hist_jet_v3 = file[jet_v3_key]
                values_jet_v3, edges_jet_v3 = hist_jet_v3.to_numpy()
                mask_v3 = edges_jet_v3[:-1] >= pt_cut
                v3_jets_counts = np.sum(values_jet_v3[mask_v3])

                if runs_to_plot is not None and run_number in runs_to_plot and plot_run_dir is not None:
                    try:
                        errors_jet_v3 = hist_jet_v3.errors()
                        hep.style.use("ATLAS")
                        fig, ax = plt.subplots(figsize=(8, 6))
                        centers = (edges_jet_v3[:-1] + edges_jet_v3[1:]) / 2
                        mask_nonzero = values_jet_v3 > 0

                        # Plot with markers and vertical error bars
                        if np.any(mask_nonzero):
                            ax.errorbar(centers[mask_nonzero], values_jet_v3[mask_nonzero],
                                        yerr=errors_jet_v3[mask_nonzero], fmt='o', color='black',
                                        markersize=4, linestyle='none')

                        ax.set_yscale('log')
                        ax.set_xlabel(r"$p_{T}$ (GeV)")
                        ax.set_ylabel("Counts")
                        ax.set_xlim(left=0)

                        lumi_entry = lumi_dict.get(run_number) if lumi_dict else None
                        if lumi_entry is not None:
                            if isinstance(lumi_entry, dict):
                                raw_lumi = lumi_entry.get('lumi_ub', 0.0)
                                run_qa_ev = lumi_entry.get('run_qa_events', 0.0)
                                if run_qa_ev > 0 and n_events <= run_qa_ev:
                                    corr_lumi = raw_lumi * (n_events / run_qa_ev)
                                    lumi_str = rf"{corr_lumi:.2e} $\mu\mathrm{{b}}^{{-1}}$"
                                else:
                                    lumi_str = "N/A"
                            elif isinstance(lumi_entry, (int, float)) and lumi_entry > 0:
                                lumi_str = rf"{lumi_entry:.2e} $\mu\mathrm{{b}}^{{-1}}$"
                            else:
                                lumi_str = "N/A"
                        else:
                            lumi_str = "N/A"

                        dur_val = duration_dict.get(run_number) if duration_dict else None
                        if dur_val is not None and dur_val > 0:
                            dur_str = f"{dur_val:.1f} min"
                        else:
                            dur_str = "N/A"

                        # Run info labels (top right)
                        text_info = (
                            f"Run: {run_number}, Duration: {dur_str}\n"
                            f"Events ($|z| < 10$ cm & MB): {n_events:.2e}\n"
                            f"Luminosity ($|z| < 10$ cm & MB): {lumi_str}\n"
                            f"Threshold ($\\geq$ {pt_cut:g} GeV): {v3_jets_counts:g}"
                        )
                        ax.text(0.95, 0.95, text_info, transform=ax.transAxes, ha='right', va='top', fontsize=15)

                        # Event selection labels (right, below run info)
                        # Text above top right plot border
                        ax.text(1.0, 1.01, rf"$R = {r_jet:g}$", transform=ax.transAxes, ha='right', va='bottom', fontsize=15)

                        # Event selection labels (right, below run info)
                        selection_text = (
                            r"Event Selection:" + "\n"
                            r"$|z| < 10$ cm & MB" + "\n"
                            r"Centrality: 0-60%" + "\n"
                            r"Good Calo-Cent" + "\n"
                            r"No Flow Failure"
                        )
                        ax.text(0.95, 0.66, selection_text, transform=ax.transAxes, ha='right', va='top', fontsize=15)

                        # Jet selection labels (right, below event selection)
                        jet_selection_text = (
                            r"Jet Selection:" + "\n"
                            r"Energy > 0" + "\n"
                            r"$p_{T} > 10$ GeV" + "\n"
                            r"$|\eta| < 1.1 - R$"
                        )
                        ax.text(0.65, 0.66, jet_selection_text, transform=ax.transAxes, ha='right', va='top', fontsize=15)

                        # Vertical line
                        ax.axvline(pt_cut, color='red', linestyle='--')

                        fig.tight_layout()
                        plt.subplots_adjust(left=0.12, bottom=0.13, top=0.95)

                        plot_path = plot_run_dir / f"run_{run_number}_{jet_v3_key}.png"
                        fig.savefig(plot_path, dpi=300)
                        plt.close(fig)
                    except Exception as e:
                        print(f"Failed to plot run {run_number}: {e}")

            return run_number, counts_above_cut, n_events, neg_jets_counts, v3_jets_counts, values_jet, values_jet_v2, values_jet_v3, edges_jet, None
    except Exception as e:
        return run_number, None, None, None, None, None, None, None, None, f"Error processing {path}: {e}"

def plot_normalized_counts(run_numbers, normalized_counts, output_dir, name, ylabel=r"Raw Counts ($p_{T} > 30$ GeV) / Event", suffix="", extra_text=None, ylim_top=None, z_values=None, z_label=None, save_pdf=False, r_jet=None):
    hep.style.use("ATLAS")

    fig, ax = plt.subplots(figsize=(10, 6))
    if z_values is not None and len(z_values) == len(run_numbers):
        sc = ax.scatter(run_numbers, normalized_counts, c=z_values, cmap='viridis', s=16, label='Data', zorder=3)
        cbar = fig.colorbar(sc, ax=ax, pad=0.02)
        if z_label:
            cbar.set_label(z_label)
    else:
        ax.plot(run_numbers, normalized_counts, marker='o', markersize=4, linestyle='none', color='black', label='Data')

    # Add a light dashed line for the average counts
    avg_counts = sum(normalized_counts) / len(normalized_counts)
    ax.axhline(avg_counts, color='gray', linestyle='--', alpha=0.5, label='Average')
    ax.legend(loc='upper center', frameon=False)

    ax.set_xlabel("Run Number", labelpad=18)
    ax.set_ylabel(ylabel)
    ax.set_title("Jets")
    if ylim_top is not None:
        ax.set_ylim(bottom=0, top=ylim_top)
    else:
        ax.set_ylim(bottom=0)

    if r_jet is not None:
        ax.text(1.0, 1.01, rf"$R = {r_jet:g}$", transform=ax.transAxes, ha='right', va='bottom', fontsize=15)

    # Calculate and display the number of runs and event selections
    total_runs = len(run_numbers)
    text_info = (
        f"Runs = {total_runs}\n"
        r"$|z| < 10$ cm & MB" + "\n"
        r"Centrality: 0-60%"
    )
    if extra_text:
        if isinstance(extra_text, (list, tuple)):
            text_info += "\n" + "\n".join(extra_text)
        else:
            text_info += f"\n{extra_text}"
    ax.text(0.05, 0.95, text_info, transform=ax.transAxes, ha='left', va='top', fontsize=18)

    plt.tight_layout()
    plt.subplots_adjust(top=0.94, bottom=0.14)

    r_tag = f"{int(round(r_jet * 10)):02d}" if r_jet is not None else ""
    image_dir = output_dir / "images" / f"r{r_tag}" if r_tag else output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    png_path = image_dir / f"{name}{suffix}.png"
    plt.savefig(png_path, dpi=300)

    if save_pdf:
        pdf_dir = output_dir / "pdf" / f"r{r_tag}" if r_tag else output_dir / "pdf"
        pdf_dir.mkdir(parents=True, exist_ok=True)
        pdf_path = pdf_dir / f"{name}{suffix}.pdf"
        plt.savefig(pdf_path)
        print(f"Saved normalized plots as {pdf_path} and {png_path}")
    else:
        print(f"Saved normalized plot as {png_path}")

    plt.close(fig)

def plot_raw_counts(run_numbers, raw_counts, output_dir, name, ylabel=r"Raw Counts ($p_{T} > 30$ GeV)", suffix="-raw", extra_text=None, z_values=None, z_label=None, save_pdf=False, r_jet=None):
    hep.style.use("ATLAS")

    fig, ax = plt.subplots(figsize=(10, 6))
    if z_values is not None and len(z_values) == len(run_numbers):
        sc = ax.scatter(run_numbers, raw_counts, c=z_values, cmap='viridis', s=16, label='Data', zorder=3)
        cbar = fig.colorbar(sc, ax=ax, pad=0.02)
        if z_label:
            cbar.set_label(z_label)
    else:
        ax.plot(run_numbers, raw_counts, marker='o', markersize=4, linestyle='none', color='black', label='Data')

    avg_raw_counts = sum(raw_counts) / len(raw_counts)
    ax.axhline(avg_raw_counts, color='gray', linestyle='--', alpha=0.5, label='Average')
    ax.legend(loc='upper right', frameon=False)

    ax.set_xlabel("Run Number", labelpad=18)
    ax.set_ylabel(ylabel)
    ax.set_title("Jets")
    ax.set_ylim(bottom=0)

    if r_jet is not None:
        ax.text(1.0, 1.01, rf"$R = {r_jet:g}$", transform=ax.transAxes, ha='right', va='bottom', fontsize=15)

    total_runs = len(run_numbers)
    text_info = (
        f"Runs = {total_runs}\n"
        r"$|z| < 10$ cm & MB" + "\n"
        r"Centrality: 0-60%"
    )
    if extra_text:
        if isinstance(extra_text, (list, tuple)):
            text_info += "\n" + "\n".join(extra_text)
        else:
            text_info += f"\n{extra_text}"
    ax.text(0.05, 0.95, text_info, transform=ax.transAxes, ha='left', va='top', fontsize=18)

    plt.tight_layout()
    plt.subplots_adjust(top=0.94, bottom=0.14)

    r_tag = f"{int(round(r_jet * 10)):02d}" if r_jet is not None else ""
    image_dir = output_dir / "images" / f"r{r_tag}" if r_tag else output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    png_path = image_dir / f"{name}{suffix}.png"
    plt.savefig(png_path, dpi=300)

    if save_pdf:
        pdf_dir = output_dir / "pdf" / f"r{r_tag}" if r_tag else output_dir / "pdf"
        pdf_dir.mkdir(parents=True, exist_ok=True)
        pdf_path = pdf_dir / f"{name}{suffix}.pdf"
        plt.savefig(pdf_path)
        print(f"Saved raw plots as {pdf_path} and {png_path}")
    else:
        print(f"Saved raw plot as {png_path}")

    plt.close(fig)

def plot_scatter_xy(x_vals, y_vals, output_dir, name, xlabel, ylabel, suffix="", save_pdf=False, r_jet=None):
    hep.style.use("ATLAS")

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(x_vals, y_vals, color='black', s=16, label='Data')

    ax.set_xlabel(xlabel, labelpad=-1)
    ax.set_ylabel(ylabel)
    ax.set_title("Runs")
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)

    total_runs = len(x_vals)
    text_info = (
        f"Runs = {total_runs}\n"
        r"$|z| < 10$ cm & MB"
    )
    ax.text(0.05, 0.95, text_info, transform=ax.transAxes, ha='left', va='top', fontsize=18)

    plt.tight_layout()
    plt.subplots_adjust(top=0.94, bottom=0.1)

    r_tag = f"{int(round(r_jet * 10)):02d}" if r_jet is not None else ""
    image_dir = output_dir / "images" / f"r{r_tag}" if r_tag else output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    png_path = image_dir / f"{name}{suffix}.png"
    plt.savefig(png_path, dpi=300)

    if save_pdf:
        pdf_dir = output_dir / "pdf" / f"r{r_tag}" if r_tag else output_dir / "pdf"
        pdf_dir.mkdir(parents=True, exist_ok=True)
        pdf_path = pdf_dir / f"{name}{suffix}.pdf"
        plt.savefig(pdf_path)
        print(f"Saved scatter plot as {pdf_path} and {png_path}")
    else:
        print(f"Saved scatter plot as {png_path}")

    plt.close(fig)

def plot_jet_pt_overlay(edges, sum_jet, sum_jet_v2, sum_jet_v3, total_runs, output_dir, name, r_jet=0.2, save_pdf=False):
    hep.style.use("ATLAS")
    r_tag = f"{int(round(r_jet * 10)):02d}" if r_jet is not None else ""
    image_dir = output_dir / "images" / f"r{r_tag}" if r_tag else output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    if save_pdf:
        pdf_dir = output_dir / "pdf" / f"r{r_tag}" if r_tag else output_dir / "pdf"
        pdf_dir.mkdir(parents=True, exist_ok=True)

    mask_past_100 = edges[:-1] >= 100
    fills_past_100 = (
        np.any(sum_jet[mask_past_100] > 0) or
        (sum_jet_v2 is not None and np.any(sum_jet_v2[mask_past_100] > 0)) or
        (sum_jet_v3 is not None and np.any(sum_jet_v3[mask_past_100] > 0))
    )

    nonzero_indices = []
    if np.any(sum_jet > 0):
        nonzero_indices.append(np.where(sum_jet > 0)[0][-1])
    if sum_jet_v2 is not None and np.any(sum_jet_v2 > 0):
        nonzero_indices.append(np.where(sum_jet_v2 > 0)[0][-1])
    if sum_jet_v3 is not None and np.any(sum_jet_v3 > 0):
        nonzero_indices.append(np.where(sum_jet_v3 > 0)[0][-1])

    max_x = edges[max(nonzero_indices) + 1] if nonzero_indices else edges[-1]
    max_x = min(edges[-1], max(100.0, float(np.ceil(max_x / 10.0) * 10)))

    configs = [{"suffix": "-pt-overlay", "xlim_right": max_x}]
    if fills_past_100:
        configs.append({"suffix": "-pt-overlay-100gev", "xlim_right": 100.0})

    for cfg in configs:
        suffix = cfg["suffix"]
        xlim_right = cfg["xlim_right"]

        fig, ax = plt.subplots(figsize=(10, 8))
        hep.histplot(sum_jet, bins=edges, ax=ax, histtype='step', color='red', label='All', linewidth=1.8)
        if sum_jet_v2 is not None:
            hep.histplot(sum_jet_v2, bins=edges, ax=ax, histtype='step', color='blue', label='Energy > 0 GeV', linewidth=1.8)
        if sum_jet_v3 is not None:
            hep.histplot(sum_jet_v3, bins=edges, ax=ax, histtype='step', color='green', label=r"$|\mathrm{calo}\ v_{2}| < 0.48$", linewidth=1.8)

        ax.set_yscale('log')
        ax.set_xlabel(r"$p_{T}$ (GeV)", labelpad=0, fontsize=22)
        ax.set_ylabel("Counts", fontsize=22)
        ax.set_xlim(left=0, right=xlim_right)

        ax.yaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=20))
        ax.yaxis.set_major_formatter(ticker.LogFormatterMathtext())
        ax.yaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs=range(2, 10), numticks=200))
        ax.yaxis.set_minor_formatter(ticker.NullFormatter())

        ax.tick_params(axis='both', which='major', labelsize=18, length=8, width=1.5)
        ax.tick_params(axis='both', which='minor', length=4, width=1.0)

        mask_range = (edges[:-1] >= 0) & (edges[:-1] < xlim_right)
        visible_counts = []
        if np.any(sum_jet[mask_range] > 0):
            visible_counts.append(sum_jet[mask_range][sum_jet[mask_range] > 0])
        if sum_jet_v2 is not None and np.any(sum_jet_v2[mask_range] > 0):
            visible_counts.append(sum_jet_v2[mask_range][sum_jet_v2[mask_range] > 0])
        if sum_jet_v3 is not None and np.any(sum_jet_v3[mask_range] > 0):
            visible_counts.append(sum_jet_v3[mask_range][sum_jet_v3[mask_range] > 0])

        if visible_counts:
            y_max = np.max(np.concatenate(visible_counts))
            if y_max > 0:
                n = int(np.floor(np.log10(y_max)))
                candidate_ticks = [k * (10.0 ** p) for p in (n, n + 1) for k in range(2, 10)]
                top_limit = next(t for t in candidate_ticks if t > y_max)
            else:
                top_limit = 10.0
            ax.set_ylim(bottom=0.5, top=top_limit)
        else:
            ax.set_ylim(bottom=0.5)

        ax.legend(loc='upper right', frameon=False, fontsize=25)
        if r_jet is not None:
            ax.text(1.0, 1.01, rf"$R = {r_jet:g}$", transform=ax.transAxes, ha='right', va='bottom', fontsize=20)

        text_info = (
            f"Runs = {total_runs}\n"
            r"Event Selection:" + "\n"
            r"$|z| < 10$ cm & MB" + "\n"
            r"Centrality: 0-60%" + "\n"
            r"Good Calo-Cent" + "\n"
            r"No Flow Failure"
        )
        ax.text(0.95, 0.72, text_info, transform=ax.transAxes, ha='right', va='top', fontsize=25)

        plt.tight_layout()
        plt.subplots_adjust(top=0.94, bottom=0.08)

        png_path = image_dir / f"{name}{suffix}.png"
        plt.savefig(png_path, dpi=300)
        if save_pdf:
            pdf_path = pdf_dir / f"{name}{suffix}.pdf"
            plt.savefig(pdf_path)
            print(f"Saved 1D overlay plot as {pdf_path} and {png_path}")
        else:
            print(f"Saved 1D overlay plot as {png_path}")
        plt.close(fig)

def process_lumi_file(path):
    path = Path(path)
    if not path.exists():
        return None, None, None, f"File not found: {path}"
    try:
        try:
            run_number = int(path.name.split('.')[0])
        except ValueError:
            match = re.search(r'\d+', path.name)
            if match:
                run_number = int(match.group())
            else:
                return None, None, None, f"Could not parse run number from {path.name}"

        with uproot.open(path) as f:
            if "hLuminosity" not in f or "hEvent" not in f:
                return run_number, None, None, f"Missing hLuminosity or hEvent in {path}"
            hlumi = f["hLuminosity"]
            hevt = f["hEvent"]
            lumi_vals = hlumi.values()
            evt_vals = hevt.values()

            if len(lumi_vals) < 1 or len(evt_vals) < 6:
                return run_number, None, None, f"Incomplete histogram bins in {path}"

            raw_lumi = float(lumi_vals[0])  # bin 1 in nb^-1
            run_qa_events = float(evt_vals[5])  # bin 6: |z| < 10 cm & MB

            # Scale raw_lumi from nb^-1 to ub^-1 (* 1000)
            lumi_ub_inv = raw_lumi * 1000.0

            return run_number, lumi_ub_inv, run_qa_events, None
    except Exception as e:
        return None, None, None, f"Error processing {path}: {e}"

def load_lumi_from_list(list_path, cache_path=None, no_cache=False):
    list_path = Path(list_path)
    if not list_path.exists():
        print(f"Warning: Lumi list file not found at {list_path}")
        return {}

    lumi_files = []
    try:
        with list_path.open('r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    lumi_files.append(Path(line))
    except Exception as e:
        print(f"Error reading lumi list {list_path}: {e}")
        return {}

    if not lumi_files:
        print(f"Warning: No files found in lumi list {list_path}")
        return {}

    cache = {}
    if not no_cache and cache_path and cache_path.exists():
        try:
            with cache_path.open("rb") as f:
                cache = pickle.load(f)
        except Exception as e:
            print(f"Warning: Could not read lumi cache {cache_path}: {e}")
            cache = {}

    lumi_dict = {}
    files_to_process = []

    for p in lumi_files:
        p_str = str(p.resolve())
        mtime = p.stat().st_mtime if p.exists() else 0
        if not no_cache and p_str in cache and cache[p_str].get("mtime") == mtime:
            entry = cache[p_str]
            run = entry["run"]
            lumi = entry["lumi_ub"]
            ev = entry["run_qa_events"]
            if run is not None and lumi is not None and ev is not None:
                lumi_dict[run] = {"lumi_ub": lumi, "run_qa_events": ev}
            continue
        files_to_process.append(p)

    if files_to_process:
        print(f"Reading luminosity from {len(files_to_process)} Event QA files ({len(lumi_dict)} loaded from cache)...")
        max_workers = min(os.cpu_count() or 4, 32)
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            new_results = list(tqdm.tqdm(executor.map(process_lumi_file, files_to_process), total=len(files_to_process)))

        for p, res in zip(files_to_process, new_results):
            run, lumi_ub, ev, err = res
            p_str = str(p.resolve())
            mtime = p.stat().st_mtime if p.exists() else 0
            cache[p_str] = {
                "mtime": mtime,
                "run": run,
                "lumi_ub": lumi_ub,
                "run_qa_events": ev,
                "err": err
            }
            if run is not None and lumi_ub is not None and ev is not None:
                lumi_dict[run] = {"lumi_ub": lumi_ub, "run_qa_events": ev}

        if not no_cache and cache_path:
            try:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                with cache_path.open("wb") as f:
                    pickle.dump(cache, f)
            except Exception as e:
                print(f"Warning: Could not write lumi cache {cache_path}: {e}")
    else:
        print(f"Loaded all {len(lumi_dict)} luminosity records from cache.")

    return lumi_dict

def main():
    parser = argparse.ArgumentParser(description="Plot integrated jet counts above pT cut per run from a list of ROOT files.")
    parser.add_argument("-f", "--file", type=Path, help="Path to a text file containing ROOT file paths (one per line).")
    parser.add_argument("-o", "--output-dir", type=Path, default=Path("."), help="Directory to save the plots (default: current directory).")
    parser.add_argument("-n", "--name", "--plot-name", dest="name", type=str, default=None, help="Base filename for output plots (default: jet_qa_counts_per_run_r<R>).")
    parser.add_argument("-p", "--pt-cut", type=float, default=30.0, help="pT threshold in GeV for integrated jet counts (default: 30.0).")
    parser.add_argument("-np", "--neg-pt-cut", type=float, default=0.0, help="pT threshold in GeV for negative energy jet counts (default: 0.0).")
    parser.add_argument("--cache-file", type=Path, default=None, help="Path to cache file for processed ROOT file data (default: <output_dir>/.jet_qa_cache_r<R>_pt<pt>_negpt<neg_pt>.pkl).")
    parser.add_argument("--no-cache", action="store_true", help="Disable caching and force re-processing of all ROOT files.")
    parser.add_argument("-l", "--lumi-list", type=Path, required=True, help="Path to text file containing Event QA ROOT file paths (one per line) for luminosity extraction.")
    parser.add_argument("--plot-runs", type=Path, default=None, help="Path to a text file containing run numbers to plot individually (one per line).")
    parser.add_argument("-r", "--r-jet", "--r", dest="r_jet", type=float, default=0.2, help="Jet resolution parameter R (default: 0.2).")
    parser.add_argument("--save-pdf", "--pdf", dest="save_pdf", action="store_true", help="Enable saving output plots in PDF format in addition to PNG.")
    parser.add_argument("files", nargs="*", type=Path, help="List of ROOT file paths")
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
        print("Error: You must provide at least one ROOT file or a text file containing ROOT file paths.")
        parser.print_help()
        sys.exit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    lumi_cache_path = args.output_dir / f".lumi_cache_{args.lumi_list.stem}.pkl"
    lumi_dict = load_lumi_from_list(args.lumi_list, cache_path=lumi_cache_path, no_cache=args.no_cache)

    run_numbers = []
    lumi_list = []
    duration_list = []
    normalized_counts = []
    lumi_normalized_counts = []
    events_list = []
    raw_counts = []
    neg_raw_counts = []
    neg_normalized_counts = []
    neg_lumi_normalized_counts = []
    v3_raw_counts = []
    v3_normalized_counts = []
    v3_lumi_normalized_counts = []

    runs_to_plot = set()
    if args.plot_runs and args.plot_runs.exists():
        try:
            with args.plot_runs.open('r') as f:
                for line in f:
                    line = line.strip()
                    if line.isdigit():
                        runs_to_plot.add(int(line))
            print(f"Loaded {len(runs_to_plot)} run numbers to plot from {args.plot_runs}")
        except Exception as e:
            print(f"Error reading plot runs file {args.plot_runs}: {e}")
            sys.exit(1)

    plot_run_dir = None
    if runs_to_plot:
        plot_run_dir = args.output_dir / "run_qa_plots"
        plot_run_dir.mkdir(parents=True, exist_ok=True)

    print(f"Found {len(file_list)} input files.")

    duration_dict = {}
    try:
        conn = psycopg2.connect(host="sphnxdaqdbreplica", dbname="daq")
        cur = conn.cursor()

        run_nums_for_query = []
        for path in file_list:
            try:
                run_num_guess = int(path.name.split('.')[0])
            except ValueError:
                match = re.search(r'\d+', path.name)
                run_num_guess = int(match.group()) if match else None
            if run_num_guess is not None:
                run_nums_for_query.append(run_num_guess)

        if run_nums_for_query:
            query = "SELECT runnumber, EXTRACT(EPOCH FROM (ertimestamp - brtimestamp)) / 60.0 FROM run WHERE runnumber = ANY(%s);"
            cur.execute(query, (run_nums_for_query,))
            for r, d in cur.fetchall():
                if d is not None:
                    duration_dict[r] = float(d)
        cur.close()
        conn.close()
    except Exception as e:
        print(f"Warning: Could not fetch run durations from database: {e}")

    cache = {}
    r_tag = f"{int(round(args.r_jet * 10)):02d}"
    pt_tag = f"{args.pt_cut:g}".replace('.', 'p')
    neg_pt_tag = f"{args.neg_pt_cut:g}".replace('.', 'p')
    default_cache_name = f".jet_qa_cache_r{r_tag}_pt{pt_tag}_negpt{neg_pt_tag}.pkl"
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

    for path in file_list:
        path = Path(path)
        resolved_str = str(path.resolve())
        mtime = path.stat().st_mtime if path.exists() else 0

        force_process = False
        if runs_to_plot:
            try:
                run_num_guess = int(path.name.split('.')[0])
            except ValueError:
                match = re.search(r'\d+', path.name)
                run_num_guess = int(match.group()) if match else None
            if run_num_guess in runs_to_plot:
                force_process = True

        if not args.no_cache and resolved_str in cache and not force_process:
            entry = cache[resolved_str]
            res = entry.get("result")
            if (entry.get("mtime") == mtime and
                entry.get("pt_cut") == args.pt_cut and
                entry.get("neg_pt_cut") == args.neg_pt_cut and
                entry.get("r_jet") == args.r_jet and
                isinstance(res, (tuple, list)) and len(res) == 10):
                results.append(entry["result"])
                continue

        files_to_process.append(path)

    if files_to_process:
        print(f"Processing {len(files_to_process)} files ({len(results)} loaded from cache)...")
        process_func = functools.partial(process_file, pt_cut=args.pt_cut, neg_pt_cut=args.neg_pt_cut, plot_run_dir=plot_run_dir, runs_to_plot=runs_to_plot, lumi_dict=lumi_dict, duration_dict=duration_dict, r_jet=args.r_jet)
        max_workers = min(os.cpu_count() or 4, 32)
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            new_results = list(tqdm.tqdm(executor.map(process_func, files_to_process), total=len(files_to_process)))

        for path, res in zip(files_to_process, new_results):
            results.append(res)
            resolved_str = str(path.resolve())
            mtime = path.stat().st_mtime if path.exists() else 0
            cache[resolved_str] = {
                "mtime": mtime,
                "pt_cut": args.pt_cut,
                "neg_pt_cut": args.neg_pt_cut,
                "r_jet": args.r_jet,
                "result": res
            }

        if not args.no_cache:
            try:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                with cache_path.open("wb") as f:
                    pickle.dump(cache, f)
                print(f"Saved updated cache to {cache_path}.")
            except Exception as e:
                print(f"Warning: Could not write cache file {cache_path}: {e}")
    else:
        print(f"All {len(results)} files loaded from cache.")

    skipped_no_lumi = []
    skipped_empty_hists = []
    skipped_jet_events_exceed_run_qa = []
    other_errors = []
    ratio_list = []
    run_qa_events_list = []

    jet_hists_to_sum = []
    jet_v2_hists_to_sum = []
    jet_v3_hists_to_sum = []
    common_edges = None

    for run_num, counts, n_events, neg_jets, v3_jets, v_jet, v_jet_v2, v_jet_v3, edges_jet, err in results:
        if err:
            # If run_num was None (e.g. from existing cache), try to extract from err
            if run_num is None:
                match = re.search(r'(\d+)\.root', str(err))
                if match:
                    run_num = int(match.group(1))

            err_lower = str(err).lower()
            if any(k in err_lower for k in ["empty", "invalid event count", "zero event", "not found", "missing"]):
                skipped_empty_hists.append(run_num if run_num is not None else err)
            else:
                other_errors.append(f"Run {run_num}: {err}" if run_num is not None else str(err))
            continue

        if run_num is not None:
            lumi_info = lumi_dict.get(run_num)
            if not lumi_info or lumi_info.get('lumi_ub', 0) <= 0 or lumi_info.get('run_qa_events', 0) <= 0:
                skipped_no_lumi.append(run_num)
                continue

            raw_lumi = lumi_info['lumi_ub']
            run_qa_ev = lumi_info['run_qa_events']

            if n_events > run_qa_ev:
                skipped_jet_events_exceed_run_qa.append((run_num, n_events, run_qa_ev))
                continue

            corr_factor = n_events / run_qa_ev
            lumi = raw_lumi * corr_factor

            norm = counts / n_events
            lumi_norm = counts / lumi

            run_numbers.append(run_num)
            lumi_list.append(lumi)
            duration_list.append(duration_dict.get(run_num, 0.0))
            normalized_counts.append(norm)
            lumi_normalized_counts.append(lumi_norm)
            events_list.append(n_events)
            raw_counts.append(counts)
            ratio_list.append(corr_factor)
            run_qa_events_list.append(run_qa_ev)

            if common_edges is None and edges_jet is not None:
                common_edges = edges_jet

            if v_jet is not None:
                jet_hists_to_sum.append(v_jet)
            if v_jet_v2 is not None:
                jet_v2_hists_to_sum.append(v_jet_v2)
            if v_jet_v3 is not None:
                jet_v3_hists_to_sum.append(v_jet_v3)

            if neg_jets is not None:
                neg_raw_counts.append(neg_jets)
                neg_normalized_counts.append(neg_jets / n_events)
                neg_lumi_normalized_counts.append(neg_jets / lumi)
            else:
                neg_raw_counts.append(0)
                neg_normalized_counts.append(0)
                neg_lumi_normalized_counts.append(0)
            if v3_jets is not None:
                v3_raw_counts.append(v3_jets)
                v3_normalized_counts.append(v3_jets / n_events)
                v3_lumi_normalized_counts.append(v3_jets / lumi)
            else:
                v3_raw_counts.append(0)
                v3_normalized_counts.append(0)
                v3_lumi_normalized_counts.append(0)

    if skipped_empty_hists:
        int_runs = sorted([r for r in skipped_empty_hists if isinstance(r, int)])
        other_skipped = [r for r in skipped_empty_hists if not isinstance(r, int)]
        print(f"\nSkipped {len(skipped_empty_hists)} runs because of empty histograms / zero events:")
        if int_runs:
            print(", ".join(map(str, int_runs)))
        if other_skipped:
            print(", ".join(map(str, other_skipped)))
        print("")

    if skipped_no_lumi:
        print(f"\nSkipped {len(skipped_no_lumi)} runs because they had no valid luminosity in the Event QA list:")
        print(", ".join(map(str, sorted(skipped_no_lumi))))
        print("")

    if skipped_jet_events_exceed_run_qa:
        print(f"\nSkipped {len(skipped_jet_events_exceed_run_qa)} runs where Jet QA events exceeded Run QA events:")
        for r, jet_ev, run_ev in sorted(skipped_jet_events_exceed_run_qa, key=lambda x: x[0]):
            diff = jet_ev - run_ev
            ratio = jet_ev / run_ev
            print(f"  Run {r}: Jet QA events = {jet_ev:g}, Run QA events = {run_ev:g} (excess = +{diff:g}, ratio = {ratio:.4f})")
        print("")

    if other_errors:
        print(f"\nEncountered {len(other_errors)} errors while processing files:")
        for e in other_errors:
            print(f"  {e}")
        print("")

    if not run_numbers:
        print("No valid data found to plot.")
        return

    # Sort by run number to ensure the plot is ordered
    sorted_pairs = sorted(zip(run_numbers, lumi_list, duration_list, normalized_counts, lumi_normalized_counts, events_list, raw_counts, neg_raw_counts, neg_normalized_counts, neg_lumi_normalized_counts, v3_raw_counts, v3_normalized_counts, v3_lumi_normalized_counts, ratio_list, run_qa_events_list))
    run_numbers, lumi_list, duration_list, normalized_counts, lumi_normalized_counts, events_list, raw_counts, neg_raw_counts, neg_normalized_counts, neg_lumi_normalized_counts, v3_raw_counts, v3_normalized_counts, v3_lumi_normalized_counts, ratio_list, run_qa_events_list = zip(*sorted_pairs)

    plot_name = args.name if args.name is not None else f"jet_qa_counts_per_run_r{r_tag}"

    # Save all processed runs to CSV sorted by V3RawCounts descending
    counts_data = [
        {
            'Run': r, 'R': args.r_jet, 'Duration': round(dur, 1),
            'Events': ev, 'RunQAEvents': run_ev, 'LumiCorrection': round(corr, 6),
            'Lumi_ub_inv': round(lumi_c_val, 6),
            'RawCounts': raw, 'NormalizedCounts': c, 'LumiNormalizedCounts': lumi_c,
            'NegRawCounts': neg_raw, 'NegNormalizedCounts': neg_norm, 'NegLumiNormalizedCounts': neg_lumi_norm,
            'V3RawCounts': v3_raw, 'V3NormalizedCounts': v3_norm, 'V3LumiNormalizedCounts': v3_lumi_norm
        }
        for r, lumi_c_val, dur, c, lumi_c, ev, raw, neg_raw, neg_norm, neg_lumi_norm, v3_raw, v3_norm, v3_lumi_norm, corr, run_ev in zip(
            run_numbers, lumi_list, duration_list, normalized_counts, lumi_normalized_counts, events_list, raw_counts,
            neg_raw_counts, neg_normalized_counts, neg_lumi_normalized_counts, v3_raw_counts, v3_normalized_counts,
            v3_lumi_normalized_counts, ratio_list, run_qa_events_list
        )
    ]

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if counts_data:
        counts_df = pd.DataFrame(counts_data)
        counts_df = counts_df.sort_values(by='V3RawCounts', ascending=False)
        csv_path = args.output_dir / f"{plot_name}.csv"
        counts_df.to_csv(csv_path, index=False)
        print(f"Saved {len(counts_df)} runs to {csv_path}")
    else:
        print("No valid runs found to write to CSV.")

    pt_str = f"{args.pt_cut:g}"
    neg_pt_str = f"{args.neg_pt_cut:g}"
    v3_extra_text = ["Jet Energy > 0 GeV", r"$|\mathrm{calo}\ v_{2}| < 0.48$"]

    configs = [
        {"z_vals": lumi_list, "z_lab": r"Luminosity ($\mu\mathrm{b}^{-1}$)", "suffix_append": ""},
        {"z_vals": duration_list, "z_lab": "Run Duration (min)", "suffix_append": "-duration"}
    ]

    for cfg in configs:
        z = cfg["z_vals"]
        zl = cfg["z_lab"]
        sa = cfg["suffix_append"]

        # We only want to plot if we have valid z-values (e.g. at least one non-zero duration if we're on the duration loop)
        if sa == "-duration" and not any(z):
            continue

        plot_normalized_counts(
            run_numbers, normalized_counts, args.output_dir, plot_name,
            ylabel=rf"Raw Counts ($p_{{T}} > {pt_str}$ GeV) / Event", suffix=f"-norm-events{sa}", z_values=z, z_label=zl, save_pdf=args.save_pdf, r_jet=args.r_jet
        )
        plot_normalized_counts(
            run_numbers, lumi_normalized_counts, args.output_dir, plot_name,
            ylabel=rf"Raw Counts ($p_{{T}} > {pt_str}$ GeV) / $\mu\mathrm{{b}}^{{-1}}$", suffix=f"-norm-lumi{sa}", z_values=z, z_label=zl, save_pdf=args.save_pdf, r_jet=args.r_jet
        )
        plot_raw_counts(
            run_numbers, raw_counts, args.output_dir, plot_name,
            ylabel=rf"Raw Counts ($p_{{T}} > {pt_str}$ GeV)", suffix=f"-raw{sa}", z_values=z, z_label=zl, save_pdf=args.save_pdf, r_jet=args.r_jet
        )

        plot_normalized_counts(
            run_numbers, v3_normalized_counts, args.output_dir, plot_name,
            ylabel=rf"Raw Counts ($p_{{T}} > {pt_str}$ GeV) / Event", suffix=f"-norm-events-v3-filters{sa}",
            extra_text=v3_extra_text, z_values=z, z_label=zl, save_pdf=args.save_pdf, r_jet=args.r_jet
        )
        plot_normalized_counts(
            run_numbers, v3_lumi_normalized_counts, args.output_dir, plot_name,
            ylabel=rf"Raw Counts ($p_{{T}} > {pt_str}$ GeV) / $\mu\mathrm{{b}}^{{-1}}$", suffix=f"-norm-lumi-v3-filters{sa}",
            extra_text=v3_extra_text, z_values=z, z_label=zl, save_pdf=args.save_pdf, r_jet=args.r_jet
        )
        plot_raw_counts(
            run_numbers, v3_raw_counts, args.output_dir, plot_name,
            ylabel=rf"Raw Counts ($p_{{T}} > {pt_str}$ GeV)", suffix=f"-v3-filters-raw{sa}",
            extra_text=v3_extra_text, z_values=z, z_label=zl, save_pdf=args.save_pdf, r_jet=args.r_jet
        )

        plot_normalized_counts(
            run_numbers, neg_normalized_counts, args.output_dir, plot_name,
            ylabel=rf"Negative Energy Jets ($p_{{T}} > {neg_pt_str}$ GeV) / Event", suffix=f"-norm-events-neg{sa}", z_values=z, z_label=zl, save_pdf=args.save_pdf, r_jet=args.r_jet
        )
        plot_normalized_counts(
            run_numbers, neg_lumi_normalized_counts, args.output_dir, plot_name,
            ylabel=rf"Negative Energy Jets ($p_{{T}} > {neg_pt_str}$ GeV) / $\mu\mathrm{{b}}^{{-1}}$", suffix=f"-norm-lumi-neg{sa}", z_values=z, z_label=zl, save_pdf=args.save_pdf, r_jet=args.r_jet
        )
        plot_raw_counts(
            run_numbers, neg_raw_counts, args.output_dir, plot_name,
            ylabel=rf"Negative Energy Jets ($p_{{T}} > {neg_pt_str}$ GeV)", suffix=f"-neg-raw{sa}", z_values=z, z_label=zl, save_pdf=args.save_pdf, r_jet=args.r_jet
        )

    # Scatter plots vs Luminosity
    plot_scatter_xy(
        lumi_list, events_list, args.output_dir, plot_name,
        xlabel=r"Luminosity ($\mu\mathrm{b}^{-1}$)",
        ylabel=r"Events ($|z| < 10$ cm & MB)",
        suffix="-events-vs-lumi",
        save_pdf=args.save_pdf,
        r_jet=args.r_jet
    )

    if any(duration_list):
        plot_scatter_xy(
            lumi_list, duration_list, args.output_dir, plot_name,
            xlabel=r"Luminosity ($\mu\mathrm{b}^{-1}$)",
            ylabel="Run Duration (min)",
            suffix="-duration-vs-lumi",
            save_pdf=args.save_pdf,
            r_jet=args.r_jet
        )

    # 1D overlay of aggregated jet pT
    if jet_hists_to_sum and common_edges is not None:
        sum_jet = np.sum(jet_hists_to_sum, axis=0)
        sum_jet_v2 = np.sum(jet_v2_hists_to_sum, axis=0) if jet_v2_hists_to_sum else None
        sum_jet_v3 = np.sum(jet_v3_hists_to_sum, axis=0) if jet_v3_hists_to_sum else None
        plot_jet_pt_overlay(
            common_edges, sum_jet, sum_jet_v2, sum_jet_v3,
            len(run_numbers), args.output_dir, plot_name,
            r_jet=args.r_jet, save_pdf=args.save_pdf
        )

    print("\nSummary:")
    print(f"  Total input files: {len(file_list)}")
    print(f"  Successfully processed and plotted: {len(run_numbers)} runs")
    if skipped_empty_hists:
        print(f"  Skipped (empty histograms / zero events): {len(skipped_empty_hists)} runs")
    if skipped_no_lumi:
        print(f"  Skipped (no valid luminosity in Event QA): {len(skipped_no_lumi)} runs")
    if skipped_jet_events_exceed_run_qa:
        print(f"  Skipped (Jet QA events > Run QA events): {len(skipped_jet_events_exceed_run_qa)} runs")
    if other_errors:
        print(f"  Encountered errors: {len(other_errors)} files")

if __name__ == "__main__":
    main()
