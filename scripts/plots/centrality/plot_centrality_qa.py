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

def make_1d_plot(hist_or_tuple, run_number, output_path, xlabel=None, ylabel=None, extra_labels=None, logy=False, xlim=None):
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

    has_positive = np.any(values > 0)

    if logy and has_positive:
        hep.histplot((values, edges), ax=ax, histtype='step', color='blue', linewidth=3)
        ax.set_yscale('log')
        ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=20))
        ax.set_ylim(bottom=0.5, top=np.max(values) * 10)
    else:
        hep.histplot((values, edges), ax=ax, histtype='step', color='blue', linewidth=3)
        ax.set_ylim(bottom=0, top=np.max(values) * 1.3 if has_positive else 10)
        if np.max(values) >= 1000:
            formatter_y = EngScalarFormatter(useMathText=True)
            formatter_y.set_powerlimits((0, 3))
            ax.yaxis.set_major_formatter(formatter_y)

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
    fig.savefig(output_path, dpi=300)
    plt.close(fig)

def process_file(path, output_dir=None, logy=False, run_subdirs=False):
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
        run_output_dir.mkdir(parents=True, exist_ok=True)

        with uproot.open(path) as file:
            plots_made = 0

            # 1. 1D Centrality plot
            if "hCentrality" in file:
                hist1d = file["hCentrality"]
                values, _ = hist1d.to_numpy()
                total_events = np.sum(values)

                extra_labels = []
                title = hist1d.title
                if title:
                    if ";" in title:
                        title = title.split(";")[0].strip()
                    cleaned_title = clean_root_latex(title)
                    if cleaned_title:
                        extra_labels.append(cleaned_title)
                extra_labels.append(f"Total: {total_events:.2e}")

                output_path = run_output_dir / f"run_{run_number}_hCentrality.png"
                make_1d_plot(
                    hist1d,
                    run_number,
                    output_path,
                    xlabel="Centrality [%]",
                    ylabel="Events",
                    extra_labels=extra_labels,
                    logy=logy,
                    xlim=(0, 100),
                )
                plots_made += 1
            else:
                print(f"Warning: 'hCentrality' not found in {path}")

            # 2. 1D Centrality Z150 Trig14 plot
            if "hCentralityZ150_Trig14" in file:
                hist1d_z150 = file["hCentralityZ150_Trig14"]
                values_z150, _ = hist1d_z150.to_numpy()
                total_events_z150 = np.sum(values_z150)

                extra_labels_z150 = []
                title_z150 = hist1d_z150.title
                if title_z150:
                    if ";" in title_z150:
                        title_z150 = title_z150.split(";")[0].strip()
                    cleaned_title_z150 = clean_root_latex(title_z150)
                    if cleaned_title_z150:
                        extra_labels_z150.append(cleaned_title_z150)
                extra_labels_z150.append(f"Total: {total_events_z150:.2e}")

                output_path_z150 = run_output_dir / f"run_{run_number}_hCentralityZ150_Trig14.png"
                make_1d_plot(
                    hist1d_z150,
                    run_number,
                    output_path_z150,
                    xlabel="Centrality [%]",
                    ylabel="Events",
                    extra_labels=extra_labels_z150,
                    logy=logy,
                    xlim=(0, 100),
                )
                plots_made += 1
            else:
                print(f"Warning: 'hCentralityZ150_Trig14' not found in {path}")

            # 3. 1D Z vertex plot (full X projection of h2ZVertexCentrality)
            if "h2ZVertexCentrality" in file:
                hist2d = file["h2ZVertexCentrality"]
                values_2d, edges_x, _ = hist2d.to_numpy()
                proj_x = np.sum(values_2d, axis=1)
                total_zvtx = np.sum(proj_x)

                extra_labels_zvtx = ["MB", f"Total: {total_zvtx:.2e}"]
                output_path_zvtx = run_output_dir / f"run_{run_number}_z_vertex.png"
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
                output_path_slices = run_output_dir / f"run_{run_number}_z_vertex_cent_slices.png"
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
