#!/usr/bin/env python3

import os
import sys
import csv
import argparse
import functools
import traceback
from pathlib import Path
import concurrent.futures

import numpy as np
import tqdm

# Ensure local imports from the same directory work reliably
script_dir = Path(__file__).resolve().parent
if str(script_dir) not in sys.path:
    sys.path.insert(0, str(script_dir))

try:
    import plot_centrality_qa
except ImportError:
    plot_centrality_qa = None

from centrality_qa_metrics import compute_centrality_average, extract_run_metrics
from centrality_qa_plots import (
    plot_heatmap,
    plot_trends,
    plot_metric_distributions,
    plot_ensemble_profile,
    plot_failure_example,
    plot_top_flat_example,
    plot_user_example,
    plot_failure_mode_metric_distribution,
    plot_centrality_1d_diagnostic
)


def write_summary_reports(data_list, output_dir):
    """
    Write CSV summary and plain text lists of good and flagged outlier runs.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "centrality_qa_summary.csv"
    flagged_list_path = output_dir / "flagged_outlier_runs.list"
    good_list_path = output_dir / "good_runs.list"

    fieldnames = [
        "run_number",
        "total_events",
        "plateau_avg",
        "rms_plat_pct",
        "max_dev_pct",
        "slope_per_10pct",
        "ratio_1",
        "ratio_1_5",
        "ratio_80",
        "chi2_ndf",
        "status",
        "file_path",
    ]

    with open(csv_path, "w", newline="") as f_csv, \
         open(flagged_list_path, "w") as f_bad, \
         open(good_list_path, "w") as f_good:

        writer = csv.DictWriter(f_csv, fieldnames=fieldnames)
        writer.writeheader()

        for d in data_list:
            row = {k: d[k] for k in fieldnames}
            # Format floats for clean CSV presentation
            for k in ["plateau_avg", "rms_plat_pct", "max_dev_pct", "slope_per_10pct", "ratio_1", "ratio_1_5", "ratio_80", "chi2_ndf"]:
                if isinstance(row[k], float) and not np.isnan(row[k]):
                    row[k] = f"{row[k]:.4f}"
            writer.writerow(row)

            if d["status"] == "GOOD":
                f_good.write(f"{d['run_number']}\n")
            else:
                f_bad.write(f"{d['run_number']}\t{d['status']}\n")

    print(f"Summary CSV written to: {csv_path}")
    print(f"Flagged runs list written to: {flagged_list_path}")
    print(f"Good runs list written to: {good_list_path}")


def generate_failure_mode_examples(metrics_list, output_dir, example_runs_per_mode=3,
                                   cent_flat_min=10.0, cent_flat_max=70.0,
                                   max_rms_pct=4.5, max_dev_pct=8.0,
                                   max_slope_per_10pct=2.5, max_central_spike=1.30,
                                   min_central_ratio=0.80):
    """
    For each failure mode, select up to `example_runs_per_mode` representative runs
    and generate their 1D centrality distribution + ratio diagnostic plots.
    Also generates a version of the relevant centrality metric distribution plot
    with vertical lines showing where the example runs fall.
    Saves plots into output_dir / "failure_examples" / <FAILURE_MODE> /
    Returns a dict mapping failure_mode -> list of example run numbers.
    """
    output_dir = Path(output_dir)
    examples_dir = output_dir / "failure_examples"

    # Configure failure modes with prioritization and domain-specific filtering:
    # 1. FLAG_SLOPE_DRIFT: extreme linear tilts across plateau
    # 2. FLAG_NON_FLAT: non-linear plateau structure (troughs, humps, ripples) excluding slope drift runs
    # 3. FLAG_CENTRAL_DROP: severe central drop (< 0.80)
    # 4. FLAG_CENTRAL_SPIKE: severe central spike (> 1.30)
    # 5. FLAG_EMPTY_OR_ZERO: zero events or zero plateau
    failure_modes_config = [
        {
            "mode": "FLAG_SLOPE_DRIFT",
            "description": "Plateau Slope Drift (|slope| > 2.5%/10% cent)",
            "filter_fn": lambda d: "FLAG_SLOPE_DRIFT" in d.get("status", "").split(";"),
            "sort_key": lambda d: abs(d.get("slope_per_10pct", 0.0)) if not np.isnan(d.get("slope_per_10pct", 0.0)) else 0.0,
            "reverse": True,
        },
        {
            "mode": "FLAG_NON_FLAT",
            "description": "Non-Flat Plateau (RMS > 4.5% or MaxDev > 8.0%)",
            # Prioritize pure non-flat runs not caught by any of the other three failure modes
            "filter_fn": lambda d: d.get("status") == "FLAG_NON_FLAT",
            "sort_key": lambda d: d.get("rms_plat_pct", 0.0) if not np.isnan(d.get("rms_plat_pct", 0.0)) else 0.0,
            "reverse": True,
        },
        {
            "mode": "FLAG_CENTRAL_DROP",
            "description": "Central Drop (R_1-5% < 0.80)",
            # Prioritize pure central drop runs not caught by any other failure mode
            "filter_fn": lambda d: d.get("status") == "FLAG_CENTRAL_DROP",
            "sort_key": lambda d: d.get("ratio_1_5", 1.0) if not np.isnan(d.get("ratio_1_5", 1.0)) else 1.0,
            "reverse": False,
        },
        {
            "mode": "FLAG_CENTRAL_SPIKE",
            "description": "Central Spike (R_1% > 1.30 or R_1-5% > 1.20)",
            # Prioritize pure central spike runs not caught by any other failure mode
            "filter_fn": lambda d: d.get("status") == "FLAG_CENTRAL_SPIKE",
            "sort_key": lambda d: max(
                d.get("ratio_1", 0.0) if not np.isnan(d.get("ratio_1", 0.0)) else 0.0,
                d.get("ratio_1_5", 0.0) if not np.isnan(d.get("ratio_1_5", 0.0)) else 0.0,
            ),
            "reverse": True,
        },
        {
            "mode": "FLAG_EMPTY_OR_ZERO",
            "description": "Empty Histogram or Zero Plateau",
            "filter_fn": lambda d: d.get("status") == "FLAG_EMPTY_OR_ZERO",
            "sort_key": lambda d: d.get("total_events", 0.0),
            "reverse": False,
        },
    ]

    examples_map = {}
    used_runs = set()

    for cfg in failure_modes_config:
        mode = cfg["mode"]
        filter_fn = cfg.get("filter_fn", lambda d: mode in d.get("status", "").split(";"))
        matching_runs = [d for d in metrics_list if filter_fn(d)]
        if not matching_runs:
            # Fallback if domain filter yielded no runs
            matching_runs = [d for d in metrics_list if mode in d.get("status", "").split(";")]

        if not matching_runs:
            examples_map[mode] = []
            continue

        # Prefer runs not already used as examples for another failure mode
        sorted_runs = sorted(
            matching_runs,
            key=lambda d: (d["run_number"] in used_runs, -cfg["sort_key"](d) if cfg["reverse"] else cfg["sort_key"](d))
        )
        selected = sorted_runs[:example_runs_per_mode]
        examples_map[mode] = [d["run_number"] for d in selected]
        for d in selected:
            used_runs.add(d["run_number"])

        mode_dir = examples_dir / mode
        mode_dir.mkdir(parents=True, exist_ok=True)

        for d in selected:
            plot_path = mode_dir / f"run_{d['run_number']}_centrality.png"
            plot_failure_example(
                metric=d,
                failure_mode=mode,
                output_path=plot_path,
                cent_flat_min=cent_flat_min,
                cent_flat_max=cent_flat_max,
            )

        # Generate the relevant metric distribution with vertical lines for the example runs
        dist_plot_path = mode_dir / "centrality_metric_distributions.png"
        plot_failure_mode_metric_distribution(
            metrics_list=metrics_list,
            failure_mode=mode,
            example_runs=[d["run_number"] for d in selected],
            output_path=dist_plot_path,
            max_rms_pct=max_rms_pct,
            max_dev_pct=max_dev_pct,
            max_slope_per_10pct=max_slope_per_10pct,
            max_central_spike=max_central_spike,
            min_central_ratio=min_central_ratio,
        )

    return examples_map


def generate_top_flat_examples(metrics_list, output_dir, n_examples=3,
                               cent_flat_min=10.0, cent_flat_max=70.0):
    """
    Select up to `n_examples` representative runs with the best / flattest centrality distributions
    (status == 'GOOD', ranked by lowest plateau RMS non-flatness) and generate their 1D
    centrality distribution + ratio diagnostic plots.
    Saves plots into output_dir / "top_flat_examples" /
    Returns a list of the selected run numbers.
    """
    output_dir = Path(output_dir)
    top_flat_dir = output_dir / "top_flat_examples"

    # Filter for GOOD runs first, or fallback to runs with valid plateau if none GOOD
    good_runs = [d for d in metrics_list if d.get("status") == "GOOD"]
    candidates = good_runs if good_runs else [
        d for d in metrics_list
        if d.get("plateau_avg", 0) > 0 and not np.isnan(d.get("rms_plat_pct", np.nan))
    ]

    if not candidates:
        print("Warning: No valid candidate runs found for top flat examples.")
        return []

    # Sort by lowest RMS non-flatness (flattest plateau)
    sorted_runs = sorted(
        candidates,
        key=lambda d: d.get("rms_plat_pct", 999.0) if not np.isnan(d.get("rms_plat_pct", 999.0)) else 999.0
    )
    selected = sorted_runs[:n_examples]
    top_flat_dir.mkdir(parents=True, exist_ok=True)

    selected_runs = []
    for rank, d in enumerate(selected, start=1):
        run_num = d["run_number"]
        selected_runs.append(run_num)
        plot_path = top_flat_dir / f"run_{run_num}_centrality.png"
        plot_top_flat_example(
            metric=d,
            rank=rank,
            output_path=plot_path,
            cent_flat_min=cent_flat_min,
            cent_flat_max=cent_flat_max,
        )

    return selected_runs


def generate_user_examples(metrics_list, output_dir, example_runs,
                           cent_flat_min=10.0, cent_flat_max=70.0):
    """
    Generate 2-panel 1D centrality diagnostic example plots (Events + Ratio to Plateau)
    for specific run numbers requested by the user.
    Saves plots into output_dir / "user_examples" / run_<run>_centrality.png
    Returns a list of run numbers that were successfully plotted.
    """
    if hasattr(metrics_list, "to_dict"):
        metrics_list = metrics_list.to_dict("records")

    if not example_runs:
        return []

    output_dir = Path(output_dir)
    user_examples_dir = output_dir / "user_examples"
    user_examples_dir.mkdir(parents=True, exist_ok=True)

    run_dict = {d.get("run_number", d.get("run")): d for d in metrics_list}
    plotted_runs = []

    for r in example_runs:
        if r not in run_dict:
            print(f"Warning: User-specified example run {r} not found in processed metrics.")
            continue
        d = run_dict[r]
        plot_path = user_examples_dir / f"run_{r}_centrality.png"
        plot_user_example(
            metric=d,
            output_path=plot_path,
            cent_flat_min=cent_flat_min,
            cent_flat_max=cent_flat_max,
            title_suffix="User Specified",
        )
        plotted_runs.append(r)

    if plotted_runs:
        print(f"User-specified example plots generated for {len(plotted_runs)} run(s) in {user_examples_dir.resolve()}")

    return plotted_runs


def print_failure_summary_table(metrics_list, output_dir=None, examples_map=None, top_flat_runs=None, user_example_runs=None):
    """
    Print an ASCII summary table of runs per failure mode to stdout
    and save it to centrality_qa_summary_table.txt in output_dir.
    """
    total_runs = len(metrics_list)
    good_runs = sum(1 for d in metrics_list if d.get("status") == "GOOD")
    flagged_runs = total_runs - good_runs

    failure_modes_info = [
        ("FLAG_CENTRAL_SPIKE", "Central Spike (R_1% > 1.30 / R_1-5% > 1.20)"),
        ("FLAG_CENTRAL_DROP",  "Central Drop (R_1-5% < 0.80)"),
        ("FLAG_NON_FLAT",      "Non-Flat Plateau (RMS > 4.5% / MaxDev > 8%)"),
        ("FLAG_SLOPE_DRIFT",   "Plateau Slope Drift (|slope| > 2.5%/10% cent)"),
        ("FLAG_EMPTY_OR_ZERO", "Empty Histogram or Zero Plateau Events"),
    ]

    mode_counts = {}
    mode_unique_counts = {}
    for mode, _ in failure_modes_info:
        cnt = sum(1 for d in metrics_list if mode in [f for f in str(d.get("status", "")).split(";") if f])
        uniq = sum(1 for d in metrics_list if [f for f in str(d.get("status", "")).split(";") if f] == [mode])
        mode_counts[mode] = cnt
        mode_unique_counts[mode] = uniq

    table_width = 98
    lines = []
    lines.append("=" * table_width)
    lines.append("CENTRALITY RUN AGGREGATE QA SUMMARY")
    lines.append("=" * table_width)
    lines.append(f"Total Runs Analyzed:     {total_runs:,}")
    pct_good = (good_runs / total_runs * 100.0) if total_runs > 0 else 0.0
    pct_flagged = (flagged_runs / total_runs * 100.0) if total_runs > 0 else 0.0
    lines.append(f"  Passed (GOOD):         {good_runs:,} ({pct_good:5.1f}%)")
    lines.append(f"  Flagged (Outliers):     {flagged_runs:,} ({pct_flagged:5.1f}%)")
    lines.append("-" * table_width)
    lines.append("FAILURE MODE BREAKDOWN:")
    lines.append(f"  {'Failure Mode':<22} {'Description':<46} {'Total':>7} {'% Runs':>8} {'Unique':>8} {'% Runs':>8}")
    lines.append("  " + "-" * (table_width - 4))

    for mode, desc in failure_modes_info:
        cnt = mode_counts[mode]
        uniq = mode_unique_counts[mode]
        pct = (cnt / total_runs * 100.0) if total_runs > 0 else 0.0
        pct_u = (uniq / total_runs * 100.0) if total_runs > 0 else 0.0
        lines.append(f"  {mode:<22} {desc:<46} {cnt:>7} {pct:>7.1f}% {uniq:>8} {pct_u:>7.1f}%")

    lines.append("-" * table_width)
    lines.append("(Note: 'Total' counts all runs triggering that flag; 'Unique' counts runs failing solely that mode)")

    if examples_map:
        lines.append("-" * table_width)
        lines.append("EXAMPLE RUNS GENERATED PER FAILURE MODE (in failure_examples/):")
        for mode, _ in failure_modes_info:
            ex_runs = examples_map.get(mode, [])
            if ex_runs:
                runs_str = ", ".join(str(r) for r in ex_runs)
                lines.append(f"  {mode:<22} -> Runs: [{runs_str}]  (dir: failure_examples/{mode}/)")
            else:
                lines.append(f"  {mode:<22} -> None flagged")

    if top_flat_runs:
        lines.append("-" * table_width)
        lines.append("TOP FLAT RUN EXAMPLES GENERATED (in top_flat_examples/):")
        run_dict = {d.get("run_number", d.get("run")): d for d in metrics_list}
        for rank, r in enumerate(top_flat_runs, start=1):
            d = run_dict.get(r, {})
            rms_val = f"{d.get('rms_plat_pct', np.nan):.2f}%" if not np.isnan(d.get('rms_plat_pct', np.nan)) else "N/A"
            slope_val = f"{d.get('slope_per_10pct', np.nan):+.2f}%" if not np.isnan(d.get('slope_per_10pct', np.nan)) else "N/A"
            evts_val = f"{float(d.get('total_events', 0)):.2e}" if d.get('total_events') is not None else "N/A"
            lines.append(f"  Rank #{rank}: Run {r:<8} (Plateau RMS: {rms_val}, Slope: {slope_val}, Evts: {evts_val})  -> top_flat_examples/run_{r}_centrality.png")

    if user_example_runs:
        lines.append("-" * table_width)
        lines.append("USER-SPECIFIED EXAMPLE RUNS GENERATED (in user_examples/):")
        run_dict = {d.get("run_number", d.get("run")): d for d in metrics_list}
        for r in user_example_runs:
            d = run_dict.get(r, {})
            status_val = d.get("status", "UNKNOWN")
            rms_val = f"{d.get('rms_plat_pct', np.nan):.2f}%" if not np.isnan(d.get('rms_plat_pct', np.nan)) else "N/A"
            slope_val = f"{d.get('slope_per_10pct', np.nan):+.2f}%" if not np.isnan(d.get('slope_per_10pct', np.nan)) else "N/A"
            evts_val = f"{float(d.get('total_events', 0)):.2e}" if d.get('total_events') is not None else "N/A"
            lines.append(f"  Run {r:<8} (Status: {status_val}, Plateau RMS: {rms_val}, Slope: {slope_val}, Evts: {evts_val})  -> user_examples/run_{r}_centrality.png")

    lines.append("=" * table_width)

    summary_text = "\n".join(lines)
    print("\n" + summary_text + "\n")

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        table_path = output_dir / "centrality_qa_summary_table.txt"
        with open(table_path, "w") as f:
            f.write(summary_text + "\n")
        print(f"Summary table text file saved to: {table_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Run Aggregate QA: Quantify Centrality Flatness, Generate Trend/Heatmap Plots, and Flag Outlier Runs."
    )
    parser.add_argument("-f", "--file", type=Path, help="Text file containing list of ROOT file paths (one per line).")
    parser.add_argument("-o", "--output-dir", type=Path, default=Path("centrality_aggregate_qa"),
                        help="Directory to save aggregate plots, reports, and detailed test QA (default: centrality_aggregate_qa).")
    parser.add_argument("--hist-name", type=str, default="hCentrality",
                        help="Name of centrality histogram to analyze (default: hCentrality).")
    parser.add_argument("--test-runs", nargs="*", type=int, default=[68144, 72020, 76020],
                        help="Run numbers to generate detailed per-run QA plots for (default: 68144 72020 76020).")
    parser.add_argument("--no-test-plots", action="store_true",
                        help="Disable generating detailed per-run plots for the test runs.")
    parser.add_argument("--plot-outliers", action="store_true",
                        help="Generate detailed per-run QA plots for all flagged outlier runs as well.")
    parser.add_argument("-j", "--workers", type=int, default=None,
                        help="Number of parallel worker processes (default: min(cpu_count(), 32)).")
    parser.add_argument("--cent-flat-min", type=float, default=10.0,
                        help="Minimum centrality [%%] for flat plateau (default: 10.0).")
    parser.add_argument("--cent-flat-max", type=float, default=70.0,
                        help="Maximum centrality [%%] for flat plateau (default: 70.0).")
    parser.add_argument("--max-rms", type=float, default=4.5,
                        help="Maximum plateau RMS non-flatness [%%] threshold before flagging NON_FLAT (default: 4.5).")
    parser.add_argument("--max-dev", type=float, default=8.0,
                        help="Maximum single-bin deviation [%%] threshold on plateau (default: 8.0).")
    parser.add_argument("--max-slope", type=float, default=2.5,
                        help="Maximum slope tolerance in %% per 10%% centrality (default: 2.5).")
    parser.add_argument("--max-central-spike", type=float, default=1.30,
                        help="Maximum allowed ratio in 1%% centrality bin (R_1%%) before flagging CENTRAL_SPIKE (default: 1.30).")
    parser.add_argument("--min-central-ratio", type=float, default=0.80,
                        help="Minimum allowed ratio in 1-5%% centrality (R_1-5%%) before flagging CENTRAL_DROP (default: 0.80).")
    parser.add_argument("--max-central-ratio", type=float, default=1.20,
                        help="Maximum allowed ratio in 1-5%% centrality (R_1-5%%) before flagging CENTRAL_SPIKE (default: 1.20).")
    parser.add_argument("--syst-floor", type=float, default=0.02,
                        help="Systematic fractional uncertainty floor for chi2 computation (default: 0.02).")
    parser.add_argument("--example-runs-per-mode", type=int, default=3,
                        help="Number of representative example run plots to generate for each failure mode (default: 3).")
    parser.add_argument("--no-failure-examples", action="store_true",
                        help="Disable generating failure mode example plots.")
    parser.add_argument("--top-flat-examples", type=int, default=3,
                        help="Number of representative example run plots to generate for top flattest runs (default: 3).")
    parser.add_argument("--no-top-flat-examples", action="store_true",
                        help="Disable generating top flat example plots.")
    parser.add_argument("--example-runs", "--user-example-runs", "--user-runs",
                        nargs="*", default=None,
                        help="Specific run number(s) specified by user to generate 2-panel 1D centrality diagnostic example plots for.")
    parser.add_argument("files", nargs="*", type=Path, help="Positional list of ROOT file paths.")
    args = parser.parse_args()

    file_list = []
    if args.files:
        file_list.extend(args.files)

    if args.file:
        try:
            with args.file.open("r") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#"):
                        file_list.append(Path(line))
        except Exception as e:
            print(f"Error reading file list {args.file}: {e}")
            sys.exit(1)

    if not file_list:
        print("Error: You must provide at least one ROOT file or a text file containing ROOT paths (-f / --file).")
        parser.print_help()
        sys.exit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    files_to_process = [Path(p) for p in file_list]
    max_workers = args.workers if args.workers is not None else min(os.cpu_count() or 4, 32)

    print(f"============================================================")
    print(f"Centrality Run Aggregate QA")
    print(f"Input files:        {len(files_to_process)}")
    print(f"Histogram:          {args.hist_name}")
    print(f"Plateau Range:      [{args.cent_flat_min}%, {args.cent_flat_max}%]")
    print(f"Max Plateau RMS:    {args.max_rms}%")
    print(f"Max Plateau Slope:  {args.max_slope}% per 10% cent")
    print(f"Central Spike Cut:  R(1%) > {args.max_central_spike}")
    print(f"Test runs:          {args.test_runs}")
    print(f"Output dir:         {args.output_dir.resolve()}")
    print(f"Worker count:       {max_workers}")
    print(f"============================================================")

    # 1. Parallel Extraction of Metrics
    extract_func = functools.partial(
        extract_run_metrics,
        hist_name=args.hist_name,
        cent_flat_min=args.cent_flat_min,
        cent_flat_max=args.cent_flat_max,
        max_rms_pct=args.max_rms,
        max_dev_pct_cut=args.max_dev,
        max_slope_per_10pct=args.max_slope,
        min_central_ratio=args.min_central_ratio,
        max_central_ratio=args.max_central_ratio,
        max_central_spike=args.max_central_spike,
        syst_floor=args.syst_floor,
    )

    print("Extracting flatness metrics across all runs...")
    metrics_list = []
    errors = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = list(tqdm.tqdm(executor.map(extract_func, files_to_process), total=len(files_to_process)))

    for m, err in results:
        if m is not None:
            metrics_list.append(m)
        if err:
            errors.append(err)

    if not metrics_list:
        print("Error: No valid histogram data could be extracted from any of the input files.")
        if errors:
            print("First 5 errors:")
            for err in errors[:5]:
                print(f"  {err}")
        sys.exit(1)

    # Sort runs chronologically by run number
    metrics_list.sort(key=lambda d: d["run_number"])
    n_total = len(metrics_list)
    n_good = sum(1 for d in metrics_list if d["status"] == "GOOD")
    n_flagged = n_total - n_good

    print(f"Extracted metrics for {n_total} runs ({n_good} GOOD, {n_flagged} FLAGGED).")
    if errors:
        print(f"Encountered {len(errors)} warnings/errors during extraction.")

    # 2. Generate Aggregate Plots
    print("Generating Run Aggregate plots...")
    plot_heatmap(
        metrics_list,
        args.output_dir / "centrality_ratio_heatmap_all_runs.png",
        test_runs=args.test_runs,
    )
    plot_trends(
        metrics_list,
        args.output_dir / "centrality_flatness_trends.png",
        test_runs=args.test_runs,
        max_rms_pct=args.max_rms,
        max_slope_per_10pct=args.max_slope,
        max_central_spike=args.max_central_spike,
        min_central_ratio=args.min_central_ratio,
    )
    plot_metric_distributions(
        metrics_list,
        args.output_dir / "centrality_metric_distributions.png",
        max_rms_pct=args.max_rms,
        max_slope_per_10pct=args.max_slope,
        max_central_spike=args.max_central_spike,
        min_central_ratio=args.min_central_ratio,
    )
    plot_ensemble_profile(
        metrics_list,
        args.output_dir / "centrality_ensemble_profile.png",
        subtitle=f"All Processed Runs (N = {n_total})",
    )
    good_metrics = [d for d in metrics_list if d.get("status") == "GOOD"]
    if good_metrics:
        plot_ensemble_profile(
            good_metrics,
            args.output_dir / "centrality_ensemble_profile_good_runs.png",
            subtitle=f"Good Runs (Passed All QA, N = {len(good_metrics)})",
        )
    else:
        print("Warning: No runs passed all QA checks. Skipping centrality_ensemble_profile_good_runs.png.")

    # 3. Write Reports
    write_summary_reports(metrics_list, args.output_dir)

    # 4. Generate Example 1D Centrality Plots for Each Failure Mode
    examples_map = None
    if not args.no_failure_examples:
        print(f"Generating example 1D centrality plots for each failure mode (up to {args.example_runs_per_mode} per mode)...")
        examples_map = generate_failure_mode_examples(
            metrics_list,
            args.output_dir,
            example_runs_per_mode=args.example_runs_per_mode,
            cent_flat_min=args.cent_flat_min,
            cent_flat_max=args.cent_flat_max,
            max_rms_pct=args.max_rms,
            max_dev_pct=args.max_dev,
            max_slope_per_10pct=args.max_slope,
            max_central_spike=args.max_central_spike,
            min_central_ratio=args.min_central_ratio,
        )

    # 4b. Generate Example 1D Centrality Plots for Top Flat Runs
    top_flat_runs = []
    if not args.no_top_flat_examples and args.top_flat_examples > 0:
        print(f"Generating example 1D centrality plots for top {args.top_flat_examples} flattest runs...")
        top_flat_runs = generate_top_flat_examples(
            metrics_list,
            args.output_dir,
            n_examples=args.top_flat_examples,
            cent_flat_min=args.cent_flat_min,
            cent_flat_max=args.cent_flat_max,
        )

    # 4c. Generate Example 1D Centrality Plots for User-Specified Runs
    user_example_runs = []
    if args.example_runs:
        user_runs_to_plot = []
        for item in args.example_runs:
            for part in str(item).replace(",", " ").split():
                try:
                    user_runs_to_plot.append(int(part))
                except ValueError:
                    print(f"Warning: Could not parse '{part}' as a run number.")
        if user_runs_to_plot:
            print(f"Generating example 1D centrality plots for {len(user_runs_to_plot)} user-specified run(s)...")
            user_example_runs = generate_user_examples(
                metrics_list,
                args.output_dir,
                example_runs=user_runs_to_plot,
                cent_flat_min=args.cent_flat_min,
                cent_flat_max=args.cent_flat_max,
            )

    # 5. Selective Detailed Plotting
    runs_to_detail = set()
    if not args.no_test_plots and args.test_runs:
        runs_to_detail.update(args.test_runs)

    if args.plot_outliers:
        flagged_runs = [d["run_number"] for d in metrics_list if d["status"] != "GOOD"]
        runs_to_detail.update(flagged_runs)

    if runs_to_detail and plot_centrality_qa is not None:
        # Find file paths corresponding to selected runs
        run_to_path = {d["run_number"]: Path(d["file_path"]) for d in metrics_list}
        target_files = [run_to_path[r] for r in runs_to_detail if r in run_to_path]

        if target_files:
            print(f"Generating detailed per-run QA plots for {len(target_files)} selected run(s)...")
            detail_output_dir = args.output_dir / "detailed_qa"
            detail_func = functools.partial(
                plot_centrality_qa.process_file,
                output_dir=detail_output_dir,
                logy=False,
                run_subdirs=True,
                cent_flat_min=args.cent_flat_min,
                cent_flat_max=args.cent_flat_max,
            )

            with concurrent.futures.ProcessPoolExecutor(max_workers=min(len(target_files), max_workers)) as executor:
                detail_results = list(tqdm.tqdm(executor.map(detail_func, target_files), total=len(target_files)))

            detail_errs = [e for e in detail_results if e is not None]
            if detail_errs:
                print(f"Detailed plotting had {len(detail_errs)} warning(s)/error(s).")
            print(f"Detailed QA plots saved in: {detail_output_dir.resolve()}")
    elif runs_to_detail and plot_centrality_qa is None:
        print("Warning: Could not import plot_centrality_qa.py; skipping detailed per-run plots.")

    # 6. Print and Save Summary Table
    print_failure_summary_table(
        metrics_list,
        output_dir=args.output_dir,
        examples_map=examples_map,
        top_flat_runs=top_flat_runs,
        user_example_runs=user_example_runs,
    )

    print(f"All Run Aggregate QA completed successfully. Results saved to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
