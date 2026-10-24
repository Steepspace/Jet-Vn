#!/usr/bin/env python3

import os
import sys
import csv
import pickle
import hashlib
import argparse
import functools
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

import centrality_qa_metrics
from centrality_qa_metrics import extract_run_metrics, R1_CUTS, R1_5_CUTS
from centrality_qa_plots import (
    plot_heatmap,
    plot_trends,
    plot_trigger_exclusive,
    plot_metric_distributions,
    plot_ensemble_profile,
    plot_failure_example,
    plot_top_flat_example,
    plot_user_example,
    plot_failure_mode_metric_distribution,
)


def failure_mode_descriptions(args):
    """(flag, description) for every failure mode, in summary-table order."""
    return [
        ("FLAG_CALIB_SHIFT", f"A 20% Class Off Flat by > {args.max_class_dev}% (1-{args.flat_max:.0f}%)"),
        ("FLAG_R1_LOW",      f"R_1% < {R1_CUTS[0]:g}"),
        ("FLAG_R1_HIGH",     f"R_1% > {R1_CUTS[1]:g}"),
        ("FLAG_R1_5_LOW",    f"R_1-5% < {R1_5_CUTS[0]:g}"),
        ("FLAG_R1_5_HIGH",   f"R_1-5% > {R1_5_CUTS[1]:g}"),
        ("FLAG_LOW_STATS",   f"< {args.min_events:.0e} Events in 1-{args.flat_max:.0f}% (incl. empty)"),
    ]


def _run_now(fn, **kwargs):
    """Default plot submitter: render immediately in this process."""
    return fn(**kwargs)


def _file_stamp(path):
    st = path.stat()
    return (st.st_mtime_ns, st.st_size)


def extract_metrics_cached(files, extract_kwargs, cache_path=None, max_workers=8):
    """
    Extract per-run metrics in parallel, reusing entries from a pickle cache for input
    files whose modification time and size are unchanged. The whole cache is ignored if
    the extraction settings or centrality_qa_metrics.py itself have changed.
    Returns (metrics_list, errors).
    """
    settings = (sorted(extract_kwargs.items()),
                hashlib.sha1(Path(centrality_qa_metrics.__file__).read_bytes()).hexdigest())
    entries = {}
    if cache_path is not None and cache_path.exists():
        try:
            with cache_path.open("rb") as f:
                cache = pickle.load(f)
            if cache.get("settings") == settings:
                entries = cache["entries"]
            else:
                print("Metric settings or code changed since the cache was written; re-extracting all runs.")
        except Exception as e:
            print(f"Warning: could not read metrics cache {cache_path}: {e}")

    results = {}
    todo = []
    for p in files:
        key = str(p)
        try:
            stamp = _file_stamp(p)
        except OSError:
            stamp = None
        hit = entries.get(key)
        if stamp is not None and hit is not None and hit["stamp"] == stamp:
            results[key] = (hit["metric"], None)
        else:
            todo.append((p, stamp))

    if cache_path is not None:
        print(f"Metrics cache: {len(results)} run(s) reused, {len(todo)} to extract ({cache_path})")

    if todo:
        extract_func = functools.partial(extract_run_metrics, **extract_kwargs)
        todo_paths = [p for p, _ in todo]
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            extracted = list(tqdm.tqdm(executor.map(extract_func, todo_paths), total=len(todo_paths)))
        for (p, stamp), (m, err) in zip(todo, extracted):
            results[str(p)] = (m, err)
            if m is not None and stamp is not None:
                entries[str(p)] = {"stamp": stamp, "metric": m}

        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = cache_path.with_name(cache_path.name + ".tmp")
            with tmp_path.open("wb") as f:
                pickle.dump({"settings": settings, "entries": entries}, f, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(tmp_path, cache_path)

    metrics_list = [m for m, _ in results.values() if m is not None]
    errors = [err for _, err in results.values() if err]
    return metrics_list, errors


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
        "flat_events",
        "flat_ref",
        "class_dev_max",
        "class_dev_signed",
        "worst_class",
        "ratio_1",
        "ratio_1_5",
        "ratio_80",
        "status",
        "file_path",
    ]
    float_fields = fieldnames[2:-2]

    with open(csv_path, "w", newline="") as f_csv, \
         open(flagged_list_path, "w") as f_bad, \
         open(good_list_path, "w") as f_good:

        writer = csv.DictWriter(f_csv, fieldnames=fieldnames)
        writer.writeheader()

        for d in data_list:
            row = {k: d[k] for k in fieldnames}
            # Format floats for clean CSV presentation
            for k in float_fields:
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


def _metric_or(d, key, default):
    v = d.get(key, default)
    return default if v is None or np.isnan(v) else v


def _select_by_severity(matching_runs, used_runs, n, sort_key, reverse):
    """Most severe runs first, preferring runs not already used as examples for another mode."""
    return sorted(
        matching_runs,
        key=lambda d: (d["run_number"] in used_runs, -sort_key(d) if reverse else sort_key(d))
    )[:n]


def _select_at_peak(matching_runs, n, bin_width=0.25):
    """
    Runs at the most populated bin of the worst-class deviation distribution, spread across the
    run range, so the examples show what a typical run of that population looks like.
    """
    devs = np.array([d["class_dev_max"] for d in matching_runs])
    bins = np.arange(devs.min(), devs.max() + bin_width, bin_width)
    if len(bins) < 2:
        return matching_runs[:n]
    counts, edges = np.histogram(devs, bins=bins)
    peak = 0.5 * (edges[np.argmax(counts)] + edges[np.argmax(counts) + 1])

    near_peak = sorted((d for d in matching_runs if abs(d["class_dev_max"] - peak) <= bin_width),
                       key=lambda d: d["run_number"])
    if len(near_peak) <= n:
        return near_peak
    picks = np.linspace(0, len(near_peak) - 1, n + 2)[1:-1].round().astype(int)
    return [near_peak[i] for i in picks]


def generate_failure_mode_examples(metrics_list, output_dir, example_runs_per_mode=3,
                                   flat_max=80.0, max_class_dev=3.5, min_events=1e5, submit=_run_now):
    """
    Select up to `example_runs_per_mode` representative runs per example set and generate their
    1D centrality distribution + ratio diagnostic plots, plus the relevant metric distribution
    with vertical lines showing where the example runs fall.
      class_dev_examples / {first_peak, second_peak, far_outliers} /: the populations of the
          worst 20% class deviation (good runs, typical miscalibrated runs, the largest deviations)
      failure_examples / <FLAG> /: one set per central-ratio flag and LOW_STATS
    Plots are handed to `submit(fn, **kwargs)`, which renders them immediately by default.
    Returns a dict mapping example label -> {"runs": [...], "dir": <relative dir>}.
    """
    output_dir = Path(output_dir)
    n = example_runs_per_mode

    def has_flag(d, mode):
        return mode in d.get("status", "").split(";")

    def by_flag(mode, severity):
        # Runs failing only this flag first, then the furthest past the cut
        return {
            "label": mode,
            "mode": mode,
            "subdir": f"failure_examples/{mode}",
            "candidates": lambda: [d for d in metrics_list if has_flag(d, mode)],
            "select": lambda runs, used: sorted(
                runs, key=lambda d: (d.get("status") != mode, d["run_number"] in used, -severity(d)))[:n],
        }

    with_dev = [d for d in metrics_list if not np.isnan(d.get("class_dev_max", np.nan))]
    example_sets = [
        {
            "label": "Class Dev: First Peak",
            "mode": "FLAG_CALIB_SHIFT",
            "subdir": "class_dev_examples/first_peak",
            "candidates": lambda: [d for d in with_dev if d.get("status") == "GOOD"],
            "select": lambda runs, used: _select_at_peak(runs, n),
        },
        {
            "label": "Class Dev: Second Peak",
            "mode": "FLAG_CALIB_SHIFT",
            "subdir": "class_dev_examples/second_peak",
            "candidates": lambda: [d for d in with_dev if d.get("status") == "FLAG_CALIB_SHIFT"],
            "select": lambda runs, used: _select_at_peak(runs, n),
        },
        {
            "label": "Class Dev: Far Outliers",
            "mode": "FLAG_CALIB_SHIFT",
            "subdir": "class_dev_examples/far_outliers",
            "candidates": lambda: with_dev,
            "select": lambda runs, used: _select_by_severity(runs, set(), n, lambda d: d["class_dev_max"], True),
        },
        by_flag("FLAG_R1_LOW", lambda d: R1_CUTS[0] - d["ratio_1"]),
        by_flag("FLAG_R1_HIGH", lambda d: d["ratio_1"] - R1_CUTS[1]),
        by_flag("FLAG_R1_5_LOW", lambda d: R1_5_CUTS[0] - d["ratio_1_5"]),
        by_flag("FLAG_R1_5_HIGH", lambda d: d["ratio_1_5"] - R1_5_CUTS[1]),
        by_flag("FLAG_LOW_STATS", lambda d: -d.get("total_events", 0.0)),
    ]

    examples_map = {}
    used_runs = set()

    for ex in example_sets:
        rel_dir = f"{ex['subdir']}/"
        matching_runs = ex["candidates"]()
        selected = ex["select"](matching_runs, used_runs) if matching_runs else []
        examples_map[ex["label"]] = {"runs": [d["run_number"] for d in selected], "dir": rel_dir}
        if not selected:
            continue
        used_runs.update(d["run_number"] for d in selected)

        mode_dir = output_dir / ex["subdir"]
        mode_dir.mkdir(parents=True, exist_ok=True)

        for d in selected:
            submit(
                plot_failure_example,
                metric=d,
                failure_mode=ex["label"],
                output_path=mode_dir / f"run_{d['run_number']}_centrality.png",
                flat_max=flat_max,
                max_class_dev=max_class_dev,
            )

        # Generate the relevant metric distribution with vertical lines for the example runs
        submit(
            plot_failure_mode_metric_distribution,
            metrics_list=metrics_list,
            failure_mode=ex["mode"],
            example_runs=[d["run_number"] for d in selected],
            output_path=mode_dir / "centrality_metric_distributions.png",
            max_class_dev=max_class_dev,
            min_events=min_events,
            title=ex["label"],
        )

    return examples_map


def generate_top_flat_examples(metrics_list, output_dir, n_examples=3,
                               flat_max=80.0, max_class_dev=3.5, submit=_run_now):
    """
    Select up to `n_examples` representative runs with the best / flattest centrality distributions
    (status == 'GOOD', ranked by lowest worst-class deviation) and generate their 1D
    centrality distribution + ratio diagnostic plots.
    Saves plots into output_dir / "top_flat_examples" /
    Returns a list of the selected run numbers.
    """
    output_dir = Path(output_dir)
    top_flat_dir = output_dir / "top_flat_examples"

    # Filter for GOOD runs first, or fallback to any run with a valid class deviation if none GOOD
    good_runs = [d for d in metrics_list if d.get("status") == "GOOD"]
    candidates = good_runs if good_runs else [
        d for d in metrics_list if not np.isnan(d.get("class_dev_max", np.nan))
    ]

    if not candidates:
        print("Warning: No valid candidate runs found for top flat examples.")
        return []

    sorted_runs = sorted(candidates, key=lambda d: _metric_or(d, "class_dev_max", 999.0))
    selected = sorted_runs[:n_examples]
    top_flat_dir.mkdir(parents=True, exist_ok=True)

    selected_runs = []
    for rank, d in enumerate(selected, start=1):
        run_num = d["run_number"]
        selected_runs.append(run_num)
        plot_path = top_flat_dir / f"run_{run_num}_centrality.png"
        submit(
            plot_top_flat_example,
            metric=d,
            rank=rank,
            output_path=plot_path,
            flat_max=flat_max,
            max_class_dev=max_class_dev,
        )

    return selected_runs


def generate_user_examples(metrics_list, output_dir, example_runs,
                           flat_max=80.0, max_class_dev=3.5, submit=_run_now):
    """
    Generate 1D centrality diagnostic example plots for specific run numbers requested by the user.
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
        submit(
            plot_user_example,
            metric=d,
            output_path=plot_path,
            flat_max=flat_max,
            max_class_dev=max_class_dev,
            title_suffix="User Specified",
        )
        plotted_runs.append(r)

    if plotted_runs:
        print(f"User-specified example plots queued for {len(plotted_runs)} run(s) in {user_examples_dir.resolve()}")

    return plotted_runs


def print_failure_summary_table(metrics_list, failure_modes_info, output_dir=None, examples_map=None, top_flat_runs=None, user_example_runs=None):
    """
    Print an ASCII summary table of runs per failure mode to stdout
    and save it to centrality_qa_summary_table.txt in output_dir.
    """
    total_runs = len(metrics_list)
    good_runs = sum(1 for d in metrics_list if d.get("status") == "GOOD")
    flagged_runs = total_runs - good_runs

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
        lines.append("EXAMPLE RUNS GENERATED (class_dev_examples/ and failure_examples/):")
        for label, ex in examples_map.items():
            if ex["runs"]:
                runs_str = ", ".join(str(r) for r in ex["runs"])
                lines.append(f"  {label:<28} -> Runs: [{runs_str}]  (dir: {ex['dir']})")
            else:
                lines.append(f"  {label:<28} -> None flagged")

    if top_flat_runs:
        lines.append("-" * table_width)
        lines.append("TOP FLAT RUN EXAMPLES GENERATED (in top_flat_examples/):")
        run_dict = {d.get("run_number", d.get("run")): d for d in metrics_list}
        for rank, r in enumerate(top_flat_runs, start=1):
            d = run_dict.get(r, {})
            dev_val = f"{d.get('class_dev_max', np.nan):.1f}%" if not np.isnan(d.get('class_dev_max', np.nan)) else "N/A"
            evts_val = f"{float(d.get('total_events', 0)):.2e}" if d.get('total_events') is not None else "N/A"
            lines.append(f"  Rank #{rank}: Run {r:<8} (Worst 20% Class: {dev_val}, Evts: {evts_val})  -> top_flat_examples/run_{r}_centrality.png")

    if user_example_runs:
        lines.append("-" * table_width)
        lines.append("USER-SPECIFIED EXAMPLE RUNS GENERATED (in user_examples/):")
        run_dict = {d.get("run_number", d.get("run")): d for d in metrics_list}
        for r in user_example_runs:
            d = run_dict.get(r, {})
            status_val = d.get("status", "UNKNOWN")
            dev_val = f"{d.get('class_dev_max', np.nan):.1f}%" if not np.isnan(d.get('class_dev_max', np.nan)) else "N/A"
            evts_val = f"{float(d.get('total_events', 0)):.2e}" if d.get('total_events') is not None else "N/A"
            lines.append(f"  Run {r:<8} (Status: {status_val}, Worst 20% Class: {dev_val}, Evts: {evts_val})  -> user_examples/run_{r}_centrality.png")

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
                        help="Name of centrality histogram to analyze (e.g. hCentrality, hCentralityZ150, hCentrality_Trig14, or 2D projection; default: hCentrality).")
    parser.add_argument("-j", "--workers", type=int, default=None,
                        help="Number of parallel worker processes (default: min(cpu_count(), 32)).")
    parser.add_argument("--flat-max", type=float, default=80.0,
                        help="Upper edge [%%] of the centrality range required to be flat (default: 80).")
    parser.add_argument("--max-class-dev", type=float, default=3.5,
                        help="Maximum deviation [%%] of any 20%%-wide centrality class from its expected "
                             "event count before flagging CALIB_SHIFT (default: 3.5).")
    parser.add_argument("--min-events", type=float, default=1e5,
                        help="Minimum events in the flatness range; runs below are flagged LOW_STATS (default: 1e5).")
    parser.add_argument("--n-examples", type=int, default=3,
                        help="Example run plots per failure mode and for the flattest runs; 0 disables them (default: 3).")
    parser.add_argument("--example-runs", nargs="*", default=None,
                        help="Extra run number(s) to make 1D centrality diagnostic plots for.")
    parser.add_argument("--test-runs", nargs="*", type=int, default=[68144, 72020, 76020],
                        help="Runs to mark on the trend/heatmap plots and make detailed per-run QA plots for; "
                             "pass the flag with no runs to disable (default: 68144 72020 76020).")
    parser.add_argument("--cache-file", type=Path, default=None,
                        help="Pickle cache of per-run metrics; unchanged input files are not re-read "
                             "(default: <output-dir>/centrality_metrics_cache.pkl).")
    parser.add_argument("--no-cache", action="store_true",
                        help="Re-extract every run without reading or writing the metrics cache.")
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
    print(f"Flatness Range:     [0%, {args.flat_max}%]")
    print(f"Max Class Dev:      {args.max_class_dev}% (20%-wide classes)")
    print(f"Min Events:         {args.min_events:.0e}")
    print(f"Test runs:          {args.test_runs}")
    print(f"Output dir:         {args.output_dir.resolve()}")
    print(f"Worker count:       {max_workers}")
    print(f"============================================================")

    # 1. Parallel Extraction of Metrics (cached)
    cache_path = None if args.no_cache else (args.cache_file or args.output_dir / "centrality_metrics_cache.pkl")
    print("Extracting flatness metrics across all runs...")
    metrics_list, errors = extract_metrics_cached(
        files_to_process,
        extract_kwargs=dict(
            hist_name=args.hist_name,
            flat_max=args.flat_max,
            max_class_dev=args.max_class_dev,
            min_events=args.min_events,
        ),
        cache_path=cache_path,
        max_workers=max_workers,
    )

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

    # 2. Render every plot in one worker pool. Jobs are queued longest first so the slow
    #    detailed per-run QA overlaps with the aggregate and example plots.
    pool = concurrent.futures.ProcessPoolExecutor(max_workers=max_workers)
    futures = {}

    def submit(fn, **kwargs):
        futures[pool.submit(fn, **kwargs)] = fn.__name__

    runs_to_detail = set(args.test_runs or [])
    detail_output_dir = args.output_dir / "detailed_qa"
    if runs_to_detail and plot_centrality_qa is not None:
        run_to_path = {d["run_number"]: Path(d["file_path"]) for d in metrics_list}
        for r in sorted(runs_to_detail):
            if r in run_to_path:
                submit(plot_centrality_qa.process_file, path=run_to_path[r],
                       output_dir=detail_output_dir, logy=False, run_subdirs=True)
    elif runs_to_detail and plot_centrality_qa is None:
        print("Warning: Could not import plot_centrality_qa.py; skipping detailed per-run plots.")

    submit(plot_metric_distributions, data_list=metrics_list,
           output_path=args.output_dir / "centrality_metric_distributions.png",
           max_class_dev=args.max_class_dev)
    submit(plot_trends, data_list=metrics_list,
           output_path=args.output_dir / "centrality_flatness_trends.png",
           test_runs=args.test_runs, max_class_dev=args.max_class_dev, flat_max=args.flat_max)
    submit(plot_trigger_exclusive, data_list=metrics_list,
           output_path=args.output_dir / "trigger_exclusive_events.png")
    submit(plot_heatmap, data_list=metrics_list,
           output_path=args.output_dir / "centrality_ratio_heatmap_all_runs.png",
           test_runs=args.test_runs)
    submit(plot_ensemble_profile, data_list=metrics_list,
           output_path=args.output_dir / "centrality_ensemble_profile.png",
           subtitle=f"All Processed Runs (N = {n_total})")
    good_metrics = [d for d in metrics_list if d.get("status") == "GOOD"]
    if good_metrics:
        submit(plot_ensemble_profile, data_list=good_metrics,
               output_path=args.output_dir / "centrality_ensemble_profile_good_runs.png",
               subtitle=f"Good Runs (Passed All QA, N = {len(good_metrics)})")
    else:
        print("Warning: No runs passed all QA checks. Skipping centrality_ensemble_profile_good_runs.png.")

    # 3. Example 1D Centrality Plots for Each Failure Mode, the Flattest Runs, and User Runs
    examples_map = None
    top_flat_runs = []
    if args.n_examples > 0:
        examples_map = generate_failure_mode_examples(
            metrics_list,
            args.output_dir,
            example_runs_per_mode=args.n_examples,
            flat_max=args.flat_max,
            max_class_dev=args.max_class_dev,
            min_events=args.min_events,
            submit=submit,
        )
        top_flat_runs = generate_top_flat_examples(
            metrics_list,
            args.output_dir,
            n_examples=args.n_examples,
            flat_max=args.flat_max,
            max_class_dev=args.max_class_dev,
            submit=submit,
        )

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
            user_example_runs = generate_user_examples(
                metrics_list,
                args.output_dir,
                example_runs=user_runs_to_plot,
                flat_max=args.flat_max,
                max_class_dev=args.max_class_dev,
                submit=submit,
            )

    # 4. Write Reports while the plots render
    write_summary_reports(metrics_list, args.output_dir)

    # 5. Wait for all plots
    print(f"Rendering {len(futures)} plot job(s) on {max_workers} worker(s)...")
    plot_errors = []
    for fut in tqdm.tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
        try:
            result = fut.result()
        except Exception as e:
            plot_errors.append(f"{futures[fut]}: {e!r}")
            continue
        # plot_centrality_qa.process_file reports problems through its return value
        if futures[fut] == "process_file" and result is not None:
            plot_errors.append(f"process_file: {result}")
    pool.shutdown()

    if plot_errors:
        print(f"{len(plot_errors)} plot job(s) reported errors:")
        for err in plot_errors[:5]:
            print(f"  {err}")
    if runs_to_detail and plot_centrality_qa is not None:
        print(f"Detailed QA plots saved in: {detail_output_dir.resolve()}")

    # 6. Print and Save Summary Table
    print_failure_summary_table(
        metrics_list,
        failure_mode_descriptions(args),
        output_dir=args.output_dir,
        examples_map=examples_map,
        top_flat_runs=top_flat_runs,
        user_example_runs=user_example_runs,
    )

    print(f"All Run Aggregate QA completed successfully. Results saved to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
