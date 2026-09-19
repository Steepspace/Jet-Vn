#!/usr/bin/env python3
"""
Find and Filter Segments Missing ZDC Calibration

This script compares merged list files against ZDC calibration list files.
For each run in the merged directory, it checks whether each segment has a
corresponding ZDC calibration segment.

If any segments in a run are missing ZDC calibration, it writes a new list file
(one per run) in the specified output directory containing only the lines from
the merged list that need ZDC calibration.

Processing is parallelized using concurrent.futures.ProcessPoolExecutor.

Example usage:
    python3 find_missing_zdc_calib.py \\
        --zdc-dir /sphenix/u/anarde/sEPD-Study/scratch/ZDC/07-01-26/lists \\
        --merged-dir /direct/sphenix+u/anarde/Documents/sPHENIX/Jet-Vn/files/run3auau/run3auau-merged-pro001_pcdb001_v001 \\
        --output-dir /direct/sphenix+u/anarde/Documents/sPHENIX/Jet-Vn/files/run3auau/run3auau-merged-missing-zdc-calib

Positional arguments are also supported:
    python3 find_missing_zdc_calib.py <zdc_dir> <merged_dir> [output_dir]
"""

import argparse
import concurrent.futures
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from tqdm import tqdm

DEFAULT_ZDC_DIR = "/sphenix/u/anarde/sEPD-Study/scratch/ZDC/07-01-26/lists"
DEFAULT_MERGED_DIR = (
    "/direct/sphenix+u/anarde/Documents/sPHENIX/Jet-Vn/files/run3auau/run3auau-merged-pro001_pcdb001_v001"
)

# Regex to extract run number from list filenames (e.g. dst_*-00067597.list)
RUN_FILENAME_PATTERN = re.compile(r"(\d+)\.list$")

# Regex to extract 5-digit (or any digits) segment number before .root
SEGMENT_PATTERN = re.compile(r"-(\d+)\.root")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Find segments in merged list files that lack corresponding ZDC calibration entries.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Positional or flag arguments
    parser.add_argument(
        "pos_zdc_dir",
        nargs="?",
        default=None,
        help="ZDC calibration directory (positional fallback)",
    )
    parser.add_argument(
        "pos_merged_dir",
        nargs="?",
        default=None,
        help="Merged files directory (positional fallback)",
    )
    parser.add_argument(
        "pos_output_dir",
        nargs="?",
        default=None,
        help="Output directory for missing lists (positional fallback)",
    )

    parser.add_argument(
        "-z",
        "--zdc-dir",
        type=str,
        default=DEFAULT_ZDC_DIR,
        help="Directory containing ZDC calibration list files",
    )
    parser.add_argument(
        "-m",
        "--merged-dir",
        type=str,
        default=DEFAULT_MERGED_DIR,
        help="Directory containing merged list files",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save the new per-run list files with missing ZDC calib segments (default: <merged_dir>-missing-zdc)",
    )
    parser.add_argument(
        "-j",
        "--workers",
        type=int,
        default=None,
        help="Number of parallel worker processes (default: min(os.cpu_count(), 16))",
    )
    parser.add_argument(
        "-r",
        "--runs",
        type=int,
        nargs="+",
        default=None,
        help="Optional: restrict checking to specific run number(s)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Scan and report missing segments without writing output files",
    )
    parser.add_argument(
        "--save-summary",
        type=str,
        default=None,
        help="Optional path to write a TSV summary of runs and missing segment counts",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print detailed per-run missing segment information to terminal",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress bar",
    )

    args = parser.parse_args()

    # Prioritize positional arguments if supplied
    if args.pos_zdc_dir:
        args.zdc_dir = args.pos_zdc_dir
    if args.pos_merged_dir:
        args.merged_dir = args.pos_merged_dir
    if args.pos_output_dir:
        args.output_dir = args.pos_output_dir

    if not args.output_dir and not args.dry_run:
        # Default output directory adjacent or suffixed
        args.output_dir = f"{args.merged_dir.rstrip('/')}-missing-zdc"

    return args


def print_table(headers: List[str], rows: List[List[str]]):
    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, val in enumerate(row):
            col_widths[i] = max(col_widths[i], len(str(val)))

    sep = "+-" + "-+-".join("-" * w for w in col_widths) + "-+"
    header_str = "| " + " | ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers)) + " |"

    print(sep)
    print(header_str)
    print(sep)
    for row in rows:
        line_str = "| " + " | ".join(str(val).ljust(col_widths[i]) for i, val in enumerate(row)) + " |"
        print(line_str)
    print(sep)


def index_zdc_directory(zdc_dir: Path) -> Dict[int, Path]:
    """
    Index all ZDC list files by integer run number.
    Returns:
        dict mapping run_number (int) -> file path (Path)
    """
    zdc_run_files: Dict[int, Path] = {}
    if not zdc_dir.is_dir():
        return zdc_run_files

    for entry in os.scandir(zdc_dir):
        if entry.is_file() and entry.name.endswith(".list"):
            match = RUN_FILENAME_PATTERN.search(entry.name)
            if match:
                run_num = int(match.group(1))
                zdc_run_files[run_num] = Path(entry.path)

    return zdc_run_files


def extract_zdc_segments(zdc_file_path: Path) -> Set[int]:
    """
    Parse a ZDC list file and extract the set of segment numbers.
    """
    segments: Set[int] = set()
    try:
        with open(zdc_file_path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                match = SEGMENT_PATTERN.search(line)
                if match:
                    segments.add(int(match.group(1)))
    except OSError as e:
        print(f"Warning: Could not read ZDC list file {zdc_file_path}: {e}", file=sys.stderr)
    return segments


def extract_segment_from_merged_line(line: str) -> Optional[int]:
    """
    Extract the segment integer from a line in a merged file.
    Merged lines are typically comma-separated:
      DST_CALOFITTING_...-00067597-00000.root,DST_ZDC_RAW_...-00067597-00000.root,...
    """
    first_part = line.split(",", 1)[0].strip()
    match = SEGMENT_PATTERN.search(first_part)
    if match:
        return int(match.group(1))
    # Fallback to scanning whole line if first part didn't match
    match = SEGMENT_PATTERN.search(line)
    if match:
        return int(match.group(1))
    return None


def process_single_run_worker(
    task: Tuple[int, str, Path, Optional[Path], Optional[Path]]
) -> Tuple[int, str, int, int, bool, Optional[str]]:
    """
    Worker function executed in parallel for each run.

    Args:
        task: (run_num, filename, merged_path, zdc_path, out_file_path)
              out_file_path is None during dry-run.

    Returns:
        (run_num, filename, total_segments, missing_count, has_zdc_list, error_msg)
    """
    run_num, filename, merged_path, zdc_path, out_file_path = task
    has_zdc_list = zdc_path is not None

    zdc_segments: Set[int] = set()
    if has_zdc_list and zdc_path is not None:
        zdc_segments = extract_zdc_segments(zdc_path)

    missing_lines: List[str] = []
    run_segment_count = 0

    try:
        with open(merged_path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line_clean = line.rstrip("\r\n")
                if not line_clean:
                    continue
                run_segment_count += 1
                seg_num = extract_segment_from_merged_line(line_clean)
                if seg_num is None:
                    missing_lines.append(line_clean)
                elif not has_zdc_list or seg_num not in zdc_segments:
                    missing_lines.append(line_clean)
    except OSError as e:
        return (run_num, filename, 0, 0, has_zdc_list, f"Read error on {merged_path}: {e}")

    missing_count = len(missing_lines)

    # If writing files is enabled and there are missing segments, write directly in worker
    if out_file_path is not None and missing_count > 0:
        try:
            with open(out_file_path, "w", encoding="utf-8") as f:
                for l in missing_lines:
                    f.write(f"{l}\n")
        except OSError as e:
            return (run_num, filename, run_segment_count, missing_count, has_zdc_list, f"Write error on {out_file_path}: {e}")

    return (run_num, filename, run_segment_count, missing_count, has_zdc_list, None)


def main():
    args = parse_args()
    start_time = time.time()

    zdc_dir = Path(args.zdc_dir).resolve()
    merged_dir = Path(args.merged_dir).resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else None

    num_workers = args.workers if args.workers is not None else min(os.cpu_count() or 4, 16)

    print("\n=======================================================")
    print(" ZDC Calibration Segment Coverage Checker")
    print("=======================================================")
    print(f"ZDC Calib Directory:  {zdc_dir}")
    print(f"Merged Directory:     {merged_dir}")
    if args.dry_run:
        print("Mode:                 DRY RUN (no output files will be written)")
    else:
        print(f"Output Directory:     {output_dir}")
    print(f"Workers:              {num_workers}")
    print("=======================================================\n")

    if not zdc_dir.is_dir():
        print(f"Error: ZDC calibration directory does not exist: {zdc_dir}", file=sys.stderr)
        sys.exit(2)

    if not merged_dir.is_dir():
        print(f"Error: Merged files directory does not exist: {merged_dir}", file=sys.stderr)
        sys.exit(2)

    # 1. Index ZDC list files
    print(f"Indexing ZDC calibration list files in {zdc_dir}...")
    zdc_run_files = index_zdc_directory(zdc_dir)
    print(f"Found {len(zdc_run_files)} ZDC calibration run list(s).\n")

    # 2. Collect merged list files
    print(f"Scanning merged files in {merged_dir}...")
    merged_files: List[Tuple[int, os.DirEntry]] = []
    for entry in os.scandir(merged_dir):
        if entry.is_file() and entry.name.endswith(".list"):
            match = RUN_FILENAME_PATTERN.search(entry.name)
            if match:
                run_num = int(match.group(1))
                if args.runs is None or run_num in args.runs:
                    merged_files.append((run_num, entry))

    merged_files.sort(key=lambda x: x[0])
    print(f"Found {len(merged_files)} merged run list(s) to process.\n")

    if not merged_files:
        print("No matching merged files found. Exiting.")
        sys.exit(0)

    # Prepare output directory if needed
    if not args.dry_run and output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

    # 3. Build tasks for parallel execution
    tasks: List[Tuple[int, str, Path, Optional[Path], Optional[Path]]] = []
    for run_num, entry in merged_files:
        zdc_path = zdc_run_files.get(run_num)
        out_file_path = (output_dir / entry.name) if (output_dir and not args.dry_run) else None
        tasks.append((run_num, entry.name, Path(entry.path), zdc_path, out_file_path))

    # 4. Execute parallel processing
    print(f"Analyzing segments across {len(tasks)} runs with {num_workers} parallel workers...")

    chunk_size = max(1, len(tasks) // (num_workers * 4)) if len(tasks) > 50 else 1

    if num_workers > 1 and len(tasks) > 1:
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
            results = list(
                tqdm(
                    executor.map(process_single_run_worker, tasks, chunksize=chunk_size),
                    total=len(tasks),
                    desc="Processing runs",
                    unit="run",
                    disable=args.no_progress,
                )
            )
    else:
        results = list(
            tqdm(
                (process_single_run_worker(t) for t in tasks),
                total=len(tasks),
                desc="Processing runs",
                unit="run",
                disable=args.no_progress,
            )
        )

    # 5. Aggregate metrics
    total_merged_segments = 0
    total_missing_segments = 0
    runs_with_missing = 0
    runs_completely_missing_zdc = 0
    runs_complete = 0
    errors: List[str] = []

    for run_num, filename, total_segs, missing_count, has_zdc_list, err in results:
        if err:
            errors.append(err)
            continue
        total_merged_segments += total_segs
        total_missing_segments += missing_count

        if not has_zdc_list:
            runs_completely_missing_zdc += 1

        if missing_count > 0:
            runs_with_missing += 1
        else:
            runs_complete += 1

    if errors:
        print(f"\n[WARNING] Encountered {len(errors)} error(s) during processing:", file=sys.stderr)
        for err in errors[:10]:
            print(f"  {err}", file=sys.stderr)
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more errors", file=sys.stderr)

    # 6. Save summary file if requested
    if args.save_summary:
        summary_path = Path(args.save_summary).resolve()
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write("run\tfilename\ttotal_segments\tmissing_segments\tstatus\n")
            for run_num, filename, total_segs, missing_count, has_zdc_list, err in results:
                if err:
                    status = "ERROR"
                elif not has_zdc_list:
                    status = "NO_ZDC_LIST"
                elif missing_count == 0:
                    status = "COMPLETE"
                else:
                    status = "PARTIAL_MISSING"
                f.write(f"{run_num}\t{filename}\t{total_segs}\t{missing_count}\t{status}\n")
        print(f"\nSaved run summary to: {summary_path}")

    elapsed_time = time.time() - start_time

    # 7. Print Summary Table
    pct_missing = (
        (total_missing_segments / total_merged_segments * 100) if total_merged_segments else 0.0
    )
    pct_runs_missing = (
        (runs_with_missing / len(merged_files) * 100) if merged_files else 0.0
    )

    headers = ["Metric", "Count", "Percentage"]
    rows = [
        ["Total Merged Runs Scanned", str(len(merged_files)), "100.00%"],
        ["Runs with ALL Segments in ZDC", str(runs_complete), f"{100 - pct_runs_missing:6.2f}%"],
        ["Runs Missing ANY ZDC Calib", str(runs_with_missing), f"{pct_runs_missing:6.2f}%"],
        ["  - Runs without any ZDC list file", str(runs_completely_missing_zdc), f"{(runs_completely_missing_zdc / len(merged_files) * 100):6.2f}%" if merged_files else "0.00%"],
        ["Total Merged Segments", str(total_merged_segments), "100.00%"],
        ["Segments Missing ZDC Calib", str(total_missing_segments), f"{pct_missing:6.2f}%"],
    ]

    print("\nSummary Statistics:")
    print_table(headers, rows)

    if not args.dry_run and output_dir:
        print(f"\nWrote {runs_with_missing} list file(s) with missing segments to: {output_dir}")

    # 8. Verbose listing of missing runs
    if args.verbose and runs_with_missing > 0:
        print("\nRuns with missing ZDC calibration segments:")
        for run_num, filename, total_segs, missing_count, has_zdc_list, err in results:
            if missing_count > 0:
                status_str = f"missing {missing_count}/{total_segs} segments"
                if not has_zdc_list:
                    status_str += " (NO ZDC LIST)"
                print(f"  Run {run_num:08d}: {status_str}")

    print(f"\nCompleted in {elapsed_time:.2f}s.\n")


if __name__ == "__main__":
    main()
