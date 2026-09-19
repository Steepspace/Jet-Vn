#!/usr/bin/env python3
"""
Find and Filter Segments Missing ZDC Calibration

This script compares merged list files against ZDC calibration list files.
For each run in the merged directory, it checks whether each segment has a
corresponding ZDC calibration segment.

Features:
1. Identify missing calibration segments:
   If any segments in a run are missing ZDC calibration, it writes a new list file
   (one per run) in the specified output directory containing only the lines from
   the merged list that need ZDC calibration.

2. Clean up extra ZDC calibration segments (--remove-extra-zdc):
   If there are segments in the ZDC calibration lists that are not present in
   the merged directory, this option deletes those extra ROOT files from disk
   and updates (or removes) the corresponding ZDC calibration list files to
   reclaim storage space.

Processing is parallelized using concurrent.futures.ProcessPoolExecutor.

Example usage:
    # Check for missing segments and write lists
    python3 find_missing_zdc_calib.py \\
        --zdc-dir /sphenix/u/anarde/sEPD-Study/scratch/ZDC/07-01-26/lists \\
        --merged-dir /direct/sphenix+u/anarde/Documents/sPHENIX/Jet-Vn/files/run3auau/run3auau-merged-pro001_pcdb001_v001 \\
        --output-dir /direct/sphenix+u/anarde/Documents/sPHENIX/Jet-Vn/files/run3auau/run3auau-merged-missing-zdc-calib

    # Dry-run check showing missing and extra segments without making any changes
    python3 find_missing_zdc_calib.py --dry-run --remove-extra-zdc

    # Remove extra ZDC calibration files and update ZDC lists
    python3 find_missing_zdc_calib.py --remove-extra-zdc
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

DEFAULT_ZDC_DIR = "/sphenix/u/anarde/sEPD-Study/scratch/ZDC/lists"
DEFAULT_MERGED_DIR = (
    "/direct/sphenix+u/anarde/Documents/sPHENIX/Jet-Vn/files/run3auau/run3auau-merged-pro001_pcdb001_v001"
)

# Regex to extract run number from list filenames (e.g. dst_*-00067597.list)
RUN_FILENAME_PATTERN = re.compile(r"(\d+)\.list$")

# Regex to extract 5-digit (or any digits) segment number before .root
SEGMENT_PATTERN = re.compile(r"-(\d+)\.root")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Find segments in merged list files that lack corresponding ZDC calibration entries, "
                    "with option to remove extra ZDC calibration segments.",
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
        "--remove-extra-zdc",
        "--clean-extra-zdc",
        dest="remove_extra_zdc",
        action="store_true",
        help="Remove segments from ZDC calib lists and delete their actual ROOT files from disk if not in merged dir",
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
        help="Scan and report missing and extra segments without writing or deleting any files",
    )
    parser.add_argument(
        "--save-summary",
        type=str,
        default=None,
        help="Optional path to write a TSV summary of runs and missing/extra segment counts",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print detailed per-run missing and extra segment information to terminal",
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
        args.output_dir = f"{args.merged_dir.rstrip('/')}-missing-zdc"

    return args


def format_bytes(size_bytes: int) -> str:
    """Format bytes into a human readable string."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.2f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.2f} MB"
    else:
        return f"{size_bytes / (1024 * 1024 * 1024):.2f} GB"


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


def extract_segment_from_line(line: str) -> Optional[int]:
    """
    Extract the segment integer from a line.
    Handles comma-separated DST files or single file paths.
    """
    first_part = line.split(",", 1)[0].strip()
    match = SEGMENT_PATTERN.search(first_part)
    if match:
        return int(match.group(1))
    match = SEGMENT_PATTERN.search(line)
    if match:
        return int(match.group(1))
    return None


def resolve_file_path(line_path: str, parent_dir: Optional[Path] = None) -> Optional[Path]:
    """Resolve file path, handling absolute paths and relative paths."""
    p = Path(line_path.strip())
    if p.exists():
        return p
    if parent_dir is not None:
        cand = parent_dir / p.name
        if cand.exists():
            return cand
    return p


def process_single_run_worker(
    task: Tuple[int, str, Optional[Path], Optional[Path], Optional[Path], bool, bool]
) -> Tuple[int, str, int, int, bool, int, int, int, int, bool, bool, Optional[str]]:
    """
    Worker function executed in parallel for each run.

    Args:
        task: (
            run_num, filename, merged_path, zdc_path, out_file_path,
            remove_extra_zdc, dry_run
        )

    Returns:
        (
            run_num, filename,
            merged_segments_count, missing_merged_count, has_zdc_list,
            zdc_segments_count, extra_zdc_count,
            deleted_root_files_count, reclaimed_bytes,
            zdc_list_updated, zdc_list_deleted,
            error_msg
        )
    """
    (
        run_num,
        filename,
        merged_path,
        zdc_path,
        out_file_path,
        remove_extra_zdc,
        dry_run,
    ) = task

    has_merged_list = merged_path is not None
    has_zdc_list = zdc_path is not None

    # 1. Read Merged List
    merged_segments: Set[int] = set()
    merged_lines: List[Tuple[Optional[int], str]] = []
    if has_merged_list and merged_path is not None:
        try:
            with open(merged_path, "r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    line_clean = line.rstrip("\r\n")
                    if not line_clean:
                        continue
                    seg_num = extract_segment_from_line(line_clean)
                    if seg_num is not None:
                        merged_segments.add(seg_num)
                    merged_lines.append((seg_num, line_clean))
        except OSError as e:
            return (
                run_num, filename, 0, 0, has_zdc_list, 0, 0, 0, 0, False, False,
                f"Read error on merged file {merged_path}: {e}"
            )

    # 2. Read ZDC List
    zdc_segments: Set[int] = set()
    zdc_lines: List[Tuple[Optional[int], str]] = []
    if has_zdc_list and zdc_path is not None:
        try:
            with open(zdc_path, "r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    line_clean = line.rstrip("\r\n")
                    if not line_clean:
                        continue
                    seg_num = extract_segment_from_line(line_clean)
                    if seg_num is not None:
                        zdc_segments.add(seg_num)
                    zdc_lines.append((seg_num, line_clean))
        except OSError as e:
            return (
                run_num, filename, len(merged_lines), 0, has_zdc_list, 0, 0, 0, 0, False, False,
                f"Read error on ZDC file {zdc_path}: {e}"
            )

    # 3. Determine Missing Merged Segments
    missing_merged_lines: List[str] = []
    for seg_num, line_str in merged_lines:
        if not has_zdc_list or seg_num is None or seg_num not in zdc_segments:
            missing_merged_lines.append(line_str)

    missing_merged_count = len(missing_merged_lines)

    # Write missing merged list if requested
    if out_file_path is not None and missing_merged_count > 0:
        try:
            out_file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_file_path, "w", encoding="utf-8") as f:
                for l in missing_merged_lines:
                    f.write(f"{l}\n")
        except OSError as e:
            return (
                run_num, filename, len(merged_lines), missing_merged_count, has_zdc_list,
                len(zdc_lines), 0, 0, 0, False, False,
                f"Write error on {out_file_path}: {e}"
            )

    # 4. Determine Extra ZDC Segments (not in merged list)
    extra_zdc_lines: List[str] = []
    keep_zdc_lines: List[str] = []

    for seg_num, line_str in zdc_lines:
        if seg_num is None or seg_num not in merged_segments:
            extra_zdc_lines.append(line_str)
        else:
            keep_zdc_lines.append(line_str)

    extra_zdc_count = len(extra_zdc_lines)
    deleted_root_files_count = 0
    reclaimed_bytes = 0
    zdc_list_updated = False
    zdc_list_deleted = False

    # 5. Clean up Extra ZDC Segments if enabled
    if remove_extra_zdc and extra_zdc_count > 0 and zdc_path is not None:
        for line_str in extra_zdc_lines:
            file_path = resolve_file_path(line_str, zdc_path.parent)
            if file_path and file_path.exists():
                try:
                    size = file_path.stat().st_size
                    reclaimed_bytes += size
                    if not dry_run:
                        file_path.unlink()
                    deleted_root_files_count += 1
                except OSError as e:
                    # Continue deleting other files even if one encounters an error
                    pass
            else:
                # File already does not exist on disk
                deleted_root_files_count += 1

        if not dry_run:
            try:
                if keep_zdc_lines:
                    # Atomically update ZDC list file with remaining lines
                    temp_zdc_path = zdc_path.with_suffix(".tmp_zdc_clean")
                    with open(temp_zdc_path, "w", encoding="utf-8") as f:
                        for l in keep_zdc_lines:
                            f.write(f"{l}\n")
                    temp_zdc_path.replace(zdc_path)
                    zdc_list_updated = True
                else:
                    # All segments were removed: unlink empty list file
                    zdc_path.unlink(missing_ok=True)
                    zdc_list_deleted = True
            except OSError as e:
                return (
                    run_num, filename, len(merged_lines), missing_merged_count, has_zdc_list,
                    len(zdc_lines), extra_zdc_count, deleted_root_files_count, reclaimed_bytes,
                    False, False, f"Failed updating ZDC list {zdc_path}: {e}"
                )
        else:
            if keep_zdc_lines:
                zdc_list_updated = True
            else:
                zdc_list_deleted = True

    return (
        run_num,
        filename,
        len(merged_lines),
        missing_merged_count,
        has_zdc_list,
        len(zdc_lines),
        extra_zdc_count,
        deleted_root_files_count,
        reclaimed_bytes,
        zdc_list_updated,
        zdc_list_deleted,
        None,
    )


def main():
    args = parse_args()
    start_time = time.time()

    zdc_dir = Path(args.zdc_dir).resolve()
    merged_dir = Path(args.merged_dir).resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else None

    num_workers = args.workers if args.workers is not None else min(os.cpu_count() or 4, 16)

    print("\n=======================================================")
    print(" ZDC Calibration Segment Coverage & Cleanup Utility")
    print("=======================================================")
    print(f"ZDC Calib Directory:  {zdc_dir}")
    print(f"Merged Directory:     {merged_dir}")
    if args.dry_run:
        print("Mode:                 DRY RUN (no output files or deletions performed)")
    else:
        print(f"Output Directory:     {output_dir}")
    if args.remove_extra_zdc:
        action_text = "WILL DELETE" if not args.dry_run else "WOULD DELETE (dry-run)"
        print(f"Extra ZDC Cleanup:    ENABLED ({action_text} extra ROOT files & update ZDC lists)")
    else:
        print("Extra ZDC Cleanup:    DISABLED (use --remove-extra-zdc to purge extra files)")
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
    merged_run_files: Dict[int, os.DirEntry] = {}
    for entry in os.scandir(merged_dir):
        if entry.is_file() and entry.name.endswith(".list"):
            match = RUN_FILENAME_PATTERN.search(entry.name)
            if match:
                run_num = int(match.group(1))
                merged_run_files[run_num] = entry

    print(f"Found {len(merged_run_files)} merged run list(s).\n")

    # 3. Determine candidate runs to process
    all_runs = set(merged_run_files.keys())
    if args.remove_extra_zdc:
        # If removing extra ZDC segments, also include any runs present in ZDC lists
        all_runs = all_runs.union(zdc_run_files.keys())

    if args.runs is not None:
        target_runs = sorted([r for r in all_runs if r in args.runs])
    else:
        target_runs = sorted(all_runs)

    if not target_runs:
        print("No matching runs to process. Exiting.")
        sys.exit(0)

    # 4. Build tasks for parallel execution
    tasks = []
    for run_num in target_runs:
        merged_entry = merged_run_files.get(run_num)
        merged_path = Path(merged_entry.path) if merged_entry else None
        zdc_path = zdc_run_files.get(run_num)

        if merged_entry:
            filename = merged_entry.name
        elif zdc_path:
            filename = zdc_path.name.replace("dst_zdc_calib-", "dst_calofitting_zdc_sepd-")
        else:
            filename = f"dst_calofitting_zdc_sepd-{run_num:08d}.list"

        out_file_path = (output_dir / filename) if (output_dir and not args.dry_run) else None

        tasks.append((
            run_num,
            filename,
            merged_path,
            zdc_path,
            out_file_path,
            args.remove_extra_zdc,
            args.dry_run,
        ))

    # 5. Execute parallel processing
    print(f"Analyzing {len(tasks)} run(s) across {num_workers} parallel workers...")

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

    # 6. Aggregate metrics
    total_merged_segments = 0
    total_missing_merged_segments = 0
    runs_with_missing_merged = 0
    runs_completely_missing_zdc = 0
    runs_complete = 0

    total_zdc_segments = 0
    total_extra_zdc_segments = 0
    runs_with_extra_zdc = 0

    total_deleted_files = 0
    total_reclaimed_bytes = 0
    total_zdc_lists_updated = 0
    total_zdc_lists_deleted = 0

    errors: List[str] = []

    for (
        run_num,
        filename,
        merged_segs,
        missing_count,
        has_zdc,
        zdc_segs,
        extra_count,
        deleted_count,
        reclaimed,
        updated_zdc,
        deleted_zdc,
        err,
    ) in results:
        if err:
            errors.append(err)
            continue

        total_merged_segments += merged_segs
        total_missing_merged_segments += missing_count

        total_zdc_segments += zdc_segs
        total_extra_zdc_segments += extra_count

        total_deleted_files += deleted_count
        total_reclaimed_bytes += reclaimed
        if updated_zdc:
            total_zdc_lists_updated += 1
        if deleted_zdc:
            total_zdc_lists_deleted += 1

        if not has_zdc and merged_segs > 0:
            runs_completely_missing_zdc += 1

        if missing_count > 0:
            runs_with_missing_merged += 1
        elif merged_segs > 0:
            runs_complete += 1

        if extra_count > 0:
            runs_with_extra_zdc += 1

    if errors:
        print(f"\n[WARNING] Encountered {len(errors)} error(s) during processing:", file=sys.stderr)
        for err in errors[:10]:
            print(f"  {err}", file=sys.stderr)
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more errors", file=sys.stderr)

    # 7. Save summary file if requested
    if args.save_summary:
        summary_path = Path(args.save_summary).resolve()
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write("run\tfilename\ttotal_merged\tmissing_in_zdc\ttotal_zdc\textra_in_zdc\tstatus\n")
            for (
                run_num, filename, merged_segs, missing_count, has_zdc,
                zdc_segs, extra_count, deleted_count, reclaimed, _, _, err
            ) in results:
                if err:
                    status = "ERROR"
                elif not has_zdc:
                    status = "NO_ZDC_LIST"
                elif missing_count == 0 and extra_count == 0:
                    status = "MATCH_PERFECT"
                elif missing_count > 0 and extra_count == 0:
                    status = "MISSING_ZDC"
                elif missing_count == 0 and extra_count > 0:
                    status = "EXTRA_ZDC"
                else:
                    status = "DIFF_BOTH"
                f.write(
                    f"{run_num}\t{filename}\t{merged_segs}\t{missing_count}\t"
                    f"{zdc_segs}\t{extra_count}\t{status}\n"
                )
        print(f"\nSaved run summary to: {summary_path}")

    elapsed_time = time.time() - start_time

    # 8. Print Summary Tables
    pct_missing = (
        (total_missing_merged_segments / total_merged_segments * 100) if total_merged_segments else 0.0
    )
    pct_runs_missing = (
        (runs_with_missing_merged / len(target_runs) * 100) if target_runs else 0.0
    )
    pct_extra_zdc = (
        (total_extra_zdc_segments / total_zdc_segments * 100) if total_zdc_segments else 0.0
    )

    headers = ["Coverage Metric", "Count", "Percentage"]
    rows = [
        ["Total Runs Evaluated", str(len(target_runs)), "100.00%"],
        ["Runs with ALL Segments in ZDC", str(runs_complete), f"{100 - pct_runs_missing:6.2f}%"],
        ["Runs Missing ANY ZDC Calib", str(runs_with_missing_merged), f"{pct_runs_missing:6.2f}%"],
        ["  - Runs without any ZDC list file", str(runs_completely_missing_zdc), f"{(runs_completely_missing_zdc / len(target_runs) * 100):6.2f}%" if target_runs else "0.00%"],
        ["Total Merged Segments", str(total_merged_segments), "100.00%"],
        ["Merged Segments Missing ZDC Calib", str(total_missing_merged_segments), f"{pct_missing:6.2f}%"],
        ["------------------------------------", "-------", "----------"],
        ["Total ZDC Calib Segments Scanned", str(total_zdc_segments), "100.00%"],
        ["Runs with Extra ZDC Segments", str(runs_with_extra_zdc), f"{(runs_with_extra_zdc / len(target_runs) * 100):6.2f}%" if target_runs else "0.00%"],
        ["Extra ZDC Segments (Not in Merged)", str(total_extra_zdc_segments), f"{pct_extra_zdc:6.2f}%"],
    ]

    print("\nSummary Statistics:")
    print_table(headers, rows)

    # Cleanup table if extra cleanup was evaluated
    if args.remove_extra_zdc or total_extra_zdc_segments > 0:
        action_label = "Cleaned (Purged)" if (args.remove_extra_zdc and not args.dry_run) else "Identified for Cleanup"
        clean_headers = ["ZDC Cleanup Metric", "Count / Value"]
        clean_rows = [
            [f"Extra ROOT Files {action_label}", str(total_deleted_files)],
            [f"Disk Space {'Freed' if (args.remove_extra_zdc and not args.dry_run) else 'Reclaimable'}", format_bytes(total_reclaimed_bytes)],
            [f"ZDC List Files Updated", str(total_zdc_lists_updated)],
            [f"ZDC List Files Removed (Empty)", str(total_zdc_lists_deleted)],
        ]
        print(f"\nExtra ZDC Cleanup Summary ({'DRY RUN - No changes made' if args.dry_run else ('ACTIVE' if args.remove_extra_zdc else 'REPORT ONLY - Run with --remove-extra-zdc to purge')}):")
        print_table(clean_headers, clean_rows)

    if not args.dry_run and output_dir:
        if runs_with_missing_merged > 0:
            print(f"\nWrote {runs_with_missing_merged} list file(s) with missing segments to: {output_dir}")
        else:
            print(f"\nNo missing segments found. Output directory was not created.")

    # 9. Verbose listing
    if args.verbose:
        if runs_with_missing_merged > 0:
            print("\nRuns with missing ZDC calibration segments:")
            for r in results:
                run_num, filename, merged_segs, missing_count, has_zdc = r[0], r[1], r[2], r[3], r[4]
                if missing_count > 0:
                    status_str = f"missing {missing_count}/{merged_segs} segments"
                    if not has_zdc:
                        status_str += " (NO ZDC LIST)"
                    print(f"  Run {run_num:08d}: {status_str}")

        if runs_with_extra_zdc > 0:
            print("\nRuns with extra ZDC calibration segments (not in merged dir):")
            for r in results:
                run_num, filename, zdc_segs, extra_count = r[0], r[1], r[5], r[6]
                if extra_count > 0:
                    print(f"  Run {run_num:08d}: {extra_count}/{zdc_segs} extra segments")

    print(f"\nCompleted in {elapsed_time:.2f}s.\n")


if __name__ == "__main__":
    main()
