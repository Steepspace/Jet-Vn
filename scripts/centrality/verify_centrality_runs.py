#!/usr/bin/env python3
"""
Verify Centrality Calibration Runs

This script inspects the subdirectories (e.g. divs, scales, vertexscales) in a
centrality calibration directory, extracts the run numbers from the root files,
and verifies whether all directories contain identical runs or if any directory
has missing, extra, or non-matching runs.

Default calibration directory:
    /sphenix/user/anarde/sEPD-Study/centrality_calib
"""

import argparse
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple


DEFAULT_CALIB_DIR = "/sphenix/user/anarde/sEPD-Study/centrality_calib"
DEFAULT_SUBDIRS = ["divs", "scales", "vertexscales"]
RUN_PATTERN = re.compile(r"(\d+)")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Verify run consistency across centrality calibration directories."
    )
    parser.add_argument(
        "-d",
        "--calib-dir",
        type=str,
        default=DEFAULT_CALIB_DIR,
        help=f"Base calibration directory (default: {DEFAULT_CALIB_DIR})",
    )
    parser.add_argument(
        "--dirs",
        nargs="+",
        default=None,
        help="Specific subdirectories to compare (defaults to all subdirectories or standard divs/scales/vertexscales)",
    )
    parser.add_argument(
        "--check-empty",
        action="store_true",
        default=True,
        help="Check for 0-byte or empty files (default: enabled)",
    )
    parser.add_argument(
        "--save-runs",
        type=str,
        default=None,
        help="Optional path to export the list of common valid runs (one per line)",
    )
    parser.add_argument(
        "--save-diff",
        type=str,
        default=None,
        help="Optional directory to save lists of missing/extra runs for each subdir",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Display full lists of missing/extra runs in the terminal output",
    )
    return parser.parse_args()


def scan_directory(
    dir_path: Path, check_empty: bool = True
) -> Tuple[Dict[int, str], List[str], List[str]]:
    """
    Scans a directory for calibration files.

    Returns:
        runs: dict mapping run_number -> filename
        non_matching_files: list of filenames that did not match the run pattern
        empty_files: list of filenames that have 0 bytes
    """
    runs: Dict[int, str] = {}
    non_matching_files: List[str] = []
    empty_files: List[str] = []

    if not dir_path.is_dir():
        return runs, non_matching_files, empty_files

    for entry in sorted(os.listdir(dir_path)):
        full_path = dir_path / entry
        if not full_path.is_file():
            continue

        match = RUN_PATTERN.search(entry)
        if match:
            run_num = int(match.group(1))
            runs[run_num] = entry
        else:
            non_matching_files.append(entry)

        if check_empty:
            try:
                if full_path.stat().st_size == 0:
                    empty_files.append(entry)
            except OSError:
                empty_files.append(entry)

    return runs, non_matching_files, empty_files


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


def main():
    args = parse_args()
    base_dir = Path(args.calib_dir).resolve()

    if not base_dir.exists():
        print(f"Error: Calibration directory does not exist: {base_dir}", file=sys.stderr)
        sys.exit(2)

    if args.dirs:
        subdirs = args.dirs
    else:
        # Detect subdirectories
        detected = [d.name for d in base_dir.iterdir() if d.is_dir()]
        # If standard subdirs exist, prefer them or use all detected subdirs
        if set(DEFAULT_SUBDIRS).issubset(set(detected)):
            subdirs = DEFAULT_SUBDIRS
        else:
            subdirs = sorted(detected)

    if not subdirs:
        print(f"Error: No subdirectories found in {base_dir}", file=sys.stderr)
        sys.exit(2)

    print(f"\n=======================================================")
    print(f" Centrality Calibration Run Consistency Verification")
    print(f"=======================================================")
    print(f"Base Directory: {base_dir}")
    print(f"Subdirectories: {', '.join(subdirs)}\n")

    dir_runs: Dict[str, Dict[int, str]] = {}
    dir_non_matching: Dict[str, List[str]] = {}
    dir_empty: Dict[str, List[str]] = {}

    for d in subdirs:
        sub_path = base_dir / d
        runs, non_match, empty = scan_directory(sub_path, check_empty=args.check_empty)
        dir_runs[d] = runs
        dir_non_matching[d] = non_match
        dir_empty[d] = empty

    all_run_sets = [set(r.keys()) for r in dir_runs.values()]
    union_runs = set.union(*all_run_sets) if all_run_sets else set()
    intersection_runs = set.intersection(*all_run_sets) if all_run_sets else set()

    # Per-directory summary table
    headers = [
        "Directory",
        "Files",
        "Unique Runs",
        "Missing Runs",
        "Extra Runs",
        "Invalid Names",
        "Empty Files",
    ]
    rows = []

    mismatches_found = False

    for d in subdirs:
        runs_set = set(dir_runs[d].keys())
        missing = union_runs - runs_set
        # Extra runs compared to the intersection
        extra = runs_set - intersection_runs
        non_match_count = len(dir_non_matching[d])
        empty_count = len(dir_empty[d])

        if missing or extra or non_match_count > 0 or empty_count > 0:
            mismatches_found = True

        rows.append([
            d,
            str(len(dir_runs[d]) + non_match_count),
            str(len(runs_set)),
            str(len(missing)),
            str(len(extra)),
            str(non_match_count),
            str(empty_count),
        ])

    print_table(headers, rows)

    # Detailed statistics
    print(f"\nGlobal Run Statistics:")
    print(f"  - Total unique runs across all directories (Union): {len(union_runs)}")
    print(f"  - Runs present in EVERY directory (Intersection):   {len(intersection_runs)}")
    if union_runs:
        sorted_union = sorted(union_runs)
        print(f"  - Run Range: {sorted_union[0]} -> {sorted_union[-1]}")

    # Check consistency
    has_discrepancy = len(union_runs) != len(intersection_runs) or mismatches_found

    if not has_discrepancy:
        print("\n[PASSED] PERFECT MATCH: All subdirectories contain identical runs with valid files.")
    else:
        print("\n[WARNING] DISCREPANCIES DETECTED across directories:")
        for d in subdirs:
            runs_set = set(dir_runs[d].keys())
            missing = sorted(union_runs - runs_set)
            extra = sorted(runs_set - intersection_runs)

            if missing:
                print(f"\n  Subdirectory '{d}' is missing {len(missing)} run(s):")
                if args.verbose or len(missing) <= 20:
                    print(f"    {missing}")
                else:
                    print(f"    First 10: {missing[:10]}")
                    print(f"    Last 10:  {missing[-10:]}")
                    print(f"    (Use -v or --verbose to see all)")

            if extra and len(subdirs) > 2:
                print(f"\n  Subdirectory '{d}' has {len(extra)} extra run(s) not in all other dirs:")
                if args.verbose or len(extra) <= 20:
                    print(f"    {extra}")
                else:
                    print(f"    First 10: {extra[:10]}")
                    print(f"    Last 10:  {extra[-10:]}")

            if dir_non_matching[d]:
                print(f"\n  Subdirectory '{d}' has {len(dir_non_matching[d])} unrecognized file(s):")
                print(f"    {dir_non_matching[d][:10]}")

            if dir_empty[d]:
                print(f"\n  Subdirectory '{d}' has {len(dir_empty[d])} 0-byte file(s):")
                print(f"    {dir_empty[d][:10]}")

    # Export options
    if args.save_runs:
        out_path = Path(args.save_runs).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            for r in sorted(intersection_runs):
                f.write(f"{r}\n")
        print(f"\nSaved {len(intersection_runs)} matching runs to: {out_path}")

    if args.save_diff and has_discrepancy:
        diff_dir = Path(args.save_diff).resolve()
        diff_dir.mkdir(parents=True, exist_ok=True)
        for d in subdirs:
            runs_set = set(dir_runs[d].keys())
            missing = sorted(union_runs - runs_set)
            if missing:
                missing_file = diff_dir / f"missing_runs_{d}.list"
                with open(missing_file, "w") as f:
                    for r in missing:
                        f.write(f"{r}\n")
                print(f"Saved missing runs for '{d}' to: {missing_file}")

    print("\nVerification completed.\n")
    sys.exit(1 if has_discrepancy else 0)


if __name__ == "__main__":
    main()
