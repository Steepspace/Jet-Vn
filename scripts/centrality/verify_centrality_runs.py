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
import concurrent.futures
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import tqdm
import uproot


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
        "--check-scales",
        action="store_true",
        default=True,
        help="Inspect 'scales' directory ROOT files for Dcentralityscale branch (default: enabled)",
    )
    parser.add_argument(
        "--no-check-scales",
        dest="check_scales",
        action="store_false",
        help="Disable checking Dcentralityscale in scales files",
    )
    parser.add_argument(
        "--save-scales",
        type=str,
        default=None,
        help="Optional directory to save lists of good (scale=1), bad (scale=0), and other scale runs",
    )
    parser.add_argument(
        "--only-good-scales",
        action="store_true",
        help="When saving runs with --save-runs, filter to only include runs with scale == 1",
    )
    parser.add_argument(
        "-j",
        "--workers",
        type=int,
        default=None,
        help="Number of parallel worker processes for checking scale files (default: min(os.cpu_count(), 16))",
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


def inspect_single_scale_file(
    item: Tuple[int, str]
) -> Tuple[int, str, Optional[float], str]:
    """
    Worker function to inspect a single scale ROOT file.

    Returns:
        (run_number, status, scale_value, message)
        where status is 'GOOD', 'BAD', 'OTHER', 'EMPTY', or 'ERROR'.
    """
    run_num, file_path_str = item
    file_path = Path(file_path_str)
    try:
        if file_path.stat().st_size == 0:
            return run_num, "EMPTY", None, "0-byte file"

        with uproot.open(file_path) as f:
            if "Multiple" in f:
                tree = f["Multiple"]
            elif len(f.keys()) > 0:
                tree = f[f.keys()[0]]
            else:
                return run_num, "ERROR", None, "No keys/trees in ROOT file"

            if "Dcentralityscale" not in tree:
                return run_num, "ERROR", None, "Missing 'Dcentralityscale' branch"

            vals = tree["Dcentralityscale"].array(library="np")
            if len(vals) == 0:
                return run_num, "ERROR", None, "Branch 'Dcentralityscale' has 0 entries"

            if len(vals) > 1:
                unique_vals = np.unique(vals)
                if len(unique_vals) > 1:
                    return run_num, "OTHER", None, f"Multiple distinct scale values: {list(unique_vals)}"

            raw_val = vals[0]
            try:
                scale_val = float(raw_val)
            except (ValueError, TypeError):
                return run_num, "OTHER", None, f"Non-numeric scale value: {raw_val}"

            if np.isnan(scale_val):
                return run_num, "OTHER", scale_val, "NaN"
            elif scale_val == 1.0:
                return run_num, "GOOD", scale_val, ""
            elif scale_val == 0.0:
                return run_num, "BAD", scale_val, ""
            else:
                return run_num, "OTHER", scale_val, f"Scale = {scale_val}"
    except Exception as e:
        return run_num, "ERROR", None, str(e)


def verify_scales_directory(
    scales_dir: Path,
    scales_runs: Dict[int, str],
    workers: Optional[int] = None,
    verbose: bool = False,
    save_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Inspects all files in scales_dir for Dcentralityscale values and prints
    the breakdown of Good (1), Bad (0), Other, and error runs.
    """
    items = [(r, str(scales_dir / fname)) for r, fname in sorted(scales_runs.items())]
    if not items:
        print(f"\nNo scale files found in {scales_dir}.")
        return {}

    num_workers = workers if workers is not None else min(os.cpu_count() or 4, 16)
    print(f"\n=======================================================")
    print(f" Centrality Scale (Dcentralityscale) Verification")
    print(f"=======================================================")
    print(f"Scales Directory: {scales_dir}")
    print(f"Inspecting {len(items)} files with {num_workers} workers...")

    if len(items) > 20 and num_workers > 1:
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
            if len(items) > 50:
                results = list(
                    tqdm.tqdm(
                        executor.map(inspect_single_scale_file, items),
                        total=len(items),
                        desc="Checking scales",
                    )
                )
            else:
                results = list(executor.map(inspect_single_scale_file, items))
    else:
        results = [inspect_single_scale_file(it) for it in items]

    good_runs: List[int] = []
    bad_runs: List[int] = []
    other_runs: Dict[int, Tuple[Optional[float], str]] = {}
    error_runs: Dict[int, str] = {}

    for run_num, status, val, msg in results:
        if status == "GOOD":
            good_runs.append(run_num)
        elif status == "BAD":
            bad_runs.append(run_num)
        elif status == "OTHER":
            other_runs[run_num] = (val, msg)
        else:
            error_runs[run_num] = msg

    total = len(items)
    pct_good = (len(good_runs) / total * 100) if total else 0.0
    pct_bad = (len(bad_runs) / total * 100) if total else 0.0
    pct_other = (len(other_runs) / total * 100) if total else 0.0
    pct_err = (len(error_runs) / total * 100) if total else 0.0

    headers = ["Classification", "Dcentralityscale", "Count", "Percentage"]
    rows = [
        ["Good", "1", str(len(good_runs)), f"{pct_good:6.2f}%"],
        ["Bad", "0", str(len(bad_runs)), f"{pct_bad:6.2f}%"],
        ["Other", "!= 0 and != 1", str(len(other_runs)), f"{pct_other:6.2f}%"],
        ["Unreadable / Error", "N/A", str(len(error_runs)), f"{pct_err:6.2f}%"],
    ]

    print("\nScale Breakdown:")
    print_table(headers, rows)

    # Highlight runs with neither 1 nor 0
    if other_runs:
        print(f"\n[ALERT] Found {len(other_runs)} run(s) with scale NEITHER 1 NOR 0:")
        for r, (val, msg) in sorted(other_runs.items()):
            print(f"    Run {r}: {msg}")
    else:
        print("\n[OK] No runs found with scale neither 1 nor 0.")

    if error_runs:
        print(f"\n[WARNING] Found {len(error_runs)} run(s) with errors reading Dcentralityscale:")
        for r, msg in sorted(error_runs.items())[:20]:
            print(f"    Run {r}: {msg}")
        if len(error_runs) > 20 and not verbose:
            print(f"    ... and {len(error_runs) - 20} more (use -v to display all)")

    # Print summary of Good vs Bad
    print(f"\nScale Summary:")
    print(f"  - Good Runs (Scale = 1): {len(good_runs)}")
    if verbose or len(good_runs) <= 20:
        if good_runs:
            print(f"    {good_runs}")
    else:
        print(f"    First 10: {good_runs[:10]}")
        print(f"    Last 10:  {good_runs[-10:]}")
        print(f"    (Use -v or --verbose to display all)")

    print(f"  - Bad Runs (Scale = 0): {len(bad_runs)}")
    if verbose or len(bad_runs) <= 20:
        if bad_runs:
            print(f"    {bad_runs}")
    else:
        print(f"    First 10: {bad_runs[:10]}")
        print(f"    Last 10:  {bad_runs[-10:]}")
        print(f"    (Use -v or --verbose to display all)")

    # Save scale lists if requested
    if save_dir:
        out_dir = Path(save_dir).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)

        good_file = out_dir / "good_scale_runs.list"
        with open(good_file, "w") as f:
            for r in good_runs:
                f.write(f"{r}\n")
        print(f"\nSaved {len(good_runs)} good scale runs to: {good_file}")

        bad_file = out_dir / "bad_scale_runs.list"
        with open(bad_file, "w") as f:
            for r in bad_runs:
                f.write(f"{r}\n")
        print(f"Saved {len(bad_runs)} bad scale runs to: {bad_file}")

        if other_runs:
            other_file = out_dir / "other_scale_runs.list"
            with open(other_file, "w") as f:
                for r, (val, msg) in sorted(other_runs.items()):
                    f.write(f"{r} {msg}\n")
            print(f"Saved {len(other_runs)} other scale runs to: {other_file}")

    return {
        "good": good_runs,
        "bad": bad_runs,
        "other": other_runs,
        "error": error_runs,
    }


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

    # Scales inspection (Dcentralityscale branch verification)
    scale_results: Dict[str, Any] = {}
    if args.check_scales:
        scales_dir_path = None
        scales_runs: Dict[int, str] = {}

        if "scales" in dir_runs:
            scales_dir_path = base_dir / "scales"
            scales_runs = dir_runs["scales"]
        elif (base_dir / "scales").is_dir():
            scales_dir_path = base_dir / "scales"
            scales_runs, _, _ = scan_directory(scales_dir_path, check_empty=args.check_empty)

        if scales_dir_path and scales_runs:
            scale_results = verify_scales_directory(
                scales_dir=scales_dir_path,
                scales_runs=scales_runs,
                workers=args.workers,
                verbose=args.verbose,
                save_dir=args.save_scales,
            )
        elif not scales_dir_path and args.verbose:
            print("\nNote: 'scales' directory not found; skipping Dcentralityscale inspection.")

    # Export options
    if args.save_runs:
        out_path = Path(args.save_runs).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        runs_to_save = intersection_runs
        if args.only_good_scales and scale_results.get("good") is not None:
            runs_to_save = runs_to_save & set(scale_results["good"])
            print(f"Filtered --save-runs to only good scale (scale=1) runs: {len(runs_to_save)} runs.")

        with open(out_path, "w") as f:
            for r in sorted(runs_to_save):
                f.write(f"{r}\n")
        print(f"\nSaved {len(runs_to_save)} matching runs to: {out_path}")

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

    if scale_results.get("other"):
        has_discrepancy = True
    if scale_results.get("error"):
        has_discrepancy = True

    print("\nVerification completed.\n")
    sys.exit(1 if has_discrepancy else 0)


if __name__ == "__main__":
    main()
