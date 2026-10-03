#!/usr/bin/env python3
"""
Merge Calofitting, ZDC Calibration, and sEPD Calibration Lists
with Coverage Analysis, Missing Calibration List Generation, and Extra Files Cleanup

This script:
1. Merges calorimeter DST segments (`dst_calofitting-*.list`) against ZDC calibration
   and sEPD calibration ROOT files, producing one merged list file per run where all
   segments match:
       <dst_calofitting>,<dst_zdc_calib_full_path>,<dst_sepd_calib_full_path>

2. Identifies segments missing ZDC and/or sEPD calibration (similar to find_missing_zdc_calib.py):
   Optionally writes per-run list files for missing ZDC segments (--write-missing / --missing-zdc-dir)
   and missing sEPD segments (--missing-sepd-dir) using raw DST lines from `dst_zdc_raw-*.list`
   and `dst_sepd_raw-*.list`.

3. Detects and cleans up extra calibration segments:
   Options `--remove-extra-zdc`, `--remove-extra-sepd`, and `--remove-extra-all` delete extra ROOT
   files from disk that are not present in the input raw lists, reclaiming storage.

Features:
- Parallelized per-run processing using `concurrent.futures.ProcessPoolExecutor`.
- Direct discovery of calibration files from run subdirectories or list files.
- Handles segment matching regardless of zero-padding variations.
- Deterministic numerical sorting of segment rows in each output list.
- Comprehensive coverage reporting and summary tables.
- Dry-run mode, selective run filtering, and TSV summary export.

Usage:
    # Run with default directories (creates merged lists):
    python3 merge_lists.py

    # Check calibration coverage in dry-run mode without writing any files:
    python3 merge_lists.py --dry-run

    # Generate merged lists AND output lists of segments missing ZDC and sEPD calibration:
    python3 merge_lists.py --write-missing

    # Custom directories for missing calibration lists:
    python3 merge_lists.py \
        --missing-zdc-dir /path/to/missing-zdc \
        --missing-sepd-dir /path/to/missing-sepd

    # Dry-run check for extra calibration files without deleting:
    python3 merge_lists.py --dry-run --remove-extra-all

    # Purge extra ZDC and sEPD calibration ROOT files from disk:
    python3 merge_lists.py --remove-extra-all

    # Restrict processing to runs in a run list file (one run per line):
    python3 merge_lists.py --run-list /path/to/runs-centrality-calibs.list

    # Exclude corrupt segments from consideration:
    python3 merge_lists.py --corrupt-list /path/to/corrupt.list

    # Check calibration coverage for runs in a run list excluding corrupt segments in dry-run mode:
    python3 merge_lists.py \
        --run-list /path/to/runs-centrality-calibs.list \
        --corrupt-list /path/to/corrupt.list \
        --dry-run
"""

import argparse
import concurrent.futures
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

try:
    from tqdm import tqdm
except ImportError:
    # Fallback if tqdm is unavailable
    def tqdm(iterable, **kwargs):
        return iterable


# Default directory paths
DEFAULT_RAW_DIR = "/sphenix/user/anarde/sEPD-Study/files/run3auau-pro001_pcdb001_v001"
DEFAULT_ZDC_DIR = "/direct/sphenix+tg+tg01/jets/anarde/run3auau/ZDC"
DEFAULT_SEPD_DIR = "/direct/sphenix+tg+tg01/jets/anarde/run3auau/sEPD"
DEFAULT_OUTPUT_DIR = (
    "/direct/sphenix+u/anarde/Documents/sPHENIX/Jet-Vn/files/run3auau/"
    "run3auau-merged-zdc-sepd-calib-pro001_pcdb001_v001"
)
DEFAULT_OUTPUT_PREFIX = "dst_calofitting_zdc_sepd"

# Regex to extract run number from list filenames
CALO_FILENAME_PATTERN = re.compile(r"dst_calofitting-(\d+)\.list$")
ZDC_RAW_FILENAME_PATTERN = re.compile(r"dst_zdc_raw-(\d+)\.list$")
SEPD_RAW_FILENAME_PATTERN = re.compile(r"dst_sepd_raw-(\d+)\.list$")

# Regex to extract integer segment number before .root
SEGMENT_PATTERN = re.compile(r"-(\d+)\.root(?:$|\s)")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Merge calorimeter, ZDC calibration, and sEPD calibration segments, "
                    "find missing calibration segments, and optionally clean up extra calibration files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Positional fallbacks for compatibility with merge_lists.sh
    parser.add_argument("pos_raw_dir", nargs="?", default=None, help="Calo/Raw directory (positional fallback)")
    parser.add_argument("pos_zdc_dir", nargs="?", default=None, help="ZDC calib directory (positional fallback)")
    parser.add_argument("pos_sepd_dir", nargs="?", default=None, help="sEPD calib directory (positional fallback)")
    parser.add_argument("pos_output_dir", nargs="?", default=None, help="Output directory (positional fallback)")
    parser.add_argument("pos_zdc_prefix", nargs="?", default=None, help="ZDC prefix (positional fallback, e.g. dst_zdc_raw)")
    parser.add_argument("pos_sepd_prefix", nargs="?", default=None, help="sEPD prefix (positional fallback, e.g. dst_sepd_raw)")

    # Input directories
    parser.add_argument(
        "-c",
        "--calo-dir",
        "--raw-dir",
        dest="raw_dir",
        type=str,
        default=DEFAULT_RAW_DIR,
        help="Directory containing dst_calofitting-*.list, dst_zdc_raw-*.list, and dst_sepd_raw-*.list files",
    )
    parser.add_argument(
        "-z",
        "--zdc-dir",
        type=str,
        default=DEFAULT_ZDC_DIR,
        help="Directory containing ZDC calibration files or run subdirectories",
    )
    parser.add_argument(
        "-s",
        "--sepd-dir",
        type=str,
        default=DEFAULT_SEPD_DIR,
        help="Directory containing sEPD calibration files or run subdirectories",
    )
    parser.add_argument(
        "--zdc-prefix",
        type=str,
        default=None,
        help="Optional prefix filter for ZDC list files (e.g. dst_zdc_raw, dst_zdc_calib)",
    )
    parser.add_argument(
        "--sepd-prefix",
        type=str,
        default=None,
        help="Optional prefix filter for sEPD list files (e.g. dst_sepd_raw, dst_sepd_calib)",
    )

    # Merged output options
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory to save merged per-run list files",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default=DEFAULT_OUTPUT_PREFIX,
        help="Prefix for merged output list files (e.g. dst_calofitting_zdc_sepd)",
    )
    parser.add_argument(
        "--skip-merged",
        action="store_true",
        help="Skip writing 3-way merged list files (useful if only checking or writing missing calibration)",
    )
    parser.add_argument(
        "--keep-empty",
        action="store_true",
        help="Create empty list file if a run has zero matching segments (default: skip empty runs)",
    )

    # Missing calibration options
    parser.add_argument(
        "--write-missing",
        action="store_true",
        help="Write list files for segments missing ZDC calibration and/or sEPD calibration",
    )
    parser.add_argument(
        "--missing-zdc-dir",
        type=str,
        default=None,
        help="Output directory for missing ZDC calibration lists (default: <output_dir>-missing-zdc when --write-missing is set)",
    )
    parser.add_argument(
        "--missing-sepd-dir",
        type=str,
        default=None,
        help="Output directory for missing sEPD calibration lists (default: <output_dir>-missing-sepd when --write-missing is set)",
    )
    parser.add_argument(
        "--missing-format",
        choices=["merged", "raw"],
        default="merged",
        help="Format of lines in missing calibration lists: 'merged' (<calo>,<zdc_raw>,<sepd_raw>) or 'raw' (single raw DST)",
    )

    # Extra calibration cleanup options
    parser.add_argument(
        "--remove-extra-zdc",
        "--clean-extra-zdc",
        dest="remove_extra_zdc",
        action="store_true",
        help="Delete extra ZDC calibration ROOT files from disk that are not present in raw lists",
    )
    parser.add_argument(
        "--remove-extra-sepd",
        "--clean-extra-sepd",
        dest="remove_extra_sepd",
        action="store_true",
        help="Delete extra sEPD calibration ROOT files from disk that are not present in raw lists",
    )
    parser.add_argument(
        "--remove-extra-all",
        "--clean-extra-all",
        dest="remove_extra_all",
        action="store_true",
        help="Delete extra ROOT files from disk for BOTH ZDC and sEPD calibration",
    )

    # Execution controls
    parser.add_argument(
        "-j",
        "--workers",
        type=int,
        default=None,
        help="Number of parallel worker processes (default: min(os.cpu_count(), 16))",
    )
    parser.add_argument(
        "-l",
        "--run-list",
        "--runs-list",
        dest="run_list",
        type=str,
        default=None,
        help="Optional: path to file containing run numbers (one run per line) to restrict processing",
    )
    parser.add_argument(
        "-r",
        "--runs",
        type=str,
        nargs="+",
        default=None,
        help="Optional: restrict processing to specific run number(s) or run list file(s)",
    )
    parser.add_argument(
        "-b",
        "--corrupt-list",
        "--corrupt-segments",
        dest="corrupt_list",
        type=str,
        nargs="+",
        default=None,
        help="Optional: path(s) to file containing corrupt segments to exclude from consideration",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Scan and report segment matches, missing lists, and extra files without writing or deleting any files",
    )
    parser.add_argument(
        "--save-summary",
        type=str,
        default=None,
        help="Optional path to write a TSV summary of runs and segment counts",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print detailed per-run match information to terminal",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress bar",
    )

    args = parser.parse_args()

    # Prioritize positional arguments if supplied
    if args.pos_raw_dir:
        args.raw_dir = args.pos_raw_dir
    if args.pos_zdc_dir:
        args.zdc_dir = args.pos_zdc_dir
    if args.pos_sepd_dir:
        args.sepd_dir = args.pos_sepd_dir
    if args.pos_output_dir:
        args.output_dir = args.pos_output_dir
    if args.pos_zdc_prefix:
        args.zdc_prefix = args.pos_zdc_prefix
    if args.pos_sepd_prefix:
        args.sepd_prefix = args.pos_sepd_prefix

    if args.remove_extra_all:
        args.remove_extra_zdc = True
        args.remove_extra_sepd = True

    # Configure missing directories
    if args.write_missing:
        if not args.missing_zdc_dir and args.output_dir:
            args.missing_zdc_dir = f"{args.output_dir.rstrip('/')}-missing-zdc"
        if not args.missing_sepd_dir and args.output_dir:
            args.missing_sepd_dir = f"{args.output_dir.rstrip('/')}-missing-sepd"
    else:
        # If user explicitly passed a missing directory, enable writing for that subsystem
        if args.missing_zdc_dir or args.missing_sepd_dir:
            args.write_missing = True

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
    """Print an ASCII table formatted cleanly."""
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


def extract_segment_from_line(line: str) -> Optional[int]:
    """
    Extract the segment integer from a filename or path.
    Handles lines like:
        DST_CALOFITTING_run3auau_pro001_pcdb001_v001-00074113-00095.root
        /path/to/DST_ZDC_CALIB_run3auau_pro001_pcdb001_v001-00074113-00095.root
    """
    first_part = line.split(",", 1)[0].strip()
    match = SEGMENT_PATTERN.search(first_part)
    if match:
        return int(match.group(1))

    # Fallback to rsplit
    try:
        base = os.path.basename(first_part)
        if base.endswith(".root") and "-" in base:
            seg_str = base.rsplit("-", 1)[1].split(".", 1)[0]
            if seg_str.isdigit():
                return int(seg_str)
    except Exception:
        pass

    return None


def index_calib_directory(directory: Path, prefix: Optional[str] = None) -> Dict[int, Path]:
    """
    Index a calibration directory (ZDC or sEPD).
    Supports:
    1. Run subdirectories named by run number (e.g. '74113' or '00074113')
    2. List files named by run number (e.g. 'dst_*-00074113.list' or 'dst_*-74113.list')

    Returns:
        dict mapping run_number (int) -> Path to run directory or list file
    """
    run_map: Dict[int, Path] = {}
    if not directory.is_dir():
        return run_map

    list_pattern = re.compile(r"(\d+)\.list$")

    try:
        for entry in os.scandir(directory):
            if entry.is_dir() and entry.name.isdigit():
                run_num = int(entry.name)
                run_map[run_num] = Path(entry.path)
            elif entry.is_file() and entry.name.endswith(".list"):
                if prefix and not (entry.name.startswith(f"{prefix}-") or entry.name.startswith(prefix)):
                    continue
                match = list_pattern.search(entry.name)
                if match:
                    run_num = int(match.group(1))
                    run_map[run_num] = Path(entry.path)
    except OSError as e:
        print(f"[WARNING] Error scanning directory {directory}: {e}", file=sys.stderr)

    return run_map


def load_segments_from_source(source_path: Optional[Path]) -> Dict[int, str]:
    """
    Load segments from a source path, which can be:
    - A directory containing ROOT files: returns dict[segment_int, full_root_path]
    - A .list file containing paths: returns dict[segment_int, line_stripped]

    Returns:
        Dict mapping segment number (int) -> line or full path (str)
    """
    segments: Dict[int, str] = {}
    if source_path is None or not source_path.exists():
        return segments

    if source_path.is_dir():
        try:
            for entry in os.scandir(source_path):
                if entry.name.endswith(".root"):
                    seg_num = extract_segment_from_line(entry.name)
                    if seg_num is not None:
                        segments[seg_num] = entry.path
        except OSError:
            pass
    elif source_path.is_file():
        try:
            with open(source_path, "r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    clean_line = line.strip()
                    if not clean_line or clean_line.startswith("#"):
                        continue
                    seg_num = extract_segment_from_line(clean_line)
                    if seg_num is not None:
                        segments[seg_num] = clean_line
        except OSError:
            pass

    return segments


def load_run_list_file(file_path: Path) -> Set[int]:
    """
    Load run numbers from a run list file, assuming one run number per line.
    Empty lines and lines starting with '#' (comments) are skipped.
    """
    if not file_path.is_file():
        raise FileNotFoundError(f"Run list file does not exist: {file_path}")

    runs: Set[int] = set()
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        for line_num, line in enumerate(f, 1):
            clean = line.strip()
            if not clean or clean.startswith("#"):
                continue
            try:
                runs.add(int(clean))
            except ValueError:
                digits = re.findall(r"\b\d+\b", clean)
                if len(digits) == 1:
                    runs.add(int(digits[0]))
                else:
                    raise ValueError(
                        f"Invalid run number on line {line_num} in {file_path}: '{clean}'. "
                        f"Expected one run number per line."
                    )
    return runs


def load_corrupt_segments_file(file_path: Path) -> Dict[int, Set[int]]:
    """
    Load corrupt segments from a file.
    Each line can be:
      - A full or relative ROOT filename, e.g.:
        DST_SEPD_RAW_run3auau_pro001_pcdb001_v001-00075811-00140.root
        /path/to/DST_CALOFITTING_run3auau_pro001_pcdb001_v001-00075811-00140.root
      - A dash/comma/space separated pair:
        75811-140, 75811,140, or 75811 140
      - Padded formats:
        00075811-00140

    Returns:
        Dict mapping run_number (int) -> set of corrupt segment numbers (Set[int])
    """
    if not file_path.is_file():
        raise FileNotFoundError(f"Corrupt segment list file does not exist: {file_path}")

    corrupt_map: Dict[int, Set[int]] = {}
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        for line_num, line in enumerate(f, 1):
            clean = line.strip()
            if not clean or clean.startswith("#"):
                continue

            # 1. Standard sPHENIX filename or path with -<run>-<seg>.root or -<run>-<seg>
            m = re.search(r"-0*([1-9]\d*)-0*(\d+)(?:\.root|\.list|\b)", clean)
            if m:
                run_num = int(m.group(1))
                seg_num = int(m.group(2))
                corrupt_map.setdefault(run_num, set()).add(seg_num)
                continue

            # 2. Comma or whitespace separated: "75811, 140" or "75811 140"
            parts = re.split(r"[, \t]+", clean)
            if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
                run_num = int(parts[0])
                seg_num = int(parts[1])
                corrupt_map.setdefault(run_num, set()).add(seg_num)
                continue

            # 3. Simple run-seg format: "75811-140"
            m = re.match(r"^0*([1-9]\d*)-0*(\d+)$", clean)
            if m:
                run_num = int(m.group(1))
                seg_num = int(m.group(2))
                corrupt_map.setdefault(run_num, set()).add(seg_num)
                continue

            print(
                f"[WARNING] Skipping unparseable line {line_num} in corrupt list {file_path}: '{clean}'",
                file=sys.stderr,
            )

    return corrupt_map


def load_raw_list(list_path: Optional[Path]) -> Dict[int, str]:
    """Load a raw segment list (e.g. dst_calofitting-*.list, dst_zdc_raw-*.list)."""
    segments: Dict[int, str] = {}
    if list_path is None or not list_path.is_file():
        return segments

    try:
        with open(list_path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                clean_line = line.strip()
                if not clean_line or clean_line.startswith("#"):
                    continue
                seg_num = extract_segment_from_line(clean_line)
                if seg_num is not None:
                    segments[seg_num] = clean_line
    except OSError:
        pass

    return segments


def process_single_run_worker(
    task: Tuple[
        int,              # run_num
        str,              # run_str_padded
        Path,             # calo_list_path
        Optional[Path],   # zdc_raw_path
        Optional[Path],   # sepd_raw_path
        Optional[Path],   # zdc_calib_source
        Optional[Path],   # sepd_calib_source
        Optional[Path],   # merged_out_path
        Optional[Path],   # missing_zdc_out_path
        Optional[Path],   # missing_sepd_out_path
        str,              # missing_format ('merged' or 'raw')
        bool,             # remove_extra_zdc
        bool,             # remove_extra_sepd
        bool,             # dry_run
        bool,             # keep_empty
        Optional[Set[int]], # corrupt_segments
    ]
) -> Tuple[
    int,            # run_num
    str,            # run_str_padded
    int,            # total_raw_count
    int,            # matched_merged_count
    bool,           # merged_written
    bool,           # has_zdc_calib
    int,            # zdc_calib_count
    int,            # missing_zdc_count
    int,            # extra_zdc_count
    bool,           # missing_zdc_written
    int,            # deleted_zdc_files
    int,            # reclaimed_zdc_bytes
    bool,           # has_sepd_calib
    int,            # sepd_calib_count
    int,            # missing_sepd_count
    int,            # extra_sepd_count
    bool,           # missing_sepd_written
    int,            # deleted_sepd_files
    int,            # reclaimed_sepd_bytes
    int,            # excluded_corrupt_count
    str,            # status_summary
    Optional[str],  # error_msg
]:
    """
    Worker function executed in parallel for each run.
    Handles segment matching, missing list writing, and extra calibration cleanup.
    """
    (
        run_num,
        run_str_padded,
        calo_list_path,
        zdc_raw_path,
        sepd_raw_path,
        zdc_calib_source,
        sepd_calib_source,
        merged_out_path,
        missing_zdc_out_path,
        missing_sepd_out_path,
        missing_format,
        remove_extra_zdc,
        remove_extra_sepd,
        dry_run,
        keep_empty,
        corrupt_segments,
    ) = task

    # 1. Load Calo List (defines reference segments)
    calo_segments = load_raw_list(calo_list_path)
    if not calo_segments:
        return (
            run_num, run_str_padded, 0, 0, False,
            zdc_calib_source is not None, 0, 0, 0, False, 0, 0,
            sepd_calib_source is not None, 0, 0, 0, False, 0, 0,
            0,
            "ERROR_READ_CALO", f"Failed to read calo list or empty: {calo_list_path}"
        )

    # Load raw ZDC and raw sEPD lists to establish base segments where all 3 exist
    zdc_raw_segments = load_raw_list(zdc_raw_path) if zdc_raw_path else {}
    sepd_raw_segments = load_raw_list(sepd_raw_path) if sepd_raw_path else {}

    # Segments where all three base segments are present
    if zdc_raw_segments and sepd_raw_segments:
        base_segments_set = set(calo_segments.keys()) & set(zdc_raw_segments.keys()) & set(sepd_raw_segments.keys())
    elif zdc_raw_segments:
        base_segments_set = set(calo_segments.keys()) & set(zdc_raw_segments.keys())
    elif sepd_raw_segments:
        base_segments_set = set(calo_segments.keys()) & set(sepd_raw_segments.keys())
    else:
        base_segments_set = set(calo_segments.keys())

    # Exclude corrupt segments
    excluded_corrupt_count = 0
    if corrupt_segments:
        for s in corrupt_segments:
            if s in base_segments_set:
                base_segments_set.discard(s)
                excluded_corrupt_count += 1
            calo_segments.pop(s, None)
            zdc_raw_segments.pop(s, None)
            sepd_raw_segments.pop(s, None)

    total_raw_count = len(base_segments_set)

    # 2. Load Calibration Segments
    has_zdc_calib = zdc_calib_source is not None and zdc_calib_source.exists()
    zdc_calib_segments = load_segments_from_source(zdc_calib_source) if has_zdc_calib else {}
    if corrupt_segments:
        for s in corrupt_segments:
            zdc_calib_segments.pop(s, None)
    zdc_calib_count = len(zdc_calib_segments)

    has_sepd_calib = sepd_calib_source is not None and sepd_calib_source.exists()
    sepd_calib_segments = load_segments_from_source(sepd_calib_source) if has_sepd_calib else {}
    if corrupt_segments:
        for s in corrupt_segments:
            sepd_calib_segments.pop(s, None)
    sepd_calib_count = len(sepd_calib_segments)

    # 3. Analyze ZDC Calibration (Missing and Extra)
    missing_zdc_segs = sorted([s for s in base_segments_set if s not in zdc_calib_segments])
    missing_zdc_count = len(missing_zdc_segs)

    extra_zdc_segs = sorted([s for s in zdc_calib_segments.keys() if s not in base_segments_set])
    extra_zdc_count = len(extra_zdc_segs)

    deleted_zdc_files = 0
    reclaimed_zdc_bytes = 0
    if remove_extra_zdc and extra_zdc_count > 0:
        for s in extra_zdc_segs:
            fpath = Path(zdc_calib_segments[s])
            if fpath.exists():
                try:
                    reclaimed_zdc_bytes += fpath.stat().st_size
                    if not dry_run:
                        fpath.unlink(missing_ok=True)
                    deleted_zdc_files += 1
                except OSError:
                    pass

    # 4. Analyze sEPD Calibration (Missing and Extra)
    missing_sepd_segs = sorted([s for s in base_segments_set if s not in sepd_calib_segments])
    missing_sepd_count = len(missing_sepd_segs)

    extra_sepd_segs = sorted([s for s in sepd_calib_segments.keys() if s not in base_segments_set])
    extra_sepd_count = len(extra_sepd_segs)

    deleted_sepd_files = 0
    reclaimed_sepd_bytes = 0
    if remove_extra_sepd and extra_sepd_count > 0:
        for s in extra_sepd_segs:
            fpath = Path(sepd_calib_segments[s])
            if fpath.exists():
                try:
                    reclaimed_sepd_bytes += fpath.stat().st_size
                    if not dry_run:
                        fpath.unlink(missing_ok=True)
                    deleted_sepd_files += 1
                except OSError:
                    pass

    # 5. Determine 3-Way Matched Segments
    common_segments = sorted(
        base_segments_set & set(zdc_calib_segments.keys()) & set(sepd_calib_segments.keys())
    )
    matched_merged_count = len(common_segments)

    # 6. Write 3-Way Merged List File
    merged_written = False
    if merged_out_path is not None and not dry_run and (matched_merged_count > 0 or keep_empty):
        try:
            merged_out_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_merged = merged_out_path.with_suffix(".tmp_merge")
            with open(tmp_merged, "w", encoding="utf-8") as out_f:
                for s in common_segments:
                    out_f.write(f"{calo_segments[s]},{zdc_calib_segments[s]},{sepd_calib_segments[s]}\n")
            tmp_merged.replace(merged_out_path)
            merged_written = True
        except OSError as e:
            return (
                run_num, run_str_padded, total_raw_count, matched_merged_count, False,
                has_zdc_calib, zdc_calib_count, missing_zdc_count, extra_zdc_count, False, deleted_zdc_files, reclaimed_zdc_bytes,
                has_sepd_calib, sepd_calib_count, missing_sepd_count, extra_sepd_count, False, deleted_sepd_files, reclaimed_sepd_bytes,
                excluded_corrupt_count,
                "ERROR_WRITE_MERGED", f"Failed to write merged list {merged_out_path}: {e}"
            )

    # 7. Write Missing ZDC Calibration List File
    missing_zdc_written = False
    if missing_zdc_out_path is not None:
        if missing_zdc_count > 0:
            if not dry_run:
                try:
                    missing_zdc_out_path.parent.mkdir(parents=True, exist_ok=True)
                    tmp_mzdc = missing_zdc_out_path.with_suffix(".tmp_mzdc")
                    with open(tmp_mzdc, "w", encoding="utf-8") as out_f:
                        for s in missing_zdc_segs:
                            if missing_format == "raw":
                                line = zdc_raw_segments[s]
                            else:
                                line = f"{calo_segments[s]},{zdc_raw_segments[s]},{sepd_raw_segments[s]}"
                            out_f.write(f"{line}\n")
                    tmp_mzdc.replace(missing_zdc_out_path)
                    missing_zdc_written = True
                except OSError as e:
                    return (
                        run_num, run_str_padded, total_raw_count, matched_merged_count, merged_written,
                        has_zdc_calib, zdc_calib_count, missing_zdc_count, extra_zdc_count, False, deleted_zdc_files, reclaimed_zdc_bytes,
                        has_sepd_calib, sepd_calib_count, missing_sepd_count, extra_sepd_count, False, deleted_sepd_files, reclaimed_sepd_bytes,
                        excluded_corrupt_count,
                        "ERROR_WRITE_MISSING_ZDC", f"Failed to write missing ZDC list {missing_zdc_out_path}: {e}"
                    )
            else:
                missing_zdc_written = True
        elif not dry_run and missing_zdc_out_path.exists():
            try:
                missing_zdc_out_path.unlink()
            except OSError:
                pass

    # 8. Write Missing sEPD Calibration List File
    missing_sepd_written = False
    if missing_sepd_out_path is not None:
        if missing_sepd_count > 0:
            if not dry_run:
                try:
                    missing_sepd_out_path.parent.mkdir(parents=True, exist_ok=True)
                    tmp_msepd = missing_sepd_out_path.with_suffix(".tmp_msepd")
                    with open(tmp_msepd, "w", encoding="utf-8") as out_f:
                        for s in missing_sepd_segs:
                            if missing_format == "raw":
                                line = sepd_raw_segments[s]
                            else:
                                line = f"{calo_segments[s]},{zdc_raw_segments[s]},{sepd_raw_segments[s]}"
                            out_f.write(f"{line}\n")
                    tmp_msepd.replace(missing_sepd_out_path)
                    missing_sepd_written = True
                except OSError as e:
                    return (
                        run_num, run_str_padded, total_raw_count, matched_merged_count, merged_written,
                        has_zdc_calib, zdc_calib_count, missing_zdc_count, extra_zdc_count, missing_zdc_written, deleted_zdc_files, reclaimed_zdc_bytes,
                        has_sepd_calib, sepd_calib_count, missing_sepd_count, extra_sepd_count, False, deleted_sepd_files, reclaimed_sepd_bytes,
                        excluded_corrupt_count,
                        "ERROR_WRITE_MISSING_SEPD", f"Failed to write missing sEPD list {missing_sepd_out_path}: {e}"
                    )
            else:
                missing_sepd_written = True
        elif not dry_run and missing_sepd_out_path.exists():
            try:
                missing_sepd_out_path.unlink()
            except OSError:
                pass

    # Overall Status Summary
    if matched_merged_count == total_raw_count and total_raw_count > 0:
        status_summary = "PERFECT_MATCH"
    elif matched_merged_count > 0:
        status_summary = f"PARTIAL_MATCH_{matched_merged_count}/{total_raw_count}"
    elif not has_zdc_calib and not has_sepd_calib:
        status_summary = "MISSING_ZDC_AND_SEPD"
    elif not has_zdc_calib:
        status_summary = "NO_ZDC_CALIB"
    elif not has_sepd_calib:
        status_summary = "NO_SEPD_CALIB"
    else:
        status_summary = "NO_COMMON_SEGMENTS"

    return (
        run_num,
        run_str_padded,
        total_raw_count,
        matched_merged_count,
        merged_written,
        has_zdc_calib,
        zdc_calib_count,
        missing_zdc_count,
        extra_zdc_count,
        missing_zdc_written,
        deleted_zdc_files,
        reclaimed_zdc_bytes,
        has_sepd_calib,
        sepd_calib_count,
        missing_sepd_count,
        extra_sepd_count,
        missing_sepd_written,
        deleted_sepd_files,
        reclaimed_sepd_bytes,
        excluded_corrupt_count,
        status_summary,
        None,
    )


def main():
    args = parse_args()
    start_time = time.time()

    raw_dir = Path(os.path.abspath(args.raw_dir))
    zdc_dir = Path(os.path.abspath(args.zdc_dir))
    sepd_dir = Path(os.path.abspath(args.sepd_dir))
    output_dir = Path(os.path.abspath(args.output_dir)) if args.output_dir else None
    missing_zdc_dir = Path(os.path.abspath(args.missing_zdc_dir)) if args.missing_zdc_dir else None
    missing_sepd_dir = Path(os.path.abspath(args.missing_sepd_dir)) if args.missing_sepd_dir else None

    num_workers = args.workers if args.workers is not None else min(os.cpu_count() or 4, 16)

    print("\n=======================================================")
    print(" Jet-Vn: Calo + ZDC + sEPD Calibration & List Manager")
    print("=======================================================")
    print(f"Raw Input Dir:    {raw_dir}")
    print(f"ZDC Calib Dir:    {zdc_dir}")
    print(f"sEPD Calib Dir:   {sepd_dir}")
    if not args.skip_merged and output_dir:
        print(f"Merged Output:    {output_dir}")
        print(f"Output Prefix:    {args.output_prefix}")
    else:
        print("Merged Output:    DISABLED")

    if args.write_missing:
        print(f"Missing ZDC Dir:  {missing_zdc_dir}")
        print(f"Missing sEPD Dir: {missing_sepd_dir}")
        print(f"Missing Format:   {args.missing_format}")
    else:
        print("Missing Lists:    DISABLED (use --write-missing to generate)")

    if args.remove_extra_zdc or args.remove_extra_sepd:
        actions = []
        if args.remove_extra_zdc:
            actions.append("ZDC")
        if args.remove_extra_sepd:
            actions.append("sEPD")
        action_text = "WILL DELETE" if not args.dry_run else "WOULD DELETE (dry-run)"
        print(f"Extra Calib Purge:ENABLED ({action_text} extra ROOT files for {', '.join(actions)})")
    else:
        print("Extra Calib Purge:DISABLED (use --remove-extra-all to purge)")

    if args.corrupt_list:
        print(f"Corrupt Segments: EXCLUDE ({len(args.corrupt_list)} list file(s))")
    else:
        print("Corrupt Segments: NONE")

    print(f"Mode:             {'DRY RUN (no files modified or written)' if args.dry_run else 'ACTIVE (writing lists)'}")
    print(f"Workers:          {num_workers}")
    print("=======================================================\n")

    # Validate directories
    if not raw_dir.is_dir():
        print(f"Error: Raw directory does not exist: {raw_dir}", file=sys.stderr)
        sys.exit(1)
    if not zdc_dir.is_dir():
        print(f"Error: ZDC directory does not exist: {zdc_dir}", file=sys.stderr)
        sys.exit(1)
    if not sepd_dir.is_dir():
        print(f"Error: sEPD directory does not exist: {sepd_dir}", file=sys.stderr)
        sys.exit(1)

    # 1. Index calibration sources
    print("Indexing ZDC calibration runs...")
    zdc_calib_map = index_calib_directory(zdc_dir, prefix=args.zdc_prefix)
    print(f"Found {len(zdc_calib_map)} ZDC calibration run(s).")

    print("Indexing sEPD calibration runs...")
    sepd_calib_map = index_calib_directory(sepd_dir, prefix=args.sepd_prefix)
    print(f"Found {len(sepd_calib_map)} sEPD calibration run(s).\n")

    # 2. Collect Calo and Raw lists from raw_dir
    print(f"Scanning lists in {raw_dir}...")
    calo_run_files: Dict[int, Tuple[str, Path]] = {}
    zdc_raw_run_files: Dict[int, Path] = {}
    sepd_raw_run_files: Dict[int, Path] = {}

    for entry in os.scandir(raw_dir):
        if not entry.name.endswith(".list") or not entry.is_file():
            continue

        m_calo = CALO_FILENAME_PATTERN.match(entry.name)
        if m_calo:
            raw_run_str = m_calo.group(1)
            run_num = int(raw_run_str)
            run_padded = f"{run_num:08d}" if len(raw_run_str) != 8 else raw_run_str
            calo_run_files[run_num] = (run_padded, Path(entry.path))
            continue

        m_zdc = ZDC_RAW_FILENAME_PATTERN.match(entry.name)
        if m_zdc:
            zdc_raw_run_files[int(m_zdc.group(1))] = Path(entry.path)
            continue

        m_sepd = SEPD_RAW_FILENAME_PATTERN.match(entry.name)
        if m_sepd:
            sepd_raw_run_files[int(m_sepd.group(1))] = Path(entry.path)
            continue

    print(f"Found {len(calo_run_files)} calofitting list(s), {len(zdc_raw_run_files)} raw ZDC list(s), and {len(sepd_raw_run_files)} raw sEPD list(s).\n")

    # 2b. Load corrupt segments if requested
    corrupt_map: Dict[int, Set[int]] = {}
    if args.corrupt_list:
        total_corrupt_files = 0
        for c_file in args.corrupt_list:
            c_path = Path(os.path.abspath(c_file))
            try:
                loaded = load_corrupt_segments_file(c_path)
                file_segs = sum(len(s) for s in loaded.values())
                print(f"Loaded {file_segs} corrupt segment(s) across {len(loaded)} run(s) from: {c_path}")
                for r_num, segs in loaded.items():
                    corrupt_map.setdefault(r_num, set()).update(segs)
                total_corrupt_files += 1
            except Exception as e:
                print(f"[ERROR] Failed to read corrupt list file '{c_file}': {e}", file=sys.stderr)
                sys.exit(1)
        total_unique_corrupt = sum(len(s) for s in corrupt_map.values())
        print(f"Total corrupt segments configured for exclusion: {total_unique_corrupt} across {len(corrupt_map)} run(s).\n")

    # 3. Filter runs if requested
    target_runs = sorted(calo_run_files.keys())
    requested_runs: Optional[Set[int]] = None

    if args.run_list:
        run_list_path = Path(os.path.abspath(args.run_list))
        try:
            requested_runs = load_run_list_file(run_list_path)
            print(f"Loaded {len(requested_runs)} run(s) from run list: {run_list_path}")
        except Exception as e:
            print(f"[ERROR] Failed to read run list file: {e}", file=sys.stderr)
            sys.exit(1)

    if args.runs:
        cli_runs: Set[int] = set()
        for item in args.runs:
            item_path = Path(item)
            if item_path.is_file():
                try:
                    file_runs = load_run_list_file(item_path)
                    print(f"Loaded {len(file_runs)} run(s) from file passed to --runs: {item_path}")
                    cli_runs.update(file_runs)
                except Exception as e:
                    print(f"[ERROR] Failed to read run list file '{item}': {e}", file=sys.stderr)
                    sys.exit(1)
            else:
                try:
                    cli_runs.add(int(item))
                except ValueError:
                    print(f"[ERROR] Invalid run number or file path passed to --runs: '{item}'", file=sys.stderr)
                    sys.exit(1)

        if requested_runs is not None:
            requested_runs = requested_runs | cli_runs
        else:
            requested_runs = cli_runs

    if requested_runs is not None:
        matched_runs = [r for r in target_runs if r in requested_runs]
        missing_runs = requested_runs - set(calo_run_files.keys())
        if missing_runs:
            print(
                f"Notice: {len(missing_runs)} of {len(requested_runs)} requested run(s) "
                f"have no calofitting list in {raw_dir}."
            )
        target_runs = matched_runs
        print(f"Filtered to {len(target_runs)} matching run(s) for processing.\n")

    if not target_runs:
        print("No matching runs to process. Exiting.")
        sys.exit(0)

    # 4. Prepare worker tasks
    tasks = []
    for run_num in target_runs:
        run_padded, calo_path = calo_run_files[run_num]
        zdc_raw_path = zdc_raw_run_files.get(run_num)
        sepd_raw_path = sepd_raw_run_files.get(run_num)

        zdc_calib_source = zdc_calib_map.get(run_num)
        sepd_calib_source = sepd_calib_map.get(run_num)

        # Output file paths
        merged_out_path = (
            (output_dir / f"{args.output_prefix}-{run_padded}.list")
            if (not args.skip_merged and output_dir)
            else None
        )
        missing_zdc_out_path = (
            (missing_zdc_dir / f"dst_calofitting_zdc_sepd-{run_padded}.list")
            if missing_zdc_dir
            else None
        )
        missing_sepd_out_path = (
            (missing_sepd_dir / f"dst_calofitting_zdc_sepd-{run_padded}.list")
            if missing_sepd_dir
            else None
        )

        corrupt_segs = corrupt_map.get(run_num)

        tasks.append((
            run_num,
            run_padded,
            calo_path,
            zdc_raw_path,
            sepd_raw_path,
            zdc_calib_source,
            sepd_calib_source,
            merged_out_path,
            missing_zdc_out_path,
            missing_sepd_out_path,
            args.missing_format,
            args.remove_extra_zdc,
            args.remove_extra_sepd,
            args.dry_run,
            args.keep_empty,
            corrupt_segs,
        ))

    # 5. Execute processing in parallel
    print(f"Processing {len(tasks)} run(s) across {num_workers} parallel workers...")

    chunk_size = max(1, len(tasks) // (num_workers * 4)) if len(tasks) > 50 else 1

    if num_workers > 1 and len(tasks) > 1:
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
            results = list(
                tqdm(
                    executor.map(process_single_run_worker, tasks, chunksize=chunk_size),
                    total=len(tasks),
                    desc="Analyzing & merging runs",
                    unit="run",
                    disable=args.no_progress,
                )
            )
    else:
        results = list(
            tqdm(
                (process_single_run_worker(t) for t in tasks),
                total=len(tasks),
                desc="Analyzing & merging runs",
                unit="run",
                disable=args.no_progress,
            )
        )

    # 6. Aggregate statistics
    total_raw_segments = 0
    total_matched_merged_segments = 0

    runs_with_matches = 0
    runs_perfect_match = 0
    runs_partial_match = 0
    runs_zero_matches = 0

    runs_missing_any_zdc = 0
    runs_completely_missing_zdc = 0
    total_missing_zdc_segments = 0
    total_zdc_calib_segments = 0
    runs_with_extra_zdc = 0
    total_extra_zdc_segments = 0
    total_deleted_zdc_files = 0
    total_reclaimed_zdc_bytes = 0

    runs_missing_any_sepd = 0
    runs_completely_missing_sepd = 0
    total_missing_sepd_segments = 0
    total_sepd_calib_segments = 0
    runs_with_extra_sepd = 0
    total_extra_sepd_segments = 0
    total_deleted_sepd_files = 0
    total_reclaimed_sepd_bytes = 0

    merged_files_written_count = 0
    missing_zdc_files_written_count = 0
    missing_sepd_files_written_count = 0
    errors: List[str] = []

    total_excluded_corrupt_segments = 0
    runs_with_corrupt_excluded = 0

    for r in results:
        (
            run_num,
            run_padded,
            raw_c,
            matched_c,
            merged_w,
            has_zdc,
            zdc_c,
            m_zdc_c,
            e_zdc_c,
            m_zdc_w,
            del_zdc,
            rec_zdc,
            has_sepd,
            sepd_c,
            m_sepd_c,
            e_sepd_c,
            m_sepd_w,
            del_sepd,
            rec_sepd,
            corrupt_c,
            status,
            err,
        ) = r

        if err:
            errors.append(err)
            continue

        if corrupt_c > 0:
            total_excluded_corrupt_segments += corrupt_c
            runs_with_corrupt_excluded += 1

        total_raw_segments += raw_c
        total_matched_merged_segments += matched_c

        if merged_w:
            merged_files_written_count += 1
        if m_zdc_w:
            missing_zdc_files_written_count += 1
        if m_sepd_w:
            missing_sepd_files_written_count += 1

        # Match status
        if status == "PERFECT_MATCH":
            runs_with_matches += 1
            runs_perfect_match += 1
        elif matched_c > 0:
            runs_with_matches += 1
            runs_partial_match += 1
        else:
            runs_zero_matches += 1

        # ZDC coverage
        total_zdc_calib_segments += zdc_c
        if not has_zdc and raw_c > 0:
            runs_completely_missing_zdc += 1
        if m_zdc_c > 0:
            runs_missing_any_zdc += 1
            total_missing_zdc_segments += m_zdc_c
        if e_zdc_c > 0:
            runs_with_extra_zdc += 1
            total_extra_zdc_segments += e_zdc_c
        total_deleted_zdc_files += del_zdc
        total_reclaimed_zdc_bytes += rec_zdc

        # sEPD coverage
        total_sepd_calib_segments += sepd_c
        if not has_sepd and raw_c > 0:
            runs_completely_missing_sepd += 1
        if m_sepd_c > 0:
            runs_missing_any_sepd += 1
            total_missing_sepd_segments += m_sepd_c
        if e_sepd_c > 0:
            runs_with_extra_sepd += 1
            total_extra_sepd_segments += e_sepd_c
        total_deleted_sepd_files += del_sepd
        total_reclaimed_sepd_bytes += rec_sepd

    if errors:
        print(f"\n[WARNING] Encountered {len(errors)} error(s) during processing:", file=sys.stderr)
        for err in errors[:10]:
            print(f"  {err}", file=sys.stderr)
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more errors", file=sys.stderr)

    # 7. Write TSV summary if requested
    if args.save_summary:
        summary_path = Path(os.path.abspath(args.save_summary))
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_path, "w", encoding="utf-8") as sum_f:
            sum_f.write(
                "run\trun_padded\ttotal_raw\tmatched_merged\t"
                "missing_zdc\ttotal_zdc\textra_zdc\t"
                "missing_sepd\ttotal_sepd\textra_sepd\t"
                "corrupt_excluded\tstatus\n"
            )
            for r in results:
                status_str = "ERROR" if r[21] else r[20]
                sum_f.write(
                    f"{r[0]}\t{r[1]}\t{r[2]}\t{r[3]}\t"
                    f"{r[7]}\t{r[6]}\t{r[8]}\t"
                    f"{r[14]}\t{r[13]}\t{r[15]}\t"
                    f"{r[19]}\t{status_str}\n"
                )
        print(f"\nSaved comprehensive run summary to: {summary_path}")

    # 8. Print Summary Tables
    total_runs_evaluated = len(target_runs)
    pct_matched_segs = (
        (total_matched_merged_segments / total_raw_segments * 100) if total_raw_segments > 0 else 0.0
    )
    pct_runs_matched = (
        (runs_with_matches / total_runs_evaluated * 100) if total_runs_evaluated > 0 else 0.0
    )

    print("\n1. Merged List Coverage:")
    merge_headers = ["Merge Metric", "Count", "Percentage"]
    merge_rows = [
        ["Total Runs Evaluated", str(total_runs_evaluated), "100.00%"],
        ["Runs with 3-Way Common Segments", str(runs_with_matches), f"{pct_runs_matched:6.2f}%"],
        ["  - Perfect match (all segments in Calo, ZDC, sEPD)", str(runs_perfect_match), f"{(runs_perfect_match / total_runs_evaluated * 100):6.2f}%" if total_runs_evaluated else "0.00%"],
        ["  - Partial match (some segments missing in calib)", str(runs_partial_match), f"{(runs_partial_match / total_runs_evaluated * 100):6.2f}%" if total_runs_evaluated else "0.00%"],
        ["Runs with Zero Matching Segments", str(runs_zero_matches), f"{(runs_zero_matches / total_runs_evaluated * 100):6.2f}%" if total_runs_evaluated else "0.00%"],
        ["----------------------------------------", "-------", "----------"],
        ["Total Raw Segments Scanned (Valid)", str(total_raw_segments), "100.00%"],
        ["Total Matched Segments Merged", str(total_matched_merged_segments), f"{pct_matched_segs:6.2f}%"],
    ]
    if total_excluded_corrupt_segments > 0 or args.corrupt_list:
        merge_rows.append(["Total Corrupt Segments Excluded", str(total_excluded_corrupt_segments), "-"])
    print_table(merge_headers, merge_rows)

    print("\n2. Calibration Coverage Summary:")
    calib_headers = ["Subsystem Calibration Metric", "Count", "Percentage"]
    pct_zdc_missing_runs = (runs_missing_any_zdc / total_runs_evaluated * 100) if total_runs_evaluated else 0.0
    pct_zdc_missing_segs = (total_missing_zdc_segments / total_raw_segments * 100) if total_raw_segments else 0.0
    pct_sepd_missing_runs = (runs_missing_any_sepd / total_runs_evaluated * 100) if total_runs_evaluated else 0.0
    pct_sepd_missing_segs = (total_missing_sepd_segments / total_raw_segments * 100) if total_raw_segments else 0.0

    calib_rows = [
        ["ZDC: Runs with ALL Segments in Calib", str(total_runs_evaluated - runs_missing_any_zdc), f"{100 - pct_zdc_missing_runs:6.2f}%"],
        ["ZDC: Runs Missing ANY Calib Segments", str(runs_missing_any_zdc), f"{pct_zdc_missing_runs:6.2f}%"],
        ["  - ZDC: Runs completely without calib", str(runs_completely_missing_zdc), f"{(runs_completely_missing_zdc / total_runs_evaluated * 100):6.2f}%" if total_runs_evaluated else "0.00%"],
        ["ZDC: Total Segments Missing Calib", str(total_missing_zdc_segments), f"{pct_zdc_missing_segs:6.2f}%"],
        ["ZDC: Runs with Extra Segments", str(runs_with_extra_zdc), f"{(runs_with_extra_zdc / total_runs_evaluated * 100):6.2f}%" if total_runs_evaluated else "0.00%"],
        ["ZDC: Total Extra Calib Segments", str(total_extra_zdc_segments), "-"],
        ["----------------------------------------", "-------", "----------"],
        ["sEPD: Runs with ALL Segments in Calib", str(total_runs_evaluated - runs_missing_any_sepd), f"{100 - pct_sepd_missing_runs:6.2f}%"],
        ["sEPD: Runs Missing ANY Calib Segments", str(runs_missing_any_sepd), f"{pct_sepd_missing_runs:6.2f}%"],
        ["  - sEPD: Runs completely without calib", str(runs_completely_missing_sepd), f"{(runs_completely_missing_sepd / total_runs_evaluated * 100):6.2f}%" if total_runs_evaluated else "0.00%"],
        ["sEPD: Total Segments Missing Calib", str(total_missing_sepd_segments), f"{pct_sepd_missing_segs:6.2f}%"],
        ["sEPD: Runs with Extra Segments", str(runs_with_extra_sepd), f"{(runs_with_extra_sepd / total_runs_evaluated * 100):6.2f}%" if total_runs_evaluated else "0.00%"],
        ["sEPD: Total Extra Calib Segments", str(total_extra_sepd_segments), "-"],
    ]
    print_table(calib_headers, calib_rows)

    # Cleanup table
    has_extra = (total_extra_zdc_segments > 0) or (total_extra_sepd_segments > 0)
    if args.remove_extra_zdc or args.remove_extra_sepd or has_extra:
        clean_action = "Purged from Disk" if (not args.dry_run and (args.remove_extra_zdc or args.remove_extra_sepd)) else "Identified for Cleanup"
        clean_headers = ["Calibration Cleanup Metric", "Count / Value"]
        clean_rows = [
            [f"Extra ZDC ROOT Files {clean_action}", str(total_deleted_zdc_files if not args.dry_run else total_extra_zdc_segments)],
            [f"ZDC Disk Space {'Freed' if not args.dry_run else 'Reclaimable'}", format_bytes(total_reclaimed_zdc_bytes)],
            [f"Extra sEPD ROOT Files {clean_action}", str(total_deleted_sepd_files if not args.dry_run else total_extra_sepd_segments)],
            [f"sEPD Disk Space {'Freed' if not args.dry_run else 'Reclaimable'}", format_bytes(total_reclaimed_sepd_bytes)],
        ]
        print(f"\n3. Calibration Cleanup Summary ({'DRY RUN - No changes made' if args.dry_run else 'ACTIVE'}):")
        print_table(clean_headers, clean_rows)

    # Output files summary
    print("\nOutput Files Generated:")
    if not args.skip_merged and output_dir:
        action = "Wrote" if not args.dry_run else "Would write (dry-run)"
        print(f"  Merged Lists:      {action} {merged_files_written_count} file(s) to: {output_dir}")
    if missing_zdc_dir:
        action = "Wrote" if not args.dry_run else "Would write (dry-run)"
        print(f"  Missing ZDC Lists: {action} {missing_zdc_files_written_count} file(s) to: {missing_zdc_dir}")
    if missing_sepd_dir:
        action = "Wrote" if not args.dry_run else "Would write (dry-run)"
        print(f"  Missing sEPD Lists:{action} {missing_sepd_files_written_count} file(s) to: {missing_sepd_dir}")

    # 9. Verbose listing
    if args.verbose:
        print("\nDetailed Per-Run Summary:")
        for r in results:
            run_num, run_padded, raw_c, matched_c = r[0], r[1], r[2], r[3]
            m_zdc, m_sepd, corrupt_c, status = r[7], r[14], r[19], r[20]
            if m_zdc > 0 or m_sepd > 0 or matched_c < raw_c or corrupt_c > 0:
                corrupt_info = f", corrupt_excluded={corrupt_c}" if corrupt_c > 0 else ""
                print(
                    f"  Run {run_padded} ({run_num}): raw={raw_c}, matched={matched_c} | missing ZDC={m_zdc}, missing sEPD={m_sepd}{corrupt_info} [{status}]"
                )

    elapsed_time = time.time() - start_time
    print(f"\nCompleted in {elapsed_time:.2f}s.\n")


if __name__ == "__main__":
    main()
