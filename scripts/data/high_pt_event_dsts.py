#!/usr/bin/env python3
"""
Build per-run DST lists and a run,event list from Jet-Ana high-pT event CSVs

This script:
1. Reads a list of Jet-Ana CSV files (columns include `run` and `event_id`).
2. Maps each event to its DST segment (event_id // events-per-segment).
3. Writes one list per run with unique segments, each line:
       <dst_calofitting>,<dst_zdc_calib_full_path>,<dst_sepd_calib_full_path>
   Segments whose ZDC or sEPD calib file does not exist are skipped and reported.
4. Writes one CSV of the unique (run, event) pairs: `run,event`.

Usage:
    python3 high_pt_event_dsts.py /sphenix/user/anarde/sEPD-Study/jetAnav3/10-26-26/high_pt_jets.list

    # Report counts and missing files without writing anything:
    python3 high_pt_event_dsts.py <input_list> --dry-run

    # Custom output directory:
    python3 high_pt_event_dsts.py <input_list> -o /path/to/output
"""

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_ZDC_DIR = "/direct/sphenix+tg+tg01/jets/anarde/run3auau/ZDC"
DEFAULT_SEPD_DIR = "/direct/sphenix+tg+tg01/jets/anarde/run3auau/sEPD"
DEFAULT_OUTPUT_SUBDIR = "high_pt_dsts"
DEFAULT_OUTPUT_PREFIX = "dst_calofitting_zdc_sepd"
DEFAULT_EVENTS_CSV = "high_pt_events.csv"
DEFAULT_EVENTS_PER_SEGMENT = 100000
DST_TAG = "run3auau_pro001_pcdb001_v001"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build per-run calofitting/ZDC/sEPD DST lists and a run,event CSV from Jet-Ana high-pT CSVs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("input_list", type=Path, help="File listing Jet-Ana CSV paths, one per line")
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=None,
        help=f"Output directory (default: <input_list parent>/{DEFAULT_OUTPUT_SUBDIR})",
    )
    parser.add_argument("--zdc-dir", type=Path, default=Path(DEFAULT_ZDC_DIR), help="ZDC calib directory (per-run subdirs)")
    parser.add_argument("--sepd-dir", type=Path, default=Path(DEFAULT_SEPD_DIR), help="sEPD calib directory (per-run subdirs)")
    parser.add_argument("--output-prefix", default=DEFAULT_OUTPUT_PREFIX, help="Prefix for per-run list files")
    parser.add_argument("--events-csv", default=DEFAULT_EVENTS_CSV, help="Name of the output run,event CSV")
    parser.add_argument(
        "--events-per-segment", type=int, default=DEFAULT_EVENTS_PER_SEGMENT, help="Events per DST segment"
    )
    parser.add_argument("--dry-run", action="store_true", help="Report only, write nothing")
    return parser.parse_args()


def read_csv_paths(list_path: Path) -> List[Path]:
    """Return unique CSV paths from the input list, preserving order."""
    seen: Set[str] = set()
    paths: List[Path] = []
    with list_path.open() as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or line in seen:
                continue
            seen.add(line)
            paths.append(Path(line))
    return paths


def read_events(csv_paths: List[Path]) -> Tuple[Set[Tuple[int, int]], int, int]:
    """Collect unique (run, event) pairs. Returns (events, n_csv_read, n_rows)."""
    events: Set[Tuple[int, int]] = set()
    n_read = 0
    n_rows = 0
    for path in csv_paths:
        try:
            with path.open(newline="") as f:
                for row in csv.DictReader(f):
                    events.add((int(row["run"]), int(row["event_id"])))
                    n_rows += 1
            n_read += 1
        except (OSError, KeyError, ValueError) as e:
            print(f"WARNING: skipping {path}: {e}", file=sys.stderr)
    return events, n_read, n_rows


def build_dst_line(run: int, segment: int, zdc_dir: Path, sepd_dir: Path) -> Tuple[str, Path, Path]:
    tag = f"{DST_TAG}-{run:08d}-{segment:05d}.root"
    zdc = zdc_dir / str(run) / f"DST_ZDC_CALIB_{tag}"
    sepd = sepd_dir / str(run) / f"DST_SEPD_CALIB_{tag}"
    return f"DST_CALOFITTING_{tag},{zdc},{sepd}", zdc, sepd


def main():
    args = parse_args()
    output_dir = args.output_dir or args.input_list.parent / DEFAULT_OUTPUT_SUBDIR

    csv_paths = read_csv_paths(args.input_list)
    events, n_read, n_rows = read_events(csv_paths)

    segments: Dict[int, Set[int]] = defaultdict(set)
    for run, event in events:
        segments[run].add(event // args.events_per_segment)

    run_lines: Dict[int, List[str]] = {}
    missing: List[Path] = []
    for run in sorted(segments):
        lines = []
        for seg in sorted(segments[run]):
            line, zdc, sepd = build_dst_line(run, seg, args.zdc_dir, args.sepd_dir)
            bad = [p for p in (zdc, sepd) if not p.is_file()]
            if bad:
                missing.extend(bad)
                continue
            lines.append(line)
        run_lines[run] = lines

    n_segments = sum(len(s) for s in segments.values())
    n_written = sum(len(v) for v in run_lines.values())

    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
        for run, lines in run_lines.items():
            if not lines:
                continue
            (output_dir / f"{args.output_prefix}-{run:08d}.list").write_text("\n".join(lines) + "\n")
        with (output_dir / args.events_csv).open("w", newline="") as f:
            writer = csv.writer(f, lineterminator="\n")
            writer.writerow(["run", "event"])
            writer.writerows(sorted(events))

    print(f"CSVs listed (unique) : {len(csv_paths)}")
    print(f"CSVs read            : {n_read}")
    print(f"Rows                 : {n_rows}")
    print(f"Unique events        : {len(events)}")
    print(f"Runs                 : {len(segments)}")
    print(f"Unique segments      : {n_segments}")
    print(f"Segments written     : {n_written}")
    print(f"Missing calib files  : {len(missing)}")
    for p in missing:
        print(f"  MISSING: {p}")
    print(f"Output dir           : {output_dir}{' (dry run, nothing written)' if args.dry_run else ''}")

    return 1 if missing or n_read < len(csv_paths) else 0


if __name__ == "__main__":
    sys.exit(main())
