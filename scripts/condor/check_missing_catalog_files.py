#!/usr/bin/env python3
"""
Condor Missing FileCatalog / Lustre Segment Checker

Identifies failed Condor jobs caused by missing input files (e.g. `getinputfiles.pl`
failures with "could not retrieve <file> from filecatalog"), extracts the run and
segment numbers, inspects the associated input list files, and queries FileCatalog
(and/or checks Lustre disk paths) to find all missing segments.

Outputs:
  - reported_missing_segments.list: Segments reported as failure points in stdout.
  - all_missing_segments.list: All candidate segments in failed jobs missing in FileCatalog.
  - all_missing_files.txt: Full LFNs missing from FileCatalog.
  - (Optional) Filters an input list directory to remove lines with missing segments.

Usage Examples:
  # Scan condor directory using an existing error summary log:
  python3 scripts/condor/check_missing_catalog_files.py -d /path/to/condor_dir -l log.txt

  # Automatically scan condor directory stdout and error files:
  python3 scripts/condor/check_missing_catalog_files.py -d /sphenix/user/anarde/sEPD-Study/event_qa/10-04-26

  # Filter an input list directory to exclude all bad segments (dry-run):
  python3 scripts/condor/check_missing_catalog_files.py -d ... -f /path/to/input_lists -n

  # Filter input lists for resubmission:
  python3 scripts/condor/check_missing_catalog_files.py -d ... -f /path/to/input_lists
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import re
import subprocess
import sys


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Extract missing segments and FileCatalog failures from Condor job logs."
    )
    parser.add_argument(
        "-d", "--condor-dir",
        type=Path,
        default=Path.cwd(),
        help="Path to Condor job directory containing stdout/, error/, and files/ (default: current directory)"
    )
    parser.add_argument(
        "-l", "--log-file",
        type=Path,
        default=None,
        help="Optional path to a log/summary file listing failed .err files (e.g. log.txt)"
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=Path,
        default=None,
        help="Directory to write output list files (default: condor directory)"
    )
    parser.add_argument(
        "--no-fc-check",
        action="store_true",
        help="Skip querying PostgreSQL FileCatalog for all candidate files in the input lists"
    )
    parser.add_argument(
        "--check-lustre",
        action="store_true",
        help="Check direct existence on Lustre filesystem for files registered in FileCatalog"
    )
    parser.add_argument(
        "-f", "--filter-input-dir",
        type=Path,
        default=None,
        help="Optional path to directory of input .list files to filter out missing run-segments"
    )
    parser.add_argument(
        "--removed-dir-name",
        type=str,
        default=None,
        help="Optional folder name for extracted missing lines (default: appends '-missing-catalog')"
    )
    parser.add_argument(
        "-n", "--dry-run",
        action="store_true",
        help="Perform a dry run without deleting logs or modifying files on disk"
    )
    parser.add_argument(
        "--clean-logs",
        action="store_true",
        help="Delete stdout/ and error/ log files corresponding to the aborted jobs"
    )
    parser.add_argument(
        "--create-resubmit",
        action="store_true",
        help="Create resubmission list files (resubmit_all_aborted.list and resubmit_available.list)"
    )
    parser.add_argument(
        "-x", "--execute-submit",
        action="store_true",
        help="Immediately submit the resubmit_available.list job via condor_submit"
    )
    parser.add_argument(
        "-j", "--threads",
        type=int,
        default=min(32, (os.cpu_count() or 4)),
        help="Number of threads for scanning log files (default: 32 or cpu count)"
    )
    return parser.parse_args()


def extract_job_info_from_err_name(fname: str) -> tuple[str, str] | None:
    """Extracts (cluster_id, process_id) from job-<cluster>-<proc>.err or .out."""
    m = re.search(r'job-(\d+)-(\d+)\.(?:err|out)', fname)
    if m:
        return m.group(1), m.group(2)
    return None


def inspect_stdout_file(out_path: Path) -> dict | None:
    """
    Parses a single stdout file for FileCatalog retrieval failures.
    Returns a dict with failure details or None if no retrieval failure.
    """
    if not out_path.is_file():
        return None

    try:
        content = out_path.read_text(errors='ignore')
    except Exception as e:
        return None

    # Check for getinputfiles / FileCatalog failure signature
    if "could not retrieve" not in content and "getinputfiles.pl failed" not in content:
        return None

    # Extract missing file reported in stdout
    m_fail = re.search(r'could not retrieve\s+(\S+)\s+from filecatalog', content)
    reported_lfn = m_fail.group(1) if m_fail else None

    # Extract run and segment from reported LFN
    reported_runseg = None
    if reported_lfn:
        seg_m = re.search(r'(\d{8}-\d{5})', reported_lfn)
        if seg_m:
            reported_runseg = seg_m.group(1)

    # Extract input list path
    m_inp = re.search(r'Reading inputs from:\s*(\S+)', content)
    input_list_str = m_inp.group(1) if m_inp else None

    return {
        "out_file": out_path,
        "reported_lfn": reported_lfn,
        "reported_runseg": reported_runseg,
        "input_list_str": input_list_str,
    }


def find_candidate_failures(condor_dir: Path, log_file: Path | None, threads: int) -> list[dict]:
    """
    Finds all jobs that failed due to FileCatalog / getinputfiles retrieval issues.
    Supports:
      - Direct failure logs (e.g. failure-log.txt with lines like 'getinputfiles failure for <file>.list ...')
      - Error summary files (e.g. log.txt with lines like 'error/job-206510-4611.err')
      - Raw lists of .list or .err/.out files
      - Auto-detection of <condor_dir>/output/failures/failure-log.txt or <condor_dir>/failures/failure-log.txt
      - Automatic parallel scan of stdout/ if no log file is provided.
    """
    stdout_dir = condor_dir / "stdout"

    # Auto-detect failure-log.txt if not explicitly passed
    if log_file is None:
        candidate_auto_logs = [
            condor_dir / "output" / "failures" / "failure-log.txt",
            condor_dir / "failures" / "failure-log.txt",
            condor_dir / "failure-log.txt",
        ]
        for auto_candidate in candidate_auto_logs:
            if auto_candidate.is_file() and auto_candidate.stat().st_size > 0:
                print(f"Auto-detected failure log at: {auto_candidate}")
                log_file = auto_candidate
                break

    out_files_to_check = []
    list_files_found = set()

    if log_file and log_file.is_file():
        print(f"Reading error/failure log from: {log_file}")
        with log_file.open('r', errors='ignore') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                # Pattern 1: Check for lines mentioning a .list file (e.g. failure-log.txt)
                m_list = re.search(r'(\S+\.list)', line)
                if m_list:
                    list_name = Path(m_list.group(1)).name
                    list_files_found.add(list_name)

                # Pattern 2: Check for directly reported .root file (e.g. 'for <file>.root in <list>')
                m_root = re.search(r'for\s+(\S+\.root)', line)
                if m_root:
                    root_name = m_root.group(1)
                    if 'direct_reported_lfns' not in locals():
                        direct_reported_lfns = set()
                    direct_reported_lfns.add(root_name)

                # Pattern 3: Check for job-<cluster>-<proc>.(err|out)
                pair = extract_job_info_from_err_name(line)
                if pair:
                    cid, pid = pair
                    candidate_out = stdout_dir / f"job-{cid}-{pid}.out"
                    if candidate_out.is_file():
                        out_files_to_check.append(candidate_out)

        # If list files were discovered from failure-log.txt, map them to stdout files or records
        if list_files_found:
            print(f"Discovered {len(list_files_found)} unique input list files in failure log.")
            # Map list filename -> process ID using jobs-*.list or jobs.list if available
            mapping = {}
            for j_file in sorted(condor_dir.glob("jobs*.list")):
                try:
                    with j_file.open('r', errors='ignore') as f:
                        for lineno, l in enumerate(f):
                            l_name = Path(l.strip()).name
                            if l_name in list_files_found and l_name not in mapping:
                                mapping[l_name] = lineno
                except Exception:
                    pass

            # Detect known cluster IDs in stdout_dir
            cluster_ids = set()
            if stdout_dir.is_dir():
                try:
                    with os.scandir(stdout_dir) as it:
                        for entry in it:
                            parts = entry.name.split('-')
                            if len(parts) >= 3 and parts[0] == "job":
                                cluster_ids.add(parts[1])
                except Exception:
                    pass

            # For each list file, locate its verified stdout file
            matched_list_names = set()
            for l_name in list_files_found:
                pid = mapping.get(l_name)
                if pid is not None and cluster_ids:
                    for cid in cluster_ids:
                        candidate = stdout_dir / f"job-{cid}-{pid}.out"
                        if candidate.is_file():
                            try:
                                with candidate.open('r', errors='ignore') as fp:
                                    head_text = "".join([fp.readline() for _ in range(6)])
                                    if l_name in head_text:
                                        out_files_to_check.append(candidate)
                                        matched_list_names.add(l_name)
                                        break
                            except Exception:
                                pass

            # For any list file that didn't match a stdout file, resolve its path directly
            unmatched = list_files_found - matched_list_names
            direct_records = []
            for l_name in unmatched:
                resolved_path = None
                for candidate_path in [
                    condor_dir / "files" / l_name,
                    condor_dir / l_name,
                    Path(l_name),
                ]:
                    if candidate_path.is_file():
                        resolved_path = candidate_path
                        break
                if resolved_path:
                    direct_records.append({
                        "out_file": None,
                        "reported_lfn": None,
                        "reported_runseg": None,
                        "input_list_str": str(resolved_path),
                    })

    if not out_files_to_check and not list_files_found:
        print(f"Scanning stdout directory: {stdout_dir} ...")
        if stdout_dir.is_dir():
            with os.scandir(stdout_dir) as it:
                for entry in it:
                    if entry.name.endswith(".out"):
                        out_files_to_check.append(Path(entry.path))

    # De-duplicate stdout files
    out_files_to_check = sorted(list(set(out_files_to_check)))
    if out_files_to_check:
        print(f"Checking {len(out_files_to_check)} output logs for retrieval errors (threads: {threads})...")

    failed_records = []
    with ThreadPoolExecutor(max_workers=threads) as executor:
        for res in executor.map(inspect_stdout_file, out_files_to_check, chunksize=128):
            if res is not None:
                failed_records.append(res)

    # Append direct records from list files if any
    if 'direct_records' in locals() and direct_records:
        failed_records.extend(direct_records)

    # Sort deterministically by cluster and process id
    def sort_key(rec):
        if rec.get("out_file"):
            pair = extract_job_info_from_err_name(rec["out_file"].name)
            return (int(pair[0]), int(pair[1])) if pair else (0, 0)
        return (999999, 999999)

    failed_records.sort(key=sort_key)
    return failed_records


def query_filecatalog_batch(lfns: list[str], chunk_size: int = 150) -> dict[str, str]:
    """
    Queries FileCatalog via psql for a list of LFNs in batches.
    Returns a dict mapping {lfn: full_file_path}.
    """
    found_records = {}
    total = len(lfns)
    if total == 0:
        return found_records

    print(f"Querying FileCatalog for {total} files across candidate jobs...")
    for i in range(0, total, chunk_size):
        chunk = lfns[i:i + chunk_size]
        in_clause = ",".join(f"'{lfn}'" for lfn in chunk)
        query = f"select lfn, full_file_path from files where lfn in ({in_clause});"
        cmd = ["psql", "-t", "-A", "-F", "|", "FileCatalog", "-c", query]
        try:
            res = subprocess.run(cmd, capture_output=True, text=True, check=False)
            if res.returncode == 0:
                for line in res.stdout.strip().splitlines():
                    if "|" in line:
                        parts = line.split("|", 1)
                        found_records[parts[0].strip()] = parts[1].strip()
            else:
                print(f"Warning: psql query returned non-zero exit code: {res.stderr.strip()}", file=sys.stderr)
        except FileNotFoundError:
            print("Error: 'psql' command not found in PATH. Skipping FileCatalog check.", file=sys.stderr)
            return found_records
        except Exception as e:
            print(f"Error running psql: {e}", file=sys.stderr)
            return found_records

    return found_records


def filter_input_list_dir(
    filter_dir: Path,
    bad_segments: list[str],
    custom_removed_dir_name: str | None = None,
    dry_run: bool = False
):
    """
    Filters out bad run-segments from input .list files in filter_dir,
    saving removed lines into a new directory. Deletes empty input list files.
    """
    filter_dir = filter_dir.resolve()
    if not filter_dir.is_dir():
        print(f"Error: Specified filter directory '{filter_dir}' is not a directory.", file=sys.stderr)
        return

    dir_name = filter_dir.name
    if custom_removed_dir_name:
        new_dir_name = custom_removed_dir_name
    elif "-pro" in dir_name:
        idx = dir_name.find("-pro")
        new_dir_name = dir_name[:idx] + "-missing-catalog" + dir_name[idx:]
    else:
        new_dir_name = dir_name + "-missing-catalog"

    removed_dir = filter_dir.parent / new_dir_name
    if not dry_run:
        removed_dir.mkdir(parents=True, exist_ok=True)

    bad_set = set(bad_segments)
    total_files_processed = 0
    total_files_modified = 0
    total_files_removed = 0
    total_lines_removed = 0

    list_files = sorted(list(filter_dir.glob("*.list")))
    if not list_files:
        list_files = sorted([f for f in filter_dir.iterdir() if f.is_file()])

    for list_file in list_files:
        total_files_processed += 1
        kept_lines = []
        removed_lines = []

        with list_file.open('r', errors='ignore') as f:
            for line in f:
                raw_line = line.rstrip('\r\n')
                if not raw_line.strip():
                    continue
                matches = re.findall(r'(\d{8}-\d{5})', raw_line)
                if any(m in bad_set for m in matches):
                    removed_lines.append(raw_line)
                else:
                    kept_lines.append(raw_line)

        if removed_lines:
            total_lines_removed += len(removed_lines)
            total_files_modified += 1
            if not dry_run:
                removed_file = removed_dir / list_file.name
                with removed_file.open('w') as f:
                    for l in removed_lines:
                        f.write(f"{l}\n")

            if not kept_lines:
                if not dry_run:
                    list_file.unlink()
                total_files_removed += 1
            else:
                if not dry_run:
                    with list_file.open('w') as f:
                        for l in kept_lines:
                            f.write(f"{l}\n")

    print("\n" + "=" * 60)
    print("INPUT LIST FILTERING SUMMARY" + (" (DRY RUN)" if dry_run else ""))
    print("=" * 60)
    print(f"Input directory:            {filter_dir}")
    print(f"Removed lines directory:    {removed_dir}")
    print(f"Total .list files inspected:{total_files_processed}")
    print(f"Files modified:             {total_files_modified}")
    print(f"Files fully empty & deleted:{total_files_removed}")
    print(f"Total lines/segments removed:{total_lines_removed}")
    print("=" * 60)


def main():
    args = parse_arguments()

    condor_dir = args.condor_dir.resolve()
    if not condor_dir.is_dir():
        print(f"Error: Condor directory '{condor_dir}' does not exist.", file=sys.stderr)
        sys.exit(1)

    output_dir = (args.output_dir or condor_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("CONDOR MISSING FILECATALOG / LUSTRE CHECKER")
    print(f"Condor Directory: {condor_dir}")
    print(f"Output Directory: {output_dir}")
    if args.log_file:
        print(f"Error Log File:   {args.log_file.resolve()}")
    print("=" * 70)

    # 1. Identify failed jobs
    failed_records = find_candidate_failures(condor_dir, args.log_file, args.threads)
    if not failed_records:
        print("No jobs with FileCatalog retrieval failures found. Exiting.")
        sys.exit(0)

    print(f"Found {len(failed_records)} jobs with FileCatalog / getinputfiles failures.\n")

    # 2. Extract reported failures and candidate input list files
    reported_runsegments = []
    reported_lfns = []
    candidate_lfns = []
    input_lists_found = []

    for rec in failed_records:
        if rec["reported_runseg"]:
            reported_runsegments.append(rec["reported_runseg"])
        if rec["reported_lfn"]:
            reported_lfns.append(rec["reported_lfn"])

        inp_str = rec["input_list_str"]
        if inp_str:
            inp_path = Path(inp_str)
            if not inp_path.is_file() and (condor_dir / "files" / inp_path.name).is_file():
                inp_path = condor_dir / "files" / inp_path.name

            if inp_path.is_file():
                input_lists_found.append(inp_path)
                with inp_path.open('r', errors='ignore') as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        parts = line.split(',')
                        # The first column is the DST LFN
                        candidate_lfns.append(parts[0].strip())

    unique_reported_segments = sorted(list(set(reported_runsegments)))
    unique_candidate_lfns = sorted(list(set(candidate_lfns)))

    print("-" * 70)
    print(f"Jobs with FileCatalog errors:      {len(failed_records)}")
    print(f"Segments explicitly reported in stdout: {len(unique_reported_segments)}")
    print(f"Total candidate files across failed jobs: {len(unique_candidate_lfns)} (across {len(set(input_lists_found))} input lists)")
    print("-" * 70)

    # 3. Query FileCatalog if enabled
    fc_found_map = {}
    all_missing_lfns = []
    all_missing_segments = []

    if not args.no_fc_check:
        fc_found_map = query_filecatalog_batch(unique_candidate_lfns)
        all_missing_lfns = [lfn for lfn in unique_candidate_lfns if lfn not in fc_found_map]

        for lfn in all_missing_lfns:
            seg_m = re.search(r'(\d{8}-\d{5})', lfn)
            if seg_m:
                all_missing_segments.append(seg_m.group(1))
            else:
                all_missing_segments.append(lfn)

        all_missing_segments = sorted(list(set(all_missing_segments)))

        # Optional Lustre check for the ones found in FC
        if args.check_lustre and fc_found_map:
            print("Checking Lustre filesystem existence for files registered in FileCatalog...")
            missing_on_lustre = []
            for lfn, fpath in fc_found_map.items():
                if not os.path.exists(fpath):
                    missing_on_lustre.append((lfn, fpath))
            if missing_on_lustre:
                print(f"Warning: {len(missing_on_lustre)} files are in FileCatalog but NOT found on Lustre disk!")
                for lfn, fpath in missing_on_lustre[:5]:
                    print(f"  {lfn} -> {fpath}")
            else:
                print("All files found in FileCatalog exist on Lustre disk.")
    else:
        all_missing_segments = unique_reported_segments
        all_missing_lfns = sorted(list(set(reported_lfns)))

    # 4. Write output files
    # A) Reported missing segments (from stdout logs)
    reported_file = output_dir / "reported_missing_segments.list"
    with reported_file.open('w') as f:
        for seg in unique_reported_segments:
            f.write(f"{seg}\n")
    print(f"\nWrote {len(unique_reported_segments)} directly reported segments to: {reported_file}")

    # B) All missing segments (from FileCatalog query across all jobs)
    all_missing_file = output_dir / "all_missing_segments.list"
    with all_missing_file.open('w') as f:
        for seg in all_missing_segments:
            f.write(f"{seg}\n")
    print(f"Wrote {len(all_missing_segments)} total missing segments to:       {all_missing_file}")

    # C) Missing filenames (.root)
    missing_files_path = output_dir / "all_missing_files.txt"
    with missing_files_path.open('w') as f:
        for lfn in all_missing_lfns:
            f.write(f"{lfn}\n")
    print(f"Wrote {len(all_missing_lfns)} missing filenames to:              {missing_files_path}")

    # 5. Display breakdown
    print("\n" + "=" * 70)
    print("DETAILED BREAKDOWN BY JOB (SAMPLE / ALL)")
    print("=" * 70)
    print(f"{'Job Log':<24} {'Reported Run-Segment':<20} {'Reported Missing LFN'}")
    print("-" * 70)
    for rec in failed_records:
        fname = rec["out_file"].name if rec.get("out_file") else Path(rec.get("input_list_str", "N/A")).name
        seg = rec["reported_runseg"] or "N/A"
        lfn = rec["reported_lfn"] or "N/A"
        print(f"{fname:<24} {seg:<20} {lfn}")

    # 6. Filter input list directory if requested
    if args.filter_input_dir:
        # Use all_missing_segments for complete filtering so resubmissions don't fail again
        filter_input_list_dir(
            args.filter_input_dir,
            all_missing_segments,
            args.removed_dir_name,
            dry_run=args.dry_run
        )

    # 7. Resubmission lists preparation
    input_list_details = {}
    for inp_p in sorted(list(set(input_lists_found))):
        try:
            with inp_p.open('r', errors='ignore') as fp:
                lfns = [line.strip().split(',')[0].strip() for line in fp if line.strip()]
                avail_count = sum(1 for lfn in lfns if lfn in fc_found_map)
                input_list_details[inp_p] = (avail_count, len(lfns))
        except Exception:
            pass

    processable_lists = [p for p, (avail, tot) in input_list_details.items() if avail > 0]
    all_aborted_lists = list(input_list_details.keys())

    # Keep the extra jobs.list columns of each job (e.g. the Q vector calib of event_qa): a
    # resubmission with the chunk path alone leaves $(input_calib) empty and shifts the arguments
    job_columns = {}
    for j_file in sorted(condor_dir.glob("jobs*.list")):
        try:
            with j_file.open('r', errors='ignore') as f:
                for l in f:
                    cols = [c.strip() for c in l.strip().split(',')]
                    if cols and cols[0]:
                        job_columns.setdefault(Path(cols[0]).name, cols[1:])
        except Exception:
            pass

    def resubmit_line(p):
        return ",".join([str(p)] + job_columns.get(p.name, []))

    resubmit_all_path = output_dir / "resubmit_all_aborted.list"
    resubmit_avail_path = output_dir / "resubmit_available.list"

    with resubmit_all_path.open('w') as fp:
        for p in all_aborted_lists:
            fp.write(f"{resubmit_line(p)}\n")
    print(f"\nWrote {len(all_aborted_lists)} aborted input lists to:     {resubmit_all_path}")

    with resubmit_avail_path.open('w') as fp:
        for p in processable_lists:
            fp.write(f"{resubmit_line(p)}\n")
    print(f"Wrote {len(processable_lists)} processable lists (with valid segments) to: {resubmit_avail_path}")

    # queue variables from the number of jobs.list columns: event_qa (input_dst[,input_calib])
    n_columns = {1 + len(job_columns.get(p.name, [])) for p in all_aborted_lists}
    known_queue_vars = {1: "input_dst", 2: "input_dst,input_calib"}
    queue_vars = known_queue_vars.get(next(iter(n_columns))) if len(n_columns) == 1 else None
    if queue_vars is None:
        print(f"\nWarning: jobs.list column counts {sorted(n_columns) or 'unknown'} do not match a known layout; "
              "take the -queue variables from the original submission (no automatic submit).", file=sys.stderr)

    # 8. Clean logs if requested
    if args.clean_logs:
        print("\n" + "=" * 70)
        print("CLEANING LOGS FOR ABORTED JOBS" + (" (DRY RUN)" if args.dry_run else ""))
        print("=" * 70)
        cleaned_out = 0
        cleaned_err = 0
        for rec in failed_records:
            out_f = rec.get("out_file")
            if out_f and out_f.is_file():
                cid_pid = extract_job_info_from_err_name(out_f.name)
                if not args.dry_run:
                    try:
                        out_f.unlink()
                        cleaned_out += 1
                    except Exception as e:
                        print(f"Could not remove {out_f}: {e}")
                else:
                    cleaned_out += 1
                if cid_pid:
                    cid, pid = cid_pid
                    err_f = condor_dir / "error" / f"job-{cid}-{pid}.err"
                    if err_f.is_file():
                        if not args.dry_run:
                            try:
                                err_f.unlink()
                                cleaned_err += 1
                            except Exception as e:
                                print(f"Could not remove {err_f}: {e}")
                        else:
                            cleaned_err += 1
        print(f"Cleaned {cleaned_out} stdout logs and {cleaned_err} error logs.")

    # 9. Print resubmission summary
    print("\n" + "=" * 70)
    print("RESUBMISSION SUMMARY & COMMANDS")
    print("=" * 70)
    print(f"Aborted jobs analyzed:          {len(failed_records)}")
    print(f"Lists with valid segments:      {len(processable_lists)} (can produce QA outputs)")
    print(f"Lists with 0 segments in FC:    {len(all_aborted_lists) - len(processable_lists)} (all segments missing)")
    print("-" * 70)
    shown_vars = queue_vars or "<queue variables of the original submission>"
    print("Recommended Condor resubmit command (for processable jobs):")
    print(f"  cd {condor_dir} && condor_submit genFun4All.sub -queue \"{shown_vars} from {resubmit_avail_path}\"")
    print("\nOr to resubmit all aborted lists (empty ones will skip via updated genFun4All_EventQA.sh):")
    print(f"  cd {condor_dir} && condor_submit genFun4All.sub -queue \"{shown_vars} from {resubmit_all_path}\"")
    print("=" * 70)

    # 10. Execute submit if requested
    if args.execute_submit and queue_vars is None:
        print("\nNot submitting: unknown jobs.list layout (see warning above).", file=sys.stderr)
    elif args.execute_submit:
        target_list = resubmit_avail_path if processable_lists else resubmit_all_path
        if args.dry_run:
            print(f"\n[DRY RUN] Would execute: condor_submit genFun4All.sub -queue \"{queue_vars} from {target_list}\"")
        else:
            submit_cmd = ["condor_submit", "genFun4All.sub", "-queue", f"{queue_vars} from {target_list}"]
            print(f"\nExecuting: {' '.join(submit_cmd)} (in {condor_dir}) ...")
            try:
                res = subprocess.run(submit_cmd, cwd=condor_dir, capture_output=True, text=True, check=False)
                print(res.stdout)
                if res.stderr:
                    print(res.stderr, file=sys.stderr)
            except Exception as e:
                print(f"Failed to execute condor_submit: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
