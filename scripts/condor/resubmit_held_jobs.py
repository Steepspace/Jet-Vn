#!/usr/bin/env python3
"""
resubmit_held_jobs.py

Identifies HTCondor jobs held due to high memory usage, extracts their corresponding
entries from a jobs list file, creates a new subset jobs list file with updated memory,
removes the held jobs and their stdout/err logs, and prints the new submission command
to run on the current node.

Author: Antigravity
Date: October 2026
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple


@dataclass
class HeldJob:
    cluster_id: int
    proc_id: int
    owner: str = ""
    hold_reason: str = ""

    @property
    def job_id(self) -> str:
        return f"{self.cluster_id}.{self.proc_id}"


def format_memory(mem: str) -> str:
    """
    Normalizes memory strings (e.g. '1' -> '1GB', '1.5' -> '1.5GB', '1024MB' -> '1024MB').
    """
    mem = str(mem).strip()
    try:
        val = float(mem)
        if val.is_integer():
            return f"{int(val)}GB"
        return f"{val}GB"
    except ValueError:
        pass

    m = re.match(r"^([\d\.]+)\s*([a-zA-Z]+)$", mem)
    if m:
        num, unit = m.group(1), m.group(2).upper()
        if unit in ("G", "GB"):
            return f"{num}GB"
        if unit in ("M", "MB"):
            return f"{num}MB"
        if unit in ("T", "TB"):
            return f"{num}TB"
    return mem


def parse_condor_hold_output(text: str) -> Tuple[List[HeldJob], Optional[str]]:
    """
    Parses output from `condor_q -hold` or log file containing condor_q output.
    Returns (list of HeldJob objects, detected schedd hostname or None).
    """
    held_jobs = []
    schedd = None

    schedd_match = re.search(r"--\s*Schedd:\s*([\w\.-]+)", text)
    if schedd_match:
        schedd = schedd_match.group(1).split(".")[0]

    for line in text.splitlines():
        line = line.strip()
        # Look for lines starting with <cluster>.<proc> <owner>
        m = re.match(r"^(\d+)\.(\d+)\s+(\S+)\s+(.*)$", line)
        if m:
            cluster_str, proc_str, owner, reason = m.groups()
            if owner.upper() == "OWNER" or cluster_str.upper() == "ID":
                continue
            cluster_id = int(cluster_str)
            proc_id = int(proc_str)
            held_jobs.append(
                HeldJob(
                    cluster_id=cluster_id,
                    proc_id=proc_id,
                    owner=owner,
                    hold_reason=reason.strip(),
                )
            )

    return held_jobs, schedd


def fetch_held_jobs(
    schedd: Optional[str] = None,
) -> Tuple[List[HeldJob], Optional[str]]:
    """
    Retrieves held jobs by querying live HTCondor via `condor_q -hold`.
    Returns (list of HeldJob objects, detected schedd hostname or None).
    """
    condor_q_bin = shutil.which("condor_q")
    if not condor_q_bin:
        print("[!] Error: 'condor_q' executable not found in PATH.", file=sys.stderr)
        return [], schedd

    cmd = [condor_q_bin, "-hold"]
    if schedd:
        cmd.extend(["-name", schedd])
    print(f"[*] Querying live HTCondor: {' '.join(cmd)}")
    try:
        res = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
        if res.returncode == 0 and res.stdout.strip():
            jobs, detected_schedd = parse_condor_hold_output(res.stdout)
            return jobs, schedd or detected_schedd
        else:
            if res.stderr.strip():
                print(f"[!] Warning: condor_q stderr: {res.stderr.strip()}")
    except Exception as e:
        print(f"[!] Error querying live condor_q: {e}", file=sys.stderr)

    return [], schedd


def find_submit_file(job_dir: Path, custom_sub: Optional[Path] = None) -> Optional[Path]:
    """
    Finds the condor submit description file (.sub) in the job directory.
    """
    if custom_sub and custom_sub.is_file():
        return custom_sub.resolve()

    # Look for standard names first
    candidates = [
        job_dir / "genFun4All.sub",
        job_dir / "condor.sub",
        job_dir / "submit.sub",
    ]
    for cand in candidates:
        if cand.is_file():
            return cand

    # Any .sub file that isn't a _resubmit.sub
    sub_files = [f for f in job_dir.glob("*.sub") if not f.name.endswith("_resubmit.sub")]
    if sub_files:
        return sub_files[0]

    return None


def extract_queue_variable(sub_file: Optional[Path], default: str = "input_dst") -> str:
    """
    Finds the $(var) used in arguments from the submit file, e.g. $(input_dst).
    """
    if not sub_file or not sub_file.is_file():
        return default

    try:
        content = sub_file.read_text(encoding="utf-8", errors="ignore")
        for line in content.splitlines():
            line = line.strip()
            if line.startswith("arguments"):
                # Search for $(var)
                m = re.findall(r"\$\((\w+)\)", line)
                for var in m:
                    if var.lower() not in ("clusterid", "process", "cluster", "item"):
                        return var
    except Exception:
        pass
    return default


def extract_log_directory(sub_file: Optional[Path], default: str = "/tmp") -> Path:
    """
    Extracts the directory where HTCondor writes the job log from the .sub file.
    """
    if sub_file and sub_file.is_file():
        try:
            content = sub_file.read_text(encoding="utf-8", errors="ignore")
            for line in content.splitlines():
                line = line.strip()
                if line.startswith("log"):
                    m = re.search(r"log\s*=\s*(\S+)", line)
                    if m:
                        log_path = Path(m.group(1))
                        return log_path.parent
        except Exception:
            pass
    return Path(default)


def remove_condor_jobs(
    job_ids: List[str],
    schedd: Optional[str] = None,
    dry_run: bool = False,
) -> bool:
    """
    Removes jobs from the HTCondor queue using `condor_rm`.
    """
    if not job_ids:
        return True

    cmd = ["condor_rm"]
    if schedd:
        cmd.extend(["-name", schedd])
    cmd.extend(job_ids)

    cmd_str = " ".join(cmd)
    if dry_run:
        print(f"[DRY-RUN] Would remove held jobs with command:\n  {cmd_str}")
        return True

    print(f"[*] Removing held jobs from Condor queue: {cmd_str}")
    try:
        res = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
            timeout=60,
        )
        if res.returncode == 0:
            print(f"[+] Successfully removed {len(job_ids)} jobs from queue.")
            if res.stdout.strip():
                print(f"    {res.stdout.strip()}")
            return True
        else:
            # If failed and schedd wasn't used or failed, try without or report
            print(f"[!] Warning: condor_rm returned exit code {res.returncode}")
            if res.stderr.strip():
                print(f"    STDERR: {res.stderr.strip()}")
            if res.stdout.strip():
                print(f"    STDOUT: {res.stdout.strip()}")
            return False
    except FileNotFoundError:
        print(f"[!] Warning: 'condor_rm' executable not found in PATH.")
        print(f"    Please run manually: {cmd_str}")
        return False
    except Exception as e:
        print(f"[!] Error running condor_rm: {e}")
        return False


def clean_job_logs(
    job_dir: Path,
    cluster_id: int,
    proc_ids: List[int],
    log_dir: Optional[Path] = None,
    dry_run: bool = False,
) -> int:
    """
    Deletes stdout, error, and condor log files for the specified (cluster_id, proc_id) jobs.
    Returns the total count of deleted files.
    """
    stdout_dir = job_dir / "stdout"
    error_dir = job_dir / "error"

    deleted_count = 0
    for proc_id in proc_ids:
        candidates = [
            stdout_dir / f"job-{cluster_id}-{proc_id}.out",
            error_dir / f"job-{cluster_id}-{proc_id}.err",
        ]
        if log_dir:
            candidates.append(log_dir / f"job-{cluster_id}-{proc_id}.log")

        for f in candidates:
            if f.is_file():
                if dry_run:
                    print(f"[DRY-RUN] Would delete: {f}")
                    deleted_count += 1
                else:
                    try:
                        f.unlink()
                        print(f"[-] Deleted log file: {f}")
                        deleted_count += 1
                    except OSError as e:
                        print(f"[!] Failed to delete {f}: {e}")

    return deleted_count


def create_resubmit_sub_file(
    original_sub: Path,
    new_memory: str,
    output_sub: Path,
    dry_run: bool = False,
) -> None:
    """
    Creates an updated submit file with new request_memory setting.
    """
    content = original_sub.read_text(encoding="utf-8")

    if re.search(r"^\s*request_memory\s*=", content, re.MULTILINE):
        new_content = re.sub(
            r"^\s*(request_memory\s*=\s*).*$",
            rf"\g<1>{new_memory}",
            content,
            flags=re.MULTILINE,
        )
    else:
        new_content = content + f"\nrequest_memory = {new_memory}\n"

    if dry_run:
        print(f"[DRY-RUN] Would create updated submit file at: {output_sub}")
        print(f"          With request_memory = {new_memory}")
    else:
        output_sub.write_text(new_content, encoding="utf-8")
        print(f"[+] Created updated submit file: {output_sub} (request_memory = {new_memory})")


def main():
    parser = argparse.ArgumentParser(
        description="Resubmit HTCondor jobs held due to memory limits, clean up failed logs, and generate resubmission commands.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Dry run to preview actions without removing jobs or files:
  python3 resubmit_held_jobs.py /gpfs02/sphenix/user/anarde/sEPD-Study/sepd_calib/10-01-26/jobs-7.list 1.0GB -n

  # Execute cleanup, generate resubmit list, and print submission command:
  python3 resubmit_held_jobs.py /gpfs02/sphenix/user/anarde/sEPD-Study/sepd_calib/10-01-26/jobs-7.list 1.0GB

  # Execute cleanup and automatically submit:
  python3 resubmit_held_jobs.py /gpfs02/sphenix/user/anarde/sEPD-Study/sepd_calib/10-01-26/jobs-7.list 1.0GB -x
        """,
    )

    parser.add_argument(
        "jobs_file",
        type=Path,
        help="Path to the original jobs list file (e.g. jobs-3.list)",
    )
    parser.add_argument(
        "memory",
        type=str,
        help="New request_memory value (e.g. '1GB', '1.5GB', '1.0', '1024MB')",
    )
    parser.add_argument(
        "-o",
        "--output-jobs-file",
        type=Path,
        default=None,
        help="Path for the new subset jobs list file (default: <jobs_file_stem>-resubmit.list)",
    )
    parser.add_argument(
        "-s",
        "--sub-file",
        type=Path,
        default=None,
        help="Path to condor submit description file (.sub) (default: auto-detected in job dir)",
    )
    parser.add_argument(
        "-c",
        "--cluster-id",
        type=int,
        default=None,
        help="Specify HTCondor cluster ID if multiple clusters exist",
    )
    parser.add_argument(
        "--schedd",
        type=str,
        default=None,
        help="HTCondor schedd daemon name (e.g. sphnxuser01)",
    )
    parser.add_argument(
        "--keep-held",
        action="store_true",
        help="Do not remove held jobs from condor queue (skip condor_rm)",
    )
    parser.add_argument(
        "--keep-logs",
        action="store_true",
        help="Do not delete stdout/error logs of held jobs",
    )
    parser.add_argument(
        "-n",
        "--dry-run",
        action="store_true",
        help="Dry run: print actions without deleting files, modifying disk, or removing jobs",
    )
    parser.add_argument(
        "-x",
        "--execute-submit",
        action="store_true",
        help="Immediately submit the new jobs after cleanup",
    )

    args = parser.parse_args()

    # 1. Validate jobs_file
    jobs_file = args.jobs_file.expanduser().resolve()
    if not jobs_file.is_file():
        print(f"[!] Error: Jobs file '{jobs_file}' does not exist.", file=sys.stderr)
        sys.exit(1)

    job_dir = jobs_file.parent
    new_memory = format_memory(args.memory)

    print("=" * 80)
    print(f" Condor Held Jobs Resubmitter")
    print("=" * 80)
    print(f"Jobs file:           {jobs_file}")
    print(f"Job directory:       {job_dir}")
    print(f"New Memory Request:  {new_memory}")
    if args.dry_run:
        print("[!] DRY-RUN MODE ENABLED: No files or queue jobs will be modified/deleted.")
    print("-" * 80)

    # Read the original jobs file
    jobs_lines = [line.strip() for line in jobs_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    total_lines = len(jobs_lines)
    print(f"[*] Read {total_lines} total lines from '{jobs_file.name}'")

    # 2. Retrieve held jobs live from HTCondor
    held_jobs, detected_schedd = fetch_held_jobs(schedd=args.schedd)

    if not held_jobs:
        print(f"[!] Error: No held jobs found in HTCondor queue via 'condor_q -hold'.", file=sys.stderr)
        sys.exit(1)

    active_schedd = args.schedd or detected_schedd
    if active_schedd:
        print(f"[*] Schedd daemon:   {active_schedd}")

    # 3. Filter held jobs to match this jobs file & cluster
    # Group by cluster ID
    clusters = sorted(list({j.cluster_id for j in held_jobs}))
    target_cluster = args.cluster_id

    if target_cluster is None:
        if len(clusters) == 1:
            target_cluster = clusters[0]
        else:
            # Check which cluster matches stdout files in job_dir
            matching_clusters = []
            for cid in clusters:
                sample_out = list((job_dir / "stdout").glob(f"job-{cid}-*.out"))
                if sample_out:
                    matching_clusters.append(cid)
            if len(matching_clusters) == 1:
                target_cluster = matching_clusters[0]
            else:
                print(f"[!] Multiple clusters found with held jobs: {clusters}")
                print(f"    Please specify which cluster to process using --cluster-id <ID>")
                sys.exit(1)

    print(f"[*] Target Cluster ID: {target_cluster}")

    target_held_jobs = [j for j in held_jobs if j.cluster_id == target_cluster]
    # Filter for memory-related hold reasons if applicable
    memory_held_jobs = [
        j for j in target_held_jobs
        if "memory" in j.hold_reason.lower() or "34/102" in j.hold_reason or not j.hold_reason
    ]
    if memory_held_jobs:
        selected_jobs = memory_held_jobs
    else:
        selected_jobs = target_held_jobs

    print(f"[*] Identified {len(selected_jobs)} held jobs for Cluster {target_cluster}")

    # Validate process indices against lines in jobs_file
    matched_entries = []
    job_ids_to_remove = []
    proc_ids_to_clean = []

    for job in sorted(selected_jobs, key=lambda x: x.proc_id):
        pid = job.proc_id
        if pid < total_lines:
            entry_line = jobs_lines[pid]
            matched_entries.append((pid, entry_line))
            job_ids_to_remove.append(job.job_id)
            proc_ids_to_clean.append(pid)
        else:
            print(f"[!] Warning: ProcId {pid} is out of bounds for {jobs_file.name} (length {total_lines})")

    if not matched_entries:
        print(f"[!] Error: No held jobs matched indices in '{jobs_file.name}'.", file=sys.stderr)
        sys.exit(1)

    # 4. Write new subset jobs list file
    output_jobs_file = (
        args.output_jobs_file.expanduser().resolve()
        if args.output_jobs_file
        else job_dir / f"{jobs_file.stem}-resubmit.list"
    )

    print("-" * 80)
    print(f"[*] Preparing new jobs file: {output_jobs_file}")
    resubmit_content = "\n".join(entry for _, entry in matched_entries) + "\n"

    if args.dry_run:
        print(f"[DRY-RUN] Would write {len(matched_entries)} lines to: {output_jobs_file}")
    else:
        output_jobs_file.write_text(resubmit_content, encoding="utf-8")
        print(f"[+] Wrote {len(matched_entries)} lines to: {output_jobs_file}")

    # Display entries preview
    print(f"[*] Entries to resubmit ({len(matched_entries)} jobs):")
    for pid, entry in matched_entries:
        print(f"    - Job {target_cluster}.{pid:04d} -> {entry}")

    # 5. Handle submit file and updated request_memory
    sub_file = find_submit_file(job_dir, args.sub_file)
    queue_var = extract_queue_variable(sub_file, default="input_dst")
    log_dir = extract_log_directory(sub_file, default=Path("/tmp/anarde/dump/sepd_calib"))

    resubmit_sub_file = None
    if sub_file:
        resubmit_sub_file = job_dir / f"{sub_file.stem}_resubmit.sub"
        create_resubmit_sub_file(sub_file, new_memory, resubmit_sub_file, dry_run=args.dry_run)
    else:
        print("[!] Note: Submit description file (.sub) not found in directory.")

    # 6. Remove held jobs from Condor queue
    print("-" * 80)
    if args.keep_held:
        print("[*] Skipping removal of held jobs (--keep-held active).")
    else:
        remove_condor_jobs(job_ids_to_remove, schedd=active_schedd, dry_run=args.dry_run)

    # 7. Remove stdout and error logs
    print("-" * 80)
    if args.keep_logs:
        print("[*] Skipping deletion of log files (--keep-logs active).")
    else:
        print(f"[*] Cleaning up old stdout and error logs for {len(proc_ids_to_clean)} jobs...")
        deleted_count = clean_job_logs(
            job_dir=job_dir,
            cluster_id=target_cluster,
            proc_ids=proc_ids_to_clean,
            log_dir=log_dir,
            dry_run=args.dry_run,
        )
        print(f"[+] Cleaned up {deleted_count} log files.")

    # 8. Formulate and print submission command
    print("=" * 80)
    print(" NEW SUBMISSION COMMAND (Run on current node):")
    print("=" * 80)

    sub_target = resubmit_sub_file.name if resubmit_sub_file else (sub_file.name if sub_file else "genFun4All.sub")
    prep_dir_cmd = f"mkdir -p {log_dir} && " if log_dir and str(log_dir).startswith("/tmp") else ""
    submit_cmd = (
        f'{prep_dir_cmd}cd {job_dir} && '
        f'condor_submit {sub_target} -queue "{queue_var} from {output_jobs_file.name}"'
    )

    print(f"\n  {submit_cmd}\n")

    if sub_file and resubmit_sub_file:
        alt_cmd = (
            f'{prep_dir_cmd}cd {job_dir} && '
            f'condor_submit {sub_file.name} request_memory={new_memory} -queue "{queue_var} from {output_jobs_file.name}"'
        )
        print(" Alternative (direct command-line memory override without editing .sub file):")
        print(f"\n  {alt_cmd}\n")

    print("=" * 80)

    # 9. Execute submission if requested
    if args.execute_submit:
        if args.dry_run:
            print(f"[DRY-RUN] Would execute submission command.")
        else:
            print(f"[*] Executing submission command...")
            try:
                res = subprocess.run(
                    ["bash", "-c", submit_cmd],
                    cwd=str(job_dir),
                    capture_output=True,
                    text=True,
                    check=False,
                )
                if res.returncode == 0:
                    print("[+] Submission successful!")
                    print(res.stdout.strip())
                else:
                    print(f"[!] Submission failed with exit code {res.returncode}")
                    if res.stderr:
                        print(f"    STDERR: {res.stderr.strip()}")
            except Exception as e:
                print(f"[!] Error executing submission: {e}")


if __name__ == "__main__":
    main()
