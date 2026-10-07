#!/usr/bin/env python3
"""
analyze_condor_logs.py

Analyzes HTCondor user log files (*.log) within a specified directory (strictly
ignoring any subdirectories). Computes and displays comprehensive statistics on:
  - Job outcomes (completed, failed, aborted, held, running, idle)
  - Exit codes and hold/abort/eviction causes
  - Memory usage for finished jobs (min, max, mean, median, percentiles, distribution)
  - Timing and duration metrics (run time, queue wait time, turnaround, CPU efficiency)
  - Execution attempts and retries (distribution, outcome matrix, most retried jobs)
  - Execute hosts and node health

Usage:
  python scripts/condor/analyze_condor_logs.py /tmp/anarde/dump/
  python scripts/condor/analyze_condor_logs.py /tmp/anarde/dump/ --top 15
  python scripts/condor/analyze_condor_logs.py /tmp/anarde/dump/ --csv results.csv --json results.json
  python scripts/condor/analyze_condor_logs.py /tmp/anarde/dump/ --plot summary.png

Date: October 2026
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import datetime
import json
import math
import os
from pathlib import Path
import re
import sys
from typing import Any, Dict, List, Optional, Tuple


# ANSI Color Codes
class Colors:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"

    @classmethod
    def disable(cls):
        cls.RESET = ""
        cls.BOLD = ""
        cls.DIM = ""
        cls.RED = ""
        cls.GREEN = ""
        cls.YELLOW = ""
        cls.BLUE = ""
        cls.MAGENTA = ""
        cls.CYAN = ""
        cls.WHITE = ""


@dataclass
class JobRecord:
    cluster_id: int
    proc_id: int
    subproc_id: int = 0
    filename: str = ""

    # Status
    status: str = "UNKNOWN"  # COMPLETED, FAILED_EXIT, FAILED_SIGNAL, ABORTED, HELD, RUNNING, IDLE, EVICTED
    exit_code: Optional[int] = None
    term_signal: Optional[int] = None
    abort_reason: Optional[str] = None
    hold_reason: Optional[str] = None
    last_event_code: Optional[str] = None

    # Attempts & Retries
    attempts: int = 0  # Number of 001 events
    retries: int = 0  # max(0, attempts - 1)
    evictions: int = 0  # Number of 004 events
    holds: int = 0  # Number of 012 events
    eviction_reasons: List[str] = field(default_factory=list)

    # Timestamps
    submit_time: Optional[datetime] = None
    first_exec_time: Optional[datetime] = None
    last_exec_time: Optional[datetime] = None
    term_time: Optional[datetime] = None
    last_event_time: Optional[datetime] = None

    # Durations (seconds)
    queue_wait_s: Optional[float] = None
    run_time_s: Optional[float] = None
    total_wall_s: Optional[float] = None

    # CPU Times (seconds)
    cpu_usr_s: float = 0.0
    cpu_sys_s: float = 0.0
    cpu_total_s: float = 0.0
    cpu_efficiency_pct: Optional[float] = None

    # Memory (MB)
    mem_used_mb: Optional[int] = None
    mem_req_mb: Optional[int] = None
    mem_alloc_mb: Optional[int] = None
    mem_rss_kb: Optional[int] = None
    mem_utilization_pct: Optional[float] = None

    # Hosts
    hosts: List[str] = field(default_factory=list)


def parse_condor_date(date_str: str) -> Optional[datetime]:
    """Parses standard Condor log timestamps."""
    s = date_str.strip()
    try:
        return datetime.strptime(s, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        pass
    try:
        dt = datetime.strptime(s, "%m/%d %H:%M:%S")
        return dt.replace(year=datetime.now().year)
    except ValueError:
        pass
    return None


def format_duration(seconds: Optional[float], compact: bool = False) -> str:
    """Formats seconds into human-readable duration."""
    if seconds is None or math.isnan(seconds):
        return "N/A"
    sec = int(round(seconds))
    if sec < 0:
        return f"-{format_duration(abs(sec), compact)}"
    if sec < 60:
        return f"{sec}s"
    m, s = divmod(sec, 60)
    h, m = divmod(m, 60)
    d, h = divmod(h, 24)
    if compact:
        if d > 0:
            return f"{d}d{h:02d}h"
        if h > 0:
            return f"{h}h{m:02d}m"
        return f"{m}m{s:02d}s"
    parts = []
    if d > 0:
        parts.append(f"{d}d")
    if h > 0 or d > 0:
        parts.append(f"{h}h")
    if m > 0 or h > 0 or d > 0:
        parts.append(f"{m}m")
    parts.append(f"{s}s")
    return " ".join(parts)


def format_mem(mb: Optional[float]) -> str:
    """Formats megabytes into human-readable MB / GB."""
    if mb is None or math.isnan(mb):
        return "N/A"
    if mb >= 1024:
        return f"{mb:.0f} MB ({mb / 1024.0:.2f} GB)"
    return f"{mb:.0f} MB"


def parse_single_log_file(file_path_str: str) -> List[JobRecord]:
    """
    Parses an individual Condor log file.
    Can handle log files with single or multiple jobs.
    """
    path = Path(file_path_str)
    try:
        with open(file_path_str, "r", errors="ignore") as f:
            content = f.read()
    except Exception:
        return []

    if not content:
        return []

    # Find all event headers:
    # Example: '000 (194706.000.000) 2026-09-30 20:01:46 Job submitted from host: ...'
    # Pattern matches 3-digit event code, job id tuple, and date time
    event_header_pattern = re.compile(
        r"^(\d{3})\s+\((\d+)\.(\d+)\.(\d+)\)\s+([\d/-]+\s+[\d:]+)",
        re.MULTILINE,
    )
    headers = list(event_header_pattern.finditer(content))
    if not headers:
        return []

    # Map jobs in this file by (cluster, proc, subproc)
    jobs_map: Dict[Tuple[int, int, int], JobRecord] = {}

    for i, match in enumerate(headers):
        event_code = match.group(1)
        cluster_id = int(match.group(2))
        proc_id = int(match.group(3))
        subproc_id = int(match.group(4))
        date_str = match.group(5)
        event_time = parse_condor_date(date_str)

        start_pos = match.start()
        end_pos = headers[i + 1].start() if i + 1 < len(headers) else len(content)
        event_body = content[start_pos:end_pos]

        key = (cluster_id, proc_id, subproc_id)
        if key not in jobs_map:
            jobs_map[key] = JobRecord(
                cluster_id=cluster_id,
                proc_id=proc_id,
                subproc_id=subproc_id,
                filename=path.name,
            )

        job = jobs_map[key]
        job.last_event_code = event_code
        if event_time:
            job.last_event_time = event_time

        # 000: Job submitted
        if event_code == "000":
            if event_time and (job.submit_time is None or event_time < job.submit_time):
                job.submit_time = event_time

        # 001: Job executing
        elif event_code == "001":
            job.attempts += 1
            if event_time:
                if job.first_exec_time is None:
                    job.first_exec_time = event_time
                job.last_exec_time = event_time

            # Extract host
            m_slot = re.search(r"SlotName:\s*([^@\s]+@)?([\w\.-]+)", event_body)
            if m_slot:
                host_str = m_slot.group(2)
                if host_str not in job.hosts:
                    job.hosts.append(host_str)
            else:
                m_host = re.search(r"alias=([\w\.-]+)", event_body)
                if m_host:
                    host_str = m_host.group(1)
                    if host_str not in job.hosts:
                        job.hosts.append(host_str)

            # Slot memory request
            if job.mem_req_mb is None:
                m_req = re.search(r"^\s*Memory\s*=\s*(\d+)", event_body, re.MULTILINE)
                if m_req:
                    job.mem_req_mb = int(m_req.group(1))

        # 004: Job evicted
        elif event_code == "004":
            job.evictions += 1
            # Eviction reason
            if "via condor_rm" in event_body:
                job.eviction_reasons.append("condor_rm")
            elif "return value" in event_body:
                m_rv = re.search(r"return value (\d+)", event_body)
                if m_rv:
                    job.eviction_reasons.append(f"exit {m_rv.group(1)}")
            elif "signal" in event_body:
                m_sig = re.search(r"signal (\d+)", event_body)
                if m_sig:
                    job.eviction_reasons.append(f"signal {m_sig.group(1)}")
            else:
                m_r = re.search(r"Reason:\s*([^\n]+)", event_body)
                if m_r:
                    job.eviction_reasons.append(m_r.group(1).strip()[:40])

            # Eviction CPU usage
            m_run_cpu = re.search(
                r"Usr\s+(\d+)\s+(\d+):(\d+):(\d+),\s*Sys\s+(\d+)\s+(\d+):(\d+):(\d+)\s*-\s*Run Remote Usage",
                event_body,
            )
            if m_run_cpu:
                d1, h1, m1, s1, d2, h2, m2, s2 = map(int, m_run_cpu.groups())
                job.cpu_usr_s += d1 * 86400 + h1 * 3600 + m1 * 60 + s1
                job.cpu_sys_s += d2 * 86400 + h2 * 3600 + m2 * 60 + s2

        # 005: Job terminated
        elif event_code == "005":
            if event_time:
                job.term_time = event_time

            # Normal or abnormal termination
            m_norm = re.search(r"Normal termination \(return value (\d+)\)", event_body)
            if m_norm:
                job.exit_code = int(m_norm.group(1))
            else:
                m_ab = re.search(r"Abnormal termination \(signal (\d+)\)", event_body)
                if m_ab:
                    job.term_signal = int(m_ab.group(1))

            # Total Remote Usage CPU
            m_tot_cpu = re.search(
                r"Usr\s+(\d+)\s+(\d+):(\d+):(\d+),\s*Sys\s+(\d+)\s+(\d+):(\d+):(\d+)\s*-\s*Total Remote Usage",
                event_body,
            )
            if m_tot_cpu:
                d1, h1, m1, s1, d2, h2, m2, s2 = map(int, m_tot_cpu.groups())
                job.cpu_usr_s = d1 * 86400 + h1 * 3600 + m1 * 60 + s1
                job.cpu_sys_s = d2 * 86400 + h2 * 3600 + m2 * 60 + s2

            # Partitionable Resources
            m_res_mem = re.search(
                r"Memory\s*\(MB\)\s*:\s*(\d+)\s+(\d+)\s+(\d+)", event_body
            )
            if m_res_mem:
                job.mem_used_mb = int(m_res_mem.group(1))
                job.mem_req_mb = int(m_res_mem.group(2))
                job.mem_alloc_mb = int(m_res_mem.group(3))

            m_time_exec = re.search(r"TimeExecute\s*\(s\)\s*:\s*(\d+)", event_body)
            if m_time_exec:
                job.run_time_s = float(m_time_exec.group(1))

        # 006: Image size updated
        elif event_code == "006":
            m_u = re.search(r"(\d+)\s+-\s+MemoryUsage of job \(MB\)", event_body)
            if m_u:
                job.mem_used_mb = int(m_u.group(1))
            m_rss = re.search(r"(\d+)\s+-\s+ResidentSetSize of job \(KB\)", event_body)
            if m_rss:
                job.mem_rss_kb = int(m_rss.group(1))

        # 009: Job aborted
        elif event_code == "009":
            if event_time:
                job.term_time = event_time
            m_ab_r = re.search(r"Job was aborted\.\s*\n\s*(.*)", event_body)
            if m_ab_r:
                job.abort_reason = m_ab_r.group(1).strip()[:60]
            else:
                job.abort_reason = "Aborted by user"

        # 012: Job held
        elif event_code == "012":
            job.holds += 1
            # Extract hold reason
            m_hr = re.search(r"Job was held\.\s*\n\s*(.*)", event_body)
            if m_hr:
                job.hold_reason = m_hr.group(1).strip()[:80]
            else:
                job.hold_reason = "Job held"

    # Post-process each job record to resolve final state and calculated metrics
    records = []
    now = datetime.now()

    for job in jobs_map.values():
        job.retries = max(0, job.attempts - 1)
        job.cpu_total_s = job.cpu_usr_s + job.cpu_sys_s

        # Resolve status
        last_code = job.last_event_code
        if last_code == "005":
            if job.exit_code == 0:
                job.status = "COMPLETED"
            elif job.exit_code is not None:
                job.status = "FAILED_EXIT"
            elif job.term_signal is not None:
                job.status = "FAILED_SIGNAL"
            else:
                job.status = "TERMINATED"
        elif last_code == "009":
            job.status = "ABORTED"
        elif last_code == "012":
            job.status = "HELD"
        elif last_code in ("001", "006"):
            job.status = "RUNNING"
        elif last_code == "000":
            job.status = "IDLE"
        elif last_code == "004":
            job.status = "EVICTED"
        else:
            job.status = f"CODE_{last_code}"

        # Queue wait time
        if job.first_exec_time and job.submit_time:
            job.queue_wait_s = max(
                0.0, (job.first_exec_time - job.submit_time).total_seconds()
            )

        # Run time (if not already set by TimeExecute)
        if job.run_time_s is None:
            if job.last_exec_time and job.term_time:
                job.run_time_s = max(
                    0.0, (job.term_time - job.last_exec_time).total_seconds()
                )
            elif job.last_exec_time and job.status == "RUNNING":
                job.run_time_s = max(
                    0.0, (now - job.last_exec_time).total_seconds()
                )

        # Total wall time
        if job.submit_time:
            end_t = job.term_time or (now if job.status == "RUNNING" else job.last_event_time)
            if end_t:
                job.total_wall_s = max(0.0, (end_t - job.submit_time).total_seconds())

        # CPU efficiency
        if job.run_time_s and job.run_time_s > 0 and job.cpu_total_s > 0:
            job.cpu_efficiency_pct = min(100.0, (job.cpu_total_s / job.run_time_s) * 100.0)

        # Memory utilization
        if job.mem_used_mb is not None and job.mem_req_mb and job.mem_req_mb > 0:
            job.mem_utilization_pct = (job.mem_used_mb / float(job.mem_req_mb)) * 100.0

        records.append(job)

    return records


def calculate_summary_stats(values: List[float]) -> Dict[str, Optional[float]]:
    """Calculates standard summary statistics (min, max, mean, median, percentiles)."""
    if not values:
        return {
            "count": 0,
            "min": None,
            "max": None,
            "mean": None,
            "median": None,
            "std": None,
            "p25": None,
            "p75": None,
            "p90": None,
            "p95": None,
            "p99": None,
            "sum": 0.0,
        }
    n = len(values)
    s_vals = sorted(values)
    mean_val = sum(s_vals) / float(n)
    sum_val = sum(s_vals)

    def percentile(p: float) -> float:
        if n == 1:
            return s_vals[0]
        k = (n - 1) * (p / 100.0)
        f = math.floor(k)
        c = math.ceil(k)
        if f == c:
            return s_vals[int(k)]
        return s_vals[int(f)] * (c - k) + s_vals[int(c)] * (k - f)

    variance = sum((x - mean_val) ** 2 for x in s_vals) / float(n) if n > 1 else 0.0
    std_val = math.sqrt(variance)

    return {
        "count": n,
        "min": s_vals[0],
        "max": s_vals[-1],
        "mean": mean_val,
        "median": percentile(50),
        "std": std_val,
        "p25": percentile(25),
        "p75": percentile(75),
        "p90": percentile(90),
        "p95": percentile(95),
        "p99": percentile(99),
        "sum": sum_val,
    }


def print_table(headers: List[str], rows: List[List[str]], alignments: Optional[List[str]] = None):
    """Prints a neatly formatted ASCII table."""
    if not rows:
        return
    col_widths = [len(h) for h in headers]
    for row in rows:
        for idx, cell in enumerate(row):
            # Strip ANSI color codes when computing width
            plain_text = re.sub(r"\033\[[0-9;]*m", "", str(cell))
            if idx < len(col_widths):
                col_widths[idx] = max(col_widths[idx], len(plain_text))
            else:
                col_widths.append(len(plain_text))

    if alignments is None:
        alignments = ["<"] * len(col_widths)

    def format_row(cells: List[str]) -> str:
        formatted = []
        for idx, cell in enumerate(cells):
            plain_text = re.sub(r"\033\[[0-9;]*m", "", str(cell))
            padding = col_widths[idx] - len(plain_text)
            align = alignments[idx] if idx < len(alignments) else "<"
            if align == ">":
                formatted.append(" " * padding + str(cell))
            elif align == "^":
                left = padding // 2
                right = padding - left
                formatted.append(" " * left + str(cell) + " " * right)
            else:
                formatted.append(str(cell) + " " * padding)
        return " | ".join(formatted)

    header_line = format_row(headers)
    separator = "-+-".join("-" * w for w in col_widths)

    print(f"  {Colors.BOLD}{header_line}{Colors.RESET}")
    print(f"  {separator}")
    for row in rows:
        print(f"  {format_row(row)}")


def generate_plot(records: List[JobRecord], output_path: str):
    """Generates a summary multi-panel graphic if matplotlib is installed."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print(f"{Colors.YELLOW}[!] Matplotlib not available; skipping plot generation.{Colors.RESET}")
        return

    finished = [r for r in records if r.status == "COMPLETED"]
    mems = [r.mem_used_mb for r in finished if r.mem_used_mb is not None]
    times_h = [r.run_time_s / 3600.0 for r in finished if r.run_time_s is not None]
    retries = [r.retries for r in records]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("HTCondor Job Analysis Summary", fontsize=16, fontweight="bold")

    # 1. Job Outcomes (Horizontal Bar Chart)
    detailed_counts: Dict[str, int] = {}
    for r in records:
        if r.status == "COMPLETED":
            lbl = "Completed (Exit 0)"
        elif r.status == "FAILED_EXIT":
            lbl = f"Failed (Exit {r.exit_code})" if r.exit_code is not None else "Failed (Exit)"
        elif r.status == "FAILED_SIGNAL":
            lbl = f"Failed (Signal {r.term_signal})" if r.term_signal is not None else "Failed (Signal)"
        elif r.status == "ABORTED":
            lbl = "Aborted (condor_rm)"
        elif r.status == "HELD":
            lbl = "Held"
        elif r.status == "RUNNING":
            lbl = "Running"
        elif r.status == "IDLE":
            lbl = "Idle / Queued"
        else:
            lbl = r.status
        detailed_counts[lbl] = detailed_counts.get(lbl, 0) + 1

    # Sort ascending so largest category is at the top of horizontal bar chart
    sorted_status = sorted(detailed_counts.items(), key=lambda x: x[1])
    labels = [s[0] for s in sorted_status]
    counts = [s[1] for s in sorted_status]

    def get_color(lbl: str) -> str:
        if "Completed" in lbl:
            return "#2ecc71"
        if "Failed" in lbl:
            return "#e74c3c"
        if "Aborted" in lbl:
            return "#f39c12"
        if "Held" in lbl:
            return "#f1c40f"
        if "Running" in lbl:
            return "#3498db"
        return "#95a5a6"

    colors_bar = [get_color(l) for l in labels]
    axes[0, 0].barh(labels, counts, color=colors_bar, edgecolor="black", alpha=0.85)
    max_c = max(counts) if counts else 1
    total_recs = len(records)
    for i, count in enumerate(counts):
        pct = (count / float(total_recs)) * 100.0 if total_recs > 0 else 0.0
        axes[0, 0].text(
            count + max_c * 0.02,
            i,
            f"{count:,} ({pct:.1f}%)",
            va="center",
            ha="left",
            fontsize=9,
            fontweight="bold",
        )
    axes[0, 0].set_xlim(0, max_c * 1.30)
    axes[0, 0].set_xlabel("Job Count", fontweight="bold")
    axes[0, 0].set_title(f"Job Outcomes Breakdown (Total: {total_recs:,})", fontweight="bold")
    axes[0, 0].grid(axis="x", linestyle="--", alpha=0.7)

    # 2. Retries Distribution (Bar Chart with count annotations)
    max_ret = max(retries) if retries else 0
    retry_levels = list(range(0, max_ret + 1))
    retry_level_counts = [sum(1 for r in retries if r == lvl) for lvl in retry_levels]
    retry_labels = [f"{lvl} retry\n({lvl+1} att.)" if lvl > 0 else "0 retries\n(1 att.)" for lvl in retry_levels]

    r_bars = axes[0, 1].bar(retry_labels, retry_level_counts, color="#3498db", edgecolor="black", alpha=0.85)
    max_rc = max(retry_level_counts) if retry_level_counts else 1
    for bar in r_bars:
        h = bar.get_height()
        pct = (h / float(total_recs)) * 100.0 if total_recs > 0 else 0.0
        axes[0, 1].text(
            bar.get_x() + bar.get_width() / 2.0,
            h + max_rc * 0.02,
            f"{h:,}\n({pct:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )
    axes[0, 1].set_ylim(0, max_rc * 1.25)
    axes[0, 1].set_xlabel("Retries (Total Attempts)", fontweight="bold")
    axes[0, 1].set_ylabel("Job Count", fontweight="bold")
    axes[0, 1].set_title("Job Retry Distribution", fontweight="bold")
    axes[0, 1].grid(axis="y", linestyle="--", alpha=0.7)

    # 3. Memory Usage for Finished Jobs
    if mems:
        axes[1, 0].hist(mems, bins=30, color="#9b59b6", edgecolor="black", alpha=0.85)
        median_mem = sorted(mems)[len(mems) // 2]
        axes[1, 0].axvline(median_mem, color="red", linestyle="--", linewidth=1.8, label=f"Median: {median_mem} MB")
        # Check if requested memory exists
        reqs = [r.mem_req_mb for r in finished if r.mem_req_mb is not None]
        if reqs:
            req_val = sorted(reqs)[len(reqs) // 2]
            axes[1, 0].axvline(req_val, color="orange", linestyle=":", linewidth=2, label=f"Requested: {req_val} MB")
        axes[1, 0].set_xlabel("Memory Usage (MB)")
        axes[1, 0].set_ylabel("Job Count")
        axes[1, 0].set_title(f"Memory Usage for Completed Jobs (N={len(mems):,})", fontweight="bold")
        axes[1, 0].legend()
        axes[1, 0].grid(axis="y", linestyle="--", alpha=0.7)
    else:
        axes[1, 0].text(0.5, 0.5, "No memory data for finished jobs", ha="center", va="center")

    # 4. Run Time for Finished Jobs
    if times_h:
        axes[1, 1].hist(times_h, bins=30, color="#1abc9c", edgecolor="black", alpha=0.85)
        median_time = sorted(times_h)[len(times_h) // 2]
        axes[1, 1].axvline(median_time, color="red", linestyle="--", linewidth=1.8, label=f"Median: {median_time:.2f} h")
        axes[1, 1].set_xlabel("Execution Time (hours)")
        axes[1, 1].set_ylabel("Job Count")
        axes[1, 1].set_title(f"Execution Time for Completed Jobs (N={len(times_h):,})", fontweight="bold")
        axes[1, 1].legend()
        axes[1, 1].grid(axis="y", linestyle="--", alpha=0.7)
    else:
        axes[1, 1].text(0.5, 0.5, "No timing data for finished jobs", ha="center", va="center")

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"\n{Colors.GREEN}[✓] Saved summary plot to: {output_path}{Colors.RESET}")


def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive HTCondor Log Analyzer for batch jobs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python %(prog)s /tmp/anarde/dump/
  python %(prog)s /tmp/anarde/dump/ --top 20
  python %(prog)s /tmp/anarde/dump/ --csv /tmp/condor_stats.csv --json /tmp/condor_stats.json
  python %(prog)s /tmp/anarde/dump/ --plot /tmp/condor_stats.png
        """,
    )
    parser.add_argument(
        "log_path",
        nargs="?",
        default=None,
        help="Path to directory containing condor log files (e.g. /tmp/anarde/dump/)",
    )
    parser.add_argument(
        "-d",
        "--dir",
        dest="log_dir_opt",
        default=None,
        help="Alternative flag to specify log directory path",
    )
    parser.add_argument(
        "--pattern",
        default="*.log",
        help="Filename pattern to match in the directory (default: '*.log')",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=10,
        help="Number of top entries to show in detailed tables (default: 10)",
    )
    parser.add_argument(
        "--status",
        choices=["all", "completed", "failed", "aborted", "held", "running", "idle"],
        default="all",
        help="Filter analysis to a specific job status (default: 'all')",
    )
    parser.add_argument(
        "--cluster",
        type=int,
        default=None,
        help="Filter analysis to a specific Cluster ID",
    )
    parser.add_argument(
        "--workers",
        "-j",
        type=int,
        default=None,
        help="Number of worker processes for parallel log parsing (default: all CPUs)",
    )
    parser.add_argument(
        "--csv",
        default=None,
        help="Export individual job records to a CSV file",
    )
    parser.add_argument(
        "--json",
        default=None,
        help="Export comprehensive statistics dictionary to a JSON file (or '-' for stdout)",
    )
    parser.add_argument(
        "--plot",
        default=None,
        help="Generate a 4-panel summary PNG figure (requires matplotlib)",
    )
    parser.add_argument(
        "--no-color",
        action="store_true",
        help="Disable ANSI color output in terminal",
    )
    parser.add_argument(
        "--brief",
        action="store_true",
        help="Print concise summary without full distributions",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Show verbose details and diagnostic notices",
    )

    args = parser.parse_args()

    # Determine directory
    target_dir_str = args.log_path or args.log_dir_opt
    if not target_dir_str:
        parser.print_help()
        print(f"\n{Colors.RED}Error: Please specify a Condor log directory path.{Colors.RESET}")
        sys.exit(1)

    log_dir = Path(target_dir_str).resolve()
    if not log_dir.exists():
        print(f"{Colors.RED}Error: Directory does not exist: {log_dir}{Colors.RESET}")
        sys.exit(1)
    if not log_dir.is_dir():
        print(f"{Colors.RED}Error: Path is not a directory: {log_dir}{Colors.RESET}")
        sys.exit(1)

    # Disable colors if requested or if output redirected
    if args.no_color or not sys.stdout.isatty():
        Colors.disable()

    # Discover files: STRICTLY regular files in log_dir (NOT ANY SUBDIR!)
    skipped_subdirs = []
    target_files = []

    pattern = args.pattern
    import fnmatch

    for entry in os.scandir(log_dir):
        if entry.is_dir():
            skipped_subdirs.append(entry.name)
        elif entry.is_file():
            if fnmatch.fnmatch(entry.name, pattern):
                target_files.append(entry.path)

    total_files_found = len(target_files)
    if total_files_found == 0:
        print(f"{Colors.YELLOW}[!] No files matching '{pattern}' found in {log_dir}.{Colors.RESET}")
        if skipped_subdirs:
            print(f"    (Skipped {len(skipped_subdirs)} subdirectories: {', '.join(sorted(skipped_subdirs)[:6])})")
        sys.exit(0)

    # Parse in parallel
    num_workers = args.workers or max(1, min(os.cpu_count() or 4, 16))
    chunk_size = max(10, min(200, total_files_found // (num_workers * 4) + 1))

    t_start = datetime.now()
    if not args.json or args.json != "-":
        print(f"{Colors.BOLD}{Colors.CYAN}Scanning {total_files_found:,} log files in {log_dir}...{Colors.RESET}")
        if skipped_subdirs:
            print(f"{Colors.DIM}  [Note: strictly ignored {len(skipped_subdirs)} subdirectories: {', '.join(sorted(skipped_subdirs)[:8])}{'...' if len(skipped_subdirs) > 8 else ''}]{Colors.RESET}")

    all_records: List[JobRecord] = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        for file_records in executor.map(parse_single_log_file, target_files, chunksize=chunk_size):
            all_records.extend(file_records)

    parse_duration = (datetime.now() - t_start).total_seconds()

    if not all_records:
        print(f"{Colors.YELLOW}[!] No valid Condor event logs parsed from {total_files_found} files.{Colors.RESET}")
        sys.exit(0)

    # Apply filters if specified
    if args.cluster is not None:
        all_records = [r for r in all_records if r.cluster_id == args.cluster]
    if args.status != "all":
        st_filter = args.status.upper()
        if st_filter == "COMPLETED":
            all_records = [r for r in all_records if r.status == "COMPLETED"]
        elif st_filter == "FAILED":
            all_records = [r for r in all_records if r.status.startswith("FAILED")]
        elif st_filter == "ABORTED":
            all_records = [r for r in all_records if r.status == "ABORTED"]
        elif st_filter == "HELD":
            all_records = [r for r in all_records if r.status == "HELD"]
        elif st_filter == "RUNNING":
            all_records = [r for r in all_records if r.status == "RUNNING"]
        elif st_filter == "IDLE":
            all_records = [r for r in all_records if r.status == "IDLE"]

    total_jobs = len(all_records)

    # Aggregate outcome counts
    clusters = sorted(list({r.cluster_id for r in all_records}))
    completed_jobs = [r for r in all_records if r.status == "COMPLETED"]
    failed_exit_jobs = [r for r in all_records if r.status == "FAILED_EXIT"]
    failed_sig_jobs = [r for r in all_records if r.status == "FAILED_SIGNAL"]
    aborted_jobs = [r for r in all_records if r.status == "ABORTED"]
    held_jobs = [r for r in all_records if r.status == "HELD"]
    running_jobs = [r for r in all_records if r.status == "RUNNING"]
    idle_jobs = [r for r in all_records if r.status == "IDLE"]
    other_jobs = [r for r in all_records if r.status not in (
        "COMPLETED", "FAILED_EXIT", "FAILED_SIGNAL", "ABORTED", "HELD", "RUNNING", "IDLE"
    )]

    # Time bounds
    submits = [r.submit_time for r in all_records if r.submit_time is not None]
    terms = [r.term_time for r in all_records if r.term_time is not None]
    min_submit = min(submits) if submits else None
    max_term = max(terms) if terms else None
    total_batch_span = (max_term - min_submit).total_seconds() if (min_submit and max_term) else None

    # Memory statistics (Finished Jobs)
    finished_mems = [float(r.mem_used_mb) for r in completed_jobs if r.mem_used_mb is not None]
    finished_mem_reqs = [float(r.mem_req_mb) for r in completed_jobs if r.mem_req_mb is not None]
    finished_mem_stats = calculate_summary_stats(finished_mems)
    finished_mem_req_stats = calculate_summary_stats(finished_mem_reqs)

    # Timing statistics (Finished Jobs)
    finished_runtimes = [r.run_time_s for r in completed_jobs if r.run_time_s is not None]
    finished_runtime_stats = calculate_summary_stats(finished_runtimes)

    finished_waits = [r.queue_wait_s for r in completed_jobs if r.queue_wait_s is not None]
    finished_wait_stats = calculate_summary_stats(finished_waits)

    finished_totals = [r.total_wall_s for r in completed_jobs if r.total_wall_s is not None]
    finished_total_stats = calculate_summary_stats(finished_totals)

    finished_cpus = [r.cpu_total_s for r in completed_jobs if r.cpu_total_s > 0]
    finished_cpu_stats = calculate_summary_stats(finished_cpus)

    finished_effs = [r.cpu_efficiency_pct for r in completed_jobs if r.cpu_efficiency_pct is not None]
    finished_eff_stats = calculate_summary_stats(finished_effs)

    # Retries statistics
    all_retries = [r.retries for r in all_records]
    completed_retries = [r.retries for r in completed_jobs]
    failed_retries = [r.retries for r in (failed_exit_jobs + failed_sig_jobs)]
    aborted_retries = [r.retries for r in aborted_jobs]

    retry_counts: Dict[int, int] = {}
    for ret in all_retries:
        retry_counts[ret] = retry_counts.get(ret, 0) + 1

    total_attempts = sum(r.attempts for r in all_records)
    total_evictions = sum(r.evictions for r in all_records)

    # CSV Export
    if args.csv:
        import csv
        with open(args.csv, "w", newline="") as f_csv:
            writer = csv.writer(f_csv)
            writer.writerow([
                "ClusterId", "ProcId", "Status", "ExitCode", "Attempts", "Retries",
                "Evictions", "MemUsed_MB", "MemReq_MB", "MemUtil_Pct", "RunTime_s",
                "QueueWait_s", "TotalWall_s", "CPUTotal_s", "CPUEff_Pct", "Hosts", "Filename"
            ])
            for r in all_records:
                writer.writerow([
                    r.cluster_id,
                    r.proc_id,
                    r.status,
                    r.exit_code if r.exit_code is not None else "",
                    r.attempts,
                    r.retries,
                    r.evictions,
                    r.mem_used_mb if r.mem_used_mb is not None else "",
                    r.mem_req_mb if r.mem_req_mb is not None else "",
                    f"{r.mem_utilization_pct:.1f}" if r.mem_utilization_pct is not None else "",
                    f"{r.run_time_s:.0f}" if r.run_time_s is not None else "",
                    f"{r.queue_wait_s:.0f}" if r.queue_wait_s is not None else "",
                    f"{r.total_wall_s:.0f}" if r.total_wall_s is not None else "",
                    f"{r.cpu_total_s:.0f}" if r.cpu_total_s > 0 else "",
                    f"{r.cpu_efficiency_pct:.1f}" if r.cpu_efficiency_pct is not None else "",
                    ",".join(r.hosts),
                    r.filename,
                ])
        print(f"{Colors.GREEN}[✓] Exported job details to CSV: {args.csv}{Colors.RESET}")

    # JSON Export
    if args.json:
        json_payload = {
            "summary": {
                "log_dir": str(log_dir),
                "total_files": total_files_found,
                "total_jobs": total_jobs,
                "clusters": clusters,
                "parse_duration_s": parse_duration,
                "min_submit": min_submit.isoformat() if min_submit else None,
                "max_term": max_term.isoformat() if max_term else None,
                "total_batch_span_s": total_batch_span,
            },
            "job_counts": {
                "completed": len(completed_jobs),
                "failed_exit": len(failed_exit_jobs),
                "failed_signal": len(failed_sig_jobs),
                "aborted": len(aborted_jobs),
                "held": len(held_jobs),
                "running": len(running_jobs),
                "idle": len(idle_jobs),
                "other": len(other_jobs),
            },
            "retries": {
                "total_attempts": total_attempts,
                "total_evictions": total_evictions,
                "distribution": retry_counts,
                "completed_avg_retries": sum(completed_retries) / len(completed_retries) if completed_retries else 0,
            },
            "memory_completed_mb": finished_mem_stats,
            "runtime_completed_s": finished_runtime_stats,
            "wait_time_completed_s": finished_wait_stats,
            "cpu_time_completed_s": finished_cpu_stats,
        }
        if args.json == "-":
            print(json.dumps(json_payload, indent=2))
            return
        with open(args.json, "w") as f_json:
            json.dump(json_payload, f_json, indent=2)
        print(f"{Colors.GREEN}[✓] Exported summary statistics to JSON: {args.json}{Colors.RESET}")

    # Optional Plot
    if args.plot:
        generate_plot(all_records, args.plot)

    # -------------------------------------------------------------
    # TERMINAL REPORT RENDERING
    # -------------------------------------------------------------
    divider = f"{Colors.DIM}{'=' * 78}{Colors.RESET}"
    sub_divider = f"{Colors.DIM}{'-' * 78}{Colors.RESET}"

    print(f"\n{divider}")
    print(f" {Colors.BOLD}{Colors.WHITE}CONDOR LOG ANALYSIS REPORT{Colors.RESET}")
    print(f"{divider}")
    print(f"  {Colors.BOLD}Target Directory:{Colors.RESET}   {log_dir}")
    print(f"  {Colors.BOLD}Files Analyzed:{Colors.RESET}     {total_files_found:,} log files (parsed in {parse_duration:.2f}s using {num_workers} workers)")
    if skipped_subdirs:
        print(f"  {Colors.DIM}Subdirs Ignored:    {len(skipped_subdirs)} (not recursed: {', '.join(sorted(skipped_subdirs)[:5])}{'...' if len(skipped_subdirs) > 5 else ''}){Colors.RESET}")
    print(f"  {Colors.BOLD}Cluster ID(s):{Colors.RESET}      {', '.join(map(str, clusters))}")
    print(f"  {Colors.BOLD}Total Jobs:{Colors.RESET}         {total_jobs:,}")
    if min_submit and max_term:
        print(f"  {Colors.BOLD}Batch Time Span:{Colors.RESET}    {min_submit.strftime('%Y-%m-%d %H:%M:%S')}  -->  {max_term.strftime('%Y-%m-%d %H:%M:%S')} ({format_duration(total_batch_span)})")

    # 1. JOB STATUS OVERVIEW
    print(f"\n{Colors.BOLD}{Colors.CYAN}--- [1] Job Status Breakdown ---{Colors.RESET}")
    status_rows = []

    def pct(count: int) -> str:
        return f"{(count / float(total_jobs)) * 100.0:.1f}%" if total_jobs > 0 else "0.0%"

    status_rows.append(["Completed (Exit 0)", f"{len(completed_jobs):,}", pct(len(completed_jobs)), f"{Colors.GREEN}SUCCESS{Colors.RESET}"])

    if failed_exit_jobs:
        # Group by exit code
        exit_code_counts: Dict[int, int] = {}
        for r in failed_exit_jobs:
            c = r.exit_code if r.exit_code is not None else -1
            exit_code_counts[c] = exit_code_counts.get(c, 0) + 1
        for code, c_count in sorted(exit_code_counts.items()):
            status_rows.append([f"Failed (Exit Code {code})", f"{c_count:,}", pct(c_count), f"{Colors.RED}ERROR{Colors.RESET}"])

    if failed_sig_jobs:
        sig_counts: Dict[int, int] = {}
        for r in failed_sig_jobs:
            s = r.term_signal if r.term_signal is not None else -1
            sig_counts[s] = sig_counts.get(s, 0) + 1
        for sig, s_count in sorted(sig_counts.items()):
            status_rows.append([f"Failed (Signal {sig})", f"{s_count:,}", pct(s_count), f"{Colors.RED}CRASH{Colors.RESET}"])

    if aborted_jobs:
        status_rows.append(["Aborted (condor_rm / cancelled)", f"{len(aborted_jobs):,}", pct(len(aborted_jobs)), f"{Colors.YELLOW}ABORTED{Colors.RESET}"])
    if held_jobs:
        status_rows.append(["Held (in queue)", f"{len(held_jobs):,}", pct(len(held_jobs)), f"{Colors.YELLOW}HELD{Colors.RESET}"])
    if running_jobs:
        status_rows.append(["Currently Running", f"{len(running_jobs):,}", pct(len(running_jobs)), f"{Colors.BLUE}RUNNING{Colors.RESET}"])
    if idle_jobs:
        status_rows.append(["Idle / Queued", f"{len(idle_jobs):,}", pct(len(idle_jobs)), f"{Colors.WHITE}QUEUED{Colors.RESET}"])
    if other_jobs:
        status_rows.append(["Other / Unknown", f"{len(other_jobs):,}", pct(len(other_jobs)), f"{Colors.DIM}OTHER{Colors.RESET}"])

    print_table(
        headers=["Outcome Status", "Count", "Percent", "State"],
        rows=status_rows,
        alignments=["<", ">", ">", "<"],
    )

    # 2. RETRIES & EXECUTION ATTEMPTS
    print(f"\n{Colors.BOLD}{Colors.CYAN}--- [2] Execution Attempts & Retries ---{Colors.RESET}")
    print(f"  {Colors.BOLD}Total Execution Starts (001):{Colors.RESET} {total_attempts:,}")
    print(f"  {Colors.BOLD}Total Job Evictions / Requeues:{Colors.RESET} {total_evictions:,}")
    avg_ret_all = sum(all_retries) / float(total_jobs) if total_jobs > 0 else 0
    avg_ret_comp = sum(completed_retries) / float(len(completed_jobs)) if completed_jobs else 0
    max_ret = max(all_retries) if all_retries else 0
    print(f"  {Colors.BOLD}Retries per Job:{Colors.RESET}               Avg: {avg_ret_all:.2f} (Completed Jobs Avg: {avg_ret_comp:.2f}) | Max: {max_ret}")

    print(f"\n  {Colors.BOLD}Retry Distribution across all jobs:{Colors.RESET}")
    retry_rows = []
    for ret_num in sorted(retry_counts.keys()):
        cnt = retry_counts[ret_num]
        attempts_label = f"{ret_num + 1} attempt{'s' if ret_num > 0 else ''}"
        # breakdown by outcome for this retry count
        c_comp = sum(1 for r in completed_jobs if r.retries == ret_num)
        c_fail = sum(1 for r in (failed_exit_jobs + failed_sig_jobs) if r.retries == ret_num)
        c_abort = sum(1 for r in aborted_jobs if r.retries == ret_num)
        outcome_detail = f"{c_comp:,} completed, {c_fail:,} failed, {c_abort:,} aborted"
        retry_rows.append([
            f"{ret_num} retry ({attempts_label})",
            f"{cnt:,}",
            pct(cnt),
            outcome_detail,
        ])
    print_table(
        headers=["Retry Level", "Job Count", "Percent", "Outcomes Breakdown"],
        rows=retry_rows,
        alignments=["<", ">", ">", "<"],
    )

    # Eviction reasons breakdown if any
    all_evict_reasons: Dict[str, int] = {}
    for r in all_records:
        for reason in r.eviction_reasons:
            all_evict_reasons[reason] = all_evict_reasons.get(reason, 0) + 1
    if all_evict_reasons:
        print(f"\n  {Colors.BOLD}Eviction / Requeue Triggers:{Colors.RESET}")
        evict_rows = [[r_name, f"{r_cnt:,}", f"{(r_cnt / float(total_evictions)) * 100.0:.1f}%"] for r_name, r_cnt in sorted(all_evict_reasons.items(), key=lambda x: x[1], reverse=True)[:5]]
        print_table(
            headers=["Eviction Reason", "Count", "Percent of Evictions"],
            rows=evict_rows,
            alignments=["<", ">", ">"],
        )

    # 3. MEMORY USAGE (Finished Jobs)
    print(f"\n{Colors.BOLD}{Colors.CYAN}--- [3] Memory Usage for Finished Jobs (Exit 0) ---{Colors.RESET}")
    if finished_mem_stats["count"] and finished_mem_stats["count"] > 0:
        req_mb = finished_mem_req_stats["median"] or (finished_mem_req_stats["mean"] or 0.0)
        req_str = f"{req_mb:.0f} MB ({req_mb / 1024.0:.2f} GB)" if req_mb > 0 else "N/A"
        print(f"  {Colors.BOLD}Finished Jobs with Memory Data:{Colors.RESET} {finished_mem_stats['count']:,} / {len(completed_jobs):,}")
        print(f"  {Colors.BOLD}Condor Request Memory (typical):{Colors.RESET} {req_str}")

        mem_table_rows = [
            ["Minimum Usage", format_mem(finished_mem_stats["min"]), f"{(finished_mem_stats['min'] / req_mb) * 100.0:.1f}%" if req_mb > 0 else "N/A"],
            ["25th Percentile (Q1)", format_mem(finished_mem_stats["p25"]), f"{(finished_mem_stats['p25'] / req_mb) * 100.0:.1f}%" if req_mb > 0 else "N/A"],
            ["Median Usage (P50)", format_mem(finished_mem_stats["median"]), f"{(finished_mem_stats['median'] / req_mb) * 100.0:.1f}%" if req_mb > 0 else "N/A"],
            ["Mean Usage (Avg)", format_mem(finished_mem_stats["mean"]), f"{(finished_mem_stats['mean'] / req_mb) * 100.0:.1f}%" if req_mb > 0 else "N/A"],
            ["75th Percentile (Q3)", format_mem(finished_mem_stats["p75"]), f"{(finished_mem_stats['p75'] / req_mb) * 100.0:.1f}%" if req_mb > 0 else "N/A"],
            ["95th Percentile (P95)", format_mem(finished_mem_stats["p95"]), f"{(finished_mem_stats['p95'] / req_mb) * 100.0:.1f}%" if req_mb > 0 else "N/A"],
            ["99th Percentile (P99)", format_mem(finished_mem_stats["p99"]), f"{(finished_mem_stats['p99'] / req_mb) * 100.0:.1f}%" if req_mb > 0 else "N/A"],
            ["Maximum Peak Usage", format_mem(finished_mem_stats["max"]), f"{(finished_mem_stats['max'] / req_mb) * 100.0:.1f}%" if req_mb > 0 else "N/A"],
            ["Std Deviation", f"{finished_mem_stats['std']:.1f} MB", "-"],
        ]
        print_table(
            headers=["Memory Metric", "Value", "Util. of Requested"],
            rows=mem_table_rows,
            alignments=["<", ">", ">"],
        )

        # Memory Bins
        mem_bins = [
            ("< 256 MB", lambda m: m < 256),
            ("256 - 512 MB", lambda m: 256 <= m < 512),
            ("512 - 768 MB", lambda m: 512 <= m < 768),
            ("768 - 1024 MB", lambda m: 768 <= m < 1024),
            ("1.0 - 1.5 GB", lambda m: 1024 <= m < 1536),
            ("1.5 - 2.0 GB", lambda m: 1536 <= m < 2048),
            ("> 2.0 GB", lambda m: m >= 2048),
        ]
        active_bins = []
        for b_label, b_fn in mem_bins:
            cnt = sum(1 for m in finished_mems if b_fn(m))
            if cnt > 0:
                active_bins.append([b_label, f"{cnt:,}", f"{(cnt / len(finished_mems)) * 100.0:.1f}%"])
        if active_bins:
            print(f"\n  {Colors.BOLD}Memory Distribution Histogram:{Colors.RESET}")
            print_table(
                headers=["Range", "Finished Jobs", "Percent"],
                rows=active_bins,
                alignments=["<", ">", ">"],
            )
    else:
        print(f"  {Colors.YELLOW}No finished jobs had memory usage events recorded.{Colors.RESET}")

    # 4. TIME NEEDED (Finished Jobs)
    print(f"\n{Colors.BOLD}{Colors.CYAN}--- [4] Time & Duration for Finished Jobs (Exit 0) ---{Colors.RESET}")
    if finished_runtime_stats["count"] and finished_runtime_stats["count"] > 0:
        total_exec_h = (finished_runtime_stats["sum"] or 0.0) / 3600.0
        total_cpu_h = (finished_cpu_stats["sum"] or 0.0) / 3600.0
        overall_cpu_eff = (total_cpu_h / total_exec_h * 100.0) if total_exec_h > 0 else 0.0

        print(f"  {Colors.BOLD}Aggregated Execution Time:{Colors.RESET} {total_exec_h:,.1f} core-hours ({total_exec_h / 24.0:,.1f} core-days)")
        if total_cpu_h > 0:
            print(f"  {Colors.BOLD}Aggregated CPU Time:{Colors.RESET}       {total_cpu_h:,.1f} CPU-hours ({total_cpu_h / 24.0:,.1f} CPU-days)")
            print(f"  {Colors.BOLD}Overall CPU Efficiency:{Colors.RESET}    {overall_cpu_eff:.1f}% (CPU Time / Execution Time)")

        time_table_rows = [
            ["Execution Wall Time (Run)", format_duration(finished_runtime_stats["min"]), format_duration(finished_runtime_stats["median"]), format_duration(finished_runtime_stats["mean"]), format_duration(finished_runtime_stats["max"])],
            ["Queue Wait Time (Pending)", format_duration(finished_wait_stats["min"]), format_duration(finished_wait_stats["median"]), format_duration(finished_wait_stats["mean"]), format_duration(finished_wait_stats["max"])],
            ["Total Turnaround (In Pool)", format_duration(finished_total_stats["min"]), format_duration(finished_total_stats["median"]), format_duration(finished_total_stats["mean"]), format_duration(finished_total_stats["max"])],
        ]
        if finished_cpu_stats["count"] and finished_cpu_stats["count"] > 0:
            time_table_rows.append([
                "Total Remote CPU Time", format_duration(finished_cpu_stats["min"]), format_duration(finished_cpu_stats["median"]), format_duration(finished_cpu_stats["mean"]), format_duration(finished_cpu_stats["max"])
            ])

        print_table(
            headers=["Timing Metric", "Min", "Median (P50)", "Mean (Avg)", "Max"],
            rows=time_table_rows,
            alignments=["<", ">", ">", ">", ">"],
        )

        # Runtime Distribution Bins
        time_bins = [
            ("< 15 min", lambda s: s < 900),
            ("15 - 60 min", lambda s: 900 <= s < 3600),
            ("1 - 2 hours", lambda s: 3600 <= s < 7200),
            ("2 - 4 hours", lambda s: 7200 <= s < 14400),
            ("4 - 8 hours", lambda s: 14400 <= s < 28800),
            ("8 - 12 hours", lambda s: 28800 <= s < 43200),
            ("> 12 hours", lambda s: s >= 43200),
        ]
        active_time_bins = []
        for t_label, t_fn in time_bins:
            cnt = sum(1 for s in finished_runtimes if t_fn(s))
            if cnt > 0:
                active_time_bins.append([t_label, f"{cnt:,}", f"{(cnt / len(finished_runtimes)) * 100.0:.1f}%"])
        if active_time_bins:
            print(f"\n  {Colors.BOLD}Execution Time Distribution:{Colors.RESET}")
            print_table(
                headers=["Duration Range", "Finished Jobs", "Percent"],
                rows=active_time_bins,
                alignments=["<", ">", ">"],
            )
    else:
        print(f"  {Colors.YELLOW}No finished jobs had execution runtime recorded.{Colors.RESET}")

    # 5. TOP JOBS TABLES (unless brief mode)
    if not args.brief and args.top > 0:
        top_n = args.top

        # Top Memory Jobs
        sorted_by_mem = sorted(
            [r for r in all_records if r.mem_used_mb is not None],
            key=lambda x: x.mem_used_mb or 0,
            reverse=True,
        )[:top_n]
        if sorted_by_mem:
            print(f"\n{Colors.BOLD}{Colors.CYAN}--- [5] Top {len(sorted_by_mem)} Peak Memory Jobs ---{Colors.RESET}")
            top_mem_rows = []
            for r in sorted_by_mem:
                top_mem_rows.append([
                    f"{r.cluster_id}.{r.proc_id}",
                    r.status,
                    format_mem(r.mem_used_mb),
                    format_mem(r.mem_req_mb),
                    f"{r.mem_utilization_pct:.1f}%" if r.mem_utilization_pct is not None else "N/A",
                    format_duration(r.run_time_s),
                    str(r.retries),
                    r.filename,
                ])
            print_table(
                headers=["Job ID", "Status", "Memory Used", "Requested", "Util %", "Run Time", "Retries", "Log File"],
                rows=top_mem_rows,
                alignments=["<", "<", ">", ">", ">", ">", ">", "<"],
            )

        # Top Longest Running Jobs
        sorted_by_time = sorted(
            [r for r in all_records if r.run_time_s is not None],
            key=lambda x: x.run_time_s or 0.0,
            reverse=True,
        )[:top_n]
        if sorted_by_time:
            print(f"\n{Colors.BOLD}{Colors.CYAN}--- [6] Top {len(sorted_by_time)} Longest Running Jobs ---{Colors.RESET}")
            top_time_rows = []
            for r in sorted_by_time:
                top_time_rows.append([
                    f"{r.cluster_id}.{r.proc_id}",
                    r.status,
                    format_duration(r.run_time_s),
                    format_duration(r.cpu_total_s) if r.cpu_total_s > 0 else "N/A",
                    f"{r.cpu_efficiency_pct:.1f}%" if r.cpu_efficiency_pct is not None else "N/A",
                    format_mem(r.mem_used_mb),
                    str(r.retries),
                    r.filename,
                ])
            print_table(
                headers=["Job ID", "Status", "Run Time", "CPU Time", "CPU Eff.", "Memory", "Retries", "Log File"],
                rows=top_time_rows,
                alignments=["<", "<", ">", ">", ">", ">", ">", "<"],
            )

        # Most Retried Jobs
        sorted_by_retries = sorted(
            [r for r in all_records if r.retries > 0],
            key=lambda x: (x.retries, x.attempts),
            reverse=True,
        )[:top_n]
        if sorted_by_retries:
            print(f"\n{Colors.BOLD}{Colors.CYAN}--- [7] Top {len(sorted_by_retries)} Most Retried Jobs ---{Colors.RESET}")
            top_retry_rows = []
            for r in sorted_by_retries:
                reasons_str = ", ".join(r.eviction_reasons[:3]) if r.eviction_reasons else "-"
                top_retry_rows.append([
                    f"{r.cluster_id}.{r.proc_id}",
                    r.status,
                    str(r.attempts),
                    str(r.retries),
                    reasons_str,
                    format_mem(r.mem_used_mb),
                    format_duration(r.run_time_s),
                    r.filename,
                ])
            print_table(
                headers=["Job ID", "Status", "Attempts", "Retries", "Eviction Reasons", "Memory", "Run Time", "Log File"],
                rows=top_retry_rows,
                alignments=["<", "<", ">", ">", "<", ">", ">", "<"],
            )

    # 6. WORKER NODES / HOSTS (Brief summary)
    all_hosts: Dict[str, int] = {}
    for r in all_records:
        for h in r.hosts:
            all_hosts[h] = all_hosts.get(h, 0) + 1
    if all_hosts and not args.brief:
        print(f"\n{Colors.BOLD}{Colors.CYAN}--- [8] Compute Hosts Summary ---{Colors.RESET}")
        print(f"  {Colors.BOLD}Total Unique Compute Hosts Used:{Colors.RESET} {len(all_hosts):,}")
        top_hosts = sorted(all_hosts.items(), key=lambda x: x[1], reverse=True)[:5]
        host_rows = [[h_name, f"{h_cnt:,}", f"{(h_cnt / float(total_attempts)) * 100.0:.1f}%"] for h_name, h_cnt in top_hosts]
        print_table(
            headers=["Execute Host", "Executions", "Percent"],
            rows=host_rows,
            alignments=["<", ">", ">"],
        )

    print(f"\n{divider}\n")


if __name__ == "__main__":
    main()
