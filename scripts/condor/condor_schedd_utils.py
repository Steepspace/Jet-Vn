#!/usr/bin/env python3
"""
condor_schedd_utils.py

Host-independent job state: asks EVERY schedd (condor_status -schedd) for the job ads of one
job directory via condor_q (queue) and condor_history (finished/removed). Unlike the user
event logs under /tmp, this works from any submit host. Shared by find_incomplete_jobs.py and
monitor_jobs.py.

Job ads are matched by Owner and Iwd (= the directory condor_submit was run from).
"""

import getpass
import os
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ATTRS = ["ClusterId", "ProcId", "JobStatus", "ExitCode", "ExitBySignal", "ExitSignal",
         "TotalSuspensions", "NumJobStarts", "CompletionDate", "LastRemoteHost", "RemoveReason", "HoldReason"]
STATUS_NAMES = {1: "idle", 2: "running", 3: "removed", 4: "completed", 5: "held", 6: "transferring", 7: "suspended"}


def condor_available() -> bool:
    return shutil.which("condor_status") is not None and shutil.which("condor_history") is not None


def list_schedds(timeout: int = 60) -> list[str]:
    try:
        out = subprocess.run(["condor_status", "-schedd", "-af", "Name"], capture_output=True, text=True,
                             timeout=timeout, check=True).stdout
    except Exception:
        return []
    return sorted({ln.strip() for ln in out.splitlines() if ln.strip()})


def _parse(line: str, schedd: str, source: str):
    p = line.split()
    if len(p) < len(ATTRS) - 3:  # last 3 free-text attrs may be missing/undefined; keep the structured ones
        return None
    try:
        cluster, proc, status = int(p[0]), int(p[1]), int(p[2])
    except ValueError:
        return None

    def num(i, default=None):
        try:
            return int(p[i])
        except (ValueError, IndexError):
            return default

    return (f"{cluster}.{proc}", dict(
        status=status, exit_code=num(3), by_signal=(p[4].lower() == "true") if len(p) > 4 else False,
        signal=num(5), suspensions=num(6, 0) or 0, starts=num(7, 0) or 0,
        completed=num(8), host=p[9] if len(p) > 9 and p[9] != "undefined" else "",
        schedd=schedd, source=source))


def _run(cmd: list, schedd: str, source: str, timeout: int):
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=True).stdout
    except Exception:
        return None
    rows = {}
    for line in out.splitlines():
        r = _parse(line, schedd, source)
        if r:
            rows[r[0]] = r[1]
    return rows


def query_job_ads(job_dirs, owner: str | None = None, timeout: int = 900, threads: int = 8):
    """
    Returns (ads, stats). ads maps 'cluster.proc' -> dict(status, exit_code, by_signal, signal,
    suspensions, starts, host, schedd, source='queue'|'history').

    CAUTION: coverage is best-effort. Remote condor_q is often refused and condor_history only
    keeps the most recent jobs of each schedd, so a job missing from ads proves nothing unless
    stats['queue_complete'] is True (every schedd's queue was readable).
    stats: schedds, history_ok, queue_ok (counts), queue_complete (bool).
    """
    owner = owner or os.environ.get("USER") or getpass.getuser()
    dirs = sorted({str(d) for d in job_dirs} | {os.path.realpath(str(d)) for d in job_dirs})
    iwd = " || ".join(f'Iwd=="{d}"' for d in dirs)
    constraint = f'Owner=="{owner}" && ({iwd})'
    schedds = list_schedds()
    stats = dict(schedds=len(schedds), history_ok=0, queue_ok=0, queue_complete=False)
    if not schedds:
        return {}, stats

    def per_schedd(s):
        base = ["-name", s, "-constraint", constraint, "-af"] + ATTRS
        return (_run(["condor_q"] + base, s, "queue", timeout),
                _run(["condor_history"] + base, s, "history", timeout))

    ads: dict = {}
    with ThreadPoolExecutor(max_workers=threads) as ex:
        for q, h in ex.map(per_schedd, schedds):
            stats["history_ok"] += h is not None
            stats["queue_ok"] += q is not None
            ads.update(h or {})
            ads.update(q or {})  # queue entries override history
    stats["queue_complete"] = stats["queue_ok"] == stats["schedds"] > 0
    return ads, stats


def describe(ad: dict) -> str:
    return (f"schedd_status={STATUS_NAMES.get(ad['status'], ad['status'])} exit={ad['exit_code']} "
            f"signal={ad['signal'] if ad['by_signal'] else '-'} suspended={ad['suspensions']}x "
            f"starts={ad['starts']} host={ad['host']}")
