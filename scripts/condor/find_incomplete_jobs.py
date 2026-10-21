#!/usr/bin/env python3
"""
find_incomplete_jobs.py

Audits a Condor job directory (as produced by genFun4All*.sh / genFun4All.sub) and finds
jobs that ended WITHOUT doing their work, even if condor says they terminated normally
(e.g. jobs suspended/evicted by an admin, nodes that vanished, killed wrappers).

For every stdout file (job-<cluster>-<proc>.out) the script combines:
  * the stdout tail         -> "Finished" line / fully-skimmed / missing-input markers
  * the condor event log    -> last terminal event (005 terminated + return value,
                               009 aborted, 012 held, 004 evicted, 010 suspended, ...)
  * optionally condor_q     -> running-looking jobs that are no longer in the queue (LOST)

Jobs are grouped by their INPUT list ("Reading inputs from: ..." in stdout), so an input
that was already redone successfully by a later cluster is NOT reported again, and inputs
in jobs.list that never produced any stdout are reported as NEVER_RAN.

Outputs (in the job directory by default):
  resubmit_incomplete.list  one input per line, directly usable with
                            condor_submit <sub> -queue "input_dst from resubmit_incomplete.list"
  incomplete_report.tsv     one row per problem input: category, job id, details

Categories:
  OK_FINISHED / OK_SKIMMED   success (not reported)
  MISSING_INPUT              catalog/data files missing; resubmitting will not help (excluded)
  TERMINATED_NO_FINISH       condor exit 0 but no "Finished" line (suspended / killed)   -> resubmit
  FAILED_EXIT                condor exit != 0 or killed by signal                          -> resubmit
  FAILED_ERROR               stdout shows ROOT crash / cp failure etc.                     -> resubmit
  REMOVED                    condor_rm'd (009)                                             -> resubmit
  HELD                       last event is a hold (use resubmit_held_jobs.py for memory)   -> resubmit with --include-held
  LOST                       no terminal event, and every schedd queue readable w/o the job -> resubmit
  STALE                      no finish, no condor log, stdout silent > --stale-hours       -> only with --include-stale
  RUNNING                    still running/idle (not resubmitted)
  NEVER_RAN                  in jobs.list but no stdout at all                             -> resubmit

Condor event logs are local to each submit host (the `log =` dir, usually under /tmp), so from
one host you only see that host's logs. Job state therefore comes from, in order:
  1. the schedds themselves: condor_history/condor_q -name <schedd> for EVERY schedd, matched by
     Owner+Iwd (works from any host; history only covers recent jobs, remote condor_q may be refused)
  2. condor event logs you can see (--condor-log-dir, repeatable; <job_dir>/logs and the local
     `log =` dir are tried automatically)
  3. stdout silence (--stale-hours) -> STALE
A job absent from all of these is only called LOST if every schedd's queue was readable.

Usage:
    python3 find_incomplete_jobs.py /path/to/job_dir
    python3 find_incomplete_jobs.py /path/to/job_dir --memory 2GB
    python3 find_incomplete_jobs.py /path/to/job_dir --condor-log-dir /tmp/anarde/dump/event_qa --condor-log-dir /path/copy_from_host2
"""

import argparse
import os
import re
import subprocess
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    from resubmit_held_jobs import extract_log_directory, extract_queue_variable, find_submit_file
except Exception:  # pragma: no cover - standalone fallback
    find_submit_file = extract_queue_variable = extract_log_directory = None
import condor_schedd_utils as csu

JOB_RE = re.compile(r"^job-(\d+)-(\d+)\.out$")
INPUT_RE = re.compile(rb"^Reading inputs from:\s*(\S+)", re.MULTILINE)
FINISH_RE = re.compile(rb"^\s*Finished(\s+successfully.*)?\s*$", re.MULTILINE)
SKIM_PROC_RE = re.compile(rb"CaloStatusSkimmer(?:::End)?\s+Total events processed:\s*([0-9,]+)")
SKIM_DONE_RE = re.compile(rb"CaloStatusSkimmer(?:::End)?\s+Total events skimmed:\s*([0-9,]+)")
MISSING_RE = re.compile(
    rb"(All segments failed to fetch|all segments failed|getinputfiles failure"
    rb"|getinputfiles\.pl failed|Fetched 0 of \d+ segments|Aborted:\s*All segments)"
)
ERROR_RE = re.compile(
    rb"(Error:\s*ROOT macro (?:crashed|failed)|Error:\s*cp failed permanently|condor scratch NOT set"
    rb"|Segmentation fault|Fatal in <|Bus error|\bAborted\b)"
)
EVENT_RE = re.compile(r"^(\d{3}) \((\d+)\.(\d+)\.\d+\) (\d{4}-\d\d-\d\d \d\d:\d\d:\d\d) (.*)$")
RETVAL_RE = re.compile(r"return value (\d+)|signal (\d+)")

# categories that are resubmitted by default
RESUBMIT = {"TERMINATED_NO_FINISH", "FAILED_EXIT", "FAILED_ERROR", "REMOVED", "LOST", "NEVER_RAN"}
OK = {"OK_FINISHED", "OK_SKIMMED"}


def read_tail(path: Path, nbytes: int) -> bytes:
    size = path.stat().st_size
    with open(path, "rb") as f:
        if size > nbytes:
            f.seek(size - nbytes)
        return f.read()


def read_head(path: Path, nbytes: int) -> bytes:
    with open(path, "rb") as f:
        return f.read(nbytes)


def parse_condor_log(path: Path) -> dict:
    """Returns last-event summary of a single-job condor user log."""
    info = {"exists": False, "last": None, "term": None, "suspended": 0, "evicted": 0, "held": False,
            "last_time": "", "host": ""}
    try:
        text = path.read_text(errors="ignore")
    except OSError:
        return info
    info["exists"] = True
    lines = text.splitlines()
    for i, line in enumerate(lines):
        m = EVENT_RE.match(line)
        if not m:
            continue
        code, _, _, when, rest = m.groups()
        if code in ("000", "001", "004", "005", "009", "010", "011", "012", "013"):
            info["last"], info["last_time"] = code, when
        if code == "001":
            info["held"] = False
        elif code == "010":
            info["suspended"] += 1
        elif code == "004":
            info["evicted"] += 1
        elif code == "012":
            info["held"] = True
        elif code == "013":
            info["held"] = False
        elif code == "005":
            rv = None
            for nxt in lines[i + 1:i + 3]:
                r = RETVAL_RE.search(nxt)
                if r:
                    rv = ("exit", int(r.group(1))) if r.group(1) is not None else ("signal", int(r.group(2)))
                    break
            info["term"] = rv or ("exit", None)
    for line in lines:
        if "SlotName" in line:
            info["host"] = line.split("@")[-1].split(".")[0].strip()
    # a job restarted after a terminal event is not terminal any more
    if info["last"] in ("001", "000", "010", "011", "004", "013") and info["term"] is not None:
        info["term"] = None
    return info


def find_condor_log(log_dirs: list, cluster: int, proc: int) -> Path | None:
    for d in log_dirs:
        p = d / f"job-{cluster}-{proc}.log"
        if p.is_file():
            return p
    return None


def classify(stdout_path: Path, log_dirs: list, tail_bytes: int, ads: dict, queue_complete: bool, stale_sec: float):
    m = JOB_RE.match(stdout_path.name)
    cluster, proc = int(m.group(1)), int(m.group(2))
    jid = f"{cluster}.{proc}"
    head = read_head(stdout_path, 65536)
    im = INPUT_RE.search(head)
    inp = im.group(1).decode() if im else None
    tail = read_tail(stdout_path, tail_bytes)

    cinfo = None
    lp = find_condor_log(log_dirs, cluster, proc)
    if lp is not None:
        cinfo = parse_condor_log(lp)

    def result(cat, detail=""):
        return dict(job=jid, input=inp, cat=cat, detail=detail, mtime=stdout_path.stat().st_mtime)

    if FINISH_RE.search(tail):
        return result("OK_FINISHED")
    pm, sm = SKIM_PROC_RE.findall(tail), SKIM_DONE_RE.findall(tail)
    if pm and sm and pm[-1] == sm[-1]:
        return result("OK_SKIMMED", f"{pm[-1].decode()} events")
    if MISSING_RE.search(tail) or MISSING_RE.search(head):
        return result("MISSING_INPUT")

    extra = ""
    ad = ads.get(jid)
    if ad:
        extra = csu.describe(ad)
        st = ad["status"]
        if st == 4:
            if ad["by_signal"] or ad["exit_code"] not in (0, None):
                return result("FAILED_EXIT", extra)
            return result("TERMINATED_NO_FINISH", extra)
        if st == 3:
            return result("REMOVED", extra)
        if st == 5:
            return result("HELD", extra)
        return result("RUNNING", extra)
    if cinfo and cinfo["exists"]:
        extra = (f"host={cinfo['host']} suspended={cinfo['suspended']} evicted={cinfo['evicted']} "
                 f"last_event={cinfo['last']}@{cinfo['last_time']}")
        if cinfo["term"] is not None:
            kind, val = cinfo["term"]
            if kind == "exit" and val == 0:
                return result("TERMINATED_NO_FINISH", extra)
            return result("FAILED_EXIT", f"{kind}={val} {extra}")
        if cinfo["last"] == "009":
            return result("REMOVED", extra)
        if cinfo["held"]:
            return result("HELD", extra)
    em = ERROR_RE.search(tail)
    if em and (cinfo is None or not cinfo["exists"] or cinfo["last"] == "005"):
        return result("FAILED_ERROR", em.group(1).decode(errors="ignore"))
    if queue_complete and (cinfo is None or not cinfo["exists"]):
        return result("LOST", "not in any schedd queue/history and no terminal event")
    if not queue_complete and (cinfo is None or not cinfo["exists"]):
        import time
        age = time.time() - stdout_path.stat().st_mtime
        if age > stale_sec:
            return result("STALE", f"no condor log; stdout silent for {age / 3600:.1f} h")
    return result("RUNNING", extra)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("job_dir", type=Path)
    ap.add_argument("--stdout-dir", type=Path, help="default: <job_dir>/stdout")
    ap.add_argument("--condor-log-dir", type=Path, action="append", default=[],
                    help="dir with condor event logs (job-C-P.log); repeatable, e.g. one copy per submit host. "
                         "Also tried: <job_dir>/logs and the 'log =' dir of the .sub if it exists on this host")
    ap.add_argument("--no-auto-logs", action="store_true",
                    help="do not auto-try <job_dir>/logs and the .sub 'log =' dir (only explicit --condor-log-dir)")
    ap.add_argument("--stale-hours", type=float, default=2.0,
                    help="for jobs with no schedd ad or condor log: stdout silent longer than this => STALE")
    ap.add_argument("--include-stale", action="store_true", help="also put STALE jobs in the resubmit list")
    ap.add_argument("--jobs-list", type=Path, help="default: <job_dir>/jobs.list (enables NEVER_RAN detection)")
    ap.add_argument("--sub", type=Path, help="submit file (default: auto-detect)")
    ap.add_argument("--no-schedd-query", action="store_true",
                    help="do not ask the schedds (condor_q/condor_history); use stdout + visible logs only")
    ap.add_argument("--include-held", action="store_true", help="also put HELD jobs in the resubmit list")
    ap.add_argument("--include-running", action="store_true", help="also put RUNNING jobs in the list (not advised)")
    ap.add_argument("--memory", help="request_memory for the printed condor_submit (e.g. 2GB)")
    ap.add_argument("-n", "--dry-run", action="store_true",
                    help="analyse and print the summary, a sample of the resubmit list and the command, but write no files")
    ap.add_argument("--tail-bytes", type=int, default=16384)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("-o", "--output", type=Path, help="resubmit list (default: <job_dir>/resubmit_incomplete.list)")
    ap.add_argument("--report", type=Path, help="TSV report (default: <job_dir>/incomplete_report.tsv)")
    args = ap.parse_args()

    jd = args.job_dir.resolve()
    sdir = args.stdout_dir or jd / "stdout"
    if not sdir.is_dir():
        print(f"stdout dir not found: {sdir}", file=sys.stderr)
        return 2
    sub = find_submit_file(jd, args.sub) if find_submit_file else args.sub
    log_dirs = list(args.condor_log_dir)
    if (jd / "logs").is_dir() and not args.no_auto_logs:
        log_dirs.append(jd / "logs")
    if sub and extract_log_directory and not args.no_auto_logs:
        cand = extract_log_directory(sub, default="")
        if str(cand) and cand.is_dir():
            log_dirs.append(cand)
    log_dirs = [d for i, d in enumerate(log_dirs) if d not in log_dirs[:i]]
    if not log_dirs:
        print("[INFO] no condor event logs visible from this host; using schedd queries + stdout staleness")
    else:
        print("condor logs : " + ", ".join(str(d) for d in log_dirs))
    print(f"stdout dir  : {sdir}")

    ads, queue_complete = {}, False
    if not args.no_schedd_query and csu.condor_available():
        ads, st = csu.query_job_ads([args.job_dir, jd])
        queue_complete = st["queue_complete"]
        print(f"schedd query: {len(ads)} job ads from {st['schedds']} schedds "
              f"(history ok on {st['history_ok']}, queue ok on {st['queue_ok']}); "
              f"coverage is partial, jobs without an ad fall back to logs/stdout")

    files = [Path(e.path) for e in os.scandir(sdir) if JOB_RE.match(e.name)]
    print(f"scanning {len(files)} stdout files ...")
    with ThreadPoolExecutor(max_workers=args.threads) as ex:
        results = list(ex.map(lambda p: classify(p, log_dirs, args.tail_bytes, ads, queue_complete, args.stale_hours * 3600), files, chunksize=64))

    # Group by input; best/most-recent instance wins
    by_input = defaultdict(list)
    no_input = []
    for r in results:
        (by_input[r["input"]] if r["input"] else no_input).append(r)

    final = {}
    for inp, rs in by_input.items():
        ok = [r for r in rs if r["cat"] in OK]
        if ok:
            final[inp] = ok[0]
            continue
        miss = [r for r in rs if r["cat"] == "MISSING_INPUT"]
        running = [r for r in rs if r["cat"] == "RUNNING"]
        if running:
            final[inp] = running[0]
        elif miss:
            final[inp] = miss[0]
        else:
            final[inp] = max(rs, key=lambda r: r["mtime"])
    for r in no_input:  # stdout without an input line: crashed before/at start
        final[f"<unknown:{r['job']}>"] = r

    jl = args.jobs_list or jd / "jobs.list"
    never = []
    if jl.is_file():
        for line in jl.read_text().splitlines():
            line = line.strip()
            if line and line not in final:
                never.append(line)
                final[line] = dict(job="-", input=line, cat="NEVER_RAN", detail="no stdout for this input", mtime=0)
    else:
        print(f"[INFO] {jl} not found; NEVER_RAN detection skipped")

    want = set(RESUBMIT)
    if args.include_held:
        want.add("HELD")
    if args.include_stale:
        want.add("STALE")
    if args.include_running:
        want.add("RUNNING")
    to_resub = sorted(i for i, r in final.items() if r["cat"] in want and not i.startswith("<unknown"))
    unknown = [r for r in final.values() if r["input"] is None or str(r["input"]).startswith("<unknown")]

    counts = Counter(r["cat"] for r in final.values())
    print("\nSummary (one row per input):")
    for cat, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"  {cat:22s} {n}")
    sus = sum(1 for r in final.values() if r["cat"] == "TERMINATED_NO_FINISH" and "suspended=0" not in r["detail"])
    if sus:
        print(f"  ({sus} TERMINATED_NO_FINISH had a suspend event)")
    if unknown:
        print(f"[WARNING] {len(unknown)} stdout files have no 'Reading inputs from' line; "
              f"cannot map to inputs: {[r['job'] for r in unknown][:5]} ...")

    outp = args.output or jd / "resubmit_incomplete.list"
    rep = args.report or jd / "incomplete_report.tsv"
    if args.dry_run:
        print(f"\n[DRY-RUN] no files written. Would write {len(to_resub)} inputs to {outp} and a report to {rep}")
        for i in to_resub[:10]:
            r = final[i]
            print(f"  {r['cat']:22s} {r['job']:>16s}  {i}  {r['detail'][:100]}")
        if len(to_resub) > 10:
            print(f"  ... and {len(to_resub) - 10} more")
    else:
        outp.write_text("".join(i + "\n" for i in to_resub))
        with open(rep, "w") as f:
            f.write("category\tjob\tinput\tdetail\n")
            for i, r in sorted(final.items(), key=lambda kv: (kv[1]["cat"], kv[1]["job"])):
                if r["cat"] not in OK:
                    f.write(f"{r['cat']}\t{r['job']}\t{i}\t{r['detail']}\n")
        print(f"\nresubmit list : {outp}  ({len(to_resub)} inputs)")
        print(f"report        : {rep}")

    if to_resub:
        qv = extract_queue_variable(sub) if extract_queue_variable else "input_dst"
        mem = f" request_memory={args.memory}" if args.memory else ""
        print("\nResubmit with:\n")
        print(f'  cd {jd} && condor_submit {sub.name if sub else "genFun4All.sub"}{mem} '
              f'-queue "{qv} from {outp.name}"\n')
        print("Note: old stdout/error/logs for these inputs are kept; the checker prefers a later "
              "successful run of the same input, so rerun this script after the resubmission finishes.")
    return 1 if to_resub else 0


if __name__ == "__main__":
    sys.exit(main())
