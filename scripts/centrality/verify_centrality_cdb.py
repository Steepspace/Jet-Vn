#!/usr/bin/env python3
"""
Verify Centrality CDB Payloads

For every run in a run list, this script resolves the Centrality,
CentralityScale and CentralityVertexScale payloads from the CDB and checks
them against exactly what CentralityReco.cc reads, i.e. that

    m_mbd_total_charge += q * getVertexScale() * m_centrality_scale

can never be 0 or NaN for vertices inside |z| < zmax, and that the 100
centrality divisions give a well-defined centile.

Reference:
    coresoftware/offline/packages/centrality/CentralityReco.cc
"""

import argparse
import collections
import concurrent.futures
import csv
import math
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import tqdm
import uproot

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "CDB"))
import check_cdb_list  # noqa: E402

PAYLOADS = ["Centrality", "CentralityScale", "CentralityVertexScale"]
NDIVS = 100  # CentralityReco::NDIVS
TREE_NAME = "Multiple"
INT_MIN = -2147483648  # CDBTTree::GetIntValue for a missing entry
RUN_IN_FILENAME = re.compile(r"_(\d+)\.root$")

STATUS_RANK = {"OK": 0, "WARN": 1, "FAIL": 2}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Verify CDB centrality payloads do not break CentralityReco for a list of runs."
    )
    parser.add_argument("run_list_file", type=Path, help="Text file containing run numbers (one per line)")
    parser.add_argument("outDir", nargs="?", type=Path, default=Path("test-centrality-cdb"), help="(optional) output directory (default: test-centrality-cdb)")
    parser.add_argument("dbtag", nargs="?", type=str, default="newcdbtag", help="(optional) database tag (default: newcdbtag)")
    parser.add_argument("-j", "--jobs", type=int, default=min(os.cpu_count() or 4, 32), help="Number of parallel jobs (default: auto)")
    parser.add_argument("--zmax", type=float, default=10.0, help="|z| < zmax [cm] window in which the effective scale must be finite and > 0 (default: 10)")
    parser.add_argument("--znominal", type=float, default=60.0, help="Nominal |z| [cm] the non-zero vertex scale is expected to cover; narrower coverage is a WARN (default: 60)")
    parser.add_argument("--zstep", type=float, default=0.5, help="z grid step [cm] used to emulate getVertexScale() (default: 0.5)")
    parser.add_argument("--scale-min", type=float, default=0.8, help="Lower bound for a plausible scale value; outside is a WARN (default: 0.8)")
    parser.add_argument("--scale-max", type=float, default=1.2, help="Upper bound for a plausible scale value; outside is a WARN (default: 1.2)")
    parser.add_argument("-n", "--dry-run", action="store_true", help="Run all checks and print the summary, but do not write any output files (outDir is ignored)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Print all FAIL/WARN runs with reasons instead of the first 20")
    return parser.parse_args()


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


class Report:
    """Collects (status, key, detail) messages for one payload."""

    def __init__(self, payload: str):
        self.payload = payload
        self.messages: List[Tuple[str, str, str]] = []

    def add(self, status: str, key: str, detail: str = ""):
        self.messages.append((status, key, detail))

    @property
    def status(self) -> str:
        if not self.messages:
            return "OK"
        return max((m[0] for m in self.messages), key=STATUS_RANK.get)


def read_cdbttree(url: str, branches: List[str], report: Report) -> Optional[Dict[str, Dict[int, Any]]]:
    """
    Reads a CDBTTree "Multiple" payload the way CDBTTree::LoadCalibrations does:
    values are keyed by the IID branch, not by row number.

    Returns {branch: {iid: value}} or None on a FAIL.
    """
    path = Path(url)
    if not path.is_file():
        report.add("FAIL", "file missing", url)
        return None
    if path.stat().st_size == 0:
        report.add("FAIL", "0-byte file", url)
        return None

    try:
        with uproot.open(path) as f:
            if TREE_NAME not in f:
                report.add("FAIL", f"no '{TREE_NAME}' tree", url)
                return None
            tree = f[TREE_NAME]
            missing = [b for b in ["IID"] + branches if b not in tree]
            if missing:
                report.add("FAIL", "missing branch", ",".join(missing))
                return None
            arrays = tree.arrays(["IID"] + branches, library="np")
    except Exception as e:
        report.add("FAIL", "unreadable", str(e))
        return None

    iids = arrays["IID"]
    if len(np.unique(iids)) != len(iids):
        report.add("WARN", "duplicate IID", "")
    return {b: dict(zip(iids.tolist(), arrays[b].tolist())) for b in branches}


def check_centrality_divs(url: str, report: Report, info: Dict[str, Any]):
    data = read_cdbttree(url, ["Fcentralitydiv"], report)
    if data is None:
        return

    # CentralityReco::Download_centralityDivisions: m_centrality_map[idiv] = GetFloatValue(idiv, "centralitydiv")
    divs = np.array([data["Fcentralitydiv"].get(i, math.nan) for i in range(NDIVS)], dtype=np.float32)
    info["div_0"] = float(divs[0])
    info["div_99"] = float(divs[-1])

    n_missing = sum(1 for i in range(NDIVS) if i not in data["Fcentralitydiv"])
    if n_missing:
        report.add("FAIL", "missing divisions", f"{n_missing} of {NDIVS} IIDs absent")
    if not np.all(np.isfinite(divs)):
        report.add("FAIL", "non-finite division", f"{int(np.sum(~np.isfinite(divs)))} NaN/inf")
        return
    if np.all(divs == 0):
        report.add("FAIL", "all divisions zero")
        return
    if np.any(divs < 0):
        report.add("FAIL", "negative division", f"min={divs.min():.4g}")

    # FillCentralityInfo picks the first i with div[i] < charge, so divs must be non-increasing
    rising = np.where(np.diff(divs) > 0)[0]
    if len(rising):
        report.add("FAIL", "divisions not non-increasing", f"first at i={rising[0]}: {divs[rising[0]]:.4g} -> {divs[rising[0] + 1]:.4g}")

    info["n_dup_divs"] = int(np.sum(np.diff(divs) == 0))


def check_centrality_scale(url: str, report: Report, info: Dict[str, Any], scale_range: Tuple[float, float]) -> Optional[float]:
    data = read_cdbttree(url, ["Dcentralityscale"], report)
    if data is None:
        return None

    # CentralityReco::Download_centralityScale: m_centrality_scale = GetDoubleValue(0, "centralityscale")
    if 0 not in data["Dcentralityscale"]:
        report.add("FAIL", "no entry IID=0")
        return None
    scale = float(data["Dcentralityscale"][0])
    info["centscale"] = scale

    if not math.isfinite(scale):
        report.add("FAIL", "scale non-finite", str(scale))
    elif scale <= 0:
        report.add("FAIL", "scale <= 0", f"{scale:.4g}")
    elif not scale_range[0] <= scale <= scale_range[1]:
        report.add("WARN", "scale out of range", f"{scale:.4g}")
    return scale


def get_vertex_scale(bins: List[Tuple[float, float, float]], z: float) -> float:
    """Emulates CentralityReco::getVertexScale (float precision, first match wins)."""
    for low, high, scale in bins:
        if low < z <= high:
            return scale
    return 0.0


def check_vertex_scale(url: str, report: Report, info: Dict[str, Any], scale_range: Tuple[float, float], zmax: float) -> Optional[List[Tuple[float, float, float]]]:
    data = read_cdbttree(url, ["Invertexbins", "Dlow_vertex", "Dhigh_vertex", "Dscale"], report)
    if data is None:
        return None

    # CentralityReco::Download_centralityVertexScales
    nbins = data["Invertexbins"].get(0, INT_MIN)
    info["n_vtx_bins"] = nbins
    if nbins == INT_MIN or nbins <= 0:
        report.add("FAIL", "invalid nvertexbins", str(nbins))
        return None
    n_entries = len(data["Dscale"])
    if nbins > n_entries:
        report.add("FAIL", "nvertexbins > entries", f"{nbins} > {n_entries}")
    elif nbins < n_entries:
        report.add("WARN", "nvertexbins < entries", f"{nbins} < {n_entries}")

    bins = []
    for iv in range(nbins):
        low = float(np.float32(data["Dlow_vertex"].get(iv, math.nan)))
        high = float(np.float32(data["Dhigh_vertex"].get(iv, math.nan)))
        scale = float(np.float32(data["Dscale"].get(iv, math.nan)))
        bins.append((low, high, scale))

        overlaps_window = high > -zmax and low < zmax
        if not (math.isfinite(low) and math.isfinite(high)):
            report.add("FAIL", "non-finite bin edge", f"bin {iv}: ({low}, {high}]")
            continue
        if low >= high:
            report.add("FAIL", "bin low >= high", f"bin {iv}: ({low}, {high}]")
        if not overlaps_window:
            continue
        if not math.isfinite(scale) or scale <= 0:
            report.add("FAIL", "vertex scale <= 0 or NaN in window", f"bin {iv}: ({low:g}, {high:g}] scale={scale:.4g}")
        elif not scale_range[0] <= scale <= scale_range[1]:
            report.add("WARN", "vertex scale out of range", f"bin {iv}: ({low:g}, {high:g}] scale={scale:.4g}")

    # Gaps / overlaps between adjacent bins
    edges = sorted((b for b in bins if math.isfinite(b[0]) and math.isfinite(b[1])), key=lambda b: b[0])
    for (lo_a, hi_a, _), (lo_b, hi_b, _) in zip(edges, edges[1:]):
        if hi_a == lo_b:
            continue
        kind = "gap" if hi_a < lo_b else "overlap"
        in_window = min(hi_a, lo_b) < zmax and max(hi_a, lo_b) > -zmax
        report.add("FAIL" if in_window else "WARN", f"bin {kind}{' in window' if in_window else ''}", f"({lo_a:g}, {hi_a:g}] / ({lo_b:g}, {hi_b:g}]")
    if edges and (edges[0][0] >= -zmax or edges[-1][1] < zmax):
        report.add("FAIL", "bins do not cover window", f"({edges[0][0]:g}, {edges[-1][1]:g}]")

    return bins


def check_effective_scale(bins: List[Tuple[float, float, float]], centscale: float, report: Report, info: Dict[str, Any], zmax: float, znominal: float, zstep: float):
    """Emulates the per-PMT multiplier getVertexScale() * m_centrality_scale on a z grid."""
    zwide = max(znominal, zmax) + 100.0
    zgrid = np.arange(-zwide, zwide + zstep / 2, zstep, dtype=np.float32)
    cs = np.float32(centscale)
    eff = np.array([np.float32(get_vertex_scale(bins, float(z))) * cs for z in zgrid], dtype=np.float32)

    good = np.isfinite(eff) & (eff > 0)
    in_window = np.abs(zgrid) < zmax
    if np.any(in_window & ~good):
        bad_z = zgrid[in_window & ~good]
        report.add("FAIL", "effective scale 0/NaN in window", f"{len(bad_z)} z points, e.g. z={bad_z[0]:g}")

    if np.any(in_window & good):
        info["eff_min"] = float(eff[in_window & good].min())
        info["eff_max"] = float(eff[in_window & good].max())

    if np.any(good):
        # Contiguous non-zero range around z=0
        idx0 = int(np.argmin(np.abs(zgrid)))
        if good[idx0]:
            lo = idx0
            while lo > 0 and good[lo - 1]:
                lo -= 1
            hi = idx0
            while hi < len(zgrid) - 1 and good[hi + 1]:
                hi += 1
            info["nonzero_zlow"] = float(zgrid[lo])
            info["nonzero_zhigh"] = float(zgrid[hi])
            if zgrid[lo] > -znominal + zstep or zgrid[hi] < znominal - zstep:
                report.add("WARN", "non-zero z range narrower than nominal", f"[{zgrid[lo]:g}, {zgrid[hi]:g}]")


def check_run(args: Tuple[int, str, float, float, float, Tuple[float, float]]) -> Dict[str, Any]:
    run, dbtag, zmax, znominal, zstep, scale_range = args
    check_cdb_list._init_cdb()

    info: Dict[str, Any] = {"run": run}
    reports = {p: Report(p) for p in PAYLOADS}
    combined = Report("Combined")

    urls: Dict[str, Optional[str]] = {}
    for p in PAYLOADS:
        try:
            url = str(check_cdb_list._ROOT.get_sphenix_cdb_url(p, int(run), dbtag))
        except Exception as e:
            url = f"DataBaseException: {e}"
        # Lookup errors come back as "DataBaseException: ..." or "BaseException: ...";
        # CentralityReco aborts on anything that is not a .root file
        if not url.endswith(".root"):
            reports[p].add("FAIL", "no CDB payload", url)
            urls[p] = None
        else:
            urls[p] = url
            m = RUN_IN_FILENAME.search(url)
            if m and int(m.group(1)) != run:
                reports[p].add("WARN", "payload from another run", m.group(1))
        info[f"url_{p}"] = urls[p] or ""

    if urls["Centrality"]:
        check_centrality_divs(urls["Centrality"], reports["Centrality"], info)
    centscale = None
    if urls["CentralityScale"]:
        centscale = check_centrality_scale(urls["CentralityScale"], reports["CentralityScale"], info, scale_range)
    bins = None
    if urls["CentralityVertexScale"]:
        bins = check_vertex_scale(urls["CentralityVertexScale"], reports["CentralityVertexScale"], info, scale_range, zmax)

    if centscale is not None and bins is not None:
        check_effective_scale(bins, centscale, combined, info, zmax, znominal, zstep)
    else:
        combined.add("FAIL", "skipped (payload unusable)")

    all_reports = list(reports.values()) + [combined]
    for r in all_reports:
        info[f"status_{r.payload}"] = r.status
    info["status"] = max((r.status for r in all_reports), key=STATUS_RANK.get)
    info["messages"] = [(r.payload, s, k, d) for r in all_reports for s, k, d in r.messages]
    return info


def read_run_list(path: Path) -> List[int]:
    runs = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                try:
                    runs.append(int(line))
                except ValueError:
                    pass
    return runs


def print_run_messages(title: str, results: List[Dict[str, Any]], status: str, verbose: bool, limit: int = 20):
    print(f"\n{title} ({len(results)}):")
    shown = results if verbose else results[:limit]
    for res in shown:
        msgs = [f"{p}: {k}{' (' + d + ')' if d else ''}" for p, s, k, d in res["messages"] if s == status]
        print(f"  Run {res['run']}: " + "; ".join(msgs))
    if len(results) > len(shown):
        print(f"  ... and {len(results) - len(shown)} more (use -v to display all)")


def main():
    args = parse_args()

    if not args.run_list_file.exists():
        print(f"Error: Could not open file {args.run_list_file}", file=sys.stderr)
        sys.exit(2)

    runs = read_run_list(args.run_list_file)
    if not runs:
        print("Error: No valid runs found in input file.", file=sys.stderr)
        sys.exit(2)

    scale_range = (args.scale_min, args.scale_max)
    print(f"\n=======================================================")
    print(f" Centrality CDB Payload Verification")
    print(f"=======================================================")
    print(f"Run list : {args.run_list_file} ({len(runs)} runs)")
    print(f"DB tag   : {args.dbtag}")
    print(f"z window : |z| < {args.zmax:g} cm (nominal non-zero coverage {args.znominal:g} cm)")
    print(f"Scale    : plausible range [{args.scale_min:g}, {args.scale_max:g}]\n")

    tasks = [(run, args.dbtag, args.zmax, args.znominal, args.zstep, scale_range) for run in runs]
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.jobs, initializer=check_cdb_list.init_worker, initargs=(args.dbtag, runs[0], PAYLOADS[0])) as executor:
        results = list(tqdm.tqdm(executor.map(check_run, tasks), total=len(tasks), desc="Checking runs"))

    by_status = {s: [r for r in results if r["status"] == s] for s in STATUS_RANK}

    # Outputs
    csv_path = args.outDir / "centrality_cdb.csv"
    if not args.dry_run:
        args.outDir.mkdir(parents=True, exist_ok=True)
        columns = ["run", "status"] + [f"status_{p}" for p in PAYLOADS + ["Combined"]] + [
            "centscale", "div_0", "div_99", "n_dup_divs", "n_vtx_bins",
            "nonzero_zlow", "nonzero_zhigh", "eff_min", "eff_max",
        ] + [f"url_{p}" for p in PAYLOADS] + ["reasons"]

        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
            writer.writeheader()
            for res in results:
                row = dict(res)
                row["reasons"] = " | ".join(f"{s} {p}: {k}{' (' + d + ')' if d else ''}" for p, s, k, d in res["messages"])
                writer.writerow(row)

        for status, fname in [("OK", "good_runs.list"), ("WARN", "warn_runs.list"), ("FAIL", "bad_runs.list")]:
            with (args.outDir / fname).open("w") as f:
                for res in by_status[status]:
                    f.write(f"{res['run']}\n")

    # Per-payload summary
    total = len(results)
    headers = ["Payload", "OK", "WARN", "FAIL"]
    rows = []
    for p in PAYLOADS + ["Combined"]:
        counts = collections.Counter(r[f"status_{p}"] for r in results)
        rows.append([p] + [f"{counts[s]} ({counts[s] / total * 100:5.1f}%)" for s in STATUS_RANK])
    rows.append(["Overall"] + [f"{len(by_status[s])} ({len(by_status[s]) / total * 100:5.1f}%)" for s in STATUS_RANK])
    print("\nStatus per payload:")
    print_table(headers, rows)

    # Reason breakdown
    reason_counts = collections.Counter((s, p, k) for r in results for p, s, k, _ in set(r["messages"]))
    if reason_counts:
        print("\nReason breakdown (runs affected):")
        print_table(
            ["Status", "Payload", "Reason", "Runs"],
            [[s, p, k, str(n)] for (s, p, k), n in sorted(reason_counts.items(), key=lambda x: (-STATUS_RANK[x[0][0]], -x[1]))],
        )

    # Value ranges over usable runs
    for key, label in [("centscale", "CentralityScale"), ("eff_min", "min eff. scale in window"), ("eff_max", "max eff. scale in window")]:
        vals = np.array([r[key] for r in results if key in r and math.isfinite(r[key])])
        if len(vals):
            print(f"{label:<28}: min={vals.min():.4f}  median={np.median(vals):.4f}  max={vals.max():.4f}  (n={len(vals)})")

    if by_status["FAIL"]:
        print_run_messages("[FAIL] runs", by_status["FAIL"], "FAIL", args.verbose)
    if by_status["WARN"]:
        print_run_messages("[WARN] runs", by_status["WARN"], "WARN", args.verbose)

    if args.dry_run:
        print("\nDry run: no output files written.")
    else:
        print(f"\nWrote: {csv_path}")
        print(f"       {args.outDir / 'good_runs.list'} ({len(by_status['OK'])})")
        print(f"       {args.outDir / 'warn_runs.list'} ({len(by_status['WARN'])})")
        print(f"       {args.outDir / 'bad_runs.list'} ({len(by_status['FAIL'])})")
    print("\nVerification completed.\n")
    sys.exit(1 if by_status["FAIL"] else 0)


if __name__ == "__main__":
    main()
