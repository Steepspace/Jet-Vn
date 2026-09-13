#!/usr/bin/env python3

import argparse
import concurrent.futures
import os
from pathlib import Path
import shutil
import sys
import tqdm
import pandas as pd

_cdb_initialized = False
_ROOT = None

def _init_cdb():
    global _cdb_initialized, _ROOT
    if _cdb_initialized:
        return
    import ROOT
    _ROOT = ROOT
    _ROOT.gSystem.Load("libsphenixnpc")
    _ROOT.gInterpreter.Declare("""
    #ifndef SPHENIX_CDB_HELPER
    #define SPHENIX_CDB_HELPER
    #include <sphenixnpc/SphenixClient.h>
    #include <sphenixnpc/CDBUtils.h>
    CDBUtils* g_sphenix_cdb = nullptr;
    std::string get_sphenix_cdb_url(const std::string& pl_type, unsigned int runnumber, const std::string& tag="newcdbtag") {
        if (!g_sphenix_cdb) {
            g_sphenix_cdb = new CDBUtils();
            g_sphenix_cdb->setGlobalTag(tag);
        }
        return g_sphenix_cdb->getUrl(pl_type, runnumber);
    }
    #endif
    """)
    _cdb_initialized = True

CDB_TYPES = [
    ("Centrality", "Centrality"),
    ("Centrality Default", "Centrality_default"),
    ("Centrality Scale", "CentralityScale"),
    ("Centrality Scale Default", "CentralityScale_default"),
    ("Centrality Vertex Scale", "CentralityVertexScale"),
    ("Centrality Vertex Scale Default", "CentralityVertexScale_default"),
    ("MBD QFIT", "MBD_QFIT"),
    ("MBD QFIT Default", "MBD_QFIT_default"),
    ("EMCal Calib Default", "CEMC_calib_ADC_to_ETower_default"),
    ("EMCal Calib", "CEMC_calib_ADC_to_ETower"),
    ("EMCal Bad Tower Map", "CEMC_BadTowerMap"),
    ("EMCal Frac Bad Chi2", "CEMC_hotTowers_fracBadChi2"),
    ("EMCal Mean Time", "CEMC_meanTime"),
    ("EMCal ZS Cross Calib", "CEMC_ZSCrossCalib"),
    ("HCALIN Frac Bad Chi2", "HCALIN_hotTowers_fracBadChi2"),
    ("HCALIN Mean Time", "HCALIN_meanTime"),
    ("HCALIN ZS Cross Calib", "HCALIN_ZSCrossCalib"),
    ("HCALOUT Frac Bad Chi2", "HCALOUT_hotTowers_fracBadChi2"),
    ("HCALOUT Mean Time", "HCALOUT_meanTime"),
    ("HCALOUT ZS Cross Calib", "HCALOUT_ZSCrossCalib"),
    ("sEPD Calib", "SEPD_NMIP_CALIB"),
    ("sEPD Event Plane", "SEPD_EventPlaneCalib")
]

def check_run(args):
    run_number, dbtag = args
    _init_cdb()
    results = {}
    for desc, name in CDB_TYPES:
        try:
            url = str(_ROOT.get_sphenix_cdb_url(str(name), int(run_number), str(dbtag)))
            has_calib = not url.startswith("DataBaseException")
            results[name] = has_calib
        except Exception as e:
            results[name] = False
    return run_number, results

def main():
    parser = argparse.ArgumentParser(description="Check CDB list for a list of runs.")
    parser.add_argument("run_list_file", type=Path, help="Text file containing run numbers (one per line)")
    parser.add_argument("outDir", nargs="?", type=Path, default=Path("test"), help="(optional) output directory (default: test)")
    parser.add_argument("dbtag", nargs="?", type=str, default="newcdbtag", help="(optional) database tag (default: newcdbtag)")

    parser.add_argument("-j", "--jobs", type=int, default=min(os.cpu_count() or 4, 32), help="Number of parallel jobs (default: auto)")

    args = parser.parse_args()

    if not args.run_list_file.exists():
        print(f"Error: Could not open file {args.run_list_file}")
        sys.exit(1)

    runnumbers = []
    with args.run_list_file.open('r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                try:
                    runnumbers.append(int(line))
                except ValueError:
                    pass

    if not runnumbers:
        print("Error: No valid runs found in input file.")
        sys.exit(1)

    total_runs = len(runnumbers)
    print(f"Checking {total_runs} runs against DB tag: {args.dbtag}...")

    tasks = [(run, args.dbtag) for run in runnumbers]
    run_calib_status = {}

    with concurrent.futures.ProcessPoolExecutor(max_workers=args.jobs) as executor:
        for run_number, results in tqdm.tqdm(executor.map(check_run, tasks), total=total_runs):
            run_calib_status[run_number] = results

    print("\nWriting individual run lists to disk...")

    if args.outDir.exists():
        if args.outDir.is_dir():
            shutil.rmtree(args.outDir)
        else:
            args.outDir.unlink()

    validDir = args.outDir / "valid_runs"
    missingDir = args.outDir / "missing_runs"
    validDir.mkdir(parents=True, exist_ok=True)
    missingDir.mkdir(parents=True, exist_ok=True)

    valid_runs = {name: [] for desc, name in CDB_TYPES}

    for run in runnumbers:
        results = run_calib_status.get(run, {})
        for desc, name in CDB_TYPES:
            if results.get(name, False):
                valid_runs[name].append(run)

    for desc, name in CDB_TYPES:
        validFilename = validDir / f"{name}_runs.txt"
        with validFilename.open("w") as f:
            for r in valid_runs[name]:
                f.write(f"{r}\n")

        missing_runs = [r for r in runnumbers if r not in valid_runs[name]]
        if missing_runs:
            missingFilename = missingDir / f"{name}_runs_missing.txt"
            with missingFilename.open("w") as f:
                for r in missing_runs:
                    f.write(f"{r}\n")

    csvFilename = args.outDir / "calibrations.csv"
    csv_data = []
    for run in runnumbers:
        row = {'Run': run}
        results = run_calib_status.get(run, {})
        for desc, name in CDB_TYPES:
            row[name] = 1 if results.get(name, False) else 0
        csv_data.append(row)

    if csv_data:
        df = pd.DataFrame(csv_data)
        df.to_csv(csvFilename, index=False)

    print("\n======================================")
    print("          CALIBRATION SUMMARY         ")
    print("======================================")
    for desc, name in CDB_TYPES:
        count = len(valid_runs[name])
        percentage = (count / total_runs) * 100.0 if total_runs > 0 else 0
        print(f"{desc:<35} : {count:>4} / {total_runs} ({percentage:>5.1f}%)")

    print("======================================")
    print("done")

if __name__ == "__main__":
    main()
