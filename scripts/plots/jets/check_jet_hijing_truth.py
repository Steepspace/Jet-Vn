#!/usr/bin/env python3
"""
Reco vs truth check of the run-31 HIJING-embedded PYTHIA8 Jet20 sample (tasks/jet-jes-comparison.md,
"Reco vs truth in the bad HIJING Jet20 events").

About 25% of the HIJING Jet20 (and Jet30) events have a leading truth R=0.4 jet below the 20 GeV generator filter,
against ~6% in run-28 p+p Jet20. This tells apart the two explanations:
  - the events are really soft: reco and truth agree, as in the control;
  - the truth record is incomplete: the calorimeter still sees the hard jet, so there is a reco jet > 20 GeV and the
    reco / truth response is far above the control at the same truth pT.
Control: HIJING Jet10 at the same leading truth R=0.4 pT. Segment N of each sample holds the same HIJING events, so the
background is identical. p+p Jet20 (HardQCD) and Jet12 (minimum bias) are the references without background, and
also show whether the generator processes change the response.

It reads the TreeFiller trees (Fun4All_HIJING_JES.C / Fun4All_PP_JES.C): reco iter jets (pt_calib >= 10 GeV), truth jets
(>= 5 GeV, |eta| < 1.5), max_pt_truth_r04. Centrality < 60% for HIJING (p+p trees store 9999).

Printed:
  1. per leading truth R=0.4 pT bin: fraction of events with a reco jet > 20 GeV in |eta| < 0.7, fraction with such a jet
     further than dR = 0.4 from every truth jet, and the median response of the leading |eta| < 0.7 truth jet
     (closest reco jet within dR < 0.2);
  2. in-window bias: events with leading truth R=0.4 in [21, 31) (the Jet20 window), response of the leading truth jet in
     bins of its pT. HIJING Jet10 is outside its own window there, used only as the unaffected control.
"""

import argparse
import concurrent.futures
import os
from pathlib import Path
import warnings
import numpy as np
import uproot
import awkward as ak

ETA_FID = 0.7  # reco and leading truth jet fiducial |eta|
MATCH_DR = 0.2  # leading truth jet to reco jet
UNMATCHED_DR = 0.4  # hard reco jet with no truth jet within this
HARD_PT = 20.0
LEAD_BINS = [(8, 12), (12, 16), (16, 20), (21, 31)]
WINDOW = (21, 31)  # HIJING Jet20 window
TRUTH_PT_BINS = [(16, 20), (20, 24), (24, 28), (28, 32)]
CORE = (0.5, 1.7)  # response range of the core mean

HIJING_LIST = "/sphenix/user/anarde/sEPD-Study/f4a_bkgsub_mc/11-19-26-hijing/tree.list"
PP_LIST = "/sphenix/user/anarde/sEPD-Study/f4a_bkgsub_mc/11-19-26-pythia/tree.list"


def files(lst, sample, n):
    """First n trees of a sample (jetNN directory in the path) in a tree list."""
    return [p for p in Path(lst).read_text().split() if f"/{sample}/" in p][:n]


def dphi(a, b):
    return (a - b + np.pi) % (2 * np.pi) - np.pi


def process(task):
    path, r_tag = task
    branches = ["max_pt_truth_r04", "centrality"] + [f"{v}_truth_{r_tag}" for v in ("pt", "eta", "phi")] + \
               [f"{v}_iter_{r_tag}" for v in ("pt_calib", "eta", "phi")]
    a = uproot.open(path)["T"].arrays(branches, library="ak")
    a = a[(a.centrality < 60) | (a.centrality >= 9999)]  # p+p trees: 9999
    tpt, teta, tphi = a[f"pt_truth_{r_tag}"], a[f"eta_truth_{r_tag}"], a[f"phi_truth_{r_tag}"]
    rpt, reta, rphi = a[f"pt_calib_iter_{r_tag}"], a[f"eta_iter_{r_tag}"], a[f"phi_iter_{r_tag}"]
    rsel = abs(reta) < ETA_FID
    rpt, reta, rphi = rpt[rsel], reta[rsel], rphi[rsel]
    lead_r = ak.fill_none(ak.max(rpt, axis=1), 0.0)  # 0 if no reco jet passes the tree cut
    # leading truth jet in |eta| < ETA_FID
    tf = abs(teta) < ETA_FID
    idx = ak.argmax(ak.where(tf, tpt, -1.0), axis=1, keepdims=True)
    lt_pt = ak.fill_none(ak.firsts(tpt[idx]), 0.0)
    lt_ok = ak.fill_none(ak.firsts(tf[idx]), False)
    lt_eta, lt_phi = ak.firsts(teta[idx]), ak.firsts(tphi[idx])
    # hard reco jets far from every truth jet
    pairs = ak.cartesian([ak.zip({"eta": reta, "phi": rphi}), ak.zip({"eta": teta, "phi": tphi})], nested=True)
    dr = np.sqrt((pairs["0"].eta - pairs["1"].eta) ** 2 + dphi(pairs["0"].phi, pairs["1"].phi) ** 2)
    dr_min = ak.fill_none(ak.min(dr, axis=2), 9.0)
    unmatched = ak.any((rpt > HARD_PT) & (dr_min > UNMATCHED_DR), axis=1)
    # response of the leading truth jet: closest reco jet within MATCH_DR
    d_lt = np.sqrt((reta - lt_eta) ** 2 + dphi(rphi, lt_phi) ** 2)
    best = ak.argmin(d_lt, axis=1, keepdims=True)
    m_dr = ak.fill_none(ak.firsts(d_lt[best]), 9.0)
    m_pt = ak.fill_none(ak.firsts(rpt[best]), 0.0)
    resp = ak.where(lt_ok & (m_dr < MATCH_DR), m_pt / lt_pt, np.nan)
    out = dict(lead4=a.max_pt_truth_r04, lead_r=lead_r, lt_pt=lt_pt, unmatched=unmatched, resp=resp)
    return {k: ak.to_numpy(v).astype(float) for k, v in out.items()}


def load(paths, r_tag, jobs):
    with concurrent.futures.ProcessPoolExecutor(max_workers=jobs) as ex:
        outs = list(ex.map(process, [(p, r_tag) for p in paths]))
    return {k: np.concatenate([o[k] for o in outs]) for k in outs[0]}


def finite(x):
    return x[np.isfinite(x)]


def main():
    parser = argparse.ArgumentParser(description="Reco vs truth in HIJING Jet20 events with the leading truth jet below the generator filter.")
    parser.add_argument("-r", "--radius", choices=["r02", "r03"], default="r03", help="Jet radius (default: r03)")
    parser.add_argument("--hijing-list", default=HIJING_LIST, help=f"HIJING tree list (default {HIJING_LIST})")
    parser.add_argument("--pp-list", default=PP_LIST, help=f"p+p tree list (default {PP_LIST})")
    parser.add_argument("--n-files", type=int, nargs=4, default=[300, 1500, 300, 3000], metavar=("H20", "H10", "PP20", "PP12"),
                        help="Trees per sample: HIJING Jet20, HIJING Jet10, p+p Jet20, p+p Jet12 (default: 300 1500 300 3000; "
                             "Jet10 and Jet12 need many for the in-window control)")
    parser.add_argument("-j", "--jobs", type=int, default=min(16, os.cpu_count() or 1), help="Parallel worker processes")
    parser.add_argument("--save-npz", type=Path, help="Save the per-event arrays to this .npz file")
    args = parser.parse_args()
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    n_h20, n_h10, n_p20, n_p12 = args.n_files
    samples = {"HIJING Jet20": files(args.hijing_list, "jet20", n_h20), "HIJING Jet10": files(args.hijing_list, "jet10", n_h10),
               "p+p Jet20": files(args.pp_list, "jet20", n_p20), "p+p Jet12": files(args.pp_list, "jet12", n_p12)}
    data = {}
    for name, paths in samples.items():
        if not paths:
            print(f"Warning: no trees for {name}, skipped")
            continue
        data[name] = load(paths, args.radius, args.jobs)
        d = data[name]
        print(f"{name}: {len(paths)} trees, {len(d['lead4'])} events, leading truth R=0.4 < 20 GeV: {np.mean(d['lead4'] < 20):.3f}")

    print(f"\nR = {args.radius}: reco iter pt_calib >= 10 GeV in |eta| < {ETA_FID}; truth jets >= 5 GeV in |eta| < 1.5")
    print(f"1. Per leading truth R=0.4 pT bin: reco jet > {HARD_PT:g} GeV, the same with no truth jet within dR {UNMATCHED_DR:g}, "
          f"median response of the leading truth jet")
    for name, d in data.items():
        for lo, hi in LEAD_BINS:
            s = (d["lead4"] >= lo) & (d["lead4"] < hi)
            if s.sum() < 50:
                continue
            resp = finite(d["resp"][s])
            print(f"  {name:<13} [{lo:>2},{hi:>2}): N {s.sum():>8}  reco > {HARD_PT:g}: {np.mean(d['lead_r'][s] > HARD_PT):.3f}"
                  f"  unmatched: {np.mean(d['unmatched'][s]):.4f}  response median {np.median(resp) if len(resp) else np.nan:.3f} (N {len(resp)})")
        print()

    print(f"2. In-window: leading truth R=0.4 in [{WINDOW[0]}, {WINDOW[1]}), response of the leading truth jet vs its pT: "
          f"median (core mean in {CORE[0]:g}-{CORE[1]:g}), fraction > 1.4")
    for lo, hi in TRUTH_PT_BINS:
        cells = []
        for name, d in data.items():
            s = (d["lead4"] >= WINDOW[0]) & (d["lead4"] < WINDOW[1]) & (d["lt_pt"] >= lo) & (d["lt_pt"] < hi)
            r = finite(d["resp"][s])
            if len(r) < 30:
                cells.append(f"{name}: N {len(r)}")
                continue
            core = r[(r > CORE[0]) & (r < CORE[1])]
            cells.append(f"{name}: N {len(r)} {np.median(r):.3f} ({core.mean():.3f}) {np.mean(r > 1.4):.3f}")
        print(f"  truth {args.radius} [{lo},{hi}): " + " | ".join(cells))

    if args.save_npz:
        np.savez(args.save_npz, **{f"{n.replace(' ', '_').replace('+', '')}__{k}": v for n, d in data.items() for k, v in d.items()})
        print(f"\nSaved: {args.save_npz}")


if __name__ == "__main__":
    main()
