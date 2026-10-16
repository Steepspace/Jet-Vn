#!/usr/bin/env python3

import argparse
import concurrent.futures
import os
from pathlib import Path
import re
import sys
import numpy as np
import uproot
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, ScalarFormatter
import mplhep as hep

from plot_jet_bkgsub_compare import save_fig, calc_ratio_errors, setup_ratio_panel

# Subtraction method metadata (label, color, marker, hEvent bin with the flow pass count)
# iternf / multnf are gated on the flow failure of iter / mult, so they share the event count
METHODS = {
    "iter": {"label": "Iterative", "color": "#984ea3", "marker": "o", "event_bin": 2},
    "iternf": {"label": "Iterative No Flow", "color": "#e41a1c", "marker": "s", "event_bin": 2},
    "mult": {"label": "Multiplicity", "color": "#377eb8", "marker": "^", "event_bin": 3},
    "multnf": {"label": "Multiplicity No Flow", "color": "#ff7f00", "marker": "D", "event_bin": 3},
}

# (name, denominator, numerator, ratio label, per-event normalization)
# Flow vs no flow share the same event selection -> raw counts
# Iter vs mult have different flow failures -> per-event yield
COMPARISONS = [
    ("iter_vs_iternf", "iternf", "iter", "Flow / No Flow", False),
    ("mult_vs_multnf", "multnf", "mult", "Flow / No Flow", False),
    ("iter_vs_mult", "iter", "mult", "Mult / Iter", True),
    ("iternf_vs_multnf", "iternf", "multnf", "Mult NF / Iter NF", True),
]

# R = 0.2 vs R = 0.3 for the same method: both come from the same events -> raw counts
R_STYLES = {
    "r02": {"label": r"$R = 0.2$", "color": "#1b9e77", "marker": "o"},
    "r03": {"label": r"$R = 0.3$", "color": "#d95f02", "marker": "s"},
}
R_COMPARE_NAME = "r02_vs_r03"
R_COMPARE_RATIO_LABEL = r"$R$=0.3 / $R$=0.2"

TEXT_INFO = (
    r"$|z| < 10$ cm & MB" "\n"
    "Good Calo-Cent\n"
    r"Centrality: 0–60%" "\n"
    "No Flow Failure\n"
    "Energy > 0\n"
    r"$|\mathrm{Calo}\ v_{2}| < 0.48$"
)


def parse_list_file(list_file):
    """Returns {run: path} from a list of ROOT files named <run>.root."""
    runs = {}
    with Path(list_file).open("r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            root_path = Path(line)
            match = re.search(r"(\d+)\.root", root_path.name)
            if match:
                runs[int(match.group(1))] = root_path
    return runs


def load_luminosity(file_path):
    """Returns the luminosity in ub^-1 for |z| < 10 cm & MBD Trig, or None if unavailable.

    It is stored in nb^-1 in the first bin of hLuminosity.
    """
    with uproot.open(file_path) as rf:
        if "hLuminosity" not in rf:
            print(f"Warning: hLuminosity not found in {file_path}")
            return None
        return float(rf["hLuminosity"].values()[0]) * 1000.0


def add_header(ax, run_number, r_jet, lumi):
    """Run, R and luminosity (with its selection) on one line above the axes."""
    text = rf"Run: {run_number}"
    if r_jet is not None:
        text += rf", $R = {r_jet:g}$"
    if lumi is not None:
        text += rf", $\mathcal{{L}} = {lumi:.2f}\ \mu\mathrm{{b}}^{{-1}}$ ($|z| < 10$ cm & MBD Trig)"
    ax.text(1.0, 1.01, text, transform=ax.transAxes, ha="right", va="bottom", fontsize=14)


def load_hists(file_path, r_tag, kind="pt"):
    """Loads hJetPtv3 (kind="pt") or the eta projection of h2JetEtav3 (kind="eta") for each method, with the flow pass event counts."""
    data = {}
    with uproot.open(file_path) as rf:
        n_events = rf["hEvent"].values()
        for method, meta in METHODS.items():
            hist_name = f"hJetPtv3_{r_tag}_{method}" if kind == "pt" else f"h2JetEtav3_{r_tag}_{method}"
            if hist_name not in rf:
                print(f"Warning: {hist_name} not found in {file_path}")
                continue
            if kind == "pt":
                h = rf[hist_name]
                vals, edges = h.to_numpy()
                errors = h.errors()
            else:
                # y-projection of (pT, eta) -> eta
                proj = rf[hist_name].to_hist().project(1)
                vals = proj.values()
                edges = proj.axes[0].edges
                variances = proj.variances()
                errors = np.sqrt(np.where(np.isnan(variances) | (variances < 0), 0.0, variances))
            data[method] = {
                "values": vals,
                "edges": edges,
                "errors": errors,
                "n_events": n_events[meta["event_bin"]],
            }
    return data


def info_text(method_label=None):
    return TEXT_INFO if method_label is None else f"{method_label}\n{TEXT_INFO}"


def plot_pair(run_number, r_jet, lumi, hists, den, num, ratio_label, per_event, output_path, ratio_ylim=None, pt_max=None, save_pdf=False, styles=METHODS, method_label=None, ylabel=None):
    hep.style.use("ATLAS")

    curves = {}
    max_x_val = 0
    for method in (den, num):
        d = hists[method]
        scale = 1.0 / d["n_events"] if per_event else 1.0
        curves[method] = {
            "values": d["values"] * scale,
            "errors": d["errors"] * scale,
            "edges": d["edges"],
        }
        nz = np.where(d["values"] > 0)[0]
        if len(nz) > 0:
            max_x_val = max(max_x_val, d["edges"][nz[-1] + 1])

    x_max = pt_max if pt_max is not None else max(max_x_val, 60)

    fig, (ax, ax_ratio) = plt.subplots(2, 1, figsize=(8, 8), gridspec_kw={"height_ratios": [3, 1], "hspace": 0.08}, sharex=True)

    for method in (den, num):
        c = curves[method]
        meta = styles[method]
        hep.histplot((c["values"], c["edges"]), ax=ax, histtype="step", color=meta["color"], linewidth=2, label=meta["label"])

    # Ratio: num / den
    d, n = curves[den], curves[num]
    edges = d["edges"]
    centers = (edges[:-1] + edges[1:]) / 2.0
    valid_mask = (n["values"] > 0) & (d["values"] > 0) & (centers <= x_max)

    all_ratios = []
    if np.any(valid_mask):
        ratio = n["values"][valid_mask] / d["values"][valid_mask]
        ratio_err = calc_ratio_errors(ratio, n["errors"][valid_mask], d["errors"][valid_mask], d["values"][valid_mask])
        meta = styles[num]
        ax_ratio.errorbar(
            centers[valid_mask], ratio, yerr=ratio_err,
            fmt=meta["marker"], color=meta["color"], markersize=4, capsize=2, linewidth=1.0,
        )
        reliable = (ratio_err < 0.25)
        if np.any(reliable):
            all_ratios.extend(ratio[reliable])

    positive = np.concatenate([c["values"][c["values"] > 0] for c in curves.values()])
    y_max = np.max(positive) if len(positive) > 0 else 1.0
    y_min = 0.5 if not per_event else (np.min(positive) * 0.5 if len(positive) > 0 else 1e-9)

    ax.set_yscale("log")
    ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=20))
    ax.set_ylim(bottom=y_min, top=y_max * 10)
    # Leave room past the furthest point (auto range only) so outliers don't sit on the plot edge
    x_right = x_max if pt_max is not None else x_max * 1.04
    ax.set_xlim(left=0, right=x_right)
    ax.set_ylabel(ylabel if ylabel is not None else (r"$\frac{1}{N_{\mathrm{evt}}}\frac{dN}{dp_{T}}$ [GeV$^{-1}$]" if per_event else "Counts"))

    add_header(ax, run_number, r_jet, lumi)
    ax.text(0.95, 0.95, info_text(method_label), transform=ax.transAxes, ha="right", va="top", fontsize=13)
    ax.legend(loc="upper right", bbox_to_anchor=(0.95, 0.55), frameon=False, fontsize=13)

    setup_ratio_panel(ax_ratio, xlabel=r"Jet $p_{T}$ [GeV]", ylabel=ratio_label, ratio_ylim=ratio_ylim, all_ratios=all_ratios, min_margin=0.06)
    plt.subplots_adjust(left=0.14, right=0.96, bottom=0.10, top=0.94)

    save_fig(fig, output_path, save_pdf)
    plt.close(fig)

    # If auto x-max is above 60 and pt_max was not explicitly set, also make a version with x-max set to 60
    if pt_max is None and max_x_val > 60:
        pt60_path = output_path.with_name(f"{output_path.stem}_pt60{output_path.suffix}")
        plot_pair(run_number, r_jet, lumi, hists, den, num, ratio_label, per_event, pt60_path,
                  ratio_ylim=ratio_ylim, pt_max=60, save_pdf=save_pdf, styles=styles, method_label=method_label, ylabel=ylabel)


def plot_eta_pair(run_number, r_jet, lumi, hists, den, num, ratio_label, per_event, output_path, ratio_ylim=None, save_pdf=False, styles=METHODS, method_label=None, eta_limit=None, ratio_eta_limit=None, ylabel=None):
    hep.style.use("ATLAS")

    # Axis range and ratio range default to the eta acceptance for this R
    if eta_limit is None:
        eta_limit = round(1.1 - r_jet, 1)
    if ratio_eta_limit is None:
        ratio_eta_limit = eta_limit

    curves = {}
    for method in (den, num):
        d = hists[method]
        bin_width = np.diff(d["edges"])
        scale = 1.0 / (d["n_events"] * bin_width) if per_event else 1.0
        curves[method] = {
            "values": d["values"] * scale,
            "errors": d["errors"] * scale,
            "edges": d["edges"],
        }

    fig, (ax, ax_ratio) = plt.subplots(2, 1, figsize=(8, 8), gridspec_kw={"height_ratios": [3, 1], "hspace": 0.08}, sharex=True)

    for method in (den, num):
        c = curves[method]
        meta = styles[method]
        hep.histplot((c["values"], c["edges"]), ax=ax, histtype="step", color=meta["color"], linewidth=2, label=meta["label"])

    # Ratio: num / den
    d, n = curves[den], curves[num]
    edges = d["edges"]
    centers = (edges[:-1] + edges[1:]) / 2.0
    valid_mask = (n["values"] > 0) & (d["values"] > 0) & (np.abs(centers) <= ratio_eta_limit)

    all_ratios = []
    if np.any(valid_mask):
        ratio = n["values"][valid_mask] / d["values"][valid_mask]
        ratio_err = calc_ratio_errors(ratio, n["errors"][valid_mask], d["errors"][valid_mask], d["values"][valid_mask])
        meta = styles[num]
        ax_ratio.errorbar(
            centers[valid_mask], ratio, yerr=ratio_err,
            fmt=meta["marker"], color=meta["color"], markersize=4, capsize=2, linewidth=1.0,
        )
        reliable = (ratio_err < 0.25) & (np.abs(centers[valid_mask]) <= (ratio_eta_limit - 0.05))
        if np.any(reliable):
            all_ratios.extend(ratio[reliable])

    y_max = max(np.max(c["values"]) for c in curves.values())

    ax.set_ylim(bottom=0, top=y_max * 1.55)
    ax.set_xlim(-eta_limit, eta_limit)
    ax.set_ylabel(ylabel if ylabel is not None else (r"$\frac{1}{N_{\mathrm{evt}}}\frac{dN}{d\eta}$" if per_event else "Counts"))

    if not per_event and y_max >= 1000:
        formatter_y = ScalarFormatter(useMathText=True)
        formatter_y.set_powerlimits((3, 3))
        ax.yaxis.set_major_formatter(formatter_y)

    add_header(ax, run_number, r_jet, lumi)
    ax.text(0.95, 0.95, info_text(method_label), transform=ax.transAxes, ha="right", va="top", fontsize=13)
    ax.legend(loc="upper left", frameon=False, fontsize=13)

    setup_ratio_panel(ax_ratio, xlabel=r"$\eta$", ylabel=ratio_label, ratio_ylim=ratio_ylim, all_ratios=all_ratios, min_margin=0.05)
    plt.subplots_adjust(left=0.14, right=0.96, bottom=0.10, top=0.94)

    save_fig(fig, output_path, save_pdf)
    plt.close(fig)


def run_task(task):
    """Makes one comparison for one run, jet radius and variable. Returns 1 on success, 0 if skipped."""
    kind, run, file_path, lumi, r_tag, r_jet, comparison, output_dir, ratio_ylim, pt_max, save_pdf = task
    name, den, num, ratio_label, per_event = comparison

    hists = load_hists(file_path, r_tag, kind)
    if den not in hists or num not in hists:
        print(f"Warning: Missing {den} or {num} for run {run} {r_tag} ({kind}), skipping")
        return 0

    hist_tag = "hJetPtv3" if kind == "pt" else "h2JetEtav3"
    out_png = output_dir / kind / name / r_tag / f"run_{run}_{hist_tag}_{r_tag}_{name}.png"
    if kind == "pt":
        plot_pair(run, r_jet, lumi, hists, den, num, ratio_label, per_event, out_png,
                  ratio_ylim=ratio_ylim, pt_max=pt_max, save_pdf=save_pdf)
    else:
        plot_eta_pair(run, r_jet, lumi, hists, den, num, ratio_label, per_event, out_png,
                      ratio_ylim=ratio_ylim, save_pdf=save_pdf)
    return 1


def run_r_compare_task(task):
    """Overlays R = 0.2 and R = 0.3 for one method, run and variable. Returns 1 on success, 0 if skipped.

    Raw counts by default (same events for both R). With shape=True each R is normalized to unit area
    (over |eta| < 0.8 for eta, the acceptance common to both R) to compare shapes only.
    """
    kind, run, file_path, lumi, method, shape, output_dir, ratio_ylim, pt_max, save_pdf = task

    hists = {r_tag: load_hists(file_path, r_tag, kind).get(method) for r_tag in R_STYLES}
    if any(h is None for h in hists.values()):
        print(f"Warning: Missing {method} for run {run} ({kind}), skipping")
        return 0

    ylabel = None
    if shape:
        # The plot functions divide by n_events (and the bin width for eta), so use the jet count in range as n_events
        for h in hists.values():
            centers = (h["edges"][:-1] + h["edges"][1:]) / 2.0
            in_range = np.ones_like(centers, dtype=bool) if kind == "pt" else (np.abs(centers) <= 0.8)
            h["n_events"] = np.sum(h["values"][in_range])
        ylabel = (r"$\frac{1}{N_{\mathrm{jets}}}\frac{dN}{dp_{T}}$ [GeV$^{-1}$]" if kind == "pt"
                  else r"$\frac{1}{N_{\mathrm{jets}}}\frac{dN}{d\eta}$")

    name = R_COMPARE_NAME + ("_shape" if shape else "")
    hist_tag = "hJetPtv3" if kind == "pt" else "h2JetEtav3"
    out_png = output_dir / kind / name / method / f"run_{run}_{hist_tag}_{method}_{name}.png"
    method_label = METHODS[method]["label"]
    if kind == "pt":
        plot_pair(run, None, lumi, hists, "r02", "r03", R_COMPARE_RATIO_LABEL, shape, out_png,
                  ratio_ylim=ratio_ylim, pt_max=pt_max, save_pdf=save_pdf, styles=R_STYLES, method_label=method_label, ylabel=ylabel)
    else:
        plot_eta_pair(run, None, lumi, hists, "r02", "r03", R_COMPARE_RATIO_LABEL, shape, out_png,
                      ratio_ylim=ratio_ylim, save_pdf=save_pdf, styles=R_STYLES, method_label=method_label,
                      eta_limit=0.9, ratio_eta_limit=0.8, ylabel=ylabel)
    return 1


def main():
    parser = argparse.ArgumentParser(
        description="Compare jet pT (hJetPtv3) and eta (h2JetEtav3) between flow and no flow, iterative and multiplicity background subtraction, and R = 0.2 and R = 0.3, run by run."
    )
    parser.add_argument(
        "-f", "--file",
        type=Path,
        default=Path("/sphenix/user/anarde/sEPD-Study/jetAnav3/10-15-26/test.list"),
        help="List of Jet-Anav3 merged ROOT files named <run>.root (default: /sphenix/user/anarde/sEPD-Study/jetAnav3/10-15-26/test.list)"
    )
    parser.add_argument(
        "-l", "--lumi-list",
        type=Path,
        default=Path("/sphenix/user/anarde/sEPD-Study/f4a_bkgsub/10-15-26/test.list"),
        help="List of Event QA ROOT files named <run>.root with hLuminosity for the luminosity label (default: /sphenix/user/anarde/sEPD-Study/f4a_bkgsub/10-15-26/test.list)"
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=Path,
        default=Path("/sphenix/user/anarde/sEPD-Study/plots/Jet-Ana/noflow_compare"),
        help="Directory to save generated comparison plots (default: /sphenix/user/anarde/sEPD-Study/plots/Jet-Ana/noflow_compare)"
    )
    parser.add_argument(
        "--runs",
        nargs="*",
        type=int,
        help="Specific run number(s) to process (default: all runs in the list)"
    )
    parser.add_argument(
        "--ratio-ylim",
        nargs=2,
        type=float,
        default=None,
        metavar=("YMIN", "YMAX"),
        help="Custom y-axis limits for the ratio subplot (default: auto-zoom around 1.0)"
    )
    parser.add_argument(
        "--pt-max",
        type=float,
        default=None,
        help="Maximum x-axis limit for Jet pT plots (default: auto; automatically generates an additional version with x-max=60 if auto-range > 60)"
    )
    parser.add_argument(
        "--no-pt",
        action="store_true",
        help="Skip the jet pT comparisons"
    )
    parser.add_argument(
        "--no-eta",
        action="store_true",
        help="Skip the jet eta comparisons"
    )
    parser.add_argument(
        "--no-r-compare",
        action="store_true",
        help="Skip the R = 0.2 vs R = 0.3 comparisons"
    )
    parser.add_argument(
        "-j", "--jobs",
        type=int,
        default=min(8, os.cpu_count() or 1),
        help="Number of parallel worker processes (default: min(8, number of CPUs); 1 runs serially)"
    )
    parser.add_argument(
        "--save-pdf",
        action="store_true",
        help="Save PDF version of plots in addition to PNG"
    )
    args = parser.parse_args()

    run_files = parse_list_file(args.file)
    if not run_files:
        print(f"Error: No ROOT files found in {args.file}")
        sys.exit(1)

    runs_to_process = sorted(args.runs) if args.runs else sorted(run_files)

    lumi_files = parse_list_file(args.lumi_list) if args.lumi_list.exists() else {}
    if not lumi_files:
        print(f"Warning: No luminosity files found in {args.lumi_list}, omitting luminosity labels")
    lumis = {}
    for run in runs_to_process:
        lumi_file = lumi_files.get(run)
        if lumi_file is None or not lumi_file.exists():
            if lumi_files:
                print(f"Warning: No luminosity file for run {run}")
            lumis[run] = None
        else:
            lumis[run] = load_luminosity(lumi_file)

    kinds = [kind for kind, skip in (("pt", args.no_pt), ("eta", args.no_eta)) if not skip]
    if not kinds:
        print("Error: Both pT and eta plots are disabled.")
        sys.exit(1)

    tasks = []
    for run in runs_to_process:
        file_path = run_files.get(run)
        if file_path is None or not file_path.exists():
            print(f"Warning: No file for run {run}, skipping")
            continue
        for r_tag, r_jet in [("r02", 0.2), ("r03", 0.3)]:
            for kind in kinds:
                for comparison in COMPARISONS:
                    tasks.append((kind, run, file_path, lumis[run], r_tag, r_jet, comparison, args.output_dir, args.ratio_ylim, args.pt_max, args.save_pdf))

    r_tasks = []
    if not args.no_r_compare:
        for run in runs_to_process:
            file_path = run_files.get(run)
            if file_path is None or not file_path.exists():
                continue
            for kind in kinds:
                for method in METHODS:
                    for shape in (False, True):
                        r_tasks.append((kind, run, file_path, lumis[run], method, shape, args.output_dir, args.ratio_ylim, args.pt_max, args.save_pdf))

    jobs = [(run_task, t) for t in tasks] + [(run_r_compare_task, t) for t in r_tasks]
    n_jobs = max(1, min(args.jobs, len(jobs)))
    print(f"Generating {len(tasks)} comparisons and {len(r_tasks)} R comparisons with {n_jobs} worker(s)")

    success_count = 0
    if n_jobs == 1:
        success_count = sum(func(t) for func, t in jobs)
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=n_jobs) as executor:
            futures = [executor.submit(func, t) for func, t in jobs]
            success_count = sum(f.result() for f in futures)

    print(f"Successfully generated {success_count}/{len(jobs)} comparisons in {args.output_dir}")


if __name__ == "__main__":
    main()
