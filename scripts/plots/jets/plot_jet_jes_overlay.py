#!/usr/bin/env python3
"""
Jet energy scale (JES) and resolution (JER) in data+MC overlay (Jet-JES outputs), differential in the 4 eta bins
of the legacy eta-dependent JES, to compare mid-rapidity with forward jets.

Each pT-hat sample (Jet12/20/30/40) is filled unweighted within its leading truth R=0.4 jet pT window by Jet-JES.
Here they are weighted by 1 / IntegratedLumi_job (h_gen_integral of the Fun4All overlay hist files) and summed.

The response pT_reco / pT_truth of matched jets is fitted with an iterative Gaussian in truth-pT bins:
JES = Gaussian mean, JER = sigma / mean. Both the raw pT and the stored pt_calib (eta-binned legacy JES) are shown.
Below --pt-trust-min the response is truncated by the tree cut (reco pt_calib >= 10 GeV).
"""

import argparse
import concurrent.futures
import csv
import os
from pathlib import Path
import re
import sys
import warnings
import numpy as np
import uproot
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mplhep as hep
from scipy.optimize import curve_fit, OptimizeWarning

from plot_jet_bkgsub_compare import save_fig, calc_ratio_errors, setup_ratio_panel, COMMON_TEXT_INFO
from plot_jet_noflow_compare import METHODS, add_header
from plot_jet_jes_z0_compare import legacy_eta_edges, eta_regions, ETA_BIN_COLORS, REGION_COLORS

SAMPLE_COLORS = {"Jet12": "#1b9e77", "Jet20": "#d95f02", "Jet30": "#7570b3", "Jet40": "#e7298a"}

RADII = {"r02": 0.2, "r03": 0.3}
CALIBS = {
    "raw": {"label": r"Raw $p_{T}$ (no JES)", "marker": "o"},
    "calib": {"label": r"Calibrated $p_{T}$ ($\eta$-binned JES)", "marker": "s"},
}

# mid / forward: Jet-JES eta bins (legacy JES bins, truth eta) and the matching eta_regions label key
REGIONS = {"mid": {"eta_bins": (1, 2), "key": "inside"}, "fwd": {"eta_bins": (0, 3), "key": "outside"}}
N_ETA_BINS = 4
CENT_CLASSES = [(0, 10), (10, 20), (20, 30), (30, 40), (40, 50), (50, 60)]
CENT_ALL = "0_60"

# Truth-pT bins of the response fits, on the 2 GeV grid of the Jet-JES histograms
PT_BINS = np.array([10, 12, 14, 16, 18, 20, 24, 28, 32, 36, 40, 46, 52, 60])
FIT_QA_BINS = [(20, 24), (32, 36), (46, 52)]
MIN_N_EFF = 50  # minimum effective entries for a fit

OVERLAY_TEXT = "Data + PYTHIA8 Overlay (Jet12–40)"
JES_YLABEL = r"$\langle p_{T}^{\mathrm{reco}} / p_{T}^{\mathrm{truth}} \rangle$ (Gaussian $\mu$)"
JER_YLABEL = r"$\sigma / \mu$ of $p_{T}^{\mathrm{reco}} / p_{T}^{\mathrm{truth}}$"
TRUTH_PT_LABEL = r"$p_{T}^{\mathrm{truth}}$ [GeV]"


def parse_sample_list(list_file):
    """Returns {sample: path} from a list of ROOT files, with the sample (JetNN) taken from the path."""
    samples = {}
    with Path(list_file).open("r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            match = re.search(r"(Jet\d+)", line)
            if not match:
                print(f"Warning: No sample (JetNN) in {line}, skipping")
                continue
            samples[match.group(1)] = Path(line)
    return dict(sorted(samples.items(), key=lambda kv: int(kv[0][3:])))


def load_weights(gen_files, samples):
    """Per-sample weight 1 / IntegratedLumi_job [pb] from h_gen_integral, with a bookkeeping table."""
    weights = {}
    print(f"{'Sample':<8}{'L_job [pb^-1]':>16}{'NEvents_job':>14}{'NEvt/NProc':>12}{'NFiles':>8}{'weight [pb]':>14}")
    for sample in samples:
        if sample not in gen_files:
            print(f"Error: {sample} not in the generator bookkeeping list")
            sys.exit(1)
        with uproot.open(gen_files[sample]) as rf:
            h = rf["h_gen_integral"]
            info = dict(zip(h.axis().labels(), h.values()))
        lumi = info["IntegratedLumi_job [pb^{-1}]"]
        if lumi <= 0:
            print(f"Error: {sample} has no integrated luminosity in {gen_files[sample]}")
            sys.exit(1)
        weights[sample] = 1.0 / lumi
        print(f"{sample:<8}{lumi:>16.4f}{info['NEvents_job']:>14.0f}{info['NEvents_job'] / info['NProcessed_file']:>12.4f}"
              f"{info['NFiles']:>8.0f}{weights[sample]:>14.4e}")
    return weights


def load_sample(path):
    """All TH1 / TH2 of a Jet-JES output: {name: {"values", "variances", "edges"}}, plus the hEvent labels."""
    hists = {}
    with uproot.open(path) as rf:
        for name, cls in rf.classnames().items():
            if not cls.startswith(("TH1", "TH2")):
                continue
            h = rf[name]
            vals, *edges = h.to_numpy()
            variances = h.variances()
            hists[name.split(";")[0]] = {"values": vals, "variances": variances if variances is not None else vals.copy(), "edges": edges}
        hists["hEvent"]["labels"] = rf["hEvent"].axis().labels()
    return hists


def scaled(h, w):
    return {"values": h["values"] * w, "variances": h["variances"] * w ** 2, "edges": h["edges"]}


def add_hists(hs):
    hs = list(hs)
    return {"values": sum(h["values"] for h in hs), "variances": sum(h["variances"] for h in hs), "edges": hs[0]["edges"]}


def stitch(samples, weights, name):
    """Sum over the samples of w_s * h_s (variance w_s^2 var_s)."""
    return add_hists(scaled(hists[name], weights[s]) for s, hists in samples.items())


def resp_name(calib, r_tag, ue, ieta, cent):
    return f"h2Resp_{calib}_{r_tag}_{ue}_eta{ieta}_cent_{cent}"


def cent_tags(cent):
    """Centrality class tags of the Jet-JES histograms in cent (a class tag or CENT_ALL)."""
    return [f"{lo}_{hi}" for lo, hi in CENT_CLASSES] if cent == CENT_ALL else [cent]


def sum_bins(hists, name_fn, eta_bins, cent):
    return add_hists(hists[name_fn(ieta, c)] for ieta in eta_bins for c in cent_tags(cent))


def pt_slice(edges, lo, hi):
    ilo, ihi = np.searchsorted(edges, [lo, hi])
    if not (np.isclose(edges[ilo], lo) and np.isclose(edges[ihi], hi)):
        raise ValueError(f"pT bin [{lo}, {hi}) is not on the histogram grid")
    return slice(ilo, ihi)


def rebin_pt(h, pt_bins):
    """TH1 of truth pT summed into pt_bins."""
    sl = [pt_slice(h["edges"][0], lo, hi) for lo, hi in zip(pt_bins[:-1], pt_bins[1:])]
    return np.array([h["values"][s].sum() for s in sl]), np.array([h["variances"][s].sum() for s in sl])


def response_in_pt(h2, lo, hi):
    """Response distribution of the TH2 (truth pT, response) for truth pT in [lo, hi)."""
    sl = pt_slice(h2["edges"][0], lo, hi)
    return h2["values"][sl].sum(axis=0), h2["variances"][sl].sum(axis=0), h2["edges"][1]


def gaussian(x, amp, mu, sigma):
    return amp * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def fit_response(vals, variances, edges, n_sigma=1.5, n_iter=3):
    """Iterative Gaussian fit in [mu - n_sigma * sigma, mu + n_sigma * sigma], seeded by the mean and RMS."""
    res = {k: np.nan for k in ("mu", "mu_err", "sigma", "sigma_err", "amp", "mean", "mean_err", "lo", "hi")}
    res["n_eff"] = 0.0
    total, total_var = vals.sum(), variances.sum()
    if total <= 0 or total_var <= 0:
        return res
    centers = 0.5 * (edges[1:] + edges[:-1])
    res["n_eff"] = total ** 2 / total_var
    mean = np.average(centers, weights=vals)
    rms = np.sqrt(np.average((centers - mean) ** 2, weights=vals))
    res["mean"], res["mean_err"] = mean, rms / np.sqrt(res["n_eff"])
    if res["n_eff"] < MIN_N_EFF:
        return res

    mu, sigma, popt, pcov = mean, rms, None, None
    for _ in range(n_iter):
        lo, hi = mu - n_sigma * sigma, mu + n_sigma * sigma
        mask = (centers >= lo) & (centers <= hi) & (variances > 0)
        if mask.sum() < 4:
            break
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", OptimizeWarning)
                popt, pcov = curve_fit(gaussian, centers[mask], vals[mask], p0=[vals[mask].max(), mu, sigma],
                                       sigma=np.sqrt(variances[mask]), absolute_sigma=True, maxfev=5000)
        except (RuntimeError, ValueError, OptimizeWarning):
            popt = None
            break
        mu, sigma = popt[1], abs(popt[2])
        # runaway fit (e.g. a response truncated by the reco pT cut): no Gaussian core
        if not (edges[0] < mu < edges[-1] and 0 < sigma < 2 * rms):
            popt = None
            break

    if popt is not None:
        err = np.sqrt(np.diag(pcov))
        if np.all(np.isfinite(err)):
            res.update(mu=mu, mu_err=err[1], sigma=sigma, sigma_err=err[2], amp=popt[0], lo=lo, hi=hi)
    return res


def jer(res):
    """sigma / mu and its error (mu and sigma errors treated as independent)."""
    r = res["sigma"] / res["mu"]
    return r, r * np.sqrt((res["sigma_err"] / res["sigma"]) ** 2 + (res["mu_err"] / res["mu"]) ** 2)


def fit_pt_bins(h2, n_sigma):
    return [fit_response(*response_in_pt(h2, lo, hi), n_sigma=n_sigma) for lo, hi in zip(PT_BINS[:-1], PT_BINS[1:])]


def field(results, key):
    return np.array([r[key] for r in results])


def pt_centers():
    return 0.5 * (PT_BINS[1:] + PT_BINS[:-1]), 0.5 * np.diff(PT_BINS)


def set_jes_ylim(ax, ys, with_one, below=0.3, above=1.2):
    """y range of the finite ys with margins (fractions of their span) for the legend and text, including 1 if with_one."""
    ys = np.concatenate([np.asarray(y, dtype=float) for y in ys])
    ys = ys[np.isfinite(ys)]
    if len(ys) == 0:
        return
    lo, hi = (min(ys.min(), 1.0), max(ys.max(), 1.0)) if with_one else (ys.min(), ys.max())
    span = max(hi - lo, 0.02)
    ax.set_ylim(lo - below * span, hi + above * span)


def shade_low_pt(ax, pt_trust_min, label=True):
    """Truth pT below which the response is truncated by the reco pt_calib >= 10 GeV tree cut."""
    ax.axvspan(PT_BINS[0], pt_trust_min, color="gray", alpha=0.15, label=r"Reco $p_{T}^{\mathrm{calib}} \geq 10$ GeV cut" if label else None)


def legend_after_band(ax, pt_trust_min, handles=None, **kwargs):
    """Legend at the top, starting right of the shaded low pT band (where the truncated points can sit high)."""
    x0 = (pt_trust_min - PT_BINS[0]) / (PT_BINS[-1] - PT_BINS[0]) + 0.01
    if handles is None:
        handles, _ = ax.get_legend_handles_labels()
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(x0, 0.99), frameon=False, **kwargs)


def info_box(ax, ue, calib=None, extra=None, x=0.95, y=0.95, ha="right", fontsize=13):
    lines = [METHODS[ue]["label"]] + ([CALIBS[calib]["label"]] if calib else []) + [OVERLAY_TEXT, COMMON_TEXT_INFO] + ([extra] if extra else [])
    ax.text(x, y, "\n".join(lines), transform=ax.transAxes, ha=ha, va="top", fontsize=fontsize)


def signed_eta_labels(r_jet):
    eta_max = round(1.1 - r_jet, 1)
    edges = [-eta_max, *legacy_eta_edges(r_jet), eta_max]
    return [rf"$[{lo:g}, {hi:g})$" for lo, hi in zip(edges[:-1], edges[1:])]


def draw_points(ax, results, key, err_key, color, marker, label, offset=0.0):
    x, xerr = pt_centers()
    y, yerr = field(results, key), field(results, err_key)
    ok = np.isfinite(y)
    ax.errorbar(x[ok] + offset, y[ok], yerr=yerr[ok], xerr=xerr[ok], fmt=marker, color=color, markersize=5, capsize=2, linewidth=1.0,
                label=label)


def draw_ratio(ax_ratio, num, den, key, err_key, color, marker):
    """num / den of a per pT bin quantity, returns the well-measured ratios for the panel limits."""
    x, xerr = pt_centers()
    n, ne, d, de = field(num, key), field(num, err_key), field(den, key), field(den, err_key)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = n / d
        err = calc_ratio_errors(ratio, ne, de, d)
    ok = np.isfinite(ratio)
    ax_ratio.errorbar(x[ok], ratio[ok], yerr=err[ok], xerr=xerr[ok], fmt=marker, color=color, markersize=4, capsize=2, linewidth=1.0)
    return list(ratio[ok & (err < 0.05)])


# --------------------------------------------------------------------------------------------------
# Stitching (all samples)
# --------------------------------------------------------------------------------------------------

def plot_lead_truth_xsec(samples, weights, run, out_path, save_pdf):
    """Luminosity-normalized leading truth R=0.4 jet pT of each sample (all events), with each sample / the next lower one."""
    hep.style.use("ATLAS")
    fig, (ax, ax_ratio) = plt.subplots(2, 1, figsize=(8, 8), gridspec_kw={"height_ratios": [3, 1], "hspace": 0.08}, sharex=True)
    xsec = {}
    for sample, hists in samples.items():
        h = hists["hMaxPtTruthR04_all"]
        widths = np.diff(h["edges"][0])
        xsec[sample] = {"values": h["values"] * weights[sample] / widths, "errors": np.sqrt(h["variances"]) * weights[sample] / widths, "edges": h["edges"][0]}
        hep.histplot((xsec[sample]["values"], xsec[sample]["edges"]), ax=ax, histtype="step", color=SAMPLE_COLORS.get(sample), linewidth=2, label=sample)

    names = list(xsec)
    for lower, higher in zip(names[:-1], names[1:]):
        lo, hi = xsec[lower], xsec[higher]
        centers = 0.5 * (lo["edges"][1:] + lo["edges"][:-1])
        ok = (lo["values"] > 0) & (hi["values"] > 0)
        ratio = hi["values"][ok] / lo["values"][ok]
        err = calc_ratio_errors(ratio, hi["errors"][ok], lo["errors"][ok], lo["values"][ok])
        ax_ratio.errorbar(centers[ok], ratio, yerr=err, fmt="o", color=SAMPLE_COLORS.get(higher), markersize=3, capsize=1, linewidth=1.0)

    ax.set_yscale("log")
    ax.set_xlim(0, 80)
    ax.set_ylabel(r"$d\sigma / dp_{T}$ [pb/GeV]")
    add_header(ax, run, None, None)
    ax.set_ylim(top=ax.get_ylim()[1] * 1e3)  # room above the spectra for the legend and text
    ax.legend(loc="upper right", frameon=False, fontsize=14)
    ax.text(0.04, 0.96, f"{OVERLAY_TEXT}\nAll events (before selection)\nLeading truth $R = 0.4$ jet", transform=ax.transAxes, ha="left", va="top", fontsize=14)
    setup_ratio_panel(ax_ratio, xlabel=r"Leading truth $R = 0.4$ jet $p_{T}$ [GeV]", ylabel="Sample / Lower", ratio_ylim=(0, 1.5))
    plt.subplots_adjust(left=0.14, right=0.96, bottom=0.10, top=0.94)
    save_fig(fig, out_path, save_pdf)
    plt.close(fig)


def plot_stitched(samples, weights, name, xlabel, ylabel_tag, run, out_path, save_pdf, text, r_jet=None, xlim=(0, 80), vlines=(), headroom=1e3):
    """Weighted contribution of each sample and the stitched sum of one TH1."""
    hep.style.use("ATLAS")
    fig, ax = plt.subplots(figsize=(8, 7))
    for sample, hists in samples.items():
        h = scaled(hists[name], weights[sample])
        widths = np.diff(h["edges"][0])
        hep.histplot((h["values"] / widths, h["edges"][0]), ax=ax, histtype="step", color=SAMPLE_COLORS.get(sample), linewidth=1.5, label=sample)
    total = stitch(samples, weights, name)
    widths = np.diff(total["edges"][0])
    hep.histplot((total["values"] / widths, total["edges"][0]), yerr=np.sqrt(total["variances"]) / widths, ax=ax, histtype="errorbar",
                 color="black", markersize=3, label="Stitched")
    for x in vlines:
        ax.axvline(x, ymax=0.75, color="gray", linestyle="--", linewidth=1)
    ax.set_yscale("log")
    ax.set_ylim(top=ax.get_ylim()[1] * headroom)  # room above the spectra for the legend and text
    ax.set_xlim(*xlim)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(rf"$d\sigma / dp_{{T}}$ [pb/GeV] ({ylabel_tag})")
    add_header(ax, run, r_jet, None)
    ax.legend(loc="upper right", frameon=False, fontsize=14)
    ax.text(0.04, 0.96, f"{OVERLAY_TEXT}\n{text}", transform=ax.transAxes, ha="left", va="top", fontsize=14)
    plt.subplots_adjust(left=0.15, right=0.96, bottom=0.12, top=0.94)
    save_fig(fig, out_path, save_pdf)
    plt.close(fig)


def sample_window(hists):
    """Leading truth R=0.4 jet pT window [lo, hi) the sample was filled with, from the last hEvent label (hi = inf if open)."""
    m = re.search(r"\[([\d.]+), ([\d.]+|#infty)\)", hists["hEvent"]["labels"][-1])
    if not m:
        raise ValueError(f"No window in hEvent label: {hists['hEvent']['labels'][-1]}")
    return float(m.group(1)), np.inf if m.group(2) == "#infty" else float(m.group(2))


def print_event_summary(samples):
    print(f"\n{'Sample':<8}" + "".join(f"{label[:26]:>28}" for label in samples[next(iter(samples))]["hEvent"]["labels"]) + f"{'Calo-MBD rejected':>20}")
    for sample, hists in samples.items():
        vals = hists["hEvent"]["values"]
        print(f"{sample:<8}" + "".join(f"{v:>28.0f}" for v in vals) + f"{(vals[1] - vals[2]) / vals[1]:>19.2%}")
    print()


# --------------------------------------------------------------------------------------------------
# JES per jet radius and subtraction method
# --------------------------------------------------------------------------------------------------

def plot_matching_qa(hists, r_tag, ue, run, out_path, save_pdf, pt_min=20):
    """Closest reco jet Delta R (before the cut) and Delta eta of the matched jets, for truth pT >= pt_min."""
    hep.style.use("ATLAS")
    r_jet = RADII[r_tag]
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, name, xlabel, cut in ((axes[0], f"h2DeltaR_{r_tag}_{ue}", r"$\Delta R$(truth, closest reco)", 0.75 * r_jet),
                                  (axes[1], f"h2DeltaEta_{r_tag}_{ue}", r"$\eta^{\mathrm{reco}} - \eta^{\mathrm{truth}}$ (matched)", None)):
        h = hists[name]
        vals, variances, edges = response_in_pt(h, pt_min, h["edges"][0][-1])
        norm = vals.sum()
        hep.histplot((vals / norm, edges), yerr=np.sqrt(variances) / norm, ax=ax, histtype="step", color=METHODS[ue]["color"], linewidth=2)
        if cut is not None:
            ax.axvline(cut, color="gray", linestyle="--", linewidth=1.5, label=rf"$0.75 R = {cut:g}$")
            ax.legend(loc="center right", frameon=False, fontsize=13)
        ax.set_yscale("log")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Normalized")
        ax.text(0.95, 0.95, f"{METHODS[ue]['label']}\n" + rf"$p_{{T}}^{{\mathrm{{truth}}}} \geq {pt_min}$ GeV", transform=ax.transAxes, ha="right", va="top", fontsize=13)
        add_header(ax, run, r_jet, None)
    plt.tight_layout()
    save_fig(fig, out_path, save_pdf)
    plt.close(fig)


def plot_match_eff(hists, r_tag, ue, run, pt_trust_min, out_path, save_pdf):
    """Fraction of fiducial truth jets with a matched reco jet, mid vs forward (0-60%)."""
    hep.style.use("ATLAS")
    r_jet = RADII[r_tag]
    labels, _ = eta_regions(r_jet)
    x, xerr = pt_centers()
    fig, ax = plt.subplots(figsize=(8, 6.5))
    for region, meta in REGIONS.items():
        truth = sum_bins(hists, lambda ieta, c: f"hTruth_{r_tag}_eta{ieta}_cent_{c}", meta["eta_bins"], CENT_ALL)
        matched = sum_bins(hists, lambda ieta, c: f"hTruthMatched_{r_tag}_{ue}_eta{ieta}_cent_{c}", meta["eta_bins"], CENT_ALL)
        t, t_var = rebin_pt(truth, PT_BINS)
        m, _ = rebin_pt(matched, PT_BINS)
        with np.errstate(divide="ignore", invalid="ignore"):
            eff = m / t
            n_eff = t ** 2 / t_var
            err = np.sqrt(np.clip(eff * (1 - eff), 0, None) / n_eff)
        ok = np.isfinite(eff)
        ax.errorbar(x[ok], eff[ok], yerr=err[ok], xerr=xerr[ok], fmt="o", color=REGION_COLORS[meta["key"]], markersize=5, capsize=2, label=labels[meta["key"]])
    shade_low_pt(ax, pt_trust_min)
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1)
    ax.set_xlim(PT_BINS[0], PT_BINS[-1])
    ax.set_ylim(0.6, 1.1)
    ax.set_xlabel(TRUTH_PT_LABEL)
    ax.set_ylabel("Matching efficiency")
    add_header(ax, run, r_jet, None)
    info_box(ax, ue, extra=rf"$\Delta R < 0.75 R$", y=0.66, fontsize=12)
    ax.legend(loc="lower right", frameon=False, fontsize=13)
    plt.subplots_adjust(left=0.14, right=0.96, bottom=0.13, top=0.93)
    save_fig(fig, out_path, save_pdf)
    plt.close(fig)


def plot_fit_qa(hists, r_tag, ue, calib, run, n_sigma, out_path, save_pdf):
    """Stitched response distributions (mid vs forward, 0-60%) with the Gaussian fits, in a few truth-pT bins."""
    hep.style.use("ATLAS")
    r_jet = RADII[r_tag]
    labels, _ = eta_regions(r_jet)
    fig, axes = plt.subplots(1, len(FIT_QA_BINS), figsize=(6 * len(FIT_QA_BINS), 6.5), sharey=True)
    y_max = 0.0
    for ax, (lo, hi) in zip(axes, FIT_QA_BINS):
        fit_lines = []
        for region, meta in REGIONS.items():
            h2 = sum_bins(hists, lambda ieta, c: resp_name(calib, r_tag, ue, ieta, c), meta["eta_bins"], CENT_ALL)
            vals, variances, edges = response_in_pt(h2, lo, hi)
            norm = vals.sum() / 100.0  # percent of the jets per bin
            if norm <= 0:
                continue
            color = REGION_COLORS[meta["key"]]
            hep.histplot((vals / norm, edges), ax=ax, histtype="step", color=color, linewidth=2, label=labels[meta["key"]])
            y_max = max(y_max, (vals / norm).max())
            res = fit_response(vals, variances, edges, n_sigma=n_sigma)
            if np.isfinite(res["mu"]):
                xs = np.linspace(res["lo"], res["hi"], 200)
                ax.plot(xs, gaussian(xs, res["amp"], res["mu"], res["sigma"]) / norm, color=color, linestyle="--", linewidth=2.5,
                        label="Gaussian fit" if region == "fwd" else None)
                fit_lines.append((rf"$\mu = {res['mu']:.3f}$" "\n" rf"$\sigma/\mu = {res['sigma'] / res['mu']:.3f}$", color))
        ax.set_xlim(0, 2.0)
        ax.set_xlabel(rf"$p_{{T}}^{{\mathrm{{reco}}}} / p_{{T}}^{{\mathrm{{truth}}}}$ ({calib})", fontsize=20)
        ax.text(0.04, 0.96, rf"${lo} \leq p_{{T}}^{{\mathrm{{truth}}}} < {hi}$ GeV", transform=ax.transAxes, ha="left", va="top", fontsize=17)
        for i, (text, color) in enumerate(fit_lines):
            ax.text(0.96, 0.96 - 0.17 * i, text, transform=ax.transAxes, ha="right", va="top", fontsize=16, color=color)
    axes[0].set_ylim(0, y_max * 1.3)
    axes[0].set_ylabel("Jets [%]", fontsize=20)
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper left", bbox_to_anchor=(0.06, 1.0), ncol=3, frameon=False, fontsize=17)
    fig.text(0.99, 0.985, rf"Run: {run}, $R = {r_jet:g}$, {METHODS[ue]['label']}, {CALIBS[calib]['label']}, 0–60%",
             ha="right", va="top", fontsize=17)
    plt.tight_layout(rect=(0, 0, 1, 0.92))
    save_fig(fig, out_path, save_pdf)
    plt.close(fig)


def plot_regions(results, r_tag, ue, calib, run, pt_trust_min, kind, out_path, save_pdf):
    """JES (kind="jes") or JER (kind="jer") vs truth pT, mid vs forward (0-60%), with forward / mid."""
    hep.style.use("ATLAS")
    r_jet = RADII[r_tag]
    labels, _ = eta_regions(r_jet)
    fig, (ax, ax_ratio) = plt.subplots(2, 1, figsize=(8, 8), gridspec_kw={"height_ratios": [3, 1], "hspace": 0.08}, sharex=True)

    series = {}
    for region, meta in REGIONS.items():
        res = results[(calib, region, CENT_ALL)]
        color = REGION_COLORS[meta["key"]]
        if kind == "jes":
            series[region] = [{"y": r["mu"], "yerr": r["mu_err"]} for r in res]
            draw_points(ax, series[region], "y", "yerr", color, "o", labels[meta["key"]])
        else:
            series[region] = [dict(zip(("y", "yerr"), jer(r))) for r in res]
            draw_points(ax, series[region], "y", "yerr", color, "o", labels[meta["key"]])

    ratios = draw_ratio(ax_ratio, series["fwd"], series["mid"], "y", "yerr", REGION_COLORS["outside"], "o")
    shade_low_pt(ax, pt_trust_min)
    shade_low_pt(ax_ratio, pt_trust_min, label=False)
    if kind == "jes":
        ax.axhline(1.0, color="gray", linestyle="--", linewidth=1)

    trusted = PT_BINS[:-1] >= pt_trust_min
    set_jes_ylim(ax, [field(s, "y")[trusted] for s in series.values()], with_one=(kind == "jes" and calib == "calib"))
    ax.set_xlim(PT_BINS[0], PT_BINS[-1])
    ax.set_ylabel(JES_YLABEL if kind == "jes" else JER_YLABEL)
    add_header(ax, run, r_jet, None)
    info_box(ax, ue, calib, fontsize=12)
    handles, _ = ax.get_legend_handles_labels()
    legend_after_band(ax, pt_trust_min, handles=handles, fontsize=12)
    setup_ratio_panel(ax_ratio, xlabel=TRUTH_PT_LABEL, ylabel="Fwd / Mid", all_ratios=ratios, min_margin=0.03)
    plt.subplots_adjust(left=0.15, right=0.96, bottom=0.10, top=0.94)
    save_fig(fig, out_path, save_pdf)
    plt.close(fig)


def plot_eta_bins(results, r_tag, ue, calib, run, pt_trust_min, out_path, save_pdf):
    """JES vs truth pT in the 4 signed legacy eta bins (0-60%), each over mid (bins 1 + 2)."""
    hep.style.use("ATLAS")
    r_jet = RADII[r_tag]
    labels = signed_eta_labels(r_jet)
    fig, (ax, ax_ratio) = plt.subplots(2, 1, figsize=(8, 8), gridspec_kw={"height_ratios": [3, 1], "hspace": 0.08}, sharex=True)
    mid = [{"y": r["mu"], "yerr": r["mu_err"]} for r in results[(calib, "mid", CENT_ALL)]]
    ratios, y_all = [], []
    for ieta in range(N_ETA_BINS):
        pts = [{"y": r["mu"], "yerr": r["mu_err"]} for r in results[(calib, f"eta{ieta}", CENT_ALL)]]
        offset = (ieta - 1.5) * 0.3
        draw_points(ax, pts, "y", "yerr", ETA_BIN_COLORS[ieta], "os^D"[ieta], labels[ieta], offset=offset)
        ratios += draw_ratio(ax_ratio, pts, mid, "y", "yerr", ETA_BIN_COLORS[ieta], "os^D"[ieta])
        y_all.append(field(pts, "y")[PT_BINS[:-1] >= pt_trust_min])
    shade_low_pt(ax, pt_trust_min, label=False)  # labelled in the mid / forward plots, keeps this legend narrow
    shade_low_pt(ax_ratio, pt_trust_min, label=False)
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1)
    set_jes_ylim(ax, y_all, with_one=(calib == "calib"))
    ax.set_xlim(PT_BINS[0], PT_BINS[-1])
    ax.set_ylabel(JES_YLABEL)
    add_header(ax, run, r_jet, None)
    info_box(ax, ue, calib, fontsize=12)
    legend_after_band(ax, pt_trust_min, fontsize=12, title=r"$\eta^{\mathrm{truth}}$ bin", title_fontsize=12)
    setup_ratio_panel(ax_ratio, xlabel=TRUTH_PT_LABEL, ylabel="Bin / Mid", all_ratios=ratios, min_margin=0.03)
    plt.subplots_adjust(left=0.15, right=0.96, bottom=0.10, top=0.94)
    save_fig(fig, out_path, save_pdf)
    plt.close(fig)


def plot_cent_grid(results, r_tag, ue, calib, run, pt_trust_min, out_path, save_pdf):
    """JES vs truth pT, mid vs forward, in each centrality class."""
    hep.style.use("ATLAS")
    r_jet = RADII[r_tag]
    labels, _ = eta_regions(r_jet)
    fig, axes = plt.subplots(2, 3, figsize=(18, 11), sharex=True, sharey=True)
    y_all = []
    for ax, (lo, hi) in zip(axes.flat, CENT_CLASSES):
        for region, meta in REGIONS.items():
            pts = [{"y": r["mu"], "yerr": r["mu_err"]} for r in results[(calib, region, f"{lo}_{hi}")]]
            draw_points(ax, pts, "y", "yerr", REGION_COLORS[meta["key"]], "o", labels[meta["key"]])
            y_all.append(field(pts, "y")[PT_BINS[:-1] >= pt_trust_min])
        shade_low_pt(ax, pt_trust_min, label=False)
        ax.axhline(1.0, color="gray", linestyle="--", linewidth=1)
        ax.text(0.95, 0.95, f"{lo}–{hi}%", transform=ax.transAxes, ha="right", va="top", fontsize=16)
    set_jes_ylim(axes[0, 0], y_all, with_one=(calib == "calib"), below=0.5, above=0.8)
    axes[0, 0].set_xlim(PT_BINS[0], PT_BINS[-1])
    # inner ticks only, so the labels of neighbouring panels don't collide
    axes[0, 0].set_xticks(np.arange(20, PT_BINS[-1], 10))
    for ax in axes[1]:
        ax.set_xlabel(TRUTH_PT_LABEL)
    for ax in axes[:, 0]:
        ax.set_ylabel(JES_YLABEL, fontsize=14)
    axes[0, 0].legend(loc="lower right", frameon=False, fontsize=12)
    axes[0, 0].text(0.0, 1.01, f"{METHODS[ue]['label']}, {CALIBS[calib]['label']}, {OVERLAY_TEXT}", transform=axes[0, 0].transAxes, ha="left", va="bottom", fontsize=14)
    add_header(axes[0, 2], run, r_jet, None)
    plt.subplots_adjust(left=0.07, right=0.98, bottom=0.08, top=0.95, wspace=0.05, hspace=0.05)
    save_fig(fig, out_path, save_pdf)
    plt.close(fig)


def fit_group(hists, r_tag, ue, n_sigma):
    """Fits per pT bin for every calib x (mid, fwd, eta0-3 at 0-60%; mid, fwd per centrality class)."""
    selections = [(region, meta["eta_bins"], CENT_ALL) for region, meta in REGIONS.items()]
    selections += [(f"eta{ieta}", (ieta,), CENT_ALL) for ieta in range(N_ETA_BINS)]
    selections += [(region, meta["eta_bins"], f"{lo}_{hi}") for lo, hi in CENT_CLASSES for region, meta in REGIONS.items()]
    results = {}
    for calib in CALIBS:
        for region, eta_bins, cent in selections:
            h2 = sum_bins(hists, lambda ieta, c: resp_name(calib, r_tag, ue, ieta, c), eta_bins, cent)
            results[(calib, region, cent)] = fit_pt_bins(h2, n_sigma)
    return results


def run_group(task):
    """All JES plots of one jet radius and subtraction method. Returns the number of figures and the table rows."""
    r_tag, ue, hists, run, pt_trust_min, n_sigma, output_dir, save_pdf = task
    results = fit_group(hists, r_tag, ue, n_sigma)
    tag = f"{r_tag}_{ue}"
    n = 0
    plot_matching_qa(hists, r_tag, ue, run, output_dir / "matching" / f"matching_qa_{tag}.png", save_pdf)
    plot_match_eff(hists, r_tag, ue, run, pt_trust_min, output_dir / "matching" / f"match_eff_{tag}.png", save_pdf)
    n += 2
    for calib in CALIBS:
        plot_fit_qa(hists, r_tag, ue, calib, run, n_sigma, output_dir / "fits" / f"fits_{tag}_{calib}.png", save_pdf)
        for kind in ("jes", "jer"):
            plot_regions(results, r_tag, ue, calib, run, pt_trust_min, kind, output_dir / "jes" / f"{kind}_mid_fwd_{tag}_{calib}.png", save_pdf)
        plot_eta_bins(results, r_tag, ue, calib, run, pt_trust_min, output_dir / "jes" / f"jes_eta_bins_{tag}_{calib}.png", save_pdf)
        plot_cent_grid(results, r_tag, ue, calib, run, pt_trust_min, output_dir / "jes_cent" / f"jes_cent_{tag}_{calib}.png", save_pdf)
        n += 5

    rows = []
    for (calib, region, cent), res in results.items():
        for (lo, hi), r in zip(zip(PT_BINS[:-1], PT_BINS[1:]), res):
            rows.append({"R": RADII[r_tag], "ue": ue, "calib": calib, "region": region, "cent": cent, "pt_lo": lo, "pt_hi": hi,
                         **{k: r[k] for k in ("mu", "mu_err", "sigma", "sigma_err", "mean", "mean_err", "n_eff")}})
    return n, rows


def group_hists(stitched, r_tag, ue):
    """The stitched histograms one (R, method) group needs."""
    keep = (f"_{r_tag}_{ue}_", f"hTruth_{r_tag}_")
    names = [k for k in stitched if any(s in k for s in keep) or k in (f"h2DeltaR_{r_tag}_{ue}", f"h2DeltaEta_{r_tag}_{ue}")]
    return {k: stitched[k] for k in names}


def main():
    parser = argparse.ArgumentParser(description="JES / JER in data+MC overlay (Jet-JES outputs), mid-rapidity vs forward eta.")
    parser.add_argument("-i", "--input-list", type=Path, default=Path("/sphenix/user/anarde/sEPD-Study/jetAna_JES/10-21-26-windows-regen/test.list"),
                        help="List of merged Jet-JES ROOT files, one per sample (JetNN in the path)")
    parser.add_argument("-g", "--gen-list", type=Path, default=Path("/sphenix/user/anarde/sEPD-Study/f4a_bkgsub/10-21-26-overlay-combined/test.list"),
                        help="List of the merged Fun4All overlay hist files with h_gen_integral, one per sample (JetNN in the path)")
    parser.add_argument("-o", "--output-dir", type=Path, default=Path("/sphenix/user/anarde/sEPD-Study/plots/Jet-Ana/jes_overlay"),
                        help="Directory to save the plots")
    parser.add_argument("--methods", nargs="+", default=["iter", "iternf"], choices=["iter", "iternf"], help="Subtraction methods (default: iter iternf)")
    parser.add_argument("--pt-trust-min", type=float, default=20.0, help="Truth pT below which the response is truncated by the reco tree cut, shaded (default: 20 GeV)")
    parser.add_argument("--n-sigma", type=float, default=1.5, help="Gaussian fit range in sigma around the mean (default: 1.5)")
    parser.add_argument("--run", type=int, default=68144, help="Data run of the overlay, for the header (default: 68144)")
    parser.add_argument("-j", "--jobs", type=int, default=min(8, os.cpu_count() or 1), help="Number of parallel worker processes")
    parser.add_argument("--save-pdf", action="store_true", help="Save PDF version of plots in addition to PNG")
    args = parser.parse_args()

    jes_files = parse_sample_list(args.input_list)
    gen_files = parse_sample_list(args.gen_list)
    if not jes_files:
        print(f"Error: No samples in {args.input_list}")
        sys.exit(1)

    weights = load_weights(gen_files, jes_files)
    samples = {sample: load_sample(path) for sample, path in jes_files.items()}
    print_event_summary(samples)

    # Stitching checks
    stitch_dir = args.output_dir / "stitching"
    sample_windows = {sample: sample_window(hists) for sample, hists in samples.items()}
    print("Windows: " + ", ".join(f"{s}: [{lo:g}, {hi:g})" for s, (lo, hi) in sample_windows.items()))
    edges = sorted(sample_windows.values())
    for (_, hi), (lo, _) in zip(edges[:-1], edges[1:]):
        if hi != lo:
            print(f"Warning: sample windows do not join ({hi:g} vs {lo:g}): the stitched spectra have a gap or overlap")
    windows = [lo for lo, _ in edges[1:]]
    plot_lead_truth_xsec(samples, weights, args.run, stitch_dir / "lead_truth_r04_xsec.png", args.save_pdf)
    plot_stitched(samples, weights, "hMaxPtTruthR04_window", r"Leading truth $R = 0.4$ jet $p_{T}$ [GeV]", "selected", args.run,
                  stitch_dir / "lead_truth_r04_stitched.png", args.save_pdf, "Event selection + sample windows", vlines=windows)
    for r_tag, r_jet in RADII.items():
        name = f"hTruth_{r_tag}_total"
        for hists in samples.values():
            hists[name] = add_hists(hists[k] for k in hists if k.startswith(f"hTruth_{r_tag}_eta"))
        eta_max = round(1.1 - r_jet, 1)
        plot_stitched(samples, weights, name, TRUTH_PT_LABEL, "selected", args.run, stitch_dir / f"truth_jet_{r_tag}_stitched.png", args.save_pdf,
                      f"Truth jets, $|\\eta| < {eta_max:g}$\n{COMMON_TEXT_INFO}", r_jet=r_jet, xlim=(0, 80), vlines=windows, headroom=1e6)

    stitched = {name: stitch(samples, weights, name) for name in samples[next(iter(samples))] if name != "hEvent"}

    tasks = [(r_tag, ue, group_hists(stitched, r_tag, ue), args.run, args.pt_trust_min, args.n_sigma, args.output_dir, args.save_pdf)
             for r_tag in RADII for ue in args.methods]
    n_jobs = max(1, min(args.jobs, len(tasks)))
    print(f"Generating JES plots for {len(tasks)} (R, method) groups with {n_jobs} worker(s)")
    if n_jobs == 1:
        outputs = [run_group(t) for t in tasks]
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=n_jobs) as executor:
            outputs = list(executor.map(run_group, tasks))

    rows = [row for _, group_rows in outputs for row in group_rows]
    table_path = args.output_dir / "jes_table.csv"
    table_path.parent.mkdir(parents=True, exist_ok=True)
    with table_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved: {table_path}")
    print(f"Generated {3 + len(RADII) + sum(n for n, _ in outputs)} plots in {args.output_dir}")


if __name__ == "__main__":
    main()
