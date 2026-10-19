#!/usr/bin/env python3
"""
Compares the jet eta distribution (h2JetEtav3) between the default JES (stored pt_calib, eta-binned
legacy JES) and the Z0 JES (Jet-Anav3 run with jes_z0_file: eta-averaged legacy JES on the raw pT),
run by run, for R = 0.2 and R = 0.3.

Both Jet-Anav3 productions must come from the same trees. The Z0 jet pT cut is raised to where its
sample is complete given the tree cut (see Z0_PT_MIN), so the shape comparison is the like-for-like one.

The pT section compares the calibrated pT spectra (pT axis of h2JetEtav3) for all eta, inside and outside the
legacy eta-bin edge, to see whether the JES switch matters only near the cut. The first bin (10-11 GeV) is not
like-for-like: the Z0 sample has no jets below Z0_PT_MIN.

The mechanism figure (one per R) explains the difference: the legacy JES is a different function of the
raw pT in each of its 4 eta bins, so the raw pT needed to pass the calibrated-pT cut steps at the bin
edges, while the Z0 JES needs the same raw pT everywhere. It reads the JES payload and a few trees for
the raw-pT spectrum and the count ratio predicted from the thresholds alone.
"""

import argparse
import concurrent.futures
import os
from pathlib import Path
import sys
import numpy as np
import uproot
import awkward as ak
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mplhep as hep

from plot_jet_bkgsub_compare import save_fig, calc_ratio_errors, setup_ratio_panel
from plot_jet_noflow_compare import METHODS, parse_list_file, load_luminosity, load_hists, plot_eta_pair, add_header, info_text

# Z0 jet pT cut applied by Jet-Anav3 (printed as "JES Z0 R = 0.X ... jet pT min") for JES_Calib_Default_47289-53864_v6
Z0_PT_MIN = {"r02": 10.19, "r03": 10.08}
DEFAULT_PT_MIN = 10.0  # Jet-Anav3 jet pT min on the stored (eta-binned) pt_calib

# CDB payload the trees were calibrated with (newcdbtag), see the Fun4All job logs
LEGACY_JES_FILE = "/cvmfs/sphenix.sdcc.bnl.gov/calibrations/sphnxpro/cdb/JES_Calib_Default/0b/08/0b0822d756de4bfc5c3793e6db0716c4_JES_Calib_Default_47289-53864_v6.root"
ETA_BIN_COLORS = ["#1b9e77", "#d95f02", "#7570b3", "#e7298a"]
Z0_COLOR = "#ff7f00"
DEFAULT_COLOR = "#984ea3"

JES_STYLES = {
    "default": {"label": r"Default JES ($\eta$-binned)", "color": "#984ea3", "marker": "o"},
    "z0": {"label": "Z0 JES", "color": "#ff7f00", "marker": "s"},
}
RATIO_LABEL = "Z0 / Default"

# pT bins for the spectra (merged from the 1 GeV bins of h2JetEtav3, pT axis 10-80 GeV)
PT_EDGES = np.concatenate([np.arange(10, 20, 1), np.arange(20, 30, 2), [30, 35, 40, 50, 60, 80]])
PT_PLOT_MAX = 40  # few jets above this in one run
REGION_COLORS = {"all": "black", "inside": "#377eb8", "outside": "#e41a1c"}


def n_events(file_path, event_bin):
    with uproot.open(file_path) as rf:
        return rf["hEvent"].values()[event_bin]


def run_task(task):
    """Makes the default vs Z0 eta comparison for one run, method and R. Returns 1 on success, 0 if skipped."""
    run, default_file, z0_file, lumi, method, r_tag, r_jet, shape, output_dir, ratio_ylim, save_pdf = task

    hists = {"default": load_hists(default_file, r_tag, "eta").get(method),
             "z0": load_hists(z0_file, r_tag, "eta").get(method)}
    if any(h is None for h in hists.values()):
        print(f"Warning: Missing {method} {r_tag} for run {run}, skipping")
        return 0

    ylabel = None
    if shape:
        # plot_eta_pair divides by n_events and the bin width: use the jet count in the acceptance
        eta_limit = round(1.1 - r_jet, 1)
        for h in hists.values():
            centers = (h["edges"][:-1] + h["edges"][1:]) / 2.0
            h["n_events"] = np.sum(h["values"][np.abs(centers) <= eta_limit])
        ylabel = r"$\frac{1}{N_{\mathrm{jets}}}\frac{dN}{d\eta}$"

    styles = {key: dict(meta) for key, meta in JES_STYLES.items()}
    styles["z0"]["label"] += rf" ($p_{{T}} \geq {Z0_PT_MIN[r_tag]:g}$ GeV)"

    name = "default_vs_z0" + ("_shape" if shape else "")
    out_png = output_dir / "eta" / name / method / r_tag / f"run_{run}_h2JetEtav3_{r_tag}_{method}_{name}.png"
    plot_eta_pair(run, r_jet, lumi, hists, "default", "z0", RATIO_LABEL, shape, out_png,
                  ratio_ylim=ratio_ylim, save_pdf=save_pdf, styles=styles, method_label=METHODS[method]["label"], ylabel=ylabel)
    return 1


def eta_regions(r_jet):
    """Labels of all eta, inside and outside the legacy eta-bin edge, and the edge."""
    edge = abs(legacy_eta_edges(r_jet)[0])
    eta_max = round(1.1 - r_jet, 1)
    labels = {"all": rf"$|\eta| < {eta_max:g}$", "inside": rf"$|\eta| < {edge:g}$", "outside": rf"${edge:g} \leq |\eta| < {eta_max:g}$"}
    return labels, edge


def load_pt_regions(file_path, r_tag, method, edge):
    """pT spectra of h2JetEtav3 (rebinned to PT_EDGES) for all eta, inside and outside the legacy edge."""
    name = f"h2JetEtav3_{r_tag}_{method}"
    with uproot.open(file_path) as rf:
        if name not in rf:
            print(f"Warning: {name} not found in {file_path}")
            return None
        vals, pt_edges, eta_edges = rf[name].to_numpy()
        variances = rf[name].variances()
    idx = np.searchsorted(pt_edges, PT_EDGES[:-1])
    if not np.allclose(pt_edges[idx], PT_EDGES[:-1]) or not np.isclose(pt_edges[-1], PT_EDGES[-1]):
        raise ValueError(f"PT_EDGES do not match the pT axis of {name}")

    abs_eta = np.abs(0.5 * (eta_edges[1:] + eta_edges[:-1]))
    masks = {"all": np.ones_like(abs_eta, dtype=bool), "inside": abs_eta < edge, "outside": abs_eta >= edge}
    out = {}
    for region, mask in masks.items():
        out[region] = {"values": np.add.reduceat(vals[:, mask].sum(axis=1), idx),
                       "errors": np.sqrt(np.add.reduceat(variances[:, mask].sum(axis=1), idx))}
    return out


def pt_ratio(spectra, region):
    """Z0 / default per pT bin (nan where either is empty). The errors treat the samples as independent, so are too large."""
    d, z = spectra["default"][region], spectra["z0"][region]
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where((d["values"] > 0) & (z["values"] > 0), z["values"] / d["values"], np.nan)
        err = calc_ratio_errors(ratio, z["errors"], d["errors"], d["values"])
    return ratio, err


def shade_incomplete_bin(ax, label=True):
    """Marks the first pT bin, where the Z0 sample is incomplete (Z0 cut above 10 GeV)."""
    ax.axvspan(PT_EDGES[0], PT_EDGES[1], color="gray", alpha=0.2, label="Z0 cut inside bin" if label else None)


def plot_pt_region(run, r_jet, lumi, method, spectra, region, region_label, z0_cut, out_path, save_pdf):
    """Default vs Z0 calibrated pT spectrum in one eta region, with the Z0 / default ratio."""
    hep.style.use("ATLAS")
    widths = np.diff(PT_EDGES)
    centers = 0.5 * (PT_EDGES[1:] + PT_EDGES[:-1])

    fig, (ax, ax_ratio) = plt.subplots(2, 1, figsize=(8, 8), gridspec_kw={"height_ratios": [3, 1], "hspace": 0.08}, sharex=True)
    positive = []
    for key, meta in JES_STYLES.items():
        d = spectra[key][region]
        label = meta["label"] + (rf" ($p_{{T}} \geq {z0_cut:g}$ GeV)" if key == "z0" else "")
        hep.histplot((d["values"] / widths, PT_EDGES), ax=ax, histtype="step", color=meta["color"], linewidth=2, label=label)
        positive.append(d["values"][d["values"] > 0] / widths[d["values"] > 0])
    positive = np.concatenate(positive)
    shade_incomplete_bin(ax)

    ratio, err = pt_ratio(spectra, region)
    ok = np.isfinite(ratio)
    ax_ratio.errorbar(centers[ok], ratio[ok], yerr=err[ok], xerr=widths[ok] / 2, fmt=JES_STYLES["z0"]["marker"],
                      color=JES_STYLES["z0"]["color"], markersize=4, capsize=2, linewidth=1.0)
    shade_incomplete_bin(ax_ratio, label=False)

    ax.set_yscale("log")
    ax.set_ylim(positive.min() * 1e-3, positive.max() * 1e2)  # room for the legend below the spectrum
    ax.set_xlim(PT_EDGES[0], PT_PLOT_MAX)
    ax.set_ylabel(r"$dN/dp_{T}$ [GeV$^{-1}$]")
    add_header(ax, run, r_jet, lumi)
    ax.text(0.95, 0.95, info_text(METHODS[method]["label"], pt_cut=region_label), transform=ax.transAxes, ha="right", va="top", fontsize=15)
    ax.legend(loc="lower left", frameon=False, fontsize=15)

    setup_ratio_panel(ax_ratio, xlabel=r"Calibrated jet $p_{T}$ [GeV]", ylabel=RATIO_LABEL, all_ratios=list(ratio[ok & (err < 0.25)]), min_margin=0.05)
    plt.subplots_adjust(left=0.14, right=0.96, bottom=0.10, top=0.94)
    save_fig(fig, out_path, save_pdf)
    plt.close(fig)


def plot_pt_ratio_regions(run, r_jet, lumi, method, spectra, labels, out_path, save_pdf):
    """Z0 / default pT spectrum ratio for all eta, inside and outside the legacy edge on one panel."""
    hep.style.use("ATLAS")
    widths = np.diff(PT_EDGES)
    centers = 0.5 * (PT_EDGES[1:] + PT_EDGES[:-1])
    offsets = {"all": 0.0, "inside": -0.15, "outside": 0.15}  # keep the error bars apart

    fig, ax = plt.subplots(figsize=(9, 6.5))
    reliable = []
    for region, label in labels.items():
        ratio, err = pt_ratio(spectra, region)
        ok = np.isfinite(ratio)
        ax.errorbar(centers[ok] + offsets[region] * widths[ok], ratio[ok], yerr=err[ok], fmt="o", color=REGION_COLORS[region],
                    markersize=5, capsize=2, linewidth=1.2, label=label)
        reliable.append(ratio[ok & (err < 0.1)])
    reliable = np.concatenate(reliable)
    shade_incomplete_bin(ax)
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1)
    ax.set_xlim(PT_EDGES[0], PT_PLOT_MAX)
    lo, hi = min(reliable.min(), 1.0), max(reliable.max(), 1.0)
    ax.set_ylim(lo - 0.1 * (hi - lo), hi + 0.45 * (hi - lo))
    ax.set_xlabel(r"Calibrated jet $p_{T}$ [GeV]")
    ax.set_ylabel(f"Jet counts {RATIO_LABEL}")
    add_header(ax, run, r_jet, lumi)
    ax.text(0.97, 0.96, METHODS[method]["label"], transform=ax.transAxes, ha="right", va="top", fontsize=16)
    ax.legend(loc="upper left", frameon=False, fontsize=15, ncol=2)
    fig.tight_layout()
    save_fig(fig, out_path, save_pdf)
    plt.close(fig)


def run_pt_task(task):
    """Makes the default vs Z0 pT spectrum comparison for one run, method and R. Returns 1 on success, 0 if skipped."""
    run, default_file, z0_file, lumi, method, r_tag, r_jet, output_dir, save_pdf = task
    labels, edge = eta_regions(r_jet)
    spectra = {"default": load_pt_regions(default_file, r_tag, method, edge), "z0": load_pt_regions(z0_file, r_tag, method, edge)}
    if any(s is None for s in spectra.values()):
        print(f"Warning: Missing {method} {r_tag} for run {run}, skipping the pT comparison")
        return 0

    out_dir = output_dir / "pt" / "default_vs_z0" / method / r_tag
    for region, label in labels.items():
        out_png = out_dir / f"run_{run}_jet_pt_{r_tag}_{method}_default_vs_z0_{region}.png"
        plot_pt_region(run, r_jet, lumi, method, spectra, region, label, Z0_PT_MIN[r_tag], out_png, save_pdf)
    plot_pt_ratio_regions(run, r_jet, lumi, method, spectra, labels, out_dir / f"run_{run}_jet_pt_{r_tag}_{method}_ratio_regions.png", save_pdf)

    for region in labels:
        ratio, _ = pt_ratio(spectra, region)
        print(f"Run {run} {method} {r_tag} {region:>7}: Z0/default per pT bin " + " ".join(
            f"[{lo:g},{hi:g}):{r:.3f}" for lo, hi, r in zip(PT_EDGES[:-1], PT_EDGES[1:], ratio) if np.isfinite(r) and hi <= 30))
    return 1


def legacy_eta_edges(r_jet):
    """Edges of JetCalib's legacy eta bins for |z| < 30 cm: 4 equal bins on [-1.2+R, 1.2-R]."""
    lo, hi = -1.2 + r_jet, 1.2 - r_jet
    return np.round([lo + (hi - lo) * k / 4 for k in (1, 2, 3)], 3)


def load_jes(jes_file, r_tag):
    """Legacy JES for one R (|z| < 30 cm): calibrated pT of the 4 eta-bin TF1s and of Z0 on a raw-pT grid,
    and the raw pT each needs to pass its calibrated-pT cut."""
    import ROOT
    fin = ROOT.TFile.Open(str(jes_file))
    if not fin or fin.IsZombie():
        raise RuntimeError(f"cannot open {jes_file}")
    name = f"JES_Calib_Func_R{r_tag[1:]}_Z0"
    funcs = [fin.Get(f"{name}_Eta{k}") for k in range(4)] + [fin.Get(name)]
    grid = np.linspace(4.0, 40.0, 721)
    calib = np.array([[f.Eval(x) for x in grid] for f in funcs])
    jes = {
        "grid": grid,
        "eta_bins": calib[:4],
        "z0": calib[4],
        "thr_default": np.array([f.GetX(DEFAULT_PT_MIN, 1, 50) for f in funcs[:4]]),
        "thr_z0": funcs[4].GetX(Z0_PT_MIN[r_tag], 1, 50),
    }
    fin.Close()
    return jes


def load_tree_jets(files, method, r_tags):
    """Raw pT and eta of the selected jets (Jet-Anav3 selection without the Calo-MBD cut)."""
    flow = "iter" if method.startswith("iter") else ("mult" if method.startswith("mult") else None)
    event_branches = ["zvtx", "centrality"] + ([f"is_flow_failure_{flow}", f"calo_v2_{flow}"] if flow else [])
    branches = event_branches + [f"{v}_{method}_{r}" for r in r_tags for v in ("pt", "eta", "e")]

    jets = {r: {"pt": [], "eta": []} for r in r_tags}
    for arr in uproot.iterate([f"{f}:T" for f in files], branches, step_size="200 MB"):
        evt = (np.abs(arr["zvtx"]) < 10) & (arr["centrality"] <= 59.5)
        if flow:
            evt = evt & ~arr[f"is_flow_failure_{flow}"] & (np.abs(arr[f"calo_v2_{flow}"]) < 0.48)
        arr = arr[evt]
        for r in r_tags:
            energy = ak.flatten(arr[f"e_{method}_{r}"]).to_numpy()
            ok = energy > 0
            jets[r]["pt"].append(ak.flatten(arr[f"pt_{method}_{r}"]).to_numpy()[ok])
            jets[r]["eta"].append(ak.flatten(arr[f"eta_{method}_{r}"]).to_numpy()[ok])
    return {r: {k: np.concatenate(v) for k, v in d.items()} for r, d in jets.items()}


def plot_mechanism(run, method, r_tag, r_jet, jes, jets, observed, n_trees, out_path, save_pdf):
    """How the two JES differ (a, b) and how that changes the eta distribution (c, d)."""
    hep.style.use("ATLAS")
    edges_leg = legacy_eta_edges(r_jet)
    edge = abs(edges_leg[0])
    eta_max = round(1.1 - r_jet, 1)
    t1, t3 = edges_leg[0], edges_leg[2]
    bin_labels = [rf"$\eta < {t1:g}$", rf"${t1:g} \leq \eta < 0$", rf"$0 \leq \eta < {t3:g}$", rf"$\eta \geq {t3:g}$"]
    thr_def, thr_z0 = jes["thr_default"], jes["thr_z0"]
    z0_cut = Z0_PT_MIN[r_tag]

    fig, axes = plt.subplots(2, 2, figsize=(16, 13))
    (ax_a, ax_b), (ax_c, ax_d) = axes

    # (a) calibration factor vs raw pT; where a factor curve crosses its cut curve is the raw-pT threshold
    grid = jes["grid"]
    for k in range(4):
        ax_a.plot(grid, jes["eta_bins"][k] / grid, color=ETA_BIN_COLORS[k], linewidth=2, label=f"Default, {bin_labels[k]}")
    ax_a.plot(grid, jes["z0"] / grid, color=Z0_COLOR, linewidth=3, linestyle="--", label=r"Z0 ($\eta$-averaged)")
    ax_a.plot(grid, DEFAULT_PT_MIN / grid, color="gray", linewidth=1.5, linestyle=":", label=rf"cut: $p_{{T}}^{{calib}} = {DEFAULT_PT_MIN:g}$ GeV")
    ax_a.plot(grid, z0_cut / grid, color=Z0_COLOR, linewidth=1.5, linestyle=":", label=rf"cut: $p_{{T}}^{{calib}} = {z0_cut:g}$ GeV (Z0)")
    for k in range(4):
        ax_a.plot(thr_def[k], DEFAULT_PT_MIN / thr_def[k], "o", color=ETA_BIN_COLORS[k], markersize=8)
    ax_a.plot(thr_z0, z0_cut / thr_z0, "s", color=Z0_COLOR, markersize=9)
    factors = np.concatenate([jes["eta_bins"].ravel() / np.tile(grid, 4), jes["z0"] / grid])
    ax_a.set_xlim(4, 25)
    ax_a.set_ylim(factors[np.tile(grid, 5) <= 25].min() * 0.97, factors[np.tile(grid, 5) >= 4].max() * 1.03)
    ax_a.set_xlabel(r"Raw jet $p_{T}$ [GeV]")
    ax_a.set_ylabel(r"$p_{T}^{calib} / p_{T}^{raw}$")
    ax_a.legend(loc="upper right", fontsize=15, frameon=False)

    # (b) raw-pT threshold vs eta: steps at the legacy bin edges for the default, flat for Z0
    eta_fine = np.linspace(-eta_max, eta_max, 2001)
    thr_eta = thr_def[np.digitize(eta_fine, edges_leg)]
    ax_b.plot(eta_fine, thr_eta, color=DEFAULT_COLOR, linewidth=2.5, label=rf"Default ($p_{{T}}^{{calib}} \geq {DEFAULT_PT_MIN:g}$ GeV)")
    ax_b.axhline(thr_z0, color=Z0_COLOR, linewidth=2.5, linestyle="--", label=rf"Z0 ($p_{{T}}^{{calib}} \geq {z0_cut:g}$ GeV)")
    ax_b.fill_between(eta_fine, thr_eta, thr_z0, where=thr_eta < thr_z0, color=DEFAULT_COLOR, alpha=0.2, label="jets only the default keeps")
    for x in edges_leg:
        ax_b.axvline(x, color="gray", linestyle=":", linewidth=1)
    for k, x in enumerate([(-eta_max + t1) / 2, t1 / 2, t3 / 2, (eta_max + t3) / 2]):
        ax_b.text(x, thr_def[k] - 0.02, f"{thr_def[k]:.2f}", color=DEFAULT_COLOR, ha="center", va="top", fontsize=17, fontweight="bold")
    ax_b.text(0, thr_z0 + 0.02, f"{thr_z0:.2f}", color=Z0_COLOR, ha="center", va="bottom", fontsize=17, fontweight="bold")
    span = thr_z0 - thr_def.min()
    ax_b.set_ylim(thr_def.min() - 1.5 * span, thr_z0 + 1.5 * span)
    ax_b.set_xlim(-eta_max, eta_max)
    ax_b.set_xlabel(r"$\eta$")
    ax_b.set_ylabel(r"Raw $p_{T}$ threshold [GeV]")
    ax_b.legend(loc="upper right", fontsize=15, frameon=False)

    # (c) raw-pT spectra inside / outside the edge: the shaded jets are the ones only the default keeps
    pt, eta = jets["pt"], jets["eta"]
    acc = np.abs(eta) < eta_max
    pt_lo = np.floor((thr_def.min() - 0.5) * 10) / 10
    pt_edges = np.arange(pt_lo, pt_lo + 3.5 + 1e-9, 0.02)
    centers = 0.5 * (pt_edges[1:] + pt_edges[:-1])
    region_colors = {"inside": "#377eb8", "outside": "#e41a1c"}
    regions = {"inside": acc & (np.abs(eta) < edge), "outside": acc & (np.abs(eta) >= edge)}
    for region, sel in regions.items():
        counts = np.histogram(pt[sel], pt_edges)[0] / np.diff(pt_edges)
        label = rf"$|\eta| < {edge:g}$" if region == "inside" else rf"${edge:g} \leq |\eta| < {eta_max:g}$"
        ax_c.stairs(counts, pt_edges, color=region_colors[region], linewidth=2, label=label)
        if region == "outside":
            shade = (centers >= thr_def[[0, 3]].min()) & (centers < thr_z0)
            ax_c.fill_between(centers, counts, step="mid", where=shade, color=DEFAULT_COLOR, alpha=0.3)
            n_def = np.sum(sel & (pt >= thr_def[np.digitize(eta, edges_leg)]))
            n_z0 = np.sum(sel & (pt >= thr_z0))
            ax_c.text(0.03, 0.97, f"Z0 keeps {n_z0 / n_def:.1%} of\nthe outer default jets",
                      transform=ax_c.transAxes, ha="left", va="top", fontsize=16, color=region_colors[region],
                      bbox=dict(facecolor="white", edgecolor="none", alpha=0.9, pad=2))
    for k in (0, 1):
        ax_c.axvline(thr_def[[0, 3]][k], color=DEFAULT_COLOR, linestyle=":", linewidth=1.5,
                     label=r"default threshold, outside" if k == 0 else None)
    ax_c.axvline(thr_def[1], color=DEFAULT_COLOR, linestyle="-.", linewidth=1.5, label="default threshold, inside")
    ax_c.axvline(thr_z0, color=Z0_COLOR, linestyle="--", linewidth=2, label="Z0 threshold")
    ax_c.set_yscale("log")
    ax_c.set_xlim(pt_edges[0], pt_edges[-1])
    ax_c.set_xlabel(r"Raw jet $p_{T}$ [GeV]")
    ax_c.set_ylabel(r"$dN/dp_{T}^{raw}$ [GeV$^{-1}$]")
    ax_c.legend(loc="upper right", fontsize=15, frameon=False)

    # (d) count ratio Z0 / default vs eta: observed (Jet-Anav3) vs predicted from the thresholds alone
    eta_edges = observed["edges"]
    eta_centers = 0.5 * (eta_edges[1:] + eta_edges[:-1])
    n_def = np.histogram(eta[acc & (pt >= thr_def[np.digitize(eta, edges_leg)])], eta_edges)[0]
    n_z0 = np.histogram(eta[acc & (pt >= thr_z0)], eta_edges)[0]
    in_acc = np.abs(eta_centers) < eta_max
    with np.errstate(divide="ignore", invalid="ignore"):
        predicted = np.where(n_def > 0, n_z0 / n_def, np.nan)
        obs = observed["z0"] / observed["default"]
        obs_err = calc_ratio_errors(obs, np.sqrt(observed["z0"]), np.sqrt(observed["default"]), observed["default"])
    ax_d.stairs(np.where(in_acc, predicted, np.nan), eta_edges, color="black", linewidth=2, label=f"predicted from thresholds ({n_trees} trees)")
    ax_d.errorbar(eta_centers[in_acc], obs[in_acc], yerr=obs_err[in_acc], fmt="s", color=Z0_COLOR, markersize=5,
                  label=f"observed, Jet-Anav3 run {run}")
    for x in edges_leg:
        ax_d.axvline(x, color="gray", linestyle=":", linewidth=1)
    ax_d.axhline(1.0, color="gray", linestyle="--", linewidth=1)
    ax_d.set_xlim(-eta_max, eta_max)
    lo = np.nanmin(np.concatenate([predicted[in_acc], obs[in_acc]]))
    ax_d.set_ylim(lo - 0.15 * (1.05 - lo), 1.05 + 0.25 * (1.05 - lo))
    ax_d.set_xlabel(r"$\eta$")
    ax_d.set_ylabel("Jet counts Z0 / Default")
    ax_d.legend(loc="upper right", fontsize=15, frameon=False)

    fig.text(0.5, 0.995, rf"Run {run}, $R = {r_jet:g}$, {METHODS[method]['label']}: default ($\eta$-binned) vs Z0 JES, $|z| < 10$ cm, 0–60%",
             ha="center", va="top", fontsize=19)
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    save_fig(fig, out_path, save_pdf)

    # Each panel on its own: add a header to every axes, then save the axes' tight bounding box
    header = rf"Run {run}, $R = {r_jet:g}$, {METHODS[method]['label']}"
    panels = [(ax_a, "jes_vs_rawpt"), (ax_b, "rawpt_threshold_vs_eta"), (ax_c, "rawpt_spectra"), (ax_d, "count_ratio_vs_eta")]
    for ax, _ in panels:
        ax.text(1.0, 1.01, header, transform=ax.transAxes, ha="right", va="bottom", fontsize=16)
    fig.suptitle("")
    for text in fig.texts:
        text.set_visible(False)
    renderer = fig.canvas.get_renderer()
    for ax, name in panels:
        # hide the other panels so none of their labels leak into the crop
        for other, _ in panels:
            other.set_visible(other is ax)
        bbox = ax.get_tightbbox(renderer).transformed(fig.dpi_scale_trans.inverted()).padded(0.1)
        panel_path = out_path.parent / "panels" / f"{out_path.stem}_{name}.png"
        panel_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(panel_path, dpi=300, bbox_inches=bbox)
        print(f"Saved: {panel_path}")
        if save_pdf:
            fig.savefig(panel_path.with_suffix(".pdf"), bbox_inches=bbox)
    plt.close(fig)


def make_mechanism_plots(args, default_files, z0_files):
    run = args.mechanism_run
    if run not in default_files or run not in z0_files:
        print(f"Warning: Mechanism run {run} is not in both lists, skipping the mechanism plots")
        return 0
    tree_files = [l.strip() for l in args.trees.read_text().splitlines() if f"/{run}/" in l][: args.max_trees]
    if not tree_files:
        print(f"Warning: No trees for run {run} in {args.trees}, skipping the mechanism plots")
        return 0

    n_made = 0
    for method in args.methods:
        print(f"Mechanism plots: reading {len(tree_files)} trees of run {run} for {method}")
        jets = load_tree_jets(tree_files, method, ["r02", "r03"])
        for r_tag, r_jet in [("r02", 0.2), ("r03", 0.3)]:
            hists = {"default": load_hists(default_files[run], r_tag, "eta").get(method),
                     "z0": load_hists(z0_files[run], r_tag, "eta").get(method)}
            if any(h is None for h in hists.values()):
                print(f"Warning: Missing {method} {r_tag} for run {run}, skipping its mechanism plot")
                continue
            observed = {"edges": hists["default"]["edges"], "default": hists["default"]["values"], "z0": hists["z0"]["values"]}
            out_png = args.output_dir / "mechanism" / method / f"run_{run}_jes_mechanism_{r_tag}_{method}.png"
            plot_mechanism(run, method, r_tag, r_jet, load_jes(args.jes_file, r_tag), jets[r_tag], observed, len(tree_files), out_png, args.save_pdf)
            n_made += 1
    return n_made


def main():
    parser = argparse.ArgumentParser(description="Compare the jet eta distribution between the default (eta-binned) JES and the Z0 JES, run by run.")
    parser.add_argument("-d", "--default-list", type=Path, default=Path("/sphenix/user/anarde/sEPD-Study/jetAnav3/10-15-26/test.list"),
                        help="List of default Jet-Anav3 merged ROOT files named <run>.root (default: 10-15-26)")
    parser.add_argument("-z", "--z0-list", type=Path, default=Path("/sphenix/user/anarde/sEPD-Study/jetAnav3/10-17-26-test/test.list"),
                        help="List of Z0 Jet-Anav3 merged ROOT files named <run>.root (default: 10-17-26-test)")
    parser.add_argument("-l", "--lumi-list", type=Path, default=Path("/sphenix/user/anarde/sEPD-Study/f4a_bkgsub/10-15-26/test.list"),
                        help="List of Event QA ROOT files named <run>.root with hLuminosity for the luminosity label")
    parser.add_argument("-o", "--output-dir", type=Path, default=Path("/sphenix/user/anarde/sEPD-Study/plots/Jet-Ana/jes_z0_compare"),
                        help="Directory to save the plots")
    parser.add_argument("--runs", nargs="*", type=int, help="Specific run number(s) to process (default: runs in both lists)")
    parser.add_argument("--methods", nargs="+", default=["iternf"], choices=list(METHODS), help="Subtraction methods (default: iternf)")
    parser.add_argument("--ratio-ylim", nargs=2, type=float, default=None, metavar=("YMIN", "YMAX"),
                        help="Custom y-axis limits for the ratio subplot (default: auto-zoom around 1.0)")
    parser.add_argument("--jes-file", type=Path, default=Path(LEGACY_JES_FILE), help="Legacy JES payload used by both productions (mechanism plots)")
    parser.add_argument("-t", "--trees", type=Path, default=Path("/sphenix/user/anarde/sEPD-Study/jetAnav3/10-15-26/tree.list"),
                        help="Tree list of the production both Jet-Anav3 runs used (mechanism plots: raw-pT spectra and predicted ratio)")
    parser.add_argument("--mechanism-run", type=int, default=68144, help="Run for the mechanism plots (default: 68144)")
    parser.add_argument("--max-trees", type=int, default=30, help="Number of trees of the mechanism run to read (default: 30)")
    parser.add_argument("--no-mechanism", action="store_true", help="Skip the mechanism plots")
    parser.add_argument("-j", "--jobs", type=int, default=min(8, os.cpu_count() or 1), help="Number of parallel worker processes")
    parser.add_argument("--save-pdf", action="store_true", help="Save PDF version of plots in addition to PNG")
    args = parser.parse_args()

    default_files = parse_list_file(args.default_list)
    z0_files = parse_list_file(args.z0_list)
    common = sorted(set(default_files) & set(z0_files))
    runs = sorted(args.runs) if args.runs else common
    if not runs:
        print(f"Error: No runs in both {args.default_list} and {args.z0_list}")
        sys.exit(1)

    lumi_files = parse_list_file(args.lumi_list) if args.lumi_list.exists() else {}

    tasks = []
    for run in runs:
        if run not in default_files or run not in z0_files:
            print(f"Warning: Run {run} is not in both lists, skipping")
            continue
        default_file, z0_file = default_files[run], z0_files[run]

        # both productions must have run over the same events
        for method in args.methods:
            event_bin = METHODS[method]["event_bin"]
            n_default, n_z0 = n_events(default_file, event_bin), n_events(z0_file, event_bin)
            if n_default != n_z0:
                print(f"Warning: Run {run} {method}: different event counts (default {n_default:.0f}, Z0 {n_z0:.0f}); the productions may not use the same trees")

        lumi_file = lumi_files.get(run)
        lumi = load_luminosity(lumi_file) if lumi_file is not None and lumi_file.exists() else None
        for method in args.methods:
            for r_tag, r_jet in [("r02", 0.2), ("r03", 0.3)]:
                for shape in (True, False):
                    tasks.append((run_task, (run, default_file, z0_file, lumi, method, r_tag, r_jet, shape, args.output_dir, args.ratio_ylim, args.save_pdf)))
                tasks.append((run_pt_task, (run, default_file, z0_file, lumi, method, r_tag, r_jet, args.output_dir, args.save_pdf)))

    n_jobs = max(1, min(args.jobs, len(tasks)))
    print(f"Generating {len(tasks)} comparisons for runs {', '.join(map(str, runs))} with {n_jobs} worker(s)")
    if n_jobs == 1:
        success_count = sum(fn(t) for fn, t in tasks)
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=n_jobs) as executor:
            success_count = sum(f.result() for f in [executor.submit(fn, t) for fn, t in tasks])

    print(f"Successfully generated {success_count}/{len(tasks)} comparisons in {args.output_dir}")

    if not args.no_mechanism:
        n_mechanism = make_mechanism_plots(args, default_files, z0_files)
        print(f"Generated {n_mechanism} mechanism plot(s) in {args.output_dir / 'mechanism'}")


if __name__ == "__main__":
    main()
