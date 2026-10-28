#!/usr/bin/env python3
"""
Constituent structure of high-pT iter-subtracted R = 0.2 jets from the detailed Fun4All_BkgSub
trees (one tree per high-pT event, with the jet tower constituents per calorimeter layer).

Selection follows the hJetPtv3 path of Jet-Anav3:
  event: centrality <= 59.5%, Calo-MBD consistency (3.5 sigma about the per-run ProfileX of
         h2TotalCaloE_MBD), no iter flow failure, |calo v2| < 0.48 (|z| < 10 cm is upstream)
  jet:   |eta| < 0.9, E > 0, stored (eta-binned) calibrated pT > pt-min

Constituents: every tower in every layer (retowered EMCal, IHCal, OHCal) is one FastJet input,
so N_const = n_EMCal + n_IHCal + n_OHCal. UE-subtracted towers can be negative, so the count of
E > 0 towers is kept as well. z_lead = (max tower E) / (jet E), both raw (uncalibrated).

Timing (JetValidationv3 with set_do_timing): t is the jet time, the energy-weighted time of the
constituent towers above 0.1 GeV, and t_<layer> the same per layer, so t lies between the layer
times. t - t0 is the jet time relative to the (main-bunch corrected) MBD t0; the band |t - t0| <
--dt-max (default 3 ns, the TimingCut MBD window) marks in-time jets.

With --event-displays, each event with a selected jet also gets tower displays (drawn with the
calo_display_utils.py panels, tower indices decoded with scripts/plots/calo/tower_info_defs.py):
  unsub_1x4    -- unsubtracted base EMCal, retowered EMCal, IHCal, OHCal
  iter_r02_1x3 -- iter-subtracted EMCal, IHCal, OHCal with the R = 0.2 jet outlines
  zoom_1x4     -- unsubtracted 1x4 cropped around the leading tower of the highest-pT selected jet
"""

import argparse
import concurrent.futures
import csv
import functools
import os
import sys
from pathlib import Path
import numpy as np
import uproot
import awkward as ak
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mplhep as hep
from matplotlib.ticker import FuncFormatter

from plot_jet_bkgsub_compare import save_fig
from plot_jet_emfrac_compare import find_trees
from calo_display_utils import JET_COLORS, get_cluster_boundary_segments, plot_calo_2d, make_1x3_calo_plot, make_1x4_calo_plot

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "calo"))
from tower_info_defs import get_emcal_ieta_iphi, get_hcal_ieta_iphi

TREE_LIST = "/sphenix/user/anarde/sEPD-Study/f4a_bkgsub/10-28-26-high-pt/tree.list"
CALO_MBD_LIST = "/sphenix/user/anarde/sEPD-Study/event_qa/10-04-26/test.list"

# Jet-Anav3 cuts
CENT_HIGH = 59.5
CALO_V2_MAX = 0.48
ETA_MAX = 0.9
ZVTX_MAX = 10.0

LAYERS = ["emcal", "ihcal", "ohcal"]
LAYER_LABELS = {"emcal": "EMCal", "ihcal": "IHCal", "ohcal": "OHCal"}
COLL = "iter_r02"

EVENT_BRANCHES = ["run", "event", "zvtx", "centrality", "t0", "emcal_energy", "ihcal_energy", "ohcal_energy",
                  "mbd_charge_south", "mbd_charge_north", "is_flow_failure_iter", "calo_v2_iter"]
JET_BRANCHES = [f"{v}_{COLL}" for v in ["pt", "pt_calib", "e", "eta", "phi", "t", "t_emcal", "t_ihcal", "t_ohcal"]]
TOWER_BRANCHES = [f"{layer}_tower_{v}_{COLL}" for layer in LAYERS for v in ["energy", "pt"]]

# event displays: full-event towers (unsubtracted and iter-subtracted) and the jet tower indices
DISPLAY_TOWERS = {
    "base": "emcal_base_tower", "retower": "emcal_retower_tower", "ihcal": "ihcal_tower", "ohcal": "ohcal_tower",
    "iter_emcal": "iter_emcal_tower", "iter_ihcal": "iter_ihcal_tower", "iter_ohcal": "iter_ohcal_tower",
}
DISPLAY_BRANCHES = [f"{b}_{v}" for b in DISPLAY_TOWERS.values() for v in ["index", "energy"]] + \
                   [f"{layer}_tower_index_{COLL}" for layer in LAYERS]
HCAL_SHAPE = (24, 64)
EMCAL_SHAPE = (96, 256)
ZOOM_HALF = 5  # HCal towers on each side of the leading tower
# retowering maps base EMCal towers to the HCal grid by geometry, not index // 4; a base phi shift
# of 2 towers matches it best, so the zoomed base EMCal window is only approximately aligned
EMCAL_PHI_SHIFT = 2

# jet features for every R = 0.2 jet in the selected events (the tree keeps calibrated pT >= 10 GeV)
FEATURE_BRANCHES = [f"iter_{layer}_tower_{v}" for layer in LAYERS for v in ["index", "energy"]] + \
                   [f"{layer}_tower_index_{COLL}" for layer in LAYERS] + [f"{v}_iter_r03" for v in ["pt", "eta", "phi"]]
FEATURE_PT_MIN = 10.0
MATCH_DR = 0.2  # R = 0.2 to R = 0.3 jet matching
SINGLE_LAYER_FRAC = 0.9
CATEGORY_LABELS = {"emcal": "EMCal > 90%", "ihcal": "IHCal > 90%", "ohcal": "OHCal > 90%", "mixed": "Mixed"}
TEXT_INFO = (
    r"$|z| < 10$ cm, Centrality: 0–60%, Calo-MBD" "\n"
    r"No Flow Failure, $|\mathrm{Calo}\ v_{2}| < 0.48$" "\n"
    r"$|\eta| < 0.9$, Energy > 0"
)


def load_trees(files, extra_branches=()):
    """Concatenates the event, jet and tower branches of all non-empty trees."""
    parts = []
    for path in files:
        with uproot.open(path) as fin:
            tree = fin["T"]
            if tree.num_entries == 0:
                continue
            parts.append(tree.arrays(EVENT_BRANCHES + JET_BRANCHES + TOWER_BRANCHES + list(extra_branches)))
    return ak.concatenate(parts) if parts else None


def load_calo_mbd_profiles(list_file, runs):
    """Per run, the mean and spread of total calo E vs MBD charge, as TH2::ProfileX(..., "s")."""
    paths = {Path(l.strip()).stem: l.strip() for l in Path(list_file).read_text().splitlines() if l.strip()}
    profiles = {}
    for run in runs:
        if str(run) not in paths:
            raise FileNotFoundError(f"run {run} not in {list_file}")
        with uproot.open(paths[str(run)]) as fin:
            h2 = fin["h2TotalCaloE_MBD"]
            counts = h2.values()  # [x, y], no flow bins: ProfileX(name, 1, -1) uses y bins 1..N
            y = 0.5 * (h2.axes[1].edges()[1:] + h2.axes[1].edges()[:-1])
            sumw = counts.sum(axis=1)
            with np.errstate(divide="ignore", invalid="ignore"):
                mean = (counts * y).sum(axis=1) / sumw
                sigma = np.sqrt(np.abs((counts * y ** 2).sum(axis=1) / sumw - mean ** 2))
            profiles[run] = (h2.axes[0].edges(), np.nan_to_num(mean), np.nan_to_num(sigma))
    return profiles


def pass_calo_mbd(events, profiles, sigma_cut):
    """Jet-Anav3 check_CaloMBD: |E_calo - mean| <= sigma_cut * sigma in the MBD charge bin."""
    total_e = ak.to_numpy(events.emcal_energy + events.ihcal_energy + events.ohcal_energy)
    mbd = ak.to_numpy(events.mbd_charge_south + events.mbd_charge_north)
    passed = np.zeros(len(events), dtype=bool)
    for i, run in enumerate(ak.to_numpy(events.run)):
        edges, mean, sigma = profiles[run]
        ibin = np.searchsorted(edges, mbd[i], side="right") - 1
        if 0 <= ibin < len(mean) and sigma[ibin] > 0:
            passed[i] = abs(total_e[i] - mean[ibin]) <= sigma_cut * sigma[ibin]
    return passed


def jet_table(events):
    """One flat record per jet with its constituent observables."""
    energy = {layer: events[f"{layer}_tower_energy_{COLL}"] for layer in LAYERS}
    tower_pt = {layer: events[f"{layer}_tower_pt_{COLL}"] for layer in LAYERS}
    all_e = ak.concatenate([energy[layer] for layer in LAYERS], axis=2)
    all_pt = ak.concatenate([tower_pt[layer] for layer in LAYERS], axis=2)

    e_jet = events[f"e_{COLL}"]
    pt_raw = events[f"pt_{COLL}"]
    lead_by_layer = np.stack([ak.to_numpy(ak.flatten(ak.fill_none(ak.max(energy[layer], axis=2), -np.inf))) for layer in LAYERS], axis=1)

    cols = {
        "run": ak.broadcast_arrays(events.run, e_jet)[0],
        "event": ak.broadcast_arrays(events.event, e_jet)[0],
        "centrality": ak.broadcast_arrays(events.centrality, e_jet)[0],
        "t0": ak.broadcast_arrays(events.t0, e_jet)[0],
        "pt_calib": events[f"pt_calib_{COLL}"],
        "pt_raw": pt_raw,
        "e": e_jet,
        "eta": events[f"eta_{COLL}"],
        "phi": events[f"phi_{COLL}"],
        "n_const": ak.num(all_e, axis=2),
        "n_pos": ak.sum(all_e > 0, axis=2),
        "e_sum": ak.sum(all_e, axis=2),
        "e_lead": ak.max(all_e, axis=2),
        "z_lead": ak.max(all_e, axis=2) / e_jet,
        "z_lead_pt": ak.max(all_pt, axis=2) / pt_raw,
        "t": events[f"t_{COLL}"],
        "dt_t0": events[f"t_{COLL}"] - events.t0,
    }
    for layer in LAYERS:
        cols[f"n_{layer}"] = ak.num(energy[layer], axis=2)
        cols[f"frac_{layer}"] = ak.sum(energy[layer], axis=2) / e_jet
        cols[f"t_{layer}"] = events[f"t_{layer}_{COLL}"]
    table = {k: ak.to_numpy(ak.fill_none(ak.flatten(v), np.nan)) for k, v in cols.items()}
    table["lead_layer"] = np.array(LAYERS)[np.argmax(lead_by_layer, axis=1)]
    return table


TICK_SIZE = 20
LEGEND_SIZE = 18
TEXT_SIZE = 16


def tighten(fig, ax):
    """Larger tick labels and narrower figure margins."""
    ax.tick_params(axis="both", which="major", labelsize=TICK_SIZE)
    fig.subplots_adjust(left=0.09, right=0.97, bottom=0.12, top=0.94)


def add_text(ax, n_jets, pt_min, extra=None):
    ax.text(1.0, 1.01, rf"Au+Au, Iterative, $R = 0.2$, $p_{{T}}^{{\mathrm{{calib}}}} > {pt_min:g}$ GeV, {n_jets} jets",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=TEXT_SIZE + 1)
    text = TEXT_INFO + (f"\n{extra}" if extra else "")
    ax.text(0.97, 0.96, text, transform=ax.transAxes, ha="right", va="top", fontsize=TEXT_SIZE)


def plot_n_const(jets, pt_min, out_path, save_pdf):
    fig, ax = plt.subplots(figsize=(10, 8))
    hi = max(jets["n_const"].max(), 1)
    edges = np.arange(-0.5, hi + 1.5 + 0.3 * hi)
    ax.hist(jets["n_const"], bins=edges, histtype="step", linewidth=2, color="#377eb8",
            label=rf"All towers, mean = {jets['n_const'].mean():.1f}")
    ax.hist(jets["n_pos"], bins=edges, histtype="step", linewidth=2, color="#e41a1c", linestyle="--",
            label=rf"Towers with $E > 0$, mean = {jets['n_pos'].mean():.1f}")
    ax.set_xlim(0, edges[-1])
    ax.set_xlabel(r"$N_{\mathrm{const}}$ (EMCal + IHCal + OHCal towers)")
    ax.set_ylabel("Jets")
    ax.set_ylim(0, ax.get_ylim()[1] * 1.2)
    ax.legend(loc="upper left", fontsize=LEGEND_SIZE, frameon=False)
    add_text(ax, len(jets["n_const"]), pt_min)
    tighten(fig, ax)
    save_fig(fig, out_path, save_pdf)
    plt.close(fig)


def plot_z_lead(jets, pt_min, out_path, save_pdf):
    fig, ax = plt.subplots(figsize=(10, 8))
    edges = np.linspace(0, 1.05, 22)
    ax.hist(jets["z_lead"], bins=edges, histtype="step", linewidth=2, color="#377eb8",
            label=rf"$E_{{\mathrm{{lead}}}} / E_{{\mathrm{{jet}}}}$, mean = {jets['z_lead'].mean():.2f}")
    ax.hist(jets["z_lead_pt"], bins=edges, histtype="step", linewidth=2, color="#ff7f00", linestyle="--",
            label=rf"$p_{{T,\mathrm{{lead}}}} / p_{{T,\mathrm{{jet}}}}^{{\mathrm{{raw}}}}$, mean = {jets['z_lead_pt'].mean():.2f}")
    ax.set_xlabel("Leading constituent fraction")
    ax.set_ylabel("Jets")
    ax.set_ylim(0, ax.get_ylim()[1] * 1.6)
    ax.legend(loc="upper left", fontsize=12, frameon=False)
    add_text(ax, len(jets["z_lead"]), pt_min)
    tighten(fig, ax)
    save_fig(fig, out_path, save_pdf)
    plt.close(fig)


def plot_layer_frac(jets, pt_min, out_path, save_pdf):
    fig, ax = plt.subplots(figsize=(10, 8))
    edges = np.linspace(-0.1, 1.1, 25)
    for layer, color in zip(LAYERS, ["#4daf4a", "#984ea3", "#e41a1c"]):
        ax.hist(np.clip(jets[f"frac_{layer}"], edges[0], edges[-1]), bins=edges, histtype="step", linewidth=2, color=color,
                label=rf"{LAYER_LABELS[layer]}, mean = {jets[f'frac_{layer}'].mean():.2f}")
    ax.set_xlabel(r"$E_{\mathrm{layer}} / E_{\mathrm{jet}}$")
    ax.set_ylabel("Jets")
    ax.set_ylim(0, ax.get_ylim()[1] * 1.6)
    ax.legend(loc="upper left", fontsize=12, frameon=False)
    add_text(ax, len(jets["e"]), pt_min)
    tighten(fig, ax)
    save_fig(fig, out_path, save_pdf)
    plt.close(fig)


LAYER_COLORS = {"emcal": "#4daf4a", "ihcal": "#984ea3", "ohcal": "#e41a1c"}


def plot_t0_correlations(jets, pt_min, out_path, save_pdf):
    """MBD t0 distribution and its correlation with the jet observables, by layer of the leading tower."""
    from scipy.stats import spearmanr

    t0 = jets["t0"]
    panels = [("pt_calib", r"Jet $p_{T}^{\mathrm{calib}}$ [GeV]"), ("z_lead", r"$E_{\mathrm{lead}} / E_{\mathrm{jet}}$"),
              ("n_const", r"$N_{\mathrm{const}}$"), ("frac_ohcal", r"$E_{\mathrm{OHCal}} / E_{\mathrm{jet}}$"),
              ("centrality", "Centrality [%]")]
    fig, axes = plt.subplots(2, 3, figsize=(20, 11))

    ax = axes[0, 0]
    lim = max(1, int(np.ceil(np.abs(t0).max())))
    edges = np.linspace(-lim, lim, 2 * lim + 1)
    ax.hist(t0, bins=edges, histtype="step", linewidth=2, color="black", label=f"All jets ({len(t0)})")
    for layer in LAYERS:
        sel = jets["lead_layer"] == layer
        if sel.any():
            ax.hist(t0[sel], bins=edges, histtype="step", linewidth=2, color=LAYER_COLORS[layer],
                    label=f"{LAYER_LABELS[layer]} leading tower ({sel.sum()})")
    ax.set_xlabel(r"MBD $t_{0}$ [ns]")
    ax.set_ylabel("Jets")
    ax.set_ylim(0, ax.get_ylim()[1] * 1.5)
    ax.legend(loc="upper left", fontsize=12, frameon=False)

    for ax, (key, label) in zip(axes.flat[1:], panels):
        for layer in LAYERS:
            sel = jets["lead_layer"] == layer
            if sel.any():
                ax.scatter(t0[sel], jets[key][sel], s=45, color=LAYER_COLORS[layer])
        rho, p = spearmanr(t0, jets[key])
        print(f"  t0 vs {key:11s} Spearman rho = {rho:6.2f} (p = {p:.2f})")
        lo, hi = ax.get_ylim()
        ax.set_ylim(lo, hi + 0.15 * (hi - lo))
        ax.text(0.03, 0.96, rf"Spearman $\rho$ = {rho:.2f} ($p$ = {p:.2f})", transform=ax.transAxes, ha="left", va="top", fontsize=13)
        ax.set_xlabel(r"MBD $t_{0}$ [ns]")
        ax.set_ylabel(label)

    fig.text(0.99, 0.995, rf"Au+Au, Iterative, $R = 0.2$, $p_{{T}}^{{\mathrm{{calib}}}} > {pt_min:g}$ GeV, {len(t0)} jets",
             ha="right", va="top", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    save_fig(fig, out_path, save_pdf)
    plt.close(fig)


def plot_eta_phi(jets, pt_min, out_path, save_pdf):
    """Jet positions, coloured by the layer of the leading tower, marker size proportional to pT."""
    from matplotlib.lines import Line2D

    fig, ax = plt.subplots(figsize=(10, 8))
    handles = []
    for layer in LAYERS:
        sel = jets["lead_layer"] == layer
        if not sel.any():
            continue
        ax.scatter(jets["eta"][sel], jets["phi"][sel], s=40 * jets["pt_calib"][sel] / pt_min, color=LAYER_COLORS[layer],
                   alpha=0.8, edgecolors="black", linewidths=0.5)
        handles.append(Line2D([], [], marker="o", linestyle="", markersize=9, color=LAYER_COLORS[layer],
                              markeredgecolor="black", label=f"{LAYER_LABELS[layer]} leading tower ({sel.sum()})"))
    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(0, 2 * np.pi + 2.0)
    ax.set_xlabel(r"Jet $\eta$")
    ax.set_ylabel(r"Jet $\phi$ [rad]")
    ax.legend(handles=handles, loc="upper center", ncol=2, fontsize=LEGEND_SIZE, frameon=False,
              title=r"Marker size $\propto p_{T}^{\mathrm{calib}}$", title_fontsize=LEGEND_SIZE)
    ax.text(1.0, 1.01, rf"Au+Au, Iterative, $R = 0.2$, $p_{{T}}^{{\mathrm{{calib}}}} > {pt_min:g}$ GeV, {len(jets['eta'])} jets",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=TEXT_SIZE + 1)
    tighten(fig, ax)
    save_fig(fig, out_path, save_pdf)
    plt.close(fig)


def plot_vs_pt(jets, key, ylabel, pt_min, out_path, save_pdf, ylim=None):
    fig, ax = plt.subplots(figsize=(10, 8))
    for layer in LAYERS:
        sel = jets["lead_layer"] == layer
        if sel.any():
            ax.scatter(jets["pt_calib"][sel], jets[key][sel], s=45, color=LAYER_COLORS[layer], edgecolor="black",
                       linewidth=0.5, label=f"{LAYER_LABELS[layer]} leading tower ({sel.sum()})")
    ax.legend(loc="lower right", fontsize=LEGEND_SIZE, frameon=False)
    ax.set_xlabel(r"Jet $p_{T}^{\mathrm{calib}}$ [GeV]")
    ax.set_ylabel(ylabel)
    if ylim:
        ax.set_ylim(*ylim)
    else:
        ax.set_ylim(0, ax.get_ylim()[1] * 1.5)
    add_text(ax, len(jets[key]), pt_min)
    tighten(fig, ax)
    save_fig(fig, out_path, save_pdf)
    plt.close(fig)


def plot_timing(jets, f, pt_min, dt_max, out_path, save_pdf):
    """
    Jet time relative to the MBD t0, for the jets above pt_min (by layer of the leading tower) and
    all jets in the selected events (by layer category) as a reference.
    """
    fig, axes = plt.subplots(2, 3, figsize=(24, 13))
    dt, dt_all = jets["dt_t0"], f["dt_t0"]
    lim = max(10, 10 * np.ceil(np.nanmax(np.abs(np.concatenate([dt, dt_all]))) / 10))
    edges = np.linspace(-lim, lim, int(lim) + 1)  # 2 ns bins

    def window(ax, vertical):
        span = ax.axvspan if vertical else ax.axhspan
        span(-dt_max, dt_max, color="gray", alpha=0.2, label=rf"$|t - t_{{0}}| < {dt_max:g}$ ns")

    def in_time(v):
        v = v[np.isfinite(v)]
        return f"{np.sum(np.abs(v) < dt_max)}/{len(v)}"

    def clipped(v):
        """Finite values, with the overflow in the edge bins."""
        return np.clip(v[np.isfinite(v)], -lim, lim)

    ax = axes[0, 0]
    ax.hist(clipped(dt_all), bins=edges, histtype="step", linewidth=2, color="black",
            label=rf"All jets, $p_{{T}} > {FEATURE_PT_MIN:g}$ GeV (in time {in_time(dt_all)})")
    for layer in LAYERS:
        sel = jets["lead_layer"] == layer
        if sel.any():
            ax.hist(clipped(dt[sel]), bins=edges, histtype="step", linewidth=2, color=LAYER_COLORS[layer],
                    label=f"{LAYER_LABELS[layer]} leading tower (in time {in_time(dt[sel])})")
    window(ax, vertical=True)
    ax.set_xlabel(r"$t_{\mathrm{jet}} - t_{0}$ [ns]")
    ax.set_ylabel("Jets")
    ax.set_ylim(0, ax.get_ylim()[1] * 1.6)
    ax.legend(loc="upper left", fontsize=12, frameon=False)

    # calo time vs MBD time: in-time jets follow the diagonal
    ax = axes[0, 1]
    ax.scatter(f["t0"], f["t"], s=12, color="#bbbbbb", label=rf"All jets, $p_{{T}} > {FEATURE_PT_MIN:g}$ GeV")
    for layer in LAYERS:
        sel = jets["lead_layer"] == layer
        if sel.any():
            ax.scatter(jets["t0"][sel], jets["t"][sel], s=45, color=LAYER_COLORS[layer], edgecolor="black", linewidth=0.5,
                       label=f"{LAYER_LABELS[layer]} leading tower ({sel.sum()})")
    x = np.array(ax.get_xlim())
    ax.fill_between(x, x - dt_max, x + dt_max, color="gray", alpha=0.2, label=rf"$|t - t_{{0}}| < {dt_max:g}$ ns")
    ax.set_xlim(*x)
    ax.set_ylim(-lim, 1.6 * lim)
    ax.set_xlabel(r"MBD $t_{0}$ [ns]")
    ax.set_ylabel(r"$t_{\mathrm{jet}}$ [ns]")
    ax.legend(loc="upper left", fontsize=12, frameon=False)

    # time of each calo layer: which layer drives an out-of-time jet
    ax = axes[0, 2]
    for layer in LAYERS:
        v = jets[f"t_{layer}"] - jets["t0"]
        ax.hist(clipped(v), bins=edges, histtype="step", linewidth=2, color=LAYER_COLORS[layer],
                label=f"{LAYER_LABELS[layer]} towers (in time {in_time(v)})")
    window(ax, vertical=True)
    ax.set_xlabel(r"$t_{\mathrm{layer}} - t_{0}$ [ns]")
    ax.set_ylabel("Jets")
    ax.set_ylim(0, ax.get_ylim()[1] * 1.6)
    ax.legend(loc="upper left", fontsize=12, frameon=False)

    ax = axes[1, 0]
    scatter_by_category(ax, f, "pt_calib", "dt_t0")
    window(ax, vertical=False)
    ax.axvline(pt_min, color="gray", linestyle="--", linewidth=1)
    ax.set_xscale("log")
    ax.set_xlim(FEATURE_PT_MIN * 0.9, f["pt_calib"].max() * 1.3)
    ax.set_ylim(-lim, 1.6 * lim)
    ax.set_xlabel(r"Jet $p_{T}^{\mathrm{calib}}$ [GeV]")
    ax.set_ylabel(r"$t_{\mathrm{jet}} - t_{0}$ [ns]")
    ax.legend(loc="upper left", ncol=2, fontsize=12, frameon=False, title="Layer fraction")

    for ax, (key, label) in zip(axes[1, 1:], [("z_lead", r"$E_{\mathrm{lead}} / E_{\mathrm{jet}}$"),
                                              ("centrality", "Centrality [%]")]):
        for layer in LAYERS:
            sel = jets["lead_layer"] == layer
            if sel.any():
                ax.scatter(jets[key][sel], dt[sel], s=45, color=LAYER_COLORS[layer], edgecolor="black", linewidth=0.5,
                           label=f"{LAYER_LABELS[layer]} leading tower ({sel.sum()})")
        window(ax, vertical=False)
        ax.set_ylim(-lim, 1.6 * lim)
        ax.set_xlabel(label)
        ax.set_ylabel(r"$t_{\mathrm{jet}} - t_{0}$ [ns]")
        ax.legend(loc="upper left", fontsize=12, frameon=False)

    fig.text(0.99, 0.995, rf"Au+Au, Iterative, $R = 0.2$, $p_{{T}}^{{\mathrm{{calib}}}} > {pt_min:g}$ GeV, {len(dt)} jets",
             ha="right", va="top", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    save_fig(fig, out_path, save_pdf)
    plt.close(fig)


def plot_dt_vs_pt(jets, pt_min, dt_max, out_path, save_pdf):
    """t - t0 vs calibrated pT for the jets above pt_min, by layer of the leading tower."""
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.axhspan(-dt_max, dt_max, color="gray", alpha=0.2, label=rf"$|t - t_{{0}}| < {dt_max:g}$ ns")
    for layer in LAYERS:
        sel = jets["lead_layer"] == layer
        if sel.any():
            ax.scatter(jets["pt_calib"][sel], jets["dt_t0"][sel], s=45, color=LAYER_COLORS[layer], edgecolor="black",
                       linewidth=0.5, label=f"{LAYER_LABELS[layer]} leading tower ({sel.sum()})")
    lim = max(10, 10 * np.ceil(np.nanmax(np.abs(jets["dt_t0"])) / 10))
    ax.set_ylim(-lim, 1.5 * lim)
    ax.set_xlabel(r"Jet $p_{T}^{\mathrm{calib}}$ [GeV]")
    ax.set_ylabel(r"$t_{\mathrm{jet}} - t_{0}$ [ns]")
    ax.legend(loc="upper left", fontsize=LEGEND_SIZE, frameon=False)
    add_text(ax, len(jets["dt_t0"]), pt_min)
    tighten(fig, ax)
    fig.subplots_adjust(left=0.13)  # room for the subscripted y label
    save_fig(fig, out_path, save_pdf)
    plt.close(fig)


def plot_dt_1d(jets, f, pt_min, dt_max, out_path, save_pdf, low_range=(10.0, 20.0)):
    """1D t - t0 (counts) for the jets above pt_min and for the jets in low_range (from all jets in the selected events)."""
    low = (f["pt_calib"] >= low_range[0]) & (f["pt_calib"] < low_range[1])
    samples = [(jets["dt_t0"], rf"$p_{{T}}^{{\mathrm{{calib}}}} > {pt_min:g}$ GeV", "#e41a1c"),
               (f["dt_t0"][low], rf"${low_range[0]:g} < p_{{T}}^{{\mathrm{{calib}}}} < {low_range[1]:g}$ GeV", "#377eb8")]

    lim = 80
    edges = np.linspace(-lim, lim, 2 * lim + 1)  # 2 ns bins, overflow in the edge bins
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.axvspan(-dt_max, dt_max, color="gray", alpha=0.2, label=rf"$|t - t_{{0}}| < {dt_max:g}$ ns")
    for dt, label, color in samples:
        dt = dt[np.isfinite(dt)]
        n_in = int(np.sum(np.abs(dt) < dt_max))
        ax.hist(np.clip(dt, -lim, lim), bins=edges, histtype="step", linewidth=2, color=color,
                label=f"{label}, {len(dt)} jets, in time {n_in}/{len(dt)} ({n_in / len(dt):.0%})")
    ax.set_xlim(-lim, lim)
    ax.set_xlabel(r"$t_{\mathrm{jet}} - t_{0}$ [ns]")
    ax.set_ylabel("Jets / 2 ns")
    ax.set_ylim(0, ax.get_ylim()[1] * 1.5)
    ax.legend(loc="upper left", fontsize=LEGEND_SIZE - 4, frameon=False)
    ax.text(1.0, 1.01, r"Au+Au, Iterative, $R = 0.2$", transform=ax.transAxes, ha="right", va="bottom", fontsize=TEXT_SIZE + 1)
    ax.text(0.97, 0.96, TEXT_INFO, transform=ax.transAxes, ha="right", va="top", fontsize=TEXT_SIZE)
    tighten(fig, ax)
    fig.subplots_adjust(left=0.13)  # room for the subscripted x/y labels
    save_fig(fig, out_path, save_pdf)
    plt.close(fig)


JET_CSV_COLS = ["run", "event", "centrality", "t0", "pt_calib", "pt_raw", "e", "eta", "phi", "n_const", "n_pos",
                "n_emcal", "n_ihcal", "n_ohcal", "e_lead", "z_lead", "z_lead_pt", "lead_layer",
                "frac_emcal", "frac_ihcal", "frac_ohcal", "t", "dt_t0", "t_emcal", "t_ihcal", "t_ohcal"]


def write_csv(jets, out_path, cols=JET_CSV_COLS):
    order = np.argsort(-jets["pt_calib"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(cols)
        for i in order:
            writer.writerow([f"{jets[c][i]:.4f}" if isinstance(jets[c][i], np.floating) else jets[c][i] for c in cols])
    print(f"Saved: {out_path}")


class TowerGrid:
    """Tower energies on an (ieta, iphi) grid, with the to_numpy() interface plot_calo_2d expects."""

    def __init__(self, values, xedges=None, yedges=None):
        self.values = values
        self.xedges = np.arange(values.shape[0] + 1) if xedges is None else xedges
        self.yedges = np.arange(values.shape[1] + 1) if yedges is None else yedges

    def to_numpy(self):
        return self.values, self.xedges, self.yedges

    def crop(self, eta_lo, eta_hi, phi_lo, phi_hi):
        """Sub-grid over [eta_lo, eta_hi) x [phi_lo, phi_hi); phi wraps around, eta is clipped."""
        eta_lo, eta_hi = max(eta_lo, 0), min(eta_hi, self.values.shape[0])
        phi = np.arange(phi_lo, phi_hi)
        return TowerGrid(self.values[eta_lo:eta_hi][:, phi % self.values.shape[1]],
                         np.arange(eta_lo, eta_hi + 1), np.arange(phi_lo, phi_hi + 1))


def tower_grid(index, energy, lookup, shape):
    """Fills tower energies into an (ieta, iphi) grid, lookup maps tower index -> (ieta, iphi)."""
    values = np.zeros(shape)
    coords = lookup[np.asarray(index, dtype=int)]
    np.add.at(values, (coords[:, 0], coords[:, 1]), np.asarray(energy, dtype=float))
    return values


@functools.cache
def tower_lookup(det):
    """Tower index -> (ieta, iphi) for the HCal grid (also retowered EMCal) or the base EMCal grid."""
    if det == "hcal":
        return np.array([get_hcal_ieta_iphi(i) for i in range(np.prod(HCAL_SHAPE))], dtype=int)
    return np.array([get_emcal_ieta_iphi(i) for i in range(np.prod(EMCAL_SHAPE))], dtype=int)


def display_payloads(events, jet_mask):
    """Per event: tower grids and the R = 0.2 jet footprints (picklable, for the plotting workers)."""
    lookup = {det: tower_lookup(det) for det in ["hcal", "emcal"]}
    payloads = []
    for iev in range(len(events)):
        ev = events[iev]
        grids = {}
        for name, branch in DISPLAY_TOWERS.items():
            det, shape = ("emcal", EMCAL_SHAPE) if name == "base" else ("hcal", HCAL_SHAPE)
            grids[name] = tower_grid(ak.to_numpy(ev[f"{branch}_index"]), ak.to_numpy(ev[f"{branch}_energy"]), lookup[det], shape)

        jets = []
        for ijet, selected in enumerate(ak.to_list(jet_mask[iev])):
            masks, frac = {}, {}
            lead_e, lead_pos = -np.inf, None
            e_jet = float(ev[f"e_{COLL}"][ijet])
            for layer in LAYERS:
                index = ak.to_numpy(ev[f"{layer}_tower_index_{COLL}"][ijet]).astype(int)
                energy = ak.to_numpy(ev[f"{layer}_tower_energy_{COLL}"][ijet])
                masks[layer] = tower_grid(index, np.ones(len(index)), lookup["hcal"], HCAL_SHAPE) > 0
                frac[layer] = float(energy.sum()) / e_jet
                if len(energy) and energy.max() > lead_e:
                    lead_e, lead_pos = energy.max(), tuple(lookup["hcal"][index[np.argmax(energy)]])
            jets.append({"selected": selected, "pt": float(ev[f"pt_calib_{COLL}"][ijet]),
                         "z_lead": lead_e / e_jet, "frac": frac, "lead": lead_pos, "masks": masks,
                         "dt_t0": float(ev[f"t_{COLL}"][ijet]) - float(ev.t0)})

        lead_jet = max((j for j in jets if j["selected"]), key=lambda j: j["pt"])
        payloads.append({"run": int(ev.run), "event": int(ev.event), "grids": grids, "jets": jets, "zoom_center": lead_jet["lead"]})
    return payloads


def make_zoom_plot(run, event, panels, output_path, logz=False):
    """1x4 of square zoomed panels (make_1x4_calo_plot is sized for the full, tall detector)."""
    hep.style.use("ATLAS")
    fig, axes = plt.subplots(1, 4, figsize=(24, 5.2))
    for ax, n_phi, (grid, title, overlays) in zip(axes, [EMCAL_SHAPE[1]] + [HCAL_SHAPE[1]] * 3, panels):
        plot_calo_2d(ax, grid, title=title, logz=logz, overlays=overlays, legend_fontsize=12)
        # the zoom window wraps in phi past the grid edge, so label the phi axis modulo the number of phi bins
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _, n=n_phi: f"{int(round(v)) % n}"))
    fig.text(0.94, 0.97, rf"Run: {run}, Event: {event}", ha="right", va="top", fontsize=15, weight="bold")
    fig.subplots_adjust(left=0.04, right=0.94, top=0.86, bottom=0.13, wspace=0.5)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def jet_label(jet, layer):
    """Legend entry: jet pT, leading tower fraction and the energy fraction in this panel's layer."""
    if not jet["selected"]:
        return "Other jets"
    frac = f"{jet['frac'][layer]:.2f}".replace("-0.00", "0.00")  # tiny negative fractions from subtracted towers
    return (rf"{jet['pt']:.0f} GeV, $z_{{\mathrm{{lead}}}}$ = {jet['z_lead']:.2f}" "\n"
            rf"$E_{{\mathrm{{{LAYER_LABELS[layer]}}}}}/E_{{\mathrm{{jet}}}}$ = {frac}, $t - t_{{0}}$ = {jet['dt_t0']:.0f} ns")


def draw_event_display(payload, out_dir, logz):
    run, event = payload["run"], payload["event"]
    grids = {k: TowerGrid(v) for k, v in payload["grids"].items()}
    stem = f"run_{run}_event_{event}"

    # one subfolder per plot type, so the same plot of every event can be paged through
    def plot_path(kind):
        kind_dir = out_dir / kind
        kind_dir.mkdir(parents=True, exist_ok=True)
        return kind_dir / f"{stem}_{kind}.png"

    make_1x4_calo_plot(run, event, grids["base"], grids["retower"], grids["ihcal"], grids["ohcal"],
                       plot_path("unsub_1x4"), logz=logz)

    # selected jets in colour with pT and leading tower fraction, the other R = 0.2 jets in grey
    xedges, yedges = np.arange(HCAL_SHAPE[0] + 1), np.arange(HCAL_SHAPE[1] + 1)
    overlays = {layer: [] for layer in LAYERS}
    n_selected = 0
    for jet in payload["jets"]:
        if jet["selected"]:
            color, ls, lw = JET_COLORS[n_selected % len(JET_COLORS)], "solid", 2.0
            n_selected += 1
        else:
            color, ls, lw = "#999999", "dashed", 1.2
        for layer in LAYERS:
            segs = get_cluster_boundary_segments(jet["masks"][layer], xedges, yedges)
            if segs:
                label = jet_label(jet, layer)
                # one legend entry per panel for the unselected jets
                lbl = "_nolegend_" if not jet["selected"] and any(o[4] == label for o in overlays[layer]) else label
                overlays[layer].append((segs, color, ls, lw, lbl))
    make_1x3_calo_plot(run, event, grids["iter_emcal"], grids["iter_ihcal"], grids["iter_ohcal"],
                       plot_path("iter_r02_1x3"), sub_type="Iter", logz=logz,
                       overlays_emcal=overlays["emcal"], overlays_ihcal=overlays["ihcal"], overlays_ohcal=overlays["ohcal"])

    ceta, cphi = payload["zoom_center"]
    eta_lo, eta_hi, phi_lo, phi_hi = ceta - ZOOM_HALF, ceta + ZOOM_HALF + 1, cphi - ZOOM_HALF, cphi + ZOOM_HALF + 1
    base = grids["base"].crop(4 * eta_lo, 4 * eta_hi, 4 * phi_lo + EMCAL_PHI_SHIFT, 4 * phi_hi + EMCAL_PHI_SHIFT)
    panels = [(base, "EMCal Base", None)]
    for key, layer, title in [("retower", "emcal", "EMCal Retower"), ("ihcal", "ihcal", "IHCal"), ("ohcal", "ohcal", "OHCal")]:
        zoom = grids[key].crop(eta_lo, eta_hi, phi_lo, phi_hi)
        overlays = []
        n_selected = 0
        for jet in payload["jets"]:
            if not jet["selected"]:
                continue
            mask = TowerGrid(jet["masks"][layer]).crop(eta_lo, eta_hi, phi_lo, phi_hi)
            segs = get_cluster_boundary_segments(mask.values, mask.xedges, mask.yedges)
            if segs:
                overlays.append((segs, JET_COLORS[n_selected % len(JET_COLORS)], "solid", 2.0, jet_label(jet, layer)))
            n_selected += 1
        panels.append((zoom, title, overlays))
    make_zoom_plot(run, event, panels, plot_path("zoom_1x4"), logz=logz)
    return out_dir


def delta_phi(a, b):
    return np.abs((a - b + np.pi) % (2 * np.pi) - np.pi)


def jet_features(events, jet_mask):
    """
    Per jet (iter R = 0.2), from the iter-subtracted towers the jets are built from:
      frac_<layer> -- layer energy fraction
      swiss        -- 1 - (sum of the 4 direct neighbours) / (leading tower), in the leading tower's layer
      r03_r02      -- raw pT of the matched (dR < 0.2) R = 0.3 jet over the raw R = 0.2 pT, NaN if unmatched
      recoil_pt    -- highest calibrated pT of another R = 0.2 jet with dphi > pi/2, 0 if none (tree keeps pT >= 10)
      t, dt_t0     -- jet time and jet time relative to the MBD t0
    """
    lookup = tower_lookup("hcal")
    neta, nphi = HCAL_SHAPE
    rows = []
    for iev in range(len(events)):
        ev = events[iev]
        grids = {layer: tower_grid(ak.to_numpy(ev[f"iter_{layer}_tower_index"]), ak.to_numpy(ev[f"iter_{layer}_tower_energy"]), lookup, HCAL_SHAPE)
                 for layer in LAYERS}
        pt, pt_raw, e, eta, phi, t = (ak.to_numpy(ev[f"{v}_{COLL}"]) for v in ["pt_calib", "pt", "e", "eta", "phi", "t"])
        pt03, eta03, phi03 = (ak.to_numpy(ev[f"{v}_iter_r03"]) for v in ["pt", "eta", "phi"])

        for ij in np.flatnonzero(ak.to_numpy(jet_mask[iev])):
            row = {"run": int(ev.run), "event": int(ev.event), "centrality": float(ev.centrality),
                   "pt_calib": pt[ij], "pt_raw": pt_raw[ij], "eta": eta[ij], "phi": phi[ij],
                   "t0": float(ev.t0), "t": t[ij], "dt_t0": t[ij] - float(ev.t0)}
            lead_e, lead_layer, lead_pos = -np.inf, None, None
            for layer in LAYERS:
                index = ak.to_numpy(ev[f"{layer}_tower_index_{COLL}"][ij]).astype(int)
                energy = ak.to_numpy(ev[f"{layer}_tower_energy_{COLL}"][ij])
                row[f"frac_{layer}"] = energy.sum() / e[ij]
                if len(energy) and energy.max() > lead_e:
                    lead_e, lead_layer, lead_pos = energy.max(), layer, lookup[index[np.argmax(energy)]]
            row["z_lead"] = lead_e / e[ij]
            row["lead_layer"] = lead_layer

            ieta, iphi = lead_pos
            grid = grids[lead_layer]
            neighbours = grid[ieta, (iphi - 1) % nphi] + grid[ieta, (iphi + 1) % nphi]
            neighbours += sum(grid[ie, iphi] for ie in (ieta - 1, ieta + 1) if 0 <= ie < neta)
            row["swiss"] = 1 - neighbours / lead_e

            dr = np.hypot(eta03 - eta[ij], delta_phi(phi03, phi[ij]))
            row["r03_r02"] = pt03[np.argmin(dr)] / pt_raw[ij] if len(dr) and dr.min() < MATCH_DR else np.nan

            away = (np.arange(len(pt)) != ij) & (e > 0) & (delta_phi(phi, phi[ij]) > np.pi / 2)
            row["recoil_pt"] = pt[away].max() if away.any() else 0.0
            row["recoil_frac"] = row["recoil_pt"] / pt[ij]
            rows.append(row)

    features = {k: np.array([r[k] for r in rows]) for k in rows[0]}
    frac_sum = features["frac_emcal"] + features["frac_ihcal"] + features["frac_ohcal"]
    dev = np.abs(frac_sum - 1)
    print(f"Layer fractions sum to 1 for {np.sum(dev <= 0.01)}/{len(dev)} jets (max deviation {dev.max():.2e})")
    category = np.full(len(rows), "mixed", dtype=object)
    for layer in LAYERS:
        category[features[f"frac_{layer}"] > SINGLE_LAYER_FRAC] = layer
    features["category"] = category
    return features


def scatter_by_category(ax, f, x, y, s=25):
    colors = {**LAYER_COLORS, "mixed": "#377eb8"}
    for cat, label in CATEGORY_LABELS.items():
        sel = f["category"] == cat
        if sel.any():
            ax.scatter(f[x][sel], f[y][sel], s=s, color=colors[cat], alpha=0.75, label=f"{label} ({sel.sum()})")


def plot_features_vs_pt(f, pt_min, out_path, save_pdf):
    panels = [("frac_ihcal", r"$E_{\mathrm{IHCal}} / E_{\mathrm{jet}}$"),
              ("swiss", r"$1 - E_{\mathrm{4\ neighbours}} / E_{\mathrm{lead}}$"),
              ("r03_r02", r"$p_{T,\ R=0.3}^{\mathrm{raw}} / p_{T,\ R=0.2}^{\mathrm{raw}}$"),
              ("recoil_frac", r"Away-side jet $p_{T}$ / jet $p_{T}$")]
    fig, axes = plt.subplots(2, 2, figsize=(18, 13))
    for ax, (key, label) in zip(axes.flat, panels):
        scatter_by_category(ax, f, "pt_calib", key)
        ax.axvline(pt_min, color="gray", linestyle="--", linewidth=1)
        ax.set_xscale("log")
        ax.set_xlim(FEATURE_PT_MIN * 0.9, f["pt_calib"].max() * 1.3)
        lo, hi = np.nanmin(f[key]), np.nanmax(f[key])
        pad = 0.05 * (hi - lo) if hi > lo else 0.1
        ax.set_ylim(lo - pad, hi + 6 * pad)
        ax.set_xlabel(r"Jet $p_{T}^{\mathrm{calib}}$ [GeV]")
        ax.set_ylabel(label)
    axes[0, 0].legend(loc="upper right", ncol=2, fontsize=12, frameon=False, title="Layer fraction")
    fig.text(0.99, 0.995, rf"Au+Au, Iterative, $R = 0.2$, all jets in the {len(np.unique(f['event']))} high-$p_{{T}}$ events, {len(f['pt_calib'])} jets",
             ha="right", va="top", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    save_fig(fig, out_path, save_pdf)
    plt.close(fig)


def plot_layer_plane(f, pt_min, out_path, save_pdf):
    """EMCal vs OHCal energy fraction, IHCal fraction = 1 - EMCal - OHCal."""
    from matplotlib.colors import LogNorm

    fig, ax = plt.subplots(figsize=(10, 8.5))
    low = f["pt_calib"] < pt_min
    sc = ax.scatter(f["frac_emcal"][low], f["frac_ohcal"][low], c=f["pt_calib"][low], cmap="viridis",
                    norm=LogNorm(FEATURE_PT_MIN, pt_min), s=22, alpha=0.85)
    ax.scatter(f["frac_emcal"][~low], f["frac_ohcal"][~low], s=90, facecolors="none", edgecolors="#e41a1c", linewidths=1.8,
               label=rf"$p_{{T}}^{{\mathrm{{calib}}}} > {pt_min:g}$ GeV ({np.sum(~low)})")
    ax.plot([0, 1], [1, 0], color="gray", linestyle="--", linewidth=1, label=r"$E_{\mathrm{IHCal}} = 0$")
    fig.colorbar(sc, ax=ax, label=rf"Jet $p_{{T}}^{{\mathrm{{calib}}}}$ [GeV] (< {pt_min:g})")
    ax.set_xlim(-0.15, 1.15)
    ax.set_ylim(-0.15, 1.45)
    ax.set_xlabel(r"$E_{\mathrm{EMCal}} / E_{\mathrm{jet}}$")
    ax.set_ylabel(r"$E_{\mathrm{OHCal}} / E_{\mathrm{jet}}$")
    ax.legend(loc="upper right", fontsize=12, frameon=False)
    ax.text(1.0, 1.01, rf"Au+Au, Iterative, $R = 0.2$, {len(f['pt_calib'])} jets", transform=ax.transAxes, ha="right", va="bottom", fontsize=13)
    save_fig(fig, out_path, save_pdf)
    plt.close(fig)


def plot_layer_plane_high_pt(f, pt_min, out_path, save_pdf):
    """EMCal vs OHCal energy fraction for the jets above pt_min only, coloured by leading-tower layer."""
    sel = f["pt_calib"] >= pt_min
    fig, ax = plt.subplots(figsize=(10, 8))
    for layer in LAYERS:
        s = sel & (f["lead_layer"] == layer)
        if s.any():
            ax.scatter(f["frac_emcal"][s], f["frac_ohcal"][s], s=60, color=LAYER_COLORS[layer], edgecolor="black", linewidth=0.5,
                       label=f"{LAYER_LABELS[layer]} leading tower ({s.sum()})")
    ax.plot([0, 1], [1, 0], color="gray", linestyle="--", linewidth=1, label=r"$E_{\mathrm{IHCal}} = 0$")
    for i, frac in enumerate((0.1, 0.9)):
        ax.axvline(frac, color="black", linestyle=":", linewidth=1, label="10% / 90%" if i == 0 else None)
        ax.axhline(frac, color="black", linestyle=":", linewidth=1)
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.1, 1.15)
    ax.set_xlabel(r"$E_{\mathrm{EMCal}} / E_{\mathrm{jet}}$")
    ax.set_ylabel(r"$E_{\mathrm{OHCal}} / E_{\mathrm{jet}}$")
    ax.legend(loc="upper right", fontsize=LEGEND_SIZE, frameon=False)
    ax.text(1.0, 1.01, rf"Au+Au, Iterative, $R = 0.2$, $p_{{T}}^{{\mathrm{{calib}}}} > {pt_min:g}$ GeV, {sel.sum()} jets",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=TEXT_SIZE + 1)
    tighten(fig, ax)
    save_fig(fig, out_path, save_pdf)
    plt.close(fig)


def print_feature_table(f, pt_min, dt_max):
    """Fraction of jets per pT bin that look like the high-pT jets (provisional thresholds)."""
    edges = [FEATURE_PT_MIN, 20, 40, pt_min, np.inf]
    matched = np.isfinite(f["r03_r02"])
    dt_name = f"|t-t0|<{dt_max:g}"
    print(f"\n{'pT_calib [GeV]':16s} {'Jets':>5s} {'single layer':>13s} {'IHCal<0.05':>11s} {'swiss>0.9':>10s} "
          f"{'R03/R02<1.05':>13s} {'no recoil':>10s} {dt_name:>10s}")
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (f["pt_calib"] >= lo) & (f["pt_calib"] < hi)
        if not sel.any():
            continue
        cols = [np.mean(f["category"][sel] != "mixed"), np.mean(f["frac_ihcal"][sel] < 0.05), np.mean(f["swiss"][sel] > 0.9),
                np.mean(f["r03_r02"][sel & matched] < 1.05) if (sel & matched).any() else np.nan, np.mean(f["recoil_pt"][sel] == 0),
                np.mean(np.abs(f["dt_t0"][sel]) < dt_max)]
        name = f"{lo:g}-{hi:g}" if np.isfinite(hi) else f">= {lo:g}"
        print(f"{name:16s} {sel.sum():5d} " + " ".join(f"{c:{w}.1%}" for c, w in zip(cols, [13, 11, 10, 13, 10, 10])))
    print(f"(single layer: one layer > {SINGLE_LAYER_FRAC:.0%} of the jet energy; R03/R02 over the {matched.sum()} matched jets)")

    # in-time fraction of the jets above pt_min by layer category: are the single-layer jets out of time?
    sel = f["pt_calib"] >= pt_min
    print(f"\npT_calib >= {pt_min:g} GeV, {dt_name} ns by layer category:")
    for cat, label in CATEGORY_LABELS.items():
        s = sel & (f["category"] == cat)
        if s.any():
            print(f"  {label:12s} {np.sum(np.abs(f['dt_t0'][s]) < dt_max):3d}/{s.sum():3d}")


def main():
    parser = argparse.ArgumentParser(description="Constituents of high-pT iter R = 0.2 jets from the detailed Fun4All_BkgSub trees.")
    parser.add_argument("-i", "--input", nargs="+", default=[TREE_LIST], help="Tree files, directories (searched for tree-*.root) or .list files.")
    parser.add_argument("--calo-mbd-list", default=CALO_MBD_LIST, help="List of per-run event QA files with h2TotalCaloE_MBD.")
    parser.add_argument("-o", "--output", default="plots/Jet-Ana/high_pt", help="Output directory.")
    parser.add_argument("--pt-min", type=float, default=80.0, help="Calibrated jet pT min [GeV].")
    parser.add_argument("--sigma-cut", type=float, default=3.5, help="Calo-MBD consistency cut in sigma.")
    parser.add_argument("--dt-max", type=float, default=3.0, help="In-time window |t_jet - t0| [ns] (shown, not cut on).")
    parser.add_argument("--pdf", action="store_true", help="Also save PDFs.")
    parser.add_argument("--event-displays", action="store_true", help="Tower displays for each event with a selected jet.")
    parser.add_argument("--logz", action="store_true", help="Logarithmic energy scale for the event displays.")
    parser.add_argument("-j", "--workers", type=int, default=0, help="Event display workers (default: auto).")
    args = parser.parse_args()

    files = find_trees(args.input)
    extra = FEATURE_BRANCHES + (DISPLAY_BRANCHES if args.event_displays else [])
    events = load_trees(files, list(dict.fromkeys(extra)))
    if events is None:
        print("No events in the trees.")
        return
    print(f"Read {len(events)} events from {len(files)} files")
    if ak.any(abs(events.zvtx) >= ZVTX_MAX):
        raise ValueError(f"events with |z| >= {ZVTX_MAX} cm; the trees should be |z| < 10 upstream")

    profiles = load_calo_mbd_profiles(args.calo_mbd_list, np.unique(ak.to_numpy(events.run)))

    # event cut flow, with the number of jets above the pT cut in the surviving events
    # (after the jet cuts, the event column counts events with at least one selected jet)
    pt_calib = events[f"pt_calib_{COLL}"]
    steps = [
        ("All events", np.ones(len(events), dtype=bool)),
        (f"Centrality <= {CENT_HIGH}%", ak.to_numpy(events.centrality <= CENT_HIGH)),
        (f"Calo-MBD {args.sigma_cut:g} sigma", pass_calo_mbd(events, profiles, args.sigma_cut)),
        ("No iter flow failure", ak.to_numpy(~events.is_flow_failure_iter)),
        (f"|calo v2| < {CALO_V2_MAX}", ak.to_numpy(abs(events.calo_v2_iter) < CALO_V2_MAX)),
    ]
    print(f"\n{'Cut':32s} {'Events':>7s} {'Jets':>6s}")
    mask_evt = np.ones(len(events), dtype=bool)
    for name, mask in steps:
        mask_evt &= mask
        print(f"{name:32s} {mask_evt.sum():7d} {ak.sum(pt_calib[mask_evt] > args.pt_min):6d}")

    events = events[mask_evt]

    # features of every jet in the selected events, as a reference for the high-pT ones
    base_mask = (events[f"pt_calib_{COLL}"] >= FEATURE_PT_MIN) & (abs(events[f"eta_{COLL}"]) < ETA_MAX) & (events[f"e_{COLL}"] > 0)
    features = jet_features(events, base_mask)

    jet_mask = events[f"pt_calib_{COLL}"] > args.pt_min
    for name, mask in [(f"|eta| < {ETA_MAX}", abs(events[f"eta_{COLL}"]) < ETA_MAX), ("E > 0", events[f"e_{COLL}"] > 0)]:
        jet_mask = jet_mask & mask
        print(f"{name:32s} {ak.sum(ak.any(jet_mask, axis=1)):7d} {ak.sum(jet_mask):6d}")

    # displays show every R = 0.2 jet in the event, so build them before the jets are masked
    payloads = []
    if args.event_displays:
        has_jet = ak.to_numpy(ak.any(jet_mask, axis=1))
        payloads = display_payloads(events[has_jet], jet_mask[has_jet])

    # keep the jet-level jagged arrays aligned with their tower lists
    for field in JET_BRANCHES + TOWER_BRANCHES:
        events[field] = events[field][jet_mask]
    jets = jet_table(events)
    if len(jets["pt_calib"]) == 0:
        print("No jets pass the selection.")
        return

    # constituent energies must add up to the jet energy
    rel = np.abs(jets["e_sum"] / jets["e"] - 1)
    if np.any(rel > 0.01):
        print(f"WARNING: {np.sum(rel > 0.01)} jets with sum of constituent E off by > 1% (max {rel.max():.2%})")
    else:
        print(f"\nConstituent E sum matches jet E for all jets (max deviation {rel.max():.2e})")

    # the jet time is the energy-weighted mean of the layer times, so it must lie between them
    layer_t = np.stack([jets[f"t_{layer}"] for layer in LAYERS], axis=1)
    has_t = np.isfinite(layer_t)
    lo = np.where(has_t, layer_t, np.inf).min(axis=1)
    hi = np.where(has_t, layer_t, -np.inf).max(axis=1)
    inside = (jets["t"] >= lo - 1e-3) & (jets["t"] <= hi + 1e-3)
    print(f"Jet time between its layer times for {inside.sum()}/{len(inside)} jets "
          f"(no timed tower: {np.sum(~has_t.any(axis=1))}, t0 NaN: {np.sum(~np.isfinite(jets['t0']))})")

    print(f"\n=== {len(jets['pt_calib'])} jets, pT_calib {jets['pt_calib'].min():.1f}-{jets['pt_calib'].max():.1f} GeV ===")
    for key, label in [("n_const", "N_const"), ("n_pos", "N_const (E > 0)"), ("n_emcal", "N EMCal"),
                       ("n_ihcal", "N IHCal"), ("n_ohcal", "N OHCal"), ("z_lead", "E_lead / E_jet"),
                       ("z_lead_pt", "pT_lead / pT_jet,raw"), ("frac_emcal", "EMCal E fraction"),
                       ("frac_ihcal", "IHCal E fraction"), ("frac_ohcal", "OHCal E fraction")]:
        v = jets[key]
        print(f"  {label:22s} mean {v.mean():7.3f}  median {np.median(v):7.3f}  min {v.min():7.3f}  max {v.max():7.3f}")
    layers, counts = np.unique(jets["lead_layer"], return_counts=True)
    print("  leading tower layer:   " + ", ".join(f"{LAYER_LABELS[l]} {c}" for l, c in zip(layers, counts)))
    for z in [0.3, 0.5, 0.7]:
        print(f"  E_lead / E_jet > {z}: {np.mean(jets['z_lead'] > z):6.1%}")
    # timing can be NaN (no tower above the timing threshold in a layer, or no t0)
    for key, label in [("dt_t0", "t - t0 [ns]"), ("t_emcal", "t EMCal [ns]"), ("t_ihcal", "t IHCal [ns]"), ("t_ohcal", "t OHCal [ns]")]:
        v = jets[key][np.isfinite(jets[key])]
        if len(v):
            print(f"  {label:22s} median {np.median(v):7.2f}  min {v.min():7.2f}  max {v.max():7.2f}  ({len(v)} jets)")
    print(f"  |t - t0| < {args.dt_max:g} ns: {np.mean(np.abs(jets['dt_t0']) < args.dt_max):6.1%}")

    hep.style.use("ATLAS")
    out_dir = Path(args.output)
    tag = f"{COLL}_pt{args.pt_min:g}"
    plot_n_const(jets, args.pt_min, out_dir / f"n_const_{tag}.png", args.pdf)
    plot_z_lead(jets, args.pt_min, out_dir / f"z_lead_{tag}.png", args.pdf)
    plot_layer_frac(jets, args.pt_min, out_dir / f"layer_frac_{tag}.png", args.pdf)
    plot_eta_phi(jets, args.pt_min, out_dir / f"eta_phi_{tag}.png", args.pdf)
    print()
    plot_t0_correlations(jets, args.pt_min, out_dir / f"t0_correlations_{tag}.png", args.pdf)
    plot_vs_pt(jets, "z_lead", r"$E_{\mathrm{lead}} / E_{\mathrm{jet}}$", args.pt_min, out_dir / f"z_lead_vs_pt_{tag}.png", args.pdf, ylim=(0, 1.2))
    plot_vs_pt(jets, "n_const", r"$N_{\mathrm{const}}$", args.pt_min, out_dir / f"n_const_vs_pt_{tag}.png", args.pdf)
    write_csv(jets, out_dir / f"jets_{tag}.csv")

    print_feature_table(features, args.pt_min, args.dt_max)
    plot_timing(jets, features, args.pt_min, args.dt_max, out_dir / f"timing_{tag}.png", args.pdf)
    plot_dt_vs_pt(jets, args.pt_min, args.dt_max, out_dir / f"dt_t0_vs_pt_{tag}.png", args.pdf)
    plot_dt_1d(jets, features, args.pt_min, args.dt_max, out_dir / f"dt_t0_1d_{tag}.png", args.pdf)
    plot_features_vs_pt(features, args.pt_min, out_dir / f"features_vs_pt_{COLL}.png", args.pdf)
    plot_layer_plane(features, args.pt_min, out_dir / f"layer_plane_{COLL}.png", args.pdf)
    plot_layer_plane_high_pt(features, args.pt_min, out_dir / f"layer_plane_{COLL}_pt{args.pt_min:g}.png", args.pdf)
    write_csv(features, out_dir / f"jets_all_{COLL}.csv",
              ["run", "event", "centrality", "pt_calib", "pt_raw", "eta", "phi", "category", "frac_emcal", "frac_ihcal",
               "frac_ohcal", "z_lead", "lead_layer", "swiss", "r03_r02", "recoil_pt", "recoil_frac", "t0", "t", "dt_t0"])

    if payloads:
        display_dir = out_dir / "event_displays"
        workers = args.workers if args.workers > 0 else min(os.cpu_count() or 4, 16)
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(draw_event_display, p, display_dir, args.logz) for p in payloads]
            for fut in concurrent.futures.as_completed(futures):
                fut.result()
        print(f"Saved event displays for {len(payloads)} events in {display_dir}")


if __name__ == "__main__":
    main()
