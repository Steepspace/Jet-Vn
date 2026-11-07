#!/usr/bin/env python3
"""Clean dijet control sample from the Jet-Dijet production (tasks/jet-dijet-control.md).

Jet-Dijet.C fills, per radius (R = 0.2, 0.3, iternf jets), 10% centrality class and raw pT_lead class, TH2s of the
subleading raw pT (and of x_J) vs delta phi of the leading and subleading jets. This script sums them over runs and:

1. Q1: event QA (jets per event and the fraction of events with a dijet, vs centrality).
2. D1: the delta phi distributions per pT_lead class and centrality, for several pT_sub thresholds.
3. D2: the purity fits. The dijets at delta phi > 3pi/4 are signal (a real back-to-back pair) plus combinatorial
   background (a subleading jet uncorrelated with the leading jet, mostly an underlying-event fluctuation). Two
   estimates of the background under the peak:
   - flat: the mean level at 5pi/12 < delta phi < 7pi/12, taken as flat (an upper bound on the purity, since the
     background is lowest near pi/2);
   - fit: B (1 + 2 V2 cos 2 delta phi) + a T(delta phi) over delta phi > pi/4, with T the signal shape of the same
     pT_lead class and pT_sub threshold in 40-60% (minus its own flat level, zero below pi/2). The iternf jets have
     no flow subtraction, so the background is modulated by the flow of the underlying event.
   The near side (delta phi < pi/4) is not fitted: it has a bump just beyond 2R.
4. D3: purity vs pT_sub threshold, D4: statistics vs threshold (total, times purity, per run), D5: x_J distributions.
5. The chosen cut per radius and centrality class: the lowest pT_sub threshold whose fit purity (all pT_lead classes
   together) reaches --purity-target. Written to dijet_cuts.csv, with the full scan in purity_scan_<r>.csv.
"""

import argparse
import concurrent.futures
import csv
import hashlib
import os
from pathlib import Path
import pickle
import sys
import numpy as np
import uproot
from scipy.optimize import least_squares
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mplhep as hep

from plot_jet_bkgsub_compare import save_fig
from plot_jet_noflow_compare import parse_list_file, load_luminosity
from plot_jet_iternf_timing import StageTimer, add_header, sci

CACHE_VERSION = 1

# (tag, R, |eta| acceptance of Jet-Dijet.C)
RADII = {
    "r02": {"r": 0.2, "eta_max": 0.9},
    "r03": {"r": 0.3, "eta_max": 0.8},
}
CENT10 = tuple((c, c + 10) for c in range(0, 60, 10))      # the classes of Jet-Dijet.C
CLASSES = ((0, 20), (20, 40), (40, 60))                    # the classes of the plots and the cuts
CLASS_COLORS = ("#1b9e77", "#d95f02", "#7570b3")
LEAD_CLS = ((7, 10), (10, 15), (15, 20), (20, 25), (25, 30), (30, 40), (40, 60), (60, 100))   # m_lead_edges
LEAD_PLOT = ((10, 15), (15, 20), (20, 30), (30, 40))      # pT_lead ranges of the D1 / D2 / D5 rows (unions of LEAD_CLS)
LEAD_COLORS = ("#999999", "#1b7837", "#e08214", "#762a83", "#2166ac", "#b2182b", "#4d4d4d", "#000000")
JET_PT_MIN = 7                                             # raw pT floor of the production [GeV]
THR_SCAN = tuple(range(7, 31))                             # pT_sub thresholds scanned [GeV]
THR_PLOT = (7, 10, 12, 15, 20)                             # thresholds of the D1 curves
THR_COLORS = ("#000000", "#e41a1c", "#377eb8", "#4daf4a", "#984ea3")
PERIPH = (40, 60)                                          # the class whose signal shape is the template
B2B_MIN = 3 * np.pi / 4                                    # back-to-back cut
FLAT_WIN = (5 * np.pi / 12, 7 * np.pi / 12)                # flat background window
FIT_MIN = np.pi / 4                                        # fit range: delta phi > FIT_MIN
V2_MAX = 0.5
MIN_FIT_DIJETS = 100                                       # fewer dijets at delta phi > 3pi/4: no fit (flat estimate used)
MIN_TEMPLATE = 50                                          # fewer template dijets: the template of the lowest threshold

EVENT_INFO = (
    r"$|z| < 10$ cm & MB" "\n"
    "Good Calo-Cent\n"
    "Iterative No Flow"
)
DPHI_LABEL = r"$\Delta\phi$ (leading, subleading)"


def jet_info(r_tag):
    return rf"$|\eta| < {RADII[r_tag]['eta_max']:g}$, Energy > 0, raw $p_{{T}} \geq {JET_PT_MIN}$ GeV"


def lead_label(lead):
    return rf"${lead[0]} \leq p_{{T,\mathrm{{lead}}}} < {lead[1]}$ GeV"


def cent_label(c):
    return f"{c[0]}–{c[1]}%"


def cent_idx(c):
    return [i for i, (lo, hi) in enumerate(CENT10) if lo >= c[0] and hi <= c[1]]


def lead_idx(lead):
    return [i for i, (lo, hi) in enumerate(LEAD_CLS) if lo >= lead[0] and hi <= lead[1]]


# Input
# ---------------------------------------------------------------------------------------------------------------------

def sub_name(r_tag, c, lead):
    return f"h2PtSubDphi_{r_tag}_cent_{c[0]}_{c[1]}_ptlead_{lead[0]}_{lead[1]}"


def xj_name(r_tag, c, lead):
    return f"h2XjDphi_{r_tag}_cent_{c[0]}_{c[1]}_ptlead_{lead[0]}_{lead[1]}"


def load_run(task):
    """Per run: hEvent, hCentrality, and per radius the stacked hists S (cent, lead, pT_sub, dphi), X (cent, lead, x_J,
    dphi), the leading pT vs centrality and the jet count vs centrality. Returns (run, data), data None on a missing hist."""
    run, path = task
    try:
        with uproot.open(path) as rf:
            d = {"hEvent": rf["hEvent"].values().astype(np.float64),
                 "hCentrality": rf["hCentrality"].values().astype(np.float64)}
            for r_tag in RADII:
                d[r_tag] = {
                    "S": np.stack([np.stack([rf[sub_name(r_tag, c, l)].values() for l in LEAD_CLS]) for c in CENT10]),
                    "X": np.stack([np.stack([rf[xj_name(r_tag, c, l)].values() for l in LEAD_CLS]) for c in CENT10]),
                    "lead": rf[f"h2LeadPtCent_{r_tag}"].values(flow=True),
                    "njets": rf[f"h2NjetsCent_{r_tag}"].values(),
                }
    except (KeyError, OSError) as e:
        print(f"Warning: run {run}: {e}, skipping")
        return run, None
    return run, d


def read_axes(path):
    with uproot.open(path) as rf:
        h = rf[sub_name("r02", CENT10[0], LEAD_CLS[0])]
        hx = rf[xj_name("r02", CENT10[0], LEAD_CLS[0])]
        return {"pt": h.axis(0).edges(), "dphi": h.axis(1).edges(), "xj": hx.axis(0).edges()}


def cache_key(tasks, lumi_paths):
    h = hashlib.sha1(f"v{CACHE_VERSION}".encode())
    for run, path in tasks:
        st = path.stat()
        h.update(f"{run}|{path}|{st.st_size}|{st.st_mtime_ns}\n".encode())
    for path in lumi_paths:
        st = path.stat()
        h.update(f"L|{path}|{st.st_size}|{st.st_mtime_ns}\n".encode())
    return h.hexdigest()[:16]


def load_inputs(tasks, lumi_files, n_jobs):
    """Sums the hists over runs; keeps per run the delta phi > 3pi/4 counts per (radius, cent, lead, pT_sub bin)."""
    axes = read_axes(tasks[0][1])
    b2b_bin = int(np.searchsorted(axes["dphi"], B2B_MIN - 1e-9))
    total, per_run, runs = {}, {r: [] for r in RADII}, []
    ex = concurrent.futures.ProcessPoolExecutor(max_workers=n_jobs) if n_jobs > 1 else None
    for run, d in (ex.map(load_run, tasks, chunksize=8) if ex else map(load_run, tasks)):
        if d is None:
            continue
        runs.append(run)
        for k in ("hEvent", "hCentrality"):
            total[k] = total.get(k, 0) + d[k]
        for r_tag in RADII:
            t = total.setdefault(r_tag, {})
            for k, v in d[r_tag].items():
                t[k] = t.get(k, 0) + v.astype(np.float64)
            per_run[r_tag].append(d[r_tag]["S"][..., b2b_bin:].sum(axis=-1).astype(np.float32))
    if ex:
        ex.shutdown()
    order = np.argsort(runs)
    runs = [runs[i] for i in order]
    per_run = {r: np.stack(v)[order] for r, v in per_run.items()}    # (run, cent, lead, pT_sub)

    lumi = None
    missing = [r for r in runs if r not in lumi_files or not lumi_files[r].exists()]
    if missing:
        print(f"Warning: no luminosity file for {len(missing)} run(s) (e.g. {missing[:5]}), omitting the luminosity label")
    else:
        lumis = [load_luminosity(lumi_files[r]) for r in runs]
        lumi = None if any(l is None for l in lumis) else float(sum(lumis))
    return {"total": total, "per_run": per_run, "runs": runs, "axes": axes, "lumi": lumi}


# Purity
# ---------------------------------------------------------------------------------------------------------------------

class Dijets:
    """Delta phi distributions and purity estimates of one radius."""

    def __init__(self, inputs, r_tag):
        self.r_tag = r_tag
        self.S = inputs["total"][r_tag]["S"]                   # (cent, lead, pT_sub, dphi)
        self.X = inputs["total"][r_tag]["X"]                   # (cent, lead, x_J, dphi)
        self.per_run = inputs["per_run"][r_tag]                # (run, cent, lead, pT_sub), delta phi > 3pi/4
        ax = inputs["axes"]
        self.pe, self.de, self.xe = ax["pt"], ax["dphi"], ax["xj"]
        self.dc = 0.5 * (self.de[1:] + self.de[:-1])
        self.b2b = self.dc > B2B_MIN
        self.flat = (self.dc > FLAT_WIN[0]) & (self.dc < FLAT_WIN[1])
        self.fit_range = self.dc > FIT_MIN
        self._cache = {}

    def pt_sel(self, thr):
        return self.pe[:-1] >= thr - 1e-9

    def dphi(self, c, lead, thr):
        """Dijets vs delta phi with pT_sub >= thr, in the classes c and pT_lead range lead."""
        s = self.S[np.ix_(cent_idx(c), lead_idx(lead))]
        return s[:, :, self.pt_sel(thr), :].sum(axis=(0, 1, 2))

    def template(self, lead, thr):
        """Signal shape of the peripheral class: minus its flat level, zero below pi/2, unit sum."""
        for t in (thr, JET_PT_MIN):
            v = self.dphi(PERIPH, lead, t)
            if v[self.b2b].sum() >= MIN_TEMPLATE:
                tm = np.clip(v - v[self.flat].mean(), 0, None)
                tm[self.dc <= np.pi / 2] = 0
                if tm.sum() > 0:
                    return tm / tm.sum()
        return None

    def estimate(self, c, lead, thr):
        """Purity estimates of one (class, pT_lead class, threshold). Background counts under delta phi > 3pi/4."""
        key = (c, lead, thr)
        if key in self._cache:
            return self._cache[key]
        v = self.dphi(c, lead, thr)
        n = v[self.b2b].sum()
        out = {"n": n, "b_flat": min(v[self.flat].mean() * self.b2b.sum(), n) if n > 0 else 0.0,
               "b_fit": np.nan, "v2": np.nan, "chi2ndf": np.nan, "fit": "none", "v": v}
        tm = self.template(lead, thr)
        if n >= MIN_FIT_DIJETS and tm is not None:
            dc, sel = self.dc, self.fit_range
            cos2 = np.cos(2 * dc)
            err = np.sqrt(np.maximum(v, 1))

            def model(p):
                return p[0] * (1 + 2 * p[2] * cos2), p[1] * tm

            def res(p):
                bk, s = model(p)
                return ((bk + s - v) / err)[sel]

            b0 = max(v[self.flat].mean(), 1e-3)
            p0 = [b0, max(n - b0 * self.b2b.sum(), 1.0), 0.05]
            r = least_squares(res, p0, bounds=([0, 0, 0], [np.inf, np.inf, V2_MAX]))
            bk, s = model(r.x)
            out.update(b_fit=min(bk[self.b2b].sum(), n), v2=r.x[2],
                       chi2ndf=(r.fun ** 2).sum() / max(sel.sum() - 3, 1), fit="ok", bk=bk, sig=s, par=r.x)
        self._cache[key] = out
        return out

    def combined(self, c, thr):
        """Sum over the pT_lead classes (each fitted with its own template): the cut is on pT_sub alone."""
        n = b_flat = b_fit = 0.0
        n_nofit = 0.0
        for lead in LEAD_CLS:
            if lead[1] <= thr:
                continue
            e = self.estimate(c, lead, thr)
            n += e["n"]
            b_flat += e["b_flat"]
            if e["fit"] == "ok":
                b_fit += e["b_fit"]
            else:                       # too few dijets for a fit: the flat estimate
                b_fit += e["b_flat"]
                n_nofit += e["n"]
        p = lambda b: 1 - b / n if n > 0 else np.nan
        return {"n": n, "p_flat": p(b_flat), "p_fit": p(b_fit), "frac_nofit": n_nofit / n if n > 0 else np.nan}

    def run_counts(self, c, thr):
        """Dijets at delta phi > 3pi/4 with pT_sub >= thr per run (all pT_lead classes)."""
        return self.per_run[:, cent_idx(c)][:, :, :, self.pt_sel(thr)].sum(axis=(1, 2, 3))


def choose_cut(D, c, target):
    for thr in THR_SCAN:
        e = D.combined(c, thr)
        if e["n"] > 0 and e["p_fit"] >= target:
            return thr, e
    return None, None


# Plots
# ---------------------------------------------------------------------------------------------------------------------

def mark_regions(ax):
    ax.axvspan(*FLAT_WIN, color="0.9", zorder=0)
    ax.axvline(B2B_MIN, color="gray", linestyle="--", linewidth=1)
    ax.axvline(FIT_MIN, color="gray", linestyle=":", linewidth=1)


def plot_q1(inputs, runs, lumi, output_path, save_pdf=False):
    hep.style.use("ATLAS")
    t = inputs["total"]
    cent = t["hCentrality"]
    cc = np.arange(len(cent))
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    for r_tag, col in zip(RADII, ("#e41a1c", "#377eb8")):
        nj = t[r_tag]["njets"]                                  # (cent, njets)
        nc = np.arange(nj.shape[1])
        with np.errstate(invalid="ignore", divide="ignore"):
            axes[0].plot(cc, (nj * nc).sum(axis=1) / nj.sum(axis=1), color=col, label=rf"$R = {RADII[r_tag]['r']:g}$")
            axes[1].plot(cc, nj[:, 2:].sum(axis=1) / nj.sum(axis=1), color=col, label=rf"$R = {RADII[r_tag]['r']:g}$")
    axes[0].set_ylabel(r"$\langle N_{\mathrm{jets}} \rangle$ per event")
    axes[1].set_ylabel("Fraction of events with a dijet")
    for ax in axes:
        ax.set_xlabel("Centrality [%]")
        ax.set_yscale("log")
        ax.legend(loc="upper right")
        ax.text(0.04, 0.04, EVENT_INFO + "\n" + rf"Jets: raw $p_{{T}} \geq {JET_PT_MIN}$ GeV, Energy > 0",
                transform=ax.transAxes, fontsize=12, va="bottom")
    add_header(axes[1], runs, None, lumi,
               extra=rf", $N_{{\mathrm{{evt}}}} = {sci(cent.sum())}$")
    fig.tight_layout()
    save_fig(fig, output_path, save_pdf)
    plt.close(fig)


def plot_d1(D, runs, lumi, output_path, save_pdf=False):
    """Rows: pT_lead ranges; columns: centrality classes; curves: pT_sub thresholds, per dijet."""
    hep.style.use("ATLAS")
    fig, axes = plt.subplots(len(LEAD_PLOT), len(CLASSES), figsize=(7 * len(CLASSES), 4.6 * len(LEAD_PLOT)), sharex=True)
    width = np.diff(D.de)
    for i, lead in enumerate(LEAD_PLOT):
        for j, c in enumerate(CLASSES):
            ax = axes[i, j]
            mark_regions(ax)
            for thr, col in zip(THR_PLOT, THR_COLORS):
                if thr >= lead[1]:
                    continue
                v = D.dphi(c, lead, thr)
                if v.sum() < 20:
                    continue
                ax.stairs(v / v.sum() / width, D.de, color=col, linewidth=1.5,
                          label=rf"$p_{{T,\mathrm{{sub}}}} \geq {thr}$: {v.sum():.3g}")
            ax.set_yscale("log")
            ax.set_xlim(0, np.pi)
            ax.legend(loc="upper left", fontsize=9, title=f"{cent_label(c)}, {lead_label(lead)}", title_fontsize=10)
            if i == len(LEAD_PLOT) - 1:
                ax.set_xlabel(DPHI_LABEL)
            if j == 0:
                ax.set_ylabel(r"$\frac{1}{N_{\mathrm{dijet}}}\frac{dN}{d\Delta\phi}$")
    add_header(axes[0, -1], runs, RADII[D.r_tag]["r"], lumi)
    fig.text(0.01, 0.995, f"{EVENT_INFO.replace(chr(10), ', ')}; {jet_info(D.r_tag)}. Gray band: flat window; dashed: "
             r"$3\pi/4$; dotted: fit range start", fontsize=11, va="top")
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    save_fig(fig, output_path, save_pdf)
    plt.close(fig)


def plot_d2(D, thr, runs, lumi, output_path, save_pdf=False):
    """The purity fits at one pT_sub threshold, per pT_lead class (rows) and centrality class (columns)."""
    hep.style.use("ATLAS")
    leads = [l for l in LEAD_CLS if l[1] > thr and l[0] >= 10 and l[1] <= 40]
    fig, axes = plt.subplots(len(leads), len(CLASSES), figsize=(7 * len(CLASSES), 4.4 * len(leads)), sharex=True, squeeze=False)
    for i, lead in enumerate(leads):
        for j, c in enumerate(CLASSES):
            ax = axes[i, j]
            mark_regions(ax)
            e = D.estimate(c, lead, thr)
            v = e["v"]
            ax.errorbar(D.dc, v, yerr=np.sqrt(v), fmt="o", color="black", markersize=3, label="Data")
            ax.axhline(e["b_flat"] / D.b2b.sum(), color="#377eb8", linestyle="--", linewidth=1.5, label="Flat bkg.")
            txt = (rf"$N(\Delta\phi > 3\pi/4) = {e['n']:.4g}$" "\n"
                   rf"$P_{{\mathrm{{flat}}}} = {1 - e['b_flat'] / e['n']:.2f}$" if e["n"] > 0 else "no dijets")
            if e["fit"] == "ok":
                ax.plot(D.dc, e["bk"], color="#e41a1c", linewidth=1.5, label=r"Bkg. $B(1 + 2V_{2}\cos 2\Delta\phi)$")
                ax.plot(D.dc, e["bk"] + e["sig"], color="#4daf4a", linewidth=1.5, label="Bkg. + 40–60% signal shape")
                txt += (rf", $P_{{\mathrm{{fit}}}} = {1 - e['b_fit'] / e['n']:.2f}$" "\n"
                        rf"$V_{{2}} = {e['v2']:.3f}$, $\chi^{{2}}/\mathrm{{ndf}} = {e['chi2ndf']:.1f}$")
            ax.text(0.03, 0.97, f"{cent_label(c)}, {lead_label(lead)}\n" + txt, transform=ax.transAxes,
                    fontsize=10, va="top")
            ax.set_xlim(0, np.pi)
            ax.set_ylim(bottom=0)
            if i == 0 and j == 0:
                ax.legend(loc="center left", fontsize=9)
            if i == len(leads) - 1:
                ax.set_xlabel(DPHI_LABEL)
            if j == 0:
                ax.set_ylabel("Dijets")
    add_header(axes[0, -1], runs, RADII[D.r_tag]["r"], lumi)
    fig.text(0.01, 0.995, rf"{jet_info(D.r_tag)}, $p_{{T,\mathrm{{sub}}}} \geq {thr}$ GeV. Fit over $\Delta\phi > \pi/4$; "
             "the 40–60% fit uses its own shape.", fontsize=11, va="top")
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    save_fig(fig, output_path, save_pdf)
    plt.close(fig)


def plot_d3(D, cuts, target, runs, lumi, output_path, save_pdf=False):
    """Purity vs pT_sub threshold per class: all pT_lead together (band: fit to flat), and per pT_lead class (fit)."""
    hep.style.use("ATLAS")
    fig, axes = plt.subplots(1, len(CLASSES), figsize=(7 * len(CLASSES), 6.5), sharey=True)
    thr = np.array(THR_SCAN)
    for ax, c in zip(axes, CLASSES):
        comb = [D.combined(c, t) for t in thr]
        pf = np.array([e["p_fit"] for e in comb])
        pz = np.array([e["p_flat"] for e in comb])
        ax.fill_between(thr, pf, pz, color="0.75", step=None, label="All $p_{T,\\mathrm{lead}}$: fit to flat")
        ax.plot(thr, pf, color="black", linewidth=2, label="All $p_{T,\\mathrm{lead}}$: fit")
        for lead, col in zip(LEAD_CLS, LEAD_COLORS):
            if lead[0] < 10 or lead[1] > 60:
                continue
            p = np.array([np.nan if t >= lead[1] else
                          (lambda e: 1 - e["b_fit"] / e["n"] if e["fit"] == "ok" else np.nan)(D.estimate(c, lead, t))
                          for t in thr])
            ax.plot(thr, p, color=col, linewidth=1.2, marker="o", markersize=3, label=lead_label(lead))
        ax.axhline(target, color="gray", linestyle="--")
        cut = cuts.get(c)
        if cut is not None:
            ax.axvline(cut, color="#e41a1c", linestyle=":")
            ax.text(cut + 0.3, 0.03, rf"$p_{{T,\mathrm{{sub}}}} \geq {cut}$", color="#e41a1c", fontsize=12)
        ax.text(0.96, 0.55, cent_label(c), transform=ax.transAxes, fontsize=18, ha="right", va="center")
        ax.set_xlabel(r"$p_{T,\mathrm{sub}}$ threshold (raw) [GeV]")
        ax.set_xlim(thr[0], thr[-1])
        ax.set_ylim(0, 1.05)
    axes[0].set_ylabel(r"Purity at $\Delta\phi > 3\pi/4$")
    axes[0].legend(loc="lower right", fontsize=9)
    add_header(axes[-1], runs, RADII[D.r_tag]["r"], lumi)
    fig.tight_layout()
    save_fig(fig, output_path, save_pdf)
    plt.close(fig)


def plot_d4(D, cuts, runs, lumi, output_path, save_pdf=False):
    """Statistics vs threshold: dijets (and times the fit purity), the median per run and the runs with >= 10 / 20."""
    hep.style.use("ATLAS")
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    thr = np.array(THR_SCAN)
    for c, col in zip(CLASSES, CLASS_COLORS):
        comb = [D.combined(c, t) for t in thr]
        n = np.array([e["n"] for e in comb])
        npf = n * np.array([e["p_fit"] for e in comb])
        axes[0].plot(thr, n, color=col, linestyle="--", label=f"{cent_label(c)}: all")
        axes[0].plot(thr, npf, color=col, linewidth=2, label=f"{cent_label(c)}: × fit purity")
        rc = np.array([D.run_counts(c, t) for t in thr])          # (thr, run)
        axes[1].plot(thr, np.median(rc, axis=1), color=col, linewidth=2, label=f"{cent_label(c)}: median per run")
        cut = cuts.get(c)
        for ax in axes:
            if cut is not None:
                ax.axvline(cut, color=col, linestyle=":")
    axes[0].set_ylabel(r"Dijets at $\Delta\phi > 3\pi/4$")
    axes[1].set_ylabel(r"Dijets per run at $\Delta\phi > 3\pi/4$")
    for ax in axes:
        ax.set_yscale("log")
        ax.set_xlabel(r"$p_{T,\mathrm{sub}}$ threshold (raw) [GeV]")
        ax.set_xlim(thr[0], thr[-1])
        ax.legend(loc="upper right", fontsize=10)
    axes[0].text(0.03, 0.03, "dotted: the chosen cut", transform=axes[0].transAxes, fontsize=11)
    add_header(axes[1], runs, RADII[D.r_tag]["r"], lumi)
    fig.tight_layout()
    save_fig(fig, output_path, save_pdf)
    plt.close(fig)


def plot_d5(D, runs, lumi, output_path, save_pdf=False):
    """x_J of the dijets at delta phi > 3pi/4 (pT_sub >= the 7 GeV floor), per pT_lead range, per class."""
    hep.style.use("ATLAS")
    fig, axes = plt.subplots(1, len(LEAD_PLOT), figsize=(6 * len(LEAD_PLOT), 6.5), sharey=True)
    for ax, lead in zip(axes, LEAD_PLOT):
        for c, col in zip(CLASSES, CLASS_COLORS):
            x = D.X[np.ix_(cent_idx(c), lead_idx(lead))][..., D.b2b].sum(axis=(0, 1, 3))
            if x.sum() > 0:
                ax.stairs(x / x.sum() / np.diff(D.xe), D.xe, color=col, linewidth=2, label=f"{cent_label(c)}: {x.sum():.3g}")
        for p in lead:
            ax.axvline(JET_PT_MIN / p, color="gray", linestyle=":")
        ax.set_xlabel(r"$x_{J} = p_{T,\mathrm{sub}} / p_{T,\mathrm{lead}}$")
        ax.set_xlim(0, 1)
        ax.legend(loc="upper left", fontsize=10, title=lead_label(lead), title_fontsize=12)
    axes[0].set_ylabel(r"$\frac{1}{N}\frac{dN}{dx_{J}}$")
    add_header(axes[-1], runs, RADII[D.r_tag]["r"], lumi)
    fig.text(0.01, 0.995, rf"{jet_info(D.r_tag)}, $\Delta\phi > 3\pi/4$, no $p_{{T,\mathrm{{sub}}}}$ cut beyond the floor. "
             rf"Dotted: $x_{{J}} = {JET_PT_MIN}\,\mathrm{{GeV}} / p_{{T,\mathrm{{lead}}}}$ at the range edges.", fontsize=11, va="top")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    save_fig(fig, output_path, save_pdf)
    plt.close(fig)


# Tables
# ---------------------------------------------------------------------------------------------------------------------

def write_scan(D, path):
    keys = ["cent_lo", "cent_hi", "lead_lo", "lead_hi", "pt_sub_min", "n_b2b", "p_flat", "p_fit", "v2", "chi2ndf", "fit",
            "frac_nofit", "median_per_run"]
    rows = []
    for c in CLASSES + CENT10:
        for thr in THR_SCAN:
            comb = D.combined(c, thr)
            rows.append({"cent_lo": c[0], "cent_hi": c[1], "lead_lo": "all", "lead_hi": "all", "pt_sub_min": thr,
                         "n_b2b": f"{comb['n']:.0f}", "p_flat": f"{comb['p_flat']:.4f}", "p_fit": f"{comb['p_fit']:.4f}",
                         "v2": "", "chi2ndf": "", "fit": "combined", "frac_nofit": f"{comb['frac_nofit']:.4f}",
                         "median_per_run": f"{np.median(D.run_counts(c, thr)):.1f}"})
            for lead in LEAD_CLS:
                if lead[1] <= thr:
                    continue
                e = D.estimate(c, lead, thr)
                rows.append({"cent_lo": c[0], "cent_hi": c[1], "lead_lo": lead[0], "lead_hi": lead[1], "pt_sub_min": thr,
                             "n_b2b": f"{e['n']:.0f}",
                             "p_flat": f"{1 - e['b_flat'] / e['n']:.4f}" if e["n"] > 0 else "",
                             "p_fit": f"{1 - e['b_fit'] / e['n']:.4f}" if e["fit"] == "ok" else "",
                             "v2": f"{e['v2']:.4f}" if e["fit"] == "ok" else "",
                             "chi2ndf": f"{e['chi2ndf']:.2f}" if e["fit"] == "ok" else "", "fit": e["fit"],
                             "frac_nofit": "", "median_per_run": ""})
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"Saved: {path}")


# Main
# ---------------------------------------------------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Clean dijet control sample: delta phi, purity and statistics vs the "
                                                 "subleading jet pT threshold, per centrality class (Jet-Dijet output).")
    parser.add_argument("-f", "--file", type=Path,
                        default=Path("/sphenix/user/anarde/sEPD-Study/jetDijet/11-06-26/test.list"),
                        help="List of Jet-Dijet merged ROOT files named <run>.root (default: %(default)s)")
    parser.add_argument("-l", "--lumi-list", type=Path,
                        default=Path("/sphenix/user/anarde/sEPD-Study/f4a_bkgsub/10-30-26/test.list"),
                        help="List of Event QA ROOT files named <run>.root with hLuminosity (default: %(default)s)")
    parser.add_argument("-o", "--output-dir", type=Path,
                        default=Path("/sphenix/user/anarde/sEPD-Study/plots/Jet-Ana/dijet_control"),
                        help="Output directory (default: %(default)s)")
    parser.add_argument("--runs", nargs="*", type=int, help="Specific run number(s) to combine (default: all runs in the list)")
    parser.add_argument("--radii", nargs="+", choices=list(RADII), default=list(RADII), help="Jet radii (default: both)")
    parser.add_argument("--purity-target", type=float, default=0.9,
                        help="Fit purity the chosen pT_sub threshold must reach, per class (default: %(default)s)")
    parser.add_argument("--fit-thresholds", nargs="+", type=int, default=[10, 15],
                        help="pT_sub thresholds with D2 fit plots, besides the chosen cuts (default: %(default)s)")
    parser.add_argument("-j", "--jobs", type=int, default=min(8, os.cpu_count() or 1),
                        help="Parallel workers for reading the runs (default: %(default)s)")
    parser.add_argument("--save-pdf", action="store_true", help="Save PDF versions in addition to PNG")
    parser.add_argument("--cache-dir", type=Path, default=None, help="Input cache directory (default: <output-dir>/cache)")
    parser.add_argument("--no-cache", action="store_true", help="Neither read nor write the input cache")
    parser.add_argument("--refresh-cache", action="store_true", help="Reread the ROOT files and overwrite the cache")
    args = parser.parse_args()
    tick = StageTimer()

    run_files = parse_list_file(args.file)
    if not run_files:
        print(f"Error: No ROOT files found in {args.file}")
        sys.exit(1)
    tasks = [(r, run_files[r]) for r in (sorted(args.runs) if args.runs else sorted(run_files))
             if r in run_files and run_files[r].exists()]
    lumi_files = parse_list_file(args.lumi_list) if args.lumi_list.exists() else {}
    lumi_paths = [lumi_files[r] for r, _ in tasks if r in lumi_files and lumi_files[r].exists()]

    cache_path = None
    if not args.no_cache:
        cache_dir = args.cache_dir if args.cache_dir is not None else args.output_dir / "cache"
        cache_path = cache_dir / f"inputs_{cache_key(tasks, lumi_paths)}.pkl"
    if cache_path is not None and cache_path.exists() and not args.refresh_cache:
        print(f"Reading the inputs from the cache {cache_path}")
        with cache_path.open("rb") as f:
            inputs = pickle.load(f)
    else:
        n_jobs = max(1, min(args.jobs, len(tasks)))
        print(f"Reading {len(tasks)} runs with {n_jobs} worker(s)")
        inputs = load_inputs(tasks, lumi_files, n_jobs)
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with cache_path.open("wb") as f:
                pickle.dump(inputs, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"Saved the input cache {cache_path}")
    tick("inputs")
    runs, lumi = inputs["runs"], inputs["lumi"]
    ev = inputs["total"]["hEvent"]
    print(f"{len(runs)} runs; events: {ev[0]:.4g} entries, {ev[1]:.4g} Good Calo-Cent, {ev[2]:.4g} selected")

    out = args.output_dir
    plot_q1(inputs, runs, lumi, out / "Q1_events.png", args.save_pdf)

    cut_rows = []
    for r_tag in args.radii:
        D = Dijets(inputs, r_tag)
        cuts = {}
        print(f"\n{r_tag}: chosen pT_sub cut (fit purity >= {args.purity_target})")
        for c in CLASSES:
            thr, e = choose_cut(D, c, args.purity_target)
            cuts[c] = thr
            if thr is None:
                print(f"  {cent_label(c)}: none up to {THR_SCAN[-1]} GeV")
                cut_rows.append({"radius": r_tag, "cent_lo": c[0], "cent_hi": c[1], "pt_sub_min": "", "n_b2b": "",
                                 "p_fit": "", "p_flat": "", "median_per_run": "", "runs_ge10": ""})
                continue
            rc = D.run_counts(c, thr)
            print(f"  {cent_label(c)}: pT_sub >= {thr} GeV, {e['n']:.0f} dijets, purity fit {e['p_fit']:.3f} / flat "
                  f"{e['p_flat']:.3f}, median {np.median(rc):.0f} per run, {np.mean(rc >= 10) * 100:.0f}% of runs with >= 10")
            cut_rows.append({"radius": r_tag, "cent_lo": c[0], "cent_hi": c[1], "pt_sub_min": thr, "n_b2b": f"{e['n']:.0f}",
                             "p_fit": f"{e['p_fit']:.4f}", "p_flat": f"{e['p_flat']:.4f}",
                             "median_per_run": f"{np.median(rc):.1f}", "runs_ge10": f"{np.mean(rc >= 10):.4f}"})

        rdir = out / r_tag
        plot_d1(D, runs, lumi, rdir / f"D1_dphi_{r_tag}.png", args.save_pdf)
        for thr in sorted(set(args.fit_thresholds) | {t for t in cuts.values() if t is not None}):
            plot_d2(D, thr, runs, lumi, rdir / f"D2_fits_{r_tag}_ptsub{thr}.png", args.save_pdf)
        plot_d3(D, cuts, args.purity_target, runs, lumi, rdir / f"D3_purity_{r_tag}.png", args.save_pdf)
        plot_d4(D, cuts, runs, lumi, rdir / f"D4_stats_{r_tag}.png", args.save_pdf)
        plot_d5(D, runs, lumi, rdir / f"D5_xj_{r_tag}.png", args.save_pdf)
        write_scan(D, rdir / f"purity_scan_{r_tag}.csv")
        tick(r_tag)

    path = out / "dijet_cuts.csv"
    with path.open("w", newline="") as f:
        f.write(f"# lowest raw pT_sub threshold [GeV] with fit purity >= {args.purity_target} at delta phi > 3pi/4, "
                f"all pT_lead; {len(runs)} runs of {args.file}\n")
        w = csv.DictWriter(f, fieldnames=list(cut_rows[0]))
        w.writeheader()
        w.writerows(cut_rows)
    print(f"Saved: {path}")


if __name__ == "__main__":
    main()
