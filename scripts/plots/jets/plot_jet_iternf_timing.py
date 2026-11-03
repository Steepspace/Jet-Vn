#!/usr/bin/env python3
"""Run-combined iternf jet plots and the leading-jet timing correction from Jet-Anav3 output (iter_is_noflow mode).

1. Raw jet pT (hJetPtv3_raw_<r>_iternf) summed over runs.
2. A model of the leading jet time t = t_jet - t0(MBD) vs OHCal energy fraction x, raw pT and centrality class c,
   fitted to p2LeadJetTimeOHCalFracRawPt_<r>_iternf_cent_<lo>_<hi> (TProfile2D per run, raw pT 10-60 GeV):

       t(x, pT, c, run) = f_c(x, pT) + a_run + b_run * x,    f_c(x, pT) = sum_{i<=4, j<=3} c_cij x^i [ln(pT / 15 GeV)]^j

   f_c is smooth in x and pT, one per centrality class. The per-run line a + b x is shared by all pT and centrality
   cells of a run; it is measured on one radius (default R = 0.3) and applied to every radius, which gets its own f_c.
   Convention: the jet-weighted means of a and b over runs are 0, so f_c carries the average level and slope.
3. A model ladder that adds one ingredient at a time (M1 f(x) -> M2 + a_run -> M3 + b_run x -> M4a + centrality,
   M4b + pT -> M5 both = the model), with plots of what each step fixes and leaves.
4. Tests of the model: degree scan, closure vs (pT, x, centrality), per-run chi2/ndf, pulls, per-run lines and their
   stability across pT and centrality, the other radius' own lines, the shapes against the data, timing RMS.

The fit uses per-run cell means of the profile. A cell's error is sigma_cell / sqrt(n_run_cell), with sigma_cell the
time RMS of the cell pooled over all runs: robust for run-cells with few jets, so the sparse high-pT cells enter the fit.
The per-run line is projected out exactly (normal equations); a pseudo-inverse handles the constant and slope that the
per-run lines and the shape share.
"""

import argparse
import concurrent.futures
import csv
import hashlib
import json
import os
from pathlib import Path
import pickle
import re
import sys
import time
import numpy as np
import uproot
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.ticker import LogLocator
import mplhep as hep

from plot_jet_bkgsub_compare import save_fig
from plot_jet_noflow_compare import parse_list_file, load_luminosity

# (tag, R, |eta| acceptance used in Jet-Anav3.C: m_jet_eta_max_r02 / m_jet_eta_max_r03)
RADII = {
    "r02": {"r": 0.2, "eta_max": 0.9},
    "r03": {"r": 0.3, "eta_max": 0.8},
}

TIMING_PT = (15, 40)                              # raw pT range of h2LeadJetTimeOHCalFrac, used for the combined 2D plot
CENT_CLASSES = ((0, 20), (20, 40), (40, 60))      # m_timing_cent_ranges of Jet-Anav3.C
CENT_COLORS = ("#1b9e77", "#d95f02", "#7570b3")
PT_REF = 15.0                                     # GeV, f_c is polynomial in ln(pT / PT_REF)
JET_PT_MIN = 10                                   # calibrated jet pT cut of the production (jet_pt_min job arg)
PT_RANGES = ((10, 15), (15, 20), (20, 30), (30, 60))
X_SLICES = ((-0.1, 0.1), (0.1, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 0.9))
X_SLICE_COLORS = ("#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e")

# Event selection shared by all histograms. In iter_is_noflow mode calo v2 = 0 and there is no flow failure,
# so the |Calo v2| and flow failure cuts of hJetPtv3 are not applied.
EVENT_INFO = (
    r"$|z| < 10$ cm & MB" "\n"
    "Good Calo-Cent\n"
    r"Centrality: 0–60%" "\n"
    "Iterative No Flow"
)

TIME_LABEL = r"Jet $t - t_{0}^{\mathrm{MBD}}$ [ns]"
FRAC_LABEL = r"OHCal Energy Fraction $x$"
PT_LABEL = r"Leading jet raw $p_{T}$ [GeV]"
MODEL_LABEL = r"$t = f_{c}(x, p_{T}) + a_{\mathrm{run}} + b_{\mathrm{run}}\,x$"

# The model ladder: (key, label, shape kind, per-run terms: 0 none, 1 offset a, 2 line a + b x)
LADDER = (
    ("M1", r"M1: $f(x)$", "x", 0),
    ("M2", r"M2: $f(x) + a_{\mathrm{run}}$", "x", 1),
    ("M3", r"M3: $f(x) + a_{\mathrm{run}} + b_{\mathrm{run}}x$", "x", 2),
    ("M4a", r"M4a: $f_{c}(x)$ + line", "x_cent", 2),
    ("M4b", r"M4b: $f(x, p_{T})$ + line", "x_pt", 2),
    ("M5", r"M5: $f_{c}(x, p_{T})$ + line", "x_pt_cent", 2),
)
LADDER_COLORS = {"M1": "#7f7f7f", "M2": "#377eb8", "M3": "#4daf4a", "M4a": "#ff7f00", "M4b": "#984ea3", "M5": "#e41a1c"}
FINAL = "M5"


def pt_hist_name(r_tag):
    return f"hJetPtv3_raw_{r_tag}_iternf"


def timing_hist_name(r_tag, pt=TIMING_PT):
    return f"h2LeadJetTimeOHCalFrac_{r_tag}_iternf_pt{pt[0]}_{pt[1]}"


def timing_status_name(r_tag):
    return f"hLeadJetTimingStatus_{r_tag}_iternf"


def profile_name(r_tag, cent):
    return f"p2LeadJetTimeOHCalFracRawPt_{r_tag}_iternf_cent_{cent[0]}_{cent[1]}"


# ---------------------------------------------------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------------------------------------------------

def th2_column_moments(h):
    """Per x bin of a TH2: [sum w, sum w t, sum w t^2, sum w^2] with t the y bin center. Shape (4, n_x)."""
    v = h.values().astype(np.float64)
    w2 = h.variances().astype(np.float64)
    ye = h.axes[1].edges()
    yc = 0.5 * (ye[:-1] + ye[1:])
    return np.stack([v.sum(1), (v * yc).sum(1), (v * yc ** 2).sum(1), w2.sum(1)])


def profile2d_moments(p):
    """Raw sums of a TProfile2D per (y = raw pT, x) cell, without flow bins: shape (4, n_y, n_x).

    [sum w (fBinEntries), sum w t (the TArrayD of the TH2D base), sum w t^2 (fSumw2), sum w^2 (fBinSumw2)].
    ROOT stores cells as global bin = ix + (nx + 2) * iy.
    """
    nx = p.member("fXaxis").member("fNbins")
    ny = p.member("fYaxis").member("fNbins")
    swt = next(np.asarray(b) for b in p.bases[0].bases if "TArray" in type(b).__name__)
    shape = (ny + 2, nx + 2)
    arrays = [np.asarray(p.member("fBinEntries")), swt, np.asarray(p.member("fSumw2")), np.asarray(p.member("fBinSumw2"))]
    return np.stack([np.asarray(a, dtype=np.float64).reshape(shape)[1:-1, 1:-1] for a in arrays])


CACHE_VERSION = 1   # bump when load_run's output changes, so old caches are not reused


def cache_key(tasks, pt_radii, t_radii, lumi_paths):
    """Hash of what the inputs depend on: the run files (path, size, mtime), the lumi files, the radii read."""
    h = hashlib.sha1(f"v{CACHE_VERSION}|{pt_radii}|{t_radii}".encode())
    for run, path, *_ in tasks:
        st = path.stat()
        h.update(f"{run}|{path}|{st.st_size}|{st.st_mtime_ns}\n".encode())
    for path in lumi_paths:
        st = path.stat()
        h.update(f"L|{path}|{st.st_size}|{st.st_mtime_ns}\n".encode())
    return h.hexdigest()[:16]


class StageTimer:
    """Prints the wall time of each stage."""

    def __init__(self):
        self.t0 = self.last = time.time()

    def __call__(self, label):
        now = time.time()
        print(f"[time] {label}: {now - self.last:.0f} s (total {now - self.t0:.0f} s)")
        self.last = now


def load_run(task):
    """Reads the per-run histograms. Returns (run, data) where data is None if a histogram is missing."""
    run, file_path, pt_radii, t_radii = task
    data = {}
    try:
        with uproot.open(file_path) as rf:
            data["n_events"] = float(rf["hCentrality"].values().sum() - rf["hCentralityCaloFail"].values().sum())
            for r_tag in pt_radii:
                h_pt = rf[pt_hist_name(r_tag)]
                # float64: the TH1F contents are float32, and the sums over thousands of runs exceed its precision
                data[r_tag] = {
                    "pt_values": h_pt.values().astype(np.float64),
                    "pt_variances": h_pt.variances().astype(np.float64),
                    "pt_edges": h_pt.axis().edges(),
                }
            for r_tag in t_radii:
                h_t = rf[timing_hist_name(r_tag)]
                t_flow = h_t.values(flow=True)
                profiles = [rf[profile_name(r_tag, c)] for c in CENT_CLASSES]
                d = data.setdefault(r_tag, {})
                d.update({
                    "t_values": h_t.values().astype(np.float64),   # full 2D, only summed for the combined plot
                    "t_mom": th2_column_moments(h_t),
                    "t_xedges": h_t.axes[0].edges(),
                    "t_yedges": h_t.axes[1].edges(),
                    "t_filled": float(rf[timing_status_name(r_tag)].values()[1]),
                    # Leading jets outside the plotted range: OHCal fraction below / above the axis, |t - t0| > 25 ns
                    "t_flow": np.array([t_flow[0].sum(), t_flow[-1].sum(),
                                        t_flow[1:-1, 0].sum() + t_flow[1:-1, -1].sum()]),
                    "p2_mom": np.stack([profile2d_moments(p) for p in profiles], axis=1),   # (4, n_cent, n_pt, n_x)
                    "p2_xedges": profiles[0].member("fXaxis").edges(),
                    "p2_ptedges": profiles[0].member("fYaxis").edges(),
                })
    except (KeyError, OSError, StopIteration) as e:
        print(f"Warning: run {run}: {e}, skipping")
        return run, None
    return run, data


# ---------------------------------------------------------------------------------------------------------------------
# Cells and the fit
# ---------------------------------------------------------------------------------------------------------------------

def build_cells(mom, xedges, ptedges, args):
    """Flattens per-run profile moments (n_runs, 4, n_cent, n_pt, n_x) into cells, with the fit inputs.

    Returns dict: mom (n_runs, 4, B), y, e, m (n_runs, B), n (jets per run-cell), co (cell coordinates), used (B,),
    run_entries (jets per run in used cells).
    """
    R = mom.shape[0]
    xc = 0.5 * (xedges[:-1] + xedges[1:])
    pc = 0.5 * (ptedges[:-1] + ptedges[1:])
    kk, pp, xx = np.meshgrid(np.arange(mom.shape[2]), pc, xc, indexing="ij")
    ii, jj = np.meshgrid(np.arange(len(pc)), np.arange(len(xc)), indexing="ij")
    co = {"k": kk.ravel(), "pt": pp.ravel(), "x": xx.ravel(),
          "ipt": np.broadcast_to(ii, kk.shape).ravel(), "ix": np.broadcast_to(jj, kk.shape).ravel()}
    mom = mom.reshape(R, 4, -1)
    s0, s1, s2, w2 = mom[:, 0], mom[:, 1], mom[:, 2], mom[:, 3]
    S0, S1, S2 = s0.sum(0), s1.sum(0), s2.sum(0)
    with np.errstate(divide="ignore", invalid="ignore"):
        sigma = np.sqrt(np.clip(S2 / S0 - (S1 / S0) ** 2, 0, None))        # pooled time RMS per cell
        y = s1 / s0
        n = np.where(w2 > 0, s0 ** 2 / w2, 0.0)
        e = sigma[None, :] / np.sqrt(n)
    used = (S0 >= args.min_cell_entries) & (sigma > 0) & (co["pt"] >= args.pt_range[0]) & (co["pt"] < args.pt_range[1])
    if args.fit_range is not None:
        used &= (co["x"] >= args.fit_range[0]) & (co["x"] <= args.fit_range[1])
    m = used[None, :] & (s0 > 0) & np.isfinite(y) & np.isfinite(e) & (e > 0)
    run_entries = np.where(m, s0, 0.0).sum(1)
    m &= (run_entries >= args.min_run_entries)[:, None]
    # contiguous copies of sum w, sum w t, sum w t^2 for the fast corrected sums (matrix-vector products)
    return {"S": [np.ascontiguousarray(a) for a in (s0, s1, s2)], "y": y, "e": e, "m": m, "n": s0, "co": co,
            "used": used, "run_entries": run_entries, "sigma": sigma, "xc": xc, "pc": pc}


def _line_ab(line, R):
    """Per-run offset a and slope b (0 where absent), and the runs that have the terms."""
    n_line = line.shape[1]
    ok = np.isfinite(line).all(1) if n_line else np.ones(R, bool)
    a = np.nan_to_num(line[:, 0]) if n_line >= 1 else np.zeros(R)
    b = np.nan_to_num(line[:, 1]) if n_line >= 2 else np.zeros(R)
    return a, b, ok


def design(co, kind, deg_x, deg_pt, n_cent):
    """Shape columns: 'x' f(x); 'x_cent' f_c(x); 'x_pt' f(x, pT); 'x_pt_cent' f_c(x, pT). Powers of x and ln(pT/15)."""
    x, L, k = co["x"], np.log(co["pt"] / PT_REF), co["k"]
    jp = range(deg_pt + 1) if kind in ("x_pt", "x_pt_cent") else range(1)
    classes = range(n_cent) if kind in ("x_cent", "x_pt_cent") else [None]
    cols = [(np.ones_like(x) if c is None else (k == c).astype(float)) * x ** i * L ** j
            for c in classes for i in range(deg_x + 1) for j in jp]
    return np.vstack(cols).T


def line_basis(co, n_line):
    """Per-run term basis per cell: [] (none), [1] (offset) or [1, x] (line). Shape (B, n_line)."""
    if n_line == 0:
        return np.zeros((len(co["x"]), 0))
    return np.vstack([np.ones_like(co["x"]), co["x"]][:n_line]).T


def weights(cells):
    m, e = cells["m"], cells["e"]
    return np.where(m, 1.0 / np.where(m, e, 1.0) ** 2, 0.0)


def finish(cells, w, s, line, L, n_params, n_fitted_per_run, ok):
    """Residuals, chi2 and ndf of a fit; per-run chi2 and points.

    The per-run terms (line, basis L) always enter the residuals; n_fitted_per_run is how many of them were fitted
    (and so count against ndf): n_line for a joint fit, 0 when they are held fixed.
    """
    y0 = np.where(cells["m"], cells["y"], 0.0)
    model = s[None, :] + (np.nan_to_num(line) @ L.T if L.shape[1] else 0.0)
    res = (y0 - model) * np.sqrt(w)
    used = w > 0
    npts = int(used.sum())
    return {"s": s, "line": line, "ok": ok, "chi2": float((res ** 2).sum()),
            "ndf": int(npts - n_fitted_per_run * int(ok.sum()) - int(n_params)), "pulls": res[used],
            "run_chi2": (res ** 2).sum(1), "run_npts": used.sum(1)}


def fit_model(cells, X, n_line):
    """y = X c + per-run terms (n_line = 0 none, 1 offset, 2 line); exact, by projecting the per-run terms out."""
    w = weights(cells)
    y0 = np.where(cells["m"], cells["y"], 0.0)
    co = cells["co"]
    R = len(y0)
    if n_line == 0:
        N = X.T @ (w.sum(0)[:, None] * X)
        rhs = X.T @ (w * y0).sum(0)
        Ni = np.linalg.pinv(N, hermitian=True)
        c = Ni @ rhs
        out = finish(cells, w, X @ c, np.zeros((R, 0)), np.zeros((len(co["x"]), 0)),
                     np.linalg.matrix_rank(N, tol=1e-9 * np.abs(N).max()), 0, np.ones(R, bool))
        out.update({"c": c, "cov": Ni})
        return out
    L = line_basis(co, n_line)
    G = np.einsum("rb,bi,bj->rij", w, L, L)
    ok = (np.linalg.det(G) > 1e-12) if n_line > 1 else (G[:, 0, 0] > 0)
    G[~ok] = np.eye(n_line)
    w[~ok] = 0.0
    Gi = np.linalg.inv(G)
    A = np.stack([(w * L[:, j]) @ X for j in range(n_line)], -1)                 # (R, K, n_line)
    ly = np.stack([(w * L[:, j] * y0).sum(1) for j in range(n_line)], -1)        # (R, n_line)
    N = X.T @ (w.sum(0)[:, None] * X) - np.einsum("rkj,rjl,rml->km", A, Gi, A)
    rhs = X.T @ (w * y0).sum(0) - np.einsum("rkj,rjl,rl->k", A, Gi, ly)
    Ni = np.linalg.pinv(N, rcond=1e-12, hermitian=True)
    c = Ni @ rhs
    s = X @ c
    line = np.einsum("rij,rj->ri", Gi, np.stack([(w * L[:, j] * (y0 - s)).sum(1) for j in range(n_line)], -1))
    line[~ok] = np.nan
    out = finish(cells, w, s, line, L, np.linalg.matrix_rank(N, tol=1e-9 * np.abs(N).max()), n_line, ok)
    out.update({"c": c, "line_cov": np.where(ok[:, None, None], Gi, np.nan)})
    return out


def fit_fixed_lines(cells, X, line):
    """Shape only, with the per-run terms (n_runs, n_line) held fixed (runs without them are dropped)."""
    w = weights(cells)
    n_line = line.shape[1]
    L = line_basis(cells["co"], n_line)
    ok = np.isfinite(line).all(1)
    w[~ok] = 0.0
    z = np.where(w > 0, cells["y"] - (np.nan_to_num(line) @ L.T if n_line else 0.0), 0.0)
    N = X.T @ (w.sum(0)[:, None] * X)
    rhs = X.T @ (w * z).sum(0)
    Ni = np.linalg.pinv(N, hermitian=True)
    c = Ni @ rhs
    out = finish(cells, w, X @ c, line, L, np.linalg.matrix_rank(N, tol=1e-9 * np.abs(N).max()), 0, ok)
    out.update({"c": c, "cov": Ni})
    return out


def fit_free_cells(cells, n_line=2, line_fixed=None, mask=None, n_iter=500, tol=1e-10):
    """Best possible shape: a free value in every cell, plus the per-run terms (alternating exact updates)."""
    w = weights(cells)
    if mask is not None:
        w = np.where(mask[None, :], w, 0.0)
    y0 = np.where(w > 0, cells["y"], 0.0)
    co = cells["co"]
    W = w.sum(0)
    if line_fixed is not None:
        n_line = line_fixed.shape[1]
        L = line_basis(co, n_line)
        ok = np.isfinite(line_fixed).all(1)
        w[~ok] = 0.0
        W = w.sum(0)
        z = y0 - np.nan_to_num(line_fixed) @ L.T
        s = np.where(W > 0, (w * z).sum(0) / np.where(W > 0, W, 1), 0.0)
        res = (z - s[None]) * np.sqrt(w)
        npts = int((w > 0).sum())
        return {"chi2": float((res ** 2).sum()), "ndf": npts - int((W > 0).sum()), "line": line_fixed, "s": s}
    L = line_basis(co, n_line)
    G = np.einsum("rb,bi,bj->rij", w, L, L)
    ok = np.linalg.det(G) > 1e-12
    G[~ok] = np.eye(n_line)
    w[~ok] = 0.0
    W = w.sum(0)
    Gi = np.linalg.inv(G)
    s = np.zeros(len(co["x"]))
    last = None
    for _ in range(n_iter):
        line = np.einsum("rij,rj->ri", Gi, np.stack([(w * L[:, j] * (y0 - s)).sum(1) for j in range(n_line)], -1))
        s = np.where(W > 0, (w * (y0 - line @ L.T)).sum(0) / np.where(W > 0, W, 1), 0.0)
        chi2 = float((((y0 - s[None] - line @ L.T) ** 2) * w).sum())
        if last is not None and abs(last - chi2) <= tol * chi2:
            break
        last = chi2
    npts = int((w > 0).sum())
    line[~ok] = np.nan
    return {"chi2": chi2, "ndf": npts - n_line * int(ok.sum()) - int((W > 0).sum()) + n_line, "line": line,
            "line_cov": np.where(ok[:, None, None], Gi, np.nan), "s": s, "ok": ok}


def center_lines(line, weights_run):
    """Subtracts the jet-weighted mean of each per-run term (the convention); returns (centered, means)."""
    ok = np.isfinite(line).all(1)
    wr = np.where(ok, weights_run, 0.0)
    means = (np.nan_to_num(line) * wr[:, None]).sum(0) / wr.sum()
    return line - means[None, :], means


# ---------------------------------------------------------------------------------------------------------------------
# Closure (pooled over runs, from moments)
# ---------------------------------------------------------------------------------------------------------------------

def pooled_corrected(cells, s, line):
    """Per cell, summed over the runs that have per-run terms: moments of t - d, d = s + a_run + b_run x.

    Expanded into matrix-vector products of the stored moments, so no (runs x cells) temporaries are made.
    Returns (c0, c1, c2) of shape (B,).
    """
    s0, s1, s2 = cells["S"]
    x = cells["co"]["x"]
    a, b, ok = _line_ab(line, len(s0))
    sv = np.zeros_like(x) if s is None else s
    w = ok.astype(float)
    wa, wb = w * a, w * b
    A0, A1, A2 = w @ s0, w @ s1, w @ s2
    Sa0, Sb0, Sa1, Sb1 = wa @ s0, wb @ s0, wa @ s1, wb @ s1
    Saa0, Sbb0, Sab0 = (wa * a) @ s0, (wb * b) @ s0, (wa * b) @ s0
    c1 = A1 - (sv * A0 + Sa0 + x * Sb0)
    c2 = (A2 - 2 * (sv * A1 + Sa1 + x * Sb1)
          + sv ** 2 * A0 + Saa0 + x ** 2 * Sbb0 + 2 * sv * Sa0 + 2 * sv * x * Sb0 + 2 * x * Sab0)
    return A0, c1, c2


def per_run_corrected_rms(cells, s, line):
    """Per run, over the used cells: RMS of t - d, d = s + a_run + b_run x (same expansion as pooled_corrected)."""
    s0, s1, s2 = cells["S"]
    x = cells["co"]["x"]
    u = cells["used"].astype(float)
    a, b, _ = _line_ab(line, len(s0))
    sv = np.zeros_like(x) if s is None else s
    T0, T1, T2 = s0 @ u, s1 @ u, s2 @ u
    Us0, Ux0, Us1, Ux1 = s0 @ (u * sv), s0 @ (u * x), s1 @ (u * sv), s1 @ (u * x)
    Uss0, Uxx0, Usx0 = s0 @ (u * sv ** 2), s0 @ (u * x ** 2), s0 @ (u * sv * x)
    n1 = T1 - (Us0 + a * T0 + b * Ux0)
    n2 = T2 - 2 * (Us1 + a * T1 + b * Ux1) + (Uss0 + a ** 2 * T0 + b ** 2 * Uxx0 + 2 * a * Us0 + 2 * b * Usx0 + 2 * a * b * Ux0)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.sqrt(np.clip(n2 / T0 - (n1 / T0) ** 2, 0, None))


def closure_stats(cells, c0, c1, c2):
    """Jet-weighted |bias|, its statistical floor, and the pooled RMS in each PT_RANGES range; the worst bias."""
    co, used = cells["co"], cells["used"]
    with np.errstate(divide="ignore", invalid="ignore"):
        sig = np.sqrt(np.clip(c2 / c0 - (c1 / c0) ** 2, 0, None))
        floor_cell = np.sqrt(2 / np.pi) * sig * np.sqrt(c0)         # N * E|noise of the mean| = sqrt(2/pi) sigma sqrt(N)
    out = {"bias": [], "floor": [], "rms": []}
    for lo, hi in PT_RANGES:
        sel = used & (co["pt"] >= lo) & (co["pt"] < hi) & (c0 > 0)
        N = c0[sel].sum()
        out["bias"].append(np.abs(c1[sel]).sum() / N if N else np.nan)
        out["floor"].append(np.nansum(floor_cell[sel]) / N if N else np.nan)
        out["rms"].append(np.sqrt(c2[sel].sum() / N - (c1[sel].sum() / N) ** 2) if N else np.nan)
    g0, g1, _, _ = group_sums(cells, c0, c1, c2)
    big = g0 >= 5000
    out["worst"] = float(np.max(np.abs(g1[big] / g0[big]))) if big.any() else 0.0
    return out


def group_sums(cells, c0, c1, c2):
    """Sums of the pooled moments per (class, pT bin, x slice) group, over the used cells (one bincount pass).

    Returns g0, g1, g2 of shape (n_cent, n_pt, n_slices) and the group shape.
    """
    co, used = cells["co"], cells["used"]
    edges = np.array([lo for lo, _ in X_SLICES] + [X_SLICES[-1][1]])
    si = np.searchsorted(edges, co["x"], side="right") - 1
    in_slice = (si >= 0) & (si < len(X_SLICES)) & (co["x"] < edges[-1])
    nK, nP, nS = int(co["k"].max()) + 1, len(cells["pc"]), len(X_SLICES)
    sel = used & in_slice
    gid = (co["k"] * nP + co["ipt"]) * nS + np.clip(si, 0, nS - 1)
    shape = (nK, nP, nS)
    sums = [np.bincount(gid[sel], weights=c[sel], minlength=nK * nP * nS).reshape(shape) for c in (c0, c1, c2)]
    return (*sums, shape)


def closure_map(cells, c0, c1, c2, min_jets=500):
    """Pooled bias and its error per (class, pT bin, x slice): dict[(k, slice index)] -> (pt, bias, err)."""
    g0, g1, g2, (nK, nP, nS) = group_sums(cells, c0, c1, c2)
    out = {}
    with np.errstate(divide="ignore", invalid="ignore"):
        mu = g1 / g0
        err = np.sqrt(np.clip(g2 / g0 - mu ** 2, 0, None) / g0)
    for k in range(nK):
        for si in range(nS):
            ok = g0[k, :, si] >= min_jets
            out[(k, si)] = (cells["pc"][ok], mu[k, ok, si], err[k, ok, si])
    return out


# ---------------------------------------------------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------------------------------------------------

def add_header(ax, runs, r_jet, lumi, x=1.0, extra=""):
    text = rf"{len(runs)} runs ({min(runs)}–{max(runs)}), $R = {r_jet:g}$"
    if lumi is not None:
        text += rf", $\mathcal{{L}} = {lumi:.1f}\ \mu\mathrm{{b}}^{{-1}}$"
    ax.text(x, 1.01, text + extra, transform=ax.transAxes, ha="right", va="bottom", fontsize=13)


def pt_info(eta_max):
    return (f"{EVENT_INFO}\n"
            rf"$|\eta| < {eta_max:g}$, Energy > 0" "\n"
            rf"Calib. $p_{{T}} \geq {JET_PT_MIN}$ GeV")


def timing_info(eta_max, pt=TIMING_PT):
    return (f"{EVENT_INFO}\n"
            rf"Leading jet: $|\eta| < {eta_max:g}$, Energy > 0" "\n"
            rf"Raw ${pt[0]} \leq p_{{T}} < {pt[1]}$ GeV")


def sci(x):
    """Mathtext scientific notation, e.g. 3.18 \\times 10^{8}."""
    exp = int(np.floor(np.log10(x))) if x > 0 else 0
    return rf"{x / 10 ** exp:.2f} \times 10^{{{exp}}}"


def line_source_text(r_tag, src):
    return "own fit" if r_tag == src else rf"from $R = {RADII[src]['r']:g}$"


def cent_label(c):
    return f"{CENT_CLASSES[c][0]}–{CENT_CLASSES[c][1]}%"


def setup_run_axis(ax, runs):
    # Label every run when few, otherwise ~40 evenly spaced runs (x is the run index, so gaps between runs are not stretched)
    step = max(1, int(np.ceil(len(runs) / 40)))
    ticks = np.arange(0, len(runs), step)
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(runs[i]) for i in ticks], rotation=90, fontsize=7)
    ax.set_xlim(-1, len(runs))
    ax.set_xlabel("Run", labelpad=6)
    ax.tick_params(axis="x", which="minor", bottom=False, top=False)


def write_csv(rows, keys, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved: {path}")


# ---------------------------------------------------------------------------------------------------------------------
# Plot 1: raw pT
# ---------------------------------------------------------------------------------------------------------------------

def plot_raw_pt(values, variances, edges, n_events, runs, r_tag, lumi, per_event, output_path, pt_max=None, save_pdf=False):
    hep.style.use("ATLAS")
    meta = RADII[r_tag]

    scale = 1.0 / (n_events * np.diff(edges)) if per_event else 1.0
    vals = values * scale
    errs = np.sqrt(variances) * scale

    nz = np.where(values > 0)[0]
    max_x_val = edges[nz[-1] + 1] if len(nz) else 60

    fig, ax = plt.subplots(figsize=(8, 7))
    hep.histplot((vals, edges), yerr=errs, ax=ax, histtype="step", color="#e41a1c", linewidth=2, label="Iterative No Flow")

    positive = vals[vals > 0]
    y_max = positive.max() if len(positive) else 1.0
    y_min = positive.min() * 0.5 if len(positive) else 0.5
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=20))
    ax.set_ylim(bottom=y_min, top=y_max * 50)
    ax.set_xlim(0, pt_max if pt_max is not None else max_x_val * 1.04)
    ax.set_xlabel(r"Jet raw $p_{T}$ [GeV]")
    ax.set_ylabel(r"$\frac{1}{N_{\mathrm{evt}}}\frac{dN}{dp_{T}}$ [GeV$^{-1}$]" if per_event else "Counts")

    add_header(ax, runs, meta["r"], lumi)
    ax.text(0.95, 0.95, pt_info(meta["eta_max"]), transform=ax.transAxes, ha="right", va="top", fontsize=14)
    ax.text(0.05, 0.05, rf"$N_{{\mathrm{{evt}}}} = {sci(n_events)}$, $N_{{\mathrm{{jets}}}} = {sci(values.sum())}$",
            transform=ax.transAxes, ha="left", va="bottom", fontsize=13)
    plt.subplots_adjust(left=0.15, right=0.96, bottom=0.12, top=0.94)
    save_fig(fig, output_path, save_pdf)
    plt.close(fig)

    if pt_max is None and max_x_val > 60:
        plot_raw_pt(values, variances, edges, n_events, runs, r_tag, lumi, per_event,
                    output_path.with_name(f"{output_path.stem}_pt60{output_path.suffix}"), pt_max=60, save_pdf=save_pdf)


# ---------------------------------------------------------------------------------------------------------------------
# Plot 2: combined timing vs OHCal fraction (15-40 GeV TH2) with the model
# ---------------------------------------------------------------------------------------------------------------------

def frac_outside_unit(values, xedges, flow):
    """Leading jets with OHCal fraction < 0 and > 1: bins of the histogram outside [0, 1] plus its x under/overflow."""
    xc = 0.5 * (xedges[:-1] + xedges[1:])
    return values[xc < 0].sum() + flow[0], values[xc > 1].sum() + flow[1]


def profile_from_moments(mom):
    """ProfileX from column moments (4, n_x): mean, error on the mean, N_eff."""
    s0, s1, s2, sw2 = mom
    with np.errstate(divide="ignore", invalid="ignore"):
        mean = s1 / s0
        neff = np.where(sw2 > 0, s0 ** 2 / sw2, 0.0)
        err = np.sqrt(np.clip(s2 / s0 - mean ** 2, 0, None) / neff)
    return mean, err, neff


def plot_timing(values, xedges, yedges, prof_raw, corr_x, model_x, runs, r_tag, src, lumi, output_path, flow,
                time_ylim=None, save_pdf=False):
    """The 15-40 GeV TH2, its profile, the profile of t - a - b x, and the model averaged over the jets in each x bin."""
    hep.style.use("ATLAS")
    meta = RADII[r_tag]
    xc = 0.5 * (xedges[:-1] + xedges[1:])

    fig, ax = plt.subplots(figsize=(9, 7))
    z = np.where(values > 0, values, np.nan)
    pcm = ax.pcolormesh(xedges, yedges, z.T, cmap="viridis", norm=LogNorm(vmin=1, vmax=np.nanmax(z)), rasterized=True)
    cbar = fig.colorbar(pcm, ax=ax, pad=0.02)
    cbar.set_label("Counts")

    raw_mean, raw_err, _ = prof_raw
    ok = np.isfinite(raw_mean)
    ax.errorbar(xc[ok], raw_mean[ok], yerr=np.nan_to_num(raw_err[ok]), fmt="o", mfc="white", mec="gray", color="gray",
                markersize=4, capsize=0, linewidth=1.0, label="Profile", zorder=3)
    c_mean, c_err = corr_x
    ok = np.isfinite(c_mean)
    ax.errorbar(xc[ok], c_mean[ok], yerr=c_err[ok], fmt="o", color="black", markersize=4, capsize=0, linewidth=1.2,
                label=r"Profile of $t - a_{\mathrm{run}} - b_{\mathrm{run}} x$", zorder=4)
    ok = np.isfinite(model_x)
    ax.plot(xc[ok], model_x[ok], "-", color="#e41a1c", linewidth=2, label=r"$f_{c}(x, p_{T})$, averaged over the jets",
            zorder=5)

    ax.set_xlim(xedges[0], xedges[-1])
    ax.set_ylim(*(time_ylim if time_ylim is not None else (yedges[0], yedges[-1])))
    ax.set_xlabel(FRAC_LABEL)
    ax.set_ylabel(TIME_LABEL)
    box = dict(facecolor="white", alpha=0.8, edgecolor="none", pad=3)
    add_header(ax, runs, meta["r"], lumi)
    ax.text(0.97, 0.97, timing_info(meta["eta_max"]), transform=ax.transAxes, ha="right", va="top", fontsize=12, bbox=box)
    ax.text(0.03, 0.03, MODEL_LABEL + "\n" + f"Per-run lines: {line_source_text(r_tag, src)}\n"
            + r"$f_{c}$ fitted on raw $p_{T}$ 10–60 GeV", transform=ax.transAxes, ha="left", va="bottom", fontsize=11,
            bbox=box)
    n_neg, n_over = frac_outside_unit(values, xedges, flow)
    total = values.sum() + flow.sum()
    ax.text(0.97, 0.03,
            f"Entries: {values.sum():.0f} in range / {total:.0f}\n"
            f"OHCal frac < 0: {n_neg:.0f} ({100 * n_neg / total:.1f}%)\n"
            f"OHCal frac > 1: {n_over:.0f} ({100 * n_over / total:.2f}%)\n"
            rf"$|t - t_{{0}}| > 25$ ns: {flow[2]:.0f}",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=11, bbox=box)
    ax.legend(loc="upper left", fontsize=11, frameon=True, framealpha=0.8, edgecolor="none")
    plt.subplots_adjust(left=0.13, right=0.97, bottom=0.12, top=0.94)
    save_fig(fig, output_path, save_pdf)
    plt.close(fig)


# ---------------------------------------------------------------------------------------------------------------------
# Plot 3: the model ladder
# ---------------------------------------------------------------------------------------------------------------------

def plot_ladder_chi2(ladder, floors, runs, lumi, output_path, save_pdf):
    """L1: chi2/ndf and pull width of each ladder model, for each radius, with the best-possible-shape floor."""
    hep.style.use("ATLAS")
    keys = [k for k, *_ in LADDER]
    labels = [lab for _, lab, *_ in LADDER]
    x = np.arange(len(keys))
    fig, (ax, axp) = plt.subplots(2, 1, figsize=(11, 9), sharex=True, gridspec_kw={"height_ratios": [2, 1], "hspace": 0.06})
    for r_tag, marker, color in (("r03", "s", "#e41a1c"), ("r02", "o", "#377eb8")):
        if r_tag not in ladder:
            continue
        chi = [ladder[r_tag][k]["chi2"] / ladder[r_tag][k]["ndf"] for k in keys]
        ps = [np.std(ladder[r_tag][k]["pulls"]) for k in keys]
        ax.plot(x, chi, marker=marker, color=color, markersize=9, linewidth=1.5, label=rf"$R = {RADII[r_tag]['r']:g}$")
        for xi, yi in zip(x, chi):          # R = 0.3 labels above, R = 0.2 below the markers
            above = r_tag == "r03"
            ax.text(xi, yi * (1.15 if above else 0.87), f"{yi:.3f}", ha="center", va="bottom" if above else "top",
                    fontsize=10, color=color)
        axp.plot(x, ps, marker=marker, color=color, markersize=8, linewidth=1.5)
        if r_tag in floors:
            ax.axhline(floors[r_tag], color=color, linestyle=":", linewidth=1.5,
                       label=rf"Best possible shape (free per cell), $R = {RADII[r_tag]['r']:g}$")
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1)
    axp.axhline(1.0, color="gray", linestyle="--", linewidth=1)
    ax.set_yscale("log")
    ax.set_ylim(0.6, ax.get_ylim()[1] * 2.5)
    ax.set_ylabel(r"$\chi^{2}/\mathrm{ndf}$")
    axp.set_ylabel("Pull std")
    axp.set_xticks(x)
    axp.set_xticklabels(labels, rotation=20, ha="right", fontsize=11)
    ax.legend(loc="upper right", frameon=False, fontsize=11)
    ax.text(0.02, 0.97, "Raw $p_{T}$ 10–60 GeV, 3 centrality classes\n$R = 0.2$ uses the per-run terms of $R = 0.3$",
            transform=ax.transAxes, ha="left", va="top", fontsize=11)
    add_header(ax, runs, 0.3, lumi, extra=r" and $R = 0.2$")
    plt.subplots_adjust(left=0.1, right=0.97, bottom=0.2, top=0.94)
    save_fig(fig, output_path, save_pdf)
    plt.close(fig)


def plot_ladder_bias(stats, runs, r_tag, lumi, output_path, save_pdf):
    """L2: jet-weighted |bias| per pT range for each ladder model, with the statistical floor."""
    hep.style.use("ATLAS")
    x = np.arange(len(PT_RANGES))
    fig, ax = plt.subplots(figsize=(10, 7))
    for key, label, *_ in LADDER:
        ax.plot(x, stats[key]["bias"], marker="o", color=LADDER_COLORS[key], linewidth=2, markersize=8, label=label)
    ax.plot(x, stats[FINAL]["floor"], "--", color="black", linewidth=1.5, label="Statistical floor (noise of the cell means)")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{lo}–{hi}" for lo, hi in PT_RANGES])
    ax.set_xlabel(PT_LABEL)
    ax.set_ylabel(r"Jet-weighted $|\langle t\rangle_{\mathrm{data}} - \langle t\rangle_{\mathrm{model}}|$ per cell [ns]")
    ax.set_ylim(ax.get_ylim()[0] * 0.7, ax.get_ylim()[1] * 4)
    ax.legend(loc="upper left", frameon=False, fontsize=10, ncol=2)
    add_header(ax, runs, RADII[r_tag]["r"], lumi)
    plt.subplots_adjust(left=0.13, right=0.97, bottom=0.12, top=0.94)
    save_fig(fig, output_path, save_pdf)
    plt.close(fig)


def plot_ladder_closure(cmap, ylim, runs, r_tag, label, lumi, output_path, save_pdf):
    """L3: pooled data - model vs raw pT, in x slices, for each centrality class (one ladder model)."""
    hep.style.use("ATLAS")
    fig, axes = plt.subplots(1, len(CENT_CLASSES), figsize=(17, 6), sharey=True, gridspec_kw={"wspace": 0.05})
    for k, ax in enumerate(axes):
        ax.axhspan(-0.05, 0.05, color="gray", alpha=0.15, linewidth=0)
        ax.axhline(0.0, color="gray", linestyle="--", linewidth=1)
        for si, ((lo, hi), color) in enumerate(zip(X_SLICES, X_SLICE_COLORS)):
            pt, b, err = cmap[(k, si)]
            if len(pt):
                ax.errorbar(pt * (1 + 0.006 * (si - 2)), b, yerr=err, fmt="o-", color=color, markersize=4, linewidth=1.0,
                            capsize=0, label=rf"${lo:g} \leq x < {hi:g}$")
        ax.set_xscale("log")
        ax.set_xlim(9.5, 62)
        ax.set_xticks([10, 15, 20, 30, 40, 60])
        ax.set_xticklabels(["10", "15", "20", "30", "40", "60"])
        ax.set_xlabel(PT_LABEL)
        ax.text(0.04, 0.95, cent_label(k), transform=ax.transAxes, ha="left", va="top", fontsize=14)
    axes[0].set_ylim(*ylim)
    axes[0].set_ylabel(r"$\langle t \rangle_{\mathrm{data}} - \langle t \rangle_{\mathrm{model}}$ [ns]")
    axes[-1].legend(loc="lower left", frameon=False, fontsize=10)
    axes[1].text(0.5, 0.95, label, transform=axes[1].transAxes, ha="center", va="top", fontsize=14)
    add_header(axes[-1], runs, RADII[r_tag]["r"], lumi)
    plt.subplots_adjust(left=0.07, right=0.98, bottom=0.13, top=0.93)
    save_fig(fig, output_path, save_pdf)
    plt.close(fig)


def plot_ladder_rms(stats, raw_rms, runs, r_tag, lumi, output_path, save_pdf):
    """L4: timing RMS pooled over runs, per pT range, raw and after each ladder model."""
    hep.style.use("ATLAS")
    x = np.arange(len(PT_RANGES))
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.plot(x, raw_rms, marker="x", color="black", linewidth=1.5, markersize=9, label=r"raw $t - t_{0}$")
    for key, label, *_ in LADDER:
        ax.plot(x, stats[key]["rms"], marker="o", color=LADDER_COLORS[key], linewidth=1.5, markersize=7, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{lo}–{hi}" for lo, hi in PT_RANGES])
    ax.set_xlabel(PT_LABEL)
    ax.set_ylabel("Timing RMS, all runs pooled [ns]")
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi + 0.6 * (hi - lo))
    ax.legend(loc="upper left", frameon=False, fontsize=10, ncol=2)
    add_header(ax, runs, RADII[r_tag]["r"], lumi)
    plt.subplots_adjust(left=0.12, right=0.97, bottom=0.12, top=0.94)
    save_fig(fig, output_path, save_pdf)
    plt.close(fig)


def plot_ladder_pulls(pull_sets, runs, r_tag, lumi, output_path, save_pdf):
    """L5: pulls of the run-cell means for each ladder model."""
    hep.style.use("ATLAS")
    fig, ax = plt.subplots(figsize=(10, 7))
    allp = np.concatenate([p for _, _, p in pull_sets])
    plim = max(5.0, min(30.0, np.ceil(np.percentile(np.abs(allp), 99.5))))
    bins = np.linspace(-plim, plim, 81)
    bw = bins[1] - bins[0]
    top = 1.0
    for key, label, p in pull_sets:
        counts, *_ = ax.hist(np.clip(p, bins[0] + 1e-9, bins[-1] - 1e-9), bins=bins, histtype="step",
                             color=LADDER_COLORS[key], linewidth=1.8, label=f"{label}: std {np.std(p):.2f}")
        top = max(top, counts.max())
    xs = np.linspace(-plim, plim, 400)
    n = len(pull_sets[-1][2])
    ax.plot(xs, n * bw / np.sqrt(2 * np.pi) * np.exp(-0.5 * xs ** 2), "--", color="black", linewidth=1.2,
            label=r"$\mathcal{N}(0, 1)$")
    ax.set_yscale("log")
    ax.set_ylim(0.5, top * 500)
    ax.set_xlim(-plim, plim)
    ax.set_xlabel(r"(Run-cell mean $-$ model) / $\sigma$")
    ax.set_ylabel("Run-cells")
    ax.legend(loc="upper left", frameon=False, fontsize=10)
    ax.text(0.98, 0.97, "(last bins incl. overflow)", transform=ax.transAxes, ha="right", va="top", fontsize=10)
    add_header(ax, runs, RADII[r_tag]["r"], lumi)
    plt.subplots_adjust(left=0.12, right=0.97, bottom=0.12, top=0.94)
    save_fig(fig, output_path, save_pdf)
    plt.close(fig)


# ---------------------------------------------------------------------------------------------------------------------
# Plot 4: tests of the model
# ---------------------------------------------------------------------------------------------------------------------

def plot_chi2_runs(rows, r_tag, src, lumi, output_path, save_pdf):
    """D1: per-run chi2/ndf of the run-cell means, for M3 (no pT, no centrality) and the model."""
    hep.style.use("ATLAS")
    runs = [r["run"] for r in rows]
    idx = np.arange(len(rows))
    m3 = np.array([r["chi2ndf_M3"] for r in rows])
    m5 = np.array([r["chi2ndf_model"] for r in rows])
    fig, ax = plt.subplots(figsize=(12, 6))
    ms = 5 if len(rows) < 500 else 2.5
    ax.plot(idx, m3, "o", color=LADDER_COLORS["M3"], markersize=ms, label=LADDER[2][1])
    ax.plot(idx, m5, "s", color=LADDER_COLORS["M5"], markersize=ms, label=f"{LADDER[-1][1]} (lines {line_source_text(r_tag, src)})")
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1)
    ax.set_yscale("log")
    vals = np.concatenate([m3, m5])
    vals = vals[np.isfinite(vals) & (vals > 0)]
    ax.set_ylim(min(0.3, vals.min() * 0.7), max(3.0, vals.max() * 5))
    ax.set_ylabel(r"$\chi^{2}/\mathrm{ndf}$ of the run's cells")
    ax.text(0.99, 0.95, f"Median: {np.nanmedian(m3):.2f} (M3), {np.nanmedian(m5):.2f} (model)",
            transform=ax.transAxes, ha="right", va="top", fontsize=13)
    setup_run_axis(ax, runs)
    add_header(ax, runs, RADII[r_tag]["r"], lumi)
    ax.legend(loc="upper left", frameon=False, fontsize=12)
    plt.subplots_adjust(left=0.08, right=0.98, bottom=0.20, top=0.93)
    save_fig(fig, output_path, save_pdf)
    plt.close(fig)


def plot_lines_runs(rows, r_tag, lumi, output_path, save_pdf, run_range=None):
    """D2: the per-run lines a and b vs run (the correction table). run_range zooms and uses the run number as x."""
    hep.style.use("ATLAS")
    if run_range is not None:
        rows = [r for r in rows if run_range[0] <= r["run"] <= run_range[1]]
        if len(rows) < 2:
            print(f"Warning: fewer than 2 runs in {run_range}, skipping zoom")
            return
    runs = [r["run"] for r in rows]
    x = np.array(runs, float) if run_range is not None else np.arange(len(rows))
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={"hspace": 0.06})
    ms = 4 if len(rows) < 500 else 2
    for ax, key, label in ((axes[0], "a", r"$a_{\mathrm{run}}$ [ns]"), (axes[1], "b", r"$b_{\mathrm{run}}$ [ns]")):
        y = np.array([r[key] for r in rows])
        e = np.array([r[f"{key}_err"] for r in rows])
        ax.axhline(0.0, color="gray", linestyle="--", linewidth=1)
        ax.errorbar(x, y, yerr=e, fmt="o", color="black", markersize=ms, capsize=0, linewidth=0.8)
        ax.set_ylabel(label)
        lo, hi = np.nanpercentile(y - e, 0.5), np.nanpercentile(y + e, 99.5)
        pad = 0.25 * (hi - lo)
        ax.set_ylim(lo - pad, hi + pad)
        ax.text(0.01, 0.95, f"RMS over runs = {np.nanstd(y):.3f} ns, median error = {np.nanmedian(e):.4f} ns",
                transform=ax.transAxes, ha="left", va="top", fontsize=12)
    axes[0].text(0.99, 0.95, r"Correction: $a_{\mathrm{run}} + b_{\mathrm{run}}\,x$ (fitted on raw $p_{T}$ 10–60 GeV)",
                 transform=axes[0].transAxes, ha="right", va="top", fontsize=12)
    if run_range is None:
        setup_run_axis(axes[1], runs)
    else:
        axes[1].set_xlabel("Run")
        axes[1].ticklabel_format(axis="x", useOffset=False, style="plain")
    add_header(axes[0], runs, RADII[r_tag]["r"], lumi)
    plt.subplots_adjust(left=0.09, right=0.98, bottom=0.15 if run_range is None else 0.09, top=0.95)
    save_fig(fig, output_path, save_pdf)
    plt.close(fig)


def plot_degree_scan(scan, floors, runs, lumi, output_path, save_pdf):
    """D4: chi2/ndf of f_c(x, pT) vs the degree in ln pT, one curve per degree in x, both radii."""
    hep.style.use("ATLAS")
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5), gridspec_kw={"wspace": 0.22})
    for ax, r_tag in zip(axes, ("r03", "r02")):
        if r_tag not in scan:
            ax.set_visible(False)
            continue
        dps = sorted({dp for _, dp in scan[r_tag]})
        for dx, color in zip(sorted({dx for dx, _ in scan[r_tag]}), ("#377eb8", "#e41a1c", "#4daf4a", "#984ea3")):
            y = [scan[r_tag][(dx, dp)] for dp in dps]
            ax.plot(dps, y, marker="o", color=color, linewidth=1.5, markersize=8, label=f"degree {dx} in $x$")
        ax.axhline(floors[r_tag], color="black", linestyle=":", linewidth=1.5, label="Best possible shape (free per cell)")
        ax.set_xticks(dps)
        ax.set_xlabel(r"Degree in $\ln(p_{T} / 15\,\mathrm{GeV})$")
        ax.set_ylabel(r"$\chi^{2}/\mathrm{ndf}$")
        ax.legend(loc="upper right", frameon=False, fontsize=11)
        ax.text(0.97, 0.55, "lines: " + ("own fit" if r_tag == "r03" else r"from $R = 0.3$ (same degrees)"),
                transform=ax.transAxes, ha="right", va="center", fontsize=11)
        add_header(ax, runs, RADII[r_tag]["r"], lumi)
    plt.subplots_adjust(left=0.08, right=0.98, bottom=0.12, top=0.93)
    save_fig(fig, output_path, save_pdf)
    plt.close(fig)


def plot_rms_runs(rows, pooled, r_tag, src, lumi, output_path, save_pdf):
    """D5: per-run timing RMS, raw / after M3 / after the model; pooled RMS per pT range in the text."""
    hep.style.use("ATLAS")
    runs = [r["run"] for r in rows]
    idx = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(12, 6))
    ms = 5 if len(rows) < 500 else 2.5
    series = (("rms_raw", "#377eb8", "o", r"$t - t_{0}$"),
              ("rms_M3", LADDER_COLORS["M3"], "^", "after M3"),
              ("rms_model", LADDER_COLORS["M5"], "s", f"after the model (lines {line_source_text(r_tag, src)})"))
    for key, color, marker, label in series:
        ax.plot(idx, [r[key] for r in rows], marker, color=color, markersize=ms, label=label)
    ax.set_ylabel("Timing RMS per run [ns]")
    vals = np.array([r[k] for r in rows for k, *_ in series])
    lo, hi = np.nanpercentile(vals, 0.5), np.nanpercentile(vals, 99.5)
    ax.set_ylim(lo - 0.3 * (hi - lo), hi + 0.9 * (hi - lo))
    lines = ["All runs pooled [ns], raw → M3 → model:"]
    for (plo, phi), a, b, c in zip(PT_RANGES, pooled["raw"], pooled["M3"], pooled["model"]):
        lines.append(f"{plo}–{phi} GeV: {a:.3f} → {b:.3f} → {c:.3f}")
    ax.text(0.99, 0.97, "\n".join(lines), transform=ax.transAxes, ha="right", va="top", fontsize=10)
    setup_run_axis(ax, runs)
    add_header(ax, runs, RADII[r_tag]["r"], lumi)
    ax.legend(loc="upper left", frameon=False, fontsize=11, ncol=3)
    plt.subplots_adjust(left=0.08, right=0.98, bottom=0.20, top=0.93)
    save_fig(fig, output_path, save_pdf)
    plt.close(fig)


def scatter_lines(axes, l1, l2, cov1, cov2, name1, name2):
    """a and b of two sets of per-run lines against each other, with the agreement in the text."""
    ok = np.isfinite(l1).all(1) & np.isfinite(l2).all(1)
    stats = []
    for ax, k, sym in ((axes[0], 0, "a"), (axes[1], 1, "b")):
        v1, v2 = l1[ok, k], l2[ok, k]
        lo, hi = np.percentile(np.concatenate([v1, v2]), [0.5, 99.5])
        ax.plot(v1, v2, "o", color="black", markersize=2, alpha=0.4)
        ax.plot([lo, hi], [lo, hi], "--", color="#e41a1c", linewidth=1.5)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        d = v2 - v1
        text = f"corr = {np.corrcoef(v1, v2)[0, 1]:.3f}\nRMS(diff) = {d.std():.3f} ns\nmean(diff) = {d.mean():+.3f} ns"
        if cov1 is not None and cov2 is not None:
            sd = np.sqrt(cov1[ok, k, k] + cov2[ok, k, k])
            text += f"\nstat. error of diff (median) = {np.median(sd):.3f} ns\npull std = {np.std(d / sd):.2f}"
            stats.append((np.std(d / sd), d.std()))
        ax.text(0.03, 0.97, text + f"\n{ok.sum()} runs", transform=ax.transAxes, ha="left", va="top", fontsize=10)
        ax.set_xlabel(rf"${sym}_{{\mathrm{{run}}}}$, {name1} [ns]")
        ax.set_ylabel(rf"${sym}_{{\mathrm{{run}}}}$, {name2} [ns]")
    return stats


def plot_line_crosscheck(own, src_line, runs, r_tag, src, lumi, output_path, save_pdf):
    """D6: per-run lines fitted on this radius vs the lines of the source radius."""
    hep.style.use("ATLAS")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5), gridspec_kw={"wspace": 0.25})
    scatter_lines(axes, src_line, own, None, None, rf"$R = {RADII[src]['r']:g}$", rf"$R = {RADII[r_tag]['r']:g}$ own fit")
    add_header(axes[1], runs, RADII[r_tag]["r"], lumi)
    plt.subplots_adjust(left=0.08, right=0.98, bottom=0.13, top=0.93)
    save_fig(fig, output_path, save_pdf)
    plt.close(fig)


def x_ranges_per_pt(cells, n_cent):
    """Populated x range of each (class, pT bin): the fit has no data outside it. -> list[class][pT bin] = (lo, hi) | None."""
    co, used = cells["co"], cells["used"]
    xw = 0.5 * np.diff(cells["xc"][:2])[0]                      # half bin width: the range covers the bins
    out = []
    for k in range(n_cent):
        row = []
        for ip in range(len(cells["pc"])):
            sel = used & (co["k"] == k) & (co["ipt"] == ip)
            row.append((float(co["x"][sel].min() - xw), float(co["x"][sel].max() + xw)) if sel.any() else None)
        out.append(row)
    return out


def plot_shapes(cells, c0, c1, c2, coef, deg_x, deg_pt, x_ranges, runs, r_tag, src, lumi, output_path, save_pdf):
    """D7: f_c(x, pT) vs x at several pT, with the pooled profile of t - a - b x in those pT bins, per class.

    Each curve is drawn only over the populated x range of its (class, pT bin): outside it the polynomial is unconstrained.
    """
    hep.style.use("ATLAS")
    co, used = cells["co"], cells["used"]
    pts = [p for p in (10.5, 12.5, 15.5, 18.5, 22.5, 27.5, 33.75) if np.any(np.isclose(co["pt"], p))]
    colors = plt.get_cmap("viridis")(np.linspace(0, 0.9, len(pts)))
    fig, axes = plt.subplots(1, len(CENT_CLASSES), figsize=(17, 6.5), sharey=True, gridspec_kw={"wspace": 0.05})
    for k, ax in enumerate(axes):
        for p, color in zip(pts, colors):
            sel = used & (co["k"] == k) & np.isclose(co["pt"], p) & (c0 >= 200)
            if not sel.any():
                continue
            mu = c1[sel] / c0[sel]
            err = np.sqrt(np.clip(c2[sel] / c0[sel] - mu ** 2, 0, None) / c0[sel])
            ax.errorbar(co["x"][sel], mu, yerr=err, fmt="o", color=color, markersize=3.5, capsize=0, linewidth=0.8)
            ip = int(co["ipt"][sel][0])
            if x_ranges[k][ip] is None:
                continue
            xs = np.linspace(*x_ranges[k][ip], 200)
            ax.plot(xs, eval_shape(coef[k], xs, np.full_like(xs, p), deg_x, deg_pt), "-", color=color, linewidth=1.8,
                    label=rf"$p_{{T}} \approx {p:g}$ GeV")
        ax.set_xlabel(FRAC_LABEL)
        ax.set_xlim(-0.3, 1.3)
        ax.text(0.04, 0.05, cent_label(k), transform=ax.transAxes, ha="left", va="bottom", fontsize=14)
    axes[0].set_ylabel(r"$t - a_{\mathrm{run}} - b_{\mathrm{run}} x$ [ns]")
    axes[0].legend(loc="upper right", frameon=False, fontsize=10)
    axes[1].text(0.5, 0.97, r"points: data (pooled over runs); lines: $f_{c}(x, p_{T})$" "\n"
                 r"(drawn over the populated $x$ range of each $p_{T}$ bin)", transform=axes[1].transAxes,
                 ha="center", va="top", fontsize=12)
    add_header(axes[-1], runs, RADII[r_tag]["r"], lumi, extra=f", lines {line_source_text(r_tag, src)}")
    plt.subplots_adjust(left=0.07, right=0.98, bottom=0.12, top=0.93)
    save_fig(fig, output_path, save_pdf)
    plt.close(fig)


def plot_line_stability(pairs, runs, r_tag, lumi, output_path, save_pdf):
    """D8: per-run lines fitted on disjoint subsets of cells (pT ranges, centrality classes) against each other."""
    hep.style.use("ATLAS")
    fig, axes = plt.subplots(len(pairs), 2, figsize=(13, 6 * len(pairs)), gridspec_kw={"wspace": 0.25, "hspace": 0.3})
    axes = np.atleast_2d(axes)
    for row, (title, (l1, cv1, n1), (l2, cv2, n2)) in zip(axes, pairs):
        stats = scatter_lines(row, l1, l2, cv1, cv2, n1, n2)
        row[0].set_title(title, fontsize=13, loc="left")
        print(f"{r_tag}: line stability, {title}: pull std of the a / b differences "
              f"{stats[0][0]:.2f} / {stats[1][0]:.2f}, RMS(diff) {stats[0][1]:.3f} / {stats[1][1]:.3f} ns")
    add_header(axes[0, 1], runs, RADII[r_tag]["r"], lumi)
    plt.subplots_adjust(left=0.08, right=0.98, bottom=0.07, top=0.95)
    save_fig(fig, output_path, save_pdf)
    plt.close(fig)


# ---------------------------------------------------------------------------------------------------------------------
# The final model's coefficients
# ---------------------------------------------------------------------------------------------------------------------

def coef_by_class(c, deg_x, deg_pt, n_cent):
    """Coefficient vector of design(kind='x_pt_cent') -> array (n_cent, deg_x + 1, deg_pt + 1)."""
    return np.asarray(c).reshape(n_cent, deg_x + 1, deg_pt + 1)


def eval_shape(coef_c, x, pt, deg_x, deg_pt):
    L = np.log(pt / PT_REF)
    return sum(coef_c[i, j] * x ** i * L ** j for i in range(deg_x + 1) for j in range(deg_pt + 1))


# ---------------------------------------------------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run-combined iternf raw jet pT, and the leading-jet timing model t = f_c(x, pT) + a_run + b_run x "
                    "(smooth shape per centrality class, per-run line), with a model ladder and tests of the model."
    )
    parser.add_argument("-f", "--file", type=Path,
                        default=Path("/sphenix/user/anarde/sEPD-Study/jetAnav3/11-02-26-v3/test.list"),
                        help="List of Jet-Anav3 merged ROOT files named <run>.root (default: %(default)s)")
    parser.add_argument("-l", "--lumi-list", type=Path,
                        default=Path("/sphenix/user/anarde/sEPD-Study/f4a_bkgsub/10-30-26/test.list"),
                        help="List of Event QA ROOT files named <run>.root with hLuminosity; the label is omitted "
                             "unless every run has one (default: %(default)s)")
    parser.add_argument("-o", "--output-dir", type=Path,
                        default=Path("/sphenix/user/anarde/sEPD-Study/plots/Jet-Ana/iternf_timing"),
                        help="Output directory (default: %(default)s)")
    parser.add_argument("--runs", nargs="*", type=int, help="Specific run number(s) to combine (default: all runs in the list)")
    parser.add_argument("--radii", nargs="+", choices=list(RADII), default=list(RADII), help="Jet radii to plot (default: both)")
    parser.add_argument("--line-source", choices=list(RADII), default="r03",
                        help="Radius the per-run lines a, b are measured from and applied to all radii (default: %(default)s)")
    parser.add_argument("--deg-x", type=int, default=4, help="Degree of f_c in x (default: %(default)s)")
    parser.add_argument("--deg-pt", type=int, default=3, help="Degree of f_c in ln(pT / 15 GeV) (default: %(default)s)")
    parser.add_argument("--pt-range", nargs=2, type=float, default=(10, 60), metavar=("PTMIN", "PTMAX"),
                        help="Raw pT range of the cells used [GeV] (default: 10 60, the profile range)")
    parser.add_argument("--fit-range", nargs=2, type=float, default=None, metavar=("XMIN", "XMAX"),
                        help="Restrict the model to this OHCal fraction range (default: all bins with enough jets)")
    parser.add_argument("--min-cell-entries", type=float, default=50,
                        help="Minimum jets in a (centrality, pT, x) cell, summed over runs, for the cell to be used (default: %(default)s)")
    parser.add_argument("--min-run-entries", type=float, default=200,
                        help="Minimum jets of a run in the used cells for the run to get a line (default: %(default)s)")
    parser.add_argument("--per-event", action="store_true", help="Normalize the raw pT spectrum per event and per GeV")
    parser.add_argument("--pt-max", type=float, default=None,
                        help="x-axis max for the raw pT plot (default: auto, plus a _pt60 version if auto > 60)")
    parser.add_argument("--time-ylim", nargs=2, type=float, default=None, metavar=("TMIN", "TMAX"),
                        help="y-axis range of the combined timing plot (default: full histogram range)")
    parser.add_argument("--zoom-runs", nargs=2, type=int, action="append", default=[], metavar=("RUNMIN", "RUNMAX"),
                        help="Also make the per-run line plot (D2) zoomed to this run range, vs run number (repeatable)")
    parser.add_argument("--no-ladder", action="store_true", help="Skip the model ladder (L1-L5)")
    parser.add_argument("--no-scan", action="store_true", help="Skip the degree scan (D4)")
    parser.add_argument("--no-stability", action="store_true", help="Skip the line stability test (D8)")
    parser.add_argument("--no-pt", action="store_true", help="Skip the raw pT plot")
    parser.add_argument("--no-timing", action="store_true", help="Skip the timing model and its tests")
    parser.add_argument("-j", "--jobs", type=int, default=min(8, os.cpu_count() or 1),
                        help="Parallel workers for reading the runs (default: %(default)s)")
    parser.add_argument("--save-pdf", action="store_true", help="Save PDF versions in addition to PNG")
    parser.add_argument("--cache-dir", type=Path, default=None,
                        help="Where the input cache is kept (default: <output-dir>/cache). The cache holds the per-run "
                             "inputs read from the ROOT files and the luminosity; it is reused only if every input file "
                             "is unchanged (path, size, mtime) and the same radii are read")
    parser.add_argument("--no-cache", action="store_true", help="Neither read nor write the input cache")
    parser.add_argument("--refresh-cache", action="store_true", help="Reread the ROOT files and overwrite the cache")
    args = parser.parse_args()
    tick = StageTimer()
    if args.deg_x < 1 or args.deg_pt < 0:
        parser.error("--deg-x must be >= 1 and --deg-pt >= 0")

    run_files = parse_list_file(args.file)
    if not run_files:
        print(f"Error: No ROOT files found in {args.file}")
        sys.exit(1)
    runs_req = sorted(args.runs) if args.runs else sorted(run_files)
    pt_radii = () if args.no_pt else tuple(args.radii)
    t_radii = () if args.no_timing else tuple(dict.fromkeys([args.line_source, *args.radii]))
    tasks = []
    for run in runs_req:
        path = run_files.get(run)
        if path is None or not path.exists():
            print(f"Warning: No file for run {run}, skipping")
            continue
        tasks.append((run, path, pt_radii, t_radii))

    lumi_files = parse_list_file(args.lumi_list) if args.lumi_list.exists() else {}
    lumi_paths = [lumi_files[r] for r, *_ in tasks if r in lumi_files and lumi_files[r].exists()]

    cache_path = None
    if not args.no_cache:
        cache_dir = args.cache_dir if args.cache_dir is not None else args.output_dir / "cache"
        cache_path = cache_dir / f"inputs_{cache_key(tasks, pt_radii, t_radii, lumi_paths)}.pkl"
    if cache_path is not None and cache_path.exists() and not args.refresh_cache:
        print(f"Reading the inputs from the cache {cache_path}")
        with cache_path.open("rb") as f:
            cached = pickle.load(f)
        loaded, t_sum, lumi = cached["loaded"], cached["t_sum"], cached["lumi"]
        runs = sorted(loaded)
    else:
        # Stream the results so only one full 2D per radius (the sum) is kept in memory
        n_jobs = max(1, min(args.jobs, len(tasks)))
        print(f"Reading {len(tasks)} runs with {n_jobs} worker(s)")
        loaded, t_sum = {}, {}
        ex = concurrent.futures.ProcessPoolExecutor(max_workers=n_jobs) if n_jobs > 1 else None
        for run, d in (ex.map(load_run, tasks, chunksize=8) if ex else map(load_run, tasks)):
            if d is None:
                continue
            for r_tag in t_radii:
                t_sum[r_tag] = t_sum.get(r_tag, 0) + d[r_tag].pop("t_values")
            loaded[run] = d
        if ex:
            ex.shutdown()
        if not loaded:
            print("Error: no runs loaded")
            sys.exit(1)
        runs = sorted(loaded)

        # Luminosity: only label it if every combined run has one, so the sum is not silently partial
        lumi = None
        if lumi_files:
            missing = [r for r in runs if r not in lumi_files or not lumi_files[r].exists()]
            if missing:
                print(f"Warning: no luminosity file for {len(missing)} run(s) (e.g. {missing[:5]}), omitting the luminosity label")
            else:
                lumis = [load_luminosity(lumi_files[r]) for r in runs]
                lumi = None if any(l is None for l in lumis) else float(sum(lumis))
        else:
            print(f"Warning: no luminosity list at {args.lumi_list}, omitting the luminosity label")

        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = cache_path.with_suffix(".tmp")
            with tmp.open("wb") as f:
                pickle.dump({"loaded": loaded, "t_sum": t_sum, "lumi": lumi}, f, protocol=pickle.HIGHEST_PROTOCOL)
            tmp.replace(cache_path)                       # atomic: an interrupted write never leaves a broken cache
            print(f"Saved the input cache: {cache_path} ({cache_path.stat().st_size / 1e6:.0f} MB)")
    tick(f"inputs ({len(runs)} runs)")

    n_events = sum(loaded[r]["n_events"] for r in runs)
    out = args.output_dir

    for r_tag in pt_radii:
        pt_vals = sum(loaded[r][r_tag]["pt_values"] for r in runs)
        pt_vars = sum(loaded[r][r_tag]["pt_variances"] for r in runs)
        print(f"{r_tag}: combined raw pT jets = {pt_vals.sum():.0f}, N_evt = {n_events:.0f}")
        plot_raw_pt(pt_vals, pt_vars, loaded[runs[0]][r_tag]["pt_edges"], n_events, runs, r_tag, lumi, args.per_event,
                    out / "pt" / f"combined_{pt_hist_name(r_tag)}.png", pt_max=args.pt_max, save_pdf=args.save_pdf)

    if args.no_timing:
        return

    src = args.line_source
    dx, dp, nK = args.deg_x, args.deg_pt, len(CENT_CLASSES)
    tdir, ddir, ldir = out / "timing", out / "timing" / "diagnostics", out / "timing" / "ladder"
    suffix_of = lambda r: f"{r}_iternf"
    order = [src] + [r for r in args.radii if r != src]

    cells = {}
    for r_tag in order:
        mom = np.stack([loaded[r][r_tag]["p2_mom"] for r in runs])          # (n_runs, 4, n_cent, n_pt, n_x)
        ref = loaded[runs[0]][r_tag]
        cells[r_tag] = build_cells(mom, ref["p2_xedges"], ref["p2_ptedges"], args)
        del mom
        for r in runs[1:]:
            del loaded[r][r_tag]["p2_mom"]                                  # the cells hold a copy
        C = cells[r_tag]
        print(f"{r_tag}: {C['n'].sum():.0f} leading jets in the profiles; {int(C['used'].sum())} cells used "
              f"({C['n'][:, C['used']].sum() / C['n'].sum():.5f} of the jets); {int(C['m'].sum())} run-cells; "
              f"{int((C['run_entries'] >= args.min_run_entries).sum())}/{len(runs)} runs with >= {args.min_run_entries:g} jets")

    tick("cells")
    # ---- The model ladder: every model on the same cells; the other radius uses the source's per-run terms of the step
    fits = {r: {} for r in order}
    for key, label, kind, n_line in LADDER:
        if args.no_ladder and key != "M3" and key != FINAL:
            continue
        S = cells[src]
        X = design(S["co"], kind, dx, dp, nK)
        fits[src][key] = fit_model(S, X, n_line)
        for r_tag in order[1:]:
            Xr = design(cells[r_tag]["co"], kind, dx, dp, nK)
            fits[r_tag][key] = fit_fixed_lines(cells[r_tag], Xr, fits[src][key]["line"])
        for r_tag in order:
            F = fits[r_tag][key]
            print(f"{r_tag}: {key:3s} {re.sub(r'[$\\{}]|mathrm', '', label):38s} chi2/ndf = {F['chi2']:.0f}/{F['ndf']} = "
                  f"{F['chi2'] / F['ndf']:.4f}, pull std {np.std(F['pulls']):.3f}")

    tick("ladder fits")
    # ---- The final model: convention for the lines, coefficients per class
    F = fits[src][FINAL]
    line_src, means = center_lines(F["line"], cells[src]["run_entries"])
    coef_vec = F["c"].copy()
    K1 = (dx + 1) * (dp + 1)
    for k in range(nK):
        coef_vec[k * K1 + 0 * (dp + 1)] += means[0]       # x^0 L^0
        coef_vec[k * K1 + 1 * (dp + 1)] += means[1]       # x^1 L^0
    lines_cov = F["line_cov"]
    models = {src: {"coef": coef_by_class(coef_vec, dx, dp, nK), "cov": None, "fit": F}}
    for r_tag in order:
        S = cells[r_tag]
        X = design(S["co"], "x_pt_cent", dx, dp, nK)
        refit = fit_fixed_lines(S, X, line_src)            # with the centered lines; for the source this is the same optimum
        if r_tag == src:
            models[r_tag]["cov"] = refit["cov"]
            dmax = np.max(np.abs((X @ refit["c"] - X @ coef_vec)[S["used"]]))
            print(f"{src}: refit with the centered lines reproduces the joint fit's shape to {dmax:.1e} ns "
                  f"(chi2 {refit['chi2']:.1f} vs {F['chi2']:.1f})")
        else:
            models[r_tag] = {"coef": coef_by_class(refit["c"], dx, dp, nK), "cov": refit["cov"], "fit": refit}
        fits[r_tag]["final_centered"] = refit
    no_line = [r for r, ok in zip(runs, F["ok"]) if not ok]
    if no_line:
        print(f"{src}: {len(no_line)} run(s) without a line (< {args.min_run_entries:g} jets in the used cells): {no_line}")

    # Per-run lines table
    rows_lines = []
    for i, run in enumerate(runs):
        ok = bool(F["ok"][i])
        rows_lines.append({
            "run": run, "status": "ok" if ok else "no_line", "jets": float(cells[src]["run_entries"][i]),
            "a": float(line_src[i, 0]) if ok else np.nan, "b": float(line_src[i, 1]) if ok else np.nan,
            "a_err": float(np.sqrt(lines_cov[i, 0, 0])) if ok else np.nan,
            "b_err": float(np.sqrt(lines_cov[i, 1, 1])) if ok else np.nan,
            "cov_ab": float(lines_cov[i, 0, 1]) if ok else np.nan,
        })
    write_csv(rows_lines, ["run", "status", "jets", "a", "b", "a_err", "b_err", "cov_ab"],
              tdir / f"per_run_lines_{suffix_of(src)}.csv")

    tick("final model")
    # ---- Floors (best possible shape) for the ladder and the degree scan
    floors = {}
    for r_tag in order:
        fl = fit_free_cells(cells[r_tag]) if r_tag == src else fit_free_cells(cells[r_tag], line_fixed=line_src)
        floors[r_tag] = fl["chi2"] / fl["ndf"]
        print(f"{r_tag}: best possible shape (free per cell, lines {line_source_text(r_tag, src)}): chi2/ndf = {floors[r_tag]:.4f}")

    tick("floors")
    # ---- Ladder plots
    ladder_stats = {}
    if not args.no_ladder:
        for r_tag in order:
            S = cells[r_tag]
            ladder_stats[r_tag] = {}
            maps = {}
            for key, label, kind, n_line in LADDER:
                Fk = fits[r_tag][key]
                c0, c1, c2 = pooled_corrected(S, Fk["s"], Fk["line"] if n_line else np.zeros((len(runs), 0)))
                ladder_stats[r_tag][key] = closure_stats(S, c0, c1, c2)
                maps[key] = closure_map(S, c0, c1, c2)
            c0, c1, c2 = pooled_corrected(S, None, np.zeros((len(runs), 0)))
            raw_rms = closure_stats(S, c0, c1, c2)["rms"]
            # common y range across the ladder so the steps can be compared
            big = max(np.nanmax(np.abs(b)) for mp in maps.values() for (_, b, _) in mp.values() if len(b))
            ylim = (-min(2.5, 1.15 * big), min(2.5, 1.15 * big))
            for key, label, *_ in LADDER:
                plot_ladder_closure(maps[key], ylim, runs, r_tag, label, lumi,
                                    ldir / f"L3_closure_{key}_{suffix_of(r_tag)}.png", args.save_pdf)
            plot_ladder_closure(maps[FINAL], (-0.25, 0.25), runs, r_tag, LADDER[-1][1] + " (zoom)", lumi,
                                ldir / f"L3_closure_{FINAL}_zoom_{suffix_of(r_tag)}.png", args.save_pdf)
            plot_ladder_bias(ladder_stats[r_tag], runs, r_tag, lumi, ldir / f"L2_bias_{suffix_of(r_tag)}.png", args.save_pdf)
            plot_ladder_rms(ladder_stats[r_tag], raw_rms, runs, r_tag, lumi, ldir / f"L4_rms_{suffix_of(r_tag)}.png", args.save_pdf)
            plot_ladder_pulls([(k, lab, fits[r_tag][k]["pulls"]) for k, lab, *_ in LADDER], runs, r_tag, lumi,
                              ldir / f"L5_pulls_{suffix_of(r_tag)}.png", args.save_pdf)
            rows = []
            for key, label, kind, n_line in LADDER:
                st, Fk = ladder_stats[r_tag][key], fits[r_tag][key]
                row = {"model": key, "label": re.sub(r"[$\\{}]|mathrm", "", label), "chi2": Fk["chi2"], "ndf": Fk["ndf"],
                       "chi2ndf": Fk["chi2"] / Fk["ndf"], "pull_std": float(np.std(Fk["pulls"])), "worst_bias": st["worst"]}
                for (lo, hi), b, fl, rm in zip(PT_RANGES, st["bias"], st["floor"], st["rms"]):
                    row[f"bias_{lo}_{hi}"], row[f"floor_{lo}_{hi}"], row[f"rms_{lo}_{hi}"] = b, fl, rm
                rows.append(row)
                print(f"{r_tag}: {key:3s} |bias| per pT range " + " ".join(f"{b:.3f}" for b in st["bias"])
                      + f" (floor " + " ".join(f"{f:.3f}" for f in st["floor"]) + f"), worst {st['worst']:.3f} ns; "
                      f"pooled RMS " + " ".join(f"{v:.3f}" for v in st["rms"]))
            write_csv(rows, list(rows[0].keys()), ldir / f"ladder_{suffix_of(r_tag)}.csv")
        plot_ladder_chi2({r: fits[r] for r in order}, floors, runs, lumi, ldir / "L1_chi2ndf.png", args.save_pdf)

    tick("ladder plots")
    # ---- Degree scan of the final model
    if not args.no_scan:
        scan = {r: {} for r in order}
        for sdx in (3, 4, 5):
            for sdp in (1, 2, 3, 4):
                S = cells[src]
                Fs = fit_model(S, design(S["co"], "x_pt_cent", sdx, sdp, nK), 2)
                scan[src][(sdx, sdp)] = Fs["chi2"] / Fs["ndf"]
                for r_tag in order[1:]:
                    Fr = fit_fixed_lines(cells[r_tag], design(cells[r_tag]["co"], "x_pt_cent", sdx, sdp, nK), Fs["line"])
                    scan[r_tag][(sdx, sdp)] = Fr["chi2"] / Fr["ndf"]
        for r_tag in order:
            print(f"{r_tag}: degree scan chi2/ndf (deg x, deg ln pT): "
                  + ", ".join(f"({a},{b}) {v:.4f}" for (a, b), v in sorted(scan[r_tag].items())))
        plot_degree_scan(scan, floors, runs, lumi, ddir / "D4_degree_scan.png", args.save_pdf)

    tick("degree scan")
    # ---- Per-radius tests of the final model
    for r_tag in args.radii:
        S = cells[r_tag]
        co = S["co"]
        suffix = suffix_of(r_tag)
        Ff = fits[r_tag]["final_centered"]
        coef = models[r_tag]["coef"]
        x_ranges = x_ranges_per_pt(S, nK)                                   # [class][pT bin] -> (lo, hi) | None

        # combined 15-40 GeV plot: TH2, its profile, profile of t - a - b x and the model averaged over the jets
        ref = loaded[runs[0]][r_tag]
        t_flow = sum(loaded[r][r_tag]["t_flow"] for r in runs)
        filled = sum(loaded[r][r_tag]["t_filled"] for r in runs)
        prof_raw = profile_from_moments(sum(loaded[r][r_tag]["t_mom"] for r in runs))
        c0l, c1l, c2l = pooled_corrected(S, None, line_src)
        sel15 = S["used"] & (co["pt"] >= TIMING_PT[0]) & (co["pt"] < TIMING_PT[1])
        nx = len(S["xc"])
        n_x = np.bincount(co["ix"][sel15], weights=c0l[sel15], minlength=nx)
        corr_x = np.bincount(co["ix"][sel15], weights=c1l[sel15], minlength=nx) / np.where(n_x > 0, n_x, np.nan)
        var_x = np.bincount(co["ix"][sel15], weights=c2l[sel15], minlength=nx) / np.where(n_x > 0, n_x, np.nan) - corr_x ** 2
        model_x = np.bincount(co["ix"][sel15], weights=(c0l * Ff["s"])[sel15], minlength=nx) / np.where(n_x > 0, n_x, np.nan)
        n_neg, n_over = frac_outside_unit(t_sum[r_tag], ref["t_xedges"], t_flow)
        print(f"{r_tag}: 15-40 GeV TH2 entries {t_sum[r_tag].sum():.0f} in range + {t_flow.sum():.0f} flow "
              f"(x < 0: {n_neg:.0f}, x > 1: {n_over:.0f}, |t| > 25: {t_flow[2]:.0f}); status 'Filled' = {filled:.0f}")
        plot_timing(t_sum[r_tag], ref["t_xedges"], ref["t_yedges"], prof_raw,
                    (corr_x, np.sqrt(np.clip(var_x, 0, None) / np.where(n_x > 0, n_x, np.nan))), model_x,
                    runs, r_tag, src, lumi, tdir / f"combined_{timing_hist_name(r_tag)}.png", t_flow,
                    time_ylim=args.time_ylim, save_pdf=args.save_pdf)

        # per-run table
        F3 = fits[r_tag]["M3"]
        rms_raw = per_run_corrected_rms(S, None, np.zeros((len(runs), 0)))
        rms_m3 = per_run_corrected_rms(S, F3["s"], F3["line"])
        rms_fin = per_run_corrected_rms(S, Ff["s"], line_src)
        rows = []
        has = np.isfinite(line_src).all(1) & (S["run_entries"] >= args.min_run_entries)
        for i, run in enumerate(runs):
            if not has[i] or Ff["run_npts"][i] < 3:
                continue
            ndf_i = Ff["run_npts"][i] - (2 if r_tag == src else 0)
            ndf3 = F3["run_npts"][i] - (2 if r_tag == src else 0)
            rows.append({"run": run, "jets": float(S["run_entries"][i]), "n_cells": int(Ff["run_npts"][i]),
                         "chi2ndf_M3": float(F3["run_chi2"][i] / max(ndf3, 1)),
                         "chi2ndf_model": float(Ff["run_chi2"][i] / max(ndf_i, 1)),
                         "a": float(line_src[i, 0]), "b": float(line_src[i, 1]),
                         "a_err": float(np.sqrt(lines_cov[i, 0, 0])), "b_err": float(np.sqrt(lines_cov[i, 1, 1])),
                         "rms_raw": float(rms_raw[i]), "rms_M3": float(rms_m3[i]), "rms_model": float(rms_fin[i])})
        print(f"{r_tag}: {len(rows)} runs in the per-run tests; median per-run chi2/ndf "
              f"{np.median([r['chi2ndf_M3'] for r in rows]):.2f} (M3) -> {np.median([r['chi2ndf_model'] for r in rows]):.2f} (model)")
        write_csv(rows, list(rows[0].keys()), ddir / f"per_run_{suffix}.csv")

        plot_chi2_runs(rows, r_tag, src, lumi, ddir / f"D1_chi2ndf_{suffix}.png", args.save_pdf)
        if r_tag == src:
            plot_lines_runs(rows, r_tag, lumi, ddir / f"D2_lines_{suffix}.png", args.save_pdf)
            for lo, hi in args.zoom_runs:
                plot_lines_runs(rows, r_tag, lumi, ddir / f"D2_lines_{suffix}_runs{lo}_{hi}.png", args.save_pdf,
                                run_range=(lo, hi))

        # D5: RMS
        pooled = {}
        for name, (s, line) in (("raw", (None, np.zeros((len(runs), 0)))), ("M3", (F3["s"], F3["line"])),
                                ("model", (Ff["s"], line_src))):
            pooled[name] = closure_stats(S, *pooled_corrected(S, s, line))["rms"]
        plot_rms_runs(rows, pooled, r_tag, src, lumi, ddir / f"D5_rms_{suffix}.png", args.save_pdf)

        # D6: this radius' own lines vs the source's
        if r_tag != src:
            own = fit_model(S, design(co, "x_pt_cent", dx, dp, nK), 2)
            own_line, _ = center_lines(own["line"], S["run_entries"])
            print(f"{r_tag}: own lines: chi2/ndf {own['chi2'] / own['ndf']:.4f} (with {src} lines "
                  f"{Ff['chi2'] / Ff['ndf']:.4f})")
            plot_line_crosscheck(own_line, line_src, runs, r_tag, src, lumi, ddir / f"D6_lines_vs_{src}_{suffix}.png",
                                 args.save_pdf)

        # D7: the shapes against the data
        plot_shapes(S, c0l, c1l, c2l, coef, dx, dp, x_ranges, runs, r_tag, src, lumi,
                    ddir / f"D7_shapes_{suffix}.png", args.save_pdf)

        tick(f"{r_tag}: tests")
        # D8: line stability across pT and centrality (free shape per cell in each subset), source radius only
        if r_tag == src and not args.no_stability:
            def subset_lines(mask):
                fl = fit_free_cells(S, mask=mask)
                ln, _ = center_lines(fl["line"], S["run_entries"])
                return ln, fl["line_cov"]
            lo_pt, hi_pt = subset_lines(co["pt"] < 15), subset_lines(co["pt"] >= 15)
            c_0, c_rest = subset_lines(co["k"] == 0), subset_lines(co["k"] > 0)
            plot_line_stability([
                (r"Raw $p_{T}$ 10–15 vs 15–60 GeV", (*lo_pt, "10–15 GeV"), (*hi_pt, "15–60 GeV")),
                ("Centrality 0–20% vs 20–60%", (*c_0, "0–20%"), (*c_rest, "20–60%")),
            ], runs, r_tag, lumi, ddir / f"D8_line_stability_{suffix}.png", args.save_pdf)

        tick(f"{r_tag}: line stability")
        # the model file
        model_json = tdir / f"model_{suffix}.json"
        with model_json.open("w") as f:
            json.dump({
                "profiles": [profile_name(r_tag, c) for c in CENT_CLASSES],
                "model": "t = f_c(x, pT) + a_run + b_run * x; t = jet t - MBD t0 [ns], x = OHCal energy fraction, "
                         "pT = leading jet raw pT [GeV]",
                "f_c": f"sum_ij coef[c][i][j] * x^i * ln(pT / {PT_REF:g})^j, i = 0..{dx}, j = 0..{dp}",
                "centrality_classes": [list(c) for c in CENT_CLASSES],
                "coefficients": models[r_tag]["coef"].tolist(),
                "covariance": models[r_tag]["cov"].tolist(),
                "apply_within": {
                    "pt": list(args.pt_range),
                    "pt_bin_edges": loaded[runs[0]][r_tag]["p2_ptedges"].tolist(),
                    "x_range_per_pt_bin": [[list(r) if r is not None else None for r in row] for row in x_ranges],
                    "note": "clamp pT to 'pt', then x to the populated range of the jet's (class, pT bin) "
                            "(x_range_per_pt_bin[class][pT bin], null = no data): the polynomial is unconstrained outside"},
                "line_source": src, "lines_file": f"per_run_lines_{suffix_of(src)}.csv",
                "convention": f"jet-weighted mean of a_run and b_run over runs is 0 ({src})",
                "chi2": Ff["chi2"] if r_tag != src else F["chi2"], "ndf": Ff["ndf"] if r_tag != src else F["ndf"],
                "floor_chi2ndf": floors[r_tag], "n_runs": int(np.isfinite(line_src).all(1).sum()),
                "pooled_rms_ns": {name: dict(zip([f"{lo}-{hi}" for lo, hi in PT_RANGES], map(float, v))) for name, v in pooled.items()},
                "settings": {k: (list(v) if isinstance(v, tuple) else v) for k, v in vars(args).items()
                             if k in ("deg_x", "deg_pt", "pt_range", "fit_range", "min_cell_entries", "min_run_entries")},
            }, f, indent=2)
        print(f"Saved: {model_json}")


if __name__ == "__main__":
    main()
