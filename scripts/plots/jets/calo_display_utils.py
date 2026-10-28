"""
Calorimeter tower display helpers: 2D tower-energy panels with jet outlines.

Used by plot_jet_high_pt_constituents.py.
"""

import copy

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
from matplotlib.colors import LogNorm, Normalize, SymLogNorm
from matplotlib.collections import LineCollection
from mpl_toolkits.axes_grid1 import make_axes_locatable

JET_COLORS = [
    "#e41a1c",  # red
    "#ff7f00",  # orange
    "#377eb8",  # blue
    "#4daf4a",  # green
    "#984ea3",  # purple
    "#a65628",  # brown
    "#f781bf",  # pink
    "#999999",  # grey
]

def get_cluster_boundary_segments(mask, xedges, yedges):
    """
    Computes 2D line segments along the outer boundary of a boolean mask of cells.
    """
    segments = []
    nx, ny = mask.shape
    for i in range(nx):
        for j in range(ny):
            if mask[i, j]:
                x0, x1 = xedges[i], xedges[i + 1]
                y0, y1 = yedges[j], yedges[j + 1]
                # Left
                if i == 0 or not mask[i - 1, j]:
                    segments.append([(x0, y0), (x0, y1)])
                # Right
                if i == nx - 1 or not mask[i + 1, j]:
                    segments.append([(x1, y0), (x1, y1)])
                # Bottom
                if j == 0 or not mask[i, j - 1]:
                    segments.append([(x0, y0), (x1, y0)])
                # Top
                if j == ny - 1 or not mask[i, j + 1]:
                    segments.append([(x0, y1), (x1, y1)])
    return segments

def plot_calo_2d(ax, hist2d, title="", xlabel=r"Tower $\eta$ bin", ylabel=r"Tower $\phi$ bin", logz=False, cmap_name="viridis", overlays=None, legend_fontsize=9):
    """
    Plots a 2D histogram on given ax with COLZ1 style (suppressing empty/0 bins)
    and optionally overlays cluster boundaries (LineCollection).
    """
    if hist2d is None:
        ax.text(0.5, 0.5, "Not Available", transform=ax.transAxes, ha='center', va='center', fontsize=14, color='gray')
        ax.set_title(title, fontsize=14)
        return None

    values, xedges, yedges = hist2d.to_numpy()

    # Mask empty / zero bins (COLZ1 effect: bins with exactly 0.0 energy)
    values_masked = np.ma.masked_where(values == 0, values)

    cmap = copy.copy(plt.get_cmap(cmap_name))
    cmap.set_bad(color='white', alpha=0.0)

    if np.all(values_masked.mask):
        mesh = ax.pcolormesh(xedges, yedges, values.T, cmap=cmap, rasterized=True)
    else:
        min_val = float(np.nanmin(values_masked))
        max_val = float(np.nanmax(values_masked))

        if logz:
            if min_val < 0:
                linthresh = max(1e-3, min(0.1, abs(min_val) / 10.0))
                norm = SymLogNorm(linthresh=linthresh, vmin=min_val, vmax=max_val)
            else:
                min_pos = float(np.nanmin(values[values > 0])) if np.any(values > 0) else 1e-3
                norm = LogNorm(vmin=min_pos, vmax=max_val)
        else:
            if min_val == max_val:
                norm = Normalize(vmin=min_val - 1, vmax=max_val + 1)
            else:
                norm = Normalize(vmin=min_val, vmax=max_val)

        mesh = ax.pcolormesh(xedges, yedges, values_masked.T, norm=norm, cmap=cmap, rasterized=True)

    # Draw overlays (cluster outlines)
    if overlays:
        for segs, color, ls, lw, lbl in overlays:
            if segs and len(segs) > 0:
                lc = LineCollection(segs, colors=color, linestyles=ls, linewidths=lw, label=lbl)
                ax.add_collection(lc)

        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(handles=handles, labels=labels, loc='upper right', fontsize=legend_fontsize, frameon=True, facecolor='white', framealpha=0.85, edgecolor='none')

    ax.set_title(title, fontsize=14, pad=8)
    ax.set_xlabel(xlabel, fontsize=13)
    ax.set_ylabel(ylabel, fontsize=13)
    ax.set_xlim(xedges[0], xedges[-1])
    ax.set_ylim(yedges[0], yedges[-1])
    ax.set_aspect('equal')

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="10%", pad=0.08)
    cbar = plt.colorbar(mesh, cax=cax)
    cbar.set_label("Energy [GeV]", fontsize=12)

    return mesh

def make_1x4_calo_plot(run, event, h_base, h_retower, h_ihcal, h_ohcal, output_path, logz=False, cmap_name="viridis"):
    """
    Creates a 1x4 panel 2D calorimeter energy display (Base EMCal, Retower EMCal, IHCal, OHCal).
    """
    hep.style.use("ATLAS")
    fig, axes = plt.subplots(1, 4, figsize=(18, 7.5))

    plot_calo_2d(axes[0], h_base, title=r"EMCal Base", logz=logz, cmap_name=cmap_name)
    plot_calo_2d(axes[1], h_retower, title=r"EMCal Retower", logz=logz, cmap_name=cmap_name)
    plot_calo_2d(axes[2], h_ihcal, title=r"IHCal", logz=logz, cmap_name=cmap_name)
    plot_calo_2d(axes[3], h_ohcal, title=r"OHCal", logz=logz, cmap_name=cmap_name)

    axes[3].text(1.0, 1.05, rf"Run: {run}, Event: {event}", transform=axes[3].transAxes, ha='right', va='bottom', fontsize=15, weight='bold')

    fig.tight_layout()
    plt.subplots_adjust(left=0.03, right=0.96, top=0.90, bottom=0.10, wspace=0.20)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)

def make_1x3_calo_plot(run, event, h_emcal, h_ihcal, h_ohcal, output_path, sub_type="Iter Sub1", logz=False, cmap_name="viridis", overlays_emcal=None, overlays_ihcal=None, overlays_ohcal=None):
    """
    Creates a 1x3 panel 2D calorimeter energy display (EMCal, IHCal, OHCal) with optional cluster overlays.
    """
    hep.style.use("ATLAS")
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 6.5))

    plot_calo_2d(axes[0], h_emcal, title=rf"{sub_type} EMCal", logz=logz, cmap_name=cmap_name, overlays=overlays_emcal)
    plot_calo_2d(axes[1], h_ihcal, title=rf"{sub_type} IHCal", logz=logz, cmap_name=cmap_name, overlays=overlays_ihcal)
    plot_calo_2d(axes[2], h_ohcal, title=rf"{sub_type} OHCal", logz=logz, cmap_name=cmap_name, overlays=overlays_ohcal)

    axes[2].text(1.0, 1.05, rf"Run: {run}, Event: {event}", transform=axes[2].transAxes, ha='right', va='bottom', fontsize=15, weight='bold')

    fig.tight_layout()
    plt.subplots_adjust(left=0.03, right=0.96, top=0.90, bottom=0.10, wspace=0.20)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
