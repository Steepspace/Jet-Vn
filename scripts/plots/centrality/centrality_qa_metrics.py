import re
from pathlib import Path

import numpy as np
import uproot


def compute_centrality_average(values, edges, cent_min=10.0, cent_max=70.0):
    """
    Compute average y-value excluding outliers at the edges (e.g. low-centrality
    turn-on / uncalibrated bins and high-centrality trigger roll-off / zeros).
    """
    bin_centers = 0.5 * (edges[:-1] + edges[1:])
    mask = (bin_centers >= cent_min) & (bin_centers <= cent_max) & (values > 0)
    if np.sum(mask) >= 3:
        vals = values[mask]
        med = np.median(vals)
        good = np.abs(vals - med) < 0.3 * med
        if np.any(good):
            return float(np.mean(vals[good]))
        return float(np.mean(vals))
    pos_vals = values[values > 0]
    if len(pos_vals) > 0:
        return float(np.median(pos_vals))
    return 0.0


def extract_run_metrics(file_path, hist_name="hCentrality", cent_flat_min=10.0, cent_flat_max=70.0,
                        max_rms_pct=4.5, max_dev_pct_cut=8.0,
                        max_slope_per_10pct=2.5, min_central_ratio=0.80,
                        max_central_ratio=1.20, max_central_spike=1.30,
                        syst_floor=0.02):
    """
    Fast extraction of scalar flatness and QA metrics from a single ROOT file.
    Does not render any plots.
    """
    path = Path(file_path)
    if not path.exists():
        return None, f"File not found: {path}"

    try:
        try:
            run_number = int(path.name.split('.')[0])
        except ValueError:
            match = re.search(r'\d+', path.name)
            if match:
                run_number = int(match.group())
            else:
                return None, f"Could not parse run number from {path.name}"

        with uproot.open(path) as f:
            if hist_name not in f:
                return None, f"Histogram '{hist_name}' not found in {path.name}"

            hist = f[hist_name]
            values, edges = hist.to_numpy()

        total_events = float(np.sum(values))
        bin_centers = 0.5 * (edges[:-1] + edges[1:])

        # Plateau average
        plateau_avg = compute_centrality_average(values, edges, cent_min=cent_flat_min, cent_max=cent_flat_max)

        flags = []

        if plateau_avg <= 0 or total_events == 0:
            return {
                "run_number": run_number,
                "file_path": str(path),
                "total_events": total_events,
                "plateau_avg": 0.0,
                "rms_plat_pct": np.nan,
                "max_dev_pct": np.nan,
                "chi2_ndf": np.nan,
                "slope_per_10pct": np.nan,
                "ratio_1": np.nan,
                "ratio_1_5": np.nan,
                "ratio_80": np.nan,
                "values": values,
                "ratios": np.zeros(len(values)),
                "bin_centers": bin_centers,
                "edges": edges,
                "status": "FLAG_EMPTY_OR_ZERO",
            }, None

        # Normalized ratio array across all bins
        ratios = np.where(values > 0, values / plateau_avg, 0.0)

        # 1. Plateau Metrics (cent_flat_min to cent_flat_max)
        plateau_mask = (bin_centers >= cent_flat_min) & (bin_centers <= cent_flat_max)
        n_plateau_bins = int(np.sum(plateau_mask))

        if n_plateau_bins > 1:
            vals_plat = values[plateau_mask]
            ratios_plat = ratios[plateau_mask]
            dev_plat = ratios_plat - 1.0

            # Statistics-independent RMS flatness across plateau (%)
            rms_plat_pct = float(np.sqrt(np.mean(dev_plat ** 2))) * 100.0

            # Peak maximum relative deviation (%)
            max_dev_pct = float(np.max(np.abs(dev_plat))) * 100.0

            # Linear slope across plateau (% variation per 10% centrality)
            x_plat = bin_centers[plateau_mask]
            poly = np.polyfit(x_plat, ratios_plat, deg=1)
            raw_slope = float(poly[0])
            slope_per_10pct = raw_slope * 10.0 * 100.0

            # Statistics-aware chi2 with systematic floor (f_syst = 0.02)
            var_plat = np.where(vals_plat > 0, vals_plat, 1.0) + (syst_floor * plateau_avg) ** 2
            chi2 = float(np.sum(((vals_plat - plateau_avg) ** 2) / var_plat))
            chi2_ndf = chi2 / (n_plateau_bins - 1)
        else:
            rms_plat_pct = np.nan
            max_dev_pct = np.nan
            slope_per_10pct = np.nan
            chi2_ndf = np.nan

        # 2. 1% Centrality Bin (Bin 1: edges 0.5 to 1.5, center 1.0)
        # Note: Bin 0 [-0.5, 0.5] represents 0% centrality, but is currently empty in all runs
        if len(ratios) > 1 and values[1] > 0:
            ratio_1 = float(ratios[1])
        else:
            ratio_1 = 0.0

        # 3. 1-5% Centrality Bins (Bins 1 to 5: edges 0.5 to 5.5, centers 1.0 to 5.0)
        # Avoids the empty 0% centrality bin (Bin 0)
        cent15_mask = (bin_centers >= 0.5) & (bin_centers <= 5.5) & (values > 0)
        if np.any(cent15_mask):
            ratio_1_5 = float(np.mean(ratios[cent15_mask]))
        else:
            ratio_1_5 = 0.0

        # 4. Peripheral Ratio (near 80% Centrality: 75-85% window)
        cent80_mask = (bin_centers >= 75.0) & (bin_centers <= 85.0) & (values > 0)
        if np.any(cent80_mask):
            ratio_80 = float(np.mean(ratios[cent80_mask]))
        else:
            ratio_80 = 0.0

        # 5. Evaluate Outlier Rules
        if not np.isnan(rms_plat_pct) and rms_plat_pct > max_rms_pct:
            flags.append("FLAG_NON_FLAT")
        elif not np.isnan(max_dev_pct) and max_dev_pct > max_dev_pct_cut:
            flags.append("FLAG_NON_FLAT")

        if not np.isnan(slope_per_10pct) and abs(slope_per_10pct) > max_slope_per_10pct:
            flags.append("FLAG_SLOPE_DRIFT")

        # Central anomaly: either sharp spike at 1% or broader peak/drop in 1-5%
        if not np.isnan(ratio_1) and ratio_1 > max_central_spike:
            flags.append("FLAG_CENTRAL_SPIKE")
        elif not np.isnan(ratio_1_5) and ratio_1_5 > max_central_ratio:
            flags.append("FLAG_CENTRAL_SPIKE")
        elif not np.isnan(ratio_1_5) and ratio_1_5 < min_central_ratio:
            flags.append("FLAG_CENTRAL_DROP")

        status = ";".join(flags) if flags else "GOOD"

        metric_dict = {
            "run_number": run_number,
            "file_path": str(path),
            "total_events": total_events,
            "plateau_avg": plateau_avg,
            "rms_plat_pct": rms_plat_pct,
            "max_dev_pct": max_dev_pct,
            "chi2_ndf": chi2_ndf,
            "slope_per_10pct": slope_per_10pct,
            "ratio_1": ratio_1,
            "ratio_1_5": ratio_1_5,
            "ratio_80": ratio_80,
            "values": values,
            "ratios": ratios,
            "bin_centers": bin_centers,
            "edges": edges,
            "status": status,
        }
        return metric_dict, None

    except Exception as e:
        return None, f"Error processing {path.name}: {e}"
