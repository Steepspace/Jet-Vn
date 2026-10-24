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


def get_centrality_hist_data(f, hist_name="hCentrality"):
    """
    Retrieve 1D centrality histogram data (values, edges, title) from an open uproot file.
    If the requested 1D histogram does not exist directly, seamlessly project it from the
    corresponding 2D vertex vs. centrality histogram (e.g. h2ZVertexCentrality*) to maintain
    full backwards compatibility.

    Returns:
        tuple: (values, edges, title) or (None, None, "") if not found.
    """
    # 1. Direct match in file
    if hist_name in f:
        item = f[hist_name]
        arr = item.to_numpy()
        if len(arr) == 2:
            values, edges = arr
            title = getattr(item, "title", "")
            return values, edges, title
        elif len(arr) == 3:
            # 2D histogram passed directly: project over |z| < 10 cm by default
            values_2d, edges_x, edges_y = arr
            z_centers = 0.5 * (edges_x[:-1] + edges_x[1:])
            mask = np.abs(z_centers) < 10.0
            values = np.sum(values_2d[mask, :], axis=0)
            edges = edges_y
            title = getattr(item, "title", "")
            return values, edges, title

    # 2. Derive base 2D name, z-range, and label from 1D hist_name conventions
    if hist_name.startswith("hCentralityZ150"):
        suffix = hist_name[len("hCentralityZ150"):]
        base_2d = f"h2ZVertexCentrality{suffix}"
        z_min, z_max = None, 150.0
        slice_desc = "|z| < 150 cm"
    elif hist_name.startswith("hCentralityZOuter"):
        suffix = hist_name[len("hCentralityZOuter"):]
        base_2d = f"h2ZVertexCentrality{suffix}"
        z_min, z_max = 10.0, 150.0
        slice_desc = "10 cm < |z| < 150 cm"
    elif hist_name.startswith("hCentrality"):
        suffix = hist_name[len("hCentrality"):]
        base_2d = f"h2ZVertexCentrality{suffix}"
        z_min, z_max = None, 10.0
        slice_desc = "|z| < 10 cm"
    elif hist_name.startswith("h2ZVertexCentrality"):
        base_2d = hist_name
        z_min, z_max = None, 10.0
        slice_desc = "|z| < 10 cm"
        suffix = hist_name[len("h2ZVertexCentrality"):]
    else:
        return None, None, ""

    if base_2d in f:
        item = f[base_2d]
        arr = item.to_numpy()
        if len(arr) == 3:
            values_2d, edges_x, edges_y = arr
            z_centers = 0.5 * (edges_x[:-1] + edges_x[1:])

            if z_min is not None and z_max is not None:
                mask = (np.abs(z_centers) >= z_min) & (np.abs(z_centers) < z_max)
            elif z_max is not None:
                mask = np.abs(z_centers) < z_max
            elif z_min is not None:
                mask = np.abs(z_centers) >= z_min
            else:
                mask = np.ones(len(z_centers), dtype=bool)

            values = np.sum(values_2d[mask, :], axis=0)
            edges = edges_y

            clean_suf = suffix.lstrip("_")
            if not clean_suf:
                title = f"{slice_desc} and MB; Centrality [%]; Events"
            elif clean_suf.endswith("_MB"):
                trig_str = clean_suf[:-3].replace("Trig", "Trig ")
                title = f"{slice_desc} and {trig_str} and MB; Centrality [%]; Events"
            else:
                trig_str = clean_suf.replace("Trig", "Trig ")
                title = f"{slice_desc} and {trig_str}; Centrality [%]; Events"

            return values, edges, title

    return None, None, ""


# The MBD centrality calibration defines 1%-wide classes covering 0-92%;
# a calibrated run populates every class equally.
N_CENT_CLASSES = 92

# Width [%] of the coarse centrality classes the calibration check averages over
COARSE_CLASS_WIDTH = 20.0

# Default local-defect cut [%]: max deviation of a 1% class from its 5-class running mean.
# Raised per run to 5 sigma_stat so low-count runs are not flagged on noise.
LOCAL_DEV_CUT_PCT = 3.5


def extract_run_metrics(file_path, hist_name="hCentrality",
                        flat_max=80.0, max_class_dev=3.0, max_local_dev_pct=LOCAL_DEV_CUT_PCT,
                        min_events=1e5, min_central_bin=0.60, min_central_ratio=0.80,
                        max_central_ratio=1.20, max_central_spike=1.30):
    """
    Fast extraction of scalar flatness and QA metrics from a single ROOT file.
    Does not render any plots.

    All ratios are relative to the flat expectation N(0-flat_max) / n_classes, i.e. the
    count every 1% class would hold if the run were perfectly calibrated.

    The primary calibration metric averages those ratios over coarse COARSE_CLASS_WIDTH-wide
    classes (0-20%, 20-40%, ...): class_dev_max is how far, in %, the worst coarse class is
    from holding its expected share of events.

    Status flags (a run can carry more than one):
      FLAG_LOW_STATS    too few events in 0-flat_max to judge (includes empty histograms)
      FLAG_CALIB_SHIFT  broad miscalibration: class_dev_max > max_class_dev
      FLAG_LOCAL_DEFECT narrow structure the coarse classes average away: a single-class
                        hole/spike or an anomalous 0-1% / 0-5% edge. defect_score is the
                        largest of these metrics in units of its own cut (> 1 fails).
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
            values, edges, _ = get_centrality_hist_data(f, hist_name=hist_name)
            if values is None:
                return None, f"Histogram '{hist_name}' (or base 2D projection) not found in {path.name}"

        values = np.asarray(values, dtype=float)
        total_events = float(np.sum(values))
        bin_centers = 0.5 * (edges[:-1] + edges[1:])

        # Bin k (center k) holds the (k-1)-k% class
        class_mask = (bin_centers > 0.5) & (bin_centers < flat_max + 0.5)
        n_classes = int(np.sum(class_mask))
        class_counts = values[class_mask]
        n_flat = float(np.sum(class_counts))

        if n_flat <= 0 or n_classes < 2:
            return {
                "run_number": run_number,
                "file_path": str(path),
                "hist_name": hist_name,
                "total_events": total_events,
                "flat_events": n_flat,
                "flat_ref": 0.0,
                "class_dev_max": np.nan,
                "class_dev_signed": np.nan,
                "worst_class": "",
                "coarse_edges": np.array([]),
                "coarse_ratios": np.array([]),
                "local_dev_pct": np.nan,
                "local_dev_cut_pct": np.nan,
                "defect_score": np.nan,
                "ratio_1": np.nan,
                "ratio_1_5": np.nan,
                "ratio_80": np.nan,
                "values": values,
                "ratios": np.zeros(len(values)),
                "bin_centers": bin_centers,
                "edges": edges,
                "status": "FLAG_LOW_STATS",
            }, None

        flat_ref = n_flat / n_classes
        ratios = values / flat_ref
        sigma_stat = 1.0 / np.sqrt(flat_ref)

        # 1. Coarse-class populations over 0-flat_max: mean ratio to flat in each wide class
        coarse_edges = np.append(np.arange(0.0, flat_max, COARSE_CLASS_WIDTH), flat_max)
        coarse_ratios = np.array([
            np.mean(ratios[(bin_centers > lo + 0.5) & (bin_centers < hi + 0.5)])
            for lo, hi in zip(coarse_edges[:-1], coarse_edges[1:])
        ])
        class_dev = (coarse_ratios - 1.0) * 100.0
        i_worst = int(np.argmax(np.abs(class_dev)))
        class_dev_max = float(np.abs(class_dev[i_worst]))
        class_dev_signed = float(class_dev[i_worst])
        worst_class = f"{coarse_edges[i_worst]:.0f}-{coarse_edges[i_worst + 1]:.0f}%"

        # 2. Local defects: deviation from a 5-class running mean. Starts at the 3-4% class so the
        #    0-1% class (covered by the central checks) does not leak into the running mean, and
        #    stops two classes short of flat_max where the running mean runs off the range.
        class_ratios = ratios[class_mask]
        running = np.convolve(class_ratios, np.ones(5) / 5.0, mode="same")
        local_dev = np.abs(class_ratios - running)[3:-2]
        local_dev_pct = float(np.max(local_dev)) * 100.0 if len(local_dev) > 0 else np.nan
        local_dev_cut_pct = max(max_local_dev_pct, 5.0 * sigma_stat * 100.0)

        # 3. 0-1% class (bin center 1) and 0-5% classes (bin centers 1-5)
        ratio_1 = float(ratios[np.argmin(np.abs(bin_centers - 1.0))])
        cent15_mask = (bin_centers > 0.5) & (bin_centers < 5.5)
        ratio_1_5 = float(np.mean(ratios[cent15_mask])) if np.any(cent15_mask) else np.nan

        # 4. Peripheral ratio (75-85% window), informational only
        cent80_mask = (bin_centers >= 75.0) & (bin_centers <= 85.0)
        ratio_80 = float(np.mean(ratios[cent80_mask])) if np.any(cent80_mask) else np.nan

        # 5. Narrow-structure score: each check in units of its own cut, so > 1 fails
        defect_terms = [
            (ratio_1 - 1.0) / (max_central_spike - 1.0),
            (1.0 - ratio_1) / (1.0 - min_central_bin),
            (ratio_1_5 - 1.0) / (max_central_ratio - 1.0),
            (1.0 - ratio_1_5) / (1.0 - min_central_ratio),
        ]
        if not np.isnan(local_dev_pct):
            defect_terms.append(local_dev_pct / local_dev_cut_pct)
        defect_score = float(max(defect_terms))

        # 6. Evaluate Outlier Rules
        flags = []
        if n_flat < min_events:
            # Too few events for the shape metrics to mean anything
            flags.append("FLAG_LOW_STATS")
        else:
            if class_dev_max > max_class_dev:
                flags.append("FLAG_CALIB_SHIFT")
            if defect_score > 1.0:
                flags.append("FLAG_LOCAL_DEFECT")

        status = ";".join(flags) if flags else "GOOD"

        metric_dict = {
            "run_number": run_number,
            "file_path": str(path),
            "hist_name": hist_name,
            "total_events": total_events,
            "flat_events": n_flat,
            "flat_ref": flat_ref,
            "class_dev_max": class_dev_max,
            "class_dev_signed": class_dev_signed,
            "worst_class": worst_class,
            "coarse_edges": coarse_edges,
            "coarse_ratios": coarse_ratios,
            "local_dev_pct": local_dev_pct,
            "local_dev_cut_pct": local_dev_cut_pct,
            "defect_score": defect_score,
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
