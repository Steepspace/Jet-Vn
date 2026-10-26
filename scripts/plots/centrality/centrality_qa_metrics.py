import re
from pathlib import Path

import numpy as np
import uproot


def flat_expectation(values, edges, flat_max=80.0):
    """
    Events per 1% bin expected if bins 1..flat_max were perfectly flat: the 1-flat_max event count
    divided by the number of those bins. Bin 0 is always empty and is excluded.
    Returns (flat_ref, n_flat, n_bins); flat_ref is 0 when the range holds no events.
    """
    values = np.asarray(values, dtype=float)
    bin_centers = 0.5 * (edges[:-1] + edges[1:])
    in_range = (bin_centers > 0.5) & (bin_centers < flat_max + 0.5)
    n_bins = int(np.sum(in_range))
    n_flat = float(np.sum(values[in_range]))
    flat_ref = n_flat / n_bins if n_bins > 0 and n_flat > 0 else 0.0
    return flat_ref, n_flat, n_bins


# |z| [cm] the MinimumBiasClassifier requires of every MB event
MB_ZVTX_MAX = 60.0

# 1D hist-name prefix -> (z_min, z_max) [cm] of the |z| slice projected from the 2D histogram.
# Checked in order, so longer prefixes come before "hCentrality".
Z_SLICES = (
    ("hCentralityZOuter", (10.0, 150.0)),
    ("hCentralityZ150", (None, 150.0)),
    ("hCentralityZ60", (None, 60.0)),
    ("hCentrality", (None, 10.0)),
    ("h2ZVertexCentrality", (None, 10.0)),
)

# The other MBD trigger bit, for the trigger-exclusive selections
OTHER_TRIGGER = {"12": "14", "14": "12"}


def get_vertex_centrality_2d(f, suffix):
    """
    (values, z_edges, cent_edges) of h2ZVertexCentrality{suffix}, or None if it is missing.

    Derived selections, built from the EventQA histograms:
      _TrigNonly_MB    MB & TrigN & !TrigM: h2ZVertexCentrality minus h2ZVertexCentrality_TrigM_MB.
                       h2ZVertexCentrality holds MB & (Trig12 | Trig14), because EventQA aborts events
                       where neither trigger fired before filling it.
      <suffix>_zwM     <suffix> with each z bin reweighted so its z distribution matches TrigM's
                       (same MB requirement), separating a trigger effect from a z-distribution effect.
                       The result holds weighted counts.
    """
    m = re.fullmatch(r"(.*)_zw(\d+)", suffix)
    if m:
        base_suffix = m.group(1)
        ref_suffix = f"_Trig{m.group(2)}" + ("_MB" if base_suffix.endswith("_MB") else "")
        base = get_vertex_centrality_2d(f, base_suffix)
        ref = get_vertex_centrality_2d(f, ref_suffix)
        if base is None or ref is None:
            return None
        n_base = base[0].sum(axis=1)
        n_ref = ref[0].sum(axis=1)
        weights = np.divide(n_ref, n_base, out=np.zeros_like(n_base, dtype=float), where=n_base > 0)
        return base[0] * weights[:, None], base[1], base[2]

    m = re.fullmatch(r"_Trig(\d+)only_MB", suffix)
    if m:
        other = OTHER_TRIGGER.get(m.group(1))
        both = get_vertex_centrality_2d(f, "")
        veto = get_vertex_centrality_2d(f, f"_Trig{other}_MB") if other else None
        if both is None or veto is None:
            return None
        return both[0] - veto[0], both[1], both[2]

    name = f"h2ZVertexCentrality{suffix}"
    if name not in f:
        return None
    arr = f[name].to_numpy()
    if len(arr) != 3:
        return None
    # TH2F contents are float32, too coarse to add and subtract counts of order 1e7 exactly
    return arr[0].astype(np.float64), arr[1], arr[2]


def _parse_hist_name(hist_name):
    """(z_min, z_max, suffix, slice_desc) for a 1D hist_name, or None if it follows no convention."""
    for prefix, (z_min, z_max) in Z_SLICES:
        if hist_name.startswith(prefix):
            # Centrality is only valid for events passing the MB classifier, which requires
            # |z| < MB_ZVTX_MAX, so no centrality selection reaches 150 cm
            z_max = min(z_max, MB_ZVTX_MAX)
            slice_desc = f"{z_min:g} cm < |z| < {z_max:g} cm" if z_min is not None else f"|z| < {z_max:g} cm"
            return z_min, z_max, hist_name[len(prefix):], slice_desc
    return None


def _selection_desc(slice_desc, suffix):
    m = re.fullmatch(r"(.*)_zw(\d+)", suffix)
    zw_desc = f" (z-weighted to Trig {m.group(2)})" if m else ""
    clean_suf = (m.group(1) if m else suffix).lstrip("_")
    trig_str = clean_suf.removesuffix("_MB").replace("Trig", "Trig ").replace("only", " only")

    if not clean_suf:
        # h2ZVertexCentrality holds only events that fired Trig 12 or Trig 14 (EventQA aborts the rest)
        sel = f"{slice_desc} and (Trig 12 | Trig 14) and MB"
    elif clean_suf.endswith("_MB"):
        sel = f"{slice_desc} and {trig_str} and MB"
    else:
        sel = f"{slice_desc} and {trig_str}"
    return f"{sel}{zw_desc}"


def describe_selection(hist_name):
    """Human-readable event selection of a hist_name, e.g. '|z| < 60 cm and Trig 14 and MB'."""
    parsed = _parse_hist_name(hist_name)
    return _selection_desc(parsed[3], parsed[2]) if parsed else hist_name


def get_centrality_hist_data(f, hist_name="hCentrality"):
    """
    Retrieve 1D centrality histogram data (values, edges, title) from an open uproot file.
    If the requested 1D histogram does not exist directly, seamlessly project it from the
    corresponding 2D vertex vs. centrality histogram (e.g. h2ZVertexCentrality*) to maintain
    full backwards compatibility.

    The name is <prefix><suffix>: the prefix sets the |z| slice (see Z_SLICES), the suffix the
    2D histogram (see get_vertex_centrality_2d), e.g. hCentralityZ60_Trig14_MB is
    |z| < 60 cm & Trig 14 (Trig 12 allowed) & MB.

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

    # 2. Derive the 2D histogram and z-range from 1D hist_name conventions
    parsed = _parse_hist_name(hist_name)
    if parsed is None:
        return None, None, ""
    z_min, z_max, suffix, slice_desc = parsed

    arr = get_vertex_centrality_2d(f, suffix)
    if arr is None:
        return None, None, ""

    values_2d, edges_x, edges_y = arr
    abs_z = np.abs(0.5 * (edges_x[:-1] + edges_x[1:]))
    mask = abs_z < z_max
    if z_min is not None:
        mask &= abs_z >= z_min

    values = np.sum(values_2d[mask, :], axis=0)
    return values, edges_y, f"{_selection_desc(slice_desc, suffix)}; Centrality [%]; Events"


def get_trigger_exclusive_counts(f, z_max=60.0):
    """
    Events passing the MB classifier with |z| < z_max that fired exactly one of the MBD triggers,
    from the EventQA hZVertex_*_MB histograms: N(12 & !14) = N(12 | 14) - N(14) and vice versa.
    n_mb counts all MB events (hZVertex, filled regardless of trigger).
    Returns NaN counts if any of the histograms is missing.
    """
    counts = {}
    for key, name in (("Trig12", "hZVertex_Trig12_MB"), ("Trig14", "hZVertex_Trig14_MB"),
                      ("Trig12_or_Trig14", "hZVertex_Trig12_or_Trig14_MB"), ("MB", "hZVertex")):
        if name not in f:
            return {"n_trig12_only": np.nan, "n_trig14_only": np.nan, "n_trig_or": np.nan, "n_mb": np.nan}
        values, edges = f[name].to_numpy()
        in_range = (edges[:-1] >= -z_max) & (edges[1:] <= z_max)
        counts[key] = float(np.sum(values[in_range], dtype=float))
    return {
        "n_trig12_only": counts["Trig12_or_Trig14"] - counts["Trig14"],
        "n_trig14_only": counts["Trig12_or_Trig14"] - counts["Trig12"],
        "n_trig_or": counts["Trig12_or_Trig14"],
        "n_mb": counts["MB"],
    }


# The MBD centrality calibration fills 1%-wide bins centred at 1-92% (bin 0 is always empty);
# a calibrated run populates every bin equally.
N_CENT_CLASSES = 92

# Width [%] of the coarse centrality classes the calibration check averages over
COARSE_CLASS_WIDTH = 20.0

# Allowed (low, high) ranges of the ratio to flat in the 1% bin and the mean of the 1-5% bins
R1_CUTS = (0.70, 1.20)
R1_5_CUTS = (0.90, 1.15)


def extract_run_metrics(file_path, hist_name="hCentrality",
                        flat_max=80.0, max_class_dev=3.5, min_events=1e5,
                        r1_cuts=R1_CUTS, r1_5_cuts=R1_5_CUTS):
    """
    Fast extraction of scalar flatness and QA metrics from a single ROOT file.
    Does not render any plots.

    All ratios are relative to the flat expectation N(1-flat_max) / n_bins, i.e. the count
    every 1% bin would hold if the run were perfectly calibrated.

    The primary calibration metric averages those ratios over coarse COARSE_CLASS_WIDTH-wide
    classes (1-20%, 21-40%, ...): class_dev_max is how far, in %, the worst coarse class is
    from holding its expected share of events.

    Status flags (a run can carry more than one):
      FLAG_LOW_STATS    too few events in 1-flat_max to judge (includes empty histograms)
      FLAG_CALIB_SHIFT  broad miscalibration: class_dev_max > max_class_dev
      FLAG_R1_LOW/HIGH  ratio in the 1% bin below/above r1_cuts
      FLAG_R1_5_LOW/HIGH mean ratio of the 1-5% bins below/above r1_5_cuts
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
            trig_counts = get_trigger_exclusive_counts(f)

        values = np.asarray(values, dtype=float)
        total_events = float(np.sum(values))
        bin_centers = 0.5 * (edges[:-1] + edges[1:])

        # Bin k is centred at k%; bins 1..flat_max make up the flatness range
        flat_ref, n_flat, n_classes = flat_expectation(values, edges, flat_max)

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
                "ratio_1": np.nan,
                "ratio_1_5": np.nan,
                "ratio_80": np.nan,
                "values": values,
                "ratios": np.zeros(len(values)),
                "bin_centers": bin_centers,
                "edges": edges,
                "status": "FLAG_LOW_STATS",
                **trig_counts,
            }, None

        ratios = values / flat_ref

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
        worst_class = f"{coarse_edges[i_worst] + 1:.0f}-{coarse_edges[i_worst + 1]:.0f}%"

        # 2. 1% bin and the mean of the 1-5% bins
        ratio_1 = float(ratios[np.argmin(np.abs(bin_centers - 1.0))])
        cent15_mask = (bin_centers > 0.5) & (bin_centers < 5.5)
        ratio_1_5 = float(np.mean(ratios[cent15_mask])) if np.any(cent15_mask) else np.nan

        # 3. Peripheral ratio (75-85% window), informational only
        cent80_mask = (bin_centers >= 75.0) & (bin_centers <= 85.0)
        ratio_80 = float(np.mean(ratios[cent80_mask])) if np.any(cent80_mask) else np.nan

        # 4. Evaluate Outlier Rules
        flags = []
        if n_flat < min_events:
            # Too few events for the shape metrics to mean anything
            flags.append("FLAG_LOW_STATS")
        else:
            if class_dev_max > max_class_dev:
                flags.append("FLAG_CALIB_SHIFT")
            if ratio_1 < r1_cuts[0]:
                flags.append("FLAG_R1_LOW")
            if ratio_1 > r1_cuts[1]:
                flags.append("FLAG_R1_HIGH")
            if ratio_1_5 < r1_5_cuts[0]:
                flags.append("FLAG_R1_5_LOW")
            if ratio_1_5 > r1_5_cuts[1]:
                flags.append("FLAG_R1_5_HIGH")

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
            "ratio_1": ratio_1,
            "ratio_1_5": ratio_1_5,
            "ratio_80": ratio_80,
            "values": values,
            "ratios": ratios,
            "bin_centers": bin_centers,
            "edges": edges,
            "status": status,
            **trig_counts,
        }
        return metric_dict, None

    except Exception as e:
        return None, f"Error processing {path.name}: {e}"
