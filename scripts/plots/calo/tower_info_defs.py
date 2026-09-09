#!/usr/bin/env python3
"""
Python port & PyROOT bridge for sPHENIX TowerInfoDefs channel mapping.
Source: offline/packages/CaloBase/TowerInfoDefs.cc

Provides bi-directional mapping between calorimeter channel (towerIndex),
detector coordinates (ieta, iphi), and 32-bit tower key for EMCal and HCal.

Features:
- Pure Python implementation (default, zero dependencies, <1ms import).
- Optional PyROOT / libcalo_io C++ bindings (via enable_pyroot(),
  --use-pyroot flag, or SPHENIX_USE_PYROOT=1 env variable).
- Full support for tower keys:
  * encode_emcal / encode_hcal: from towerIndex OR (ieta, iphi) -> tower_key
  * decode_emcal / decode_hcal: from tower_key OR (ieta, iphi) -> towerIndex
  * get_calo_tower_eta_bin / get_calo_tower_phi_bin: from tower_key -> ieta / iphi
"""

import argparse
import os
import sys

# ---------------------------------------------------------
# PyROOT C++ Bridge Backend Configuration
# ---------------------------------------------------------
USE_PYROOT = os.environ.get("SPHENIX_USE_PYROOT", "0").lower() in ("1", "true", "yes")
_pyroot_initialized = False
_ROOT = None


def enable_pyroot(enable=True):
    """Enable or disable using C++ TowerInfoDefs via PyROOT."""
    global USE_PYROOT
    USE_PYROOT = enable
    if enable:
        _init_pyroot()


def _init_pyroot():
    """Lazily load ROOT, libcalo_io, and declare TowerInfoDefs.h."""
    global _pyroot_initialized, _ROOT
    if _pyroot_initialized:
        return
    try:
        import ROOT
        _ROOT = ROOT
        _ROOT.gSystem.Load("libcalo_io")
        _ROOT.gInterpreter.Declare("#include <calobase/TowerInfoDefs.h>")
        _pyroot_initialized = True
    except Exception as e:
        raise RuntimeError(f"Failed to initialize PyROOT TowerInfoDefs: {e}") from e


# ---------------------------------------------------------
# Tower Key Utilities (Bit manipulation)
# ---------------------------------------------------------
def get_calo_tower_eta_bin(key):
    """
    Extract eta bin (high 16 bits) from calorimeter tower key.
    Matches TowerInfoDefs::getCaloTowerEtaBin.
    """
    return int(key) >> 16


def get_calo_tower_phi_bin(key):
    """
    Extract phi bin (low 16 bits) from calorimeter tower key.
    Matches TowerInfoDefs::getCaloTowerPhiBin.
    """
    return int(key) & 0xFFFF


def get_calo_tower_key_coords(key):
    """Extract (ieta, iphi) tuple from calorimeter tower key."""
    return get_calo_tower_eta_bin(key), get_calo_tower_phi_bin(key)


# Aliases matching C++ TowerInfoDefs exact function names
getCaloTowerEtaBin = get_calo_tower_eta_bin
getCaloTowerPhiBin = get_calo_tower_phi_bin


# ---------------------------------------------------------
# Pure Python EMCal Hardware / Channel Mapping Constants
# ---------------------------------------------------------
EMCADC = [
    [62, 60, 46, 44, 30, 28, 14, 12],
    [63, 61, 47, 45, 31, 29, 15, 13],
    [58, 56, 42, 40, 26, 24, 10, 8],
    [59, 57, 43, 41, 27, 25, 11, 9],
    [54, 52, 38, 36, 22, 20, 6, 4],
    [55, 53, 39, 37, 23, 21, 7, 5],
    [50, 48, 34, 32, 18, 16, 2, 0],
    [51, 49, 35, 33, 19, 17, 3, 1],
]

EMCAL_ETAMAP = [0] * 64
EMCAL_PHIMAP = [0] * 64
for _j in range(8):
    for _k in range(8):
        _ch = EMCADC[_j][_k]
        EMCAL_ETAMAP[_ch] = _j
        EMCAL_PHIMAP[_ch] = _k

EMCAL_ETABINOFFSET = [24, 0, 48, 72]
EMCAL_ETABINMAP = [1, 0, 2, 3]

# ---------------------------------------------------------
# Pure Python HCal Hardware / Channel Mapping Constants
# ---------------------------------------------------------
HCALADC = [
    [0, 1],
    [2, 3],
    [4, 5],
    [6, 7],
    [8, 9],
    [10, 11],
    [12, 13],
    [14, 15],
]

HCAL_ETAMAP = [0] * 16
HCAL_PHIMAP = [0] * 16
for _j in range(8):
    for _k in range(2):
        _ch = HCALADC[_j][_k]
        HCAL_ETAMAP[_ch] = _j
        HCAL_PHIMAP[_ch] = _k

HCAL_ETABINOFFSET = [0, 8, 16, 0]
HCAL_PHIBINOFFSET = [0, 2, 4, 6]


# ---------------------------------------------------------
# EMCal Mapping Functions
# ---------------------------------------------------------
def get_emcal_ieta_iphi(tower_index, use_pyroot=None):
    """
    Convert EMCal tower index (0..24575) to (ieta, iphi).
    Matches TowerInfoDefs::encode_emcal(towerIndex).
    """
    if use_pyroot or (use_pyroot is None and USE_PYROOT):
        _init_pyroot()
        key = _ROOT.TowerInfoDefs.encode_emcal(int(tower_index))
        return int(_ROOT.TowerInfoDefs.getCaloTowerEtaBin(key)), int(_ROOT.TowerInfoDefs.getCaloTowerPhiBin(key))

    channels_per_sector = 64
    supersector = 64 * 12
    nchannelsperpacket = 64 * 3
    maxphibin = 7
    maxetabin = 23

    supersectornumber = tower_index // supersector
    remainder = tower_index % supersector
    packet = remainder // nchannelsperpacket
    if packet < 0 or packet > 3:
        return None, None

    packet_rem = remainder % nchannelsperpacket
    interfaceboard = packet_rem // channels_per_sector
    interfaceboard_channel = packet_rem % channels_per_sector

    localphibin = EMCAL_PHIMAP[interfaceboard_channel]
    if packet in (0, 1):
        localphibin = maxphibin - localphibin

    localetabin = EMCAL_ETAMAP[interfaceboard_channel]
    packet_etabin = localetabin + 8 * interfaceboard
    if packet in (0, 1):
        packet_etabin = maxetabin - packet_etabin

    globaletabin = packet_etabin + EMCAL_ETABINOFFSET[packet]
    globalphibin = localphibin + supersectornumber * 8
    return globaletabin, globalphibin


def encode_emcal(arg1, arg2=None, use_pyroot=None):
    """
    Encode EMCal tower index OR (ieta, iphi) into a 32-bit tower key.
    Matches TowerInfoDefs::encode_emcal(towerIndex) and encode_emcal(etabin, phibin).

    Usage:
        key = encode_emcal(tower_index)
        key = encode_emcal(ieta, iphi)
    """
    if arg2 is not None:
        ieta, iphi = int(arg1), int(arg2)
        if use_pyroot or (use_pyroot is None and USE_PYROOT):
            _init_pyroot()
            return int(_ROOT.TowerInfoDefs.encode_emcal(ieta, iphi))
        return iphi + (ieta << 16)
    else:
        tower_index = int(arg1)
        if use_pyroot or (use_pyroot is None and USE_PYROOT):
            _init_pyroot()
            return int(_ROOT.TowerInfoDefs.encode_emcal(tower_index))
        ieta, iphi = get_emcal_ieta_iphi(tower_index, use_pyroot=False)
        if ieta is None or iphi is None:
            return None
        return iphi + (ieta << 16)


get_emcal_tower_key = encode_emcal


def decode_emcal(arg1, arg2=None, use_pyroot=None):
    """
    Convert EMCal tower key OR (ieta, iphi) to channel tower index (0..24575).
    Matches TowerInfoDefs::decode_emcal(tower_key).

    Usage:
        index = decode_emcal(tower_key)
        index = decode_emcal(ieta, iphi)
    """
    if arg2 is not None:
        ieta, iphi = int(arg1), int(arg2)
        tower_key = iphi + (ieta << 16)
    else:
        tower_key = int(arg1)
        ieta = tower_key >> 16
        iphi = tower_key & 0xFFFF

    if use_pyroot or (use_pyroot is None and USE_PYROOT):
        _init_pyroot()
        return int(_ROOT.TowerInfoDefs.decode_emcal(tower_key))

    channels_per_sector = 64
    supersector = 64 * 12
    nchannelsperpacket = 64 * 3
    maxphibin = 7
    maxetabin = 23

    packet = EMCAL_ETABINMAP[int(ieta) // 24]
    localetabin = ieta - EMCAL_ETABINOFFSET[packet]
    localphibin = iphi % 8
    supersectornumber = iphi // 8

    if packet in (0, 1):
        localetabin = maxetabin - localetabin
    ib = localetabin // 8
    if packet in (0, 1):
        localphibin = maxphibin - localphibin
    localetabin = localetabin % 8

    localindex = EMCADC[localetabin][localphibin]
    index = localindex + channels_per_sector * ib + packet * nchannelsperpacket + supersector * supersectornumber
    return index


# ---------------------------------------------------------
# HCal Mapping Functions
# ---------------------------------------------------------
def get_hcal_ieta_iphi(tower_index, use_pyroot=None):
    """
    Convert HCal tower index (0..1535) to (ieta, iphi).
    Matches TowerInfoDefs::encode_hcal(towerIndex).
    """
    if use_pyroot or (use_pyroot is None and USE_PYROOT):
        _init_pyroot()
        key = _ROOT.TowerInfoDefs.encode_hcal(int(tower_index))
        return int(_ROOT.TowerInfoDefs.getCaloTowerEtaBin(key)), int(_ROOT.TowerInfoDefs.getCaloTowerPhiBin(key))

    channels_per_sector = 16
    supersector = 16 * 4 * 3
    nchannelsperpacket = channels_per_sector * 4

    supersectornumber = tower_index // supersector
    remainder = tower_index % supersector
    packet = remainder // nchannelsperpacket
    if packet < 0 or packet > 3:
        return None, None

    packet_rem = remainder % nchannelsperpacket
    interfaceboard = packet_rem // channels_per_sector
    interfaceboard_channel = packet_rem % channels_per_sector

    localphibin = HCAL_PHIMAP[interfaceboard_channel] + HCAL_PHIBINOFFSET[interfaceboard]
    localetabin = HCAL_ETAMAP[interfaceboard_channel]
    packet_etabin = localetabin

    globaletabin = packet_etabin + HCAL_ETABINOFFSET[packet]
    globalphibin = localphibin + supersectornumber * 8
    return globaletabin, globalphibin


def encode_hcal(arg1, arg2=None, use_pyroot=None):
    """
    Encode HCal tower index OR (ieta, iphi) into a 32-bit tower key.
    Matches TowerInfoDefs::encode_hcal(towerIndex) and encode_hcal(etabin, phibin).

    Usage:
        key = encode_hcal(tower_index)
        key = encode_hcal(ieta, iphi)
    """
    if arg2 is not None:
        ieta, iphi = int(arg1), int(arg2)
        if use_pyroot or (use_pyroot is None and USE_PYROOT):
            _init_pyroot()
            return int(_ROOT.TowerInfoDefs.encode_hcal(ieta, iphi))
        return iphi + (ieta << 16)
    else:
        tower_index = int(arg1)
        if use_pyroot or (use_pyroot is None and USE_PYROOT):
            _init_pyroot()
            return int(_ROOT.TowerInfoDefs.encode_hcal(tower_index))
        ieta, iphi = get_hcal_ieta_iphi(tower_index, use_pyroot=False)
        if ieta is None or iphi is None:
            return None
        return iphi + (ieta << 16)


get_hcal_tower_key = encode_hcal


def decode_hcal(arg1, arg2=None, use_pyroot=None):
    """
    Convert HCal tower key OR (ieta, iphi) to channel tower index (0..1535).
    Matches TowerInfoDefs::decode_hcal(tower_key).

    Usage:
        index = decode_hcal(tower_key)
        index = decode_hcal(ieta, iphi)
    """
    if arg2 is not None:
        ieta, iphi = int(arg1), int(arg2)
        tower_key = iphi + (ieta << 16)
    else:
        tower_key = int(arg1)
        ieta = tower_key >> 16
        iphi = tower_key & 0xFFFF

    if use_pyroot or (use_pyroot is None and USE_PYROOT):
        _init_pyroot()
        return int(_ROOT.TowerInfoDefs.decode_hcal(tower_key))

    channels_per_sector = 16
    supersector = 16 * 4 * 3
    nchannelsperpacket = channels_per_sector * 4

    localphibin = iphi % 8
    supersectornumber = iphi // 8
    packet = int(ieta) // 8
    localetabin = ieta % 8
    ib = localphibin // 2
    localphibin = localphibin - HCAL_PHIBINOFFSET[ib]

    localindex = HCALADC[localetabin][localphibin]
    index = localindex + channels_per_sector * ib + packet * nchannelsperpacket + supersector * supersectornumber
    return index


# ---------------------------------------------------------
# Generic Dispatch Functions
# ---------------------------------------------------------
def get_calo_tower_ieta_iphi(tower_index, det="EMCal", use_pyroot=None):
    """
    Generic lookup from towerIndex to (ieta, iphi).
    det: "EMCal" (or any string containing "EMCal"), or "HCal" / "IHCal" / "OHCal".
    """
    if "HCal" in det:
        return get_hcal_ieta_iphi(tower_index, use_pyroot=use_pyroot)
    return get_emcal_ieta_iphi(tower_index, use_pyroot=use_pyroot)


def get_calo_tower_key(arg1, arg2=None, det="EMCal", use_pyroot=None):
    """
    Generic lookup to get 32-bit tower key from either:
      - towerIndex: get_calo_tower_key(tower_index, det="EMCal")
      - coordinates: get_calo_tower_key(ieta, iphi, det="EMCal")
    """
    if "HCal" in det:
        return encode_hcal(arg1, arg2, use_pyroot=use_pyroot)
    return encode_emcal(arg1, arg2, use_pyroot=use_pyroot)


def get_calo_tower_index(arg1, arg2=None, det="EMCal", use_pyroot=None):
    """
    Generic reverse lookup from tower key OR (ieta, iphi) to towerIndex.

    Usage:
        get_calo_tower_index(tower_key, det="EMCal")
        get_calo_tower_index(ieta, iphi, det="EMCal")
    """
    if "HCal" in det:
        return decode_hcal(arg1, arg2, use_pyroot=use_pyroot)
    return decode_emcal(arg1, arg2, use_pyroot=use_pyroot)


# ---------------------------------------------------------
# sPHENIX CDB Bad Tower Map Support
# ---------------------------------------------------------
_CDB_URL_CACHE = {}
_BAD_TOWER_CACHE = {}
_cdb_initialized = False


def _init_cdb():
    """Lazily load libsphenixnpc and declare C++ singleton CDB helper."""
    global _cdb_initialized, _ROOT
    if _cdb_initialized:
        return
    import ROOT
    _ROOT = ROOT
    _ROOT.gSystem.Load("libsphenixnpc")
    _ROOT.gInterpreter.Declare("""
    #ifndef SPHENIX_CDB_HELPER
    #define SPHENIX_CDB_HELPER
    #include <sphenixnpc/SphenixClient.h>
    #include <sphenixnpc/CDBUtils.h>
    CDBUtils* g_sphenix_cdb = nullptr;
    std::string get_sphenix_cdb_url(const std::string& pl_type, unsigned int runnumber, const std::string& tag="newcdbtag") {
        if (!g_sphenix_cdb) {
            g_sphenix_cdb = new CDBUtils();
            g_sphenix_cdb->setGlobalTag(tag);
        }
        return g_sphenix_cdb->getUrl(pl_type, runnumber);
    }
    #endif
    """)
    _cdb_initialized = True


def get_cdb_calibration_url(payload_type, run_number, dbtag="newcdbtag"):
    """
    Query the sPHENIX CDB for the calibration ROOT file path matching run_number and payload_type.
    Uses sphenixnpc / CDBUtils via PyROOT.
    Cached per (payload_type, run_number, dbtag).
    """
    cache_key = (str(payload_type), int(run_number), str(dbtag))
    if cache_key in _CDB_URL_CACHE:
        return _CDB_URL_CACHE[cache_key]

    try:
        _init_cdb()
        url = str(_ROOT.get_sphenix_cdb_url(str(payload_type), int(run_number), str(dbtag)))
        if url.startswith("DataBaseException") or not url.endswith(".root"):
            url = None
    except Exception as e:
        print(f"Warning: Failed to query CDB for {payload_type} (run {run_number}): {e}")
        url = None

    _CDB_URL_CACHE[cache_key] = url
    return url


def get_bad_tower_map(run_number, det="CEMC", dbtag="newcdbtag"):
    """
    Load the bad tower map calibration tree for a given run and detector.
    det: 'CEMC' (EMCal), 'HCALIN', or 'HCALOUT' (or full payload type like 'CEMC_BadTowerMap').
    Returns a dict mapping towerKey (IID) -> {'sigma': float, 'status': int}.
    Cached per (det, run_number, dbtag).
    """
    cache_key = (str(det), int(run_number), str(dbtag))
    if cache_key in _BAD_TOWER_CACHE:
        return _BAD_TOWER_CACHE[cache_key]

    pl_type = det if det.endswith("_BadTowerMap") else f"{det}_BadTowerMap"
    url = get_cdb_calibration_url(pl_type, run_number, dbtag=dbtag)
    if not url:
        _BAD_TOWER_CACHE[cache_key] = {}
        return {}

    try:
        import uproot
        import numpy as np

        with uproot.open(url) as f:
            tree = f["Multiple"] if "Multiple" in f else f[f.keys()[0]]
            sigma_branch = next((b for b in tree.keys() if b.endswith("_sigma")), None)
            branches = ["IID"]
            if sigma_branch:
                branches.append(sigma_branch)
            if "Istatus" in tree:
                branches.append("Istatus")

            data = tree.arrays(branches, library="np")
            iids = data["IID"]
            sigmas = data[sigma_branch] if sigma_branch else np.full(len(iids), np.nan)
            statuses = data["Istatus"] if "Istatus" in data else np.zeros(len(iids), dtype=int)

            result = {}
            for iid, s, st in zip(iids, sigmas, statuses):
                result[int(iid)] = {"sigma": float(s), "status": int(st)}

            _BAD_TOWER_CACHE[cache_key] = result
            return result
    except Exception as e:
        print(f"Warning: Failed to load BadTowerMap from {url}: {e}")
        _BAD_TOWER_CACHE[cache_key] = {}
        return {}


get_cemc_bad_tower_map = get_bad_tower_map


# ---------------------------------------------------------
# sPHENIX CDB Hot Towers fracBadChi2 Support
# ---------------------------------------------------------
_FRAC_BAD_CHI2_CACHE = {}


def get_frac_bad_chi2_map(run_number, det="CEMC", dbtag="newcdbtag"):
    """
    Load the hot towers fracBadChi2 calibration tree for a given run and detector.
    det: 'CEMC' (EMCal), 'HCALIN', or 'HCALOUT' (or full payload type like 'CEMC_hotTowers_fracBadChi2').
    Returns a dict mapping towerKey (IID) -> float(Ffraction).
    Cached per (det, run_number, dbtag).
    """
    cache_key = (str(det), int(run_number), str(dbtag))
    if cache_key in _FRAC_BAD_CHI2_CACHE:
        return _FRAC_BAD_CHI2_CACHE[cache_key]

    if det.endswith("_hotTowers_fracBadChi2"):
        pl_type = det
    elif "EMCal" in det or det == "CEMC":
        pl_type = "CEMC_hotTowers_fracBadChi2"
    elif "HCALIN" in det:
        pl_type = "HCALIN_hotTowers_fracBadChi2"
    elif "HCALOUT" in det:
        pl_type = "HCALOUT_hotTowers_fracBadChi2"
    else:
        pl_type = f"{det}_hotTowers_fracBadChi2"

    url = get_cdb_calibration_url(pl_type, run_number, dbtag=dbtag)
    if not url:
        _FRAC_BAD_CHI2_CACHE[cache_key] = {}
        return {}

    try:
        import uproot

        with uproot.open(url) as f:
            tree = f["Multiple"] if "Multiple" in f else f[f.keys()[0]]
            data = tree.arrays(["IID", "Ffraction"], library="np")
            result = {int(iid): float(fr) for iid, fr in zip(data["IID"], data["Ffraction"])}
            _FRAC_BAD_CHI2_CACHE[cache_key] = result
            return result
    except Exception as e:
        print(f"Warning: Failed to load hotTowers_fracBadChi2 from {url}: {e}")
        _FRAC_BAD_CHI2_CACHE[cache_key] = {}
        return {}


get_cemc_frac_bad_chi2_map = get_frac_bad_chi2_map


# ---------------------------------------------------------
# Command-line Interface
# ---------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Convert between sPHENIX calorimeter tower index, (ieta, iphi), and tower key."
    )
    parser.add_argument("index", nargs="?", type=int, help="Tower index to convert.")
    parser.add_argument("--det", choices=["EMCal", "HCal"], default="EMCal", help="Calorimeter detector (default: EMCal).")
    parser.add_argument("--eta", type=int, help="ieta coordinate.")
    parser.add_argument("--phi", type=int, help="iphi coordinate.")
    parser.add_argument(
        "--key",
        type=lambda x: int(x, 0),
        help="Tower key (decimal or 0x hex) to decode.",
    )
    parser.add_argument(
        "--run",
        type=int,
        help="Optional run number to query CDB BadTowerMap & fracBadChi2.",
    )
    parser.add_argument(
        "--cdbtag",
        default="newcdbtag",
        help="CDB global tag (default: newcdbtag).",
    )
    parser.add_argument(
        "--use-pyroot",
        action="store_true",
        help="Use C++ TowerInfoDefs via PyROOT instead of pure Python.",
    )
    args = parser.parse_args()

    backend = "PyROOT (C++)" if args.use_pyroot else "Pure Python"

    cdb_info = {}
    chi2_info = {}
    if args.run is not None:
        cdb_det = "CEMC" if "EMCal" in args.det else "HCALIN"
        cdb_info = get_bad_tower_map(args.run, det=cdb_det, dbtag=args.cdbtag)
        chi2_info = get_frac_bad_chi2_map(args.run, det=cdb_det, dbtag=args.cdbtag)

    def _format_cdb_str(key_val):
        if not cdb_info and not chi2_info:
            return ""
        parts = []
        info = cdb_info.get(key_val)
        if info is not None:
            parts.append(f"z-score={info['sigma']:+.2f}, status={info['status']}")
        chi2_val = chi2_info.get(key_val)
        if chi2_val is not None:
            c_str = f"{chi2_val:.2e}" if 0 < abs(chi2_val) < 0.01 else f"{chi2_val:.2f}"
            parts.append(f"frac badChi2={c_str}")
        return ", " + ", ".join(parts) if parts else ""

    if args.key is not None:
        eta, phi = get_calo_tower_key_coords(args.key)
        idx = get_calo_tower_index(args.key, det=args.det, use_pyroot=args.use_pyroot)
        cdb_str = _format_cdb_str(args.key)
        print(
            f"[{backend}] {args.det} towerKey={args.key} (hex: {hex(args.key)}) -> "
            f"(ieta={eta}, iphi={phi}), towerIndex={idx}{cdb_str}"
        )
    elif args.eta is not None and args.phi is not None:
        idx = get_calo_tower_index(args.eta, args.phi, det=args.det, use_pyroot=args.use_pyroot)
        key = get_calo_tower_key(args.eta, args.phi, det=args.det, use_pyroot=args.use_pyroot)
        cdb_str = _format_cdb_str(key)
        print(
            f"[{backend}] {args.det} (ieta={args.eta}, iphi={args.phi}) -> "
            f"towerIndex={idx}, towerKey={key} (hex: {hex(key)}){cdb_str}"
        )
    elif args.index is not None:
        eta, phi = get_calo_tower_ieta_iphi(args.index, det=args.det, use_pyroot=args.use_pyroot)
        key = get_calo_tower_key(args.index, det=args.det, use_pyroot=args.use_pyroot)
        cdb_str = _format_cdb_str(key)
        print(
            f"[{backend}] {args.det} towerIndex={args.index} -> "
            f"(ieta={eta}, iphi={phi}), towerKey={key} (hex: {hex(key)}){cdb_str}"
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
