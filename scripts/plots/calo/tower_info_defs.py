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
import concurrent.futures
import csv
import functools
import os
from pathlib import Path
import re
import sys

import numpy as np
from scipy.optimize import LinearConstraint, milp
import tqdm
import uproot


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
# sPHENIX CDB ADC-to-ETower Calibration Factor Support
# ---------------------------------------------------------
_CALIB_ADC_TO_ETOWER_CACHE = {}


def get_calib_adc_to_etower_map(run_number, det="CEMC", dbtag="newcdbtag"):
    """
    Load the ADC-to-ETower calibration factor tree for a given run and detector.
    det: 'CEMC' (or 'EMCal'), 'HCALIN' (or 'IHCal'), 'HCALOUT' (or 'OHCal'),
         or a full payload type like 'CEMC_calib_ADC_to_ETower'.
    Returns a dict mapping towerKey (IID) -> float(calibration factor).
    Cached per (det, run_number, dbtag).
    """
    cache_key = (str(det), int(run_number), str(dbtag))
    if cache_key in _CALIB_ADC_TO_ETOWER_CACHE:
        return _CALIB_ADC_TO_ETOWER_CACHE[cache_key]

    if det.endswith("_calib_ADC_to_ETower"):
        pl_type = det
    elif "EMCal" in det or det == "CEMC":
        pl_type = "CEMC_calib_ADC_to_ETower"
    elif "OHCal" in det or "HCALOUT" in det:
        pl_type = "HCALOUT_calib_ADC_to_ETower"
    elif "IHCal" in det or "HCALIN" in det:
        pl_type = "HCALIN_calib_ADC_to_ETower"
    else:
        pl_type = f"{det}_calib_ADC_to_ETower"

    url = get_cdb_calibration_url(pl_type, run_number, dbtag=dbtag)
    if not url:
        url = get_cdb_calibration_url(f"{pl_type}_default", run_number, dbtag=dbtag)

    if not url:
        _CALIB_ADC_TO_ETOWER_CACHE[cache_key] = {}
        return {}

    try:
        with uproot.open(url) as f:
            tree = f["Multiple"] if "Multiple" in f else f[f.keys()[0]]

            # Find the calibration branch name
            calib_branch = None
            for cand in [
                f"F{pl_type}",
                f"F{pl_type.replace('_default', '')}",
                f"F{det}_calib_ADC_to_ETower",
                f"{det}_calib_ADC_to_ETower",
                "calib",
                "Fcalib",
            ]:
                if cand in tree.keys():
                    calib_branch = cand
                    break

            if not calib_branch:
                for b in tree.keys():
                    b_clean = b.split(";")[0]
                    if "calib" in b_clean.lower() and b_clean not in ("IID", "Istatus", "status"):
                        calib_branch = b
                        break

            if not calib_branch:
                for b in tree.keys():
                    b_clean = b.split(";")[0]
                    if b_clean not in ("IID", "Istatus", "status"):
                        calib_branch = b
                        break

            if not calib_branch or "IID" not in tree:
                print(f"Warning: Could not identify calibration branches in {url} (branches: {tree.keys()})")
                _CALIB_ADC_TO_ETOWER_CACHE[cache_key] = {}
                return {}

            data = tree.arrays(["IID", calib_branch], library="np")
            result = {int(iid): float(val) for iid, val in zip(data["IID"], data[calib_branch])}
            _CALIB_ADC_TO_ETOWER_CACHE[cache_key] = result
            return result
    except Exception as e:
        print(f"Warning: Failed to load calib_ADC_to_ETower from {url}: {e}")
        _CALIB_ADC_TO_ETOWER_CACHE[cache_key] = {}
        return {}


get_cemc_calib_adc_to_etower_map = get_calib_adc_to_etower_map
get_calo_calib_adc_to_etower_map = get_calib_adc_to_etower_map
get_calibration_map = get_calib_adc_to_etower_map


def get_calo_tower_calib_factor(arg1, arg2=None, run_number=None, det="EMCal", dbtag="newcdbtag", use_pyroot=None):
    """
    Convenience function to get the ADC-to-ETower calibration factor for a given tower.
    Usage:
        get_calo_tower_calib_factor(tower_index, run_number=68144, det="EMCal")
        get_calo_tower_calib_factor(ieta, iphi, run_number=68144, det="EMCal")
        get_calo_tower_calib_factor(tower_key, run_number=68144, det="EMCal")
    """
    if run_number is None and arg2 is not None and arg2 > 1000:
        run_number = arg2
        arg2 = None

    if run_number is None:
        raise ValueError("run_number must be specified to look up tower calibration factor.")

    if arg2 is not None:
        tower_key = get_calo_tower_key(arg1, arg2, det=det, use_pyroot=use_pyroot)
    else:
        tower_key = get_calo_tower_key(arg1, det=det, use_pyroot=use_pyroot)

    calib_map = get_calib_adc_to_etower_map(run_number, det=det, dbtag=dbtag)
    val = calib_map.get(tower_key)
    if val is None and arg2 is None:
        val = calib_map.get(int(arg1))
    return val


def query_single_run_tower_cdb(run_number, key_val, cdb_det="CEMC", dbtag="newcdbtag"):
    """
    Query CDB BadTowerMap, fracBadChi2, and calib_ADC_to_ETower for a single run and tower key.
    Safe for execution across worker processes in ProcessPoolExecutor.
    Returns: (run_number, info_dict_or_None, frac_bad_chi2_float_or_None, calib_factor_float_or_None)
    """
    info = None
    chi2_val = None
    calib_val = None
    try:
        bad_map = get_bad_tower_map(run_number, det=cdb_det, dbtag=dbtag)
        info = bad_map.get(key_val)
        if info is None:
            idx = get_calo_tower_index(key_val, det=cdb_det)
            if idx is not None:
                info = bad_map.get(idx)
    except Exception:
        pass

    try:
        chi2_map = get_frac_bad_chi2_map(run_number, det=cdb_det, dbtag=dbtag)
        chi2_val = chi2_map.get(key_val)
        if chi2_val is None:
            idx = get_calo_tower_index(key_val, det=cdb_det)
            if idx is not None:
                chi2_val = chi2_map.get(idx)
    except Exception:
        pass

    try:
        calib_map = get_calib_adc_to_etower_map(run_number, det=cdb_det, dbtag=dbtag)
        calib_val = calib_map.get(key_val)
        if calib_val is None:
            idx = get_calo_tower_index(key_val, det=cdb_det)
            if idx is not None:
                calib_val = calib_map.get(idx)
    except Exception:
        pass

    return run_number, info, chi2_val, calib_val


def extract_runs_from_file(file_path):
    """
    Read run numbers from a text file.
    Supports:
    - Integer run numbers (one per line, space-separated, or comma-separated)
    - ROOT file paths (extracts run number from filename, e.g. .../68144.root)
    - Comment lines starting with # and inline comments after #
    """
    p = Path(file_path)
    if not p.is_file():
        raise FileNotFoundError(f"Run list file '{file_path}' not found.")

    extracted = []
    with p.open("r") as f:
        for line in f:
            line = line.split("#")[0].strip()
            if not line:
                continue
            # If line looks like a file path or ROOT filename
            if "/" in line or line.endswith(".root"):
                name = Path(line).name
                try:
                    extracted.append(int(name.split(".")[0]))
                except ValueError:
                    match = re.search(r"\d+", name)
                    if match:
                        extracted.append(int(match.group()))
            else:
                for part in re.split(r"[\s,]+", line):
                    part = part.strip()
                    if not part:
                        continue
                    try:
                        extracted.append(int(part))
                    except ValueError:
                        match = re.search(r"\d+", part)
                        if match:
                            extracted.append(int(match.group()))
    return extracted


def _query_run_towers_worker(run_number, target_keys, cdb_det="CEMC", dbtag="newcdbtag"):
    """
    Query CDB BadTowerMap for a single run, returning data only for target_keys.
    Returns: (run_number, {tower_key: {'sigma': float, 'status': int}}, error_str_or_None)
    """
    try:
        bad_map = get_bad_tower_map(run_number, det=cdb_det, dbtag=dbtag)
        res = {}
        for key in target_keys:
            info = bad_map.get(key)
            if info is None:
                idx = get_calo_tower_index(key, det=cdb_det)
                if idx is not None:
                    info = bad_map.get(idx)
            if info is not None:
                res[key] = info
        return run_number, res, None
    except Exception as e:
        return run_number, {}, str(e)


def load_towers_from_csv(csv_path, det="EMCal"):
    """
    Load calorimeter towers from a CSV file.
    Supports (ieta, iphi), (towerIndex), or (towerKey).
    Also extracts run numbers if a run column exists.
    Returns: (tower_entries, extracted_runs)
    where tower_entries is a list of dict:
      {'ieta': ieta, 'iphi': iphi, 'index': idx, 'key': key}
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    with open(csv_path, "r") as f:
        raw_lines = [l.strip() for l in f if l.strip() and not l.strip().startswith("#")]

    if not raw_lines:
        return [], []

    first_line = raw_lines[0]
    has_header = False
    for col_name in ["eta", "phi", "tower", "index", "key", "run"]:
        if col_name in first_line.lower():
            has_header = True
            break

    ieta_col = None
    iphi_col = None
    tower_col = None
    key_col = None
    run_col = None
    start_idx = 0

    if has_header:
        start_idx = 1
        reader = csv.reader([first_line])
        header = [c.strip().lower() for c in next(reader)]
        for idx, col in enumerate(header):
            if col in ("ieta", "eta", "etabin", "i_eta") or ("eta" in col and "theta" not in col):
                ieta_col = idx
            elif col in ("iphi", "phi", "phibin", "i_phi") or "phi" in col:
                iphi_col = idx
            elif col in ("tower", "tower_index", "towerindex", "tower_id", "towerid", "index"):
                tower_col = idx
            elif col in ("towerkey", "tower_key", "key", "iid"):
                key_col = idx
            elif col in ("run", "runnumber", "run_number", "run_id", "runid") or (col.startswith("run") and "count" not in col and "frac" not in col):
                run_col = idx
    else:
        sample = first_line
        delim = "," if "," in sample else ("\t" if "\t" in sample else None)
        parts = [p.strip() for p in sample.split(delim)] if delim else sample.split()
        if len(parts) >= 3:
            ieta_col, iphi_col, run_col = 0, 1, 2
        elif len(parts) == 2:
            ieta_col, iphi_col = 0, 1
        elif len(parts) == 1:
            tower_col = 0

    if ieta_col is None and tower_col is None and key_col is None:
        ieta_col, iphi_col = 0, 1

    towers = []
    seen_keys = set()
    extracted_runs = []

    for line_num, line in enumerate(raw_lines[start_idx:], start=start_idx + 1):
        delim = "," if "," in line else ("\t" if "\t" in line else None)
        parts = [p.strip() for p in line.split(delim)] if delim else line.split()

        req_cols = [c for c in [ieta_col, iphi_col, tower_col, key_col] if c is not None]
        if not req_cols or len(parts) < max(req_cols) + 1:
            continue

        try:
            ieta, iphi, idx, key = None, None, None, None
            if key_col is not None and (ieta_col is None or iphi_col is None):
                key = int(parts[key_col], 0)
                ieta, iphi = get_calo_tower_key_coords(key)
                idx = get_calo_tower_index(key, det=det)
            elif tower_col is not None and (ieta_col is None or iphi_col is None):
                idx = int(parts[tower_col])
                ieta, iphi = get_calo_tower_ieta_iphi(idx, det=det)
                key = get_calo_tower_key(idx, det=det)
            else:
                ieta = int(parts[ieta_col])
                iphi = int(parts[iphi_col])
                idx = get_calo_tower_index(ieta, iphi, det=det)
                key = get_calo_tower_key(ieta, iphi, det=det)

            if key not in seen_keys:
                seen_keys.add(key)
                towers.append({
                    "ieta": ieta,
                    "iphi": iphi,
                    "index": idx,
                    "key": key,
                })

            if run_col is not None and len(parts) > run_col:
                r_str = parts[run_col]
                try:
                    r_val = int(float(r_str))
                    if r_val not in extracted_runs:
                        extracted_runs.append(r_val)
                except ValueError:
                    pass
        except Exception:
            continue

    return towers, extracted_runs


def solve_minimum_set_cover(tower_keys, candidate_runs_per_tower):
    """
    Find the minimum set of runs S such that for every tower T in tower_keys,
    S contains at least one run from candidate_runs_per_tower[T].
    Uses essential runs reduction, exact MILP (via scipy if available), or exact branch-and-bound.
    """
    valid_towers = [t for t in tower_keys if candidate_runs_per_tower.get(t)]
    if not valid_towers:
        return []

    tower_to_runs = {t: set(candidate_runs_per_tower[t]) for t in valid_towers}
    run_to_towers = {}
    for t, runs in tower_to_runs.items():
        for r in runs:
            if r not in run_to_towers:
                run_to_towers[r] = set()
            run_to_towers[r].add(t)

    selected_runs = set()
    uncovered_towers = set(valid_towers)

    # Reduction 1: Essential runs (towers with only 1 candidate run)
    changed = True
    while changed:
        changed = False
        essential = set()
        for t in list(uncovered_towers):
            avail_runs = tower_to_runs[t] & set(run_to_towers.keys())
            if len(avail_runs) == 1:
                r = next(iter(avail_runs))
                essential.add(r)

        for r in essential:
            selected_runs.add(r)
            cov = run_to_towers.get(r, set())
            uncovered_towers -= cov
            run_to_towers.pop(r, None)
            changed = True

    if not uncovered_towers:
        return sorted(selected_runs)

    # Try exact ILP via scipy
    try:
        cand_runs_list = list(run_to_towers.keys())
        uncov_towers_list = list(uncovered_towers)
        run_idx_map = {r: i for i, r in enumerate(cand_runs_list)}

        n_runs = len(cand_runs_list)
        n_towers = len(uncov_towers_list)

        c = np.ones(n_runs)
        A = np.zeros((n_towers, n_runs))
        for i, t in enumerate(uncov_towers_list):
            for r in tower_to_runs[t]:
                if r in run_idx_map:
                    A[i, run_idx_map[r]] = 1.0

        constraints = LinearConstraint(A, lb=1.0, ub=np.inf)
        integrality = np.ones(n_runs)
        res = milp(c=c, integrality=integrality, constraints=constraints, bounds=(0, 1))
        if res.success:
            chosen = [cand_runs_list[i] for i, val in enumerate(res.x) if val > 0.5]
            selected_runs.update(chosen)
            return sorted(selected_runs)
    except Exception:
        pass

    # Fallback: Branch and bound search
    cand_runs = list(run_to_towers.keys())
    greedy_selected = set()
    greedy_uncovered = set(uncovered_towers)
    while greedy_uncovered:
        best_r = max(cand_runs, key=lambda r: len(run_to_towers.get(r, set()) & greedy_uncovered))
        greedy_selected.add(best_r)
        greedy_uncovered -= run_to_towers.get(best_r, set())

    best_solution = list(greedy_selected)
    best_size = len(best_solution)

    def bnb(remaining_towers, current_runs):
        nonlocal best_solution, best_size
        if not remaining_towers:
            if len(current_runs) < best_size:
                best_size = len(current_runs)
                best_solution = list(current_runs)
            return

        if len(current_runs) >= best_size - 1:
            return

        t = min(remaining_towers, key=lambda tow: len(tower_to_runs[tow] & set(cand_runs)))
        branch_runs = sorted(
            [r for r in tower_to_runs[t] if r in cand_runs],
            key=lambda r: len(run_to_towers.get(r, set()) & remaining_towers),
            reverse=True,
        )

        for r in branch_runs:
            cov = run_to_towers.get(r, set())
            bnb(remaining_towers - cov, current_runs | {r})

    bnb(set(uncovered_towers), set())
    selected_runs.update(best_solution)
    return sorted(selected_runs)


def find_smallest_runs_for_towers(
    tower_list,
    runs,
    det="EMCal",
    dbtag="newcdbtag",
    workers=None,
    verbose=True,
):
    """
    Given a list of towers and a list of runs, finds the smallest set of runs where:
    - towers in the list have the highest |z-score| and status = 0 (good).

    tower_list: list of dicts with 'key', 'ieta', 'iphi', 'index' (from load_towers_from_csv)
                or list of (ieta, iphi) tuples.
    runs: list of integer run numbers.
    det: 'EMCal' / 'CEMC', 'IHCal' / 'HCALIN', or 'OHCal' / 'HCALOUT'.
    dbtag: CDB tag.
    workers: number of parallel worker processes.

    Returns dict with:
      'selected_runs': list of int,
      'run_coverage': dict mapping run -> list of covered tower info dicts,
      'tower_stats': dict mapping tower_key -> dict of stats,
      'uncovered_towers': list of tower info dicts without any status=0 run,
      'total_towers': int,
      'all_runs_queried': list of runs queried.
    """
    if "EMCal" in det or "CEMC" in det:
        cdb_det = "CEMC"
    elif "OHCal" in det or "HCALOUT" in det:
        cdb_det = "HCALOUT"
    elif "IHCal" in det or "HCALIN" in det:
        cdb_det = "HCALIN"
    else:
        cdb_det = "CEMC"

    # Standardize tower_list
    normalized_towers = []
    tower_by_key = {}
    for item in tower_list:
        if isinstance(item, dict):
            key = item.get("key")
            ieta = item.get("ieta")
            iphi = item.get("iphi")
            idx = item.get("index")
            if key is None:
                key = get_calo_tower_key(ieta, iphi, det=det)
            if idx is None:
                idx = get_calo_tower_index(ieta, iphi, det=det)
            entry = {"ieta": ieta, "iphi": iphi, "index": idx, "key": key}
        elif isinstance(item, (tuple, list)):
            if len(item) == 2:
                ieta, iphi = item
                idx = get_calo_tower_index(ieta, iphi, det=det)
                key = get_calo_tower_key(ieta, iphi, det=det)
            else:
                ieta, iphi, idx = item[:3]
                key = get_calo_tower_key(ieta, iphi, det=det)
            entry = {"ieta": ieta, "iphi": iphi, "index": idx, "key": key}
        else:
            key = int(item)
            ieta, iphi = get_calo_tower_key_coords(key)
            idx = get_calo_tower_index(key, det=det)
            entry = {"ieta": ieta, "iphi": iphi, "index": idx, "key": key}

        if entry["key"] not in tower_by_key:
            tower_by_key[entry["key"]] = entry
            normalized_towers.append(entry)

    target_keys = list(tower_by_key.keys())
    unique_runs = list(dict.fromkeys(runs))

    if not unique_runs:
        return {
            "selected_runs": [],
            "run_coverage": {},
            "tower_stats": {},
            "uncovered_towers": normalized_towers,
            "total_towers": len(normalized_towers),
            "all_runs_queried": [],
        }

    if verbose:
        print(f"Querying CDB ({cdb_det}, tag '{dbtag}') across {len(unique_runs)} run(s) for {len(target_keys)} tower(s)...")

    max_workers = workers if workers else min(os.cpu_count() or 4, 32, len(unique_runs))
    worker_func = functools.partial(
        _query_run_towers_worker,
        target_keys=target_keys,
        cdb_det=cdb_det,
        dbtag=dbtag,
    )

    run_results = {}
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        if tqdm is not None and len(unique_runs) > 3 and verbose:
            results_iter = tqdm.tqdm(
                executor.map(worker_func, unique_runs),
                total=len(unique_runs),
                desc="Querying CDB",
            )
        else:
            results_iter = executor.map(worker_func, unique_runs)

        for r_num, t_map, err in results_iter:
            if err and verbose:
                print(f"Warning: Failed to load BadTowerMap for run {r_num}: {err}")
            run_results[r_num] = t_map

    # Analyze per-tower results across all runs
    candidate_runs = {}
    tower_stats = {}
    uncovered_towers = []

    for key in target_keys:
        good_records = []
        had_any_record = False

        for r in unique_runs:
            t_info = run_results.get(r, {}).get(key)
            if t_info is not None:
                had_any_record = True
                st = t_info.get("status")
                sig = t_info.get("sigma")
                if st == 0 and sig is not None and not (isinstance(sig, float) and np.isnan(sig)):
                    good_records.append((r, float(sig), abs(float(sig))))

        if not good_records:
            reason = "Status was non-zero in all queried runs (always bad)" if had_any_record else "No CDB BadTowerMap record found"
            uncovered_towers.append({**tower_by_key[key], "reason": reason})
        else:
            max_abs_z = max(rec[2] for rec in good_records)
            best_runs = [rec[0] for rec in good_records if abs(rec[2] - max_abs_z) < 1e-4]
            candidate_runs[key] = best_runs
            tower_stats[key] = {
                "tower": tower_by_key[key],
                "max_abs_z": max_abs_z,
                "best_runs": best_runs,
                "good_records_count": len(good_records),
                "run_data": {rec[0]: {"sigma": rec[1], "status": 0} for rec in good_records},
            }

    valid_keys = list(candidate_runs.keys())
    selected_runs = solve_minimum_set_cover(valid_keys, candidate_runs)

    # Organize coverage breakdown: assign each covered tower to one of the selected runs
    run_coverage = {r: [] for r in selected_runs}
    if selected_runs:
        # Sort runs by number of candidates they can cover descending
        sorted_sel = sorted(selected_runs, key=lambda r: sum(1 for k in valid_keys if r in candidate_runs[k]), reverse=True)
        assigned_keys = set()
        for r in sorted_sel:
            for k in valid_keys:
                if k not in assigned_keys and r in candidate_runs[k]:
                    assigned_keys.add(k)
                    t_entry = tower_by_key[k]
                    sig = tower_stats[k]["run_data"][r]["sigma"]
                    run_coverage[r].append({
                        **t_entry,
                        "sigma": sig,
                        "status": 0,
                        "max_abs_z": tower_stats[k]["max_abs_z"],
                    })

    return {
        "selected_runs": selected_runs,
        "run_coverage": run_coverage,
        "tower_stats": tower_stats,
        "uncovered_towers": uncovered_towers,
        "total_towers": len(normalized_towers),
        "all_runs_queried": unique_runs,
    }


def print_smallest_runs_report(result, det="CEMC", dbtag="newcdbtag"):
    """Format and print the summary report for find_smallest_runs_for_towers."""
    selected_runs = result["selected_runs"]
    run_coverage = result["run_coverage"]
    uncovered = result["uncovered_towers"]
    total_towers = result["total_towers"]
    covered_count = total_towers - len(uncovered)
    total_runs = len(result["all_runs_queried"])

    print("\n" + "=" * 80)
    print("Smallest Set of Runs for Towers with Highest |z-score| and Status = 0 (Good)")
    print("=" * 80)
    print(f"Detector:         {det} (CDB Tag: {dbtag})")
    print(f"Towers requested: {total_towers}")
    print(f"Runs analyzed:    {total_runs}")
    print(f"Runs selected:    {len(selected_runs)} (covers {covered_count}/{total_towers} towers at peak |z-score| with status=0)")
    print(f"Selected Runs:    {', '.join(map(str, selected_runs)) if selected_runs else 'None'}")
    print("-" * 80)

    if selected_runs:
        print("Run Breakdown:")
        for r in selected_runs:
            t_list = run_coverage.get(r, [])
            print(f"\n  Run {r} (covers {len(t_list)} tower{'s' if len(t_list) != 1 else ''}):")
            for t in t_list:
                z_str = f"{t['sigma']:+.2f}"
                abs_z = f"{abs(t['sigma']):.2f}"
                print(f"    - Tower (ieta={t['ieta']:2d}, iphi={t['iphi']:3d}, idx={t['index']:5d}, key={hex(t['key'])}): z-score = {z_str} (|z| = {abs_z}), status = {t['status']}")

    if uncovered:
        print("\n" + "-" * 80)
        print(f"Uncovered Towers ({len(uncovered)}): [No run in list had status = 0 (good)]")
        for t in uncovered:
            print(f"    - Tower (ieta={t['ieta']:2d}, iphi={t['iphi']:3d}, idx={t['index']:5d}, key={hex(t['key'])}): {t.get('reason', 'No good status')}")

    print("=" * 80)
    if selected_runs:
        print("Selected runs (space-separated):")
        print(" ".join(map(str, selected_runs)))
        print("=" * 80)


def save_smallest_runs_csv(result, output_path):
    """
    Save the determined (ieta, iphi, run) results to a CSV file.
    Columns: ieta, iphi, run, tower_index, tower_key, z_score, status
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    run_coverage = result.get("run_coverage", {})
    for r, towers in run_coverage.items():
        for t in towers:
            rows.append({
                "ieta": t["ieta"],
                "iphi": t["iphi"],
                "run": r,
                "tower_index": t.get("index", ""),
                "tower_key": int(t["key"]) if "key" in t else "",
                "z_score": f"{t['sigma']:.4f}" if "sigma" in t and t["sigma"] is not None else "",
                "status": t.get("status", 0),
            })

    # Sort by ieta, iphi for clean consistent ordering
    rows.sort(key=lambda x: (x["ieta"], x["iphi"]))

    fieldnames = ["ieta", "iphi", "run", "tower_index", "tower_key", "z_score", "status"]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved determined results for {len(rows)} tower(s) to CSV: {output_path}")
    if result.get("uncovered_towers"):
        print(f"Note: {len(result['uncovered_towers'])} uncovered tower(s) were excluded from CSV because no run had status=0 (good).")


# ---------------------------------------------------------
# Command-line Interface
# ---------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Convert between sPHENIX calorimeter tower index, (ieta, iphi), and tower key, or find smallest set of runs for a tower CSV."
    )
    parser.add_argument("index", nargs="?", type=int, help="Tower index to convert.")
    parser.add_argument("--det", choices=["EMCal", "CEMC", "HCal", "IHCal", "OHCal", "HCALIN", "HCALOUT"], default="EMCal", help="Calorimeter detector (default: EMCal).")
    parser.add_argument("--eta", type=int, help="ieta coordinate.")
    parser.add_argument("--phi", type=int, help="iphi coordinate.")
    parser.add_argument(
        "--key",
        type=lambda x: int(x, 0),
        help="Tower key (decimal or 0x hex) to decode.",
    )
    parser.add_argument(
        "-c",
        "--csv",
        "--tower-csv",
        "--towers-csv",
        type=Path,
        dest="csv",
        help="Path to CSV file containing (ieta,iphi) tower list.",
    )
    parser.add_argument(
        "-o",
        "--output",
        "--output-runs",
        type=Path,
        dest="output_runs",
        help="Optional text file path to write the selected run numbers (one per line).",
    )
    parser.add_argument(
        "--output-csv",
        "--out-csv",
        type=Path,
        dest="output_csv",
        help="Optional CSV file path to write the determined results (ieta, iphi, run, tower_index, tower_key, z_score, status).",
    )
    parser.add_argument(
        "-f",
        "--file",
        "--run-file",
        "--run-list",
        type=Path,
        help="Optional text file containing run numbers or ROOT file paths (one per line).",
    )
    parser.add_argument(
        "--run",
        "--runs",
        nargs="+",
        action="extend",
        help="Optional run number(s) or run list file(s) to query CDB BadTowerMap, fracBadChi2, & calib_ADC_to_ETower.",
    )
    parser.add_argument(
        "-j",
        "--jobs",
        "--workers",
        dest="workers",
        type=int,
        default=None,
        help="Number of parallel worker processes to query CDB (default: min(cpu_count, 32)).",
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

    # --- Mode 1: CSV of (ieta, iphi) towers ---
    if args.csv is not None:
        try:
            csv_towers, csv_runs = load_towers_from_csv(args.csv, det=args.det)
        except Exception as e:
            print(f"Error loading tower CSV '{args.csv}': {e}")
            sys.exit(1)

        if not csv_towers:
            print(f"Warning: No valid towers found in CSV file '{args.csv}'.")
            return

        print(f"Loaded {len(csv_towers)} unique tower(s) from {args.csv}.")

        runs = []
        if args.file:
            try:
                runs.extend(extract_runs_from_file(args.file))
            except Exception as e:
                print(f"Error reading run file '{args.file}': {e}")
                sys.exit(1)

        if args.run:
            for r_arg in args.run:
                if Path(r_arg).is_file():
                    try:
                        runs.extend(extract_runs_from_file(r_arg))
                    except Exception as e:
                        print(f"Error reading run file '{r_arg}': {e}")
                        sys.exit(1)
                    continue

                for r_str in str(r_arg).split(","):
                    r_str = r_str.strip()
                    if r_str:
                        try:
                            runs.append(int(r_str))
                        except ValueError:
                            print(f"Error: Invalid run number or file '{r_str}'")
                            sys.exit(1)

        if not runs and csv_runs:
            runs.extend(csv_runs)

        runs = list(dict.fromkeys(runs))

        if not runs:
            print(f"No run list provided. Displaying first {min(len(csv_towers), 10)} tower(s) from CSV:")
            for t in csv_towers[:10]:
                print(f"  (ieta={t['ieta']}, iphi={t['iphi']}) -> towerIndex={t['index']}, towerKey={t['key']} (hex: {hex(t['key'])})")
            if len(csv_towers) > 10:
                print(f"  ... and {len(csv_towers) - 10} more tower(s).")
            print("\nTo find the smallest set of runs with highest |z-score| and status=0, provide a run list via -f/--file or --run.")
            return

        result = find_smallest_runs_for_towers(
            csv_towers,
            runs,
            det=args.det,
            dbtag=args.cdbtag,
            workers=args.workers,
            verbose=True,
        )

        print_smallest_runs_report(result, det=args.det, dbtag=args.cdbtag)

        if args.output_runs:
            out_p = Path(args.output_runs)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            with open(out_p, "w") as f:
                for r in result["selected_runs"]:
                    f.write(f"{r}\n")
            print(f"\nSaved {len(result['selected_runs'])} selected run(s) to {out_p}")

        if args.output_csv:
            save_smallest_runs_csv(result, args.output_csv)
        elif args.output_runs and str(args.output_runs).lower().endswith(".csv"):
            save_smallest_runs_csv(result, args.output_runs)

        return

    # --- Mode 2: Single tower conversion / lookup ---
    backend = "PyROOT (C++)" if args.use_pyroot else "Pure Python"

    if args.key is not None:
        target_key = args.key
        eta, phi = get_calo_tower_key_coords(target_key)
        idx = get_calo_tower_index(target_key, det=args.det, use_pyroot=args.use_pyroot)
        base = (
            f"[{backend}] {args.det} towerKey={target_key} (hex: {hex(target_key)}) -> "
            f"(ieta={eta}, iphi={phi}), towerIndex={idx}"
        )
    elif args.eta is not None and args.phi is not None:
        idx = get_calo_tower_index(args.eta, args.phi, det=args.det, use_pyroot=args.use_pyroot)
        target_key = get_calo_tower_key(args.eta, args.phi, det=args.det, use_pyroot=args.use_pyroot)
        base = (
            f"[{backend}] {args.det} (ieta={args.eta}, iphi={args.phi}) -> "
            f"towerIndex={idx}, towerKey={target_key} (hex: {hex(target_key)})"
        )
    elif args.index is not None:
        eta, phi = get_calo_tower_ieta_iphi(args.index, det=args.det, use_pyroot=args.use_pyroot)
        target_key = get_calo_tower_key(args.index, det=args.det, use_pyroot=args.use_pyroot)
        base = (
            f"[{backend}] {args.det} towerIndex={args.index} -> "
            f"(ieta={eta}, iphi={phi}), towerKey={target_key} (hex: {hex(target_key)})"
        )
    else:
        parser.print_help()
        return

    runs = []
    if args.file:
        try:
            runs.extend(extract_runs_from_file(args.file))
        except Exception as e:
            print(f"Error reading run file '{args.file}': {e}")
            sys.exit(1)

    if args.run:
        for r_arg in args.run:
            if Path(r_arg).is_file():
                try:
                    runs.extend(extract_runs_from_file(r_arg))
                except Exception as e:
                    print(f"Error reading run file '{r_arg}': {e}")
                    sys.exit(1)
                continue

            for r_str in str(r_arg).split(","):
                r_str = r_str.strip()
                if r_str:
                    try:
                        runs.append(int(r_str))
                    except ValueError:
                        print(f"Error: Invalid run number or file '{r_str}'")
                        sys.exit(1)
        runs = list(dict.fromkeys(runs))

    if not runs:
        print(base)
        return

    if "EMCal" in args.det or "CEMC" in args.det:
        cdb_dets = ["CEMC"]
    elif "OHCal" in args.det or "HCALOUT" in args.det:
        cdb_dets = ["HCALOUT"]
    elif "IHCal" in args.det or "HCALIN" in args.det:
        cdb_dets = ["HCALIN"]
    else:
        cdb_dets = ["HCALIN", "HCALOUT"]

    def format_cdb_parts(info, chi2_val, calib_val):
        parts = []
        if info is not None:
            parts.append(f"z-score={info['sigma']:+.2f}, status={info['status']}")
        if chi2_val is not None:
            c_str = f"{chi2_val:.2e}" if 0 < abs(chi2_val) < 0.01 else f"{chi2_val:.2f}"
            parts.append(f"frac badChi2={c_str}")
        if calib_val is not None:
            if 0 < abs(calib_val) < 0.001 or abs(calib_val) >= 10000:
                calib_str = f"{calib_val:.4e}"
            else:
                calib_str = f"{calib_val:.6g}"
            parts.append(f"calib={calib_str} GeV/ADC")
        return parts

    if len(runs) == 1:
        r = runs[0]
        run_parts = []
        for det_name in cdb_dets:
            _, info, chi2_val, calib_val = query_single_run_tower_cdb(r, target_key, cdb_det=det_name, dbtag=args.cdbtag)
            p = format_cdb_parts(info, chi2_val, calib_val)
            if p:
                prefix = f"{det_name}: " if len(cdb_dets) > 1 else ""
                run_parts.append(f"{prefix}{', '.join(p)}")
        cdb_str = f", {'; '.join(run_parts)}" if run_parts else ""
        print(f"{base}{cdb_str}")
        return

    max_workers = args.workers if args.workers else min(os.cpu_count() or 4, 32, len(runs))
    if len(cdb_dets) == 1:
        worker_func = functools.partial(
            query_single_run_tower_cdb,
            key_val=target_key,
            cdb_det=cdb_dets[0],
            dbtag=args.cdbtag,
        )
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            if tqdm is not None and len(runs) > 3:
                results = list(tqdm.tqdm(executor.map(worker_func, runs), total=len(runs), desc="Querying CDB"))
            else:
                results = list(executor.map(worker_func, runs))

        print(base)
        for r, info, chi2_val, calib_val in results:
            parts = format_cdb_parts(info, chi2_val, calib_val)
            out_str = ", ".join(parts) if parts else "no CDB record"
            print(f"  Run {r}: {out_str}")
    else:
        def worker_multi(r):
            det_res = {}
            for det_name in cdb_dets:
                _, info, chi2_val, calib_val = query_single_run_tower_cdb(r, target_key, cdb_det=det_name, dbtag=args.cdbtag)
                det_res[det_name] = (info, chi2_val, calib_val)
            return r, det_res

        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            if tqdm is not None and len(runs) > 3:
                results = list(tqdm.tqdm(executor.map(worker_multi, runs), total=len(runs), desc="Querying CDB"))
            else:
                results = list(executor.map(worker_multi, runs))

        print(base)
        for r, det_res in results:
            run_parts = []
            for det_name, (info, chi2_val, calib_val) in det_res.items():
                p = format_cdb_parts(info, chi2_val, calib_val)
                if p:
                    run_parts.append(f"{det_name}: {', '.join(p)}")
            out_str = "; ".join(run_parts) if run_parts else "no CDB record"
            print(f"  Run {r}: {out_str}")


if __name__ == "__main__":
    main()
