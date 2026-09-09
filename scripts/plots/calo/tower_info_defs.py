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
        "--use-pyroot",
        action="store_true",
        help="Use C++ TowerInfoDefs via PyROOT instead of pure Python.",
    )
    args = parser.parse_args()

    backend = "PyROOT (C++)" if args.use_pyroot else "Pure Python"

    if args.key is not None:
        eta, phi = get_calo_tower_key_coords(args.key)
        idx = get_calo_tower_index(args.key, det=args.det, use_pyroot=args.use_pyroot)
        print(
            f"[{backend}] {args.det} towerKey={args.key} (hex: {hex(args.key)}) -> "
            f"(ieta={eta}, iphi={phi}), towerIndex={idx}"
        )
    elif args.eta is not None and args.phi is not None:
        idx = get_calo_tower_index(args.eta, args.phi, det=args.det, use_pyroot=args.use_pyroot)
        key = get_calo_tower_key(args.eta, args.phi, det=args.det, use_pyroot=args.use_pyroot)
        print(
            f"[{backend}] {args.det} (ieta={args.eta}, iphi={args.phi}) -> "
            f"towerIndex={idx}, towerKey={key} (hex: {hex(key)})"
        )
    elif args.index is not None:
        eta, phi = get_calo_tower_ieta_iphi(args.index, det=args.det, use_pyroot=args.use_pyroot)
        key = get_calo_tower_key(args.index, det=args.det, use_pyroot=args.use_pyroot)
        print(
            f"[{backend}] {args.det} towerIndex={args.index} -> "
            f"(ieta={eta}, iphi={phi}), towerKey={key} (hex: {hex(key)})"
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
