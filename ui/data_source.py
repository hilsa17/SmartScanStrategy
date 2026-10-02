"""
Loads the real TSRD busy-map data once at import time (from the cached
.npz if present — fast, no .h5 files needed — or from backend/data/h5/*.h5
if you've dropped the raw captures there and the cache doesn't exist yet),
and hands out a fresh BusyMapSource per live-demo session so every
connected viewer gets independent episode playback over the same
underlying real recordings.

This replaces the earlier synthetic emitter simulator: the live demo now
sweeps real recorded pulse activity, not scripted fake emitters.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import radar_data as rd  # noqa: E402

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
H5_DIR = DATA_DIR / "h5"          # raw .h5 captures — NOT committed (380MB, too large for git)
CACHE_DIR = DATA_DIR / "cache"     # precomputed busy maps — committed (a few hundred KB)

N_BANDS = 32
F_LO, F_HI = 0.5, 18.0     # GHz — matches metadata/receiver/freq_range_mhz in the captures
SLOT_US = 1000.0            # 1 ms slots
MIN_PULSES = 3
N_SLOTS = 200                # slots per episode window shown in the live demo
N_TRAIN_FILES, N_TEST_FILES = 100, 50   # upper bounds; we only have 19 files total


def _args():
    p = argparse.ArgumentParser()
    rd.add_data_args(p)
    return p.parse_args([
        "--data-root", str(H5_DIR),
        "--f-lo", str(F_LO), "--f-hi", str(F_HI), "--n-bands", str(N_BANDS),
        "--slot-us", str(SLOT_US), "--min-pulses", str(MIN_PULSES), "--n-slots", str(N_SLOTS),
        "--n-train-files", str(N_TRAIN_FILES), "--n-test-files", str(N_TEST_FILES),
        "--cache-dir", str(CACHE_DIR),
    ])


_a = _args()
_datasets = rd.build_datasets(_a)
TEST_MAPS = _datasets["TSRD"]["test"].maps
TRAIN_MAPS = _datasets["TSRD"]["train"].maps
BAND_WIDTH_GHZ = (F_HI - F_LO) / N_BANDS


def band_label(band: int) -> str:
    lo = F_LO + band * BAND_WIDTH_GHZ
    hi = lo + BAND_WIDTH_GHZ
    return f"{lo:.2f}-{hi:.2f} GHz"


def new_source():
    """A fresh, independent episode-sampling source over the held-out TEST
    captures — the same split final_result.json's numbers were measured on."""
    return rd.BusyMapSource(TEST_MAPS, N_SLOTS)
