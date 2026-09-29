"""
dsp_link.py  --  the bridge between the DSP files and the schedulers (bandits / RL)
===================================================================================
The DSP files (sieve, estimator, kalman, manager) work on individual pulses. The schedulers
(bandits_all.py, rl_all.py) work on a busy map (slot x band). This file joins the two:

    pulses of one .h5 file
      -> DSP confirms a rotating radar (sieve -> estimator -> bursts line up in phase?)
      -> Kalman filter + TrackManager predict when its beam will hit us next
      -> a TIMETABLE: forced[slot] = band DSP wants the receiver on (or -1 = "DSP has no opinion")

DSP DOMINANCE = the schedulers obey the timetable first. Only slots with forced = -1 are
decided by the bandit / RL agent.
"""

import numpy as np
import h5py

from sieve import filter_mainbeam_and_convert
from estimator import estimate_t_scan
from kalman import CircularKalmanFilter
from manager import TrackManager
from radar_data import BusyMapSource, find_h5, pulses_to_busy_map, _to_ghz

CONFIRM_S = 10.0     # DSP studies the first 10 s of pulses before it may confirm a radar
MIN_BURSTS = 4       # ... and must have seen at least this many beam passes
MIN_ALIGN = 0.8      # ... and the passes must line up in phase (1.0 = perfectly periodic, ~0.4 = random)
MAINBEAM_PCT = 90    # mainbeam = the strongest 10% of the cluster's pulses (sidelobes are weaker)


def bursts_of(toas, t_scan):
    """Group mainbeam pulse times into beam passes. Returns (centre time, duration) of every pass."""
    groups = np.split(toas, np.where(np.diff(toas) > t_scan / 2)[0] + 1)
    return np.array([g.mean() for g in groups]), np.array([g[-1] - g[0] for g in groups])


def dsp_timetable(path, a):
    """One .h5 file -> (busy map, forced-band timetable, info dict about what DSP concluded)."""
    with h5py.File(path, "r") as f:
        d = f["data"][()].astype(np.float64)          # columns: ToA(us), Freq(MHz), PW, AoA(deg), Amp(dB)
    busy = pulses_to_busy_map(d[:, 0], _to_ghz(d[:, 1]), a)
    forced = np.full(len(busy), -1, dtype=np.int64)
    info = {"confirmed": False}

    t_us, f_mhz, aoa, amp = d[:, 0] - d[:, 0].min(), d[:, 1], d[:, 3], d[:, 4]
    bw_mhz = (a.f_hi - a.f_lo) / a.n_bands * 1e3
    band = np.floor((f_mhz - a.f_lo * 1e3) / bw_mhz).astype(int)
    cell = band * 100 + np.floor((aoa + 180) / 10).astype(int)       # cluster = (band, 10-degree bearing bin)
    early = (t_us < CONFIRM_S * 1e6) & (band >= 0) & (band < a.n_bands)
    cells, n = np.unique(cell[early], return_counts=True)

    for c in cells[np.argsort(-n)[:3]]:                              # try the 3 busiest clusters
        m = early & (cell == c)
        f0, a0 = np.median(f_mhz[m]), np.median(aoa[m])
        thr = np.percentile(amp[m], MAINBEAM_PCT)                    # amplitude limit for "mainbeam"
        allm = (cell == c)
        pdws = [{"toa": x, "centre_freq": y, "aoa": z, "amplitude": w}
                for x, y, z, w in zip(t_us[allm], f_mhz[allm], aoa[allm], amp[allm])]
        toas, _ = filter_mainbeam_and_convert(pdws, f0, a0, freq_tol=bw_mhz / 2, aoa_tol=5.0, min_amp_db=thr)
        seen = toas[toas < CONFIRM_S]
        if len(seen) < 30:
            continue
        t_scan = estimate_t_scan(seen[::max(1, len(seen) // 3000)])   # thin out: the estimator is slow on 1e5 pulses
        centres, taus = bursts_of(toas, t_scan)
        k = np.searchsorted(centres, CONFIRM_S)                       # passes DSP has seen in its study period
        if k < MIN_BURSTS:
            continue
        align = abs(np.exp(-2j * np.pi * centres[:k] / t_scan).sum()) / k
        if align < MIN_ALIGN:
            continue

        # ---- confirmed: track the beam with the Kalman filter, one beam pass at a time ----
        mgr, tid = TrackManager(), f"band{c // 100}"
        mgr.add_track(tid, CircularKalmanFilter(0.0, t_scan), t_scan, 0.0)
        mgr.record_hit(tid, 0.0, taus[0])                             # first pass: phase = 0 by definition
        t_state, windows, hits = centres[0], [], 0
        for i in range(1, k):                                         # study period: learn from every pass
            mgr.tracks[tid].kf.predict(centres[i] - t_state)
            mgr.record_hit(tid, 0.0, taus[i]); t_state = centres[i]
        while tid in mgr.tracks and t_state < t_us.max() / 1e6:       # after that: predict, then check the prediction
            mgr.tracks[tid].kf.predict(t_scan / 2)                    # ask half a turn after the last pass (phase ~ pi, no wrap trouble)
            t_now = t_state + t_scan / 2
            req = mgr.get_next_interrupts(t_now)[0]
            lo, hi = req["start_time"], req["start_time"] + req["duration"]
            windows.append((lo, hi))
            j = np.searchsorted(centres, lo)                          # first real pass at or after window start
            if j < len(centres) and centres[j] <= hi:                 # HIT: the beam came inside the window
                mgr.tracks[tid].kf.predict(centres[j] - t_now)
                mgr.record_hit(tid, 0.0, taus[j]); t_state = centres[j]; hits += 1
            else:                                                     # MISS: nothing came, coast to the window centre
                mgr.record_miss(tid, (lo + hi) / 2 - t_now); t_state = (lo + hi) / 2
            mgr.prune_ghosts()
        slot_s = a.slot_us / 1e6
        for lo, hi in windows:
            forced[max(0, int(lo / slot_s)): int(np.ceil(hi / slot_s)) + 1] = c // 100
        info = {"confirmed": True, "band": int(c // 100), "t_scan": float(t_scan), "align": float(align),
                "windows": len(windows), "hits": hits, "forced_fraction": float((forced >= 0).mean())}
        break
    return busy, forced, info


class DspSource(BusyMapSource):
    """Same busy-map windows as BusyMapSource, but every window also carries its slice of the DSP timetable.
    Episodes start after DSP's study period (CONFIRM_S), because before that DSP has nothing to say."""

    def __init__(self, maps, tables, n_slots, first_slot, use_dsp):
        self.maps, self.tables, self.n_slots, self.first, self.use_dsp = maps, tables, n_slots, first_slot, use_dsp
        self.n_bands = maps[0].shape[1]
        self.window, self.table, self.t = None, None, 0

    def start_episode(self, seed):
        rng = np.random.default_rng(seed)
        i = int(rng.integers(len(self.maps)))
        s = int(rng.integers(self.first, len(self.maps[i]) - self.n_slots + 1))
        self.window = self.maps[i][s:s + self.n_slots]
        self.table = self.tables[i][s:s + self.n_slots] if self.use_dsp else None
        self.t = 0

    def forced_band(self):
        return -1 if self.table is None else int(self.table[self.t])


def dsp_sources(a):
    """Runs DSP on every TEST file. Returns two sources that serve the SAME episodes:
    "plain" (agent alone) and "with_dsp" (DSP timetable overrides the agent), plus what DSP concluded per file."""
    files = find_h5(a.data_root, "test", a.rx_mode)[:a.n_test_files]
    runs = [dsp_timetable(p, a) for p in files]
    info = []
    for p, (_, forced, i) in zip(files, runs):
        info.append({"file": p.name, **i})
        print(f"[DSP] {p.name}: " + (f"radar confirmed in band {i['band']}, T_scan = {i['t_scan']:.3f} s, "
              f"{i['hits']}/{i['windows']} predicted windows caught the beam -> DSP controls {i['forced_fraction']:.1%} of slots"
              if i["confirmed"] else "no rotating radar confirmed -> bandit/RL keeps full control"))
    maps, tables = [r[0] for r in runs], [r[1] for r in runs]
    first = int(CONFIRM_S * 1e6 / a.slot_us)
    return {"plain": DspSource(maps, tables, a.n_slots, first, use_dsp=False),
            "with_dsp": DspSource(maps, tables, a.n_slots, first, use_dsp=True), "info": info}
