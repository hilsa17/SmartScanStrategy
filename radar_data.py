import argparse
from pathlib import Path

import numpy as np


# ============================================================
# SECTION 0: settings you can change from the command line
# ============================================================

def add_data_args(p: argparse.ArgumentParser):
    g = p.add_argument_group("data settings")
    g.add_argument("--data-root", default=None,
                   help="folder where you downloaded the .h5 files. Leave empty to use fake emitters.")
    g.add_argument("--rx-mode", default="stare",
                   help="which files to use. 'stare' sees the whole spectrum, so it is our ground truth")
    g.add_argument("--f-lo", type=float, default=2.0, help="lowest frequency to watch (GHz)")
    g.add_argument("--f-hi", type=float, default=18.0, help="highest frequency to watch (GHz)")
    g.add_argument("--n-bands", type=int, default=32,
                   help="number of bands. 32 bands over 2-18 GHz = 500 MHz per band")
    g.add_argument("--slot-us", type=float, default=1000.0,
                   help="length of one time slot in microseconds (1000 us = 1 ms)")
    g.add_argument("--min-pulses", type=int, default=3,
                   help="pulses needed in a box to call it busy. "
                        "Make this bigger if the printed 'busy fraction' is above 0.4")
    g.add_argument("--n-slots", type=int, default=200, help="time slots in one episode")
    g.add_argument("--n-train-files", type=int, default=100, help="how many train files to read at most")
    g.add_argument("--n-test-files", type=int, default=50, help="how many test files to read at most")
    g.add_argument("--cache-dir", default="cache", help="processed busy maps are saved here (so 2nd run is fast)")
    g = p.add_argument_group("receiver settings")
    g.add_argument("--rx-pd", type=float, default=0.95,
                   help="chance the receiver notices a signal that IS there. 1.0 = perfect receiver")
    g.add_argument("--rx-pfa", type=float, default=0.02,
                   help="chance the receiver says 'signal!' when the band is EMPTY (noise). 0.0 = perfect receiver")
    return p


# ============================================================
# SECTION 1: read .h5 files and build the busy map
# ============================================================

def find_h5(root, split, mode):
    """Find the .h5 files whose path contains the mode (stare) and the split (train/test)."""
    root = Path(root)
    split_names = {"train": ("train",), "test": ("test",), "validation": ("val",)}[split]
    out = []
    for p in sorted(root.rglob("*.h5")):
        parts = [s.lower() for s in p.relative_to(root).parts]
        if any(mode in s for s in parts) and any(s.startswith(n) for s in parts for n in split_names):
            out.append(p)
    return out


def _find_columns(h5file):
    """Find which column is arrival time (ToA) and which is frequency (CF)."""
    names = []
    try:
        raw = h5file["metadata/feature_names"][()]
        names = [(x.decode() if isinstance(x, bytes) else str(x)).lower() for x in np.atleast_1d(raw)]
    except Exception:
        pass
    toa = next((i for i, n in enumerate(names) if "toa" in n or "time" in n), 0)   # default: column 0
    cf = next((i for i, n in enumerate(names) if n.startswith("cf") or "freq" in n), 1)  # default: column 1
    return toa, cf, names


def _to_ghz(cf):
    """Guess the unit of the frequency column and convert to GHz."""
    typical = float(np.nanmedian(cf))
    if typical > 1e6:
        return cf / 1e9     # it was in Hz
    if typical > 100:
        return cf / 1e3     # it was in MHz
    return cf               # already GHz


def pulses_to_busy_map(toa_us, cf_ghz, a):
    """Turn a list of pulses into the busy map (rows = time slots, columns = bands)."""
    good = np.isfinite(toa_us) & np.isfinite(cf_ghz)
    toa_us, cf_ghz = toa_us[good], cf_ghz[good]
    if toa_us.size == 0:
        return None
    band_width = (a.f_hi - a.f_lo) / a.n_bands
    slot = ((toa_us - toa_us.min()) // a.slot_us).astype(np.int64)     # which time slot each pulse is in
    n_slots_total = int(slot.max()) + 1
    band = np.floor((cf_ghz - a.f_lo) / band_width).astype(np.int64)   # which band each pulse is in
    inside = (band >= 0) & (band < a.n_bands)                         # ignore pulses outside 2-18 GHz
    counts = np.bincount(slot[inside] * a.n_bands + band[inside], minlength=n_slots_total * a.n_bands)
    counts = counts.reshape(n_slots_total, a.n_bands)                 # pulses per (slot, band) box
    return counts >= a.min_pulses


def load_dataset_split(a, split, max_files):
    import h5py
    cache = Path(a.cache_dir) / (f"busy_{split}_{a.rx_mode}_{a.n_bands}b_{a.f_lo}-{a.f_hi}GHz_"
                                 f"{int(a.slot_us)}us_{a.min_pulses}p_{max_files}f.npz")
    if cache.exists():
        z = np.load(cache)
        maps = [z[k] for k in sorted(z.files, key=lambda s: int(s.split("_")[1]))]
        print(f"[data] {split}: loaded {len(maps)} saved busy maps from {cache}")
        return maps

    files = find_h5(a.data_root, split, a.rx_mode)
    if not files:
        seen = [str(p.relative_to(a.data_root)) for p in list(Path(a.data_root).rglob("*.h5"))[:5]]
        raise FileNotFoundError(
            f"No '{a.rx_mode}' '{split}' .h5 files found under {a.data_root}. "
            f"Some .h5 files I can see: {seen}. Check --data-root and --rx-mode.")
    files = files[:max_files]

    maps = []
    for i, path in enumerate(files):
        try:
            with h5py.File(path, "r") as f:
                toa_col, cf_col, names = _find_columns(f)
                arr = f["data"][()]
            toa = arr[:, toa_col].astype(np.float64)
            cf = _to_ghz(arr[:, cf_col].astype(np.float64))
            if i == 0:   # print a quick check so you can see if the units look right
                print(f"[data] first file: {path.name}  shape={arr.shape}  columns={names or 'unknown'}")
                print(f"[data]   time span = {np.ptp(toa)/1e6:.2f} s (assuming microseconds)   "
                      f"frequency range = {np.nanmin(cf):.2f} to {np.nanmax(cf):.2f} GHz")
            busy = pulses_to_busy_map(toa, cf, a)
        except Exception as e:      # a bad file should not stop the whole run
            print(f"[data] skipping {path.name}: {e}")
            continue
        if busy is not None and len(busy) >= a.n_slots:
            maps.append(busy)
        if (i + 1) % 10 == 0:
            print(f"[data] {split}: done {i+1}/{len(files)} files")
    if not maps:
        raise RuntimeError("No usable files (all empty or shorter than --n-slots).")
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, *maps)
    print(f"[data] {split}: {len(maps)} busy maps saved to {cache}")
    return maps


# ============================================================
# SECTION 2: fake emitters (old code, used only when there is no dataset)
# ============================================================

class Emitter:
    """A simple fake emitter that is in ONE band at a time."""

    def __init__(self, n_bands=20, mode="fixed_periodic", seed=0, p_stay=0.85):
        self.n_bands, self.mode, self.p_stay = n_bands, mode, p_stay
        self.rng = np.random.default_rng(seed)
        self._t = 0
        self.state = int(self.rng.integers(0, n_bands))

    def step(self):
        if self.mode == "fixed_periodic":        # goes band 0,1,2,... again and again
            self.state = self._t % self.n_bands
        elif self.mode == "pseudo_random":       # jumps to a random band every slot
            self.state = int(self.rng.integers(0, self.n_bands))
        elif self.mode == "markov_restless":     # mostly stays, sometimes jumps
            if self.rng.random() > self.p_stay:
                self.state = int(self.rng.integers(0, self.n_bands))
        else:
            raise ValueError(f"unknown emitter mode {self.mode}")
        self._t += 1
        return self.state


def fake_busy_maps(mode, n_bands, n_maps, length, seed0):
    maps = []
    for i in range(n_maps):
        em = Emitter(n_bands, mode, seed=seed0 + i)
        m = np.zeros((length, n_bands), dtype=bool)
        for t in range(length):
            m[t, em.step()] = True
        maps.append(m)
    return maps


# ============================================================
# SECTION 3: small helper classes
# ============================================================

class BusyMapSource:
    """Gives the agent one time slot at a time.
    Each episode = a random window of n_slots rows from one of the busy maps.
    The window depends only on the seed, so every algorithm gets the SAME windows (fair comparison)."""

    def __init__(self, maps, n_slots):
        self.maps = [m for m in maps if len(m) >= n_slots]
        self.n_slots = n_slots
        self.n_bands = self.maps[0].shape[1]
        self.window, self.t = None, 0

    @property
    def busy_fraction(self):
        return float(np.mean([m.mean() for m in self.maps]))

    def start_episode(self, seed):
        rng = np.random.default_rng(seed)
        m = self.maps[int(rng.integers(len(self.maps)))]
        start = int(rng.integers(0, len(m) - self.n_slots + 1))
        self.window, self.t = m[start:start + self.n_slots], 0

    def next_slot(self):
        row = self.window[self.t]
        self.t += 1
        return row          # True/False for every band: which bands are really busy now


class Receiver:
    """Our receiver. It looks at ONE band per slot and is not perfect:
    it can miss a signal (rx_pd < 1) and can raise a false alarm (rx_pfa > 0)."""

    def __init__(self, pd=0.95, pfa=0.02, seed=0):
        self.pd, self.pfa = pd, pfa
        self.rng = np.random.default_rng(seed)

    def observe(self, band_is_busy):
        chance = self.pd if band_is_busy else self.pfa
        return bool(self.rng.random() < chance)     # True = receiver says "I see a signal"


def learn_stickiness(maps):
    """Learn from the training data how a band changes from one slot to the next:
         p01 = chance a band becomes busy, given it was empty
         p11 = chance a band stays busy,   given it was busy
    We average over all bands, because each pulse train has its own random emitters
    (which band is busy changes between files) but 'how sticky busy is' stays similar."""
    n_empty = n_empty_to_busy = n_busy = n_busy_to_busy = 0.0
    for m in maps:
        now, nxt = m[:-1], m[1:]
        n_empty += (~now).sum();  n_empty_to_busy += (~now & nxt).sum()
        n_busy += now.sum();      n_busy_to_busy += (now & nxt).sum()
    p01 = float(np.clip((n_empty_to_busy + 1) / (n_empty + 2), 1e-3, 1 - 1e-3))
    p11 = float(np.clip((n_busy_to_busy + 1) / (n_busy + 2), 1e-3, 1 - 1e-3))
    return p01, p11


class EpisodeLogger:
    """Keeps score for one algorithm. Numbers we report:
       P_int = fraction of slots where we looked at a busy band AND noticed it (intercept rate)
       Pd    = when the band we looked at was busy, how often did we notice it
       Pfa   = when the band we looked at was empty, how often did we wrongly say 'signal!'
       TTFI  = number of slots until our first intercept (lower is better)
       avg_reward = average reward per slot"""

    def __init__(self, name):
        self.name, self.records = name, []
        self._reset()

    def _reset(self):
        self.n = 0
        self.busy_looks = self.noticed_busy = self.empty_looks = self.false_alarms = 0
        self.reward = 0.0
        self.first = None

    def log_slot(self, t, was_busy, noticed, reward):
        self.n += 1
        self.reward += reward
        if was_busy:
            self.busy_looks += 1
            if noticed:
                self.noticed_busy += 1
                if self.first is None:
                    self.first = t
        else:
            self.empty_looks += 1
            self.false_alarms += int(noticed)

    def end_episode(self):
        self.records.append({
            "P_int": self.noticed_busy / max(1, self.n),
            "Pd": self.noticed_busy / self.busy_looks if self.busy_looks else np.nan,
            "Pfa": self.false_alarms / self.empty_looks if self.empty_looks else np.nan,
            "TTFI": self.first if self.first is not None else self.n,
            "avg_reward": self.reward / max(1, self.n),
        })
        self._reset()

    def summary(self):
        def mean(k):
            v = np.array([r[k] for r in self.records], dtype=float)
            return float(np.nanmean(v)) if np.isfinite(v).any() else float("nan")
        return {k: mean(k) for k in ("P_int", "Pd", "Pfa", "TTFI", "avg_reward")}


# ============================================================
# SECTION 4: one function that prepares all data
# ============================================================

def build_datasets(a):
    """Returns {name: {"train": BusyMapSource, "test": BusyMapSource}}"""
    out = {}
    if a.data_root:
        train_maps = load_dataset_split(a, "train", a.n_train_files)
        test_maps = load_dataset_split(a, "test", a.n_test_files)
        out["TSRD"] = {"train": BusyMapSource(train_maps, a.n_slots), "test": BusyMapSource(test_maps, a.n_slots)}
        frac = out["TSRD"]["test"].busy_fraction
        print(f"[data] {a.n_bands} bands, {a.slot_us:.0f} us slots, min_pulses={a.min_pulses}: "
              f"busy fraction = {frac:.3f}  (a random guess hits about this often)")
        if frac > 0.4:
            print("[data] WARNING: too many boxes are busy, so even random guessing works. "
                  "Increase --min-pulses or decrease --slot-us.")
        if frac < 0.005:
            print("[data] WARNING: almost nothing is busy. Decrease --min-pulses or increase --slot-us.")
    else:
        print("[data] no --data-root given -> using fake emitters")
        for mode in ("fixed_periodic", "pseudo_random", "markov_restless"):
            tr = fake_busy_maps(mode, a.n_bands, 20, 2000, seed0=10_000)
            te = fake_busy_maps(mode, a.n_bands, 20, 2000, seed0=20_000)
            out[mode] = {"train": BusyMapSource(tr, a.n_slots), "test": BusyMapSource(te, a.n_slots)}
    return out