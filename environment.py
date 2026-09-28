"""
RF environment: ground-truth spectrum simulator.
Same logic as the JS prototype, ported so the backend and any future
hardware-in-the-loop feed share one contract with the scheduler.
"""
import random
import numpy as np

N_BANDS = 16

EMITTER_DEFS = [
    dict(name="Fixed Radar A", type="fixed", band=2, p_on=0.10, p_off=0.35),
    dict(name="Fixed Radar B", type="fixed", band=12, p_on=0.06, p_off=0.30),
    dict(name="Hopper α", type="hopper", bands=[1, 4, 7, 10, 13], p_hop=0.5, p_on=0.5, p_off=0.6),
    dict(name="Hopper β", type="hopper", bands=[3, 6, 9, 14], p_hop=0.35, p_on=0.4, p_off=0.55),
    dict(name="Periodic Scanner (threat)", type="scan", period=5, p_on=0.9, p_off=0.9),
    dict(name="Comms Burst", type="sporadic", bands=[0, 5, 8, 11, 15], p_on=0.04, p_off=0.5),
]


class Emitter:
    def __init__(self, d):
        self.__dict__.update(d)
        self.on = False
        self.cur_band = self.band if self.type == "fixed" else (self.bands[0] if hasattr(self, "bands") else 0)
        self.on_since = -1
        self.counted_this_burst = False
        self.scan_idx = 0

    def band_now(self):
        if self.type == "fixed":
            return self.band
        if self.type == "scan":
            return self.scan_idx
        return self.cur_band

    def tick(self, step):
        if self.type == "fixed":
            if not self.on and random.random() < self.p_on:
                self.on, self.on_since, self.counted_this_burst = True, step, False
            elif self.on and random.random() < self.p_off:
                self.on = False
        elif self.type == "hopper":
            if random.random() < self.p_hop:
                self.cur_band = random.choice(self.bands)
            if not self.on and random.random() < self.p_on:
                self.on, self.on_since, self.counted_this_burst = True, step, False
            elif self.on and random.random() < self.p_off:
                self.on = False
        elif self.type == "scan":
            if step % self.period == 0:
                self.scan_idx = (self.scan_idx + 1) % N_BANDS
                self.on = random.random() < self.p_on
                if self.on:
                    self.on_since, self.counted_this_burst = step, False
        elif self.type == "sporadic":
            if not self.on and random.random() < self.p_on:
                self.on = True
                self.cur_band = random.choice(self.bands)
                self.on_since, self.counted_this_burst = step, False
            elif self.on and random.random() < self.p_off:
                self.on = False


class RFEnvironment:
    def __init__(self, n_bands=N_BANDS):
        self.n_bands = n_bands
        self.emitters = [Emitter(d) for d in EMITTER_DEFS]
        self.step = 0
        # decayed per-band activity stats, used as scheduler context features
        self.decay = 0.95
        self.band_activity_ema = np.zeros(n_bands)
        self.band_hit_ema = np.zeros(n_bands)
        self.band_last_active_gap = np.full(n_bands, 999.0)

    def tick(self):
        """Advance ground truth by one step. Returns bool array, len n_bands."""
        active = np.zeros(self.n_bands, dtype=bool)
        for em in self.emitters:
            em.tick(self.step)
            if em.on:
                active[em.band_now()] = True
        self.band_activity_ema = self.decay * self.band_activity_ema + (1 - self.decay) * active
        self.band_last_active_gap = np.where(active, 0, self.band_last_active_gap + 1)
        self.step += 1
        return active

    def context_matrix(self):
        """Feature vector per band for the contextual bandit: [bias, activity_ema, hit_ema, recency, global_rate]."""
        gap_norm = np.clip(self.band_last_active_gap / 20.0, 0, 1)
        global_rate = float(self.band_activity_ema.mean())
        ctx = np.stack([
            np.ones(self.n_bands),
            self.band_activity_ema,
            self.band_hit_ema,
            1.0 - gap_norm,
            np.full(self.n_bands, global_rate),
        ], axis=1)
        return ctx

    def register_hit(self, band):
        onehot = np.zeros(self.n_bands)
        onehot[band] = 1
        self.band_hit_ema = self.decay * self.band_hit_ema + (1 - self.decay) * onehot

    def emitter_snapshot(self):
        return [dict(name=e.name, type=e.type, on=e.on, band=e.band_now(),
                     on_since=e.on_since, counted=e.counted_this_burst) for e in self.emitters]
