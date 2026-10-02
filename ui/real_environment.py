"""
Wraps a radar_data.BusyMapSource (real recorded TSRD pulse activity) into
the same tick() / context_matrix() / register_hit() contract the scheduler
and metrics engine use — so swapping the synthetic simulator for real data
required no changes to scheduler.py or metrics_engine.py.
"""
import numpy as np

from . import data_source


class RealRFEnvironment:
    def __init__(self, source, seed0: int = 0):
        self.source = source
        self.n_bands = source.n_bands
        self._seed = seed0
        self.step = 0
        self.decay = 0.95
        self.band_activity_ema = np.zeros(self.n_bands)
        self.band_hit_ema = np.zeros(self.n_bands)
        self.band_last_active_gap = np.full(self.n_bands, 999.0)
        self._start_episode()

    def _start_episode(self):
        self.source.start_episode(self._seed)
        self._seed += 1

    def tick(self) -> np.ndarray:
        if self.source.window is None or self.source.t >= self.source.n_slots:
            self._start_episode()
        active = self.source.next_slot().copy()
        self.band_activity_ema = self.decay * self.band_activity_ema + (1 - self.decay) * active
        self.band_last_active_gap = np.where(active, 0, self.band_last_active_gap + 1)
        self.step += 1
        return active

    def context_matrix(self) -> np.ndarray:
        gap_norm = np.clip(self.band_last_active_gap / 20.0, 0, 1)
        global_rate = float(self.band_activity_ema.mean())
        return np.stack([
            np.ones(self.n_bands),
            self.band_activity_ema,
            self.band_hit_ema,
            1.0 - gap_norm,
            np.full(self.n_bands, global_rate),
        ], axis=1)

    def register_hit(self, band: int):
        onehot = np.zeros(self.n_bands)
        onehot[band] = 1
        self.band_hit_ema = self.decay * self.band_hit_ema + (1 - self.decay) * onehot

    def active_band_snapshot(self, active: np.ndarray):
        """List of currently-active bands with their real GHz range, for the GUI."""
        return [
            dict(band=int(b), ghz=data_source.band_label(int(b)))
            for b in np.flatnonzero(active)
        ]
