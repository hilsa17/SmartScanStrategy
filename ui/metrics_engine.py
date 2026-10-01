"""
Evaluation Metrics Engine
==========================
Single source of truth for every figure of merit used across the project:
the live scheduler demo (ws_live.py) and the final trained-model results
(results_loader.py) both call into this module, so a judge looking at
either surface sees numbers computed the same way, by the same code.

Figures of merit (per Adamy, EW 101 — receiver performance metrics):

  P_int   Probability of Intercept   = intercepted pulses / total transmitted pulses
  TTFI    Time-to-First-Intercept    = latency from emitter power-on to first detect
  Pd      Probability of Detection   = true positives / actual positives   (energy-detector confusion matrix)
  Pfa     Probability of False Alarm = false positives / actual negatives
  IRE     Interception Ratio Error   = | observed intercept ratio − ground-truth activity ratio |
  n_duty  Receiver duty-cycle eff.   = T_dwell / (T_dwell + T_tune)
  avg_reward                        = mean per-step scheduler reward (hit − false-alarm cost − retune cost)

Two call shapes are supported:
  1. `summarize_step_log(steps)` — raw per-step dicts from a live run
     (what the WebSocket scheduler demo produces every tick).
  2. Fields already aggregated offline by a training/eval script (what
     final_result.json contains per policy) are passed straight through
     by `normalize_offline_row()` so both paths render through identical
     frontend code.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import numpy as np

TUNE_MS_DEFAULT = 28.0


@dataclass
class FigureOfMerit:
    p_int: float
    ttfi: Optional[float]
    pd: Optional[float] = None
    pfa: Optional[float] = None
    ire: Optional[float] = None
    n_duty: Optional[float] = None
    avg_reward: Optional[float] = None
    hits: int = 0
    total_pulses: int = 0
    false_alarms: int = 0
    nonactive_dwells: int = 0

    def as_dict(self):
        return {k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in self.__dict__.items()}


class MetricsEngine:
    """
    Stateful accumulator: feed it one dwell/step at a time (hit or miss,
    false-alarm flag, reward), it maintains running totals and returns the
    current figure-of-merit snapshot on demand. Used by the live demo so
    numbers update every tick without recomputing from full history.
    """

    def __init__(self, dwell_ms: float = 140.0, tune_ms: float = TUNE_MS_DEFAULT):
        self.dwell_ms = dwell_ms
        self.tune_ms = tune_ms
        self.step = 0
        self.total_pulses = 0
        self.hits = 0
        self.false_alarms = 0
        self.nonactive_dwells = 0
        self.cum_reward = 0.0
        self._ttfi_samples: list[float] = []
        self._active_ring: list[np.ndarray] = []
        self._ring_cap = 200

    def record_step(self, active_mask: np.ndarray, band_watch: int, hit: bool,
                     false_alarm: bool = False, reward: float = 0.0,
                     ttfi_sample: Optional[float] = None):
        self.step += 1
        n_active = int(active_mask.sum())
        self.total_pulses += n_active
        self._active_ring.append(active_mask.copy())
        if len(self._active_ring) > self._ring_cap:
            self._active_ring.pop(0)

        if hit:
            self.hits += 1
            if ttfi_sample is not None:
                self._ttfi_samples.append(ttfi_sample)
                if len(self._ttfi_samples) > 200:
                    self._ttfi_samples.pop(0)
        else:
            self.nonactive_dwells += 1
            if false_alarm:
                self.false_alarms += 1
        self.cum_reward += reward

    def snapshot(self) -> FigureOfMerit:
        p_int = self.hits / self.total_pulses if self.total_pulses else 0.0
        ttfi = float(np.mean(self._ttfi_samples)) if self._ttfi_samples else None
        if self._active_ring:
            gt_ratio = float(np.mean(self._active_ring))
        else:
            gt_ratio = 0.0
        obs_ratio = self.hits / self.step if self.step else 0.0
        ire = abs(obs_ratio - gt_ratio)
        n_duty = self.dwell_ms / (self.dwell_ms + self.tune_ms)
        pfa = self.false_alarms / self.nonactive_dwells if self.nonactive_dwells else 0.0
        avg_reward = self.cum_reward / self.step if self.step else 0.0
        return FigureOfMerit(
            p_int=p_int, ttfi=ttfi, pfa=pfa, ire=ire, n_duty=n_duty,
            avg_reward=avg_reward, hits=self.hits, total_pulses=self.total_pulses,
            false_alarms=self.false_alarms, nonactive_dwells=self.nonactive_dwells,
        )


def normalize_offline_row(row: dict) -> dict:
    """
    Take one row of an offline evaluation result (e.g. final_result.json /
    rl_results.csv — already-aggregated P_int, Pd, Pfa, TTFI, avg_reward
    per policy) and fill in whatever this engine can add on top (IRE and
    n_duty are receiver/run-level quantities, not something you can derive
    after the fact from an aggregated row, so they're left null for offline
    rows — only the live demo computes them, since it has step-level
    ground truth).
    """
    return {
        "policy": row.get("policy"),
        "p_int": row.get("P_int"),
        "pd": row.get("Pd"),
        "pfa": row.get("Pfa"),
        "ttfi": row.get("TTFI"),
        "avg_reward": row.get("avg_reward"),
        "ire": None,
        "n_duty": None,
    }


def dsp_confirmation_summary(dsp_rows: list[dict]) -> dict:
    """Aggregate the per-config DSP confirmation records into a scoreboard."""
    total = len(dsp_rows)
    confirmed = [r for r in dsp_rows if r.get("confirmed")]
    n_confirmed = len(confirmed)
    if confirmed:
        avg_align = float(np.mean([r["align"] for r in confirmed]))
        avg_forced_fraction = float(np.mean([r["forced_fraction"] for r in confirmed]))
        total_hits = sum(r["hits"] for r in confirmed)
        total_windows = sum(r["windows"] for r in confirmed)
        hit_rate = total_hits / total_windows if total_windows else 0.0
    else:
        avg_align = avg_forced_fraction = hit_rate = 0.0
    return dict(
        total=total, confirmed=n_confirmed,
        confirmation_rate=(n_confirmed / total if total else 0.0),
        avg_align=avg_align, avg_forced_fraction=avg_forced_fraction,
        dsp_hit_rate=hit_rate,
    )
