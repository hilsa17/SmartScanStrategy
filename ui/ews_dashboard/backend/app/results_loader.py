"""
Loads the artifacts an offline training/eval run produced (final_result.json,
the stable-baselines3-style Monitor CSV, and the DSP confirmation log) and
shapes them for the frontend. This is the "system integration" half of the
task: it's the seam between whatever trains the model offline and the
real-time GUI that presents the result.
"""
import csv
import json
from pathlib import Path

import numpy as np

from .metrics_engine import normalize_offline_row, dsp_confirmation_summary

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RESULT_JSON = DATA_DIR / "final_result.json"
MONITOR_CSV = DATA_DIR / "train_monitor.csv"

_SMOOTH_WINDOW = 15


def load_final_result() -> dict:
    with open(RESULT_JSON) as f:
        raw = json.load(f)
    rows = [normalize_offline_row(r) for r in raw["rows"]]
    best = raw.get("best_policy")
    dsp_rows = raw.get("dsp") or []
    return dict(
        dataset=raw.get("dataset"),
        algo=raw.get("algo"),
        best_policy=best,
        rows=rows,
        dsp_summary=dsp_confirmation_summary(dsp_rows),
        dsp_rows=dsp_rows,
    )


def load_training_curve() -> dict:
    """
    Parses a stable-baselines3 Monitor CSV: first line is a '#'-prefixed
    JSON header (t_start, env_id), second line is the real column header
    'r,l,t', then one row per completed episode.
    """
    if not MONITOR_CSV.exists():
        return dict(episodes=[], reward=[], smoothed=[])
    with open(MONITOR_CSV) as f:
        lines = f.readlines()
    # drop the leading '#{...}' metadata line if present
    start = 1 if lines and lines[0].startswith("#") else 0
    reader = csv.DictReader(lines[start:])
    rewards = []
    for row in reader:
        try:
            rewards.append(float(row["r"]))
        except (KeyError, ValueError):
            continue
    rewards = np.array(rewards)
    if len(rewards) == 0:
        return dict(episodes=[], reward=[], smoothed=[])
    kernel = np.ones(_SMOOTH_WINDOW) / _SMOOTH_WINDOW
    smoothed = np.convolve(rewards, kernel, mode="valid")
    # pad the front so smoothed[] lines up 1:1 with episodes[] for plotting
    pad = np.full(_SMOOTH_WINDOW - 1, smoothed[0])
    smoothed = np.concatenate([pad, smoothed])
    return dict(
        episodes=list(range(len(rewards))),
        reward=[round(float(x), 3) for x in rewards],
        smoothed=[round(float(x), 3) for x in smoothed],
    )
