import os
import csv
import math
import argparse
from collections import deque

import numpy as np
import matplotlib.pyplot as plt

from radar_data import add_data_args, build_datasets, Receiver, EpisodeLogger, learn_stickiness

# ============================================================
# SECTION 1: simple sweeps (the "dumb" methods we must beat)
# ============================================================

class LinearSawtooth:
    """Look at band 0, 1, 2, ... then start again."""
    def __init__(self, n_bands, **_):
        self.n, self.t = n_bands, 0

    def select(self):
        b = self.t % self.n
        self.t += 1
        return b

    def update(self, band, reward):
        pass


class SteppedLO:
    """Like the sawtooth, but jumps `step` bands each time (0, 3, 6, ...)."""
    def __init__(self, n_bands, step=3, **_):
        self.n, self.step, self.t = n_bands, step, 0

    def select(self):
        b = (self.t * self.step) % self.n
        self.t += 1
        return b

    def update(self, band, reward):
        pass

class UniformRandom:
    """Look at a random band every time."""
    def __init__(self, n_bands, seed=0, **_):
        self.n, self.rng = n_bands, np.random.default_rng(seed)

    def select(self):
        return int(self.rng.integers(self.n))

    def update(self, band, reward):
        pass


# ============================================================
# SECTION 2: UCB family
# ============================================================

class UCB1:
    def __init__(self, n_bands, c=2.0, **_):
        self.n, self.c = n_bands, c
        self.counts = np.zeros(n_bands)     # how many times each band was looked at
        self.values = np.zeros(n_bands)     # average reward of each band
        self.t = 0

    def select(self):
        self.t += 1
        untried = np.where(self.counts == 0)[0]
        if len(untried):                    # first, try every band once
            return int(untried[0])
        bonus = np.sqrt(self.c * math.log(self.t) / self.counts)
        return int(np.argmax(self.values + bonus))

    def update(self, band, reward):
        self.counts[band] += 1
        self.values[band] += (reward - self.values[band]) / self.counts[band]


class DiscountedUCB:
    """Same as UCB1 but old results slowly fade (multiplied by gamma every slot),
    so the agent follows a signal that moves around."""
    def __init__(self, n_bands, c=2.0, gamma=0.98, **_):
        self.n, self.c, self.gamma = n_bands, c, gamma
        self.counts = np.zeros(n_bands)     # faded count of looks
        self.sums = np.zeros(n_bands)       # faded sum of rewards

    def select(self):
        untried = np.where(self.counts < 1e-9)[0]
        if len(untried):
            return int(untried[0])
        total = max(self.counts.sum(), 1.0 + 1e-9)
        average = self.sums / self.counts
        return int(np.argmax(average + np.sqrt(self.c * math.log(total) / self.counts)))

    def update(self, band, reward):
        self.counts *= self.gamma
        self.sums *= self.gamma
        self.counts[band] += 1
        self.sums[band] += reward


class SlidingWindowUCB:
    """Same as UCB1 but only the LAST `window` looks are remembered. Everything older is forgotten."""
    def __init__(self, n_bands, c=2.0, window=50, **_):
        self.n, self.c, self.window = n_bands, c, window
        self.recent = deque()               # last `window` (band, reward) pairs
        self.counts = np.zeros(n_bands)
        self.sums = np.zeros(n_bands)
        self.t = 0

    def select(self):
        self.t += 1
        untried = np.where(self.counts == 0)[0]
        if len(untried):                    # a band that was forgotten gets looked at again
            return int(untried[0])
        bonus = np.sqrt(self.c * math.log(min(self.t, self.window)) / self.counts)
        return int(np.argmax(self.sums / self.counts + bonus))

    def update(self, band, reward):
        self.recent.append((band, reward))
        self.counts[band] += 1
        self.sums[band] += reward
        if len(self.recent) > self.window:  # forget the oldest look
            old_band, old_reward = self.recent.popleft()
            self.counts[old_band] -= 1
            self.sums[old_band] -= old_reward