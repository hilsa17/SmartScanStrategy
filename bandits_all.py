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

