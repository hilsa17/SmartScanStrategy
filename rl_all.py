import os
import csv
import argparse

import numpy as np
try:
    import pandas as pd
except Exception:  # pragma: no cover - optional dependency on some systems
    pd = None
try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - optional dependency on some systems
    plt = None
import torch
import torch.nn as nn
import gymnasium as gym
from gymnasium import spaces

from radar_data import add_data_args, build_datasets, Receiver, EpisodeLogger

torch.set_num_threads(1)
# ============================================================
# SECTION 1: LSTM predictor (guesses which bands will be busy next slot)
# ============================================================
class BusyLSTM(nn.Module):
    def __init__(self, n_bands, hidden=64):
        super().__init__()
        self.lstm = nn.LSTM(input_size=2 * n_bands, hidden_size=hidden, batch_first=True)
        self.head = nn.Linear(hidden, n_bands)

    def forward(self, x):                      # x shape: [batch, past slots, 2*n_bands]
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])        # one score per band


def encode_one_slot(band, noticed, n_bands):
    """Small vector describing what happened in one slot."""
    v = np.zeros(2 * n_bands, dtype=np.float32)
    v[band] = float(noticed)                   # first half: did we noyice a sinal in this band
    v[n_bands + band] = 1.0                    # second half: which band we listened to
    return v


class BusyPredictor:
    def __init__(self, n_bands, window=32):
        self.B, self.W = n_bands, window       # window = how many past slots the LSTM looks at
        self.model = BusyLSTM(n_bands)
        self.model.eval()

    def _make_batch(self, maps, n, rng):
        B, W = self.B, self.W
        X = np.zeros((n, W, 2 * B), dtype=np.float32)
        Y = np.zeros((n, B), dtype=np.float32)
        idx = np.arange(W)
        for i in range(n):
            m = maps[int(rng.integers(len(maps)))]
            start = int(rng.integers(0, len(m) - W))
            looks = rng.integers(0, B, size=W)                 # pretend we listened to random bands
            X[i, idx, looks] = m[start:start + W][idx, looks]  # we only know the band we listened to
            X[i, idx, B + looks] = 1.0
            Y[i] = m[start + W]                                # answer = which bands were busy next
        return torch.from_numpy(X), torch.from_numpy(Y)
    def pretrain(self, maps, steps=2000, batch=128, seed=0):
        rng = np.random.default_rng(seed)
        opt = torch.optim.Adam(self.model.parameters(), lr=2e-3)
        loss_fn = nn.BCEWithLogitsLoss()
        self.model.train()
        for k in range(steps):
            X, Y = self._make_batch(maps, batch, rng)
            opt.zero_grad()
            loss = loss_fn(self.model(X), Y)
            loss.backward()
            opt.step()
            if (k + 1) % 500 == 0:
                print(f"[predictor] step {k+1}/{steps}  loss = {loss.item():.4f}")
        self.model.eval()

    @torch.no_grad()
    def predict(self, history):                # history shape: [window, 2*n_bands]
        x = torch.from_numpy(history).unsqueeze(0)
        return torch.sigmoid(self.model(x)).squeeze(0).numpy().astype(np.float32)

    @torch.no_grad()
    def accuracy(self, maps, n=2000, seed=123):
        """% correct predictions: is the band the predictor likes best really busy next slot?
        Also returns the score of a random guess for comparison."""
        X, Y = self._make_batch(maps, n, np.random.default_rng(seed))
        best_band = torch.sigmoid(self.model(X)).argmax(1)
        return float(Y[torch.arange(n), best_band].mean()), float(Y.mean())