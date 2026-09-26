
import os
import csv
import math
import argparse
from collections import deque

import numpy as np
import matplotlib.pyplot as plt

from radar_data import add_data_args, build_datasets, Receiver, EpisodeLogger, learn_stickiness

# Every agent below has the same two functions:
#    select()               -> tells which band to look at
#    update(band, reward)   -> tells the agent what happened (reward 1 = signal noticed, 0 = nothing)


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
# UCB = "average reward so far  +  a bonus for bands we have not tried much".
# The bonus makes the agent explore; the average makes it exploit.
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


# ============================================================
# SECTION 3: Exp3 (for the case where the enemy may behave against us)
# It keeps a weight for each band, picks bands randomly (good weight = more likely),
# and raises the weight of bands that gave a reward.
# ============================================================

class Exp3:
    def __init__(self, n_bands, gamma=0.1, seed=0, **_):
        self.n, self.gamma = n_bands, gamma
        self.log_w = np.zeros(n_bands)      # log of the weights (avoids very big numbers)
        self.rng = np.random.default_rng(seed)
        self.p = np.full(n_bands, 1.0 / n_bands)

    def select(self):
        w = np.exp(self.log_w - self.log_w.max())
        self.p = (1 - self.gamma) * w / w.sum() + self.gamma / self.n   # gamma part = always explore a bit
        return int(self.rng.choice(self.n, p=self.p))

    def update(self, band, reward):
        self.log_w[band] += self.gamma * (reward / self.p[band]) / self.n


# ============================================================
# SECTION 4: belief methods (Myopic and Whittle Index)
#
# Each band is either EMPTY or BUSY, and it can change even when we are not looking
# (this is the "restless" part). For every band we keep a BELIEF = our guess of the
# chance that the band is busy right now.
#   - When we look at a band, the belief is corrected using what we saw.
#   - When we do NOT look at a band, the belief slowly drifts towards its long-term average.
# p01 and p11 (how sticky busy/empty is) are learned from the TRAIN data.
# ============================================================

def whittle_table(p01, p11, pd, pfa, beta=0.95, grid=201, n_sub=51, iters=80):
    """Builds the Whittle Index score for every possible belief value (0 to 1).

    Simple explanation: imagine ONE band alone. Every slot we can either
       - LOOK at it (we may get reward), or
       - SKIP it and get a small fixed payment m instead.
    The Whittle index of a belief = the smallest payment m at which skipping becomes the
    better choice. A higher index means "this band is really worth looking at".
    We find it by trying many values of m and solving the small problem each time
    (value iteration on a grid of belief values)."""
    w = np.linspace(0.0, 1.0, grid)                                   # belief values 0 ... 1
    drift = lambda x: x * p11 + (1 - x) * p01                         # belief after one unwatched slot
    reward_now = w * pd + (1 - w) * pfa                               # chance we "notice a signal"
    belief_if_noticed = w * pd / np.maximum(w * pd + (1 - w) * pfa, 1e-12)
    belief_if_not = w * (1 - pd) / np.maximum(w * (1 - pd) + (1 - w) * (1 - pfa), 1e-12)
    next_if_noticed, next_if_not, next_if_skip = drift(belief_if_noticed), drift(belief_if_not), drift(w)
    payments = np.linspace(0.0, 1.0, n_sub)
    skip_is_better = np.zeros((n_sub, grid), dtype=bool)
    V = np.zeros(grid)                                                # value of each belief
    for k, m in enumerate(payments):
        for _ in range(iters):
            look_value = reward_now + beta * (reward_now * np.interp(next_if_noticed, w, V)
                                              + (1 - reward_now) * np.interp(next_if_not, w, V))
            skip_value = m + beta * np.interp(next_if_skip, w, V)
            V = np.maximum(look_value, skip_value)
        skip_is_better[k] = skip_value >= look_value
    # index = smallest payment where skipping wins
    return np.where(skip_is_better.any(0), payments[skip_is_better.argmax(0)], 1.0)


class BeliefAgent:
    """Keeps a belief for every band. Children only decide how to turn beliefs into a choice."""

    def __init__(self, n_bands, p01, p11, pd, pfa, seed=0, **_):
        self.n, self.p01, self.p11, self.pd, self.pfa = n_bands, p01, p11, pd, pfa
        self.belief = np.full(n_bands, p01 / (1 - p11 + p01))         # start at the long-term average
        self.rng = np.random.default_rng(seed)

    def score(self):
        raise NotImplementedError

    def select(self):
        tiny_noise = 1e-9 * self.rng.random(self.n)                   # breaks ties randomly
        return int(np.argmax(self.score() + tiny_noise))

    def update(self, band, reward):
        noticed = reward > 0.5
        b = self.belief[band]
        if noticed:      # Bayes rule: how likely is "busy" after seeing a signal?
            new_b = b * self.pd / max(b * self.pd + (1 - b) * self.pfa, 1e-12)
        else:            # ... after NOT seeing a signal?
            new_b = b * (1 - self.pd) / max(b * (1 - self.pd) + (1 - b) * (1 - self.pfa), 1e-12)
        self.belief[band] = new_b
        self.belief = self.belief * self.p11 + (1 - self.belief) * self.p01   # all bands drift one step


class MyopicBelief(BeliefAgent):
    """Simply look at the band with the highest belief."""
    def score(self):
        return self.belief


class WhittleAgent(BeliefAgent):
    """Look at the band with the highest Whittle index (looked up from the table)."""
    def __init__(self, n_bands, p01, p11, pd, pfa, table=None, **kw):
        super().__init__(n_bands, p01, p11, pd, pfa, **kw)
        self.table = table                  # same table for every band

    def score(self):
        last = len(self.table) - 1
        pos = np.clip(self.belief, 0, 1) * last
        lo = np.minimum(pos.astype(int), last - 1)
        frac = pos - lo
        return self.table[lo] * (1 - frac) + self.table[lo + 1] * frac   # straight-line lookup in the table


# ============================================================
# SECTION 5: run one algorithm on the test windows
# ============================================================

def run_agent(make_agent, name, source, a, n_episodes):
    logger = EpisodeLogger(name)
    for ep in range(n_episodes):
        source.start_episode(seed=ep)                     # same window for every algorithm
        receiver = Receiver(a.rx_pd, a.rx_pfa, seed=ep)
        agent = make_agent(ep)
        for t in range(a.n_slots):
            busy_now = source.next_slot()                 # truth: which bands are busy (agent cannot see this)
            band = agent.select()
            was_busy = bool(busy_now[band])
            noticed = receiver.observe(was_busy)          # what the receiver reports
            reward = 1.0 if noticed else 0.0
            agent.update(band, reward)
            logger.log_slot(t, was_busy, noticed, reward)
        logger.end_episode()
    return logger


# ============================================================
# SECTION 6: run everything, print table, save CSV + chart
# ============================================================

def main():
    ap = argparse.ArgumentParser(description="Bandit algorithms on the radar dataset")
    add_data_args(ap)
    ap.add_argument("--n-episodes", "--n-eval-episodes", dest="n_episodes", type=int, default=200,
                    help="test episodes per algorithm")
    ap.add_argument("--dataset", default=None,
                    help="fake emitters only: fixed_periodic | pseudo_random | markov_restless")
    a = ap.parse_args()

    os.makedirs("results", exist_ok=True)
    datasets = build_datasets(a)
    if a.dataset is not None:
        if a.dataset not in datasets:
            raise ValueError(f"Unknown dataset '{a.dataset}'. Available: {sorted(datasets)}")
        datasets = {a.dataset: datasets[a.dataset]}
    B = a.n_bands
    all_loggers, rows = [], []

    for name, data in datasets.items():
        print(f"\n[{name}] learning how sticky the bands are from the train data...")
        p01, p11 = learn_stickiness(data["train"].maps)
        print(f"  p01 = {p01:.4f} (empty -> busy)   p11 = {p11:.4f} (busy -> busy)")
        table = whittle_table(p01, p11, a.rx_pd, a.rx_pfa)

        algorithms = {
            "sawtooth":     lambda ep: LinearSawtooth(B),
            "stepped_lo":   lambda ep: SteppedLO(B, step=3),
            "uniform_rand": lambda ep: UniformRandom(B, seed=ep),
            "UCB1":         lambda ep: UCB1(B),
            "D-UCB":        lambda ep: DiscountedUCB(B, gamma=0.98),
            "SW-UCB":       lambda ep: SlidingWindowUCB(B, window=50),
            "Exp3":         lambda ep: Exp3(B, seed=ep),
            "Myopic":       lambda ep: MyopicBelief(B, p01, p11, a.rx_pd, a.rx_pfa, seed=ep),
            "Whittle":      lambda ep: WhittleAgent(B, p01, p11, a.rx_pd, a.rx_pfa, table=table, seed=ep),
        }
        for algo_name, make in algorithms.items():
            print(f"  running {algo_name} ...")
            all_loggers.append(run_agent(make, f"{algo_name}|{name}", data["test"], a, a.n_episodes))

    print(f"\n{'algorithm | data':30s} {'P_int':>8s} {'Pd':>8s} {'Pfa':>8s} {'TTFI':>8s} {'avg_rew':>9s}")
    print("-" * 76)
    for lg in all_loggers:
        s = lg.summary()
        rows.append({"algo": lg.name, **s})
        print(f"{lg.name:30s} {s['P_int']:8.3f} {s['Pd']:8.3f} {s['Pfa']:8.3f} {s['TTFI']:8.2f} {s['avg_reward']:9.3f}")

    csv_path = os.path.join("results", "bandits_results.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["algo", "P_int", "Pd", "Pfa", "TTFI", "avg_reward"])
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved table -> {csv_path}")

    # chart for slides: first dataset only
    first = list(datasets)[0]
    part = [r for r in rows if r["algo"].endswith(f"|{first}")]
    names = [r["algo"].split("|")[0] for r in part]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].bar(names, [r["P_int"] for r in part])
    axes[0].set_title(f"Intercept rate P_int (higher is better) [{first}]")
    axes[1].bar(names, [r["TTFI"] for r in part])
    axes[1].set_title(f"Slots to first intercept, TTFI (lower is better) [{first}]")
    for ax in axes:
        ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    chart_path = os.path.join("results", "bandits_comparison.png")
    plt.savefig(chart_path, dpi=150)
    print(f"Saved chart -> {chart_path}")


if __name__ == "__main__":
    main()