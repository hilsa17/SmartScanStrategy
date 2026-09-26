
import os
import csv
import json
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
#
# Input for each past slot: [what we noticed in the band we listened to | which band we listened to]
# Output: for every band, the chance that it will be busy in the NEXT slot.
# It is trained before the RL training, on the train files, using random listening choices.
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
    v[band] = float(noticed)                   # first half: did we notice a signal in this band
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


# ============================================================
# SECTION 2: the game (Gymnasium environment)
# ============================================================

class SpectrumEnv(gym.Env):
    """One step = one time slot.
    Action      : which band to listen to (0 ... n_bands-1)
    Observation : for each band, how recently we NOTICED a signal there,
                  for each band, how recently we LISTENED there,
                  (+ the predictor's guess for each band, if used).
                  'How recently' = exp(-slots_ago / 10): 1.0 means just now, near 0 means long ago.
    Reward      : +1 if we noticed a signal, -0.05 if not."""

    metadata = {"render_modes": []}

    def __init__(self, source, rx_pd=0.95, rx_pfa=0.02, predictor=None, seed_offset=0):
        super().__init__()
        self.src, self.predictor = source, predictor
        self.B, self.n_slots = source.n_bands, source.n_slots
        self.rx_pd, self.rx_pfa, self.seed_offset = rx_pd, rx_pfa, seed_offset
        self._episode = 0
        self.action_space = spaces.Discrete(self.B)
        size = 2 * self.B + (self.B if predictor is not None else 0)
        self.observation_space = spaces.Box(0.0, 1.0, shape=(size,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        ep_seed = seed if seed is not None else self.seed_offset + self._episode
        self._episode += 1
        self.src.start_episode(ep_seed)
        self.receiver = Receiver(self.rx_pd, self.rx_pfa, seed=ep_seed)
        self.t = 0
        self.slots_since_noticed = np.full(self.B, 1e3, dtype=np.float32)
        self.slots_since_listened = np.full(self.B, 1e3, dtype=np.float32)
        if self.predictor is not None:
            self.history = np.zeros((self.predictor.W, 2 * self.B), dtype=np.float32)
        return self._observation(), {}

    def _observation(self):
        obs = [np.exp(-self.slots_since_noticed / 10.0), np.exp(-self.slots_since_listened / 10.0)]
        if self.predictor is not None:
            obs.append(self.predictor.predict(self.history))
        return np.concatenate(obs).astype(np.float32)

    def step(self, action):
        band = int(action)
        busy_now = self.src.next_slot()
        was_busy = bool(busy_now[band])
        noticed = self.receiver.observe(was_busy)

        self.slots_since_noticed += 1
        self.slots_since_listened += 1
        self.slots_since_listened[band] = 0
        if noticed:
            self.slots_since_noticed[band] = 0
        if self.predictor is not None:                       # add this slot to the predictor's history
            self.history = np.roll(self.history, -1, axis=0)
            self.history[-1] = encode_one_slot(band, noticed, self.B)

        reward = 1.0 if noticed else -0.05
        self.t += 1
        done_by_time = self.t >= self.n_slots
        return self._observation(), reward, False, done_by_time, {"was_busy": was_busy, "noticed": noticed}


# ============================================================
# SECTION 3: train and test
# ============================================================

def train_agent(env, algo, timesteps, monitor_path):
    from stable_baselines3 import DQN, PPO
    from stable_baselines3.common.monitor import Monitor

    env = Monitor(env, filename=monitor_path)          # Monitor saves the reward of every training episode
    if algo == "dqn":
        model = DQN("MlpPolicy", env, learning_rate=1e-3, buffer_size=50_000, learning_starts=1000,
                    batch_size=64, gamma=0.95, train_freq=4, target_update_interval=500,
                    exploration_fraction=0.3, exploration_final_eps=0.05, verbose=1, device="cpu")
    else:
        model = PPO("MlpPolicy", env, learning_rate=3e-4, n_steps=1024, batch_size=64, gamma=0.95,
                    verbose=1, device="cpu")
    model.learn(total_timesteps=timesteps)
    return model


def test_policy(policy, name, env, n_slots, n_episodes):
    """policy(observation, t) -> band.  Seeds 0,1,2,... give the same test windows as bandits_all.py."""
    logger = EpisodeLogger(name)
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=ep)
        for t in range(n_slots):
            obs, reward, terminated, truncated, info = env.step(policy(obs, t))
            logger.log_slot(t, info["was_busy"], info["noticed"], reward)
            if terminated or truncated:
                break
        logger.end_episode()
    return logger


def infer_best_dataset_from_bandits(csv_path):
    """Read results/bandits_results.csv and return the best-performing dataset name."""
    if not os.path.exists(csv_path):
        return None
    best_name = None
    best_score = float("-inf")
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = (row.get("algo") or "").strip()
            if not name or "|" not in name:
                continue
            dataset = name.split("|", 1)[1]
            try:
                score = float(row.get("avg_reward", row.get("avg_rew", "-inf")))
            except ValueError:
                continue
            if score > best_score:
                best_score = score
                best_name = dataset
    return best_name


# ============================================================
# SECTION 4: run everything, print results, save CSV + charts
# ============================================================

def run_rl_pipeline(a=None, confirmation=None):
    if a is None:
        ap = argparse.ArgumentParser(description="RL scheduler on the radar dataset")
        add_data_args(ap)
        ap.add_argument("--algo", choices=["dqn", "ppo"], default="dqn")
        ap.add_argument("--timesteps", type=int, default=100_000,
                        help="training steps (50k = quick test, 100k-300k = real results)")
        ap.add_argument("--n-eval-episodes", "--n-episodes", dest="n_eval_episodes", type=int, default=50)
        ap.add_argument("--no-predictor", action="store_true", help="do not use the LSTM predictor")
        ap.add_argument("--predictor-steps", type=int, default=2000, help="training steps for the LSTM")
        ap.add_argument("--dataset", default=None,
                        help="fake emitters only: fixed_periodic | pseudo_random | markov_restless")
        a = ap.parse_args()

    if confirmation is not None:
        if confirmation.get("confirmed"):
            print(f"[DSP-confirmation] confirmatory score = {confirmation.get('score', 'n/a'):.3f}")
            if confirmation.get("dataset") is not None:
                a.dataset = confirmation["dataset"]
                print(f"[DSP-confirmation] overriding dataset to '{a.dataset}'")
        else:
            print("[DSP-confirmation] no confirmatory data received; keeping existing RL settings")

    if a.dataset is None:
        bandit_csv = os.path.join("results", "bandits_results.csv")
        hinted_dataset = infer_best_dataset_from_bandits(bandit_csv)
        if hinted_dataset is not None:
            a.dataset = hinted_dataset
            print(f"[bandit-link] using dataset '{a.dataset}' from {bandit_csv} as the RL input")

    os.makedirs("results", exist_ok=True)
    datasets = build_datasets(a)
    if a.dataset is not None:
        if a.dataset not in datasets:
            raise ValueError(f"Unknown dataset '{a.dataset}'. Available: {sorted(datasets)}")
        datasets = {a.dataset: datasets[a.dataset]}
    name = a.dataset or list(datasets)[0]
    train_src, test_src = datasets[name]["train"], datasets[name]["test"]
    B = a.n_bands
    print(f"\n=== data: {name} | agent: {a.algo.upper()} | predictor: {not a.no_predictor} ===")

    predictor = None
    if not a.no_predictor:
        predictor = BusyPredictor(B, window=32)
        predictor.pretrain(train_src.maps, steps=a.predictor_steps)
        correct, random_guess = predictor.accuracy(test_src.maps)
        print(f"[predictor] correct predictions on test data = {correct:.3f}  (random guess = {random_guess:.3f})")

    train_env = SpectrumEnv(train_src, a.rx_pd, a.rx_pfa, predictor, seed_offset=1_000_000)
    monitor_path = os.path.join("results", "train_monitor")
    print(f"\nTraining {a.algo.upper()} for {a.timesteps} steps...\n")
    model = train_agent(train_env, a.algo, a.timesteps, monitor_path)
    model.save("dqn_spectrum" if a.algo == "dqn" else "ppo_spectrum")

    print("\nTesting on the TEST files...")
    test_env = SpectrumEnv(test_src, a.rx_pd, a.rx_pfa, predictor)
    rl_policy = lambda obs, t: int(model.predict(obs, deterministic=True)[0])
    sawtooth = lambda obs, t: t % B
    rng = np.random.default_rng(0)
    random_policy = lambda obs, t: int(rng.integers(B))
    loggers = [test_policy(rl_policy, a.algo.upper(), test_env, a.n_slots, a.n_eval_episodes),
               test_policy(sawtooth, "sawtooth", test_env, a.n_slots, a.n_eval_episodes),
               test_policy(random_policy, "uniform_random", test_env, a.n_slots, a.n_eval_episodes)]

    print(f"\n{'policy':16s} {'P_int':>8s} {'Pd':>8s} {'Pfa':>8s} {'TTFI':>8s} {'avg_rew':>9s}")
    print("-" * 62)
    rows = []
    for lg in loggers:
        s = lg.summary()
        rows.append({"policy": lg.name, **s})
        print(f"{lg.name:16s} {s['P_int']:8.3f} {s['Pd']:8.3f} {s['Pfa']:8.3f} {s['TTFI']:8.2f} {s['avg_reward']:9.3f}")

    csv_path = os.path.join("results", "rl_results.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["policy", "P_int", "Pd", "Pfa", "TTFI", "avg_reward"])
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved results -> {csv_path}")

    final_summary = {
        "dataset": name,
        "algo": a.algo,
        "best_policy": max(rows, key=lambda r: r["avg_reward"] if np.isfinite(r["avg_reward"]) else float('-inf'))["policy"],
        "rows": rows,
        "dsp_confirmation": confirmation,
    }

    final_json = os.path.join("results", "final_result.json")
    with open(final_json, "w") as f:
        json.dump(final_summary, f, indent=2)
    print(f"Saved final output -> {final_json}")

    if plt is None:
        print("[plot] skipping chart generation because matplotlib is unavailable in this environment")
    else:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        monitor_csv = monitor_path + ".monitor.csv"
        if pd is not None:
            mon = pd.read_csv(monitor_csv, skiprows=1)
            rewards = mon["r"].tolist()
        else:
            rewards = []
            with open(monitor_csv, newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if "r" in row and row["r"] not in (None, ""):
                        rewards.append(float(row["r"]))
        if rewards:
            smooth = np.convolve(rewards, np.ones(max(1, len(rewards) // 20)) / max(1, len(rewards) // 20), mode="same")
            axes[0].plot(smooth)
        axes[0].set_xlabel("training episode")
        axes[0].set_ylabel("episode reward (smoothed)")
        axes[0].set_title(f"{a.algo.upper()} training curve [{name}]")
        axes[1].bar([r["policy"] for r in rows], [r["P_int"] for r in rows])
        axes[1].set_title("Intercept rate P_int on test data (higher is better)")
        plt.tight_layout()
        chart_path = os.path.join("results", "rl_training_curve.png")
        plt.savefig(chart_path, dpi=150)
        print(f"Saved chart -> {chart_path}")

    return final_summary


def main():
    ap = argparse.ArgumentParser(description="RL scheduler on the radar dataset")
    add_data_args(ap)
    ap.add_argument("--algo", choices=["dqn", "ppo"], default="dqn")
    ap.add_argument("--timesteps", type=int, default=100_000,
                    help="training steps (50k = quick test, 100k-300k = real results)")
    ap.add_argument("--n-eval-episodes", "--n-episodes", dest="n_eval_episodes", type=int, default=50)
    ap.add_argument("--no-predictor", action="store_true", help="do not use the LSTM predictor")
    ap.add_argument("--predictor-steps", type=int, default=2000, help="training steps for the LSTM")
    ap.add_argument("--dataset", default=None,
                    help="fake emitters only: fixed_periodic | pseudo_random | markov_restless")
    a = ap.parse_args()
    run_rl_pipeline(a)


if __name__ == "__main__":
    main()