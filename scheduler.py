"""
Scheduler interface. Every scheduler implements select_band(context, step) -> int
and update(band, reward, context). Swap implementations without touching
environment.py, metrics.py or the websocket loop in main.py.
"""
import numpy as np


class BaseScheduler:
    name = "base"

    def select_band(self, context: np.ndarray, step: int) -> int:
        raise NotImplementedError

    def update(self, band: int, reward: float, context: np.ndarray):
        pass


class OpenLoopSweep(BaseScheduler):
    """Baseline: fixed round-robin sweep, ignores context entirely."""
    name = "open-loop"

    def __init__(self, n_bands):
        self.n_bands = n_bands
        self.ptr = 0

    def select_band(self, context, step):
        b = self.ptr
        self.ptr = (self.ptr + 1) % self.n_bands
        return b


class LinUCBScheduler(BaseScheduler):
    """
    Contextual bandit (Disjoint LinUCB, Li et al. 2010). One linear model per
    band/arm. This is the 'small trained model' — it updates online every
    step from hit/miss reward, no offline training data required, which
    matters for a hackathon where you have no real emitter dataset yet.
    Swap for an LSTM (see LSTMScheduler stub below) once you have logged
    sequences and time to train offline.
    """
    name = "linucb-bandit"

    def __init__(self, n_bands, ctx_dim, alpha=1.3, forced_sweep_every=5):
        self.n_bands = n_bands
        self.alpha = alpha
        self.forced_sweep_every = forced_sweep_every
        self.ptr = 0
        self.A = [np.eye(ctx_dim) for _ in range(n_bands)]
        self.b = [np.zeros(ctx_dim) for _ in range(n_bands)]

    def select_band(self, context, step):
        # forced coverage floor: guarantees every band gets watched
        # periodically even while confidently exploiting, so a brand-new
        # emitter on a "cold" band is never permanently starved.
        if self.forced_sweep_every and step % self.forced_sweep_every == 0:
            b = self.ptr
            self.ptr = (self.ptr + 1) % self.n_bands
            return b
        best, best_score = 0, -1e18
        for i in range(self.n_bands):
            A_inv = np.linalg.inv(self.A[i])
            theta = A_inv @ self.b[i]
            x = context[i]
            mean = float(theta @ x)
            bonus = self.alpha * float(np.sqrt(max(x @ A_inv @ x, 0.0)))
            score = mean + bonus
            if score > best_score:
                best_score, best = score, i
        return best

    def update(self, band, reward, context):
        x = context[band]
        self.A[band] += np.outer(x, x)
        self.b[band] += reward * x


class LSTMScheduler(BaseScheduler):
    """
    Stretch-goal stub. Same interface, so it drops in without touching the
    websocket loop. Real version: maintain a rolling window of per-band
    (active, watched, hit) history, feed it through a small PyTorch LSTM
    that outputs a score per band each step, argmax to pick, and do an
    online policy-gradient (REINFORCE) or supervised next-activity-
    prediction update on `.update()`. Left unimplemented here on purpose —
    training needs a logged run first; use LinUCBScheduler to generate that
    log (each websocket message already contains full ground truth + action
    + reward, so just dump the stream to a .jsonl file with a client script).
    """
    name = "lstm (stub)"

    def __init__(self, n_bands, ctx_dim):
        self.n_bands = n_bands
        self._fallback = LinUCBScheduler(n_bands, ctx_dim)

    def select_band(self, context, step):
        return self._fallback.select_band(context, step)

    def update(self, band, reward, context):
        self._fallback.update(band, reward, context)


def make_scheduler(mode: str, n_bands: int, ctx_dim: int) -> BaseScheduler:
    if mode == "smart":
        return LinUCBScheduler(n_bands, ctx_dim)
    if mode == "lstm":
        return LSTMScheduler(n_bands, ctx_dim)
    return OpenLoopSweep(n_bands)
