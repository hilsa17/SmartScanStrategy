"""
Run:  uvicorn main:app --reload
Open: http://localhost:8000
"""
import asyncio
import json
import random
from collections import deque

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles

from environment import RFEnvironment, N_BANDS
from scheduler import make_scheduler

CTX_DIM = 5
TUNE_MS = 28          # T_tune, receiver switch/settle overhead
P_FA_BASE = 0.03       # per-dwell false-trigger probability on an inactive band (noise floor)
SWITCH_COST = 0.02     # reward penalty for re-tuning to a different band
FALSE_ALARM_COST = 0.05

app = FastAPI()


class Session:
    """One simulation + scheduler instance per connected client."""
    def __init__(self):
        self.env = RFEnvironment(N_BANDS)
        self.mode = "open"
        self.scheduler = make_scheduler(self.mode, N_BANDS, CTX_DIM)
        self.dwell_ms = 140
        self.playing = False
        self.prev_band = None

        self.total_pulses = 0
        self.hits = 0
        self.false_alarms = 0
        self.nonactive_dwells = 0
        self.cum_reward = 0.0
        self.ttfi_samples = deque(maxlen=60)
        self.ring_active = deque(maxlen=90)   # for IRE ground-truth ratio over the visible window

    def set_mode(self, mode):
        self.mode = mode
        self.scheduler = make_scheduler(mode, N_BANDS, CTX_DIM)
        self.prev_band = None

    def reset(self):
        self.__init__()

    def step_once(self):
        active = self.env.tick()
        n_active = int(active.sum())
        self.total_pulses += n_active
        self.ring_active.append(active.copy())

        ctx = self.env.context_matrix()
        band = self.scheduler.select_band(ctx, self.env.step)
        hit = bool(active[band])

        false_alarm = False
        if not hit:
            self.nonactive_dwells += 1
            if random.random() < P_FA_BASE:
                false_alarm = True
                self.false_alarms += 1

        reward = 0.0
        if hit:
            self.hits += 1
            reward += 1.0
            self.env.register_hit(band)
            for em in self.env.emitters:
                if em.on and em.band_now() == band and not em.counted_this_burst:
                    self.ttfi_samples.append(self.env.step - em.on_since)
                    em.counted_this_burst = True
        if false_alarm:
            reward -= FALSE_ALARM_COST
        if self.prev_band is not None and self.prev_band != band:
            reward -= SWITCH_COST
        self.cum_reward += reward
        self.scheduler.update(band, reward, ctx)
        self.prev_band = band

        return dict(
            step=self.env.step,
            mode=self.mode,
            band_watch=band,
            hit=hit,
            false_alarm=false_alarm,
            active=[bool(a) for a in active],
            emitters=self.env.emitter_snapshot(),
            metrics=self.metrics(),
        )

    def metrics(self):
        p_int = self.hits / self.total_pulses if self.total_pulses else 0.0
        ttfi = float(np.mean(self.ttfi_samples)) if self.ttfi_samples else None
        if self.ring_active:
            arr = np.array(self.ring_active)
            gt_ratio = float(arr.mean())
        else:
            gt_ratio = 0.0
        obs_ratio = self.hits / self.env.step if self.env.step else 0.0
        ire = abs(obs_ratio - gt_ratio)
        n_duty = self.dwell_ms / (self.dwell_ms + TUNE_MS)
        pfa = self.false_alarms / self.nonactive_dwells if self.nonactive_dwells else 0.0
        avg_reward = self.cum_reward / self.env.step if self.env.step else 0.0
        return dict(
            p_int=p_int, hits=self.hits, total_pulses=self.total_pulses,
            ttfi=ttfi, ire=ire, n_duty=n_duty,
            pfa=pfa, false_alarms=self.false_alarms,
            avg_reward=avg_reward,
        )


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    sess = Session()
    try:
        while True:
            # drain any pending control messages without blocking the sim loop
            try:
                while True:
                    raw = await asyncio.wait_for(ws.receive_text(), timeout=0.001)
                    msg = json.loads(raw)
                    if msg["type"] == "set_mode":
                        sess.set_mode(msg["mode"])
                    elif msg["type"] == "set_speed":
                        sess.dwell_ms = int(msg["ms"])
                    elif msg["type"] == "play":
                        sess.playing = True
                    elif msg["type"] == "pause":
                        sess.playing = False
                    elif msg["type"] == "reset":
                        sess.reset()
            except asyncio.TimeoutError:
                pass

            if sess.playing:
                payload = sess.step_once()
                await ws.send_text(json.dumps(payload))
                await asyncio.sleep(sess.dwell_ms / 1000)
            else:
                await asyncio.sleep(0.05)
    except WebSocketDisconnect:
        pass


app.mount("/", StaticFiles(directory="static", html=True), name="static")
