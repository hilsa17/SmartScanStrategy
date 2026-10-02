"""
Interactive live demo: open-loop sweep vs. LinUCB contextual-bandit
scheduler against REAL recorded TSRD radar activity (not a synthetic
simulator), streamed over a WebSocket. Episodes are sampled from the
held-out TEST captures — the same split final_result.json's numbers were
measured on — so what you see here and the headline numbers on the Final
Results tab are honest about using the same evaluation data.
"""
import asyncio
import json
import random

from fastapi import WebSocket, WebSocketDisconnect

from . import data_source
from .real_environment import RealRFEnvironment
from .scheduler import make_scheduler
from .metrics_engine import MetricsEngine

CTX_DIM = 5
P_FA_BASE = 0.03
SWITCH_COST = 0.02
FALSE_ALARM_COST = 0.05


class LiveSession:
    def __init__(self):
        self.n_bands = data_source.N_BANDS
        self.env = RealRFEnvironment(data_source.new_source(), seed0=random.randint(0, 100_000))
        self.mode = "open"
        self.scheduler = make_scheduler(self.mode, self.n_bands, CTX_DIM)
        self.dwell_ms = 140
        self.playing = False
        self.prev_band = None
        self.engine = MetricsEngine(dwell_ms=self.dwell_ms)
        # TTFI has no labeled "emitter power-on" event in real busy-map data
        # (unlike the old synthetic emitters), so it's derived from band
        # activity transitions: a band going idle->active starts the clock,
        # the first hit on that band while it's still active stops it.
        self.band_active_since = [None] * self.n_bands
        self.band_counted = [False] * self.n_bands

    def set_mode(self, mode):
        self.mode = mode
        self.scheduler = make_scheduler(mode, self.n_bands, CTX_DIM)
        self.prev_band = None

    def set_speed(self, ms):
        self.dwell_ms = ms
        self.engine.dwell_ms = ms

    def reset(self):
        self.__init__()

    def step_once(self):
        active = self.env.tick()

        for b in range(self.n_bands):
            if active[b]:
                if self.band_active_since[b] is None:
                    self.band_active_since[b] = self.env.step
                    self.band_counted[b] = False
            else:
                self.band_active_since[b] = None

        ctx = self.env.context_matrix()
        band = self.scheduler.select_band(ctx, self.env.step)
        hit = bool(active[band])

        false_alarm = False
        reward = 0.0
        ttfi_sample = None
        if hit:
            reward += 1.0
            self.env.register_hit(band)
            if self.band_active_since[band] is not None and not self.band_counted[band]:
                ttfi_sample = self.env.step - self.band_active_since[band]
                self.band_counted[band] = True
        else:
            if random.random() < P_FA_BASE:
                false_alarm = True
                reward -= FALSE_ALARM_COST
        if self.prev_band is not None and self.prev_band != band:
            reward -= SWITCH_COST

        self.engine.record_step(active, band, hit, false_alarm, reward, ttfi_sample)
        self.scheduler.update(band, reward, ctx)
        self.prev_band = band

        return dict(
            step=self.env.step,
            mode=self.mode,
            n_bands=self.n_bands,
            band_watch=band,
            band_watch_ghz=data_source.band_label(band),
            hit=hit,
            false_alarm=false_alarm,
            active=[bool(a) for a in active],
            active_bands=self.env.active_band_snapshot(active),
            metrics=self.engine.snapshot().as_dict(),
        )


async def ws_live_endpoint(ws: WebSocket):
    await ws.accept()
    sess = LiveSession()
    try:
        while True:
            try:
                while True:
                    raw = await asyncio.wait_for(ws.receive_text(), timeout=0.001)
                    msg = json.loads(raw)
                    if msg["type"] == "set_mode":
                        sess.set_mode(msg["mode"])
                    elif msg["type"] == "set_speed":
                        sess.set_speed(int(msg["ms"]))
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
