"""
Interactive live demo: open-loop sweep vs. LinUCB contextual-bandit
scheduler against a simulated RF environment, streamed over a WebSocket.
This is the "try it yourself" surface for judges; the trained DQN result
in results_loader.py is the real, offline-evaluated headline number.
"""
import asyncio
import json
import random

from fastapi import WebSocket, WebSocketDisconnect

from .environment import RFEnvironment, N_BANDS
from .scheduler import make_scheduler
from .metrics_engine import MetricsEngine

CTX_DIM = 5
P_FA_BASE = 0.03
SWITCH_COST = 0.02
FALSE_ALARM_COST = 0.05


class LiveSession:
    def __init__(self):
        self.env = RFEnvironment(N_BANDS)
        self.mode = "open"
        self.scheduler = make_scheduler(self.mode, N_BANDS, CTX_DIM)
        self.dwell_ms = 140
        self.playing = False
        self.prev_band = None
        self.engine = MetricsEngine(dwell_ms=self.dwell_ms)

    def set_mode(self, mode):
        self.mode = mode
        self.scheduler = make_scheduler(mode, N_BANDS, CTX_DIM)
        self.prev_band = None

    def set_speed(self, ms):
        self.dwell_ms = ms
        self.engine.dwell_ms = ms

    def reset(self):
        self.__init__()

    def step_once(self):
        active = self.env.tick()
        ctx = self.env.context_matrix()
        band = self.scheduler.select_band(ctx, self.env.step)
        hit = bool(active[band])

        false_alarm = False
        ttfi_sample = None
        reward = 0.0
        if hit:
            reward += 1.0
            self.env.register_hit(band)
            for em in self.env.emitters:
                if em.on and em.band_now() == band and not em.counted_this_burst:
                    ttfi_sample = self.env.step - em.on_since
                    em.counted_this_burst = True
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
            band_watch=band,
            hit=hit,
            false_alarm=false_alarm,
            active=[bool(a) for a in active],
            emitters=self.env.emitter_snapshot(),
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
