# EW Smart Scan — Evaluation Console

Evaluation Metrics Engine, System Integration &
Real-Time GUI for the Smart Scan Strategy project. Presents the trained
DQN scheduler's real results (on TSRD) alongside a live, interactive
open-loop-vs-bandit demo, from one FastAPI backend and one page.

## Run locally

```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload
```
Open **http://localhost:8000**.

## Run with Docker

```bash
docker build -t ew-console .
docker run -p 8000:8000 ew-console
```

## What's on the page

**Final Results tab** (loads real data from `backend/data/`, produced by
offline training run):
- Policy comparison table — DQN, DQN+DSP, sawtooth, uniform_random on
  TSRD: P_int, Pd, Pfa, TTFI, avg_reward, best policy highlighted.
- P_int bar chart and the DQN training curve (parsed live from the
  Monitor CSV, smoothed with a rolling mean — not a static screenshot).
- DSP confirmation log: confirmed/total, average RL↔DSP band alignment,
  DSP hit rate, forced-exploration fraction, and the per-config list.

**Live Scheduler Demo tab**: open-loop-sweep vs. a LinUCB contextual bandit,
running against **real recorded TSRD radar pulses** — not a synthetic
simulator. Every episode samples a random window from the held-out TEST
captures (the same split `final_result.json`'s numbers were measured on),
using your own `radar_data.py` to turn raw pulses into the 32-band busy map
the scheduler watches. The "Active Bands" panel shows real GHz ranges
(0.5–18 GHz, 32 × ~0.55 GHz bands) pulled live from the capture, not
scripted emitter archetypes.

## Where the live-demo data actually comes from

`backend/app/data_source.py` loads a precomputed "busy map" cache
(`backend/data/cache/*.npz`, built by your `radar_data.py` from the raw
`.h5` captures) at server startup. Each cell `busy[t, b]` is `True` if ≥3
real pulses landed in band `b` during 1ms slot `t`. `real_environment.py`
wraps that into the same `tick()`/`context_matrix()` interface the
scheduler already used, so no scheduler or metrics code had to change —
only the data underneath it did.

```bash
mkdir -p backend/data/h5
cp /path/to/your/config_*.h5 backend/data/h5/
rm backend/data/cache/*.npz   # force a rebuild
uvicorn app.main:app --reload  # rebuilds cache on next startup
```



