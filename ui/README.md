# EW Smart Scan — Evaluation Console

SIH Task 06 deliverable: Evaluation Metrics Engine, System Integration &
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
your offline training run):
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

**The raw `.h5` files are NOT committed** (19 files, ~380MB total — over
GitHub's comfort zone). Only the tiny precomputed cache is committed
(`backend/data/cache/`, a few hundred KB), which is all the app needs to
run. If you want to regenerate that cache from the original captures (e.g.
with more files, different band count, different slot size):

```bash
mkdir -p backend/data/h5
cp /path/to/your/config_*.h5 backend/data/h5/
rm backend/data/cache/*.npz   # force a rebuild
uvicorn app.main:app --reload  # rebuilds cache on next startup
```
Adjust `N_BANDS`, `F_LO`/`F_HI`, `SLOT_US`, `MIN_PULSES` at the top of
`data_source.py` if you want a different configuration — just note the
cache filename encodes those settings, so changing them triggers a
rebuild automatically (old cache files are simply ignored, not overwritten;
delete them if you want to reclaim the space).

## Repo layout

```
backend/
  app/
    main.py              FastAPI app: REST + websocket + static mount
    metrics_engine.py    figures-of-merit engine (P_int, TTFI, Pd, Pfa, IRE, n_duty, avg_reward)
    results_loader.py    loads final_result.json + Monitor CSV -> JSON for the Final Results tab
    radar_data.py         YOUR loader, unmodified: .h5 -> busy maps, train/test split, caching
    data_source.py         loads the busy-map cache at startup, hands out episode sources
    real_environment.py    wraps a busy-map source into tick()/context_matrix() for the scheduler
    scheduler.py           OpenLoopSweep / LinUCBScheduler / LSTMScheduler stub
    ws_live.py              live demo websocket loop (real data, see section above)
  data/
    final_result.json   <- your offline training run's output, drop a new one here to update the dashboard
    rl_results.csv
    train_monitor.csv
    cache/*.npz          <- precomputed real busy maps (committed, tiny)
    h5/                   <- raw .h5 captures go here if you want to rebuild cache (NOT committed)
  requirements.txt
frontend/
  index.html            single-page tabbed dashboard, vanilla JS, no build step
Dockerfile
```

## Updating with a new training run

Overwrite the three files in `backend/data/` with your latest run's
outputs (same schema: `final_result.json` needs `dataset`, `algo`,
`best_policy`, `rows` (per-policy P_int/Pd/Pfa/TTFI/avg_reward), and `dsp`
(per-config confirmation records); `train_monitor.csv` is a standard
stable-baselines3 `Monitor` wrapper CSV). No code changes needed — restart
the server and the dashboard reflects the new numbers.

## Push to git

```bash
git init
git add .
git commit -m "EW smart scan evaluation console"
git remote add origin <your-repo-url>
git push -u origin main
```
`.gitignore` already excludes `__pycache__/`, venvs, and `.env`. The
`backend/data/` result files are real artifacts, not secrets — they're
committed on purpose so a fresh clone works immediately.

## Notes on IRE and n_duty for offline (trained-model) rows

IRE (interception ratio error vs ground truth) and n_duty (receiver duty
cycle) are computed from step-level ground truth, which the live demo has
and an already-aggregated `final_result.json` row does not. They show as
`null`/omitted for the offline policy table for that reason — if you want
them for the final numbers too, log the per-step active-band mask during
evaluation (not just the final aggregated row) and feed it through
`metrics_engine.MetricsEngine` the same way `ws_live.py` does; then add it
to `final_result.json` and `results_loader.normalize_offline_row` will
pass it through automatically.
