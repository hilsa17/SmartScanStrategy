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

**Live Scheduler Demo tab**: the interactive open-loop-sweep-vs-LinUCB-bandit
console from the earlier prototype, for judges to try live. Kept because a
working, prodable demo is a different kind of evidence than a results
table — use whichever your judges respond to.

## Repo layout

```
backend/
  app/
    main.py            FastAPI app: REST + websocket + static mount
    metrics_engine.py  figures-of-merit engine (P_int, TTFI, Pd, Pfa, IRE, n_duty, avg_reward)
    results_loader.py  loads final_result.json + Monitor CSV -> JSON for the frontend
    environment.py     simulated RF environment (live demo only)
    scheduler.py        OpenLoopSweep / LinUCBScheduler / LSTMScheduler stub (live demo only)
    ws_live.py           live demo websocket loop
  data/
    final_result.json   <- your offline training run's output, drop a new one here to update the dashboard
    rl_results.csv
    train_monitor.csv
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
