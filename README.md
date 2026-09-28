# ES Scheduler Console — local backend

## Run it (3 commands)

```bash
cd ews_backend
pip install -r requirements.txt
uvicorn main:app --reload
```

Open **http://localhost:8000** in your browser. That's the whole demo —
FastAPI serves `static/index.html` and streams simulation state over
`ws://localhost:8000/ws`.

Click **Start**, toggle **Open-loop sweep** vs **LinUCB bandit** live, drag
the dwell-time slider. Everything you see (waterfall, P_int, TTFI, IRE,
n_duty, P_fa, avg reward/step) is computed server-side in `main.py` and
pushed to the browser — nothing is simulated in JS anymore.

## File map

| File | What it is |
|---|---|
| `environment.py` | RF ground-truth simulator: 16 bands, 6 emitter archetypes (fixed, frequency-agile hopper, periodic self-scanner, sporadic burst). `context_matrix()` is the feature vector the scheduler reads each step. |
| `scheduler.py` | `BaseScheduler` interface + `OpenLoopSweep` (baseline) + `LinUCBScheduler` (the ML scheduler — a contextual bandit, trains online, no dataset needed) + `LSTMScheduler` stub for later. |
| `main.py` | FastAPI app. `Session` class owns one environment + one scheduler per connected browser tab, runs the step loop, computes all figures of merit, and speaks the websocket protocol below. |
| `static/index.html` | The dashboard UI. Pure JS, no framework — connects to `/ws`, renders the waterfall on canvas, updates metric cards. |

## Websocket protocol

**Server → client**, one JSON message per simulated step:
```json
{
  "step": 812,
  "mode": "smart",
  "band_watch": 7,
  "hit": true,
  "false_alarm": false,
  "active": [false, true, ...],           // 16 bools, ground truth this step
  "emitters": [{"name": "...", "type": "...", "on": true, "band": 7}, ...],
  "metrics": {"p_int": 0.31, "ttfi": 2.4, "ire": 0.02, "n_duty": 0.833,
              "pfa": 0.028, "avg_reward": 0.41}
}
```

**Client → server** control messages:
```json
{"type": "set_mode", "mode": "smart"}   // or "open", "lstm"
{"type": "set_speed", "ms": 140}
{"type": "play"}
{"type": "pause"}
{"type": "reset"}
```

## 1. Swapping in a trained model instead of the bandit

`LinUCBScheduler` already *is* a trained model — it's a per-band linear
regressor updated online every step from reward, which is the right scope
for a hackathon (no offline dataset, no training wait, and it's a
legitimate, citable technique — this is the standard "multi-armed bandit
scheduling" framing your literature reference (Adamy, *EW 101*) discusses
under receiver search strategy). For judges who want to see "real ML":

- Point at `scheduler.py`, show `select_band()`/`update()` are the only two
  methods any scheduler needs to implement, and that `OpenLoopSweep`,
  `LinUCBScheduler`, and the `LSTMScheduler` stub all satisfy the same
  interface — swapping is a one-line change in `make_scheduler()`.
- To actually finish `LSTMScheduler`: every websocket message already has
  everything a training example needs (context implicitly via `active` +
  `band_watch` + `hit`). Add a small script that opens the `/ws` endpoint
  as a client (or just log inside `Session.step_once`), dump ~5–10k steps
  to `log.jsonl`, then train a 1-layer LSTM offline (PyTorch) to predict
  next-step-active-band-probabilities, and at inference time argmax that
  distribution inside `select_band()`. That's a post-hackathon stretch —
  don't attempt it live unless you have hours, not minutes, left.

## 2. Why not wire the *published claude.ai artifact* to this backend

The artifact link runs in a locked-down preview sandbox: it's only allowed
to load scripts from a short CDN allowlist and cannot open arbitrary
network/WebSocket connections to `localhost` or anywhere else. That's why
this backend ships its own `static/index.html` instead of trying to make
the claude.ai link talk to your laptop. Use the claude.ai artifact link
only as a passive "here's the concept" tab (e.g. on your phone while your
laptop runs the real thing on the projector); demo the *real* integrated
system from `http://localhost:8000`.

## 3. PyQt6 (optional — only if a judge specifically wants a native window)

Fastest path, no UI rewrite: embed the same page in a `QWebEngineView`.

```python
# pyqt_shell.py — run this INSTEAD of opening a browser tab, backend still needed
import sys, subprocess, time
from PyQt6.QtWidgets import QApplication, QMainWindow
from PyQt6.QtWebEngineWidgets import QWebEngineView
from PyQt6.QtCore import QUrl

subprocess.Popen(["uvicorn", "main:app"])  # starts the FastAPI backend
time.sleep(1.5)

app = QApplication(sys.argv)
win = QMainWindow()
win.setWindowTitle("ES Scheduler Console")
view = QWebEngineView()
view.load(QUrl("http://localhost:8000"))
win.setCentralWidget(view)
win.resize(1280, 860)
win.show()
sys.exit(app.exec())
```
Needs `pip install PyQt6 PyQt6-WebEngine`. This gets you a literal desktop
window for the "PyQt6" line on your slide without touching any of the
dashboard code above — same HTML/JS, just displayed in a native frame
instead of a browser tab.
