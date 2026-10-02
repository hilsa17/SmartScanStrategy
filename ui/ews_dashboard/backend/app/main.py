"""
Run:  uvicorn app.main:app --reload --app-dir backend
Open: http://localhost:8000
"""
from pathlib import Path

from fastapi import FastAPI, WebSocket
from fastapi.responses import FileResponse

from . import results_loader
from .ws_live import ws_live_endpoint

app = FastAPI(title="EW Smart Scan — Evaluation Console")

FRONTEND_DIR = Path(__file__).resolve().parent.parent.parent / "frontend"


@app.get("/api/results")
def api_results():
    """Final trained-policy comparison table (DQN, DQN+DSP, sawtooth, uniform_random on TSRD)."""
    return results_loader.load_final_result()


@app.get("/api/training-curve")
def api_training_curve():
    """Smoothed DQN training reward curve, parsed from the Monitor CSV."""
    return results_loader.load_training_curve()


@app.websocket("/ws/live")
async def ws_live(ws: WebSocket):
    await ws_live_endpoint(ws)


@app.get("/", include_in_schema=False)
def serve_index():
    """Single-file frontend — served explicitly (not via a root Mount) so it
    can never shadow the /ws/live websocket route."""
    return FileResponse(FRONTEND_DIR / "index.html")
