"""FastAPI bridge: serves the live engine to the syndicate dashboard.

Endpoints
---------
GET  /health              liveness + mode (sim | kafka)
GET  /api/book            current book snapshot (no sim advance)
POST /api/approve         record bookrunner approval of the latest rec
WS   /ws                  live payloads at ~1 Hz (sim tick + cached rec)
GET  /                    the dashboard (docs/index.html)
GET  /static/...          dashboard assets (single-file: not needed yet)

Run:
    python -m dcm_engine.bridge.app
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

from dcm_engine.bridge.engine import get_engine
from dcm_engine.core.pricing import bond_price_from_yield

logger = logging.getLogger(__name__)

DASHBOARD_PATH = Path(__file__).resolve().parents[2] / "docs" / "dashboard.html"
LANDING_PATH = Path(__file__).resolve().parents[2] / "docs" / "index.html"
BROADCAST_INTERVAL = 1.0  # seconds between WS pushes
MUTATION_LOCK = asyncio.Lock()


class ApproveRequest(BaseModel):
    """Body for POST /api/approve."""

    approver: str = Field(min_length=2, max_length=64)
    tranche_id: str | None = Field(default=None, description="defaults to the live tranche")


app = FastAPI(title="DCM Syndicate Pricing Bridge", version="1.0.0")
_engine = get_engine()
_approvals: list[dict[str, Any]] = []


@app.get("/health")
async def health() -> JSONResponse:
    """Liveness probe. Reports whether Kafka is attached or we're simulating."""
    import os as _os

    mode = "kafka" if _os.environ.get("DCM_KAFKA_BOOTSTRAP") else "sim"
    rec = _engine.current()["recommendation"]
    return JSONResponse(
        {
            "status": "ok",
            "mode": mode,
            "model": _engine.model_info,
            "has_recommendation": rec is not None,
            "approvals_recorded": len(_approvals),
        }
    )


@app.get("/api/book")
async def get_book() -> JSONResponse:
    """Current book snapshot without advancing the simulation."""
    return JSONResponse(_engine.current())


@app.post("/api/approve")
async def approve(req: ApproveRequest) -> JSONResponse:
    """Human bookrunner approval gate: records who approved the live rec."""
    payload = _engine.current()
    rec = payload["recommendation"]
    if rec is None:
        raise HTTPException(status_code=409, detail="no recommendation finalized yet")
    tranche_id = req.tranche_id or payload["tranche"]["tranche_id"]
    if rec["tranche_id"] != tranche_id:
        raise HTTPException(status_code=409, detail="tranche mismatch vs live recommendation")
    entry = {
        "approver": req.approver,
        "tranche_id": tranche_id,
        "spread_bps": rec["final_spread_bps"],
        "approved_at": payload["server_time"],
    }
    _approvals.append(entry)
    logger.info("APPROVAL recorded: %s", entry)
    return JSONResponse({"status": "approved", **entry})


@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket) -> None:
    """Push live payloads at ~1 Hz. Each frame: book + cached recommendation."""
    await websocket.accept()
    logger.info("dashboard client connected")
    try:
        while True:
            # tick_and_recommend advances the sim and refreshes the rec on TTL.
            payload = await asyncio.to_thread(_engine.tick_and_recommend)
            # Enrich with derived pricing (price for the recommended spread).
            rec = payload["recommendation"]
            if rec:
                rec["final_price"] = round(
                    bond_price_from_yield(
                        face=100.0,
                        ytm=rec["final_yield"],
                        coupon_rate=payload["tranche"]["coupon"],
                        years=payload["tranche"]["tenor_years"],
                    ),
                    3,
                )
            await websocket.send_text(json.dumps(payload))
            await asyncio.sleep(BROADCAST_INTERVAL)
    except WebSocketDisconnect:
        logger.info("dashboard client disconnected")
    except Exception:
        logger.exception("ws handler error")
        try:
            await websocket.close(code=1011)
        except Exception:  # noqa: BLE001 - client already gone; nothing to do
            logger.debug("websocket close after error also failed")


@app.get("/", response_class=HTMLResponse)
async def landing() -> HTMLResponse:
    """Serve the landing page (docs/index.html)."""
    if not LANDING_PATH.exists():
        raise HTTPException(status_code=404, detail="landing page not built")
    return HTMLResponse(LANDING_PATH.read_text(encoding="utf-8"))


@app.get("/dashboard", response_class=HTMLResponse)
@app.get("/dashboard.html", response_class=HTMLResponse)
async def dashboard() -> HTMLResponse:
    """Serve the live syndicate dashboard (docs/dashboard.html).

    Both /dashboard and /dashboard.html are routed so the landing page's
    relative links work identically on this server and on GitHub Pages.
    """
    if not DASHBOARD_PATH.exists():
        raise HTTPException(status_code=404, detail="dashboard not built")
    return HTMLResponse(DASHBOARD_PATH.read_text(encoding="utf-8"))


def main() -> None:
    """Entrypoint: uvicorn on :8123 by default (DCM_PORT to override)."""
    import uvicorn

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    port = int(os.environ.get("DCM_PORT", "8123"))
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="info")


if __name__ == "__main__":
    main()
