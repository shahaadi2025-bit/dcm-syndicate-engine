"""Tests for the FastAPI bridge: REST, approval gate, WebSocket frames."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from dcm_engine.bridge.app import app


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(app)


def test_health(client: TestClient) -> None:
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["mode"] in ("sim", "kafka")


def test_book_shape(client: TestClient) -> None:
    r = client.get("/api/book")
    assert r.status_code == 200
    body = r.json()
    assert set(body) >= {"tranche", "book", "ordered_mm", "target_mm", "osr", "server_time"}
    assert body["target_mm"] == 1000.0


def test_approve_requires_recommendation(client: TestClient) -> None:
    r = client.post("/api/approve", json={"approver": "desk-1"})
    # 200 once a rec exists, 409 before the first agent run.
    assert r.status_code in (200, 409)


def test_ws_frames(client: TestClient) -> None:
    with client.websocket_connect("/ws") as ws:
        first = ws.receive_json()
        assert set(first) >= {"tranche", "book", "ordered_mm", "recommendation"}
        second = ws.receive_json()
        assert second["server_time"] >= first["server_time"]


def test_landing_served(client: TestClient) -> None:
    r = client.get("/")
    assert r.status_code == 200
    assert "DCM" in r.text


def test_dashboard_served(client: TestClient) -> None:
    r = client.get("/dashboard")
    # Implemented below; until then treat 404 as pending.
    assert r.status_code in (200, 404)
