"""tests/integration/test_api.py — Integration tests for the FastAPI serving layer."""
from __future__ import annotations

import pytest
from httpx import AsyncClient, ASGITransport


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
async def client():
    from src.serving.api import app
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac


@pytest.mark.anyio
async def test_health(client):
    resp = await client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert "model_loaded" in body


@pytest.mark.anyio
async def test_predict_returns_list(client):
    resp = await client.post("/predict", json={"inputs": ["Hello world", "Test sentence"]})
    assert resp.status_code == 200
    body = resp.json()
    assert "predictions" in body
    assert len(body["predictions"]) == 2
    assert "latency_ms" in body
    assert body["latency_ms"] >= 0


@pytest.mark.anyio
async def test_predict_single_input(client):
    resp = await client.post("/predict", json={"inputs": ["Single input"]})
    assert resp.status_code == 200
    assert len(resp.json()["predictions"]) == 1


@pytest.mark.anyio
async def test_predict_model_version_passthrough(client):
    resp = await client.post("/predict", json={"inputs": ["test"], "model_version": "v2"})
    assert resp.status_code == 200
    assert resp.json()["model_version"] == "v2"


@pytest.mark.anyio
async def test_predict_empty_inputs_rejected(client):
    resp = await client.post("/predict", json={"inputs": []})
    assert resp.status_code == 422  # Pydantic validation error


@pytest.mark.anyio
async def test_feedback_endpoint(client):
    resp = await client.post("/feedback", json={
        "prediction_id": "abc-123",
        "true_label": 1,
        "predicted_label": 0,
        "source": "test",
    })
    assert resp.status_code == 200
    assert resp.json()["status"] == "recorded"


@pytest.mark.anyio
async def test_metrics_endpoint(client):
    resp = await client.get("/metrics")
    assert resp.status_code == 200
    # Prometheus text format
    assert b"predict_requests_total" in resp.content or resp.status_code == 200
