"""Smoke test for the bootstrap helper FastAPI app."""

from fastapi.testclient import TestClient

from andamentum.chunker.types import NextUnitResult
from benchmarks.chunker.app.main import app, _set_executor_factory


def _fake_executor_factory(model: str):
    """Returns an executor that always returns one canned unit."""
    items = [
        NextUnitResult(
            found=True,
            title="Greeting",
            start_anchor="Hello world",
            end_anchor="end here.",
            kind="prose",
        )
    ]

    async def executor(*, instructions, user_message, output_type, validators):
        return items.pop(0) if items else NextUnitResult(found=False, skip_to="x")

    return executor


def test_index_serves_html():
    client = TestClient(app)
    r = client.get("/")
    assert r.status_code == 200
    assert "<html" in r.text.lower()
    assert "chunker" in r.text.lower()


def test_chunk_endpoint_returns_units(monkeypatch):
    _set_executor_factory(_fake_executor_factory)
    client = TestClient(app)
    r = client.post(
        "/api/chunk",
        json={
            "text": "Hello world. This is a test. End here.",
            "domain": "general",
            "model": "fake:test",
            "window_size": 200,
            "lookahead": 50,
        },
    )
    assert r.status_code == 200, r.text
    data = r.json()
    assert "units" in data
    assert len(data["units"]) >= 1
    u = data["units"][0]
    assert u["title"] == "Greeting"
    assert "source_start" in u
    assert "source_end" in u


def test_chunk_endpoint_validates_model_required():
    client = TestClient(app)
    r = client.post(
        "/api/chunk",
        json={"text": "x", "domain": "general"},  # missing model
    )
    assert r.status_code in (400, 422)
