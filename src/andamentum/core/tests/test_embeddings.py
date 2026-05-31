"""Tests for the embedding-budget discovery and reader helpers.

The chunker relies on these to subdivide oversized paragraphs before
embedding them. If the discovery / fallback paths break, the chunker
silently regresses to its old "send 12k chars to a 2k-context model"
behavior and Ollama returns 500.
"""

from __future__ import annotations

import warnings

import httpx
import pytest

from andamentum.core.embeddings import (
    DEFAULT_MAX_EMBED_CHARS,
    discover_input_budget_chars,
    infer_input_budget_chars,
    make_ollama_embedder,
)


def _show_payload(context_length: int | None, prefix: str = "gemma3") -> dict:
    """Build a fake /api/show JSON payload."""
    info: dict = {f"{prefix}.embedding_length": 768}
    if context_length is not None:
        info[f"{prefix}.context_length"] = context_length
    return {"model_info": info}


def test_discover_input_budget_chars_parses_context_length(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _show_payload(2048, prefix="gemma3")

    class _FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return payload

    class _FakeClient:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def __enter__(self) -> "_FakeClient":
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def post(self, url: str, json: dict) -> _FakeResponse:
            assert url.endswith("/api/show")
            assert json == {"name": "embeddinggemma:latest"}
            return _FakeResponse()

    monkeypatch.setattr(httpx, "Client", _FakeClient)
    budget = discover_input_budget_chars("embeddinggemma:latest")
    # 2048 tokens × 2 chars/token = 4096 (conservative for dense scientific text)
    assert budget == 4096


def test_discover_input_budget_chars_handles_alt_arch_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Different model archs use different prefixes — we match by suffix."""
    payload = _show_payload(512, prefix="nomic-bert-moe")

    class _FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return payload

    class _FakeClient:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def __enter__(self) -> "_FakeClient":
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def post(self, url: str, json: dict) -> _FakeResponse:
            return _FakeResponse()

    monkeypatch.setattr(httpx, "Client", _FakeClient)
    assert discover_input_budget_chars("nomic-embed-text-v2-moe:latest") == 1024


def test_discover_input_budget_chars_falls_back_on_http_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeClient:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def __enter__(self) -> "_FakeClient":
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def post(self, url: str, json: dict) -> object:
            raise httpx.ConnectError("ollama down")

    monkeypatch.setattr(httpx, "Client", _FakeClient)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        budget = discover_input_budget_chars("anything:latest")
    assert budget == DEFAULT_MAX_EMBED_CHARS
    assert any("Could not discover" in str(w.message) for w in caught)


def test_discover_input_budget_chars_falls_back_when_no_context_length(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Some odd models may not expose context_length — degrade gracefully."""
    payload = {"model_info": {"some.other_field": 42}}

    class _FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return payload

    class _FakeClient:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def __enter__(self) -> "_FakeClient":
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def post(self, url: str, json: dict) -> _FakeResponse:
            return _FakeResponse()

    monkeypatch.setattr(httpx, "Client", _FakeClient)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        budget = discover_input_budget_chars("weirdmodel:latest")
    assert budget == DEFAULT_MAX_EMBED_CHARS
    assert any("no .context_length" in str(w.message) for w in caught)


def test_infer_input_budget_chars_reads_attribute() -> None:
    async def fake_embed(texts: list[str]) -> list[list[float]]:
        return [[0.1] * 4 for _ in texts]

    fake_embed.input_budget_chars = 4096  # type: ignore[attr-defined]
    assert infer_input_budget_chars(fake_embed) == 4096


def test_infer_input_budget_chars_falls_back_on_missing_attribute() -> None:
    async def bare_embed(texts: list[str]) -> list[list[float]]:
        return [[0.1] * 4 for _ in texts]

    assert infer_input_budget_chars(bare_embed) == DEFAULT_MAX_EMBED_CHARS
    assert infer_input_budget_chars(None) == DEFAULT_MAX_EMBED_CHARS
    assert infer_input_budget_chars(bare_embed, fallback=999) == 999


def test_make_ollama_embedder_stamps_explicit_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When input_budget_chars is supplied, no /api/show call is made."""

    class _BoomClient:
        def __init__(self, *args: object, **kwargs: object) -> None:
            raise AssertionError("should not be called when budget is explicit")

        def __enter__(self) -> "_BoomClient":  # pragma: no cover
            return self

        def __exit__(self, *args: object) -> None:  # pragma: no cover
            return None

    monkeypatch.setattr(httpx, "Client", _BoomClient)
    embedder = make_ollama_embedder(model="anything:latest", input_budget_chars=8000)
    assert infer_input_budget_chars(embedder) == 8000


async def test_embedder_halves_on_context_length_overflow(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If Ollama reports the input still exceeds context (because the static
    chars/token ratio was too generous for this specific text), the embedder
    halves the input and averages the two halves' embeddings rather than
    crashing the pipeline."""
    seen_lengths: list[int] = []

    class _FakeResponse:
        def __init__(
            self, status_code: int, json_data: dict | None = None, text: str = ""
        ) -> None:
            self.status_code = status_code
            self._json = json_data or {}
            self.text = text

        def json(self) -> dict:
            return self._json

    class _FakeAsyncClient:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        async def post(self, url: str, json: dict) -> _FakeResponse:
            text: str = json["prompt"]
            seen_lengths.append(len(text))
            # Anything over 100 chars "exceeds context"; smaller succeeds.
            if len(text) > 100:
                return _FakeResponse(
                    500, text='{"error":"the input length exceeds the context length"}'
                )
            return _FakeResponse(200, json_data={"embedding": [float(len(text))] * 3})

    monkeypatch.setattr(httpx, "AsyncClient", _FakeAsyncClient)
    embedder = make_ollama_embedder(model="fakemodel:latest", input_budget_chars=200)
    # 400 chars → first call 500s, then halves → 200, both 500 again, halve to
    # 100, both succeed. Final: averaged embedding of 4 quarter-pieces.
    out = await embedder(["x" * 400])
    assert len(out) == 1 and len(out[0]) == 3
    # Averaged 100-char halves => each component = 100 (the constant we encoded).
    assert all(v == pytest.approx(100.0) for v in out[0])
    # We saw 1 (400) + 2 (200) + 4 (100) = 7 calls total.
    assert seen_lengths.count(400) == 1
    assert seen_lengths.count(200) == 2
    assert seen_lengths.count(100) == 4
