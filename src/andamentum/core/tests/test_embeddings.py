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
    OllamaEmbedder,
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


async def test_ollama_embedder_truncates_oversized_text_before_batch_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A single dense chunk must not be able to take the whole batch down.

    document_store used to send a chunk's full, untruncated text into a
    batch ``/api/embed`` call; a dense OCR'd table tokenizing far past the
    char:token estimate would make Ollama reject the whole batch. The fix is
    to cap every text to the discovered budget before it goes out.
    """
    seen_inputs: list[list[str]] = []

    class _FakeAsyncClient:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        async def post(self, url: str, json: dict) -> object:
            texts: list[str] = json["input"]
            seen_inputs.append(texts)

            class _Resp:
                def raise_for_status(self) -> None:
                    return None

                def json(self) -> dict:
                    return {"embeddings": [[1.0, 2.0] for _ in texts]}

            return _Resp()

    monkeypatch.setattr(httpx, "AsyncClient", _FakeAsyncClient)
    embedder = OllamaEmbedder(model="fakemodel:latest", input_budget_chars=10)

    result = await embedder.embed_batch(["short", "x" * 500])

    assert len(result) == 2
    # The oversized text was capped to the budget before it left this process.
    assert seen_inputs[0] == ["short", "x" * 10]


def _recording_client(seen_inputs: list[list[str]]) -> type:
    """Fake AsyncClient that records each ``/api/embed`` input list.

    Each returned vector carries the length and the first digit of the text
    it was made from, so a test can check that vectors come back in input
    order across several requests.
    """

    class _FakeAsyncClient:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        async def post(self, url: str, json: dict) -> object:
            texts: list[str] = json["input"]
            seen_inputs.append(texts)

            class _Resp:
                def raise_for_status(self) -> None:
                    return None

                def json(self) -> dict:
                    return {"embeddings": [[float(len(t)), float(t[0])] for t in texts]}

            return _Resp()

    return _FakeAsyncClient


async def test_ollama_embedder_splits_large_batch_by_item_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Many chunks must go out as several requests, in order.

    A 1,238-chunk document sent as ONE ``/api/embed`` request made a
    single-slot local Ollama drop the connection (raw EOF, no status code),
    independent of any single chunk's length. The cap has to sit on this
    side, and the vectors must still land at the index of their text.
    """
    seen_inputs: list[list[str]] = []
    monkeypatch.setattr(httpx, "AsyncClient", _recording_client(seen_inputs))
    embedder = OllamaEmbedder(
        model="fakemodel:latest", input_budget_chars=1000, max_batch_items=64
    )

    texts = [str(i % 10) * (i + 1) for i in range(150)]
    result = await embedder.embed_batch(texts)

    assert [len(g) for g in seen_inputs] == [64, 64, 22]
    assert [g for group in seen_inputs for g in group] == texts
    assert [(int(v[0]), int(v[1])) for v in result] == [
        (len(t), int(t[0])) for t in texts
    ]


async def test_ollama_embedder_splits_large_batch_by_char_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A few long texts must also split when their total passes the char cap."""
    seen_inputs: list[list[str]] = []
    monkeypatch.setattr(httpx, "AsyncClient", _recording_client(seen_inputs))
    embedder = OllamaEmbedder(
        model="fakemodel:latest",
        input_budget_chars=100,
        max_batch_items=64,
        max_batch_chars=100,
    )

    texts = ["1" * 60, "2" * 60, "3" * 60]
    result = await embedder.embed_batch(texts)

    assert [len(g) for g in seen_inputs] == [1, 1, 1]
    assert len(result) == 3
    assert [int(v[1]) for v in result] == [1, 2, 3]


async def test_ollama_embedder_one_text_over_char_budget_still_goes_out(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A single text longer than ``max_batch_chars`` forms its own group."""
    seen_inputs: list[list[str]] = []
    monkeypatch.setattr(httpx, "AsyncClient", _recording_client(seen_inputs))
    embedder = OllamaEmbedder(
        model="fakemodel:latest", input_budget_chars=100, max_batch_chars=10
    )

    result = await embedder.embed_batch(["7" * 50])

    assert seen_inputs == [["7" * 50]]
    assert len(result) == 1


async def test_ollama_embedder_short_response_is_loud(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fewer vectors than inputs must raise, not misalign chunks silently."""

    class _FakeAsyncClient:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        async def post(self, url: str, json: dict) -> object:
            class _Resp:
                def raise_for_status(self) -> None:
                    return None

                def json(self) -> dict:
                    return {"embeddings": [[1.0]]}

            return _Resp()

    monkeypatch.setattr(httpx, "AsyncClient", _FakeAsyncClient)
    embedder = OllamaEmbedder(model="fakemodel:latest", input_budget_chars=100)

    with pytest.raises(RuntimeError, match="1 vectors for 2 inputs"):
        await embedder.embed_batch(["a", "b"])


async def test_ollama_embedder_embed_text_and_close(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    closed = False

    class _FakeAsyncClient:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        async def post(self, url: str, json: dict) -> object:
            class _Resp:
                def raise_for_status(self) -> None:
                    return None

                def json(self) -> dict:
                    return {"embeddings": [[3.0, 4.0]]}

            return _Resp()

        async def aclose(self) -> None:
            nonlocal closed
            closed = True

    monkeypatch.setattr(httpx, "AsyncClient", _FakeAsyncClient)
    embedder = OllamaEmbedder(model="fakemodel:latest", input_budget_chars=100)

    assert await embedder.embed_text("hello") == [3.0, 4.0]
    await embedder.close()
    assert closed
