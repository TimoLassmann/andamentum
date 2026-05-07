"""Shared Ollama embedding client.

Single home for the HTTP plumbing every sub-module needs:
- ``make_ollama_embedder`` — long-lived client + semaphore, for code paths
  that issue many small embed calls (chunker's semantic split).
- ``embed_texts`` — one-shot fresh-client batch embed with a char-cap safety
  net, for code paths that embed a small batch then move on.
- ``embed_documents`` — embed long documents by chunking, returning
  per-document chunk embeddings (max-sim pairwise comparison).
- ``cosine_similarity`` — with the 1e-8 epsilon every callsite uses.
- ``chunk_text`` — overlapping-window splitter aligned with document_store's
  500-token convention (≈2000 chars).

Sub-modules (chunker, epistemic, ...) re-export the names they need from
here so callers don't have to know which module owns the embedder.

Architecture: shared infrastructure, lazy httpx import, no andamentum deps.
"""

from __future__ import annotations

import asyncio
import warnings
from typing import Awaitable, Callable

import httpx
import numpy as np

# Public type alias: any caller can supply their own embedder by matching this.
# Embedders built by ``make_ollama_embedder`` additionally carry an
# ``input_budget_chars`` attribute that downstream callers (e.g. the chunker)
# read via ``infer_input_budget_chars`` to pre-size their inputs.
EmbeddingFn = Callable[[list[str]], Awaitable[list[list[float]]]]

DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_EMBEDDING_MODEL = "embeddinggemma:latest"

# Aligned with document_store's 500-token / 4-chars-per-token chunking.
DEFAULT_MAX_EMBED_CHARS = 2000
DEFAULT_OVERLAP_CHARS = 200

# Conservative chars-per-token ratio for translating a model's reported
# context length (tokens) into a char budget. The typical English-prose
# average is ~4 chars/token, but dense scientific text (units, references,
# technical vocab) tokenizes closer to ~2.5 chars/token, and equation- /
# unicode-heavy text can drop below that. 2.0 is the safe pre-budget; the
# embedder also has a runtime halving safety net for any text that still
# overflows (see ``_one``).
_CHARS_PER_TOKEN = 2.0


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two vectors. Returns 0.0 for empty input."""
    if not a or not b:
        return 0.0
    av = np.asarray(a, dtype=np.float32)
    bv = np.asarray(b, dtype=np.float32)
    denom = float(np.linalg.norm(av) * np.linalg.norm(bv)) + 1e-8
    return float(np.dot(av, bv) / denom)


def chunk_text(
    text: str,
    *,
    max_chars: int = DEFAULT_MAX_EMBED_CHARS,
    overlap: int = DEFAULT_OVERLAP_CHARS,
) -> list[str]:
    """Split text into overlapping char-windows for embedding."""
    if not text or len(text) <= max_chars:
        return [text or ""]
    chunks: list[str] = []
    stride = max_chars - overlap
    for start in range(0, len(text), stride):
        chunk = text[start : start + max_chars]
        if chunk.strip():
            chunks.append(chunk)
        if start + max_chars >= len(text):
            break
    return chunks or [text[:max_chars]]


def discover_input_budget_chars(
    model: str,
    *,
    base_url: str = DEFAULT_OLLAMA_URL,
    timeout: float = 5.0,
) -> int:
    """Query Ollama's ``/api/show`` for the model's context length.

    Returns ``int(context_length * _CHARS_PER_TOKEN)``. Falls back to
    ``DEFAULT_MAX_EMBED_CHARS`` and emits a warning if the endpoint is
    unreachable, errors, or doesn't expose a context length for this model.

    Ollama exposes the context length under ``model_info["<arch>.context_length"]``
    where ``<arch>`` varies by model (``gemma3``, ``nomic-bert-moe``, ...).
    We match by suffix rather than hard-coding the prefix.
    """
    try:
        with httpx.Client(timeout=timeout) as client:
            resp = client.post(f"{base_url}/api/show", json={"name": model})
            resp.raise_for_status()
            payload = resp.json()
    except (httpx.HTTPError, ValueError) as e:
        warnings.warn(
            f"Could not discover context length for embedding model "
            f"'{model}' ({e!r}); falling back to {DEFAULT_MAX_EMBED_CHARS} "
            f"chars. Pass input_budget_chars= to override.",
            stacklevel=2,
        )
        return DEFAULT_MAX_EMBED_CHARS

    info = payload.get("model_info") or {}
    for key, value in info.items():
        if key.endswith(".context_length") and isinstance(value, int) and value > 0:
            return int(value * _CHARS_PER_TOKEN)

    warnings.warn(
        f"Ollama /api/show for '{model}' returned no .context_length entry; "
        f"falling back to {DEFAULT_MAX_EMBED_CHARS} chars. Pass "
        f"input_budget_chars= to override.",
        stacklevel=2,
    )
    return DEFAULT_MAX_EMBED_CHARS


def infer_input_budget_chars(
    embedding_fn: EmbeddingFn | None,
    *,
    fallback: int = DEFAULT_MAX_EMBED_CHARS,
) -> int:
    """Read the embedder's advertised input budget.

    Embedders constructed by ``make_ollama_embedder`` carry an
    ``input_budget_chars`` attribute set from the model's reported context
    length. User-supplied bare callables (e.g. test fakes) may not — in
    that case we return ``fallback``.
    """
    return int(getattr(embedding_fn, "input_budget_chars", fallback))


def make_ollama_embedder(
    *,
    model: str = DEFAULT_EMBEDDING_MODEL,
    base_url: str = DEFAULT_OLLAMA_URL,
    max_concurrent: int = 8,
    timeout: float = 60.0,
    input_budget_chars: int | None = None,
) -> EmbeddingFn:
    """Build a long-lived embedder bound to one Ollama HTTP client.

    The returned coroutine accepts a list of strings and returns one vector
    per string. Concurrency is bounded so we don't overwhelm Ollama on long
    documents. The client lives for the life of the embedder — pair this
    with code paths that embed many batches before tear-down.

    The returned callable also carries an ``input_budget_chars`` attribute
    (discovered from Ollama's ``/api/show`` unless explicitly supplied) so
    downstream callers can pre-size their inputs to fit the model's context.
    """
    sem = asyncio.Semaphore(max_concurrent)
    client = httpx.AsyncClient(timeout=timeout)

    if input_budget_chars is None:
        input_budget_chars = discover_input_budget_chars(model, base_url=base_url)

    async def _one(text: str) -> list[float]:
        # Hold the semaphore only while we're issuing the HTTP call;
        # any recursion (overflow → halve-and-retry) re-acquires it
        # cleanly without deadlocking against ``max_concurrent``.
        async with sem:
            r = await client.post(
                f"{base_url}/api/embeddings",
                json={"model": model, "prompt": text},
            )
            status = r.status_code
            body = r.text if status >= 400 else None
            embedding = r.json()["embedding"] if status < 400 else None

        if embedding is not None:
            return embedding

        # Runtime safety net: if the model reports an over-context input
        # despite our pre-budget, halve and average. Mean-pooling halves is
        # the same approximation ``embed_documents`` uses for long-doc
        # embeddings; for the chunker's cosine-drop signal between adjacent
        # paragraphs it is more than adequate. Bail at a tiny ``text`` so a
        # truly broken Ollama response can't recurse forever.
        body_lc = (body or "").lower()
        if "context length" in body_lc and len(text) > 64:
            mid = len(text) // 2
            left, right = await asyncio.gather(_one(text[:mid]), _one(text[mid:]))
            return [(a + b) / 2.0 for a, b in zip(left, right)]

        raise RuntimeError(
            f"Ollama embedding call returned {status} "
            f"for model={model!r}, input_chars={len(text)}, "
            f"advertised_budget_chars={input_budget_chars}. "
            f"Response body: {(body or '')[:300]!r}"
        )

    async def embed(texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        return await asyncio.gather(*(_one(t) for t in texts))

    embed.input_budget_chars = input_budget_chars  # type: ignore[attr-defined]
    return embed


async def embed_texts(
    texts: list[str],
    *,
    model: str,
    base_url: str = DEFAULT_OLLAMA_URL,
    max_chars: int = DEFAULT_MAX_EMBED_CHARS,
    timeout: float = 30.0,
) -> list[list[float]]:
    """One-shot embed a batch of short texts via a fresh Ollama client.

    Each text is truncated to ``max_chars`` as a safety net. For long
    documents use ``embed_documents``.

    Raises:
        RuntimeError: If Ollama is unreachable or returns an error.
    """
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            embeddings: list[list[float]] = []
            for text in texts:
                resp = await client.post(
                    f"{base_url}/api/embeddings",
                    json={"model": model, "prompt": text[:max_chars]},
                )
                resp.raise_for_status()
                embeddings.append(resp.json()["embedding"])
            return embeddings
    except httpx.ConnectError as e:
        raise RuntimeError(
            f"Cannot connect to Ollama at {base_url}. Start Ollama with: ollama serve"
        ) from e
    except httpx.HTTPStatusError as e:
        raise RuntimeError(
            f"Ollama embedding failed (HTTP {e.response.status_code}). "
            f"Ensure model '{model}' is pulled: ollama pull {model}"
        ) from e


async def embed_documents(
    texts: list[str],
    *,
    model: str,
    base_url: str = DEFAULT_OLLAMA_URL,
) -> list[list[list[float]]]:
    """Embed long documents by chunking, returning per-document chunk embeddings.

    Each document is split into overlapping chunks of ``DEFAULT_MAX_EMBED_CHARS``;
    chunks are embedded as a single flat batch; results are reshaped back to
    one list of chunk-vectors per input document. Callers use the chunk
    embeddings for max-sim pairwise comparison (best cosine between any
    chunk pair of two documents).
    """
    doc_chunks: list[list[str]] = [chunk_text(t) for t in texts]
    flat: list[str] = [c for chunks in doc_chunks for c in chunks]
    flat_embeds = await embed_texts(flat, model=model, base_url=base_url)

    out: list[list[list[float]]] = []
    idx = 0
    for chunks in doc_chunks:
        out.append(flat_embeds[idx : idx + len(chunks)])
        idx += len(chunks)
    return out
