"""Embedding cache that sits outside every store.

Keyed by (model name, model digest, sha256 of the text). A new store layout
re-reads vectors from here and embeds nothing; only a new embedding model or
changed passage text costs model calls. The digest is part of the key so that
pulling a new build of the same tag cannot silently mix two models' vectors.
"""

from __future__ import annotations

import asyncio
import hashlib
import sqlite3
import struct
from pathlib import Path
from typing import Awaitable, Callable

import httpx

from andamentum.core.embeddings import DEFAULT_OLLAMA_URL, OllamaEmbedder

EmbedFn = Callable[[list[str]], Awaitable[list[list[float]]]]


def _text_key(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _pack(vec: list[float]) -> bytes:
    return struct.pack(f"{len(vec)}f", *vec)


def _unpack(blob: bytes) -> list[float]:
    return list(struct.unpack(f"{len(blob) // 4}f", blob))


def ollama_model_digest(model: str, *, base_url: str = DEFAULT_OLLAMA_URL) -> str:
    """The local digest of an Ollama model tag, from ``/api/tags`` (no model load)."""
    r = httpx.get(f"{base_url}/api/tags", timeout=10.0)
    r.raise_for_status()
    wanted = model if ":" in model else f"{model}:latest"
    for m in r.json()["models"]:
        if m["name"] == wanted:
            return m["digest"]
    raise LookupError(f"Ollama has no model {wanted!r}; pull it first")


class EmbeddingCache:
    def __init__(
        self, path: Path, *, model: str, digest: str, embed_fn: EmbedFn
    ) -> None:
        self.path = path
        self.model = model
        self.digest = digest
        self._embed_fn = embed_fn
        self.misses = 0  # texts embedded by this instance
        path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(path) as db:
            db.execute(
                "CREATE TABLE IF NOT EXISTS embeddings ("
                " model TEXT, digest TEXT, text_sha256 TEXT, vector BLOB,"
                " PRIMARY KEY (model, digest, text_sha256))"
            )

    @classmethod
    def for_ollama(cls, path: Path, *, model: str) -> "EmbeddingCache":
        embedder: OllamaEmbedder | None = None
        # The model is local: a run with --concurrency N must still send it
        # one request at a time.
        one_at_a_time = asyncio.Lock()

        async def embed(texts: list[str]) -> list[list[float]]:
            # Created on the first miss only, so a fully cached run never
            # touches the model (OllamaEmbedder probes it on construction).
            nonlocal embedder
            async with one_at_a_time:
                if embedder is None:
                    embedder = OllamaEmbedder(model=model)
                return await embedder.embed_batch(texts)

        return cls(path, model=model, digest=ollama_model_digest(model), embed_fn=embed)

    def _lookup(self, keys: list[str]) -> dict[str, list[float]]:
        found: dict[str, list[float]] = {}
        with sqlite3.connect(self.path) as db:
            for i in range(0, len(keys), 500):
                part = keys[i : i + 500]
                marks = ",".join("?" * len(part))
                rows = db.execute(
                    f"SELECT text_sha256, vector FROM embeddings WHERE model = ? "
                    f"AND digest = ? AND text_sha256 IN ({marks})",
                    (self.model, self.digest, *part),
                ).fetchall()
                found.update((k, _unpack(v)) for k, v in rows)
        return found

    async def get_or_compute(self, texts: list[str]) -> list[list[float]]:
        """Vectors for ``texts`` in order; only cache misses reach the model."""
        keys = [_text_key(t) for t in texts]
        found = self._lookup(sorted(set(keys)))
        todo: dict[str, str] = {}
        for k, t in zip(keys, texts, strict=True):
            if k not in found:
                todo.setdefault(k, t)
        if todo:
            todo_keys = list(todo)
            vectors = await self._embed_fn([todo[k] for k in todo_keys])
            if len(vectors) != len(todo_keys):
                raise RuntimeError(
                    f"embedder returned {len(vectors)} vectors for {len(todo_keys)} texts"
                )
            with sqlite3.connect(self.path) as db:
                db.executemany(
                    "INSERT OR REPLACE INTO embeddings VALUES (?, ?, ?, ?)",
                    [
                        (self.model, self.digest, k, _pack(v))
                        for k, v in zip(todo_keys, vectors, strict=True)
                    ],
                )
            self.misses += len(todo_keys)
            # Round-trip through float32 so the run that fills the cache sees
            # exactly the vectors every later run reads back.
            found.update(
                (k, _unpack(_pack(v))) for k, v in zip(todo_keys, vectors, strict=True)
            )
        return [found[k] for k in keys]
