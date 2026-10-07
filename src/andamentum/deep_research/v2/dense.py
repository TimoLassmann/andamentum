"""Dense similarity: the question, each sub-question and each candidate as
vectors of a local embedding model, compared by cosine.

The ``rrf_dense`` rank strategy (``strategies.RrfDenseRank``) reads two lists
from it per sub-question: the candidates ranked by cosine against the
sub-question, and by cosine against the question. A candidate's text is its
title and abstract (or the search snippet), cut at ``DENSE_MAX_CHARS``.

The embedding model is an explicit argument of ``research()``
(``embedding_model=``, CLI ``--embedding-model NAME``); nothing here names a
default. ``OllamaDenseEmbedder`` sends one ``/api/embed`` request per batch of
at most ``EMBED_BATCH`` texts, one batch after another, under the lock that
serialises every decision-model request to the same Ollama server
(``core.decisions.server_lock``): a local embedding call is never concurrent
with a decision call. Vectors are cached in the run's store by (model digest or
name, sha256 of the text) (``ResearchStore.cached_embeddings``), so a rerun, a
frozen arm over the same store or a shared cache does not embed a text twice.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Protocol

import httpx

from andamentum.core.decisions import find_digest, server_lock
from andamentum.core.embeddings import (
    DEFAULT_OLLAMA_URL,
    OllamaEmbedder,
    cosine_similarity,
)

from .store import ResearchStore
from .types import Candidate, RunStats, SubQuestion

# A candidate's text (title, then abstract or snippet) is cut here before it is
# embedded; the question and sub-questions are far shorter.
DENSE_MAX_CHARS = 1500
# Texts per embedding request (one ``/api/embed`` call each).
EMBED_BATCH = 64

Pair = tuple[str, str]  # (work_id, subquestion_id)


class EmbeddingError(RuntimeError):
    """The embedding model could not be reached or answered unusably."""


class Embedder(Protocol):
    """One embedding model. ``embed`` sends ONE request for all *texts* (the
    caller batches) and returns one vector per text, in order. ``name`` is the
    model; ``digest`` identifies its weights when the server says (the cache
    key prefers it)."""

    name: str
    digest: str | None

    async def embed(self, texts: list[str]) -> list[list[float]]: ...


def _plain_name(name: str) -> str:
    return name.removeprefix("ollama:")


class OllamaDenseEmbedder:
    """A local Ollama embedding model behind ``core.embeddings.OllamaEmbedder``,
    every request under the server's decision-model lock."""

    def __init__(
        self, *, name: str, digest: str | None, base_url: str = DEFAULT_OLLAMA_URL
    ) -> None:
        self.name = _plain_name(name)
        self.digest = digest
        self.base_url = base_url
        self._client = OllamaEmbedder(
            model=self.name, base_url=base_url, input_budget_chars=DENSE_MAX_CHARS
        )

    @classmethod
    async def open(
        cls, name: str, *, base_url: str = DEFAULT_OLLAMA_URL
    ) -> OllamaDenseEmbedder:
        """The model *name* on the server at *base_url*, its digest read from
        ``/api/tags``. Raises ``EmbeddingError`` when the server cannot be
        reached or does not list the model (with the ``ollama pull`` fix)."""
        plain = _plain_name(name)
        try:
            async with server_lock(base_url), httpx.AsyncClient(timeout=10.0) as http:
                response = await http.get(f"{base_url.rstrip('/')}/api/tags")
                response.raise_for_status()
                listed = response.json().get("models")
        except (httpx.HTTPError, ValueError) as error:
            raise EmbeddingError(
                f"cannot reach Ollama at {base_url} for the embedding model "
                f"{plain!r} ({error!r}); is `ollama serve` running?"
            ) from error
        digest = find_digest(listed, plain)
        if digest is None:
            raise EmbeddingError(
                f"Ollama does not list the embedding model {plain!r}; "
                f"run `ollama pull {plain}`"
            )
        return cls(name=plain, digest=digest, base_url=base_url)

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if len(texts) > EMBED_BATCH:
            raise ValueError(f"at most {EMBED_BATCH} texts a request, not {len(texts)}")
        try:
            async with server_lock(self.base_url):
                return await self._client.embed_batch(texts)
        except (httpx.HTTPError, RuntimeError, KeyError, ValueError) as error:
            raise EmbeddingError(
                f"embedding with {self.name!r} failed: {type(error).__name__}: {error}"
            ) from error

    async def aclose(self) -> None:
        await self._client.close()


def text_sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def cache_model_key(embedder: Embedder) -> str:
    """What the cache files a vector under: the digest when known, else the name."""
    return embedder.digest or embedder.name


def candidate_text(candidate: Candidate) -> str:
    """What is embedded of a candidate: title, then abstract (or snippet)."""
    work = candidate.work
    return f"{work.title}\n{work.abstract or candidate.snippet}".strip()[
        :DENSE_MAX_CHARS
    ]


@dataclass
class EmbedOutcome:
    """Vectors by text sha256, and what it cost: requests sent, texts embedded
    and texts served from the cache."""

    vectors: dict[str, list[float]] = field(default_factory=dict)
    requests: int = 0
    embedded: int = 0
    cache_hits: int = 0
    batches: list[dict[str, float | int]] = field(default_factory=list)


async def embed_cached(
    texts: Sequence[str],
    *,
    embedder: Embedder,
    store: ResearchStore,
    clock: Callable[[], float],
) -> EmbedOutcome:
    """The vector of every text in *texts*, from the store's cache where it
    holds one, else embedded in requests of at most ``EMBED_BATCH`` texts, one
    after another, and cached. Raises ``EmbeddingError``."""
    outcome = EmbedOutcome()
    key = cache_model_key(embedder)
    wanted = {text_sha(t): t for t in texts}
    outcome.vectors.update(store.cached_embeddings(model=key, shas=list(wanted)))
    outcome.cache_hits = len(outcome.vectors)
    missing = [
        (sha, text) for sha, text in wanted.items() if sha not in outcome.vectors
    ]
    for start in range(0, len(missing), EMBED_BATCH):
        batch = missing[start : start + EMBED_BATCH]
        began = clock()
        vectors = await embedder.embed([text for _, text in batch])
        if len(vectors) != len(batch):
            raise EmbeddingError(
                f"{embedder.name!r} returned {len(vectors)} vectors for {len(batch)} texts"
            )
        fresh = {sha: vector for (sha, _), vector in zip(batch, vectors, strict=True)}
        store.cache_embeddings(model=key, vectors=fresh)
        outcome.vectors.update(fresh)
        outcome.requests += 1
        outcome.embedded += len(batch)
        outcome.batches.append({"texts": len(batch), "seconds": clock() - began})
    return outcome


@dataclass
class DenseUpdate:
    """New cosine similarities: per (work, sub-question) against the
    sub-question, per work against the question; and the embedding cost."""

    subquestion: dict[Pair, float] = field(default_factory=dict)
    question: dict[str, float] = field(default_factory=dict)
    embed: EmbedOutcome = field(default_factory=EmbedOutcome)


async def similarities(
    pairs: Sequence[Pair],
    *,
    question: str,
    subquestions: Sequence[SubQuestion],
    candidates: Mapping[str, Candidate],
    known: Mapping[Pair, float],
    embedder: Embedder,
    store: ResearchStore,
    stats: RunStats,
    phase: str,
    clock: Callable[[], float],
) -> DenseUpdate:
    """Cosine similarities for the *pairs* not in *known*: each candidate
    against its sub-question and against the question. The embedding requests
    are counted in *stats* (``embedding_calls``, under *phase* in
    ``step_embedding_calls``; ``embedding_cache_hits``). Raises
    ``EmbeddingError``."""
    texts = {s.id: s.text for s in subquestions}
    # a record with neither title nor abstract has nothing to embed: it gets no
    # dense similarity (and so no place in the dense lists)
    todo = [
        p
        for p in pairs
        if p not in known
        and p[0] in candidates
        and p[1] in texts
        and candidate_text(candidates[p[0]])
    ]
    update = DenseUpdate()
    if not todo:
        return update
    by_work = {w: candidate_text(candidates[w]) for w, _ in todo}
    wanted = [question, *texts.values(), *by_work.values()]
    outcome = await embed_cached(wanted, embedder=embedder, store=store, clock=clock)
    update.embed = outcome
    stats.embedding_calls += outcome.requests
    stats.embedding_cache_hits += outcome.cache_hits
    if outcome.requests:
        key = phase or "other"
        stats.step_embedding_calls[key] = (
            stats.step_embedding_calls.get(key, 0) + outcome.requests
        )
    vector = outcome.vectors
    question_vector = vector[text_sha(question)]
    for work_id, subquestion_id in todo:
        own = vector[text_sha(by_work[work_id])]
        update.subquestion[(work_id, subquestion_id)] = cosine_similarity(
            own, vector[text_sha(texts[subquestion_id])]
        )
        update.question[work_id] = cosine_similarity(own, question_vector)
    return update
