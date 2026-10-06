"""Cross-encoder reranking and the selection rule that consumes its scores.

Why this exists at all. :func:`~.search.search_unified` ranks documents by a
Reciprocal Rank Fusion sum. An RRF score is an artefact of the ranks that fed
it: it has no absolute meaning, it is not comparable between two queries, and
it cannot answer "is result 7 actually relevant, or merely seventh". So every
caller of :func:`~.public.search` has to guess a ``limit`` and hope. A
cross-encoder score can answer that question, which turns "how many results"
from a constant the caller picked into a property of the query's own evidence.

Why a cross-encoder rather than the embeddings already in the store. A
bi-encoder embeds query and passage separately and compares two vectors. Its
training objective only ever optimises *relative* ranking, so its cosine
similarities are not calibrated across queries, and embedding spaces are
anisotropic — unrelated pairs sit at surprisingly high cosine similarity. Feed
query and passage through one model *together*, with cross-attention over both,
and the score becomes discriminative instead: measured on a real corpus,
mastheads and copyright notices scored 0.0000–0.0003 while topically relevant
passages scored 0.02 and up.

Two functions, split on purpose:

- :func:`rerank` needs the model. It is the only thing here that imports
  ``torch`` and ``transformers`` and it does so lazily, inside the call.
- :func:`select_relevant` needs nothing but arithmetic. It is importable, and
  testable, on an install with no ``rerank`` extra — which matters, because it
  is the half that decides what a caller actually sees.

**Nothing in the store calls either of these.** :func:`~.public.search` does
not rerank and must not start: it is called by CLIs and by pipelines that run
it many times per task, and a default rerank would silently charge every one of
them a 0.6B model's import, its multi-second load, and its resident memory —
and would return different results depending on whether an optional extra
happened to be installed. Reranking is a step a caller takes deliberately,
after search, on the candidates it chose.

Cost, stated plainly. The model is resident in this process for its lifetime,
which is real memory on a unified-memory machine and is a separate budget from
the Ollama-served models (nothing in ``ollama ps`` accounts for it). Scoring is
compute-bound and scales with the number of candidates. andamentum resolves
torch from PyTorch's CPU index, so outside macOS this runs on CPU.

Fixed input shapes, by construction. On Apple silicon, PyTorch's GPU backend
keeps a compiled graph for every distinct input shape it sees and never frees
it (pytorch/pytorch#181213, open at the time of writing; ``torch.mps.empty_cache``
does not reach that memory). The usual way to run a cross-encoder pads each
batch to its own longest pair, so almost every batch has a new shape and a long
run grows without bound: measured here, about 0.5 GB per question until the
process hit the device limit, while the same 40 questions at one fixed shape
held a flat 5.9 GB footprint. So this module never sends a shape the model has
not seen before: every pair is padded to the smallest of a fixed set of length
buckets, and every batch is filled to that bucket's fixed size. The set of
shapes is finite and known before the first call, which bounds the memory on
every backend. Padding is masked out by the attention mask, so a pair scores
the same as it would padded to its own length.
"""

from __future__ import annotations

import asyncio
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

_EXTRA_HINT = (
    "Cross-encoder reranking needs the 'rerank' extra. "
    "Install it with: uv sync --extra rerank  (or: pip install 'andamentum[rerank]')"
)

#: Pair lengths are padded up to one of these token counts (each capped at the
#: model's own maximum, which is always the last bucket). Powers of two, so a
#: pair never carries more than twice its own length in padding.
_LENGTH_BUCKETS = (64, 128, 256, 512, 1024, 2048, 4096, 8192)

#: Tokens per batch. A bucket's batch size is this divided by its length, so
#: every batch does about the same work and holds about the same activations:
#: 128 pairs at 64 tokens, 16 at 512, 1 at 8192.
_TOKENS_PER_BATCH = 8192


class _Scorer(Protocol):
    """What :func:`_score` needs from a model. :class:`_CrossEncoder` is the
    real one; tests pass a stub, so the batching logic runs without torch."""

    buckets: tuple[int, ...]

    def token_lengths(self, query: str, texts: Sequence[str]) -> list[int]: ...

    def score_batch(
        self, query: str, texts: Sequence[str], *, length: int
    ) -> list[float]: ...


@dataclass(frozen=True)
class _Batch:
    length: int  # every pair in the batch is padded to exactly this many tokens
    size: int  # the batch always holds exactly this many pairs
    indices: list[int]  # positions of the real pairs; the rest repeat the last one


def _buckets_for(max_length: int) -> tuple[int, ...]:
    """The length buckets for a model whose input limit is ``max_length``."""
    if max_length < 1:
        raise ValueError(f"model input limit must be positive, got {max_length}")
    return tuple(b for b in _LENGTH_BUCKETS if b < max_length) + (max_length,)


def _batch_size(length: int) -> int:
    return max(1, _TOKENS_PER_BATCH // length)


def _allowed_shapes(buckets: Sequence[int]) -> set[tuple[int, int]]:
    """Every (batch size, length) a model with these buckets can ever receive."""
    return {(_batch_size(b), b) for b in buckets}


def _plan_batches(lengths: Sequence[int], *, buckets: Sequence[int]) -> list[_Batch]:
    """Group pairs into fixed-shape batches by the smallest bucket that fits.

    ``lengths`` are token counts after truncation, so none exceeds the last
    bucket. Raises rather than invent a new shape if one somehow does.
    """
    groups: dict[int, list[int]] = {}
    for i, n in enumerate(lengths):
        bucket = next((b for b in buckets if n <= b), None)
        if bucket is None:
            raise ValueError(
                f"pair of {n} tokens exceeds the model limit {buckets[-1]}; "
                "the tokenizer should have truncated it"
            )
        groups.setdefault(bucket, []).append(i)
    batches: list[_Batch] = []
    for bucket in sorted(groups):
        size = _batch_size(bucket)
        members = groups[bucket]
        for start in range(0, len(members), size):
            batches.append(_Batch(bucket, size, members[start : start + size]))
    return batches


def _score(model: _Scorer, query: str, texts: Sequence[str]) -> list[float]:
    """Scores for ``texts`` in input order, computed only at fixed shapes."""
    lengths = model.token_lengths(query, texts)
    if len(lengths) != len(texts):
        raise RuntimeError("tokenizer returned a different number of pairs")
    scores: list[float | None] = [None] * len(texts)
    for batch in _plan_batches(lengths, buckets=model.buckets):
        # Fill a short batch with copies of its last real pair, so its shape is
        # the bucket's fixed shape; their scores are discarded below.
        rows = batch.indices + [batch.indices[-1]] * (batch.size - len(batch.indices))
        out = model.score_batch(query, [texts[i] for i in rows], length=batch.length)
        if len(out) != batch.size:
            raise RuntimeError(
                f"model returned {len(out)} scores for {batch.size} pairs"
            )
        for i, value in zip(batch.indices, out):
            scores[i] = value
    if any(v is None for v in scores):  # pragma: no cover - guarded by the planner
        raise RuntimeError("a pair was left unscored")
    return [float(v) for v in scores if v is not None]


class _CrossEncoder:
    """A single-logit sequence-classification model that only sees fixed shapes.

    Loaded with ``transformers`` directly rather than through a wrapper that
    pads each batch to its own longest pair, because the padding is exactly
    what this class has to control. Scores are ``sigmoid(logit)``: the selection
    rule below applies an absolute floor, and a floor is only meaningful on a
    bounded scale.
    """

    def __init__(self, model_name: str) -> None:
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - depends on install shape
            raise ImportError(f"{_EXTRA_HINT} ({exc})") from exc

        self._torch = torch
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.tokenizer: Any = AutoTokenizer.from_pretrained(model_name)
        net: Any = AutoModelForSequenceClassification.from_pretrained(model_name)
        if net.config.num_labels != 1:
            raise ValueError(
                f"{model_name} has {net.config.num_labels} output labels; "
                "reranking needs a cross-encoder with a single relevance logit"
            )
        self.net = net.to(self.device).eval()
        self.max_length = self._input_limit(model_name)
        self.buckets = _buckets_for(self.max_length)

    def _input_limit(self, model_name: str) -> int:
        """The model's token limit, from the tokenizer, checked against the model."""
        limit = int(self.tokenizer.model_max_length)
        positions = getattr(self.net.config, "max_position_embeddings", None)
        if limit > 1_000_000:  # transformers' sentinel for "not recorded"
            if positions is None:
                raise ValueError(f"{model_name} records no input length limit")
            limit = int(positions)
        if positions is not None and limit > positions:
            raise ValueError(
                f"{model_name}: tokenizer limit {limit} exceeds the model's "
                f"{positions} position embeddings"
            )
        return limit

    def token_lengths(self, query: str, texts: Sequence[str]) -> list[int]:
        encoded = self.tokenizer(
            [query] * len(texts),
            list(texts),
            truncation=True,
            max_length=self.max_length,
        )
        return [len(ids) for ids in encoded["input_ids"]]

    def score_batch(
        self, query: str, texts: Sequence[str], *, length: int
    ) -> list[float]:
        torch = self._torch
        # A pair's truncated length never exceeds its bucket (the planner put it
        # there), so truncating at ``length`` cuts exactly what token_lengths
        # cut, and padding="max_length" pads every row to the bucket.
        features = self.tokenizer(
            [query] * len(texts),
            list(texts),
            truncation=True,
            max_length=length,
            padding="max_length",
            return_tensors="pt",
        )
        shape = tuple(features["input_ids"].shape)
        if shape != (len(texts), length):
            raise RuntimeError(
                f"tokenizer produced shape {shape}, expected {(len(texts), length)}"
            )
        with torch.inference_mode():
            logits = self.net(
                **{k: v.to(self.device) for k, v in features.items()}
            ).logits
        return torch.sigmoid(logits[:, 0].float()).cpu().tolist()


# Keyed by model name so switching models inside one process reloads rather
# than silently scoring with the previous one. Loading weights costs tens of
# seconds, which is worth paying once per process and not once per call.
_model: _CrossEncoder | None = None
_model_name: str | None = None


def _reranker(model_name: str) -> _CrossEncoder:
    """The process-lifetime cross-encoder for ``model_name``."""
    global _model, _model_name
    if _model is not None and _model_name == model_name:
        return _model
    _model = _CrossEncoder(model_name)
    _model_name = model_name
    return _model


def _release_gpu_cache() -> None:
    """Return the tensor memory PyTorch cached for the last call to the system.

    This frees cached *tensor* buffers only. It does not free the per-shape
    graph memory described in the module docstring; the fixed shapes are what
    bound that. Looks torch up rather than importing it: a model that ran has
    already imported it, and a stub model must not pull torch in.
    """
    torch = sys.modules.get("torch")
    if torch is not None and torch.backends.mps.is_available():
        torch.mps.empty_cache()


def _predict(model: _Scorer, query: str, texts: Sequence[str]) -> list[float]:
    try:
        return _score(model, query, texts)
    finally:
        _release_gpu_cache()


def reset_reranker_cache() -> None:
    """Drop the cached model, releasing its memory.

    For a long-lived process that has finished reranking and wants the
    memory back, and for tests that must not inherit another test's model.
    """
    global _model, _model_name
    _model = None
    _model_name = None


async def rerank(
    query: str,
    texts: Sequence[str],
    *,
    reranker_model: str,
) -> list[float]:
    """Score each of ``texts`` against ``query`` with a cross-encoder.

    Returns one score per text, **in the order the texts were given** — never
    sorted, never filtered. The caller owns its own objects and zips the scores
    back onto them, so this function never has to know what a candidate is; it
    works equally on store chunks, on web passages, or on anything else with
    text. Pair it with :func:`select_relevant` to turn the scores into a
    decision.

    ``reranker_model`` is any Hugging Face cross-encoder id with a single
    relevance logit (a sequence-classification model with ``num_labels=1``).
    There is no default: which model produced a score is part of what the score
    means, and the absolute floor a caller passes to :func:`select_relevant` is
    calibrated against one specific model. See :data:`SUGGESTED_RERANKER_MODEL`
    for a starting point that is a suggestion and not a fallback.

    Raises:
        ImportError: if the ``rerank`` extra is not installed.
    """
    if not texts:
        return []
    model = _reranker(reranker_model)
    # Compute-bound, not I/O-bound. Off the event loop thread so it does not
    # block whatever else the process is running concurrently.
    return await asyncio.to_thread(_predict, model, query, list(texts))


# A starting point, not a default: no function here reads it. BAAI/bge-reranker-v2-m3
# is Apache-2.0, 0.6B parameters, and the strongest BEIR score among small local
# cross-encoders at evaluation time (~0.66, against ~0.58 for
# ms-marco-MiniLM-L-12-v2). A caller who changes it must re-calibrate the
# ``min_relevance`` floor, because that floor is a property of the model's scale.
SUGGESTED_RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"


# How far the curve must depart from its own chord, in normalised units, before
# the farthest point counts as a break rather than as rounding noise. Both axes
# are scaled to [0, 1], so this reads as "the bend must be at least 1% of the
# unit square". Small enough that a genuinely S-shaped curve clears it by more
# than an order of magnitude; large enough that an evenly declining curve does
# not.
_MIN_ELBOW_DEPARTURE = 0.01


def _elbow_index(scores: Sequence[float]) -> int | None:
    """Index of the natural break in a descending-sorted score curve.

    Geometric knee-finding: normalise both axes to [0, 1], then take the point
    at the greatest perpendicular distance from the straight line joining the
    curve's first and last points. A Kneedle-style fit is the standard tool
    here and was tried first, but it assumes one consistent curve direction
    throughout; a real reranker score curve is S-shaped (a confident plateau, a
    steep drop, a noise-floor plateau) and Kneedle reported a degenerate knee at
    index 0 on real data. Greatest-departure-from-the-chord does not assume
    which way the curve bends, only where it leaves the straight line.

    Returns ``None`` when there is no bend to find — fewer than three points,
    every score identical, or a curve so close to straight that the farthest
    point is noise rather than a break. The caller keeps everything in that
    case; an arbitrary index is worse than no index.

    That last case is not hypothetical, and it is why
    :data:`_MIN_ELBOW_DEPARTURE` exists. On a straight descending line every
    point is equidistant from the chord, ``max`` picks the first by tie order,
    and the result is a confident-looking "cut after the top result" that is
    really an argmax over floating-point dust. Scores that decline evenly are a
    real and common shape — a corpus where everything is somewhat relevant
    produces one — and the honest answer there is that the curve has no break,
    not that the break is at index 0.
    """
    n = len(scores)
    if n <= 2:
        return None
    lowest = min(scores)
    highest = max(scores)
    span = highest - lowest
    if span < 1e-12:
        return None

    # Normalised coordinates: x evenly spaced over [0, 1], y scaled to [0, 1].
    xs = [i / (n - 1) for i in range(n)]
    ys = [(score - lowest) / span for score in scores]
    x0, y0 = xs[0], ys[0]
    x1, y1 = xs[-1], ys[-1]
    chord = ((y1 - y0) ** 2 + (x1 - x0) ** 2) ** 0.5
    if chord < 1e-12:
        return None

    best_index = 0
    best_distance = -1.0
    for i, (x, y) in enumerate(zip(xs, ys, strict=True)):
        distance = abs((y1 - y0) * x - (x1 - x0) * y + x1 * y0 - y1 * x0) / chord
        if distance > best_distance:
            best_distance = distance
            best_index = i
    if best_distance < _MIN_ELBOW_DEPARTURE:
        return None
    return best_index


def select_relevant[T](
    scored: Sequence[tuple[T, float]],
    *,
    min_relevance: float,
    buffer: int = 0,
) -> list[T]:
    """Which of ``scored`` are relevant enough to act on, most relevant first.

    Generic over the item type: this is selection over precomputed
    ``(item, score)`` pairs and knows nothing about chunks. It imports nothing
    optional, so it works without the ``rerank`` extra.

    Two independent checks, and **both are required**:

    - The elbow decides *how many*: the natural break in this query's own
      sorted score curve, plus ``buffer`` extra items so a genuinely relevant
      item that landed just past the boundary is not lost to noise at the edge.
      The elbow errs generous by design, and by up to one item structurally:
      the geometric method finds the *corner* of the curve, and whether the
      corner is the last good item or the first bad one depends on how long
      each plateau is — an index alone cannot say which. So a clean
      five-good-then-thirty-bad curve keeps six, not five. Do not tighten this
      into a guess about which side of the corner you are on; the floor below
      is what removes the extra item, and it removes it on evidence.
    - ``min_relevance`` is an absolute floor applied regardless of what the
      elbow found. It is not redundant, it is the safety: normalising a set of
      near-identical noise scores to [0, 1] manufactures visible structure out
      of nothing. Measured directly — 50 synthetic near-zero noise scores
      produced a confident-looking elbow that kept 49 of them. The floor is the
      only thing that excludes that case.

    ``min_relevance`` is required and has no default, because the right value
    is a property of the reranker model and of the corpus, not of this library.
    A number calibrated on one corpus (0.01, against a noise floor of 0.0003)
    would be a domain assumption wearing the clothes of a constant. Calibrate it
    by scoring real candidates for a real query and looking at where the tail is.
    """
    if buffer < 0:
        raise ValueError(f"buffer must be >= 0, got {buffer}")
    ranked = sorted(scored, key=lambda pair: pair[1], reverse=True)
    scores = [score for _, score in ranked]
    elbow = _elbow_index(scores)
    cutoff = len(ranked) if elbow is None else elbow + 1 + buffer
    return [item for item, score in ranked[:cutoff] if score >= min_relevance]
