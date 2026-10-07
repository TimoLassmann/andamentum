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
  ``sentence_transformers`` and it does so lazily, inside the call.
- :func:`select_relevant` needs nothing but arithmetic, so it is testable
  without loading a model.

**Where the store calls them.** :func:`~.public.search` calls :func:`rerank`
on its fused top candidates, scoring each document by its best passage, so
every search pays for the model. That was a measured choice: on a multi-hop
benchmark the rerank gave the largest single gain of anything tried.
:func:`select_relevant` is never called by the store; deciding how many
results are relevant stays with the caller.

Cost, stated plainly. The model is resident in this process for its lifetime,
which is real memory on a unified-memory machine and is a separate budget from
the Ollama-served models (nothing in ``ollama ps`` accounts for it). Scoring is
compute-bound and scales with the number of candidates. andamentum resolves
torch from PyTorch's CPU index, so outside macOS this runs on CPU.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    from sentence_transformers import CrossEncoder

# Keyed by model name so switching models inside one process reloads rather
# than silently scoring with the previous one. Loading weights costs tens of
# seconds, which is worth paying once per process and not once per call.
_model: CrossEncoder | None = None
_model_name: str | None = None


def _reranker(model_name: str) -> CrossEncoder:
    """The process-lifetime cross-encoder for ``model_name``."""
    global _model, _model_name
    if _model is not None and _model_name == model_name:
        return _model

    # Imported here, not at module level: torch and the model load only when a
    # search first reranks, never on ``import andamentum.document_store``.
    import torch
    from sentence_transformers import CrossEncoder

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    # Sigmoid, not the model's raw logits: the selection rule below applies an
    # absolute floor, and a floor is only meaningful on a bounded scale.
    _model = CrossEncoder(model_name, device=device, activation_fn=torch.nn.Sigmoid())
    _model_name = model_name
    return _model


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

    ``reranker_model`` is any ``sentence_transformers`` cross-encoder id. There
    is no default: which model produced a score is part of what the score means,
    and the absolute floor a caller passes to :func:`select_relevant` is
    calibrated against one specific model. See :data:`SUGGESTED_RERANKER_MODEL`
    for a starting point that is a suggestion and not a fallback.

    """
    if not texts:
        return []
    model = _reranker(reranker_model)
    pairs = [(query, text) for text in texts]
    # predict() is compute-bound, not I/O-bound. Off the event loop thread so
    # it does not block whatever else the process is running concurrently.
    scores = await asyncio.to_thread(model.predict, pairs, show_progress_bar=False)
    return [float(score) for score in scores]


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
    ``(item, score)`` pairs and knows nothing about chunks or models.

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
