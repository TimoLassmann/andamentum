"""Tests for cross-encoder reranking and the selection rule.

The selection rule is tested exhaustively and with no model, because it is the
half that decides what a caller sees. :func:`rerank` itself is tested against a stubbed
cross-encoder — a real one downloads ~2 GB of weights, which the default suite
must never do.
"""

from __future__ import annotations

import random

import pytest

from andamentum.document_store import reranking as rerank_module
from andamentum.document_store.reranking import (
    SUGGESTED_RERANKER_MODEL,
    _elbow_index,
    rerank,
    reset_reranker_cache,
    select_relevant,
)


@pytest.fixture(autouse=True)
def _clean_model_cache():
    """No test inherits another test's cached model."""
    reset_reranker_cache()
    yield
    reset_reranker_cache()


# ── select_relevant ──────────────────────────────────────────────────────


def test_returns_most_relevant_first_regardless_of_input_order():
    scored = [("c", 0.1), ("a", 0.9), ("b", 0.5)]
    assert select_relevant(scored, min_relevance=0.0) == ["a", "b", "c"]


def test_floor_excludes_everything_below_it():
    scored = [("a", 0.9), ("b", 0.005), ("c", 0.001)]
    assert select_relevant(scored, min_relevance=0.01) == ["a"]


def test_floor_alone_can_return_nothing():
    """A query with no real evidence returns an empty list, not a best guess."""
    scored = [("a", 0.0004), ("b", 0.0002), ("c", 0.0001)]
    assert select_relevant(scored, min_relevance=0.01) == []


def test_floor_excludes_a_confident_looking_elbow_over_pure_noise():
    """The regression this floor exists for.

    Normalising near-identical noise to [0, 1] manufactures visible structure,
    so the elbow fires confidently on a corpus that holds nothing relevant at
    all. The floor is the only thing that excludes that case, which is why the
    two checks are independent and both required.
    """
    rng = random.Random(1234)
    scores = sorted((0.0001 + rng.random() * 0.0004 for _ in range(50)), reverse=True)
    noise = [(f"chunk-{i}", score) for i, score in enumerate(scores)]

    elbow = _elbow_index(scores)
    assert elbow is not None, "the elbow DOES fire on noise — that is the point"
    assert len(select_relevant(noise, min_relevance=0.0)) < len(noise), (
        "and it keeps a confident-looking subset"
    )

    assert select_relevant(noise, min_relevance=0.01) == []


def test_elbow_cuts_the_tail_off_an_s_shaped_curve():
    """A plateau, a steep drop, a noise floor — the real shape of a score curve.

    Five good items and thirty bad ones, and the elbow keeps six: the geometric
    method finds the corner, and here the corner is the first bad item rather
    than the last good one. That single extra is structural and documented, not
    a bug to tune away — removing it is the floor's job, and the floor does it
    on evidence rather than on a guess about which side of the corner it is on.
    """
    scored = [
        *[(f"top-{i}", 0.9 - i * 0.01) for i in range(5)],
        *[(f"tail-{i}", 0.02 - i * 0.0001) for i in range(30)],
    ]
    kept = select_relevant(scored, min_relevance=0.0)
    assert len(kept) == 6
    assert kept[:5] == [f"top-{i}" for i in range(5)]

    # The floor removes the extra, and the tail with it.
    assert select_relevant(scored, min_relevance=0.5) == [f"top-{i}" for i in range(5)]


def test_buffer_keeps_extra_items_past_the_break():
    scored = [
        *[(f"top-{i}", 0.9 - i * 0.01) for i in range(5)],
        *[(f"tail-{i}", 0.02 - i * 0.0001) for i in range(30)],
    ]
    base = select_relevant(scored, min_relevance=0.0)
    kept = select_relevant(scored, min_relevance=0.0, buffer=3)
    assert len(kept) == len(base) + 3
    assert kept[: len(base)] == base


def test_buffer_defaults_to_none_extra():
    scored = [("a", 0.9), ("b", 0.88), ("c", 0.02), ("d", 0.019)]
    assert select_relevant(scored, min_relevance=0.0) == select_relevant(
        scored, min_relevance=0.0, buffer=0
    )


def test_negative_buffer_is_rejected():
    with pytest.raises(ValueError, match="buffer must be >= 0"):
        select_relevant([("a", 1.0)], min_relevance=0.0, buffer=-1)


def test_empty_input_returns_empty():
    assert select_relevant([], min_relevance=0.01) == []


def test_single_item_is_kept_when_it_clears_the_floor():
    assert select_relevant([("a", 0.5)], min_relevance=0.01) == ["a"]
    assert select_relevant([("a", 0.005)], min_relevance=0.01) == []


def test_identical_scores_keep_everything_above_the_floor():
    """No bend to find, so the elbow abstains and the floor decides alone."""
    scored = [(f"c{i}", 0.4) for i in range(10)]
    assert len(select_relevant(scored, min_relevance=0.01)) == 10


def test_selection_is_generic_over_the_item_type():
    """Nothing here knows what a chunk is."""

    class Passage:
        def __init__(self, name: str) -> None:
            self.name = name

    a, b = Passage("a"), Passage("b")
    assert select_relevant([(a, 0.9), (b, 0.05)], min_relevance=0.5) == [a]


# ── _elbow_index ─────────────────────────────────────────────────────────


def test_elbow_abstains_on_too_few_points():
    assert _elbow_index([]) is None
    assert _elbow_index([1.0]) is None
    assert _elbow_index([1.0, 0.5]) is None


def test_elbow_abstains_when_every_score_is_identical():
    assert _elbow_index([0.4] * 10) is None


def test_elbow_finds_the_break_in_a_step_curve():
    """The index is the corner — here the first item of the low plateau."""
    scores = [0.9, 0.89, 0.88, 0.02, 0.019, 0.018]
    assert _elbow_index(scores) == 3


def test_elbow_abstains_on_an_evenly_declining_curve():
    """A straight line has no corner, and index 0 would be a lie.

    Every point on a straight line is equidistant from its own chord, so an
    unguarded argmax returns 0 and the caller cuts to a single result. Scores
    that decline evenly are a real shape — a corpus where everything is
    somewhat relevant produces one — and the honest answer is "no break".
    """
    assert _elbow_index([0.9 - i * 0.02 for i in range(10)]) is None


def test_evenly_declining_scores_keep_everything_above_the_floor():
    scored = [(f"c{i}", 0.9 - i * 0.02) for i in range(10)]
    assert len(select_relevant(scored, min_relevance=0.0)) == 10


# ── rerank ───────────────────────────────────────────────────────────────


class _StubCrossEncoder:
    """Scores by how many query words a text contains. Deterministic, no weights.

    Implements the same two methods the real model exposes, and records the
    shape of every batch it is asked to score.
    """

    buckets = rerank_module._buckets_for(512)

    def __init__(self) -> None:
        self.shapes: list[tuple[int, int]] = []

    def token_lengths(self, query, texts):  # noqa: ANN001
        # Two special tokens plus one per word, truncated like a tokenizer would.
        return [min(2 + len(query.split()) + len(t.split()), 512) for t in texts]

    def score_batch(self, query, texts, *, length):  # noqa: ANN001
        self.shapes.append((len(texts), length))
        words = set(query.lower().split())
        return [
            sum(1 for word in text.lower().split() if word in words) / 10.0
            for text in texts
        ]


@pytest.fixture
def stub_model(monkeypatch: pytest.MonkeyPatch) -> _StubCrossEncoder:
    stub = _StubCrossEncoder()
    monkeypatch.setattr(rerank_module, "_reranker", lambda _name: stub)
    return stub


async def test_rerank_returns_one_score_per_text_in_input_order(stub_model):
    texts = ["nothing at all", "the query words here", "query"]
    scores = await rerank("query words", texts, reranker_model="stub")
    assert len(scores) == len(texts)
    assert scores[0] < scores[2] < scores[1]


async def test_rerank_never_sorts_or_filters(stub_model):
    """The order contract: the caller zips scores back onto its own objects."""
    texts = ["query query query", "irrelevant", "query"]
    scores = await rerank("query", texts, reranker_model="stub")
    assert scores == [0.3, 0.0, 0.1]


async def test_rerank_on_empty_input_never_loads_a_model(stub_model):
    assert await rerank("anything", [], reranker_model="stub") == []
    assert stub_model.shapes == []


async def test_rerank_returns_plain_floats(stub_model):
    scores = await rerank("query", ["query"], reranker_model="stub")
    assert all(type(score) is float for score in scores)


async def test_rerank_and_select_compose(stub_model):
    chunks = ["mitochondrial dna repair", "copyright notice", "dna repair mechanism"]
    scores = await rerank("dna repair", chunks, reranker_model="stub")
    kept = select_relevant(
        list(zip(chunks, scores, strict=True)), min_relevance=0.05, buffer=0
    )
    assert "copyright notice" not in kept
    assert kept[0] in {"mitochondrial dna repair", "dna repair mechanism"}


async def test_rerank_releases_the_gpu_cache_after_every_call(stub_model, monkeypatch):
    """The cached tensor buffers go back after each call (the shape-keyed graph
    memory is bounded separately, by the fixed shapes tested below)."""
    released: list[int] = []
    monkeypatch.setattr(rerank_module, "_release_gpu_cache", lambda: released.append(1))
    await rerank("q", ["a", "b"], reranker_model="stub")
    await rerank("q", ["c"], reranker_model="stub")
    assert len(released) == 2


async def test_rerank_releases_the_gpu_cache_even_when_predict_fails(monkeypatch):
    class Broken(_StubCrossEncoder):
        def score_batch(self, query, texts, *, length):  # noqa: ANN001
            raise RuntimeError("MPS backend out of memory")

    released: list[int] = []
    monkeypatch.setattr(rerank_module, "_reranker", lambda _name: Broken())
    monkeypatch.setattr(rerank_module, "_release_gpu_cache", lambda: released.append(1))
    with pytest.raises(RuntimeError, match="out of memory"):
        await rerank("q", ["a"], reranker_model="stub")
    assert released == [1]


# ── fixed input shapes (pytorch/pytorch#181213) ──────────────────────────


def test_buckets_end_at_the_model_limit():
    assert rerank_module._buckets_for(512) == (64, 128, 256, 512)
    assert rerank_module._buckets_for(300) == (64, 128, 256, 300)
    assert rerank_module._buckets_for(8192)[-1] == 8192
    assert rerank_module._buckets_for(40) == (40,)
    with pytest.raises(ValueError):
        rerank_module._buckets_for(0)


def test_planner_uses_only_the_allowed_shapes_and_covers_every_pair_once():
    buckets = rerank_module._buckets_for(8192)
    allowed = rerank_module._allowed_shapes(buckets)
    rng = random.Random(7)
    for _ in range(200):
        lengths = [rng.randint(1, 8192) for _ in range(rng.randint(1, 300))]
        batches = rerank_module._plan_batches(lengths, buckets=buckets)
        assert {(b.size, b.length) for b in batches} <= allowed
        covered = sorted(i for b in batches for i in b.indices)
        assert covered == list(range(len(lengths)))
        for b in batches:
            assert 1 <= len(b.indices) <= b.size
            assert all(lengths[i] <= b.length for i in b.indices)
            smaller = [x for x in buckets if x < b.length]
            assert all(lengths[i] > smaller[-1] for i in b.indices) if smaller else True


def test_planner_refuses_a_pair_longer_than_the_model_limit():
    with pytest.raises(ValueError, match="exceeds the model limit"):
        rerank_module._plan_batches([10, 513], buckets=(64, 512))


async def test_many_calls_of_varying_length_only_ever_send_the_fixed_shapes(stub_model):
    """The regression that matters: shapes must not grow with the input."""
    rng = random.Random(11)
    words = ["query", "other", "text", "word"]
    for _ in range(300):
        texts = [
            " ".join(rng.choice(words) for _ in range(rng.randint(1, 600)))
            for _ in range(rng.randint(1, 60))
        ]
        await rerank("query words", texts, reranker_model="stub")
    seen = set(stub_model.shapes)
    assert seen <= rerank_module._allowed_shapes(stub_model.buckets)
    assert len(seen) <= len(stub_model.buckets)


async def test_filler_rows_are_scored_but_never_returned(stub_model):
    texts = [
        "query",
        "nothing",
        "query query",
    ]  # one short batch of 3 in a 128-row shape
    scores = await rerank("query", texts, reranker_model="stub")
    assert scores == [0.1, 0.0, 0.2]
    assert stub_model.shapes == [(128, 64)]


async def test_a_wrong_number_of_scores_is_an_error_not_a_misalignment(monkeypatch):
    class Short(_StubCrossEncoder):
        def score_batch(self, query, texts, *, length):  # noqa: ANN001
            return [0.5] * (len(texts) - 1)

    monkeypatch.setattr(rerank_module, "_reranker", lambda _name: Short())
    monkeypatch.setattr(rerank_module, "_release_gpu_cache", lambda: None)
    with pytest.raises(RuntimeError, match="scores for"):
        await rerank("q", ["a", "b"], reranker_model="stub")


def test_suggested_model_is_a_suggestion_not_a_default():
    """No function may fill in a reranker model on the caller's behalf.

    A score's meaning and the calibrated floor both belong to one specific
    model, so a hidden default would silently change what a caller's
    ``min_relevance`` means.
    """
    import inspect

    signature = inspect.signature(rerank)
    assert signature.parameters["reranker_model"].default is inspect.Parameter.empty
    assert signature.parameters["reranker_model"].kind is inspect.Parameter.KEYWORD_ONLY


def test_min_relevance_has_no_default():
    """The floor is corpus- and model-specific; the library must not guess it."""
    import inspect

    signature = inspect.signature(select_relevant)
    assert signature.parameters["min_relevance"].default is inspect.Parameter.empty


# ── the real model ───────────────────────────────────────────────────────


@pytest.mark.rerank
async def test_a_real_cross_encoder_separates_junk_from_relevant_content():
    """The claim this whole module rests on, checked against real weights.

    Deselected by default: it downloads ~2 GB. Run it with
    ``uv run pytest -m rerank``. Everything else here stubs the model, so this
    is the only test that would catch a sentence-transformers API change or a
    model that no longer discriminates.

    The assertion is deliberately about the *gap*, not about absolute numbers.
    Junk and relevant content must land orders of magnitude apart — that
    separation is what makes an absolute floor meaningful at all, and it is the
    property a bi-encoder's cosine similarity does not have.
    """
    query = "What is the role of mitochondrial dysfunction in muscle wasting?"
    junk = "Copyright 2026. All rights reserved. Reproduction prohibited."
    relevant = (
        "Impaired mitochondrial respiration in skeletal muscle drives "
        "proteolysis and accelerates muscle wasting in cachexia."
    )

    scores = await rerank(
        query, [junk, relevant], reranker_model=SUGGESTED_RERANKER_MODEL
    )

    junk_score, relevant_score = scores
    assert relevant_score > junk_score * 10, (junk_score, relevant_score)
    assert select_relevant(
        [("junk", junk_score), ("relevant", relevant_score)],
        min_relevance=0.01,
    ) == ["relevant"]


def _varied_texts(rng: random.Random, n: int, *, max_words: int) -> list[str]:
    words = "the muscle cell protein energy repair signal pathway loss of mass".split()
    return [
        " ".join(rng.choice(words) for _ in range(rng.randint(1, max_words)))
        for _ in range(n)
    ]


@pytest.mark.rerank
async def test_fixed_shape_scores_equal_the_library_cross_encoder():
    """Padding to a bucket must not change a score: the attention mask hides it.

    Compares against sentence-transformers' CrossEncoder, which pads each batch
    to its own longest pair, on pairs that straddle several bucket borders.
    """
    import torch
    from sentence_transformers import CrossEncoder

    query = "What is the role of mitochondrial dysfunction in muscle wasting?"
    texts = _varied_texts(random.Random(3), 60, max_words=700)
    ours = await rerank(query, texts, reranker_model=SUGGESTED_RERANKER_MODEL)

    model = rerank_module._reranker(SUGGESTED_RERANKER_MODEL)
    library = CrossEncoder(
        SUGGESTED_RERANKER_MODEL, device=model.device, activation_fn=torch.nn.Sigmoid()
    )
    theirs = library.predict([(query, t) for t in texts], show_progress_bar=False)
    lengths = model.token_lengths(query, texts)
    assert (
        len(
            {
                b.length
                for b in rerank_module._plan_batches(lengths, buckets=model.buckets)
            }
        )
        >= 3
    )
    worst = max(abs(a - float(b)) for a, b in zip(ours, theirs, strict=True))
    assert worst < 1e-4, worst


@pytest.mark.rerank
async def test_gpu_memory_stays_flat_over_many_calls_of_varying_length():
    """The regression for pytorch/pytorch#181213 on the real model and device.

    Before fixed shapes, each question of varying length added about 0.5 GB of
    GPU driver memory that empty_cache could not return.
    """
    import torch

    if not torch.backends.mps.is_available():
        pytest.skip("the shape-keyed graph cache is an Apple GPU (MPS) issue")
    rng = random.Random(5)
    query = "What is the role of mitochondrial dysfunction in muscle wasting?"
    model = rerank_module._reranker(SUGGESTED_RERANKER_MODEL)
    # Warm up: one call per bucket up to 1024 tokens, so every shape used below
    # has been compiled once.
    for bucket in [b for b in model.buckets if b <= 1024]:
        await rerank(
            query, ["word " * (bucket - 30)], reranker_model=SUGGESTED_RERANKER_MODEL
        )
    before = torch.mps.driver_allocated_memory()
    for _ in range(150):
        texts = _varied_texts(rng, rng.randint(1, 50), max_words=600)
        await rerank(query, texts, reranker_model=SUGGESTED_RERANKER_MODEL)
    growth = torch.mps.driver_allocated_memory() - before
    assert growth < 0.5 * 2**30, f"driver memory grew {growth / 2**30:.2f} GB"
