"""Tests for the novelty / prior-work check (Step 10).

Three surfaces:

1. Schemas (NoveltyClaim, NoveltyClaimList) round-trip cleanly.
2. NoveltyCheck node passes through when ``check_novelty=False`` (the
   default), runs the extractor + per-claim deep_research when True.
3. Disk caching avoids redundant deep_research runs across calls.

deep_research itself is stubbed — we don't actually hit the network in
unit tests. The integration with ``deep_research.check_novelty`` is
exercised by patching ``check_novelty`` and ``run_research`` in the
``andamentum.deep_research`` namespace.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest import mock

from andamentum.whetstone.v2.agents.novelty_claim_extractor import (
    NoveltyClaim,
    NoveltyClaimList,
)
from andamentum.whetstone.v2.deps import ReviewDeps
from andamentum.whetstone.v2.nodes.novelty_check import (
    NoveltyCheck,
    _report_to_finding,
)
from andamentum.whetstone.v2.state import ReviewState


# ── Schemas ────────────────────────────────────────────────────────────


def test_novelty_claim_round_trip():
    c = NoveltyClaim(
        claim_text="We present the first method for X.",
        short_summary="first method for X",
        why_load_bearing="core abstract claim",
    )
    dumped = c.model_dump()
    assert dumped["claim_text"].startswith("We present")
    assert NoveltyClaim.model_validate(dumped) == c


def test_novelty_claim_list_default_empty():
    lst = NoveltyClaimList()
    assert lst.claims == []


# ── NoveltyCheck pass-through when off ─────────────────────────────────


async def test_pass_through_when_check_novelty_off():
    """Without state.check_novelty=True, the node returns EditSections
    immediately without running any LLM call."""
    state = ReviewState(source="x", check_novelty=False, markdown="some text")
    deps = ReviewDeps(model="fake:test")

    class FakeCtx:
        state: ReviewState
        deps: ReviewDeps

    ctx = FakeCtx()
    ctx.state = state
    ctx.deps = deps

    next_node = await NoveltyCheck().run(ctx)  # type: ignore[arg-type]
    # Expect an EditSections instance — type checked by pydantic-graph
    assert next_node.__class__.__name__ == "EditSections"
    # No LLM calls when off
    assert state.llm_calls == 0
    # No findings appended
    assert state.findings == []


async def test_pass_through_when_markdown_empty():
    """Even with check_novelty=True, an empty manuscript skips."""
    state = ReviewState(source="x", check_novelty=True, markdown="   ")
    deps = ReviewDeps(model="fake:test")

    class FakeCtx:
        state: ReviewState
        deps: ReviewDeps

    ctx = FakeCtx()
    ctx.state = state
    ctx.deps = deps

    next_node = await NoveltyCheck().run(ctx)  # type: ignore[arg-type]
    assert next_node.__class__.__name__ == "EditSections"


# ── NoveltyCheck full run with mocked agents ───────────────────────────


async def test_extracts_then_calls_deep_research(tmp_path: Path):
    """When the extractor returns 2 claims, the node calls per-claim and
    appends a Finding for each NoveltyReport that shows prior work."""
    state = ReviewState(
        source="x",
        check_novelty=True,
        markdown="We present the first XYZ. Furthermore, we propose a novel ABC.",
        novelty_cache_dir=tmp_path,
    )
    deps = ReviewDeps(model="fake:test")

    extractor_output = NoveltyClaimList(
        claims=[
            NoveltyClaim(
                claim_text="We present the first XYZ.",
                short_summary="first XYZ method",
                why_load_bearing="abstract claim",
            ),
            NoveltyClaim(
                claim_text="Novel ABC technique.",
                short_summary="novel ABC technique",
                why_load_bearing="results headline",
            ),
        ]
    )

    class FakeAgent:
        def __init__(self, output):
            self.output = output

        async def run(self, _prompt):
            captured_output = self.output

            class _R:
                output = captured_output

            return _R()

    # Stub the per-claim novelty checker to return a "not novel" report
    fake_report = {
        "claim": "first XYZ method",
        "is_novel": False,
        "confidence": 0.85,
        "assessment": "Strong prior work exists.",
        "similar_work": [
            {
                "title": "Earlier XYZ paper",
                "url": "https://example.com/x",
                "relevance": "direct",
                "summary": "Earlier work showing XYZ.",
            }
        ],
        "sources": ["https://example.com/x"],
        "search_queries_used": [],
    }

    class FakeCtx:
        state: ReviewState
        deps: ReviewDeps

    ctx = FakeCtx()
    ctx.state = state
    ctx.deps = deps

    from andamentum.whetstone.v2.nodes import novelty_check as nc_mod

    with mock.patch.object(
        nc_mod,
        "build_pydantic_ai_agent",
        lambda *_a, **_k: FakeAgent(extractor_output),
    ), mock.patch.object(
        nc_mod, "_check_one_claim", mock.AsyncMock(return_value=fake_report)
    ):
        next_node = await NoveltyCheck().run(ctx)  # type: ignore[arg-type]

    assert next_node.__class__.__name__ == "EditSections"
    # Two claims → up to two findings (both flagged "not novel" by the
    # stub, so both surface)
    assert len(state.findings) == 2
    # All findings have category="novelty" and severity major (high conf)
    for f in state.findings:
        assert f.category == "novelty"
        assert f.severity == "major"


async def test_skips_finding_when_claim_is_novel(tmp_path: Path):
    """If deep_research confirms the claim IS novel, no finding is
    surfaced — surfacing 'looks novel' would clutter the output."""
    state = ReviewState(
        source="x",
        check_novelty=True,
        markdown="We present the first XYZ.",
        novelty_cache_dir=tmp_path,
    )
    deps = ReviewDeps(model="fake:test")

    extractor_output = NoveltyClaimList(
        claims=[
            NoveltyClaim(
                claim_text="We present the first XYZ.",
                short_summary="first XYZ",
                why_load_bearing="abstract",
            ),
        ]
    )

    class FakeAgent:
        async def run(self, _prompt):
            class _R:
                output = extractor_output

            return _R()

    fake_report_novel = {
        "claim": "first XYZ",
        "is_novel": True,
        "confidence": 0.9,
        "assessment": "No prior work found.",
        "similar_work": [],
        "sources": [],
        "search_queries_used": [],
    }

    class FakeCtx:
        state: ReviewState
        deps: ReviewDeps

    ctx = FakeCtx()
    ctx.state = state
    ctx.deps = deps

    from andamentum.whetstone.v2.nodes import novelty_check as nc_mod

    with mock.patch.object(
        nc_mod,
        "build_pydantic_ai_agent",
        lambda *_a, **_k: FakeAgent(),
    ), mock.patch.object(
        nc_mod, "_check_one_claim", mock.AsyncMock(return_value=fake_report_novel)
    ):
        await NoveltyCheck().run(ctx)  # type: ignore[arg-type]

    assert state.findings == []


async def test_per_claim_failure_does_not_abort_node(tmp_path: Path):
    """A single claim's deep_research call crashing should not poison
    the rest — the surviving claims should still produce findings."""
    state = ReviewState(
        source="x",
        check_novelty=True,
        markdown="We present the first XYZ. Novel ABC.",
        novelty_cache_dir=tmp_path,
    )
    deps = ReviewDeps(model="fake:test")

    extractor_output = NoveltyClaimList(
        claims=[
            NoveltyClaim(
                claim_text="A.",
                short_summary="A",
                why_load_bearing="x",
            ),
            NoveltyClaim(
                claim_text="B.",
                short_summary="B",
                why_load_bearing="x",
            ),
        ]
    )

    class FakeAgent:
        async def run(self, _prompt):
            class _R:
                output = extractor_output

            return _R()

    call_count = 0

    async def flaky_check(**kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise RuntimeError("simulated deep_research failure")
        return {
            "claim": "B",
            "is_novel": False,
            "confidence": 0.8,
            "assessment": "Found prior work for B.",
            "similar_work": [],
            "sources": [],
            "search_queries_used": [],
        }

    class FakeCtx:
        state: ReviewState
        deps: ReviewDeps

    ctx = FakeCtx()
    ctx.state = state
    ctx.deps = deps

    from andamentum.whetstone.v2.nodes import novelty_check as nc_mod

    with mock.patch.object(
        nc_mod,
        "build_pydantic_ai_agent",
        lambda *_a, **_k: FakeAgent(),
    ), mock.patch.object(nc_mod, "_check_one_claim", flaky_check):
        await NoveltyCheck().run(ctx)  # type: ignore[arg-type]

    # First claim crashed — no finding. Second claim succeeded — one finding.
    assert len(state.findings) == 1
    assert state.findings[0].title.startswith("Novelty claim contradicted")


# ── Disk cache ─────────────────────────────────────────────────────────


async def test_cache_hit_avoids_recomputation(tmp_path: Path):
    """When a cached result exists for a claim's hash, the deep_research
    pipeline is skipped."""
    from andamentum.whetstone.v2.nodes.novelty_check import _check_one_claim

    # Pre-populate the cache. Hash of short_summary "first XYZ" lives at
    # a stable path.
    import hashlib

    claim = NoveltyClaim(
        claim_text="We present the first XYZ.",
        short_summary="first XYZ",
        why_load_bearing="x",
    )
    cache_key = hashlib.sha256(claim.short_summary.encode("utf-8")).hexdigest()[:16]
    cache_path = tmp_path / f"{cache_key}.json"
    cache_path.write_text(
        json.dumps(
            {
                "claim": "first XYZ",
                "is_novel": False,
                "confidence": 0.95,
                "assessment": "From cache",
                "similar_work": [],
                "sources": [],
                "search_queries_used": [],
            }
        )
    )

    deps = ReviewDeps(model="fake:test")

    # If deep_research was called, this would fail with an import or
    # network error. The cache hit should bypass it entirely.
    result = await _check_one_claim(
        claim=claim,
        deps=deps,
        cache_dir=tmp_path,
        search_depth=1,
    )

    assert result["is_novel"] is False
    assert result["assessment"] == "From cache"


# ── Adapter ────────────────────────────────────────────────────────────


def test_adapter_high_confidence_not_novel_emits_major():
    claim = NoveltyClaim(
        claim_text="We present the first method.",
        short_summary="first method",
        why_load_bearing="abstract",
    )
    report = {
        "is_novel": False,
        "confidence": 0.85,
        "assessment": "Definitely not novel.",
        "similar_work": [],
    }
    f = _report_to_finding(claim, report)
    assert f is not None
    assert f.severity == "major"


def test_adapter_medium_confidence_not_novel_emits_moderate():
    claim = NoveltyClaim(
        claim_text="x",
        short_summary="x",
        why_load_bearing="x",
    )
    report = {
        "is_novel": False,
        "confidence": 0.5,
        "assessment": "Some related work.",
        "similar_work": [],
    }
    f = _report_to_finding(claim, report)
    assert f is not None
    assert f.severity == "moderate"


def test_adapter_low_confidence_not_novel_emits_minor():
    claim = NoveltyClaim(
        claim_text="x",
        short_summary="x",
        why_load_bearing="x",
    )
    report = {
        "is_novel": False,
        "confidence": 0.2,
        "assessment": "Maybe related work.",
        "similar_work": [],
    }
    f = _report_to_finding(claim, report)
    assert f is not None
    assert f.severity == "minor"


def test_adapter_novel_returns_none():
    claim = NoveltyClaim(
        claim_text="x",
        short_summary="x",
        why_load_bearing="x",
    )
    report = {
        "is_novel": True,
        "confidence": 0.9,
        "assessment": "No prior work.",
        "similar_work": [],
    }
    assert _report_to_finding(claim, report) is None
