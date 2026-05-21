"""Tests for the Consolidate node.

The adjudication agent is stubbed via patching ``build_pydantic_ai_agent``;
embeddings are injected via ``deps.embedding_fn`` so no Ollama is needed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal
from unittest.mock import patch

from andamentum.whetstone.agents.consolidate import SameOrDistinct
from andamentum.whetstone.deps import ReviewDeps
from andamentum.whetstone.nodes.consolidate import Consolidate
from andamentum.whetstone.schemas import Finding, Quote
from andamentum.whetstone.state import ReviewState


# ── Stubs ─────────────────────────────────────────────────────────────────


@dataclass
class _Result:
    output: SameOrDistinct


class _FakeAgent:
    """Returns a fixed same/distinct verdict for every adjudication call."""

    def __init__(self, relation: str):
        self._relation = relation

    async def run(self, prompt: str):
        return _Result(output=SameOrDistinct(relation=self._relation))  # type: ignore[arg-type]


async def _embed_identical(texts: list[str]) -> list[list[float]]:
    """Every claim gets the same vector → cosine 1.0 (max similarity)."""
    return [[1.0, 0.0, 0.0] for _ in texts]


async def _embed_orthogonal(texts: list[str]) -> list[list[float]]:
    """Each claim gets a distinct orthogonal vector → cosine 0 (no sim edge)."""
    return [
        [1.0 if i == k else 0.0 for k in range(len(texts))] for i in range(len(texts))
    ]


def _ctx(state: ReviewState, deps: ReviewDeps):
    @dataclass
    class _Ctx:
        state: ReviewState
        deps: ReviewDeps

    return _Ctx(state=state, deps=deps)


def _finding(
    title,
    *,
    start=0,
    end=10,
    perspective=None,
    section="s1",
    source: Literal["deterministic", "investigate", "challenged"] = "investigate",
):
    return Finding(
        title=title,
        severity="moderate",
        confidence="medium",
        rationale=f"rationale for {title}",
        quotes=[Quote(section_id=section, char_start=start, char_end=end, text=title)],
        sections_involved=[section],
        source=source,
        perspective=perspective,
    )


# ── Tests ──────────────────────────────────────────────────────────────────


async def test_same_verdict_merges_two_findings() -> None:
    state = ReviewState(source="x")
    state.challenged_findings = [
        _finding("claim unsupported", perspective="rigorous"),
        _finding("claim lacks evidence", start=2, end=12, perspective="skeptic"),
    ]
    deps = ReviewDeps(model="stub", embedding_fn=_embed_identical)

    with patch(
        "andamentum.whetstone.nodes.consolidate.build_pydantic_ai_agent",
        return_value=_FakeAgent("same"),
    ):
        await Consolidate().run(_ctx(state, deps))  # type: ignore[arg-type]

    # Two findings → one merged, with both perspectives recorded + confidence bump.
    assert len(state.challenged_findings) == 1
    merged = state.challenged_findings[0]
    assert merged.corroborated_by == ["rigorous", "skeptic"]
    assert merged.confidence == "high"


async def test_cross_section_pairs_held_off() -> None:
    # Identical embeddings (cosine 1.0) but DIFFERENT sections → not a
    # candidate; cross-section merging is held off. Agent must not run.
    state = ReviewState(source="x")
    state.challenged_findings = [
        _finding("claim unsupported", section="s1", perspective="rigorous"),
        _finding("claim unsupported", section="s2", perspective="skeptic"),
    ]
    deps = ReviewDeps(model="stub", embedding_fn=_embed_identical)

    def _boom(*a, **k):  # pragma: no cover - asserts non-invocation
        raise AssertionError("no cross-section pair should be adjudicated")

    with patch(
        "andamentum.whetstone.nodes.consolidate.build_pydantic_ai_agent",
        side_effect=_boom,
    ):
        await Consolidate().run(_ctx(state, deps))  # type: ignore[arg-type]

    assert len(state.challenged_findings) == 2  # both survive, unmerged


async def test_distinct_verdict_keeps_both() -> None:
    state = ReviewState(source="x")
    state.challenged_findings = [
        _finding("passive voice", perspective="rigorous"),
        _finding("overclaim", start=2, end=12, perspective="skeptic"),
    ]
    deps = ReviewDeps(model="stub", embedding_fn=_embed_identical)

    with patch(
        "andamentum.whetstone.nodes.consolidate.build_pydantic_ai_agent",
        return_value=_FakeAgent("distinct"),
    ):
        await Consolidate().run(_ctx(state, deps))  # type: ignore[arg-type]

    assert len(state.challenged_findings) == 2


async def test_deterministic_rollup_needs_no_llm() -> None:
    state = ReviewState(source="x")
    state.deterministic_findings = [
        Finding(
            title="Passive voice",
            severity="minor",
            confidence="high",
            rationale="passive",
            quotes=[Quote(section_id="s1", char_start=i * 20, char_end=i * 20 + 5, text=f"q{i}")],
            sections_involved=["s1"],
            source="deterministic",
            category="style:passive",
        )
        for i in range(5)
    ]
    # No challenged findings; orthogonal embeds so no semantic edges form. The
    # rollup happens deterministically; the agent must never be called.
    deps = ReviewDeps(model="stub", embedding_fn=_embed_orthogonal)

    def _boom(*a, **k):  # pragma: no cover - asserts non-invocation
        raise AssertionError("adjudication agent should not run for pure rollup")

    with patch(
        "andamentum.whetstone.nodes.consolidate.build_pydantic_ai_agent",
        side_effect=_boom,
    ):
        await Consolidate().run(_ctx(state, deps))  # type: ignore[arg-type]

    assert len(state.deterministic_findings) == 1
    assert state.deterministic_findings[0].title == "5× Passive voice"


async def test_novelty_findings_pass_through_untouched() -> None:
    state = ReviewState(source="x")
    novelty = _finding("novel claim", source="challenged")
    novelty = novelty.model_copy(update={"category": "novelty"})
    state.challenged_findings = [novelty]
    deps = ReviewDeps(model="stub", embedding_fn=_embed_identical)

    with patch(
        "andamentum.whetstone.nodes.consolidate.build_pydantic_ai_agent",
        return_value=_FakeAgent("same"),
    ):
        await Consolidate().run(_ctx(state, deps))  # type: ignore[arg-type]

    assert len(state.challenged_findings) == 1
    assert state.challenged_findings[0].category == "novelty"
