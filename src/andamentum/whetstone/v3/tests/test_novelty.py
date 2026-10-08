"""Novelty 3-node pipeline for v3.

Coverage:
  - the extractor runs ALWAYS; only search and judge are gated on
    check_novelty, because only they talk to a party that was not already
    reading the document
  - check_novelty=True invokes extractor, deep_research, and judge
  - target cap bounds the number of deep_research calls
  - severity scaling: ≥0.7 → major / high, ≥0.4 → moderate / medium,
    else minor / low
  - is_novel=True verdicts produce NO Finding (silence on confirmation)
  - per-target deep_research crash does not abort the loop
  - findings flow through the existing graph into ReviewResult.findings
"""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import AsyncMock, patch


from andamentum.whetstone.v3.graph import (
    FlagNoveltyTargets,
    JudgeNovelty,
    RunNoveltySearches,
    V3Deps,
    V3State,
)
from andamentum.whetstone.v3.novelty import (
    NoveltyEvidence,
    NoveltyTarget,
    SimilarWorkRef,
    flag_novelty_targets,
    judge_novelty,
    run_novelty_searches,
    verdicts_to_findings,
)


# ── Helpers ────────────────────────────────────────────────────────────────


_TARGET = NoveltyTarget(
    claim_text="We present the first end-to-end neural framework for X.",
    short_summary="First end-to-end neural framework for X",
    why_load_bearing="Contribution statement in the abstract.",
)


@dataclass
class _Ctx:
    state: V3State
    deps: V3Deps


# ── Pure-function tests ────────────────────────────────────────────────────


def test_judge_novelty_high_confidence_yields_major_severity() -> None:
    """confidence >= 0.7 -> major severity, high confidence band."""
    ev = NoveltyEvidence(
        target=_TARGET, is_novel=False, confidence=0.85, assessment="..."
    )
    verdicts = judge_novelty([ev])
    assert len(verdicts) == 1
    assert verdicts[0].severity == "major"
    assert verdicts[0].confidence_band == "high"
    assert verdicts[0].is_novel is False


def test_judge_novelty_moderate_confidence_yields_moderate_severity() -> None:
    """0.4 <= confidence < 0.7 -> moderate severity, medium band."""
    ev = NoveltyEvidence(
        target=_TARGET, is_novel=False, confidence=0.55, assessment="..."
    )
    verdicts = judge_novelty([ev])
    assert verdicts[0].severity == "moderate"
    assert verdicts[0].confidence_band == "medium"


def test_judge_novelty_low_confidence_yields_minor_severity() -> None:
    """confidence < 0.4 -> minor severity, low band."""
    ev = NoveltyEvidence(
        target=_TARGET, is_novel=False, confidence=0.2, assessment="..."
    )
    verdicts = judge_novelty([ev])
    assert verdicts[0].severity == "minor"
    assert verdicts[0].confidence_band == "low"


def test_judge_novelty_is_novel_true_produces_no_finding() -> None:
    """is_novel=True yields verdict with severity=None;
    verdicts_to_findings drops it (silence on confirmation)."""
    ev = NoveltyEvidence(
        target=_TARGET, is_novel=True, confidence=0.95, assessment="confirmed novel"
    )
    verdicts = judge_novelty([ev])
    assert verdicts[0].severity is None
    findings = verdicts_to_findings(verdicts)
    assert findings == []


def test_judge_novelty_evidence_with_error_skipped() -> None:
    """A NoveltyEvidence carrying an error (the per-target search
    crashed) produces no verdict — we have no signal to report on."""
    ev = NoveltyEvidence(target=_TARGET, error="deep_research timeout")
    verdicts = judge_novelty([ev])
    assert verdicts == []


def test_verdicts_to_findings_maps_severity_into_finding() -> None:
    """Contradicted claim → v3.review.Finding with criterion='Novelty'
    and severity matching the verdict."""
    ev = NoveltyEvidence(
        target=_TARGET,
        is_novel=False,
        confidence=0.85,
        assessment="...",
        similar_work=[
            SimilarWorkRef(
                title="A 2024 paper on X",
                url="https://example.org/x",
                relevance="direct",
                summary="Already presents an end-to-end neural pipeline.",
            )
        ],
    )
    verdicts = judge_novelty([ev])
    findings = verdicts_to_findings(verdicts)
    assert len(findings) == 1
    f = findings[0]
    assert f.criterion == "Novelty"
    assert f.severity == "major"
    assert "Literature search" in f.issue


# ── Node-level tests (graph wiring) ────────────────────────────────────────


async def test_the_claims_are_found_even_when_nothing_will_verify_them() -> None:
    """The extractor runs with check_novelty=False, which is the default.

    This is the split the design turns on. FINDING what a draft claims as new
    costs one call to the model that has already read every word of it, so it
    discloses nothing further. VERIFYING those claims sends them to a search
    engine, which was never reading the document — a step to take deliberately
    on an unpublished manuscript, not one to inherit from a default.

    So the list is always produced and never checked unless asked, and the
    report says which of the two it is holding.
    """
    state = V3State(source="some markdown")
    deps = V3Deps(agent_model="stub", check_novelty=False)
    with patch(
        "andamentum.whetstone.v3.graph.flag_novelty_targets",
        new=AsyncMock(return_value=[_TARGET]),
    ) as mock_extract:
        result = await FlagNoveltyTargets().run(_Ctx(state, deps))
    assert isinstance(result, RunNoveltySearches)
    mock_extract.assert_awaited_once()
    assert state.novelty_targets == [_TARGET]


async def test_a_later_round_of_a_reviewer_list_does_not_flag_again() -> None:
    """flag_novelty=False skips the extractor call entirely.

    The CLI sets it for the second and later rounds of a reviewer list: the
    draft's claims do not change because another reviewer read it, and the
    old behaviour spent one full-draft model call per extra round to produce
    a list that was discarded.
    """
    state = V3State(source="some markdown")
    deps = V3Deps(agent_model="stub", flag_novelty=False)
    with patch(
        "andamentum.whetstone.v3.graph.flag_novelty_targets",
        new=AsyncMock(side_effect=AssertionError("no call may be spent")),
    ) as mock_extract:
        result = await FlagNoveltyTargets().run(_Ctx(state, deps))
    assert isinstance(result, RunNoveltySearches)
    mock_extract.assert_not_called()
    assert state.novelty_targets == []


async def test_nothing_is_searched_unless_asked() -> None:
    """The guard moved to the node that talks to the outside world, and it is
    the only thing standing between an unpublished draft and a search engine."""
    state = V3State(source="x", novelty_targets=[_TARGET])
    deps = V3Deps(agent_model="stub", check_novelty=False)
    with patch(
        "andamentum.whetstone.v3.graph.run_novelty_searches",
        new=AsyncMock(side_effect=AssertionError("nothing may leave the machine")),
    ) as mock_search:
        await RunNoveltySearches().run(_Ctx(state, deps))
    mock_search.assert_not_called()
    assert state.novelty_evidence == []


async def test_check_novelty_true_invokes_extractor_with_cap() -> None:
    """check_novelty=True calls flag_novelty_targets with the configured
    cap, and stores the result in state.novelty_targets."""
    state = V3State(source="some markdown")
    deps = V3Deps(agent_model="stub", check_novelty=True, novelty_target_cap=4)
    targets = [_TARGET]
    with patch(
        "andamentum.whetstone.v3.graph.flag_novelty_targets",
        new=AsyncMock(return_value=targets),
    ) as mock_extract:
        await FlagNoveltyTargets().run(_Ctx(state, deps))
    mock_extract.assert_awaited_once()
    # Confirm cap was passed through
    kwargs = mock_extract.call_args.kwargs
    assert kwargs["cap"] == 4
    assert state.novelty_targets == targets


async def test_run_novelty_searches_node_invokes_helper() -> None:
    """RunNoveltySearches forwards targets + search_depth to the helper."""
    state = V3State(source="...", novelty_targets=[_TARGET])
    deps = V3Deps(agent_model="stub", check_novelty=True, novelty_search_depth=3)
    evidence = [NoveltyEvidence(target=_TARGET, is_novel=False, confidence=0.8)]
    with patch(
        "andamentum.whetstone.v3.graph.run_novelty_searches",
        new=AsyncMock(return_value=evidence),
    ) as mock_search:
        result = await RunNoveltySearches().run(_Ctx(state, deps))
    assert isinstance(result, JudgeNovelty)
    mock_search.assert_awaited_once()
    assert mock_search.call_args.kwargs["search_depth"] == 3
    assert state.novelty_evidence == evidence


async def test_run_novelty_searches_skips_when_no_targets() -> None:
    """When the extractor returned [] (no novelty claims to verify),
    RunNoveltySearches does not call deep_research."""
    state = V3State(source="...", novelty_targets=[])
    deps = V3Deps(agent_model="stub", check_novelty=True)
    with patch(
        "andamentum.whetstone.v3.graph.run_novelty_searches",
        new=AsyncMock(side_effect=AssertionError("should not be called")),
    ) as mock_search:
        result = await RunNoveltySearches().run(_Ctx(state, deps))
    assert isinstance(result, JudgeNovelty)
    mock_search.assert_not_called()


async def test_judge_novelty_node_extends_findings_list() -> None:
    """JudgeNovelty appends novelty findings (Finding objects with
    criterion='Novelty') to state.findings — no displacement of
    existing criterion findings."""
    state = V3State(source="...")
    state.novelty_evidence = [
        NoveltyEvidence(target=_TARGET, is_novel=False, confidence=0.85)
    ]
    deps = V3Deps(agent_model="stub", check_novelty=True)
    await JudgeNovelty().run(_Ctx(state, deps))
    assert len(state.findings) == 1
    assert state.findings[0].criterion == "Novelty"


async def test_run_novelty_searches_helper_isolates_per_target_failures() -> None:
    """If deep_research crashes for one target, others still produce
    evidence — the failed target gets error=... in its NoveltyEvidence."""
    targets = [
        NoveltyTarget(
            claim_text="claim 1",
            short_summary="claim 1 summary",
            why_load_bearing="why 1",
        ),
        NoveltyTarget(
            claim_text="claim 2",
            short_summary="claim 2 summary",
            why_load_bearing="why 2",
        ),
    ]

    from andamentum.deep_research.v2.novelty import NoveltyReport

    call_count = [0]

    async def _stub_run_novelty_check(claim, **_kw):
        call_count[0] += 1
        if call_count[0] == 1:
            raise RuntimeError("simulated deep_research crash")
        return NoveltyReport(
            claim=claim, is_novel=False, confidence=0.8, assessment="found prior work"
        )

    with patch(
        "andamentum.deep_research.v2.run_novelty_check_v2",
        new=_stub_run_novelty_check,
    ):
        evidence = await run_novelty_searches(
            targets, agent_model="stub", search_depth=1
        )
    assert len(evidence) == 2
    assert evidence[0].error is not None
    assert evidence[1].error is None
    assert evidence[1].is_novel is False


async def test_flag_novelty_targets_empty_source_returns_no_targets() -> None:
    """A whitespace-only source skips the extractor LLM call entirely."""
    targets = await flag_novelty_targets([], "   \n\n   ", agent_model="stub", cap=8)
    assert targets == []
