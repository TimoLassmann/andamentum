"""End-to-end test of Phases 2-6 with fake LLM agents.

Patches ``build_pydantic_ai_agent`` so each agent name returns a fake
async ``run`` method that produces canned structured outputs. This lets
us exercise the full graph (Skim → InvestigateLoop → Challenge →
AuthorQuestions → Synthesise) without making any network calls.

These tests are the proof that Phases 2-6 are wired correctly:
the data flows from one node to the next, perspectives are tracked,
findings accumulate, challenges apply verdicts, the registry dispatches
to ``investigate_internal``, etc.
"""

from dataclasses import dataclass
from typing import Any

import pytest

from andamentum.whetstone.v2 import review_document
from andamentum.whetstone.v2.agents import (
    AuthorQuestionOutput,
    ChallengeVerdict,
    InvestigationOutput,
    ReviewSummary,
    SkimHypothesis,
    SkimOutput,
    SkimSection,
)


PAPER = """## 1 Introduction

This paper studies Reinforcement Learning (RL) applied to bipedal walking.
We had N = 50 participants in our user study, and cite prior work [1, 42].
As shown in Figure 1, the results are striking.

## 2 Methods

We compare two variants of RL on the same benchmark.
Across N=48 trials, the new method outperforms baselines significantly.
Figure 1: Comparison of accuracy across methods.

## References

[1] First Author. (2020). Title one.
[2] Second Author. (2021).
"""


@dataclass
class _FakeRunResult:
    output: Any


class _FakeAgent:
    """Fake pydantic-ai Agent. Returns a canned output for every .run() call."""

    def __init__(self, output: Any):
        self.output = output
        self.calls: list[str] = []

    async def run(self, prompt: str) -> _FakeRunResult:
        self.calls.append(prompt)
        return _FakeRunResult(output=self.output)


@pytest.fixture
def patched_agents(monkeypatch):
    """Patch ``build_pydantic_ai_agent`` to return a fake per name.

    Returns a dict the test customises BEFORE calling review_document().
    """
    canned: dict[str, Any] = {}

    def fake_build(name: str, model: Any) -> _FakeAgent:
        if name not in canned:
            raise AssertionError(f"agent {name!r} called but no canned output set")
        return _FakeAgent(output=canned[name])

    # Patch at every site the agent is built (not just one — each node
    # imports the helper directly).
    import andamentum.whetstone.v2.agents as agents_mod
    import andamentum.whetstone.v2.investigators.internal as inv_mod
    import andamentum.whetstone.v2.nodes.author_questions as aq_mod
    import andamentum.whetstone.v2.nodes.challenge as ch_mod
    import andamentum.whetstone.v2.nodes.skim as sk_mod
    import andamentum.whetstone.v2.nodes.synthesise as sy_mod

    for mod in (agents_mod, inv_mod, aq_mod, ch_mod, sk_mod, sy_mod):
        monkeypatch.setattr(mod, "build_pydantic_ai_agent", fake_build, raising=True)
    return canned


# ── Phase 2: Skim emits hypotheses, Investigate produces findings ──────


async def test_skim_then_investigate_produces_findings(patched_agents):
    """Single-perspective: skim emits 2 hypotheses, investigator finds 1."""
    patched_agents["skim"] = SkimOutput(
        enriched_sections=[
            SkimSection(section_id="sec_001", one_line_gist="Intro of RL setup."),
            SkimSection(section_id="sec_002", one_line_gist="Methods comparing variants."),
        ],
        hypotheses=[
            SkimHypothesis(
                text="Is the sample size in §1 consistent with §2?",
                priority="high",
                relevant_section_ids=["sec_001", "sec_002"],
            ),
            SkimHypothesis(
                text="Does Figure 1 actually compare what the methods say?",
                priority="medium",
                relevant_section_ids=["sec_002"],
            ),
        ],
    )
    patched_agents["investigate"] = InvestigationOutput(
        decision="finding",
        finding_title="Sample size mismatch between sections",
        finding_severity="major",
        finding_confidence="high",
        finding_rationale="§1 says N=50, §2 says N=48.",
        finding_quotes=["N = 50", "N=48"],
        finding_sections=["sec_001", "sec_002"],
    )
    # Challenge says everything stands as-is.
    patched_agents["challenge"] = ChallengeVerdict(verdict="stand", reason="ok")
    # No author questions for this run.
    patched_agents["author_question"] = AuthorQuestionOutput(
        question="placeholder", why="n/a", sections_involved=[]
    )
    patched_agents["synthesise"] = ReviewSummary(
        executive_summary="Two issues found.",
        major_findings_summary="Sample-size mismatch between §1 and §2.",
        moderate_findings_summary="No moderate findings.",
        minor_findings_summary="No minor findings.",
    )

    result = await review_document(PAPER, model="fake:test")

    # Skim's output added findings via the investigate agent
    assert len(result.findings) >= 1
    assert any("Sample size" in f.title for f in result.findings)
    # Quotes were located in source via find_anchor
    finding = next(f for f in result.findings if "Sample size" in f.title)
    assert len(finding.quotes) >= 1
    # Summary was filled in by Synthesise
    assert "Two issues found" in result.summary
    # Metrics show LLM activity
    assert result.metrics.llm_calls >= 3  # skim + investigate + challenge + synthesise


async def test_challenge_withdraw_drops_finding(patched_agents):
    """A challenger that says 'withdraw' removes the finding from the result."""
    patched_agents["skim"] = SkimOutput(
        enriched_sections=[],
        hypotheses=[
            SkimHypothesis(
                text="Q",
                priority="high",
                relevant_section_ids=["sec_001"],
            )
        ],
    )
    patched_agents["investigate"] = InvestigationOutput(
        decision="finding",
        finding_title="False alarm",
        finding_severity="major",  # high enough to be challenged
        finding_confidence="high",
        finding_rationale="claim",
        finding_quotes=["RL"],  # locatable in the paper
        finding_sections=["sec_001"],
    )
    patched_agents["challenge"] = ChallengeVerdict(
        verdict="withdraw", reason="actually that's fine"
    )
    patched_agents["author_question"] = AuthorQuestionOutput(
        question="x", why="x", sections_involved=[]
    )
    patched_agents["synthesise"] = ReviewSummary(
        executive_summary="Nothing major.",
        major_findings_summary="No major findings.",
        moderate_findings_summary="No moderate findings.",
        minor_findings_summary="No minor findings.",
    )

    result = await review_document(PAPER, model="fake:test")

    # The withdrawn finding should not appear in result.findings
    assert not any("False alarm" in f.title for f in result.findings)


async def test_challenge_weaken_lowers_confidence(patched_agents):
    """A 'weaken' verdict drops confidence by one tier."""
    patched_agents["skim"] = SkimOutput(
        enriched_sections=[],
        hypotheses=[
            SkimHypothesis(
                text="Q", priority="high", relevant_section_ids=["sec_001"]
            )
        ],
    )
    patched_agents["investigate"] = InvestigationOutput(
        decision="finding",
        finding_title="Suspicious claim",
        finding_severity="moderate",  # challengeable
        finding_confidence="high",
        finding_rationale="rationale",
        finding_quotes=["RL"],
        finding_sections=["sec_001"],
    )
    patched_agents["challenge"] = ChallengeVerdict(
        verdict="weaken", reason="evidence is partial"
    )
    patched_agents["author_question"] = AuthorQuestionOutput(
        question="x", why="x", sections_involved=[]
    )
    patched_agents["synthesise"] = ReviewSummary(
        executive_summary="One weak claim.",
        major_findings_summary="No major findings.",
        moderate_findings_summary="One moderate, weakened.",
        minor_findings_summary="No minor findings.",
    )

    result = await review_document(PAPER, model="fake:test")

    weakened = next(f for f in result.findings if "Suspicious claim" in f.title)
    assert weakened.confidence == "medium"  # was "high"
    assert weakened.source == "challenged"
    assert "Challenged:" in weakened.rationale


async def test_disabled_challenge_keeps_findings_unchanged(patched_agents):
    """challenge=False skips the Challenge phase entirely."""
    patched_agents["skim"] = SkimOutput(
        enriched_sections=[],
        hypotheses=[
            SkimHypothesis(
                text="Q", priority="high", relevant_section_ids=["sec_001"]
            )
        ],
    )
    patched_agents["investigate"] = InvestigationOutput(
        decision="finding",
        finding_title="Strong finding",
        finding_severity="major",
        finding_confidence="high",
        finding_rationale="solid",
        finding_quotes=["RL"],
        finding_sections=["sec_001"],
    )
    patched_agents["author_question"] = AuthorQuestionOutput(
        question="x", why="x", sections_involved=[]
    )
    patched_agents["synthesise"] = ReviewSummary(
        executive_summary="One finding.",
        major_findings_summary="One major finding.",
        moderate_findings_summary="No moderate findings.",
        minor_findings_summary="No minor findings.",
    )

    # Note: challenge=False ⇒ "challenge" agent never invoked, no need to set it.
    result = await review_document(PAPER, model="fake:test", challenge=False)

    finding = next(f for f in result.findings if "Strong finding" in f.title)
    # source remains "investigate" because Challenge node was skipped
    assert finding.source == "investigate"
    assert finding.confidence == "high"


# ── Phase 5: Panel mode (multi-perspective) ────────────────────────────


async def test_panel_mode_tags_findings_with_perspective(patched_agents):
    """Two perspectives → two skim runs → findings tagged by perspective."""
    patched_agents["skim"] = SkimOutput(
        enriched_sections=[],
        hypotheses=[
            SkimHypothesis(
                text="Per-persona Q",
                priority="medium",
                relevant_section_ids=["sec_001"],
            )
        ],
    )
    patched_agents["investigate"] = InvestigationOutput(
        decision="finding",
        finding_title="One finding per perspective",
        finding_severity="minor",  # below challenge threshold → kept as is
        finding_confidence="medium",
        finding_rationale="rationale",
        finding_quotes=["RL"],
        finding_sections=["sec_001"],
    )
    patched_agents["author_question"] = AuthorQuestionOutput(
        question="x", why="x", sections_involved=[]
    )
    patched_agents["synthesise"] = ReviewSummary(
        executive_summary="Two perspectives looked.",
        major_findings_summary="No major findings.",
        moderate_findings_summary="No moderate findings.",
        minor_findings_summary="One minor each.",
    )

    result = await review_document(
        PAPER,
        model="fake:test",
        perspectives=["statistician", "writer"],
    )

    # Two perspectives → at least 2 findings (one per perspective)
    perspectives_seen = {f.perspective for f in result.findings if f.perspective}
    assert "statistician" in perspectives_seen
    assert "writer" in perspectives_seen


# ── Phase 6: AuthorQuestions when budget exhausted ─────────────────────


async def test_author_questions_emitted_for_unresolved_hypotheses(patched_agents):
    """A budget of 1 with 2 hypotheses leaves one open → becomes a question."""
    patched_agents["skim"] = SkimOutput(
        enriched_sections=[],
        hypotheses=[
            SkimHypothesis(
                text="High-priority Q",
                priority="high",
                relevant_section_ids=["sec_001"],
            ),
            SkimHypothesis(
                text="Lower-priority Q",
                priority="low",
                relevant_section_ids=["sec_001"],
            ),
        ],
    )
    patched_agents["investigate"] = InvestigationOutput(
        decision="unfounded",
        unfounded_reason="no evidence found",
    )
    patched_agents["author_question"] = AuthorQuestionOutput(
        question="Could you clarify the lower-priority point?",
        why="couldn't resolve from text",
        sections_involved=["sec_001"],
    )
    patched_agents["synthesise"] = ReviewSummary(
        executive_summary="Mostly unresolvable.",
        major_findings_summary="No major findings.",
        moderate_findings_summary="No moderate findings.",
        minor_findings_summary="No minor findings.",
    )

    result = await review_document(
        PAPER, model="fake:test", hypothesis_budget=1
    )

    # Budget=1 ⇒ only one hypothesis investigated; the other stays "open".
    # AuthorQuestions should turn it into a question.
    assert len(result.author_questions) >= 1
    assert "lower-priority" in result.author_questions[0].question.lower()


# ── Phase 1 still works (no model passed) ──────────────────────────────


async def test_phase_1_path_still_works_without_model():
    """Don't pass a model → ChunkAndScan returns End[ReviewResult] directly,
    no LLM phases run. This is the deterministic-only path from Phase 1."""
    result = await review_document(PAPER)  # no model arg
    assert result.metrics.llm_calls == 0
    assert result.findings == []  # no LLM-driven findings
    assert result.summary == ""  # no synthesis
    # But deterministic findings still populated
    assert len(result.deterministic_findings) > 0
