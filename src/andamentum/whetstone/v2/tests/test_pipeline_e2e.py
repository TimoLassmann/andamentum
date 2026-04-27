"""End-to-end pipeline test for the new critical-review flow.

Stubs every LLM-calling agent and runs the full graph:

    HarvestSource → ChunkAndScan → CriticalRead → ReflectAndInvestigate
                                  → EditSections → Challenge
                                  → AuthorQuestions → Synthesise

Confirms the data flows correctly: lenses produce findings, the loop
consolidates them, challenge optionally refines them, summary is
written.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from andamentum.whetstone.v2 import review_document
from andamentum.whetstone.v2.agents import (
    AuthorQuestionOutput,
    ChallengeVerdict,
    InvestigatorOutput,
    LensIssueProposal,
    LensReadOutput,
    NoteUpdate,
    ReflectionOutput,
    ReflectionTask,
    ReviewSummary,
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
    def __init__(self, output: Any):
        self.output = output
        self.calls: list[str] = []

    async def run(self, prompt: str) -> _FakeRunResult:
        self.calls.append(prompt)
        return _FakeRunResult(output=self.output)


@pytest.fixture
def patched_agents(monkeypatch):
    """Patch ``build_pydantic_ai_agent`` everywhere a node imports it."""
    canned: dict[str, Any] = {}

    def fake_build(name: str, model: Any) -> _FakeAgent:
        # Lens agents share an output schema — match by lens.<name> prefix.
        if name.startswith("lens.") and "lens" in canned:
            return _FakeAgent(output=canned["lens"])
        if name not in canned:
            raise AssertionError(
                f"agent {name!r} called but no canned output set"
            )
        return _FakeAgent(output=canned[name])

    import andamentum.whetstone.v2.agents as agents_mod
    import andamentum.whetstone.v2.nodes.author_questions as aq_mod
    import andamentum.whetstone.v2.nodes.challenge as ch_mod
    import andamentum.whetstone.v2.nodes.critical_read as cr_mod
    import andamentum.whetstone.v2.nodes.reflect_and_investigate as ri_mod
    import andamentum.whetstone.v2.nodes.synthesise as sy_mod

    for mod in (agents_mod, aq_mod, ch_mod, cr_mod, ri_mod, sy_mod):
        monkeypatch.setattr(
            mod, "build_pydantic_ai_agent", fake_build, raising=True
        )
    return canned


# ── Lens reads → findings appear in pool ───────────────────────────────


async def test_lens_findings_land_in_pool(patched_agents) -> None:
    """One lens × N sections → N findings (one per section per lens)."""
    patched_agents["lens"] = LensReadOutput(
        issues=[
            LensIssueProposal(
                title="prose is unclear here",
                severity="minor",
                confidence="medium",
                rationale="The sentence about RL is hard to parse.",
                quote_text="Reinforcement Learning",
                category="argument-flow",
            )
        ]
    )
    # Reflection returns nothing → loop exits round 1.
    patched_agents["reflection"] = ReflectionOutput(tasks=[])
    patched_agents["author_question"] = AuthorQuestionOutput(
        question="x", why="x", sections_involved=[]
    )
    patched_agents["synthesise"] = ReviewSummary(
        executive_summary="some prose issues",
        major_findings_summary="No major findings.",
        moderate_findings_summary="No moderate findings.",
        minor_findings_summary="One minor.",
    )

    result = await review_document(
        PAPER, model="fake:test", perspectives=["writer"]
    )

    # 3 sections × 1 lens = 3 findings.
    assert len(result.findings) == 3
    assert all(f.perspective == "writer" for f in result.findings)
    assert all("prose is unclear" in f.title for f in result.findings)


# ── Reflection produces a task → investigation refines a finding ───────


async def test_reflection_loop_refines_finding(patched_agents) -> None:
    """Reflection picks one note; investigator refines it; result reflects refinement."""
    patched_agents["lens"] = LensReadOutput(
        issues=[
            LensIssueProposal(
                title="vague claim",
                severity="moderate",
                confidence="low",
                rationale="The claim is not specific enough.",
                quote_text="Reinforcement Learning",
                category="evidence",
            )
        ]
    )

    # First reflection round: pick the first finding for refinement.
    # We don't know the auto-generated id yet; the controller passes the
    # task to the investigator regardless of id, but the investigator
    # has to refer to the right id. We use a side-effect closure to
    # peek at state once it's built.
    captured: dict[str, str] = {}

    class _DynamicReflectionAgent:
        async def run(self, prompt: str):
            # Extract a note id from the prompt — they look like "[abc12345]".
            import re

            m = re.search(r"\[([0-9a-f]{8})\]", prompt)
            if m and "first_id" not in captured:
                captured["first_id"] = m.group(1)
                captured["first_section"] = re.search(
                    r"\[[0-9a-f]{8}\] [^\s]+ (sec_\d+):", prompt
                ).group(1) if re.search(r"\[[0-9a-f]{8}\] [^\s]+ (sec_\d+):", prompt) else "sec_001"
                return _FakeRunResult(output=ReflectionOutput(tasks=[
                    ReflectionTask(
                        description="Verify and refine this vague claim.",
                        section_ids=[captured["first_section"]],
                        related_note_ids=[captured["first_id"]],
                    )
                ]))
            # Subsequent rounds: nothing more to do.
            return _FakeRunResult(output=ReflectionOutput(tasks=[]))

    class _DynamicInvestigatorAgent:
        async def run(self, prompt: str):
            return _FakeRunResult(output=InvestigatorOutput(
                updates=[NoteUpdate(
                    note_id=captured.get("first_id", "ghost"),
                    action="refine",
                    refined_title="vague claim — specifically about RL participants",
                    refined_severity="moderate",
                    refined_confidence="medium",
                    refined_rationale="The number 50 doesn't match downstream methods.",
                    refined_quote_text="N = 50 participants",
                    refined_quote_section_id=captured.get("first_section", "sec_001"),
                )]
            ))

    def fake_build(name: str, model: Any):
        if name.startswith("lens."):
            return _FakeAgent(output=patched_agents["lens"])
        if name == "reflection":
            return _DynamicReflectionAgent()
        if name == "investigator":
            return _DynamicInvestigatorAgent()
        if name in patched_agents:
            return _FakeAgent(output=patched_agents[name])
        raise AssertionError(f"agent {name!r} not stubbed")

    import andamentum.whetstone.v2.agents as agents_mod
    import andamentum.whetstone.v2.nodes.author_questions as aq_mod
    import andamentum.whetstone.v2.nodes.challenge as ch_mod
    import andamentum.whetstone.v2.nodes.critical_read as cr_mod
    import andamentum.whetstone.v2.nodes.reflect_and_investigate as ri_mod
    import andamentum.whetstone.v2.nodes.synthesise as sy_mod

    import unittest.mock as mock

    patched_agents["author_question"] = AuthorQuestionOutput(
        question="x", why="x", sections_involved=[]
    )
    patched_agents["challenge"] = ChallengeVerdict(verdict="stand", reason="ok")
    patched_agents["synthesise"] = ReviewSummary(
        executive_summary="One refined finding.",
        major_findings_summary="No major findings.",
        moderate_findings_summary="One moderate, refined.",
        minor_findings_summary="No minor findings.",
    )

    with mock.patch.multiple(
        agents_mod, build_pydantic_ai_agent=fake_build
    ), mock.patch.multiple(
        aq_mod, build_pydantic_ai_agent=fake_build
    ), mock.patch.multiple(
        ch_mod, build_pydantic_ai_agent=fake_build
    ), mock.patch.multiple(
        cr_mod, build_pydantic_ai_agent=fake_build
    ), mock.patch.multiple(
        ri_mod, build_pydantic_ai_agent=fake_build
    ), mock.patch.multiple(
        sy_mod, build_pydantic_ai_agent=fake_build
    ):
        result = await review_document(
            PAPER, model="fake:test", perspectives=["rigorous"]
        )

    # At least one finding survives, and one was refined to mention "RL participants".
    refined = [f for f in result.findings if "RL participants" in f.title]
    assert len(refined) >= 1


# ── Round cap is respected ─────────────────────────────────────────────


async def test_round_cap_terminates_loop(patched_agents) -> None:
    """Reflection always returns one task → loop stops at the round cap."""
    patched_agents["lens"] = LensReadOutput(issues=[
        LensIssueProposal(
            title="x",
            severity="minor",
            confidence="low",
            rationale="...",
            quote_text="Reinforcement Learning",
        )
    ])
    patched_agents["author_question"] = AuthorQuestionOutput(
        question="x", why="x", sections_involved=[]
    )
    patched_agents["synthesise"] = ReviewSummary(
        executive_summary="x",
        major_findings_summary="No major findings.",
        moderate_findings_summary="No moderate findings.",
        minor_findings_summary="No minor findings.",
    )

    # Reflection returns one task every round (forcing the cap to be hit).
    # Investigator returns no-op.
    call_counter = {"n": 0}

    class _AlwaysOneTask:
        async def run(self, prompt: str):
            call_counter["n"] += 1
            return _FakeRunResult(output=ReflectionOutput(tasks=[
                ReflectionTask(
                    description=f"do something {call_counter['n']}",
                    section_ids=["sec_001"],
                )
            ]))

    class _NoopInvestigator:
        async def run(self, prompt: str):
            return _FakeRunResult(output=InvestigatorOutput())

    def fake_build(name: str, model: Any):
        if name.startswith("lens."):
            return _FakeAgent(output=patched_agents["lens"])
        if name == "reflection":
            return _AlwaysOneTask()
        if name == "investigator":
            return _NoopInvestigator()
        if name in patched_agents:
            return _FakeAgent(output=patched_agents[name])
        raise AssertionError(f"agent {name!r} not stubbed")

    import andamentum.whetstone.v2.agents as agents_mod
    import andamentum.whetstone.v2.nodes.author_questions as aq_mod
    import andamentum.whetstone.v2.nodes.challenge as ch_mod
    import andamentum.whetstone.v2.nodes.critical_read as cr_mod
    import andamentum.whetstone.v2.nodes.reflect_and_investigate as ri_mod
    import andamentum.whetstone.v2.nodes.synthesise as sy_mod
    import unittest.mock as mock

    with mock.patch.multiple(
        agents_mod, build_pydantic_ai_agent=fake_build
    ), mock.patch.multiple(
        aq_mod, build_pydantic_ai_agent=fake_build
    ), mock.patch.multiple(
        ch_mod, build_pydantic_ai_agent=fake_build
    ), mock.patch.multiple(
        cr_mod, build_pydantic_ai_agent=fake_build
    ), mock.patch.multiple(
        ri_mod, build_pydantic_ai_agent=fake_build
    ), mock.patch.multiple(
        sy_mod, build_pydantic_ai_agent=fake_build
    ):
        result = await review_document(
            PAPER,
            model="fake:test",
            perspectives=["rigorous"],
            challenge=False,  # skip challenge for simplicity
        )

    # Round cap = 3 (the default), so reflection should be called exactly 3 times.
    assert call_counter["n"] == 3
    # Result still produced.
    assert result.summary
