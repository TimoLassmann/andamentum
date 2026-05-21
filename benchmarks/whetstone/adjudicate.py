"""Judge-model adjudication + blinded human worksheet.

A strong judge model aligns the two arms' findings, buckets each
(both / a_only / b_only) and tags it (critical|minor, cross_section|local) per
the pre-registered rubric. Because an LLM judging LLM output is circular, we
also emit a BLINDED worksheet: the subset that matters (every b_only-critical
item plus a sample) shown with the arm relabelled "System 1/2" via a per-paper
random mapping, for a human to verify. The de-blinding key is saved separately.
"""

from __future__ import annotations

import logging
import random

from pydantic import BaseModel, Field

from .types import AdjudicatedFinding, ArmOutput, PaperResult

logger = logging.getLogger("whetstone.bench")


_RUBRIC = """RUBRIC:
- bucket: which review(s) raised this issue. "both" = both raised it (match
  issues that are the same even when worded differently); "a_only" = only the
  first review; "b_only" = only the second review.
- severity: "critical" = a problem that undermines a central claim, result, or
  the soundness of the work (unsupported headline claim, methodological flaw,
  internal contradiction, missing control the conclusions depend on).
  "minor" = style, clarity, or a peripheral point.
- locality: "cross_section" = recognising it requires reading and connecting
  MULTIPLE sections (e.g. abstract claim vs results). "local" = visible within
  a single section."""

_JUDGE_PROMPT = (
    """You are aligning two independent reviews of the SAME manuscript and \
classifying every distinct issue they raise.

Review A (a section-by-section reviewer) and Review B (a whole-document \
reviewer) are given below. Produce ONE entry per distinct issue: its text, \
which review(s) raised it, and its severity and locality.

"""
    + _RUBRIC
)


class _AdjudicationList(BaseModel):
    items: list[AdjudicatedFinding] = Field(default_factory=list)


def _format_arm(out: ArmOutput) -> str:
    if not out.findings:
        return "  (no findings)"
    return "\n".join(f"  - {f.title}: {f.detail}" for f in out.findings)


async def adjudicate(
    arm_a: ArmOutput, arm_b: ArmOutput, *, model: str
) -> list[AdjudicatedFinding]:
    from andamentum.core.agents import AgentDefinition, build_pydantic_ai_agent
    from andamentum.core.models import resolve_model

    defn = AgentDefinition(
        name="bench_judge",
        prompt=_JUDGE_PROMPT,
        output_model=_AdjudicationList,
        retries=2,
        output_retries=2,
    )
    agent = build_pydantic_ai_agent(defn, resolve_model(model))
    prompt = (
        f"REVIEW A (section-by-section):\n{_format_arm(arm_a)}\n\n"
        f"REVIEW B (whole-document):\n{_format_arm(arm_b)}\n\n"
        f"Classify every distinct issue."
    )
    result = await agent.run(prompt)
    items = result.output.items  # type: ignore[attr-defined]
    logger.info("[judge] %d adjudicated issue(s)", len(items))
    return list(items)


class _VerdictMatch(BaseModel):
    matches: bool = Field(
        description="True if A's synthesis covers the same central problems as "
        "B's top weaknesses."
    )


async def judge_verdict_match(a_verdict: str, b_verdict: str, *, model: str) -> bool:
    from andamentum.core.agents import AgentDefinition, build_pydantic_ai_agent
    from andamentum.core.models import resolve_model

    defn = AgentDefinition(
        name="bench_verdict_match",
        prompt=(
            "You compare two summaries of a manuscript's main problems. Decide "
            "whether the FIRST identifies the same central weaknesses as the "
            "SECOND (it may add more, but must cover the core ones)."
        ),
        output_model=_VerdictMatch,
        retries=2,
        output_retries=2,
    )
    agent = build_pydantic_ai_agent(defn, resolve_model(model))
    result = await agent.run(
        f"FIRST (whetstone synthesis):\n{a_verdict}\n\n"
        f"SECOND (whole-document top weaknesses):\n{b_verdict}"
    )
    return bool(result.output.matches)  # type: ignore[attr-defined]


# ── Blinded human worksheet ─────────────────────────────────────────────────

_BUCKET_TO_SYSTEM = {"a_only": "A", "b_only": "B"}


def build_worksheet(
    result: PaperResult, *, seed: int, sample_minor: int = 3
) -> tuple[str, dict[str, str]]:
    """Markdown worksheet + de-blinding key for one paper.

    Includes every ``b_only`` critical item (the headline subset) plus a small
    sample of others, with the arm relabelled System 1/2 via a per-paper random
    mapping so the human's judgement is unbiased. The key maps System↔arm.
    """
    rng = random.Random(f"{result.paper.slug}:{seed}")
    # Per-paper blind mapping of arm → System label.
    flip = rng.random() < 0.5
    arm_to_system = (
        {"A": "System 1", "B": "System 2"}
        if flip
        else {
            "A": "System 2",
            "B": "System 1",
        }
    )
    key = {v: k for k, v in arm_to_system.items()}  # System label → arm

    headline = [
        f
        for f in result.adjudications
        if f.bucket == "b_only" and f.severity == "critical"
    ]
    others = [f for f in result.adjudications if f not in headline]
    rng.shuffle(others)
    selected = headline + others[:sample_minor]

    lines = [
        f"## {result.paper.slug}",
        "",
        "For each issue, mark whether you agree it is real, and your own "
        "severity/locality. Systems are anonymised.",
        "",
    ]
    for i, f in enumerate(selected, 1):
        if f.bucket == "both":
            who = "both systems"
        else:
            who = arm_to_system[_BUCKET_TO_SYSTEM[f.bucket]] + " only"
        lines += [
            f"### {i}. {f.text}",
            f"- raised by: **{who}**",
            f"- judge tags: {f.severity}, {f.locality}",
            "- your verdict (real? severity? locality?): ____",
            "",
        ]
    return "\n".join(lines), key
