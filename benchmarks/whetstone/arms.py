"""The review arms, all run on the SAME model.

Arm A — whetstone v2 (``review_document``), the chunked pipeline.
Arm B — one of:
  • ``run_arm_b``    — single whole-document critical-review prompt (baseline).
  • ``run_arm_b_v3`` — whetstone v3 (whole-document, SPECS criteria, gap loop).

All take the model as a string and resolve it through the shared
``core.models`` infrastructure, so ``ollama:…`` / ``openai:…`` / ``bedrock:…``
all work. They consume the identical harvested markdown (the path for A, the
text for B) so extraction can't confound the comparison.
"""

from __future__ import annotations

import logging
from pathlib import Path

from pydantic import BaseModel, Field

from .types import ArmFinding, ArmOutput, PaperRef

logger = logging.getLogger("whetstone.bench")


# ── Arm A: whetstone ────────────────────────────────────────────────────


async def run_arm_a(ref: PaperRef, *, model: str) -> ArmOutput:
    """Whetstone review. Findings = LLM + deterministic; verdict = synthesis.

    NOTE: post-Phase-I (v2 deletion), the v2 lens pipeline this arm
    originally compared against no longer exists. The arm now invokes
    the consolidated v3 ``review_document`` (the new canonical name);
    arm A and arm B (v3) therefore exercise the same review path and
    will produce identical output. Keep this arm in place as a guard
    against accidental regression of the public ``review_document``
    surface but expect arm A vs arm B results to converge.
    """
    from andamentum.whetstone import review_document

    assert ref.markdown_path, f"{ref.slug}: harvest before running arm A"
    result = await review_document(str(Path(ref.markdown_path)), model=model)
    findings = [
        ArmFinding(title=f.title, detail=f.rationale)
        for f in result.findings
        if f.category != "novelty"
    ]
    logger.info("[arm A] %s → %d finding(s)", ref.slug, len(findings))
    return ArmOutput(arm="A", findings=findings, verdict=result.summary or "")


# ── Arm B: whole-document baseline ──────────────────────────────────────


class _WholeDocReview(BaseModel):
    """Arm B's output schema — deliberately the same shape as A's payload."""

    findings: list[ArmFinding] = Field(
        default_factory=list,
        description="Every substantive issue with the manuscript.",
    )
    central_weaknesses: list[str] = Field(
        default_factory=list,
        description="The 3 most important problems, most critical first.",
    )


_ARM_B_PROMPT = """You are an expert, critical reviewer. You are given the \
FULL TEXT of a manuscript. Read all of it and review it as a whole — pay \
particular attention to issues that only surface across sections: claims made \
in one place that the evidence elsewhere does not support, internal \
contradictions, an abstract or introduction that overstates the results, an \
evaluation that does not test the headline claim, and missing pieces the \
conclusions depend on.

List every substantive issue as a finding (title + one or two sentences of \
detail), and give the 3 most important problems as central_weaknesses (most \
critical first). Be specific and grounded in the text; do not invent issues."""


async def run_arm_b(ref: PaperRef, *, model: str) -> ArmOutput:
    """Whole-document critical review by the same model."""
    from andamentum.core.agents import AgentDefinition, build_pydantic_ai_agent
    from andamentum.core.models import resolve_model

    assert ref.markdown_path, f"{ref.slug}: harvest before running arm B"
    text = Path(ref.markdown_path).read_text(encoding="utf-8")

    defn = AgentDefinition(
        name="whole_doc_reviewer",
        prompt=_ARM_B_PROMPT,
        output_model=_WholeDocReview,
        retries=2,
        output_retries=2,
    )
    agent = build_pydantic_ai_agent(defn, resolve_model(model))
    result = await agent.run(f"MANUSCRIPT (full text):\n\n{text}")
    out: _WholeDocReview = result.output  # type: ignore[assignment]
    verdict = "\n".join(f"{i}. {w}" for i, w in enumerate(out.central_weaknesses, 1))
    logger.info("[arm B] %s → %d finding(s)", ref.slug, len(out.findings))
    return ArmOutput(arm="B", findings=list(out.findings), verdict=verdict)


# ── Arm B (alt): whetstone v3 ───────────────────────────────────────────


async def run_arm_b_v3(ref: PaperRef, *, model: str) -> ArmOutput:
    """Whetstone v3 review (whole-document, criterion-cascade, gap loop).

    Post-Phase-I this is the same review path as arm A (the v3 graph is
    now the only review path; ``review_document`` is just an alias for
    ``run_review_v3``'s file-entry wrapper). Kept as an explicit
    benchmark arm so historical comparisons against arm A's findings
    remain interpretable.
    """
    from andamentum.whetstone import run_review

    assert ref.markdown_path, f"{ref.slug}: harvest before running arm B (v3)"
    text = Path(ref.markdown_path).read_text(encoding="utf-8")
    result = await run_review(text, model=model)
    findings = [
        ArmFinding(title=f.title, detail=f.rationale)
        for f in result.findings
        if f.category != "novelty"
    ]
    logger.info("[arm B/v3] %s → %d finding(s)", ref.slug, len(findings))
    return ArmOutput(arm="B", findings=findings, verdict=result.summary or "")
