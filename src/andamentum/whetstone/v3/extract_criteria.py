"""Decompose free-text guidelines / reviewer briefs into v3 Criteria.

Single async API: ``extract_criteria_from_guidelines(prose, *, model) ->
list[Criterion]``. One LLM call. Replaces v2's
``extract_checkable_items`` agent + ``custom_reviewer`` runtime-schema
mechanism with the simpler ``Criterion`` model already used by the
v3 cascade.

The cost shape: 1 extraction call + N criterion-review calls (one per
extracted criterion). This is more expensive than v2's custom mode
(which fired one combined-schema call for all criteria) but matches the
v3 cascade shape — every other v3 criterion set already runs sequentially
with each criterion getting its own call.
"""

from __future__ import annotations

import logging
from typing import cast

from pydantic import BaseModel, Field

from andamentum.core.agents import AgentDefinition, build_pydantic_ai_agent
from andamentum.core.models import resolve_model

from .criteria import Criterion

logger = logging.getLogger("andamentum.whetstone.v3")

_DEFAULT_MAX_CRITERIA: int = 8


class _ExtractedCriterion(BaseModel):
    """Flat, small-Ollama-friendly schema for one extracted criterion."""

    name: str = Field(
        description="Short title for the criterion — 1 to 4 words. e.g. "
        "'Reproducibility', 'Conflict of Interest', 'Statistical reporting'."
    )
    questions: list[str] = Field(
        description="2 to 5 atomic yes/no questions a reviewer would ask "
        "of the document to evaluate this criterion. Each question must "
        "be answerable by reading the document alone."
    )


class _ExtractionResult(BaseModel):
    """Output schema for the extractor agent."""

    criteria: list[_ExtractedCriterion] = Field(
        description="The extracted review criteria. Order matters — most "
        "important first. Aim for 4 to 8 items; cap at the max_criteria "
        "the caller requests."
    )


_PROMPT = (
    "You extract checkable review criteria from a free-text reviewer "
    "brief, journal author-guideline document, style guide, or "
    "evaluation rubric.\n\n"
    "For each genuinely distinct review axis the source describes, "
    "emit ONE criterion: a short name (1-4 words) plus 2-5 atomic "
    "yes/no questions a reviewer would ask of the document to evaluate "
    "that criterion.\n\n"
    "Quality bar:\n"
    "- Each question must be answerable by reading the document alone. "
    "Do NOT generate questions that require external knowledge (e.g. "
    "'does this match the journal's house style?').\n"
    "- Each criterion must cover a distinct axis. Do not split one axis "
    "across multiple criteria, and do not bundle multiple axes into one.\n"
    "- Do NOT invent criteria the source does not imply.\n"
    "- Prefer short names (1-4 words) over long descriptive ones.\n"
    "- Aim for 4-8 criteria total. If the source is short, fewer is fine.\n"
    "- If the source covers more than the cap, prioritise the criteria "
    "most relevant to a reviewer reading a draft.\n"
)


async def extract_criteria_from_guidelines(
    prose: str,
    *,
    model: str,
    max_criteria: int = _DEFAULT_MAX_CRITERIA,
) -> list[Criterion]:
    """Decompose ``prose`` into a ``list[Criterion]`` via one LLM call.

    Each extracted criterion becomes a ``Criterion(name=..., questions=...,
    facets=["claims", "gists"])`` ready to feed the v3 criterion cascade.

    Raises
    ------
    ValueError
        If ``prose`` is empty or whitespace-only.
    RuntimeError
        If the LLM returns no criteria (the extractor failed to find
        anything actionable in the source). Per the no-silent-failures
        rule, the caller is expected to fall back explicitly rather than
        receive a silently-empty list.
    """
    stripped = prose.strip()
    if not stripped:
        raise ValueError(
            "extract_criteria_from_guidelines: prose is empty. Pass the "
            "free-text guidelines you want decomposed into criteria, or "
            "use the document-type default by omitting guidelines_text."
        )

    defn = AgentDefinition(
        name="v3_extract_criteria",
        prompt=_PROMPT,
        output_model=_ExtractionResult,
        retries=2,
        output_retries=2,
    )
    agent = build_pydantic_ai_agent(defn, resolve_model(model))
    result = await agent.run(
        f"GUIDELINES / BRIEF:\n{stripped}\n\nExtract up to {max_criteria} criteria."
    )
    extracted = cast(_ExtractionResult, result.output).criteria

    if not extracted:
        raise RuntimeError(
            "extract_criteria_from_guidelines: the extractor returned no "
            "criteria. The source may be too short or too abstract for "
            "the model to find actionable axes. Either rewrite the "
            "guidelines with more concrete rules, pass an explicit "
            "criteria=... list, or omit guidelines_text to use the "
            "document-type default."
        )

    # Cap and wrap into Criterion objects with safe default facets.
    capped = extracted[:max_criteria]
    out = [
        Criterion(
            name=ec.name.strip() or f"Criterion {i + 1}",
            questions=[q.strip() for q in ec.questions if q.strip()],
            facets=["claims", "gists"],
        )
        for i, ec in enumerate(capped)
    ]
    logger.info(
        "[v3.extract_criteria] %d criteria extracted from %d-char prose",
        len(out),
        len(stripped),
    )
    return out
