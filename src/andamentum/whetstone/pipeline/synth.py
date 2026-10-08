"""Synthesis + self-critique, and the adapter to whetstone's ReviewResult.

Synthesise turns the gated findings into one structured review (synopsis /
strengths / weaknesses); critique-and-revise rereads that draft against the
document model and removes anything the text doesn't support. The result is
adapted into `whetstone.schemas.ReviewResult` so all three existing renderers
consume it unchanged — v3's only job is to fill that contract.
"""

from __future__ import annotations

import logging
from typing import cast

from pydantic import BaseModel, Field

from andamentum.core.agents import AgentDefinition, build_pydantic_ai_agent
from andamentum.core.models import resolve_model

from ..schemas import Edit
from ..schemas import Finding as WFinding
from ..schemas import (
    DeclinedFinding,
    NoveltyClaim,
    Quote,
    ReviewMetrics,
    ReviewResult,
    SectionCard,
)
from .model import DocumentModel
from .review import Finding

logger = logging.getLogger("andamentum.whetstone.v3")


class StructuredReview(BaseModel):
    synopsis: str = ""
    strengths: list[str] = Field(default_factory=list)
    weaknesses: list[str] = Field(default_factory=list)


def _findings_block(findings: list[Finding]) -> str:
    return (
        "\n".join(
            f"  - ({f.criterion}/{f.severity}) {f.issue}"
            + (f" Fix: {f.fix}" if f.fix else "")
            + f" — quote: {f.quote!r}"
            for f in findings
        )
        or "  (no findings)"
    )


def _synopsis_length_band(source: str) -> str:
    """Pick a synopsis length appropriate to the document size.

    Bands are tuned against the empirical observation that all four models
    obeyed the previous fixed "2-3 sentences" instruction but produced
    summaries that were too tight for ≥5000-word ML papers (could not
    name where issues clustered).

    Word-count thresholds:
      - ≤1000 words: 1 sentence (short tech notes, single-page memos)
      - 1000-5000 words: 2-4 sentences (standard paper, conference-style)
      - >5000 words: 4-8 sentences (long manuscript, multi-contribution paper)
    """
    word_count = len(source.split())
    if word_count <= 1000:
        return "1 sentence"
    if word_count <= 5000:
        return "2-4 sentences"
    return "4-8 sentences"


async def synthesise(
    model: DocumentModel, findings: list[Finding], *, agent_model: str
) -> StructuredReview:
    synopsis_length = _synopsis_length_band(model.source)
    defn = AgentDefinition(
        name="v3_synthesise",
        prompt=(
            f"Write one concise structured review of a document from the findings "
            f"and section gists. synopsis: {synopsis_length} on what the document "
            f"is and its overall state — name where the issues cluster if there "
            f"are several, not just the global stance. strengths: short bullet "
            f"points. weaknesses: short bullet points that reflect the findings, "
            f"most important first, each one plain sentence telling the author "
            f"what to fix and where. Write to the author as 'you', in plain "
            f"words, without the names of review criteria. Do not invent issues "
            f"beyond the findings."
        ),
        output_model=StructuredReview,
        retries=2,
        output_retries=2,
    )
    agent = build_pydantic_ai_agent(defn, resolve_model(agent_model))
    gists = "\n".join(f"  - {g.title}: {g.gist}" for g in model.gists)
    res = await agent.run(
        f"FINDINGS:\n{_findings_block(findings)}\n\nSECTIONS:\n{gists}"
    )
    from ._metrics import bump_from_result

    bump_from_result(res)
    return cast(StructuredReview, res.output)


async def critique_and_revise(
    model: DocumentModel,
    draft: StructuredReview,
    findings: list[Finding],
    *,
    agent_model: str,
) -> StructuredReview:
    """Reread the draft against the document; drop unsupported statements.

    Validates the three parts of the draft against three different ground
    truths:

    - **synopsis / strengths** are positive-character assertions ABOUT the
      document. Validated against ``model.gists`` (what each section is)
      and ``model.claims`` (what the author explicitly asserts). A
      strength that contradicts the gists/claims is hallucinated.
    - **weaknesses** are absence- or flaw-based reviewer notes that the
      author would NOT assert themselves. Validated against the
      ``findings`` list (which has already passed the hallucination gate
      in ``gate.py``). A weakness with no corresponding finding is
      hallucinated; a weakness that mirrors a real finding is kept even
      if the author's claims don't mention the gap.

    The previous version validated everything against ``model.claims``
    alone, which structurally cannot support absence-based weaknesses —
    the author doesn't claim what's missing. Empirical audit found
    5/13 weaknesses in the smoke runs were absence-based and 4/13
    typo/presentation; both classes would be silently deleted by the
    old critique step.
    """
    defn = AgentDefinition(
        name="v3_critique_revise",
        prompt=(
            "You are reviewing a draft review for faithfulness to the document.\n\n"
            "Use THREE ground-truth blocks:\n"
            "  - SECTION GISTS: what each section is about.\n"
            "  - AUTHOR CLAIMS: what the author explicitly asserts.\n"
            "  - SUPPORTED FINDINGS: reviewer-side observations the system has "
            "already verified against the source (the hallucination gate has "
            "already run on these).\n\n"
            "Validate the draft's three parts against these ground truths:\n"
            "  - synopsis: must be consistent with section gists and author claims. "
            "Remove or soften any sentence that contradicts them.\n"
            "  - strengths: must be supported by gists or author claims. Remove "
            "any strength the document doesn't back up.\n"
            "  - weaknesses: must correspond to a listed finding. Do NOT drop a "
            "weakness just because the author doesn't assert the gap "
            'themselves — absence-based weaknesses ("no baseline comparison", '
            '"lacks confidence intervals") and typo/presentation issues '
            '("broken notation") are exactly the kind the findings list '
            "exists to legitimise. Only drop a weakness if no finding supports "
            "it.\n\n"
            "Return the corrected structured review. Preserve what is well-supported."
        ),
        output_model=StructuredReview,
        retries=2,
        output_retries=2,
    )
    agent = build_pydantic_ai_agent(defn, resolve_model(agent_model))
    gists_block = (
        "\n".join(f"  - {g.title}: {g.gist}" for g in model.gists) or "  (none)"
    )
    claims_block = "\n".join(f"  - {c.quote}" for c in model.claims) or "  (none)"
    findings_block = _findings_block(findings)
    res = await agent.run(
        f"DRAFT REVIEW:\n{draft.model_dump_json(indent=2)}\n\n"
        f"SECTION GISTS:\n{gists_block}\n\n"
        f"AUTHOR CLAIMS:\n{claims_block}\n\n"
        f"SUPPORTED FINDINGS:\n{findings_block}"
    )
    from ._metrics import bump_from_result

    bump_from_result(res)
    return cast(StructuredReview, res.output)


# ── Adapter → whetstone ReviewResult (renderer contract) ────────────────────


def synopsis_of(summary: str) -> str:
    """The synopsis paragraph of a summary :func:`_flatten` wrote, or ``""``.

    The inverse of the first part of :func:`_flatten`, kept beside it so the
    format has one owner. A multi-reviewer summary quotes each reviewer's
    synopsis; its strengths and weaknesses stay out, because the weaknesses
    restate findings that the combined report already lists.
    """
    text = summary.strip()
    if text.startswith("## Summary"):
        text = text[len("## Summary") :].lstrip()
    return text.split("\n## ", 1)[0].strip()


def _flatten(review: StructuredReview) -> str:
    parts = [f"## Summary\n\n{review.synopsis}".rstrip()]
    if review.strengths:
        parts.append("## Strengths\n\n" + "\n".join(f"- {s}" for s in review.strengths))
    if review.weaknesses:
        parts.append(
            "## Weaknesses\n\n" + "\n".join(f"- {w}" for w in review.weaknesses)
        )
    return "\n\n".join(parts)


def _to_wfinding(f: Finding, model: DocumentModel) -> WFinding:
    quotes: list[Quote] = []
    if f.span is not None:
        # Span and Quote are both document coordinates now, so there is
        # nothing to rebase. The docx renderer passes these offsets to the
        # surgery layer unchanged; subtracting the section's start here would
        # anchor every annotation that many characters too early.
        quotes.append(
            Quote(
                section_id=f.span.section_id,
                char_start=f.span.start,
                char_end=f.span.end,
                text=f.quote,
            )
        )
    # The headline is the reviewer's own few words for the problem. The
    # criterion stays out of the title: it headed every finding it raised
    # alike, so three comments on one document all carried the same criterion
    # name as their heading, over three different complaints.
    # It is kept in ``category``. A finding with no headline (an older caller,
    # or a model that left it empty) falls back to the criterion rather than
    # repeating the issue text, which every renderer already shows as the body.
    title = f.headline.strip() or f.criterion or "Finding"
    return WFinding(
        id=f.id,
        title=title,
        severity=f.severity,
        confidence="medium",
        rationale=f.issue,
        fix=f.fix.strip(),
        quotes=quotes,
        sections_involved=[f.span.section_id] if f.span else [],
        source="investigate",
        category=f.criterion.lower(),
    )


def to_review_result(
    model: DocumentModel,
    findings: list[Finding],
    review: StructuredReview,
    edits: list[Edit] | None = None,
    declined: list[DeclinedFinding] | None = None,
    novelty_claims: list[NoveltyClaim] | None = None,
    novelty_checked: bool = False,
    llm_calls: int = 0,
    gap_rounds_used: int = 0,
    failed_criteria: list[str] | None = None,
    total_criteria: int = 0,
    soft_failure_threshold: float = 1.0,
) -> ReviewResult:
    edits = edits or []
    failed = list(failed_criteria or [])
    # Aggregate-failure gate (dialect L7): a run where most of the cascade
    # crashed is not green. Callers that don't track the cascade leave
    # total_criteria=0, which never trips the gate.
    degraded_reason = ""
    if total_criteria > 0 and len(failed) / total_criteria >= soft_failure_threshold:
        degraded_reason = (
            f"{len(failed)}/{total_criteria} review criteria crashed and "
            f"contributed no findings: {', '.join(failed)}"
        )
    gist_by_section = {g.section_id: g.gist for g in model.gists}
    return ReviewResult(
        summary=_flatten(review),
        findings=[_to_wfinding(f, model) for f in findings],
        edits=list(edits),
        declined_findings=list(declined or []),
        novelty_claims=list(novelty_claims or []),
        novelty_checked=novelty_checked,
        failed_criteria=failed,
        degraded=bool(degraded_reason),
        degraded_reason=degraded_reason,
        document_map=[
            SectionCard(
                section_id=s.id,
                title=s.title,
                one_line_gist=gist_by_section.get(s.id, ""),
            )
            for s in model.sections
        ],
        metrics=ReviewMetrics(
            llm_calls=llm_calls,
            investigated_findings_count=len(findings),
            sections_processed=len(model.sections),
            edits_count=len(edits),
            reflection_rounds_used=gap_rounds_used,
        ),
    )
