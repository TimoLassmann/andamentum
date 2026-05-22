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

from ..schemas import Finding as WFinding
from ..schemas import Quote, ReviewMetrics, ReviewResult, SectionCard
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
            f"  - ({f.criterion}/{f.severity}) {f.issue} — quote: {f.quote!r}"
            for f in findings
        )
        or "  (no findings)"
    )


async def synthesise(
    model: DocumentModel, findings: list[Finding], *, agent_model: str
) -> StructuredReview:
    defn = AgentDefinition(
        name="v3_synthesise",
        prompt=(
            "Write one concise structured review of a document from the findings "
            "and section gists. synopsis: 2-3 sentences on what the document is "
            "and its overall state. strengths / weaknesses: short bullet points; "
            "weaknesses should reflect the findings, most important first. Do not "
            "invent issues beyond the findings."
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
    return cast(StructuredReview, res.output)


async def critique_and_revise(
    model: DocumentModel, draft: StructuredReview, *, agent_model: str
) -> StructuredReview:
    """Reread the draft against the document; drop unsupported statements."""
    defn = AgentDefinition(
        name="v3_critique_revise",
        prompt=(
            "You are reviewing a draft review for faithfulness to the document. "
            "Remove or soften any statement the document's claims do not support, "
            "fix factual slips, and return the corrected structured review. Keep "
            "what is well-supported."
        ),
        output_model=StructuredReview,
        retries=2,
        output_retries=2,
    )
    agent = build_pydantic_ai_agent(defn, resolve_model(agent_model))
    claims = "\n".join(f"  - {c.quote}" for c in model.claims) or "  (none)"
    res = await agent.run(
        f"DRAFT REVIEW:\n{draft.model_dump_json(indent=2)}\n\n"
        f"DOCUMENT CLAIMS (ground truth):\n{claims}"
    )
    return cast(StructuredReview, res.output)


# ── Adapter → whetstone ReviewResult (renderer contract) ────────────────────


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
        section = model.section_by_id(f.span.section_id)
        base = section.start if section else 0
        quotes.append(
            Quote(
                section_id=f.span.section_id,
                char_start=f.span.start - base,
                char_end=f.span.end - base,
                text=f.quote,
            )
        )
    # A v3 Finding carries one `issue` statement — no separate title/rationale.
    # The criterion is the short label; the issue is the body. Mapping the same
    # text into both fields duplicates it, because every renderer shows title
    # AND rationale (docx concatenates them, markdown/html stack them).
    title = f.criterion or "Finding"
    return WFinding(
        title=title,
        severity=f.severity,
        confidence="medium",
        rationale=f.issue,
        quotes=quotes,
        sections_involved=[f.span.section_id] if f.span else [],
        source="investigate",
        category=f.criterion.lower(),
    )


def to_review_result(
    model: DocumentModel, findings: list[Finding], review: StructuredReview
) -> ReviewResult:
    gist_by_section = {g.section_id: g.gist for g in model.gists}
    return ReviewResult(
        summary=_flatten(review),
        findings=[_to_wfinding(f, model) for f in findings],
        document_map=[
            SectionCard(
                section_id=s.id,
                title=s.title,
                one_line_gist=gist_by_section.get(s.id, ""),
            )
            for s in model.sections
        ],
        metrics=ReviewMetrics(
            investigated_findings_count=len(findings),
            sections_processed=len(model.sections),
        ),
    )
