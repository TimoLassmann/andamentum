"""Generic criterion review (agent) + verify-findings (deterministic).

One review function, run per criterion in the active set. Each reads its
projection of the document model + asks the criterion's atomic questions, and
emits findings with verbatim quotes. VerifyFindings then locates every quote in
the source and drops any that can't be found (the hallucination gate for
findings, mirroring the digest).
"""

from __future__ import annotations

import asyncio
import logging
from typing import Literal, cast

from pydantic import BaseModel, Field

from andamentum.core.agents import AgentDefinition, build_pydantic_ai_agent
from andamentum.core.models import resolve_model

from .criteria import Criterion
from .locate import locate
from .model import DocumentModel, Span

logger = logging.getLogger("andamentum.whetstone.v3")

# Dropped from 5 → 2: the 5 SPECS criterion reviews all firing at once
# was the most obvious burst contributor (every review-crash log shows
# all 5 criteria crashing simultaneously).
_MAX_CONCURRENT = 2

Severity = Literal["minor", "moderate", "major"]


class Finding(BaseModel):
    criterion: str
    issue: str
    quote: str
    severity: Severity = "moderate"
    span: Span | None = None  # filled by verify_findings


class _RawFinding(BaseModel):
    issue: str = Field(description="Short description of the problem (1-2 sentences).")
    quote: str = Field(description="A VERBATIM span from the document it concerns.")
    severity: Severity = "moderate"


class _CriterionFindings(BaseModel):
    findings: list[_RawFinding] = Field(default_factory=list)


_PROMPT = """You are reviewing a document on ONE criterion. Answer the criterion's \
questions and report ONLY real problems — not a checklist of everything.

For each problem: a short issue description, a VERBATIM quote from the document \
it concerns (copied exactly — non-verbatim quotes are dropped), and a severity \
(minor / moderate / major). If the document is sound on this criterion, return \
no findings."""


def _project(criterion: Criterion, model: DocumentModel) -> str:
    """The criterion's slice of the document model, as prompt text."""
    parts: list[str] = []
    if "claims" in criterion.facets:
        claims = "\n".join(f"  - {c.quote}" for c in model.claims) or "  (none)"
        parts.append(f"CLAIMS:\n{claims}")
    if "gists" in criterion.facets:
        gists = "\n".join(f"  - {g.title}: {g.gist}" for g in model.gists) or "  (none)"
        parts.append(f"SECTION GISTS:\n{gists}")
    if "citations" in criterion.facets:
        markers = sorted({c.marker for c in model.citations})
        parts.append(f"CITATIONS PRESENT: {', '.join(markers) or '(none)'}")
    return "\n\n".join(parts)


async def review_criterion(
    criterion: Criterion,
    model: DocumentModel,
    *,
    agent_model: str,
    full_text: str | None = None,
) -> list[Finding]:
    defn = AgentDefinition(
        name=f"v3_review_{criterion.name.lower()}",
        prompt=_PROMPT,
        output_model=_CriterionFindings,
        retries=2,
        output_retries=2,
    )
    agent = build_pydantic_ai_agent(defn, resolve_model(agent_model))
    questions = "\n".join(f"  {i}. {q}" for i, q in enumerate(criterion.questions, 1))
    # "Raw text if it fits": when the caller passes the full document, include it
    # as extra grounding alongside the compact projection.
    full = f"\n\nFULL DOCUMENT:\n{full_text}" if full_text else ""
    prompt = (
        f"CRITERION: {criterion.name}\n\nQUESTIONS:\n{questions}\n\n"
        f"{_project(criterion, model)}{full}\n\n"
        f"Report real problems for the {criterion.name} criterion."
    )
    result = await agent.run(prompt)
    raw = cast(_CriterionFindings, result.output).findings
    return [
        Finding(
            criterion=criterion.name, issue=r.issue, quote=r.quote, severity=r.severity
        )
        for r in raw
    ]


async def run_criteria(
    criteria: list[Criterion],
    model: DocumentModel,
    *,
    agent_model: str,
    full_text: str | None = None,
) -> list[Finding]:
    """Fan the generic review over the active criterion set (parallel)."""
    sem = asyncio.Semaphore(_MAX_CONCURRENT)

    async def one(c: Criterion) -> list[Finding]:
        async with sem:
            try:
                return await review_criterion(
                    c, model, agent_model=agent_model, full_text=full_text
                )
            except Exception as exc:
                logger.warning("[v3.review] %s crashed: %s", c.name, exc)
                return []

    results = await asyncio.gather(*[one(c) for c in criteria])
    return [f for fs in results for f in fs]


def verify_findings(findings: list[Finding], model: DocumentModel) -> list[Finding]:
    """Locate every finding's quote in the source; drop unanchorable; set span.

    Deterministic. A finding's quote is matched against the whole source (the
    reviewer doesn't reliably know section ids); the section is resolved from
    the located offset."""
    kept: list[Finding] = []
    for f in findings:
        loc = locate(f.quote, model.source)
        if loc is None:
            logger.info(
                "[v3.verify_findings] dropped (not in source): %r", f.quote[:60]
            )
            continue
        section = next((s for s in model.sections if s.start <= loc[0] < s.end), None)
        f.span = Span(
            section_id=section.id if section else "?", start=loc[0], end=loc[1]
        )
        kept.append(f)
    return kept
