"""Generic criterion review (agent) + verify-findings (deterministic).

One review function, run per criterion in the active set. Each reads its
projection of the document model + asks the criterion's atomic questions, and
emits findings with verbatim quotes. VerifyFindings then locates every quote in
the source and drops any that can't be found (the hallucination gate for
findings, mirroring the digest).
"""

from __future__ import annotations

import logging
from typing import Literal, cast

from pydantic import BaseModel, Field

from andamentum.core.agents import AgentDefinition, build_pydantic_ai_agent
from andamentum.core.models import resolve_model

from .criteria import Criterion
from .locate import locate
from .model import DocumentModel, Span

logger = logging.getLogger("andamentum.whetstone.v3")

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


_PROMPT = """You are reviewing a document on ONE criterion. Work through each \
of the criterion's questions and surface every real problem you find. Don't \
pad with trivia, but also don't omit a substantive issue because you've \
already flagged a few.

You may also be shown PRIOR-STAGE FINDINGS — issues that earlier criteria in \
the cascade already raised. Don't re-list those. Where relevant, connect your \
findings to them (e.g. an evaluation gap that compounds a story-level overclaim \
already flagged, or a correctness issue that explains a presentation problem \
already noted). The cascade exists so later stages can reason across the \
document rather than re-discovering what's already on the table.

For a typical academic paper expect roughly 3-6 findings per criterion — \
fewer if the document is genuinely sound on this axis, more if real issues \
stack up. Don't artificially limit yourself; the consolidation step downstream \
will merge any near-duplicates.

For each problem: a short issue description, a VERBATIM quote from the document \
it concerns (copied exactly — non-verbatim quotes are dropped), and a severity \
(minor / moderate / major)."""


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
    prior_findings: list[Finding] | None = None,
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
    # SPECS-style cascade: prior criteria's findings are surfaced so this stage
    # can connect/build instead of re-discovering. Don't re-list them.
    prior = ""
    if prior_findings:
        lines = "\n".join(
            f"  - [{f.criterion}/{f.severity}] {f.issue}" for f in prior_findings
        )
        prior = f"\n\nPRIOR-STAGE FINDINGS (already raised — connect, don't repeat):\n{lines}"
    prompt = (
        f"CRITERION: {criterion.name}\n\nQUESTIONS:\n{questions}\n\n"
        f"{_project(criterion, model)}{full}{prior}\n\n"
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
    """Run the criterion set as a sequential SPECS-style cascade.

    Each criterion sees the accumulated findings of all earlier criteria, so
    later stages can connect to issues earlier ones raised (e.g. Correctness
    builds on Story's overclaim flag; Significance threads through Evaluations'
    baseline gap) rather than re-discovering each criterion's slice in
    isolation. This mirrors the AAAI-26 SPECS pipeline shape.

    Previously parallel; the cascade trades wall-clock for cross-criterion
    coherence — the load-bearing benefit on papers where issues thread across
    criteria. A single criterion crashing is caught and logged so the rest of
    the cascade still benefits from the findings already accumulated.
    """
    accumulated: list[Finding] = []
    for c in criteria:
        try:
            stage = await review_criterion(
                c,
                model,
                agent_model=agent_model,
                full_text=full_text,
                prior_findings=accumulated,
            )
            logger.info("[v3.review] %s → %d finding(s)", c.name, len(stage))
            accumulated.extend(stage)
        except Exception as exc:
            logger.warning("[v3.review] %s crashed: %s", c.name, exc)
            continue
    return accumulated


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
