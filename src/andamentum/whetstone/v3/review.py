"""Generic criterion review (agent + tools) + verify-findings (deterministic).

One review function, run per criterion in the active set. The agent receives:

- a SECTIONS table-of-contents block (id + title + size + gist),
- CLAIMS BY SECTION (verbatim spans grouped under their origin section),
- CITATIONS PRESENT (when the criterion's facets request it),
- prior-stage findings (from the SPECS cascade),

and two layer-1 tools (``read_section``, ``search_paper`` — see
``whetstone/v3/tools.py``) it can call to investigate the source beyond
what the digest gave it. Findings come back as ``_RawFinding``s with
verbatim quotes; VerifyFindings then locates each quote in the source
and drops any that can't be anchored (the hallucination gate).
"""

from __future__ import annotations

import logging
from typing import Literal, cast

from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.usage import UsageLimits

from andamentum.core.models import resolve_model

from .criteria import Criterion
from .locate import locate
from .model import DocumentModel, Span
from .tools import DocDeps, read_section, search_paper

logger = logging.getLogger("andamentum.whetstone.v3")

Severity = Literal["minor", "moderate", "major"]

# Per-criterion budgets. The cascade runs five criteria sequentially, so
# the per-call budget needs to be tight enough that one stalled stage
# doesn't dominate the run, but generous enough to allow a few tool
# calls when the agent wants to investigate.
#
# request_limit = total model requests (initial + tool turns + final).
# tool_calls_limit = caps tool-call iterations specifically.
# total_tokens_limit = secondary backstop against runaway prompt growth
#   when read_section drags large sections back into the context.
_REQUEST_LIMIT = 18
_TOOL_CALLS_LIMIT = 10
_TOTAL_TOKENS_LIMIT = 80_000


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
(minor / moderate / major).

You have two tools available to investigate the source beyond the digest:
  - read_section(section_id) — read a section in full when the gist isn't \
enough.
  - search_paper(query) — find where a term appears in the paper; pass \
regex=True for patterns like (limitation|caveat|weakness) or Theorem [0-9]+.

Use them when:
  - the digest doesn't tell you enough to answer a criterion question;
  - prior-stage findings draw attention to a section worth reading in full;
  - you're considering flagging an absence — verify with a search first.

Section ids are in the SECTIONS block (e.g. 4.2, abstract, sec_004)."""


def _project(criterion: Criterion, model: DocumentModel) -> str:
    """The criterion's slice of the document model, as prompt text.

    The SECTIONS block is always emitted (regardless of the criterion's
    declared facets) because the agent needs section ids to invoke
    ``read_section``. Claims and citations remain facet-gated.
    """
    parts: list[str] = []

    # SECTIONS — always emitted. Doubles as the navigation aid for
    # `read_section` (gives the agent every valid section_id) and as
    # the document outline (size hints for where the substance lives,
    # gists for semantic flavour).
    gist_by_id = {g.section_id: g.gist for g in model.gists}
    if model.sections:
        section_lines = []
        for s in model.sections:
            gist = gist_by_id.get(s.id, "").strip()
            gist_part = f" — {gist}" if gist else ""
            section_lines.append(
                f"  - [{s.id}] {s.title} ({len(s.text):,} chars){gist_part}"
            )
        parts.append("SECTIONS (id | title | size | gist):\n" + "\n".join(section_lines))
    else:
        parts.append("SECTIONS: (none)")

    # CLAIMS BY SECTION — facet-gated. Groups verbatim claims under
    # their origin section_id so the agent sees "what each section
    # asserts" without having to read every section.
    if "claims" in criterion.facets:
        claims_by_section: dict[str, list[str]] = {}
        for c in model.claims:
            claims_by_section.setdefault(c.span.section_id, []).append(c.quote)
        if claims_by_section:
            lines: list[str] = []
            # Walk sections in their declared order so the output is stable
            # and matches the SECTIONS block above.
            for s in model.sections:
                section_claims = claims_by_section.get(s.id)
                if not section_claims:
                    continue
                lines.append(f"  [{s.id}]:")
                for quote in section_claims:
                    lines.append(f"    - {quote!r}")
            parts.append("CLAIMS BY SECTION:\n" + "\n".join(lines))
        else:
            parts.append("CLAIMS BY SECTION: (none)")

    # CITATIONS PRESENT — facet-gated, unchanged from earlier.
    if "citations" in criterion.facets:
        markers = sorted({c.marker for c in model.citations})
        parts.append(f"CITATIONS PRESENT: {', '.join(markers) or '(none)'}")

    return "\n\n".join(parts)


def _build_agent(criterion: Criterion, agent_model: str) -> Agent[DocDeps, _CriterionFindings]:
    """Construct the criterion-review agent with tools + typed deps.

    Layer-1 tools (read_section, search_paper) are unconditionally
    attached for every criterion — they're free, universal, and
    pure-Python. Layer 2/3 tools (deferred) would be attached based on
    ``criterion.tools``; that field exists but is empty for SPECS today.
    """
    return Agent(
        resolve_model(agent_model),
        instructions=_PROMPT,
        output_type=_CriterionFindings,
        deps_type=DocDeps,
        tools=[read_section, search_paper],
        retries=2,
        output_retries=2,
    )


async def review_criterion(
    criterion: Criterion,
    model: DocumentModel,
    *,
    agent_model: str,
    prior_findings: list[Finding] | None = None,
) -> list[Finding]:
    agent = _build_agent(criterion, agent_model)
    questions = "\n".join(f"  {i}. {q}" for i, q in enumerate(criterion.questions, 1))
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
        f"{_project(criterion, model)}{prior}\n\n"
        f"Report real problems for the {criterion.name} criterion."
    )
    result = await agent.run(
        prompt,
        deps=DocDeps(document_model=model),
        usage_limits=UsageLimits(
            request_limit=_REQUEST_LIMIT,
            tool_calls_limit=_TOOL_CALLS_LIMIT,
            total_tokens_limit=_TOTAL_TOKENS_LIMIT,
        ),
    )
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
) -> list[Finding]:
    """Run the criterion set as a sequential SPECS-style cascade.

    Each criterion sees the accumulated findings of all earlier criteria, so
    later stages can connect to issues earlier ones raised (e.g. Correctness
    builds on Story's overclaim flag; Significance threads through Evaluations'
    baseline gap) rather than re-discovering each criterion's slice in
    isolation. This mirrors the AAAI-26 SPECS pipeline shape.

    The cascade trades wall-clock for cross-criterion coherence — the
    load-bearing benefit on papers where issues thread across criteria. A
    single criterion crashing is caught and logged so the rest of the cascade
    still benefits from the findings already accumulated.
    """
    accumulated: list[Finding] = []
    for c in criteria:
        try:
            stage = await review_criterion(
                c,
                model,
                agent_model=agent_model,
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
