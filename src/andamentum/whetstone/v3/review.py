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
from typing import Any, Literal, cast

from pydantic import BaseModel, Field
from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.exceptions import UnexpectedModelBehavior, UsageLimitExceeded
from pydantic_ai.usage import UsageLimits

from andamentum.core.models import resolve_model
from andamentum.core.text_match import find_span

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
_REQUEST_LIMIT = 200
_TOOL_CALLS_LIMIT = 100
_TOTAL_TOKENS_LIMIT = 1_000_000


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

Every finding must be author-actionable: the issue description should name what \
the author would change, add, or verify to resolve it. If you cannot say what a \
fix would look like — even abstractly ("add a power-analysis sentence", \
"reconcile the n=50 vs n=48 mismatch", "drop the 'first-ever' claim or cite a \
prior-art search") — the finding is too vague to keep; either sharpen it or \
omit it.

Severity rubric — pick the level by what the author would have to do:
  - major: the paper's conclusions, validity, or reproducibility are at stake. \
Ignoring this would leave the work wrong, unsupported, or unusable. Author \
must address before the paper is sound.
  - moderate: a real weakness that a competent reader will notice and that \
weakens the paper, but conclusions survive if it stays. Author should fix to \
strengthen the work.
  - minor: a local improvement — wording, typo, formatting, a single sentence \
that could be sharper. Safe to ignore; nice to fix.

When uncertain between two severity tiers, pick the lower one.

You have two tools available to investigate the source beyond the digest:
  - read_section(section_id) — read a section in full when the gist isn't \
enough.
  - search_paper(query) — find where a term appears in the paper; pass \
regex=True for patterns like (limitation|caveat|weakness) or Theorem [0-9]+.

Use them when:
  - the digest doesn't tell you enough to answer a criterion question;
  - prior-stage findings draw attention to a section worth reading in full;
  - you're considering flagging an absence — verify with a search first.

Tool-use discipline:
  - Don't re-read a section you've already loaded in this conversation. \
Refer back to the earlier tool result; the section text is already in your \
context.
  - When a search returns zero hits, treat the absence as the answer. Do not \
rephrase the same negative-result query with adjacent terms — if the concept \
isn't there, broaden ONCE if needed (e.g. add a synonym) and then accept the \
finding.
  - When a search returns hits but you want more context around them, call \
read_section on the matching section rather than re-searching with a longer \
query that pulls in the surrounding text.

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
        parts.append(
            "SECTIONS (id | title | size | gist):\n" + "\n".join(section_lines)
        )
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


# Max validator-driven re-quote attempts. The validator fires on every
# structured-output completion; while ctx.retry < this value AND any
# quote fails to anchor in the source, the validator raises ModelRetry
# and the model gets another chance to re-quote. Once ctx.retry reaches
# this value, the validator stops pushing and returns only the anchored
# findings — verify_findings remains the deterministic floor.
#
# Set to 2 so the model gets up to two re-quote attempts (ctx.retry 0
# and 1) before we accept whatever anchors. output_retries on the agent
# is 3 = 2 validator retries + 1 reserve for pydantic-ai's own
# structured-output coercion.
_VALIDATOR_REQUOTE_ATTEMPTS = 2


def make_anchor_validator(
    source: str,
    output_class: type,
    *,
    max_attempts: int = _VALIDATOR_REQUOTE_ATTEMPTS,
):
    """Build an ``output_validator`` with lock-and-refine semantics.

    The previous design asked the model to regenerate its ENTIRE
    structured output when any single quote missed; smoke logs showed
    the model reintroducing errors in previously-anchored quotes during
    retry (validator fired with 4 anchored on attempt 0, then 3 anchored
    on attempt 1 — net loss of 1 good finding).

    Lock-and-refine fixes this. The returned closure accumulates
    anchored findings across retry attempts (closure state survives
    within one ``agent.run`` call but is fresh per criterion). On each
    attempt:

    1. Findings with a quote that anchors in ``source`` get added to the
       lock (keyed by quote text — duplicates are ignored).
    2. If any quote in the current attempt is still unanchored AND
       ``ctx.retry < max_attempts``, raise ``ModelRetry`` with the
       count of locked findings and the verbatim text of the unanchored
       ones. The model is told to refine ONLY the bad ones; locked
       findings will be preserved regardless of what it emits next.
    3. Otherwise return ``output_class(findings=list(locked.values()))``
       — the accumulated lock, never the most-recent attempt's output.

    The validator's contract: signal does not regress across retries.
    Anchored findings from attempt 0 survive even if the model botches
    its retry. The deterministic ``verify_findings`` pass below stays as
    the safety net.

    ``output_class`` is the structured-output type the agent returns
    (``_CriterionFindings`` for the cascade, ``_ReexamineFindings`` for
    the gap loop). Items in its ``findings`` list must have a
    ``.quote: str`` attribute.
    """
    locked: dict[str, Any] = {}  # closure state: quote → finding

    async def validator(ctx: RunContext[Any], output: Any) -> Any:
        if ctx.partial_output:
            return output

        # Add newly-anchored findings to the lock. Key by quote text so
        # a model that re-emits a previously-locked finding doesn't
        # double-count.
        for f in output.findings:
            if f.quote in locked:
                continue
            if find_span(f.quote, source) is not None:
                locked[f.quote] = f

        # What's in the model's current attempt that didn't anchor?
        unanchored = [f.quote for f in output.findings if f.quote not in locked]

        if unanchored and ctx.retry < max_attempts:
            preview = "\n".join(f"  - {q!r}" for q in unanchored[:5])
            logger.info(
                "[v3.validator] %d locked, %d unanchored on attempt %d — refining bad ones",
                len(locked),
                len(unanchored),
                ctx.retry,
            )
            raise ModelRetry(
                f"LOCKED FINDINGS ({len(locked)} already verified verbatim "
                f"in the source — these will be preserved). DO NOT re-emit "
                f"or rewrite them in your next output; the system already "
                f"has them.\n\n"
                f"FINDINGS TO FIX ({len(unanchored)} quote(s) are not "
                f"present verbatim in the source — re-quote each exactly "
                f"from the document, or remove the finding):\n{preview}"
            )

        if unanchored:
            logger.info(
                "[v3.validator] attempts exhausted; keeping %d locked, dropping %d unanchored",
                len(locked),
                len(unanchored),
            )
        return output_class(findings=list(locked.values()))

    return validator


def _build_agent(
    criterion: Criterion, agent_model: str
) -> Agent[DocDeps, _CriterionFindings]:
    """Construct the criterion-review agent with tools + typed deps.

    Layer-1 tools (read_section, search_paper) are unconditionally
    attached for every criterion — they're free, universal, and
    pure-Python. Layer 2/3 tools (deferred) would be attached based on
    ``criterion.tools``; that field exists but is empty for SPECS today.

    ``output_retries=3`` makes room for the lock-and-refine validator's
    two re-quote attempts (see ``_VALIDATOR_REQUOTE_ATTEMPTS``) plus one
    reserve for pydantic-ai's own structured-output coercion. The
    validator itself is registered in ``review_criterion`` per call,
    because its closure state (the lock) must reset per criterion.
    """
    return Agent(
        resolve_model(agent_model),
        instructions=_PROMPT,
        output_type=_CriterionFindings,
        deps_type=DocDeps,
        tools=[read_section, search_paper],
        retries=2,
        output_retries=3,
    )


async def review_criterion(
    criterion: Criterion,
    model: DocumentModel,
    *,
    agent_model: str,
    prior_findings: list[Finding] | None = None,
) -> list[Finding]:
    agent = _build_agent(criterion, agent_model)
    # Lock-and-refine validator: fresh per criterion (closure state
    # must reset between criteria). Anchored findings accumulate
    # across retry attempts within this run; only unanchored ones are
    # re-requested from the model on retry.
    agent.output_validator(make_anchor_validator(model.source, _CriterionFindings))
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
    from ._metrics import bump_from_result

    bump_from_result(result)
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
) -> tuple[list[Finding], list[str]]:
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

    Returns ``(findings, failed_criteria)`` where ``failed_criteria`` names the
    criteria that crashed and contributed nothing — so the caller can surface
    the partial coverage in the result rather than presenting a silently
    incomplete review as complete.
    """
    accumulated: list[Finding] = []
    failed: list[str] = []
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
        except UnexpectedModelBehavior as exc:
            # Per-tool retry caps, output_retries exhaustion, content filter,
            # IncompleteToolCall, and provider-side oddities (e.g. the Ollama
            # "invalid message content type: <nil>" HTTP 400) all surface here.
            # When the provider attached a response body, log the first 500
            # chars — enough to identify what went wrong upstream.
            body = getattr(exc, "body", None)
            body_part = f" — body: {str(body)[:500]!r}" if body else ""
            logger.warning(
                "[v3.review] %s: model behaviour error (%s)%s",
                c.name,
                exc,
                body_part,
            )
            failed.append(c.name)
            continue
        except UsageLimitExceeded as exc:
            logger.warning("[v3.review] %s: usage limit hit (%s)", c.name, exc)
            failed.append(c.name)
            continue
        except Exception as exc:
            logger.warning("[v3.review] %s crashed: %s", c.name, exc)
            failed.append(c.name)
            continue
    if failed:
        logger.warning(
            "[v3.review] %d/%d criteria produced no findings (crashed): %s",
            len(failed),
            len(criteria),
            ", ".join(failed),
        )
    return accumulated, failed


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
