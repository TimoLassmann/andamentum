"""Generic criterion review (agent + tools) + verify-findings (deterministic).

One review function, run per criterion in the active set. The agent receives:

- a SECTIONS table-of-contents block (id + title + size + gist),
- CLAIMS BY SECTION (verbatim spans grouped under their origin section),
- CITATIONS — how the document cites and how often, as the reference
  checker reads it (when the criterion's facets request it),
- prior-stage findings (from the SPECS cascade),

and two layer-1 tools (``read_section``, ``search_paper`` — see
``whetstone/pipeline/tools.py``) it can call to investigate the source beyond
what the digest gave it. Findings come back as ``_RawFinding``s with
verbatim quotes; VerifyFindings then locates each quote in the source
and drops any that can't be anchored (the hallucination gate).
"""

from __future__ import annotations

import logging
import uuid
from typing import Any, Protocol, cast

from pydantic import BaseModel, Field
from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.exceptions import UnexpectedModelBehavior, UsageLimitExceeded
from pydantic_ai.usage import UsageLimits

from andamentum.core.models import resolve_model
from andamentum.core.text_match import find_span

from .criteria import Criterion
from .digest import citations_line
from .locate import anchor_span
from .model import DocumentModel, Severity, Span
from .prompt_blocks import (
    ACTIONABILITY_BAR,
    FIX_FIELD,
    HEADLINE_FIELD,
    ISSUE_FIELD,
    QUOTE_FIELD,
    SECTION_IDS_NOTE,
    SEVERITY_RUBRIC,
    TOOL_DISCIPLINE,
    TOOLS_AVAILABLE,
    WRITING_RULES,
    brief_block,
    split_fix,
    claims_block,
    sections_block,
)
from .tools import DocDeps, read_section, search_paper

logger = logging.getLogger("andamentum.whetstone.pipeline")

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
    issue: str  # what is wrong here and what it costs a reader
    quote: str
    severity: Severity = "moderate"
    span: Span | None = None  # filled by verify_findings
    # The problem in a few words, for the comment's heading. Empty when the
    # producer wrote none; the public finding then falls back to the criterion.
    headline: str = ""
    fix: str = ""  # one action for the author, starting with a verb
    # Stable through the pipeline, and reused as the public ``Finding.id``, so
    # the editor's answer to a finding can name it after synthesis has run.
    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:8])


# Field order is generation order: the passage first, then what is wrong with
# it and what to do, and the headline LAST — a heading written before the
# problem is worked out summarises a guess.
class _RawFinding(BaseModel):
    quote: str = Field(description=QUOTE_FIELD)
    issue: str = Field(description=ISSUE_FIELD)
    fix: str = Field(default="", description=FIX_FIELD)
    headline: str = Field(default="", description=HEADLINE_FIELD)
    severity: Severity = "moderate"


class _CriterionFindings(BaseModel):
    findings: list[_RawFinding] = Field(default_factory=list)


# The blocks interpolated below are shared with checklist mode and live in
# ``prompt_blocks``; the prose around them is review-only. The cascade
# paragraph in particular must NOT move there — probes are independent by
# design and are never told what another probe found.
_PROMPT = f"""You are reviewing a document on ONE criterion. Work through each \
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

For each problem: a VERBATIM quote from the document it concerns (copied \
exactly — non-verbatim quotes are dropped), the issue, the fix, a short \
headline, and a severity (minor / moderate / major).

{ACTIONABILITY_BAR}

{WRITING_RULES}

{SEVERITY_RUBRIC}

{TOOLS_AVAILABLE}

Use them when:
  - the digest doesn't tell you enough to answer a criterion question;
  - prior-stage findings draw attention to a section worth reading in full;
  - you're considering flagging an absence — verify with a search first.

{TOOL_DISCIPLINE}

{SECTION_IDS_NOTE}"""


def _project(criterion: Criterion, model: DocumentModel) -> str:
    """The criterion's slice of the document model, as prompt text.

    The SECTIONS block is always emitted (regardless of the criterion's
    declared facets) because the agent needs section ids to invoke
    ``read_section``. Claims and citations remain facet-gated.

    The two block builders are shared with checklist mode; the facet gating is
    not, because a probe has no facet vocabulary (see
    ``checklist.run_probes._project``).
    """
    parts: list[str] = [sections_block(model)]

    if "claims" in criterion.facets:
        parts.append(claims_block(model))

    # CITATIONS — facet-gated. Not shared: checklist mode omits it, and does
    # not read the citations at all.
    if "citations" in criterion.facets and model.citations is not None:
        parts.append(citations_line(model.citations))

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


class _QuotedFinding(Protocol):
    """Structural type for the items in an anchor-validated output's
    ``findings`` list — ``_RawFinding`` and ``_ReexamineFinding`` both
    satisfy it. The validator needs the verbatim quote to anchor, and the
    issue to tell two findings on one passage apart."""

    quote: str
    issue: str


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
       lock. The lock is keyed by quote text ACROSS attempts and by
       ``(quote, issue)`` WITHIN one: a quote locked on an earlier attempt
       is the model re-emitting what it was told to leave alone, and its
       re-emission (usually reworded) is skipped; two findings on one quote
       in the same attempt are two observations about one passage, and
       both are kept. Keying by quote alone dropped the second of those
       silently — a lost finding, not a removed duplicate.
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
    # Closure state: quote → the findings locked on it, in arrival order.
    locked: dict[str, list[_QuotedFinding]] = {}

    async def validator(ctx: RunContext[Any], output: Any) -> Any:
        if ctx.partial_output:
            return output

        # Newly-anchored findings from THIS attempt, grouped by quote. They
        # join the lock only after the loop, so a quote first seen in this
        # attempt can carry several distinct findings, while a quote locked on
        # an earlier attempt admits nothing more (see step 1 above).
        arrived: dict[str, list[_QuotedFinding]] = {}
        for f in output.findings:
            if f.quote in locked:
                continue
            if find_span(f.quote, source) is None:
                continue
            same_quote = arrived.setdefault(f.quote, [])
            if all(g.issue.strip() != f.issue.strip() for g in same_quote):
                same_quote.append(f)
        locked.update(arrived)

        # What's in the model's current attempt that didn't anchor?
        unanchored = [f.quote for f in output.findings if f.quote not in locked]
        n_locked = sum(len(group) for group in locked.values())

        if unanchored and ctx.retry < max_attempts:
            preview = "\n".join(f"  - {q!r}" for q in unanchored[:5])
            logger.info(
                "[validator] %d locked, %d unanchored on attempt %d — refining bad ones",
                n_locked,
                len(unanchored),
                ctx.retry,
            )
            raise ModelRetry(
                f"LOCKED FINDINGS ({n_locked} already verified verbatim "
                f"in the source — these will be preserved). DO NOT re-emit "
                f"or rewrite them in your next output; the system already "
                f"has them.\n\n"
                f"FINDINGS TO FIX ({len(unanchored)} quote(s) are not "
                f"present verbatim in the source — re-quote each exactly "
                f"from the document, or remove the finding):\n{preview}"
            )

        if unanchored:
            logger.info(
                "[validator] attempts exhausted; keeping %d locked, dropping %d unanchored",
                n_locked,
                len(unanchored),
            )
        return output_class(findings=[f for group in locked.values() for f in group])

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
        retries={"tools": 2, "output": 3},
        # Pinned, not inherited: pydantic-ai v2 moved the default from 'early' to
        # 'graceful', which would run any still-pending read_section/search_paper
        # call *after* the model has already produced its findings. This is the only
        # tool-registering agent in the package, so it is the only place the new
        # default could bite. Held at v1 behaviour so the v2 uplift stays
        # behaviour-preserving; adopting 'graceful' is a separate, evaluated change.
        end_strategy="early",
    )


async def review_criterion(
    criterion: Criterion,
    model: DocumentModel,
    *,
    agent_model: str,
    prior_findings: list[Finding] | None = None,
    brief: str = "",
) -> list[Finding]:
    """Run ONE criterion against the document.

    *brief* is the active reviewer profile's stance, rendered at the top of the
    user prompt by :func:`~.prompt_blocks.brief_block`. Empty by default, which
    renders nothing at all — the default reviewer's prompt is byte-identical to
    the one this function built before profiles existed.
    """
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
        f"{brief_block(brief)}"
        f"CRITERION: {criterion.name}\n\nQUESTIONS:\n{questions}\n\n"
        f"{_project(criterion, model)}{prior}\n\n"
        f"Report real problems for the {criterion.name} criterion."
    )
    result = await agent.run(
        prompt,
        deps=DocDeps(document_model=model, label=criterion.name),
        usage_limits=UsageLimits(
            request_limit=_REQUEST_LIMIT,
            tool_calls_limit=_TOOL_CALLS_LIMIT,
            total_tokens_limit=_TOTAL_TOKENS_LIMIT,
        ),
    )
    from ._metrics import bump_from_result

    bump_from_result(result)
    raw = cast(_CriterionFindings, result.output).findings
    findings: list[Finding] = []
    for r in raw:
        issue, fix = split_fix(r.issue, r.fix)
        findings.append(
            Finding(
                criterion=criterion.name,
                issue=issue,
                quote=r.quote,
                severity=r.severity,
                headline=r.headline,
                fix=fix,
            )
        )
    return findings


async def run_criteria(
    criteria: list[Criterion],
    model: DocumentModel,
    *,
    agent_model: str,
    brief: str = "",
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

    *brief* is the reviewer profile's stance and is passed unchanged to every
    criterion in the cascade: one stance per run, not one per question.
    """
    accumulated: list[Finding] = []
    failed: list[str] = []
    for index, c in enumerate(criteria, start=1):
        # Logged BEFORE the call, not just after: a criterion that stalls or
        # loops is exactly when someone reads the log, and until it returns
        # the only output is unattributed tool lines.
        logger.info("[review] criterion %d/%d %s", index, len(criteria), c.name)
        try:
            stage = await review_criterion(
                c,
                model,
                agent_model=agent_model,
                prior_findings=accumulated,
                brief=brief,
            )
            logger.info("[review] %s → %d finding(s)", c.name, len(stage))
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
                "[review] %s: model behaviour error (%s)%s",
                c.name,
                exc,
                body_part,
            )
            failed.append(c.name)
            continue
        except UsageLimitExceeded as exc:
            logger.warning("[review] %s: usage limit hit (%s)", c.name, exc)
            failed.append(c.name)
            continue
        except Exception as exc:
            logger.warning("[review] %s crashed: %s", c.name, exc)
            failed.append(c.name)
            continue
    if failed:
        logger.warning(
            "[review] %d/%d criteria produced no findings (crashed): %s",
            len(failed),
            len(criteria),
            ", ".join(failed),
        )
    return accumulated, failed


def verify_findings(findings: list[Finding], model: DocumentModel) -> list[Finding]:
    """Locate every finding's quote in the source; drop unanchorable; set span.

    Deterministic. A finding's quote is matched against the whole source (the
    reviewer doesn't reliably know section ids); the section is resolved from
    the located offset. Both of those steps are ``locate.anchor_span`` — this
    function is the drop policy, and nothing more."""
    kept: list[Finding] = []
    for f in findings:
        span = anchor_span(f.quote, model)
        if span is None:
            logger.info(
                "[verify_findings] dropped (not in source): %r", f.quote[:60]
            )
            continue
        f.span = span
        kept.append(f)
    return kept
