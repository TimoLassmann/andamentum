"""Run a probe bank — one agent call per probe, strictly sequential.

This is the checklist-mode analogue of :func:`..review.run_criteria`, and the
differences from it are the design, not accidents of a port.

**Probes are independent.** ``run_criteria`` threads each criterion's findings
into the next one's prompt (the SPECS cascade) so later stages can connect to
what earlier ones raised. Nothing of the sort happens here, deliberately: a
probe's output must depend only on the document and the probe, or the same bank
run twice on the same draft stops producing the same table — and that
comparability is the entire reason checklist mode exists beside review mode. It
looks like an omission, so it is called out again at the loop that would
otherwise be the natural place to add it.

**Two output schemas, not one.** ``presence`` probes return a verbatim quote;
``absence`` probes return the missing item plus the search terms that would
disprove it. A single schema carrying both (quote *or* missing_item, whichever
applies) fills badly on the small models this mode targets — they populate the
fields they were shown, so showing them four fields when two apply produces
half-filled findings. Each shape therefore gets its own flat schema and its own
closing instruction.

**Verification is not done here.** Quotes are not anchored and search terms are
not checked in this module; :mod:`.verify` owns both gates. The review
pipeline's lock-and-refine output validator is deliberately *not* attached: it
buys quote quality by spending up to two extra model requests per call and
making the output depend on retry timing, which is precisely the
nondeterminism this mode trades away. The deterministic gate downstream is the
floor, and here it is also the ceiling.

What is shared with review mode is everything that was expensive to tune: the
author-actionability bar, the severity rubric, the tool menu and the tool-use
discipline all come from :mod:`..prompt_blocks`, which both modes import. Only
the check-scoped framing and the two shape tails are written here.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import TypeVar, cast

from pydantic import BaseModel, Field, ValidationError
from pydantic_ai import Agent
from pydantic_ai.exceptions import UnexpectedModelBehavior, UsageLimitExceeded
from pydantic_ai.usage import UsageLimits

from andamentum.core.models import resolve_model

from .._metrics import bump_from_result
from ..model import DocumentModel, Severity
from ..prompt_blocks import (
    ACTIONABILITY_BAR,
    FIX_FIELD,
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
from ..tools import DocDeps, read_section, search_paper
from .probes import Probe, ProbeFinding, ProbeShape, normalise_tag

logger = logging.getLogger("andamentum.whetstone.pipeline")

# Per-probe budgets, tighter than the criterion cascade's (200 / 100 / 1M).
# A probe asks ONE narrowly-scoped question, so it needs far fewer tool turns
# than an open-ended criterion; and a bank is nine to eleven calls rather than
# five, so a single probe that starts looping must not be able to consume the
# whole run's wall-clock before the limit stops it.
#
# request_limit = total model requests (initial + tool turns + final).
# tool_calls_limit = caps tool-call iterations specifically.
# total_tokens_limit = backstop against runaway prompt growth when read_section
#   drags large sections back into the context.
_REQUEST_LIMIT = 60
_TOOL_CALLS_LIMIT = 30
_TOTAL_TOKENS_LIMIT = 400_000

_OutputT = TypeVar("_OutputT", bound=BaseModel)


# ── output schemas ──────────────────────────────────────────────────────────
#
# Field descriptions are the highest-leverage text in this module: they are
# what a small model reads when it decides what to put in a slot, and they are
# read far more reliably than an equivalent sentence buried in the prompt.


class _PresenceFinding(BaseModel):
    """One instance of a defect that is present in the text."""

    issue: str = Field(
        description="What is wrong in this passage and what it costs a "
        "reader, in 1-2 plain sentences written to the author. Do not "
        "restate the quote."
    )
    fix: str = Field(default="", description=FIX_FIELD)
    quote: str = Field(
        description="A VERBATIM span copied exactly from the document — the "
        "text the defect sits in. Copy it character-for-character; a quote "
        "that is not present in the source is discarded."
    )
    severity: Severity = "moderate"
    tag: str = Field(
        default="",
        description="One label from the allowed list in the prompt. Leave "
        "empty when no list was given or none of the labels fits.",
    )


class _AbsenceFinding(BaseModel):
    """One instance of something the document should contain and does not."""

    issue: str = Field(
        description="What is missing and what its absence costs a reader, in "
        "1-2 plain sentences written to the author."
    )
    fix: str = Field(default="", description=FIX_FIELD)
    missing_item: str = Field(
        description="The specific thing that is absent, named concretely — "
        "'a justification for the n=50 sample size', not 'methodological "
        "detail'."
    )
    search_terms: list[str] = Field(
        default_factory=list,
        description="2 to 4 concrete terms you would expect to find in the "
        "document if this item were present. They are used to verify your "
        "claim: if any of them appears, this finding is discarded.",
    )
    severity: Severity = "moderate"
    tag: str = Field(
        default="",
        description="One label from the allowed list in the prompt. Leave "
        "empty when no list was given or none of the labels fits.",
    )


class _PresenceFindings(BaseModel):
    findings: list[_PresenceFinding] = Field(default_factory=list)


class _AbsenceFindings(BaseModel):
    findings: list[_AbsenceFinding] = Field(default_factory=list)


# ── prompt assembly ─────────────────────────────────────────────────────────

# The interpolated blocks are shared with review mode and live in
# ``..prompt_blocks`` — the actionability bar, the severity rubric, the tool
# menu and the tool-use discipline were tuned against real drafts, and a change
# to any of them has to reach both modes. Only the check-scoped opening and the
# "use them when" bullets are probe-specific and written here.
_BASE_PROMPT = f"""You are running ONE named check over a document. This is not \
a general review: report only instances of the specific defect described in the \
CHECK block, and ignore every other problem the draft may have. Other checks \
cover those.

{ACTIONABILITY_BAR}

{WRITING_RULES}

{SEVERITY_RUBRIC}

{TOOLS_AVAILABLE}

Use them when:
  - the digest doesn't tell you enough to decide whether something counts as \
an instance of this check;
  - you are considering reporting something as absent — verify with a search \
first.

{TOOL_DISCIPLINE}

{SECTION_IDS_NOTE}"""

_PRESENCE_TAIL = """
For each instance report: the issue, the fix, a VERBATIM quote copied exactly \
from the document, and a severity. The quote is how the finding is anchored back to the \
draft — a quote that is not present in the source character-for-character is \
discarded, along with the finding attached to it, so copy rather than \
paraphrase."""

_ABSENCE_TAIL = """
This check is about something that is MISSING, so there is nothing to quote. \
For each instance report: the issue, the fix, the missing item itself, 2 to 4 \
search terms, and a severity. The search terms are how the finding is checked: the \
document is searched for them, and if ANY of them appears the finding is \
discarded as mistaken. Choose terms you are confident would appear if the item \
were present, and search the document yourself before reporting."""


def _instructions_for(shape: ProbeShape) -> str:
    """The static half of the prompt — everything that varies only by shape.

    Split this way so the per-probe prompt carries nothing but the probe: the
    agent is rebuilt per probe anyway, but keeping the invariant text in
    ``instructions`` makes the diff between two probes' prompts exactly the
    probe.

    Typed as ``ProbeShape`` rather than ``str`` so a third shape cannot reach
    the ``else`` arm and silently receive the absence instructions: adding one
    to the literal makes pyright reject every call site until each is updated.
    """
    return _BASE_PROMPT + (_PRESENCE_TAIL if shape == "presence" else _ABSENCE_TAIL)


def _project(model: DocumentModel) -> str:
    """The document substrate every probe is given, as prompt text.

    The blocks themselves are shared with review mode (``..prompt_blocks``);
    what differs is the selection. ``review._project`` gates claims and
    citations on the criterion's declared facets, and a probe has no facet
    vocabulary — adding one would make the bank, which is meant to be swappable
    data, depend on the criterion type.

    SECTIONS is always emitted because it is also the ``read_section``
    navigation aid — without it the agent has no valid section ids. CLAIMS is
    always emitted because "what does this document assert" is the substrate
    every probe reasons over, quota'd or not.

    Citations are omitted even though probes read them — ``unattributed_source``
    in the external-communication bank asks for exactly that. The citations
    block is a deduplicated list of markers with no positions, which answers
    "does this document cite anything" and not "does THIS statement have a
    source near it"; only the second question is the probe's, and
    ``search_paper`` answers it against the real text.
    """
    return f"{sections_block(model)}\n\n{claims_block(model)}"


def _quota_clause(probe: Probe) -> str:
    """How many instances to return — the field that carries the mode.

    A quota turns the task into a ranking with a fixed output size, which is
    what small models do well. No quota has to explicitly bless the empty list,
    or the model invents an instance rather than return nothing.
    """
    if probe.quota is not None:
        return (
            f"Find the {probe.quota} MOST SERIOUS instances. Return exactly "
            f"{probe.quota} unless the document genuinely contains fewer."
        )
    return (
        "Report every instance you find. If there are none, return an empty "
        "list — that is a valid and useful answer."
    )


def _tags_clause(probe: Probe) -> str:
    """The closed label set, when the probe declares one."""
    if not probe.tags:
        return ""
    allowed = ", ".join(probe.tags)
    return (
        f"\n\nFor each instance, set `tag` to exactly ONE of: {allowed}. "
        f"Anything outside that list is discarded."
    )


def _build_prompt(probe: Probe, model: DocumentModel, *, brief: str = "") -> str:
    """Assemble the per-probe prompt: the stance, the check, the quota, the
    substrate.

    *brief* is the active reviewer profile's stance and renders nothing when it
    is empty, which is the default and every run that names no reviewer — the
    prompt for a given probe and document is then byte-identical to the one
    this function built before profiles existed, and two such runs stay
    comparable row for row.
    """
    return (
        f"{brief_block(brief)}"
        f"CHECK: {probe.title}\n\n{probe.instruction.strip()}\n\n"
        f"{_quota_clause(probe)}{_tags_clause(probe)}\n\n"
        f"{_project(model)}"
    )


# ── agent + result conversion ───────────────────────────────────────────────


def _build_agent(
    *, output_type: type[_OutputT], instructions: str, agent_model: str
) -> Agent[DocDeps, _OutputT]:
    """Construct a probe agent with the layer-1 tools and typed deps.

    Generic over the output type so the two shapes share one construction site
    instead of drifting apart — the tool wiring, retry policy and
    ``end_strategy`` are properties of *this pipeline*, not of a shape.

    ``end_strategy="early"`` is pinned for the same reason as in
    :mod:`..review`: pydantic-ai v2 changed the default to ``graceful``, which
    would run a still-pending ``read_section`` *after* the model has already
    produced its findings.

    No output validator is registered. See the module docstring — anchoring is
    :mod:`.verify`'s deterministic job here, not a retry loop's.
    """
    return Agent(
        resolve_model(agent_model),
        instructions=instructions,
        output_type=output_type,
        deps_type=DocDeps,
        tools=[read_section, search_paper],
        retries={"tools": 2, "output": 2},
        end_strategy="early",
    )


def _finalise(
    probe: Probe, raw_items: Sequence[_PresenceFinding | _AbsenceFinding]
) -> list[ProbeFinding]:
    """Convert one probe's raw output into ``ProbeFinding``s.

    Both shapes funnel through here so tag normalisation and the malformed-
    finding gate exist once. ``ProbeFinding``'s validator enforces the shape
    invariant, so a half-filled response raises on construction; that is caught
    per finding rather than per probe, because a model that botches one of
    three findings has still produced two usable ones and dropping all three
    would lose real signal.
    """
    findings: list[ProbeFinding] = []
    dropped_tags = 0
    for item in raw_items:
        tag = normalise_tag(item.tag, probe.tags)
        # Only count a drop against a probe that HAS a closed set. An untagged
        # probe was never asked for a tag, so a model volunteering one is not a
        # failure to follow the set — counting it would corrupt the very
        # measurement the drop rate exists to provide.
        if probe.tags and item.tag.strip() and not tag:
            dropped_tags += 1
        issue, fix = split_fix(item.issue, item.fix)
        try:
            if isinstance(item, _PresenceFinding):
                finding = ProbeFinding(
                    probe_id=probe.id,
                    issue=issue,
                    fix=fix,
                    severity=item.severity,
                    tag=tag,
                    quote=item.quote,
                )
            else:
                finding = ProbeFinding(
                    probe_id=probe.id,
                    issue=issue,
                    fix=fix,
                    severity=item.severity,
                    tag=tag,
                    missing_item=item.missing_item,
                    search_terms=tuple(
                        term for term in (t.strip() for t in item.search_terms) if term
                    ),
                )
        except ValidationError as exc:
            logger.warning(
                "[checklist] %s: dropped a malformed finding (%r): %s",
                probe.id,
                item.issue[:80],
                exc,
            )
            continue
        findings.append(finding)

    if dropped_tags:
        # Not a cosmetic loss: the rate at which a model emits tags outside a
        # probe's closed set is the cheapest available read on whether that
        # model can run the bank at all, so it is reported rather than absorbed.
        logger.warning(
            "[checklist] %s: dropped %d/%d tag(s) outside the allowed set (%s)",
            probe.id,
            dropped_tags,
            len(raw_items),
            ", ".join(probe.tags),
        )
    return findings


def _first_sentence(text: str, *, limit: int = 150) -> str:
    """The opening sentence of a probe instruction, for the log line.

    Enough to recognise which check is running without reprinting the whole
    instruction, which runs to a paragraph and would bury the tool lines it is
    meant to give context to.
    """
    flat = " ".join(text.split())
    cut = flat.find(". ")
    sentence = flat[: cut + 1] if 0 < cut <= limit else flat
    if len(sentence) > limit:
        sentence = sentence[: limit - 1].rstrip() + "…"
    return sentence


def _log_probe_start(probe: Probe, *, index: int, total: int) -> None:
    """Announce a probe BEFORE it runs, naming what it is about to look for.

    Logged up front rather than only on completion. A probe holding a local
    model for minutes emits nothing but tool lines while it works, and the
    reader needs to know whose they are — which is precisely when the log is
    being read at all.
    """
    quota = f"quota {probe.quota}" if probe.quota is not None else "no quota"
    logger.info(
        "[checklist] probe %d/%d %s (%s, %s)\n    %s",
        index,
        total,
        probe.id,
        probe.shape,
        quota,
        _first_sentence(probe.instruction),
    )


# ── public surface ──────────────────────────────────────────────────────────


async def run_probe(
    probe: Probe, model: DocumentModel, *, agent_model: str, brief: str = ""
) -> list[ProbeFinding]:
    """Run ONE probe against the document — a single agent call plus tools.

    The shape branch is explicit rather than table-driven because the two paths
    differ in their output type, and collapsing them behind one loosely-typed
    schema is exactly the thing this design rejects.
    """
    prompt = _build_prompt(probe, model, brief=brief)
    deps = DocDeps(document_model=model, label=probe.id)
    limits = UsageLimits(
        request_limit=_REQUEST_LIMIT,
        tool_calls_limit=_TOOL_CALLS_LIMIT,
        total_tokens_limit=_TOTAL_TOKENS_LIMIT,
    )
    instructions = _instructions_for(probe.shape)

    if probe.shape == "presence":
        presence_agent = _build_agent(
            output_type=_PresenceFindings,
            instructions=instructions,
            agent_model=agent_model,
        )
        presence_result = await presence_agent.run(
            prompt, deps=deps, usage_limits=limits
        )
        bump_from_result(presence_result)
        return _finalise(
            probe, cast(_PresenceFindings, presence_result.output).findings
        )

    absence_agent = _build_agent(
        output_type=_AbsenceFindings,
        instructions=instructions,
        agent_model=agent_model,
    )
    absence_result = await absence_agent.run(prompt, deps=deps, usage_limits=limits)
    bump_from_result(absence_result)
    return _finalise(probe, cast(_AbsenceFindings, absence_result.output).findings)


async def run_probes(
    probes: list[Probe], model: DocumentModel, *, agent_model: str, brief: str = ""
) -> tuple[list[ProbeFinding], list[str]]:
    """Run a whole bank, one probe at a time.

    Returns ``(findings, failed_probe_ids)``. A probe that crashes is logged
    and skipped; every other probe still runs, and the caller gets the list of
    ids that contributed nothing so partial coverage can be shown as partial
    rather than presented as a clean bill of health.

    Cost is exactly one agent call per probe (plus whatever tool turns each
    one takes), which is what makes a run's price predictable before it starts.

    *brief* is the active reviewer profile's stance, passed unchanged to every
    probe in the bank — one stance per run, exactly as review mode passes one
    stance to every criterion.

    The three ``except`` arms below mirror ``..review.run_criteria``'s on
    purpose: both loops face the same three failure modes from the same model
    layer, and a fix to one (a newly-observed provider error, a better log
    line) belongs in the other. They are kept as two readable loops rather than
    one shared runner behind a callback — twenty lines of duplication is the
    cheaper price than an indirection both call sites would have to be read
    through.
    """
    collected: list[ProbeFinding] = []
    failed: list[str] = []
    for index, probe in enumerate(probes, start=1):
        _log_probe_start(probe, index=index, total=len(probes))
        # Strictly sequential, for two independent reasons.
        #
        # 1. INDEPENDENCE (the design): nothing from an earlier probe is passed
        #    into a later one. This is NOT the review cascade — no accumulated
        #    findings, no cross-probe context. It looks like a missing feature;
        #    it is the feature. A probe whose output depends on what a previous
        #    probe happened to find is no longer comparable run-to-run, and
        #    comparability is the whole product here. Two probes landing on the
        #    same passage is handled downstream by `corroborate`, which can only
        #    call that co-location meaningful because the probes could not see
        #    each other.
        # 2. SERIALISATION (the constraint): concurrent local-model calls are
        #    forbidden project-wide; Ollama serialises them anyway, so
        #    gathering would buy nothing and break the rule.
        try:
            found = await run_probe(probe, model, agent_model=agent_model, brief=brief)
            logger.info("[checklist] %s → %d finding(s)", probe.id, len(found))
            collected.extend(found)
        except UnexpectedModelBehavior as exc:
            # Per-tool retry caps, output-retry exhaustion, content filters,
            # IncompleteToolCall, and provider-side oddities (e.g. the Ollama
            # "invalid message content type: <nil>" HTTP 400) all land here.
            # When the provider attached a response body, log the first 500
            # chars — enough to identify what went wrong upstream.
            body = getattr(exc, "body", None)
            body_part = f" — body: {str(body)[:500]!r}" if body else ""
            logger.warning(
                "[checklist] %s: model behaviour error (%s)%s",
                probe.id,
                exc,
                body_part,
            )
            failed.append(probe.id)
            continue
        except UsageLimitExceeded as exc:
            logger.warning("[checklist] %s: usage limit hit (%s)", probe.id, exc)
            failed.append(probe.id)
            continue
        except Exception as exc:
            logger.warning("[checklist] %s crashed: %s", probe.id, exc)
            failed.append(probe.id)
            continue
    if failed:
        logger.warning(
            "[checklist] %d/%d probe(s) produced no findings (crashed): %s",
            len(failed),
            len(probes),
            ", ".join(failed),
        )
    return collected, failed
