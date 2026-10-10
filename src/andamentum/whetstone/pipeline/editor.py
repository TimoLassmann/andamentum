"""Editor phase.

One LLM call per section (≤5 concurrent). Each call returns 0–8
``EditProposal`` records — verbatim original + proposed rewrite +
rationale + severity + confidence. Proposals whose ``original_text``
cannot be anchored against the section source are silently dropped
(LLM hallucination guard). The remaining proposals become
:class:`andamentum.whetstone.schemas.Edit` objects ready for the
docx track-changes renderer.

Public API:

    DEFAULT_EDITOR_CRITERIA          — module-level default list
    EditProposal, EditorOutput       — LLM I/O schemas
    EDITOR_PROMPT                    — agent prompt (calibrated on
                                       small Ollama models)
    run_editor(sections, *, criteria, agent_model) -> list[Edit]
    edit_findings(sections, findings, *, model, criteria) — one pass over
                                       several reviewers' findings

Off by default. Wired into the graph via the optional ``editor=True``
kwarg on :func:`run_review`.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Literal, cast

from pydantic import BaseModel, Field
from pydantic_ai import ModelRetry, RunContext
from pydantic_ai.exceptions import ModelHTTPError, UnexpectedModelBehavior

from andamentum.core.agents import AgentDefinition, build_pydantic_ai_agent
from andamentum.core.models import resolve_model

from ..schemas import Edit
from ..schemas import Finding as WFinding
from .locate import locate
from .model import Section, Span
from .prompt_blocks import brief_block
from .review import Finding

logger = logging.getLogger("andamentum.whetstone.pipeline")


_MAX_CONCURRENT_EDITORS: int = 5
DEFAULT_EDITOR_CRITERIA: list[str] = ["clarity", "concision", "grammar"]


# ── LLM I/O schemas ─────────────────────────────────────────────────────────

EDITOR_PROMPT = """You are a careful editor proposing concrete rewrites to ONE section of a draft.

You receive:
  • the section's full text
  • a list of editorial criteria to apply (e.g. clarity, concision, grammar)

Your job: emit 0–8 EditProposal records. Each is a CONCRETE rewrite of a
specific span — the original text VERBATIM and your proposed replacement.

# What to look for

Apply these checks as you read, scaled to the criteria in your input:

## Grammar & spelling
  • Subject-verb agreement: "the data show" (not "shows"); "each of the
    students has" (not "have")
  • Verb-tense consistency within a paragraph
  • Pronoun-antecedent agreement; clear antecedents (no orphan "this"/"it")
  • Pronoun case: "between you and me" (not "I")
  • Sentence fragments, run-ons, comma splices
  • Parallel structure in lists and series
  • Dangling and misplaced modifiers
  • Apostrophe usage in possessives and contractions
  • Standard punctuation and capitalisation

## Academic style
  • Eliminate filler words — every word should contribute
  • Replace vague language with specific terms ("many studies" →
    "17 of 23 studies"; "significantly" → "by 34%")
  • Use active voice where it clarifies agency
  • Strengthen weak verb constructions ("performed an analysis of" →
    "analysed")
  • Match certainty of claims to strength of evidence; do not over-hedge
    OR over-claim
  • Eliminate excessive nominalisation that hides verbs

## Polish & consistency
  • Consistent terminology, formatting, and capitalisation
  • Strengthen transitions between sentences and paragraphs
  • Uniform citation style
  • Remove distracting small inconsistencies

# Hard rules

  • original_text MUST be copied verbatim from the section. Do not
    paraphrase, summarise, or reword the original. Copy exactly.
  • new_text is your proposed rewrite of that span. It can be longer or
    shorter, but should fit naturally in the surrounding prose.
  • One edit = one focused change. Don't bundle multiple unrelated rewrites
    into one EditProposal — split them.
  • Rationale: ONE short, plain sentence per edit, written to the author,
    saying what was wrong with the original ("'Significantly' gave no size;
    the number does"). Do not name an editorial criterion.
  • severity: minor (cosmetic), moderate (clarity/correctness), major
    (changes meaning, removes ambiguity, fixes an error).
  • confidence: low (judgement call), medium (clearly better), high
    (objective improvement — typo, broken sentence, unambiguous error).

# Soft rules

  • If the section is already well-written, return an empty edits list.
    Returning empty is honest; returning weak edits to look productive
    wastes the author's time.
  • Don't propose edits that remove substantive content. You're sharpening
    prose, not censoring claims.
  • Don't propose edits that change the technical meaning unless the
    original is clearly wrong (and then mark severity=major).
  • Preserve the author's voice. Don't standardise informal language
    that's working as a rhetorical choice.
  • Don't change discipline-specific terminology unless it's used
    incorrectly.

# Reviewer findings

When the input lists REVIEWER FINDINGS, they are the priority. Another reader
has already gone through this document and said what is wrong with it; your
job is to answer them where a rewrite can, and to say plainly where it cannot.

  • Emit an entry for every listed finding, with `addresses` set to its
    number. One entry is the usual answer. Emit a second only when the
    finding names a problem in two SEPARATE places that need two rewrites.
  • Set can_fix=false when rewriting a span cannot answer the finding — it
    needs data you do not have, a new paragraph, a restructure, or a decision
    only the author can make. In `why`, say in a few plain words what a
    rewrite cannot supply ("the value is not in the text", "the reviewers
    disagree"). Do NOT restate the finding's fix — the author already has it.
    THIS IS THE RIGHT
    ANSWER FOR MOST STRUCTURAL FINDINGS, and it is more useful than a
    cosmetic change that leaves the problem in place while looking addressed.
  • When you can fix it, the rewrite must actually answer the finding. If your
    change would leave a reader with the same objection, set can_fix=false
    instead.
  • Two findings on the SAME sentence: produce one rewrite that answers both,
    attached to the lower-numbered one, and set can_fix=false on the other
    with `why` naming the entry that covers it. Never two rewrites of one
    sentence — only one can be applied and the other is discarded.
  • Two findings that pull in OPPOSITE directions — one saying a claim is
    overstated, another saying it is undersold — cannot both be satisfied.
    Set can_fix=false on both and say so. Choosing between them is the
    author's decision, not yours.
  • Findings outrank the editorial criteria. Where concision would undo what
    a finding asks for, the finding wins and you skip the style edit.

# Your own editorial pass

Edits of your own (`addresses=0`) are a SEPARATE job, and the findings do not
reduce it. Read the section again against the editorial criteria and propose
the edits you would have proposed had the finding list been empty — no more
than that, and no fewer. A section with three findings still gets its grammar
and concision pass, and a finding you had to decline does not cancel it.

One limit: an edit of your own must not touch a span a finding-directed
rewrite already covers. Only one change can be applied to a span, so the
second would be discarded.

Output an EditorOutput holding BOTH — your answer to every finding, and your
own edits. Either may be empty on its own merits, and neither is empty
because the other one is long."""


class EditProposal(BaseModel):
    """One response: a rewrite of a verbatim span, or a reasoned refusal.

    Findings and free-standing style edits share one flat list rather than
    having a structure each, because small models fill a flat list reliably
    and lose their place in nested ones.
    """

    addresses: int = Field(
        default=0,
        description=(
            "The number of the REVIEWER FINDING this responds to, as listed in "
            "the input. Use 0 for an edit of your own that no finding asked "
            "for."
        ),
    )
    can_fix: bool = Field(
        default=True,
        description=(
            "False when the finding cannot be answered by rewriting a span — "
            "it needs new content, a restructure, or a decision only the "
            "author can make. Then leave original_text and new_text empty. "
            "Saying so is the correct answer, not a failure."
        ),
    )
    title: str = Field(
        default="",
        description="Short label for this edit, e.g. 'Tighten sentence'.",
    )
    rationale: str = Field(
        description=(
            "One plain sentence to the author: what was wrong with the "
            "original. Do not name an editorial criterion."
        )
    )
    why: str = Field(
        default="",
        description=(
            "When can_fix is false: a few plain words on what a rewrite "
            "cannot supply — e.g. 'the value is not in the text', 'needs a "
            "new paragraph', 'the reviewers disagree'. Not the fix; the "
            "author already has that."
        ),
    )
    severity: Literal["minor", "moderate", "major"] = Field(
        default="minor",
        description=(
            "minor = cosmetic. moderate = clarity/correctness. "
            "major = changes meaning, removes ambiguity, or fixes an "
            "actual error."
        ),
    )
    confidence: Literal["low", "medium", "high"] = Field(
        default="medium",
        description=(
            "low = judgement call. medium = clearly better. "
            "high = objective improvement (typo, broken sentence, "
            "unambiguous error)."
        ),
    )
    original_text: str = Field(
        default="",
        description=(
            "Verbatim span from the section to replace. Copy exactly. Empty "
            "when can_fix is false."
        ),
    )
    new_text: str = Field(
        default="",
        description="Your proposed rewrite of that span. Empty when can_fix is false.",
    )


class EditorOutput(BaseModel):
    """Editor agent's flat output for ONE section."""

    edits: list[EditProposal] = Field(
        default_factory=list,
        description=(
            "0–8 edits for this section. Quality over quantity — return "
            "an empty list if the section is already well-written. "
            "Don't propose weak edits to look productive."
        ),
    )


# ── Output validation ───────────────────────────────────────────────────────

# How many times the validator pushes back before it accepts what it has. Two
# gives the model a first correction and a second look; a third rarely changed
# anything in the review agent and costs a full generation on a local model.
_VALIDATOR_ATTEMPTS = 2


def make_edit_validator(
    section_text: str,
    finding_count: int,
    *,
    max_attempts: int = _VALIDATOR_ATTEMPTS,
):
    """Tell the editor what it got wrong, instead of discarding it in silence.

    Every rule this schema depends on lives in the prompt, because none of it
    is expressible in JSON: ``addresses`` must index the finding list, a
    refusal must leave its text fields empty, a rewrite must quote the section
    character for character, and every listed finding must get an answer. A
    small model breaks one of them and the old code dropped the entry without
    telling anyone — the finding it was answering came back as "the proposed
    rewrite did not quote the document verbatim", which is true and useless.

    This is :func:`~andamentum.whetstone.pipeline.review.make_anchor_validator`
    applied to the editor's shape, and it is the same bargain: entries that
    pass are LOCKED and survive whatever the model does next, so a retry can
    only add. The review agent adopted lock-and-refine after smoke logs caught
    the model reintroducing errors into quotes it had already got right; there
    is no reason to expect the editor to behave differently.

    Locked by ``(addresses, original_text)`` rather than by finding number, so
    a finding answered in two separate sentences keeps both entries.
    """
    locked: dict[tuple[int, str], EditProposal] = {}

    async def validator(ctx: RunContext[Any], output: EditorOutput) -> EditorOutput:
        if ctx.partial_output:
            return output

        problems: list[str] = []
        for prop in output.edits:
            if not 0 <= prop.addresses <= finding_count:
                problems.append(
                    f"addresses={prop.addresses} on {prop.title!r}: the list "
                    f"has {finding_count} finding(s), so use 1-{finding_count} "
                    f"or 0 for an edit of your own"
                )
                continue
            if prop.addresses and not prop.can_fix:
                locked.setdefault((prop.addresses, ""), prop)
                continue
            if not prop.original_text.strip():
                problems.append(
                    f"{prop.title!r}: can_fix is true but original_text is "
                    f"empty — quote the span you are replacing, or set "
                    f"can_fix=false and say why"
                )
                continue
            key = (prop.addresses, prop.original_text)
            if key in locked or locate(prop.original_text, section_text) is not None:
                locked.setdefault(key, prop)
                continue
            problems.append(
                f"not found in the section, copy it exactly: "
                f"{prop.original_text[:100]!r}"
            )

        answered = {number for number, _ in locked if number}
        unanswered = [n for n in range(1, finding_count + 1) if n not in answered]

        if (problems or unanswered) and ctx.retry < max_attempts:
            parts = [
                f"LOCKED ({len(locked)} entry/entries already accepted — they "
                f"are kept whatever you send next. DO NOT re-emit them.)"
            ]
            if problems:
                listed = "\n".join(f"  - {p}" for p in problems[:6])
                parts.append(f"FIX THESE ({len(problems)}):\n{listed}")
            if unanswered:
                parts.append(
                    f"NOT YET ANSWERED: finding(s) "
                    f"{', '.join(str(n) for n in unanswered)}. Every listed "
                    f"finding needs an entry — a rewrite, or can_fix=false "
                    f"with a reason in `why`."
                )
            logger.info(
                "[editor.validator] attempt %d: %d locked, %d problem(s), "
                "%d unanswered%s",
                ctx.retry,
                len(locked),
                len(problems),
                len(unanswered),
                # Name the first one. A count alone told us qwen3.8 broke a
                # rule and not which of the four, which is the question you
                # need answered when you read the log a month later.
                f" — first: {problems[0]}" if problems else "",
            )
            raise ModelRetry("\n\n".join(parts))

        if problems or unanswered:
            logger.info(
                "[editor.validator] attempts spent: keeping %d, dropped %d "
                "malformed, %d finding(s) unanswered",
                len(locked),
                len(problems),
                len(unanswered),
            )
        return EditorOutput(edits=list(locked.values()))

    return validator


# ── Per-section runner ──────────────────────────────────────────────────────


@dataclass(frozen=True)
class Declined:
    """A finding the editor could not answer with a rewrite, and why.

    Recorded rather than dropped. A finding with no edit is work left to the
    author, and saying which ones those are — with the reason — is the
    difference between a review that answers itself and one that only looks
    like it did.
    """

    section_id: str
    criterion: str
    issue: str
    why: str
    finding_id: str = ""  # the id of the finding declined, for the renderers


def _passage(finding: Finding, section: Section) -> str:
    """The document's own words for what a finding points at.

    Falls back to the stored quote only when the finding has no span — which
    in review mode means it is document-level, and there is no passage.
    """
    span = finding.span
    if span is None:
        return finding.quote
    start = span.start - section.start
    end = span.end - section.start
    if 0 <= start < end <= len(section.text):
        return section.text[start:end]
    # A span outside the section it was routed to is a bug upstream, not
    # something to paper over silently.
    logger.warning(
        "[editor] finding span %d-%d falls outside section %s (%d-%d); "
        "showing the reviewer's quote instead",
        span.start,
        span.end,
        section.id,
        section.start,
        section.end,
    )
    return finding.quote


async def _run_editor_on_section(
    section: Section,
    criteria: list[str],
    findings: list[Finding],
    *,
    agent_model: str,
    brief: str = "",
) -> tuple[list[Edit], list[Declined]]:
    """One LLM call → the edits it proposed and the findings it declined.

    Offsets come back in DOCUMENT coordinates. An EditProposal whose
    original_text cannot be located in the section is dropped — the agent
    paraphrased instead of quoting, and a span that is not in the source
    cannot be anchored.

    *brief* is the active reviewer profile's stance, rendered at the top of the
    user prompt by :func:`~.prompt_blocks.brief_block`. Empty by default and
    then rendered as nothing, so the default reviewer's editor prompt is
    byte-identical to the one this function built before profiles existed.
    """
    defn = AgentDefinition(
        name="editor",
        prompt=EDITOR_PROMPT,
        output_model=EditorOutput,
        retries=2,
        # Two validator pushes plus a reserve for pydantic-ai's own
        # structured-output coercion, matching the review agent's budget.
        # Two was too tight to correct anything: a small model that broke one
        # coupling rule spent both attempts on the schema and the section came
        # back empty.
        output_retries=4,
    )
    model = resolve_model(agent_model)
    agent = build_pydantic_ai_agent(defn, model)

    # In document order, so two findings on one sentence are adjacent in the
    # list and the agent can see that it must answer them with one rewrite.
    ordered = sorted(findings, key=lambda f: f.span.start if f.span else 0)
    if ordered:
        # The passage comes from the SECTION, not from `finding.quote`. The
        # quote is the reviewer's transcription and is a lossy copy of what it
        # points at — measured on paper/FINDINGS.md, 3 of 7 differed from the
        # text at their own span (a dropped `**`, a collapsed line break,
        # altered indentation). The anchoring step recovered the true location
        # regardless, so the span is authoritative and the quote is not.
        # Showing the transcription invites the editor to echo it back and
        # rewrite text the document does not contain.
        listed = "\n".join(
            f"{number}. [{finding.criterion}/{finding.severity}] {finding.issue}"
            + (f" Suggested fix: {finding.fix}" if finding.fix else "")
            + f'\n   concerning: "{_passage(finding, section)}"'
            for number, finding in enumerate(ordered, start=1)
        )
        findings_block = (
            f"REVIEWER FINDINGS for this section — answer each one:\n{listed}\n\n"
            f"Answer every finding above. THEN, separately, edit this section "
            f"against the editorial criteria with addresses=0, exactly as you "
            f"would if the list above were empty. The findings do not reduce "
            f"that second pass.\n\n"
        )
    else:
        findings_block = (
            "REVIEWER FINDINGS for this section: none. Apply the editorial "
            "criteria alone, with addresses=0 on every entry.\n\n"
        )

    user_prompt = (
        f"{brief_block(brief)}"
        f"SECTION TITLE: {section.title}\n"
        f"SECTION ID: {section.id}\n\n"
        f"{findings_block}"
        f"EDITORIAL CRITERIA TO APPLY:\n"
        f"{', '.join(criteria) or '(none specified — use general editing judgement)'}\n\n"
        f"SECTION TEXT (quote VERBATIM from below — copy original_text exactly):\n"
        f"--- BEGIN ---\n"
        f"{section.text}\n"
        f"--- END ---\n\n"
        f"An empty list is fine if there are no findings and the section is "
        f"already well-written."
    )
    # Registered per call: the lock is closure state and must start empty for
    # each section. The fallback agent shares it, so entries the first run
    # accepted stay accepted.
    validator = make_edit_validator(section.text, len(ordered))
    agent.output_validator(validator)

    try:
        res = await agent.run(user_prompt)
    except (UnexpectedModelBehavior, ModelHTTPError) as exc:
        # gemma4 on Ollama returns an empty reply to a tool-call request often
        # enough (measured: 6 of 24 editor calls) to spend every output retry;
        # the same request without a tool always came back with text. So the
        # schema goes in the prompt and the run is made again, as
        # core.agents.AgentRunner does.
        logger.info(
            "[editor] section %s: tool-based output failed (%s), falling back "
            "to PromptedOutput",
            section.id,
            exc,
        )
        fallback = build_pydantic_ai_agent(defn, model, prompted=True)
        fallback.output_validator(validator)
        res = await fallback.run(user_prompt)
    from ._metrics import bump_from_result

    bump_from_result(res)
    output = cast(EditorOutput, res.output)

    edits: list[Edit] = []
    declined: list[Declined] = []
    # Which findings got any response at all. Used ONLY to notice silence — a
    # second rewrite for the same finding is not dropped here. One finding can
    # legitimately need two changes in different sentences, and where two
    # rewrites really do collide the surgery layer refuses one and reports it.
    # Deduplicating by finding number as well threw away edits that never
    # conflicted with anything.
    answered: set[int] = set()

    logger.debug(
        "[editor] %s: %d finding(s) in, %d proposal(s) back (%d of the "
        "editor's own, %d declined)",
        section.id,
        len(ordered),
        len(output.edits),
        sum(1 for p in output.edits if p.addresses == 0),
        sum(1 for p in output.edits if not p.can_fix),
    )

    for prop in output.edits:
        addressed = (
            ordered[prop.addresses - 1] if 1 <= prop.addresses <= len(ordered) else None
        )

        if addressed is not None:
            answered.add(prop.addresses)

        if not prop.can_fix or not prop.original_text or not prop.new_text:
            if addressed is not None:
                declined.append(
                    Declined(
                        section_id=section.id,
                        criterion=addressed.criterion,
                        issue=addressed.issue,
                        why=prop.why or prop.rationale or "no reason given",
                        finding_id=addressed.id,
                    )
                )
            continue

        span = locate(prop.original_text, section.text)
        if span is None:
            logger.debug(
                "[editor] dropped unanchored edit in %s: %r",
                section.id,
                prop.original_text[:60],
            )
            if addressed is not None:
                # The finding is not answered after all: say so rather than
                # letting it look handled because an edit was attempted.
                declined.append(
                    Declined(
                        section_id=section.id,
                        criterion=addressed.criterion,
                        issue=addressed.issue,
                        why="the proposed rewrite did not quote the document verbatim",
                        finding_id=addressed.id,
                    )
                )
            continue

        start, end = span
        # The agent read one section, so `locate` answers in section
        # coordinates; `Edit` records document coordinates, because that is
        # what the docx renderer hands to the surgery layer as a span. The
        # section's own offset is the whole conversion — and doing it here,
        # once, is why a phrase that occurs in three sections still lands in
        # the one the editor was reading.
        edits.append(
            Edit(
                title=prop.title or _title_from(prop.rationale),
                severity=prop.severity,
                confidence=prop.confidence,
                rationale=prop.rationale,
                section_id=section.id,
                char_start=section.start + start,
                char_end=section.start + end,
                original_text=section.text[start:end],
                new_text=prop.new_text,
                # The link, not a copy of the finding's text: the renderers
                # show the rewrite with the finding it answers. Appending the
                # finding to the rationale said the same thing twice.
                finding_id=addressed.id if addressed is not None else None,
            )
        )

    # A finding the agent simply omitted is not answered either. Silence is
    # not a decision, and it would otherwise read as one.
    for number, finding in enumerate(ordered, start=1):
        if number not in answered:
            declined.append(
                Declined(
                    section_id=section.id,
                    criterion=finding.criterion,
                    issue=finding.issue,
                    why="the editor did not respond to this finding",
                    finding_id=finding.id,
                )
            )

    return edits, declined


# ── Orchestrator (semaphore-bounded fanout) ─────────────────────────────────


def _title_from(rationale: str) -> str:
    """A label for an edit the model left untitled: the rationale's first words.

    The title is only a label, so a model that leaves it out (some do, every
    time and after every retry) must not cost the section its edits.
    """
    words = rationale.split()
    if not words:
        return "(untitled edit)"
    return " ".join(words[:8]) + (" …" if len(words) > 8 else "")


async def run_editor(
    sections: list[Section],
    *,
    criteria: list[str],
    agent_model: str,
    findings: list[Finding] | None = None,
    brief: str = "",
) -> tuple[list[Edit], list[Declined]]:
    """Run the editor across every section, bounded by
    ``_MAX_CONCURRENT_EDITORS=5`` parallel calls. Per-section failures are
    logged and treated as 'no edits for that section' — they do not abort the
    run.

    Findings are routed to the section their span falls in, so each call sees
    the problems raised about the text in front of it and nothing else. A
    finding with no span is never sent: it names something missing, or the
    document as a whole, and inviting a rewrite for it is inviting invention.

    *brief* is the active reviewer profile's stance, passed unchanged to every
    section's call.
    """
    if not sections:
        return [], []

    by_section: dict[str, list[Finding]] = {}
    unanchored = 0
    for finding in findings or []:
        if finding.span is None:
            unanchored += 1
            continue
        by_section.setdefault(finding.span.section_id, []).append(finding)
    if unanchored:
        logger.info(
            "[editor] %d finding(s) not sent to the editor — they have no "
            "span, so there is no passage to rewrite",
            unanchored,
        )

    sem = asyncio.Semaphore(_MAX_CONCURRENT_EDITORS)

    async def _one(section: Section) -> tuple[list[Edit], list[Declined]]:
        async with sem:
            try:
                return await _run_editor_on_section(
                    section,
                    criteria,
                    by_section.get(section.id, []),
                    agent_model=agent_model,
                    brief=brief,
                )
            except Exception as exc:
                logger.warning(
                    "[editor] section %s crashed (%s) — no edits emitted",
                    section.id,
                    exc,
                )
                return [], []

    per_section = await asyncio.gather(*[_one(s) for s in sections])
    edits: list[Edit] = []
    declined: list[Declined] = []
    for section_edits, section_declined in per_section:
        edits.extend(section_edits)
        declined.extend(section_declined)

    directed = sum(1 for e in edits if e.finding_id is not None)
    logger.info(
        "[editor] %d edit(s) across %d section(s) — %d answering a finding, "
        "%d of the editor's own; %d finding(s) need the author (criteria: %s)",
        len(edits),
        len(sections),
        directed,
        len(edits) - directed,
        len(declined),
        ", ".join(criteria) or "(none)",
    )
    for entry in declined:
        logger.info(
            "[editor] no local fix — [%s] %s\n              because: %s",
            entry.criterion,
            entry.issue[:70],
            entry.why,
        )
    return edits, declined


# ── Final editor over several reviewers' findings ───────────────────────────


def _disagreement_brief(disagreements: list[str]) -> str:
    """The part of a reconciliation the editor must act on: by not acting.

    Two reviewers who disagree about a passage cannot both be answered by a
    rewrite, and choosing between them is the author's judgement. The editor
    prompt already declines two findings that pull in opposite directions when
    both are in its list; this names the disagreements the reconciliation
    found across reviewers, which a per-section list of findings would not
    make visible on its own.
    """
    if not disagreements:
        return ""
    listed = "\n".join(f"  • {d}" for d in disagreements)
    return (
        "The reviewers DISAGREE on the following points. Do not rewrite a "
        "passage to settle one of them: set can_fix=false on any finding that "
        "takes a side, and say that the reviewers disagree. The author "
        "chooses.\n"
        f"{listed}"
    )


def to_editor_finding(finding: WFinding) -> Finding | None:
    """A rendered finding back into the shape the editor reads.

    The public :class:`~andamentum.whetstone.schemas.Finding` is what a round
    of review returns; the editor reads the pipeline's own
    :class:`~.review.Finding`. The mapping is the inverse of
    :func:`~.synth._to_wfinding` — criterion from the category, headline from
    the title, issue from the rationale, span from the first quote — and it
    loses nothing the editor uses. The id is carried across, so the edits it
    produces link back to the finding the reader sees. A finding with no quote
    names something missing or the document as a whole, and
    :func:`run_editor` would not send it to any section, so it is dropped here
    with the same reasoning.

    The reviewer's identity is written into the issue, because the editor
    reads several reviewers' findings in one list and "two reviewers said
    this" and "one reviewer said this twice" should not look alike.
    """
    if not finding.quotes:
        return None
    quote = finding.quotes[0]
    issue = finding.rationale
    if finding.perspective:
        issue = f"[{finding.perspective}] {issue}"
    return Finding(
        id=finding.id,
        criterion=finding.category or finding.title,
        headline=finding.title,
        issue=issue,
        fix=finding.fix,
        quote=quote.text,
        severity=finding.severity,
        span=Span(
            section_id=quote.section_id, start=quote.char_start, end=quote.char_end
        ),
    )


async def edit_findings(
    sections: list[Section],
    findings: list[WFinding],
    *,
    model: str,
    criteria: list[str],
    brief: str = "",
    disagreements: list[str] | None = None,
) -> tuple[list[Edit], list[Declined]]:
    """One editor pass over the findings of several reviewers.

    This is the editor a LIST of reviewers gets. Each reviewer in the list
    reads the document and raises findings; none of them rewrites anything.
    When the last has finished, this runs once over the unchanged text with
    every finding in front of it — so two findings on one sentence meet in one
    call and get one rewrite, two that pull in opposite directions are both
    declined, and the free-standing style pass happens once, over prose that
    will not move again. Per-reviewer editing did none of those: five
    reviewers each rewrote the same abstract, each re-editing the last one's
    insertions.

    *sections* are the reviewed document's own sections, in its coordinates.
    That is also why the findings can be pooled at all: with no reviewer
    editing, every round read the same text, so every finding's offsets
    address the same document. *findings* are the public findings of all
    rounds, each tagged with its reviewer through ``perspective``.
    *disagreements* are the points the reconciliation found the reviewers
    divided on, rendered into the brief as passages not to settle.
    """
    internal = [f for f in (to_editor_finding(w) for w in findings) if f is not None]
    skipped = len(findings) - len(internal)
    if skipped:
        logger.info(
            "[editor] %d finding(s) not sent to the editor — they quote no "
            "passage, so there is nothing to rewrite",
            skipped,
        )
    full_brief = "\n\n".join(
        part
        for part in (brief.strip(), _disagreement_brief(disagreements or []))
        if part
    )
    return await run_editor(
        sections,
        criteria=criteria,
        agent_model=model,
        findings=internal,
        brief=full_brief,
    )
