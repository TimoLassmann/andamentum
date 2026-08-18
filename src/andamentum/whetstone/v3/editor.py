"""Editor phase for v3.

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
    EDITOR_PROMPT                    — agent prompt (copied verbatim
                                       from v2's editor agent — already
                                       calibrated on small Ollama models)
    run_editor(sections, *, criteria, agent_model) -> list[Edit]

Off by default. Wired into the graph via the optional ``editor=True``
kwarg on :func:`run_review_v3`.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Literal, cast

from pydantic import BaseModel, Field

from andamentum.core.agents import AgentDefinition, build_pydantic_ai_agent
from andamentum.core.models import resolve_model

from ..schemas import Edit
from .locate import locate
from .model import Section
from .review import Finding

logger = logging.getLogger("andamentum.whetstone.v3")


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
  • Rationale: ONE short sentence per edit explaining why your version
    is better, citing the criterion it addresses.
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
    only the author can make. Give one sentence in `why`. THIS IS THE RIGHT
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
        description="Short label for this edit, e.g. 'Tighten sentence'."
    )
    rationale: str = Field(description="One sentence: why your version is better.")
    why: str = Field(
        default="",
        description=(
            "When can_fix is false: one sentence on what the finding needs "
            "that a rewrite cannot give — data, a new paragraph, a "
            "restructure, or a decision only the author can make."
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


async def _run_editor_on_section(
    section: Section,
    criteria: list[str],
    findings: list[Finding],
    *,
    agent_model: str,
) -> tuple[list[Edit], list[Declined]]:
    """One LLM call → the edits it proposed and the findings it declined.

    Offsets come back in DOCUMENT coordinates. An EditProposal whose
    original_text cannot be located in the section is dropped — the agent
    paraphrased instead of quoting, and a span that is not in the source
    cannot be anchored.
    """
    defn = AgentDefinition(
        name="v3_editor",
        prompt=EDITOR_PROMPT,
        output_model=EditorOutput,
        retries=2,
        output_retries=2,
    )
    agent = build_pydantic_ai_agent(defn, resolve_model(agent_model))

    # In document order, so two findings on one sentence are adjacent in the
    # list and the agent can see that it must answer them with one rewrite.
    ordered = sorted(findings, key=lambda f: f.span.start if f.span else 0)
    if ordered:
        listed = "\n".join(
            f"{number}. [{finding.criterion}/{finding.severity}] {finding.issue}\n"
            f'   concerning: "{finding.quote}"'
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
    res = await agent.run(user_prompt)
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
        "[v3.editor] %s: %d finding(s) in, %d proposal(s) back (%d of the "
        "editor's own, %d declined)",
        section.id,
        len(ordered),
        len(output.edits),
        sum(1 for p in output.edits if p.addresses == 0),
        sum(1 for p in output.edits if not p.can_fix),
    )

    for prop in output.edits:
        addressed = ordered[prop.addresses - 1] if 1 <= prop.addresses <= len(ordered) else None

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
                    )
                )
            continue

        span = locate(prop.original_text, section.text)
        if span is None:
            logger.debug(
                "[v3.editor] dropped unanchored edit in %s: %r",
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
                    )
                )
            continue

        start, end = span
        rationale = prop.rationale
        if addressed is not None:
            rationale = f"{rationale} (addresses: {addressed.issue})"
        # The agent read one section, so `locate` answers in section
        # coordinates; `Edit` records document coordinates, because that is
        # what the docx renderer hands to the surgery layer as a span. The
        # section's own offset is the whole conversion — and doing it here,
        # once, is why a phrase that occurs in three sections still lands in
        # the one the editor was reading.
        edits.append(
            Edit(
                title=prop.title or "(untitled edit)",
                severity=prop.severity,
                confidence=prop.confidence,
                rationale=rationale,
                section_id=section.id,
                char_start=section.start + start,
                char_end=section.start + end,
                original_text=section.text[start:end],
                new_text=prop.new_text,
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
                )
            )

    return edits, declined


# ── Orchestrator (semaphore-bounded fanout) ─────────────────────────────────


async def run_editor(
    sections: list[Section],
    *,
    criteria: list[str],
    agent_model: str,
    findings: list[Finding] | None = None,
) -> tuple[list[Edit], list[Declined]]:
    """Run the editor across every section, bounded by
    ``_MAX_CONCURRENT_EDITORS=5`` parallel calls. Per-section failures are
    logged and treated as 'no edits for that section' — they do not abort the
    run.

    Findings are routed to the section their span falls in, so each call sees
    the problems raised about the text in front of it and nothing else. A
    finding with no span is never sent: it names something missing, or the
    document as a whole, and inviting a rewrite for it is inviting invention.
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
            "[v3.editor] %d finding(s) not sent to the editor — they have no "
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
                )
            except Exception as exc:
                logger.warning(
                    "[v3.editor] section %s crashed (%s) — no edits emitted",
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

    directed = sum(1 for e in edits if "(addresses:" in e.rationale)
    logger.info(
        "[v3.editor] %d edit(s) across %d section(s) — %d answering a finding, "
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
            "[v3.editor] no local fix — [%s] %s\n              because: %s",
            entry.criterion,
            entry.issue[:70],
            entry.why,
        )
    return edits, declined
