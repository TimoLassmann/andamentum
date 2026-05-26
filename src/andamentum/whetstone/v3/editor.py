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
from typing import Literal, cast

from pydantic import BaseModel, Field

from andamentum.core.agents import AgentDefinition, build_pydantic_ai_agent
from andamentum.core.models import resolve_model

from ..schemas import Edit
from .locate import locate
from .model import Section

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

Output an EditorOutput with the edits list (possibly empty)."""


class EditProposal(BaseModel):
    """One proposed edit on a verbatim span of section text."""

    title: str = Field(
        description="Short label for this edit, e.g. 'Tighten sentence'."
    )
    rationale: str = Field(description="One sentence: why your version is better.")
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
        description="Verbatim span from the section to replace. Copy exactly."
    )
    new_text: str = Field(description="Your proposed rewrite of that span.")


class EditorOutput(BaseModel):
    """Editor agent's flat output: 0–8 EditProposals for ONE section."""

    edits: list[EditProposal] = Field(
        default_factory=list,
        description=(
            "0–8 edits for this section. Quality over quantity — return "
            "an empty list if the section is already well-written. "
            "Don't propose weak edits to look productive."
        ),
    )


# ── Per-section runner ──────────────────────────────────────────────────────


async def _run_editor_on_section(
    section: Section, criteria: list[str], *, agent_model: str
) -> list[Edit]:
    """One LLM call → list[Edit] with section-local char offsets. Drops any
    EditProposal whose original_text can't be located in section.text."""
    defn = AgentDefinition(
        name="v3_editor",
        prompt=EDITOR_PROMPT,
        output_model=EditorOutput,
        retries=2,
        output_retries=2,
    )
    agent = build_pydantic_ai_agent(defn, resolve_model(agent_model))
    user_prompt = (
        f"SECTION TITLE: {section.title}\n"
        f"SECTION ID: {section.id}\n\n"
        f"EDITORIAL CRITERIA TO APPLY:\n"
        f"{', '.join(criteria) or '(none specified — use general editing judgement)'}\n\n"
        f"SECTION TEXT (quote VERBATIM from below — copy original_text exactly):\n"
        f"--- BEGIN ---\n"
        f"{section.text}\n"
        f"--- END ---\n\n"
        f"Emit 0–8 EditProposals. An empty list is fine if the section "
        f"is already well-written."
    )
    res = await agent.run(user_prompt)
    from ._metrics import bump_from_result

    bump_from_result(res)
    output = cast(EditorOutput, res.output)

    edits: list[Edit] = []
    for prop in output.edits:
        if not prop.original_text or not prop.new_text:
            continue
        span = locate(prop.original_text, section.text)
        if span is None:
            logger.debug(
                "[v3.editor] dropped unanchored edit in %s: %r",
                section.id,
                prop.original_text[:60],
            )
            continue
        start, end = span
        edits.append(
            Edit(
                title=prop.title or "(untitled edit)",
                severity=prop.severity,
                confidence=prop.confidence,
                rationale=prop.rationale,
                section_id=section.id,
                char_start=start,
                char_end=end,
                original_text=section.text[start:end],
                new_text=prop.new_text,
            )
        )
    return edits


# ── Orchestrator (semaphore-bounded fanout) ─────────────────────────────────


async def run_editor(
    sections: list[Section], *, criteria: list[str], agent_model: str
) -> list[Edit]:
    """Run the editor agent across every section, bounded by
    ``_MAX_CONCURRENT_EDITORS=5`` parallel calls. Per-section failures
    are logged and treated as 'no edits for that section' — they do not
    abort the run."""
    if not sections:
        return []
    sem = asyncio.Semaphore(_MAX_CONCURRENT_EDITORS)

    async def _one(s: Section) -> list[Edit]:
        async with sem:
            try:
                return await _run_editor_on_section(
                    s, criteria, agent_model=agent_model
                )
            except Exception as exc:
                logger.warning(
                    "[v3.editor] section %s crashed (%s) — no edits emitted",
                    s.id,
                    exc,
                )
                return []

    per_section = await asyncio.gather(*[_one(s) for s in sections])
    flat: list[Edit] = []
    for batch in per_section:
        flat.extend(batch)
    logger.info(
        "[v3.editor] %d edits across %d sections (criteria: %s)",
        len(flat),
        len(sections),
        ", ".join(criteria) or "(none)",
    )
    return flat
