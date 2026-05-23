"""Node: ReflectAndInvestigate — the bounded reflection loop.

Replaces the old ``InvestigateLoop`` (which ran on speculative
hypotheses). The new flow runs a deep-research-style closed loop, hard-
capped at ``state.reflection_round_cap`` rounds (default 3):

  Round N:
    1. ONE reflection call. Senior reviewer sees the document map, the
       full issue pool, and the descriptions of every task already run.
       Returns up to 10 ReflectionTasks. Empty → exit loop.
    2. ONE investigator call PER task (parallel). Each call gets the
       full original text of the named sections plus the current notes
       the task is about (presented as observations to verify, not as
       facts). Returns NoteUpdates and optional NewNotes.
    3. Programmatic anchor check. Refinements with un-verifiable quotes
       are rejected; new notes with un-verifiable quotes are dropped.
    4. Apply surviving outcomes to the pool.

After the loop exits (empty reflection or round cap), control passes
to ``EditSections``.

The discipline that prevents the loop from drifting:
  • Every issue at every moment carries an anchor-verified quote. The
    chain traces back to the manuscript at every link.
  • The investigator only ever sees source text + current note state.
    It does not see prior conclusions, the reflection prompt, or other
    investigators' outputs.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

from pydantic_graph import BaseNode, GraphRunContext

from ..agents import (
    InvestigatorOutput,
    ReflectionOutput,
    ReflectionTask,
    build_pydantic_ai_agent,
)
from ..anchoring import anchor_quote
from ..deps import ReviewDeps
from ..schemas import Finding, ReviewResult
from ..state import FailedTask, ReviewState

if TYPE_CHECKING:
    from ..structural.types import SectionRef
    from .novelty_check import NoveltyCheck


logger = logging.getLogger("andamentum.whetstone")
# Dropped from 4 → 2 to avoid stale-connection / NAT-table saturation.
_MAX_CONCURRENT_INVESTIGATIONS = 2


@dataclass
class ReflectAndInvestigate(BaseNode[ReviewState, ReviewDeps, ReviewResult]):
    """Run the bounded reflect → investigate → apply loop."""

    async def run(
        self, ctx: GraphRunContext[ReviewState, ReviewDeps]
    ) -> "NoveltyCheck":
        ctx.state.current_phase = "reflect_investigate"
        cap = ctx.state.reflection_round_cap
        sections_by_id: dict[str, "SectionRef"] = {
            s.id: s for s in ctx.state.sections
        }
        notes_by_id: dict[str, Finding] = {f.id: f for f in ctx.state.findings}

        for round_idx in range(1, cap + 1):
            ctx.state.reflection_round = round_idx
            logger.info(
                "[reflect] round %d/%d — %d note(s) in pool",
                round_idx,
                cap,
                len(notes_by_id),
            )

            try:
                tasks = await _run_reflection(ctx.deps, ctx.state, notes_by_id)
            except Exception as exc:
                logger.warning("[reflect] reflection call crashed: %s", exc)
                break
            ctx.state.llm_calls += 1
            if not tasks:
                logger.info(
                    "[reflect] round %d — nothing to do, exiting loop", round_idx
                )
                break
            logger.info("[reflect] round %d — %d task(s)", round_idx, len(tasks))

            # Track tasks for the next round's reflection prompt so it can
            # avoid duplicates.
            for t in tasks:
                ctx.state.prior_task_descriptions.append(t.description)

            sem = asyncio.Semaphore(_MAX_CONCURRENT_INVESTIGATIONS)

            async def run_one(task: ReflectionTask) -> InvestigatorOutput | None:
                async with sem:
                    try:
                        return await _run_investigation(
                            ctx.deps, task, sections_by_id, notes_by_id
                        )
                    except Exception as exc:
                        logger.warning(
                            "[investigate] task crashed: %s", exc
                        )
                        ctx.state.failed_tasks.append(
                            FailedTask(description=task.description, error=str(exc))
                        )
                        return None

            results = await asyncio.gather(*[run_one(t) for t in tasks])
            ctx.state.llm_calls += sum(1 for r in results if r is not None)

            for task, result in zip(tasks, results):
                if result is None:
                    continue
                _apply_investigator_result(
                    result, task, sections_by_id, notes_by_id
                )

        ctx.state.findings = list(notes_by_id.values())
        logger.info(
            "[reflect] loop done after %d round(s) — %d note(s) survive",
            ctx.state.reflection_round,
            len(ctx.state.findings),
        )

        # Optional novelty-check phase before edit/challenge/synthesise. Gated
        # by ``state.check_novelty``; the node passes through when off.
        from .novelty_check import NoveltyCheck

        return NoveltyCheck()


# ── Reflection call ─────────────────────────────────────────────────────


async def _run_reflection(
    deps: ReviewDeps,
    state: ReviewState,
    notes_by_id: dict[str, Finding],
) -> list[ReflectionTask]:
    """One reflection call. Returns the list of tasks for this round."""
    map_lines = "\n".join(
        f"  • {c.section_id} — {c.title}: {c.one_line_gist}"
        for c in state.document_map
    ) or "  (no document map available)"

    note_lines = []
    for note in notes_by_id.values():
        quote_blurb = ""
        if note.quotes:
            q_text = note.quotes[0].text.replace("\n", " ")
            if len(q_text) > 140:
                q_text = q_text[:137] + "…"
            quote_blurb = f"\n      quote: {q_text!r}"
        section_id = note.sections_involved[0] if note.sections_involved else "?"
        note_lines.append(
            f"  [{note.id}] ({note.perspective}|{note.severity}|"
            f"{note.confidence}|{note.category}) {section_id}: {note.title}"
            f"\n      {note.rationale}{quote_blurb}"
        )

    prior_block = "\n".join(
        f"  • {desc}" for desc in state.prior_task_descriptions
    ) or "  (none — this is the first round)"

    prompt = f"""DOCUMENT MAP:
{map_lines}

CURRENT NOTES ({len(notes_by_id)} total):
{chr(10).join(note_lines) or "  (none)"}

TASKS ALREADY RUN IN EARLIER ROUNDS:
{prior_block}

This is round {state.reflection_round} of {state.reflection_round_cap}.
Decide what — if anything — is worth a closer look. Return at most 10 tasks."""

    agent = build_pydantic_ai_agent("reflection", deps.model)
    result = await agent.run(prompt)
    output = cast(ReflectionOutput, result.output)
    return output.tasks


# ── Investigation call ──────────────────────────────────────────────────


async def _run_investigation(
    deps: ReviewDeps,
    task: ReflectionTask,
    sections_by_id: dict[str, "SectionRef"],
    notes_by_id: dict[str, Finding],
) -> InvestigatorOutput:
    """One investigator call for one task. The investigator gets full
    section text(s) and current notes — nothing else."""
    cited_sections = [
        sections_by_id[sid] for sid in task.section_ids if sid in sections_by_id
    ]
    related_notes = [
        notes_by_id[nid] for nid in task.related_note_ids if nid in notes_by_id
    ]

    sections_block = "\n\n".join(
        f"--- BEGIN {s.id} ({s.title}) ---\n{s.text}\n--- END {s.id} ---"
        for s in cited_sections
    ) or "(task named no valid sections — investigator should return empty output)"

    note_lines = []
    for note in related_notes:
        section_id = note.sections_involved[0] if note.sections_involved else "?"
        note_lines.append(
            f"  [{note.id}] ({note.perspective}|{note.severity}|"
            f"{note.confidence}) {section_id}: {note.title}\n"
            f"      {note.rationale}"
        )
    notes_block = "\n".join(note_lines) or (
        "  (no related notes — task is purely about raising new ones)"
    )

    prompt = f"""TASK FROM SENIOR REVIEWER:
{task.description}

SECTION TEXT(S) — your only evidence:
{sections_block}

NOTES TO VERIFY (from junior reviewers — observations, NOT facts):
{notes_block}

Decide outcomes. Return updates for each note above and any new notes
you raise. Every quote must be verbatim from one of the section texts
above."""

    agent = build_pydantic_ai_agent("investigator", deps.model)
    result = await agent.run(prompt)
    return cast(InvestigatorOutput, result.output)


# ── Apply outcomes (anchor-checked) ─────────────────────────────────────


def _apply_investigator_result(
    result: InvestigatorOutput,
    task: ReflectionTask,
    sections_by_id: dict[str, "SectionRef"],
    notes_by_id: dict[str, Finding],
) -> None:
    """Mutate ``notes_by_id`` in place with anchor-verified outcomes.

    Refinements whose quote can't be anchored are silently rejected —
    the original note stays unchanged. New notes whose quote can't be
    anchored are dropped.
    """
    fed_section_ids = set(task.section_ids)

    # Updates first.
    for upd in result.updates:
        note = notes_by_id.get(upd.note_id)
        if note is None:
            continue  # task referenced a stale id; ignore

        if upd.action == "keep":
            continue
        if upd.action == "drop":
            del notes_by_id[upd.note_id]
            continue
        if upd.action == "refine":
            # Only allow refinement quotes from sections the task actually fed.
            if upd.refined_quote_section_id not in fed_section_ids:
                continue
            sec = sections_by_id.get(upd.refined_quote_section_id)
            if sec is None:
                continue
            quote = anchor_quote(upd.refined_quote_text, sec.text, sec.id)
            if quote is None:
                # Quote can't be verified — reject the refinement; note
                # stays unchanged.
                continue
            update_dict: dict[str, object] = {
                "title": upd.refined_title or note.title,
                "rationale": upd.refined_rationale or note.rationale,
                "quotes": [quote],
                "source": "investigate",
            }
            if upd.refined_severity is not None:
                update_dict["severity"] = upd.refined_severity
            if upd.refined_confidence is not None:
                update_dict["confidence"] = upd.refined_confidence
            notes_by_id[upd.note_id] = note.model_copy(update=update_dict)

    # New notes second. Each must reference a fed section AND its quote
    # must anchor in that section's text.
    for new in result.new_notes:
        if new.quote_section_id not in fed_section_ids:
            continue  # named a section the task didn't feed; reject
        sec = sections_by_id.get(new.quote_section_id)
        if sec is None:
            continue
        quote = anchor_quote(new.quote_text, sec.text, sec.id)
        if quote is None:
            continue
        finding = Finding(
            title=new.title,
            severity=new.severity,
            confidence=new.confidence,
            rationale=new.rationale,
            quotes=[quote],
            sections_involved=[sec.id],
            source="investigate",
            perspective="reflection",  # signals the investigator raised it
            category=new.category,
        )
        notes_by_id[finding.id] = finding
