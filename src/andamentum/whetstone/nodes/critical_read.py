"""Node: CriticalRead — replaces Skim.

Each configured lens reads either ONE section at a time (the default,
``LENS_MULTI_SECTION[lens] is False``) or the WHOLE document at once
(``LENS_MULTI_SECTION[lens] is True``). The latter is for lenses whose
job is inherently cross-section — terminology drift, contradicting
prose claims, etc.

Issues are wrapped into ``Finding``s on the fly: the lens fills the six
flat fields (title / severity / confidence / rationale / quote_text /
category); the controller fills the section_id(s), the lens name (as
``perspective``), and turns ``quote_text`` into an anchored ``Quote``.

Lens-emitted quotes that don't appear verbatim in the document are
silently dropped. The Finding still surfaces — the verbatim quote is
preferred but not strictly required at the lens stage.

After the parallel reads complete, control passes to
``ReflectAndInvestigate``, which runs the bounded reflection loop.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

from pydantic_graph import BaseNode, GraphRunContext

from ..agents import LensReadOutput, build_pydantic_ai_agent
from ..agents.lens_prompts import LENS_MULTI_SECTION, LENS_TARGET_SECTIONS
from ..agents.section_kinds import classify_section_kind
from ..anchoring import anchor_quote
from ..deps import ReviewDeps
from ..schemas import Finding, ReviewResult
from ..state import ReviewState

if TYPE_CHECKING:
    from ..structural.types import SectionRef
    from .reflect_and_investigate import ReflectAndInvestigate


logger = logging.getLogger("andamentum.whetstone")

# Concurrency cap on parallel lens-reading calls. Dropped from 4 → 2
# because high concurrency was causing stale-connection / NAT-table
# saturation against the OpenAI edge (waves of `Connection error` in
# batches of exactly N parallel calls). Ollama serialises internally,
# so this doesn't hurt local throughput either.
_MAX_CONCURRENT = 2


@dataclass
class CriticalRead(BaseNode[ReviewState, ReviewDeps, ReviewResult]):
    """Run every per-section lens × section pair plus every multi-section
    lens × document, all as parallel LLM calls."""

    async def run(
        self, ctx: GraphRunContext[ReviewState, ReviewDeps]
    ) -> "ReflectAndInvestigate":
        ctx.state.current_phase = "critical_read"
        sections = await _reviewable_sections(ctx.deps, ctx.state)
        all_lenses = ctx.state.perspectives
        per_section_lenses = [
            lens for lens in all_lenses if not LENS_MULTI_SECTION.get(lens, False)
        ]
        multi_section_lenses = [
            lens for lens in all_lenses if LENS_MULTI_SECTION.get(lens, False)
        ]
        total = len(sections) * len(per_section_lenses) + len(multi_section_lenses)
        logger.info(
            "[critical_read] %d section × %d per-section lens + %d multi-section "
            "lens = %d reads (concurrency=%d)",
            len(sections),
            len(per_section_lenses),
            len(multi_section_lenses),
            total,
            _MAX_CONCURRENT,
        )

        sem = asyncio.Semaphore(_MAX_CONCURRENT)
        completed = 0

        async def read_section(section: "SectionRef", lens: str) -> list[Finding]:
            nonlocal completed
            async with sem:
                doc_context = _build_document_context(
                    ctx.state.document_map, section.id
                )
                try:
                    findings = await _run_lens(
                        ctx.deps, section, lens, doc_context=doc_context
                    )
                except Exception as exc:
                    logger.warning(
                        "[critical_read] %s × %s crashed: %s",
                        section.id,
                        lens,
                        exc,
                    )
                    return []
                completed += 1
                logger.info(
                    "[critical_read] %d/%d done — %s × %s: %d issue(s)",
                    completed,
                    total,
                    section.title or section.id,
                    lens,
                    len(findings),
                )
                return findings

        async def read_document(lens: str) -> list[Finding]:
            nonlocal completed
            async with sem:
                try:
                    findings = await _run_multi_section_lens(
                        ctx.deps, sections, lens
                    )
                except Exception as exc:
                    logger.warning(
                        "[critical_read] whole-doc × %s crashed: %s", lens, exc
                    )
                    return []
                completed += 1
                logger.info(
                    "[critical_read] %d/%d done — whole-doc × %s: %d issue(s)",
                    completed,
                    total,
                    lens,
                    len(findings),
                )
                return findings

        per_section_tasks = [
            read_section(s, lens)
            for s in sections
            for lens in per_section_lenses
            if _lens_targets_section(lens, s)
        ]
        multi_section_tasks = [read_document(lens) for lens in multi_section_lenses]

        results = await asyncio.gather(*per_section_tasks, *multi_section_tasks)
        for findings in results:
            ctx.state.findings.extend(findings)
        ctx.state.llm_calls += sum(1 for r in results if r is not None)
        logger.info(
            "[critical_read] done — %d issue(s) in pool",
            len(ctx.state.findings),
        )

        from .reflect_and_investigate import ReflectAndInvestigate

        return ReflectAndInvestigate()


async def _reviewable_sections(
    deps: ReviewDeps, state: ReviewState
) -> list["SectionRef"]:
    """Sections worth reviewing as prose, per the section classifier.

    Classifies every section once (``review`` / ``reference`` / ``boilerplate``),
    records the reviewable ids on the state so ReconcileClaims reuses them, and
    returns only the ``review`` sections. General — no heading/keyword/position
    or document-type assumption; a bibliography or boilerplate block is skipped
    wherever it sits.
    """
    sections = state.sections
    sem = asyncio.Semaphore(_MAX_CONCURRENT)

    async def classify(section: "SectionRef") -> tuple[str, str]:
        async with sem:
            try:
                kind = await _classify_section(deps, section)
            except Exception as exc:
                logger.warning(
                    "[classify] %s crashed: %s — treating as reviewable",
                    section.id,
                    exc,
                )
                kind = "review"
        return section.id, kind

    results = await asyncio.gather(*[classify(s) for s in sections])
    reviewable = {sid for sid, kind in results if kind == "review"}
    state.reviewable_section_ids = reviewable
    skipped = len(sections) - len(reviewable)
    if skipped:
        logger.info(
            "[classify] %d section(s) reviewable, %d skipped "
            "(reference/boilerplate)",
            len(reviewable),
            skipped,
        )
    return [s for s in sections if s.id in reviewable]


async def _classify_section(deps: ReviewDeps, section: "SectionRef") -> str:
    """One section-classifier call. Returns 'review' | 'reference' | 'boilerplate'."""
    from ..agents import SectionClass

    snippet = section.text[:1200]
    prompt = f"""SECTION TITLE: {section.title}

SECTION TEXT (start):
{snippet}

Classify this section."""
    agent = build_pydantic_ai_agent("section_classifier", deps.model)
    result = await agent.run(prompt)
    return cast(SectionClass, result.output).kind


def _lens_targets_section(lens: str, section: "SectionRef") -> bool:
    """Whether ``lens`` should run against ``section``.

    Lenses with no entry in ``LENS_TARGET_SECTIONS`` run against every
    section (the default). Lenses with an entry run only against
    sections whose classified kind is in the entry's set.
    """
    targets = LENS_TARGET_SECTIONS.get(lens)
    if not targets:
        return True
    return classify_section_kind(section.title) in targets


def _build_document_context(document_map, current_section_id: str) -> str:
    """Compact preamble giving a per-section lens awareness of the WHOLE
    document — so it doesn't flag content as "missing" when it merely lives
    in another section the agent can't see.

    Lists every section's title (+ a short gist) as a table of contents,
    marks the section currently under review, and instructs the lens to
    comment only on the visible text. Domain-agnostic; ~one line per
    section. Empty string when no map is available.
    """
    if not document_map:
        return ""
    total = len(document_map)
    position = next(
        (i + 1 for i, c in enumerate(document_map) if c.section_id == current_section_id),
        None,
    )
    toc_lines: list[str] = []
    for c in document_map:
        marker = "▶ " if c.section_id == current_section_id else "  "
        gist = (c.one_line_gist or "").strip()
        if len(gist) > 80:
            gist = gist[:77] + "…"
        toc_lines.append(f"{marker}{c.title}" + (f" — {gist}" if gist else ""))
    toc = "\n".join(toc_lines)
    pos = f"section {position} of {total}" if position else "one section"
    return (
        f"DOCUMENT CONTEXT — you are reviewing {pos} of a larger document.\n"
        f"The document contains these sections (you can only see the text of "
        f"the one marked ▶):\n"
        f"{toc}\n\n"
        f"Review ONLY the section text below. Do NOT report something as "
        f"missing from the document (e.g. \"no methods section\", \"no "
        f"references\", \"no discussion\") — it may exist in another section "
        f"listed above that you cannot see here.\n\n"
    )


async def _run_lens(
    deps: ReviewDeps,
    section: "SectionRef",
    lens: str,
    doc_context: str = "",
) -> list[Finding]:
    """One lens call against one section. Returns Findings.

    Two dispatch paths:

    * Sub-graph lenses (``strunk``, …) live in ``whetstone/lenses/`` and
      have their own pydantic-graph DAGs. We hand the section + model
      directly to their entrypoint and let them produce ``Finding``s.
    * Persona / prompt-based lenses (the original seven) take the
      existing path: build a pydantic-ai agent from the registered
      ``AgentDefinition`` and convert ``LensIssueProposal``s into
      ``Finding``s here.

    ``doc_context`` is the whole-document preamble (see
    :func:`_build_document_context`) prepended to the prose-lens prompt so
    the agent knows what exists in sections it cannot see.
    """
    from ..lenses import SUBGRAPH_LENS_ENTRYPOINTS

    if lens in SUBGRAPH_LENS_ENTRYPOINTS:
        return await SUBGRAPH_LENS_ENTRYPOINTS[lens](section, model=deps.model)

    prompt = f"""{doc_context}SECTION ID: {section.id}
SECTION TITLE: {section.title}

SECTION TEXT — your only evidence; quote VERBATIM:
--- BEGIN ---
{section.text}
--- END ---

Now read this section as a {lens} reviewer and emit your issues."""

    agent = build_pydantic_ai_agent(f"lens.{lens}", deps.model)
    result = await agent.run(prompt)
    output = cast(LensReadOutput, result.output)

    findings: list[Finding] = []
    for proposal in output.issues:
        quote = (
            anchor_quote(proposal.quote_text, section.text, section.id)
            if proposal.quote_text
            else None
        )
        findings.append(
            Finding(
                title=proposal.title,
                severity=proposal.severity,
                confidence=proposal.confidence,
                rationale=proposal.rationale,
                quotes=[quote] if quote else [],
                sections_involved=[section.id],
                source="investigate",
                perspective=lens,
                category=proposal.category,
            )
        )
    return findings


async def _run_multi_section_lens(
    deps: ReviewDeps,
    sections: list["SectionRef"],
    lens: str,
) -> list[Finding]:
    """One lens-agent call against the WHOLE document.

    Used for lenses whose job is inherently cross-section (terminology
    drift, contradicting prose claims, claim-emphasis shift). Each
    section is shown verbatim with its id and title; the lens picks
    which sections each issue spans.

    Anchoring: the proposal's quote_text is searched across every
    section (in order) — the first match wins. The Finding's
    sections_involved is left empty if the lens couldn't anchor
    cross-section, but we still surface the finding because the lens's
    rationale carries most of the value here.
    """
    section_blocks = "\n\n".join(
        f"=== {s.id} ({s.title}) ===\n{s.text}" for s in sections
    )
    prompt = f"""DOCUMENT — your only evidence; quote VERBATIM. Each
section is shown with its id and title between ``===`` markers:

--- BEGIN DOCUMENT ---
{section_blocks}
--- END DOCUMENT ---

Now read this document as a {lens} reviewer and emit your issues. Every
issue must span 2+ sections."""

    agent = build_pydantic_ai_agent(f"lens.{lens}", deps.model)
    result = await agent.run(prompt)
    output = cast(LensReadOutput, result.output)

    findings: list[Finding] = []
    for proposal in output.issues:
        quote = None
        anchored_section_id: str | None = None
        if proposal.quote_text:
            for s in sections:
                quote = anchor_quote(proposal.quote_text, s.text, s.id)
                if quote is not None:
                    anchored_section_id = s.id
                    break
        # We can't recover ALL sections the lens reasoned over from quote
        # alone — only the one we anchored. The lens's rationale will
        # mention the other side of the inconsistency in prose.
        sections_involved = [anchored_section_id] if anchored_section_id else []
        findings.append(
            Finding(
                title=proposal.title,
                severity=proposal.severity,
                confidence=proposal.confidence,
                rationale=proposal.rationale,
                quotes=[quote] if quote else [],
                sections_involved=sections_involved,
                source="investigate",
                perspective=lens,
                category=proposal.category or "consistency",
            )
        )
    return findings
