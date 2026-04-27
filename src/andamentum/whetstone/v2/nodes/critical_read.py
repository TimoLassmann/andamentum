"""Node: CriticalRead — replaces Skim.

Each configured lens reads each section in full and emits issues.
Issues are wrapped into ``Finding``s on the fly: the lens fills the six
flat fields (title / severity / confidence / rationale / quote_text /
category); the controller fills the section, the lens name (as
``perspective``), and turns ``quote_text`` into an anchored ``Quote``.

Lens-emitted quotes that don't appear verbatim in the section are
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
from ..anchoring import anchor_quote
from ..deps import ReviewDeps
from ..schemas import Finding, ReviewResult
from ..state import ReviewState

if TYPE_CHECKING:
    from ..structural.types import SectionRef
    from .reflect_and_investigate import ReflectAndInvestigate


logger = logging.getLogger("andamentum.whetstone.v2")

# Concurrency cap on parallel lens-reading calls. Tuned for small local
# Ollama models which serialise above ~4 concurrent requests anyway.
_MAX_CONCURRENT = 4


@dataclass
class CriticalRead(BaseNode[ReviewState, ReviewDeps, ReviewResult]):
    """Run every (lens × section) pair as a parallel LLM call."""

    async def run(
        self, ctx: GraphRunContext[ReviewState, ReviewDeps]
    ) -> "ReflectAndInvestigate":
        ctx.state.current_phase = "critical_read"
        sections = ctx.state.sections
        lenses = ctx.state.perspectives
        total = len(sections) * len(lenses)
        logger.info(
            "[critical_read] %d section × %d lens = %d reads (concurrency=%d)",
            len(sections),
            len(lenses),
            total,
            _MAX_CONCURRENT,
        )

        sem = asyncio.Semaphore(_MAX_CONCURRENT)
        completed = 0

        async def read_one(section: "SectionRef", lens: str) -> list[Finding]:
            nonlocal completed
            async with sem:
                try:
                    findings = await _run_lens(ctx.deps, section, lens)
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

        results = await asyncio.gather(
            *[read_one(s, lens) for s in sections for lens in lenses]
        )
        for findings in results:
            ctx.state.findings.extend(findings)
        ctx.state.llm_calls += sum(1 for r in results if r is not None)
        logger.info(
            "[critical_read] done — %d issue(s) in pool",
            len(ctx.state.findings),
        )

        from .reflect_and_investigate import ReflectAndInvestigate

        return ReflectAndInvestigate()


async def _run_lens(
    deps: ReviewDeps,
    section: "SectionRef",
    lens: str,
) -> list[Finding]:
    """One lens-agent call against one section. Returns Findings."""
    prompt = f"""SECTION ID: {section.id}
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
