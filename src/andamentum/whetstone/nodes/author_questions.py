"""Node: AuthorQuestions.

After the reflection loop has done its work, some findings will
inevitably remain "high-stakes but uncertain" — typically major or
moderate severity findings whose confidence is still low because the
section text alone couldn't fully resolve them.

For those findings, ask the author_question agent to formulate one
sharp question only the document's author can answer. The investigator
already exhausted what could be checked from the source; what's left
is something only the author knows.

Capped at ``_MAX_AUTHOR_QUESTIONS`` — too many questions = noise.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

from pydantic_graph import BaseNode, GraphRunContext

from ..agents import AuthorQuestionOutput, build_pydantic_ai_agent
from ..deps import ReviewDeps
from ..schemas import AuthorQuestion, Finding, ReviewResult
from ..state import ReviewState

if TYPE_CHECKING:
    from .reconcile_claims import ReconcileClaims


logger = logging.getLogger("andamentum.whetstone")

_MAX_AUTHOR_QUESTIONS = 8
_MAX_CONCURRENT = 4


@dataclass
class AuthorQuestions(BaseNode[ReviewState, ReviewDeps, ReviewResult]):
    """Turn unresolved high-stakes findings into author-facing questions."""

    async def run(
        self, ctx: GraphRunContext[ReviewState, ReviewDeps]
    ) -> "ReconcileClaims":
        ctx.state.current_phase = "author_questions"

        unresolved = _select_unresolved(ctx.state.challenged_findings or ctx.state.findings)
        unresolved = unresolved[:_MAX_AUTHOR_QUESTIONS]
        if not unresolved:
            logger.info("[author_questions] no unresolved findings — skipping")
            from .reconcile_claims import ReconcileClaims

            return ReconcileClaims()

        logger.info(
            "[author_questions] formulating %d question(s) for the author",
            len(unresolved),
        )

        sections_by_id = {s.id: s for s in ctx.state.sections}
        sem = asyncio.Semaphore(_MAX_CONCURRENT)

        async def one(f: Finding) -> AuthorQuestion | None:
            async with sem:
                try:
                    return await _build_question(ctx.deps, f, sections_by_id)
                except Exception as exc:
                    logger.warning(
                        "[author_questions] one question failed: %s", exc
                    )
                    return None

        questions = await asyncio.gather(*[one(f) for f in unresolved])
        good = [q for q in questions if q is not None]
        ctx.state.author_questions.extend(good)
        ctx.state.llm_calls += len(good)
        logger.info(
            "[author_questions] done — %d question(s) emitted", len(good)
        )

        from .reconcile_claims import ReconcileClaims

        return ReconcileClaims()


def _select_unresolved(findings: list[Finding]) -> list[Finding]:
    """Pick findings that look genuinely unresolvable from the source alone.

    A finding is "unresolved" when:
      • severity is major or moderate (the question is worth asking), AND
      • confidence is low (the source couldn't settle it).

    Sort by severity (major first) then by section position so the most
    important questions come first.
    """
    severity_order = {"major": 3, "moderate": 2, "minor": 1}
    candidates = [
        f for f in findings
        if f.severity in ("major", "moderate") and f.confidence == "low"
    ]
    candidates.sort(key=lambda f: -severity_order.get(f.severity, 0))
    return candidates


async def _build_question(
    deps: ReviewDeps,
    finding: Finding,
    sections_by_id,
) -> AuthorQuestion:
    cited = [
        sections_by_id[sid]
        for sid in finding.sections_involved
        if sid in sections_by_id
    ]
    sections_text = "\n\n".join(
        f"--- {s.id} ({s.title}) ---\n{s.text[:3000]}"
        for s in cited
    )
    quote_blurb = ""
    if finding.quotes:
        quote_blurb = f'\nVERBATIM QUOTE the reviewers cited: "{finding.quotes[0].text}"'

    prompt = f"""UNRESOLVED FINDING:
title:      {finding.title}
severity:   {finding.severity}
confidence: {finding.confidence}
rationale:  {finding.rationale}{quote_blurb}

RELEVANT SECTIONS (excerpts):
{sections_text or "(none cited or not found in document)"}

Formulate ONE sharp question for the author. One sentence. Specific.
Reference section ids when possible."""

    agent = build_pydantic_ai_agent("author_question", deps.model)
    result = await agent.run(prompt)
    output = cast(AuthorQuestionOutput, result.output)
    return AuthorQuestion(
        question=output.question,
        why=output.why,
        sections_involved=output.sections_involved or finding.sections_involved,
    )
