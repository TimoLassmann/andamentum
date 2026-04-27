"""Node: AuthorQuestions.

For every hypothesis still ``status == "open"`` after InvestigateLoop
exhausted its budget, AND for every hypothesis that landed
``status == "unfounded"`` with a meaningful reason, ask the
``author_question_agent`` to formulate one sharp question only the
document's author can answer.

Caps the number of questions asked so a poorly-skimmed document
doesn't bury the user under 50 trivial clarifications.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING

from pydantic_graph import BaseNode, GraphRunContext

from ..agents import build_pydantic_ai_agent, AuthorQuestionOutput
from ..deps import ReviewDeps
from ..schemas import AuthorQuestion, Hypothesis, ReviewResult
from ..state import ReviewState

if TYPE_CHECKING:
    from .synthesise import Synthesise


# Cap on how many questions to surface. Too many questions = noise.
_MAX_AUTHOR_QUESTIONS = 8
_MAX_CONCURRENT = 4


@dataclass
class AuthorQuestions(BaseNode[ReviewState, ReviewDeps, ReviewResult]):
    """Turn unresolved hypotheses into sharp author-facing questions."""

    async def run(
        self, ctx: GraphRunContext[ReviewState, ReviewDeps]
    ) -> "Synthesise":
        ctx.state.current_phase = "author_questions"

        unresolved = _select_unresolved(ctx.state.hypotheses)[:_MAX_AUTHOR_QUESTIONS]
        if not unresolved:
            from .synthesise import Synthesise

            return Synthesise()

        sections_by_id = {s.id: s for s in ctx.state.sections}
        sem = asyncio.Semaphore(_MAX_CONCURRENT)

        async def one(h: Hypothesis) -> AuthorQuestion | None:
            async with sem:
                try:
                    return await _build_question(ctx.deps, h, sections_by_id)
                except Exception:
                    return None  # don't crash the whole graph on one failure

        questions = await asyncio.gather(*[one(h) for h in unresolved])
        good = [q for q in questions if q is not None]
        ctx.state.author_questions.extend(good)
        ctx.state.llm_calls += len(good)

        from .synthesise import Synthesise

        return Synthesise()


def _select_unresolved(hypotheses: list[Hypothesis]) -> list[Hypothesis]:
    """Pick hypotheses that look genuinely unresolvable (not crashed).

    Prefer ``open`` (budget-exhausted) over ``unfounded`` so the most
    important questions surface first. Within each, sort by priority.
    """
    priority_order = {"high": 3, "medium": 2, "low": 1}
    open_ones = sorted(
        (h for h in hypotheses if h.status == "open"),
        key=lambda h: -priority_order.get(h.priority, 0),
    )
    unfounded_high = sorted(
        (h for h in hypotheses if h.status == "unfounded" and h.priority == "high"),
        key=lambda h: -priority_order.get(h.priority, 0),
    )
    return open_ones + unfounded_high


async def _build_question(
    deps: ReviewDeps,
    hypothesis: Hypothesis,
    sections_by_id,
) -> AuthorQuestion:
    cited = [
        sections_by_id[sid]
        for sid in hypothesis.relevant_section_ids
        if sid in sections_by_id
    ]
    sections_text = "\n\n".join(
        f"--- {s.id} ({s.title}) ---\n{s.text[:3000]}"
        for s in cited
    )
    prompt = f"""UNRESOLVED HYPOTHESIS:
{hypothesis.text}
(priority: {hypothesis.priority}; status: {hypothesis.status})

RELEVANT SECTIONS (excerpts):
{sections_text or "(none cited or not found in document)"}

Formulate ONE sharp question for the author. One sentence. Specific.
Reference section ids when possible."""

    agent = build_pydantic_ai_agent("author_question", deps.model)
    result = await agent.run(prompt)
    from typing import cast

    output = cast(AuthorQuestionOutput, result.output)
    return AuthorQuestion(
        question=output.question,
        why=output.why,
        sections_involved=output.sections_involved or hypothesis.relevant_section_ids,
    )
