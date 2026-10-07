"""Answering over retrieved passages: interpretation, kept apart from search.

Search (``v2.agent.search_agentic``, ``SectionStore.round0``) returns passages
and never writes prose. The functions here take passages a caller already has
and write a short answer with citations. No search code imports this module.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from typing import Any

from pydantic_ai.models import Model
from pydantic_ai.settings import ModelSettings

from ...core.agents import run_agent_with_usage
from .render import passages_text
from .store import Hit, SectionStore
from .types import Answer

#: A model id (resolved through ``core.models.resolve_model``) or a ready model object.
ModelLike = str | Model

_ANSWER_INSTRUCTIONS = (
    "Answer the question using only the passages given. Put in text only the shortest "
    "phrase that answers the question: a name, a date, a number or a short noun phrase, "
    "never a sentence. If the passages make an answer likely but not certain, still give it "
    "and set confidence to low. Write unknown only when no passage is relevant to the "
    "question. Cite each passage you used with its path and, as two separate integers, "
    "line_from and line_to, copied from the passage label. Set confidence to high, medium or "
    "low according to how well the passages support the answer."
)

_CLOSED_BOOK_INSTRUCTIONS = (
    "Answer the question from your own knowledge; no passages are given. Put in text only "
    "the shortest phrase that answers the question: a name, a date, a number or a short noun "
    "phrase, never a sentence. If you are not sure, still give your best answer and set "
    "confidence to low. Leave citations empty."
)


def prompts_sha256() -> str:
    """Fingerprint of every instruction text this module sends; run records keep it."""
    texts = (_ANSWER_INSTRUCTIONS, _CLOSED_BOOK_INSTRUCTIONS)
    return hashlib.sha256("\x00".join(texts).encode()).hexdigest()


def _citation_validator(store: SectionStore) -> Callable[[Answer], Answer]:
    """Reject an answer whose citation names no real path or line range.

    The model then gets a retry with the document's real line count, instead
    of a citation that points nowhere being passed on as evidence.
    """
    from pydantic_ai import ModelRetry

    def check(answer: Answer) -> Answer:
        problems = []
        for c in answer.citations:
            try:
                sections = store.outline(c.path)
            except KeyError:
                problems.append(f"{c.path} is not a document in the store")
                continue
            last = max((e.line_to for e in sections), default=0)
            if not 1 <= c.line_from <= c.line_to <= last:
                problems.append(
                    f"{c.path}:{c.line_from}-{c.line_to} is outside the document's lines 1-{last}"
                )
        if problems:
            raise ModelRetry(
                "Fix these citations, copying path and lines from the passage labels: "
                + "; ".join(problems)
            )
        return answer

    return check


async def answer_from_hits(
    question: str,
    hits: list[Hit],
    *,
    model: ModelLike,
    store: SectionStore | None = None,
    model_settings: ModelSettings | None = None,
) -> tuple[Answer, Any]:
    """One synthesis call over the given hits; returns ``(answer, usage)``.

    With ``store``, citations are checked against it and an invalid one is sent
    back to the model for correction (costing one more request).
    """
    answer, usage = await run_agent_with_usage(
        model,
        instructions=_ANSWER_INSTRUCTIONS,
        output_type=Answer,
        user_message=f"Question: {question}\n\nPassages:\n{passages_text(hits)}",
        validators=[_citation_validator(store)] if store is not None else None,
        model_settings=model_settings,
    )
    return answer, usage


async def answer_closed_book(
    question: str, *, model: ModelLike, model_settings: ModelSettings | None = None
) -> tuple[Answer, Any]:
    """One call with no passages: the floor that retrieval has to beat."""
    answer, usage = await run_agent_with_usage(
        model,
        instructions=_CLOSED_BOOK_INSTRUCTIONS,
        output_type=Answer,
        user_message=f"Question: {question}",
        model_settings=model_settings,
    )
    return answer, usage


__all__ = ["answer_closed_book", "answer_from_hits", "prompts_sha256"]
