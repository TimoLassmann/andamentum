"""Answering over retrieved passages: interpretation, kept apart from search.

Search (``Store.search``, ``AgentSearcher``) returns passages and never writes
prose. The functions here take passages a caller already has and write a
short answer with citations. Only the benchmark uses them.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from typing import Any, Literal

from pydantic import BaseModel
from pydantic_ai import ModelRetry
from pydantic_ai.models import Model
from pydantic_ai.settings import ModelSettings

from ...core.agents import run_agent_with_usage
from ..store import Store
from ..types import Hit, line_range

#: A model id (resolved through ``core.models.resolve_model``) or a ready model object.
ModelLike = str | Model

_PASSAGE_CHARS = 1200

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


class Citation(BaseModel):
    path: str
    line_from: int
    line_to: int


class Answer(BaseModel):
    """A short answer with citations."""

    text: str  # short answer, a few words where possible
    citations: list[Citation] = []
    confidence: Literal["high", "medium", "low"]


def prompts_sha256() -> str:
    """Fingerprint of every instruction text this module sends; run records keep it."""
    texts = (_ANSWER_INSTRUCTIONS, _CLOSED_BOOK_INSTRUCTIONS)
    return hashlib.sha256("\x00".join(texts).encode()).hexdigest()


async def label(store: Store, h: Hit) -> str:
    """``key:line_from-line_to`` for a hit (the document id when it has no key)."""
    doc = await store.get(h.doc_id, rev=h.rev)
    if doc is None:
        raise KeyError(h.doc_id)
    a, b = line_range(doc.text, start=h.start, end=h.end)
    return f"{doc.key or doc.doc_id}:{a}-{b}"


async def passages_text(store: Store, hits: list[Hit]) -> str:
    """Each hit as its label and up to 1200 characters of text."""
    if not hits:
        return "(no passages)"
    parts = [f"[{await label(store, h)}]\n{h.text[:_PASSAGE_CHARS]}" for h in hits]
    return "\n\n".join(parts)


def _citation_validator(line_counts: dict[str, int]) -> Callable[[Answer], Answer]:
    """Reject an answer whose citation names no real document or line range.

    The model then gets a retry with the document's real line count, instead
    of a citation that points nowhere being passed on as evidence.
    """

    def check(answer: Answer) -> Answer:
        problems = []
        for c in answer.citations:
            if c.path not in line_counts:
                problems.append(f"{c.path} is not a document in the store")
                continue
            last = line_counts[c.path]
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
    store: Store,
    model_settings: ModelSettings | None = None,
) -> tuple[Answer, Any]:
    """One synthesis call over the given hits; returns ``(answer, usage)``.

    Citations are checked against the store; an invalid one is sent back to
    the model for correction (costing one more request).
    """
    line_counts: dict[str, int] = {}
    for h in hits:
        doc = await store.get(h.doc_id)
        if doc is not None:
            line_counts[doc.key or doc.doc_id] = doc.text.count("\n") + 1
    answer, usage = await run_agent_with_usage(
        model,
        instructions=_ANSWER_INSTRUCTIONS,
        output_type=Answer,
        user_message=f"Question: {question}\n\nPassages:\n{await passages_text(store, hits)}",
        validators=[_citation_validator(line_counts)],
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


__all__ = ["Answer", "Citation", "answer_closed_book", "answer_from_hits", "prompts_sha256"]
