"""Query decomposition: the cheaper multi-hop baseline the agent loop must beat.

One LLM call splits a multi-hop question into ordered single-hop
sub-questions. Round 0 runs for the question and for each sub-question, the
section rankings are fused, the fused top 50 is reranked against the ORIGINAL
question, and one synthesis call answers from the top passages: two LLM calls.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from ...core.agents import run_agent_with_usage
from .agent import EmbedFn, ModelLike, answer_from_hits
from .store import Hit, SectionStore, rrf
from .types import Answer

MAX_SUB_QUESTIONS = 4
FUSED_TOP = 50

_INSTRUCTIONS = (
    "Split a multi-hop question into the ordered single-hop sub-questions "
    "needed to answer it. Each sub-question must be a short, self-contained "
    "search question about one fact. Put the sub-question that must be "
    f"answered first at the start. Use at most {MAX_SUB_QUESTIONS}. If the "
    "question is already single-hop, return it unchanged as the only "
    "sub-question."
)


class Decomposition(BaseModel):
    sub_questions: list[str] = Field(max_length=MAX_SUB_QUESTIONS)


async def decompose(question: str, *, model: ModelLike) -> tuple[Decomposition, Any]:
    """One structured call; returns the decomposition and its usage."""
    result, usage = await run_agent_with_usage(
        model,
        instructions=_INSTRUCTIONS,
        output_type=Decomposition,
        user_message=f"Question: {question}",
    )
    return result, usage


def _counts(usage: Any) -> tuple[int, int, int]:
    return (
        int(getattr(usage, "requests", 0) or 0),
        int(getattr(usage, "input_tokens", 0) or 0),
        int(getattr(usage, "output_tokens", 0) or 0),
    )


async def retrieve_decomposed(
    question: str,
    *,
    store: SectionStore,
    model: ModelLike,
    embed_fn: EmbedFn,
    reranker_model: str | None,
    k_final: int = 10,
) -> tuple[list[Hit], Answer, tuple[int, int, int], Decomposition]:
    """Decompose, retrieve per sub-question, fuse, rerank, answer.

    Returns ``(ranked hits, answer, (llm_calls, tokens_in, tokens_out),
    decomposition)``.
    """
    decomposition, usage_d = await decompose(question, model=model)
    queries = [question] + [
        q for q in decomposition.sub_questions if q.strip() and q != question
    ]
    vectors = await embed_fn(queries)
    by_id: dict[int, Hit] = {}
    rankings: list[list[int]] = []
    for query, vec in zip(queries, vectors, strict=True):
        hits = store.round0(query, query_embedding=vec, use_exact=False)
        for h in hits:
            by_id.setdefault(h.section_id, h)
        rankings.append([h.section_id for h in hits])
    fused = [by_id[sid] for sid, _ in rrf(rankings, depth=FUSED_TOP)]
    if reranker_model is not None:
        fused = await store.rerank(question, fused, reranker_model=reranker_model)
    answer, usage_a = await answer_from_hits(question, fused[:k_final], model=model)
    d, a = _counts(usage_d), _counts(usage_a)
    counts = (d[0] + a[0], d[1] + a[1], d[2] + a[2])
    return fused, answer, counts, decomposition
