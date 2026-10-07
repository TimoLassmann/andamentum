"""Query decomposition, scripted with a FunctionModel over a fake store."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pytest
from pydantic import ValidationError
from pydantic_ai.messages import ModelMessage, ModelResponse, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from andamentum.document_store.bench import arms
from andamentum.document_store.v2 import Decomposition, Hit, retrieve_decomposed
from andamentum.document_store.v2.decompose import decompose

_ROWS = {
    1: ("a.md", "Alpha", "Jane Doe founded SaltLog."),
    2: ("b.md", "Beta", "SaltLog is based in Springfield."),
    3: ("c.md", "Gamma", "Unrelated gardening notes."),
}


def _hit(sid: int, score: float = 1.0) -> Hit:
    path, heading, text = _ROWS[sid]
    return Hit(sid, path, heading, 1, 3, text, score)


class FakeStore:
    """round0 answers per query from a script; rerank reverses and records."""

    def __init__(self, script: dict[str, list[int]]) -> None:
        self.script = script
        self.queries: list[str] = []
        self.rerank_queries: list[str] = []

    def round0(
        self,
        query: str,
        *,
        query_embedding: Sequence[float],
        use_exact: bool = True,
        **kw: Any,
    ) -> list[Hit]:
        assert use_exact is False
        self.queries.append(query)
        return [_hit(s) for s in self.script[query]]

    async def rerank(
        self, query: str, hits: list[Hit], *, reranker_model: str
    ) -> list[Hit]:
        self.rerank_queries.append(query)
        return list(reversed(hits))


async def _embed(texts: list[str]) -> list[list[float]]:
    return [[0.0, 1.0] for _ in texts]


def _model(subs: list[str], seen: list[str] | None = None) -> FunctionModel:
    def fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        props = info.output_tools[0].parameters_json_schema.get("properties", {})
        name = info.output_tools[0].name
        if "sub_questions" in props:
            return ModelResponse(parts=[ToolCallPart(name, {"sub_questions": subs})])
        if seen is not None:
            seen.append(str(messages[-1]))
        return ModelResponse(
            parts=[
                ToolCallPart(
                    name, {"text": "Springfield", "citations": [], "confidence": "high"}
                )
            ]
        )

    return FunctionModel(fn)


QUESTION = "Where is the company founded by Jane Doe based?"
SCRIPT = {
    QUESTION: [1, 3],
    "Who founded SaltLog?": [1],
    "Where is SaltLog based?": [2, 3],
}


async def test_retrieves_per_sub_question_fuses_and_reranks_against_original() -> None:
    store = FakeStore(SCRIPT)
    hits, answer, counts, dec = await retrieve_decomposed(
        QUESTION,
        store=store,  # type: ignore[arg-type]
        model=_model(["Who founded SaltLog?", "Where is SaltLog based?"]),  # type: ignore[arg-type]
        embed_fn=_embed,
        reranker_model="fake",
    )
    assert store.queries == [
        QUESTION,
        "Who founded SaltLog?",
        "Where is SaltLog based?",
    ]
    assert store.rerank_queries == [QUESTION]
    assert {h.section_id for h in hits} == {1, 2, 3}
    # RRF puts section 1 (two rankings) first; the fake reranker reverses it last.
    assert hits[-1].section_id == 1
    assert answer.text == "Springfield"
    assert counts[0] == 2 and dec.sub_questions[0] == "Who founded SaltLog?"


async def test_synthesis_sees_only_top_k_final() -> None:
    seen: list[str] = []
    await retrieve_decomposed(
        QUESTION,
        store=FakeStore(SCRIPT),  # type: ignore[arg-type]
        model=_model(["Who founded SaltLog?", "Where is SaltLog based?"], seen),  # type: ignore[arg-type]
        embed_fn=_embed,
        reranker_model=None,
        k_final=1,
    )
    assert "Jane Doe founded SaltLog." in seen[0]
    assert "gardening" not in seen[0] and "Springfield." not in seen[0]


async def test_single_hop_returning_itself_is_not_searched_twice() -> None:
    store = FakeStore(SCRIPT)
    await retrieve_decomposed(
        QUESTION,
        store=store,  # type: ignore[arg-type]
        model=_model([QUESTION]),  # type: ignore[arg-type]
        embed_fn=_embed,
        reranker_model=None,
    )
    assert store.queries == [QUESTION]


async def test_decompose_returns_usage() -> None:
    dec, usage = await decompose(QUESTION, model=_model(["a?"]))
    assert dec.sub_questions == ["a?"] and usage.requests == 1


def test_more_than_four_sub_questions_is_rejected() -> None:
    with pytest.raises(ValidationError):
        Decomposition(sub_questions=["a", "b", "c", "d", "e"])


async def test_arm_needs_driver_model() -> None:
    arm = arms.ARMS["v2_decompose"]
    with pytest.raises(ValueError, match="driver-model"):
        await arm.rank(None, None, None)  # type: ignore[arg-type]
