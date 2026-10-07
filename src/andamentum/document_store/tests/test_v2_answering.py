"""Answering over passages, kept apart from search: the citation check and its retry."""

from __future__ import annotations

from typing import Any

from pydantic_ai.messages import ModelMessage, ModelResponse, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from andamentum.document_store.v2 import Hit, answer_from_hits
from andamentum.document_store.v2.store import OutlineEntry


class _Store:
    def outline(self, path: str) -> list[OutlineEntry]:
        if path != "a.md":
            raise KeyError(path)
        return [OutlineEntry(1, "Alpha", 1, 1, 5)]


_HITS = [Hit(1, "a.md", "Alpha", 1, 5, "Apollo launched in 1969.", 1.0)]


def _answers(*citations: list[dict[str, Any]]) -> FunctionModel:
    queue = list(citations)

    def fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        cites = queue.pop(0) if len(queue) > 1 else queue[0]
        args = {"text": "1969", "citations": cites, "confidence": "high"}
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, args)])

    return FunctionModel(fn)


async def test_answer_cites_the_passage() -> None:
    model = _answers([{"path": "a.md", "line_from": 1, "line_to": 5}])
    answer, usage = await answer_from_hits(
        "When?",
        _HITS,
        model=model,
        store=_Store(),  # type: ignore[arg-type]
    )
    assert answer.text == "1969"
    assert usage.requests == 1


async def test_a_citation_that_points_nowhere_is_sent_back() -> None:
    model = _answers(
        [{"path": "nope.md", "line_from": 1, "line_to": 2}],
        [{"path": "a.md", "line_from": 1, "line_to": 5}],
    )
    answer, usage = await answer_from_hits(
        "When?",
        _HITS,
        model=model,
        store=_Store(),  # type: ignore[arg-type]
    )
    assert [c.path for c in answer.citations] == ["a.md"]
    assert usage.requests == 2


def test_search_code_does_not_import_answering() -> None:
    import ast
    import inspect

    from andamentum.document_store.v2 import agent

    tree = ast.parse(inspect.getsource(agent))
    imported = [
        node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
    ]
    assert not any("answering" in m for m in imported)
