"""Answering over passages, kept apart from search: the citation check and its retry."""

from __future__ import annotations

import ast
import inspect
from pathlib import Path
from typing import Any

import pytest
from pydantic_ai.messages import ModelMessage, ModelResponse, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from andamentum.document_store import Hit, Store, agent
from andamentum.document_store.bench import answering
from andamentum.document_store.bench.answering import (
    Answer,
    Citation,
    answer_closed_book,
    answer_from_hits,
    label,
    passages_text,
    prompts_sha256,
)

_TEXT = "# Alpha\n\nApollo launched in 1969.\nIt landed on the Moon.\n"


@pytest.fixture
async def store(tmp_path: Path):
    s = await Store.open(tmp_path / "answering.db")
    doc_id = await s.put(_TEXT, key="a.md")
    yield s, doc_id
    await s.close()


def _hit(doc_id: str, text: str = "Apollo launched in 1969.") -> Hit:
    start = _TEXT.index(text)
    return Hit(doc_id, 1, start, start + len(text), text, 1.0)


def _answers(*citations: list[dict[str, Any]]) -> FunctionModel:
    queue = list(citations)

    def fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        cites = queue.pop(0) if len(queue) > 1 else queue[0]
        args = {"text": "1969", "citations": cites, "confidence": "high"}
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, args)])

    return FunctionModel(fn)


async def test_label_is_the_key_and_line_range(store) -> None:
    s, doc_id = store
    assert await label(s, _hit(doc_id)) == "a.md:3-3"
    two_lines = "Apollo launched in 1969.\nIt landed"
    assert await label(s, _hit(doc_id, two_lines)) == "a.md:3-4"


async def test_label_of_a_missing_document_is_an_error(store) -> None:
    s, _ = store
    with pytest.raises(KeyError):
        await label(s, Hit("nope", 1, 0, 1, "x", 1.0))


async def test_passages_text_labels_and_truncates(store) -> None:
    s, doc_id = store
    assert await passages_text(s, []) == "(no passages)"
    long_hit = Hit(doc_id, 1, 0, 5, "x" * 3000, 1.0)
    text = await passages_text(s, [_hit(doc_id), long_hit])
    assert text.startswith("[a.md:3-3]\nApollo launched in 1969.")
    assert "x" * 1200 in text and "x" * 1201 not in text


async def test_answer_cites_the_passage(store) -> None:
    s, doc_id = store
    model = _answers([{"path": "a.md", "line_from": 3, "line_to": 3}])
    answer, usage = await answer_from_hits(
        "When?", [_hit(doc_id)], model=model, store=s
    )
    assert answer.text == "1969"
    assert answer.citations == [Citation(path="a.md", line_from=3, line_to=3)]
    assert usage.requests == 1


async def test_a_citation_that_points_nowhere_is_sent_back(store) -> None:
    s, doc_id = store
    model = _answers(
        [{"path": "nope.md", "line_from": 1, "line_to": 2}],
        [{"path": "a.md", "line_from": 3, "line_to": 3}],
    )
    answer, usage = await answer_from_hits(
        "When?", [_hit(doc_id)], model=model, store=s
    )
    assert [c.path for c in answer.citations] == ["a.md"]
    assert usage.requests == 2


async def test_a_citation_outside_the_documents_lines_is_sent_back(store) -> None:
    s, doc_id = store
    model = _answers(
        [{"path": "a.md", "line_from": 3, "line_to": 99}],
        [{"path": "a.md", "line_from": 4, "line_to": 3}],  # reversed range
        [{"path": "a.md", "line_from": 3, "line_to": 4}],
    )
    answer, usage = await answer_from_hits(
        "When?", [_hit(doc_id)], model=model, store=s
    )
    assert answer.citations[0].line_to == 4
    assert usage.requests == 3


def test_citation_validator_accepts_valid_and_retries_invalid() -> None:
    from pydantic_ai import ModelRetry

    check = answering._citation_validator({"a.md": 4})
    ok = Answer(
        text="x",
        citations=[Citation(path="a.md", line_from=1, line_to=4)],
        confidence="low",
    )
    assert check(ok) is ok
    assert check(Answer(text="x", confidence="low")).citations == []  # none is fine
    bad = Answer(
        text="x",
        citations=[
            Citation(path="a.md", line_from=0, line_to=2),
            Citation(path="b.md", line_from=1, line_to=1),
        ],
        confidence="low",
    )
    with pytest.raises(ModelRetry, match="b.md is not a document") as exc:
        check(bad)
    assert "a.md:0-2 is outside the document's lines 1-4" in str(exc.value)


async def test_closed_book_makes_one_call_and_sends_no_passages() -> None:
    seen: list[str] = []

    def fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        seen.append(str(messages))
        args = {"text": "Jane Doe", "confidence": "low"}
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, args)])

    answer, usage = await answer_closed_book("Who?", model=FunctionModel(fn))
    assert answer.text == "Jane Doe" and answer.citations == []
    assert usage.requests == 1
    assert "Question: Who?" in seen[0] and "Passages" not in seen[0]


def test_prompts_fingerprint_is_stable_hex() -> None:
    assert prompts_sha256() == prompts_sha256()
    assert len(prompts_sha256()) == 64


def test_search_code_does_not_import_answering() -> None:
    tree = ast.parse(inspect.getsource(agent))
    imported = [
        node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
    ]
    assert not any("answering" in m for m in imported)
