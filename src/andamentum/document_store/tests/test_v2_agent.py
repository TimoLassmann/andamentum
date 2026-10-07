"""The v2 agent loop, scripted with a FunctionModel over an in-memory fake store."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from pydantic_ai.messages import (
    ModelMessage,
    ModelResponse,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel

from andamentum.document_store.v2 import AgentBudget, Answer, Hit, run_agentic
from andamentum.document_store.v2.agent import answer_from_hits
from andamentum.document_store.v2.store import DocEntry, OutlineEntry

# (section_id, path, heading, line_from, line_to, text)
_SECTIONS = [
    (1, "a.md", "Alpha", 1, 5, "Apollo launched in 1969 from Florida."),
    (2, "a.md", "Alpha two", 6, 10, "The crew of Apollo numbered three."),
    (3, "b.md", "Beta", 1, 5, "Unrelated notes about gardening."),
    (4, "c.md", "Gamma", 1, 5, "The SaltLog service stores salt records."),
    (5, "d.md", "Delta", 1, 5, "Jane Doe wrote the SaltLog design."),
]


def _hit(row: tuple[int, str, str, int, int, str], score: float = 1.0) -> Hit:
    sid, path, heading, lf, lt, text = row
    return Hit(sid, path, heading, lf, lt, text, score)


class FakeStore:
    """The store methods the agent uses, over in-memory data."""

    def __init__(self) -> None:
        self.notes: list[tuple[str, int, int, int, str]] = []
        self.decisions: list[dict[str, Any]] = []
        self.rerank_calls = 0
        self.round0_hits = [_hit(_SECTIONS[0], 0.9), _hit(_SECTIONS[2], 0.5)]

    def round0(
        self,
        query: str,
        *,
        query_embedding: Sequence[float],
        use_exact: bool = True,
        **kw: Any,
    ) -> list[Hit]:
        return list(self.round0_hits)

    async def rerank(
        self, query: str, hits: list[Hit], *, reranker_model: str
    ) -> list[Hit]:
        self.rerank_calls += 1
        return list(reversed(hits))

    def search(
        self,
        query: str,
        *,
        k: int = 10,
        path_prefix: str = "",
        meta: dict[str, str] | None = None,
    ) -> list[Hit]:
        words = query.lower().split()
        rows = [
            r
            for r in _SECTIONS
            if r[1].startswith(path_prefix) and any(w in r[5].lower() for w in words)
        ]
        return [_hit(r) for r in rows][:k]

    def similar(
        self,
        query_embedding: Sequence[float],
        *,
        k: int = 10,
        path_prefix: str = "",
        meta: dict[str, str] | None = None,
    ) -> list[Hit]:
        return [_hit(r) for r in _SECTIONS if r[1].startswith(path_prefix)][:k]

    def grep(
        self, pattern: str, *, path_prefix: str = "", limit: int = 20
    ) -> list[Hit]:
        import re

        rx = re.compile(pattern)
        return [_hit(r) for r in _SECTIONS if rx.search(r[5])][:limit]

    def list_documents(self, *, prefix: str = "", limit: int = 100) -> list[DocEntry]:
        paths = sorted({r[1] for r in _SECTIONS if r[1].startswith(prefix)})
        return [DocEntry(p, p, sum(1 for r in _SECTIONS if r[1] == p)) for p in paths][
            :limit
        ]

    def glob(self, pattern: str, *, limit: int = 100) -> list[DocEntry]:
        return self.list_documents(limit=limit)

    def outline(self, path: str) -> list[OutlineEntry]:
        rows = [r for r in _SECTIONS if r[1] == path]
        if not rows:
            raise KeyError(path)
        return [OutlineEntry(r[0], r[2], 1, r[3], r[4]) for r in rows]

    def read(self, path: str, *, offset: int = 1, limit: int = 60) -> str:
        rows = [r for r in _SECTIONS if r[1] == path]
        if not rows:
            raise KeyError(path)
        return "\n".join(f"{i}: {r[5]}" for i, r in enumerate(rows, start=1))

    def note(
        self, *, run_id: str, section_id: int, line_from: int, line_to: int, claim: str
    ) -> int:
        self.notes.append((run_id, section_id, line_from, line_to, claim))
        return len(self.notes)

    def evidence(self, run_id: str) -> list[tuple[int, int, int, str]]:
        return [(n[1], n[2], n[3], n[4]) for n in self.notes if n[0] == run_id]

    def log_decision(
        self,
        *,
        run_id: str,
        step: int,
        kind: str,
        features: dict[str, Any],
        choice: str,
        decided_by: str,
        confidence: float | None = None,
    ) -> None:
        self.decisions.append(
            {
                "run_id": run_id,
                "step": step,
                "kind": kind,
                "features": features,
                "choice": choice,
            }
        )

    def section_path(self, section_id: int) -> str:
        return next(r[1] for r in _SECTIONS if r[0] == section_id)


async def _embed(texts: list[str]) -> list[list[float]]:
    return [[0.0, 1.0] for _ in texts]


def _final(info: AgentInfo, **args: Any) -> ModelResponse:
    return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, args)])


def _answer(
    info: AgentInfo, text: str = "1969", citations: list[dict[str, Any]] | None = None
) -> ModelResponse:
    return _final(info, text=text, citations=citations or [], confidence="high")


def scripted(
    *,
    enough: bool,
    steps: Sequence[Callable[[AgentInfo], ModelResponse]] = (),
    seen_returns: list[str] | None = None,
) -> FunctionModel:
    """Stop call -> Sufficiency; loop agent (has function tools) -> next step; else synthesis."""
    queue = list(steps)

    def fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        props = info.output_tools[0].parameters_json_schema.get("properties", {})
        if "enough" in props:
            return _final(
                info, enough=enough, missing=None if enough else "the launch crew"
            )
        if info.function_tools:
            if seen_returns is not None:
                for m in messages:
                    for p in getattr(m, "parts", []):
                        if isinstance(p, ToolReturnPart) and isinstance(p.content, str):
                            seen_returns.append(p.content)
            return (queue.pop(0) if len(queue) > 1 else queue[0])(info)
        return _answer(info, text="synthesised")

    return FunctionModel(fn)


def _tool(name: str, **args: Any) -> Callable[[AgentInfo], ModelResponse]:
    return lambda info: ModelResponse(parts=[ToolCallPart(name, args)])


async def _run(
    model: FunctionModel,
    store: FakeStore,
    *,
    reranker: str | None = None,
    budget: AgentBudget = AgentBudget(),
):
    return await run_agentic(
        "When did Apollo launch?",
        store=store,  # type: ignore[arg-type]
        model=model,  # type: ignore[arg-type]
        embed_fn=_embed,
        reranker_model=reranker,
        run_id="r1",
        budget=budget,
    )


async def test_answers_after_round0() -> None:
    store = FakeStore()
    result = await _run(scripted(enough=True), store)
    assert not result.entered_loop
    assert result.answer.text == "synthesised"
    assert result.llm_calls == 2  # stop decision + synthesis
    assert result.tokens_in > 0 and result.tokens_out > 0
    assert result.ranked_paths == ["a.md", "b.md"]
    assert [d["kind"] for d in store.decisions] == ["stop_after_round0"]
    assert store.decisions[0]["choice"] == "answer"
    assert [t["kind"] for t in result.trace] == [
        "round0",
        "stop_after_round0",
        "answer_from_hits",
    ]


async def test_loop_search_read_note_answer() -> None:
    store = FakeStore()
    cite = [{"path": "c.md", "line_from": 1, "line_to": 5}]
    model = scripted(
        enough=False,
        steps=[
            _tool("search", keywords="saltlog"),
            _tool("read", path="d.md"),
            _tool("note", path="d.md", line_from=1, line_to=5, claim="Jane wrote it"),
            lambda info: _answer(info, text="Jane Doe", citations=cite),
        ],
    )
    result = await _run(model, store)
    assert result.entered_loop
    assert result.answer.text == "Jane Doe"
    assert store.decisions[0]["choice"] == "loop"
    assert [d["kind"] for d in store.decisions[1:]] == [
        "tool:search",
        "tool:read",
        "tool:note",
    ]
    assert store.notes == [("r1", 5, 1, 5, "Jane wrote it")]
    # cited -> noted -> tool hits (search: c, d) -> round 0 (a, b)
    assert result.ranked_paths == ["c.md", "d.md", "a.md", "b.md"]
    kinds = [t["kind"] for t in result.trace]
    assert kinds == [
        "round0",
        "stop_after_round0",
        "tool:search",
        "tool:read",
        "tool:note",
    ]
    assert result.trace[2]["result_paths"] == ["c.md:1-5", "d.md:1-5"]
    assert result.llm_calls == 1 + 4  # stop decision + four loop requests


async def test_ranked_paths_order_tools_then_round0() -> None:
    store = FakeStore()
    model = scripted(
        enough=False,
        steps=[
            _tool("grep", regex="Jane"),  # d
            _tool("search", keywords="crew"),  # a (section 2)
            lambda info: _answer(info),
        ],
    )
    result = await _run(model, store)
    assert result.ranked_paths == ["d.md", "a.md", "b.md"]


async def test_tool_budget_spent_returns_error_to_model() -> None:
    store = FakeStore()
    returns: list[str] = []
    model = scripted(
        enough=False,
        steps=[
            _tool("search", keywords="apollo"),
            _tool("search", keywords="crew"),
            lambda info: _answer(info),
        ],
        seen_returns=returns,
    )
    result = await _run(model, store, budget=AgentBudget(max_tool_calls=1))
    assert "ERROR: tool budget spent; answer now" in returns
    errors = [t for t in result.trace if t.get("error")]
    assert len(errors) == 1 and errors[0]["kind"] == "tool:search"
    # the refused call is not logged as a successful decision
    assert [d["kind"] for d in store.decisions] == ["stop_after_round0", "tool:search"]


async def test_request_limit_falls_back_to_answer_from_hits() -> None:
    store = FakeStore()
    model = scripted(
        enough=False, steps=[_tool("search", keywords="saltlog")]
    )  # never answers
    result = await _run(model, store, budget=AgentBudget(max_requests=3))
    assert result.entered_loop
    assert result.answer.text == "synthesised"
    assert result.trace[-1]["kind"] == "request_limit_fallback"
    assert result.llm_calls == 1 + 3 + 1  # stop + three loop requests + synthesis
    assert "c.md" in result.ranked_paths


async def test_tool_exception_is_returned_and_traced() -> None:
    store = FakeStore()
    returns: list[str] = []
    model = scripted(
        enough=False,
        steps=[_tool("outline", path="missing.md"), lambda info: _answer(info)],
        seen_returns=returns,
    )
    result = await _run(model, store)
    assert any(r.startswith("ERROR:") and "missing.md" in r for r in returns)
    err = [t for t in result.trace if t["kind"] == "tool:outline"]
    assert len(err) == 1 and err[0]["error"].startswith("ERROR: KeyError")
    assert result.answer.text == "1969"  # the run itself still completes


async def test_note_without_overlapping_section_errors() -> None:
    store = FakeStore()
    model = scripted(
        enough=False,
        steps=[
            _tool("note", path="a.md", line_from=50, line_to=60, claim="x"),
            lambda info: _answer(info),
        ],
    )
    result = await _run(model, store)
    assert store.notes == []
    assert any(t.get("error") for t in result.trace if t["kind"] == "tool:note")


async def test_tools_rerank_candidates_when_reranker_set() -> None:
    store = FakeStore()
    model = scripted(
        enough=False,
        steps=[_tool("search", keywords="apollo saltlog"), lambda info: _answer(info)],
    )
    await _run(model, store, reranker="fake", budget=AgentBudget(tool_k=2))
    assert store.rerank_calls == 2  # round 0 + the search tool


async def test_tools_do_not_rerank_when_disabled() -> None:
    store = FakeStore()
    model = scripted(
        enough=False,
        steps=[_tool("search", keywords="apollo"), lambda info: _answer(info)],
    )
    await _run(
        model, store, reranker="fake", budget=AgentBudget(tool_rerank_candidates=0)
    )
    assert store.rerank_calls == 1  # round 0 only


async def test_similar_and_listing_tools_run() -> None:
    store = FakeStore()
    model = scripted(
        enough=False,
        steps=[
            _tool("similar", text="apollo"),
            _tool("list", prefix=""),
            _tool("glob", pattern="*.md"),
            lambda info: _answer(info),
        ],
    )
    result = await _run(model, store)
    assert [t["kind"] for t in result.trace[2:]] == [
        "tool:similar",
        "tool:list",
        "tool:glob",
    ]
    assert not any(t.get("error") for t in result.trace)


async def test_answer_from_hits_returns_usage() -> None:
    model = FunctionModel(lambda messages, info: _answer(info, text="x"))
    answer, usage = await answer_from_hits("q?", [_hit(_SECTIONS[0])], model=model)  # type: ignore[arg-type]
    assert isinstance(answer, Answer) and usage.requests == 1
