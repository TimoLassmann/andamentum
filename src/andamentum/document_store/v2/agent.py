"""Agentic retrieval over the v2 store: round 0, a stop decision, then a tool loop.

Round 0 (no LLM) retrieves and reranks. One structured call decides whether
that evidence is enough; if so one synthesis call answers. Otherwise a
pydantic-ai agent with store tools (list, glob, grep, search, similar,
outline, read, note) loops until it answers. The LLM makes every decision.

A failed tool call is returned to the model as an ``ERROR: ...`` string and
recorded in the trace; a failed run raises. Only the request limit has a
defined fallback: one synthesis call over everything seen so far, recorded
in the trace.
"""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from pydantic_ai import Agent
from pydantic_ai.exceptions import UsageLimitExceeded
from pydantic_ai.models import Model
from pydantic_ai.usage import RunUsage, UsageLimits

from ...core.agents import run_agent_with_usage
from ...core.models import resolve_model
from .store import Hit, SectionStore
from .types import Answer, Citation, Sufficiency

EmbedFn = Callable[[list[str]], Awaitable[list[list[float]]]]

#: A model id (resolved through ``core.models.resolve_model``) or a ready model object.
ModelLike = str | Model

_SNIPPET_CHARS = 300
_PASSAGE_CHARS = 1200
_FALLBACK_MAX_HITS = 20
_READ_MAX_CHARS = 6000


@dataclass(frozen=True)
class AgentBudget:
    max_tool_calls: int = 12  # hard cap on tool calls inside the loop
    max_requests: int = 15  # hard cap on LLM requests inside the loop (UsageLimits)
    round0_k: int = 10  # passages shown to the LLM from round 0
    tool_k: int = 8  # hits returned by search/similar/grep to the LLM
    tool_rerank_candidates: int = 20  # tool hits reranked before returning (0 = none)

    def __post_init__(self) -> None:
        # Each loop request issues at least one tool call or the answer, and an
        # invalid citation costs one retry request. The tool budget must run out
        # before the request budget, so the agent always gets to answer from its
        # own evidence instead of being cut off mid-search.
        if self.max_requests < self.max_tool_calls + 3:
            raise ValueError(
                f"max_requests ({self.max_requests}) must be at least "
                f"max_tool_calls + 3 ({self.max_tool_calls + 3})"
            )


@dataclass
class AgentResult:
    answer: Answer
    ranked_paths: list[str]
    llm_calls: int
    tokens_in: int
    tokens_out: int
    seconds: float
    entered_loop: bool
    trace: list[dict[str, Any]] = field(default_factory=list)


def _passages(hits: list[Hit]) -> str:
    if not hits:
        return "(no passages)"
    return "\n\n".join(f"[{h.citation}]\n{h.text[:_PASSAGE_CHARS]}" for h in hits)


def _render_hit(h: Hit) -> str:
    snippet = " ".join(h.text.split())[:_SNIPPET_CHARS]
    return f"{h.citation} | {h.heading} | {snippet}"


def _render_hits(hits: list[Hit]) -> str:
    return "\n".join(_render_hit(h) for h in hits) if hits else "(no results)"


_ANSWER_FORMAT = (
    "Put in text only the shortest phrase that answers the question: a name, a date, a "
    "number or a short noun phrase, never a sentence. If the passages make an answer likely "
    "but not certain, still give it and set confidence to low. Write unknown only when no "
    "passage is relevant to the question. Cite each passage you used with its path and, as "
    "two separate integers, line_from and line_to, copied from the passage label. Set "
    "confidence to high, medium or low according to how well the passages support the answer."
)

_ANSWER_INSTRUCTIONS = (
    "Answer the question using only the passages given. " + _ANSWER_FORMAT
)

_STOP_INSTRUCTIONS = (
    "You judge retrieved evidence. Given a question and passages, set enough to true only if "
    "the passages contain what is needed to answer the question. Otherwise set enough to "
    "false and say in missing what is still needed."
)

_LOOP_INSTRUCTIONS = (
    "You are a research agent over a document store. Answer the question with the tools "
    "provided: list and glob browse paths, grep matches a regular expression, search matches "
    "keywords, similar matches by meaning, outline shows a document's sections, read shows "
    "lines of a document, note records a passage as evidence. Hits are shown as "
    "path:line_from-line_to. Use the evidence already gathered and look only for what is "
    "missing. For a question with several steps, find each intermediate entity first and "
    "then search for the next step by its name. You have at most {max_tool_calls} tool "
    "calls in total; every tool result states how many are left. Answer as soon as the "
    "evidence is enough, and when no calls are left, answer with the evidence you have. "
) + _ANSWER_FORMAT


async def answer_from_hits(
    question: str,
    hits: list[Hit],
    *,
    model: ModelLike,
    store: SectionStore | None = None,
) -> tuple[Answer, Any]:
    """One synthesis call over the given hits; returns ``(answer, usage)``.

    With ``store``, citations are checked against it and an invalid one is sent
    back to the model for correction (costing one more request).
    """
    answer, usage = await run_agent_with_usage(
        model,
        instructions=_ANSWER_INSTRUCTIONS,
        output_type=Answer,
        user_message=f"Question: {question}\n\nPassages:\n{_passages(hits)}",
        validators=[_citation_validator(store)] if store is not None else None,
    )
    return answer, usage


def _dedup(paths: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for p in paths:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def _citation_trace(store: SectionStore, answer: Answer) -> list[dict[str, Any]]:
    """Each citation of the answer with whether it names a real path and line range."""
    out: list[dict[str, Any]] = []
    for c in answer.citations:
        try:
            sections = store.outline(c.path)
            resolves = (
                1 <= c.line_from <= c.line_to
                and bool(sections)
                and c.line_to <= max(e.line_to for e in sections)
            )
        except KeyError:
            resolves = False
        out.append(
            {
                "path": c.path,
                "line_from": c.line_from,
                "line_to": c.line_to,
                "resolves": resolves,
            }
        )
    return out


def _citation_validator(store: SectionStore) -> Callable[[Answer], Answer]:
    """Reject an answer whose citation names no real path or line range.

    The model then gets one retry with the document's real line count, instead
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


class _Counter:
    """Sums pydantic-ai usage objects from every LLM call of one run."""

    def __init__(self) -> None:
        self.llm_calls = 0
        self.tokens_in = 0
        self.tokens_out = 0

    def add(self, usage: Any) -> None:
        self.llm_calls += int(usage.requests)
        self.tokens_in += int(usage.input_tokens)
        self.tokens_out += int(usage.output_tokens)


async def run_agentic(
    question: str,
    *,
    store: SectionStore,
    model: ModelLike,
    embed_fn: EmbedFn,
    reranker_model: str | None,
    run_id: str,
    budget: AgentBudget = AgentBudget(),
) -> AgentResult:
    started = time.perf_counter()
    counter = _Counter()
    trace: list[dict[str, Any]] = []

    # 1. Round 0 -------------------------------------------------------------
    query_embedding = (await embed_fn([question]))[0]
    hits0 = store.round0(question, query_embedding=query_embedding, use_exact=False)
    if reranker_model:
        hits0 = await store.rerank(question, hits0, reranker_model=reranker_model)
    top0 = hits0[: budget.round0_k]
    trace.append(
        {
            "step": 0,
            "kind": "round0",
            "args": {"reranked": bool(reranker_model)},
            "result_paths": [h.citation for h in top0],
        }
    )

    # 2. Stop decision -------------------------------------------------------
    sufficiency, usage = await run_agent_with_usage(
        model,
        instructions=_STOP_INSTRUCTIONS,
        output_type=Sufficiency,
        user_message=f"Question: {question}\n\nPassages:\n{_passages(top0)}",
    )
    counter.add(usage)
    choice = "answer" if sufficiency.enough else "loop"
    store.log_decision(
        run_id=run_id,
        step=0,
        kind="stop_after_round0",
        features={
            "n_hits": len(hits0),
            "top_scores": [round(h.score, 4) for h in top0[:3]],
        },
        choice=choice,
        decided_by=model if isinstance(model, str) else model.model_name,
    )
    trace.append(
        {
            "step": 0,
            "kind": "stop_after_round0",
            "args": {"enough": sufficiency.enough, "missing": sufficiency.missing},
            "choice": choice,
        }
    )

    round0_paths = [h.path for h in hits0]

    # 3. Enough: synthesise from round 0 ----------------------------------------
    if sufficiency.enough:
        answer, usage = await answer_from_hits(question, top0, model=model, store=store)
        counter.add(usage)
        trace.append(
            {"step": 1, "kind": "answer_from_hits", "args": {"n_hits": len(top0)}}
        )
        trace.append(
            {
                "step": 2,
                "kind": "answer",
                "args": {"confidence": answer.confidence},
                "citations": _citation_trace(store, answer),
            }
        )
        ranked = _dedup([c.path for c in answer.citations] + round0_paths)
        return AgentResult(
            answer=answer,
            ranked_paths=ranked,
            llm_calls=counter.llm_calls,
            tokens_in=counter.tokens_in,
            tokens_out=counter.tokens_out,
            seconds=time.perf_counter() - started,
            entered_loop=False,
            trace=trace,
        )

    # 4. The agent loop ---------------------------------------------------------
    seen: dict[int, Hit] = {
        h.section_id: h for h in top0
    }  # for the request-limit fallback
    tool_paths: list[str] = []  # paths of every tool hit, in call order
    state = {"calls": 0, "step": 0}

    def _finish_tool(name: str, args: dict[str, Any], hits: list[Hit] | None) -> None:
        for h in hits or []:
            seen.setdefault(h.section_id, h)
            tool_paths.append(h.path)

    async def _rerank_hits(hits: list[Hit]) -> list[Hit]:
        if reranker_model and budget.tool_rerank_candidates > 0 and hits:
            hits = await store.rerank(question, hits, reranker_model=reranker_model)
        return hits[: budget.tool_k]

    def _candidates() -> int:
        if reranker_model and budget.tool_rerank_candidates > 0:
            return max(budget.tool_rerank_candidates, budget.tool_k)
        return budget.tool_k

    async def _call(
        name: str,
        args: dict[str, Any],
        fn: Callable[[], Awaitable[tuple[str, list[Hit] | None]]],
    ) -> str:
        """Run one tool under the budget; every outcome is logged and traced."""
        state["step"] += 1
        step = state["step"]
        compact = ", ".join(f"{k}={v!r}" for k, v in args.items())
        if state["calls"] >= budget.max_tool_calls:
            msg = "ERROR: no tool calls left; answer now with the evidence you have"
            trace.append(
                {"step": step, "kind": f"tool:{name}", "args": args, "error": msg}
            )
            return msg
        state["calls"] += 1
        try:
            text, hits = await fn()
        except Exception as exc:  # returned to the model and traced, not swallowed
            msg = f"ERROR: {type(exc).__name__}: {exc}"
            trace.append(
                {"step": step, "kind": f"tool:{name}", "args": args, "error": msg}
            )
            store.log_decision(
                run_id=run_id,
                step=step,
                kind=f"tool:{name}",
                features={"error": msg},
                choice=compact,
                decided_by=decided_by,
            )
            return msg
        _finish_tool(name, args, hits)
        left = budget.max_tool_calls - state["calls"]
        text = (
            f"{text}\n[tool calls left: {left}]"
            if left
            else f"{text}\n[no tool calls left: answer now with the evidence you have]"
        )
        trace.append(
            {
                "step": step,
                "kind": f"tool:{name}",
                "args": args,
                "result_paths": [h.citation for h in hits or []],
            }
        )
        store.log_decision(
            run_id=run_id,
            step=step,
            kind=f"tool:{name}",
            features={"n_hits": len(hits or [])},
            choice=compact,
            decided_by=decided_by,
        )
        return text

    decided_by = model if isinstance(model, str) else model.model_name
    agent: Agent[None, Answer] = Agent(
        resolve_model(model) if isinstance(model, str) else model,
        instructions=_LOOP_INSTRUCTIONS.format(max_tool_calls=budget.max_tool_calls),
        output_type=Answer,
        retries={"tools": 3, "output": 5},
    )
    agent.output_validator(_citation_validator(store))

    @agent.tool_plain(name="list")
    async def list_documents(prefix: str = "") -> str:
        """List documents whose path starts with prefix (empty prefix lists the first documents)."""

        async def go() -> tuple[str, list[Hit] | None]:
            docs = store.list_documents(prefix=prefix, limit=50)
            lines = [f"{d.path} | {d.title} | {d.n_sections} sections" for d in docs]
            return ("\n".join(lines) or "(no documents)"), None

        return await _call("list", {"prefix": prefix}, go)

    @agent.tool_plain
    async def glob(pattern: str) -> str:
        """List documents whose path matches a glob pattern such as 'papers/*2020*'."""

        async def go() -> tuple[str, list[Hit] | None]:
            docs = store.glob(pattern, limit=50)
            lines = [f"{d.path} | {d.title} | {d.n_sections} sections" for d in docs]
            return ("\n".join(lines) or "(no documents)"), None

        return await _call("glob", {"pattern": pattern}, go)

    @agent.tool_plain
    async def grep(regex: str, path_prefix: str = "") -> str:
        """Find passages matching a regular expression, optionally within a path prefix."""

        async def go() -> tuple[str, list[Hit] | None]:
            hits = await _rerank_hits(
                store.grep(regex, path_prefix=path_prefix, limit=_candidates())
            )
            return _render_hits(hits), hits

        return await _call("grep", {"regex": regex, "path_prefix": path_prefix}, go)

    @agent.tool_plain
    async def search(keywords: str, path_prefix: str = "") -> str:
        """Keyword search (BM25) over passages, optionally within a path prefix."""

        async def go() -> tuple[str, list[Hit] | None]:
            hits = await _rerank_hits(
                store.search(keywords, k=_candidates(), path_prefix=path_prefix)
            )
            return _render_hits(hits), hits

        return await _call(
            "search", {"keywords": keywords, "path_prefix": path_prefix}, go
        )

    @agent.tool_plain
    async def similar(text: str, path_prefix: str = "") -> str:
        """Find passages similar in meaning to text, optionally within a path prefix."""

        async def go() -> tuple[str, list[Hit] | None]:
            embedding = (await embed_fn([text]))[0]
            hits = await _rerank_hits(
                store.similar(embedding, k=_candidates(), path_prefix=path_prefix)
            )
            return _render_hits(hits), hits

        return await _call("similar", {"text": text, "path_prefix": path_prefix}, go)

    @agent.tool_plain
    async def outline(path: str) -> str:
        """Show the sections of one document with their line ranges."""

        async def go() -> tuple[str, list[Hit] | None]:
            entries = store.outline(path)
            lines = [
                f"{'  ' * max(e.level - 1, 0)}{e.heading} | lines {e.line_from}-{e.line_to}"
                for e in entries
            ]
            return ("\n".join(lines) or "(no sections)"), None

        return await _call("outline", {"path": path}, go)

    @agent.tool_plain
    async def read(path: str, offset: int = 1, limit: int = 60) -> str:
        """Read lines of a document starting at the 1-based line offset."""

        async def go() -> tuple[str, list[Hit] | None]:
            return store.read(path, offset=offset, limit=limit)[:_READ_MAX_CHARS], None

        return await _call("read", {"path": path, "offset": offset, "limit": limit}, go)

    @agent.tool_plain
    async def note(path: str, line_from: int, line_to: int, claim: str) -> str:
        """Record a passage (path and line range) as evidence for the claim it supports."""

        async def go() -> tuple[str, list[Hit] | None]:
            overlapping = [
                e
                for e in store.outline(path)
                if e.line_from <= line_to and e.line_to >= line_from
            ]
            if not overlapping:
                raise ValueError(
                    f"no section of {path} overlaps lines {line_from}-{line_to}"
                )
            best = max(
                overlapping,
                key=lambda e: min(e.line_to, line_to) - max(e.line_from, line_from),
            )
            store.note(
                run_id=run_id,
                section_id=best.section_id,
                line_from=line_from,
                line_to=line_to,
                claim=claim,
            )
            return f"noted {path}:{line_from}-{line_to}", None

        return await _call(
            "note",
            {"path": path, "line_from": line_from, "line_to": line_to, "claim": claim},
            go,
        )

    async def _fallback_hits() -> list[Hit]:
        """The agent's noted evidence first, then everything it saw, best first."""
        noted_ids = [sid for sid, _, _, _ in store.evidence(run_id)]
        missing = [sid for sid in noted_ids if sid not in seen]
        for h in store.hits(missing):
            seen[h.section_id] = h
        noted = [seen[sid] for sid in dict.fromkeys(noted_ids)]
        rest = [h for sid, h in seen.items() if sid not in set(noted_ids)]
        if reranker_model and rest:
            rest = await store.rerank(question, rest, reranker_model=reranker_model)
        return (noted + rest)[:_FALLBACK_MAX_HITS]

    prompt = (
        f"Question: {question}\n\nEvidence from the first retrieval:\n{_passages(top0)}\n\n"
        f"Still missing: {sufficiency.missing or '(not stated)'}"
    )
    loop_usage = RunUsage()
    try:
        result = await agent.run(
            prompt,
            usage=loop_usage,
            usage_limits=UsageLimits(request_limit=budget.max_requests),
        )
        answer = result.output
        counter.add(loop_usage)
    except UsageLimitExceeded as exc:
        counter.add(loop_usage)
        fallback_hits = await _fallback_hits()
        trace.append(
            {
                "step": state["step"] + 1,
                "kind": "request_limit_fallback",
                "args": {"n_hits": len(fallback_hits)},
                "error": str(exc),
            }
        )
        answer, usage = await answer_from_hits(
            question, fallback_hits, model=model, store=store
        )
        counter.add(usage)

    trace.append(
        {
            "step": state["step"] + 2,
            "kind": "answer",
            "args": {"confidence": answer.confidence},
            "citations": _citation_trace(store, answer),
        }
    )

    # 5. Ranked paths -----------------------------------------------------------
    noted = [store.section_path(sid) for sid, _, _, _ in store.evidence(run_id)]
    ranked = _dedup(
        [c.path for c in answer.citations] + noted + tool_paths + round0_paths
    )
    return AgentResult(
        answer=answer,
        ranked_paths=ranked,
        llm_calls=counter.llm_calls,
        tokens_in=counter.tokens_in,
        tokens_out=counter.tokens_out,
        seconds=time.perf_counter() - started,
        entered_loop=True,
        trace=trace,
    )


__all__ = [
    "AgentBudget",
    "AgentResult",
    "Answer",
    "Citation",
    "EmbedFn",
    "answer_from_hits",
    "run_agentic",
]
