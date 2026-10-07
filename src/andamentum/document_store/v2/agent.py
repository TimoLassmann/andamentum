"""Agent-run search over the v2 store: round 0, a stop decision, then a tool loop.

The output is search results, the same shape as the current store's
``search``: documents, best first, each with its passages and a score. The
agent never answers the request and never writes prose that reaches the
caller. It decides only where to look and which passages the request needs;
the results are then built deterministically from those passages.

Round 0 (no LLM) retrieves and reranks. One structured call decides whether
those passages cover the request; if so they are the results. Otherwise a
pydantic-ai agent with store tools (list, glob, grep, search, similar,
outline, read, note) searches until it finishes with a ``Selection``: the
passages the request needs, by path and line range. A selection may name only
passages a tool showed it, so it cannot point at a source it never saw.

Results, in this order: the selected passages, then every other passage the
agent saw, each group by reranker score against the request, then the rest of
round 0 in its reranked order. Passages are grouped into documents at the
position of each document's first passage.

A failed tool call is returned to the model as an ``ERROR: ...`` string and
recorded in the trace. If the loop runs out of requests, or the model cannot
produce a valid selection, the passages it noted become the selection; the
trace records which case it was. No extra model call is made for it.
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic_ai import Agent
from pydantic_ai.exceptions import UnexpectedModelBehavior, UsageLimitExceeded
from pydantic_ai.models import Model
from pydantic_ai.settings import ModelSettings
from pydantic_ai.usage import RunUsage, UsageLimits

from ...core.agents import run_agent_with_usage
from ...core.models import resolve_model
from .render import hit_lines, passages_text
from .store import Hit, SectionStore
from .types import PassageRef, Selection, Sufficiency

EmbedFn = Callable[[list[str]], Awaitable[list[list[float]]]]

#: A model id (resolved through ``core.models.resolve_model``) or a ready model object.
ModelLike = str | Model

#: Where a result's first passage came from: selected by the agent, seen by
#: the agent, or only in round 0.
MatchType = Literal["selected", "seen", "round0"]

_READ_MAX_CHARS = 6000


@dataclass(frozen=True)
class AgentBudget:
    max_tool_calls: int = 12  # hard cap on tool calls inside the loop
    max_requests: int = 15  # hard cap on LLM requests inside the loop (UsageLimits)
    round0_k: int = 10  # passages shown to the LLM from round 0
    tool_k: int = 8  # hits returned by search/similar/grep to the LLM
    tool_rerank_candidates: int = 20  # tool hits reranked before returning (0 = none)

    def __post_init__(self) -> None:
        # Each loop request issues at least one tool call or the selection, and an
        # invalid selection costs one retry request. The tool budget must run out
        # before the request budget, so the agent always gets to finish with its
        # own selection instead of being cut off mid-search.
        if self.max_requests < self.max_tool_calls + 3:
            raise ValueError(
                f"max_requests ({self.max_requests}) must be at least "
                f"max_tool_calls + 3 ({self.max_tool_calls + 3})"
            )


@dataclass(frozen=True)
class DocumentResult:
    """One document of the results, with its passages best first."""

    path: str
    title: str
    score: float  # the score of its first passage
    match_type: MatchType
    passages: list[Hit]


@dataclass
class AgentSearchResult:
    results: list[DocumentResult]
    llm_calls: int
    tokens_in: int
    tokens_out: int
    seconds: float
    entered_loop: bool
    trace: list[dict[str, Any]] = field(default_factory=list)

    @property
    def ranked_paths(self) -> list[str]:
        return [d.path for d in self.results]


_STOP_INSTRUCTIONS = (
    "You judge search results. Given a search request and passages, set enough to true only "
    "if the passages contain everything the request needs; for a request with several "
    "steps, every step. Otherwise set enough to false and say in missing what is still "
    "needed."
)

_SIMILAR_TOOL = "similar matches by meaning, "

_LOOP_INSTRUCTIONS = (
    "You search a document store for the passages a request needs. You do not answer the "
    "request. Tools: list and glob browse paths, grep matches a regular expression, search "
    "matches keywords, {similar}outline shows a document's sections, read shows lines of a "
    "document, note records a passage as evidence. Hits are shown as "
    "path:line_from-line_to. Use the passages already found and look only for what is "
    "missing. For a request with several steps, find each intermediate entity first and "
    "then search for the next step by its name. You have at most {max_tool_calls} tool "
    "calls in total; every tool result states how many are left. Finish as soon as the "
    "passages cover the request, and when no calls are left, finish with what you have. "
    "To finish, list the passages the request needs, most important first, each with its "
    "path and, as two separate integers, line_from and line_to copied from the hit label. "
    "List only passages a tool showed you."
)


def prompts_sha256() -> str:
    """Fingerprint of every instruction text this module sends; run records keep it."""
    texts = (_STOP_INSTRUCTIONS, _LOOP_INSTRUCTIONS, _SIMILAR_TOOL)
    return hashlib.sha256("\x00".join(texts).encode()).hexdigest()


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


def _overlapping(store: SectionStore, ref: PassageRef) -> list[int]:
    """Section ids of ``ref.path`` that overlap the ref's line range; [] if none."""
    try:
        entries = store.outline(ref.path)
    except KeyError:
        return []
    return [
        e.section_id
        for e in entries
        if e.line_from <= ref.line_to and e.line_to >= ref.line_from
    ]


async def search_agentic(
    query: str,
    *,
    store: SectionStore,
    model: ModelLike,
    embed_fn: EmbedFn | None,
    reranker_model: str | None,
    run_id: str,
    limit: int = 10,
    budget: AgentBudget = AgentBudget(),
    model_settings: ModelSettings | None = None,
) -> AgentSearchResult:
    """Search the store for ``query``; the agent decides where to look.

    Returns at most ``limit`` documents. With ``embed_fn=None`` the agent works
    without embeddings: round 0 uses the lexical legs only and the ``similar``
    tool is not offered.
    """
    started = time.perf_counter()
    counter = _Counter()
    trace: list[dict[str, Any]] = []
    decided_by = model if isinstance(model, str) else model.model_name

    # Every passage the run has, and its reranker score against the query.
    hits: dict[int, Hit] = {}
    scores: dict[int, float] = {}

    async def rerank(candidates: list[Hit]) -> list[Hit]:
        if not (reranker_model and candidates):
            return candidates
        reranked = await store.rerank(query, candidates, reranker_model=reranker_model)
        for h in reranked:
            scores[h.section_id] = h.score
        return reranked

    # 1. Round 0 -------------------------------------------------------------
    query_embedding = (await embed_fn([query]))[0] if embed_fn is not None else None
    hits0 = await rerank(
        store.round0(query, query_embedding=query_embedding, use_exact=False)
    )
    hits.update({h.section_id: h for h in hits0})
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
        user_message=f"Search request: {query}\n\nPassages:\n{passages_text(top0)}",
        model_settings=model_settings,
    )
    counter.add(usage)
    choice = "round0" if sufficiency.enough else "loop"
    store.log_decision(
        run_id=run_id,
        step=0,
        kind="stop_after_round0",
        features={
            "n_hits": len(hits0),
            "top_scores": [round(h.score, 4) for h in top0[:3]],
        },
        choice=choice,
        decided_by=decided_by,
    )
    trace.append(
        {
            "step": 0,
            "kind": "stop_after_round0",
            "args": {"enough": sufficiency.enough, "missing": sufficiency.missing},
            "choice": choice,
        }
    )

    selected: list[int] = []
    seen: list[int] = []
    entered_loop = not sufficiency.enough

    # 3. The agent loop ---------------------------------------------------------
    if entered_loop:
        selected, seen = await _loop(
            query,
            store=store,
            model=model,
            decided_by=decided_by,
            embed_fn=embed_fn,
            reranker_model=reranker_model,
            run_id=run_id,
            budget=budget,
            model_settings=model_settings,
            top0=top0,
            missing=sufficiency.missing,
            rerank=rerank,
            hits=hits,
            counter=counter,
            trace=trace,
        )

    # 4. Results, deterministically ---------------------------------------------
    missing_hits = [sid for sid in dict.fromkeys(selected + seen) if sid not in hits]
    hits.update({h.section_id: h for h in store.hits(missing_hits)})
    await rerank(
        [hits[sid] for sid in dict.fromkeys(selected + seen) if sid not in scores]
    )

    def score(sid: int) -> float:
        return scores.get(sid, hits[sid].score)

    chosen = sorted(dict.fromkeys(selected), key=lambda sid: -score(sid))
    others = sorted(
        (sid for sid in dict.fromkeys(seen) if sid not in set(chosen)),
        key=lambda sid: -score(sid),
    )
    tail = [h.section_id for h in hits0 if h.section_id not in set(chosen + others)]
    tiers: list[tuple[int, MatchType]] = []
    groups: tuple[tuple[list[int], MatchType], ...] = (
        (chosen, "selected"),
        (others, "seen"),
        (tail, "round0"),
    )
    for group, tier in groups:
        tiers.extend((sid, tier) for sid in group)

    order: list[str] = []
    grouped: dict[str, list[tuple[int, MatchType]]] = {}
    for sid, tier in tiers:
        path = hits[sid].path
        if path not in grouped:
            order.append(path)
            grouped[path] = []
        grouped[path].append((sid, tier))
    order = order[:limit]
    titles = store.titles(order)
    results = [
        DocumentResult(
            path=path,
            title=titles.get(path, ""),
            score=score(grouped[path][0][0]),
            match_type=grouped[path][0][1],
            passages=[
                Hit(
                    h.section_id,
                    h.path,
                    h.heading,
                    h.line_from,
                    h.line_to,
                    h.text,
                    score(h.section_id),
                )
                for h in (hits[sid] for sid, _ in grouped[path])
            ],
        )
        for path in order
    ]
    trace.append(
        {
            "step": len(trace),
            "kind": "results",
            "args": {"selected": len(chosen), "seen": len(others), "round0": len(tail)},
            "result_paths": [d.path for d in results],
        }
    )
    return AgentSearchResult(
        results=results,
        llm_calls=counter.llm_calls,
        tokens_in=counter.tokens_in,
        tokens_out=counter.tokens_out,
        seconds=time.perf_counter() - started,
        entered_loop=entered_loop,
        trace=trace,
    )


async def _loop(
    query: str,
    *,
    store: SectionStore,
    model: ModelLike,
    decided_by: str,
    embed_fn: EmbedFn | None,
    reranker_model: str | None,
    run_id: str,
    budget: AgentBudget,
    model_settings: ModelSettings | None,
    top0: list[Hit],
    missing: str | None,
    rerank: Callable[[list[Hit]], Awaitable[list[Hit]]],
    hits: dict[int, Hit],
    counter: _Counter,
    trace: list[dict[str, Any]],
) -> tuple[list[int], list[int]]:
    """Run the tool loop; returns ``(selected, seen)`` section ids."""
    # Every section a tool showed the agent, in the order it was shown.
    shown: list[int] = [h.section_id for h in top0]
    state = {"calls": 0, "step": 0}

    def show(section_ids: list[int]) -> None:
        shown.extend(sid for sid in section_ids if sid not in set(shown))

    async def tool_hits(candidates: list[Hit]) -> list[Hit]:
        if reranker_model and budget.tool_rerank_candidates > 0:
            candidates = await rerank(candidates)
        found = candidates[: budget.tool_k]
        hits.update({h.section_id: h for h in found})
        show([h.section_id for h in found])
        return found

    def n_candidates() -> int:
        if reranker_model and budget.tool_rerank_candidates > 0:
            return max(budget.tool_rerank_candidates, budget.tool_k)
        return budget.tool_k

    async def call(
        name: str,
        args: dict[str, Any],
        fn: Callable[[], Awaitable[tuple[str, list[Hit] | None]]],
    ) -> str:
        """Run one tool under the budget; every outcome is logged and traced."""
        state["step"] += 1
        step = state["step"]
        compact = ", ".join(f"{k}={v!r}" for k, v in args.items())
        if state["calls"] >= budget.max_tool_calls:
            msg = "ERROR: no tool calls left; finish now with the passages you have"
            trace.append(
                {"step": step, "kind": f"tool:{name}", "args": args, "error": msg}
            )
            return msg
        state["calls"] += 1
        try:
            text, found = await fn()
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
        left = budget.max_tool_calls - state["calls"]
        text = (
            f"{text}\n[tool calls left: {left}]"
            if left
            else f"{text}\n[no tool calls left: finish now with the passages you have]"
        )
        trace.append(
            {
                "step": step,
                "kind": f"tool:{name}",
                "args": args,
                "result_paths": [h.citation for h in found or []],
            }
        )
        store.log_decision(
            run_id=run_id,
            step=step,
            kind=f"tool:{name}",
            features={"n_hits": len(found or [])},
            choice=compact,
            decided_by=decided_by,
        )
        return text

    agent: Agent[None, Selection] = Agent(
        resolve_model(model) if isinstance(model, str) else model,
        instructions=_LOOP_INSTRUCTIONS.format(
            max_tool_calls=budget.max_tool_calls,
            similar=_SIMILAR_TOOL if embed_fn is not None else "",
        ),
        output_type=Selection,
        retries={"tools": 3, "output": 5},
    )

    @agent.output_validator
    def check(selection: Selection) -> Selection:
        """Reject a selection that names a passage no tool showed the agent."""
        from pydantic_ai import ModelRetry

        problems = [
            f"{r.path}:{r.line_from}-{r.line_to}"
            for r in selection.passages
            if not set(_overlapping(store, r)) & set(shown)
        ]
        if problems:
            raise ModelRetry(
                "These passages were never shown to you or do not exist: "
                + "; ".join(problems)
                + ". List only passages from tool results, copying path and lines "
                "from the hit labels."
            )
        return selection

    @agent.tool_plain(name="list")
    async def list_documents(prefix: str = "") -> str:
        """List documents whose path starts with prefix (empty prefix lists the first documents)."""

        async def go() -> tuple[str, list[Hit] | None]:
            docs = store.list_documents(prefix=prefix, limit=50)
            lines = [f"{d.path} | {d.title} | {d.n_sections} sections" for d in docs]
            return ("\n".join(lines) or "(no documents)"), None

        return await call("list", {"prefix": prefix}, go)

    @agent.tool_plain
    async def glob(pattern: str) -> str:
        """List documents whose path matches a glob pattern such as 'papers/*2020*'."""

        async def go() -> tuple[str, list[Hit] | None]:
            docs = store.glob(pattern, limit=50)
            lines = [f"{d.path} | {d.title} | {d.n_sections} sections" for d in docs]
            return ("\n".join(lines) or "(no documents)"), None

        return await call("glob", {"pattern": pattern}, go)

    @agent.tool_plain
    async def grep(regex: str, path_prefix: str = "") -> str:
        """Find passages matching a regular expression, optionally within a path prefix."""

        async def go() -> tuple[str, list[Hit] | None]:
            found = await tool_hits(
                store.grep(regex, path_prefix=path_prefix, limit=n_candidates())
            )
            return hit_lines(found), found

        return await call("grep", {"regex": regex, "path_prefix": path_prefix}, go)

    @agent.tool_plain
    async def search(keywords: str, path_prefix: str = "") -> str:
        """Keyword search (BM25) over passages, optionally within a path prefix."""

        async def go() -> tuple[str, list[Hit] | None]:
            found = await tool_hits(
                store.search(keywords, k=n_candidates(), path_prefix=path_prefix)
            )
            return hit_lines(found), found

        return await call(
            "search", {"keywords": keywords, "path_prefix": path_prefix}, go
        )

    if embed_fn is not None:
        embed = embed_fn

        @agent.tool_plain
        async def similar(text: str, path_prefix: str = "") -> str:
            """Find passages similar in meaning to text, optionally within a path prefix."""

            async def go() -> tuple[str, list[Hit] | None]:
                embedding = (await embed([text]))[0]
                found = await tool_hits(
                    store.similar(embedding, k=n_candidates(), path_prefix=path_prefix)
                )
                return hit_lines(found), found

            return await call("similar", {"text": text, "path_prefix": path_prefix}, go)

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

        return await call("outline", {"path": path}, go)

    @agent.tool_plain
    async def read(path: str, offset: int = 1, limit: int = 60) -> str:
        """Read lines of a document starting at the 1-based line offset."""

        async def go() -> tuple[str, list[Hit] | None]:
            text = store.read(path, offset=offset, limit=limit)[:_READ_MAX_CHARS]
            show(
                _overlapping(
                    store,
                    PassageRef(path=path, line_from=offset, line_to=offset + limit - 1),
                )
            )
            return text, None

        return await call("read", {"path": path, "offset": offset, "limit": limit}, go)

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
            show([best.section_id])
            return f"noted {path}:{line_from}-{line_to}", None

        return await call(
            "note",
            {"path": path, "line_from": line_from, "line_to": line_to, "claim": claim},
            go,
        )

    prompt = (
        f"Search request: {query}\n\nPassages from the first search:\n"
        f"{passages_text(top0)}\n\nStill missing: {missing or '(not stated)'}"
    )
    loop_usage = RunUsage()
    try:
        result = await agent.run(
            prompt,
            usage=loop_usage,
            usage_limits=UsageLimits(request_limit=budget.max_requests),
            model_settings=model_settings,
        )
        counter.add(loop_usage)
        refs = result.output.passages
        selected = [
            sid for r in refs for sid in _overlapping(store, r) if sid in set(shown)
        ]
        trace.append(
            {
                "step": state["step"] + 1,
                "kind": "selection",
                "args": {"n_passages": len(refs)},
                "result_paths": [f"{r.path}:{r.line_from}-{r.line_to}" for r in refs],
            }
        )
    except (UsageLimitExceeded, UnexpectedModelBehavior) as exc:
        # The loop ran out of requests, or the model could not produce a valid
        # selection after every retry. What it noted is its own judgement of
        # the evidence, so the noted passages become the selection.
        counter.add(loop_usage)
        selected = [sid for sid, _, _, _ in store.evidence(run_id)]
        trace.append(
            {
                "step": state["step"] + 1,
                "kind": (
                    "request_limit_fallback"
                    if isinstance(exc, UsageLimitExceeded)
                    else "invalid_selection_fallback"
                ),
                "args": {"n_noted": len(selected)},
                "error": str(exc),
            }
        )
    return selected, shown


__all__ = [
    "AgentBudget",
    "AgentSearchResult",
    "DocumentResult",
    "EmbedFn",
    "MatchType",
    "prompts_sha256",
    "search_agentic",
]
