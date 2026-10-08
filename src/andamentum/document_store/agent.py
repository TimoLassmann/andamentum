"""Agent-run search: the default search first, a stop decision, then a tool loop.

``AgentSearcher`` is an opt-in searcher. Its output has the same shape as the
default searcher's: passages, best first. The agent never answers the request
and never writes prose that reaches the caller. It decides only where to look
and which passages the request needs; the result is then built
deterministically from those passages.

Round 0 is the inner searcher's candidates, reranked against the request (no
LLM). One structured call decides whether the top passages cover the request;
if so they are the result. Otherwise a pydantic-ai agent with store tools
(list, glob, grep, search, outline, read, note) searches until it finishes
with a ``Selection``: the passages the request needs, by document and line
range. A selection may name only passages a tool showed it, so it cannot point
at a source it never saw. The ``search`` tool is the inner searcher itself,
reranked against the tool's own query.

Result order: the selected passages, then every other passage the agent saw,
each group by reranker score against the request, then the rest of round 0 in
its reranked order. Every ``Hit.score`` is the reranker score against the
request, so it means the same as in the default searcher; the order carries
the agent's judgement. Use ``group_by_document`` to group, which keeps that
order.

A failed tool call is returned to the model as an ``ERROR: ...`` string and
recorded in the trace. If the loop runs out of requests, or the model cannot
produce a valid selection, the passages it noted become the selection; the
trace records which case it was. No extra model call is made for it.
"""

from __future__ import annotations

import hashlib
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel
from pydantic_ai import Agent, ModelRetry
from pydantic_ai.exceptions import UnexpectedModelBehavior, UsageLimitExceeded
from pydantic_ai.models import Model
from pydantic_ai.settings import ModelSettings
from pydantic_ai.usage import RunUsage, UsageLimits

from ..core.agents import run_agent_with_usage
from ..core.models import resolve_model
from .index import IndexedChunk, Index
from .searchers import HybridRerankSearcher
from .types import Hit, Where, line_range

if TYPE_CHECKING:
    from .store import Store

#: A model id (resolved through ``core.models.resolve_model``) or a ready model object.
ModelLike = str | Model

_READ_MAX_CHARS = 6000
_SNIPPET_CHARS = 300
_PASSAGE_CHARS = 1200
_LIST_LIMIT = 50


class Sufficiency(BaseModel):
    """The stop decision after round 0: do the passages cover the request?"""

    enough: bool
    missing: str | None = None  # what the passages lack; seeds the search loop


class PassageRef(BaseModel):
    """One passage, by document and line range."""

    doc: str
    line_from: int
    line_to: int


class Selection(BaseModel):
    """The search agent's last step: the passages the request needs, most important first."""

    passages: list[PassageRef] = []


@dataclass(frozen=True)
class AgentBudget:
    max_tool_calls: int = 12  # hard cap on tool calls inside the loop
    max_requests: int = 15  # hard cap on LLM requests inside the loop (UsageLimits)
    round0_k: int = 10  # passages shown to the LLM from round 0
    tool_k: int = 8  # hits returned by search/grep to the LLM

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


@dataclass
class AgentRun:
    """What the last ``search`` call did: model usage and a step-by-step trace."""

    llm_calls: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    entered_loop: bool = False
    trace: list[dict[str, Any]] = field(default_factory=list)

    def add(self, usage: Any) -> None:
        self.llm_calls += int(usage.requests)
        self.tokens_in += int(usage.input_tokens)
        self.tokens_out += int(usage.output_tokens)


_STOP_INSTRUCTIONS = (
    "You judge search results. Given a search request and passages, set enough to true only "
    "if the passages contain everything the request needs; for a request with several "
    "steps, every step. Otherwise set enough to false and say in missing what is still "
    "needed."
)

_LOOP_INSTRUCTIONS = (
    "You search a document store for the passages a request needs. You do not answer the "
    "request. Tools: list and glob browse document names, grep matches a regular expression, "
    "search finds passages by meaning and keywords, outline shows a document's sections, "
    "read shows lines of a document, note records a passage as evidence. Hits are shown as "
    "doc:line_from-line_to. Use the passages already found and look only for what is "
    "missing. For a request with several steps, find each intermediate entity first and "
    "then search for the next step by its name. You have at most {max_tool_calls} tool "
    "calls in total; every tool result states how many are left. Finish as soon as the "
    "passages cover the request, and when no calls are left, finish with what you have. "
    "To finish, list the passages the request needs, most important first, each with its "
    "doc and, as two separate integers, line_from and line_to copied from the hit label. "
    "List only passages a tool showed you."
)


def prompts_sha256() -> str:
    """Fingerprint of every instruction text this module sends; run records keep it."""
    texts = (_STOP_INSTRUCTIONS, _LOOP_INSTRUCTIONS)
    return hashlib.sha256("\x00".join(texts).encode()).hexdigest()


class _Docs:
    """Per-run view of documents: names, texts and line numbers, cached."""

    def __init__(self, store: Store, index: Index) -> None:
        self.store = store
        self.index = index
        self._text: dict[str, str] = {}
        self._name: dict[str, str] = {}

    async def text(self, doc_id: str) -> str:
        if doc_id not in self._text:
            doc = await self.store.get(doc_id)
            if doc is None:
                raise KeyError(f"no document {doc_id}")
            self._text[doc_id] = doc.text
            self._name[doc_id] = doc.key or doc.doc_id
        return self._text[doc_id]

    async def name(self, doc_id: str) -> str:
        await self.text(doc_id)
        return self._name[doc_id]

    async def doc_id(self, name: str) -> str:
        """A document named by its key, or by its id."""
        found = await self.store.resolve(name)
        if found is None and await self.store.get(name) is not None:
            found = name
        if found is None:
            raise KeyError(f"no document {name!r}")
        return found

    async def lines(self, h: Hit | IndexedChunk) -> tuple[int, int]:
        return line_range(await self.text(h.doc_id), start=h.start, end=h.end)

    async def label(self, h: Hit | IndexedChunk) -> str:
        a, b = await self.lines(h)
        return f"{await self.name(h.doc_id)}:{a}-{b}"


def _key_filter(store: Store, where: Where | None, *, pattern: str | None, glob: bool) -> tuple[str, list[object]] | None:
    """Documents matching ``where`` whose name (key, else id) has this prefix or glob."""
    if not where and not pattern:
        return None
    sql, params = store.where_docs(where)
    if pattern:
        if glob:
            sql += " AND COALESCE(d.key, d.id) GLOB ?"
            params.append(pattern)
        else:
            escaped = pattern.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
            sql += " AND COALESCE(d.key, d.id) LIKE ? ESCAPE '\\'"
            params.append(escaped + "%")
    return sql, params


class AgentSearcher:
    """Model-driven search on top of an inner :class:`HybridRerankSearcher`.

    ``model`` has no default. ``last_run`` holds the usage and trace of the
    most recent ``search`` call; concurrent callers use ``search_with_run``.
    """

    def __init__(
        self,
        *,
        model: ModelLike,
        inner: HybridRerankSearcher,
        budget: AgentBudget = AgentBudget(),
        model_settings: ModelSettings | None = None,
    ) -> None:
        self.model = model
        self.inner = inner
        self.budget = budget
        self.model_settings = model_settings
        self.last_run = AgentRun()

    def indexes(self) -> Sequence[Index]:
        return self.inner.indexes()

    async def search(
        self, store: Store, query: str, *, limit: int, where: Where | None
    ) -> list[Hit]:
        hits, self.last_run = await self.search_with_run(
            store, query, limit=limit, where=where
        )
        return hits

    async def search_with_run(
        self, store: Store, query: str, *, limit: int, where: Where | None
    ) -> tuple[list[Hit], AgentRun]:
        """``search`` plus this call's usage and trace; safe to run concurrently."""
        run = AgentRun()
        budget = self.budget
        docs = _Docs(store, self.inner.index)
        allowed = store.where_docs(where) if where else None

        # Every passage the run has, keyed by (doc_id, start, end), and its
        # reranker score against the request.
        hits: dict[tuple[str, int, int], Hit] = {}
        scores: dict[tuple[str, int, int], float] = {}

        def key(h: Hit) -> tuple[str, int, int]:
            return (h.doc_id, h.start, h.end)

        async def rerank_to_request(candidates: list[Hit]) -> list[Hit]:
            reranked = await self.inner.rerank(store, query, candidates)
            for h in reranked:
                scores[key(h)] = h.score
            return reranked

        # 1. Round 0 ---------------------------------------------------------
        cands = await self.inner.candidates(
            store, query, depth=max(self.inner.depth, limit), allowed=allowed
        )
        hits0 = await rerank_to_request(cands)
        hits.update({key(h): h for h in hits0})
        top0 = hits0[: budget.round0_k]
        run.trace.append(
            {"step": 0, "kind": "round0", "result_paths": [await docs.label(h) for h in top0]}
        )

        # 2. Stop decision ---------------------------------------------------
        sufficiency, usage = await run_agent_with_usage(
            self.model,
            instructions=_STOP_INSTRUCTIONS,
            output_type=Sufficiency,
            user_message=f"Search request: {query}\n\nPassages:\n{await self._passages(docs, top0)}",
            model_settings=self.model_settings,
        )
        run.add(usage)
        run.entered_loop = not sufficiency.enough
        run.trace.append(
            {
                "step": 0,
                "kind": "stop_after_round0",
                "args": {"enough": sufficiency.enough, "missing": sufficiency.missing},
                "choice": "loop" if run.entered_loop else "round0",
            }
        )

        selected: list[tuple[str, int, int]] = []
        seen: list[tuple[str, int, int]] = []
        if run.entered_loop:
            selected, seen = await self._loop(
                query, store=store, docs=docs, where=where, top0=top0,
                missing=sufficiency.missing, hits=hits, run=run,
            )

        # 3. Result, deterministically -----------------------------------------
        unscored = [hits[k] for k in dict.fromkeys(selected + seen) if k not in scores]
        await rerank_to_request(unscored)

        def score(k: tuple[str, int, int]) -> float:
            return scores.get(k, hits[k].score)

        chosen = sorted(dict.fromkeys(selected), key=lambda k: -score(k))
        others = sorted(
            (k for k in dict.fromkeys(seen) if k not in set(chosen)), key=lambda k: -score(k)
        )
        tail = [key(h) for h in hits0 if key(h) not in set(chosen + others)]
        order = chosen + others + tail
        run.trace.append(
            {
                "step": len(run.trace),
                "kind": "results",
                "args": {"selected": len(chosen), "seen": len(others), "round0": len(tail)},
            }
        )
        out: list[Hit] = []
        for k in order[:limit]:
            h = hits[k]
            out.append(Hit(h.doc_id, h.rev, h.start, h.end, h.text, score(k), h.heading_path))
        return out, run

    async def _passages(self, docs: _Docs, hits: list[Hit]) -> str:
        """Each hit as its label and up to 1200 characters of text."""
        if not hits:
            return "(no passages)"
        return "\n\n".join(
            [f"[{await docs.label(h)}]\n{h.text[:_PASSAGE_CHARS]}" for h in hits]
        )

    async def _hit_lines(self, docs: _Docs, hits: Sequence[Hit]) -> str:
        if not hits:
            return "(no results)"
        lines = []
        for h in hits:
            snippet = " ".join(h.text.split())[:_SNIPPET_CHARS]
            heading = " > ".join(h.heading_path)
            lines.append(f"{await docs.label(h)} | {heading} | {snippet}")
        return "\n".join(lines)

    async def _loop(
        self,
        query: str,
        *,
        store: Store,
        docs: _Docs,
        where: Where | None,
        top0: list[Hit],
        missing: str | None,
        hits: dict[tuple[str, int, int], Hit],
        run: AgentRun,
    ) -> tuple[list[tuple[str, int, int]], list[tuple[str, int, int]]]:
        """Run the tool loop; returns ``(selected, seen)`` passage keys."""
        budget = self.budget
        index = self.inner.index
        db = store.db
        shown: list[tuple[str, int, int]] = [(h.doc_id, h.start, h.end) for h in top0]
        noted: list[tuple[str, int, int]] = []
        state = {"calls": 0, "step": 0}

        def remember(found: Sequence[Hit]) -> None:
            for h in found:
                k = (h.doc_id, h.start, h.end)
                hits.setdefault(k, h)
                if k not in set(shown):
                    shown.append(k)

        def as_hit(c: IndexedChunk, score: float = 0.0) -> Hit:
            return Hit(c.doc_id, c.rev, c.start, c.end, c.text, score, c.heading_path)

        async def overlapping(ref: PassageRef) -> list[IndexedChunk]:
            """Chunks of ``ref.doc`` that overlap its line range; [] if none."""
            try:
                doc_id = await docs.doc_id(ref.doc)
            except KeyError:
                return []
            out = []
            for c in index.document_chunks(db, doc_id):
                a, b = await docs.lines(c)
                if a <= ref.line_to and b >= ref.line_from:
                    out.append(c)
            return out

        async def call(
            name: str, args: dict[str, Any], fn: Callable[[], Awaitable[tuple[str, Sequence[Hit]]]]
        ) -> str:
            """Run one tool under the budget; every outcome is traced."""
            state["step"] += 1
            step = state["step"]
            if state["calls"] >= budget.max_tool_calls:
                msg = "ERROR: no tool calls left; finish now with the passages you have"
                run.trace.append({"step": step, "kind": f"tool:{name}", "args": args, "error": msg})
                return msg
            state["calls"] += 1
            try:
                text, found = await fn()
            except Exception as exc:  # returned to the model and traced, not swallowed
                msg = f"ERROR: {type(exc).__name__}: {exc}"
                run.trace.append({"step": step, "kind": f"tool:{name}", "args": args, "error": msg})
                return msg
            remember(found)
            left = budget.max_tool_calls - state["calls"]
            text = (
                f"{text}\n[tool calls left: {left}]"
                if left
                else f"{text}\n[no tool calls left: finish now with the passages you have]"
            )
            run.trace.append(
                {
                    "step": step,
                    "kind": f"tool:{name}",
                    "args": args,
                    "result_paths": [await docs.label(h) for h in found],
                }
            )
            return text

        agent: Agent[None, Selection] = Agent(
            resolve_model(self.model) if isinstance(self.model, str) else self.model,
            instructions=_LOOP_INSTRUCTIONS.format(max_tool_calls=budget.max_tool_calls),
            output_type=Selection,
            retries={"tools": 3, "output": 5},
        )

        @agent.output_validator
        async def check(selection: Selection) -> Selection:
            """Reject a selection that names a passage no tool showed the agent."""
            problems = []
            for r in selection.passages:
                ks = {(c.doc_id, c.start, c.end) for c in await overlapping(r)}
                if not ks & set(shown):
                    problems.append(f"{r.doc}:{r.line_from}-{r.line_to}")
            if problems:
                raise ModelRetry(
                    "These passages were never shown to you or do not exist: "
                    + "; ".join(problems)
                    + ". List only passages from tool results, copying doc and lines "
                    "from the hit labels."
                )
            return selection

        async def doc_list(flt: tuple[str, list[object]] | None) -> str:
            sql = (
                "SELECT d.id, COALESCE(d.key, d.id) AS name FROM documents d "
                "WHERE d.deleted_at IS NULL"
            )
            params: list[object] = []
            if flt is not None:
                sql += f" AND d.id IN ({flt[0]})"
                params = list(flt[1])
            sql += " ORDER BY name LIMIT ?"
            rows = db.execute(sql, [*params, _LIST_LIMIT]).fetchall()
            out = []
            for doc_id, name in rows:
                n = len(index.document_chunks(db, doc_id))
                doc = await store.get(doc_id)
                t = doc.metadata.get("title", "") if doc else ""
                out.append(f"{name} | {t} | {n} sections")
            return "\n".join(out) or "(no documents)"

        @agent.tool_plain(name="list")
        async def list_documents(prefix: str = "") -> str:
            """List documents whose name starts with prefix (empty prefix lists the first documents)."""

            async def go() -> tuple[str, Sequence[Hit]]:
                return await doc_list(_key_filter(store, where, pattern=prefix, glob=False)), []

            return await call("list", {"prefix": prefix}, go)

        @agent.tool_plain
        async def glob(pattern: str) -> str:
            """List documents whose name matches a glob pattern such as 'papers/*2020*'."""

            async def go() -> tuple[str, Sequence[Hit]]:
                return await doc_list(_key_filter(store, where, pattern=pattern, glob=True)), []

            return await call("glob", {"pattern": pattern}, go)

        @agent.tool_plain
        async def grep(regex: str, prefix: str = "") -> str:
            """Find passages matching a regular expression, optionally within documents whose name starts with prefix."""

            async def go() -> tuple[str, Sequence[Hit]]:
                found = [
                    as_hit(c, float(n))
                    for c, n in index.grep(
                        db, regex, limit=budget.tool_k,
                        allowed=_key_filter(store, where, pattern=prefix, glob=False),
                    )
                ]
                return await self._hit_lines(docs, found), found

            return await call("grep", {"regex": regex, "prefix": prefix}, go)

        @agent.tool_plain
        async def search(query_text: str, prefix: str = "") -> str:
            """Search passages by meaning and keywords, optionally within documents whose name starts with prefix."""

            async def go() -> tuple[str, Sequence[Hit]]:
                cands = await self.inner.candidates(
                    store, query_text, depth=self.inner.depth,
                    allowed=_key_filter(store, where, pattern=prefix, glob=False),
                )
                found = (await self.inner.rerank(store, query_text, cands))[: budget.tool_k]
                return await self._hit_lines(docs, found), found

            return await call("search", {"query_text": query_text, "prefix": prefix}, go)

        @agent.tool_plain
        async def outline(doc: str) -> str:
            """Show the sections of one document with their line ranges."""

            async def go() -> tuple[str, Sequence[Hit]]:
                doc_id = await docs.doc_id(doc)
                lines = []
                for c in index.document_chunks(db, doc_id):
                    a, b = await docs.lines(c)
                    depth = max(len(c.heading_path) - 1, 0)
                    heading = c.heading_path[-1] if c.heading_path else "(start)"
                    lines.append(f"{'  ' * depth}{heading} | lines {a}-{b}")
                return ("\n".join(lines) or "(no sections)"), []

            return await call("outline", {"doc": doc}, go)

        @agent.tool_plain
        async def read(doc: str, offset: int = 1, limit: int = 60) -> str:
            """Read lines of a document starting at the 1-based line offset."""

            async def go() -> tuple[str, Sequence[Hit]]:
                if offset < 1 or limit < 1:
                    raise ValueError("offset and limit must be >= 1")
                doc_id = await docs.doc_id(doc)
                lines = (await docs.text(doc_id)).split("\n")[offset - 1 : offset - 1 + limit]
                # Whole lines up to the size cap; only those count as shown.
                shown_lines: list[str] = []
                size = 0
                for i, line in enumerate(lines):
                    numbered = f"{offset + i}: {line}"
                    if shown_lines and size + len(numbered) + 1 > _READ_MAX_CHARS:
                        break
                    shown_lines.append(numbered[:_READ_MAX_CHARS])
                    size += len(numbered) + 1
                if not shown_lines:
                    return "(no lines)", []
                last = offset + len(shown_lines) - 1
                seen_now = await overlapping(PassageRef(doc=doc, line_from=offset, line_to=last))
                return "\n".join(shown_lines), [as_hit(c) for c in seen_now]

            return await call("read", {"doc": doc, "offset": offset, "limit": limit}, go)

        @agent.tool_plain
        async def note(doc: str, line_from: int, line_to: int, claim: str) -> str:
            """Record a passage (doc and line range) as evidence for the claim it supports."""

            async def go() -> tuple[str, Sequence[Hit]]:
                if line_from > line_to:
                    raise ValueError("line_from must be <= line_to")
                cs = await overlapping(PassageRef(doc=doc, line_from=line_from, line_to=line_to))
                if not cs:
                    raise ValueError(f"no section of {doc} overlaps lines {line_from}-{line_to}")
                best, best_overlap = cs[0], -1
                for c in cs:
                    a, b = await docs.lines(c)
                    ov = min(b, line_to) - max(a, line_from)
                    if ov > best_overlap:
                        best, best_overlap = c, ov
                noted.append((best.doc_id, best.start, best.end))
                return f"noted {doc}:{line_from}-{line_to}", [as_hit(best)]

            return await call(
                "note", {"doc": doc, "line_from": line_from, "line_to": line_to, "claim": claim}, go
            )

        prompt = (
            f"Search request: {query}\n\nPassages from the first search:\n"
            f"{await self._passages(docs, top0)}\n\nStill missing: {missing or '(not stated)'}"
        )
        loop_usage = RunUsage()
        try:
            result = await agent.run(
                prompt,
                usage=loop_usage,
                usage_limits=UsageLimits(request_limit=budget.max_requests),
                model_settings=self.model_settings,
            )
            run.add(loop_usage)
            refs = result.output.passages
            selected = []
            for r in refs:
                for c in await overlapping(r):
                    k = (c.doc_id, c.start, c.end)
                    if k in set(shown):
                        selected.append(k)
            run.trace.append(
                {
                    "step": state["step"] + 1,
                    "kind": "selection",
                    "args": {"n_passages": len(refs)},
                    "result_paths": [f"{r.doc}:{r.line_from}-{r.line_to}" for r in refs],
                }
            )
        except (UsageLimitExceeded, UnexpectedModelBehavior) as exc:
            # The loop ran out of requests, or the model could not produce a valid
            # selection after every retry. What it noted is its own judgement of
            # the evidence, so the noted passages become the selection.
            run.add(loop_usage)
            selected = list(noted)
            run.trace.append(
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
