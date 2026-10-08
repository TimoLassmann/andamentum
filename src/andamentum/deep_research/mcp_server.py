"""The MCP server: ``andamentum-research mcp --store PATH`` puts the research engine and its store in reach of an agent.

A stdio server over the ``mcp`` SDK. Every tool writes only to the store.

Tools:

- ``search_works`` and ``expand_citations`` find scholarly works (OpenAlex and
  the other adapters) and one-hop references or citations; ``search_web``
  searches the web (needs ``--searxng-url`` or ``--browser-cdp-url``). All
  three merge duplicates and keep the works in the store.
- ``read_work`` returns a stored work with its passages. With ``fetch`` it first
  checks Crossref for a retraction, loads the text (full text where open,
  else the abstract, else the search snippet) and stores the most relevant
  window of it as passages, with no model. With ``subquestion`` and ``run_id``
  it runs the engine's reader on the work: one notes call over the window,
  quotes checked in code, findings recorded in the run (a quote that cannot
  be found is kept and marked unverified).
- ``record_finding`` stores a claim with the verbatim quote behind it, after
  checking that the quote occurs in one of the work's passages (or in its
  abstract, when the licence allows the abstract to be used).
- ``get_run`` reads a run's rows back; ``research`` runs the whole engine
  against the server's store.

Calls that use a model run one at a time: a local model serves one request at
a time.

Resources: ``research://run/<run_id>``, one per run in the store.
"""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any, Literal

import anyio
from mcp import types
from mcp.server.lowlevel import Server
from mcp.server.lowlevel.helper_types import ReadResourceContents
from mcp.server.stdio import stdio_server
from pydantic import (
    AnyUrl,
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    ValidationError,
)
from pydantic_ai.models import Model

from andamentum.references import Work

from .ask import Asker
from .engine import research
from .identity import merge
from .read import ReadDeps, load_page, locate_quote, read_page
from .sources import (
    ExpandingSource,
    SourceClient,
    SourceConfig,
    SourceError,
    SourceSet,
    build_sources,
)
from .store import ResearchStore, StoreError
from .types import (
    Budget,
    Candidate,
    Finding,
    Mode,
    Passage,
    RunStats,
    SubQuestion,
)

SERVER_NAME = "andamentum-research"
RUN_URI_PREFIX = "research://run/"
DEFAULT_LIMIT = 20
MAX_LIMIT = 200
DEFAULT_READ_CHARS = 20_000
MCP_RUN_MODE = "mcp"
SCHOLARLY_SOURCES: tuple[str, ...] = (
    "openalex",
    "europepmc",
    "arxiv",
    "semantic_scholar",
)
_FINDING_ID = re.compile(r"F(\d+)")

JsonObject = dict[str, JsonValue]


class _Params(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SearchWorksParams(_Params):
    query: str = Field(min_length=1, description="keywords for the scholarly sources")
    sources: list[Literal["openalex", "europepmc", "arxiv", "semantic_scholar"]] = (
        Field(
            default_factory=lambda: ["openalex"],
            description=f"which sources to ask: {', '.join(SCHOLARLY_SOURCES)}",
        )
    )
    limit: int = Field(
        default=DEFAULT_LIMIT, ge=1, le=MAX_LIMIT, description="results per source"
    )


class SearchWebParams(_Params):
    query: str = Field(min_length=1, description="what to search the web for")
    limit: int = Field(default=DEFAULT_LIMIT, ge=1, le=MAX_LIMIT)


class ExpandCitationsParams(_Params):
    work_id: str = Field(min_length=1, description="a work in the store")
    direction: Literal["references", "citations", "both"] = "both"
    limit: int = Field(default=DEFAULT_LIMIT, ge=1, le=MAX_LIMIT)
    source: Literal["openalex", "europepmc", "semantic_scholar"] = Field(
        default="openalex", description="the source that knows the citation graph"
    )


class ReadWorkParams(_Params):
    work_id: str = Field(min_length=1, description="a work in the store")
    fetch: bool = Field(
        default=False,
        description="check for a retraction, load the text (full text when open, "
        "else the abstract) and store its passages; no model is used",
    )
    subquestion: str | None = Field(
        default=None,
        description="read the work for this question with the engine's reader "
        "(passage selection, finding extraction, quotes checked); needs run_id",
    )
    run_id: str | None = Field(
        default=None, description="the run the reader's findings are recorded in"
    )
    max_chars: int = Field(
        default=DEFAULT_READ_CHARS,
        ge=1,
        description="cap on the passage text returned",
    )


class RecordFindingParams(_Params):
    run_id: str = Field(min_length=1, description="a run in the store")
    subquestion_id: str = Field(min_length=1)
    work_id: str = Field(min_length=1, description="the work the quote is from")
    claim: str = Field(min_length=1, description="what the work says, in one sentence")
    quote: str = Field(
        min_length=1, description="verbatim text from the work that states the claim"
    )


class GetRunParams(_Params):
    run_id: str | None = Field(
        default=None, description="the run to read; omit to list every run"
    )
    kind: str | None = Field(default=None, description="only rows of this kind")


class ResearchParams(_Params):
    question: str = Field(min_length=1)
    mode: Mode = "both"
    budget: Budget = Field(default_factory=Budget)


def _jsonable(value: BaseModel | list[BaseModel]) -> JsonValue:
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    return json.loads(value.model_dump_json())


class ResearchTools:
    """The tools, bound to one store, its HTTP client and the server's model."""

    def __init__(
        self,
        store: ResearchStore,
        *,
        model: str | Model | None = None,
        sources: SourceConfig | None = None,
        client: SourceClient | None = None,
    ) -> None:
        if client is not None and client.store is not store:
            raise ValueError("client belongs to another store")
        self.store = store
        self.model = model
        self.config = sources or SourceConfig()
        self.client = client or SourceClient(
            store, pdf_extraction=self.config.pdf_extraction
        )
        self._owns_client = client is None
        self._model_lock = anyio.Lock()

    async def aclose(self) -> None:
        if self._owns_client:
            await self.client.aclose()

    # -- what the tools share ---------------------------------------------------

    def _sources(
        self, mode: Mode, *, scholarly: tuple[str, ...] | None = None
    ) -> SourceSet:
        config = self.config
        if scholarly is not None:
            config = config.model_copy(update={"scholarly": scholarly})
        return build_sources(config, store=self.store, mode=mode, client=self.client)

    async def _asker(self, run_id: str) -> Asker:
        """The run's models behind one Asker, counted from zero for this call."""
        return Asker(
            model=self.model,
            store=self.store,
            run_id=run_id,
            budget=Budget(),
            stats=RunStats(),
        )

    def _work(self, work_id: str) -> Work:
        resolved = self.store.resolve_work_id(work_id)
        work = self.store.get_work(resolved) if resolved else None
        if work is None:
            raise ValueError(f"no work {work_id!r} in the store")
        return work

    def _run(self, run_id: str) -> None:
        if self.store.get_run(run_id) is None:
            raise ValueError(f"no run {run_id!r} in the store")

    def _start_run(self, *, question: str, tool: str) -> str:
        return self.store.start_run(
            question=question,
            mode=MCP_RUN_MODE,
            config={"tool": tool},
            models={"model": self.model if isinstance(self.model, str) else None},
        )

    def _subquestion(self, run_id: str, text: str) -> SubQuestion:
        """The run's sub-question with this text, recorded with the next free
        id when the run has none, so several calls into one run never collide."""
        existing = [
            SubQuestion.model_validate(i.data)
            for i in self.store.run_items(run_id, "subquestion")
        ]
        for subquestion in existing:
            if subquestion.text == text:
                return subquestion
        subquestion = SubQuestion(id=f"Q{len(existing) + 1}", text=text)
        self.store.add_run_item(run_id, "subquestion", subquestion.id, subquestion)
        return subquestion

    def _next_finding_number(self, run_id: str) -> int:
        numbers = [
            int(found.group(1))
            for item in self.store.run_items(run_id, "finding")
            if (found := _FINDING_ID.fullmatch(item.ref_id))
        ]
        return max(numbers, default=0) + 1

    def _keep(self, candidates: list[Candidate]) -> list[JsonValue]:
        """Merge duplicates, store the works, and describe them."""
        described: list[JsonValue] = []
        for candidate in merge(candidates):
            work = self.store.upsert_work(candidate.work)
            row: JsonObject = {
                "work_id": work.work_id,
                "title": work.title,
                "year": work.year,
                "doi": work.doi,
                "url": work.url,
                "access": work.access,
                "retracted": work.retracted,
                "sources": list(candidate.sources),
                "snippet": candidate.snippet if work.redistributable else "",
            }
            described.append(row)
        return described

    # -- the tools ----------------------------------------------------------

    async def search_works(self, params: SearchWorksParams) -> JsonObject:
        names = tuple(dict.fromkeys(params.sources))
        sources = self._sources("scholarly", scholarly=names)
        found: list[Candidate] = []
        errors: JsonObject = {}
        for source in sources.search:
            try:
                found += await source.search(params.query, limit=params.limit)
            except SourceError as error:  # reported to the caller, not hidden
                errors[source.name] = error.reason
        if errors and len(errors) == len(sources.search):
            raise ValueError(
                "every source failed: "
                + "; ".join(f"{n}: {e}" for n, e in errors.items())
            )
        return {"works": self._keep(found), "source_errors": errors}

    async def search_web(self, params: SearchWebParams) -> JsonObject:
        if not (self.config.searxng_url or self.config.browser_cdp_url):
            raise ValueError(
                "search_web needs a SearxNG instance or a Chrome: start the server "
                "with --searxng-url or --browser-cdp-url"
            )
        (web,) = self._sources("web").search
        found = await web.search(params.query, limit=params.limit)
        return {"works": self._keep(found), "source_errors": {}}

    async def expand_citations(self, params: ExpandCitationsParams) -> JsonObject:
        work = self._work(params.work_id)
        sources = self._sources("scholarly", scholarly=(params.source,))
        expanding: list[ExpandingSource] = [
            s for s in sources.expand if s.name == params.source
        ]
        if not expanding:
            raise ValueError(f"the {params.source} source does not follow citations")
        source = expanding[0]
        directions: tuple[Literal["references", "citations"], ...] = (
            ("references", "citations")
            if params.direction == "both"
            else (params.direction,)
        )
        found: JsonObject = {}
        for direction in directions:
            follow = (
                source.references if direction == "references" else source.citations
            )
            found[direction] = self._keep(await follow(work, limit=params.limit))
        return {"work_id": work.work_id, **found}

    async def read_work(self, params: ReadWorkParams) -> JsonObject:
        work = self._work(params.work_id)
        if params.subquestion is not None and params.run_id is None:
            raise ValueError("reading for a subquestion needs run_id")
        notes: list[JsonValue] = []
        findings: list[Finding] = []
        if params.fetch or params.subquestion is not None:
            run_id = params.run_id or self._start_run(
                question=work.title or work.work_id, tool="read_work"
            )
            self._run(run_id)
            if params.subquestion is not None and self.model is None:
                raise ValueError(
                    "reading for a subquestion needs a model: start the server "
                    "with --model"
                )
            async with self._model_lock:
                asker = await self._asker(run_id)
                sources = self._sources(
                    "both"
                    if self.config.searxng_url or self.config.browser_cdp_url
                    else "scholarly"
                )
                deps = ReadDeps(
                    asker=asker,
                    crossref=sources.crossref,
                    providers=sources.text,
                    window_chars=asker.budget.window_chars,
                    fetched_bytes=lambda: self.client.fetched_bytes,
                )
                candidate = Candidate(work=work, sources=["store"])
                if params.subquestion is None:
                    reading = await load_page(
                        candidate,
                        subquestion=SubQuestion(
                            id="Q0", text=work.title or work.work_id
                        ),
                        queries=[],
                        deps=deps,
                    )
                else:
                    subquestion = self._subquestion(run_id, params.subquestion)
                    reading = await read_page(
                        candidate, subquestion=subquestion, queries=[], deps=deps
                    )
                    first = self._next_finding_number(run_id)
                    for number, raw in enumerate(reading.findings, start=first):
                        finding = Finding.from_raw(raw, number=number)
                        self.store.add_run_item(
                            run_id, "finding", finding.finding_id, finding
                        )
                        findings.append(finding)
            notes = [*reading.notes, *reading.failed_notes, *reading.degraded]
            if reading.excluded:
                notes.append(reading.excluded)
            work = self._work(work.work_id)
        remaining = params.max_chars
        passages: list[JsonValue] = []
        for passage in self.store.passages_for_work(work.work_id):
            if remaining <= 0:
                break
            passages.append(
                {
                    "passage_id": passage.passage_id,
                    "start": passage.start,
                    "end": passage.end,
                    "text": passage.text[:remaining],
                }
            )
            remaining -= len(passage.text)
        return {
            "work": _jsonable(work),
            "abstract": work.abstract[: params.max_chars]
            if work.redistributable
            else "",
            "passages": passages,
            "findings": _jsonable(list(findings)),
            "notes": notes,
        }

    async def record_finding(self, params: RecordFindingParams) -> JsonObject:
        self._run(params.run_id)
        work = self._work(params.work_id)
        passage, quote = self._passage_with(work, params.quote)
        finding = Finding(
            finding_id=f"F{self._next_finding_number(params.run_id)}",
            subquestion_id=params.subquestion_id,
            claim=params.claim,
            quote=quote,
            passage_id=passage.passage_id,
            work_id=work.work_id,
        )
        self.store.add_run_item(params.run_id, "finding", finding.finding_id, finding)
        return {"finding": _jsonable(finding), "passage": _jsonable(passage)}

    def _passage_with(self, work: Work, quote: str) -> tuple[Passage, str]:
        """The stored passage containing *quote* and the passage's own words for
        it (``read.locate_quote``: exact, then case, whitespace, quote marks and
        dashes folded). The abstract becomes a passage when it holds the quote
        and its licence lets it be used."""
        for passage in self.store.passages_for_work(work.work_id):
            span = locate_quote(quote, passage.text)
            if span is not None:
                return passage, passage.text[span[0] : span[1]]
        if work.abstract and work.redistributable:
            span = locate_quote(quote, work.abstract)
            if span is not None:
                passage = Passage.make(
                    work_id=work.work_id, start=0, text=work.abstract
                )
                self.store.add_passages([passage])
                return passage, passage.text[span[0] : span[1]]
        raise ValueError(
            f"the quote does not occur in any passage of {work.work_id}; read the "
            "work (read_work with fetch) and quote it verbatim"
        )

    async def get_run(self, params: GetRunParams) -> JsonObject:
        if params.run_id is None:
            runs: list[JsonValue] = []
            for r in self.store.list_runs():
                row: JsonObject = {
                    "run_id": r.run_id,
                    "question": r.question,
                    "mode": r.mode,
                    "started_at": r.started_at,
                    "finished_at": r.finished_at,
                    "degraded": list(r.degraded),
                }
                runs.append(row)
            return {"runs": runs}
        return self.run_document(params.run_id, kind=params.kind)

    def run_document(self, run_id: str, *, kind: str | None = None) -> JsonObject:
        record = self.store.get_run(run_id)
        if record is None:
            raise ValueError(f"no run {run_id!r} in the store")
        items = [i for i in self.store.run_items(run_id) if kind in (None, i.kind)]
        return {
            "run": _jsonable(record),
            "items": [
                {"kind": i.kind, "ref_id": i.ref_id, "data": i.data} for i in items
            ],
        }

    async def research(self, params: ResearchParams) -> JsonObject:
        if self.model is None:
            raise ValueError("research needs a model: start the server with --model")
        async with self._model_lock:
            result = await research(
                params.question,
                mode=params.mode,
                model=self.model,
                store=self.store,
                budget=params.budget,
                sources=self.config,
                source_client=self.client,
            )
        data = _jsonable(result)
        assert isinstance(data, dict)
        return data


Handler = Callable[[ResearchTools, Any], Awaitable[JsonObject]]

# tool name -> (params model, handler, description)
_TOOLS: dict[str, tuple[type[_Params], Handler, str]] = {
    "search_works": (
        SearchWorksParams,
        lambda t, p: t.search_works(p),
        "Search scholarly sources (OpenAlex by default) for works. Duplicates are "
        "merged and every work is kept in the store; returns work ids.",
    ),
    "search_web": (
        SearchWebParams,
        lambda t, p: t.search_web(p),
        "Search the web (Chrome over CDP or SearxNG). Results are kept in the store as works.",
    ),
    "expand_citations": (
        ExpandCitationsParams,
        lambda t, p: t.expand_citations(p),
        "One hop along the citation graph from a stored work: the works it cites "
        "and the works that cite it. New works are kept in the store.",
    ),
    "read_work": (
        ReadWorkParams,
        lambda t, p: t.read_work(p),
        "A stored work with its abstract and passages. With fetch, checks for a "
        "retraction and stores the most relevant passages of its text (full text "
        "when open, else the abstract or snippet). With subquestion and run_id, "
        "also extracts findings whose quotes are checked against the passages "
        "(a quote not found is kept, marked unverified) and records them in the run.",
    ),
    "record_finding": (
        RecordFindingParams,
        lambda t, p: t.record_finding(p),
        "Store a claim with the verbatim quote that supports it. Refused unless "
        "the quote occurs in one of the work's passages (or its abstract, when "
        "its licence allows).",
    ),
    "get_run": (
        GetRunParams,
        lambda t, p: t.get_run(p),
        "Read a run's rows (sub-questions, findings, log) from the store; without a run id, list the runs.",
    ),
    "research": (
        ResearchParams,
        lambda t, p: t.research(p),
        "Run the whole research engine on a question and return the result: "
        "findings with quotes, the cited report, its "
        "verification flags and the run log (`log`: what happened at each step, "
        "also kept in the store as the run item log/run). Needs a model.",
    ),
}


def list_tools() -> list[types.Tool]:
    return [
        types.Tool(name=name, description=text, inputSchema=model.model_json_schema())
        for name, (model, _, text) in _TOOLS.items()
    ]


def _structured(data: JsonObject, *, is_error: bool = False) -> types.CallToolResult:
    return types.CallToolResult(
        content=[types.TextContent(type="text", text=json.dumps(data, indent=2))],
        structuredContent=data,
        isError=is_error,
    )


def build_server(tools: ResearchTools) -> Server:
    """The MCP server over *tools*, with the runs of its store as resources."""
    store = tools.store
    server: Server = Server(SERVER_NAME)

    @server.list_tools()
    async def _list_tools() -> list[types.Tool]:
        return list_tools()

    @server.call_tool(validate_input=False)
    async def _call_tool(
        name: str, arguments: dict[str, object]
    ) -> types.CallToolResult:
        # An exception from a tool reaches the caller as an error result (the
        # SDK reports it); nothing here swallows it.
        if name not in _TOOLS:
            raise ValueError(f"unknown tool {name!r}")
        params_model, handler, _ = _TOOLS[name]
        try:
            params = params_model.model_validate(arguments)
        except ValidationError as error:
            raise ValueError(f"{name}: invalid arguments: {error}") from error
        try:
            return _structured(await handler(tools, params))
        except (StoreError, SourceError) as error:
            raise ValueError(str(error)) from error

    @server.list_resources()
    async def _list_resources() -> list[types.Resource]:
        return [
            types.Resource(
                uri=AnyUrl(f"{RUN_URI_PREFIX}{run.run_id}"),
                name=run.run_id,
                description=run.question[:200],
                mimeType="application/json",
            )
            for run in store.list_runs()
        ]

    @server.read_resource()
    async def _read_resource(uri: AnyUrl) -> list[ReadResourceContents]:
        text = str(uri)
        if not text.startswith(RUN_URI_PREFIX):
            raise ValueError(f"unknown resource {uri}")
        document = tools.run_document(text[len(RUN_URI_PREFIX) :])
        return [
            ReadResourceContents(
                content=json.dumps(document, indent=2), mime_type="application/json"
            )
        ]

    return server


async def serve_stdio(tools: ResearchTools) -> None:
    server = build_server(tools)
    try:
        async with stdio_server() as (read, write):
            await server.run(read, write, server.create_initialization_options())
    finally:
        await tools.aclose()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="andamentum-research mcp",
        description="Serve the research engine and its store over MCP (stdio); "
        "example: claude mcp add research -- andamentum-research mcp --store research.db",
    )
    parser.add_argument(
        "--store",
        default=None,
        metavar="PATH",
        help="the research store (a SQLite file; blobs go beside it). Without it "
        "the store lives in memory and nothing persists",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="LLM for reading and the research tool",
    )
    parser.add_argument(
        "--searxng-url", default=None, help="SearXNG instance for search_web"
    )
    parser.add_argument(
        "--browser-cdp-url",
        default=None,
        metavar="URL",
        help="search the web (and read pages the plain fetch cannot) through this "
        "Chrome instead of SearxNG, e.g. http://127.0.0.1:9222",
    )
    return parser


def main(argv: list[str]) -> None:
    """Run the server on stdio; *argv* is what follows ``mcp``."""
    import os

    from dotenv import load_dotenv

    load_dotenv()  # secrets only (API keys, contact mail); configuration is flags
    args = build_parser().parse_args(argv)
    store = ResearchStore(Path(args.store)) if args.store else ResearchStore.in_memory()
    sources = SourceConfig(
        mailto=os.environ.get("OPENALEX_EMAIL") or None,
        openalex_api_key=os.environ.get("OPENALEX_API_KEY") or None,
        semantic_scholar_api_key=os.environ.get("SEMANTIC_SCHOLAR_API_KEY") or None,
        searxng_url=args.searxng_url,
        browser_cdp_url=args.browser_cdp_url,
    )
    tools = ResearchTools(store, model=args.model, sources=sources)
    try:
        anyio.run(serve_stdio, tools)
    finally:
        store.close()


__all__ = [
    "RUN_URI_PREFIX",
    "ResearchTools",
    "build_server",
    "list_tools",
    "main",
    "serve_stdio",
]
