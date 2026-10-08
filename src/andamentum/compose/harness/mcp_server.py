"""The MCP server: ``andamentum-word mcp`` lets Claude Code or Claude for Word drive the harness.

A stdio server over the ``mcp`` SDK. Every tool writes to a COPY of the
document and never to the file it was given.

Tools:

- ``open_document`` and ``show_section`` read a ``.docx`` (sections with their
  paragraph ids; one section in full with its comments).
- ``apply_edits`` applies a ``Transaction`` through the safe writer, to ``out``;
  it refuses citation operations (those go through ``insert_citation``, which
  verifies the work first).
- ``check``, ``draft``, ``refresh``, ``respond``, ``find_citation`` and
  ``insert_citation`` are the verbs of the same names (``find-citation`` and
  ``insert-citation`` for the last two). Their input schema is the verb's
  params model, and the verb is looked up in the registry at call time, so a
  verb registered after the server started works at once and one that is not
  registered answers "not available" instead of failing the server.

A verb tool runs the verb's handler as the CLI would, with its output captured.
Exit code 2 (usage) is a tool error and so is a crash (the traceback is in the
error text); the other codes (1 errors found or items left for the authors) are results the
caller reads. The input ``.docx`` files a
call names are hashed before and after, and a call that changed one fails
loudly; an ``out`` that is also an input is refused up front.

Resource ``compose://report/last``: the last verb run (tool, arguments, exit
code, captured output) and the text of the markdown report it printed, if any.
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import re
import threading
import time
import traceback
from pathlib import Path

import anyio
import anyio.to_thread
from mcp import types
from mcp.server.lowlevel import Server
from mcp.server.lowlevel.helper_types import ReadResourceContents
from mcp.server.stdio import stdio_server
from pydantic import AnyUrl, BaseModel, ConfigDict, Field, ValidationError

from .docx.reader import DocxReadError
from .docx.writer import DocxAdapter
from .kernel import (
    DocumentView,
    Paragraph,
    ProjectStore,
    Transaction,
    Verb,
    get_verb,
)
from .kernel.verbs import load_builtin_verbs
from .project import project_dir, write_copy

SERVER_NAME = "andamentum-word"
REPORT_URI = "compose://report/last"
EXIT_USAGE = 2

# tool name -> verb name; the verbs that exist as tools of their own
VERB_TOOLS: dict[str, str] = {
    "check": "check",
    "draft": "draft",
    "refresh": "refresh",
    "respond": "respond",
    "find_citation": "find-citation",
    "insert_citation": "insert-citation",
}

# Inputs a verb tool may be given are only trusted to stay unmodified when they
# are .docx files; ``out`` is the one field that is allowed to be overwritten.
_OUTPUT_FIELDS = frozenset({"out"})


class _Params(BaseModel):
    model_config = ConfigDict(extra="forbid")


class OpenDocumentParams(_Params):
    path: Path = Field(description="the .docx to read")


class ShowSectionParams(_Params):
    path: Path = Field(description="the .docx to read")
    heading: str = Field(
        description="the section's heading text (case-insensitive; a unique "
        "substring is accepted)"
    )


class ApplyEditsParams(_Params):
    path: Path = Field(description="the .docx the paragraph ids were read from")
    out: Path = Field(description="where to write the edited COPY (never the input)")
    transaction: Transaction = Field(
        description="the operations (tracked edits, comments, replies, binds)"
    )
    project: Path | None = Field(
        default=None, description="folder that keeps the ledger and logs (default: state/ beside the output)"
    )


# ------------------------------------------------------------------- reading


def _open(path: Path) -> DocumentView:
    if not path.is_file() or path.suffix.lower() != ".docx":
        raise ValueError(f"{path} is not a .docx file")
    try:
        return DocxAdapter().open(path)
    except DocxReadError as exc:
        raise ValueError(str(exc)) from exc


def _sections(view: DocumentView) -> list[tuple[Paragraph | None, list[Paragraph]]]:
    """Heading paragraph (None for the text before the first heading) and body.

    The same split as ``DocumentView.sections``, which reports heading text and
    body ids only; the heading's own id and level are needed here.
    """
    found: list[tuple[Paragraph | None, list[Paragraph]]] = []
    for paragraph in view.paragraphs:
        if paragraph.heading_level is not None:
            found.append((paragraph, []))
        else:
            if not found:
                found.append((None, []))
            found[-1][1].append(paragraph)
    return found


def _paragraph_json(paragraph: Paragraph) -> dict[str, object]:
    return {
        "id": paragraph.id,
        "style": paragraph.style,
        "heading_level": paragraph.heading_level,
        "text": paragraph.text,
    }


def open_document(params: OpenDocumentParams) -> dict[str, object]:
    view = _open(params.path)
    return {
        "path": str(params.path),
        "project_id": view.project_id,
        "paragraphs": len(view.paragraphs),
        "comments": len(view.comments),
        "anchors": len(view.anchors),
        "citation_fields": sum(1 for f in view.fields if f.kind == "citation"),
        "sections": [
            {
                "heading": "" if heading is None else heading.text,
                "heading_id": None if heading is None else heading.id,
                "level": None if heading is None else heading.heading_level,
                "paragraph_ids": [p.id for p in body],
            }
            for heading, body in _sections(view)
        ],
    }


def show_section(params: ShowSectionParams) -> dict[str, object]:
    view = _open(params.path)
    sections = [(h, b) for h, b in _sections(view) if h is not None]
    wanted = params.heading.strip().casefold()
    matches = [(h, b) for h, b in sections if h.text.strip().casefold() == wanted]
    if not matches:
        matches = [
            (h, b) for h, b in sections if wanted and wanted in h.text.casefold()
        ]
    if not matches:
        headings = ", ".join(repr(h.text) for h, _ in sections) or "none"
        raise ValueError(f"no section {params.heading!r}; headings: {headings}")
    if len(matches) > 1:
        headings = ", ".join(repr(h.text) for h, _ in matches)
        raise ValueError(f"heading {params.heading!r} is ambiguous: {headings}")
    heading, body = matches[0]
    ids = {heading.id, *(p.id for p in body)}
    return {
        "path": str(params.path),
        "heading": _paragraph_json(heading),
        "paragraphs": [_paragraph_json(p) for p in body],
        "comments": [c.model_dump() for c in view.comments if c.paragraph_id in ids],
        "anchors": [a.model_dump() for a in view.anchors if a.paragraph_id in ids],
        "fields": [
            f.model_dump(exclude={"data"}) for f in view.fields if f.paragraph_id in ids
        ],
    }


# ------------------------------------------------------------------- writing


def apply_edits(params: ApplyEditsParams) -> dict[str, object]:
    if params.out.resolve() == params.path.resolve():
        raise ValueError("out must not be the input file; edits are written to a copy")
    citation_ops = [
        op.op
        for op in params.transaction.ops
        if op.op in ("insert_citation", "update_bibliography")
    ]
    if citation_ops:
        raise ValueError(
            f"apply_edits does not write citations ({', '.join(sorted(set(citation_ops)))}): "
            "a work enters a document only after it is verified; use the "
            "insert_citation tool"
        )
    adapter = DocxAdapter()
    view = _open(params.path)
    store = ProjectStore(project_dir(params.out, params.project))
    project_id = store.init(params.path, project_id=view.project_id).id
    result = write_copy(
        adapter, view, params.transaction, out=params.out, project_id=project_id
    )
    store.append_event(
        {
            "type": "transaction",
            "source": params.transaction.source,
            "author": params.transaction.author,
            "ops": len(params.transaction.ops),
            "applied": result.applied,
            "refused": result.refused,
            "output": str(params.out),
        }
    )
    return json.loads(result.model_dump_json())


# --------------------------------------------------------------------- verbs


class VerbRun(BaseModel):
    """What one verb call produced (also the body of the last-report resource)."""

    model_config = ConfigDict(extra="forbid")

    tool: str
    verb: str
    arguments: dict[str, object]
    exit_code: int
    stdout: str
    stderr: str
    seconds: float
    report_markdown: str | None = None


def _docx_inputs(params: BaseModel) -> list[Path]:
    """The ``.docx`` files a verb's params name as inputs."""
    found: list[Path] = []
    for name in type(params).model_fields:
        if name in _OUTPUT_FIELDS:
            continue
        value = getattr(params, name)
        for item in value if isinstance(value, list) else [value]:
            if isinstance(item, Path) and item.suffix.lower() == ".docx":
                found.append(item)
    return found


def _digests(paths: list[Path]) -> dict[Path, str]:
    return {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths if p.is_file()}


def _printed_report(stdout: str) -> str | None:
    """The markdown report the verb printed a path to, when that file exists."""
    for candidate in re.findall(r"\S+\.md\b", stdout):
        path = Path(candidate)
        if path.is_file():
            return path.read_text(encoding="utf-8")
    return None


def _run_handler(verb: Verb, params: BaseModel) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        try:
            code = verb.handler(params)
        except Exception as exc:  # the caller gets the crash with its traceback
            raise RuntimeError(
                f"the {verb.name!r} verb crashed: {exc!r}\n{traceback.format_exc()}"
            ) from exc
    return code, out.getvalue(), err.getvalue()


# The handlers print to the process's stdout, which is swapped while one runs.
_VERB_LOCK = threading.Lock()


def run_verb_tool(tool: str, arguments: dict[str, object]) -> VerbRun:
    """Run the verb behind *tool* with *arguments*; raise ValueError when it cannot run."""
    load_builtin_verbs()
    name = VERB_TOOLS[tool]
    try:
        verb = get_verb(name)
    except KeyError:
        raise ValueError(
            f"the {name!r} verb is not available in this build (not registered)"
        ) from None
    unknown = sorted(set(arguments) - set(verb.params.model_fields))
    if unknown:
        raise ValueError(f"{tool}: unknown argument(s): {', '.join(unknown)}")
    try:
        params = verb.params.model_validate(arguments)
    except ValidationError as exc:
        raise ValueError(f"{tool}: invalid arguments: {exc}") from exc

    inputs = _docx_inputs(params)
    out = getattr(params, "out", None)
    if isinstance(out, Path) and any(out.resolve() == p.resolve() for p in inputs):
        raise ValueError("out must not be an input file; the tools write copies")
    before = _digests(inputs)

    started = time.monotonic()
    with _VERB_LOCK:
        code, stdout, stderr = _run_handler(verb, params)
    changed = [
        str(p) for p, digest in _digests(inputs).items() if before.get(p) != digest
    ]
    if changed:
        raise RuntimeError(f"{name} modified its input: {', '.join(changed)}")
    return VerbRun(
        tool=tool,
        verb=name,
        arguments={k: v for k, v in json.loads(params.model_dump_json()).items()},
        exit_code=code,
        stdout=stdout,
        stderr=stderr,
        seconds=round(time.monotonic() - started, 2),
        report_markdown=_printed_report(stdout),
    )


# -------------------------------------------------------------------- server

_NATIVE: dict[str, tuple[type[_Params], str]] = {
    "open_document": (
        OpenDocumentParams,
        "Read a .docx: sections with their paragraph ids, counts of comments, "
        "anchors and citation fields. The ids are what apply_edits addresses.",
    ),
    "show_section": (
        ShowSectionParams,
        "Show one section of a .docx in full: paragraphs with ids and styles, "
        "its comments, anchors and fields.",
    ),
    "apply_edits": (
        ApplyEditsParams,
        "Apply a transaction (tracked replace/insert/delete, comments, replies) "
        "to a COPY of the document at out. Returns what was applied and refused.",
    ),
}

_ANY_OBJECT: dict[str, object] = {"type": "object", "additionalProperties": True}


def _verb_tool(tool: str) -> types.Tool:
    name = VERB_TOOLS[tool]
    try:
        verb = get_verb(name)
    except KeyError:
        return types.Tool(
            name=tool,
            description=f"The {name!r} verb. Not registered in this build yet; "
            "a call answers once it is.",
            inputSchema=_ANY_OBJECT,
        )
    return types.Tool(
        name=tool,
        description=f"{verb.summary}. Writes copies; the input is never modified.",
        inputSchema=verb.params.model_json_schema(),
    )


def list_tools() -> list[types.Tool]:
    load_builtin_verbs()
    tools = [
        types.Tool(name=name, description=text, inputSchema=model.model_json_schema())
        for name, (model, text) in _NATIVE.items()
    ]
    tools += [_verb_tool(tool) for tool in VERB_TOOLS]
    return tools


def _error(message: str) -> types.CallToolResult:
    return types.CallToolResult(
        content=[types.TextContent(type="text", text=message)], isError=True
    )


def _structured(
    data: dict[str, object], *, is_error: bool = False
) -> types.CallToolResult:
    return types.CallToolResult(
        content=[types.TextContent(type="text", text=json.dumps(data, indent=2))],
        structuredContent=data,
        isError=is_error,
    )


def _call_native(name: str, arguments: dict[str, object]) -> dict[str, object]:
    model, _ = _NATIVE[name]
    try:
        params = model.model_validate(arguments)
    except ValidationError as exc:
        raise ValueError(f"{name}: invalid arguments: {exc}") from exc
    if isinstance(params, OpenDocumentParams):
        return open_document(params)
    if isinstance(params, ShowSectionParams):
        return show_section(params)
    assert isinstance(params, ApplyEditsParams)
    return apply_edits(params)


def build_server() -> Server:
    """The MCP server with its tools and the last-report resource."""
    server: Server = Server(SERVER_NAME)
    last: dict[str, VerbRun] = {}

    @server.list_tools()
    async def _list_tools() -> list[types.Tool]:
        return list_tools()

    @server.call_tool(validate_input=False)
    async def _call_tool(
        name: str, arguments: dict[str, object]
    ) -> types.CallToolResult:
        # An exception from a tool reaches the caller as an error result (the
        # SDK reports it); nothing here swallows it.
        if name in _NATIVE:
            return _structured(
                await anyio.to_thread.run_sync(_call_native, name, arguments)
            )
        if name in VERB_TOOLS:
            run = await anyio.to_thread.run_sync(run_verb_tool, name, arguments)
            last["run"] = run
            data = json.loads(run.model_dump_json(exclude={"report_markdown"}))
            return _structured(data, is_error=run.exit_code == EXIT_USAGE)
        return _error(f"unknown tool {name!r}")

    @server.list_resources()
    async def _list_resources() -> list[types.Resource]:
        return [
            types.Resource(
                uri=AnyUrl(REPORT_URI),
                name="last-report",
                description="The last verb run: output and its markdown report.",
                mimeType="text/markdown",
            )
        ]

    @server.read_resource()
    async def _read_resource(uri: AnyUrl) -> list[ReadResourceContents]:
        if str(uri) != REPORT_URI:
            raise ValueError(f"unknown resource {uri}")
        run = last.get("run")
        if run is None:
            text = "No report yet: no verb has run in this session.\n"
        else:
            text = run.report_markdown or (
                f"# {run.verb} (exit code {run.exit_code})\n\n"
                f"```\n{run.stdout}{run.stderr}```\n"
            )
        return [ReadResourceContents(content=text, mime_type="text/markdown")]

    return server


async def serve_stdio() -> None:
    server = build_server()
    async with stdio_server() as (read, write):
        await server.run(read, write, server.create_initialization_options())


# ---------------------------------------------------------------- the verb


class McpParams(BaseModel):
    """The ``mcp`` verb takes no arguments."""


def _handle(params: McpParams) -> int:
    anyio.run(serve_stdio)
    return 0


MCP_VERB = Verb(
    name="mcp",
    summary="serve the harness to Claude Code or Claude for Word over MCP (stdio); "
    "example: claude mcp add word -- andamentum-word mcp",
    params=McpParams,
    handler=_handle,
)

VERBS: tuple[Verb, ...] = (MCP_VERB,)

__all__ = [
    "REPORT_URI",
    "VERBS",
    "VERB_TOOLS",
    "VerbRun",
    "build_server",
    "run_verb_tool",
    "serve_stdio",
]
