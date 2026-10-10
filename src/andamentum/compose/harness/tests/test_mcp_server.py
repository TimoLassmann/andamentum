"""The MCP server through the SDK's in-memory client: tools, safety and the report resource."""

from __future__ import annotations

import functools
import hashlib
import importlib
import inspect
import sys
import shutil
from collections.abc import Iterator
from pathlib import Path

import pytest
from mcp import ClientSession, types
from mcp.shared.memory import create_connected_server_and_client_session
from pydantic import BaseModel, Field

from andamentum.word import cli
from andamentum.compose.harness import mcp_server
from andamentum.compose.harness.bench.planted import generate
from andamentum.compose.harness.docx.reader import open_docx
from andamentum.compose.harness.kernel.verbs import load_builtin_verbs
from andamentum.compose.harness.kernel import Verb, all_checks, get_verb, verbs

ALL_TOOLS = {
    "open_document",
    "show_section",
    "apply_edits",
    "check",
    "draft",
    "refresh",
    "respond",
    "find_citation",
    "insert_citation",
}


@pytest.fixture(scope="module")
def planted(tmp_path_factory: pytest.TempPathFactory) -> Path:
    folder = tmp_path_factory.mktemp("planted")
    generate(folder, seed=0)
    return folder


@pytest.fixture
def manuscript(planted: Path, tmp_path: Path) -> Path:
    path = tmp_path / "paper.docx"
    shutil.copy(planted / "d01_mixed_a.docx", path)
    return path


@pytest.fixture(autouse=True)
def registered_checks() -> None:
    """``test_kernel_types`` leaves the check registry empty; put the checks back."""
    load_builtin_verbs()  # importing the check verb registers the checks
    if all_checks():
        return
    for name in sorted(sys.modules):
        if name.startswith("andamentum.compose.harness.checks."):
            importlib.reload(sys.modules[name])
    assert all_checks()


STUBBED = ("find_citation", "insert_citation", "respond", "refresh")


@pytest.fixture(autouse=True)
def stub_names(monkeypatch: pytest.MonkeyPatch) -> None:
    """Point the tools other work registers verbs for at names nothing registers.

    The real verbs may exist by now; these tests need a verb that is absent
    until a test registers its stub.
    """
    for tool in STUBBED:
        monkeypatch.setitem(mcp_server.VERB_TOOLS, tool, f"stub-{tool}")


@pytest.fixture(autouse=True)
def restore_registry() -> Iterator[None]:
    """Tests register stub verbs; leave the registry as it was."""
    saved = dict(verbs._VERBS)
    yield
    verbs._VERBS.clear()
    verbs._VERBS.update(saved)


def with_client(test):
    """Run an async test with a connected in-memory client as its first argument.

    The session is opened inside the test's own task: an async-generator fixture
    would enter and leave the SDK's cancel scopes in different tasks.
    """

    @functools.wraps(test)
    async def wrapper(*args, **kwargs):
        async with create_connected_server_and_client_session(
            mcp_server.build_server()
        ) as session:
            await test(session, *args, **kwargs)

    parameters = list(inspect.signature(test).parameters.values())[1:]
    wrapper.__signature__ = inspect.Signature(parameters)  # type: ignore[attr-defined]
    return wrapper


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def data(result: types.CallToolResult) -> dict[str, object]:
    assert result.structuredContent is not None, result.content
    return result.structuredContent


def message(result: types.CallToolResult) -> str:
    (block,) = result.content
    assert isinstance(block, types.TextContent)
    return block.text


class StubParams(BaseModel):
    query: str = Field(
        json_schema_extra={"positional": True}, description="what to find"
    )
    file: Path | None = None
    out: Path | None = None


def stub_verb(tool: str, handler=None) -> Verb:
    def default(params: StubParams) -> int:
        print(f"found {params.query}")
        return 0

    verb = Verb(
        name=mcp_server.VERB_TOOLS[tool],
        summary="a stub",
        params=StubParams,
        handler=handler or default,
    )
    verbs.register_verb(verb)
    return verb


# ------------------------------------------------------------------ listing


@with_client
async def test_lists_every_tool_with_schemas_from_the_verbs(client: ClientSession):
    tools = {t.name: t for t in (await client.list_tools()).tools}
    assert set(tools) == ALL_TOOLS
    assert "file" in tools["check"].inputSchema["properties"]
    assert tools["check"].description and "commented copy" in tools["check"].description
    assert set(tools["apply_edits"].inputSchema["required"]) == {
        "path",
        "out",
        "transaction",
    }
    assert get_verb("mcp").name == "mcp"


@with_client
async def test_a_verb_registered_later_changes_the_listing(client: ClientSession):
    before = {t.name: t for t in (await client.list_tools()).tools}
    assert before["find_citation"].inputSchema == mcp_server._ANY_OBJECT
    stub_verb("find_citation")
    after = {t.name: t for t in (await client.list_tools()).tools}
    assert "query" in after["find_citation"].inputSchema["properties"]


# ------------------------------------------------------------------ reading


@with_client
async def test_open_document_and_show_section(client: ClientSession, manuscript: Path):
    view = open_docx(manuscript)
    opened = data(await client.call_tool("open_document", {"path": str(manuscript)}))
    assert opened["paragraphs"] == len(view.paragraphs)
    sections = opened["sections"]
    assert isinstance(sections, list) and sections
    headings = [s["heading"] for s in sections if s["heading"]]
    assert headings
    listed = [pid for s in sections for pid in s["paragraph_ids"]]
    assert listed and set(listed) <= {p.id for p in view.paragraphs}

    shown = data(
        await client.call_tool(
            "show_section", {"path": str(manuscript), "heading": headings[0].upper()}
        )
    )
    assert shown["heading"]["text"] == headings[0]  # type: ignore[index]
    texts = {p.id: p.text for p in view.paragraphs}
    for paragraph in shown["paragraphs"]:  # type: ignore[attr-defined]
        assert paragraph["text"] == texts[paragraph["id"]]


@with_client
async def test_show_section_unknown_heading_lists_the_headings(
    client: ClientSession, manuscript: Path
):
    result = await client.call_tool(
        "show_section", {"path": str(manuscript), "heading": "no such heading"}
    )
    assert result.isError
    assert "headings:" in message(result)


@with_client
async def test_open_document_rejects_a_non_docx(client: ClientSession, tmp_path: Path):
    other = tmp_path / "notes.txt"
    other.write_text("x")
    result = await client.call_tool("open_document", {"path": str(other)})
    assert result.isError and "not a .docx" in message(result)


# ------------------------------------------------------------------ writing


def _first_word_replacement(manuscript: Path, new: str) -> dict[str, object]:
    view = open_docx(manuscript)
    paragraph = next(p for p in view.paragraphs if p.heading_level is None and p.text)
    word = paragraph.text.split(" ")[0]
    return {
        "ops": [
            {
                "op": "replace",
                "paragraph_id": paragraph.id,
                "start": 0,
                "end": len(word),
                "text": new,
                "expect": word,
            }
        ],
        "author": "tester",
        "reason": "test",
        "source": "test",
    }


@with_client
async def test_apply_edits_writes_a_copy_and_leaves_the_input(
    client: ClientSession, manuscript: Path, tmp_path: Path
):
    before = sha(manuscript)
    out = tmp_path / "edited.docx"
    transaction = _first_word_replacement(manuscript, "Zzyzx")
    result = data(
        await client.call_tool(
            "apply_edits",
            {"path": str(manuscript), "out": str(out), "transaction": transaction},
        )
    )
    assert result["applied"] == 1 and result["refused"] == []
    assert sha(manuscript) == before
    # the copy carries the project id, so the next verb finds the same project
    assert open_docx(out).project_id is not None
    ops = transaction["ops"]
    assert isinstance(ops, list)
    pid = ops[0]["paragraph_id"]  # type: ignore[index]
    assert (
        open_docx(out)
        .paragraphs[int(str(pid).split("-")[0][1:])]
        .text.startswith("Zzyzx")
    )


@with_client
async def test_apply_edits_refuses_to_overwrite_the_input(
    client: ClientSession, manuscript: Path
):
    before = sha(manuscript)
    result = await client.call_tool(
        "apply_edits",
        {
            "path": str(manuscript),
            "out": str(manuscript),
            "transaction": _first_word_replacement(manuscript, "Zzyzx"),
        },
    )
    assert result.isError and "must not be the input" in message(result)
    assert sha(manuscript) == before


@with_client
async def test_apply_edits_refuses_an_existing_copy(
    client: ClientSession, manuscript: Path, tmp_path: Path
):
    out = tmp_path / "edited.docx"
    out.write_bytes(b"edited in Word")
    result = await client.call_tool(
        "apply_edits",
        {
            "path": str(manuscript),
            "out": str(out),
            "transaction": _first_word_replacement(manuscript, "Zzyzx"),
        },
    )
    assert result.isError and "never overwrites" in message(result)
    assert out.read_bytes() == b"edited in Word"


@with_client
async def test_apply_edits_reports_invalid_operations(
    client: ClientSession, manuscript: Path, tmp_path: Path
):
    result = await client.call_tool(
        "apply_edits",
        {
            "path": str(manuscript),
            "out": str(tmp_path / "x.docx"),
            "transaction": {
                "ops": [{"op": "nonsense"}],
                "author": "a",
                "reason": "r",
                "source": "s",
            },
        },
    )
    assert result.isError and "invalid arguments" in message(result)


# -------------------------------------------------------------------- verbs


@with_client
async def test_check_tool_runs_the_verb_and_the_report_is_a_resource(
    client: ClientSession, manuscript: Path
):
    empty = await client.read_resource(mcp_server.REPORT_URI)  # type: ignore[arg-type]
    assert "No report yet" in empty.contents[0].text  # type: ignore[union-attr]

    before = sha(manuscript)
    run = data(
        await client.call_tool(
            "check",
            {
                "file": str(manuscript),
                "offline": True,
                "out": str(manuscript.with_name("paper.checked.docx")),
            },
        )
    )
    assert (
        run["verb"] == "check" and run["exit_code"] == 1
    )  # errors found, not a failure
    assert "commented copy" in str(run["stdout"])
    assert sha(manuscript) == before
    assert manuscript.with_name("paper.checked.docx").is_file()

    report = await client.read_resource(mcp_server.REPORT_URI)  # type: ignore[arg-type]
    text = report.contents[0].text  # type: ignore[union-attr]
    assert text.startswith("# Check report")


@with_client
async def test_check_tool_usage_error_is_a_tool_error(
    client: ClientSession, tmp_path: Path
):
    result = await client.call_tool(
        "check",
        {
            "file": str(tmp_path / "missing.docx"),
            "offline": True,
            "out": str(tmp_path / "checked.docx"),
        },
    )
    assert result.isError and data(result)["exit_code"] == 2


@with_client
async def test_check_tool_rejects_unknown_arguments(
    client: ClientSession, manuscript: Path
):
    result = await client.call_tool("check", {"file": str(manuscript), "bogus": 1})
    assert result.isError and "unknown argument" in message(result)
    assert not manuscript.with_name("paper.checked.docx").exists()


@with_client
async def test_verb_missing_at_call_time_then_found_once_registered(
    client: ClientSession,
):
    missing = await client.call_tool("find_citation", {"query": "sodium"})
    assert missing.isError and "not available" in message(missing)

    stub_verb("find_citation")  # registered while the server is running
    found = await client.call_tool("find_citation", {"query": "sodium"})
    assert not found.isError
    assert data(found)["stdout"] == "found sodium\n"
    assert data(found)["exit_code"] == 0


@with_client
async def test_verb_arguments_are_validated(client: ClientSession):
    stub_verb("insert_citation")
    result = await client.call_tool("insert_citation", {})
    assert result.isError and "invalid arguments" in message(result)


@with_client
async def test_out_equal_to_an_input_is_refused(
    client: ClientSession, manuscript: Path
):
    stub_verb("insert_citation")
    result = await client.call_tool(
        "insert_citation",
        {"query": "q", "file": str(manuscript), "out": str(manuscript)},
    )
    assert result.isError and "must not be an input" in message(result)


@with_client
async def test_a_verb_that_modifies_its_input_fails_loudly(
    client: ClientSession, manuscript: Path
):
    def vandal(params: StubParams) -> int:
        assert params.file is not None
        params.file.write_bytes(params.file.read_bytes() + b"x")
        return 0

    stub_verb("insert_citation", vandal)
    result = await client.call_tool(
        "insert_citation", {"query": "q", "file": str(manuscript)}
    )
    assert result.isError and "modified its input" in message(result)


@with_client
async def test_a_crashing_verb_is_a_tool_error_with_the_traceback(
    client: ClientSession,
):
    def boom(params: StubParams) -> int:
        raise RuntimeError("kaput")

    stub_verb("respond", boom)
    result = await client.call_tool("respond", {"query": "q"})
    assert result.isError
    assert "kaput" in message(result) and "Traceback" in message(result)


@with_client
async def test_exit_codes_above_the_failures_are_results(client: ClientSession):
    stub_verb("refresh", lambda params: 4)
    result = await client.call_tool("refresh", {"query": "q"})
    assert not result.isError and data(result)["exit_code"] == 4


@with_client
async def test_unknown_tool(client: ClientSession):
    result = await client.call_tool("nope", {})
    assert result.isError


# ------------------------------------------------------------------- the CLI


def test_the_mcp_verb_is_a_subcommand():
    args = cli.build_parser().parse_args(["mcp"])
    assert args.command == "mcp"


async def test_the_stdio_server_answers_and_keeps_stdout_clean(manuscript: Path):
    """The real process: handler output must not corrupt the protocol stream."""
    from mcp.client.stdio import StdioServerParameters, stdio_client

    server = StdioServerParameters(
        command=sys.executable,
        args=["-m", "andamentum.word.cli", "mcp"],
    )
    async with stdio_client(server) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            names = {t.name for t in (await session.list_tools()).tools}
            assert ALL_TOOLS <= names
            result = await session.call_tool(
                "check",
                {
                    "file": str(manuscript),
                    "offline": True,
                    "out": str(manuscript.with_name("checked.docx")),
                },
            )
            assert data(result)["exit_code"] == 1


@with_client
async def test_apply_edits_refuses_citations_of_unverified_works(
    client: ClientSession, manuscript: Path, tmp_path: Path
):
    paragraph = open_docx(manuscript).paragraphs[0]
    fabricated = {"id": "x", "title": "Invented study", "DOI": "10.0000/fake"}
    result = await client.call_tool(
        "apply_edits",
        {
            "path": str(manuscript),
            "out": str(tmp_path / "x.docx"),
            "transaction": {
                "ops": [
                    {
                        "op": "insert_citation",
                        "paragraph_id": paragraph.id,
                        "at": 0,
                        "items": [{"csl": fabricated}],
                    },
                    {"op": "update_bibliography"},
                ],
                "author": "a",
                "reason": "r",
                "source": "s",
            },
        },
    )
    assert result.isError and "insert_citation tool" in message(result)
    assert not (tmp_path / "x.docx").exists()
