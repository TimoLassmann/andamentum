"""andamentum-word: the parser, the exit codes, and "writes only into --out"."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from andamentum.compose.harness.bindings import bind as bind_module
from andamentum.compose.harness.tests._bindings_helpers import (
    EFFECT,
    make_manuscript,
    make_results,
)
from andamentum.compose.harness.tests.test_bindings_bind import PICKS, StubModel
from andamentum.word import cli

WRITERS = ("check", "respond", "refresh", "bind-numbers", "draft", "insert-citation")
PRINTERS = ("find-citation", "paragraphs", "mcp")


def _subparsers(parser: argparse.ArgumentParser) -> dict[str, argparse.ArgumentParser]:
    action = next(
        a for a in parser._actions if isinstance(a, argparse._SubParsersAction)
    )
    return dict(action.choices)


def test_the_parser_has_the_documented_subcommands():
    assert set(_subparsers(cli.build_parser())) == {*WRITERS, *PRINTERS}


def test_every_flag_has_help_text():
    for name, sub in _subparsers(cli.build_parser()).items():
        for action in sub._actions:
            assert action.help, (
                f"{name}: {action.option_strings or action.dest} has no help"
            )


@pytest.mark.parametrize("name", WRITERS)
def test_a_command_that_writes_requires_the_short_and_long_out_flag(name: str):
    sub = _subparsers(cli.build_parser())[name]
    out = next(a for a in sub._actions if "--out" in a.option_strings)
    assert "-o" in out.option_strings and out.required


@pytest.mark.parametrize("name", PRINTERS)
def test_a_command_that_only_prints_has_no_out_flag(name: str):
    sub = _subparsers(cli.build_parser())[name]
    assert not any("--out" in a.option_strings for a in sub._actions)


def test_no_command_has_a_project_or_report_location_flag():
    for name, sub in _subparsers(cli.build_parser()).items():
        flags = {o for a in sub._actions for o in a.option_strings}
        assert not flags & {"--project", "--report", "--cache", "--output"}, name


def test_draft_offers_exactly_these_flags():
    """Mutation: add back a role model (--drafter, --judge), --max-rounds or
    --research-mode and this fails."""
    sub = _subparsers(cli.build_parser())["draft"]
    flags = {o for a in sub._actions for o in a.option_strings if o.startswith("--")}
    assert flags == {
        "--help", "--out", "--model", "--into", "--genre", "--source",
        "--results", "--library", "--zotero", "--template", "--offline",
        "--confirm-own-sources", "--mailto", "--research", "--searxng-url",
        "--browser", "--json",
    }  # fmt: skip


@pytest.mark.parametrize(
    ("extra", "research", "mode"),
    [
        ([], False, None),
        (["--research", "auto"], True, None),  # the genre chooses where to search
        (["--research", "both"], True, "both"),
    ],
)
def test_draft_research_is_one_flag(tmp_path, monkeypatch, extra, research, mode):
    from andamentum import word

    seen: dict[str, object] = {}

    async def fake_draft(task, **kwargs):
        seen.update(kwargs)
        raise RuntimeError("stop here")

    monkeypatch.setattr(word, "draft_document", fake_draft)
    argv = ["draft", "t", "-o", str(tmp_path / "d"), "--model", "m", *extra]
    assert cli.main(argv) == 4
    assert seen["research"] is research and seen["research_mode"] == mode
    assert "max_rounds" not in seen and "drafter" not in seen


def test_parsing_a_check_command():
    args = cli.build_parser().parse_args(
        ["check", "paper.docx", "-o", "checked", "--results", "r", "--offline"]
    )
    assert args.command == "check" and args.offline and str(args.out) == "checked"


def test_a_missing_out_is_a_usage_error():
    with pytest.raises(SystemExit) as exc:
        cli.main(["check", "paper.docx"])
    assert exc.value.code == 2


def test_a_missing_input_exits_3_and_a_bad_results_folder_exits_3(tmp_path, capsys):
    assert (
        cli.main(["check", str(tmp_path / "nope.docx"), "-o", str(tmp_path / "o")]) == 3
    )
    assert "error:" in capsys.readouterr().err
    notes = tmp_path / "notes.txt"
    notes.write_text("x")
    assert cli.main(["check", str(notes), "-o", str(tmp_path / "o")]) == 3
    paper = make_manuscript(tmp_path / "paper.docx")
    code = cli.main(
        [
            "check",
            str(paper),
            "-o",
            str(tmp_path / "o"),
            "--results",
            str(tmp_path / "no"),
        ]
    )
    assert code == 3
    assert not (tmp_path / "o").exists()


def test_a_corrupt_docx_exits_3(tmp_path, capsys):
    bad = tmp_path / "bad.docx"
    bad.write_bytes(b"not a zip")
    assert cli.main(["check", str(bad), "-o", str(tmp_path / "o")]) == 3
    assert capsys.readouterr().err.startswith("error:")


def test_a_crash_exits_4(tmp_path, monkeypatch, capsys):
    from andamentum.compose.harness import check as check_module

    async def boom(params, **_kw):
        raise RuntimeError("network down")

    monkeypatch.setattr(check_module, "run_check", boom)
    paper = make_manuscript(tmp_path / "paper.docx")
    assert cli.main(["check", str(paper), "-o", str(tmp_path / "o"), "--offline"]) == 4
    assert "error: RuntimeError: network down" in capsys.readouterr().err


def _tree(root: Path) -> set[Path]:
    return {p for p in root.rglob("*")}


def test_the_writers_create_files_only_inside_out(tmp_path, monkeypatch, capsys):
    """check, bind-numbers, refresh and respond: the whole tree outside --out is untouched."""
    monkeypatch.setattr(bind_module, "run_agent_with_usage", StubModel(PICKS))
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    paper = make_manuscript(inputs / "paper.docx")
    results = make_results(inputs / "results")
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    out = tmp_path / "elsewhere" / "out"
    outside = _tree(tmp_path) | {out, out.parent}
    before_inputs = {p: p.read_bytes() for p in inputs.rglob("*") if p.is_file()}

    def run(*argv: str) -> int:
        code = cli.main([*argv, "-o", str(out)])
        capsys.readouterr()
        return code

    assert run("check", str(paper), "--offline") in (0, 1)
    assert (
        run("bind-numbers", str(paper), "--results", str(results), "--model", "stub:m")
        == 0
    )
    bound = out / "paper.bound.docx"
    assert bound.is_file()
    assert run("refresh", str(bound), "--results", str(results)) == 0
    assert run("respond", str(paper), "--offline") == 0

    created = _tree(tmp_path) - outside
    assert created, "nothing was written"
    assert all(out in p.parents for p in created), sorted(
        str(p) for p in created if out not in p.parents
    )
    assert (out / "state" / "project.yaml").is_file()
    assert list(cwd.iterdir()) == []  # nothing in the current folder
    after_inputs = {p: p.read_bytes() for p in inputs.rglob("*") if p.is_file()}
    assert after_inputs == before_inputs  # nothing next to the input, input unchanged


#: A statistics error the offline checks find: t(20) = 2.10 cannot give p = .90.
_PLANTED = [
    ("Pilot", "Heading1"),
    (
        "The group difference was large (t(20) = 2.10, p = .90) in the pilot sample.",
        None,
    ),
]


def test_a_planted_error_exits_1_and_still_writes_the_report(tmp_path, capsys):
    """Mutation: make `check` return 0 whatever the report holds (or write nothing
    when errors are found) and this fails."""
    paper = make_manuscript(tmp_path / "paper.docx", _PLANTED)
    out = tmp_path / "o"
    assert cli.main(["check", str(paper), "-o", str(out), "--offline"]) == 1
    capsys.readouterr()
    report = json.loads((out / "paper.checked.json").read_text())
    assert report["counts"]["error"] >= 1
    assert (out / "paper.checked.md").is_file() and (
        out / "paper.checked.docx"
    ).is_file()


def test_a_clean_document_exits_0(tmp_path, capsys):
    paper = make_manuscript(
        tmp_path / "paper.docx",
        [("Pilot", "Heading1"), ("Nothing numeric here.", None)],
    )
    assert cli.main(["check", str(paper), "-o", str(tmp_path / "o"), "--offline"]) == 0
    capsys.readouterr()


def test_offline_makes_no_network_call(tmp_path, monkeypatch, capsys):
    """Mutation: let an offline check open a socket (e.g. run the citation lookup
    despite --offline) and the raising connect turns the exit code into 4."""
    import socket

    import httpx

    def refuse(*_a, **_k):
        raise AssertionError("network used under --offline")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", refuse)
    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", refuse)
    paper = make_manuscript(tmp_path / "paper.docx", _PLANTED)
    assert cli.main(["check", str(paper), "-o", str(tmp_path / "o"), "--offline"]) == 1
    capsys.readouterr()


def test_json_prints_the_result_instead_of_text(tmp_path, capsys):
    paper = make_manuscript(tmp_path / "paper.docx")
    code = cli.main(
        ["check", str(paper), "-o", str(tmp_path / "o"), "--offline", "--json"]
    )
    assert code in (0, 1)
    data = json.loads(capsys.readouterr().out)
    assert data["report"]["counts"].keys() == {"error", "warning", "info"}


async def test_the_python_functions_write_only_into_out(tmp_path):
    from andamentum import word

    paper = make_manuscript(tmp_path / "in" / "paper.docx")
    before = _tree(tmp_path)
    run = await word.check_document(paper, out=tmp_path / "out", offline=True)
    assert run.output == tmp_path / "out" / "paper.checked.docx"
    created = _tree(tmp_path) - before
    assert all(tmp_path / "out" in p.parents or p == tmp_path / "out" for p in created)
    with pytest.raises(word.WordInputError):
        await word.check_document(tmp_path / "missing.docx", out=tmp_path / "out2")
    assert issubclass(word.WordInputError, word.WordError)
    assert issubclass(word.WordUsageError, word.WordError)


def test_the_public_surface():
    import andamentum.word as word

    assert set(word.__all__) == {
        "check_document",
        "respond_to_reviews",
        "refresh_numbers",
        "bind_numbers",
        "draft_document",
        "find_citation",
        "insert_citation",
        "CheckRun",
        "RespondRun",
        "RefreshRun",
        "BindRun",
        "DraftResult",
        "CitationSearch",
        "FoundWork",
        "UnverifiedWork",
        "CitationInsertion",
        "WordError",
        "WordUsageError",
        "WordInputError",
        "list_paragraphs",
        "ParagraphInfo",
    }
    assert len(word.__all__) == len(set(word.__all__))
    for name in word.__all__:
        assert getattr(word, name) is not None
    assert EFFECT  # the synthetic helper text is shared with the harness tests


@pytest.mark.parametrize("where", ["web", "scholarly", "both"])
def test_draft_research_passes_each_place(tmp_path, monkeypatch, where):
    from andamentum import word

    seen: dict[str, object] = {}

    async def fake_draft(task, **kwargs):
        seen.update(kwargs)
        raise RuntimeError("stop here")

    monkeypatch.setattr(word, "draft_document", fake_draft)
    argv = ["draft", "t", "-o", str(tmp_path / "d"), "--model", "m", "--research", where]
    assert cli.main(argv) == 4
    assert seen["research"] is True and seen["research_mode"] == where


def test_draft_offline_refuses_a_web_source(tmp_path, monkeypatch, capsys):
    from andamentum import word

    async def never(task, **kwargs):
        raise AssertionError("the draft must not start")

    monkeypatch.setattr(word, "draft_document", never)
    argv = ["draft", "t", "-o", str(tmp_path / "d"), "--model", "m", "--offline",
            "--source", "https://example.org/saltlog.html"]
    assert cli.main(argv) == 2
    assert "--offline rules out fetching https://example.org/saltlog.html" in capsys.readouterr().err
