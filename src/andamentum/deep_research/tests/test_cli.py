"""``andamentum-research``: flags to arguments, output, exit codes."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

from .. import cli
from ..sources import SourceConfig
from ..types import Budget, ResearchResult


def _result(*, degraded: list[str] | None = None) -> ResearchResult:
    return ResearchResult(
        run_id="r_1",
        question="Why Apollo?",
        mode="scholarly",
        report_markdown="# Why Apollo?\n\nIt is fast [F1].\n",
        stop_reason="answered",
        degraded=degraded or [],
    )


@pytest.fixture(autouse=True)
def _in_tmp(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Nothing is written into the current folder unless --out names it."""
    monkeypatch.chdir(tmp_path)


@pytest.fixture
def calls(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    seen: list[dict[str, Any]] = []

    async def research(question: str, **kwargs: Any) -> ResearchResult:
        seen.append({"question": question, **kwargs})
        return _result()

    monkeypatch.setattr(cli, "research", research)
    for name in ("OPENALEX_API_KEY", "OPENALEX_EMAIL", "SEMANTIC_SCHOLAR_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(cli, "load_dotenv", lambda: None)
    return seen


def test_flags_become_the_research_arguments(
    calls: list[dict[str, Any]], capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    store = tmp_path / "research.db"
    cli.main(
        [
            "ask",
            "Why Apollo?",
            "--mode",
            "both",
            "--model",
            "ollama:gemma4:31b-cloud",
            "--store",
            str(store),
            "--searxng-url",
            "http://127.0.0.1:4070",
            "--browser",
            "--sources",
            "europepmc,openalex",
            "--tdm-host",
            "journals.example",
            "--max-rounds",
            "2",
            "--max-wall-s",
            "90.5",
            "--quiet",
        ]
    )
    (call,) = calls
    assert call["question"] == "Why Apollo?"
    assert call["mode"] == "both"
    assert call["model"] == "ollama:gemma4:31b-cloud"
    assert call["store"] == store
    assert call["budget"] == Budget(max_rounds=2, max_wall_s=90.5)
    sources: SourceConfig = call["sources"]
    assert sources.scholarly == ("europepmc", "openalex")
    assert sources.searxng_url == "http://127.0.0.1:4070"
    assert sources.browser is True
    assert sources.tdm_allowed_hosts == frozenset({"journals.example"})
    assert sources.openalex_api_key is None
    assert "config" not in call and "decision_model" not in call
    out = capsys.readouterr()
    assert out.out == "# Why Apollo?\n\nIt is fast [F1].\n"
    assert "stop: answered" in out.err


@pytest.mark.parametrize("flag", ["--preset", "--decision-model", "--embedding-model"])
def test_the_flags_of_the_removed_steps_are_usage_errors(
    flag: str, calls: list[dict[str, Any]]
) -> None:
    with pytest.raises(SystemExit) as stopped:
        cli.main(["ask", "q", "--mode", "scholarly", "--model", "m", flag, "x"])
    assert stopped.value.code == 2 and calls == []


def test_secrets_come_from_the_environment_only(
    calls: list[dict[str, Any]], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENALEX_API_KEY", "k-open")
    monkeypatch.setenv("OPENALEX_EMAIL", "jane@example.org")
    monkeypatch.setenv("SEMANTIC_SCHOLAR_API_KEY", "k-s2")
    cli.main(["ask", "q", "--mode", "scholarly", "--model", "m", "--quiet"])
    sources: SourceConfig = calls[0]["sources"]
    assert sources.openalex_api_key == "k-open"
    assert sources.mailto == "jane@example.org"
    assert sources.semantic_scholar_api_key == "k-s2"
    assert calls[0]["store"] is None


def test_without_out_nothing_is_written(
    calls: list[dict[str, Any]], capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    assert cli.main(["ask", "q", "--mode", "scholarly", "--model", "m", "--quiet"]) == 0
    assert capsys.readouterr().out.startswith("# Why Apollo?")
    assert list(tmp_path.iterdir()) == []


def test_out_is_a_folder_with_the_report_and_the_log(
    calls: list[dict[str, Any]], tmp_path: Path
) -> None:
    out = tmp_path / "run"
    cli.main(
        ["ask", "q", "--mode", "scholarly", "--model", "m", "-o", str(out), "--quiet"]
    )
    assert sorted(p.name for p in out.iterdir()) == ["log.jsonl", "log.md", "report.md"]
    assert (out / "report.md").read_text().startswith("# Why Apollo?")
    assert [p.name for p in tmp_path.iterdir()] == ["run"]


def test_json_output_into_the_folder(
    calls: list[dict[str, Any]], tmp_path: Path
) -> None:
    out = tmp_path / "run"
    cli.main(
        [
            "ask",
            "q",
            "--mode",
            "scholarly",
            "--model",
            "m",
            "--json",
            "-o",
            str(out),
            "--quiet",
        ]
    )
    assert json.loads((out / "report.json").read_text())["stop_reason"] == "answered"
    assert not (out / "report.md").exists()


def test_a_degraded_run_exits_1_after_writing_the_result(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    async def research(question: str, **kwargs: Any) -> ResearchResult:
        return _result(degraded=["web search failed for 'x': HTTP 500"])

    monkeypatch.setattr(cli, "research", research)
    monkeypatch.setattr(cli, "load_dotenv", lambda: None)
    code = cli.main(
        ["ask", "q", "--mode", "web", "--browser", "--model", "m", "--quiet"]
    )
    assert code == cli.EXIT_DEGRADED == 1
    out = capsys.readouterr()
    assert out.out.startswith("# Why Apollo?")
    assert "web search failed" in out.err


def test_a_failed_run_exits_4_with_an_error_line(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    async def research(question: str, **kwargs: Any) -> ResearchResult:
        raise RuntimeError("model unreachable")

    monkeypatch.setattr(cli, "research", research)
    monkeypatch.setattr(cli, "load_dotenv", lambda: None)
    code = cli.main(["ask", "q", "--mode", "scholarly", "--model", "m", "--quiet"])
    assert code == 4
    assert capsys.readouterr().err.startswith("error: RuntimeError: model unreachable")


def test_a_missing_local_corpus_exits_3(
    calls: list[dict[str, Any]], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = cli.main(
        ["ask", "q", "--mode", "scholarly", "--model", "m", "--sources", "local_corpus"]
        + ["--local-corpus", str(tmp_path / "none")]
    )
    assert code == 3 and calls == []
    assert capsys.readouterr().err.startswith("error:")


@pytest.mark.parametrize(
    "argv",
    [
        ["ask", "q", "--model", "m"],  # no mode
        ["ask", "q", "--mode", "scholarly"],  # no model
        [
            "ask",
            "q",
            "--mode",
            "scholarly",
            "--model",
            "m",
            "--sources",
            "openalex,nope",
        ],
        ["ask", "q", "--mode", "web", "--model", "m"],  # web needs an address
        [
            "ask",
            "q",
            "--mode",
            "scholarly",
            "--model",
            "m",
            "--sources",
            "local_corpus",
        ],
        ["ask", "q", "--mode", "scholarly", "--model", "m", "--md"],
        ["no-such-command"],
        [],
    ],
)
def test_usage_errors_exit_2(
    argv: list[str], calls: list[dict[str, Any]], capsys: pytest.CaptureFixture[str]
) -> None:
    try:
        code = cli.main(argv)
    except SystemExit as exited:
        code = exited.code
    assert code == 2 and calls == []


def test_a_question_that_starts_with_mcp_is_a_question(
    calls: list[dict[str, Any]],
) -> None:
    cli.main(
        [
            "ask",
            "mcp servers compared",
            "--mode",
            "scholarly",
            "--model",
            "m",
            "--quiet",
        ]
    )
    assert [c["question"] for c in calls] == ["mcp servers compared"]


def test_the_question_can_come_from_stdin(
    calls: list[dict[str, Any]], monkeypatch: pytest.MonkeyPatch
) -> None:
    import io

    monkeypatch.setattr(sys, "stdin", io.StringIO("Why Apollo?\n"))
    cli.main(["ask", "-", "--mode", "scholarly", "--model", "m", "--quiet"])
    assert [c["question"] for c in calls] == ["Why Apollo?"]


def test_every_budget_flag_has_a_plain_description() -> None:
    ask = cli.build_parser()._subparsers._group_actions[0].choices["ask"]  # type: ignore[union-attr]
    helps = {a.dest: a.help for a in ask._actions}
    for name in Budget.model_fields:
        assert helps[name] and "default:" in helps[name]
        assert helps[name].split(" (default")[0] != ""


def test_the_cache_subcommands_run_the_cache_job(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    assert cli.main(["cache", "info", "--cache", str(tmp_path / "none.db")]) == 3


def test_ask_is_the_default_command(calls: list[dict[str, Any]]) -> None:
    code = cli.main(["Why Apollo?", "--mode", "scholarly", "--model", "m", "--quiet"])
    assert code == 0
    assert [c["question"] for c in calls] == ["Why Apollo?"]


def test_the_default_command_leaves_flags_and_other_commands_alone(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as stop:
        cli.main(["--version"])
    assert stop.value.code == 0
    assert capsys.readouterr().out.startswith("andamentum-research ")
    assert cli.main([]) == cli.EXIT_USAGE


def test_the_help_says_web_mode_needs_an_address(
    capsys: pytest.CaptureFixture[str],
) -> None:
    for argv in (["--help"], ["ask", "--help"]):
        with pytest.raises(SystemExit):
            cli.main(argv)
        text = " ".join(capsys.readouterr().out.split())
        assert "--searxng-url URL or --browser" in text or (
            "--searxng-url" in text and "--browser" in text and "web" in text
        )
    with pytest.raises(SystemExit) as stop:
        cli.main(["q", "--mode", "web", "--model", "m"])
    assert stop.value.code == 2


def test_mailto_wins_over_the_environment(
    calls: list[dict[str, Any]], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENALEX_EMAIL", "env@example.org")
    cli.main(["q", "--mode", "scholarly", "--model", "m", "--quiet"])
    cli.main(
        ["q", "--mode", "scholarly", "--model", "m", "--quiet",
         "--mailto", "flag@example.org"]
    )  # fmt: skip
    assert [c["sources"].mailto for c in calls] == [
        "env@example.org",
        "flag@example.org",
    ]
