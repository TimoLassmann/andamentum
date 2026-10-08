"""``andamentum-research``: flags to arguments, output, exit status."""

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
    """Every run writes its log; without --out it lands in the current directory."""
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
            "Why Apollo?",
            "--mode",
            "both",
            "--model",
            "ollama:gemma4:31b-cloud",
            "--store",
            str(store),
            "--searxng-url",
            "http://127.0.0.1:4070",
            "--browser-cdp-url",
            "http://127.0.0.1:9222",
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
    assert sources.browser_cdp_url == "http://127.0.0.1:9222"
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
        cli.main(["q", "--mode", "scholarly", "--model", "m", flag, "x"])
    assert stopped.value.code == 2 and calls == []


def test_secrets_come_from_the_environment_only(
    calls: list[dict[str, Any]], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENALEX_API_KEY", "k-open")
    monkeypatch.setenv("OPENALEX_EMAIL", "jane@example.org")
    monkeypatch.setenv("SEMANTIC_SCHOLAR_API_KEY", "k-s2")
    cli.main(["q", "--mode", "scholarly", "--model", "m", "--quiet"])
    sources: SourceConfig = calls[0]["sources"]
    assert sources.openalex_api_key == "k-open"
    assert sources.mailto == "jane@example.org"
    assert sources.semantic_scholar_api_key == "k-s2"
    assert calls[0]["store"] is None


def test_json_output_to_a_file(calls: list[dict[str, Any]], tmp_path: Path) -> None:
    out = tmp_path / "result.json"
    cli.main(["q", "--mode", "web", "--model", "m", "--json", "--out", str(out)])
    assert json.loads(out.read_text())["stop_reason"] == "answered"


def test_a_degraded_run_exits_4_after_writing_the_result(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    async def research(question: str, **kwargs: Any) -> ResearchResult:
        return _result(degraded=["web search failed for 'x': HTTP 500"])

    monkeypatch.setattr(cli, "research", research)
    monkeypatch.setattr(cli, "load_dotenv", lambda: None)
    with pytest.raises(SystemExit) as exited:
        cli.main(["q", "--mode", "web", "--model", "m", "--quiet"])
    assert exited.value.code == cli.EXIT_DEGRADED
    out = capsys.readouterr()
    assert out.out.startswith("# Why Apollo?")
    assert "web search failed" in out.err


def test_an_error_exits_1(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    async def research(question: str, **kwargs: Any) -> ResearchResult:
        raise ValueError("web mode needs searxng_url")

    monkeypatch.setattr(cli, "research", research)
    monkeypatch.setattr(cli, "load_dotenv", lambda: None)
    with pytest.raises(SystemExit) as exited:
        cli.main(["q", "--mode", "web", "--model", "m", "--quiet"])
    assert exited.value.code == 1
    assert "searxng_url" in capsys.readouterr().err


@pytest.mark.parametrize(
    "argv",
    [
        ["q", "--model", "m"],  # no mode
        ["q", "--mode", "web"],  # no model
        ["q", "--mode", "web", "--model", "m", "--sources", "openalex,nope"],
        ["q", "--mode", "web", "--model", "m", "--json", "--md"],
    ],
)
def test_usage_errors_exit_2(
    argv: list[str], calls: list[dict[str, Any]], capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as exited:
        cli.main(argv)
    assert exited.value.code == 2 and calls == []


def test_the_question_is_the_first_argument_and_argv_defaults_to_the_process(
    calls: list[dict[str, Any]], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["andamentum-research", "Why Apollo?", "--mode", "web", "--model", "m"],
    )
    cli.main()
    assert [c["question"] for c in calls] == ["Why Apollo?"]


def test_the_mcp_subcommand_goes_to_the_mcp_server(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from .. import mcp_server

    seen: list[list[str]] = []
    monkeypatch.setattr(mcp_server, "main", seen.append)
    cli.main(["mcp", "--store", "s.db"])
    assert seen == [["--store", "s.db"]]


def test_the_cache_subcommands_go_to_the_cache_cli(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from .. import cache_cli

    seen: list[list[str]] = []
    monkeypatch.setattr(cache_cli, "main", seen.append)
    cli.main(["cache-info", "--cache", "c.db"])
    assert seen == [["cache-info", "--cache", "c.db"]]
