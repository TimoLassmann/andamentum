"""``andamentum-research v2``: flags to arguments, output, exit status."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

from .. import cli
from ..config import PRESETS
from ..sources import SourceConfig
from ..types import Budget, ResearchResult


def _result(*, degraded: list[str] | None = None) -> ResearchResult:
    return ResearchResult(
        run_id="r_1",
        question="Why Apollo?",
        mode="scholarly",
        report_markdown="# Why Apollo?\n\nIt is fast [F1].\n",
        stop_reason="covered",
        degraded=degraded or [],
    )


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
            "--decision-model",
            "nimble:9b",
            "--store",
            str(store),
            "--searxng-url",
            "http://127.0.0.1:4070",
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
    assert call["decision_model"] == "nimble:9b"
    assert call["store"] == store
    assert call["budget"] == Budget(max_rounds=2, max_wall_s=90.5)
    sources: SourceConfig = call["sources"]
    assert sources.scholarly == ("europepmc", "openalex")
    assert sources.searxng_url == "http://127.0.0.1:4070"
    assert sources.tdm_allowed_hosts == frozenset({"journals.example"})
    assert sources.openalex_api_key is None
    assert call["config"] is PRESETS["baseline"]  # the default preset
    out = capsys.readouterr()
    assert out.out == "# Why Apollo?\n\nIt is fast [F1].\n"
    assert "stop: covered" in out.err


def test_the_preset_flag_picks_the_configuration(calls: list[dict[str, Any]]) -> None:
    cli.main(
        ["q", "--mode", "scholarly", "--model", "m", "--preset", "criteria", "--quiet"]
    )
    assert calls[0]["config"] is PRESETS["criteria"]
    with pytest.raises(SystemExit) as stopped:
        cli.main(["q", "--mode", "scholarly", "--model", "m", "--preset", "nope"])
    assert stopped.value.code == 2


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
    assert calls[0]["store"] is None and calls[0]["decision_model"] is None


def test_json_output_to_a_file(calls: list[dict[str, Any]], tmp_path: Path) -> None:
    out = tmp_path / "result.json"
    cli.main(["q", "--mode", "web", "--model", "m", "--json", "--out", str(out)])
    assert json.loads(out.read_text())["stop_reason"] == "covered"


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


def test_the_research_cli_dispatches_the_v2_subcommand(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from andamentum.deep_research import cli as research_cli

    seen: list[list[str]] = []
    monkeypatch.setattr(cli, "main", seen.append)
    monkeypatch.setattr(
        sys, "argv", ["andamentum-research", "v2", "q", "--mode", "web"]
    )
    research_cli.main()
    assert seen == [["q", "--mode", "web"]]
