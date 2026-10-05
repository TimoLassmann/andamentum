"""R1 reliability: ambient state stays out of the library path.

No load_dotenv, no env vars, no home-directory cache, no container start in
library code; those are explicit CLI behaviour.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

import pytest

import andamentum.deep_research as deep_research_pkg
from andamentum.deep_research import cli, searxng
from andamentum.deep_research.models import EvidenceReport

PACKAGE_DIR = Path(deep_research_pkg.__file__).parent
V2_DIR = PACKAGE_DIR / "v2"


def _legacy_sources() -> list[Path]:
    return [
        p
        for p in PACKAGE_DIR.rglob("*.py")
        if "tests" not in p.parts and V2_DIR not in p.parents
    ]


def test_load_dotenv_only_in_the_cli() -> None:
    offenders = [
        p.name
        for p in _legacy_sources()
        if "load_dotenv" in p.read_text(encoding="utf-8") and p.name != "cli.py"
    ]
    assert offenders == []


def test_no_environment_reads_in_legacy_library() -> None:
    pattern = re.compile(r"os\.(getenv|environ)|Path\.home\(\)|expanduser")
    offenders = [
        p.name
        for p in _legacy_sources()
        if pattern.search(p.read_text(encoding="utf-8"))
    ]
    assert offenders == []


def test_searxng_manager_needs_an_explicit_state_dir() -> None:
    with pytest.raises(TypeError):
        searxng.SearxngManager()  # type: ignore[call-arg]


def test_searxng_manager_creates_nothing_until_asked(tmp_path: Path) -> None:
    state = tmp_path / "state"
    manager = searxng.SearxngManager(state_dir=state)
    assert not state.exists()
    manager.write_minimal_settings()
    assert (state / "settings.yml").is_file()


async def test_run_research_does_not_load_dotenv_or_start_searxng(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import dotenv

    from andamentum.deep_research import graph as graph_module
    from andamentum.deep_research import orchestrator

    def forbidden(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("library code must not do this")

    class _Graph:
        async def run(self, **kwargs: Any) -> EvidenceReport:
            return EvidenceReport(
                evidence_summary="s",
                key_findings=["f"],
                sources=["u"],
                total_searches_performed=0,
                total_pages_fetched=0,
                iterations_required=0,
            )

    class _Backend:
        async def search(self, query: str, max_results: int = 10) -> list:
            return []

        async def fetch_page(self, url: str) -> Any:
            raise AssertionError("not used")

    monkeypatch.setattr(dotenv, "load_dotenv", forbidden)
    monkeypatch.setattr(searxng.SearxngManager, "__init__", forbidden)
    monkeypatch.setattr(searxng.SearxngManager, "ensure_running", forbidden)
    monkeypatch.setattr(graph_module, "research_graph", _Graph())

    result = await orchestrator.run_research(
        "q",
        model="test",
        searxng_url="http://127.0.0.1:4070",
        backend=_Backend(),
    )
    assert result.errors.search_errors == 0


def _args(**overrides: Any) -> argparse.Namespace:
    base: dict[str, Any] = {
        "start_searxng": True,
        "searxng_state_dir": None,
        "searxng_url": "http://127.0.0.1:4071",
        "verbose": False,
    }
    base.update(overrides)
    return argparse.Namespace(**base)


def test_cli_start_searxng_needs_a_state_dir() -> None:
    with pytest.raises(SystemExit, match="--searxng-state-dir"):
        cli._start_searxng(_args())


def test_cli_start_searxng_uses_url_port_and_given_state_dir(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    started: dict[str, Any] = {}

    class _Manager:
        def __init__(self, *, state_dir: Path, host_port: int) -> None:
            started["state_dir"] = state_dir
            started["port"] = host_port

        def is_running(self) -> bool:
            return False

        def ensure_running(self) -> None:
            started["ran"] = True

    monkeypatch.setattr(searxng, "SearxngManager", _Manager)
    cli._start_searxng(_args(searxng_state_dir=tmp_path))
    assert started == {"state_dir": tmp_path, "port": 4071, "ran": True}


def test_cli_does_not_start_searxng_without_the_flag() -> None:
    args = cli._build_parser().parse_args(["a question"])
    assert args.start_searxng is False
    assert args.searxng_state_dir is None


def test_cli_main_loads_dotenv(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr(cli, "load_dotenv", lambda: calls.append("loaded"))
    monkeypatch.setattr(cli.sys, "argv", ["andamentum-research", "agents"])
    cli.main()  # the "agents" path returns before dotenv; run the real path:
    monkeypatch.setattr(cli.sys, "argv", ["andamentum-research", "--version"])
    with pytest.raises(SystemExit):
        cli.main()
    assert calls == ["loaded"]
