"""The SaltLog example project: it builds, reads back, and the offline fixtures serve it."""

from __future__ import annotations

import json
import re
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

from ._helpers import copy_of

from andamentum.compose.harness.bench.fixture_http import scheme_only_url_check
from andamentum.compose.harness.check import CheckParams, run_check
from andamentum.compose.harness.docx.reader import open_docx
from andamentum.compose.harness.examples import (
    EXAMPLE_DIR,
    ExampleProject,
    fixture_transport_for_example,
    make_example,
)


@pytest.fixture
def project(tmp_path: Path) -> ExampleProject:
    return make_example(tmp_path / "saltlog")


def test_make_example_writes_every_file(project: ExampleProject):
    for path in (project.notes, project.library, project.reviews, project.collaborator):
        assert path.is_file()
    assert sorted(p.name for p in project.results.iterdir()) == [
        "intake.csv",
        "summary.csv",
    ]


def test_library_is_csl_json_with_fixture_dois(project: ExampleProject):
    library = json.loads(project.library.read_text(encoding="utf-8"))
    assert len(library) == 3
    fixtures = json.loads((EXAMPLE_DIR / "fixtures" / "crossref.json").read_text())
    for item in library:
        assert {"id", "type", "title", "author", "issued", "DOI"} <= set(item)
        assert item["DOI"] in fixtures["works"]
        assert item["DOI"] in fixtures["updates"]
        assert item["title"].lower() in fixtures["search"]


def test_collaborator_docx_carries_what_the_example_promises(project: ExampleProject):
    view = open_docx(project.collaborator)
    texts = [p.text for p in view.paragraphs]
    assert any("adults logged every meal" in t and "52" in t for t in texts)
    assert not any("48 adults" in t for t in texts)  # the hand edit replaced it
    threads = {c.id: c for c in view.comments}
    assert len(threads) == 4
    assert sum(1 for c in threads.values() if c.parent_id) == 1
    assert [c.resolved for c in view.comments].count(True) == 1
    assert (
        sum(1 for f in view.fields if f.manager == "zotero" and f.kind == "citation")
        == 3
    )
    assert any(f.kind == "bibliography" for f in view.fields)


def test_hand_edited_number_disagrees_with_the_results(project: ExampleProject):
    summary = (project.results / "summary.csv").read_text(encoding="utf-8")
    assert "completed,48" in summary
    assert "52" in " ".join(p.text for p in open_docx(project.collaborator).paragraphs)


def test_make_example_is_repeatable(tmp_path: Path):
    first = make_example(tmp_path / "a").collaborator
    second = make_example(tmp_path / "b").collaborator
    assert [p.text for p in open_docx(first).paragraphs] == [
        p.text for p in open_docx(second).paragraphs
    ]


async def test_check_runs_on_the_example_from_fixtures(project: ExampleProject):
    before = project.collaborator.read_bytes()
    result = await run_check(
        CheckParams(out=copy_of(project.collaborator, "checked"), file=project.collaborator, results=None),
        transport=fixture_transport_for_example(),
        url_check=scheme_only_url_check,
    )
    report = result.report
    assert project.collaborator.read_bytes() == before
    # the hand edit shows up: 52 is in the abstract and nowhere in the body
    errors = [d for d in report.diagnostics if d.severity == "error"]
    assert [d.check for d in errors] == ["abstract_body"]
    assert "52" in errors[0].message
    assert report.exit_code == 1
    assert {"citations", "statistics", "crossrefs"} <= set(report.checks_run)
    assert report.network_calls > 0
    assert not any(d.data.get("kind") == "unresolved" for d in report.diagnostics)


# ------------------------------------------------------------------ the guide

GUIDES = Path(__file__).resolve().parents[5] / "docs"
GUIDE = GUIDES / "word.md"


def guide_commands(guide: Path = GUIDE, tool: str = "andamentum-word") -> list[list[str]]:
    """Every ``<tool> ...`` command of the ``bash`` code blocks of a guide, split into words."""
    text = guide.read_text(encoding="utf-8")
    commands: list[list[str]] = []
    for block in re.findall(r"```bash\n(.*?)```", text, flags=re.DOTALL):
        for command in re.sub(r"\\\n\s*", " ", block).splitlines():
            words = shlex.split(command, comments=True)[1:]
            if command.startswith(tool) and "--help" not in words:
                commands.append(words)
    return commands


def test_the_guide_has_commands():
    verbs = {c[0] for c in guide_commands()}
    assert {
        "check",
        "draft",
        "refresh",
        "bind-numbers",
        "respond",
        "find-citation",
        "insert-citation",
        "mcp",
    } <= verbs


@pytest.mark.parametrize("words", guide_commands(), ids=lambda w: " ".join(w[:2]))
def test_every_command_in_the_guide_parses_against_the_cli(words: list[str]):
    """The guide cannot drift from the CLI: each documented command is accepted as written."""
    from andamentum.word.cli import build_parser

    args = build_parser().parse_args(words)
    assert args.command


@pytest.mark.parametrize(
    "words",
    guide_commands(GUIDES / "compose.md", "andamentum-compose"),
    ids=lambda w: " ".join(w[:2]),
)
def test_every_command_in_the_compose_guide_parses_against_the_cli(words: list[str]):
    from andamentum.compose.cli import build_parser

    args = build_parser().parse_args(words)
    assert args.command


def test_the_guide_names_the_example_files_it_builds(project: ExampleProject):
    text = GUIDE.read_text(encoding="utf-8")
    for path in project.root.iterdir():
        assert path.name in text, path.name


def test_the_example_builds_from_the_command_the_guide_gives(tmp_path: Path):
    command = [
        sys.executable,
        "-m",
        "andamentum.compose.harness.examples",
        str(tmp_path / "demo"),
    ]
    subprocess.run(command, check=True, capture_output=True)
    assert (tmp_path / "demo" / "collaborator.docx").is_file()
