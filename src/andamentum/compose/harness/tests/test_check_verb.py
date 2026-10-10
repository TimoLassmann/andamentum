"""The check verb end to end: a planted manuscript in, a commented copy and a report out."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest

from andamentum.compose.harness.bench.fixture_http import (
    fixture_transport,
    scheme_only_url_check,
)
from andamentum.compose.harness.bench.planted import generate
from andamentum.compose.harness.check import (
    CheckParams,
    CheckUsageError,
    run_check,
)
from andamentum.compose.harness.docx.reader import open_docx
from andamentum.compose.harness.kernel import ProjectStore
from ._helpers import clear_outputs, copy_of
from andamentum.core.decisions.testing import StubDecisions, noul_answer


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


@pytest.fixture
def clean(planted: Path, tmp_path: Path) -> Path:
    path = tmp_path / "clean.docx"
    shutil.copy(planted / "c01_clean.docx", path)
    return path


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


async def run(path: Path, *, decision_handle=None, **kw):
    kw.setdefault("out", copy_of(path, "checked"))
    return await run_check(
        CheckParams(file=path, **kw),
        transport=fixture_transport(),
        url_check=scheme_only_url_check,
        decision_handle=decision_handle,
    )


async def test_finds_the_planted_defects_and_writes_the_copy(manuscript: Path):
    before = sha(manuscript)
    result = await run(manuscript)
    report = result.report
    assert sha(manuscript) == before  # the input is never modified
    assert result.output == manuscript.with_name("paper.checked.docx")
    assert report.counts["error"] >= 2 and report.exit_code == 1
    kinds = {d.data.get("kind") for d in report.diagnostics}
    assert {"unresolved", "url", "dangling_target"} <= kinds
    assert "citations" in report.checks_run and "crossrefs" in report.checks_run

    copy = open_docx(result.output)
    ours = [c for c in copy.comments if c.author == "andamentum check"]
    assert len(ours) == report.comments_written > 0
    assert report.comments_refused == []
    assert any("23.4%" in c.text for c in ours)
    assert all(c.text.startswith("[") for c in ours)
    # the original text is untouched by the comments
    assert [p.text for p in copy.paragraphs] == [
        p.text for p in open_docx(manuscript).paragraphs
    ]


async def test_report_files_events_and_snapshot(manuscript: Path):
    result = await run(manuscript)
    assert result.report_md == manuscript.with_name("paper.checked.md")
    payload = json.loads(result.report_json.read_text("utf-8"))
    assert payload["counts"] == result.report.counts
    assert payload["network_calls"] > 0 and payload["model_calls"] == 0
    markdown = result.report_md.read_text("utf-8")
    assert "## citations" in markdown and "Skipped checks" in markdown
    assert "needs --results" in markdown

    store = ProjectStore(manuscript.with_name("state"))
    events = store.read_events()
    assert [e["type"] for e in events] == ["transaction", "check_run"]
    assert events[0]["author"] == "andamentum check"
    assert store.snapshot_log() and store.snapshot_log()[0].startswith("check:")
    assert store.meta().id == result.report.project_id
    assert open_docx(result.output).project_id == result.report.project_id


async def test_second_run_uses_the_project_caches(manuscript: Path):
    first = await run(manuscript)
    clear_outputs(first.output)
    second = await run(manuscript)
    assert first.report.checks_from_cache == []
    assert set(second.report.checks_from_cache) == {
        "abstract_body",
        "crossrefs",
        "statistics",
        "structure",
    }
    assert second.report.network_calls == 0
    assert second.report.http_cache_hits > 0
    assert second.report.diagnostics == first.report.diagnostics
    assert len(ProjectStore(manuscript.with_name("state")).read_events()) == 4


async def test_offline_skips_network_checks_and_says_why(manuscript: Path):
    result = await run(manuscript, offline=True)
    report = result.report
    assert "citations" not in report.checks_run
    assert any(s.id == "citations" and "offline" in s.reason for s in report.skipped)
    assert report.network_calls == 0


async def test_clean_file_exits_zero(clean: Path):
    result = await run(clean)
    assert result.report.counts["error"] == 0
    assert result.report.exit_code == 0
    assert result.report.comments_written == len(result.report.diagnostics) - sum(
        1 for d in result.report.diagnostics if d.paragraph_id is None
    )


async def test_choosing_checks(manuscript: Path):
    result = await run(manuscript, checks=["statistics,crossrefs"])
    assert result.report.checks_run == ["crossrefs", "statistics"]
    assert {s.id for s in result.report.skipped} >= {"citations", "numbers"}


def always_yes() -> StubDecisions:
    return StubDecisions(
        lambda state, questions: {k: noul_answer(0.99) for k in questions}
    )


async def test_model_judged_diagnostics_are_advisory(
    manuscript: Path,
):
    stub = always_yes()
    result = await run(manuscript, decision_handle=stub.model())
    gate = [d for d in result.report.diagnostics if d.check == "needs_citation"]
    assert gate and all(d.severity == "info" and d.tier == 2 for d in gate)
    assert result.report.model_calls > 0
    assert result.report.input_tokens > 0
    assert "needs_citation" in result.report.checks_run


async def test_usage_errors(manuscript: Path, tmp_path: Path):
    with pytest.raises(CheckUsageError, match="not a .docx"):
        await run(tmp_path / "missing.docx")
    with pytest.raises(CheckUsageError, match="unknown check"):
        await run(manuscript, checks=["nope"])
    with pytest.raises(CheckUsageError, match="needs --results"):
        await run(manuscript, checks=["numbers"])
    with pytest.raises(CheckUsageError, match="must not be the input"):
        await run(manuscript, out=manuscript)
    with pytest.raises(CheckUsageError, match="is not a folder"):
        await run(manuscript, results=tmp_path / "nowhere")


async def test_the_project_id_travels_with_the_copy(manuscript: Path, tmp_path: Path):
    project = tmp_path / "shared.harness"
    await run(manuscript, project=project, offline=True)
    # the copy carries the project's id, so checking it again joins the project
    again = await run(
        manuscript.with_name("paper.checked.docx"),
        project=project,
        offline=True,
        out=tmp_path / "again.docx",
    )
    assert again.report.project_id == ProjectStore(project).meta().id


def _field_document(path: Path, payload: str) -> Path:
    from andamentum.compose.harness.spike.make import (
        build_docx,
        complex_field,
        paragraph,
        run as text_run,
    )

    body = paragraph(
        text_run("Sodium fell ")
        + complex_field(f"ADDIN ZOTERO_ITEM CSL_CITATION {payload}", "(Doe 2020)")
        + text_run(".")
    )
    return build_docx(path, body)


async def test_a_fixed_field_payload_is_not_served_from_the_cache(tmp_path: Path):
    path = tmp_path / "cite.docx"
    _field_document(path, "{not json")
    first = await run(path, offline=True, checks=["crossrefs"])
    assert any(
        d.data.get("kind") == "unparseable_field" for d in first.report.diagnostics
    )

    # The display text is unchanged; only the field's payload is repaired.
    _field_document(path, json.dumps({"citationItems": [{"id": "x"}]}))
    clear_outputs(first.output)
    second = await run(path, offline=True, checks=["crossrefs"])
    assert second.report.checks_from_cache == []
    assert not any(
        d.data.get("kind") == "unparseable_field" for d in second.report.diagnostics
    )


async def test_the_project_checks_yaml_holds_the_structure_limits(tmp_path: Path):
    path = tmp_path / "cite.docx"
    _field_document(path, json.dumps({"citationItems": [{"id": "x"}]}))
    project = tmp_path / "state"
    project.mkdir()
    (project / "checks.yaml").write_text("structure:\n  max_words: 1\n")
    result = await run(path, offline=True, checks=["structure"])
    assert [d.check for d in result.report.diagnostics] == ["structure"]
