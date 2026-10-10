"""Output versions in a Word command's folder (``compose.harness.outputs``).

An earlier copy that is unchanged since the tool wrote it is written over; one
that changed is kept and the run writes ``-2``, ``-3`` and so on. A run on the
original while a changed copy of it sits in the folder says which copy.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from andamentum.compose.harness.outputs import (
    FINGERPRINTS,
    changed_copies,
    copy_names,
    finish_outputs,
    plan_outputs,
)
from andamentum.compose.harness.spike.make import make_spike_docx
from andamentum.word import api


@pytest.fixture
def paper(tmp_path: Path) -> Path:
    return make_spike_docx(tmp_path / "paper.docx")


def _names(folder: Path) -> list[str]:
    return sorted(p.name for p in folder.iterdir() if p.is_file())


def test_an_unchanged_copy_is_written_over(paper: Path, tmp_path: Path, capsys):
    from andamentum.word import cli

    out = tmp_path / "out"
    args = ["check", str(paper), "--out", str(out), "--offline"]
    assert cli.main(args) in (0, 1)
    assert cli.main(args) in (0, 1)
    assert _names(out) == ["paper.checked.docx", "paper.checked.json", "paper.checked.md"]
    assert "note:" not in capsys.readouterr().err
    assert not any((out / "state" / ".replaced").iterdir())


def test_a_changed_copy_is_kept_and_the_next_version_written(
    paper: Path, tmp_path: Path, capsys
):
    from andamentum.word import cli

    out = tmp_path / "out"
    args = ["check", str(paper), "--out", str(out), "--offline"]
    assert cli.main(args) in (0, 1)
    edited = out / "paper.checked.docx"
    edited.write_bytes(edited.read_bytes() + b"\0")  # saved again in Word
    capsys.readouterr()

    assert cli.main(args) in (0, 1)
    err = capsys.readouterr().err
    assert edited.read_bytes().endswith(b"\0")
    assert (out / "paper.checked-2.docx").exists()
    assert (out / "paper.checked-2.md").exists() and (out / "paper.checked-2.json").exists()
    assert f"kept {edited}" in err and "writing" in err and "paper.checked-2.docx" in err
    assert f"{edited} changed after andamentum wrote it" in err
    assert "run on it instead of" in err

    # -2 is unchanged, so the next run writes over it, not -3
    assert cli.main(args) in (0, 1)
    assert not (out / "paper.checked-3.docx").exists()

    # once -2 changes too, the run goes on to -3
    second = out / "paper.checked-2.docx"
    second.write_bytes(second.read_bytes() + b"\0")
    assert cli.main(args) in (0, 1)
    assert (out / "paper.checked-3.docx").exists()
    assert second.read_bytes().endswith(b"\0")


def test_a_file_with_no_fingerprint_is_kept(paper: Path, tmp_path: Path):
    from andamentum.word import cli

    out = tmp_path / "out"
    out.mkdir()
    (out / "paper.checked.md").write_text("my notes", encoding="utf-8")
    assert cli.main(["check", str(paper), "--out", str(out), "--offline"]) in (0, 1)
    assert (out / "paper.checked.md").read_text(encoding="utf-8") == "my notes"
    assert (out / "paper.checked-2.docx").exists()
    assert not (out / "paper.checked.docx").exists()


def test_running_on_the_changed_copy_itself_gives_no_warning(
    paper: Path, tmp_path: Path, capsys
):
    from andamentum.word import cli

    out = tmp_path / "out"
    assert cli.main(["check", str(paper), "--out", str(out), "--offline"]) in (0, 1)
    edited = out / "paper.checked.docx"
    edited.write_bytes(edited.read_bytes() + b"\0")
    capsys.readouterr()

    assert cli.main(["check", str(edited), "--out", str(out), "--offline"]) in (0, 1)
    assert "changed after andamentum wrote it" not in capsys.readouterr().err
    assert (out / "paper.checked.checked.docx").exists()


def test_a_run_that_writes_no_copy_puts_the_replaced_files_back(tmp_path: Path):
    names = copy_names(tmp_path, "paper.cited", report=False)
    state = tmp_path / "state"
    first = plan_outputs(names, state=state)
    first.files[0].write_bytes(b"cited")
    finish_outputs(first)
    assert (state / FINGERPRINTS).exists()

    second = plan_outputs(names, state=state)  # unchanged: moved aside
    assert second.files == first.files and not first.files[0].exists()
    finish_outputs(second)  # the run wrote nothing (a work did not verify)
    assert first.files[0].read_bytes() == b"cited"


def test_changed_copies_lists_only_changed_copies_of_the_input(tmp_path: Path):
    state = tmp_path / "state"
    for stem in ("paper.checked", "paper.bound", "other.checked"):
        plan = plan_outputs(copy_names(tmp_path, stem), state=state)
        for path in plan.files:
            path.write_bytes(b"x")
        finish_outputs(plan)
    (tmp_path / "paper.bound.docx").write_bytes(b"edited")
    (tmp_path / "other.checked.docx").write_bytes(b"edited")
    found = changed_copies(tmp_path / "paper.docx", folder=tmp_path, state=state)
    assert found == [tmp_path / "paper.bound.docx"]


def test_respond_and_draft_names_carry_the_version():
    folder = Path("out")
    respond = api._respond_names(folder, Path("paper.docx"), letter=True)
    assert [p.name for p in respond(2)] == [
        "paper.responded-2.docx",
        "paper.responded-2.md",
        "paper.responded-2.json",
        "response-letter-2.docx",
        "response-letter-2.md",
    ]
    assert len(api._respond_names(folder, Path("paper.docx"), letter=False)(1)) == 3
    assert [p.name for p in api._draft_names(folder)(3)] == [
        "draft-3.docx",
        "draft-3.report.md",
        "draft-3.report.json",
    ]
