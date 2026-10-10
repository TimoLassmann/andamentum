"""A verb never writes over a file: a copy from an earlier run may hold edits.

Each verb that writes a copy refuses, before anything is written (``state/``
included), when the copy or its report already exists. The refusal comes before
the document is opened, so an empty ``paper.docx`` is enough here.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from andamentum.compose.harness.bindings.bind import BindParams, run_bind
from andamentum.compose.harness.bindings.refresh import RefreshParams, run_refresh
from andamentum.compose.harness.check import CheckParams, run_check
from andamentum.compose.harness.errors import WordUsageError
from andamentum.compose.harness.literature.verbs import InsertParams, run_insert
from andamentum.compose.harness.respond.run import RespondParams, run_respond
from andamentum.compose.harness.spike.make import make_spike_docx


async def _check(file: Path, out: Path) -> object:
    return await run_check(CheckParams(file=file, out=out, offline=True))


async def _respond(file: Path, out: Path) -> object:
    return await run_respond(RespondParams(file=file, out=out))


async def _refresh(file: Path, out: Path) -> object:
    return run_refresh(RefreshParams(file=file, out=out, results=file.parent))


async def _bind(file: Path, out: Path) -> object:
    return await run_bind(BindParams(file=file, out=out, results=file.parent, model="m"))


async def _insert(file: Path, out: Path) -> object:
    params = InsertParams(file=file, out=out, paragraph_id="p0", at=0, doi=["10.5555/x"])
    return await run_insert(params)


VERBS: dict[str, tuple[Callable, str, bool]] = {
    # name: (runner, copy suffix, writes a report beside the copy)
    "check": (_check, "checked", True),
    "respond": (_respond, "responded", True),
    "refresh": (_refresh, "refreshed", True),
    "bind-numbers": (_bind, "bound", True),
    "insert-citation": (_insert, "cited", False),
}


def _taken_names(name: str) -> list[str]:
    _, suffix, report = VERBS[name]
    names = [f"paper.{suffix}.docx"]
    if report:
        names += [f"paper.{suffix}.md", f"paper.{suffix}.json"]
    return names


@pytest.mark.parametrize(
    ("name", "taken"),
    [(name, taken) for name in VERBS for taken in _taken_names(name)],
)
async def test_a_taken_name_is_refused_and_nothing_is_written(
    tmp_path: Path, name: str, taken: str
):
    run, suffix, _ = VERBS[name]
    file = tmp_path / "in" / "paper.docx"
    file.parent.mkdir()
    file.write_bytes(b"")
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    (out_dir / taken).write_text("edited in Word", encoding="utf-8")

    with pytest.raises(WordUsageError, match="never overwrites"):
        await run(file, out_dir / f"paper.{suffix}.docx")

    assert sorted(p.name for p in out_dir.iterdir()) == [taken]
    assert (out_dir / taken).read_text(encoding="utf-8") == "edited in Word"


def test_the_command_line_exits_2_and_keeps_the_copy(tmp_path: Path, capsys):
    from andamentum.word import cli

    file = make_spike_docx(tmp_path / "paper.docx")
    out_dir = tmp_path / "out"
    args = ["check", str(file), "--out", str(out_dir), "--offline"]
    assert cli.main(args) in (0, 1)
    copy = out_dir / "paper.checked.docx"
    copy.write_bytes(b"edited in Word")
    capsys.readouterr()

    assert cli.main(args) == 2
    assert "never overwrites" in capsys.readouterr().err
    assert copy.read_bytes() == b"edited in Word"
