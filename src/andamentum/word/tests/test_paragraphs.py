"""andamentum-word paragraphs: read-only list of paragraph ids."""

from __future__ import annotations

import json

import pytest

from andamentum.compose.harness.errors import WordInputError
from andamentum.compose.harness.tests._bindings_helpers import make_manuscript
from andamentum.word import cli, list_paragraphs


async def test_list_paragraphs_returns_ids_and_text(tmp_path):
    paper = make_manuscript(tmp_path / "paper.docx")
    before = paper.read_bytes()
    items = await list_paragraphs(paper)
    assert items and all(item.id and item.text for item in items)
    assert any(item.heading_level is not None for item in items)
    assert paper.read_bytes() == before


async def test_list_paragraphs_rejects_a_missing_file(tmp_path):
    with pytest.raises(WordInputError):
        await list_paragraphs(tmp_path / "nope.docx")


def test_paragraphs_command_prints_and_writes_nothing(tmp_path, capsys):
    paper = make_manuscript(tmp_path / "paper.docx")
    assert cli.main(["paragraphs", str(paper)]) == 0
    out = capsys.readouterr().out
    assert out.strip()
    assert cli.main(["paragraphs", str(paper), "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert {"id", "style", "heading_level", "text"} <= set(data[0])
    assert sorted(p.name for p in tmp_path.iterdir()) == ["paper.docx"]


def test_paragraphs_missing_file_exits_3(tmp_path, capsys):
    assert cli.main(["paragraphs", str(tmp_path / "nope.docx")]) == 3
    assert "error:" in capsys.readouterr().err
