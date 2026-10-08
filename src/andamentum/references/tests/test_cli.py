"""``andamentum-references`` end to end, offline."""

from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from andamentum.references.cli import build_parser, main

DRAFT = """Introduction

Rare diseases are common [1]. Costs are high [3].

References

1. Yang G, Cintina I (2022) The national economic burden of rare disease in the United States in 2019. Orphanet J Rare Dis 17:163
2. Zurynski Y, Deverell M (2017) Australian children living with rare diseases. Orphanet J Rare Dis 12:68
"""

CLEAN = """Introduction

Rare diseases are common [1]. Costs are high [2].

References

1. Yang G, Cintina I (2022) The national economic burden of rare disease in the United States in 2019. Orphanet J Rare Dis 17:163
2. Zurynski Y, Deverell M (2017) Australian children living with rare diseases. Orphanet J Rare Dis 12:68
"""


def _run(argv, capsys):
    code = main(argv)
    out = capsys.readouterr()
    return code, out.out, out.err


def test_problems_are_listed_and_exit_one(tmp_path: Path, capsys) -> None:
    draft = tmp_path / "draft.md"
    draft.write_text(DRAFT)
    code, out, _ = _run([str(draft), "--offline"], capsys)
    assert code == 1
    assert "Citation 3 is not in the list" in out
    assert "Reference 2 is never cited" in out
    assert "Crossref lookup: off" in out


def test_a_clean_draft_exits_zero(tmp_path: Path, capsys) -> None:
    draft = tmp_path / "draft.md"
    draft.write_text(CLEAN)
    code, out, _ = _run([str(draft), "--offline"], capsys)
    assert code == 0
    assert "No problems found." in out


def test_json_output_carries_the_text_of_each_problem(tmp_path: Path, capsys) -> None:
    draft = tmp_path / "draft.md"
    draft.write_text(DRAFT)
    code, out, _ = _run([str(draft), "--offline", "--json"], capsys)
    assert code == 1
    data = json.loads(out)
    assert data["found_list"] is True
    assert any("[3]" in p["text"] for p in data["problems"])


def test_out_writes_the_file_and_nothing_to_stdout(tmp_path: Path, capsys) -> None:
    draft = tmp_path / "draft.md"
    draft.write_text(CLEAN)
    dest = tmp_path / "sub" / "result.txt"
    code, out, _ = _run([str(draft), "--offline", "-o", str(dest)], capsys)
    assert code == 0
    assert out == ""
    assert "No problems found." in dest.read_text()


def test_stdin_is_read_as_markdown(monkeypatch, capsys) -> None:
    monkeypatch.setattr("sys.stdin", io.StringIO(DRAFT))
    code, out, _ = _run(["-", "--offline"], capsys)
    assert code == 1
    assert "Citation 3 is not in the list" in out


def test_no_reference_list_is_reported_and_exits_one(tmp_path: Path, capsys) -> None:
    draft = tmp_path / "draft.md"
    draft.write_text("Just some prose with no list.\n")
    code, out, _ = _run([str(draft), "--offline"], capsys)
    assert code == 1
    assert "No reference list was found" in out


def test_a_missing_file_exits_three(tmp_path: Path, capsys) -> None:
    code, _, err = _run([str(tmp_path / "nope.md")], capsys)
    assert code == 3
    assert err.startswith("error:")


def test_a_word_file_goes_through_harvest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    seen: list[Path] = []

    async def fake_extract(source, **_kwargs):
        seen.append(Path(source))
        return CLEAN

    monkeypatch.setattr("andamentum.harvest.extract", fake_extract)
    draft = tmp_path / "draft.docx"
    draft.write_bytes(b"not read: harvest is stubbed")
    code, out, _ = _run([str(draft), "--offline"], capsys)
    assert code == 0
    assert seen == [draft]


def test_an_unreadable_word_file_exits_three(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    from andamentum.harvest import ExtractionError

    async def failing(source, **_kwargs):
        raise ExtractionError("broken")

    monkeypatch.setattr("andamentum.harvest.extract", failing)
    draft = tmp_path / "draft.docx"
    draft.write_bytes(b"x")
    code, _, err = _run([str(draft), "--offline"], capsys)
    assert code == 3
    assert "broken" in err


def test_an_unavailable_decision_model_stops_the_run_before_reading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    from andamentum.core.decisions import DecisionModel, DecisionModelUnavailable

    async def unavailable(*_args: object, **_kwargs: object) -> DecisionModel:
        raise DecisionModelUnavailable("run: ollama pull nimble:9b")

    monkeypatch.setattr(DecisionModel, "open", unavailable)
    draft = tmp_path / "draft.md"
    draft.write_text(CLEAN)
    code, out, err = _run(
        [str(draft), "--offline", "--decision-model", "nimble:9b"], capsys
    )
    assert code == 4
    assert out == ""
    assert "ollama pull nimble:9b" in err


def test_a_model_flag_has_no_default() -> None:
    args = build_parser().parse_args(["draft.md"])
    assert args.model is None and args.decision_model is None
