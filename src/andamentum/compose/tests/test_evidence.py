from __future__ import annotations

import pytest

from andamentum.compose.evidence import (
    RawFact,
    RawFacts,
    check_quotes,
    extract_facts,
    find_quote,
    ingest_sources,
    stored_source_path,
)
from andamentum.compose.log import ActionLog
from andamentum.compose.schemas import Source
from andamentum.whetstone import check_confidentiality  # noqa: F401

from ._stub import install, make_brief

TEXT = "Apollo Labs will run the pilot.\nIt starts on 1   March  2027 in Perth."


def test_find_quote_exact():
    span = find_quote(TEXT, "starts on")
    assert span is not None and TEXT[span[0] : span[1]] == "starts on"


def test_find_quote_normalised_whitespace_and_case():
    span = find_quote(TEXT, "STARTS ON 1 march 2027")
    assert span is not None
    assert TEXT[span[0] : span[1]] == "starts on 1   March  2027"


def test_find_quote_absent_and_empty():
    assert find_quote(TEXT, "ends in July") is None
    assert find_quote(TEXT, "   ") is None


def test_check_quotes_drops_absent_and_numbers_globally():
    raw = [
        RawFact(statement="Apollo runs it", quote="Apollo Labs will run"),
        RawFact(statement="invented", quote="costs ten dollars"),
        RawFact(statement="Start date", quote="1 March 2027"),
    ]
    facts, dropped = check_quotes(raw, source_id="src1", text=TEXT, first_number=5)
    assert dropped == 1
    assert [f.id for f in facts] == ["f5", "f6"]
    assert facts[1].quote == "1   March  2027"
    assert TEXT[facts[0].start : facts[0].end] == facts[0].quote


async def test_extract_facts_caps_and_logs_drop(monkeypatch, tmp_path):
    many = [RawFact(statement=f"s{i}", quote="Apollo Labs") for i in range(40)]
    many.append(RawFact(statement="bad", quote="nonsense text"))
    install(monkeypatch, [RawFacts(facts=many)])
    src = Source(
        id="src1", path="a.md", title="A", sha256="0" * 64, chars=len(TEXT), kind="file"
    )
    log = ActionLog(tmp_path)
    facts = await extract_facts(
        src, TEXT, brief=make_brief(), first_number=1, model="m", log=log
    )
    assert len(facts) == 25  # capped before the quote check
    events = [r for r in log.records() if r.get("event") == "facts_dropped"]
    assert not events  # the bad one was beyond the cap


async def test_extract_facts_records_dropped(monkeypatch, tmp_path):
    install(
        monkeypatch,
        [
            RawFacts(
                facts=[
                    RawFact(statement="ok", quote="Apollo Labs"),
                    RawFact(statement="bad", quote="nonsense text"),
                ]
            )
        ],
    )
    src = Source(
        id="src2", path="a.md", title="A", sha256="0" * 64, chars=len(TEXT), kind="file"
    )
    log = ActionLog(tmp_path)
    facts = await extract_facts(
        src, TEXT, brief=make_brief(), first_number=3, model="m", log=log
    )
    assert [f.id for f in facts] == ["f3"]
    ev = [r for r in log.records() if r.get("event") == "facts_dropped"]
    assert ev and ev[0]["dropped"] == 1


async def test_ingest_markdown_and_csv(tmp_path):
    md = tmp_path / "note.md"
    md.write_text("# Apollo note\n\nThe pilot starts in March.\n", encoding="utf-8")
    csv = tmp_path / "data.csv"
    csv.write_text("year,value\n2025,1\n2026,2\n", encoding="utf-8")
    bundle = tmp_path / "b"
    sources = await ingest_sources([str(md), str(csv)], bundle_dir=bundle)
    assert [s.id for s in sources] == ["src1", "src2"]
    assert sources[0].title == "Apollo note"
    assert (
        stored_source_path(bundle, sources[0]).read_text().startswith("# Apollo note")
    )
    stored_csv = stored_source_path(bundle, sources[1])
    assert stored_csv.suffix == ".csv" and stored_csv.read_text() == csv.read_text()
    again = await ingest_sources([str(md), str(csv)], bundle_dir=bundle)
    assert [s.sha256 for s in again] == [s.sha256 for s in sources]


async def test_ingest_confidentiality_raises_unless_confirmed(tmp_path):
    f = tmp_path / "secret.md"
    f.write_text("# Doc\n\nCONFIDENTIAL - do not distribute.\n", encoding="utf-8")
    with pytest.raises(Exception):
        check_confidentiality(f.read_text())
    with pytest.raises(Exception):
        await ingest_sources([str(f)], bundle_dir=tmp_path / "b")
    got = await ingest_sources(
        [str(f)], bundle_dir=tmp_path / "b2", confirm_own_sources=True
    )
    assert len(got) == 1
