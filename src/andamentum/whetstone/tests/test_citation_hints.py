"""What the Word file says about citations that its text does not."""

from __future__ import annotations

from pathlib import Path

from docx import Document
from docx.oxml.ns import qn
from lxml import etree  # type: ignore

from andamentum.whetstone.docx.citation_hints import citation_hints
from andamentum.whetstone.reviewed_document import open_source


async def _hints(path: Path):
    document = await open_source(str(path))
    return document.text, citation_hints(document.docx_path, expected_text=document.text)


async def test_superscript_runs_become_hints_at_their_offsets(tmp_path: Path) -> None:
    doc = Document()
    p = doc.add_paragraph()
    p.add_run("Diagnosis is slow")
    p.add_run("12").font.superscript = True
    p.add_run(". Area of 3 m")
    p.add_run("2").font.superscript = True
    path = tmp_path / "raised.docx"
    doc.save(str(path))

    text, hints = await _hints(path)
    assert [(h.kind, text[h.start : h.end]) for h in hints] == [
        ("superscript", "12"),
        ("superscript", "2"),
    ]


async def test_a_zotero_field_result_is_one_citation_hint(tmp_path: Path) -> None:
    """begin / instruction / separate / result / end, as Zotero writes it."""
    doc = Document()
    p = doc.add_paragraph()
    p.add_run("Rare diseases are common ")
    runs = [p.add_run() for _ in range(5)]
    runs[0]._r.append(_fld("begin"))
    instr = etree.SubElement(runs[1]._r, qn("w:instrText"))
    instr.text = ' ADDIN ZOTERO_ITEM CSL_CITATION {"citationItems":[]} '
    runs[2]._r.append(_fld("separate"))
    runs[3].text = "(Smith et al., 2020)"
    runs[4]._r.append(_fld("end"))
    p.add_run(".")
    path = tmp_path / "zotero.docx"
    doc.save(str(path))

    text, hints = await _hints(path)
    fields = [h for h in hints if h.kind == "field_citation"]
    assert [text[h.start : h.end] for h in fields] == ["(Smith et al., 2020)"]
    assert "ZOTERO_ITEM" in fields[0].payload


async def test_a_field_that_is_not_a_citation_is_ignored(tmp_path: Path) -> None:
    doc = Document()
    p = doc.add_paragraph()
    runs = [p.add_run() for _ in range(5)]
    runs[0]._r.append(_fld("begin"))
    instr = etree.SubElement(runs[1]._r, qn("w:instrText"))
    instr.text = " REF _Ref123 \\h "
    runs[2]._r.append(_fld("separate"))
    runs[3].text = "Figure 2"
    runs[4]._r.append(_fld("end"))
    path = tmp_path / "ref.docx"
    doc.save(str(path))

    _, hints = await _hints(path)
    assert hints == []


async def test_a_different_text_gets_no_hints(tmp_path: Path) -> None:
    doc = Document()
    doc.add_paragraph().add_run("x1").font.superscript = True
    path = tmp_path / "x.docx"
    doc.save(str(path))
    assert citation_hints(path, expected_text="something else") == []


def _fld(kind: str):
    element = etree.Element(qn("w:fldChar"))
    element.set(qn("w:fldCharType"), kind)
    return element
