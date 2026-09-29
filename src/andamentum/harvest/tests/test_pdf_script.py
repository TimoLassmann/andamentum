"""Raised text and page footnotes read from a PDF's own characters.

The fixtures are two small PDFs rendered by LibreOffice from the reference
checker's style test set (``benchmarks/references``), with the markdown
docling made of each saved beside them — so these tests read the PDF with
pypdfium2 and never load docling.
"""

from __future__ import annotations

from pathlib import Path

from andamentum.harvest import api as api_mod
from andamentum.harvest import extract
from andamentum.harvest.fetch import Fetched
from andamentum.harvest.pdf_script import analyse_pdf, mark_markdown

FIXTURES = Path(__file__).parent / "fixtures"


def _load(name: str) -> tuple[bytes, str]:
    return (FIXTURES / f"{name}.pdf").read_bytes(), (
        FIXTURES / f"{name}.docling.md"
    ).read_text()


def test_raised_runs_are_every_superscript_and_nothing_else() -> None:
    pdf, _ = _load("superscript_citations")
    script = analyse_pdf(pdf)
    # Two affiliation marks, nine citations, "10⁵", "m²", one repeated citation.
    assert [r.text for r in script.raised] == [
        "1",
        "2",
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
        "7",
        "8",
        "9",
        "5",
        "2",
        "2",
    ]
    assert script.raised[2].before.endswith("patients")
    # Its numbered reference list starts at the foot of page 1, and its
    # numbers pair with raised marks — but stop short of the page's highest
    # mark, so it is not taken for footnotes.
    assert script.footnotes == []


def test_footnotes_are_found_by_pairing_marks_and_numbers() -> None:
    """This PDF sets its notes at body size with plain numbers; the pairing of
    each mark with the note of its number finds them anyway, and a note's
    line starting "1, Springer…" (a wrapped "no. 1") does not break it."""
    pdf, _ = _load("footnote_citations")
    script = analyse_pdf(pdf)
    assert [(n.page, n.number) for n in script.footnotes] == [
        (1, i) for i in range(1, 9)
    ] + [(2, i) for i in range(9, 16)]
    assert script.footnotes[3].text == "Ibid., 14."
    assert script.footnotes[12].text.endswith("https://doi.org/10.1038/nbt.1754.")
    # Footnote numbers inside the note block are not body marks.
    assert len(script.raised) == 15


def test_superscripts_become_sup_tags_with_docling_spacing_closed() -> None:
    pdf, markdown = _load("superscript_citations")
    marked, stats = mark_markdown(markdown, analyse_pdf(pdf))
    assert stats.unplaced == 0 and stats.marked == 14
    assert "Ada Lovelace<sup>1</sup>" in marked
    assert "most patients<sup>1</sup>" in marked
    assert "10<sup>5</sup> cells on 3 m<sup>2</sup>" in marked


def test_footnotes_become_markdown_footnotes_and_leave_the_body() -> None:
    pdf, markdown = _load("footnote_citations")
    marked, stats = mark_markdown(markdown, analyse_pdf(pdf))
    assert stats.notes == 15 and stats.unplaced == 0
    assert "most patients.[^1-1]" in marked
    assert "[^1-4]: Ibid., 14." in marked
    body = marked.split("\n[^1-1]:")[0]
    # The notes' text, which docling merged into the body, is gone from it
    # (docling writes 'CADD' where the PDF has “CADD”; both are matched).
    assert (
        "CADD: Predicting the Deleteriousness" not in body.split("## Bibliography")[0]
    )


async def test_the_option_is_off_by_default_and_changes_nothing(monkeypatch) -> None:
    """Every existing caller — document_store, proofread, deep_research —
    gets exactly the markdown docling made."""
    pdf, markdown = _load("superscript_citations")

    async def fake_resolve(source, **_kwargs):
        return Fetched(data=pdf, format="pdf", source_url="x.pdf")

    async def fake_docling(data, **_kwargs):
        return markdown

    monkeypatch.setattr(api_mod, "resolve", fake_resolve)
    monkeypatch.setattr(api_mod, "extract_with_docling", fake_docling)
    assert await extract("x.pdf") == markdown
    marked = await extract("x.pdf", preserve_superscripts=True)
    assert "<sup>" in marked


def test_a_mark_docling_dropped_is_placed_after_its_preceding_text() -> None:
    """Docling can leave a raised digit out of its sentence and set it down
    elsewhere. The PDF still says where it belongs."""
    from andamentum.harvest.pdf_script import PdfFootnote, PdfScript, RaisedRun

    markdown = (
        "Visual inspection of alignments remains useful. Further work followed.\n\n"
        "9Li, 'Fast and accurate short,' 22.\n\n9\n"
    )
    script = PdfScript(
        raised=[
            RaisedRun(
                page=2,
                text="9",
                before="ofalignmentsremainsuseful.",
                attached_before=True,
                attached_after=False,
            )
        ],
        footnotes=[
            PdfFootnote(page=2, number=9, text="Li, “Fast and accurate short,” 22.")
        ],
    )
    marked, stats = mark_markdown(markdown, script)
    assert stats.moved == 1 and stats.unplaced == 0
    assert "remains useful.[^2-9] Further" in marked
    assert "[^2-9]: Li, “Fast and accurate short,” 22." in marked
    assert "9Li," not in marked.split("[^2-9]:")[0]
