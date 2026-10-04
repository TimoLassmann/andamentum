"""A bundle .docx is sectionised by whetstone into the bundle's own sections.

compose maps whetstone findings back to bundle sections by title, so the
titles whetstone reads out of the rendered .docx must equal the bundle's, in
order. The bundle stays free of a whetstone import; the check lives here.

Whetstone bands its sections by size (6000 characters). A document longer than
that splits at its ``Heading 1`` paragraphs, and the Word ``Title`` paragraph
shows up as a leading piece named after the document. A document shorter than
the band is ONE section named after the document — compose cannot map findings
by section title there, and the last test pins that so the behaviour is known.
"""

from andamentum.compose.bundle import Bundle
from andamentum.whetstone import open_source

TITLES = ["Summary", "Findings", "Next steps"]
SENTENCE = (
    "Apollo Labs measured a twelve percent drop in median latency across the pilot. "
)


def _bundle(path, *, repeats: int) -> Bundle:
    bundle = Bundle.create(path, title="Apollo Pilot")
    for title in TITLES:
        paragraphs = [
            f"{SENTENCE * 4}Item {n} of {title.lower()}.".strip()
            for n in range(repeats)
        ]
        bundle.add_section(title=title, body="\n\n".join(paragraphs), status="drafted")
    return bundle


async def test_whetstone_section_titles_equal_bundle_titles_in_order(tmp_path):
    bundle = _bundle(tmp_path / "doc.compose", repeats=8)  # ~2.5k chars a section
    docx = bundle.render(tmp_path / "doc.docx", format="docx")

    document = await open_source(str(docx))
    try:
        titles = [s.title for s in document.sections]
        assert titles[0] == bundle.title  # the Title paragraph's own piece
        assert titles[1:] == [s.title for s in bundle.sections()]
        assert titles[1:] == TITLES
        for section in document.sections:
            assert section.text == document.text[section.start : section.end]
    finally:
        document.cleanup()


async def test_whetstone_reads_lists_and_tables_inside_their_section(tmp_path):
    bundle = Bundle.create(tmp_path / "doc.compose", title="Apollo Pilot")
    filler = "\n\n".join([SENTENCE * 4] * 12)
    bundle.add_section(
        title="Summary", body=f"{filler}\n\n- Faster reads\n- Fewer retries"
    )
    bundle.add_section(
        title="Findings", body=f"{filler}\n\n| Week | Errors |\n|---|---|\n| 1 | 4 |"
    )
    docx = bundle.render(tmp_path / "doc.docx", format="docx")

    document = await open_source(str(docx))
    try:
        by_title = {s.title: s.text for s in document.sections}
        assert "Fewer retries" in by_title["Summary"]
        assert (
            "Errors" in by_title["Findings"]
            and "Fewer retries" not in by_title["Findings"]
        )
    finally:
        document.cleanup()


async def test_short_document_is_one_whetstone_section_named_after_the_document(
    tmp_path,
):
    bundle = _bundle(tmp_path / "doc.compose", repeats=1)
    docx = bundle.render(tmp_path / "doc.docx", format="docx")

    document = await open_source(str(docx))
    try:
        assert [s.title for s in document.sections] == [bundle.title]
        for title in TITLES:
            assert title in document.sections[0].text
    finally:
        document.cleanup()
