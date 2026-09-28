"""Write real Word footnotes into a .docx that has none yet.

python-docx has no footnote API. A caller that wants footnotes writes a
placeholder run ``⟦FNn⟧`` where each mark goes, saves, and calls
:func:`add_footnotes` with the notes' texts in order: each placeholder run
becomes a ``<w:footnoteReference>``, and a ``footnotes.xml`` part is added
with its relationship and content type. The placeholder is not text the
reader sees — it is replaced whole — so the document's projection is the
text with the marks where Word shows them.
"""

from __future__ import annotations

import re
import zipfile
from io import BytesIO
from pathlib import Path
from xml.sax.saxutils import escape

PLACEHOLDER = "⟦FN{n}⟧"
_W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
_REL = "http://schemas.openxmlformats.org/officeDocument/2006/relationships/footnotes"
_TYPE = "application/vnd.openxmlformats-officedocument.wordprocessingml.footnotes+xml"


def add_footnotes(path: str | Path, notes: list[str]) -> None:
    """Replace placeholders ``⟦FN1⟧ … ⟦FNn⟧`` in *path* with footnotes."""
    if not notes:
        return
    path = Path(path)
    buffer = BytesIO()
    with zipfile.ZipFile(path) as source, zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as target:
        if "word/footnotes.xml" in source.namelist():
            raise ValueError(f"{path} already has footnotes; add_footnotes writes a new part")
        for item in source.infolist():
            body = source.read(item.filename)
            if item.filename == "word/document.xml":
                text = body.decode("utf-8")
                for n in range(1, len(notes) + 1):
                    text, count = re.subn(
                        rf"<w:r>(?:<w:rPr>(?:(?!</w:rPr>).)*</w:rPr>)?<w:t[^>]*>{re.escape(PLACEHOLDER.format(n=n))}</w:t></w:r>",
                        '<w:r><w:rPr><w:vertAlign w:val="superscript"/></w:rPr>'
                        f'<w:footnoteReference w:id="{n}"/></w:r>',
                        text,
                    )
                    if count != 1:
                        raise ValueError(f"footnote placeholder {n} found {count} times")
                body = text.encode("utf-8")
            elif item.filename == "word/_rels/document.xml.rels":
                body = body.replace(
                    b"</Relationships>",
                    f'<Relationship Id="rIdAndamentumFootnotes" Type="{_REL}" Target="footnotes.xml"/></Relationships>'.encode(),
                )
            elif item.filename == "[Content_Types].xml":
                body = body.replace(
                    b"</Types>",
                    f'<Override PartName="/word/footnotes.xml" ContentType="{_TYPE}"/></Types>'.encode(),
                )
            target.writestr(item, body)
        parts = [
            '<w:footnote w:type="separator" w:id="-1"><w:p><w:r><w:separator/></w:r></w:p></w:footnote>',
            '<w:footnote w:type="continuationSeparator" w:id="0"><w:p><w:r><w:continuationSeparator/></w:r></w:p></w:footnote>',
        ] + [
            f'<w:footnote w:id="{n}"><w:p><w:r><w:t xml:space="preserve">{escape(note)}</w:t></w:r></w:p></w:footnote>'
            for n, note in enumerate(notes, start=1)
        ]
        target.writestr(
            "word/footnotes.xml",
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            f'<w:footnotes xmlns:w="{_W}">{"".join(parts)}</w:footnotes>',
        )
    path.write_bytes(buffer.getvalue())
