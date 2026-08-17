"""The document under review, and the Word file its offsets address.

One rule holds the whole pipeline together: **the reviewer reads the text of
the document being annotated.** Get that right and a quote's offsets are
already coordinates in the .docx — there is nothing left to match, align, or
score for similarity.

There are two ways to be sure of it, and this module is both of them:

``.docx`` input
    Project the document to text and review the projection. The provenance
    comes from READING the file: every character in what the reviewer sees is
    a character Word stores, in a run this pipeline can point at.

everything else
    Harvest to markdown, WRITE a baseline ``.docx`` from that markdown, then
    review the projection of the file that was written. The baseline is
    generated before the review rather than after it, so the text the reviewer
    reads is the text the annotations land in — not a sibling rendering of it.

What is not on the list is the third option the pipeline used to take:
harvest to markdown, review the markdown, generate a baseline separately, and
work out afterwards where each quote went. Measured on one real manuscript,
that lost 15 of 43 rewrites — every one reporting a nearest match at 0.97 or
better, because Docling wrote ``aim 3,4 .`` where Word held
``aim3,4.``. It also harvested twice, and for an ambiguous HTML page
harvest races two extractors, so the two harvests need not even agree.

The document is a resource: a non-``.docx`` source leaves a baseline file
behind. Whoever calls :func:`open_source` owns it and calls
:meth:`ReviewedDocument.cleanup`; the entry points that open one themselves
clean up before returning.
"""

from __future__ import annotations

import logging
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger("andamentum.whetstone")

# Extensions that make a bare string look like a file the caller meant us to
# read, rather than a blob of markdown passed inline.
_PATH_EXTENSIONS: frozenset[str] = frozenset(
    {".md", ".markdown", ".txt", ".html", ".htm", ".pdf", ".docx", ".pptx", ".tex"}
)

# A blank line, allowing the whitespace an editor may have left on it.
_BLOCK_BREAK = re.compile(r"\n[ \t]*\n")


@dataclass(frozen=True)
class ReviewedDocument:
    """What the reviewer reads, plus the Word file it is the projection of."""

    text: str
    """The document as a reader sees it — and the coordinate system for every
    ``Quote`` and ``Edit`` the review produces."""

    sections: list  # list[andamentum.whetstone.v3.model.Section]
    """Heading-delimited units, indexing into :attr:`text`."""

    docx_path: Path
    """The file the annotations are written into. Its body projects to
    :attr:`text`, exactly — :func:`~.docx.annotate.apply_annotations` refuses
    to run if it does not."""

    docx_is_temp: bool
    """True when this module generated the file and the caller should remove
    it (see :meth:`cleanup`)."""

    header_and_note_text: str = ""
    """Reader-visible text outside the body: running headers, footers,
    footnotes. Carries no offsets and is never reviewed. It exists for the
    confidentiality tripwire, because a "CONFIDENTIAL — FOR PEER REVIEW ONLY"
    stamp is usually a header, and reading the body directly would otherwise
    walk straight past it."""

    warnings: list[str] = field(default_factory=list)

    @property
    def confidentiality_text(self) -> str:
        """Everything the marker scan should see."""
        if not self.header_and_note_text:
            return self.text
        return f"{self.header_and_note_text}\n{self.text}"

    def cleanup(self) -> None:
        """Remove the generated baseline, if this document owns one."""
        if not self.docx_is_temp:
            return
        try:
            self.docx_path.unlink()
        except OSError:
            pass


async def open_source(source: str) -> ReviewedDocument:
    """Resolve *source* to a document whose text and .docx are the same thing.

    *source* is a path, a URL, or raw markdown content. A string that looks
    like a path but is not there raises ``FileNotFoundError`` rather than being
    reviewed as if the path itself were the manuscript.
    """
    path = Path(source)
    if path.exists() and path.suffix.lower() == ".docx":
        return _open_docx(path, is_temp=False)

    markdown = await _harvest_or_treat_as_markdown(source)
    baseline = _write_baseline_docx(markdown)
    logger.info(
        "[source] not a .docx — reviewing the projection of a generated baseline (%s)",
        baseline,
    )
    return _open_docx(baseline, is_temp=True)


def _open_docx(path: Path, *, is_temp: bool) -> ReviewedDocument:
    # Imported here: the .docx reader pulls in lxml, and a caller that only
    # wants the path heuristics should not pay for it.
    from .docx.reading import auxiliary_text
    from .v3.docx_source import load_docx

    document = load_docx(path)
    warnings: list[str] = []
    if len(document.sections) <= 1 and len(document.text) > _ONE_SECTION_IS_ODD:
        # Word says which paragraphs are headings in their style. A document
        # that names none — bold-and-bigger body text instead — reads as one
        # unit, and every stage downstream then works on a single 40-page
        # chunk. It is legitimate input, but it is not what the author expects
        # to have happened, so say so.
        warnings.append(
            f"{path.name} has no paragraphs styled as headings, so the whole "
            f"document ({len(document.text)} characters) is one section. "
            f"Applying Word's Heading 1/2/3 styles will give a finer review."
        )
        logger.warning("[source] %s", warnings[-1])
    return ReviewedDocument(
        text=document.text,
        sections=document.sections,
        docx_path=path,
        docx_is_temp=is_temp,
        header_and_note_text=auxiliary_text(path),
        warnings=warnings,
    )


# Below this size a one-section document is simply a short document.
_ONE_SECTION_IS_ODD = 20_000


def _write_baseline_docx(markdown: str) -> Path:
    """Write *markdown* out as a plain .docx: headings and paragraphs.

    Deliberately minimal. The output is not a rendering of the markdown, it is
    a document to annotate — and because the review then reads THIS file's
    projection rather than the markdown, nothing downstream depends on how
    faithful the rendering was. What matters is only that each block became one
    paragraph, and that a ``#`` line became a paragraph Word calls a heading,
    so the sectioniser finds the same structure the markdown had.
    """
    from docx import Document  # python-docx

    document = Document()
    for block in _BLOCK_BREAK.split(markdown):
        block = block.strip()
        if not block:
            continue
        level = len(block) - len(block.lstrip("#"))
        if level and block[level : level + 1].isspace():
            document.add_heading(block[level:].strip(), level=min(level, 9))
        else:
            document.add_paragraph(block)

    handle = tempfile.NamedTemporaryFile(
        delete=False, suffix=".docx", prefix="whetstone-baseline-"
    )
    handle.close()
    out = Path(handle.name)
    document.save(str(out))
    return out


def _is_url(source: str) -> bool:
    return source.startswith(("http://", "https://", "file://"))


def _looks_like_filesystem_path(source: str) -> bool:
    """Heuristic: does this string look like a file path the caller meant
    us to read from disk? Used to distinguish a path that doesn't exist
    (loud-fail) from raw markdown content the caller passed directly.

    True if the string is short, single-line, and starts with a path
    prefix OR ends with a recognised file extension. Conservatively
    biased: a string that clearly looks like a content blob (multi-line,
    long, no slashes) is not treated as a path attempt.
    """
    if not source or "\n" in source:
        return False
    if source.startswith(("/", "./", "../", "~/")):
        return True
    if len(source) > 500:
        return False
    return Path(source).suffix.lower() in _PATH_EXTENSIONS


async def _harvest_or_treat_as_markdown(source: str) -> str:
    """Resolve ``source`` to markdown content. If it points to an existing
    file, harvest it. If it looks like a file path but doesn't exist,
    raise FileNotFoundError loudly — silently treating a missing path as
    raw markdown content (the v3 bug fixed in 2026-05-26) made the LLM
    review the string ``"/tmp/missing.md"`` as if it were the manuscript.
    Only fall back to "treat source as raw markdown" when the string
    clearly is not a path attempt.

    A URL goes to harvest as-is. It used to fall through to the raw-markdown
    branch — ``Path("https://…/paper").exists()`` is False and the string has
    no recognised extension — so the review ran on the URL itself while the
    baseline .docx was built from the fetched page."""
    from andamentum.harvest import extract

    if _is_url(source):
        return await extract(source)
    path = Path(source)
    if path.exists():
        return await extract(path)
    if _looks_like_filesystem_path(source):
        raise FileNotFoundError(
            f"input file not found: {source!r}. The string looks like a "
            "path attempt (starts with /, ./, ../, ~/, or ends in a "
            "recognised extension like .md / .docx / .pdf). To pass raw "
            "markdown content directly, include newlines or omit the "
            "file extension; otherwise check the path."
        )
    return source
