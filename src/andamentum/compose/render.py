"""Rendering: evidence markers become citations (or vanish), outputs are written.

Deterministic. The drafter marks fact use as ``[[f12]]`` right after the sentence
that relies on it. Here, and only here, those markers turn into numbered
citations (genres that cite) or disappear (genres that do not), the outputs are
written to ``<bundle>/out/``, and provenance is applied. compose never sends
anything: the email output is a text file.
"""

from __future__ import annotations

import re
import shutil
import tempfile
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import NamedTuple

from andamentum.compose.bundle import Bundle, Section
from andamentum.whetstone import provenance_line, stamp_provenance

from .genres import OutputFormat, Genre
from .schemas import Brief, Fact, Source

MARKER_RE = re.compile(r"\[\[(f\d+)\]\]")
SOURCES_TITLE = "Sources"
OUT_DIR = "out"
_VALID_FORMATS = ("docx", "md", "html", "txt")


class ProvenanceRequired(ValueError):
    """Provenance was switched off for a genre that does not allow it."""


def marker_ids(text: str) -> list[str]:
    """Fact ids cited in *text*, in order of appearance (repeats kept)."""
    return MARKER_RE.findall(text)


def _tidy_line(line: str) -> str:
    """Close the gap a removed marker leaves, without touching other spacing."""
    stripped = line.strip(" ")
    lead = line[: len(line) - len(line.lstrip(" "))]
    return lead + re.sub(r"(?<=\S) {2,}(?=\S)", " ", stripped)


def _replace_markers(text: str, replace: Callable[[re.Match[str]], str]) -> str:
    out: list[str] = []
    for line in text.split("\n"):
        if "[[" not in line:
            out.append(line)
            continue
        line = re.sub(r"[ \t]+(?=\[\[f\d+\]\])", " ", line)
        out.append(_tidy_line(MARKER_RE.sub(replace, line)))
    return "\n".join(out)


def strip_markers(text: str) -> str:
    """Remove every ``[[fN]]`` marker."""
    cleaned = _replace_markers(text, lambda m: "")
    return re.sub(r" +([.,;:!?])", r"\1", cleaned) if "[[" in text else cleaned


def source_numbers(
    sections: Sequence[Section], *, facts: Sequence[Fact]
) -> dict[str, int]:
    """Source id -> citation number, by first appearance across the sections in order."""
    by_id = {f.id: f for f in facts}
    numbers: dict[str, int] = {}
    for section in sorted(sections, key=lambda s: s.order):
        for fact_id in marker_ids(section.body):
            fact = by_id.get(fact_id)
            if fact is not None and fact.source_id not in numbers:
                numbers[fact.source_id] = len(numbers) + 1
    return numbers


def citation_filter(
    numbers: dict[str, int], *, facts: Sequence[Fact]
) -> Callable[[str], str]:
    """A ``text_filter``: ``[[f12]]`` becomes ``[n]``; unknown ids vanish."""
    by_id = {f.id: f for f in facts}

    def replace(match: re.Match[str]) -> str:
        fact = by_id.get(match.group(1))
        if fact is None or fact.source_id not in numbers:
            return ""
        return f"[{numbers[fact.source_id]}]"

    def apply(text: str) -> str:
        replaced = _replace_markers(text, replace)
        # Two facts from one source in a row cite it once.
        collapsed = re.sub(r"(\[\d+\])(?:\s*\1)+", r"\1", replaced)
        return re.sub(r" +([.,;:!?])", r"\1", collapsed) if "[[" in text else collapsed

    return apply


def sources_body(numbers: dict[str, int], *, sources: Sequence[Source]) -> str:
    """The numbered list under the Sources heading: title and file name or URL.

    Only ``Source.label`` is printed, never ``Source.path``, so a finished
    document does not reveal the author's directory layout.
    """
    by_id = {s.id: s for s in sources}
    lines = []
    for source_id, number in sorted(numbers.items(), key=lambda kv: kv[1]):
        source = by_id.get(source_id)
        if source is None:
            label = source_id
        elif source.kind == "research":
            label = source.label or "Web research report"
        elif source.label and source.label != source.title:
            label = f"{source.title} ({source.label})"
        else:
            label = source.title
        lines.append(f"{number}. {label}")
    return "\n".join(lines)


def plain_text(sections: Sequence[Section], *, title: str = "") -> str:
    """The document as a reader sees it, markers removed: title, then each section."""
    parts = [title] if title else []
    for section in sorted(sections, key=lambda s: s.order):
        parts.append(section.title)
        parts.append(strip_markers(section.body))
    return "\n\n".join(p for p in parts if p.strip())


def _email_text(
    sections: Sequence[Section], *, brief: Brief, text_filter: Callable[[str], str]
) -> str:
    ordered = sorted(sections, key=lambda s: s.order)
    if len(ordered) == 1:
        body = text_filter(ordered[0].body).strip()
    else:
        body = "\n\n".join(f"{s.title}\n{text_filter(s.body).strip()}" for s in ordered)
    return f"Subject: {brief.email_subject}\n\n{body}\n"


def _working_copy(bundle: Bundle, *, into: Path, sources_text: str) -> Bundle:
    """A throwaway copy of the bundle that carries the Sources section.

    Rendering the copy keeps every format on one code path and leaves the real
    bundle's sections untouched, so checks and resume never see "Sources".
    """
    for name in ("bundle.yaml", "references.yaml"):
        if (bundle.path / name).exists():
            shutil.copy2(bundle.path / name, into / name)
    for name in ("sections", "figures"):
        if (bundle.path / name).exists():
            shutil.copytree(bundle.path / name, into / name)
    copy = Bundle.open(into)
    if sources_text:
        last = copy.sections()[-1].id
        copy.add_section(
            title=SOURCES_TITLE, body=sources_text, after=last, status="drafted"
        )
    return copy


def _provenance_footer(path: Path, *, fmt: str, model: str) -> None:
    line = provenance_line(model=model)
    if fmt in ("md", "html"):
        footer = f"\n<!-- {line} -->\n"
    else:
        footer = f"\n--\n{line}\n"
    with path.open("a", encoding="utf-8") as fh:
        fh.write(footer)


class ShippedView(NamedTuple):
    """The document as a reader receives it, ready to render."""

    bundle: Bundle
    text_filter: Callable[[str], str]


@contextmanager
def shipped_view(
    bundle: Bundle,
    *,
    genre: Genre,
    facts: Sequence[Fact],
    sources: Sequence[Source],
) -> Iterator[ShippedView]:
    """What ships: citations (or none) in the text and, for genres that cite, the
    Sources section, on a throwaway copy of the bundle.

    Delivery and the critic both render through this, so the critic reviews
    exactly the document the reader gets. Render the copy with
    ``image_root=bundle.path``; the copy is deleted when the block ends.
    """
    sections = bundle.sections()
    if genre.cites:
        numbers = source_numbers(sections, facts=facts)
        text_filter = citation_filter(numbers, facts=facts)
        sources_text = sources_body(numbers, sources=sources)
    else:
        text_filter = strip_markers
        sources_text = ""
    with tempfile.TemporaryDirectory(prefix="compose-render-") as tmp:
        copy_dir = Path(tmp) / "bundle"
        copy_dir.mkdir()
        yield ShippedView(
            _working_copy(bundle, into=copy_dir, sources_text=sources_text),
            text_filter,
        )


def render_outputs(
    bundle: Bundle,
    *,
    brief: Brief,
    genre: Genre,
    facts: Sequence[Fact],
    sources: Sequence[Source],
    model: str,
    formats: Sequence[str] | None = None,
    provenance: bool = True,
) -> list[Path]:
    """Write the finished document to ``<bundle>/out/`` in each requested format.

    *formats* defaults to ``genre.outputs``. *model* is recorded as the producing
    model in the provenance marks. Raises ``ProvenanceRequired`` when
    ``provenance=False`` is asked of a genre that requires it, and ``ValueError``
    for an unknown format.
    """
    if not provenance and genre.provenance_required:
        raise ProvenanceRequired(
            f"genre {genre.name!r} requires provenance markers; "
            "they cannot be switched off"
        )
    wanted: list[OutputFormat] = []
    for fmt in formats if formats is not None else genre.outputs:
        if fmt not in _VALID_FORMATS:
            raise ValueError(
                f"unknown output format {fmt!r}; valid: {', '.join(_VALID_FORMATS)}"
            )
        if fmt not in wanted:
            wanted.append(fmt)  # type: ignore[arg-type]

    sections = bundle.sections()
    out_dir = bundle.path / OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    with shipped_view(bundle, genre=genre, facts=facts, sources=sources) as view:
        copy, text_filter = view
        for fmt in wanted:
            if fmt == "txt" and brief.email_subject:
                target = out_dir / "email.txt"
                target.write_text(
                    _email_text(sections, brief=brief, text_filter=text_filter),
                    encoding="utf-8",
                )
            else:
                target = out_dir / f"document.{fmt}"
                copy.render(
                    target,
                    format=fmt,
                    text_filter=text_filter,
                    image_root=bundle.path,
                )
            if provenance:
                if fmt == "docx":
                    stamp_provenance(target, model=model)
                else:
                    _provenance_footer(target, fmt=fmt, model=model)
            written.append(target)
    return written
