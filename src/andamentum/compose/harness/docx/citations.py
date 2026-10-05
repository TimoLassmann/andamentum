"""Citations and the reference list as docx operations.

``InsertCitation`` and ``UpdateBibliography`` are kernel operations with no
single XML shape: which one the writer writes depends on the document. This
module decides that (:func:`expand_citations`) and builds the pieces the writer
puts in the tree. Nothing here edits XML in place; the writer does that, so the
tracked-change and verification rules stay in one place.

Two modes, chosen per transaction:

``zotero``
    The document already holds Zotero fields, or an item carries a Zotero
    ``uri``. A citation is a complex field ``ADDIN ZOTERO_ITEM CSL_CITATION
    {json}`` whose ``citationItems[].itemData`` is the CSL JSON (``uris`` only
    for an item that came from Zotero), inside one tracked insertion; its
    result is formatted text in the document's style. The reference list is the
    ``ADDIN ZOTERO_BIBL`` field: an existing one gets the missing entries
    appended to its result as a tracked insertion (separated by line breaks;
    Zotero reflows and reorders on the next refresh); with none, a heading
    "References" (unless the document has one) and a new field paragraph are
    inserted, tracked.
``plain``
    Any other document. A citation is formatted text, a tracked insertion; the
    reference list is a "References" section the harness maintains: entries
    already listed are left alone, the missing ones are inserted as tracked
    paragraphs (alphabetical when the list already is, else at the end).

A citation is expanded into the kernel's ``Insert`` (the position handling,
field-edge hoisting and refusal rules of the writer all apply) and a
:class:`FieldPlan` the writer applies to the insertion it produced; the
reference list is expanded into ``InsertParagraph`` ops plus, for an existing
Zotero bibliography, the entries to append.

Limits: EndNote fields are never written (reading only); numbering of plain
numeric citations follows the order works are added, so a citation inserted
before an earlier one leaves the list out of first-citation order, which the
``citations`` check reports; the list of an existing Zotero bibliography is
only ever added to.
"""

from __future__ import annotations

import copy
import json
import re
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from lxml import etree  # type: ignore

from ..kernel.csl import (
    CitationStyle,
    bibliography_entry,
    citation_text,
    detect_style,
    title_of,
    work_key,
    doi_of,
)
from ..kernel.ops import (
    CitationItem,
    Insert,
    InsertCitation,
    InsertParagraph,
    Operation,
    Transaction,
    UpdateBibliography,
)
from ..kernel.view import DocumentView, FieldToken
from .field_spans import Field
from .ooxml import XML_NS, w

CSL_SCHEMA = (
    "https://github.com/citation-style-language/schema/raw/master/csl-citation.json"
)
ITEM_INSTRUCTION = "ADDIN ZOTERO_ITEM CSL_CITATION "
BIBLIOGRAPHY_INSTRUCTION = (
    'ADDIN ZOTERO_BIBL {"uncited":[],"omitted":[],"custom":[]} CSL_BIBLIOGRAPHY'
)
_SECTION_NAMES = frozenset(
    {"references", "bibliography", "literature cited", "works cited", "reference list"}
)
_LEADING_NUMBER = re.compile(r"^\s*[\[(]?(\d+)[\].)]?\s")


@dataclass(frozen=True)
class FieldPlan:
    """A citation insertion to turn into a field once surgery has written it."""

    instruction: str
    display: str


@dataclass
class Expansion:
    """A transaction of kernel ops the writer understands, plus what it must add."""

    txn: Transaction
    origin: list[int]  # derived op index -> index of the op it came from
    citation_fields: dict[int, FieldPlan] = field(default_factory=dict)
    new_bibliography: int | None = None  # derived InsertParagraph index of the field
    new_bibliography_entries: list[str] = field(default_factory=list)
    append_entries: list[str] = field(default_factory=list)  # to an existing field
    bibliography_done: bool = False

    @property
    def has_citation_work(self) -> bool:
        return bool(
            self.citation_fields
            or self.new_bibliography is not None
            or self.append_entries
        )


# ------------------------------------------------------------------ expansion


def expand_citations(
    view: DocumentView,
    txn: Transaction,
    *,
    style_ids: Mapping[str, str],
    skip: frozenset[int] = frozenset(),
) -> Expansion:
    """Rewrite citation operations into kernel operations (see the module docstring).

    *style_ids* is the document's lower-cased style name and id to style id map
    (a "References" heading uses ``heading 1`` only when the document has it).
    *skip* holds the indexes of ``InsertCitation`` operations the writer refused
    on an earlier pass: they are left out, so the bibliography lists only the
    works that are cited in the copy.
    """
    if not any(isinstance(op, InsertCitation | UpdateBibliography) for op in txn.ops):
        return Expansion(txn=txn, origin=list(range(len(txn.ops))))

    cited = [
        op
        for index, op in enumerate(txn.ops)
        if isinstance(op, InsertCitation) and index not in skip
    ]
    uses_zotero = any(f.manager == "zotero" for f in view.fields) or any(
        item.uri for op in cited for item in op.items
    )
    style = detect_style(
        [f.display for f in view.fields if f.kind == "citation" and f.manager != "none"]
        or [f.display for f in view.fields if f.kind == "citation"]
    )
    section = _section(view)
    works = _works(view, cited, include_fields=uses_zotero)
    listed, numbers = _numbering(view, section, works, zotero=uses_zotero)

    ops: list[Operation] = []
    origin: list[int] = []
    fields: dict[int, FieldPlan] = {}
    expansion = Expansion(txn=txn, origin=origin)
    for index, op in enumerate(txn.ops):
        if index in skip:
            continue
        if isinstance(op, InsertCitation):
            text = citation_text(op.items, style, numbers=numbers)
            if uses_zotero:
                fields[len(ops)] = FieldPlan(
                    instruction=zotero_instruction(op.items, text), display=text
                )
            ops.append(Insert(paragraph_id=op.paragraph_id, at=op.at, text=text))
            origin.append(index)
        elif isinstance(op, UpdateBibliography):
            derived = _bibliography_ops(
                view,
                _Context(
                    section=section,
                    works=works,
                    listed=listed,
                    numbers=numbers,
                    style=style,
                    style_ids=style_ids,
                    zotero=uses_zotero,
                ),
                expansion,
                first=len(ops),
            )
            ops.extend(derived)
            origin.extend([index] * len(derived))
        else:
            ops.append(op)
            origin.append(index)
    expansion.txn = Transaction(
        ops=ops, author=txn.author, reason=txn.reason, source=txn.source
    )
    expansion.citation_fields = fields
    return expansion


def _works(
    view: DocumentView, cited: Sequence[InsertCitation], *, include_fields: bool
) -> dict[str, dict[str, object]]:
    """Work key to CSL, in the order numbers are given: the document's Zotero
    items first (document order), then the transaction's."""
    works: dict[str, dict[str, object]] = {}
    if include_fields:
        for token in view.fields:
            if token.kind == "citation" and token.manager == "zotero":
                for csl in _field_items(token):
                    works.setdefault(work_key(csl), csl)
    for op in cited:
        for item in op.items:
            works.setdefault(work_key(item.csl), dict(item.csl))
    return works


def _field_items(token: FieldToken) -> list[dict[str, object]]:
    items = token.data.get("citationItems")
    if not isinstance(items, list):
        return []
    found: list[dict[str, object]] = []
    for item in items:
        data = item.get("itemData") if isinstance(item, dict) else None
        if isinstance(data, dict):
            found.append({str(k): v for k, v in data.items()})
    return found


def _normalise(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def _represents(text: str, csl: Mapping[str, object]) -> bool:
    """Does reference text *text* already list the work *csl*?"""
    doi = doi_of(csl)
    if doi and doi in text.lower():
        return True
    title = _normalise(title_of(csl))
    return bool(title) and title in _normalise(text)


@dataclass
class _Section:
    heading: int | None  # paragraph index of the heading, None when absent
    entries: list[int]  # paragraph indexes of the non-empty entries
    last: int  # paragraph index new paragraphs go after


def _section(view: DocumentView) -> _Section:
    """The reference list: the paragraphs after a "References" heading up to the
    next heading. Absent, ``heading`` is None and ``last`` the document's end."""
    paragraphs = view.paragraphs
    for index, paragraph in enumerate(paragraphs):
        if paragraph.text.strip().casefold().rstrip(":") not in _SECTION_NAMES:
            continue
        if paragraph.heading_level is None and len(paragraph.text) > 40:
            continue
        entries: list[int] = []
        for later in range(index + 1, len(paragraphs)):
            if paragraphs[later].heading_level is not None:
                break
            if paragraphs[later].text.strip():
                entries.append(later)
        return _Section(index, entries, entries[-1] if entries else index)
    return _Section(None, [], len(paragraphs) - 1)


def _numbering(
    view: DocumentView,
    section: _Section,
    works: Mapping[str, Mapping[str, object]],
    *,
    zotero: bool,
) -> tuple[dict[str, int], dict[str, int]]:
    """``(listed, numbers)``: work key to the paragraph index of its plain entry,
    and work key to its reference number.

    A Zotero document numbers works in order of first appearance (Zotero
    renumbers on refresh anyway). A plain document keeps the number of an entry
    already in the list (its own leading number, else its position) and gives
    each new work the next one.
    """
    if zotero:
        return {}, {key: i + 1 for i, key in enumerate(works)}
    listed: dict[str, int] = {}
    entry_number: dict[int, int] = {}
    for position, index in enumerate(section.entries):
        found = _LEADING_NUMBER.match(view.paragraphs[index].text)
        entry_number[index] = int(found.group(1)) if found else position + 1
    for key, csl in works.items():
        for index in section.entries:
            if _represents(view.paragraphs[index].text, csl):
                listed[key] = index
                break
    counter = max([len(section.entries), *entry_number.values()])
    numbers: dict[str, int] = {}
    for key in works:
        if key in listed:
            numbers[key] = entry_number[listed[key]]
        else:
            counter += 1
            numbers[key] = counter
    return listed, numbers


@dataclass
class _Context:
    section: _Section
    works: Mapping[str, Mapping[str, object]]
    listed: Mapping[str, int]
    numbers: Mapping[str, int]
    style: CitationStyle
    style_ids: Mapping[str, str]
    zotero: bool


def _bibliography_ops(
    view: DocumentView, ctx: _Context, expansion: Expansion, *, first: int
) -> list[Operation]:
    """The ops of one ``UpdateBibliography``; a second one in a transaction does nothing."""
    if expansion.bibliography_done:
        return []
    expansion.bibliography_done = True
    numeric = ctx.style.kind == "numeric"

    def entry(key: str) -> str:
        return bibliography_entry(
            ctx.works[key], ctx.style, number=ctx.numbers[key] if numeric else None
        )

    if ctx.zotero:
        existing = next(
            (
                f
                for f in view.fields
                if f.kind == "bibliography" and f.manager == "zotero"
            ),
            None,
        )
        if existing is not None:
            expansion.append_entries = [
                entry(key)
                for key, csl in ctx.works.items()
                if not _represents(existing.display, csl)
            ]
            return []
        keys = list(ctx.works)
    else:
        keys = [key for key in ctx.works if key not in ctx.listed]
    if not keys:
        return []
    if numeric:
        keys.sort(key=lambda key: ctx.numbers[key])
    else:
        keys.sort(key=lambda key: entry(key).casefold())

    paragraphs = view.paragraphs
    section = ctx.section
    ops: list[Operation] = []
    last = paragraphs[section.last].id
    if section.heading is None:
        heading = "heading 1" if "heading 1" in ctx.style_ids else None
        ops.append(
            InsertParagraph(after_paragraph_id=last, text="References", style=heading)
        )
    if ctx.zotero:
        entries = [entry(key) for key in keys]
        ops.append(InsertParagraph(after_paragraph_id=last, text=" ".join(entries)))
        expansion.new_bibliography = first + len(ops) - 1
        expansion.new_bibliography_entries = entries
        return ops

    donor = paragraphs[section.entries[-1]].style if section.entries else "Normal"
    entry_style = (
        donor if donor.lower() in ctx.style_ids and donor != "Normal" else None
    )
    texts = [paragraphs[i].text.casefold() for i in section.entries]
    place_alphabetically = not numeric and bool(texts) and texts == sorted(texts)
    for key in keys:
        text = entry(key)
        anchor = last
        if place_alphabetically:
            before = [
                i
                for i in section.entries
                if paragraphs[i].text.casefold() <= text.casefold()
            ]
            anchor = (
                paragraphs[before[-1]].id
                if before
                else paragraphs[section.heading or 0].id
            )
        ops.append(
            InsertParagraph(after_paragraph_id=anchor, text=text, style=entry_style)
        )
    return ops


# ------------------------------------------------------------------ Zotero XML


def zotero_instruction(items: Sequence[CitationItem], display: str) -> str:
    """``ADDIN ZOTERO_ITEM CSL_CITATION {json}`` for *items*.

    ``itemData`` is the item's CSL JSON; ``uris`` is written only for an item
    that has a ``uri`` (it came from Zotero).
    """
    cited: list[dict[str, object]] = []
    for item in items:
        data = dict(item.csl)
        identifier = data.get("id") or work_key(item.csl)
        data["id"] = identifier
        entry: dict[str, object] = {"id": identifier, "itemData": data}
        if item.uri:
            entry["uris"] = [item.uri]
        if item.locator:
            entry["locator"] = item.locator
            entry["label"] = "page"
        if item.prefix:
            entry["prefix"] = item.prefix
        if item.suffix:
            entry["suffix"] = item.suffix
        cited.append(entry)
    payload = {
        "citationID": uuid.uuid4().hex[:8],
        "properties": {
            "formattedCitation": display,
            "plainCitation": display,
            "noteIndex": 0,
        },
        "citationItems": cited,
        "schema": CSL_SCHEMA,
    }
    return ITEM_INSTRUCTION + json.dumps(
        payload, ensure_ascii=False, separators=(",", ":")
    )


def _run(rpr: Any | None) -> Any:
    run = etree.Element(w("r"))
    if rpr is not None:
        run.append(copy.deepcopy(rpr))
    return run


def _fldchar(kind: str, rpr: Any | None) -> Any:
    run = _run(rpr)
    etree.SubElement(run, w("fldChar")).set(w("fldCharType"), kind)
    return run


def _text_run(text: str, rpr: Any | None) -> Any:
    run = _run(rpr)
    element = etree.SubElement(run, w("t"))
    element.set(f"{{{XML_NS}}}space", "preserve")
    element.text = text
    return run


def _break_run(rpr: Any | None) -> Any:
    run = _run(rpr)
    etree.SubElement(run, w("br"))
    return run


def field_runs(instruction: str, result: Sequence[Any], rpr: Any | None) -> list[Any]:
    """The runs of a complex field: begin, instruction, separate, *result*, end.

    *result* is a list of ready runs (text and line breaks).
    """
    code = _run(rpr)
    element = etree.SubElement(code, w("instrText"))
    element.set(f"{{{XML_NS}}}space", "preserve")
    element.text = f" {instruction} "
    return [
        _fldchar("begin", rpr),
        code,
        _fldchar("separate", rpr),
        *result,
        _fldchar("end", rpr),
    ]


def entry_runs(
    entries: Sequence[str], rpr: Any | None, *, leading_break: bool
) -> list[Any]:
    """Text runs for *entries* with a line break before each (before the first
    only when *leading_break*)."""
    runs: list[Any] = []
    for position, text in enumerate(entries):
        if position or leading_break:
            runs.append(_break_run(rpr))
        runs.append(_text_run(text, rpr))
    return runs


def convert_to_field(run: Any, plan: FieldPlan) -> None:
    """Turn the tracked insertion of ``plan.display`` into a Zotero citation field.

    *run* is the run surgery wrote for the insertion (a child of its ``w:ins``)
    and must read exactly the display text. Anything else means the tree is not
    what the writer planned, and raises ``ValueError``.
    """
    insertion = run.getparent()
    shown = "".join(t.text or "" for t in run.iter(w("t")))
    if insertion is None or insertion.tag != w("ins") or shown != plan.display:
        raise ValueError(
            f"expected a tracked run reading {plan.display!r}, found {shown!r}"
        )
    rpr = run.find(w("rPr"))
    position = list(insertion).index(run)
    insertion.remove(run)
    result = [_text_run(plan.display, rpr)]
    for offset, new in enumerate(field_runs(plan.instruction, result, rpr)):
        insertion.insert(position + offset, new)


def new_bibliography_paragraph(paragraph: Any, entries: Sequence[str]) -> None:
    """Rewrite the text of a tracked inserted paragraph as a Zotero bibliography field.

    *paragraph* is ``tracked_paragraph`` output: its one ``w:ins`` holds one run
    that reads the entries joined by a space.
    """
    insertion = paragraph.find(w("ins"))
    if insertion is None:
        raise ValueError("the inserted paragraph has no tracked insertion")
    runs = [child for child in insertion if child.tag == w("r")]
    if len(runs) != 1:
        raise ValueError("the inserted paragraph does not hold exactly one run")
    rpr = runs[0].find(w("rPr"))
    insertion.remove(runs[0])
    for new in field_runs(
        BIBLIOGRAPHY_INSTRUCTION, entry_runs(entries, rpr, leading_break=False), rpr
    ):
        insertion.append(new)


def field_instruction(field_: Field) -> str:
    """The instruction text of a complex field (the pieces between its begin and
    separate runs); empty for a simple field or one built across parents."""
    if field_.simple or field_.sep is None:
        return ""
    pieces: list[str] = []
    for sibling in field_.begin.itersiblings():
        if sibling is field_.sep:
            return "".join(pieces)
        pieces.extend(t.text or "" for t in sibling.iter(w("instrText")))
    return ""
