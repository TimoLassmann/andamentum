"""Apply edits by cutting the runs a span covers, and nothing else.

The rule this module exists to enforce: **an edit may modify only the runs its
resolved span covers.** Every other node in the document is left where it is,
untouched — line breaks, tabs, hyperlinks, content controls, footnote
references, bookmarks, earlier tracked changes, and whatever Word adds next.

The design this replaces read each paragraph as a ``<w:t>`` join, edited that
string, and rebuilt the paragraph from it by diffing tokens. Rebuilding from a
lossy projection destroys everything the projection dropped: measured, one edit
took a paragraph's ``<w:br>`` count from 1 to 0 and welded "keep" and "and"
into "keepand". It also flattened any tracked change already in the paragraph,
silently accepting a co-author's revisions and erasing their name.

Nothing is rebuilt here, so none of that is expressible.

Two further consequences fall out rather than being added:

*Marks stay narrow.* The token diff is not gone, it is scoped to the span. So
replacing ``"earns its place"`` with ``"earns its keep"`` marks ``place`` and
``keep``, not the whole phrase — and a line break inside an unchanged part of
the span is simply never in a cut.

*Paragraph boundaries are ordinary characters.* Because a paragraph break is an
atom, an edit that spans one merges the paragraphs, and a ``\\n`` in the
replacement text splits one. Both are tracked changes Word understands, and
neither needed a rule of its own.
"""

from __future__ import annotations

import copy
import datetime
import difflib
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from lxml import etree  # type: ignore

from .atoms import Atom, AtomKind, qn
from .textmap import AtomSpan, TextMap

logger = logging.getLogger("andamentum.whetstone")

# Whitespace runs are tokens in their own right, so a diff that only changes
# spacing marks the spacing and not the words either side of it.
_TOKEN_RE = re.compile(r"\s+|\S+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text)


# ── XML construction ────────────────────────────────────────────────────


def _timestamp() -> str:
    return datetime.datetime.now().replace(microsecond=0).isoformat()


def _text_run(text: str, rpr: Any | None) -> Any:
    run = etree.Element(qn("r"))
    if rpr is not None:
        run.append(copy.deepcopy(rpr))
    element = etree.SubElement(run, qn("t"))
    element.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
    element.text = text
    return run


def _change(kind: str, *, author: str, cid: int) -> Any:
    container = etree.Element(qn(kind))
    container.set(qn("id"), str(cid))
    container.set(qn("author"), author)
    container.set(qn("date"), _timestamp())
    return container


def _insertion(text: str, rpr: Any | None, *, author: str, cid: int) -> Any:
    container = _change("ins", author=author, cid=cid)
    container.append(_text_run(text, rpr))
    return container


def _deletion_of_text(text: str, rpr: Any | None, *, author: str, cid: int) -> Any:
    container = _change("del", author=author, cid=cid)
    run = etree.Element(qn("r"))
    if rpr is not None:
        run.append(copy.deepcopy(rpr))
    element = etree.SubElement(run, qn("delText"))
    element.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
    element.text = text
    container.append(run)
    return container


def _deletion_of_run(run: Any, *, author: str, cid: int) -> Any:
    """Wrap a whole run (e.g. one holding a ``<w:br>``) in a deletion."""
    container = _change("del", author=author, cid=cid)
    moved = copy.deepcopy(run)
    for text_element in moved.findall(qn("t")):
        text_element.tag = qn("delText")
    container.append(moved)
    return container


#: Where change ids start in a document that carries none. Arbitrary, and high
#: enough to be recognisable as ours in a hand-inspected file.
FIRST_CHANGE_ID = 1000


def next_change_id(root: Any) -> int:
    """One above every revision id already in the document under *root*.

    Word groups a revision by its ``w:id``: accepting one change accepts every
    element sharing that id. So a second round that restarts its numbering
    welds its own edits onto the first round's — measured on a two-round run,
    ``<w:ins id="1000" author="A">`` and ``<w:del id="1000" author="B">`` ended
    up in one file, where accepting A's insertion also accepted B's deletion of
    it.

    A revision marker is recognised by carrying both an id and an author, which
    is the signature of every one of them (``w:ins``, ``w:del``, ``w:moveFrom``,
    ``w:rPrChange``, …) rather than a list this has to keep up to date.
    """
    id_attr, author_attr = qn("id"), qn("author")
    used = [
        element.get(id_attr)
        for element in root.iter()
        if isinstance(element.tag, str)
        and element.get(author_attr) is not None
        and element.get(id_attr) is not None
    ]
    numeric = [int(value) for value in used if value.isdigit()]
    return max(numeric) + 1 if numeric else FIRST_CHANGE_ID


def _mark_paragraph_mark(p_elem: Any, kind: str, *, author: str, cid: int) -> None:
    """Mark a paragraph's own mark as inserted or deleted.

    Deleting the mark is how Word represents "these two paragraphs merge when
    the change is accepted"; inserting it is how it represents a new paragraph.
    Neither moves any content, which is what keeps the operation local.
    """
    ppr = p_elem.find(qn("pPr"))
    if ppr is None:
        ppr = etree.Element(qn("pPr"))
        p_elem.insert(0, ppr)
    rpr = ppr.find(qn("rPr"))
    if rpr is None:
        rpr = etree.SubElement(ppr, qn("rPr"))
    for existing in rpr.findall(qn(kind)):
        rpr.remove(existing)
    rpr.insert(0, _change(kind, author=author, cid=cid))


# ── the plan ────────────────────────────────────────────────────────────


@dataclass
class _RunCuts:
    """Every change to make inside one run, in run-local character offsets."""

    deletions: list[tuple[int, int]] = field(default_factory=list)
    insertions: list[tuple[int, str]] = field(default_factory=list)
    delete_whole_run: bool = False
    # Offsets that must end up as node boundaries even if nothing is cut there,
    # so a paragraph split has an element to sit after.
    split_points: set[int] = field(default_factory=set)


@dataclass
class Edit:
    """One resolved edit: replace what *span* covers with *new_text*."""

    span: AtomSpan
    new_text: str
    author: str


@dataclass
class EditReport:
    applied: int = 0
    conflicted: list[tuple[Edit, Edit]] = field(default_factory=list)
    paragraphs_merged: int = 0
    paragraphs_created: int = 0


def _diff_operations(old: str, new: str) -> list[tuple[str, int, int, str]]:
    """Character-range operations turning *old* into *new*, at token granularity.

    Returns ``(kind, lo, hi, text)`` where ``lo``/``hi`` are offsets into *old*
    and ``text`` is what to insert there. Token granularity is what keeps the
    tracked marks narrow: one word changed shows as one word changed.
    """
    old_tokens, new_tokens = _tokenize(old), _tokenize(new)
    old_starts, position = [], 0
    for token in old_tokens:
        old_starts.append(position)
        position += len(token)
    old_starts.append(position)

    operations: list[tuple[str, int, int, str]] = []
    matcher = difflib.SequenceMatcher(None, old_tokens, new_tokens, autojunk=False)
    for kind, i1, i2, j1, j2 in matcher.get_opcodes():
        if kind == "equal":
            continue
        lo, hi = old_starts[i1], old_starts[i2]
        operations.append((kind, lo, hi, "".join(new_tokens[j1:j2])))
    return operations


def _anchor_for_insertion(
    atoms: Sequence[Atom], index: int
) -> tuple[Any, int] | None:
    """Where to put text inserted at atom *index*: ``(run, run-local offset)``.

    Prefers to append after the preceding character so the insertion inherits
    that run's formatting and sits inside the same paragraph.
    """
    previous = atoms[index - 1] if index > 0 else None
    if previous is not None and previous.kind is AtomKind.TEXT:
        return previous.run, previous.offset + 1
    following = atoms[index] if index < len(atoms) else None
    if following is not None and following.kind is AtomKind.TEXT:
        return following.run, following.offset
    return None


def _compile(
    textmap: TextMap, edits: Sequence[Edit]
) -> tuple[
    dict[Any, _RunCuts],
    list[tuple[Any, str, str]],
    list[tuple[Any, int, list[str], str]],
]:
    """Turn resolved edits into per-run cuts plus paragraph-level operations.

    Compiling everything before touching the tree is what makes each run
    rewritable exactly once — so no operation can invalidate another's address.
    """
    atoms = textmap.atoms
    cuts: dict[Any, _RunCuts] = {}
    paragraph_marks: list[tuple[Any, str, str]] = []  # (p_elem, kind, author)
    splits: list[tuple[Any, int, list[str], str]] = []  # (run, offset, tails, author)

    def cuts_for(run: Any) -> _RunCuts:
        if run not in cuts:
            cuts[run] = _RunCuts()
        return cuts[run]

    for edit in edits:
        old = textmap.slice(edit.span)
        for kind, lo, hi, inserted in _diff_operations(old, edit.new_text):
            if kind in ("delete", "replace"):
                for atom in atoms[edit.span.start + lo : edit.span.start + hi]:
                    if atom.kind is AtomKind.PARAGRAPH_BREAK:
                        paragraph_marks.append((atom.paragraph, "del", edit.author))
                    elif atom.kind is AtomKind.STRUCTURAL:
                        cuts_for(atom.run).delete_whole_run = True
                    else:
                        cuts_for(atom.run).deletions.append(
                            (atom.offset, atom.offset + 1)
                        )
            if kind in ("insert", "replace") and inserted:
                anchor = _anchor_for_insertion(atoms, edit.span.start + lo)
                if anchor is None:
                    logger.warning(
                        "[surgery] nowhere to attach inserted text %r", inserted[:40]
                    )
                    continue
                run, offset = anchor
                head, *tails = inserted.split("\n")
                if head:
                    cuts_for(run).insertions.append((offset, head))
                if tails:
                    # Each newline starts a paragraph. The anchor run has to be
                    # cut at this offset even when nothing else happens there,
                    # so the split has a node to sit after.
                    cuts_for(run).split_points.add(offset)
                    splits.append((run, offset, tails, edit.author))
    return cuts, paragraph_marks, splits


def _merge_ranges(ranges: Iterable[tuple[int, int]]) -> list[tuple[int, int]]:
    """Collapse adjacent and overlapping ranges so each becomes one mark."""
    ordered = sorted(ranges)
    merged: list[tuple[int, int]] = []
    for low, high in ordered:
        if merged and low <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], high))
        else:
            merged.append((low, high))
    return merged


def _rewrite_run(
    run: Any, plan: _RunCuts, *, author: str, cid: int
) -> tuple[int, dict[int, Any]]:
    """Replace *run* in its parent with the nodes its cuts describe.

    Returns the next free change id and a map from run-local offset to the last
    node that ends there — the address a paragraph split uses to find the
    element it should follow.
    """
    parent = run.getparent()
    at = parent.index(run)
    rpr = run.find(qn("rPr"))
    ends_at: dict[int, Any] = {}

    if plan.delete_whole_run:
        replacement = [_deletion_of_run(run, author=author, cid=cid)]
        cid += 1
    else:
        text_element = run.find(qn("t"))
        text = (text_element.text or "") if text_element is not None else ""
        deletions = _merge_ranges(plan.deletions)
        insertions = sorted(plan.insertions)

        replacement = []
        cursor = 0
        # Walk the run left to right, emitting kept text, deletions and
        # insertions in the order they occur.
        boundaries = sorted(
            {0, len(text)}
            | plan.split_points
            | {low for low, _ in deletions}
            | {high for _, high in deletions}
            | {position for position, _ in insertions}
        )
        for boundary in boundaries:
            if boundary > cursor:
                chunk = text[cursor:boundary]
                if chunk:
                    covering = next(
                        (d for d in deletions if d[0] <= cursor and boundary <= d[1]),
                        None,
                    )
                    if covering is not None:
                        replacement.append(
                            _deletion_of_text(chunk, rpr, author=author, cid=cid)
                        )
                        cid += 1
                    else:
                        replacement.append(_text_run(chunk, rpr))
                    ends_at[boundary] = replacement[-1]
                cursor = boundary
            for position, inserted in insertions:
                if position == boundary:
                    replacement.append(
                        _insertion(inserted, rpr, author=author, cid=cid)
                    )
                    cid += 1
                    # Last writer wins: a split at this offset must follow the
                    # newly inserted text, not the kept text before it.
                    ends_at[boundary] = replacement[-1]

    for offset, node in enumerate(replacement):
        parent.insert(at + offset, node)
    parent.remove(run)
    return cid, ends_at


def split_run_at(run: Any, offset: int) -> tuple[Any, Any]:
    """Cut *run* in two at *offset*, returning ``(head, tail)``.

    The head is the original element, truncated — so any address already held
    against it stays valid for offsets before the cut. The tail is a new run
    with the same formatting, inserted immediately after.
    """
    element = run.find(qn("t"))
    text = (element.text or "") if element is not None else ""
    if element is None or offset <= 0 or offset >= len(text):
        return run, run

    rpr = run.find(qn("rPr"))
    tail = _text_run(text[offset:], rpr)
    element.text = text[:offset]
    element.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
    run.addnext(tail)
    return run, tail


def isolate_span(atoms: Sequence[Atom], span: AtomSpan) -> tuple[Any, Any] | None:
    """Split boundary runs so *span* occupies whole runs; return its ends.

    A comment brackets runs, not characters, so a range over part of a run
    would either overshoot or — when two comments share one run — nest inside
    another comment's markers. Splitting first gives each range its own runs.

    The right-hand cut is made before the left so that, when both fall in the
    same run, the left cut still sees the offsets it was given.
    """
    covered = [a for a in atoms[span.start : span.end] if a.run is not None]
    if not covered:
        return None
    first, last = covered[0], covered[-1]

    end_run = last.run
    if last.kind is AtomKind.TEXT:
        end_run, _ = split_run_at(last.run, last.offset + 1)
    start_run = first.run
    if first.kind is AtomKind.TEXT:
        _, start_run = split_run_at(first.run, first.offset)
        # When both ends were in one run, the second cut left the original
        # element holding the prefix; the span's own text moved to the tail.
        if end_run is first.run:
            end_run = start_run
    return start_run, end_run


def _split_paragraph_after(node: Any, *, author: str, cid: int) -> Any:
    """Start a new paragraph immediately after *node*.

    Everything after *node* moves into a fresh paragraph carrying the same
    properties, and the original paragraph's mark is marked inserted — Word's
    representation of "a paragraph break was added here".
    """
    p_elem = node.getparent()
    while p_elem is not None and p_elem.tag != qn("p"):
        node, p_elem = p_elem, p_elem.getparent()
    if p_elem is None:
        return None

    new_paragraph = etree.Element(qn("p"))
    ppr = p_elem.find(qn("pPr"))
    if ppr is not None:
        new_paragraph.append(copy.deepcopy(ppr))

    index = p_elem.index(node)
    for sibling in list(p_elem)[index + 1 :]:
        p_elem.remove(sibling)
        new_paragraph.append(sibling)
    p_elem.addnext(new_paragraph)
    _mark_paragraph_mark(p_elem, "ins", author=author, cid=cid)
    return new_paragraph


def _first_change_id(textmap: TextMap) -> int:
    """Where this batch of changes starts numbering, read off the document."""
    for atom in textmap.atoms:
        if atom.paragraph is not None:
            return next_change_id(atom.paragraph.getroottree().getroot())
    return FIRST_CHANGE_ID


def apply_edits(
    textmap: TextMap, edits: Sequence[Edit], *, first_id: int | None = None
) -> EditReport:
    """Apply every edit in *edits*, which must have been resolved against *textmap*.

    Overlapping edits are refused rather than composed: applied one at a time
    against running text, two edits over one sentence could both succeed and
    leave it saying something neither reviewer proposed.

    *first_id* defaults to :func:`next_change_id` over the document the textmap
    addresses, rather than to a constant. A constant is what let a second round
    reuse the first round's revision ids; deriving it here means no caller can
    forget to.
    """
    report = EditReport()
    accepted: list[Edit] = []
    for edit in edits:
        clash = next(
            (
                other
                for other in accepted
                if other.span.start < edit.span.end
                and edit.span.start < other.span.end
            ),
            None,
        )
        if clash is None:
            accepted.append(edit)
        else:
            report.conflicted.append((edit, clash))
            logger.warning(
                "[surgery] edit DROPPED — its span overlaps one already accepted.\n"
                "         dropped: %r\n         kept   : %r",
                textmap.slice(edit.span)[:70],
                textmap.slice(clash.span)[:70],
            )
    report.applied = len(accepted)
    if not accepted:
        return report

    cuts, paragraph_marks, splits = _compile(textmap, accepted)

    cid = _first_change_id(textmap) if first_id is None else first_id
    written: dict[Any, dict[int, Any]] = {}
    for run, plan in cuts.items():
        cid, ends_at = _rewrite_run(run, plan, author=accepted[0].author, cid=cid)
        written[run] = ends_at

    for p_elem, kind, author in paragraph_marks:
        _mark_paragraph_mark(p_elem, kind, author=author, cid=cid)
        cid += 1
        report.paragraphs_merged += 1

    # Splits last: they move nodes between paragraphs, so every run rewrite has
    # to be finished before the tree is reshaped. Reversed so an earlier split
    # cannot move the reference node of a later one out from under it.
    for run, offset, tails, author in reversed(splits):
        reference = written.get(run, {}).get(offset)
        if reference is None:
            logger.warning("[surgery] no anchor for a paragraph split; skipped")
            continue
        for tail in tails:
            new_paragraph = _split_paragraph_after(reference, author=author, cid=cid)
            cid += 1
            if new_paragraph is None:
                break
            report.paragraphs_created += 1
            reference = _insertion(tail, None, author=author, cid=cid)
            cid += 1
            new_paragraph.insert(
                0 if new_paragraph.find(qn("pPr")) is None else 1, reference
            )
    return report
