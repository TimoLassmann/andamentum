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

    # A paragraph mark carries at most one revision of each kind, so a second
    # proposal cannot be added alongside the first. Leave the first alone: it
    # already says the mark is deleted (or inserted), so the outcome is the
    # same, and overwriting it would restamp a co-author's revision with this
    # author's name — the very thing this module exists to stop.
    if rpr.find(qn(kind)) is not None:
        return
    rpr.insert(0, _change(kind, author=author, cid=cid))


# ── the plan ────────────────────────────────────────────────────────────


@dataclass
class _RunCuts:
    """Every change to make inside one run, in run-local character offsets."""

    # Each cut carries the reviewer that asked for it. Holding one author per
    # element was too coarse: two reviewers editing different words of the same
    # ``<w:t>`` had the second one's change stamped with the first one's name.
    deletions: list[tuple[int, int, str]] = field(default_factory=list)
    # (offset, text, author, id of the Edit that asked for it)
    insertions: list[tuple[int, str, str, int]] = field(default_factory=list)
    delete_whole_run: bool = False
    whole_run_author: str = ""
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
    # The ``<w:ins>`` nodes each edit produced, keyed by ``id(edit)``. A
    # comment whose anchor a rewrite consumed can bracket these instead of
    # being dropped — the reader then sees the new wording and the reason for
    # it together. Reported rather than searched for: the nodes are known at
    # the moment they are built, and looking them up afterwards would be the
    # deleted coordinate system growing back.
    insertions: dict[int, list[Any]] = field(default_factory=dict)  # <w:r> runs


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
    """Where to put text inserted at atom *index*: ``(<w:t> element, offset)``.

    Prefers to append after the preceding character so the insertion inherits
    that run's formatting and sits inside the same paragraph.
    """
    previous = atoms[index - 1] if index > 0 else None
    if previous is not None and previous.kind is AtomKind.TEXT:
        return previous.element, previous.offset + 1
    following = atoms[index] if index < len(atoms) else None
    if following is not None and following.kind is AtomKind.TEXT:
        return following.element, following.offset
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

    def cuts_for(element: Any) -> _RunCuts:
        if element not in cuts:
            cuts[element] = _RunCuts()
        return cuts[element]

    for edit in edits:
        old = textmap.slice(edit.span)
        for kind, lo, hi, inserted in _diff_operations(old, edit.new_text):
            if kind in ("delete", "replace"):
                for atom in atoms[edit.span.start + lo : edit.span.start + hi]:
                    if atom.kind is AtomKind.PARAGRAPH_BREAK:
                        paragraph_marks.append((atom.paragraph, "del", edit.author))
                    elif atom.kind is AtomKind.STRUCTURAL:
                        structural = cuts_for(atom.element)
                        structural.delete_whole_run = True
                        structural.whole_run_author = edit.author
                    else:
                        cuts_for(atom.element).deletions.append(
                            (atom.offset, atom.offset + 1, edit.author)
                        )
            if kind in ("insert", "replace") and inserted:
                anchor = _anchor_for_insertion(atoms, edit.span.start + lo)
                if anchor is None:
                    logger.warning(
                        "[surgery] nowhere to attach inserted text %r", inserted[:40]
                    )
                    continue
                element, offset = anchor
                head, *tails = inserted.split("\n")
                if head:
                    cuts_for(element).insertions.append(
                        (offset, head, edit.author, id(edit))
                    )
                if tails:
                    # Each newline starts a paragraph. The anchor run has to be
                    # cut at this offset even when nothing else happens there,
                    # so the split has a node to sit after.
                    cuts_for(element).split_points.add(offset)
                    splits.append((element, offset, tails, edit.author))
    return cuts, paragraph_marks, splits


def _merge_ranges(
    ranges: Iterable[tuple[int, int, str]],
) -> list[tuple[int, int, str]]:
    """Collapse adjacent and overlapping ranges so each becomes one mark.

    Only ranges from the same reviewer merge. Two reviewers deleting adjacent
    words are two revisions, and folding them into one would attribute both to
    whichever came first.
    """
    merged: list[tuple[int, int, str]] = []
    for low, high, author in sorted(ranges):
        if merged and low <= merged[-1][1] and author == merged[-1][2]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], high), author)
        else:
            merged.append((low, high, author))
    return merged


def _rewrite_run(
    run: Any, plans: dict[Any, _RunCuts], produced: dict[int, list[Any]], *, cid: int
) -> tuple[int, dict[Any, dict[int, Any]]]:
    """Rebuild *run* once, applying the cuts for every element inside it.

    A run is not one string. It may hold several ``<w:t>`` children with a
    ``<w:br>``, a ``<w:tab>`` or a ``<w:lastRenderedPageBreak/>`` between them,
    and Word writes the last of those on every save — so a real manuscript is
    full of runs with more than one child.

    Two earlier versions got this wrong in different ways. The first read
    ``run.find(qn("t"))`` and replaced the whole run, deleting every other
    child: measured, one edit turned "The first half<br>the second half." into
    "The initial half", losing a sentence with no tracked deletion and
    reporting success. The second keyed cuts by element but still replaced the
    run per element, so two cut ``<w:t>`` children in one run meant the second
    pass operated on a run the first had already detached.

    Hence one pass per run: walk its children in order, cut the ones that have
    cuts, and carry everything else across untouched. Children with no cuts are
    accumulated so an untouched stretch stays a single run rather than being
    shattered into one run per node.
    """
    parent = run.getparent()
    at = parent.index(run)
    rpr = run.find(qn("rPr"))
    replacement: list[Any] = []
    ends_at: dict[Any, dict[int, Any]] = {}
    carried: list[Any] = []

    def flush() -> Any | None:
        """Emit the accumulated untouched children as one run."""
        nonlocal carried
        if not carried:
            return None
        kept = etree.Element(qn("r"))
        if rpr is not None:
            kept.append(copy.deepcopy(rpr))
        for node in carried:
            kept.append(copy.deepcopy(node))
        carried = []
        replacement.append(kept)
        return kept

    def emit_text(text: str) -> Any:
        """Emit kept text, merged with anything carried before it."""
        nonlocal carried
        kept = etree.Element(qn("r"))
        if rpr is not None:
            kept.append(copy.deepcopy(rpr))
        for node in carried:
            kept.append(copy.deepcopy(node))
        carried = []
        written = etree.SubElement(kept, qn("t"))
        written.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
        written.text = text
        replacement.append(kept)
        return kept

    for child in run:
        if child is rpr:
            continue
        plan = plans.get(child)
        if plan is None:
            carried.append(child)
            continue

        if plan.delete_whole_run:
            flush()
            replacement.append(
                _wrapped_deletion(child, rpr, author=plan.whole_run_author, cid=cid)
            )
            cid += 1
            continue

        text = child.text or ""
        deletions = _merge_ranges(plan.deletions)
        insertions = sorted(plan.insertions)
        boundaries = sorted(
            {0, len(text)}
            | plan.split_points
            | {low for low, _, _ in deletions}
            | {high for _, high, _ in deletions}
            | {offset for offset, _, _, _ in insertions}
        )
        marks: dict[int, Any] = {}
        cursor = 0
        for boundary in boundaries:
            if boundary > cursor:
                chunk = text[cursor:boundary]
                if chunk:
                    covering = next(
                        (d for d in deletions if d[0] <= cursor and boundary <= d[1]),
                        None,
                    )
                    if covering is not None:
                        flush()
                        replacement.append(
                            _deletion_of_text(chunk, rpr, author=covering[2], cid=cid)
                        )
                        cid += 1
                    else:
                        emit_text(chunk)
                    marks[boundary] = replacement[-1]
                cursor = boundary
            for offset, inserted, inserted_by, owner in insertions:
                if offset == boundary:
                    flush()
                    replacement.append(
                        _insertion(inserted, rpr, author=inserted_by, cid=cid)
                    )
                    # The run inside, not the container: a caller bracketing
                    # a comment across it needs runs, and reaching into the
                    # XML for one is exactly the searching this design removed.
                    produced.setdefault(owner, []).extend(
                        replacement[-1].findall(qn("r"))
                    )
                    cid += 1
                    # Last writer wins: a split at this offset must follow the
                    # newly inserted text, not the kept text before it.
                    marks[boundary] = replacement[-1]
        ends_at[child] = marks

    flush()

    for index, node in enumerate(replacement):
        parent.insert(at + index, node)
    parent.remove(run)
    return cid, ends_at


def _wrapped_deletion(element: Any, rpr: Any | None, *, author: str, cid: int) -> Any:
    """Delete one structural child (a break, a tab) as a tracked change."""
    container = _change("del", author=author, cid=cid)
    run = etree.Element(qn("r"))
    if rpr is not None:
        run.append(copy.deepcopy(rpr))
    run.append(copy.deepcopy(element))
    container.append(run)
    return container


def split_run_at(element: Any, offset: int) -> tuple[Any, Any]:
    """Cut the run holding *element* at *offset* within it; return (head, tail).

    The head is the original run, truncated — so any address already held
    against it stays valid for offsets before the cut. Siblings after
    *element* move to the tail, because they follow the cut in reading order.
    """
    text = element.text or ""
    if offset <= 0 or offset >= len(text):
        return element.getparent(), element.getparent()

    run = element.getparent()
    rpr = run.find(qn("rPr"))
    siblings = [child for child in run if child is not rpr]
    after = siblings[siblings.index(element) + 1 :]

    tail = etree.Element(qn("r"))
    if rpr is not None:
        tail.append(copy.deepcopy(rpr))
    written = etree.SubElement(tail, qn("t"))
    written.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
    written.text = text[offset:]
    for node in after:
        run.remove(node)
        tail.append(node)

    element.text = text[:offset]
    element.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
    run.addnext(tail)
    return run, tail


def split_at_boundaries(atoms: Sequence[Atom], spans: Iterable[AtomSpan]) -> None:
    """Cut runs so every span in *spans* starts and ends on a run boundary.

    A comment brackets runs, not characters, so a range covering part of a run
    would either overshoot or — when two comments share a run — nest inside
    another comment's markers.

    Every boundary is cut before any is read back, which is what makes
    overlapping and identical spans safe. The alternative, isolating one span
    at a time, has to rebase the second cut whenever both ends of a span share
    one ``<w:t>``: the first cut moves the text the second was measured
    against. That rebasing was wrong twice — once addressing runs, once
    addressing elements — and splitting everything first makes the case
    unrepresentable rather than merely handled.

    Descending order means cuts inside one run are made right to left, so an
    earlier cut never moves the text a later one is measured against.

    **Re-derive the atom stream afterwards.** A cut moves characters into a new
    element while the atoms built before it still point at the old one, so
    ``run_bounds`` would read the wrong run. The projection is unchanged by a
    split, so the spans themselves stay valid across the rebuild.
    """
    boundaries = sorted(
        {edge for span in spans for edge in (span.start, span.end)}, reverse=True
    )
    for index in boundaries:
        if index >= len(atoms):
            continue
        atom = atoms[index]
        if atom.kind is AtomKind.TEXT and atom.element is not None and atom.offset > 0:
            split_run_at(atom.element, atom.offset)


def run_bounds(
    atoms: Sequence[Atom], span: AtomSpan
) -> tuple[Any, Any] | None:
    """The first and last run *span* covers, or None if it covers none.

    Call after :func:`split_at_boundaries`, which guarantees the span occupies
    whole runs. Read from the element rather than ``atom.run``: splitting a
    boundary moves text into a new run, so the run an atom was built with may
    no longer hold that character, while the element always does.
    """
    covered = [
        atom for atom in atoms[span.start : span.end] if atom.element is not None
    ]
    if not covered:
        return None
    return covered[0].element.getparent(), covered[-1].element.getparent()


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
    # Grouped by run, because a run is rebuilt as a whole: two cut ``<w:t>``
    # children in one run must be handled in the same pass or the second finds
    # a run the first already replaced.
    by_run: dict[Any, dict[Any, _RunCuts]] = {}
    for element, plan in cuts.items():
        by_run.setdefault(element.getparent(), {})[element] = plan

    written: dict[Any, dict[int, Any]] = {}
    for run, plans in by_run.items():
        cid, ends_at = _rewrite_run(run, plans, report.insertions, cid=cid)
        written.update(ends_at)

    for p_elem, kind, author in paragraph_marks:
        _mark_paragraph_mark(p_elem, kind, author=author, cid=cid)
        cid += 1
        report.paragraphs_merged += 1

    # Splits last: they move nodes between paragraphs, so every run rewrite has
    # to be finished before the tree is reshaped. Reversed so an earlier split
    # cannot move the reference node of a later one out from under it.
    for element, offset, tails, author in reversed(splits):
        reference = written.get(element, {}).get(offset)
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
