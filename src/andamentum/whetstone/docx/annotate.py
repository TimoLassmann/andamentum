"""Annotate a document at coordinates that are already its own.

Every other way of putting a reviewer's remark into a Word file has to answer
"where is this text?" first. This one does not, because the reviewer read the
document's own projection — :func:`~.atoms.projection` of its paragraphs — and
one atom is one character, so an offset into what the reviewer read IS an atom
index. There is no matcher here, no normalisation, no similarity threshold and
no nearest-match fallback: the caller either hands over addresses in this
document or it hands over nothing.

That is not a refinement of the search, it is the removal of it. Searching for
a quote in a converted rendering of the document lost 15 of 43 rewrites on one
real manuscript — every one of them reporting a nearest match at 0.97 or
better, because the converter wrote ``aim 3,4 .`` where Word held
``aim3,4.``. A 0.97 near-miss is not a matcher to tune; it is a coordinate
system to delete.

Two guarantees, both loud:

*The document must be the one that was reviewed.* The projection is compared
against the text the caller says it annotated. A mismatch raises
:class:`CoordinateError` rather than degrading into a search — a silent
fallback would be the deleted coordinate system growing back.

*An offset must be an offset in it.* Out-of-range spans raise too. A dropped
annotation is a thinner review; a misplaced one is a reviewer pointing at a
sentence the author did not write.

Order of operations, and why it is the reverse of :mod:`.patch_editor`'s.
Comments are bracketed first, rewrites applied second:

1. Runs are split at every note boundary. Splitting a run does not change a
   single character of the projection, so all the addresses still hold — the
   rebuilt stream is checked against the same text to keep that a fact rather
   than a belief.
2. Each note takes the runs its span now exactly covers.
3. :func:`~.surgery.apply_edits` runs last, because it REMOVES the runs an edit
   covers (:func:`~.surgery._rewrite_run`) and a comment already bracketing one
   of those would be left pointing at a detached element.

Which is also why a note overlapping a rewrite is dropped: the rewrite is about
to consume its anchor. The test is one integer comparison, and it reaches one
character to the left of the rewrite, because
:func:`~.surgery._anchor_for_insertion` attaches inserted text to the run of
the character BEFORE the span — so a note ending exactly where a rewrite begins
shares a run with it after all.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Sequence

from .atoms import Atom, document_atoms, projection
from .exceptions import DocumentReviewError
from .low_level import DocxEditor, PlacedComment
from .surgery import (
    Edit,
    apply_edits,
    run_bounds,
    split_at_boundaries,
    translate_span,
)
from .textmap import AtomSpan, TextMap

logger = logging.getLogger("andamentum.whetstone")


class CoordinateError(DocumentReviewError):
    """An annotation's offsets are not addresses in this document."""


@dataclass(frozen=True)
class Rewrite:
    """Replace ``[start, end)`` of the reviewed text with *new_text*."""

    start: int
    end: int
    new_text: str
    author: str
    label: str = ""  # what to call it in a log line or a warning


@dataclass(frozen=True)
class Note:
    """Attach a comment to ``[start, end)`` of the reviewed text."""

    start: int
    end: int
    body: str
    author: str
    label: str = ""


@dataclass
class AnnotationReport:
    """What reached the document, and what did not.

    ``shadowed_notes`` and ``conflicted_rewrites`` are the only two ways an
    annotation can fail to arrive, and both are decisions this module made on
    purpose. "Could not find the text" is not on the list any more.
    """

    total: int = 0
    applied_rewrites: int = 0
    applied_notes: int = 0
    conflicted_rewrites: list[Rewrite] = field(default_factory=list)
    shadowed_notes: list[Note] = field(default_factory=list)
    # Notes whose span a rewrite shifted or partly replaced. They ARE placed —
    # counted in applied_notes — and listed separately only so a caller can say
    # the comment now covers revised text.
    moved_notes: list[Note] = field(default_factory=list)
    paragraphs_merged: int = 0
    paragraphs_created: int = 0
    processing_time: float = 0.0

    @property
    def applied(self) -> int:
        return self.applied_rewrites + self.applied_notes


def apply_annotations(
    editor: DocxEditor,
    *,
    rewrites: Sequence[Rewrite],
    notes: Sequence[Note],
    reviewed_text: str,
) -> AnnotationReport:
    """Write *rewrites* and *notes* into *editor*'s document at their offsets.

    *reviewed_text* is the text the offsets were taken against. It is not a
    hint — it is checked, and a document that does not project to exactly that
    string is refused.
    """
    started = time.time()
    report = AnnotationReport(total=len(rewrites) + len(notes))

    atoms = document_atoms(editor.paragraph_elements)
    _require_projection(projection(atoms), reviewed_text)
    for annotation in (*rewrites, *notes):
        _require_addressable(annotation, len(atoms))

    # Rewrites first. They replace runs, so a comment bracketing one would be
    # pointing at a detached element — and placing comments first meant a note
    # over a rewritten passage had to be dropped or moved onto the insertion,
    # which pinned a sentence-level finding to a two-word change.
    applied: list[Edit] = []
    if rewrites:
        edits = [
            Edit(
                span=AtomSpan(rewrite.start, rewrite.end),
                new_text=rewrite.new_text,
                author=rewrite.author,
            )
            for rewrite in rewrites
        ]
        origin = {id(edit): rewrite for edit, rewrite in zip(edits, rewrites)}
        edit_report = apply_edits(TextMap(atoms), edits)
        report.applied_rewrites = edit_report.applied
        report.conflicted_rewrites = [
            origin[id(loser)] for loser, _winner in edit_report.conflicted
        ]
        report.paragraphs_merged = edit_report.paragraphs_merged
        report.paragraphs_created = edit_report.paragraphs_created
        applied = edit_report.applied_edits

    if notes:
        # Every note's span, in the coordinates the document now has. Exact
        # arithmetic over the applied rewrites — see `translate_span`. A note
        # whose passage merely CONTAINS a rewrite still covers its whole
        # passage, which is what the reviewer was talking about.
        atoms = document_atoms(editor.paragraph_elements)
        spans = {
            id(note): translate_span(AtomSpan(note.start, note.end), applied)
            for note in notes
        }
        split_at_boundaries(atoms, list(spans.values()))
        # Splitting a run moves no characters, so the spans still hold. The
        # atoms do not: a cut puts characters in a new element.
        atoms = document_atoms(editor.paragraph_elements)

        for note in notes:
            span = spans[id(note)]
            bounds = run_bounds(atoms, span) if len(span) else None
            if bounds is None:
                # Nothing survives for it to bracket: a span holding only a
                # paragraph break, or one a rewrite emptied by deleting the
                # passage outright.
                report.shadowed_notes.append(note)
                logger.info(
                    "[annotate] note not placed — nothing left to anchor to: %r",
                    note.label or note.body[:60],
                )
                continue
            start_run, end_run = bounds
            editor.placed_comments.append(
                PlacedComment(
                    author=note.author,
                    text=note.body,
                    start_run=start_run,
                    end_run=end_run,
                )
            )
            report.applied_notes += 1
            if span != AtomSpan(note.start, note.end):
                report.moved_notes.append(note)

    report.processing_time = time.time() - started
    logger.info(
        "[annotate] %d/%d applied (%d rewrite(s), %d comment(s), %d of them "
        "moved onto a rewrite's new text); %d note(s) with nothing to anchor "
        "to, %d rewrite(s) dropped as overlapping",
        report.applied,
        report.total,
        report.applied_rewrites,
        report.applied_notes,
        len(report.moved_notes),
        len(report.shadowed_notes),
        len(report.conflicted_rewrites),
    )
    return report


# ── the two refusals ────────────────────────────────────────────────────


def _require_projection(actual: str, expected: str, *, why: str = "") -> None:
    if actual == expected:
        return
    at = next(
        (i for i, (a, b) in enumerate(zip(actual, expected)) if a != b),
        min(len(actual), len(expected)),
    )
    context = " after " + why if why else ""
    raise CoordinateError(
        f"the document does not read as the text these offsets were taken "
        f"against{context}. Every annotation is an address in that text, so "
        f"there is nothing safe to do with it here — searching for the quote "
        f"instead is the failure mode this path exists to remove.\n"
        f"  first difference at character {at}\n"
        f"  document : {actual[max(0, at - 30) : at + 30]!r}\n"
        f"  reviewed : {expected[max(0, at - 30) : at + 30]!r}\n"
        f"  lengths  : {len(actual)} vs {len(expected)}"
    )


def _require_addressable(annotation: Rewrite | Note, length: int) -> None:
    if 0 <= annotation.start < annotation.end <= length:
        return
    raise CoordinateError(
        f"annotation span [{annotation.start}, {annotation.end}) is not a "
        f"range in a document of {length} characters "
        f"({annotation.label or 'unlabelled'})"
    )


# ── shadowing, boundaries, bracketing ───────────────────────────────────


def _slice(atoms: Sequence[Atom], start: int, end: int) -> str:
    return "".join(atom.char for atom in atoms[start:end])
