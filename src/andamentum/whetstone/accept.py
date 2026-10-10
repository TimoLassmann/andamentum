"""Apply the edits a reader accepted to the markdown they were made on.

A review of a markdown note addresses the projection of a generated ``.docx``
(see :mod:`.reviewed_document`), not the markdown: heading marks, block breaks
and HTML entities differ between the two. Writing the edits through Word, as
:func:`~.reviewed_document.carry_forward_edits` does, hands back that projection
and loses the markdown. So this applies each edit to the markdown itself: it
finds ``original_text`` there and replaces it.

The projection keeps a block's inline markup as written (``**bold**``, links),
so most spans are found verbatim. A span that is not (it crosses a heading mark
or a decoded entity) is not applied, and the result says so. Every edit given
is in exactly one of ``applied`` and ``not_applied``; nothing is dropped.
"""

from __future__ import annotations

from collections.abc import Sequence

from pydantic import BaseModel, Field

from .schemas import Edit


class NotApplied(BaseModel):
    """An accepted edit that could not be applied, and why."""

    id: str
    reason: str


class AcceptedEdits(BaseModel):
    """What :func:`accept_edits` returns."""

    text: str  # the markdown with the applied edits
    applied: list[str] = Field(default_factory=list)  # edit ids, in input order
    not_applied: list[NotApplied] = Field(default_factory=list)


def accept_edits(markdown: str, edits: Sequence[Edit]) -> AcceptedEdits:
    """Apply *edits* to *markdown*; report each one that could not be applied.

    Each edit's ``original_text`` is looked up in *markdown*. When it occurs
    more than once, the occurrence nearest the edit's own ``char_start`` is
    taken: the projection the offset was measured in differs from the markdown
    only by a few characters per block. Two edits whose spans overlap cannot
    both apply; the one given first wins. Raises ``ValueError`` when two edits
    share an id, because the report could not then name each one.
    """
    ids = [edit.id for edit in edits]
    if len(set(ids)) != len(ids):
        raise ValueError("two edits share an id; each edit must have its own id")

    kept: list[tuple[int, int, Edit]] = []
    not_applied: list[NotApplied] = []
    for edit in edits:
        span = _locate(markdown, edit)
        if isinstance(span, str):
            not_applied.append(NotApplied(id=edit.id, reason=span))
            continue
        start, end = span
        clash = next((other for s, e, other in kept if start < e and s < end), None)
        if clash is not None:
            not_applied.append(
                NotApplied(id=edit.id, reason=f"it overlaps edit {clash.id}, which was applied")
            )
            continue
        kept.append((start, end, edit))

    text = markdown
    for start, end, edit in sorted(kept, key=lambda k: k[0], reverse=True):
        text = text[:start] + edit.new_text + text[end:]
    applied = {edit.id for _, _, edit in kept}
    return AcceptedEdits(
        text=text,
        applied=[i for i in ids if i in applied],
        not_applied=not_applied,
    )


def _locate(markdown: str, edit: Edit) -> tuple[int, int] | str:
    """The span of *edit* in *markdown*, or the reason there is none."""
    needle = edit.original_text
    if not needle:
        return "it names no text to replace"
    starts: list[int] = []
    at = markdown.find(needle)
    while at != -1:
        starts.append(at)
        at = markdown.find(needle, at + 1)
    if not starts:
        return "its text is not in the note as written (it may cross markdown formatting)"
    start = min(starts, key=lambda s: abs(s - edit.char_start))
    return start, start + len(needle)
