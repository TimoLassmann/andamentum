"""From a rewritten paragraph to span operations.

The model returns the whole paragraph; code diffs it against the current text
at word level and emits the smallest ``Replace`` / ``Insert`` / ``Delete``
operations, so a one-word change marks one word and everything else stays
byte-identical. A rewrite that would touch a citation field, add a citation,
add a line break or change nothing is refused here rather than half-applied by
the writer. A work enters a document only through ``insert-citation``, which
verifies it; a rewrite that types a new "(Smith, 2019)" or "[7]" would bypass that.
"""

from __future__ import annotations

import re
from collections import Counter
from difflib import SequenceMatcher

from ..kernel import Delete, FieldToken, Insert, Paragraph, Replace
from ..kernel.csl import TYPED_CITATION

_TOKEN = re.compile(r"\s+|\w+|[^\w\s]")

SpanOp = Replace | Insert | Delete


class EditRefused(Exception):
    """The proposed paragraph cannot be turned into safe span operations."""


def span_ops(
    paragraph: Paragraph, new_text: str, *, fields: list[FieldToken]
) -> list[SpanOp]:
    """Operations that turn ``paragraph.text`` into *new_text*."""
    old = paragraph.text
    new = new_text.strip()
    if "\n" in new or "\r" in new:
        raise EditRefused("the rewrite spans more than one paragraph")
    if not old.strip():
        raise EditRefused("the paragraph has no text to attach a change to")
    if not new:
        raise EditRefused("the rewrite is empty")
    if new == old:
        raise EditRefused("the rewrite changes nothing")
    before, after = _TOKEN.findall(old), _TOKEN.findall(new)
    starts = [0]
    for token in before:
        starts.append(starts[-1] + len(token))
    ops: list[SpanOp] = []
    matcher = SequenceMatcher(None, before, after, autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        start, end = starts[i1], starts[i2]
        text = "".join(after[j1:j2])
        if tag == "replace":
            ops.append(
                Replace(
                    paragraph_id=paragraph.id,
                    start=start,
                    end=end,
                    text=text,
                    expect=old[start:end],
                )
            )
        elif tag == "delete":
            ops.append(
                Delete(
                    paragraph_id=paragraph.id,
                    start=start,
                    end=end,
                    expect=old[start:end],
                )
            )
        elif tag == "insert":
            ops.append(Insert(paragraph_id=paragraph.id, at=start, text=text))
    for token in fields:
        if token.paragraph_id != paragraph.id:
            continue
        for op in ops:
            if isinstance(op, Insert):
                touches = token.start < op.at < token.end
            else:
                touches = op.start < token.end and op.end > token.start
            if touches:
                raise EditRefused(
                    f"the rewrite would change the field {token.display!r}"
                )
    added = Counter(TYPED_CITATION.findall(new)) - Counter(TYPED_CITATION.findall(old))
    if added:
        raise EditRefused(
            f"the rewrite adds a citation ({next(iter(added))!r}); "
            "a work is cited with insert-citation, which verifies it"
        )
    return ops
