"""Locate text in a document, once, and get back an address.

One index over one atom stream, shared by everything that needs to find text:
the edit resolver and the comment anchorer both call :meth:`TextMap.find`, and
both get an :class:`AtomSpan` back. Previously the two used different matchers
over different representations — edits a raw case-sensitive substring test on a
lossy string, comments a normalised index over runs — so a target could resolve
for one and miss for the other. On a real review that cost 16 of 27 edits.

Normalisation is not implemented here. It is
:func:`andamentum.core.text_match.normalize_for_match`, the repo's single
answer to "is this text verbatim in that text", applied to the projection and
to the target alike. Because one atom contributes exactly one character to the
projection, a match maps straight back to atom indices with no second
coordinate system in between.

Ambiguity is a refusal, not a coin toss: a target occurring twice identifies no
single passage, and anchoring to the wrong one costs the reader's trust in
every other annotation in the document.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass
from typing import Sequence

from andamentum.core.text_match import normalize_for_match

from .atoms import Atom, projection

# Markup a markdown converter invents that Word never stored as text: a list
# marker at the start of a line (Word generates it from the list style) and the
# pipes delimiting a table row (Word holds each cell as its own paragraph).
_LIST_MARKER = re.compile(r"^\s*(?:\d+[.)]|[-*•])\s+")


def strip_converter_decoration(target: str) -> str:
    """Drop markup that provably exists on one side only.

    Only reachable from the ``apply-patches`` path now — a pattern supplied by
    an outside agent may well have come from a converted rendering. The review
    path never sees a converter's output and never searches at all.

    Not a fuzzy relaxation — the retry is still an exact match and still cannot
    land on text that is not there. Pipes become spaces rather than vanishing,
    because ``| Validation | 94% |`` against two Word cell paragraphs joined by
    a space would otherwise become ``Validation94%`` and match nothing.
    """
    return _LIST_MARKER.sub("", target).replace("|", " ")


@dataclass(frozen=True)
class AtomSpan:
    """A half-open range of atom indices: ``[start, end)``."""

    start: int
    end: int

    def __len__(self) -> int:
        return self.end - self.start


class TextMap:
    """A searchable view of an atom stream."""

    def __init__(self, stream: Sequence[Atom]) -> None:
        self._atoms: list[Atom] = list(stream)
        self._text = projection(self._atoms)
        self._normalized, self._index_map = normalize_for_match(self._text)

    @property
    def atoms(self) -> list[Atom]:
        return self._atoms

    @property
    def text(self) -> str:
        """The document as a reader sees it, with changes accepted."""
        return self._text

    @property
    def normalized(self) -> str:
        return self._normalized

    def slice(self, span: AtomSpan) -> str:
        return "".join(a.char for a in self._atoms[span.start : span.end])

    def find_all(self, target: str, *, limit: int = 8) -> list[AtomSpan]:
        """Every place *target* occurs, up to *limit*.

        Bounded because callers only ever need "one, or more than one", and a
        short target can occur hundreds of times.
        """
        normalized_target, _ = normalize_for_match(target)
        normalized_target = normalized_target.strip()
        if not normalized_target:
            return []

        spans: list[AtomSpan] = []
        width = len(normalized_target)
        position = self._normalized.find(normalized_target)
        while position >= 0 and len(spans) < limit:
            start = self._index_map[position]
            end = self._index_map[position + width - 1] + 1
            spans.append(AtomSpan(start, end))
            position = self._normalized.find(normalized_target, position + 1)
        return spans

    def find(self, target: str) -> AtomSpan | None:
        """Locate *target* — only if it occurs exactly once.

        Tried as given, then with converter decoration removed. Returns None
        for "not there" and for "there more than once"; callers that can
        disambiguate should use :meth:`find_all`.
        """
        spans = self.find_all(target, limit=2)
        if len(spans) == 1:
            return spans[0]
        if len(spans) > 1:
            return None  # ambiguous as written; stripping can only widen it
        undecorated = strip_converter_decoration(target)
        if undecorated == target:
            return None
        spans = self.find_all(undecorated, limit=2)
        return spans[0] if len(spans) == 1 else None

    def closest(self, target: str, *, window_pad: int = 30) -> tuple[float, str]:
        """Best-effort nearest region, for diagnostics when :meth:`find` fails.

        Reporting only — never used to place anything.
        """
        normalized_target, _ = normalize_for_match(target)
        normalized_target = normalized_target.strip()
        if not normalized_target or not self._normalized:
            return 0.0, ""
        matcher = difflib.SequenceMatcher(
            None, normalized_target, self._normalized, autojunk=False
        )
        block = matcher.find_longest_match(
            0, len(normalized_target), 0, len(self._normalized)
        )
        if block.size == 0:
            return 0.0, ""
        # Score the best LOCAL window: a whole-document ratio is meaninglessly
        # tiny for a long target even when its region matches well.
        width = len(normalized_target)
        window_start = max(0, block.b - block.a)
        window = self._normalized[window_start : window_start + width]
        score = difflib.SequenceMatcher(
            None, normalized_target, window, autojunk=False
        ).ratio()
        low = max(0, block.b - window_pad)
        high = min(len(self._normalized), block.b + block.size + window_pad)
        return score, self._normalized[low:high]
