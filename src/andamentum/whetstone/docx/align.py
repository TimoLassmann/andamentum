"""Map every character of the converted text back to the document it came from.

The reviewer reads markdown produced by a converter; the annotations land in a
.docx. The two say the same thing in different notation — Docling writes
``aim 3,4 .`` where Word holds ``aim3,4.``, adds ``*`` around italics,
and invents a ``|---|---|`` row for a table. Searching for a quote from one
inside the other therefore fails on text that is plainly there: on a real
manuscript, 15 of 43 rewrites were lost that way, every one of them reporting a
nearest match at 0.97 similarity or better.

Nothing needs searching. The pipeline already knows where each quote sits in
the markdown — ``Quote`` and ``Edit`` carry character offsets. The only missing
piece is where each markdown character came from, and that is one alignment.

Position also settles what a score never can. A sentence appearing in both the
abstract and the introduction is two different places, and a search has no way
to choose; the first occurrence in the markdown maps to the first in the
document because that is what the alignment says.

**Anchored, not brute force.** A whole-text edit distance is O(nm/w) — 2.5
seconds for a thesis-length document. But the two strings are ~99% identical:
a converter changes notation, it does not rewrite prose. So most of the text
already matches exactly. Phrases occurring exactly once in both strings are
unambiguous seeds; keeping the colinear subset and running the edit distance
only between consecutive seeds turns the cost linear — measured 58x faster at
273k characters, producing an identical map.

**Verify what you map.** Where a converter MOVES a block, the seeds for it fall
out of order, are dropped from the chain, and the region maps somewhere
plausible and wrong. So a mapped span is a hypothesis: callers must check the
document text against the quote before using it. :func:`map_span` returns the
span; deciding whether to trust it belongs to the caller that holds the quote.
"""

from __future__ import annotations

import bisect
import re
from dataclasses import dataclass
from typing import Optional, Sequence

from rapidfuzz.distance import Levenshtein

_WORD = re.compile(r"\S+")

# Phrase length for a seed, in words. Three is long enough that a match is
# rarely coincidental and short enough that seeds stay dense in ordinary prose.
DEFAULT_SEED_WORDS = 3


@dataclass(frozen=True)
class Seed:
    """A phrase occurring exactly once in each string, at these offsets."""

    source_start: int
    target_start: int
    length: int


def _phrase_positions(text: str, words_per_seed: int) -> dict[str, list[int]]:
    """Every n-word phrase in *text*, mapped to the offsets it starts at."""
    words = [(m.start(), m.end()) for m in _WORD.finditer(text)]
    positions: dict[str, list[int]] = {}
    for index in range(len(words) - words_per_seed + 1):
        start = words[index][0]
        end = words[index + words_per_seed - 1][1]
        positions.setdefault(text[start:end], []).append(start)
    return positions


def find_seeds(source: str, target: str, *, words_per_seed: int) -> list[Seed]:
    """Phrases that occur exactly once in *source* and exactly once in *target*.

    Uniqueness on both sides is what makes a seed unambiguous: a phrase
    appearing twice anywhere gives no information about which copy is which.
    """
    source_positions = _phrase_positions(source, words_per_seed)
    target_positions = _phrase_positions(target, words_per_seed)
    seeds: list[Seed] = []
    for phrase, starts in source_positions.items():
        if len(starts) != 1:
            continue
        target_starts = target_positions.get(phrase)
        if target_starts is None or len(target_starts) != 1:
            continue
        seeds.append(
            Seed(
                source_start=starts[0],
                target_start=target_starts[0],
                length=len(phrase),
            )
        )
    seeds.sort(key=lambda seed: seed.source_start)
    return seeds


def colinear_chain(seeds: Sequence[Seed]) -> list[Seed]:
    """The longest run of seeds that advances in both strings together.

    Seeds arrive ordered by position in the source. A seed that is unique in
    both but sits in a block the converter moved would pull the alignment
    sideways, so only a subsequence increasing in the target too is kept —
    the classic longest-increasing-subsequence chain, in O(n log n).
    """
    if not seeds:
        return []

    chain_ends: list[int] = []  # smallest target-end achieving each chain length
    chain_tail: list[int] = []  # index of the seed at each chain length
    predecessor: list[int] = []

    for position, seed in enumerate(seeds):
        slot = bisect.bisect_left(chain_ends, seed.target_start)
        predecessor.append(chain_tail[slot - 1] if slot else -1)
        end = seed.target_start + seed.length
        if slot == len(chain_ends):
            chain_ends.append(end)
            chain_tail.append(position)
        elif end < chain_ends[slot]:
            chain_ends[slot] = end
            chain_tail[slot] = position

    chain: list[Seed] = []
    cursor = chain_tail[-1]
    while cursor >= 0:
        chain.append(seeds[cursor])
        cursor = predecessor[cursor]
    chain.reverse()

    # Discard residual overlap so consecutive seeds bound a well-formed gap.
    kept: list[Seed] = []
    for seed in chain:
        if kept:
            previous = kept[-1]
            if (
                seed.source_start < previous.source_start + previous.length
                or seed.target_start < previous.target_start + previous.length
            ):
                continue
        kept.append(seed)
    return kept


class OffsetMap:
    """Where each character of *source* sits in *target*, or nowhere.

    ``None`` at a position means the converter invented that character — a
    ``#``, a ``*``, a table pipe, a space it added around a superscript. Those
    correctly map nowhere, because they exist nowhere in the document.
    """

    def __init__(
        self, source: str, target: str, *, words_per_seed: int = DEFAULT_SEED_WORDS
    ) -> None:
        self._source = source
        self._target = target
        self._to_target: list[Optional[int]] = [None] * len(source)
        self._seeds_used = 0
        self._build(words_per_seed)

    def _build(self, words_per_seed: int) -> None:
        chain = colinear_chain(find_seeds(self._source, self._target, words_per_seed=words_per_seed))
        self._seeds_used = len(chain)

        source_cursor = target_cursor = 0
        for seed in chain:
            self._align_gap(
                source_cursor, seed.source_start, target_cursor, seed.target_start
            )
            self._record_equal(seed.source_start, seed.target_start, seed.length)
            source_cursor = seed.source_start + seed.length
            target_cursor = seed.target_start + seed.length
        self._align_gap(
            source_cursor, len(self._source), target_cursor, len(self._target)
        )

    def _record_equal(self, source_start: int, target_start: int, length: int) -> None:
        for step in range(length):
            self._to_target[source_start + step] = target_start + step

    def _align_gap(
        self, source_lo: int, source_hi: int, target_lo: int, target_hi: int
    ) -> None:
        """Edit distance over one gap only — bounded by the seeds either side."""
        if source_hi <= source_lo or target_hi <= target_lo:
            return
        opcodes = Levenshtein.opcodes(
            self._source[source_lo:source_hi], self._target[target_lo:target_hi]
        )
        for opcode in opcodes:
            if opcode.tag != "equal":
                continue
            source_from = int(opcode.src_start)
            source_to = int(opcode.src_end)
            target_from = int(opcode.dest_start)
            self._record_equal(
                source_lo + source_from,
                target_lo + target_from,
                source_to - source_from,
            )

    @property
    def seeds_used(self) -> int:
        return self._seeds_used

    @property
    def coverage(self) -> float:
        """Fraction of source characters that exist in the target."""
        if not self._to_target:
            return 0.0
        mapped = sum(1 for offset in self._to_target if offset is not None)
        return mapped / len(self._to_target)

    def map_span(self, start: int, end: int) -> Optional[tuple[int, int]]:
        """The target span a source span covers, or None if it maps nowhere.

        Characters the converter invented are skipped rather than blocking the
        span: a quote wrapped in ``*`` still identifies the prose between them.
        The result is a hypothesis — see this module's docstring on moved
        blocks — so callers must verify it against the text they expected.

        Whitespace at the edges is trimmed first. Where an alignment is not
        unique it is always whitespace that is ambiguous — the converter's
        ``\\n\\n`` against the document's single break gives two candidates for
        one slot, and two optimal alignments can pick different ones. Trimming
        makes the span independent of that choice instead of merely usually
        unaffected by it.
        """
        start = max(0, start)
        end = min(end, len(self._to_target))
        while start < end and self._source[start].isspace():
            start += 1
        while end > start and self._source[end - 1].isspace():
            end -= 1

        present = [
            offset for offset in self._to_target[start:end] if offset is not None
        ]
        if not present:
            return None
        return min(present), max(present) + 1
