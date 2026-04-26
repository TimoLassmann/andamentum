"""Utility functions for sequence alignment."""

from __future__ import annotations


def hamming_distance(a: str, b: str) -> int:
    """Number of positions at which the two strings differ.

    Both inputs must be the same length; raises ValueError otherwise.
    """
    if len(a) != len(b):
        raise ValueError(f"length mismatch: {len(a)} vs {len(b)}")
    return sum(1 for x, y in zip(a, b) if x != y)


def levenshtein(a: str, b: str) -> int:
    """Edit distance between two strings.

    Uses the standard dynamic programming algorithm. O(len(a) * len(b)).
    """
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        curr = [i]
        for j, cb in enumerate(b, start=1):
            ins = curr[j - 1] + 1
            dele = prev[j] + 1
            sub = prev[j - 1] + (0 if ca == cb else 1)
            curr.append(min(ins, dele, sub))
        prev = curr
    return prev[-1]


class AlignmentScore:
    """Track an alignment's score components."""

    def __init__(self, matches: int = 0, mismatches: int = 0, gaps: int = 0):
        self.matches = matches
        self.mismatches = mismatches
        self.gaps = gaps

    def total(self) -> int:
        """Score = 2 * matches - mismatches - 2 * gaps (toy scoring)."""
        return 2 * self.matches - self.mismatches - 2 * self.gaps
