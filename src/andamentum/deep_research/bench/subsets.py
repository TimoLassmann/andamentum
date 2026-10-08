"""Deterministic subsets: the same ids give the same sample on every machine.

An item is ranked by ``sha256(seed + id)``, so a subset depends on the ids and
the seed only, never on the order a file lists them in, and a larger ``n``
extends a smaller one (the first 30 of the 100 are the 30).
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence


def _rank(item_id: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{item_id}".encode()).hexdigest()


def select_subset(item_ids: Sequence[str], n: int, *, seed: int) -> list[str]:
    """The *n* ids with the lowest hash rank, in rank order.

    Raises ``ValueError`` for a duplicate id or a negative *n*; asking for more
    than there are returns all of them.
    """
    if n < 0:
        raise ValueError("n must not be negative")
    if len(set(item_ids)) != len(item_ids):
        raise ValueError("item ids must be unique")
    return sorted(item_ids, key=lambda i: _rank(i, seed))[:n]
