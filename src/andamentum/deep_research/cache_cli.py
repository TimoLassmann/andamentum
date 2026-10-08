"""``andamentum-research cache info`` and ``cache prune``: look at, and trim, an HTTP cache.

    andamentum-research cache info --cache PATH
    andamentum-research cache prune --cache PATH [--max-bytes N]

``PATH`` is a research store file (the ``--cache`` of ``andamentum-research``). The cache is capped on every write (``store.HTTP_CACHE_MAX_BYTES``);
``cache prune`` applies a cap by hand, for example a smaller one. A missing file is an error
(exit 3) and is never created. The parser lives in ``cli.py``.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

from .store import ResearchStore


def _when(stamp: float | None) -> str:
    if stamp is None:
        return "-"
    return datetime.fromtimestamp(stamp, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def _mb(n: int) -> str:
    return f"{n / (1024 * 1024):.2f} MB"


def _open(path: str, *, max_bytes: int | None = None) -> ResearchStore | None:
    target = Path(path)
    if not target.is_file():
        print(f"error: no cache at {target}", file=sys.stderr)
        return None
    return ResearchStore(target, max_cache_bytes=max_bytes)


def run(args: argparse.Namespace) -> int:
    """Run ``cache info`` or ``cache prune`` from parsed arguments; return the
    exit status (3 when the cache file does not exist)."""
    if args.cache_command == "info":
        opened = _open(args.cache)
        if opened is None:
            return 3
        with opened as store:
            stats = store.cache_stats()
        print(f"cache: {args.cache}")
        print(
            f"size:  {_mb(stats.bytes)} of {_mb(stats.max_bytes)} ({stats.bytes} bytes)"
        )
        print(f"rows:  {stats.rows}")
        for source, n in stats.by_source.items():
            print(f"  {source}: {n}")
        print(f"oldest fetched: {_when(stats.oldest)}")
        print(f"newest fetched: {_when(stats.newest)}")
        return 0
    opened = _open(args.cache)
    if opened is None:
        return 3
    with opened as store:
        before = store.cache_bytes()
        done = store.prune_cache(args.max_bytes)
        after = store.cache_bytes()
    print(
        f"pruned {done.expired} expired and {done.evicted} least recently used rows, "
        f"{done.blobs_deleted} blobs deleted; {_mb(before)} -> {_mb(after)}"
    )
    return 0
