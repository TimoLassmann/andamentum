"""``andamentum-research cache-info`` and ``cache-prune``: look at, and trim, an HTTP cache.

    andamentum-research cache-info --cache PATH
    andamentum-research cache-prune --cache PATH [--max-bytes N]

``PATH`` is a research store file (the ``--cache`` of ``andamentum-research``). The cache is capped on every write (``store.HTTP_CACHE_MAX_BYTES``);
``cache-prune`` applies a cap by hand, for example a smaller one. A missing file is an error
and is never created.
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


def _open(path: str, *, max_bytes: int | None = None) -> ResearchStore:
    target = Path(path)
    if not target.is_file():
        print(f"Error: no cache at {target}", file=sys.stderr)
        sys.exit(1)
    return ResearchStore(target, max_cache_bytes=max_bytes)


def main(argv: list[str]) -> None:
    """Run the subcommand named by ``argv[0]``; the rest are its flags."""
    parser = argparse.ArgumentParser(prog="andamentum-research")
    sub = parser.add_subparsers(dest="command", required=True)
    info = sub.add_parser("cache-info", help="size, rows per source, oldest and newest")
    info.add_argument("--cache", required=True, metavar="PATH")
    prune = sub.add_parser(
        "cache-prune", help="evict expired, then least recently used rows"
    )
    prune.add_argument("--cache", required=True, metavar="PATH")
    prune.add_argument(
        "--max-bytes",
        type=int,
        default=None,
        metavar="N",
        help="the cap to bring the cache under (default: the store's cap, 200 MB)",
    )
    args = parser.parse_args(argv)
    if args.command == "cache-info":
        with _open(args.cache) as store:
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
        return
    with _open(args.cache) as store:
        before = store.cache_bytes()
        done = store.prune_cache(args.max_bytes)
        after = store.cache_bytes()
    print(
        f"pruned {done.expired} expired and {done.evicted} least recently used rows, "
        f"{done.blobs_deleted} blobs deleted; {_mb(before)} -> {_mb(after)}"
    )
