"""``bench cache import``: merge the HTTP caches of existing stores into the shared cache.

Earlier freezes and runs left their source responses in many separate store
files (one per snapshot, one per scratch run). This reads the ``http_cache``
rows of each (``ResearchStore.import_cache``: the freshest row per key wins, an
expired row is not imported, the licence tag travels with the row) so a later
freeze finds them instead of downloading them again. The files read are not
changed. A file that is not a research store (no ``http_cache`` table) is
reported, not an error.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from pydantic import BaseModel

from ..store import CacheImport, CacheImportCounts, ResearchStore


def cache_databases(sources: Iterable[Path], *, exclude: Path) -> list[Path]:
    """The database files named by *sources*: a file as it is, a folder searched
    (recursively, in name order) for ``*.db``. *exclude* (the shared cache) and
    duplicates are left out."""
    skip = exclude.resolve()
    found: dict[Path, None] = {}
    for source in sources:
        if source.is_dir():
            candidates = sorted(source.rglob("*.db"))
        elif source.is_file():
            candidates = [source]
        else:
            raise FileNotFoundError(f"no such file or folder: {source}")
        for path in candidates:
            resolved = path.resolve()
            if resolved != skip:
                found[resolved] = None
    return list(found)


class SourceTotals(BaseModel):
    imported: int = 0
    expired: int = 0
    kept_existing: int = 0
    unreadable: int = 0


@dataclass
class ImportSummary:
    """Every file's outcome and the totals by source."""

    files: list[CacheImport] = field(default_factory=list)

    def by_source(self) -> dict[str, SourceTotals]:
        totals: dict[str, SourceTotals] = {}
        for report in self.files:
            for name, counts in report.by_source.items():
                total = totals.setdefault(name, SourceTotals())
                total.imported += counts.imported
                total.expired += counts.expired
                total.kept_existing += counts.kept_existing
                total.unreadable += counts.unreadable
        return dict(sorted(totals.items()))

    def lines(self) -> list[str]:
        problems = [f for f in self.files if f.problem]
        readable = len(self.files) - len(problems)
        out = [
            f"cache import: {len(self.files)} files, {readable} with an http_cache"
            f", {len(problems)} not a research store"
        ]
        for report in problems:
            out.append(f"  skipped {report.path}: {report.problem}")
        by_source = self.by_source()
        out.append(
            f"  {'source':22s} {'imported':>9s} {'expired':>8s} "
            f"{'older':>8s} {'unreadable':>10s}"
        )
        for name, t in by_source.items():
            out.append(
                f"  {name:22s} {t.imported:9d} {t.expired:8d} "
                f"{t.kept_existing:8d} {t.unreadable:10d}"
            )
        out.append(
            f"  {'total':22s} {sum(t.imported for t in by_source.values()):9d} "
            f"{sum(t.expired for t in by_source.values()):8d} "
            f"{sum(t.kept_existing for t in by_source.values()):8d} "
            f"{sum(t.unreadable for t in by_source.values()):10d}"
        )
        out.append(
            "  older = a row as fresh or fresher was already in the cache "
            "(or imported earlier in this run)"
        )
        return out


def import_stores(shared: ResearchStore, databases: Sequence[Path]) -> ImportSummary:
    """Merge each of *databases* into *shared*, in order."""
    summary = ImportSummary()
    for database in databases:
        summary.files.append(shared.import_cache(database))
    return summary


__all__ = [
    "CacheImportCounts",
    "ImportSummary",
    "cache_databases",
    "import_stores",
]
