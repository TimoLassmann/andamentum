"""SYNERGY: labelled systematic-review screening records (CC0).

Each review is a list of OpenAlex records marked included or excluded by the
review's authors after full-text screening. The review's own published title
is the research question the screening arms are asked about. Files come from
the ``v1.0`` tag of the ``asreview/synergy-dataset`` repository (the DataverseNL
copy of the same release sits behind a bot wall that refuses scripted
downloads), one review at a time, under the directory the caller names.

``sample_records`` takes a stratified, hash-ranked subset that keeps the review's
prevalence of included records, so work saved on the subset is comparable to
the full review's.
"""

from __future__ import annotations

import csv
import io
import json
import re
import zipfile
from pathlib import Path

from pydantic import BaseModel, Field

from andamentum.references import Work, normalise_doi, normalise_openalex_id

from .download import (
    SYNERGY_CARD,
    DatasetMissing,
    DownloadError,
    Downloader,
    save_file,
)
from .subsets import select_subset

RAW_URL = "https://raw.githubusercontent.com/asreview/synergy-dataset/v1.0/{key}/{name}"
SYNERGY_DIR = "synergy"
DEFAULT_REVIEWS: tuple[str, ...] = (
    "Donners_2021",
    "Sep_2021",
    "Meijboom_2021",
    "Menon_2022",
)
MAX_WORKS_PARTS = 20
MIN_INCLUDED = 10
_KEY = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_\-]*$")
_PLAIN_FILES = ("labels.csv", "metadata_publication.json", "CITATION.txt")


class ScreeningRecord(BaseModel):
    record_id: str = Field(min_length=1, description="The OpenAlex id, e.g. W123.")
    title: str
    abstract: str
    year: int | None
    venue: str
    doi: str | None
    included: bool

    def to_work(self) -> Work:
        """The record as a ``Work`` (with its id as ``work_id``) for the screeners."""
        return Work(
            work_id=self.record_id,
            openalex_id=normalise_openalex_id(self.record_id),
            doi=normalise_doi(self.doi),
            title=self.title,
            year=self.year,
            venue=self.venue,
            abstract=self.abstract,
            access="abstract" if self.abstract else "metadata",
        )


class ScreeningReview(BaseModel):
    key: str
    question: str = Field(min_length=1)
    records: list[ScreeningRecord]
    n_records_total: int
    n_included_total: int
    seed: int


def _check_key(key: str) -> None:
    if not _KEY.match(key):
        raise ValueError(f"{key!r} is not a SYNERGY review key such as 'Donners_2021'")


async def download_synergy(
    root: Path, *, downloader: Downloader, reviews: tuple[str, ...] = DEFAULT_REVIEWS
) -> list[str]:
    """Fetch the labels, works and publication record of each review.

    Returns the files written, relative to *root*. A key that is not a review
    of the release fails at its first file with a 404 ``DownloadError``.
    """
    written: list[str] = []
    for key in reviews:
        _check_key(key)
        for name in _PLAIN_FILES:
            url = RAW_URL.format(key=key, name=name)
            relative = f"{SYNERGY_DIR}/{key}/{name}"
            save_file(
                root, relative, await downloader.get(url), url=url, card=SYNERGY_CARD
            )
            written.append(relative)
        for part in range(1, MAX_WORKS_PARTS + 1):
            name = f"works_{part}.zip"
            url = RAW_URL.format(key=key, name=name)
            data = await downloader.get_optional(url)
            if data is None:
                if part == 1:
                    raise DownloadError(
                        url, "the review has no works_1.zip", status=404
                    )
                break
            relative = f"{SYNERGY_DIR}/{key}/{name}"
            save_file(root, relative, data, url=url, card=SYNERGY_CARD)
            written.append(relative)
    return written


def abstract_text(inverted_index: dict[str, list[int]] | None) -> str:
    """OpenAlex stores an abstract as word -> positions; put the words back in order."""
    if not inverted_index:
        return ""
    placed = {
        position: word
        for word, positions in inverted_index.items()
        for position in positions
    }
    return " ".join(placed[p] for p in sorted(placed))


def _venue(work: dict[str, object]) -> str:
    for slot in ("primary_location", "host_venue"):
        location = work.get(slot)
        if not isinstance(location, dict):
            continue
        source = location.get("source")
        for holder in (source, location):
            if isinstance(holder, dict):
                name = holder.get("display_name")
                if isinstance(name, str) and name:
                    return name
    return ""


def _read_works(folder: Path) -> dict[str, dict[str, object]]:
    parts = sorted(folder.glob("works_*.zip"))
    if not parts:
        raise DatasetMissing(f"no works_*.zip in {folder}")
    works: dict[str, dict[str, object]] = {}
    for part in parts:
        with zipfile.ZipFile(part) as archive:
            for member in sorted(archive.namelist()):
                if not (member.startswith("works_") and member.endswith(".json")):
                    continue
                for work in json.loads(archive.read(member)):
                    short = normalise_openalex_id(str(work.get("id") or ""))
                    if short is not None:
                        works[short] = work
    return works


def _int_or_none(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def load_review(root: Path, key: str, *, seed: int = 0) -> ScreeningReview:
    """Read one downloaded review: every labelled record joined to its work.

    Raises ``DatasetMissing`` when the review is not in *root*, and
    ``ValueError`` for a label that is not 0 or 1, a labelled record with no
    work, or a review with no title.
    """
    _check_key(key)
    folder = root / SYNERGY_DIR / key
    if not (folder / "labels.csv").exists():
        raise DatasetMissing(
            f"the review {key} is not in {root}; run: uv run python -m "
            f"andamentum.deep_research.v2.bench download synergy --dir {root} "
            f"--reviews {key}"
        )
    publication = json.loads((folder / "metadata_publication.json").read_text("utf-8"))
    title = publication.get("title") or publication.get("display_name")
    if not isinstance(title, str) or not title.strip():
        raise ValueError(f"{key}: metadata_publication.json has no title")
    works = _read_works(folder)
    records: list[ScreeningRecord] = []
    reader = csv.DictReader(io.StringIO((folder / "labels.csv").read_text("utf-8")))
    for row in reader:
        label = row["label_included"].strip()
        if label not in ("0", "1"):
            raise ValueError(f"{key}: label_included is {label!r}, not 0 or 1")
        short = normalise_openalex_id(row["openalex_id"])
        work = works.get(short) if short else None
        if short is None or work is None:
            raise ValueError(f"{key}: no work for labelled record {row['openalex_id']}")
        index = work.get("abstract_inverted_index")
        records.append(
            ScreeningRecord(
                record_id=short,
                title=str(work.get("title") or ""),
                abstract=abstract_text(index if isinstance(index, dict) else None),
                year=_int_or_none(work.get("publication_year")),
                venue=_venue(work),
                doi=normalise_doi(str(work.get("doi") or row.get("doi") or "")),
                included=label == "1",
            )
        )
    if len({r.record_id for r in records}) != len(records):
        raise ValueError(f"{key}: labels.csv lists a record more than once")
    return ScreeningReview(
        key=key,
        question=title.strip(),
        records=records,
        n_records_total=len(records),
        n_included_total=sum(r.included for r in records),
        seed=seed,
    )


def sample_records(
    records: list[ScreeningRecord],
    max_records: int | None,
    *,
    seed: int,
    min_included: int = MIN_INCLUDED,
) -> list[ScreeningRecord]:
    """A stratified, deterministic subset of at most *max_records* records.

    The included share of the review is kept (rounded), but never fewer than
    *min_included* included records (all of them when there are fewer), so a
    sparse review still has something to find. Records keep their file order.
    ``None``, or a number at least the size of the review, returns every record.
    """
    if max_records is None or max_records >= len(records):
        return list(records)
    if max_records < 1:
        raise ValueError("max_records must be at least 1")
    included = [r for r in records if r.included]
    excluded = [r for r in records if not r.included]
    wanted = max(min_included, round(max_records * len(included) / len(records)))
    k_included = min(len(included), wanted, max_records)
    k_excluded = min(len(excluded), max_records - k_included)
    chosen = set(
        select_subset([r.record_id for r in included], k_included, seed=seed)
    ) | set(select_subset([r.record_id for r in excluded], k_excluded, seed=seed))
    return [r for r in records if r.record_id in chosen]
