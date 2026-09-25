"""Which entry a citation points to. Deterministic.

A number points to the entry with that number. An author-year work points to
the entry whose FIRST author has that surname and whose publication year (the
first year the entry prints) is that year, with the a/b suffix choosing between
two such entries. Names are compared without accents or case, so "Garcia" in
the text finds "García" in the list.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

from .models import CitedWork, ReferenceEntry


def normalise(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text)
    plain = "".join(c for c in decomposed if not unicodedata.combining(c))
    return " ".join(re.sub(r"[^a-z0-9]+", " ", plain.lower()).split())


def first_author_is(surname: str, entry: ReferenceEntry) -> bool:
    """Whether *entry*'s author part starts with *surname*.

    The surname is tried whole and then without its leading words, longest
    first: the scan keeps every capitalised word ("Nguengang Wakap"), and in
    running text that includes a sentence-initial one ("As Turro"). A
    leading "The" in the entry is skipped, for "The GTEx Consortium".
    """
    words = normalise(surname).split()
    lead = normalise(entry.lead or entry.text[:120])
    for i in range(len(words)):
        name = " ".join(words[i:])
        for start in (lead, lead.removeprefix("the ")):
            if start == name or start.startswith(name + " "):
                return True
    return False


@dataclass
class WorkResolution:
    """What one cited work resolved to."""

    status: str  # "match" | "year_mismatch" | "ambiguous" | "missing"
    entries: list[ReferenceEntry] = field(default_factory=list)


def resolve_work(work: CitedWork, entries: list[ReferenceEntry]) -> WorkResolution:
    by_author = [e for e in entries if first_author_is(work.surname, e)]
    if not by_author:
        return WorkResolution("missing")
    same_year = [e for e in by_author if e.year == work.year]
    if not same_year:
        return WorkResolution("year_mismatch", by_author)
    if len(same_year) == 1:
        return WorkResolution("match", same_year)
    if work.suffix:
        suffixed = [e for e in same_year if e.year_suffix == work.suffix]
        if len(suffixed) == 1:
            return WorkResolution("match", suffixed)
        # Lists often drop the suffix; then "a" is the first such entry.
        index = ord(work.suffix) - ord("a")
        if not any(e.year_suffix for e in same_year) and 0 <= index < len(same_year):
            return WorkResolution("match", [same_year[index]])
    return WorkResolution("ambiguous", same_year)


def numbered(entries: list[ReferenceEntry]) -> dict[int, list[ReferenceEntry]]:
    out: dict[int, list[ReferenceEntry]] = {}
    for e in entries:
        if e.number is not None:
            out.setdefault(e.number, []).append(e)
    return out


def resolve_note_work(work: CitedWork, entries: list[ReferenceEntry]) -> WorkResolution:
    """A work named in a note, against the bibliography.

    By first-author surname, then by title: one author often has several
    works in a bibliography, and a note's short form names the title, not the
    year. The title is matched fuzzily (a short form shortens it; quotation
    marks and capitals differ between note and entry).
    """
    from rapidfuzz import fuzz

    by_author = [e for e in entries if first_author_is(work.surname, e)]
    if not by_author:
        return WorkResolution("missing")
    title = normalise(work.title)
    if title:
        titled = [
            e for e in by_author if fuzz.partial_ratio(title, normalise(e.text)) >= 85
        ]
        if not titled:
            return WorkResolution("missing", by_author)
        by_author = titled
    if len(by_author) > 1:
        return WorkResolution("ambiguous", by_author)
    entry = by_author[0]
    if work.year is not None and entry.year is not None and work.year != entry.year:
        return WorkResolution("year_mismatch", [entry])
    return WorkResolution("match", [entry])
