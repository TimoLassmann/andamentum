"""The offline checks: citations against the reference list.

No network and no model. Each problem is located at the citation or entry it
concerns and says, in plain words to the author, what is wrong and what to do.
Numbered and author-year documents share the list-side checks (never cited,
listed twice) and differ in how a citation finds its entry.
"""

from __future__ import annotations

import re

from .models import Candidate, Problem, ReferenceEntry, Style
from .resolve import normalise, numbered, resolve_work


def check_citations(
    entries: list[ReferenceEntry],
    citations: list[Candidate],
    *,
    style: Style,
    complete: bool = True,
) -> list[Problem]:
    """Every mismatch between the accepted citations and the list.

    *complete* is False when some candidates could not be judged (no model
    was given for them). An entry then may be cited by one of those, so
    "never cited" is not reported at all rather than reported wrongly.
    """
    if style == "author_year":
        problems, cited = _author_year(entries, citations)
    else:
        problems, cited = _numeric(entries, citations)
    # "Never cited" means something only when citations were found at all: a
    # document whose citations could not be read would otherwise report every
    # entry in its list.
    if citations and complete:
        problems += _never_cited(entries, cited)
    problems += _duplicate_works(entries)
    if style != "author_year":
        problems += _out_of_order(entries, citations)
    return problems


# ── numbered documents ──────────────────────────────────────────────────────


def _numeric(
    entries: list[ReferenceEntry], citations: list[Candidate]
) -> tuple[list[Problem], set[int]]:
    problems: list[Problem] = []
    by_number = numbered(entries)
    cited_entries: set[int] = set()
    reported: set[int] = set()
    for citation in citations:
        for n in citation.numbers:
            for entry in by_number.get(n, []):
                cited_entries.add(id(entry))
        new = [n for n in citation.numbers if n not in by_number and n not in reported]
        if not new:
            continue
        reported.update(new)
        listed = ", ".join(str(n) for n in new)
        plural = len(new) > 1
        problems.append(
            Problem(
                kind="cited_not_listed",
                severity="major",
                headline=f"Citation {listed} is not in the list"
                if not plural
                else f"Citations {listed} are not in the list",
                issue=(
                    f"You cite {listed} here, but the reference list has no "
                    f"entry numbered {listed}, so a reader cannot find the "
                    f"source{'s' if plural else ''}."
                ),
                fix=f"Add the missing entr{'ies' if plural else 'y'}, or correct the number.",
                start=citation.start,
                end=citation.end,
            )
        )
    for number, same in sorted(by_number.items()):
        for extra in same[1:]:
            problems.append(
                Problem(
                    kind="duplicate_number",
                    severity="major",
                    headline=f"Two entries are numbered {number}",
                    issue=(
                        f"Two entries share the number {number}, so a "
                        f"citation of {number} points to both."
                    ),
                    fix="Renumber the list so each entry has its own number.",
                    start=extra.start,
                    end=extra.end,
                    entry_number=number,
                )
            )
    return problems, cited_entries


def _out_of_order(
    entries: list[ReferenceEntry], citations: list[Candidate]
) -> list[Problem]:
    """Citations that introduce a number out of first-citation order.

    Only checked when the list LOOKS numbered in order of first citation —
    when most new numbers do arrive in sequence. An alphabetical list numbered
    by author has no such order, and flagging every citation in it would be
    noise. The first break is reported and the rest are not: one out-of-order
    citation shifts every number after it, and listing each consequence would
    bury the cause.
    """
    listed = {e.number for e in entries if e.number is not None}
    first_seen: list[tuple[int, Candidate]] = []
    seen: set[int] = set()
    for citation in citations:
        for number in citation.numbers:
            if number in listed and number not in seen:
                seen.add(number)
                first_seen.append((number, citation))
    if len(first_seen) < 3:
        return []
    in_sequence = sum(1 for i, (number, _) in enumerate(first_seen) if number == i + 1)
    if in_sequence < 0.6 * len(first_seen):
        return []
    for i, (number, citation) in enumerate(first_seen):
        if number != i + 1:
            return [
                Problem(
                    kind="out_of_order",
                    severity="minor",
                    headline=f"Citation {number} appears before {i + 1}",
                    issue=(
                        f"The list is numbered in order of first citation, but "
                        f"{number} is cited here before {i + 1} has been cited, "
                        f"so the numbering from this point on is out of sequence."
                    ),
                    fix="Renumber the references in order of first citation.",
                    start=citation.start,
                    end=citation.end,
                    entry_number=number,
                )
            ]
    return []


# ── author-year documents ───────────────────────────────────────────────────


def _author_year(
    entries: list[ReferenceEntry], citations: list[Candidate]
) -> tuple[list[Problem], set[int]]:
    problems: list[Problem] = []
    cited: set[int] = set()
    reported: set[tuple[str, int, str]] = set()
    for citation in citations:
        for work in citation.works:
            resolution = resolve_work(work, entries)
            # A citation with the wrong year still means its author's entry:
            # it is reported once, as a year mismatch — not a second time as
            # an entry nobody cites.
            if resolution.status != "year_mismatch" or len(resolution.entries) == 1:
                cited.update(id(e) for e in resolution.entries)
            key = (normalise(work.surname), work.year, work.suffix)
            if resolution.status == "match" or key in reported:
                continue
            reported.add(key)
            label = f"{work.surname} {work.year}{work.suffix}"
            if resolution.status == "missing":
                problems.append(
                    Problem(
                        kind="cited_not_listed",
                        severity="major",
                        headline=f"{label} is not in the list",
                        issue=(
                            f"You cite {label} here, but no entry in the "
                            f"reference list has {work.surname} as first "
                            f"author, so a reader cannot find the source."
                        ),
                        fix="Add the entry, or correct the author's name.",
                        start=citation.start,
                        end=citation.end,
                    )
                )
            elif resolution.status == "year_mismatch":
                listed_years = sorted({e.year for e in resolution.entries if e.year})
                years = " or ".join(str(y) for y in listed_years)
                problems.append(
                    Problem(
                        kind="year_mismatch",
                        severity="moderate",
                        headline=f"Year differs from the list ({work.year} vs {years})",
                        issue=(
                            f"You cite {label}, but the list has "
                            f"{work.surname} {years}, so the citation and the "
                            f"entry disagree about the year."
                        ),
                        fix="Correct the year in the citation or in the entry.",
                        start=citation.start,
                        end=citation.end,
                    )
                )
            else:  # ambiguous
                problems.append(
                    Problem(
                        kind="ambiguous_citation",
                        severity="moderate",
                        headline=f"{label} could be {len(resolution.entries)} entries",
                        issue=(
                            f"The list has {len(resolution.entries)} entries by "
                            f"{work.surname} from {work.year}, so a reader "
                            f"cannot tell which one this citation means."
                        ),
                        fix=(
                            f"Add a letter to the year in the list and in each "
                            f"citation ({work.year}a, {work.year}b)."
                        ),
                        start=citation.start,
                        end=citation.end,
                    )
                )
    return problems, cited


# ── both ────────────────────────────────────────────────────────────────────


def _never_cited(entries: list[ReferenceEntry], cited: set[int]) -> list[Problem]:
    problems: list[Problem] = []
    for entry in entries:
        if id(entry) in cited:
            continue
        name = f"Reference {entry.number}" if entry.number is not None else "This reference"
        problems.append(
            Problem(
                kind="listed_not_cited",
                severity="moderate",
                headline=f"{name} is never cited",
                issue="This entry is in the list, but no citation in the text points to it.",
                fix="Cite it where it supports the text, or remove it from the list.",
                start=entry.start,
                end=entry.end,
                entry_number=entry.number,
            )
        )
    return problems


def _duplicate_works(entries: list[ReferenceEntry]) -> list[Problem]:
    problems: list[Problem] = []
    first_by_key: dict[str, ReferenceEntry] = {}
    for entry in entries:
        keys = [f"doi:{entry.doi.lower()}"] if entry.doi else []
        keys.append(f"text:{re.sub(r'[^a-z0-9]+', ' ', entry.text.lower()).strip()}")
        earlier = next((first_by_key[k] for k in keys if k in first_by_key), None)
        if earlier is not None:
            which = f"entry {earlier.number}" if earlier.number is not None else "an earlier entry"
            problems.append(
                Problem(
                    kind="duplicate_entry",
                    severity="moderate",
                    headline="The same work is listed twice",
                    issue=f"This entry is the same work as {which}, so it is listed twice.",
                    fix="Remove this entry and cite the earlier one instead.",
                    start=entry.start,
                    end=entry.end,
                    entry_number=entry.number,
                )
            )
            continue
        for key in keys:
            first_by_key.setdefault(key, entry)
    return problems
