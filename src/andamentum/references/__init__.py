"""Check a draft's references: how it cites, what it cites, and whether the
works exist.

One async entry point::

    report = await check_references(text, sections=..., hints=..., model=...)

Four stages:

1. **Find** the reference list and split it into entries, numbered or not
   (:mod:`.parse`), and collect every possible citation of every kind —
   numbers in brackets or parentheses, superscript numbers (from the file's
   formatting, passed as *hints*), digits glued to words, author-year
   citations, reference-manager fields (:mod:`.scan`).
2. **Decide** how the document cites, and which candidates are citations
   (:mod:`.decide`). Evidence first: what points to real entries, and whether
   the list is numbered. A model is asked only when the evidence is mixed,
   with a bounded sample and a flat answer, so a local model can do it. With
   no *model*, what the evidence cannot settle is reported as undecided —
   never guessed.
3. **Check** the citations against the list (:mod:`.check`): cited but not
   listed, never cited, the same work twice, and per style — numbers out of
   first-citation order; an author-year citation whose year differs from its
   entry, or that fits two entries.
4. **Look up** each entry in Crossref unless ``lookup=False``
   (:mod:`.lookup`): does the work exist, and do its title, year and first
   author agree. Only entry text is sent, never the draft.

Every problem carries offsets into *text*, so a caller holding the document in
the same coordinates attaches it to the exact citation or entry.

Depends on ``pydantic``, ``httpx``, ``rapidfuzz`` and — for the model calls
only — ``andamentum.core``. Imports no other andamentum sub-module.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

import httpx

from .check import check_citations
from .decide import decide_style, judge_candidates
from .lookup import lookup_entries
from .models import (
    Candidate,
    CitedWork,
    Hint,
    Judged,
    LookupResult,
    Match,
    Problem,
    ReferenceEntry,
    ReferenceReport,
    StyleDecision,
)
from .parse import find_reference_list, split_entries
from .scan import scan

__all__ = [
    "Candidate",
    "CitedWork",
    "Hint",
    "Judged",
    "LookupResult",
    "Match",
    "Problem",
    "ReferenceEntry",
    "ReferenceReport",
    "StyleDecision",
    "check_references",
]


async def check_references(
    text: str,
    *,
    sections: Sequence[tuple[str, int, int]] | None = None,
    hints: list[Hint] | None = None,
    model: str | None = None,
    lookup: bool = True,
    mailto: str | None = None,
    client: httpx.AsyncClient | None = None,
) -> ReferenceReport:
    """Check the references of *text*.

    *sections* — ``(title, start, end)`` per section, when the caller knows
    them — locates the reference list by its heading, and the first section's
    end bounds the author block (where raised numbers are affiliations).
    *hints* carry what the file's format knows and the text does not:
    superscript runs and reference-manager fields. *model* is used only when
    the evidence is mixed; ``None`` means such cases are reported as
    undecided. *mailto* joins Crossref's polite pool; *client* lets a caller
    (or a test) supply the HTTP client.
    """
    span = find_reference_list(text, sections=sections)
    if span is None:
        return ReferenceReport(found_list=False)
    start, end = span
    entries = split_entries(text, start=start, end=end)
    candidates = scan(text, start=0, end=start, hints=hints) + scan(
        text, start=end, end=len(text), hints=hints, first_id=10_000
    )
    report = ReferenceReport(found_list=True, list_start=start, list_end=end, entries=entries)
    author_block_end = _author_block_end(sections, text)
    report.style = await decide_style(
        text, candidates, entries, model=model, author_block_end=author_block_end
    )
    if report.style.style != "undecided":
        report.judged = await judge_candidates(
            text,
            candidates,
            entries,
            style=report.style.style,
            model=model,
            author_block_end=author_block_end,
        )
        report.problems = check_citations(
            entries,
            report.citations,
            style=report.style.style,
            complete=not any(j.reason.startswith("not judged") for j in report.judged),
        )
    if lookup and entries:
        report.looked_up = True
        report.lookups = await lookup_entries(entries, client=client, mailto=mailto)
        for entry, result in zip(entries, report.lookups):
            problem = _lookup_problem(result, entry)
            if problem is not None:
                report.problems.append(problem)
    return report


_FIRST_HEADING = re.compile(
    r"^[ \t#]*(abstract|summary|introduction|background|keywords)\b",
    re.IGNORECASE | re.MULTILINE,
)


def _author_block_end(sections: Sequence[tuple[str, int, int]] | None, text: str) -> int:
    """Where the title and author block ends, or 0 when there is none.

    With sections, the end of the first one (capped: a document with no
    headings is one long section, and its "author block" is not all of it).
    Without, the first line that opens the body — Abstract, Introduction and
    the like. A text with neither has no author block, and no raised number
    in it is read as an affiliation mark.
    """
    if sections and len(sections) > 1:
        first = sorted(sections, key=lambda s: s[1])[0]
        return min(first[2], 3000)
    heading = _FIRST_HEADING.search(text)
    return heading.start() if heading else 0


def _lookup_problem(result: LookupResult, entry: ReferenceEntry) -> Problem | None:
    """A lookup result as a problem with the entry, or None when it is fine
    or was not checked."""
    if result.status == "not_found" and entry.doi:
        return Problem(
            kind="doi_not_found",
            severity="major",
            headline="The DOI does not resolve",
            issue=(
                f"The DOI {entry.doi} leads nowhere, so a reader following it "
                f"cannot reach the source."
            ),
            fix="Copy the correct DOI from the publisher's page for this work.",
            start=entry.start,
            end=entry.end,
            entry_number=entry.number,
        )
    if result.status == "not_found":
        return Problem(
            kind="not_found",
            severity="moderate",
            headline="No published record found",
            issue=(
                "Crossref, which indexes most journals and publishers, has no "
                "work whose title matches this entry, so a reader may not be "
                "able to find it. Some conference papers and reports are not "
                "indexed there."
            ),
            fix=(
                "Check the entry against the source and add its DOI, or remove "
                "it if you cannot find the work."
            ),
            start=entry.start,
            end=entry.end,
            entry_number=entry.number,
        )
    if result.status == "details_differ" and result.match is not None:
        different_work = result.detail.startswith("the DOI points")
        return Problem(
            kind="details_differ",
            severity="major" if different_work else "moderate",
            headline="The DOI is for a different work"
            if different_work
            else "Details differ from the published record",
            issue=(
                f"Crossref matches this entry to “{result.match.title}”, "
                f"but {result.detail}."
                if not different_work
                else f"You cite this work, but {result.detail}."
            ),
            fix="Check the entry against the published record and correct it.",
            start=entry.start,
            end=entry.end,
            entry_number=entry.number,
        )
    return None
