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

Also the home of three pieces of shared bibliographic code, usable alone:
:mod:`.identifiers` (DOI / PMID / PMCID / arXiv / ISBN / URL extraction),
:mod:`.crossref` (by-DOI fetch and search as CSL JSON) and :mod:`.pubmed`
(PubMed XML to records, and the efetch call).

Depends on ``pydantic``, ``httpx``, ``rapidfuzz`` and — for the model calls
only — ``andamentum.core``. Imports no other andamentum sub-module.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

import httpx

from andamentum.core.decisions import DecisionModel

from .check import check_citations
from .cited import resolve_cited_work, resolve_entry
from .crossref import CrossrefError, fetch_work, normalise_doi, search_crossref, to_csl
from .decide import decide_style, judge_candidates
from .http import Cache, ReferenceHttp, UrlHealth, UrlStatus
from .identifiers import Identifiers, extract_identifiers
from .lookup import lookup_entries
from .models import (
    LOOKUP_KINDS,
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
from .parse import DOI, find_reference_list, split_entries
from .pubmed import PubMedRecord, fetch_pubmed, parse_pubmed_xml
from .retraction import NoticeKind, RetractionStatus, UpdateNotice, lookup_retraction
from .scan import pandoc_notes, scan

__all__ = [
    "LOOKUP_KINDS",
    "Cache",
    "CrossrefError",
    "Identifiers",
    "PubMedRecord",
    "extract_identifiers",
    "fetch_pubmed",
    "fetch_work",
    "normalise_doi",
    "parse_pubmed_xml",
    "search_crossref",
    "to_csl",
    "NoticeKind",
    "ReferenceHttp",
    "RetractionStatus",
    "UpdateNotice",
    "UrlHealth",
    "UrlStatus",
    "lookup_retraction",
    "resolve_cited_work",
    "resolve_entry",
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
    decision_model: DecisionModel | None = None,
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

    *decision_model* — an opened ``core.decisions.DecisionModel`` — answers
    the candidates the rules leave open instead of *model* (yes when
    ``p >= 0.5``). ``None`` (the default) changes nothing. See :mod:`.gate`.
    """
    span = find_reference_list(text, sections=sections)
    has_notes = any(h.kind == "note" for h in hints or []) or bool(
        pandoc_notes(text)[0]
    )
    if span is None and not has_notes:
        return ReferenceReport(found_list=False)
    if span is None:
        # A notes-only document (no bibliography): the notes are checked on
        # their own, and their full citations are what is looked up.
        start = end = len(text)
        entries = []
    else:
        start, end = span
        # Markdown footnote definitions after the list are notes, not entries.
        definitions = [a for a, _ in pandoc_notes(text)[1] if start <= a < end]
        if definitions:
            end = min(definitions)
        entries = split_entries(text, start=start, end=end)
    candidates = scan(text, start=0, end=start, hints=hints)
    if end < len(text):
        # What follows the list (appendices, legends). Not scanned when
        # nothing does: a note mark at the very end would be read twice.
        candidates += scan(text, start=end, end=len(text), hints=hints, first_id=10_000)
    report = ReferenceReport(
        found_list=span is not None, list_start=start, list_end=end, entries=entries
    )
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
            decision_model=decision_model,
        )
        report.problems = check_citations(
            entries,
            report.citations,
            style=report.style.style,
            complete=not any(j.reason.startswith("not judged") for j in report.judged),
            # Every note, in order — an "Ibid." refers to the note before it,
            # whatever that note was.
            notes=[j.candidate for j in report.judged if j.candidate.source == "note"],
        )
    to_look_up = entries or _full_notes(report)
    if lookup and to_look_up:
        report.looked_up = True
        report.lookups = await lookup_entries(to_look_up, client=client, mailto=mailto)
        for entry, result in zip(to_look_up, report.lookups):
            problem = _lookup_problem(result, entry)
            if problem is not None:
                report.problems.append(problem)
    return report


_FIRST_HEADING = re.compile(
    r"^[ \t#]*(abstract|summary|introduction|background|keywords)\b",
    re.IGNORECASE | re.MULTILINE,
)


def _full_notes(report: ReferenceReport) -> list[ReferenceEntry]:
    """With no bibliography, the full-form notes stand in for its entries:
    each is looked up by its own text, and a problem lands on its note mark.
    One per work, the first full citation of it."""
    seen: set[tuple[tuple[str, str], ...]] = set()
    out: list[ReferenceEntry] = []
    for judged in report.judged:
        note = judged.candidate
        if not judged.is_citation or note.source != "note" or note.note_form != "full":
            continue
        key = tuple(sorted((w.surname, w.title) for w in note.works))
        if key in seen or len(note.works) != 1:
            continue
        seen.add(key)
        body = " ".join(note.note_text.split())
        doi = DOI.search(body)
        out.append(
            ReferenceEntry(
                number=None,
                text=body,
                doi=doi.group(1).rstrip(".,;)") if doi else "",
                start=note.start,
                end=note.end,
                year=note.works[0].year,
                years=[note.works[0].year] if note.works[0].year else [],
            )
        )
    return out


def _author_block_end(
    sections: Sequence[tuple[str, int, int]] | None, text: str
) -> int:
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
