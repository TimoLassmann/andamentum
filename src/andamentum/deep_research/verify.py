"""VERIFY: every report passes through these checks, in code, before it is returned.

Cited finding ids exist, verified quotes are in their passages, unverified
quotes are flagged and marked, retractions, links and DOIs. No model.

- **Finding ids.** Every id the report cites is a finding of the run.
- **Quotes.** A verified finding's quote is still in its passage (a sanity check
  that should never fire). A cited finding whose quote could not be found in its
  page is flagged once (``unverified_quote``), and a sentence whose cited
  findings are all unverified is marked.
- **Retraction.** No cited finding rests on a retracted work, and no DOI in the
  report is retracted (Crossref, through ``references``).
- **Links and DOIs.** Every URL and DOI in the report resolves. Only a settled
  "not there" (404, 410, an unresolvable host, a refused address) is a flag; a
  timeout, a server error or a bot block proves nothing, so such a link is
  listed as unchecked instead.

Sentences are the stretches of text that end at a ``[F12]`` citation. The
``Sources`` section and the headings are not sentences; the fixed lines the
writer builds in code (``write.CODE_LINE_PREFIXES``) carry no ids by design.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from andamentum.core.agents import gather_llm
from andamentum.references import (
    ReferenceHttp,
    Work,
    lookup_retraction,
    normalise_doi,
)

from .read import locate_quote
from .types import Finding, Passage, VerifyFlag
from .write import SOURCES_HEADING

UNVERIFIED_MARK = " [quote not verified]"
MAX_LINKS_CHECKED = 100
LINK_CONCURRENCY = 8

# The brackets the writer puts around finding ids.
_CITATION = re.compile(r"\[\s*(F\d+(?:\s*[,;]\s*F\d+)*)\s*\]")
_FINDING_ID = re.compile(r"F\d+")
_LIST_MARKER = re.compile(r"\s*(?:[-*+]|\d+[.)])\s+")
_URL = re.compile(r"https?://[^\s<>\"')\]]+")
_DOI = re.compile(r"\b10\.\d{4,9}/[^\s<>\"')\]]+")
_TRAILING = ".,;:!?"
# Statuses that settle that a link does not lead anywhere.
_DEAD = frozenset({"not_found", "dns_failure", "blocked"})


@dataclass(frozen=True)
class ReportUnit:
    """A sentence of the report: its text, the finding ids it cites, where it ends."""

    text: str
    finding_ids: tuple[str, ...]
    end: int


@dataclass
class VerifyOutcome:
    report_markdown: str
    flags: list[VerifyFlag] = field(default_factory=list)
    degraded: list[str] = field(default_factory=list)
    # Links and DOIs that could not be checked (a timeout, a bot block): listed, not flagged.
    unchecked: list[str] = field(default_factory=list)


def report_units(report: str) -> list[ReportUnit]:
    """The sentences of the report body, each ending at its citation.

    Headings, blank lines and everything from the ``Sources`` heading on are
    skipped. Text after the last citation of a line, if it holds words, is a
    sentence with no ids. Citations that follow each other (``[F1][F2]``)
    belong to one sentence.
    """
    units: list[ReportUnit] = []
    offset = 0
    for line in report.splitlines(keepends=True):
        start, offset = offset, offset + len(line)
        stripped = line.strip()
        if stripped == SOURCES_HEADING:
            break
        if not stripped or stripped.startswith("#"):
            continue
        body = line.rstrip("\r\n")
        marker = _LIST_MARKER.match(body)
        position = marker.end() if marker else len(body) - len(body.lstrip())
        first_of_line = len(units)
        for found in _CITATION.finditer(body, position):
            text = body[position : found.start()]
            end = found.end()
            while end < len(body) and body[end] in ".;,":
                end += 1
            ids = tuple(_FINDING_ID.findall(found.group(1)))
            if not text.strip(" .;,") and len(units) > first_of_line:
                last = units[-1]
                units[-1] = ReportUnit(
                    last.text, tuple(dict.fromkeys(last.finding_ids + ids)), start + end
                )
            else:
                units.append(ReportUnit(text.strip(" ;,"), ids, start + end))
            position = end
        tail = body[position:].strip()
        if re.search(r"\w", tail):
            units.append(ReportUnit(tail, (), start + len(body.rstrip())))
    return units


def check_quotes(
    findings: Sequence[Finding], passages: Mapping[str, Passage]
) -> list[VerifyFlag]:
    """A ``quote_missing`` flag for each VERIFIED finding whose quote is not in its
    passage (it should never fire); unverified findings are skipped here."""
    flags: list[VerifyFlag] = []
    for finding in findings:
        if not finding.quote_verified:
            continue
        passage = passages.get(finding.passage_id)
        if passage is None:
            flags.append(
                VerifyFlag(
                    kind="quote_missing",
                    ref=finding.finding_id,
                    detail="its passage is not available",
                )
            )
        elif locate_quote(finding.quote, passage.text) is None:
            flags.append(
                VerifyFlag(
                    kind="quote_missing",
                    ref=finding.finding_id,
                    detail="the quote is not in its passage",
                )
            )
    return flags


async def verify_report(
    report_markdown: str,
    *,
    findings: Sequence[Finding],
    passages: Mapping[str, Passage],
    works: Mapping[str, Work],
    http: ReferenceHttp,
) -> VerifyOutcome:
    """Run every check on *report_markdown*: cited ids exist, verified quotes are in
    their passages, cited unverified findings are flagged (``unverified_quote``, and
    ``UNVERIFIED_MARK`` on a sentence whose cited findings are all unverified),
    retracted works and DOIs, dead links and unresolved DOIs. Returns the report
    (marked) and the flags."""
    by_id = {f.finding_id: f for f in findings}
    flags = check_quotes(findings, passages)
    units = report_units(report_markdown)

    seen_unknown: set[str] = set()
    seen_unverified: set[str] = set()
    retracted_works: set[str] = set()
    unmarked: list[ReportUnit] = []
    for unit in units:
        known = [by_id[i] for i in unit.finding_ids if i in by_id]
        for finding_id in unit.finding_ids:
            if finding_id not in by_id and finding_id not in seen_unknown:
                seen_unknown.add(finding_id)
                flags.append(
                    VerifyFlag(
                        kind="unknown_finding",
                        ref=finding_id,
                        detail="the report cites a finding that does not exist",
                    )
                )
        for finding in known:
            if not finding.quote_verified and finding.finding_id not in seen_unverified:
                seen_unverified.add(finding.finding_id)
                flags.append(
                    VerifyFlag(
                        kind="unverified_quote",
                        ref=finding.finding_id,
                        detail="the quote could not be found in its page",
                    )
                )
            work = works.get(finding.work_id)
            if (
                work is not None
                and work.retracted == "retracted"
                and finding.work_id not in retracted_works
            ):
                retracted_works.add(finding.work_id)
                flags.append(
                    VerifyFlag(
                        kind="retracted",
                        ref=finding.work_id,
                        detail=f"cited by {finding.finding_id}: {work.title}",
                    )
                )
        if known and all(not f.quote_verified for f in known):
            unmarked.append(unit)

    link_flags, unchecked = await check_links(report_markdown, http=http)
    return VerifyOutcome(
        report_markdown=_mark(report_markdown, unmarked),
        flags=flags + link_flags,
        unchecked=unchecked,
    )


def _mark(report: str, units: Sequence[ReportUnit]) -> str:
    marked = report
    for end in sorted({u.end for u in units}, reverse=True):
        if not marked.startswith(UNVERIFIED_MARK, end):
            marked = marked[:end] + UNVERIFIED_MARK + marked[end:]
    return marked


# -- links and DOIs ---------------------------------------------------------


def links_in(report: str) -> tuple[list[str], list[str]]:
    """The distinct URLs and DOIs written in *report*, in order of appearance.

    A ``doi.org`` URL counts as its DOI, not as a URL.
    """
    dois: list[str] = []
    for found in _DOI.finditer(report):
        doi = normalise_doi(found.group(0))
        if doi is not None and doi not in dois:
            dois.append(doi)
    urls: list[str] = []
    for found in _URL.finditer(report):
        url = found.group(0).rstrip(_TRAILING)
        if normalise_doi(url) is None and url not in urls:
            urls.append(url)
    return urls, dois


async def check_links(
    report: str, *, http: ReferenceHttp
) -> tuple[list[VerifyFlag], list[str]]:
    """Flags for dead links, unresolved DOIs and retracted DOIs; and the links
    that could not be settled either way."""
    urls, dois = links_in(report)
    flags: list[VerifyFlag] = []
    unchecked: list[str] = []
    kept_urls = urls[:MAX_LINKS_CHECKED]
    kept_dois = dois[: max(MAX_LINKS_CHECKED - len(kept_urls), 0)]
    omitted = urls[len(kept_urls) :] + dois[len(kept_dois) :]
    unchecked += [f"{x}: over the limit of {MAX_LINKS_CHECKED} links" for x in omitted]
    urls, dois = kept_urls, kept_dois

    healths = await gather_llm(
        [http.check_url(u) for u in urls], concurrency=LINK_CONCURRENCY
    )
    for health in healths:
        if health.status in _DEAD:
            flags.append(
                VerifyFlag(
                    kind="dead_link",
                    ref=health.url,
                    detail=f"{health.status}: {health.detail}".strip(": "),
                )
            )
        elif health.status != "ok":
            unchecked.append(f"{health.url}: {health.status} {health.detail}".strip())

    for doi in dois:
        status = await lookup_retraction(doi, http=http)
        if status.status == "service_error":
            unchecked.append(f"doi {doi}: {status.detail}")
        elif status.status == "not_found":
            # Crossref knows only its own registrations; doi.org resolves them all.
            health = await http.check_url(f"https://doi.org/{doi}")
            if health.status in _DEAD:
                flags.append(
                    VerifyFlag(kind="doi_unresolved", ref=doi, detail=status.detail)
                )
            elif health.status != "ok":
                unchecked.append(f"doi {doi}: {health.status} {health.detail}".strip())
        elif status.retracted:
            flags.append(
                VerifyFlag(
                    kind="retracted", ref=doi, detail="Crossref lists a retraction"
                )
            )
    return flags, unchecked
