"""VERIFY: every report passes through these checks before it is returned.

Checks, in code unless noted:

- **Quotes.** Every finding's quote occurs in its passage (``read.locate_quote``).
- **Finding ids.** Every id the report cites is a finding of the run.
- **Claim support** (decision model or language model): for each report
  sentence, do the quotes of the findings it cites state what it claims? A
  sentence that fails, or that cites nothing, is flagged ``unsupported`` and
  marked in the returned report.
- **Retraction.** No cited finding rests on a retracted work, and no DOI in the
  report is retracted (Crossref, through ``references``).
- **Links and DOIs.** Every URL and DOI in the report resolves. Only a
  settled "not there" (404, 410, an unresolvable host, a refused address)
  is a flag; a timeout, a server error or a bot block proves nothing, so
  such a link is listed as unchecked instead.
- **Independence.** A sentence whose findings come from fewer than
  ``MIN_INDEPENDENT_SOURCES`` independent sources is flagged ``single_source``.

Sentences are the stretches of text that end at a ``[F12]`` citation. The
``Sources`` section and the headings are not sentences; the fixed lines the
writer builds in code (``write.CODE_LINE_PREFIXES``) carry no ids by design.
Every flag is written to the run's items.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from pydantic import BaseModel, Field

from andamentum.core.agents import gather_llm
from andamentum.references import (
    ReferenceHttp,
    Work,
    lookup_retraction,
    normalise_doi,
)

from .ask import MODEL_ERRORS, Asker, BudgetExhausted
from .coverage import independent_sources
from .read import locate_quote
from .types import MIN_INDEPENDENT_SOURCES, Finding, Passage, VerifyFlag
from .write import CODE_LINE_PREFIXES, SOURCES_HEADING

UNSUPPORTED_MARK = " [unsupported]"
CLAIM_QUESTION = (
    "Do the quotes state what the statement claims? Answer no when the statement "
    "says more than the quotes support, says something they do not say, or "
    "gives a number or name they do not contain."
)
# States put to the claim-support check at a time (one budget check per chunk).
CLAIM_CHUNK = 8
MAX_LINKS_CHECKED = 100
LINK_CONCURRENCY = 8

# The brackets the writer puts around finding ids; the same form as
# ``ResearchResult.cited_work_ids`` reads.
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


def _is_code_line(unit: ReportUnit) -> bool:
    return not unit.finding_ids and unit.text.startswith(CODE_LINE_PREFIXES)


class VerifyOutcome(BaseModel):
    """What verification found. ``report_markdown`` is the report with
    unsupported sentences marked; ``unchecked`` lists links that could not be
    settled; ``degraded`` says which checks could not run."""

    flags: list[VerifyFlag] = Field(default_factory=list)
    report_markdown: str = ""
    unchecked: list[str] = Field(default_factory=list)
    degraded: list[str] = Field(default_factory=list)
    sentences_checked: int = 0


def check_quotes(
    findings: Sequence[Finding], passages: Mapping[str, Passage]
) -> list[VerifyFlag]:
    """A ``quote_missing`` flag for each finding whose quote is not in its passage."""
    flags: list[VerifyFlag] = []
    for finding in findings:
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
    asker: Asker,
    http: ReferenceHttp,
) -> VerifyOutcome:
    """Run every check on *report_markdown* and record the flags in the store."""
    by_id = {f.finding_id: f for f in findings}
    flags = check_quotes(findings, passages)
    degraded: list[str] = []
    units = report_units(report_markdown)

    seen_unknown: set[str] = set()
    for unit in units:
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

    cited = [by_id[i] for u in units for i in u.finding_ids if i in by_id]
    retracted_works: set[str] = set()
    for finding in cited:
        work = works.get(finding.work_id)
        if work is not None and work.retracted == "retracted":
            if finding.work_id not in retracted_works:
                retracted_works.add(finding.work_id)
                flags.append(
                    VerifyFlag(
                        kind="retracted",
                        ref=finding.work_id,
                        detail=f"cited by {finding.finding_id}: {work.title}",
                    )
                )

    checkable: list[tuple[ReportUnit, list[Finding]]] = []
    unsupported: list[ReportUnit] = []
    for unit in units:
        if _is_code_line(unit):
            continue
        known = [by_id[i] for i in unit.finding_ids if i in by_id]
        if not known:
            unsupported.append(unit)
            flags.append(
                VerifyFlag(
                    kind="unsupported",
                    ref=unit.text,
                    detail="cites no finding"
                    if not unit.finding_ids
                    else "cites no known finding",
                )
            )
            continue
        checkable.append((unit, known))
        if (
            independent_sources(known, _works_by_finding(known, works))
            < MIN_INDEPENDENT_SOURCES
        ):
            flags.append(
                VerifyFlag(
                    kind="single_source",
                    ref=unit.text,
                    detail=f"rests on {', '.join(f.finding_id for f in known)}, "
                    "one independent source",
                )
            )

    checked = 0
    for chunk_start in range(0, len(checkable), CLAIM_CHUNK):
        chunk = checkable[chunk_start : chunk_start + CLAIM_CHUNK]
        try:
            answers = await asker.ask_each(
                "claim_support",
                instruction=CLAIM_QUESTION,
                states=[
                    {"statement": unit.text, "quotes": [f.quote for f in known]}
                    for unit, known in chunk
                ],
            )
        except BudgetExhausted as spent:
            left = len(checkable) - chunk_start
            degraded.append(f"claim support not checked for {left} sentences: {spent}")
            break
        except MODEL_ERRORS as error:
            left = len(checkable) - chunk_start
            degraded.append(
                f"claim support not checked for {left} sentences: the model failed "
                f"({type(error).__name__}: {error})"
            )
            break
        checked += len(chunk)
        for (unit, known), p in zip(chunk, answers):
            if p < 0.5:
                unsupported.append(unit)
                flags.append(
                    VerifyFlag(
                        kind="unsupported",
                        ref=unit.text,
                        detail=f"the quotes of {', '.join(f.finding_id for f in known)} "
                        "do not state it",
                    )
                )

    link_flags, unchecked = await check_links(report_markdown, http=http)
    flags += link_flags

    for flag in flags:
        asker.store.add_run_item(
            asker.run_id, "flag", f"verify:{flag.kind}:{flag.ref[:200]}", flag
        )
    return VerifyOutcome(
        flags=flags,
        report_markdown=_mark(report_markdown, unsupported),
        unchecked=unchecked,
        degraded=degraded,
        sentences_checked=checked,
    )


def _works_by_finding(
    findings: Sequence[Finding], works: Mapping[str, Work]
) -> Mapping[str, Work]:
    return {f.work_id: works[f.work_id] for f in findings if f.work_id in works}


def _mark(report: str, unsupported: Sequence[ReportUnit]) -> str:
    marked = report
    for end in sorted({u.end for u in unsupported}, reverse=True):
        marked = marked[:end] + UNSUPPORTED_MARK + marked[end:]
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
