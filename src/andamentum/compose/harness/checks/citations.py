"""Do the cited works exist, are they retracted, do the URLs resolve (spec section 4).

Needs the network (one cached HTTP client from the run context). Every answer
is a hint for a person, never a verdict: an unresolved work or a dead URL is a
warning ("could not verify"), because a lookup can miss a real work and a
server can refuse a robot. Only a retraction notice is an error.

What is read, as cited works:

- **Reference-manager fields** (Zotero, Mendeley, EndNote): the DOI, title,
  authors and year the field's data carries. The diagnostic sits on the field.
- **Plain-text reference entries**: the reference list found and split by
  ``andamentum.references`` (the package's one citation reader). The
  diagnostic sits on the entry. A work cited by a field and listed as an entry
  is looked up once and reported at each place.

What is checked, per distinct work: existence and metadata through the
two-stage ``resolve_cited_work`` / ``resolve_entry`` (search, then compare
title, authors, year); for a work that resolves, retraction, expression of
concern and correction through ``lookup_retraction`` (error for a retraction,
warning for the others). URLs written in the text (not DOI links, which the
existence check covers) go through ``ReferenceHttp.check_url``: 404/410, a
server error, a name that does not resolve and a timeout are warnings with
their own wording; other 4xx answers (401, 403, 429) are info, since they are
usually a robot block.

Coverage limits: a URL stored only as a hyperlink target (the display text
being something else) is not read, because the view carries no link targets;
a plain-text citation outside a reference list is not looked up (only reported
when no entry of the document's reference list names its first author); the existence
check cannot tell a missing record from a Crossref gap, which is why it warns.
Lookups run one after another.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from andamentum.references import (
    LookupResult,
    ReferenceEntry,
    ReferenceHttp,
    UrlHealth,
    lookup_retraction,
    resolve_cited_work,
    resolve_entry,
)
from andamentum.references.models import CitedWork
from andamentum.references.parse import find_reference_list, split_entries
from andamentum.references.resolve import resolve_work

from ..kernel import (
    CheckContext,
    CheckSpec,
    Diagnostic,
    DocumentView,
    FieldToken,
    Severity,
    register,
)

_MANAGERS = {"zotero", "endnote", "mendeley"}
_URL = re.compile(r"https?://[^\s<>\"]+")
_DOI_HOSTS = {"doi.org", "dx.doi.org"}
_TRAILING = ".,;:)]}'”’"


@dataclass(frozen=True)
class _Cited:
    """One place a work is cited, with what is known about it."""

    paragraph_id: str
    start: int
    end: int
    shown: str  # the field display or the entry text, for evidence
    doi: str = ""
    title: str = ""
    authors: tuple[str, ...] = ()
    year: int | None = None
    entry: ReferenceEntry | None = None

    @property
    def key(self) -> str:
        if self.doi:
            return "doi:" + self.doi.lower()
        return "title:" + re.sub(r"\W+", " ", self.title.lower()).strip()


@dataclass
class _Verdict:
    resolution: LookupResult
    doi: str
    notices: list[tuple[str, str, str]]  # (kind, source, notice doi)
    retraction_error: str = ""


def _clean_doi(value: object) -> str:
    text = str(value or "").strip()
    for prefix in (
        "https://doi.org/",
        "http://doi.org/",
        "https://dx.doi.org/",
        "doi:",
    ):
        if text.lower().startswith(prefix):
            text = text[len(prefix) :]
    return text.rstrip(_TRAILING)


def _year_of(item: dict[str, object]) -> int | None:
    issued = item.get("issued")
    if isinstance(issued, dict):
        parts = issued.get("date-parts")
        if isinstance(parts, list) and parts and isinstance(parts[0], list):
            first = parts[0][0] if parts[0] else None
            if isinstance(first, (int, str)) and str(first).isdigit():
                return int(first)
    return None


def _csl_items(token: FieldToken) -> list[_Cited]:
    out: list[_Cited] = []
    items = token.data.get("citationItems")
    if not isinstance(items, list):
        return out
    for item in items:
        data = item.get("itemData") if isinstance(item, dict) else None
        if not isinstance(data, dict):
            continue
        authors = tuple(
            str(a.get("family") or a.get("literal") or "")
            for a in data.get("author", [])  # type: ignore[union-attr]
            if isinstance(a, dict)
        )
        title = data.get("title")
        out.append(
            _Cited(
                paragraph_id=token.paragraph_id,
                start=token.start,
                end=token.end,
                shown=token.display,
                doi=_clean_doi(data.get("DOI")),
                title=title if isinstance(title, str) else "",
                authors=tuple(a for a in authors if a),
                year=_year_of(data),
            )
        )
    return out


def _endnote_items(token: FieldToken) -> list[_Cited]:
    out: list[_Cited] = []
    cites = token.data.get("cites")
    if not isinstance(cites, list):
        return out
    for cite in cites:
        if not isinstance(cite, dict):
            continue
        authors = cite.get("authors")
        families = tuple(
            str(a).split(",")[0].strip()
            for a in (authors if isinstance(authors, list) else [])
        ) or ((str(cite.get("author", "")),) if cite.get("author") else ())
        year = str(cite.get("record_year") or cite.get("year") or "")
        out.append(
            _Cited(
                paragraph_id=token.paragraph_id,
                start=token.start,
                end=token.end,
                shown=token.display,
                doi=_clean_doi(cite.get("doi")),
                title=str(cite.get("title") or ""),
                authors=tuple(a for a in families if a),
                year=int(year) if year.isdigit() else None,
            )
        )
    return out


def _field_works(view: DocumentView) -> list[_Cited]:
    works: list[_Cited] = []
    for token in view.fields:
        if token.kind != "citation" and token.manager != "endnote":
            continue
        if token.manager not in _MANAGERS or "parse_error" in token.data:
            continue  # crossrefs reports a field that does not parse
        if token.manager == "endnote":
            works.extend(_endnote_items(token))
        else:
            works.extend(_csl_items(token))
    return works


def _entry_works(view: DocumentView) -> list[_Cited]:
    text = view.text()
    span = find_reference_list(text)
    if span is None:
        return []
    lengths = {p.id: len(p.text) for p in view.paragraphs}
    works: list[_Cited] = []
    for entry in split_entries(text, start=span[0], end=span[1]):
        paragraph_id, local = view.locate(entry.start)
        end = min(local + (entry.end - entry.start), lengths[paragraph_id])
        cleaned = entry.model_copy(update={"doi": _clean_doi(entry.doi)})
        works.append(
            _Cited(
                paragraph_id=paragraph_id,
                start=local,
                end=end,
                shown=entry.text[:300],
                doi=cleaned.doi,
                entry=cleaned,
            )
        )
    return works


def _urls(view: DocumentView) -> list[tuple[str, str, int, int]]:
    """``(url, paragraph id, start, end)`` for every non-DOI URL in the text."""
    found: list[tuple[str, str, int, int]] = []
    for paragraph in view.paragraphs:
        for match in _URL.finditer(paragraph.text):
            url = match.group(0).rstrip(_TRAILING)
            host = re.sub(r"^https?://", "", url).split("/")[0].lower()
            if host in _DOI_HOSTS:
                continue
            found.append((url, paragraph.id, match.start(), match.start() + len(url)))
    return found


class CitationsCheck:
    spec = CheckSpec(id="citations", tier=0, scope="document", needs=["network"])
    version = "2"

    async def run(self, view: DocumentView, ctx: CheckContext) -> list[Diagnostic]:
        if ctx.http is None:
            raise ValueError("the citations check needs the run's HTTP client")
        http = ctx.http
        works = _field_works(view) + _entry_works(view)
        verdicts: dict[str, _Verdict] = {}
        for work in works:
            if work.key not in verdicts and (work.doi or work.title or work.entry):
                verdicts[work.key] = await self._verify(work, http)
        out: list[Diagnostic] = []
        service_errors: list[str] = []
        for work in works:
            verdict = verdicts.get(work.key)
            if verdict is None:
                continue
            out.extend(self._work_diagnostics(work, verdict, service_errors))
        if service_errors:
            out.append(
                self._diag(
                    "warning",
                    f"Could not verify {len(service_errors)} cited work(s): "
                    f"the lookup failed ({service_errors[0]}).",
                    evidence=sorted(set(service_errors))[:5],
                    data={"kind": "lookup_failed"},
                )
            )
        out.extend(self._unlisted(view))
        out.extend(await self._url_diagnostics(view, http))
        return out

    def _unlisted(self, view: DocumentView) -> list[Diagnostic]:
        """Plain-text author-year citations no reference-list entry answers.

        Such a citation names a work nothing in the document lets a reader (or
        the existence check) find, so it stands unverified.
        """
        entries = [w.entry for w in _entry_works(view) if w.entry is not None]
        out: list[Diagnostic] = []
        for token in view.fields:
            if token.kind != "citation" or token.manager != "none":
                continue
            works = token.data.get("works")
            if not isinstance(works, list):
                continue
            for raw in works:
                if not isinstance(raw, dict):
                    continue
                work = CitedWork.model_validate(raw)
                if resolve_work(work, entries).status != "missing":
                    continue
                reason = (
                    "no entry of the reference list names it"
                    if entries
                    else "the document has no reference list"
                )
                out.append(
                    Diagnostic(
                        check=self.spec.id,
                        tier=self.spec.tier,
                        severity="warning",
                        paragraph_id=token.paragraph_id,
                        start=token.start,
                        end=token.end,
                        message=f"Could not verify the citation {token.display!r}: {reason}.",
                        evidence=[token.display],
                        data={"kind": "unlisted", "surname": work.surname},
                    )
                )
        return out

    # ------------------------------------------------------------ works

    async def _verify(self, work: _Cited, http: ReferenceHttp) -> _Verdict:
        if work.entry is not None:
            resolution = await resolve_entry(work.entry, http=http)
        else:
            resolution = await resolve_cited_work(
                http=http,
                doi=work.doi,
                title=work.title,
                authors=list(work.authors),
                year=work.year,
            )
        doi = work.doi or (resolution.match.doi if resolution.match else "")
        notices: list[tuple[str, str, str]] = []
        retraction_error = ""
        if resolution.status in ("found", "details_differ") and doi:
            status = await lookup_retraction(doi, http=http)
            if status.status != "checked":
                retraction_error = status.detail or status.status
            elif not status.is_notice:
                notices = [(n.kind, n.source, n.notice_doi) for n in status.notices]
        return _Verdict(
            resolution=resolution,
            doi=doi,
            notices=notices,
            retraction_error=retraction_error,
        )

    def _work_diagnostics(
        self, work: _Cited, verdict: _Verdict, service_errors: list[str]
    ) -> list[Diagnostic]:
        out: list[Diagnostic] = []
        resolution = verdict.resolution
        label = verdict.doi or work.title or work.shown
        evidence = [work.shown]
        if resolution.status == "not_found":
            out.append(
                self._at(
                    work,
                    "warning",
                    f"Could not verify {label!r}: {resolution.detail or 'no matching work found'}.",
                    evidence + [resolution.detail],
                    {"kind": "unresolved", "doi": verdict.doi},
                )
            )
        elif resolution.status == "details_differ":
            out.append(
                self._at(
                    work,
                    "warning",
                    f"Could not verify the details of {label!r}: {resolution.detail}.",
                    evidence + [resolution.detail],
                    {"kind": "details_differ", "doi": verdict.doi},
                )
            )
        elif resolution.status == "service_error":
            service_errors.append(resolution.detail or "no detail")
        if verdict.retraction_error:
            out.append(
                self._at(
                    work,
                    "warning",
                    f"Retraction status of {label!r} could not be checked "
                    f"({verdict.retraction_error}).",
                    evidence + [verdict.retraction_error],
                    {"kind": "lookup_failed", "doi": verdict.doi},
                )
            )
        for kind, source, notice_doi in verdict.notices:
            severity: Severity = "error" if kind == "retraction" else "warning"
            name = kind.replace("_", " ")
            out.append(
                self._at(
                    work,
                    severity,
                    f"The cited work {label!r} has a {name} notice"
                    + (f" ({source})" if source else "")
                    + ".",
                    evidence + ([f"notice: {notice_doi}"] if notice_doi else []),
                    {"kind": kind, "doi": verdict.doi, "source": source},
                )
            )
        return out

    # ------------------------------------------------------------- urls

    async def _url_diagnostics(
        self, view: DocumentView, http: ReferenceHttp
    ) -> list[Diagnostic]:
        health: dict[str, UrlHealth] = {}
        out: list[Diagnostic] = []
        for url, paragraph_id, start, end in _urls(view):
            if url not in health:
                health[url] = await http.check_url(url)
            found = _url_diagnostic(health[url])
            if found is None:
                continue
            severity, message = found
            out.append(
                Diagnostic(
                    check=self.spec.id,
                    tier=self.spec.tier,
                    severity=severity,
                    paragraph_id=paragraph_id,
                    start=start,
                    end=end,
                    message=message,
                    evidence=[url, health[url].detail],
                    data={"kind": "url", "status": health[url].status},
                )
            )
        return out

    # ---------------------------------------------------------- builders

    def _at(
        self,
        work: _Cited,
        severity: Severity,
        message: str,
        evidence: list[str],
        data: dict[str, object],
    ) -> Diagnostic:
        return Diagnostic(
            check=self.spec.id,
            tier=self.spec.tier,
            severity=severity,
            paragraph_id=work.paragraph_id,
            start=work.start,
            end=work.end,
            message=message,
            evidence=[e for e in evidence if e],
            data=data,
        )

    def _diag(
        self,
        severity: Severity,
        message: str,
        *,
        evidence: list[str],
        data: dict[str, object],
    ) -> Diagnostic:
        return Diagnostic(
            check=self.spec.id,
            tier=self.spec.tier,
            severity=severity,
            paragraph_id=None,
            start=None,
            end=None,
            message=message,
            evidence=evidence,
            data=data,
        )


def _url_diagnostic(health: UrlHealth) -> tuple[Severity, str] | None:
    code = health.http_status
    match health.status:
        case "ok":
            return None
        case "not_found":
            return "warning", f"The URL does not exist (HTTP {code})."
        case "server_error":
            return "warning", f"The URL's server reports an error (HTTP {code})."
        case "dns_failure":
            return "warning", "The URL's host name does not resolve."
        case "timeout":
            return "warning", "The URL did not answer in time; could not verify it."
        case "client_error":
            return (
                "info",
                f"Could not verify the URL (HTTP {code}); many sites refuse robots.",
            )
        case "blocked":
            return "warning", f"The URL was not requested: {health.detail}."
        case _:
            return "warning", f"Could not verify the URL: {health.detail}."


register(CitationsCheck())
