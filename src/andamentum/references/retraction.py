"""Has a cited work been retracted, corrected, or flagged?

Crossref records, on the updated work, the notices that update it
(``updated-by``: each with the notice's DOI, a ``type`` and a ``source`` —
``retraction-watch`` or ``publisher``). The notice carries the reverse link
(``update-to``). One lookup by the cited work's DOI reads the first; a DOI
that is itself a notice is reported as such, from its ``update-to``.

Only retraction, expression of concern and correction are reported; other
update types (new versions, addenda) are not a reason to doubt the work.
Crossref's own type names are folded: withdrawal reads as a retraction,
erratum and corrigendum as a correction.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from .crossref import get_work
from .works import normalise_doi
from .http import ReferenceHttp

NoticeKind = Literal["retraction", "expression_of_concern", "correction"]

_KINDS: dict[str, NoticeKind] = {
    "retraction": "retraction",
    "withdrawal": "retraction",
    "removal": "retraction",
    "expression_of_concern": "expression_of_concern",
    "expression-of-concern": "expression_of_concern",
    "correction": "correction",
    "erratum": "correction",
    "corrigendum": "correction",
}


class UpdateNotice(BaseModel):
    kind: NoticeKind
    notice_doi: str = ""  # the DOI of the retraction notice, when given
    source: str = ""  # "retraction-watch", "publisher", ...
    label: str = ""
    date: str = ""  # ISO date as far as Crossref gives it


class RetractionStatus(BaseModel):
    """What Crossref says about updates to one DOI.

    ``service_error`` and ``not_found`` mean the DOI was not checked; neither
    says the work is sound.
    """

    doi: str
    status: Literal["checked", "not_found", "service_error"]
    notices: list[UpdateNotice] = Field(default_factory=list)
    is_notice: bool = False  # the DOI is itself a notice of another work
    detail: str = ""

    @property
    def retracted(self) -> bool:
        return any(n.kind == "retraction" for n in self.notices)


async def lookup_retraction(doi: str, *, http: ReferenceHttp) -> RetractionStatus:
    """Update notices Crossref holds for *doi*."""
    doi = normalise_doi(doi) or doi.strip()
    result = await get_work(doi, http=http)
    if isinstance(result, str):
        return RetractionStatus(doi=doi, status="service_error", detail=result)
    code, body = result
    if code == 404:
        return RetractionStatus(
            doi=doi, status="not_found", detail=f"the DOI {doi} does not resolve"
        )
    message = body.get("message", {})
    notices = [n for n in map(_notice, message.get("updated-by") or []) if n]
    own = [n for n in map(_notice, message.get("update-to") or []) if n]
    return RetractionStatus(
        doi=doi,
        status="checked",
        notices=notices,
        is_notice=bool(own),
    )


def _notice(item: dict) -> UpdateNotice | None:
    kind = _KINDS.get(str(item.get("type", "")).lower())
    if kind is None:
        return None
    updated = item.get("updated") or {}
    parts = (updated.get("date-parts") or [[]])[0]
    date = "-".join(f"{p:02d}" if i else str(p) for i, p in enumerate(parts))
    return UpdateNotice(
        kind=kind,
        notice_doi=str(item.get("DOI", "")),
        source=str(item.get("source", "")),
        label=str(item.get("label", "")),
        date=date,
    )
