"""Verify a work before it can be cited.

Two stages, both through ``andamentum.references``: the work must exist and its
details must agree (``resolve_cited_work``: a DOI is fetched, a work without
one is searched by title and authors; a returned record counts only when its
title is the cited one), and it must not be retracted (``lookup_retraction``).
Only a :class:`VerifiedWork` may enter a document. Everything else is an
:class:`Unverified` carrying the reason, which becomes a question for the
author. A lookup that could not be done is ``unchecked``, never a pass.
"""

from __future__ import annotations

from andamentum.references import ReferenceHttp, lookup_retraction, resolve_cited_work

from .models import Unverified, VerifiedWork, Work


async def resolve(work: Work, *, http: ReferenceHttp) -> VerifiedWork | Unverified:
    """Does *work* exist, agree with its record, and stand unretracted?"""
    found = await resolve_cited_work(
        http=http,
        doi=work.doi,
        title=work.title,
        authors=work.authors,
        year=work.year,
    )
    if found.status == "not_found":
        return Unverified(
            work=work,
            kind="not_found",
            reason=found.detail or "no such work in Crossref",
        )
    if found.status == "details_differ":
        return Unverified(work=work, kind="details_differ", reason=found.detail)
    if found.status == "service_error" or found.match is None:
        return Unverified(
            work=work, kind="unchecked", reason=f"could not be checked: {found.detail}"
        )
    doi = found.match.doi or work.doi
    if not doi:
        return Unverified(
            work=work,
            kind="unchecked",
            reason="the matching record has no DOI to check for a retraction",
        )
    status = await lookup_retraction(doi, http=http)
    if status.status != "checked":
        return Unverified(
            work=work,
            kind="unchecked",
            reason=f"retraction status unknown: {status.detail}",
        )
    if status.retracted:
        return Unverified(
            work=work,
            kind="retracted",
            reason=f"{doi} has been retracted",
            notices=status.notices,
        )
    checked = work
    if not work.doi:
        checked = work.model_copy(update={"csl": {**work.csl, "DOI": doi}})
    return VerifiedWork(work=checked, notices=status.notices)
