"""Bibliographic enricher: the Crossref or PubMed record of a document."""

from __future__ import annotations

from typing import Any

from andamentum.references import (
    PubMedRecord,
    ReferenceHttp,
    extract_identifiers,
    fetch_pubmed,
    fetch_work,
)

import httpx

from ..store import Store
from ._base import EnrichError, EnrichResult, load, skipped, write_key

KEY = "bib"


def _pubmed_to_csl(record: PubMedRecord) -> dict[str, Any]:
    """A CSL-like dict from a PubMed record; empty fields are left out."""
    csl: dict[str, Any] = {
        "type": "article-journal",
        "title": record.title,
        "container-title": record.journal,
        "volume": record.volume,
        "issue": record.issue,
        "page": record.pages,
        "abstract": record.abstract,
        "DOI": record.doi,
        "PMID": record.pmid,
        "PMCID": record.pmcid,
    }
    if record.authors:
        csl["author"] = [{"literal": a} for a in record.authors]
    if record.year.isdigit():
        csl["issued"] = {"date-parts": [[int(record.year)]]}
    return {k: v for k, v in csl.items() if v}


async def enrich_bibliographic(
    store: Store,
    doc_id: str,
    *,
    http: ReferenceHttp,
    pubmed_client: httpx.AsyncClient | None = None,
    api_key: str | None = None,
) -> EnrichResult:
    """Write the bibliographic record under ``metadata["bib"]``.

    The identifier comes from ``metadata["identifiers"]`` when the identifiers
    enricher has run, else from the text. A DOI is looked up on Crossref
    through ``http``; with no DOI, a PMID is looked up on PubMed (``api_key``
    is a credential the caller passes; nothing is read from the environment).
    With neither, nothing is written and the result says so. A failed lookup
    raises (``CrossrefError``, ``httpx.HTTPStatusError``, :class:`EnrichError`).
    """
    doc = await load(store, doc_id)
    stored = doc.metadata.get("identifiers")
    if isinstance(stored, dict) and (stored.get("doi") or stored.get("pmid")):
        doi, pmid = stored.get("doi"), stored.get("pmid")
    else:
        found = extract_identifiers(doc.text)
        doi, pmid = found.doi, found.pmid

    if doi:
        csl = dict(await fetch_work(doi, http=http))
        value = {**csl, "source": "crossref", "id": doi}
    elif pmid:
        records = await fetch_pubmed([pmid], client=pubmed_client, api_key=api_key)
        if not records:
            raise EnrichError(f"PubMed has no record for PMID {pmid}")
        value = {**_pubmed_to_csl(records[0]), "source": "pubmed", "id": pmid}
    else:
        return skipped(doc, key=KEY, reason="skipped: no identifier")
    return await write_key(store, doc, key=KEY, value=value)
