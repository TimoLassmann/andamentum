"""Cochrane Library Evidence Provider.

Searches for Cochrane systematic reviews and meta-analyses via PubMed
E-utilities. Each review synthesizes findings from multiple randomized
controlled trials on a specific clinical question.

API: PubMed E-utilities with journal filter "Cochrane Database Syst Rev"
No authentication required.

Architecture: Layer 1 (standalone package)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..preflight import CheckResult

from andamentum.references import PubMedRecord, parse_pubmed_xml

from ..operations import GatheredEvidence

logger = logging.getLogger(__name__)

EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"


class CochraneProvider:
    """Evidence provider for Cochrane systematic reviews via PubMed E-utilities."""

    description = (
        "Cochrane systematic reviews and meta-analyses — the highest level of "
        "clinical evidence. Each review synthesizes findings from multiple "
        "randomized controlled trials on a specific clinical question. Use for "
        "any claim about clinical interventions, treatment effectiveness, drug "
        "safety, or public health interventions where a systematic review may "
        "exist. Example queries: 'exercise interventions for preventing falls "
        "in older adults', 'statins for primary prevention of cardiovascular "
        "disease', 'antibiotics for acute otitis media in children'."
    )

    query_guidance = (
        'The query goes to NCBI esearch with `AND "Cochrane Database Syst '
        'Rev"[Journal]` appended automatically. Full PubMed query syntax is '
        "supported (Boolean, MeSH, [tiab]/[ti]/[ab], [au], [pdat], phrase "
        "quoting, wildcards), but the corpus is already pre-filtered to "
        "Cochrane systematic reviews — keep queries shorter and more topic-"
        "focused than for PubMed.\n"
        "\n"
        "Query styles that all work:\n"
        "- Short topic phrase: metformin type 2 diabetes\n"
        "- Topic plus intervention: exercise falls older adults\n"
        '- MeSH-anchored: "Metformin"[MeSH] AND "Diabetes Mellitus, Type 2"[MeSH]\n'
        '- Title-tagged: "metformin"[ti] AND "diabetes"[ti]\n'
        "- Topic plus outcome: aspirin myocardial infarction primary prevention\n"
        "- Drug class plus condition: SGLT2 inhibitor heart failure\n"
        "\n"
        "Adding [pt] filters for systematic reviews is redundant — every result "
        "is one. 3–7 tokens is usually optimal. The `site:` operator does not "
        "work."
    )

    query_examples: list[tuple[str, str | None]] = [
        (
            "effectiveness of metformin in type 2 diabetes",
            "metformin type 2 diabetes",
        ),
        (
            "exercise interventions for preventing falls in older adults",
            "exercise falls older adults",
        ),
        (
            "SGLT2 inhibitors for heart failure outcomes",
            "SGLT2 inhibitor heart failure",
        ),
        (
            "statins for primary cardiovascular prevention",
            '"Hydroxymethylglutaryl-CoA Reductase Inhibitors"[MeSH] AND "primary prevention"',
        ),
        (
            "antibiotics for acute otitis media in children",
            "antibiotics acute otitis media children",
        ),
        # Out-of-domain — basic biology, no clinical question
        (
            "molecular mechanism of histone modification in chromatin remodeling",
            None,
        ),
        # Out-of-domain — drug compound metadata
        (
            "SMILES structure of remdesivir",
            None,
        ),
        # Out-of-domain — gene-phenotype association
        (
            "BRCA1 mutations and breast cancer risk",
            None,
        ),
    ]
    output_kind = "assertion_evidence"
    independence_group = "systematic_reviews"
    provider_contract_version = 1

    def __init__(self, max_results: int = 10):
        self.max_results = max_results

    async def check_health(self) -> "CheckResult":
        """Test PubMed E-utilities reachability with Cochrane journal filter."""
        import time

        import httpx

        from ..preflight import CheckResult

        t0 = time.monotonic()
        try:
            params: dict[str, Any] = {
                "db": "pubmed",
                "term": '"Cochrane Database Syst Rev"[Journal]',
                "retmax": "1",
                "retmode": "json",
            }

            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(
                    f"{EUTILS_BASE}/esearch.fcgi", params=params
                )
                elapsed = (time.monotonic() - t0) * 1000
                if response.status_code == 200:
                    data = response.json()
                    count = int(data.get("esearchresult", {}).get("count", "0"))
                    if count > 0:
                        return CheckResult(
                            name="CochraneProvider",
                            status="pass",
                            message=f"API reachable ({elapsed:.0f}ms)",
                            elapsed_ms=elapsed,
                        )
                    return CheckResult(
                        name="CochraneProvider",
                        status="fail",
                        message="No Cochrane reviews found in PubMed",
                        elapsed_ms=elapsed,
                    )
                return CheckResult(
                    name="CochraneProvider",
                    status="fail",
                    message=f"HTTP {response.status_code}",
                    elapsed_ms=elapsed,
                )
        except Exception as e:
            elapsed = (time.monotonic() - t0) * 1000
            return CheckResult(
                name="CochraneProvider",
                status="fail",
                message=str(e),
                elapsed_ms=elapsed,
            )

    async def gather(self, query: str) -> list[GatheredEvidence]:
        """Search PubMed for Cochrane systematic reviews matching the query."""
        import httpx

        gathered: list[GatheredEvidence] = []

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                # Step 1: Search for PMIDs filtered to Cochrane reviews
                search_params: dict[str, Any] = {
                    "db": "pubmed",
                    "term": f'{query} AND "Cochrane Database Syst Rev"[Journal]',
                    "retmax": str(self.max_results),
                    "retmode": "json",
                    "sort": "relevance",
                }

                search_resp = await client.get(
                    f"{EUTILS_BASE}/esearch.fcgi", params=search_params
                )
                if search_resp.status_code != 200:
                    return []

                search_data = search_resp.json()
                pmids = search_data.get("esearchresult", {}).get("idlist", [])
                if not pmids:
                    return []

                # Step 2: Fetch article details as XML
                fetch_params: dict[str, Any] = {
                    "db": "pubmed",
                    "id": ",".join(pmids),
                    "rettype": "abstract",
                    "retmode": "xml",
                }

                fetch_resp = await client.get(
                    f"{EUTILS_BASE}/efetch.fcgi", params=fetch_params
                )
                if fetch_resp.status_code != 200:
                    return []

                gathered = self._parse_articles(fetch_resp.text)

        except Exception as e:
            logger.warning(f"CochraneProvider query failed for '{query}': {e}")

        return gathered

    def _parse_articles(self, xml_text: str) -> list[GatheredEvidence]:
        """Parse PubMed XML response into GatheredEvidence items."""
        try:
            records = parse_pubmed_xml(xml_text)
        except ValueError:
            return []
        gathered: list[GatheredEvidence] = []
        for record in records:
            result = self._to_evidence(record)
            if result is not None:
                gathered.append(result)
        return gathered

    def _to_evidence(self, record: PubMedRecord) -> GatheredEvidence | None:
        """Wrap one parsed PubMed record as GatheredEvidence."""
        pmid, title, authors, doi = (
            record.pmid,
            record.title,
            record.authors,
            record.doi,
        )
        if not title:
            return None

        # Structured abstract sections
        abstract_sections: dict[str, str] = {}
        content_parts: list[str] = [title]
        if authors:
            if len(authors) > 5:
                content_parts.append(
                    f"Authors: {', '.join(authors[:5])} (et al, {len(authors)} authors total)"
                )
            else:
                content_parts.append(f"Authors: {', '.join(authors)}")
        for label, text in record.abstract_sections:
            if label:
                abstract_sections[label] = text
                content_parts.append(f"{label}: {text}")
            else:
                content_parts.append(text)

        pub_date = record.year

        # Build identifiers
        identifiers: dict[str, str] = {}
        if pmid:
            identifiers["pmid"] = pmid
        if doi:
            identifiers["doi"] = doi

        # Source reference
        source_ref = f"doi:{doi}" if doi else f"PMID:{pmid}"

        return GatheredEvidence(
            content="\n".join(content_parts),
            source_ref=source_ref,
            source_type="cochrane",
            evidence_kind="systematic_review",
            identifiers=identifiers,
            structured_data={
                "title": title,
                "authors": authors,
                "pub_date": pub_date,
                "abstract_sections": abstract_sections,
            },
            quality_score=None,
            quality_metadata={
                "journal": "Cochrane Database Syst Rev",
                "publication_types": ["Systematic Review"],
            },
            limitations=[],
        )
