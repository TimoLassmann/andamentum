"""Monarch Initiative Evidence Provider.

Provides structured gene-disease association data from the Monarch Initiative
(https://monarchinitiative.org/), which aggregates curated data from multiple
biomedical databases.

REST API: https://api-v3.monarchinitiative.org/v3/api/

Architecture: Layer 1 (standalone package)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..preflight import CheckResult

from ..operations import GatheredEvidence

logger = logging.getLogger(__name__)

# Monarch API base URL
MONARCH_API = "https://api-v3.monarchinitiative.org/v3/api"

# Association categories worth fetching (disease/phenotype focus).
# Gene-gene interactions (PairwiseGeneToGeneInteraction, BioGRID) are
# deliberately excluded -- the epistemic system verifies claims, not
# protein interaction networks.
_ASSOCIATION_CATEGORIES = [
    "biolink:CausalGeneToDiseaseAssociation",
    "biolink:CorrelatedGeneToDiseaseAssociation",
    "biolink:GeneToPhenotypicFeatureAssociation",
    "biolink:DiseaseToPhenotypicFeatureAssociation",
]

# Map Monarch category URIs to short evidence_kind strings.
_CATEGORY_TO_KIND: dict[str, str] = {
    "biolink:CausalGeneToDiseaseAssociation": "causal_gene_disease",
    "biolink:CorrelatedGeneToDiseaseAssociation": "correlated_gene_disease",
    "biolink:GeneToPhenotypicFeatureAssociation": "gene_phenotype",
    "biolink:DiseaseToPhenotypicFeatureAssociation": "disease_phenotype",
}


class MonarchProvider:
    """Evidence provider using Monarch Initiative for gene-disease associations.

    Gathers curated gene-disease and gene-phenotype associations with rich
    metadata including knowledge source, evidence codes (ECO), and publications.
    Quality score is always None -- the system's quality agent assesses quality,
    not the provider.
    No auth required. No documented rate limit.
    """

    description = (
        "Curated gene–disease and gene–phenotype associations aggregated from "
        "OMIM, HPO (Human Phenotype Ontology), Orphanet, ClinVar, and model organism "
        "databases by the Monarch Initiative. Best for questions about which genes "
        "are linked to which diseases, phenotype-driven rare disease diagnosis, "
        "variant–disease significance, and cross-species orthology of disease genes. "
        "Example queries: 'genes associated with hypertrophic cardiomyopathy', "
        "'phenotypes caused by COL1A1 mutations', 'rare diseases linked to mitochondrial "
        "complex I deficiency', 'clinical significance of BRCA1 c.5266dupC variant'."
    )

    query_guidance = (
        "The query goes to Monarch's `/search` `q` parameter. Accepts: gene "
        "symbols, disease names, phenotype terms, ontology IDs (HGNC:, MONDO:, "
        "HP:, OMIM:, ORPHA:), and variants.\n"
        "\n"
        "Query styles that all work:\n"
        "- Gene symbol: BRCA1\n"
        "- Gene plus disease: BRCA1 breast cancer\n"
        "- Disease name: cystic fibrosis\n"
        "- HPO phenotype term: intellectual disability\n"
        "- MONDO disease ID: MONDO:0008029\n"
        "- HGNC gene ID: HGNC:1100\n"
        "- HPO phenotype ID: HP:0001263\n"
        "- Variant: c.5266dupC BRCA1\n"
        "\n"
        "Monarch is curated gene-disease-phenotype association data, not "
        "literature. Use only for 'what genes are linked to X' or 'what "
        "phenotypes does mutation in Y cause'. 1–3 token queries are optimal."
    )

    query_examples: list[tuple[str, str | None]] = []
    output_kind = "structured_record"
    independence_group = "genetics_structured"
    provider_contract_version = 1

    def __init__(self, max_results: int = 10):
        self.max_results = max_results

    async def check_health(self) -> "CheckResult":
        """Test Monarch API reachability."""
        import time

        import httpx

        from ..preflight import CheckResult

        t0 = time.monotonic()
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                # Use the same params as production _search() so preflight catches real failures
                response = await client.get(
                    f"{MONARCH_API}/search",
                    params=[
                        ("q", "BRCA1"),
                        ("limit", "1"),
                        ("category", "biolink:Gene"),
                        ("category", "biolink:Disease"),
                    ],
                )
                elapsed = (time.monotonic() - t0) * 1000
                if response.status_code == 200:
                    return CheckResult(
                        name="MonarchProvider",
                        status="pass",
                        message=f"API reachable ({elapsed:.0f}ms)",
                        elapsed_ms=elapsed,
                    )
                return CheckResult(
                    name="MonarchProvider",
                    status="fail",
                    message=f"HTTP {response.status_code}",
                    elapsed_ms=elapsed,
                )
        except Exception as e:
            elapsed = (time.monotonic() - t0) * 1000
            return CheckResult(
                name="MonarchProvider",
                status="fail",
                message=str(e),
                elapsed_ms=elapsed,
            )

    async def gather(self, query: str) -> list[GatheredEvidence]:
        """Search Monarch for gene-disease and gene-phenotype associations.

        1. Search for entities matching *query* (genes and diseases).
        2. For each entity ID, fetch associations filtered to disease/phenotype
           categories (excluding gene-gene interactions).
        3. Build rich GatheredEvidence from each association.

        Args:
            query: Natural language query (e.g., "BRCA1 breast cancer")

        Returns:
            List of GatheredEvidence with structured association data
        """
        import httpx

        gathered: list[GatheredEvidence] = []

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                # Step 1: Search for entities to get IDs
                search_results = await self._search(client, query)
                gathered.extend(search_results)

                # Step 2: Fetch associations for each entity
                entity_ids = self._extract_entity_ids(search_results, query)
                entity_total = len(entity_ids)
                for entity_id in entity_ids[:3]:
                    assoc_results = await self._get_associations(client, entity_id)
                    # Surface total entity count in quality_metadata when truncated
                    if entity_total > 3:
                        for item in assoc_results:
                            if item.quality_metadata is not None:
                                item.quality_metadata["entity_total"] = entity_total
                            else:
                                item.quality_metadata = {"entity_total": entity_total}
                    gathered.extend(assoc_results)
                if entity_total > 3:
                    logger.debug(
                        "Monarch: showing associations for 3 of %d entities",
                        entity_total,
                    )

        except Exception as e:
            # Honest empty-result on API failure. The previous behaviour
            # was to return a single phantom GatheredEvidence with
            # ``source_ref=query`` (the search query string) and
            # synthesised content "Monarch Initiative search failed: ...".
            # Per the provider contract (see CONTRIBUTING.md) and the
            # Strategy 2 deletion in ExtractEvidenceOperation (commit
            # ee4bbb8), providers must return ``[]`` on error rather than
            # placeholder content — placeholder evidence pollutes the
            # downstream evidence pool with non-paper content that the
            # judge cannot meaningfully evaluate.
            logger.warning(f"Monarch Initiative query failed for '{query}': {e}")

        return gathered[: self.max_results]

    async def _search(self, client: Any, query: str) -> list[GatheredEvidence]:
        """Search Monarch for entities matching a query."""
        try:
            response = await client.get(
                f"{MONARCH_API}/search",
                params=[
                    ("q", query),
                    ("limit", str(self.max_results)),
                    ("category", "biolink:Gene"),
                    ("category", "biolink:Disease"),
                ],
            )

            if response.status_code != 200:
                logger.debug(f"Monarch search returned {response.status_code}")
                return []

            data = response.json()
            items = data.get("items", [])
            gathered: list[GatheredEvidence] = []

            for item in items:
                name = item.get("name", "")
                category = item.get("category", "")
                description = item.get("description", "")
                item_id = item.get("id", "")

                if not name:
                    continue

                content_parts = [f"**{name}** ({category})"]
                if description:
                    content_parts.append(description)
                if item_id:
                    content_parts.append(f"ID: {item_id}")

                gathered.append(
                    GatheredEvidence(
                        content="\n".join(content_parts),
                        source_ref=(
                            f"https://monarchinitiative.org/{item_id}"
                            if item_id
                            else name
                        ),
                        source_type="monarch",
                        evidence_kind="entity_metadata",
                        identifiers={"monarch_id": item_id} if item_id else {},
                        structured_data={
                            "name": name,
                            "category": category,
                            "description": description,
                        },
                        quality_score=None,
                        quality_metadata={
                            "source": "monarch_initiative",
                            "entity_id": item_id,
                            "category": category,
                        },
                        limitations=[
                            "Entity metadata only; see associations for evidence details"
                        ],
                    )
                )

            return gathered

        except Exception as e:
            logger.debug(f"Monarch search failed: {e}")
            return []

    async def _get_associations(
        self, client: Any, entity_id: str
    ) -> list[GatheredEvidence]:
        """Get disease/phenotype associations for a specific entity.

        Fetches associations filtered to causal/correlated gene-disease and
        gene/disease-phenotype categories.  Gene-gene interactions
        (PairwiseGeneToGeneInteraction) are excluded via query-parameter
        filtering.
        """
        try:
            # Build params with category filtering
            params: list[tuple[str, str]] = [
                ("subject", entity_id),
                ("limit", str(self.max_results)),
            ]
            for cat in _ASSOCIATION_CATEGORIES:
                params.append(("category", cat))

            response = await client.get(
                f"{MONARCH_API}/association",
                params=params,
            )

            if response.status_code != 200:
                # Try as object instead of subject
                params_obj: list[tuple[str, str]] = [
                    ("object", entity_id),
                    ("limit", str(self.max_results)),
                ]
                for cat in _ASSOCIATION_CATEGORIES:
                    params_obj.append(("category", cat))

                response = await client.get(
                    f"{MONARCH_API}/association",
                    params=params_obj,
                )
                if response.status_code != 200:
                    return []

            data = response.json()
            items = data.get("items", [])
            gathered: list[GatheredEvidence] = []

            for assoc in items:
                evidence = self._parse_association(assoc, entity_id)
                if evidence is not None:
                    gathered.append(evidence)

            return gathered

        except Exception as e:
            logger.debug(f"Monarch association query failed for {entity_id}: {e}")
            return []

    def _parse_association(
        self, assoc: dict[str, Any], entity_id: str
    ) -> GatheredEvidence | None:
        """Parse a single association item into GatheredEvidence.

        Handles both the real Monarch v3 API format (flat fields: subject,
        subject_label, object, object_label) and dict-based formats used in
        tests.
        """
        # Extract subject/object -- handle both flat and nested dict formats
        raw_subject = assoc.get("subject", "")
        raw_object = assoc.get("object", "")

        if isinstance(raw_subject, dict):
            subject_name = raw_subject.get("name", raw_subject.get("id", "unknown"))
            subject_id = raw_subject.get("id", "")
        else:
            subject_id = str(raw_subject) if raw_subject else ""
            subject_name = assoc.get("subject_label", subject_id)

        if isinstance(raw_object, dict):
            object_name = raw_object.get("name", raw_object.get("id", "unknown"))
            object_id = raw_object.get("id", "")
        else:
            object_id = str(raw_object) if raw_object else ""
            object_name = assoc.get("object_label", object_id)

        predicate = assoc.get("predicate", "associated_with")
        category = assoc.get("category", "")
        primary_knowledge_source = assoc.get("primary_knowledge_source", "")
        knowledge_level = assoc.get("knowledge_level", "")
        evidence_count = assoc.get("evidence_count", 0)
        has_evidence = assoc.get("has_evidence", []) or []
        publications = assoc.get("publications", []) or []
        publications_links = assoc.get("publications_links", []) or []
        negated = assoc.get("negated", False)

        # Also accept legacy test field
        evidence_types = assoc.get("evidence_types", []) or []
        if not has_evidence and evidence_types:
            has_evidence = evidence_types

        # Build rich content
        content_parts: list[str] = []

        # Main assertion
        predicate_display = predicate.replace("biolink:", "").replace("_", " ")
        content_parts.append(f"{subject_name} {predicate_display} {object_name}")

        # Knowledge source context
        if primary_knowledge_source:
            source_short = (
                primary_knowledge_source.split(":")[-1]
                if ":" in primary_knowledge_source
                else primary_knowledge_source
            )
            knowledge_desc = ""
            if knowledge_level:
                knowledge_desc = f", {knowledge_level.replace('_', ' ')}"
            content_parts[0] += f" ({source_short}{knowledge_desc})"

        # Publications
        pmids = [p for p in publications if isinstance(p, str) and "PMID" in p.upper()]
        if pmids:
            suffix = f" ({len(pmids)} total)" if len(pmids) > 5 else ""
            content_parts.append(f"Publications: {', '.join(pmids[:5])}{suffix}")

        # Evidence type (ECO codes)
        if has_evidence:
            eco_strs = [str(e) for e in has_evidence]
            content_parts.append(f"Evidence type: {', '.join(eco_strs)}")

        content = ". ".join(content_parts) + "."

        # Determine source_ref -- prefer PubMed URL
        source_ref = ""
        if publications_links:
            # Pick first PubMed link
            for link in publications_links:
                link_url = link.get("url", "") if isinstance(link, dict) else str(link)
                if "pubmed" in link_url.lower() or "ncbi" in link_url.lower():
                    source_ref = link_url
                    break
            if not source_ref:
                first = publications_links[0]
                source_ref = (
                    first.get("url", str(first))
                    if isinstance(first, dict)
                    else str(first)
                )
        if not source_ref and pmids:
            pmid_num = pmids[0].replace("PMID:", "").strip()
            source_ref = f"https://pubmed.ncbi.nlm.nih.gov/{pmid_num}"
        if not source_ref:
            source_ref = (
                f"https://monarchinitiative.org/{subject_id}"
                if subject_id
                else f"monarch:{entity_id}"
            )

        # Map category to evidence_kind
        evidence_kind = _CATEGORY_TO_KIND.get(category, "genetic_association")

        # Extract PMID for identifiers
        identifiers: dict[str, str] = {}
        if subject_id:
            identifiers["subject_id"] = subject_id
        if object_id:
            identifiers["object_id"] = object_id
        if pmids:
            identifiers["pmid"] = pmids[0]

        # Structured data preserving API fields
        structured_data: dict[str, Any] = {
            "predicate": predicate,
            "category": category,
            "primary_knowledge_source": primary_knowledge_source,
            "knowledge_level": knowledge_level,
            "evidence_count": evidence_count,
            "has_evidence": has_evidence,
            "negated": negated,
        }

        # Quality metadata for the quality agent
        quality_metadata: dict[str, Any] = {
            "primary_knowledge_source": primary_knowledge_source,
            "knowledge_level": knowledge_level,
            "evidence_count": evidence_count,
            "predicate": predicate,
            "category": category,
        }

        # Limitations
        limitations: list[str] = []
        if evidence_count == 0 and not publications:
            limitations.append("No supporting publications cited")
        if negated:
            limitations.append("This is a NEGATED association")
        if knowledge_level == "not_provided":
            limitations.append("Knowledge level not provided by source")

        return GatheredEvidence(
            content=content,
            source_ref=source_ref,
            source_type="monarch",
            evidence_kind=evidence_kind,
            identifiers=identifiers,
            structured_data=structured_data,
            quality_score=None,
            quality_metadata=quality_metadata,
            limitations=limitations,
        )

    @staticmethod
    def _extract_entity_ids(
        search_results: list[GatheredEvidence], query: str
    ) -> list[str]:
        """Extract entity IDs from search results for association lookup."""
        ids: list[str] = []
        for result in search_results:
            if result.quality_metadata:
                eid = result.quality_metadata.get("entity_id", "")
                if eid:
                    ids.append(eid)
        return ids
