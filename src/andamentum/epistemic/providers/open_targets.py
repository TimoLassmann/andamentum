"""Open Targets Evidence Provider.

Queries the Open Targets Platform GraphQL API for individual evidence
items linking targets to diseases. Returns per-evidence data with
literature references, extracted text, confidence levels, and data
source attribution — not just aggregate association scores.

Data sources integrated (as of Platform release 26.03):
  europepmc (literature mining), eva (ClinVar genetic), eva_somatic,
  intogen (cancer drivers), cancer_gene_census, cancer_biomarkers,
  genomics_england (GEL panels), gene2phenotype, clingen,
  orphanet, uniprot_variants, uniprot_literature, impc (mouse models),
  clinical_precedence (drug evidence).

API docs: https://platform-docs.opentargets.org/data-access/graphql-api
No authentication required.

Architecture: Layer 1 (standalone package)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..preflight import CheckResult

from ..operations import GatheredEvidence

logger = logging.getLogger(__name__)

OT_API = "https://api.platform.opentargets.org/api/v4/graphql"

# Map Open Targets datatypeId to evidence_kind values used by the
# epistemic system. These align with the GatheredEvidence.evidence_kind
# vocabulary documented in operations/base.py.
_DATATYPE_TO_KIND: dict[str, str] = {
    "literature": "literature_mining",
    "genetic_association": "genetic_evidence",
    "somatic_mutation": "somatic_evidence",
    "animal_model": "animal_model",
    "clinical": "clinical_evidence",
    "genetic_literature": "curated_genetic",
}


class OpenTargetsProvider:
    """Evidence provider using Open Targets Platform GraphQL API.

    Queries individual evidence items (not just aggregate scores) so the
    epistemic system can judge each piece of evidence independently.
    """

    description = (
        "Integrated drug target evidence from the Open Targets Platform, combining "
        "genetic associations (GWAS), somatic mutations (cancer), literature co-mentions, "
        "pathway membership, drug-target interactions, tractability, and "
        "target-disease association scores. Best for questions about which proteins "
        "or genes are therapeutic targets for a given disease, drug repurposing "
        "opportunities, pathway-level target evaluation, and druggability assessment. "
        "Example queries: 'therapeutic targets for Alzheimer's disease with genetic "
        "support', 'druggable targets in KRAS-mutant colorectal cancer', 'pathway "
        "evidence linking TNF signaling to rheumatoid arthritis', 'target tractability "
        "for PCSK9 in cardiovascular disease'."
    )

    query_guidance = (
        "The query goes to Open Targets GraphQL `search(queryString: $q)`. "
        "Accepts: target names, gene symbols, disease names, pathway names, "
        "ontology IDs (ENSG, MONDO, EFO, HP).\n"
        "\n"
        "Query styles that all work:\n"
        "- Gene symbol: KRAS\n"
        "- Disease name: Alzheimer's disease\n"
        "- Disease plus qualifier: idiopathic pulmonary fibrosis\n"
        "- Ensembl gene ID: ENSG00000133703\n"
        "- EFO disease ID: EFO_0000270\n"
        "- Pathway name: TNF signaling\n"
        "- Drug-disease pair: tofacitinib rheumatoid arthritis\n"
        "\n"
        "Returns target-disease associations, druggability, GWAS signal, and "
        "drug-target interactions. Not for literature. 1–3 token queries are "
        "optimal."
    )

    query_examples: list[tuple[str, str | None]] = []
    output_kind = "structured_record"
    independence_group = "genetics_structured"
    provider_contract_version = 1

    def __init__(self, max_results: int = 10):
        self.max_results = max_results

    async def check_health(self) -> "CheckResult":
        """Test Open Targets API reachability."""
        import time

        import httpx

        from ..preflight import CheckResult

        t0 = time.monotonic()
        try:
            query = "{ meta { dataVersion { year month } } }"
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(OT_API, json={"query": query})
                elapsed = (time.monotonic() - t0) * 1000
                if response.status_code == 200:
                    data = response.json().get("data", {})
                    version = data.get("meta", {}).get("dataVersion", {})
                    ver_str = f"{version.get('year', '?')}.{version.get('month', '?')}"
                    return CheckResult(
                        name="OpenTargetsProvider",
                        status="pass",
                        message=f"API reachable, data v{ver_str} ({elapsed:.0f}ms)",
                        elapsed_ms=elapsed,
                    )
                return CheckResult(
                    name="OpenTargetsProvider",
                    status="fail",
                    message=f"HTTP {response.status_code}",
                    elapsed_ms=elapsed,
                )
        except Exception as e:
            elapsed = (time.monotonic() - t0) * 1000
            return CheckResult(
                name="OpenTargetsProvider",
                status="fail",
                message=str(e),
                elapsed_ms=elapsed,
            )

    async def gather(self, query: str) -> list[GatheredEvidence]:
        """Search Open Targets and fetch individual evidence items."""
        import httpx

        gathered: list[GatheredEvidence] = []

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                # Step 1: Search for entities (targets and diseases)
                search_results = await self._search_entities(client, query)

                targets = [h for h in search_results if h.get("entity") == "target"]
                diseases = [h for h in search_results if h.get("entity") == "disease"]

                # Step 2: For target hits, get evidence across their top diseases
                for target in targets[: self.max_results]:
                    target_id = target.get("id", "")
                    # Get top associated disease IDs for this target
                    disease_ids = await self._get_top_disease_ids(client, target_id)
                    if disease_ids:
                        items = await self._get_evidence_items(
                            client, target_id, disease_ids
                        )
                        gathered.extend(items)

                # Step 3: For disease hits, get evidence across their top targets
                for disease in diseases[: self.max_results]:
                    disease_id = disease.get("id", "")
                    target_ids = await self._get_top_target_ids(client, disease_id)
                    target_total = len(target_ids)
                    for tid in target_ids[:3]:
                        items = await self._get_evidence_items(
                            client, tid, [disease_id]
                        )
                        # Surface total target count in each item's quality_metadata
                        for item in items:
                            if item.quality_metadata is not None:
                                item.quality_metadata["target_total"] = target_total
                            else:
                                item.quality_metadata = {"target_total": target_total}
                        gathered.extend(items)
                    if target_total > 3:
                        logger.debug(
                            "Disease %s: showing 3 of %d targets",
                            disease_id,
                            target_total,
                        )

        except Exception as e:
            logger.warning(f"Open Targets query failed for '{query}': {e}")

        return gathered[: self.max_results]

    async def _search_entities(self, client: Any, query: str) -> list[dict[str, Any]]:
        """Search Open Targets for targets or diseases matching query."""
        gql = """
        query Search($q: String!, $size: Int!) {
            search(queryString: $q, page: {size: $size, index: 0}) {
                hits {
                    id
                    entity
                    name
                    description
                }
            }
        }
        """
        try:
            resp = await client.post(
                OT_API,
                json={
                    "query": gql,
                    "variables": {"q": query, "size": self.max_results},
                },
            )
            if resp.status_code != 200:
                return []
            data = resp.json()
            return data.get("data", {}).get("search", {}).get("hits", [])
        except Exception as e:
            logger.debug(f"Open Targets search failed: {e}")
            return []

    async def _get_top_disease_ids(
        self, client: Any, target_id: str, n: int = 5
    ) -> list[str]:
        """Get top-N associated disease IDs for a target."""
        gql = """
        query TopDiseases($id: String!, $size: Int!) {
            target(ensemblId: $id) {
                associatedDiseases(page: {size: $size, index: 0}) {
                    rows { disease { id } }
                }
            }
        }
        """
        try:
            resp = await client.post(
                OT_API,
                json={"query": gql, "variables": {"id": target_id, "size": n}},
            )
            if resp.status_code != 200:
                return []
            data = resp.json()
            rows = (
                data.get("data", {})
                .get("target", {})
                .get("associatedDiseases", {})
                .get("rows", [])
            )
            return [r["disease"]["id"] for r in rows if r.get("disease", {}).get("id")]
        except Exception:
            return []

    async def _get_top_target_ids(
        self, client: Any, disease_id: str, n: int = 3
    ) -> list[str]:
        """Get top-N associated target IDs for a disease."""
        gql = """
        query TopTargets($id: String!, $size: Int!) {
            disease(efoId: $id) {
                associatedTargets(page: {size: $size, index: 0}) {
                    rows { target { id } }
                }
            }
        }
        """
        try:
            resp = await client.post(
                OT_API,
                json={"query": gql, "variables": {"id": disease_id, "size": n}},
            )
            if resp.status_code != 200:
                return []
            data = resp.json()
            rows = (
                data.get("data", {})
                .get("disease", {})
                .get("associatedTargets", {})
                .get("rows", [])
            )
            return [r["target"]["id"] for r in rows if r.get("target", {}).get("id")]
        except Exception:
            return []

    async def _get_evidence_items(
        self,
        client: Any,
        target_id: str,
        disease_ids: list[str],
    ) -> list[GatheredEvidence]:
        """Fetch individual evidence items for a target-disease pair."""
        gql = """
        query Evidence($ensemblId: String!, $efoIds: [String!]!, $size: Int!) {
            target(ensemblId: $ensemblId) {
                approvedSymbol
                approvedName
                evidences(efoIds: $efoIds, size: $size) {
                    rows {
                        id
                        datasourceId
                        datatypeId
                        score
                        diseaseFromSource
                        targetFromSource
                        literature
                        publicationYear
                        publicationFirstAuthor
                        confidence
                        textMiningSentences {
                            text
                            section
                        }
                        urls {
                            url
                            niceName
                        }
                        variantRsId
                        studyId
                        clinicalStage
                        drugFromSource
                    }
                }
            }
        }
        """
        gathered: list[GatheredEvidence] = []

        try:
            resp = await client.post(
                OT_API,
                json={
                    "query": gql,
                    "variables": {
                        "ensemblId": target_id,
                        "efoIds": disease_ids,
                        "size": self.max_results * 3,
                    },
                },
            )
            if resp.status_code != 200:
                return []

            data = resp.json()
            target = data.get("data", {}).get("target")
            if not target:
                return []

            target_symbol = target.get("approvedSymbol", "")
            target_name = target.get("approvedName", "")
            rows = target.get("evidences", {}).get("rows", [])

            for row in rows:
                evidence = self._parse_evidence_row(row, target_symbol, target_name)
                if evidence:
                    gathered.append(evidence)

        except Exception as e:
            logger.debug(f"Open Targets evidence query failed: {e}")

        return gathered

    def _parse_evidence_row(
        self,
        row: dict[str, Any],
        target_symbol: str,
        target_name: str,
    ) -> GatheredEvidence | None:
        """Parse a single evidence row into a GatheredEvidence item."""
        datasource = row.get("datasourceId", "")
        datatype = row.get("datatypeId", "")
        score = row.get("score", 0) or 0
        disease = row.get("diseaseFromSource", "") or ""
        literature = row.get("literature") or []
        confidence = row.get("confidence") or ""
        sentences = row.get("textMiningSentences") or []
        urls = row.get("urls") or []

        # Build human-readable content
        content_parts: list[str] = []

        # Header: what target-disease pair, from which source
        content_parts.append(
            f"{target_symbol} ({target_name}) — {disease} [{datasource}, {datatype}]"
        )

        # Text mining sentences are the richest content
        if sentences:
            for s in sentences[:5]:
                section = s.get("section", "")
                text = s.get("text", "")
                if text:
                    prefix = f"[{section}] " if section else ""
                    content_parts.append(f"{prefix}{text}")
            if len(sentences) > 5:
                content_parts.append(f"({len(sentences) - 5} more sentences not shown)")

        # For non-literature evidence, add available context
        if not sentences:
            if row.get("variantRsId"):
                content_parts.append(f"Variant: {row['variantRsId']}")
            if row.get("clinicalStage"):
                content_parts.append(f"Clinical stage: {row['clinicalStage']}")
            if row.get("drugFromSource"):
                content_parts.append(f"Drug: {row['drugFromSource']}")
            if row.get("studyId"):
                content_parts.append(f"Study: {row['studyId']}")

        content = "\n".join(content_parts)
        if not content.strip():
            return None

        # Source reference: prefer PubMed URL, fall back to OT URLs
        source_ref = ""
        pmids = [lit for lit in literature if lit and lit.strip()]
        if pmids:
            source_ref = f"https://pubmed.ncbi.nlm.nih.gov/{pmids[0]}/"
        elif urls:
            source_ref = urls[0].get("url", "")
        if not source_ref:
            source_ref = (
                f"https://platform.opentargets.org/evidence/{row.get('id', '')}"
            )

        # Evidence kind from datatype
        evidence_kind = _DATATYPE_TO_KIND.get(datatype, "database_record")

        # Identifiers for dedup and cross-referencing
        identifiers: dict[str, str] = {}
        if pmids:
            identifiers["pmid"] = pmids[0]
        identifiers["ot_evidence_id"] = row.get("id", "")
        identifiers["ensembl_id"] = ""  # filled by caller context
        identifiers["datasource"] = datasource

        return GatheredEvidence(
            content=content,
            source_ref=source_ref,
            source_type="open_targets",
            evidence_kind=evidence_kind,
            identifiers=identifiers,
            structured_data={
                "datasource": datasource,
                "datatype": datatype,
                "target_symbol": target_symbol,
                "target_name": target_name,
                "disease": disease,
                "evidence_score": score,
                "confidence": confidence,
                "publication_year": row.get("publicationYear"),
                "first_author": row.get("publicationFirstAuthor"),
                "variant": row.get("variantRsId"),
                "drug": row.get("drugFromSource"),
                "clinical_stage": row.get("clinicalStage"),
            },
            quality_score=None,
            quality_metadata={
                "evidence_score": score,
                "confidence": confidence,
                "datasource": datasource,
            },
            limitations=(
                ["Text-mined from literature; not manually curated"]
                if datasource == "europepmc"
                else []
            ),
        )
