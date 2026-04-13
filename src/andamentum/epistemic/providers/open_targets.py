"""Open Targets Evidence Provider.

Queries the Open Targets Platform GraphQL API for disease-target associations,
integrating genetic, somatic, literature, and drug evidence.

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


class OpenTargetsProvider:
    """Evidence provider using Open Targets Platform GraphQL API."""

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
                    return CheckResult(
                        name="OpenTargetsProvider",
                        status="pass",
                        message=f"API reachable ({elapsed:.0f}ms)",
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
            return CheckResult(name="OpenTargetsProvider", status="fail", message=str(e), elapsed_ms=elapsed)

    async def gather(self, query: str) -> list[GatheredEvidence]:
        """Search Open Targets for disease-target associations."""
        import httpx

        gathered: list[GatheredEvidence] = []

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                # Step 1: Search for entities (targets or diseases)
                search_results = await self._search_entities(client, query)

                # Step 2: For each target-disease pair, get association data
                for entity in search_results[: self.max_results]:
                    entity_id = entity.get("id", "")
                    entity_type = entity.get("entity", "")

                    if entity_type == "target":
                        assocs = await self._get_target_associations(client, entity_id)
                    elif entity_type == "disease":
                        assocs = await self._get_disease_associations(client, entity_id)
                    else:
                        continue

                    gathered.extend(assocs)

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

    async def _get_target_associations(self, client: Any, target_id: str) -> list[GatheredEvidence]:
        """Get disease associations for a target."""
        gql = """
        query TargetAssociations($id: String!, $size: Int!) {
            target(ensemblId: $id) {
                id
                approvedSymbol
                approvedName
                associatedDiseases(page: {size: $size, index: 0}) {
                    rows {
                        disease {
                            id
                            name
                        }
                        score
                        datasourceScores {
                            id
                            score
                        }
                    }
                }
            }
        }
        """
        return await self._fetch_associations(client, gql, {"id": target_id, "size": 5}, "target")

    async def _get_disease_associations(self, client: Any, disease_id: str) -> list[GatheredEvidence]:
        """Get target associations for a disease."""
        gql = """
        query DiseaseAssociations($id: String!, $size: Int!) {
            disease(efoId: $id) {
                id
                name
                associatedTargets(page: {size: $size, index: 0}) {
                    rows {
                        target {
                            id
                            approvedSymbol
                            approvedName
                        }
                        score
                        datasourceScores {
                            id
                            score
                        }
                    }
                }
            }
        }
        """
        return await self._fetch_associations(client, gql, {"id": disease_id, "size": 5}, "disease")

    async def _fetch_associations(
        self, client: Any, gql: str, variables: dict, query_type: str
    ) -> list[GatheredEvidence]:
        """Execute a GraphQL query and parse association results."""
        gathered: list[GatheredEvidence] = []

        try:
            resp = await client.post(OT_API, json={"query": gql, "variables": variables})
            if resp.status_code != 200:
                return []

            data = resp.json().get("data", {})

            if query_type == "target":
                entity = data.get("target", {})
                target_symbol = entity.get("approvedSymbol", "")
                target_name = entity.get("approvedName", "")
                rows = entity.get("associatedDiseases", {}).get("rows", [])

                for row in rows:
                    disease = row.get("disease", {})
                    score = row.get("score", 0)
                    ds_scores = {ds["id"]: ds["score"] for ds in row.get("datasourceScores", []) if ds.get("score")}

                    disease_name = disease.get("name", "")
                    disease_id = disease.get("id", "")

                    content = f"{target_symbol} ({target_name}) is associated with {disease_name} (score: {score:.2f})"
                    if ds_scores:
                        top_sources = sorted(ds_scores.items(), key=lambda x: x[1], reverse=True)[:3]
                        sources_str = ", ".join(f"{s[0]}: {s[1]:.2f}" for s in top_sources)
                        content += f". Evidence: {sources_str}"

                    gathered.append(
                        GatheredEvidence(
                            content=content,
                            source_ref=f"{entity.get('id', '')}-{disease_id}",
                            source_type="open_targets",
                            evidence_kind="genetic_association",
                            identifiers={
                                "ensembl_id": entity.get("id", ""),
                                "disease_id": disease_id,
                                "gene_symbol": target_symbol,
                            },
                            structured_data={
                                "target": {"symbol": target_symbol, "name": target_name},
                                "disease": {"name": disease_name, "id": disease_id},
                                "association_score": score,
                                "datasource_scores": ds_scores,
                            },
                            quality_score=min(0.5 + score * 0.4, 0.9),
                            quality_metadata={"association_score": score},
                        )
                    )

            elif query_type == "disease":
                entity = data.get("disease", {})
                disease_name = entity.get("name", "")
                rows = entity.get("associatedTargets", {}).get("rows", [])

                for row in rows:
                    target = row.get("target", {})
                    score = row.get("score", 0)
                    ds_scores = {ds["id"]: ds["score"] for ds in row.get("datasourceScores", []) if ds.get("score")}

                    target_symbol = target.get("approvedSymbol", "")
                    target_name = target.get("approvedName", "")
                    target_id = target.get("id", "")

                    content = f"{target_symbol} ({target_name}) is associated with {disease_name} (score: {score:.2f})"

                    gathered.append(
                        GatheredEvidence(
                            content=content,
                            source_ref=f"{target_id}-{entity.get('id', '')}",
                            source_type="open_targets",
                            evidence_kind="genetic_association",
                            identifiers={
                                "ensembl_id": target_id,
                                "disease_id": entity.get("id", ""),
                                "gene_symbol": target_symbol,
                            },
                            structured_data={
                                "target": {"symbol": target_symbol, "name": target_name},
                                "disease": {"name": disease_name, "id": entity.get("id", "")},
                                "association_score": score,
                                "datasource_scores": ds_scores,
                            },
                            quality_score=min(0.5 + score * 0.4, 0.9),
                            quality_metadata={"association_score": score},
                        )
                    )

        except Exception as e:
            logger.debug(f"Open Targets association query failed: {e}")

        return gathered
