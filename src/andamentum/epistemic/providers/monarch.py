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


class MonarchProvider:
    """Evidence provider using Monarch Initiative for gene-disease associations.

    Quality score is pre-populated at 0.7 (curated aggregated database).
    No auth required. No documented rate limit.
    """

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
        """Search Monarch for gene-disease associations.

        Uses both association and search endpoints.

        Args:
            query: Natural language query (e.g., "BRCA1 breast cancer")

        Returns:
            List of GatheredEvidence with structured association data
        """
        import httpx

        gathered: list[GatheredEvidence] = []

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                # Strategy 1: Search endpoint for broad queries
                search_results = await self._search(client, query)
                gathered.extend(search_results)

                # Strategy 2: Association endpoint if we found entity IDs
                entity_ids = self._extract_entity_ids(search_results, query)
                for entity_id in entity_ids[:3]:
                    assoc_results = await self._get_associations(client, entity_id)
                    gathered.extend(assoc_results)

        except Exception as e:
            logger.warning(f"Monarch Initiative query failed for '{query}': {e}")
            if not gathered:
                return [
                    GatheredEvidence(
                        content=f"Monarch Initiative search failed: {e}",
                        source_ref=query,
                        source_type="monarch_initiative",
                        quality_score=0.0,
                        limitations=[f"API error: {e}"],
                    )
                ]

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
                        source_ref=f"https://monarchinitiative.org/{item_id}"
                        if item_id
                        else name,
                        source_type="monarch_initiative",
                        evidence_kind="gene_disease",
                        identifiers={"monarch_id": item_id} if item_id else {},
                        structured_data={
                            "name": name,
                            "category": category,
                            "description": description,
                        },
                        quality_score=0.7,
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
        """Get associations for a specific entity (gene or disease)."""
        try:
            response = await client.get(
                f"{MONARCH_API}/association",
                params={
                    "subject": entity_id,
                    "limit": self.max_results,
                },
            )

            if response.status_code != 200:
                response = await client.get(
                    f"{MONARCH_API}/association",
                    params={
                        "object": entity_id,
                        "limit": self.max_results,
                    },
                )
                if response.status_code != 200:
                    return []

            data = response.json()
            items = data.get("items", [])
            gathered: list[GatheredEvidence] = []

            for assoc in items:
                subject = assoc.get("subject", {})
                obj = assoc.get("object", {})
                predicate = assoc.get("predicate", "associated_with")
                evidence_types = assoc.get("evidence_types", [])
                sources = assoc.get("publications", []) or assoc.get("sources", [])

                subject_name = subject.get("name", subject.get("id", "unknown"))
                object_name = obj.get("name", obj.get("id", "unknown"))
                subject_id = subject.get("id", "")
                object_id = obj.get("id", "")

                content_parts = [
                    f"**Association**: {subject_name} — {predicate} — {object_name}",
                    f"Subject: {subject_name} ({subject_id})",
                    f"Object: {object_name} ({object_id})",
                ]

                if evidence_types:
                    content_parts.append(
                        f"Evidence types: {', '.join(str(e) for e in evidence_types)}"
                    )
                if sources:
                    source_strs = [str(s) for s in sources[:5]]
                    content_parts.append(f"Sources: {', '.join(source_strs)}")

                gathered.append(
                    GatheredEvidence(
                        content="\n".join(content_parts),
                        source_ref=f"https://monarchinitiative.org/{subject_id}"
                        if subject_id
                        else f"monarch:{entity_id}",
                        source_type="monarch_initiative",
                        evidence_kind="gene_disease",
                        identifiers={
                            k: v
                            for k, v in [
                                ("subject_id", subject_id),
                                ("object_id", object_id),
                            ]
                            if v
                        },
                        structured_data={
                            "subject_name": subject_name,
                            "object_name": object_name,
                            "predicate": predicate,
                            "evidence_types": evidence_types,
                        },
                        quality_score=0.7,
                        quality_metadata={
                            "source": "monarch_initiative",
                            "subject_id": subject_id,
                            "object_id": object_id,
                            "predicate": predicate,
                            "evidence_types": evidence_types,
                        },
                        limitations=[
                            "Curated database association; strength of evidence varies by source"
                        ],
                    )
                )

            return gathered

        except Exception as e:
            logger.debug(f"Monarch association query failed for {entity_id}: {e}")
            return []

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
