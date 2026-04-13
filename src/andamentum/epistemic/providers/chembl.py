"""ChEMBL Evidence Provider.

Searches the EMBL-EBI ChEMBL database for bioactivity data, drug mechanisms,
and compound-target interactions.

API docs: https://www.ebi.ac.uk/chembl/api/data/docs
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

CHEMBL_API = "https://www.ebi.ac.uk/chembl/api/data"


class ChEMBLProvider:
    """Evidence provider using the ChEMBL REST API."""

    def __init__(self, max_results: int = 10):
        self.max_results = max_results

    async def check_health(self) -> "CheckResult":
        """Test ChEMBL API reachability."""
        import time

        import httpx

        from ..preflight import CheckResult

        t0 = time.monotonic()
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(
                    f"{CHEMBL_API}/molecule/search.json",
                    params={"q": "aspirin", "limit": 1},
                )
                elapsed = (time.monotonic() - t0) * 1000
                if response.status_code == 200:
                    return CheckResult(
                        name="ChEMBLProvider",
                        status="pass",
                        message=f"API reachable ({elapsed:.0f}ms)",
                        elapsed_ms=elapsed,
                    )
                return CheckResult(
                    name="ChEMBLProvider",
                    status="fail",
                    message=f"HTTP {response.status_code}",
                    elapsed_ms=elapsed,
                )
        except Exception as e:
            elapsed = (time.monotonic() - t0) * 1000
            return CheckResult(
                name="ChEMBLProvider", status="fail", message=str(e), elapsed_ms=elapsed
            )

    async def gather(self, query: str) -> list[GatheredEvidence]:
        """Search ChEMBL for molecules and their bioactivity data."""
        import httpx

        gathered: list[GatheredEvidence] = []

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                # Search for molecules
                mol_resp = await client.get(
                    f"{CHEMBL_API}/molecule/search.json",
                    params={
                        "q": query,
                        "limit": self.max_results,
                    },
                )
                if mol_resp.status_code != 200:
                    return []

                data = mol_resp.json()
                molecules = data.get("molecules", [])

                for mol in molecules[: self.max_results]:
                    chembl_id = mol.get("molecule_chembl_id", "")
                    pref_name = mol.get("pref_name") or ""
                    raw_phase = mol.get("max_phase")
                    try:
                        max_phase = int(raw_phase) if raw_phase is not None else None
                    except (ValueError, TypeError):
                        max_phase = None
                    molecule_type = mol.get("molecule_type", "")
                    first_approval = mol.get("first_approval")

                    if not chembl_id:
                        continue

                    # Get mechanism of action if available
                    mechanism = await self._get_mechanism(client, chembl_id)

                    # Get top bioactivities
                    activities = await self._get_activities(client, chembl_id)

                    # Build content
                    content_parts = []
                    name_str = pref_name or chembl_id
                    content_parts.append(f"{name_str} ({chembl_id})")
                    if max_phase is not None:
                        phase_labels = {
                            0: "Preclinical",
                            1: "Phase I",
                            2: "Phase II",
                            3: "Phase III",
                            4: "Approved",
                        }
                        content_parts.append(
                            f"Max phase: {phase_labels.get(max_phase, str(max_phase))}"
                        )
                    if mechanism:
                        content_parts.append(
                            f"Mechanism: {mechanism.get('description', '')}"
                        )
                    if activities:
                        act_strs = [
                            f"{a['type']} = {a['value']} {a['units']} ({a['target']})"
                            for a in activities[:3]
                        ]
                        content_parts.append(f"Activity: {'; '.join(act_strs)}")

                    # Quality based on data completeness and phase
                    quality = 0.6
                    if max_phase and max_phase >= 4:
                        quality = 0.8
                    elif max_phase and max_phase >= 2:
                        quality = 0.7

                    identifiers: dict[str, str] = {"chembl_id": chembl_id}

                    gathered.append(
                        GatheredEvidence(
                            content="\n".join(content_parts),
                            source_ref=chembl_id,
                            source_type="chembl",
                            evidence_kind="bioactivity",
                            identifiers=identifiers,
                            structured_data={
                                "molecule_name": pref_name,
                                "molecule_type": molecule_type,
                                "max_phase": max_phase,
                                "first_approval": first_approval,
                                "mechanism": mechanism,
                                "activities": activities,
                            },
                            quality_score=quality,
                            quality_metadata={
                                "max_phase": max_phase,
                                "activity_count": len(activities),
                            },
                        )
                    )

        except Exception as e:
            logger.warning(f"ChEMBL query failed for '{query}': {e}")

        return gathered

    async def _get_mechanism(
        self, client: Any, chembl_id: str
    ) -> dict[str, Any] | None:
        """Get mechanism of action for a molecule."""
        try:
            resp = await client.get(
                f"{CHEMBL_API}/mechanism.json",
                params={
                    "molecule_chembl_id": chembl_id,
                    "limit": 1,
                },
            )
            if resp.status_code == 200:
                data = resp.json()
                mechs = data.get("mechanisms", [])
                if mechs:
                    m = mechs[0]
                    return {
                        "description": m.get("mechanism_of_action", ""),
                        "target_chembl_id": m.get("target_chembl_id", ""),
                        "action_type": m.get("action_type", ""),
                    }
        except Exception:
            pass
        return None

    async def _get_activities(
        self, client: Any, chembl_id: str
    ) -> list[dict[str, Any]]:
        """Get top bioactivities for a molecule."""
        try:
            resp = await client.get(
                f"{CHEMBL_API}/activity.json",
                params={
                    "molecule_chembl_id": chembl_id,
                    "limit": 5,
                    "pchembl_value__isnull": "false",
                },
            )
            if resp.status_code == 200:
                data = resp.json()
                activities = []
                for act in data.get("activities", []):
                    activities.append(
                        {
                            "type": act.get("standard_type", ""),
                            "value": act.get("standard_value"),
                            "units": act.get("standard_units", ""),
                            "target": act.get("target_pref_name", ""),
                            "target_chembl_id": act.get("target_chembl_id", ""),
                        }
                    )
                return activities
        except Exception:
            pass
        return []
