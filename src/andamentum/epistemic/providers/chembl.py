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

    description = (
        "Curated database of bioactive drug-like small molecules from EMBL-EBI, with "
        "quantitative bioactivity data, drug mechanisms, ADMET properties, and "
        "compound-target interactions. Contains IC50, EC50, Ki, Kd values, SMILES "
        "structures, ChEMBL IDs, binding assays, and approved drug indications. Best "
        "for questions about specific chemical compounds, drug potency, medicinal "
        "chemistry, quantitative pharmacology, and structure-activity relationships. "
        "Example queries: 'IC50 of imatinib against BCR-ABL kinase', 'mechanism of "
        "action of pembrolizumab', 'SMILES structure and bioactivity of remdesivir', "
        "'EC50 values for ACE inhibitors on angiotensin converting enzyme'."
    )

    query_guidance = (
        "The query goes to ChEMBL's `/molecule/search.json` `q` parameter. "
        "Accepts: compound generic name, trade name, synonym, ChEMBL ID, drug "
        "INN, IUPAC name. SMILES and InChI substring search uses different "
        "endpoints not exposed by this adapter — do NOT pass SMILES strings.\n"
        "\n"
        "Query styles that all work:\n"
        "- Generic name: imatinib\n"
        "- Trade name: Gleevec\n"
        "- ChEMBL ID: CHEMBL941\n"
        "- Synonym or development code: STI571\n"
        "- Drug class member: pembrolizumab\n"
        "- Compound plus synonym: metformin glucophage\n"
        "\n"
        "This is a compound search, not a literature search — returns "
        "molecular structures, bioactivity (IC50, EC50, Ki), and mechanism. "
        "Use only when the question explicitly asks for compound-level "
        "pharmacology. 1–3 token compound names are optimal; verbose natural-"
        "language descriptions return nothing."
    )

    query_examples: list[tuple[str, str | None]] = []
    output_kind = "structured_record"
    independence_group = "chemistry_structured"
    provider_contract_version = 1

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

                    # Get top bioactivities sorted by pchembl_value
                    activities = await self._get_activities(client, chembl_id)

                    # Get molecule details (SMILES + ADMET properties)
                    mol_details = await self._get_molecule_details(client, chembl_id)
                    smiles = mol_details.get("smiles")
                    admet = mol_details.get("admet", {})

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
                    if admet:
                        admet_parts = []
                        if admet.get("mw_freebase") is not None:
                            admet_parts.append(f"MW: {admet['mw_freebase']}")
                        if admet.get("alogp") is not None:
                            admet_parts.append(f"LogP: {admet['alogp']}")
                        if admet.get("ro5_violations") is not None:
                            admet_parts.append(
                                f"Lipinski violations: {admet['ro5_violations']}"
                            )
                        if admet_parts:
                            content_parts.append(", ".join(admet_parts))
                    if mechanism:
                        content_parts.append(
                            f"Mechanism: {mechanism.get('description', '')}"
                        )
                    if activities:
                        act_strs = [
                            (
                                f"{a['type']} = {a['value']} {a['units']}"
                                f" (pChEMBL: {a['pchembl_value']})"
                                f" [{a['target']} / {a['target_chembl_id']}]"
                            )
                            for a in activities[:3]
                        ]
                        suffix = (
                            f" (showing top 3 of {len(activities)})"
                            if len(activities) > 3
                            else ""
                        )
                        content_parts.append(f"Activity: {'; '.join(act_strs)}{suffix}")

                    identifiers: dict[str, str] = {"chembl_id": chembl_id}
                    if smiles:
                        identifiers["smiles"] = smiles

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
                                "smiles": smiles,
                                "admet": admet,
                                "mechanism": mechanism,
                                "activities": activities,
                            },
                            quality_score=None,
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
        """Get top bioactivities for a molecule, sorted by pchembl_value descending."""
        try:
            resp = await client.get(
                f"{CHEMBL_API}/activity.json",
                params={
                    "molecule_chembl_id": chembl_id,
                    "limit": 25,
                    "pchembl_value__isnull": "false",
                },
            )
            if resp.status_code == 200:
                data = resp.json()
                activities = []
                for act in data.get("activities", []):
                    raw_pchembl = act.get("pchembl_value")
                    try:
                        pchembl = (
                            float(raw_pchembl) if raw_pchembl is not None else None
                        )
                    except (ValueError, TypeError):
                        pchembl = None
                    if pchembl is None:
                        continue
                    activities.append(
                        {
                            "type": act.get("standard_type", ""),
                            "value": act.get("standard_value"),
                            "units": act.get("standard_units", ""),
                            "pchembl_value": pchembl,
                            "target": act.get("target_pref_name", ""),
                            "target_chembl_id": act.get("target_chembl_id", ""),
                        }
                    )
                activities.sort(key=lambda a: a["pchembl_value"], reverse=True)
                return activities
        except Exception:
            pass
        return []

    async def _get_molecule_details(
        self, client: Any, chembl_id: str
    ) -> dict[str, Any]:
        """Get molecule SMILES and ADMET-related properties."""
        try:
            resp = await client.get(f"{CHEMBL_API}/molecule/{chembl_id}.json")
            if resp.status_code == 200:
                data = resp.json()
                smiles = None
                structs = data.get("molecule_structures") or {}
                if structs:
                    smiles = structs.get("canonical_smiles")

                admet: dict[str, Any] = {}
                props = data.get("molecule_properties") or {}
                for key in (
                    "alogp",
                    "hba",
                    "hbd",
                    "psa",
                    "ro5_violations",
                    "mw_freebase",
                ):
                    val = props.get(key)
                    if val is not None:
                        try:
                            admet[key] = float(val) if "." in str(val) else int(val)
                        except (ValueError, TypeError):
                            admet[key] = val

                return {"smiles": smiles, "admet": admet}
        except Exception:
            pass
        return {"smiles": None, "admet": {}}
