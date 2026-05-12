"""ClinicalTrials.gov Evidence Provider.

Searches the ClinicalTrials.gov API v2 for clinical study records.
Returns structured trial metadata with phase, status, endpoints, enrollment.

API docs: https://clinicaltrials.gov/data-api/api
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

CT_API = "https://clinicaltrials.gov/api/v2/studies"

# Phase → quality score (higher phase = more evidence)
_PHASE_QUALITY: dict[str, float] = {
    "PHASE4": 0.85,
    "PHASE3": 0.85,
    "PHASE2": 0.70,
    "PHASE1": 0.55,
    "EARLY_PHASE1": 0.45,
    "NA": 0.50,
}


class ClinicalTrialsProvider:
    """Evidence provider using ClinicalTrials.gov API v2."""

    description = (
        "Registry of FDA-regulated and international clinical trials from "
        "ClinicalTrials.gov. Contains trial protocols, eligibility criteria, primary "
        "and secondary endpoints, enrollment numbers, phase (I/II/III/IV), sponsor "
        "information, recruitment status, and posted results. Best for questions about "
        "ongoing or completed clinical studies in humans, trial design, patient "
        "eligibility, endpoint selection, recruitment, and comparative trial data. "
        "Example queries: 'ongoing phase III trials for semaglutide in heart failure', "
        "'eligibility criteria for CAR-T cell therapy trials in lymphoma', 'primary "
        "endpoints of EMPA-REG OUTCOME study', 'recruiting clinical trials for "
        "pancreatic cancer immunotherapy'."
    )

    query_guidance = (
        "The query goes to ClinicalTrials.gov v2 as the `query.term` parameter. "
        "Inside `query.term`, the AREA[FieldName]value syntax scopes to specific "
        "fields: Condition, Intervention, BriefTitle, OfficialTitle, Sponsor, "
        "OverallStatus, StudyType, Phase, OutcomeMeasure, LocationCountry, "
        'NCTId. Boolean (AND, OR, NOT), phrase quoting ("...").\n'
        "\n"
        "Query styles that all work:\n"
        "- Plain text: metformin type 2 diabetes\n"
        "- Field-scoped intervention plus condition: AREA[Intervention]metformin "
        'AND AREA[Condition]"type 2 diabetes"\n'
        "- Phase-filtered: AREA[Intervention]semaglutide AND AREA[Phase]Phase3\n"
        '- Outcome-targeted: AREA[OutcomeMeasure]"HbA1c" AND '
        "AREA[Intervention]metformin\n"
        "- Sponsor plus condition: AREA[Sponsor]Pfizer AND AREA[Condition]melanoma\n"
        '- Status filter: AREA[Intervention]"CAR-T" AND '
        "AREA[OverallStatus]Recruiting\n"
        "- NCT ID lookup: AREA[NCTId]NCT04183440\n"
        "\n"
        "This provider returns trial registrations, not literature — it is not "
        "the right place to look for systematic reviews or meta-analyses. The "
        "`site:` operator does not work."
    )

    query_examples: list[tuple[str, str | None]] = [
        (
            "ongoing phase III trials of semaglutide in heart failure",
            'AREA[Intervention]semaglutide AND AREA[Condition]"heart failure" AND AREA[Phase]Phase3',
        ),
        (
            "eligibility criteria for CAR-T cell therapy trials in lymphoma",
            'AREA[Intervention]"CAR-T" AND AREA[Condition]lymphoma',
        ),
        (
            "trials measuring HbA1c outcomes with metformin in type 2 diabetes",
            'AREA[OutcomeMeasure]"HbA1c" AND AREA[Intervention]metformin AND AREA[Condition]"type 2 diabetes"',
        ),
        (
            "trials by Pfizer in melanoma",
            "AREA[Sponsor]Pfizer AND AREA[Condition]melanoma",
        ),
        (
            "recruiting CAR-T trials",
            'AREA[Intervention]"CAR-T" AND AREA[OverallStatus]Recruiting',
        ),
        (
            "what is the design of NCT04183440",
            "AREA[NCTId]NCT04183440",
        ),
        # Out-of-domain — fundamental biology, no human trials
        (
            "role of H2A.Z histone variant in yeast chromatin",
            None,
        ),
        # Out-of-domain — preclinical / animal-model only
        (
            "efficacy of compound X in mouse xenograft models of glioblastoma",
            None,
        ),
        # Out-of-domain — published literature, not trials
        (
            "mechanism of action of pembrolizumab",
            None,
        ),
    ]
    output_kind = "trial_registration"
    independence_group = "clinical_registry"
    provider_contract_version = 1

    def __init__(self, max_results: int = 10):
        self.max_results = max_results

    async def check_health(self) -> "CheckResult":
        """Test ClinicalTrials.gov API reachability."""
        import time

        import httpx

        from ..preflight import CheckResult

        t0 = time.monotonic()
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(
                    CT_API,
                    params={"query.term": "test", "pageSize": 1, "format": "json"},
                )
                elapsed = (time.monotonic() - t0) * 1000
                if response.status_code == 200:
                    return CheckResult(
                        name="ClinicalTrialsProvider",
                        status="pass",
                        message=f"API reachable ({elapsed:.0f}ms)",
                        elapsed_ms=elapsed,
                    )
                return CheckResult(
                    name="ClinicalTrialsProvider",
                    status="fail",
                    message=f"HTTP {response.status_code}",
                    elapsed_ms=elapsed,
                )
        except Exception as e:
            elapsed = (time.monotonic() - t0) * 1000
            return CheckResult(
                name="ClinicalTrialsProvider",
                status="fail",
                message=str(e),
                elapsed_ms=elapsed,
            )

    async def gather(self, query: str) -> list[GatheredEvidence]:
        """Search ClinicalTrials.gov for relevant studies."""
        import httpx

        gathered: list[GatheredEvidence] = []

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(
                    CT_API,
                    params={
                        "query.term": query,
                        "pageSize": self.max_results,
                        "format": "json",
                    },
                )
                if response.status_code != 200:
                    return []

                data = response.json()
                studies = data.get("studies", [])

                for study in studies:
                    protocol = study.get("protocolSection", {})
                    if not protocol:
                        continue

                    has_results = study.get("hasResults", False)
                    gathered.append(
                        self._parse_study(protocol, has_results=has_results)
                    )

        except Exception as e:
            logger.warning(f"ClinicalTrials.gov query failed for '{query}': {e}")

        return gathered

    def _parse_study(
        self, protocol: dict[str, Any], *, has_results: bool = False
    ) -> GatheredEvidence:
        """Parse a single study protocol section."""
        id_module = protocol.get("identificationModule", {})
        status_module = protocol.get("statusModule", {})
        design_module = protocol.get("designModule", {})
        description_module = protocol.get("descriptionModule", {})
        conditions_module = protocol.get("conditionsModule", {})
        interventions_module = protocol.get("armsInterventionsModule", {})
        outcomes_module = protocol.get("outcomesModule", {})
        sponsor_module = protocol.get("sponsorCollaboratorsModule", {})

        nct_id = id_module.get("nctId", "")
        title = id_module.get("officialTitle") or id_module.get("briefTitle", "")
        status = status_module.get("overallStatus", "")
        phases = design_module.get("phases", [])
        phase = phases[0] if phases else "NA"
        enrollment_info = design_module.get("enrollmentInfo", {})
        enrollment = enrollment_info.get("count")
        enrollment_type = enrollment_info.get("type", "")
        brief_summary = description_module.get("briefSummary", "")
        conditions = conditions_module.get("conditions", [])

        # Interventions
        interventions = []
        for arm in interventions_module.get("interventions", []):
            interventions.append(
                {
                    "type": arm.get("type", ""),
                    "name": arm.get("name", ""),
                }
            )

        # Primary outcomes
        primary_endpoints = []
        for outcome in outcomes_module.get("primaryOutcomes", []):
            primary_endpoints.append(
                {
                    "measure": outcome.get("measure", ""),
                    "time_frame": outcome.get("timeFrame", ""),
                }
            )

        # Secondary outcomes
        secondary_endpoints = []
        for outcome in outcomes_module.get("secondaryOutcomes", []):
            secondary_endpoints.append(
                {
                    "measure": outcome.get("measure", ""),
                    "time_frame": outcome.get("timeFrame", ""),
                }
            )

        # Other outcomes
        other_endpoints = []
        for outcome in outcomes_module.get("otherOutcomes", []):
            other_endpoints.append(
                {
                    "measure": outcome.get("measure", ""),
                    "time_frame": outcome.get("timeFrame", ""),
                }
            )

        # Sponsor
        lead_sponsor = sponsor_module.get("leadSponsor", {})
        sponsor = lead_sponsor.get("name", "")

        # Build human-readable content
        content_parts = [title]
        if phase != "NA":
            content_parts.append(f"Phase: {phase.replace('PHASE', 'Phase ')}")
        content_parts.append(f"Status: {status}")
        if conditions:
            suffix = f" (and {len(conditions) - 3} more)" if len(conditions) > 3 else ""
            content_parts.append(f"Conditions: {', '.join(conditions[:3])}{suffix}")
        if interventions:
            names = [i["name"] for i in interventions[:3]]
            suffix = (
                f" (and {len(interventions) - 3} more)"
                if len(interventions) > 3
                else ""
            )
            content_parts.append(f"Interventions: {', '.join(names)}{suffix}")
        if enrollment:
            content_parts.append(f"Enrollment: {enrollment}")
        if brief_summary:
            content_parts.append(f"\n{brief_summary}")

        # Quality scoring
        quality = _PHASE_QUALITY.get(phase, 0.50)
        if has_results:
            quality = min(quality + 0.1, 0.95)
        if status in ("COMPLETED", "ACTIVE_NOT_RECRUITING"):
            quality = min(quality + 0.05, 0.95)

        return GatheredEvidence(
            content="\n".join(content_parts),
            source_ref=nct_id,
            source_type="clinicaltrials",
            evidence_kind="clinical_trial",
            identifiers={"nct_id": nct_id} if nct_id else {},
            structured_data={
                "title": title,
                "phase": phase,
                "status": status,
                "enrollment": enrollment,
                "enrollment_type": enrollment_type,
                "conditions": conditions,
                "interventions": interventions,
                "primary_endpoints": primary_endpoints,
                "secondary_endpoints": secondary_endpoints,
                "other_endpoints": other_endpoints,
                "sponsor": sponsor,
                "has_results": has_results,
                "brief_summary": brief_summary,
            },
            quality_score=None,
            quality_metadata={
                "phase": phase,
                "has_results": has_results,
                "status": status,
            },
        )
