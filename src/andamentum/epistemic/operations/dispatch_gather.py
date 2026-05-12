"""Description-driven evidence gathering — initial-gather operation
plus a shared helper used by the follow-up investigation operation.

This module holds:

1. ``DispatchGatherOperation`` — the initial gather. Driven by
   ``PlanEvidence``, run once per objective at the start of inquiry.
   Iterates the objective's sub-investigations (or the clarified
   question if there's no decomposition) and feeds each one to
   ``dispatch_and_persist_for_text``.

2. ``dispatch_and_persist_for_text`` — a module-level helper that runs
   one search target through ``gather_evidence_new`` and persists the
   returned ``GatheredEvidence`` items as Evidence entities with content
   + quality scoring already attached. Both ``DispatchGatherOperation``
   (initial gather) and ``InvestigateClaimOperation`` (follow-up rounds)
   call this — there is one routing+persistence implementation, used
   in both places.

Scope discipline: neither operation falls back to a legacy path or
fabricates evidence when providers return nothing. Zero-yield is
honest — the retrieval-health check in ``ExtractEvidence`` /
``ExtractNewEvidence`` marks the objective ``retrieval_failed``.
"""

from __future__ import annotations

import logging
from typing import Any

from .base import BaseOperation, OperationInput, OperationResult
from .evidence import ExtractEvidenceOperation

from ..entities import Evidence, Objective

logger = logging.getLogger(__name__)


async def dispatch_and_persist_for_text(
    op: BaseOperation,
    text: str,
    *,
    objective_id: str,
    providers: dict[str, Any],
    core_runner: Any,
    sub_investigation_id: str | None = None,
    depends_on_claim_id: str | None = None,
    created_by: str = "dispatch",
) -> list[str]:
    """Run one search target through the description-driven dispatch
    path and persist each returned ``GatheredEvidence`` as an Evidence
    entity. Returns the list of created Evidence entity ids.

    Both the initial-gather operation and the follow-up investigation
    operation funnel through this function — there is one routing +
    persistence implementation. The ``text`` argument is whatever the
    caller wants the dispatch agent to interpret as a search target:

    - Initial gather passes a sub-investigation's ``seed_claim`` (or
      the clarified question when there's no decomposition).
    - Investigation passes one **intent** from the rewritten
      ``epistemic_investigate_claim`` agent — a natural-language
      description of an evidence-gap angle.

    The dispatch agent's prompt names its input "claim" but the
    contract is "natural-language description of what to look for"; an
    intent is the same shape.

    Args:
        op: The calling operation. Used as a context bag for
            ``repo``, ``agent_runner``, ``evidence_gatherer``,
            ``quality_scorer``, and ``embedding_model`` — the
            dependencies needed to score evidence consistently with
            the legacy ExtractEvidence path.
        text: The search target the dispatch agent sees as "claim".
        objective_id: Objective the evidence belongs to.
        providers: ``{name: instance}`` provider registry.
        core_runner: The core.agents.AgentRunner-shaped runner the
            dispatch agent uses (``op.agent_runner.core_runner`` on
            the DefaultAgentRunner).
        sub_investigation_id: Propagated onto each Evidence so
            MultiSeedClaim's per-claim filter sees it. Initial gather
            sets this from ``sub.id``; investigation propagates it
            from the claim under investigation.
        depends_on_claim_id: Set by investigation only — ties each
            new Evidence to the claim that triggered the round.
            Initial gather leaves this None.
        created_by: Audit tag on the new Evidence entities.
    """
    from ..dispatch import gather_evidence_new

    gathered = await gather_evidence_new(
        claim=text, providers=providers, agent_runner=core_runner
    )

    extract_helper = ExtractEvidenceOperation(
        op.repo,
        op.agent_runner,
        evidence_gatherer=op.evidence_gatherer,
        quality_scorer=op.quality_scorer,
        embedding_model=op.embedding_model,
    )

    created_ids: list[str] = []
    for g in gathered:
        evidence = Evidence(
            objective_id=objective_id,
            source_type=g.source_type or "unknown",
            source_ref=g.source_ref,
            extracted=True,
            sub_investigation_id=sub_investigation_id,
            depends_on_claim_id=depends_on_claim_id,
            created_by=created_by,
        )
        extract_helper._fill_evidence_from_gathered(evidence, g)
        await extract_helper._score_evidence(evidence, g)
        await op.repo.save(evidence)
        created_ids.append(evidence.entity_id)
    return created_ids


class DispatchGatherOperation(BaseOperation):
    """Initial evidence gather: dispatch each sub-investigation (or the
    objective itself when there's no decomposition) through the
    description-driven routing layer.

    Requires:

    - ``self.agent_runner`` — a ``DefaultAgentRunner``. Its
      ``core_runner`` property is forwarded to ``gather_evidence_new``
      because that path uses the definition-based ``run(defn,
      **kwargs)`` shape.
    - ``providers`` dict on construction — provider instances to
      dispatch to.
    """

    entity_type = "objective"

    def __init__(
        self,
        *args: object,
        providers: dict[str, object] | None = None,
        **kwargs: object,
    ) -> None:
        super().__init__(*args, **kwargs)  # type: ignore[arg-type]
        self._dispatch_providers: dict[str, object] = providers or {}

    async def execute(self, work: OperationInput) -> OperationResult:
        objective = await self.repo.get("objective", work.entity_id)

        if not isinstance(objective, Objective):
            return OperationResult(
                success=False,
                entity_id=work.entity_id,
                message="Entity is not Objective",
            )

        if objective.phase != "analyzed":
            return OperationResult(
                success=True,
                entity_id=work.entity_id,
                message=f"Phase {objective.phase} is not 'analyzed'",
            )

        if not self.agent_runner:
            raise RuntimeError(
                "DispatchGatherOperation requires an agent_runner — the "
                "description-driven dispatch agent is the routing decision."
            )

        if not self._dispatch_providers:
            raise RuntimeError(
                "DispatchGatherOperation requires a non-empty providers "
                "dict (passed through EpistemicDeps.providers)."
            )

        core_runner: Any = getattr(
            self.agent_runner, "core_runner", self.agent_runner
        )

        clarified = objective.clarified_question or objective.description
        sub_investigations = (
            objective.decomposition.sub_investigations
            if objective.decomposition
            else []
        )
        if sub_investigations:
            work_list: list[tuple[str | None, str]] = [
                (sub.id, sub.seed_claim or clarified)
                for sub in sub_investigations
            ]
        else:
            work_list = [(None, clarified)]

        created_entities: list[str] = []
        for sub_id, claim_text in work_list:
            ev_ids = await dispatch_and_persist_for_text(
                self,
                claim_text,
                objective_id=objective.entity_id,
                providers=self._dispatch_providers,  # type: ignore[arg-type]
                core_runner=core_runner,
                sub_investigation_id=sub_id,
                created_by="dispatch_gather",
            )
            created_entities.extend(ev_ids)

        objective.phase = "planned"
        await self.repo.save(objective)

        sub_count = len(sub_investigations) if sub_investigations else 1
        plan_msg = (
            f"Description-driven dispatch: {len(created_entities)} evidence "
            f"items persisted across {sub_count} sub-investigation(s)"
        )

        return OperationResult(
            success=True,
            entity_id=objective.entity_id,
            message=plan_msg,
            created_entities=created_entities,
        )
