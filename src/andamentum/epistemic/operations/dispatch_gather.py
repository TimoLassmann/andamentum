"""Description-driven evidence gathering — new dispatch path.

Phase 4 of the description-driven-dispatch PRD
(``docs/superpowers/plans/2026-05-12-description-driven-provider-
dispatch.md``).

This operation is the new-mode counterpart to the legacy pair
``PlanTaskOperation`` (creates Evidence stubs via three-agent chain)
plus ``ExtractEvidenceOperation`` (fills each stub by calling its
provider). It collapses both into one step:

1. For each sub-investigation (or the objective itself when no
   decomposition exists), call ``dispatch.gather_evidence_new``.
2. ``gather_evidence_new`` runs the description-driven dispatch agent
   once per provider in parallel: each agent decides whether to abstain
   or commit one or two native-syntax queries, then provider HTTP calls
   fan out in parallel.
3. Returned ``GatheredEvidence`` items are persisted as Evidence
   entities with ``extracted=True`` and quality scoring already applied
   — so the downstream ``ExtractEvidence`` node harmlessly finds no
   ``extracted=False`` stubs and proceeds.

Scope discipline: this operation never falls back to the legacy path,
never silently drops sub-investigations, and never fabricates evidence
when providers return nothing. Zero-yield is honest — the retrieval-
health check in ``ExtractEvidence`` will mark the objective
``retrieval_failed``.

Architecture: operations are pure transforms (P1). This operation
reads the objective, does its work, and writes Evidence entities.
The graph (``PlanEvidence`` node, branching on ``state.dispatch_mode``)
controls when it runs.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from .base import BaseOperation, OperationInput, OperationResult
from .evidence import ExtractEvidenceOperation

from ..entities import Evidence, Objective

if TYPE_CHECKING:
    from .base import GatheredEvidence

logger = logging.getLogger(__name__)


class DispatchGatherOperation(BaseOperation):
    """Run the description-driven dispatch path end-to-end on one objective.

    Reads ``objective.decomposition`` to determine the per-sub-claim
    fan-out, runs ``gather_evidence_new`` per sub-claim, persists each
    returned ``GatheredEvidence`` as an Evidence entity with content +
    quality score attached.

    Requires:

    - ``self.agent_runner`` — a ``DefaultAgentRunner`` (the graph's
      standard agent runner). Its ``core_runner`` property is forwarded
      to ``gather_evidence_new`` because that path uses the
      definition-based ``run(defn, **kwargs)`` shape rather than the
      name-based wrapper.
    - ``providers`` dict on the constructor's ``embedding_model`` side
      isn't enough — we read it directly off ``self`` via the bespoke
      constructor argument below.
    """

    entity_type = "objective"

    def __init__(
        self,
        *args: object,
        providers: dict[str, object] | None = None,
        **kwargs: object,
    ) -> None:
        # Pass-through to BaseOperation; we just snapshot ``providers``
        # as an extra attribute so execute() can hand it to
        # gather_evidence_new without reaching into the deps object.
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
                "dict (passed through EpistemicDeps.providers). Set "
                "dispatch_mode='legacy' or supply providers."
            )

        # Reach the core AgentRunner protocol: gather_evidence_new builds
        # the AgentDefinition itself and calls runner.run(defn, **kwargs),
        # which is not the name-based wrapper DefaultAgentRunner exposes.
        core_runner: Any = getattr(
            self.agent_runner, "core_runner", self.agent_runner
        )

        # Build the per-sub-claim work list. Multi-seed decomposition
        # produces N (sub_id, claim_text) pairs; otherwise a single
        # (None, clarified_question) pair drives one dispatch sweep.
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

        # Run dispatch + gather once per sub-claim. Sub-claims are
        # iterated sequentially because gather_evidence_new already
        # parallelises across providers; serialising at the sub-claim
        # level keeps API call bursts bounded and per-claim traces
        # debuggable.
        from ..dispatch import gather_evidence_new

        created_entities: list[str] = []
        total_gathered = 0
        for sub_id, claim_text in work_list:
            gathered = await gather_evidence_new(
                claim=claim_text,
                providers=self._dispatch_providers,
                agent_runner=core_runner,
            )
            total_gathered += len(gathered)
            for g in gathered:
                ev_id = await self._persist_evidence(
                    objective=objective,
                    gathered=g,
                    sub_investigation_id=sub_id,
                )
                created_entities.append(ev_id)

        objective.phase = "planned"
        await self.repo.save(objective)

        sub_count = len(sub_investigations) if sub_investigations else 1
        plan_msg = (
            f"Description-driven dispatch: {total_gathered} evidence items "
            f"persisted across {sub_count} sub-investigation(s)"
        )

        return OperationResult(
            success=True,
            entity_id=objective.entity_id,
            message=plan_msg,
            created_entities=created_entities,
        )

    async def _persist_evidence(
        self,
        *,
        objective: Objective,
        gathered: "GatheredEvidence",
        sub_investigation_id: str | None,
    ) -> str:
        """Persist one GatheredEvidence as an Evidence entity.

        Uses ``ExtractEvidenceOperation``'s fill + score helpers so the
        scoring logic is shared with the legacy path — including the
        OpenAlex bibliometric resolution, agent assessment fallback,
        and provider-supplied score chain. The evidence is marked
        ``extracted=True`` because content is already attached.
        """
        evidence = Evidence(
            objective_id=objective.entity_id,
            source_type=gathered.source_type or "unknown",
            source_ref=gathered.source_ref,
            extracted=True,
            sub_investigation_id=sub_investigation_id,
        )
        # Instantiate ExtractEvidenceOperation only to reuse its
        # fill/score helpers — we do NOT call its execute(). Constructed
        # with the same dependencies this operation already holds.
        extract_helper = ExtractEvidenceOperation(
            self.repo,
            self.agent_runner,
            evidence_gatherer=self.evidence_gatherer,
            quality_scorer=self.quality_scorer,
            embedding_model=self.embedding_model,
        )
        extract_helper._fill_evidence_from_gathered(evidence, gathered)
        await extract_helper._score_evidence(evidence, gathered)
        await self.repo.save(evidence)
        return evidence.entity_id
