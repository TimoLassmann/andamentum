"""Claim proposal operations (Phase 4).

Clusters evidence, extracts assertions, and drafts claims. Contains the
shared ``select_top_k_evidence`` function used by both ProposeClaimsOperation
and ScrutiniseClaimOperation (in scrutiny.py).

Depends on: base (BaseOperation, OperationResult, DEDUP_SIMILARITY_THRESHOLD, GatheredEvidence)
Operates on: Objective, Evidence, Claim entities
"""

from typing import TYPE_CHECKING, Optional

from .base import BaseOperation, OperationResult, DEDUP_SIMILARITY_THRESHOLD, WorkItem

from ..dedup import deduplicate_evidence
from ..entities import (
    Claim,
    ClaimStage,
    Evidence,
    Objective,
)

if TYPE_CHECKING:
    from ..repository import EpistemicRepository


# ══════════════════════════════════════════════════════════════════════════════
# EVIDENCE SELECTION
# Shared by ProposeClaimsOperation (initial) and ScrutiniseClaimOperation
# (after investigation). Groups similar evidence, ranks by quality, selects
# the most informative subset.
# ══════════════════════════════════════════════════════════════════════════════

EVIDENCE_TOP_K = 5  # Maximum clusters to process per cycle


async def select_top_k_evidence(
    repo: "EpistemicRepository",
    extracted: list[Evidence],
    top_k: int = EVIDENCE_TOP_K,
    embedding_model: Optional[str] = None,
) -> list[Evidence]:
    """Select the most informative evidence subset via cluster-ranked top-K.

    Clusters evidence by semantic similarity (HDBSCAN), ranks clusters by
    the best quality_score of any member, and selects the top-K clusters.
    Within each cluster, representatives are: the medoid (most central),
    up to 3 boundary members (most diverse), and the best-quality member.

    When HDBSCAN finds no cluster structure (all singletons), this
    naturally degrades to top-K-by-quality selection.

    Called from two places:
    - ProposeClaimsOperation: clusters initial evidence before claim proposal
    - ScrutiniseClaimOperation: clusters investigation-fetched evidence before
      re-scrutiny

    Args:
        repo: Repository for saving updated evidence entities
        extracted: All extracted, non-invalidated evidence to select from
        top_k: Maximum number of clusters to process

    Returns:
        Filtered list containing only representative evidence
    """
    import logging as _logging

    _sel_log = _logging.getLogger(__name__ + ".select_evidence")

    _sel_log.warning(
        "[select_top_k_evidence] Called with %d evidence items, top_k=%d",
        len(extracted),
        top_k,
    )

    if len(extracted) < 2:
        _sel_log.warning("[select_top_k_evidence] < 2 items, returning as-is")
        return extracted

    import uuid as _uuid

    # Step 1: Cluster by semantic similarity
    evidence_texts = [e.extracted_content or "" for e in extracted]
    if not embedding_model:
        raise RuntimeError(
            "embedding_model is required for evidence deduplication. Pass embedding_model= to create_operations()."
        )
    clusters = await deduplicate_evidence(
        evidence_texts, min_cluster_size=2, embedding_model=embedding_model
    )

    _sel_log.warning(
        "[select_top_k_evidence] HDBSCAN produced %d clusters from %d items "
        "(singletons=%d, multi-member=%d)",
        len(clusters),
        len(extracted),
        sum(1 for c in clusters if c.count == 1),
        sum(1 for c in clusters if c.count > 1),
    )

    # Step 2: Augment representatives with best-quality member per cluster
    for cluster in clusters:
        best_quality_idx = max(
            cluster.member_indices,
            key=lambda i: extracted[i].quality_score or 0.0,
        )
        if best_quality_idx not in cluster.representative_indices:
            cluster.representative_indices.append(best_quality_idx)

    # Step 3: Rank clusters by best member quality (descending)
    clusters.sort(
        key=lambda c: max(
            (extracted[i].quality_score or 0.0) for i in c.member_indices
        ),
        reverse=True,
    )

    # Step 4: Select top-K clusters
    k = min(top_k, len(clusters))
    selected_clusters = clusters[:k]
    deferred_clusters = clusters[k:]

    _sel_log.warning(
        "[select_top_k_evidence] Selected %d clusters (top-K=%d), deferred %d clusters",
        len(selected_clusters),
        k,
        len(deferred_clusters),
    )

    # Step 5: Mark evidence entities
    for cluster in selected_clusters:
        cluster_id = _uuid.uuid4().hex[:12]
        rep_set = set(cluster.representative_indices)

        for idx in cluster.member_indices:
            ev = extracted[idx]
            ev.cluster_id = cluster_id

            if idx in rep_set:
                ev.cluster_status = "representative"
                ev.corroboration_count = cluster.count
                ev.corroborating_sources = [
                    extracted[j].source_ref
                    for j in cluster.member_indices
                    if j != idx and extracted[j].source_ref
                ]
            else:
                ev.cluster_status = "corroborative"

            await repo.save(ev)

    for cluster in deferred_clusters:
        cluster_id = _uuid.uuid4().hex[:12]
        for idx in cluster.member_indices:
            ev = extracted[idx]
            ev.cluster_status = "deferred"
            ev.cluster_id = cluster_id
            await repo.save(ev)

    # Step 6: Return only representatives
    representatives = [e for e in extracted if e.cluster_status == "representative"]
    _sel_log.warning(
        "[select_top_k_evidence] Result: %d representatives, %d corroborative, %d deferred (from %d total)",
        len(representatives),
        sum(1 for e in extracted if e.cluster_status == "corroborative"),
        sum(1 for e in extracted if e.cluster_status == "deferred"),
        len(extracted),
    )
    return representatives


class ProposeClaimsOperation(BaseOperation):
    """Propose claims from extracted evidence via 3-step decomposition.

    Step 1: extract_assertion — narrow agent, one call per evidence item
    Step 2: cluster_assertions — deterministic embedding-based clustering
    Step 3: draft_claim — narrow agent, one call per cluster

    Falls back to epistemic_propose_claims (monolithic) when no agent_runner
    is available, and to a single cluster when embeddings are unavailable.
    """

    entity_type = "objective"

    async def execute(self, work: WorkItem) -> OperationResult:
        objective = await self.repo.get("objective", work.entity_id)

        if not isinstance(objective, Objective):
            return OperationResult(
                success=False,
                entity_id=work.entity_id,
                message="Entity is not Objective",
            )

        if objective.claims_proposed:
            return OperationResult(
                success=True,
                entity_id=work.entity_id,
                message="Claims already proposed",
            )

        # Get all extracted evidence
        evidence_list = await self.repo.query(
            "evidence",
            objective_id=objective.entity_id,
            extracted=True,
        )

        # Filter to valid extracted evidence
        extracted = [
            e
            for e in evidence_list
            if isinstance(e, Evidence)
            and e.extracted
            and e.extracted_content
            and not e.invalidated
        ]

        # ── Relevance screening: filter evidence by research question ───
        # One cheap LLM call per evidence item. Prevents off-topic papers
        # (keyword-only matches) from consuming downstream compute.
        clarified = objective.clarified_question or objective.description

        if self.agent_runner and extracted:
            relevant: list[Evidence] = []
            for ev in extracted:
                try:
                    screen = await self.run_agent(
                        "epistemic_screen_relevance",
                        research_question=clarified,
                        evidence_content=ev.extracted_content,
                        source_info=f"[{ev.source_type}] {ev.source_ref}",
                    )
                    if screen.is_relevant:
                        relevant.append(ev)
                except Exception:
                    relevant.append(ev)  # Screening failed — include by default
            extracted = relevant

        if not self.agent_runner:
            # No agent runner: create a placeholder claim
            evidence_entity_ids = [
                e.entity_id for e in evidence_list if isinstance(e, Evidence)
            ]
            claim = Claim(
                objective_id=objective.entity_id,
                statement=f"[Placeholder claim for: {objective.description[:50]}]",
                scope="specific",
                stage=ClaimStage.HYPOTHESIS,
                evidence_ids=list(evidence_entity_ids),
            )
            await self.repo.save(claim)

            objective.claims_proposed = True
            objective.phase = "claims_proposed"
            await self.repo.save(objective)

            return OperationResult(
                success=True,
                entity_id=objective.entity_id,
                message="Proposed 1 claims",
                created_entities=[claim.entity_id],
            )

        # If no external evidence, create a world_knowledge evidence stub
        if not extracted:
            wk_evidence = Evidence(
                objective_id=objective.entity_id,
                source_type="world_knowledge",
                source_ref="LLM training data",
                extracted_content=f"Draw on your knowledge to propose claims about: {objective.description}",
                extracted=True,
            )
            await self.repo.save(wk_evidence)
            extracted = [wk_evidence]

        # ── Evidence selection: cluster-ranked top-K ────────────────────────
        # Clusters evidence by semantic similarity, ranks clusters by quality,
        # selects top-K clusters, and returns representative evidence only.
        extracted = await select_top_k_evidence(
            self.repo, extracted, embedding_model=self.embedding_model
        )

        # ── Step 1: Extract one assertion per evidence ──────────────────────
        # Descriptive-drift defense lives in the extract_assertion agent's own
        # prompt (with good/bad examples). No second-opinion validator here —
        # that was over-eager on small models and rejected legitimate findings.
        assertions: list[tuple[str, str]] = []  # (assertion_text, evidence_id)
        for ev in extracted:
            try:
                result = await self.run_agent(
                    "epistemic_extract_assertion",
                    evidence_content=ev.extracted_content,
                    research_question=clarified,
                )
                assertion_text = result.assertion
            except Exception:
                assertion_text = None  # Agent call failed — skip this evidence

            if assertion_text:
                assertions.append((assertion_text, ev.entity_id))

        if not assertions:
            # All extractions failed — mark done with no claims
            objective.claims_proposed = True
            objective.phase = "claims_proposed"
            await self.repo.save(objective)
            return OperationResult(
                success=False,
                entity_id=work.entity_id,
                message="No assertions extracted from evidence",
            )

        # ── Step 2: Cluster assertions (deterministic) ──────────────────────
        assertion_texts = [a[0] for a in assertions]

        from ..similarity import embed_and_group

        if not self.embedding_model:
            raise RuntimeError(
                "embedding_model is required for assertion clustering. Pass embedding_model= to create_operations()."
            )
        clusters = await embed_and_group(
            assertion_texts,
            threshold=DEDUP_SIMILARITY_THRESHOLD,
            embedding_model=self.embedding_model,
        )

        # ── Step 3: Draft one claim per cluster ─────────────────────────────
        # Falsifiability + aligned-to-question defense lives in the draft_claim
        # agent's own prompt (with good/bad examples). No second-opinion
        # validator here — over-eager on small models.
        created_claims: list[str] = []
        for cluster_indices in clusters:
            cluster_assertions = [assertions[i] for i in cluster_indices]
            cluster_texts = [a[0] for a in cluster_assertions]
            cluster_evidence_ids = [a[1] for a in cluster_assertions]
            joined_assertions = "\n".join(f"- {t}" for t in cluster_texts)

            claim_statement = None
            claim_scope = None

            try:
                result = await self.run_agent(
                    "epistemic_draft_claim",
                    assertions=joined_assertions,
                    research_question=clarified,
                )
                claim_statement = result.statement
                claim_scope = result.scope
            except Exception:
                continue  # Agent call failed — skip this cluster

            if not claim_statement:
                continue  # Agent returned empty — skip this cluster

            try:
                claim = Claim(
                    objective_id=objective.entity_id,
                    statement=claim_statement,
                    scope=claim_scope,
                    stage=ClaimStage.HYPOTHESIS,
                    evidence_ids=cluster_evidence_ids,
                )
                await self.repo.save(claim)
                created_claims.append(claim.entity_id)

                # ── Judge each evidence item against the claim ─────────
                from ..judge import judge_evidence as _judge

                for eid in cluster_evidence_ids:
                    ev = await self.repo.get("evidence", eid)
                    if ev.support_judgment is not None:
                        continue
                    judgment = await _judge(
                        claim_statement=claim.statement,
                        claim_scope=claim.scope,
                        evidence_content=ev.extracted_content or "",
                        evidence_source=f"{ev.source_type}: {ev.source_ref}",
                        runner=self.agent_runner,
                    )
                    verdict = judgment.verdict.lower().strip()
                    if verdict not in ("supports", "contradicts", "no_bearing"):
                        verdict = "no_bearing"
                    ev.support_judgment = verdict
                    ev.judgment_reasoning = judgment.reasoning
                    await self.repo.save(ev)

            except Exception:
                pass  # Best effort per cluster

        objective.claims_proposed = True
        objective.phase = "claims_proposed"
        await self.repo.save(objective)

        return OperationResult(
            success=True,
            entity_id=objective.entity_id,
            message=f"Proposed {len(created_claims)} claims from {len(assertions)} assertions in {len(clusters)} clusters",
            created_entities=created_claims,
        )
