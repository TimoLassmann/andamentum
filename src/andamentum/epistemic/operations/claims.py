"""Claim proposal operations.

Clusters evidence, extracts assertions, and drafts claims. Contains the
shared ``select_top_k_evidence`` function used by both ProposeClaimsOperation
and ScrutiniseClaimOperation (in scrutiny.py).

Depends on: base (BaseOperation, OperationResult, DEDUP_SIMILARITY_THRESHOLD, GatheredEvidence)
Operates on: Objective, Evidence, Claim entities
"""

from typing import TYPE_CHECKING, Optional

from .base import (
    BaseOperation,
    DEDUP_SIMILARITY_THRESHOLD,
    OperationInput,
    OperationResult,
)

from ..dedup import deduplicate_evidence
from ..entities import (
    Claim,
    ClaimStage,
    Evidence,
    Objective,
)

if TYPE_CHECKING:
    from ..dedup import EvidenceCluster
    from ..repository import EpistemicRepository


# ══════════════════════════════════════════════════════════════════════════════
# EVIDENCE SELECTION
# Shared by ProposeClaimsOperation (initial) and ScrutiniseClaimOperation
# (after investigation). Groups similar evidence, ranks by quality, selects
# the most informative subset.
# ══════════════════════════════════════════════════════════════════════════════

# Maximum number of representative evidence items each LLM-touching panel
# operation will inspect (assess_convergence's per-rep classify and pairwise
# independence; deductive validation; computational verification; abductive
# integration; synthesis). Bounds per-claim LLM cost where the consumer's
# work scales with rep count. Independent of clustering: every cluster gets
# a representative chosen, posterior weighting reads them all, this cap only
# limits which subset is sent to those specific LLM panels.
LLM_PANEL_CAP = 10

# Backwards-compatibility alias. Older callers may import EVIDENCE_TOP_K
# directly. Kept until external consumers (paper harness, integration runs)
# migrate to LLM_PANEL_CAP.
EVIDENCE_TOP_K = LLM_PANEL_CAP


async def top_n_representatives(
    evidence: list[Evidence],
    n: int = LLM_PANEL_CAP,
    *,
    claim_text: str | None = None,
    embedding_model: str | None = None,
) -> list[Evidence]:
    """Pick the top N evidence pieces, ranked by claim-relevance.

    The IBE chain (and other LLM panel consumers) need the N pieces of
    evidence that speak most directly to the claim under investigation.
    Ranking by ``quality_score`` — the previous behaviour — selects on
    source reliability and extraction completeness, which is the wrong
    axis: a high-quality piece that's only tangentially relevant gets
    passed through, while a directly-relevant piece from a less-rigorous
    source gets demoted. SciFact case 1163 trace showed the quality
    ranking preserving 3:1 supports majority but the IBE chain still
    committing contradicts — the deeper bug is in the loveliness scorer,
    but the quality-based filter wasn't doing IBE any favours either.

    When ``claim_text`` and ``embedding_model`` are both provided, ranks
    by cosine similarity between each piece's content embedding and the
    claim text embedding. This is the same Reichenbach common-cause-at-
    retrieval principle applied at the adversarial-search rerank
    (commit 0f039ef): the stable signal is the claim itself, not the
    transient quality_score.

    When either is None, falls back to ranking by ``quality_score`` (the
    legacy behaviour) so test paths and other callers without embedding
    infrastructure don't crash. Production paths should provide both.

    Tiebreaker is ``source_ref`` so selection is deterministic across
    re-runs on the same evidence base.
    """
    if not evidence:
        return []

    if not claim_text or not embedding_model:
        # Legacy quality-score ranking — preserved for callers without
        # embedding infrastructure (notably some test paths).
        ranked = sorted(
            evidence,
            key=lambda e: (-(e.quality_score or 0.0), e.source_ref or ""),
        )
        return ranked[:n]

    # Embed claim + every candidate in one batch. embeddinggemma-2 at
    # 768-dim is fast enough that batching all evidence per call is fine.
    from ..embeddings import embed_texts
    from ..similarity import cosine_similarity

    texts = [claim_text] + [ev.extracted_content or "" for ev in evidence]
    try:
        embeddings = await embed_texts(texts, model=embedding_model)
    except RuntimeError:
        # Embedding endpoint unreachable — fall back to quality-score
        # rather than crash the IBE chain. The downstream consumer will
        # still operate on a sensible (if less-relevance-optimal)
        # candidate set.
        ranked = sorted(
            evidence,
            key=lambda e: (-(e.quality_score or 0.0), e.source_ref or ""),
        )
        return ranked[:n]

    claim_emb = embeddings[0]
    candidate_embs = embeddings[1:]
    scored = list(
        zip(evidence, (cosine_similarity(claim_emb, ce) for ce in candidate_embs))
    )
    # Sort by similarity descending; tiebreak on source_ref for stable
    # output across re-runs.
    scored.sort(key=lambda pair: (-pair[1], pair[0].source_ref or ""))
    return [ev for ev, _ in scored[:n]]


def _independence_group(ev: Evidence) -> str:
    """The Reichenbach independence class of one evidence item's source.

    This is the existing per-provider ``independence_group`` concept
    (``providers/*.py`` class attributes — e.g. arXiv and bioRxiv are both
    ``"preprint_archive"``, so two hits from them are one source, not two;
    see ``providers/CONTRIBUTING.md`` and FINDINGS.md §4.5), looked up by
    ``ev.source_type``. Falls back to the raw ``source_type`` when it does
    not name a registered provider (e.g. an offline benchmark corpus
    provider), so an unrecognised source is at minimum its own singleton
    group rather than silently merging into a group it was never shown to
    share.

    Local import: every provider module does ``from ..operations import
    GatheredEvidence``, so ``providers`` depends on ``operations`` at
    import time — a module-level ``from ..providers import ...`` here
    would be a circular import. Deferred to call time, matching this
    file's existing pattern for optional/cross-cutting imports (see
    ``embed_and_group`` and ``judge_evidence`` below).
    """
    from ..providers import PROVIDER_REGISTRY

    provider_cls = PROVIDER_REGISTRY.get(ev.source_type)
    group = getattr(provider_cls, "independence_group", None)
    return group or ev.source_type or "unknown"


def _direction_independence_groups(
    extracted: list[Evidence],
    clusters: list["EvidenceCluster"],
    cluster_ids: list[str],
) -> dict[str, set[tuple[str, str]]]:
    """For each judged direction, the set of genuinely separate sources.

    A "genuinely separate source" (membership unit) is a
    ``(cluster_id, independence_group)`` pair: two evidence items count as
    the SAME source only when they are both near-duplicates by content
    (HDBSCAN put them in the same cluster) AND from the same provider
    independence class. Either axis differing — a different content
    cluster, or a different independence_group — makes them independent
    in the Reichenbach sense, so each contributes its own membership unit.

    Scoped by ``support_judgment`` (falling back to the literal string
    ``"unjudged"``) because only evidence agreeing on the SAME direction
    can corroborate one another — a contradicting cluster is not
    corroboration. ``ProposeClaimsOperation`` calls this function BEFORE
    any evidence has been judged (judging happens per-cluster, after claim
    drafting), so at that call site every item shares the "unjudged"
    bucket; this is not a regression, since the pre-fix code was equally
    unable to scope by direction at that point in the pipeline.
    """
    groups: dict[str, set[tuple[str, str]]] = {}
    for cluster, cluster_id in zip(clusters, cluster_ids):
        by_direction: dict[str, set[str]] = {}
        for idx in cluster.member_indices:
            ev = extracted[idx]
            direction = ev.support_judgment or "unjudged"
            by_direction.setdefault(direction, set()).add(_independence_group(ev))
        for direction, indep_groups in by_direction.items():
            bucket = groups.setdefault(direction, set())
            bucket.update((cluster_id, group) for group in indep_groups)
    return groups


async def select_top_k_evidence(
    repo: "EpistemicRepository",
    extracted: list[Evidence],
    top_k: Optional[int] = None,  # accepted for backwards compatibility; ignored
    embedding_model: Optional[str] = None,
) -> tuple[list[Evidence], int, int]:
    """Cluster evidence and pick representatives for every cluster.

    HDBSCAN groups evidence by semantic similarity. Inside each cluster we
    promote the medoid, up to 3 boundary members, and the best-quality
    member to ``cluster_status="representative"``; the rest become
    ``cluster_status="corroborative"``.

    Two DIFFERENT signals get written onto every representative, and they
    must not be confused (plan: docs/epistemic/PRINCIPLE_FIDELITY_PLAN.md
    §4.4):

    - ``retrieval_cluster_size`` — the raw HDBSCAN cluster size: how many
      near-duplicate items were collapsed into this representative. A
      retrieval/redundancy count, nothing more.
    - ``corroboration_count`` — the number of GENUINELY INDEPENDENT
      sources (Reichenbach common-cause) agreeing with this
      representative's direction: distinct ``(content cluster,
      independence_group)`` pairs, via ``_direction_independence_groups``.
      This is what the posterior and scrutiny weight amplify by
      ``1 + log(n)`` — many near-duplicate documents from one source no
      longer inflate it; genuinely separate sources do.

    No cap on the number of clusters: every cluster contributes
    representatives. Cost-bounded LLM panel operations (convergence
    classification, pairwise independence, deductive validation,
    computational verification, integration, synthesis) apply
    ``LLM_PANEL_CAP`` themselves, sorted by quality, so cost stays comparable
    to the previous top-K behaviour.

    Called from three places:
    - ProposeClaimsOperation: clusters initial evidence before claim proposal
      (before any evidence-claim judgment exists — see
      ``_direction_independence_groups``'s "unjudged" bucket)
    - ScrutiniseClaimOperation: clusters investigation-fetched evidence before
      re-scrutiny (claim-scoped, judged)
    - RunVerification (graph/nodes.py): re-clusters a SUPPORTED claim's
      evidence before verification (claim-scoped, judged)

    Args:
        repo: Repository for saving updated evidence entities
        extracted: All extracted, non-invalidated evidence to select from
        top_k: Accepted for backwards compatibility but ignored. The cap was
            removed to stop discarding completed work; consumers cap LLM use.
        embedding_model: Embedding model id for HDBSCAN.

    Returns:
        3-tuple of (representatives, total_clusters, deferred_count).
        ``deferred_count`` is always 0 — kept in the signature so existing
        callers don't break. Every cluster now produces representatives.
    """
    del top_k  # accepted for backwards compatibility; clustering is no longer capped
    import logging as _logging

    _sel_log = _logging.getLogger(__name__ + ".select_evidence")

    _sel_log.warning(
        "[select_top_k_evidence] Called with %d evidence items",
        len(extracted),
    )

    if len(extracted) < 2:
        _sel_log.warning("[select_top_k_evidence] < 2 items, returning as-is")
        return extracted, 1, 0

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

    # Step 3: Mark evidence entities. Every cluster contributes representatives;
    # there is no top-K filter here.
    cluster_ids: list[str] = []
    for cluster in clusters:
        cluster_id = _uuid.uuid4().hex[:12]
        cluster_ids.append(cluster_id)
        rep_set = set(cluster.representative_indices)

        for idx in cluster.member_indices:
            ev = extracted[idx]
            ev.cluster_id = cluster_id

            if idx in rep_set:
                ev.cluster_status = "representative"
                # Retrieval-time redundancy (near-duplicate volume) — NOT
                # corroboration. See the field's docstring.
                ev.retrieval_cluster_size = cluster.count
                ev.corroborating_sources = [
                    extracted[j].source_ref
                    for j in cluster.member_indices
                    if j != idx and extracted[j].source_ref
                ]
            else:
                ev.cluster_status = "corroborative"

    # Step 4: Reichenbach corroboration — independence, not redundancy
    # (plan §4.4). corroboration_count is this module's sole write of the
    # field; a field-ownership guard enforces that (see
    # tests/test_field_ownership.py).
    direction_groups = _direction_independence_groups(extracted, clusters, cluster_ids)
    for cluster in clusters:
        for idx in cluster.representative_indices:
            ev = extracted[idx]
            direction = ev.support_judgment or "unjudged"
            ev.corroboration_count = len(direction_groups[direction])

    # Persist every touched item once, now that all its fields are final.
    touched_indices = {idx for cluster in clusters for idx in cluster.member_indices}
    for idx in touched_indices:
        await repo.save(extracted[idx])

    representatives = [e for e in extracted if e.cluster_status == "representative"]
    total_clusters = len(clusters)
    _sel_log.warning(
        "[select_top_k_evidence] Result: %d representatives, %d corroborative (from %d total, %d clusters)",
        len(representatives),
        sum(1 for e in extracted if e.cluster_status == "corroborative"),
        len(extracted),
        total_clusters,
    )
    # deferred_count returned as 0 for backwards-compat; the concept is retired.
    return representatives, total_clusters, 0


class ProposeClaimsOperation(BaseOperation):
    """Propose claims from extracted evidence via 3-step decomposition.

    Step 1: extract_assertion — narrow agent, one call per evidence item
    Step 2: cluster_assertions — deterministic embedding-based clustering
    Step 3: draft_claim — narrow agent, one call per cluster

    Falls back to epistemic_propose_claims (monolithic) when no agent_runner
    is available, and to a single cluster when embeddings are unavailable.
    """

    entity_type = "objective"

    async def execute(self, work: OperationInput) -> OperationResult:
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
                screen = await self.run_agent(
                    "epistemic_screen_relevance",
                    research_question=clarified,
                    evidence_content=ev.extracted_content,
                    source_info=f"[{ev.source_type}] {ev.source_ref}",
                )
                if screen.is_relevant:
                    relevant.append(ev)
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
        extracted, _total_clusters, _deferred_count = await select_top_k_evidence(
            self.repo, extracted, embedding_model=self.embedding_model
        )

        # ── Step 1: Extract one assertion per evidence ──────────────────────
        # Descriptive-drift defense lives in the extract_assertion agent's own
        # prompt (with good/bad examples). No second-opinion validator here —
        # that was over-eager on small models and rejected legitimate findings.
        assertions: list[tuple[str, str]] = []  # (assertion_text, evidence_id)
        for ev in extracted:
            result = await self.run_agent(
                "epistemic_extract_assertion",
                evidence_content=ev.extracted_content,
                research_question=clarified,
            )
            assertion_text = result.assertion

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

            result = await self.run_agent(
                "epistemic_draft_claim",
                assertions=joined_assertions,
                research_question=clarified,
            )
            claim_statement = result.statement
            claim_scope = result.scope

            if not claim_statement:
                continue  # Agent returned empty — skip this cluster

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
            from ..judge import apply_judgment, judge_evidence as _judge

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
                apply_judgment(ev, judgment)
                await self.repo.save(ev)

        objective.claims_proposed = True
        objective.phase = "claims_proposed"
        await self.repo.save(objective)

        deferred_text = (
            f" ({_deferred_count} clusters deferred)" if _deferred_count > 0 else ""
        )
        return OperationResult(
            success=True,
            entity_id=objective.entity_id,
            message=(
                f"Proposed {len(created_claims)} claims from {len(assertions)} assertions"
                f" in {len(clusters)} clusters"
                f" ({len(extracted)} of {_total_clusters} evidence clusters selected{deferred_text})"
            ),
            created_entities=created_claims,
        )
