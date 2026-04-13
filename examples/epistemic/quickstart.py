"""Quickstart example for mosaic-epistemic.

Demonstrates core mechanics without requiring an LLM:
  - Create a repository with in-memory storage
  - Define an objective, evidence, and claims
  - Validate promotion through stage gates
  - Use the pattern scheduler to discover pending work

Run:
    uv run python packages/epistemic/examples/quickstart.py
"""

import asyncio

from epistemic import (
    InMemoryStorageBackend,
    EpistemicRepository,
    Objective,
    Evidence,
    Claim,
    ClaimStage,
    Uncertainty,
    UncertaintyType,
    UncertaintyScope,
    BLOCKING_TYPES,
    validate_promotion,
    get_next_stage,
    check_degeneracy,
    quality_weighted_evidence_sum,
    compute_confidence_score,
    PatternScheduler,
)


async def main() -> None:
    # ── 1. Storage & Repository ──────────────────────────────────────────
    backend = InMemoryStorageBackend()
    repo = EpistemicRepository(backend)

    # ── 2. Objective ─────────────────────────────────────────────────────
    obj = Objective(
        objective_id="obj-1",
        description="Does spaced repetition improve long-term retention?",
        phase="planned",
    )
    await repo.save(obj)
    print(f"Objective: {obj.description}")

    # ── 3. Evidence ──────────────────────────────────────────────────────
    e1 = Evidence(
        objective_id="obj-1",
        source_type="journal",
        extracted_content="Cepeda et al. (2006) meta-analysis of spacing effect, d=0.46",
        quality_score=0.85,
        extracted=True,
    )
    e2 = Evidence(
        objective_id="obj-1",
        source_type="journal",
        extracted_content="Karpicke & Roediger (2008): retrieval practice boosts retention",
        quality_score=0.80,
        extracted=True,
    )
    await repo.save(e1)
    await repo.save(e2)
    print(f"\nEvidence collected: {await repo.count('evidence')}")

    # ── 4. Claims ────────────────────────────────────────────────────────
    claim = Claim(
        objective_id="obj-1",
        statement="Spaced repetition improves long-term retention",
        stage=ClaimStage.HYPOTHESIS,
        scrutiny_verdict="pass",
        evidence_ids=[e1.entity_id, e2.entity_id],
    )
    await repo.save(claim)
    print(f"\nClaim: {claim.statement}")
    print(f"  Stage: {claim.stage.value}")
    print(f"  Evidence count: {claim.evidence_count}")

    # ── 5. Quality-weighted evidence ─────────────────────────────────────
    evidence_sum = await quality_weighted_evidence_sum(claim, repo)
    print(f"  Quality-weighted evidence sum: {evidence_sum:.2f}")

    # ── 6. Stage gate validation ─────────────────────────────────────────
    target = get_next_stage(claim.stage)
    print(f"\nAttempting promotion: {claim.stage.value} -> {target.value}")

    gate_result = await validate_promotion(
        claim=claim,
        target_stage=target,
        repo=repo,
    )
    print(f"  Gate passed: {gate_result.passed}")
    print(f"  Reason: {gate_result.reason}")

    if gate_result.passed:
        from_stage = claim.stage
        claim.stage = target
        claim.record_promotion(from_stage, target, "Gate passed")
        await repo.save(claim)
        print(f"  New stage: {claim.stage.value}")

    # ── 7. Confidence score ──────────────────────────────────────────────
    avg_quality = evidence_sum / max(claim.evidence_count, 1)
    confidence = compute_confidence_score(stage=claim.stage, avg_quality=avg_quality)
    print(f"\nConfidence score: {confidence:.2f}")

    # ── 8. Uncertainty ───────────────────────────────────────────────────
    u = Uncertainty(
        objective_id="obj-1",
        description="Effect size may vary across age groups",
        uncertainty_type=UncertaintyType.UNKNOWN,
        scope=UncertaintyScope.CLAIM,
    )
    await repo.save(u)
    is_blocking = u.uncertainty_type in BLOCKING_TYPES
    print(f"\nUncertainty: {u.description}")
    print(f"  Type: {u.uncertainty_type.value}, Blocking: {is_blocking}")

    # ── 9. Degeneracy check ──────────────────────────────────────────────
    warnings = check_degeneracy(claim)
    print(f"\nDegeneracy warnings: {len(warnings)}")
    for code, msg in warnings:
        print(f"  [{code}] {msg}")

    # ── 10. Pattern scheduler ────────────────────────────────────────────
    scheduler = PatternScheduler(repo)
    work = await scheduler.get_pending_work(objective_id="obj-1")
    print(f"\nPending work items: {len(work)}")
    for w in work:
        print(f"  {w.operation} on {w.entity_type} {w.entity_id[:8]}...")

    print("\nDone.")


if __name__ == "__main__":
    asyncio.run(main())
