"""Run epistemic integration tests — verify philosophical pathways fire.

Usage:
    uv run python packages/epistemic/integration_tests/run.py --model openai:gpt-5.4-mini
    uv run python packages/epistemic/integration_tests/run.py --tradition doyle --model openai:gpt-5.4-mini
    uv run python packages/epistemic/integration_tests/run.py --list
    uv run python packages/epistemic/integration_tests/run.py --model openai:gpt-5.4-mini --keep

These are NOT unit tests. Each test runs the full pipeline with real LLM calls
and real web search, then inspects the database to verify that specific
epistemological traditions actually fired. Expect ~3-5 minutes per test.
"""

import argparse
import asyncio
import json
import sqlite3
import sys
import tempfile
from pathlib import Path

# Ensure this script's directory is on the path for the local questions module
sys.path.insert(0, str(Path(__file__).parent))
from questions import PATHWAY_TESTS, PathwayTest  # type: ignore[import-not-found]


# ── Database inspection ────────────────────────────────────────────────


def _get_execution_steps(db_path: str) -> list[dict]:
    """Load all execution steps from the database."""
    db = sqlite3.connect(db_path)
    rows = db.execute(
        "SELECT metadata FROM documents "
        "WHERE deleted_at IS NULL "
        "AND json_extract(metadata, '$.epistemic_type') = 'execution_step' "
        "ORDER BY CAST(json_extract(metadata, '$.step_number') AS INTEGER)"
    ).fetchall()
    db.close()
    return [json.loads(r[0]) for r in rows]


def _get_entities(db_path: str, entity_type: str) -> list[dict]:
    """Load all entities of a given type from the database."""
    db = sqlite3.connect(db_path)
    rows = db.execute(
        "SELECT metadata FROM documents "
        "WHERE deleted_at IS NULL "
        "AND json_extract(metadata, '$.epistemic_type') = ?",
        (entity_type,),
    ).fetchall()
    db.close()
    return [json.loads(r[0]) for r in rows]


# ── Diagnostic summaries ──────────────────────────────────────────────


async def print_diagnostics(db_path: str, tradition: str) -> None:
    """Print a rich diagnostic summary of what happened during the run."""
    if not Path(db_path).exists():
        print("  (no database to inspect)")
        return

    steps = _get_execution_steps(db_path)
    claims = _get_entities(db_path, "claim")
    evidence = _get_entities(db_path, "evidence")
    uncertainties = _get_entities(db_path, "uncertainty")
    objective = _get_entities(db_path, "objective")

    # ── General run overview ──────────────────────────────────────────
    print(f"\n  {'─' * 56}")
    print("  Run Overview")
    print(f"  {'─' * 56}")

    total_ops = len(steps)
    succeeded = sum(1 for s in steps if s.get("success"))
    failed = sum(1 for s in steps if not s.get("success"))
    print(f"  Operations:    {total_ops} total, {succeeded} succeeded, {failed} failed")
    print(f"  Evidence:      {len(evidence)} items")
    print(f"  Claims:        {len(claims)}")
    print(f"  Uncertainties: {len(uncertainties)}")

    # Question classification
    if objective:
        qtype = objective[0].get("question_type", "unknown")
        print(f"  Question type: {qtype}")

    # Operation profile (counts per type)
    op_counts: dict[str, int] = {}
    for s in steps:
        op = s.get("operation", "?")
        op_counts[op] = op_counts.get(op, 0) + 1
    print("\n  Operation counts:")
    for op, count in sorted(op_counts.items(), key=lambda x: -x[1]):
        print(f"    {op:<35} {count:>3}x")

    # ── Claim details ─────────────────────────────────────────────────
    print(f"\n  {'─' * 56}")
    print("  Claims")
    print(f"  {'─' * 56}")
    for c in claims:
        stmt = c.get("statement", "?")[:65]
        stage = c.get("stage", "?")
        adv = c.get("adversarial_balance")
        adv_str = f"adv={adv:.2f}" if adv is not None else "adv=n/a"
        mod = c.get("modification_count", 0)
        ev_count = len(c.get("evidence_ids", []))
        verdict = c.get("scrutiny_verdict", "n/a")
        abandoned = " ABANDONED" if c.get("abandoned") else ""
        print(f"  [{stage:<12}] {stmt}")
        print(
            f"    scrutiny={verdict}  {adv_str}  evidence={ev_count}  mods={mod}{abandoned}"
        )

        # Promotion history (shows demotions)
        for h in c.get("promotion_history", []):
            direction = (
                "demoted"
                if _stage_rank(h.get("to", "")) < _stage_rank(h.get("from", ""))
                else "promoted"
            )
            print(
                f"    {direction}: {h.get('from')} -> {h.get('to')}  {h.get('justification', '')[:60]}"
            )

        # Predictions
        preds = c.get("predictions", [])
        if preds:
            print(f"    predictions: {len(preds)}")
            for p in preds:
                has_falsif = "yes" if p.get("failure_criteria") else "NO"
                spec = p.get("specificity")
                spec_str = f"spec={spec:.2f}" if spec is not None else ""
                print(
                    f"      - {p.get('statement', '?')[:55]}  falsif={has_falsif} {spec_str}"
                )

    # ── Evidence quality ──────────────────────────────────────────────
    print(f"\n  {'─' * 56}")
    print("  Evidence")
    print(f"  {'─' * 56}")
    invalidated = [e for e in evidence if e.get("invalidated")]
    judged = [e for e in evidence if e.get("support_judgment")]
    supports = [e for e in judged if e.get("support_judgment") == "supports"]
    contradicts = [e for e in judged if e.get("support_judgment") == "contradicts"]
    qualities: list[float] = [
        float(e["quality_score"])
        for e in evidence
        if e.get("quality_score") is not None
    ]

    print(
        f"  Total: {len(evidence)}  Judged: {len(judged)} (supports={len(supports)}, contradicts={len(contradicts)})"
    )
    print(f"  Invalidated: {len(invalidated)}")
    if qualities:
        avg_q = sum(qualities) / len(qualities)
        min_q = min(qualities)
        max_q = max(qualities)
        print(f"  Quality: avg={avg_q:.2f}  min={min_q:.2f}  max={max_q:.2f}")
    for inv in invalidated:
        print(
            f"    INVALIDATED: {inv.get('source_ref', '?')[:50]}  reason={inv.get('invalidation_reason', '?')[:40]}"
        )

    # ── TMS activity (Doyle) ──────────────────────────────────────────
    reval_steps = [s for s in steps if s.get("operation") == "revalidate_claim"]
    if reval_steps:
        print(f"\n  {'─' * 56}")
        print("  TMS Activity (Doyle)")
        print(f"  {'─' * 56}")
        for s in reval_steps:
            msg = s.get("message", "")
            changes = s.get("state_changes", [])
            print(f"  {msg}")
            for change in changes:
                if "stage:" in change.lower() or "adversarial" in change.lower():
                    print(f"    {change}")

    # ── Investigation cycles (Peirce) ─────────────────────────────────
    invest_steps = [s for s in steps if s.get("operation") == "investigate_claim"]
    if invest_steps:
        print(f"\n  {'─' * 56}")
        print("  Investigation Cycles (Peirce)")
        print(f"  {'─' * 56}")
        for s in invest_steps:
            created = s.get("created_entities", [])
            print(f"  {s.get('message', '')}  created={len(created)} stubs")

    # ── Scrutiny verdicts ─────────────────────────────────────────────
    scrutiny_steps = [s for s in steps if s.get("operation") == "scrutinise_claim"]
    if scrutiny_steps:
        print(f"\n  {'─' * 56}")
        print("  Scrutiny Verdicts")
        print(f"  {'─' * 56}")
        for s in scrutiny_steps:
            msg = s.get("message", "")
            print(f"  {msg}")

    # ── Contrastive evaluation (Lipton) ───────────────────────────────
    contrastive_steps = [
        s for s in steps if s.get("operation") == "contrastive_evaluation"
    ]
    if contrastive_steps:
        print(f"\n  {'─' * 56}")
        print("  Contrastive Evaluation (Lipton)")
        print(f"  {'─' * 56}")
        for s in contrastive_steps:
            print(f"  {s.get('message', '')}")

    # ── Adversarial search results ────────────────────────────────────
    adv_steps = [s for s in steps if s.get("operation") == "adversarial_search"]
    if adv_steps:
        print(f"\n  {'─' * 56}")
        print("  Adversarial Search")
        print(f"  {'─' * 56}")
        for s in adv_steps:
            print(f"  {s.get('message', '')}")

    # ── Promotion attempts ────────────────────────────────────────────
    promote_steps = [s for s in steps if s.get("operation") == "promote_claim"]
    if promote_steps:
        successes = [s for s in promote_steps if s.get("success")]
        failures = [s for s in promote_steps if not s.get("success")]
        print(f"\n  {'─' * 56}")
        print(f"  Promotions: {len(successes)} succeeded, {len(failures)} blocked")
        print(f"  {'─' * 56}")
        for s in successes:
            print(f"  OK: {s.get('message', '')}")
        for s in failures:
            print(f"  BLOCKED: {s.get('message', '')}")

    # ── Uncertainties summary ─────────────────────────────────────────
    blocking = [u for u in uncertainties if u.get("is_blocking")]
    resolved = [u for u in uncertainties if u.get("resolution") is not None]
    contrastive_uncerts = [
        u for u in uncertainties if u.get("created_by") == "contrastive_evaluation"
    ]
    if uncertainties:
        print(f"\n  {'─' * 56}")
        print(
            f"  Uncertainties: {len(uncertainties)} total, {len(blocking)} blocking, {len(resolved)} resolved"
        )
        print(f"  {'─' * 56}")
        if contrastive_uncerts:
            print(f"  Contrastive observations: {len(contrastive_uncerts)}")
            for u in contrastive_uncerts:
                bl = "BLOCKING" if u.get("is_blocking") else "caveat"
                print(f"    [{bl}] {u.get('description', '')[:70]}")

    # ── Confidence scoring ─────────────────────────────────────────────
    obj_id = objective[0].get("objective_id") if objective else None
    if obj_id:
        try:
            from andamentum.epistemic.confidence import (
                compute_answer_confidence,
                compute_posterior,
            )
            from andamentum.epistemic.repository import EpistemicRepository

            db_dir = str(Path(db_path).parent)
            db_name = Path(db_path).stem
            repo = await EpistemicRepository.for_database(db_name, db_dir=Path(db_dir))

            ac = await compute_answer_confidence(repo, obj_id)
            print(f"\n  {'─' * 56}")
            print("  Answer Confidence")
            print(f"  {'─' * 56}")
            print(f"  Overall: {ac.confidence:.2f} ({ac.level.upper()})")
            print(f"  Checks: {ac.passes} passed, {ac.failures} failed")
            for check in ac.checks:
                status = "PASS" if check.passed else "FAIL"
                print(f"    [{status}] {check.name}: {check.detail}")

            posterior = await compute_posterior(repo, obj_id)
            if posterior is not None:
                print(f"\n  {'─' * 56}")
                print("  Posterior confidence")
                print(f"  {'─' * 56}")
                print(f"  Confidence: {posterior.posterior:.2%}")
                print(
                    f"  Claims supported: {posterior.supporting_count}  Contradicted: {posterior.contradicting_count}"
                )
                print(f"  {posterior.explanation}")
        except Exception as e:
            print(f"\n  Confidence calculation failed: {e}")

    print()


def _stage_rank(stage: str) -> int:
    """Numeric rank for stage comparison."""
    return {
        "hypothesis": 0,
        "supported": 1,
        "provisional": 2,
        "robust": 3,
        "actionable": 4,
    }.get(stage, -1)


# ── Verification checks ───────────────────────────────────────────────


def verify_doyle(db_path: str) -> list[str]:
    """Verify Doyle TMS fired: revalidate_claim operation executed, claim demoted."""
    failures = []
    steps = _get_execution_steps(db_path)
    reval_steps = [s for s in steps if s.get("operation") == "revalidate_claim"]

    if not reval_steps:
        failures.append("revalidate_claim never fired — TMS was not triggered")
        return failures

    # Check that at least one revalidation actually demoted
    demoted = [s for s in reval_steps if "demoted" in s.get("message", "").lower()]
    if not demoted:
        failures.append("revalidate_claim fired but no claim was demoted")

    # AGM check: verify evidence links preserved after demotion
    claims = _get_entities(db_path, "claim")
    for claim in claims:
        if claim.get("modification_count", 0) > 0:
            if not claim.get("evidence_ids"):
                failures.append(
                    f"Claim {claim.get('statement', '')[:50]} was demoted but lost all evidence links (AGM violation)"
                )

    return failures


def verify_peirce(db_path: str) -> list[str]:
    """Verify Peirce inquiry cycling: investigation fired after scrutiny doubt."""
    failures = []
    steps = _get_execution_steps(db_path)

    investigate_steps = [s for s in steps if s.get("operation") == "investigate_claim"]
    if not investigate_steps:
        failures.append("investigate_claim never fired — no inquiry cycling occurred")
        return failures

    # Check that scrutiny re-ran after investigation
    scrutiny_steps = [s for s in steps if s.get("operation") == "scrutinise_claim"]
    if len(scrutiny_steps) < 2:
        failures.append(
            f"Only {len(scrutiny_steps)} scrutiny runs — expected re-scrutiny after investigation"
        )

    return failures


def verify_lipton(db_path: str) -> list[str]:
    """Verify Lipton contrastive evaluation fired."""
    failures = []
    steps = _get_execution_steps(db_path)

    contrastive_steps = [
        s for s in steps if s.get("operation") == "contrastive_evaluation"
    ]
    if not contrastive_steps:
        failures.append("contrastive_evaluation never fired")

    return failures


def verify_kahneman(db_path: str) -> list[str]:
    """Verify Kahneman independence: per-evidence scrutiny produced results."""
    failures = []
    steps = _get_execution_steps(db_path)

    scrutiny_steps = [s for s in steps if s.get("operation") == "scrutinise_claim"]
    if not scrutiny_steps:
        failures.append("scrutinise_claim never fired")

    # Check that uncertainties were created (evidence of issue identification)
    uncertainties = _get_entities(db_path, "uncertainty")
    if not uncertainties:
        failures.append(
            "No uncertainties created — scrutiny found no issues in any evidence"
        )

    return failures


def verify_tetlock(db_path: str) -> list[str]:
    """Verify Tetlock predictions: generate_prediction fired with falsification criteria."""
    failures = []
    steps = _get_execution_steps(db_path)

    pred_steps = [s for s in steps if s.get("operation") == "generate_prediction"]
    if not pred_steps:
        # Provide diagnostic context for why it didn't fire
        claims = _get_entities(db_path, "claim")
        stages = [c.get("stage", "?") for c in claims]
        failures.append(
            f"generate_prediction never fired — no claim reached ROBUST stage. "
            f"Claim stages: {stages}. "
            f"This is the hardest pathway to trigger; requires 3+ evidence, "
            f"all verification tracks complete, and independent domains."
        )
        return failures

    # Check that predictions were actually stored
    claims = _get_entities(db_path, "claim")
    claims_with_preds = [c for c in claims if c.get("predictions")]
    if not claims_with_preds:
        failures.append("generate_prediction fired but no predictions stored on claims")
        return failures

    # Check falsification criteria
    for claim in claims_with_preds:
        for pred in claim.get("predictions", []):
            if not pred.get("failure_criteria"):
                failures.append(
                    f"Prediction '{pred.get('statement', '')[:50]}' has no falsification criteria"
                )

    return failures


VERIFIERS = {
    "doyle": verify_doyle,
    "peirce": verify_peirce,
    "lipton": verify_lipton,
    "kahneman": verify_kahneman,
    "tetlock": verify_tetlock,
}


# ── Runner ─────────────────────────────────────────────────────────────


async def run_single_test(
    test: PathwayTest,
    model: str,
    keep: bool = False,
    db_dir: str | None = None,
) -> tuple[bool, list[str], str]:
    """Run a single integration test. Returns (passed, failures, db_path)."""
    from andamentum.epistemic.cli_handlers import handle_ask

    project_name = f"integration_{test.tradition}"
    actual_db_dir = db_dir or tempfile.mkdtemp(prefix="epistemic_integration_")

    try:
        await handle_ask(
            question=test.question,
            name=project_name,
            model=model,
            keep=True,  # Always keep during test — we need to inspect
            verbose=False,
            trace="none",
            db_dir=actual_db_dir,
        )
    except Exception as e:
        return False, [f"Pipeline crashed: {e}"], ""

    db_path = str(Path(actual_db_dir) / f"{project_name}.db")

    # Print diagnostics regardless of pass/fail
    await print_diagnostics(db_path, test.tradition)

    # Generate HTML report
    try:
        from andamentum.document_store import DocumentStore
        from andamentum.epistemic.report_generator import ReportGenerator

        store = DocumentStore.for_database(project_name, db_dir=Path(actual_db_dir))
        await store.initialize()
        gen = ReportGenerator(store, project_name)
        report_path = Path(actual_db_dir) / f"{project_name}_report.html"
        if await gen.save_html(output_path=report_path, model_name=model):
            print(f"\n  Report: {report_path}")
    except Exception as e:
        print(f"\n  [warning] Report generation failed: {e}")

    # Run verification
    verifier = VERIFIERS.get(test.tradition)
    if not verifier:
        return (
            True,
            [f"No verifier for tradition '{test.tradition}' — skipped"],
            db_path,
        )

    failures = verifier(db_path)
    passed = len(failures) == 0

    # Cleanup unless --keep
    if not keep and passed:
        try:
            Path(db_path).unlink(missing_ok=True)
        except Exception:
            pass

    return passed, failures, db_path


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run epistemic integration tests (full pipeline, real LLM calls)"
    )
    parser.add_argument("--model", help="LLM model (e.g., openai:gpt-5.4-mini)")
    parser.add_argument(
        "--tradition", help="Run only this tradition (e.g., doyle, peirce)"
    )
    parser.add_argument(
        "--list", action="store_true", help="List available tests and exit"
    )
    parser.add_argument(
        "--keep", action="store_true", help="Keep databases after tests"
    )
    parser.add_argument("--db-dir", help="Directory for test databases (default: temp)")
    args = parser.parse_args()

    if args.list:
        print("\nAvailable integration tests:\n")
        for test in PATHWAY_TESTS:
            print(f"  {test.tradition:<12}  {test.question[:70]}")
        print(f"\nTotal: {len(PATHWAY_TESTS)} tests")
        print()
        return

    if not args.model:
        parser.error("--model is required when running tests")

    tests = PATHWAY_TESTS
    if args.tradition:
        tests = [t for t in PATHWAY_TESTS if t.tradition == args.tradition]
        if not tests:
            print(f"Unknown tradition: {args.tradition}")
            print(f"Available: {[t.tradition for t in PATHWAY_TESTS]}")
            sys.exit(1)

    print(f"\nModel: {args.model}")
    print(f"Tests: {len(tests)}")
    print()

    results: list[tuple[str, bool, list[str], str]] = []

    for test in tests:
        print(f"{'=' * 60}")
        print(f"  {test.tradition.upper()}: {test.question[:50]}...")
        print(f"  Rationale: {test.rationale[:70]}...")
        print(f"{'=' * 60}")

        passed, failures, db_path = await run_single_test(
            test, args.model, keep=args.keep, db_dir=args.db_dir
        )
        results.append((test.tradition, passed, failures, db_path))

        if passed:
            print("  RESULT: PASSED")
        else:
            print("  RESULT: FAILED")
            for f in failures:
                print(f"    - {f}")

        if db_path and args.keep:
            print(f"  Database: {db_path}")
        print()

    # Summary
    print(f"\n{'=' * 60}")
    print("  SUMMARY")
    print(f"{'=' * 60}")

    total = len(results)
    passed_count = sum(1 for _, p, _, _ in results if p)

    for tradition, passed, failures, db_path in results:
        status = "PASSED" if passed else "FAILED"
        print(f"  {tradition:<12}  {status}")
        if not passed:
            for f in failures:
                print(f"    - {f[:80]}")

    print(f"\n  {passed_count}/{total} traditions verified")
    print()

    if passed_count < total:
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
