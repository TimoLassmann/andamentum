"""Run all agent evals, print reports, and cache results for tracking.

Usage:
    cd packages/epistemic
    uv run python agent_evals/run_all.py
    uv run python agent_evals/run_all.py --model bedrock:claude-haiku-4-5
    uv run python agent_evals/run_all.py --agent scrutinise_claim
    uv run python agent_evals/run_all.py --history          # show past runs
    uv run python agent_evals/run_all.py --compare 2        # compare last 2 runs
"""

import argparse
import asyncio
import importlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# Ensure the evals package is importable
EVALS_DIR = Path(__file__).parent
sys.path.insert(0, str(EVALS_DIR.parent))
sys.path.insert(0, str(EVALS_DIR))

RESULTS_DIR = EVALS_DIR / "results"

EVAL_MODULES = [
    "scrutinise_claim.eval",
    "evidence_quality.eval",
    "extract_evidence.eval",
    "propose_claims.eval",
    "adversarial_search.eval",
    "write_answer.eval",
]


# ── Result serialization ────────────────────────────────────────────────


def _serialize_report(report, agent_name: str, model: str) -> dict:
    """Extract a JSON-serializable summary from an EvaluationReport."""
    cases = []
    for case in report.cases:
        case_data = {
            "name": case.name,
            "task_duration": round(case.task_duration, 2),
            "total_duration": round(case.total_duration, 2),
            "assertions": {
                k: {"value": v.value, "reason": v.reason}
                for k, v in case.assertions.items()
            },
            "scores": {
                k: {"value": v.value, "reason": v.reason}
                for k, v in case.scores.items()
            },
            "labels": {
                k: {"value": v.value, "reason": v.reason}
                for k, v in case.labels.items()
            },
        }
        cases.append(case_data)

    # Compute aggregate stats
    all_assertions = []
    for c in cases:
        all_assertions.extend(v["value"] for v in c["assertions"].values())
    total = len(all_assertions)
    passed = sum(1 for v in all_assertions if v is True)

    averages = report.averages()
    avg_assertions = None
    avg_duration = None
    if averages:
        avg_assertions = averages.assertions
        avg_duration = round(averages.task_duration or 0, 2)

    return {
        "agent": agent_name,
        "model": model,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "assertions_passed": passed,
        "assertions_total": total,
        "pass_rate": round(passed / total * 100, 1) if total else 0,
        "avg_assertions": avg_assertions,
        "avg_task_duration": avg_duration,
        "num_cases": len(cases),
        "num_failures": len(report.failures),
        "cases": cases,
    }


def _save_result(result: dict, run_id: str) -> Path:
    """Save a single agent's eval result to the results directory."""
    run_dir = RESULTS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / f"{result['agent']}.json"
    path.write_text(json.dumps(result, indent=2, default=str))
    return path


def _save_run_manifest(run_id: str, model: str, agents: list[dict]) -> Path:
    """Save a manifest summarizing the entire run."""
    run_dir = RESULTS_DIR / run_id
    manifest = {
        "run_id": run_id,
        "model": model,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "agents": [
            {
                "name": a["agent"],
                "pass_rate": a["pass_rate"],
                "assertions": f"{a['assertions_passed']}/{a['assertions_total']}",
                "cases": a["num_cases"],
                "failures": a["num_failures"],
                "avg_duration": a.get("avg_task_duration"),
            }
            for a in agents
        ],
        "overall_pass_rate": round(
            sum(a["assertions_passed"] for a in agents)
            / max(sum(a["assertions_total"] for a in agents), 1)
            * 100,
            1,
        ),
    }
    path = run_dir / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2))
    return path


# ── History & comparison ─────────────────────────────────────────────────


def _list_runs() -> list[dict]:
    """Load all run manifests, sorted newest first."""
    if not RESULTS_DIR.exists():
        return []
    runs = []
    for manifest_path in sorted(RESULTS_DIR.glob("*/manifest.json"), reverse=True):
        runs.append(json.loads(manifest_path.read_text()))
    return runs


def _print_history(limit: int = 20) -> None:
    """Print a table of past eval runs."""
    runs = _list_runs()[:limit]
    if not runs:
        print("No eval history found.")
        return

    print(f"\n{'Run ID':<22} {'Model':<40} {'Pass Rate':>10} {'Agents':>8}")
    print("-" * 82)
    for run in runs:
        print(
            f"{run['run_id']:<22} "
            f"{run['model'][:39]:<40} "
            f"{run['overall_pass_rate']:>9}% "
            f"{len(run['agents']):>7}"
        )
        for a in run["agents"]:
            print(
                f"  {'':20} {a['name']:<30} {a['pass_rate']:>6}%  ({a['assertions']})"
            )
    print()


def _print_comparison(n: int = 2) -> None:
    """Compare the last N runs side by side."""
    runs = _list_runs()[:n]
    if len(runs) < 2:
        print(f"Need at least 2 runs to compare, found {len(runs)}.")
        return

    # Collect all agent names across runs
    all_agents = []
    for run in runs:
        for a in run["agents"]:
            if a["name"] not in all_agents:
                all_agents.append(a["name"])

    header = f"{'Agent':<25}"
    for run in runs:
        ts = run["run_id"][:10]
        model_short = run["model"].split(":")[-1][:20]
        header += f" {ts} ({model_short}):>25"
    print(f"\n{header}")
    print("-" * (25 + 25 * len(runs)))

    for agent_name in all_agents:
        row = f"{agent_name:<25}"
        values = []
        for run in runs:
            match = next((a for a in run["agents"] if a["name"] == agent_name), None)
            if match:
                row += f" {match['pass_rate']:>6}% ({match['assertions']})"
                values.append(match["pass_rate"])
            else:
                row += f" {'—':>22}"
                values.append(None)

        # Add delta if we have two values
        if len(values) >= 2 and values[0] is not None and values[1] is not None:
            delta = values[0] - values[1]
            arrow = "↑" if delta > 0 else "↓" if delta < 0 else "="
            row += f"  {arrow} {abs(delta):.1f}%"
        print(row)

    # Overall
    row = f"{'OVERALL':<25}"
    for run in runs:
        row += f" {run['overall_pass_rate']:>6}%"
    print("-" * (25 + 25 * len(runs)))
    print(row)
    print()


# ── Main ─────────────────────────────────────────────────────────────────


async def main() -> None:
    parser = argparse.ArgumentParser(description="Run epistemic agent evals")
    parser.add_argument(
        "--model", default=None, help="Override model (e.g., bedrock:claude-haiku-4-5)"
    )
    parser.add_argument(
        "--agent",
        default=None,
        help="Run only this agent's eval (e.g., scrutinise_claim)",
    )
    parser.add_argument(
        "--max-concurrency", type=int, default=3, help="Max concurrent cases per eval"
    )
    parser.add_argument("--no-cache", action="store_true", help="Don't save results")
    parser.add_argument(
        "--history", action="store_true", help="Show past runs and exit"
    )
    parser.add_argument(
        "--compare",
        type=int,
        nargs="?",
        const=2,
        help="Compare last N runs (default: 2)",
    )
    args = parser.parse_args()

    if args.history:
        _print_history()
        return

    if args.compare:
        _print_comparison(args.compare)
        return

    if args.model:
        os.environ["EPISTEMIC_EVAL_MODEL"] = args.model

    modules = EVAL_MODULES
    if args.agent:
        modules = [m for m in modules if args.agent in m]
        if not modules:
            print(f"No eval found for agent: {args.agent}")
            print(f"Available: {[m.split('.')[0] for m in EVAL_MODULES]}")
            sys.exit(1)

    from conftest import get_eval_model

    model = get_eval_model()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    print(f"\nModel: {model}")
    print(f"Run:   {run_id}")
    print(f"Evals: {len(modules)}\n")

    agent_results = []

    for mod_path in modules:
        agent_name = mod_path.split(".")[0]
        print(f"{'=' * 60}")
        print(f"  {agent_name}")
        print(f"{'=' * 60}")
        try:
            mod = importlib.import_module(mod_path)
            report = await mod.run_eval(max_concurrency=args.max_concurrency)
            report.print(include_input=False, include_output=True, include_reasons=True)

            result = _serialize_report(report, agent_name, model)
            agent_results.append(result)

            if not args.no_cache:
                path = _save_result(result, run_id)
                print(f"  Saved: {path.relative_to(EVALS_DIR)}")
        except Exception as e:
            print(f"  FAILED: {e}")
            agent_results.append(
                {
                    "agent": agent_name,
                    "model": model,
                    "pass_rate": 0,
                    "assertions_passed": 0,
                    "assertions_total": 0,
                    "num_cases": 0,
                    "num_failures": 1,
                    "error": str(e),
                }
            )
        print()

    # Save run manifest
    if not args.no_cache and agent_results:
        manifest_path = _save_run_manifest(run_id, model, agent_results)
        print(f"Run manifest: {manifest_path.relative_to(EVALS_DIR)}")

    # Print summary
    print(f"\n{'=' * 60}")
    print("  SUMMARY")
    print(f"{'=' * 60}")
    total_passed = sum(a.get("assertions_passed", 0) for a in agent_results)
    total_assertions = sum(a.get("assertions_total", 0) for a in agent_results)
    for a in agent_results:
        status = f"{a.get('pass_rate', 0)}%" if "error" not in a else "ERROR"
        print(
            f"  {a['agent']:<25} {status:>8}  ({a.get('assertions_passed', 0)}/{a.get('assertions_total', 0)})"
        )
    print(f"  {'─' * 45}")
    overall = round(total_passed / max(total_assertions, 1) * 100, 1)
    print(f"  {'OVERALL':<25} {overall:>7}%  ({total_passed}/{total_assertions})")
    print()

    if not args.no_cache:
        print("View history:  uv run python agent_evals/run_all.py --history")
        print("Compare runs:  uv run python agent_evals/run_all.py --compare")
        print()


if __name__ == "__main__":
    asyncio.run(main())
