"""What do ``expected_score`` / ``expected_preference`` buy — through the REAL module?

Unlike ``run.py`` (which reads raw elicited distributions), this drives the
actual public entry points ``andamentum.llm_judge.judge_score`` /
``judge_compare`` and compares the discrete verdict the module reports
against the continuous field it now also exposes.

- SCORE (SciFact): does ``expected_score`` rank SUPPORT above CONTRADICT
  better than the argmax ``overall`` does? AUROC(expected_score; SUPPORT vs
  CONTRADICT), plus 3-way argmax accuracy for reference.
- COMPARE (JudgeBench): does ``expected_preference`` (>0.5 -> a) beat the
  argmax ``winner`` (ties counted wrong, gold has none)? Accuracy + tie-rate.

Every call is cached to ``results/field_cache.jsonl`` keyed by
(task, model, id) so the run resumes for free and the metrics recompute
offline. Usage::

    uv run python benchmarks/judge_scoring/field_benchmark.py \
        --model openai:gpt-5.4-nano --model openai:gpt-5.4-mini \
        --score-n 60 --compare-n 50
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from dotenv import load_dotenv

from andamentum.llm_judge import Criterion, judge_compare, judge_score

HERE = Path(__file__).parent
DATA = HERE / "data_powered.json"
CACHE = HERE / "results" / "field_cache.jsonl"

# One factual-support axis. meets = supported, fails = contradicted, partial =
# not enough info — matching the data's expected_verdict mapping.
SUPPORT_CRITERION = Criterion(
    name="support",
    description=(
        "The EVIDENCE supports the CLAIM: 'meets' = the evidence supports it, "
        "'fails' = the evidence contradicts it, 'partial' = the evidence gives "
        "no clear information either way."
    ),
)


# ── tiny metric kit (no sklearn) ─────────────────────────────────────────


def auroc(pos: list[float], neg: list[float]) -> float | None:
    """Mann-Whitney AUROC: P(a random pos scores above a random neg), ties=0.5."""
    if not pos or not neg:
        return None
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def balanced(items: list[dict], key: str, per_class: int) -> list[dict]:
    """First ``per_class`` items of each distinct value of ``key`` — stable,
    deterministic, no RNG (the data file is already shuffled)."""
    buckets: dict[str, list[dict]] = {}
    for it in items:
        buckets.setdefault(it[key], []).append(it)
    out: list[dict] = []
    for _, group in sorted(buckets.items()):
        out.extend(group[:per_class])
    return out


# ── cache ────────────────────────────────────────────────────────────────


def load_cache() -> dict[tuple[str, str, str], dict]:
    cache: dict[tuple[str, str, str], dict] = {}
    if CACHE.exists():
        for line in CACHE.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            cache[(rec["task"], rec["model"], rec["id"])] = rec["result"]
    return cache


def append_cache(task: str, model: str, item_id: str, result: dict) -> None:
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    with CACHE.open("a", encoding="utf-8") as fh:
        fh.write(
            json.dumps(
                {"task": task, "model": model, "id": item_id, "result": result}
            )
            + "\n"
        )


# ── runners ──────────────────────────────────────────────────────────────


async def run_score(model: str, items: list[dict], cache: dict) -> list[dict]:
    rows: list[dict] = []
    for i, ex in enumerate(items):
        key = ("score", model, ex["id"])
        if key in cache:
            res = cache[key]
        else:
            r = await judge_score(
                f"CLAIM: {ex['claim']}\n\nEVIDENCE: {ex['evidence']}",
                criteria=[SUPPORT_CRITERION],
                model=model,
            )
            res = {"overall": r.overall, "expected_score": r.expected_score}
            append_cache("score", model, ex["id"], res)
            print(
                f"  [score {model.split(':')[-1]}] {i + 1}/{len(items)} "
                f"gold={ex['gold_label']:<10} overall={res['overall']:<8} "
                f"E={res['expected_score']:.3f}"
            )
        rows.append({**ex, **res})
    return rows


async def run_compare(model: str, items: list[dict], cache: dict) -> list[dict]:
    rows: list[dict] = []
    for i, ex in enumerate(items):
        key = ("compare", model, ex["id"])
        if key in cache:
            res = cache[key]
        else:
            r = await judge_compare(
                ex["response_a"],
                ex["response_b"],
                context=ex["question"],
                model=model,
            )
            res = {"winner": r.winner, "expected_preference": r.expected_preference}
            append_cache("compare", model, ex["id"], res)
            print(
                f"  [compare {model.split(':')[-1]}] {i + 1}/{len(items)} "
                f"gold={ex['gold']} winner={res['winner']:<4} "
                f"E={res['expected_preference']:.3f}"
            )
        rows.append({**ex, **res})
    return rows


# ── reporting ────────────────────────────────────────────────────────────


def report_score(model: str, rows: list[dict]) -> dict:
    n = len(rows)
    argmax_acc = sum(r["overall"] == r["expected_verdict"] for r in rows) / n
    sup = [r["expected_score"] for r in rows if r["gold_label"] == "SUPPORT"]
    con = [r["expected_score"] for r in rows if r["gold_label"] == "CONTRADICT"]
    a = auroc(sup, con)
    # Risk-coverage on the binary SUPPORT/CONTRADICT slice: keep the most
    # confident |expected_score - 0.5| and measure directional accuracy.
    binary = [r for r in rows if r["gold_label"] in ("SUPPORT", "CONTRADICT")]
    binary.sort(key=lambda r: abs(r["expected_score"] - 0.5), reverse=True)
    cov = {}
    for frac in (1.0, 0.8, 0.6, 0.4):
        k = max(1, int(len(binary) * frac))
        kept = binary[:k]
        correct = sum(
            (r["expected_score"] > 0.5) == (r["gold_label"] == "SUPPORT") for r in kept
        )
        cov[frac] = correct / k
    return {
        "model": model,
        "n": n,
        "argmax_accuracy_3way": argmax_acc,
        "auroc_expected_score": a,
        "risk_coverage": cov,
        "n_support": len(sup),
        "n_contradict": len(con),
    }


def report_compare(model: str, rows: list[dict]) -> dict:
    n = len(rows)
    argmax_correct = sum(r["winner"] == r["gold"] for r in rows)
    tie_rate = sum(r["winner"] == "tie" for r in rows) / n
    exp_correct = sum(
        ("a" if r["expected_preference"] > 0.5 else "b") == r["gold"] for r in rows
    )
    return {
        "model": model,
        "n": n,
        "argmax_accuracy": argmax_correct / n,
        "argmax_tie_rate": tie_rate,
        "expected_pref_accuracy": exp_correct / n,
    }


async def main() -> None:
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="append", required=True)
    ap.add_argument("--score-n", type=int, default=60, help="per class (x3)")
    ap.add_argument("--compare-n", type=int, default=50, help="per class (x2)")
    args = ap.parse_args()

    data = json.loads(DATA.read_text(encoding="utf-8"))
    score_items = balanced(data["score"], "gold_label", args.score_n // 3)
    compare_items = balanced(data["compare"], "gold", args.compare_n // 2)
    cache = load_cache()

    print(
        f"score n={len(score_items)} (balanced), "
        f"compare n={len(compare_items)} (balanced), models={args.model}\n"
    )

    summary: dict[str, list] = {"score": [], "compare": []}
    for model in args.model:
        print(f"=== {model} ===")
        srows = await run_score(model, score_items, cache)
        crows = await run_compare(model, compare_items, cache)
        summary["score"].append(report_score(model, srows))
        summary["compare"].append(report_compare(model, crows))

    out = HERE / "results" / "field_benchmark.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n" + "=" * 72)
    print("SCORE — expected_score vs argmax overall")
    print(
        f"{'model':<22}{'argmax acc(3way)':>18}{'AUROC(E; S vs C)':>18}"
        f"{'RC@60%':>9}{'RC@40%':>9}"
    )
    for s in summary["score"]:
        rc = s["risk_coverage"]
        a = f"{s['auroc_expected_score']:.3f}" if s["auroc_expected_score"] else "n/a"
        print(
            f"{s['model']:<22}{s['argmax_accuracy_3way']:>17.0%}{a:>18}"
            f"{rc[0.6]:>8.0%}{rc[0.4]:>9.0%}"
        )
    print("\nCOMPARE — expected_preference vs argmax winner")
    print(
        f"{'model':<22}{'argmax acc':>12}{'tie-rate':>10}"
        f"{'E-pref acc':>12}{'delta':>8}"
    )
    for c in summary["compare"]:
        delta = c["expected_pref_accuracy"] - c["argmax_accuracy"]
        print(
            f"{c['model']:<22}{c['argmax_accuracy']:>11.0%}{c['argmax_tie_rate']:>10.0%}"
            f"{c['expected_pref_accuracy']:>11.0%}{delta:>+8.0%}"
        )
    print("=" * 72)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    asyncio.run(main())
