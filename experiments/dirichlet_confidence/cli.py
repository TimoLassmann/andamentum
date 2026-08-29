"""CLI for the mixture-of-Dirichlets SciFact confidence benchmark.

Example:

    uv run python -m experiments.dirichlet_confidence.cli \
        --scifact-path /path/to/scifact \
        --n-claims 60 \
        --models ollama:gemma3:27b openai:gpt-5.1 openai:gpt-5.1-codex openai:gpt-5.1-codex-mini \
        --out experiments/dirichlet_confidence/results/run1.json

Elicitations are cached under ``--cache-dir`` keyed by (claim, model, phrasing),
so reruns re-pay for nothing already gathered.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from pathlib import Path

from dotenv import load_dotenv

from .benchmark import run_benchmark
from .scifact import balanced_subset, label_counts, load_scifact

# Default ensemble: one local model + the three gpt-5.1 variants you named.
# All four were smoke-tested live and return the verbalized schema. The OpenAI
# models route through the Responses API (the codex family is responses-only);
# the local slot uses gemma4:31b-nvfp4 (gemma3:27b is not pulled locally).
# Swap any of these via --models.
DEFAULT_MODELS = [
    "ollama:gemma4:31b-nvfp4",
    "openai:gpt-5.1",
    "openai:gpt-5.1-codex",
    "openai:gpt-5.1-codex-mini",
]


def _fmt_row(name: str, m: dict) -> str:
    if not m or m.get("n", 0) == 0:
        return f"  {name:<28} (no data)"
    extra = ""
    if "sampled_nll" in m:
        extra = f"  sampledNLL={m['sampled_nll']:.3f}"
    return (
        f"  {name:<28} acc={m['accuracy']:.3f}  ECE={m['ece']:.3f}  "
        f"Brier={m['brier']:.3f}  NLL={m['nll']:.3f}{extra}  (n={m['n']})"
    )


def render_report(results: dict) -> str:
    ds = results["dataset"]
    lines = [
        "=" * 78,
        "Mixture-of-Dirichlets confidence — SciFact benchmark",
        "=" * 78,
        f"examples: {ds['n_examples']}  mixtures built: {ds['n_mixtures_built']}  "
        f"labels: {ds['labels']}",
        f"models: {', '.join(ds['models'])}",
        f"fitted s: {results['s_fit']:.1f}  (on {results['n_dev']} dev mixtures)",
        "",
        "Per-model (ask each model directly):",
    ]
    for model, blk in results["per_model"].items():
        lines.append(f"  [{model}]")
        lines.append(_fmt_row("    single prompt", blk["single_prompt"]))
        lines.append(_fmt_row("    multi-phrasing mean", blk["multi_phrasing_mean"]))
    lines += [
        "",
        "Configurations:",
        _fmt_row("A single prompt", results["A_single"]),
        _fmt_row("B multi-phrasing mean", results["B_multi_phrasing_mean"]),
        _fmt_row("C full point mixture", results["C_full_point_mixture"]),
        _fmt_row("D Dirichlet mixture", results["D_dirichlet_mixture"]),
        "",
        "Variance decomposition (test mean, summed over classes):",
    ]
    vd = results["mean_variance_decomposition"]
    lines.append(
        f"  within_component={vd['within_component']:.4f}  "
        f"within_model(phrasing)={vd['within_model']:.4f}  "
        f"between_model(family)={vd['between_model']:.4f}"
    )
    lines += ["", "Top disagreement claims (read these — Lever 7 sanity check):"]
    for r in results["top_disagreement"][:5]:
        lines.append(
            f"  [{r['label']}] within={r['within_model']:.3f} "
            f"between={r['between_model']:.3f} mean={r['pooled_mean']}"
        )
        lines.append(f"      {r['claim'][:100]}")
    lines.append("=" * 78)
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scifact-path", required=True, help="SciFact dir or flat .jsonl"
    )
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--n-claims", type=int, default=60)
    parser.add_argument("--cache-dir", default="experiments/dirichlet_confidence/cache")
    parser.add_argument("--out", default=None, help="write results JSON here")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--alpha0", type=float, default=0.5)
    parser.add_argument("--s-initial", type=float, default=50.0)
    parser.add_argument("--dev-frac", type=float, default=0.3)
    parser.add_argument("--mc-samples", type=int, default=2000)
    parser.add_argument("--concurrency", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--no-noinfo", action="store_true", help="exclude NOINFO examples"
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    load_dotenv()

    examples = load_scifact(args.scifact_path, include_noinfo=not args.no_noinfo)
    print(f"loaded {len(examples)} examples; labels={label_counts(examples)}")
    examples = balanced_subset(examples, args.n_claims, seed=args.seed)
    print(f"using {len(examples)} balanced examples; labels={label_counts(examples)}")

    cache_path = Path(args.cache_dir) / "elicitations.jsonl"
    results = asyncio.run(
        run_benchmark(
            examples,
            args.models,
            cache_path,
            temperature=args.temperature,
            s_initial=args.s_initial,
            alpha0=args.alpha0,
            dev_frac=args.dev_frac,
            n_mc_samples=args.mc_samples,
            concurrency=args.concurrency,
            seed=args.seed,
        )
    )

    print("\n" + render_report(results))

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(results, indent=2))
        print(f"\nresults written to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
