"""GATHER — the only step that calls models. Run rarely; it caches everything.

This is the expensive half of the springboard. It collects two things into a
single append-only JSONL cache and then stops:

  1. BASELINE rows — each baseline model (default: gpt-5.4, gpt-5.4-mini,
     gpt-5.4-nano) asked ONCE per claim, temperature 0, using phrasing 0.
     These are the fixed reference points. They do not change while we develop
     the method, so we never want to re-pay for them.

  2. METHOD rows — the method model (default: gpt-5.4-nano) asked under K
     DIFFERENT PHRASINGS of the same question, all at temperature 0. The
     phrasings are the diversity source (Lever 1 of the walkthrough): same
     question, different wording, deterministic decoding. The spread between
     phrasings is *phrasing brittleness*, which the analysis step turns into an
     uncertainty band.

Idempotent: it loads the cache, computes exactly what is missing, fetches only
that, and appends. Re-running when nothing is missing makes ZERO calls. So the
90 baseline rows already gathered are reused untouched.

The maths (zero-removal, Dirichlet, Monte Carlo, metrics) lives in the separate
``analyze.py`` — that step makes no API calls and is where you iterate.

RUN
===
    uv run python -m experiments.dirichlet_confidence.gather \
        --scifact-path experiments/dirichlet_confidence/data/scifact \
        --n-claims 30
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiments.dirichlet_confidence.cache_io import (  # noqa: E402
    CACHE_PATH,
    ElicitationCache,
)
from experiments.dirichlet_confidence.elicit import elicit_row  # noqa: E402
from experiments.dirichlet_confidence.phrasings import PHRASINGS  # noqa: E402
from experiments.dirichlet_confidence.scifact import (  # noqa: E402
    balanced_subset,
    label_counts,
    load_scifact,
)

console = Console()

BASELINE_MODELS = ["openai:gpt-5.4", "openai:gpt-5.4-mini", "openai:gpt-5.4-nano"]
METHOD_MODEL = "openai:gpt-5.4-nano"
# How many phrasings to use for the method's ensemble. K<=len(PHRASINGS).
DEFAULT_N_PHRASINGS = 5


def _short(model: str) -> str:
    return model.split(":", 1)[-1]


async def run(args: argparse.Namespace) -> None:
    load_dotenv()

    examples = load_scifact(args.scifact_path)
    console.print(f"Loaded {len(examples)} SciFact examples ({label_counts(examples)})")
    examples = balanced_subset(examples, args.n_claims, seed=args.seed)
    console.print(
        f"Using {len(examples)} balanced examples ({label_counts(examples)})\n"
    )

    cache = ElicitationCache(Path(args.cache))
    n_phrasings = min(args.n_phrasings, len(PHRASINGS))

    # Build the full list of (kind, claim, model, phrasing_id, temperature) the
    # experiment needs, then keep only what the cache is missing.
    #   - baseline: phrasing 0, temp 0, one per baseline model
    #   - method:   phrasings 0..K-1, temp 0, method model
    wanted: list[tuple[str, object, str, int, float]] = []
    for ex in examples:
        for m in args.baseline_models:
            wanted.append(("baseline", ex, m, 0, 0.0))
        for k in range(n_phrasings):
            wanted.append(("method", ex, args.method_model, k, 0.0))

    missing = [w for w in wanted if not cache.has(w[0], w[1].claim_id, w[2], w[3])]

    console.print(
        f"Cache: {cache.size()} rows on disk · "
        f"{len(wanted)} needed for this run · "
        f"[bold]{len(missing)} to fetch[/bold] "
        f"({len(wanted) - len(missing)} already cached, reused free)\n"
    )
    if not missing:
        console.print("[green]Nothing to fetch — cache is complete for this run.[/green]")
        return

    sem = asyncio.Semaphore(args.concurrency)
    done = 0
    lock = asyncio.Lock()

    async def fetch(kind, ex, model, phrasing_id, temperature) -> None:
        nonlocal done
        async with sem:
            row = await elicit_row(ex.claim, ex.evidence, model, phrasing_id, temperature)
        async with lock:
            cache.put(kind, ex.claim_id, model, phrasing_id, temperature, row)
            done += 1
            ok = "ok" if row is not None else "[red]FAILED[/red]"
            console.print(
                f"  [{done}/{len(missing)}] {kind:8s} {_short(model):16s} "
                f"phrasing {phrasing_id}  claim {ex.claim_id:>12}  {ok}"
            )

    with console.status("Fetching missing elicitations…"):
        await asyncio.gather(*(fetch(*w) for w in missing))

    console.print(
        f"\n[green]Done.[/green] Cache now has {cache.size()} rows at {args.cache}.\n"
        f"Next: run [bold]analyze[/bold] (no API calls) to score the arms."
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scifact-path", required=True)
    p.add_argument("--n-claims", type=int, default=30)
    p.add_argument("--baseline-models", nargs="+", default=BASELINE_MODELS)
    p.add_argument("--method-model", default=METHOD_MODEL)
    p.add_argument("--n-phrasings", type=int, default=DEFAULT_N_PHRASINGS)
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--cache", default=str(CACHE_PATH))
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(argv)
    asyncio.run(run(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
