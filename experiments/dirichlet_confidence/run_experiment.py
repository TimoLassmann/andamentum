"""Confidence experiment: cloud-model baselines vs a local high-temperature ensemble.

WHAT THIS RUNS
==============

Two ways of getting a confidence estimate, scored on the SAME SciFact claims:

1. BASELINE — three cloud models, each asked ONCE.
   openai:gpt-5.4 / gpt-5.4-mini / gpt-5.4-nano each distribute 100 points across
   SUPPORT / CONTRADICT / NOINFO (temperature 0). The points on the chosen class
   are that model's confidence. Each model is scored on its own.

2. METHOD (the thing we are testing) — one local model, asked MANY times.
   ollama:gemma4:31b-nvfp4 is asked the same claim N times at a higher
   temperature. Because the answers vary, we get not just a confidence but an
   uncertainty band on it (how much the samples disagreed). The N answers are
   pooled with the mixture-of-Dirichlets maths in mixture.py.

Only the baseline calls cost money (N_claims x 3). The ensemble is local/free.
Every call is cached to JSONL, so reruns re-pay for nothing.

The display shows ONE readable block per claim as it finishes, then summary
tables at the end.

RUN
===
    uv run python -m experiments.dirichlet_confidence.run_experiment \
        --scifact-path experiments/dirichlet_confidence/data/scifact \
        --n-claims 30 \
        --ensemble-samples 10 --ensemble-temperature 0.8
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.exceptions import ModelHTTPError, UnexpectedModelBehavior
from pydantic_ai.output import PromptedOutput
from pydantic_ai.settings import ModelSettings
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# Make the package importable whether invoked as a module or a file path.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from andamentum.core import resolve_model  # noqa: E402

from experiments.dirichlet_confidence import metrics  # noqa: E402
from experiments.dirichlet_confidence.mixture import (  # noqa: E402
    MixtureOfDirichlets,
    build_mixture,
    fit_concentration_s,
    repair_row,
)
from experiments.dirichlet_confidence.phrasings import (  # noqa: E402
    PHRASINGS,
    SYSTEM_INSTRUCTIONS,
    render,
)
from experiments.dirichlet_confidence.scifact import (  # noqa: E402
    Example,
    balanced_subset,
    label_counts,
    load_scifact,
)

console = Console()

BASELINE_MODELS = ["openai:gpt-5.4", "openai:gpt-5.4-mini", "openai:gpt-5.4-nano"]
METHOD_MODEL = "openai:gpt-5.4-nano"
# The ensemble asks the method model once per temperature. Spanning low->high
# temperature is what produces the answer-to-answer spread that becomes the
# uncertainty band. Verified empirically that gpt-5.4-nano's output does vary
# across these.
ENSEMBLE_TEMPERATURES = [0.3, 0.7, 1.0]
PHRASING_ID = 0  # the single fixed "distribute 100 points" phrasing

CLASSES = ["SUPPORT", "CONTRADICT", "NOINFO"]


def _short(model: str) -> str:
    return model.split(":", 1)[-1]


def _pred(row: list[float] | None) -> str | None:
    return None if row is None else CLASSES[int(np.argmax(row))]


def _conf(row: list[float] | None) -> float | None:
    """Confidence in the chosen class = its share of the row total (0..1)."""
    if row is None:
        return None
    arr = np.asarray(row, dtype=float)
    return float(arr.max() / arr.sum()) if arr.sum() > 0 else None


def _mark(pred: str | None, gold: str) -> str:
    if pred is None:
        return "[red]failed[/red]"
    return "[green]correct[/green]" if pred == gold else "[red]wrong[/red]"


# ---------------------------------------------------------------------------
# Elicitation (ask one model for a verbalized distribution)
# ---------------------------------------------------------------------------


class VerbalizedDistribution(BaseModel):
    support: int = Field(ge=0, le=100)
    contradict: int = Field(ge=0, le=100)
    noinfo: int = Field(ge=0, le=100)

    def to_row(self) -> list[float]:
        return [float(self.support), float(self.contradict), float(self.noinfo)]


def _agent(model: str, *, prompted: bool) -> Agent:
    resolved = resolve_model(model) if model.startswith("ollama:") else model
    output_type = (
        PromptedOutput(VerbalizedDistribution) if prompted else VerbalizedDistribution
    )
    return Agent(resolved, instructions=SYSTEM_INSTRUCTIONS, output_type=output_type)


async def elicit(
    claim: str, evidence: str, model: str, temperature: float
) -> list[float] | None:
    """Ask one model once. Returns the raw 0-100 row, or None if it fails."""
    prompt = render(PHRASINGS[PHRASING_ID], claim, evidence)
    settings = ModelSettings(temperature=temperature)
    try:
        result = await _agent(model, prompted=False).run(prompt, model_settings=settings)
        return result.output.to_row()
    except (UnexpectedModelBehavior, ModelHTTPError):
        try:
            result = await _agent(model, prompted=True).run(prompt, model_settings=settings)
            return result.output.to_row()
        except Exception:  # noqa: BLE001
            return None
    except Exception:  # noqa: BLE001
        return None


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------


def _key(phase: str, claim_id: str, model: str, sample: int) -> str:
    return f"{phase}\x1f{claim_id}\x1f{model}\x1f{sample}"


def _load_cache(path: Path) -> dict[str, list[float] | None]:
    cache: dict[str, list[float] | None] = {}
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                cache[_key(r["phase"], r["claim_id"], r["model"], r["sample"])] = r["row"]
    return cache


def _append_cache(path: Path, phase: str, claim_id: str, model: str, sample: int, row) -> None:
    with path.open("a") as fh:
        fh.write(
            json.dumps(
                {
                    "phase": phase,
                    "claim_id": claim_id,
                    "model": model,
                    "sample": sample,
                    "row": row,
                }
            )
            + "\n"
        )


async def _get(
    phase: str,
    ex: Example,
    model: str,
    sample,
    temperature: float,
    cache: dict,
    cache_path: Path,
    lock: asyncio.Lock,
) -> list[float] | None:
    """Get one elicitation, from cache or fresh. ``sample`` is the cache slot id
    (an int for baseline; for the ensemble it encodes the temperature, e.g.
    ``"t0.7"``, so the cache is robust if the temperature list is reordered)."""
    k = _key(phase, ex.claim_id, model, sample)
    if k in cache:
        return cache[k]
    row = await elicit(ex.claim, ex.evidence, model, temperature)
    async with lock:
        cache[k] = row
        _append_cache(cache_path, phase, ex.claim_id, model, sample, row)
    return row


# ---------------------------------------------------------------------------
# Per-claim gather + render
# ---------------------------------------------------------------------------


async def gather_claim(
    ex: Example, args, cache: dict, cache_path: Path, lock: asyncio.Lock, status=None
) -> tuple[dict[str, list[float] | None], list[tuple[float, list[float] | None]]]:
    """Gather the baseline answers and the ensemble replicates for one claim.

    Returns (baseline_by_model, [(temperature, row), ...]). ``status`` is an
    optional rich Status updated as each call lands, so there's no silent wait.
    """
    def _tick(stage: str) -> None:
        if status is not None:
            status.update(f"[dim]{stage}[/dim]")

    n_models = len(args.baseline_models)
    _tick(f"asking {n_models} cloud baseline models…")
    baseline_rows = await asyncio.gather(
        *(
            _get("baseline", ex, m, 0, args.baseline_temperature, cache, cache_path, lock)
            for m in args.baseline_models
        )
    )
    baseline = dict(zip(args.baseline_models, baseline_rows))

    temps = args.ensemble_temperatures
    sem = asyncio.Semaphore(args.method_concurrency)
    done = 0

    async def one(temp: float) -> tuple[float, list[float] | None]:
        nonlocal done
        async with sem:
            row = await _get(
                "ensemble", ex, args.method_model, f"t{temp}", temp, cache, cache_path, lock
            )
        done += 1
        _tick(
            f"{_short(args.method_model)} ensemble · replicate {done}/{len(temps)} (temp {temp})"
        )
        return temp, row

    ensemble = list(await asyncio.gather(*(one(t) for t in temps)))
    return baseline, ensemble


def render_claim(idx: int, total: int, ex: Example, baseline: dict, ensemble: list, alpha0: float) -> None:
    gold = ex.label_name
    console.rule(f"[bold]Claim {idx} of {total}[/bold]")
    console.print(f'"{ex.claim}"')
    console.print(f"True answer: [bold]{gold}[/bold]\n")

    # Baseline: each cloud model asked once.
    t = Table(title="Baseline — each cloud model asked once")
    t.add_column("Model")
    t.add_column("Says")
    t.add_column("Confidence", justify="right")
    t.add_column("Result")
    for model in baseline:
        row = baseline[model]
        pred = _pred(row)
        conf = _conf(row)
        t.add_row(
            _short(model),
            pred or "—",
            f"{conf * 100:.0f}%" if conf is not None else "—",
            _mark(pred, gold),
        )
    console.print(t)

    # Method: the same weak model asked once per temperature.
    model_name = _short(ensemble_model())
    console.print(
        f"\n[bold]Method[/bold] — {model_name} ensemble "
        f"(one ask at each of {len(ensemble)} temperatures):"
    )
    et = Table(show_header=True)
    et.add_column("Temperature", justify="right")
    et.add_column("Says")
    et.add_column("Confidence", justify="right")
    for temp, row in ensemble:
        pred = _pred(row)
        conf = _conf(row)
        et.add_row(
            f"{temp}",
            pred or "—",
            f"{conf * 100:.0f}%" if conf is not None else "—",
        )
    console.print(et)

    good = [r for _, r in ensemble if r is not None]
    mix = build_mixture(
        [{"model": "m", "phrasing_id": i, "row": r} for i, r in enumerate(good)],
        alpha0=alpha0,
    )
    if mix is not None:
        pooled = mix.pooled_mean()
        ci = int(np.argmax(pooled))
        pred = CLASSES[ci]
        std = float(np.sqrt(mix.variance_decomposition()["total"][ci]))
        stable = (
            "the replicates agreed — trustworthy"
            if std < 0.10
            else "the replicates disagreed — treat with caution"
        )
        console.print(
            f"  [bold]Verdict: {pred}, {pooled[ci] * 100:.0f}% confident "
            f"(± {std * 100:.0f}%)[/bold] → {_mark(pred, gold)}"
        )
        console.print(f"  [dim](± is how much the replicates disagreed; {stable})[/dim]")
    console.print()


# Tiny module-level handle so render_claim can name the method model without
# threading args through (set once in run()).
_ARGS: argparse.Namespace | None = None


def ensemble_model() -> str:
    return _ARGS.method_model if _ARGS else METHOD_MODEL


# ---------------------------------------------------------------------------
# Scoring (unchanged maths; reuses mixture.py + metrics.py)
# ---------------------------------------------------------------------------


def _score(probs: np.ndarray, labels: np.ndarray) -> dict:
    preds = probs.argmax(axis=1)
    return {
        "accuracy": metrics.accuracy(probs, labels),
        "ece": metrics.expected_calibration_error(probs.max(axis=1), preds == labels),
        "brier": metrics.brier_score(probs, labels),
        "nll": metrics.negative_log_likelihood(probs, labels),
        "n": int(len(labels)),
    }


def score_baseline(model, examples, cache, alpha0) -> dict:
    probs, labels = [], []
    for ex in examples:
        row = cache.get(_key("baseline", ex.claim_id, model, 0))
        if row is None:
            continue
        p = repair_row(np.array(row), alpha0=alpha0)
        if p is None:
            continue
        probs.append(p)
        labels.append(ex.label)
    return _score(np.array(probs), np.array(labels)) if probs else {"n": 0}


def build_ensemble_mixtures(examples, cache, model, temperatures, alpha0) -> dict:
    out = {}
    for ex in examples:
        rows = [
            cache.get(_key("ensemble", ex.claim_id, model, f"t{t}")) for t in temperatures
        ]
        good = [r for r in rows if r is not None]
        out[ex.claim_id] = build_mixture(
            [{"model": model, "phrasing_id": i, "row": r} for i, r in enumerate(good)],
            alpha0=alpha0,
        )
    return out


# ---------------------------------------------------------------------------
# Summary tables
# ---------------------------------------------------------------------------


def baseline_summary(results: dict[str, dict]) -> Table:
    t = Table(title="SUMMARY · Baseline cloud models (one ask each)")
    t.add_column("Model")
    for c in ("Accuracy", "Calibration error", "Brier", "NLL", "Claims"):
        t.add_column(c, justify="right")
    for model, m in results.items():
        if m.get("n", 0) == 0:
            t.add_row(_short(model), "—", "—", "—", "—", "0")
            continue
        t.add_row(
            _short(model),
            f"{m['accuracy']:.0%}",
            f"{m['ece']:.3f}",
            f"{m['brier']:.3f}",
            f"{m['nll']:.3f}",
            str(m["n"]),
        )
    return t


def _cell(m: dict, key: str, fmt: str) -> str:
    if not m or m.get("n", 0) == 0 or key not in m:
        return "—"
    return format(m[key], fmt)


def head_to_head(model: str, single: dict, point: dict, dirichlet: dict, s_fit: float) -> Table:
    """The headline comparison: the SAME weak model, asked once vs run through the
    ensemble. Lower calibration error / Brier / NLL for the ensemble = the method
    improved that model's confidence."""
    t = Table(title=f"HEAD-TO-HEAD · {_short(model)}: single ask vs ensemble method")
    t.add_column("Approach")
    for c in ("Accuracy", "Calibration error", "Brier", "NLL", "Sampled-NLL"):
        t.add_column(c, justify="right")
    t.add_row(
        "Single ask (baseline)",
        _cell(single, "accuracy", ".0%"),
        _cell(single, "ece", ".3f"),
        _cell(single, "brier", ".3f"),
        _cell(single, "nll", ".3f"),
        "—",
    )
    t.add_row(
        "Ensemble (point mixture)",
        _cell(point, "accuracy", ".0%"),
        _cell(point, "ece", ".3f"),
        _cell(point, "brier", ".3f"),
        _cell(point, "nll", ".3f"),
        _cell(point, "sampled_nll", ".3f"),
    )
    t.add_row(
        f"Ensemble (Dirichlet, s={s_fit:.0f})",
        _cell(dirichlet, "accuracy", ".0%"),
        _cell(dirichlet, "ece", ".3f"),
        _cell(dirichlet, "brier", ".3f"),
        _cell(dirichlet, "nll", ".3f"),
        _cell(dirichlet, "sampled_nll", ".3f"),
    )
    return t


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def run(args: argparse.Namespace) -> dict:
    global _ARGS
    _ARGS = args

    examples = load_scifact(args.scifact_path)
    console.print(
        f"Loaded [bold]{len(examples)}[/bold] SciFact examples "
        f"(labels: {label_counts(examples)})"
    )
    examples = balanced_subset(examples, args.n_claims, seed=args.seed)
    console.print(
        f"Using [bold]{len(examples)}[/bold] balanced examples "
        f"(labels: {label_counts(examples)})\n"
    )

    cache_path = Path(args.cache)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache = _load_cache(cache_path)
    lock = asyncio.Lock()

    temps = args.ensemble_temperatures
    console.print(
        Panel(
            f"Baseline : {', '.join(_short(m) for m in args.baseline_models)} "
            f"— each asked once per claim ({len(examples) * len(args.baseline_models)} calls)\n"
            f"Method   : {_short(args.method_model)} ensemble "
            f"— one ask at each temperature {temps} "
            f"({len(examples) * len(temps)} calls)\n"
            f"Head-to-head: '{_short(args.method_model)}' appears in BOTH — its single\n"
            f"             baseline ask vs its ensemble — the core comparison.\n"
            f"Cache    : {cache_path}  (already-done calls are reused, no recharge)",
            title="What will run",
        )
    )
    console.print()

    # Process one claim at a time so a clean block appears as each completes.
    # A live status line ticks through the calls so there's never a silent wait.
    for i, ex in enumerate(examples, 1):
        with console.status(f"Claim {i}/{len(examples)}…") as status:
            baseline, ensemble = await gather_claim(
                ex, args, cache, cache_path, lock, status=status
            )
        render_claim(i, len(examples), ex, baseline, ensemble, args.alpha0)

    # --- score baselines ---
    baseline_results = {
        m: score_baseline(m, examples, cache, args.alpha0) for m in args.baseline_models
    }

    # --- method: dev/test split, fit s, score ---
    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(len(examples))
    n_dev = max(1, int(len(examples) * args.dev_frac))
    dev = [examples[i] for i in idx[:n_dev]]
    test = [examples[i] for i in idx[n_dev:]] or dev

    mixtures = build_ensemble_mixtures(
        examples, cache, args.method_model, args.ensemble_temperatures, args.alpha0
    )
    dev_pairs = [(mixtures[e.claim_id], e.label) for e in dev if mixtures[e.claim_id]]
    s_fit = fit_concentration_s(dev_pairs) if dev_pairs else 50.0
    test_pairs = [(mixtures[e.claim_id], e.label) for e in test if mixtures[e.claim_id]]

    point = dirichlet = {"n": 0}
    vd = {"within_component": 0.0, "within_model": 0.0, "between_model": 0.0}
    if test_pairs:
        probs = np.array([m.pooled_mean() for m, _ in test_pairs])
        labels = np.array([y for _, y in test_pairs])
        point = _score(probs, labels)
        dirichlet = dict(point)
        point["sampled_nll"] = metrics.mean_sampled_nll(test_pairs, 1e6, args.mc_samples, rng)
        dirichlet["sampled_nll"] = metrics.mean_sampled_nll(test_pairs, s_fit, args.mc_samples, rng)
        acc = {k: [] for k in vd}
        for m, _ in test_pairs:
            d = m.with_s(s_fit).variance_decomposition()
            for k in vd:
                acc[k].append(float(d[k].sum()))
        vd = {k: float(np.mean(v)) for k, v in acc.items()}

    # --- final summary ---
    console.rule("[bold]Overall results[/bold]")
    console.print(baseline_summary(baseline_results))
    console.print()
    # Headline: same model, single ask vs ensemble. The method model's own
    # single-ask score comes from the baseline pass (it's in baseline_models).
    single = baseline_results.get(args.method_model, {"n": 0})
    console.print(head_to_head(args.method_model, single, point, dirichlet, s_fit))
    console.print()
    console.print(
        f"Average uncertainty band the ensemble produced (within-model spread): "
        f"[bold]{vd['within_model']:.4f}[/bold]"
    )
    console.print()
    console.print(
        Panel(
            f"The head-to-head asks: does running [bold]{_short(args.method_model)}[/bold] as an\n"
            "ensemble give better-calibrated confidence than asking it once?\n\n"
            "'Accuracy'          = how often the chosen class is right.\n"
            "'Calibration error' = does a stated 70% really mean 70%? (lower = better)\n"
            "'Brier'             = overall quality of the probabilities (lower = better).\n"
            "'NLL'/'Sampled-NLL' = penalise confident-and-wrong (lower = better).\n"
            "If the ensemble rows beat the single-ask row on calibration/Brier, the\n"
            "method improved this model's confidence.\n"
            f"At {len(examples)} claims these numbers are indicative, not conclusive.",
            title="How to read this",
        )
    )

    return {
        "baseline": baseline_results,
        "method": {"point": point, "dirichlet": dirichlet, "s_fit": s_fit, "variance": vd},
        "head_to_head": {
            "model": args.method_model,
            "single_ask": single,
            "ensemble_point": point,
            "ensemble_dirichlet": dirichlet,
        },
        "config": {
            "n_claims": len(examples),
            "baseline_models": args.baseline_models,
            "method_model": args.method_model,
            "ensemble_temperatures": args.ensemble_temperatures,
        },
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scifact-path", required=True)
    p.add_argument("--n-claims", type=int, default=30)
    p.add_argument("--baseline-models", nargs="+", default=BASELINE_MODELS)
    p.add_argument("--method-model", default=METHOD_MODEL)
    p.add_argument("--baseline-temperature", type=float, default=0.0)
    p.add_argument(
        "--ensemble-temperatures", nargs="+", type=float, default=ENSEMBLE_TEMPERATURES,
        help="one ensemble replicate per temperature (the spread becomes the band)",
    )
    p.add_argument("--alpha0", type=float, default=0.5)
    p.add_argument("--dev-frac", type=float, default=0.3)
    p.add_argument("--mc-samples", type=int, default=2000)
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument(
        "--method-concurrency", type=int, default=4,
        help="concurrent ensemble calls (use 1 for local/ollama models)",
    )
    p.add_argument("--cache", default="experiments/dirichlet_confidence/cache/experiment.jsonl")
    p.add_argument("--out", default=None)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(argv)

    load_dotenv()
    results = asyncio.run(run(args))
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results, indent=2))
        console.print(f"\nResults written to [bold]{out}[/bold]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
