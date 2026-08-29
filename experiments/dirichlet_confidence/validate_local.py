"""LOCAL-MODEL VALIDATION — does single-call verbalized confidence survive on
the models the epistemic system actually runs?

WHY THIS EXISTS
===============

The main dirichlet_confidence study showed that the entropy of ONE verbalized
histogram (claim x evidence -> 0-100 over SUPPORT/CONTRADICT/NOINFO) is a useful
"am I wrong?" detector — but it showed this on OpenAI models, and its single
SMALL model (nano) failed. The epistemic system runs on small LOCAL models.

Before changing anything in `andamentum.epistemic`, this answers the one gating
question: on `gemma4:12b-mxfp8` and `gpt-oss:20b`, does a single histogram's
entropy rank the model's WRONG judgments above its right ones (error-detection
AUROC well above 0.5)? And does the model even PRODUCE graded histograms, or is
it so overconfident (always [100,0,0]) that the entropy signal is degenerate?

This experiment imports NOTHING from `andamentum.epistemic`. It reuses only this
folder's harness (elicit, metrics) and data (scifact). It costs ZERO cloud spend
— both models are local Ollama.

THE YARDSTICK
=============
error-detection AUROC: treat "the judgment is wrong" as the alarm condition and
the single-call entropy as the alarm score. 0.5 = entropy is noise; >~0.65 with a
CI excluding 0.5 = a real, usable warning light worth wiring in.

USAGE
=====
    # 0. Measure per-call latency (1 call per model) to estimate run time.
    uv run python -m experiments.dirichlet_confidence.validate_local probe

    # 1. Gather (local Ollama; no cloud cost). Cached, so reruns are free and
    #    extending --n-claims later only pays for the new claims.
    uv run python -m experiments.dirichlet_confidence.validate_local gather --n-claims 60

    # 2. Analyze (NO model calls; run as often as you like).
    uv run python -m experiments.dirichlet_confidence.validate_local analyze --n-claims 60

    # gather + analyze in one go:
    uv run python -m experiments.dirichlet_confidence.validate_local run --n-claims 60
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pydantic import BaseModel, Field  # noqa: E402
from pydantic_ai import Agent  # noqa: E402
from pydantic_ai.exceptions import ModelHTTPError, UnexpectedModelBehavior  # noqa: E402
from pydantic_ai.output import PromptedOutput  # noqa: E402
from pydantic_ai.settings import ModelSettings  # noqa: E402

from andamentum.core import resolve_model  # noqa: E402

from experiments.dirichlet_confidence import metrics  # noqa: E402
from experiments.dirichlet_confidence.elicit import elicit_row  # noqa: E402
from experiments.dirichlet_confidence.scifact import (  # noqa: E402
    Example,
    balanced_subset,
    label_counts,
    load_scifact,
)

console = Console()

# The two local target models, exactly as the user named them.
MODELS = ["ollama:gemma4:12b-mxfp8", "ollama:gpt-oss:20b"]

# One call per claim: phrasing 0 (the points-distribution framing), temperature 0
# for reproducibility. The detector is the entropy WITHIN that single histogram.
PHRASING_ID = 0
TEMPERATURE = 0.0
SCIFACT_PATH = "experiments/dirichlet_confidence/data/scifact"

# Each arm has its own cache + results file so they never collide. "baseline"
# reuses the files the first run already populated.
ARMS = {
    "baseline": {
        "cache": Path("experiments/dirichlet_confidence/cache/validate_local.jsonl"),
        "results": Path("experiments/dirichlet_confidence/results/validate_local.json"),
    },
    "engineered": {
        "cache": Path("experiments/dirichlet_confidence/cache/validate_local_engineered.jsonl"),
        "results": Path("experiments/dirichlet_confidence/results/validate_local_engineered.json"),
    },
}

# Back-compat aliases (probe + default paths).
CACHE = ARMS["baseline"]["cache"]
RESULTS = ARMS["baseline"]["results"]

_SEP = "\x1f"

# ---------------------------------------------------------------------------
# ENGINEERED ARM — three levers against gemma's 82% one-hot degeneracy:
#   1. reasoning field FIRST in the schema -> forces chain-of-thought before
#      the numbers are committed (field order = generation order), emulating
#      what the reasoning model gpt-oss does natively.
#   2. per-field descriptions stating each bucket's meaning + the sum-to-100
#      contract at the field level (baseline schema fields are bare).
#   3. a calibration-PRESERVING nudge: permit confidence, but reserve 0/100 for
#      genuinely unambiguous evidence and put real weight on NOINFO when the
#      evidence is only partially relevant. (NOT "always spread" — that would
#      just flatten entropy and destroy the signal.)
# Same question wording as baseline (points-distribution) so the only variables
# are the framing, the schema, and the forced reasoning.
# ---------------------------------------------------------------------------

ENGINEERED_INSTRUCTIONS = (
    "You are a careful scientific claim-verification assistant. You are given a "
    "CLAIM and an EVIDENCE passage drawn from a scientific abstract. Decide how "
    "the evidence bears on the claim, across exactly three categories:\n"
    "  SUPPORT    — the evidence supports the claim\n"
    "  CONTRADICT — the evidence contradicts the claim\n"
    "  NOINFO     — the evidence neither supports nor contradicts the claim\n\n"
    "First reason briefly about what the evidence actually establishes relative "
    "to the claim — what it directly states, what it only implies, and what it "
    "leaves open. Then express your judgment as three integers from 0 to 100 "
    "(one per category) that together sum to 100.\n\n"
    "These three numbers are a probability distribution over the categories, not "
    "a single choice. Use intermediate values to express genuine uncertainty: "
    "reserve extremes like 0 or 100 for cases where the evidence is truly "
    "unambiguous. If the evidence is only partially relevant, or related but not "
    "decisive, put real weight on NOINFO. If it leans one way without settling "
    "the matter, split your points to reflect that residual doubt. Report the "
    "reasoning first, then the three numbers in support, contradict, noinfo."
)

_SUM_NOTE = "support + contradict + noinfo must equal 100."


class EngineeredDistribution(BaseModel):
    """Reasoning-first verbalized distribution over the three SciFact classes."""

    reasoning: str = Field(
        description="Brief step-by-step reasoning about what the evidence "
        "establishes relative to the claim, BEFORE assigning any scores."
    )
    support: int = Field(
        ge=0, le=100,
        description=f"Belief points (0-100) that the evidence SUPPORTS the claim. {_SUM_NOTE}",
    )
    contradict: int = Field(
        ge=0, le=100,
        description=f"Belief points (0-100) that the evidence CONTRADICTS the claim. {_SUM_NOTE}",
    )
    noinfo: int = Field(
        ge=0, le=100,
        description=f"Belief points (0-100) that the evidence gives NO INFORMATION about the claim. {_SUM_NOTE}",
    )

    def to_row(self) -> list[float]:
        return [float(self.support), float(self.contradict), float(self.noinfo)]


_ENGINEERED_PROMPT = (
    "CLAIM:\n{claim}\n\nEVIDENCE:\n{evidence}\n\n"
    "Distribute 100 points across the categories SUPPORT, CONTRADICT, and "
    "NOINFO according to how the evidence bears on the claim."
)


def _engineered_agent(model: str, *, prompted: bool) -> Agent:
    resolved = resolve_model(model) if model.startswith("ollama:") else model
    output_type = (
        PromptedOutput(EngineeredDistribution) if prompted else EngineeredDistribution
    )
    return Agent(resolved, instructions=ENGINEERED_INSTRUCTIONS, output_type=output_type)


async def elicit_engineered(
    claim: str, evidence: str, model: str, temperature: float
) -> list[float] | None:
    """Engineered-arm elicitation: same question, reasoning-first schema + nudge."""
    prompt = _ENGINEERED_PROMPT.format(claim=claim, evidence=evidence)
    settings = ModelSettings(temperature=temperature)
    try:
        result = await _engineered_agent(model, prompted=False).run(prompt, model_settings=settings)
        return result.output.to_row()
    except (UnexpectedModelBehavior, ModelHTTPError):
        try:
            result = await _engineered_agent(model, prompted=True).run(prompt, model_settings=settings)
            return result.output.to_row()
        except Exception:  # noqa: BLE001
            return None
    except Exception:  # noqa: BLE001
        return None


async def _elicit(arm: str, claim: str, evidence: str, model: str) -> list[float] | None:
    if arm == "engineered":
        return await elicit_engineered(claim, evidence, model, TEMPERATURE)
    return await elicit_row(claim, evidence, model, PHRASING_ID, TEMPERATURE)


def _short(model: str) -> str:
    return model.split(":", 1)[-1]


# ---------------------------------------------------------------------------
# Cache: keyed by (model, claim_id). One histogram row per pair.
# ---------------------------------------------------------------------------


class Cache:
    def __init__(self, path: Path = CACHE) -> None:
        self.path = Path(path)
        self._rows: dict[str, list[float] | None] = {}
        if self.path.exists():
            for line in self.path.read_text().splitlines():
                if line.strip():
                    r = json.loads(line)
                    self._rows[f"{r['model']}{_SEP}{r['claim_id']}"] = r["row"]

    def has(self, model: str, claim_id: str) -> bool:
        return f"{model}{_SEP}{claim_id}" in self._rows

    def get(self, model: str, claim_id: str) -> list[float] | None:
        return self._rows.get(f"{model}{_SEP}{claim_id}")

    def put(self, model: str, claim_id: str, row: list[float] | None) -> None:
        self._rows[f"{model}{_SEP}{claim_id}"] = row
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as fh:
            fh.write(json.dumps({"model": model, "claim_id": claim_id, "row": row}) + "\n")


# ---------------------------------------------------------------------------
# Math helpers
# ---------------------------------------------------------------------------


def _normed(row: list[float] | None) -> np.ndarray | None:
    """Normalise a raw 0-100 histogram to a probability vector (None if invalid)."""
    if row is None:
        return None
    arr = np.asarray(row, dtype=float)
    total = arr.sum()
    if total <= 0:
        return None
    return arr / total


def _entropy(p: np.ndarray) -> float:
    """Shannon entropy in nats, normalised to [0, 1] by log(C). Higher = less sure."""
    p = np.clip(p, 1e-12, 1.0)
    return float(-(p * np.log(p)).sum() / np.log(len(p)))


# ---------------------------------------------------------------------------
# Gather
# ---------------------------------------------------------------------------


async def _gather(examples: list[Example], cache: Cache, arm: str, models: list[str]) -> None:
    total = len(examples) * len(models)
    done = 0
    t0 = time.perf_counter()
    for model in models:
        for ex in examples:
            done += 1
            if cache.has(model, ex.claim_id):
                continue
            row = await _elicit(arm, ex.claim, ex.evidence, model)
            cache.put(model, ex.claim_id, row)
            elapsed = time.perf_counter() - t0
            console.print(
                f"[dim]{done}/{total}[/dim] [{arm}] {_short(model)} {ex.claim_id} "
                f"-> {row}  [dim]({elapsed:.0f}s elapsed)[/dim]"
            )


# ---------------------------------------------------------------------------
# Analyze
# ---------------------------------------------------------------------------


def _boot_auroc(score: np.ndarray, correct: np.ndarray, seed: int = 42, n: int = 2000):
    """Point + 95% bootstrap CI for error-detection AUROC."""
    rng = np.random.default_rng(seed)
    point = metrics.error_detection_auroc(score, correct)
    boots = []
    m = len(score)
    for _ in range(n):
        idx = rng.integers(0, m, m)
        val = metrics.error_detection_auroc(score[idx], correct[idx])
        if not np.isnan(val):
            boots.append(val)
    lo, hi = (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))) if boots else (float("nan"), float("nan"))
    return point, lo, hi


def _analyze_model(model: str, examples: list[Example], cache: Cache) -> dict | None:
    probs, labels, entropies, conf = [], [], [], []
    n_missing = 0
    for ex in examples:
        p = _normed(cache.get(model, ex.claim_id))
        if p is None:
            n_missing += 1
            continue
        probs.append(p)
        labels.append(ex.label)
        entropies.append(_entropy(p))
        conf.append(float(p.max()))
    if not probs:
        return None

    probs_arr = np.vstack(probs)
    labels_arr = np.asarray(labels)
    entropies_arr = np.asarray(entropies)
    conf_arr = np.asarray(conf)
    correct = probs_arr.argmax(axis=1) == labels_arr

    point, lo, hi = _boot_auroc(entropies_arr, correct)

    # Degeneracy: how often is the histogram effectively one-hot (max >= 0.95),
    # i.e. entropy ~ 0? If this is high, the entropy detector cannot work.
    degenerate = float((conf_arr >= 0.95).mean())

    return {
        "model": model,
        "n": int(len(probs)),
        "n_missing": int(n_missing),
        "accuracy": metrics.accuracy(probs_arr, labels_arr),
        "n_wrong": int((~correct).sum()),
        "ece": metrics.expected_calibration_error(conf_arr, correct),
        "error_detection_auroc": {"point": point, "lo": lo, "hi": hi},
        "mean_entropy": float(entropies_arr.mean()),
        "degeneracy_rate": degenerate,
        "risk_coverage": metrics.risk_coverage(entropies_arr, correct),
    }


def _render(out: dict) -> None:
    console.print(Panel.fit(
        f"Local-model validation — single-call verbalized confidence\n"
        f"n_claims requested: {out['config']['n_claims']}  ·  arm: {out['config'].get('arm', 'baseline')}  ·  temp {TEMPERATURE}",
        style="bold",
    ))
    for model, r in out["results"].items():
        if r is None:
            console.print(f"[red]{_short(model)}: no usable rows[/red]")
            continue
        a = r["error_detection_auroc"]
        verdict = (
            "[green]usable signal (CI excludes 0.5)[/green]" if a["lo"] > 0.5
            else "[yellow]inconclusive / no signal[/yellow]"
        )
        t = Table(title=f"{_short(model)}  (n={r['n']}, wrong={r['n_wrong']}, missing={r['n_missing']})")
        t.add_column("metric"); t.add_column("value", justify="right")
        t.add_row("accuracy", f"{r['accuracy']:.3f}")
        t.add_row("ECE (calibration err)", f"{r['ece']:.3f}")
        t.add_row("error-detection AUROC", f"{a['point']:.3f}  [{a['lo']:.3f}, {a['hi']:.3f}]")
        t.add_row("mean entropy", f"{r['mean_entropy']:.3f}")
        t.add_row("degeneracy (max>=.95)", f"{r['degeneracy_rate']:.0%}")
        console.print(t)
        console.print(f"  verdict: {verdict}\n")
        rc = "  ".join(f"{x['coverage']:.0%}:{x['accuracy']:.2f}" for x in r["risk_coverage"])
        console.print(f"  [dim]risk-coverage (keep most-confident first) {rc}[/dim]\n")


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------


async def _probe() -> None:
    """One call per model; report latency to ground the run-time estimate."""
    exs = balanced_subset(load_scifact(SCIFACT_PATH), 3)
    ex = exs[0]
    console.print(Panel.fit("Latency probe — 1 call per model (includes cold model load)", style="bold"))
    for model in MODELS:
        t0 = time.perf_counter()
        row = await elicit_row(ex.claim, ex.evidence, model, PHRASING_ID, TEMPERATURE)
        dt = time.perf_counter() - t0
        console.print(f"{_short(model):20s} {dt:6.1f}s  -> {row}")
    console.print(
        "\n[dim]Note: first call per model includes one-time load into memory; "
        "steady-state per-call time is lower. gpt-oss is a reasoning model and "
        "is the slower of the two.[/dim]"
    )


def _examples(n: int) -> list[Example]:
    exs = balanced_subset(load_scifact(SCIFACT_PATH), n)
    console.print(f"[dim]Loaded {len(exs)} claims  ·  labels {label_counts(exs)}[/dim]")
    return exs


def _analyze(n: int, arm: str, models: list[str]) -> dict:
    cache = Cache(ARMS[arm]["cache"])
    exs = _examples(n)
    out = {
        "config": {"n_claims": n, "arm": arm, "temperature": TEMPERATURE, "models": models},
        "results": {m: _analyze_model(m, exs, cache) for m in models},
    }
    results_path = ARMS[arm]["results"]
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(out, indent=2))
    _render(out)
    console.print(f"[dim]results -> {results_path}[/dim]")
    return out


def _compare(n: int, models: list[str]) -> None:
    """Baseline vs engineered, side by side, per model — the 'how much does
    prompt engineering help?' readout."""
    base = _analyze(n, "baseline", models)
    eng = _analyze(n, "engineered", models)
    console.print(Panel.fit("BASELINE vs ENGINEERED  (Δ = engineered − baseline)", style="bold"))
    for model in models:
        b, e = base["results"].get(model), eng["results"].get(model)
        if not b or not e:
            console.print(f"[red]{_short(model)}: missing an arm[/red]")
            continue
        t = Table(title=_short(model))
        for c in ("metric", "baseline", "engineered", "Δ"):
            t.add_column(c, justify="right" if c != "metric" else "left")
        rows = [
            ("accuracy", b["accuracy"], e["accuracy"], "{:+.3f}"),
            ("ECE (lower=better)", b["ece"], e["ece"], "{:+.3f}"),
            ("error-det AUROC", b["error_detection_auroc"]["point"], e["error_detection_auroc"]["point"], "{:+.3f}"),
            ("mean entropy", b["mean_entropy"], e["mean_entropy"], "{:+.3f}"),
            ("degeneracy rate", b["degeneracy_rate"], e["degeneracy_rate"], "{:+.0%}"),
        ]
        for name, bv, ev, fmt in rows:
            pct = name == "degeneracy rate"
            t.add_row(
                name,
                f"{bv:.0%}" if pct else f"{bv:.3f}",
                f"{ev:.0%}" if pct else f"{ev:.3f}",
                fmt.format(ev - bv),
            )
        console.print(t)
        ba, ea = b["error_detection_auroc"], e["error_detection_auroc"]
        console.print(
            f"  baseline AUROC CI [{ba['lo']:.3f}, {ba['hi']:.3f}]  ·  "
            f"engineered AUROC CI [{ea['lo']:.3f}, {ea['hi']:.3f}]\n"
        )


def _grid(n: int, models: list[str]) -> None:
    """All model x arm cells in ONE table: rows = metrics, columns = the four
    (model, arm) combinations side by side."""
    exs = _examples(n)
    alias = {"ollama:gemma4:12b-mxfp8": "gemma", "ollama:gpt-oss:20b": "gpt-oss"}
    arm_label = {"baseline": "default", "engineered": "engineered"}
    cells: list[tuple[str, dict | None]] = []  # (column label, analysis dict)
    for model in models:
        for arm in ("baseline", "engineered"):
            cache = Cache(ARMS[arm]["cache"])
            r = _analyze_model(model, exs, cache)
            label = f"{alias.get(model, _short(model))}\n{arm_label[arm]}"
            cells.append((label, r))

    def fmt(r: dict | None, key: str, kind: str) -> str:
        if r is None:
            return "—"
        if kind == "auroc":
            a = r["error_detection_auroc"]
            return f"{a['point']:.3f}\n[{a['lo']:.2f},{a['hi']:.2f}]"
        v = r[key]
        return f"{v:.0%}" if kind == "pct" else f"{v:.3f}"

    t = Table(title=f"All cells — {n} balanced SciFact claims, single call")
    t.add_column("metric")
    for label, _ in cells:
        t.add_column(label, justify="right")
    spec = [
        ("accuracy", "accuracy", "f3"),
        ("n_wrong", "n_wrong", "int"),
        ("ECE (↓ better)", "ece", "f3"),
        ("error-det AUROC", "error_detection_auroc", "auroc"),
        ("mean entropy", "mean_entropy", "f3"),
        ("degeneracy (one-hot)", "degeneracy_rate", "pct"),
    ]
    for name, key, kind in spec:
        row = [name]
        for _, r in cells:
            if kind == "int":
                row.append(str(r["n_wrong"]) if r else "—")
            else:
                row.append(fmt(r, key, kind))
        t.add_row(*row)
    console.print(t)
    console.print(
        "[dim]usable signal = error-det AUROC CI excludes 0.50. "
        "AUROC CIs are wide (~9-12 errors); ECE/degeneracy resolve on all 60.[/dim]"
    )


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="mode", required=True)
    sub.add_parser("probe")
    for name in ("gather", "analyze", "run"):
        sp = sub.add_parser(name)
        sp.add_argument("--n-claims", type=int, default=60)
        sp.add_argument("--arm", choices=list(ARMS), default="baseline")
        sp.add_argument("--models", nargs="+", default=None,
                        help="override model list (default: both)")
    cp = sub.add_parser("compare")
    cp.add_argument("--n-claims", type=int, default=60)
    cp.add_argument("--models", nargs="+", default=None)
    gp = sub.add_parser("grid")
    gp.add_argument("--n-claims", type=int, default=60)
    gp.add_argument("--models", nargs="+", default=None)
    args = p.parse_args(argv)

    models = getattr(args, "models", None) or MODELS

    if args.mode == "probe":
        asyncio.run(_probe())
    elif args.mode == "gather":
        asyncio.run(_gather(_examples(args.n_claims), Cache(ARMS[args.arm]["cache"]), args.arm, models))
    elif args.mode == "analyze":
        _analyze(args.n_claims, args.arm, models)
    elif args.mode == "run":
        asyncio.run(_gather(_examples(args.n_claims), Cache(ARMS[args.arm]["cache"]), args.arm, models))
        _analyze(args.n_claims, args.arm, models)
    elif args.mode == "compare":
        _compare(args.n_claims, models)
    elif args.mode == "grid":
        _grid(args.n_claims, models)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
