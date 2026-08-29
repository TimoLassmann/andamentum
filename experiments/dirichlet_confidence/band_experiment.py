"""BAND VALIDATION — does the uncertainty band flag a model's WRONG judgments?

WHY THIS EXISTS
===============

Before wiring the band into the epistemic system, we test its premise on a
labelled proxy for the system's most load-bearing micro-decision: judging a
piece of evidence against a claim (SciFact: claim + abstract -> SUPPORT /
CONTRADICT / NOINFO, with gold labels). The band is only worth integrating if a
BIGGER band reliably means a WRONG judgment.

We also settle a practical question that decides HOW integration would work:

  • PHRASING band   — ask the SAME question 5 different WAYS (temperature 0).
                      Each integration site would need hand-crafted paraphrases.
  • TEMPERATURE band — ask the SAME single prompt 5 times at temperature 0.7.
                      A fully generic wrapper: point it at any existing agent,
                      change nothing about its prompt, get a band for free.

In BOTH arms every call returns a full histogram (the 0-100 distribution over
the three classes); the K histograms are pooled with the same
mixture-of-Dirichlets machinery. The arms differ only in what creates the spread
between the K histograms.

THE YARDSTICK
=============

error-detection AUROC: treat "the judgment is wrong" as the alarm condition and
the band as the alarm score. 0.5 = the band is noise; 1.0 = the band perfectly
ranks every wrong judgment above every right one. If temperature ~= phrasing and
both are well above 0.5, integration can be a generic, transparent drop-in.

This experiment is fully separate from the epistemic system — it imports none of
it. It only reuses this folder's math (mixture, metrics) and data (scifact).

USAGE
=====
    # 1. Gather (only the temperature arm costs new calls; phrasing is reused).
    uv run python -m experiments.dirichlet_confidence.band_experiment gather \
        --scifact-path experiments/dirichlet_confidence/data/scifact --n-claims 30

    # 2. Analyze (NO API calls; run as often as you like).
    uv run python -m experiments.dirichlet_confidence.band_experiment analyze \
        --n-claims 30
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiments.dirichlet_confidence import metrics  # noqa: E402
from experiments.dirichlet_confidence.elicit import elicit_row  # noqa: E402
from experiments.dirichlet_confidence.mixture import build_mixture  # noqa: E402
from experiments.dirichlet_confidence.scifact import (  # noqa: E402
    Example,
    balanced_subset,
    label_counts,
    load_scifact,
)

console = Console()

NANO = "openai:gpt-5.4-nano"
CLASSES = ["SUPPORT", "CONTRADICT", "NOINFO"]

BAND_CACHE = Path("experiments/dirichlet_confidence/cache/band_experiment.jsonl")
# The phrasing arm was already gathered by the main experiment; reuse it.
PHRASING_SOURCE_CACHE = Path("experiments/dirichlet_confidence/cache/experiment.jsonl")

# Arm definitions: (name, n_variants, temperature, fixed_phrasing_id_or_None)
#   phrasing    -> variant i uses phrasing i, temperature 0
#   temperature -> variant i is sample i of phrasing 0, temperature T
DEFAULT_K = 5
DEFAULT_TEMPERATURE = 0.7


def _short(model: str) -> str:
    return model.split(":", 1)[-1]


# ---------------------------------------------------------------------------
# Cache (own schema: keyed by arm + claim + variant)
# ---------------------------------------------------------------------------

_SEP = "\x1f"


def _key(arm: str, claim_id: str, variant: int) -> str:
    return f"{arm}{_SEP}{claim_id}{_SEP}{variant}"


class BandCache:
    def __init__(self, path: Path = BAND_CACHE) -> None:
        self.path = Path(path)
        self._rows: dict[str, list[float] | None] = {}
        if self.path.exists():
            for line in self.path.read_text().splitlines():
                if line.strip():
                    r = json.loads(line)
                    self._rows[_key(r["arm"], r["claim_id"], r["variant"])] = r["row"]

    def has(self, arm: str, claim_id: str, variant: int) -> bool:
        return _key(arm, claim_id, variant) in self._rows

    def get(self, arm: str, claim_id: str, variant: int) -> list[float] | None:
        return self._rows.get(_key(arm, claim_id, variant))

    def size(self) -> int:
        return len(self._rows)

    def put(
        self, arm: str, claim_id: str, variant: int, temperature: float,
        phrasing_id: int, row: list[float] | None,
    ) -> None:
        self._rows[_key(arm, claim_id, variant)] = row
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as fh:
            fh.write(
                json.dumps(
                    {
                        "arm": arm,
                        "claim_id": claim_id,
                        "variant": variant,
                        "temperature": temperature,
                        "phrasing_id": phrasing_id,
                        "row": row,
                    }
                )
                + "\n"
            )

    def seed_phrasing_from(self, source: Path, model: str, k: int) -> int:
        """Reuse already-gathered phrasing rows (kind='method', phrasing_id k)
        from the main experiment cache so we don't re-pay. Idempotent."""
        if not source.exists():
            return 0
        seeded = 0
        for line in source.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("kind") != "method" or r.get("model") != model:
                continue
            pid = r.get("phrasing_id")
            if pid is None or pid >= k:
                continue
            if self.has("phrasing", r["claim_id"], pid):
                continue
            self.put("phrasing", r["claim_id"], pid, 0.0, pid, r["row"])
            seeded += 1
        return seeded


# ---------------------------------------------------------------------------
# Gather (the only step that calls the model)
# ---------------------------------------------------------------------------


async def gather(args: argparse.Namespace) -> None:
    load_dotenv()
    examples = balanced_subset(
        load_scifact(args.scifact_path), args.n_claims, seed=args.seed
    )
    console.print(f"Using {len(examples)} balanced claims ({label_counts(examples)})\n")

    cache = BandCache(Path(args.cache))
    seeded = cache.seed_phrasing_from(PHRASING_SOURCE_CACHE, args.model, args.k)
    if seeded:
        console.print(f"[dim]Reused {seeded} phrasing rows from the main cache.[/dim]")

    # Build the work list. variant i: phrasing arm -> phrasing i, temp 0;
    # temperature arm -> sample i of phrasing 0, temp T.
    jobs: list[tuple[str, Example, int, int, float]] = []
    for ex in examples:
        for i in range(args.k):
            jobs.append(("phrasing", ex, i, i, 0.0))
        for i in range(args.k):
            jobs.append(("temperature", ex, i, 0, args.temperature))

    missing = [j for j in jobs if not cache.has(j[0], j[1].claim_id, j[2])]
    console.print(
        f"Need {len(jobs)} elicitations · {len(jobs) - len(missing)} cached · "
        f"[bold]{len(missing)} to fetch[/bold] "
        f"(phrasing arm mostly reused; temperature arm is the new spend)\n"
    )
    if not missing:
        console.print("[green]Nothing to fetch — cache complete.[/green]")
        return

    sem = asyncio.Semaphore(args.concurrency)
    lock = asyncio.Lock()
    done = 0

    async def fetch(arm, ex, variant, phrasing_id, temperature):
        nonlocal done
        async with sem:
            row = await elicit_row(ex.claim, ex.evidence, args.model, phrasing_id, temperature)
        async with lock:
            cache.put(arm, ex.claim_id, variant, temperature, phrasing_id, row)
            done += 1
            ok = "ok" if row is not None else "[red]FAILED[/red]"
            console.print(
                f"  [{done}/{len(missing)}] {arm:11s} variant {variant} "
                f"claim {ex.claim_id:>12}  {ok}"
            )

    with console.status("Fetching…"):
        await asyncio.gather(*(fetch(*j) for j in missing))
    console.print(f"\n[green]Done.[/green] Cache: {cache.size()} rows. Now run `analyze`.")


# ---------------------------------------------------------------------------
# Analyze (no API calls)
# ---------------------------------------------------------------------------


def _claim_band(rows: list[list[float] | None], alpha0: float):
    """Pool K histograms for one claim -> (predicted_class, band) or None.

    The band is the std of the predicted class's probability ACROSS the K
    elicitations (sqrt of the mixture's between-component variance) — i.e. how
    much the K histograms disagreed on the chosen answer. It is s-free: it
    measures real spread between elicitations, not Monte-Carlo noise.
    """
    good = [r for r in rows if r is not None]
    if len(good) < 2:  # need >= 2 histograms to have any spread
        return None
    mix = build_mixture(
        [{"model": "m", "phrasing_id": i, "row": r} for i, r in enumerate(good)],
        alpha0=alpha0,
    )
    if mix is None:
        return None
    pooled = mix.pooled_mean()
    ci = int(np.argmax(pooled))
    band = float(np.sqrt(mix.variance_decomposition()["between_component"][ci]))
    return ci, band


def _arm_scores(arm: str, examples: list[Example], cache: BandCache, k: int, alpha0: float):
    preds, bands, correct = [], [], []
    n_missing = 0
    for ex in examples:
        rows = [cache.get(arm, ex.claim_id, v) for v in range(k)]
        if all(r is None for r in rows):
            n_missing += 1
            continue
        res = _claim_band(rows, alpha0)
        if res is None:
            n_missing += 1
            continue
        ci, band = res
        preds.append(ci)
        bands.append(band)
        correct.append(ci == ex.label)
    if not preds:
        return None
    bands = np.array(bands)
    correct = np.array(correct)
    return {
        "n": len(preds),
        "n_missing": n_missing,
        "accuracy": float(correct.mean()),
        "error_detection_auroc": metrics.error_detection_auroc(bands, correct),
        "mean_band_right": float(bands[correct].mean()) if correct.any() else float("nan"),
        "mean_band_wrong": float(bands[~correct].mean()) if (~correct).any() else float("nan"),
        "n_wrong": int((~correct).sum()),
        "risk_coverage": metrics.risk_coverage(bands, correct),
    }


def _fmt(v, fmt=".3f"):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "n/a"
    return format(v, fmt)


def analyze(args: argparse.Namespace) -> dict:
    examples = balanced_subset(
        load_scifact(args.scifact_path), args.n_claims, seed=args.seed
    )
    cache = BandCache(Path(args.cache))
    cache.seed_phrasing_from(PHRASING_SOURCE_CACHE, args.model, args.k)

    console.print(
        Panel(
            f"Task: evidence judgment on {len(examples)} SciFact claims "
            f"({label_counts(examples)})\n"
            f"Model: {_short(args.model)}  ·  every call returns a full histogram\n"
            f"Arms (K={args.k} histograms each, same pooling):\n"
            f"  • phrasing    — {args.k} different wordings, temperature 0\n"
            f"  • temperature — 1 fixed wording, {args.k} samples at temperature {args.temperature}\n"
            "Yardstick: does a bigger band flag a WRONG judgment? (error-detection AUROC)\n"
            "Cache: this analysis makes NO API calls.",
            title="Band validation experiment",
        )
    )

    arms = {
        "phrasing": _arm_scores("phrasing", examples, cache, args.k, args.alpha0),
        "temperature": _arm_scores("temperature", examples, cache, args.k, args.alpha0),
    }

    # Headline table.
    t = Table(title="Does the band flag wrong judgments? (the integration question)")
    t.add_column("Arm (diversity source)")
    t.add_column("Claims", justify="right")
    t.add_column("Accuracy", justify="right")
    t.add_column("Error-detect AUROC ↑", justify="right")
    t.add_column("band RIGHT ↓", justify="right")
    t.add_column("band WRONG ↑", justify="right")
    for name, s in arms.items():
        if s is None:
            t.add_row(name, "0", "—", "[yellow]not gathered[/yellow]", "—", "—")
            continue
        t.add_row(
            name,
            str(s["n"]),
            _fmt(s["accuracy"], ".0%"),
            _fmt(s["error_detection_auroc"]),
            _fmt(s["mean_band_right"], ".4f"),
            _fmt(s["mean_band_wrong"], ".4f"),
        )
    console.print()
    console.print(t)

    # Risk-coverage per gathered arm.
    for name, s in arms.items():
        if s is None:
            continue
        rc = Table(title=f"Risk-coverage · {name} arm (drop highest-band claims first)")
        rc.add_column("Keep", justify="right")
        rc.add_column("Kept", justify="right")
        rc.add_column("Accuracy on kept", justify="right")
        for r in s["risk_coverage"]:
            rc.add_row(f"{r['coverage']:.0%}", str(r["n"]), f"{r['accuracy']:.0%}")
        console.print()
        console.print(rc)

    # Verdict guidance.
    ph = arms["phrasing"]
    te = arms["temperature"]
    lines = [
        "AUROC near 0.5 = the band is noise; toward 1.0 = it reliably flags errors.",
        "'band WRONG' should be clearly larger than 'band RIGHT'.",
        "",
    ]
    # Verdict logic. "Useful" = clears the CHANCE_FLOOR (meaningfully above 0.5).
    # "Close" = within MATCH_TOL of each other. We report which arms are useful
    # first, then the practicality tradeoff between them.
    CHANCE_FLOOR = 0.6
    MATCH_TOL = 0.07
    if ph and te:
        pa = ph["error_detection_auroc"]
        ta = te["error_detection_auroc"]
        if not np.isnan(pa) and not np.isnan(ta):
            ph_ok = pa > CHANCE_FLOOR
            te_ok = ta > CHANCE_FLOOR
            if ph_ok and te_ok and abs(pa - ta) <= MATCH_TOL:
                lines.append(
                    f"→ BOTH flag errors and are close (phrasing {pa:.2f}, temperature "
                    f"{ta:.2f}). The generic TEMPERATURE wrapper (no per-site prompts) is "
                    "viable — GREEN LIGHT for the drop-in form."
                )
            elif ph_ok and te_ok:
                better = "phrasing" if pa > ta else "temperature"
                lines.append(
                    f"→ BOTH flag errors (phrasing {pa:.2f}, temperature {ta:.2f}); "
                    f"{better} is sharper by {abs(pa - ta):.2f}. Tradeoff: the generic "
                    "TEMPERATURE wrapper works but gives up some sharpness vs hand-crafted "
                    "PHRASINGS. Both are usable — choose per site."
                )
            elif ph_ok and not te_ok:
                lines.append(
                    f"→ Only PHRASING clears the bar ({pa:.2f} vs temperature {ta:.2f}): "
                    "integration would need hand-crafted paraphrases per site. RECONSIDER "
                    "the generic wrapper."
                )
            elif te_ok and not ph_ok:
                lines.append(
                    f"→ Only TEMPERATURE clears the bar ({ta:.2f} vs phrasing {pa:.2f}) — "
                    "unusual; worth re-checking the phrasings. Generic wrapper still viable."
                )
            else:
                lines.append(
                    f"→ NEITHER clears {CHANCE_FLOOR:.2f} (phrasing {pa:.2f}, temperature "
                    f"{ta:.2f}): the band does not reliably flag errors on this judgment "
                    "type. DON'T integrate yet."
                )
    elif ph and not te:
        pa = ph["error_detection_auroc"]
        verdict = "flags errors" if (not np.isnan(pa) and pa > CHANCE_FLOOR) else "is weak"
        lines.append(
            f"→ Only the phrasing arm is gathered (AUROC {pa:.2f}, {verdict}). Run "
            "`gather` to add the temperature arm for the full comparison."
        )
    lines.append("")
    nwrong = (ph or te or {}).get("n_wrong", 0) if (ph or te) else 0
    lines.append(
        f"CAVEAT: small n — AUROC rests on only ~{nwrong} wrong judgments. "
        "Scale up before believing it. A LOW band never proves correctness "
        "(a model can be stably wrong); the band is only ever a suspicion-raiser."
    )
    console.print()
    console.print(Panel("\n".join(lines), title="What this means for integration"))

    out = {
        "arms": arms,
        "config": {
            "n_claims": len(examples), "model": args.model, "k": args.k,
            "temperature": args.temperature, "alpha0": args.alpha0,
        },
    }
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(out, indent=2, default=float))
        console.print(f"\nResults written to [bold]{args.out}[/bold]")
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--scifact-path", default="experiments/dirichlet_confidence/data/scifact")
    common.add_argument("--n-claims", type=int, default=30)
    common.add_argument("--model", default=NANO)
    common.add_argument("--k", type=int, default=DEFAULT_K, help="histograms per claim per arm")
    common.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    common.add_argument("--alpha0", type=float, default=0.5)
    common.add_argument("--cache", default=str(BAND_CACHE))
    common.add_argument("--seed", type=int, default=42)

    g = sub.add_parser("gather", parents=[common], help="call the model (temperature arm)")
    g.add_argument("--concurrency", type=int, default=4)

    a = sub.add_parser("analyze", parents=[common], help="score the bands (no API calls)")
    a.add_argument("--out", default=None)

    args = p.parse_args(argv)
    if args.cmd == "gather":
        asyncio.run(gather(args))
    else:
        analyze(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
