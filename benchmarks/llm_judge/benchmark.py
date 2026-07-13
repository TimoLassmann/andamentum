"""Illustrative benchmark for ``andamentum.llm_judge`` with live rich output.

Runs the module's two entry points over ~20 curated examples pulled from the
source experiments (see ``extract_examples.py`` / ``examples.json``):

- ``judge_compare`` over 10 JudgeBench pairs (objective which-is-better gold).
- ``judge_score`` over 10 SciFact claims (SUPPORT/CONTRADICT/NOINFO gold,
  scored under one 'factual accuracy' criterion mapped to meets/partial/fails).

For every example it prints, live: the inputs, a spinner while the judge runs,
then the verdict with confidence/doubt bars, the order-consistency and
needs_review flags, and a correct/incorrect mark against the gold label. A
summary at the end reports accuracy, mean confidence/doubt, and — the point of
the whole module — accuracy split by the ``needs_review`` gate.

This calls a real LLM. Pick a model explicitly (no default):

    # fast single judge (local)
    uv run python benchmarks/llm_judge/benchmark.py --model ollama:gemma4:31b-nvfp4

    # panel of three (an agreement gate)
    uv run python benchmarks/llm_judge/benchmark.py \\
        --model ollama:gemma4:31b-nvfp4 --model ollama:gpt-oss:20b --model ollama:gemma4:26b-nvfp4

    # same model repeated as a panel (sampling-temperature agreement gate)
    uv run python benchmarks/llm_judge/benchmark.py --model ollama:gemma4:31b-nvfp4 --repeat 3

    # a quick two-example smoke of each task
    uv run python benchmarks/llm_judge/benchmark.py --model openai:gpt-5.4-nano --limit 2
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path
from typing import Sequence

from dotenv import load_dotenv
from rich.box import ROUNDED, SIMPLE
from rich.console import Console, Group
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

from andamentum.llm_judge import (
    CompareResult,
    Criterion,
    ScoreResult,
    judge_compare,
    judge_score,
)

load_dotenv()

_HERE = Path(__file__).resolve().parent
console = Console()

# One criterion that maps cleanly onto SciFact's SUPPORT/CONTRADICT/NOINFO gold,
# so judge_score's meets/partial/fails roll-up is directly checkable.
FACTUAL_ACCURACY = Criterion(
    name="factual accuracy",
    description=(
        "The statement is factually correct and supported by the provided evidence: "
        "'meets' if the evidence supports it, 'fails' if the evidence contradicts it, "
        "'partial' if the evidence is insufficient to decide."
    ),
)

_VERDICT_COLOR = {
    "a": "cyan",
    "b": "magenta",
    "tie": "yellow",
    "meets": "green",
    "partial": "yellow",
    "fails": "red",
}


def _trunc(text: str, limit: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _bar(value: float, width: int = 14) -> Text:
    """A 0-1 value as a small colored block bar."""
    value = value or 0.0  # normalize negative zero (one-hot entropy) to 0.00
    filled = round(value * width)
    if value >= 0.66:
        color = "green"
    elif value >= 0.4:
        color = "yellow"
    else:
        color = "red"
    bar = Text("█" * filled, style=color)
    bar.append("░" * (width - filled), style="grey37")
    bar.append(f" {value:.2f}", style="bold")
    return bar


def _flag(label: str, value: bool, *, true_is_good: bool) -> Text:
    good = value if true_is_good else not value
    mark = "yes" if value else "no"
    return Text(f"{label}: {mark}", style="green" if good else "red")


def _correct_mark(correct: bool) -> Text:
    return (
        Text("  ✓ CORRECT", style="bold green")
        if correct
        else Text("  ✗ INCORRECT", style="bold red")
    )


def _verdict_text(verdict: str) -> Text:
    return Text(verdict.upper(), style=f"bold {_VERDICT_COLOR.get(verdict, 'white')}")


def _judges_table(judges) -> Table:
    table = Table(box=SIMPLE, show_edge=False, pad_edge=False, expand=False)
    table.add_column("judge", style="dim")
    table.add_column("verdict")
    table.add_column("conf", justify="right")
    table.add_column("doubt", justify="right")
    table.add_column("order", justify="center")
    for j in judges:
        order = (
            "—"
            if j.order_consistent is None
            else ("ok" if j.order_consistent else "FLIP")
        )
        table.add_row(
            _trunc(j.model, 28),
            _verdict_text(j.verdict),
            f"{j.confidence:.2f}",
            f"{j.doubt:.2f}",
            Text(order, style="red" if order == "FLIP" else "dim"),
        )
    return table


def _result_metrics(
    *,
    confidence: float,
    doubt: float,
    needs_review: bool,
    order_consistent: bool | None,
) -> Table:
    table = Table(box=None, show_header=False, pad_edge=False)
    table.add_column(justify="right", style="dim")
    table.add_column()
    table.add_row("confidence", _bar(confidence))
    table.add_row("doubt", _bar(doubt))
    if order_consistent is not None:
        table.add_row(
            "", _flag("order_consistent", order_consistent, true_is_good=True)
        )
    table.add_row("", _flag("needs_review", needs_review, true_is_good=False))
    return table


# --------------------------------------------------------------------------- #
# compare
# --------------------------------------------------------------------------- #
async def run_compare(
    ex: dict, model: str | Sequence[str], index: int, total: int
) -> dict:
    console.print(
        Rule(
            Text.assemble(
                (f" compare {index}/{total} ", "bold white on blue"),
                (
                    f"  id={_trunc(ex['id'], 20)}  source={ex['source']}  gold={ex['gold'].upper()} ",
                    "dim",
                ),
            ),
            align="left",
        )
    )
    console.print(
        Panel(
            Group(
                Text.assemble(("Q: ", "bold"), _trunc(ex["question"], 260)),
                Text.assemble(("A: ", "bold cyan"), _trunc(ex["response_a"], 200)),
                Text.assemble(("B: ", "bold magenta"), _trunc(ex["response_b"], 200)),
            ),
            box=ROUNDED,
            border_style="grey37",
            padding=(0, 1),
        )
    )

    start = time.monotonic()
    with console.status(
        "[dim]judging (both A-first and B-first orders)…", spinner="dots"
    ):
        result: CompareResult = await judge_compare(
            ex["response_a"], ex["response_b"], context=ex["question"], model=model
        )
    elapsed = time.monotonic() - start

    correct = result.winner == ex["gold"]
    header = Text.assemble(
        ("winner ", "bold"),
        _verdict_text(result.winner),
        (f"   (gold {ex['gold'].upper()})", "dim"),
        _correct_mark(correct),
        (f"   {elapsed:.1f}s", "dim"),
    )
    body = [
        header,
        _result_metrics(
            confidence=result.confidence,
            doubt=result.doubt,
            needs_review=result.needs_review,
            order_consistent=result.order_consistent,
        ),
    ]
    if result.judges:
        body.append(Text("panel votes:", style="dim"))
        body.append(_judges_table(result.judges))
    body.append(
        Text.assemble(("why: ", "dim"), (_trunc(result.reasoning, 220), "italic dim"))
    )
    console.print(
        Panel(
            Group(*body),
            box=ROUNDED,
            border_style="green" if correct else "red",
            padding=(0, 1),
        )
    )
    console.print()
    return {
        "correct": correct,
        "confidence": result.confidence,
        "doubt": result.doubt,
        "needs_review": result.needs_review,
        "order_consistent": result.order_consistent,
    }


# --------------------------------------------------------------------------- #
# score
# --------------------------------------------------------------------------- #
async def run_score(
    ex: dict,
    model: str | Sequence[str],
    criteria: list[Criterion],
    index: int,
    total: int,
) -> dict:
    console.print(
        Rule(
            Text.assemble(
                (f" score {index}/{total} ", "bold white on dark_green"),
                (
                    f"  id={_trunc(ex['id'], 20)}  gold={ex['gold_label']} → expect {ex['expected_verdict'].upper()} ",
                    "dim",
                ),
            ),
            align="left",
        )
    )
    context = f"Task: judge whether the scientific claim is accurate, given the evidence.\n\nEvidence:\n{ex['evidence']}"
    console.print(
        Panel(
            Group(
                Text.assemble(("claim: ", "bold"), _trunc(ex["claim"], 260)),
                Text.assemble(
                    ("evidence: ", "bold"), (_trunc(ex["evidence"], 260), "dim")
                ),
            ),
            box=ROUNDED,
            border_style="grey37",
            padding=(0, 1),
        )
    )

    start = time.monotonic()
    n = len(criteria)
    with console.status(
        f"[dim]judging ({n} criterion call{'s' if n != 1 else ''})…", spinner="dots"
    ):
        result: ScoreResult = await judge_score(
            ex["claim"], context=context, criteria=criteria, model=model
        )
    elapsed = time.monotonic() - start

    correct = result.overall == ex["expected_verdict"]
    header = Text.assemble(
        ("overall ", "bold"),
        _verdict_text(result.overall),
        (f"   (expect {ex['expected_verdict'].upper()})", "dim"),
        _correct_mark(correct),
        (f"   {elapsed:.1f}s", "dim"),
    )
    crit = Table(box=SIMPLE, show_edge=False, pad_edge=False)
    crit.add_column("criterion", style="dim")
    crit.add_column("meets", justify="right", style="green")
    crit.add_column("partial", justify="right", style="yellow")
    crit.add_column("fails", justify="right", style="red")
    crit.add_column("why", overflow="ellipsis", max_width=52)
    for cs in result.per_criterion:
        crit.add_row(
            _trunc(cs.criterion, 22),
            str(cs.meets),
            str(cs.partial),
            str(cs.fails),
            Text(_trunc(cs.reasoning, 90), style="italic dim"),
        )
    body = [
        header,
        _result_metrics(
            confidence=result.confidence,
            doubt=result.doubt,
            needs_review=result.needs_review,
            order_consistent=None,
        ),
        crit,
    ]
    if result.judges:
        body.append(Text("panel votes:", style="dim"))
        body.append(_judges_table(result.judges))
    console.print(
        Panel(
            Group(*body),
            box=ROUNDED,
            border_style="green" if correct else "red",
            padding=(0, 1),
        )
    )
    console.print()
    return {
        "correct": correct,
        "confidence": result.confidence,
        "doubt": result.doubt,
        "needs_review": result.needs_review,
    }


# --------------------------------------------------------------------------- #
# summary
# --------------------------------------------------------------------------- #
def _summary_table(title: str, records: list[dict]) -> Table:
    table = Table(title=title, box=ROUNDED, title_style="bold", title_justify="left")
    table.add_column("metric", style="dim")
    table.add_column("value", justify="right")
    n = len(records)
    acc = sum(r["correct"] for r in records) / n
    table.add_row("examples", str(n))
    table.add_row("accuracy", f"{acc:.0%}  ({sum(r['correct'] for r in records)}/{n})")
    table.add_row("mean confidence", f"{sum(r['confidence'] for r in records) / n:.2f}")
    table.add_row("mean doubt", f"{sum(r['doubt'] for r in records) / n:.2f}")
    review = [r for r in records if r["needs_review"]]
    clean = [r for r in records if not r["needs_review"]]
    table.add_row("flagged needs_review", f"{len(review)}/{n}")
    # The module's core claim: the gate should concentrate errors in the flagged set.
    if clean:
        table.add_row(
            "accuracy | not flagged",
            f"{sum(r['correct'] for r in clean) / len(clean):.0%}  ({len(clean)} items)",
            style="green",
        )
    if review:
        table.add_row(
            "accuracy | flagged",
            f"{sum(r['correct'] for r in review) / len(review):.0%}  ({len(review)} items)",
            style="yellow",
        )
    return table


def _resolve_model(models: list[str], repeat: int) -> str | list[str]:
    if len(models) == 1 and repeat > 1:
        return models * repeat
    if len(models) == 1:
        return models[0]
    return models


async def main_async(args: argparse.Namespace) -> None:
    data = json.loads((_HERE / "examples.json").read_text())
    model = _resolve_model(args.model, args.repeat)
    is_panel = not isinstance(model, str)
    criteria = None if args.default_criteria else [FACTUAL_ACCURACY]

    console.print(
        Panel(
            Group(
                Text.assemble(("model: ", "bold"), (str(model), "cyan")),
                Text.assemble(
                    ("mode:  ", "bold"),
                    (
                        "panel (agreement gate)" if is_panel else "fast (single judge)",
                        "white",
                    ),
                ),
                Text.assemble(("tasks: ", "bold"), (args.task, "white")),
                Text.assemble(
                    ("score criteria: ", "bold"),
                    (
                        "module DEFAULT_CRITERIA"
                        if args.default_criteria
                        else "single 'factual accuracy' (mapped to SciFact gold)",
                        "dim",
                    ),
                ),
            ),
            title="andamentum.llm_judge — illustrative benchmark",
            box=ROUNDED,
            border_style="blue",
        )
    )
    if args.default_criteria and args.task in ("score", "both"):
        console.print(
            "[yellow]note:[/yellow] with --default-criteria the score verdict is not "
            "comparable to the SciFact gold, so score accuracy is not meaningful.\n"
        )

    compare_records: list[dict] = []
    score_records: list[dict] = []

    if args.task in ("compare", "both"):
        items = data["compare"][: args.limit] if args.limit else data["compare"]
        console.print(
            Rule("[bold blue]judge_compare — which output is better", style="blue")
        )
        for i, ex in enumerate(items, 1):
            compare_records.append(await run_compare(ex, model, i, len(items)))

    if args.task in ("score", "both"):
        items = data["score"][: args.limit] if args.limit else data["score"]
        console.print(
            Rule(
                "[bold dark_green]judge_score — score one output against criteria",
                style="green",
            )
        )
        for i, ex in enumerate(items, 1):
            score_records.append(
                await run_score(ex, model, criteria or [], i, len(items))
            )

    console.print(Rule("[bold]summary"))
    if compare_records:
        console.print(_summary_table("judge_compare", compare_records))
    if score_records and not args.default_criteria:
        console.print(_summary_table("judge_score", score_records))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Illustrative rich-console benchmark for andamentum.llm_judge.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--model",
        action="append",
        required=True,
        metavar="MODEL_ID",
        help="Model id (any pydantic-ai id). Repeat for a panel. One model = fast path.",
    )
    p.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="Repeat a single --model N times as a same-model panel (agreement gate).",
    )
    p.add_argument("--task", choices=("compare", "score", "both"), default="both")
    p.add_argument(
        "--limit", type=int, default=0, help="Cap examples per task (0 = all)."
    )
    p.add_argument(
        "--default-criteria",
        action="store_true",
        help="Score against the module DEFAULT_CRITERIA instead of the single "
        "factual-accuracy criterion (accuracy vs SciFact gold then not meaningful).",
    )
    return p


def main() -> None:
    args = build_parser().parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
