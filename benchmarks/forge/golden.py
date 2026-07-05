"""Tier 3 — golden-task correctness: build the system, RUN it, score the OUTPUT.

Tier 2 scores ``audit.works`` — holes filled, generated tests pass, dialect-clean. That is
a *smoke* bar, not a correctness bar: a workflow can be structurally perfect and
semantically wrong (the canonical example: a "summarise each article" system that
processed ONE item passed Tier 2). Tier 3 closes that gap: ``run_golden`` builds the
system end-to-end, executes the rendered package's CLI (``python -m <name> "<input>"
--model <id>``) on a real input, and scores whether the output actually covers the task.

The rubric is deterministic: each :class:`GoldenCase` carries ``marker_groups`` — the
output must contain at least one marker from EVERY group, case-insensitive
(:func:`score_output`, pure). All four golden inputs are text-only, so the built systems
execute offline-of-the-web (they fit the subprocess sandbox — no network nodes).

An outcome is ``correct`` only when the audit says works AND the run exits 0 AND every
marker group is covered.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from andamentum.forge import run_forge
from andamentum.forge.agents import AgentSink
from andamentum.forge.sandbox import SandboxPort

#: Lines of combined output kept for the report when a run fails or misses markers.
TAIL_LINES = 15


@dataclass
class GoldenCase:
    """One golden task: a brief, a real input, and the marker rubric its output must hit."""

    key: str
    brief: str
    input_text: str
    marker_groups: list[
        tuple[str, ...]
    ]  # output must contain >=1 marker from EVERY group
    note: str = ""


@dataclass
class GoldenOutcome:
    """What one golden run produced.

    ``kind`` ∈ {"correct", "wrong_output", "run_failed", "build_failed", "refused"}.
    ``correct`` requires all three: ``works`` (the sandbox-audit verdict), run exit 0,
    and every marker group covered.
    """

    kind: str
    works: bool | None = None
    covered: list[bool] = field(default_factory=list)
    output_tail: str = ""
    seconds: float = 0.0
    error: str = ""


GOLDEN_CASES: list[GoldenCase] = [
    GoldenCase(
        key="reduce",
        brief=(
            "Given several meeting notes, one per line, summarise each note and "
            "combine the summaries into one brief."
        ),
        input_text=(
            "Kestrel kickoff: Priya Raghavan committed to shipping the Zephyr "
            "dashboard by March.\n"
            "Budget sync: Marcus Oyelaran flagged a forty-thousand-dollar overrun "
            "on the Bluewattle cloud migration.\n"
            "Hiring huddle: Ingrid Solheim approved two new engineers for the "
            "Tanami data platform."
        ),
        marker_groups=[
            ("kestrel", "priya", "raghavan", "zephyr"),
            ("marcus", "oyelaran", "bluewattle", "overrun"),
            ("ingrid", "solheim", "tanami", "hiring"),
        ],
        note="the historical Tier-2 blind spot: one-item processing passes the smoke bar",
    ),
    GoldenCase(
        key="per_item",
        brief=(
            "Given several people's names, one per line, write a short friendly "
            "greeting for each name."
        ),
        input_text="Aroha\nBenedikt\nChinwe",
        marker_groups=[("aroha",), ("benedikt",), ("chinwe",)],
        note="every item must surface in the output, not just the first",
    ),
    GoldenCase(
        key="sequence",
        brief="Summarise the document into three bullet points.",
        input_text=(
            "Project Heronwatch is a five-year effort to track grey heron migration "
            "across the Camargue wetlands in southern France. The team, led by "
            "ornithologist Dr. Lena Vasquez, designed a solar-assisted GPS tracker "
            "that weighs just nine grams, light enough to sit on a heron's back "
            "without changing its flight behaviour. In the first field season the "
            "group fitted trackers to 240 herons captured at twelve roost sites, "
            "making it the largest single-species tagging campaign ever attempted "
            "in the region. Early telemetry shows the birds split into two distinct "
            "flyways, one hugging the Rhone valley and one crossing directly over "
            "the Mediterranean toward Tunisia. The tracker's battery design is the "
            "project's quiet triumph: a supercapacitor buffer lets each unit run "
            "for eighteen months between full recharges, surviving the low-light "
            "winter weeks when the solar panel gathers almost nothing."
        ),
        marker_groups=[
            ("heronwatch", "nine gram", "nine-gram", "9 gram", "9-gram"),
            ("240", "camargue"),
            ("eighteen", "18", "battery", "supercapacitor"),
        ],
        note="a faithful summary must carry the document's three distinctive facts",
    ),
    GoldenCase(
        key="branch",
        brief="Classify a support ticket's urgency and route it to the matching team.",
        input_text=(
            "URGENT: the production database has been down since 02:00 UTC. All "
            "customers are affected and error rates are at 100 percent. Enterprise "
            "clients cannot log in and we are losing transactions every minute."
        ),
        marker_groups=[
            ("urgent", "high", "critical", "p1", "sev1", "severity 1"),
            (
                "database",
                "infrastructure",
                "engineering",
                "ops",
                "platform",
                "sre",
                "on-call",
                "dba",
            ),
        ],
        note="an obviously-urgent ticket must be classified urgent and routed somewhere sane",
    ),
]


def score_output(case: GoldenCase, output: str) -> tuple[list[bool], bool]:
    """Score ``output`` against the case's rubric — pure, deterministic.

    Returns ``(covered, all_covered)``: one bool per marker group (True when the output
    contains at least one of the group's markers, case-insensitive) and the overall pass.
    """
    low = output.lower()
    covered = [
        any(marker.lower() in low for marker in group) for group in case.marker_groups
    ]
    return covered, all(covered)


def _tail(text: str, lines: int = TAIL_LINES) -> str:
    """The last ``lines`` lines of ``text`` — the report-friendly slice of a run's output."""
    return "\n".join(text.strip().splitlines()[-lines:])


async def run_golden(
    case: GoldenCase,
    *,
    model: str,
    dest_root: Path,
    sandbox_backend: str = "subprocess",
    timeout: int = 600,
    sink: AgentSink | None = None,
    sandbox: SandboxPort | None = None,
) -> GoldenOutcome:
    """Build the system for ``case``, execute it on the case's input, score the output.

    Builds via the full pipeline (``stop_after="audit"``) into ``dest_root/<key>``, then
    runs the rendered package's own CLI out-of-process (``python -m <name> "<input>"
    --model <id>``) with the package dir + this repo's ``src`` on ``PYTHONPATH`` and a
    minimal ``PATH``. A ``ValueError`` from forge is a refusal; any other build exception
    is ``build_failed``; a non-zero exit is ``run_failed``. Passing ``sink`` / ``sandbox``
    drives the build with stubs (the offline self-test path — execution is then
    monkeypatched, see ``test_golden_offline.py``).
    """
    start = time.monotonic()
    dest = Path(dest_root) / case.key
    try:
        result = await run_forge(
            case.brief,
            model=model,
            dest=dest,
            stop_after="audit",
            sandbox_backend=sandbox_backend,
            sink=sink,
            sandbox=sandbox,
        )
    except ValueError as exc:
        return GoldenOutcome(
            kind="refused", seconds=time.monotonic() - start, error=str(exc)
        )
    except Exception as exc:  # a crash in render/build/audit, not a refusal
        return GoldenOutcome(
            kind="build_failed", seconds=time.monotonic() - start, error=str(exc)
        )

    works = bool(result.audit.works) if result.audit is not None else False

    # Execute the built package the way an operator would — its own CLI, out of process.
    repo_src = Path(__file__).resolve().parents[2] / "src"
    env = dict(os.environ)  # the resolved provider's API key must flow through
    env["PYTHONPATH"] = os.pathsep.join([str(dest), str(repo_src)])
    env["PATH"] = "/usr/bin:/bin"
    argv = [sys.executable, "-m", result.spec.name, case.input_text, "--model", model]
    try:
        proc = subprocess.run(
            argv,
            cwd=dest,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return GoldenOutcome(
            kind="run_failed",
            works=works,
            seconds=time.monotonic() - start,
            error=f"timed out after {timeout}s",
        )

    combined_tail = _tail(proc.stdout + "\n" + proc.stderr)
    if proc.returncode != 0:
        return GoldenOutcome(
            kind="run_failed",
            works=works,
            output_tail=combined_tail,
            seconds=time.monotonic() - start,
            error=f"exit code {proc.returncode}",
        )

    covered, all_covered = score_output(case, proc.stdout)
    kind = "correct" if (works and all_covered) else "wrong_output"
    return GoldenOutcome(
        kind=kind,
        works=works,
        covered=covered,
        output_tail=combined_tail,
        seconds=time.monotonic() - start,
    )


def render_golden(
    results: list[tuple[GoldenCase, list[GoldenOutcome]]], *, model: str
) -> str:
    """Render the golden-corpus results as a markdown report.

    One section per case: kind, works, covered n/m, seconds — plus the output tail when a
    run fell short. The overall line counts fully-``correct`` runs.
    """
    if not results:
        return "_no golden results_\n"

    lines: list[str] = []
    lines.append("# Forge golden-task report (Tier 3)\n")
    lines.append(f"Model: `{model}`\n")

    correct = 0
    total = 0
    for case, outcomes in results:
        lines.append(f"## {case.key}\n")
        lines.append(f"Brief: {case.brief}\n")
        for outcome in outcomes:
            total += 1
            if outcome.kind == "correct":
                correct += 1
            n_covered = sum(outcome.covered)
            n_groups = len(case.marker_groups)
            works = "—" if outcome.works is None else str(outcome.works)
            lines.append(
                f"- kind: **{outcome.kind}** · works: {works} · "
                f"covered {n_covered}/{n_groups} · {outcome.seconds:.1f}s"
            )
            if outcome.error:
                lines.append(f"  - error: {outcome.error}")
            if outcome.kind != "correct" and outcome.output_tail:
                lines.append("\n  ```")
                for tail_line in outcome.output_tail.splitlines():
                    lines.append(f"  {tail_line}")
                lines.append("  ```")
        lines.append("")

    lines.append(f"**Overall: correct {correct}/{total}.**\n")
    return "\n".join(lines)
