"""Command-line adapter for forge — design, build, and audit an agentic system.

    andamentum-forge build  "Manage my reading list."  --model anthropic:claude-haiku-4-5 --out ./out
    andamentum-forge design "Triage support tickets."   --model anthropic:claude-haiku-4-5

``build`` runs the full pipeline (design → render → author code → sandbox audit);
``design`` stops at the validated spec. The model resolves via ``--model`` or
``$ANDAMENTUM_MAIN_LLM_MODEL`` (no hidden default). Code execution during the audit runs
in a Podman container by default; ``--sandbox subprocess`` opts out (no host isolation).
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from andamentum.core import resolve_model_from_args

from .graph import run_forge
from .schemas import AuditRound, ForgeResult


def _self_correction_lines(history: list[AuditRound]) -> list[str]:
    """Human trace of the audit self-correction loop, from the round history.

    Emitted only when the loop actually fired (more than one audit pass) — a clean
    first build produces a single-entry history and no extra output. Pure so it is
    unit-testable without constructing a whole ForgeResult."""
    if len(history) <= 1:
        return []
    rebuilds = len(history) - 1
    plural = "s" if rebuilds != 1 else ""
    lines = [
        f"\nself-correction: {len(history)} audit passes ({rebuilds} rebuild{plural})"
    ]
    for r in history:
        state = "clean" if not r.failing_checks else f"failed — {r.failing_checks}"
        line = f"  pass {r.index}: {state}"
        if r.rebuild_targets:
            line += f"   → re-authored: {', '.join(r.rebuild_targets)}"
        lines.append(line)
    return lines


def _print_summary(result: ForgeResult) -> None:
    spec = result.spec
    heads = sum(1 for n in spec.nodes if n.kind.value == "head")
    print(f"\nsystem:  {spec.name}   (reached: {result.stage_reached})")
    print(f"purpose: {spec.description}")
    print(f"nodes:   {len(spec.nodes)} ({heads} head, {len(spec.nodes) - heads} spine)")
    print(
        f"agents:  {len(spec.agents)}   entities: {len(spec.entities)}   loop-caps: {len(spec.loop_caps)}"
    )
    for note in result.notes:
        print(f"note: {note}")

    if result.report is not None:
        verdict = "ok" if result.report.works else "INCOMPLETE"
        print(f"\nrender check: {verdict}  (score {result.report.score:.2f})")
        for c in result.report.checks:
            print(f"  [{'pass' if c.passed else 'FAIL'}] {c.name}: {c.detail}")

    if result.build is not None:
        b = result.build
        print(
            f"\nbuild: {len(b.filled)} node bodies authored, {len(b.unfillable)} unfillable"
        )
        for u in b.unfillable:
            print(f"  [FAIL] {u.node}: {u.last_error}")

    if result.audit is not None:
        a = result.audit
        rebuilds = max(len(result.audit_history) - 1, 0)
        suffix = ""
        if rebuilds:
            outcome = "converged" if a.works else "settled"
            suffix = (
                f"  ({outcome} after {rebuilds} rebuild{'s' if rebuilds != 1 else ''})"
            )
        print(f"\naudit: {'WORKS' if a.works else 'INCOMPLETE'}{suffix}")
        for c in a.checks:
            print(f"  [{'pass' if c.passed else 'FAIL'}] {c.name}: {c.detail}")
        if a.requirements is not None and a.requirements.gaps:
            print("  requirement gaps:")
            for g in a.requirements.gaps:
                print(f"    - {g}")
        if a.critic is not None and a.critic.issues:
            print("  critic issues:")
            for i in a.critic.issues:
                print(f"    - {i.node}: {i.issue}" if i.node else f"    - {i.issue}")
        for line in _self_correction_lines(result.audit_history):
            print(line)

    if result.rendered_files:
        print(
            f"\nrendered {len(result.rendered_files)} files under {Path(result.rendered_files[0]).parent}"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="andamentum-forge", description="Build agentic systems from a brief."
    )
    parser.add_argument(
        "command",
        choices=["build", "design"],
        help="build = full pipeline; design = spec only",
    )
    parser.add_argument("brief", help="the natural-language brief")
    parser.add_argument(
        "--model",
        default=None,
        help="pydantic-ai model id (or set ANDAMENTUM_MAIN_LLM_MODEL)",
    )
    parser.add_argument(
        "--out", default="out", help="output directory for `build` (default: ./out)"
    )
    parser.add_argument(
        "--stop-after",
        choices=["render", "build", "audit"],
        default="audit",
        help="how far to run the build pipeline (default: audit — the full authoring + verification)",
    )
    parser.add_argument(
        "--sandbox",
        choices=["podman", "subprocess"],
        default="podman",
        help="code-execution backend for the audit (default: podman — host-isolated)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="show a live progress dashboard while the pipeline runs",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="emit the full ForgeResult (incl. self-correction audit_history) as JSON to stdout; disables the live dashboard",
    )
    args = parser.parse_args(argv)

    model = resolve_model_from_args(args.model)
    if args.command == "design":
        dest, stop_after = None, "design"
    else:
        dest, stop_after = Path(args.out), args.stop_after

    reporter = None
    if args.verbose and not args.json:
        # --json owns stdout; a live dashboard would corrupt the JSON stream.
        from rich.console import Console

        from .reporter import RichReporter

        reporter = RichReporter(
            Console(),
            brief=args.brief,
            model=model,
            dest=str(dest) if dest is not None else None,
        )

    err: Exception | None = None
    result = None
    if reporter is not None:
        reporter.start()
    try:
        result = asyncio.run(
            run_forge(
                args.brief,
                model=model,
                dest=dest,
                stop_after=stop_after,
                sandbox_backend=args.sandbox,
                reporter=reporter,
            )
        )
    except ValueError as e:
        err = e
    finally:
        if reporter is not None:
            reporter.stop()

    if err is not None or result is None:
        print(f"Error: {err}", file=sys.stderr)
        return 1

    if args.json:
        print(result.model_dump_json(indent=2))
    else:
        _print_summary(result)
    if result.audit is not None:
        return 0 if result.audit.works else 1
    if result.report is not None:
        return 0 if result.report.works else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
