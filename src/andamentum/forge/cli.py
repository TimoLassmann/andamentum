"""Command line for ``andamentum-forge``: build a small function package from a brief.

Forge builds small runnable function packages: a plain-language brief becomes a
validated design, a rendered Python package, code written by a model for each
step, and a test run of the package in a sandbox. It refuses briefs that ask
for an app, an agent or a service (anything that keeps running and is driven
from outside) and says how to reshape them.

    andamentum-forge build  "Summarise a reading list." --model anthropic:claude-haiku-4-5 --out ./out
    andamentum-forge design "Triage support tickets."   --model anthropic:claude-haiku-4-5

``build`` runs the whole pipeline (design, render, write the code, sandbox test)
into the ``--out`` folder; ``design`` stops at the validated design and writes
nothing. The sandbox is a Podman container by default; ``--sandbox subprocess``
runs the generated code on this machine without isolation.

Exit codes: 0 success; 1 the pipeline ran and a check failed (the rendered package
is incomplete, a step could not be written, the sandbox test failed); 2 usage
error, including a brief forge refuses; 3 the brief could not be read; 4 the run
failed (model or sandbox error, or a crash).
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from collections.abc import Sequence
from pathlib import Path

from .schemas import ForgeResult

_EXIT_CODES = """\
exit codes:
  0  success
  1  the pipeline ran and a check failed (incomplete package, an unwritten step, a failed test)
  2  usage error, including a brief forge refuses (not a small function)
  3  the brief could not be read
  4  the run failed: model or sandbox error, or a crash
"""


def _epilog(*examples: str) -> str:
    """Examples first, then the exit codes."""
    head = "example:" if len(examples) == 1 else "examples:"
    lines = "\n".join(f"  {e}" for e in examples)
    return f"{head}\n{lines}\n\n{_EXIT_CODES}"


_M = "--model anthropic:claude-haiku-4-5"
_BRIEF = '"Count the words in a text and return the ten most common."'
_BUILD = f"andamentum-forge build {_BRIEF} \\\n      {_M} -o wordcount"
_BUILD_LOCAL = (
    f"andamentum-forge build {_BRIEF} \\\n"
    f"      {_M} -o wordcount --stop-after render --verbose"
)
_DESIGN = f"andamentum-forge design {_BRIEF} \\\n      {_M} --json"

_ABOUT = (
    "Build a small runnable function package from a plain-language brief. "
    "Forge refuses apps, agents and services, which keep running and are "
    "driven from outside."
)
_ABOUT_DESIGN = (
    "Validate a design for a function package from a plain-language brief; "
    "writes nothing. Forge refuses apps, agents and services, which keep "
    "running and are driven from outside."
)


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
        print(f"\naudit: {'WORKS' if a.works else 'INCOMPLETE'}")
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

    if result.rendered_files:
        print(
            f"\nrendered {len(result.rendered_files)} files under {Path(result.rendered_files[0]).parent}"
        )


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "brief",
        help="What the function should do, in plain language ('-' reads it from standard input)",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Model that designs and writes the code, for example anthropic:claude-haiku-4-5",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show a live progress display while the pipeline runs (not with --json)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the full result as JSON on stdout instead of the summary",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="andamentum-forge",
        description=_ABOUT,
        epilog=_epilog(_BUILD, _DESIGN),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    from andamentum import __version__ as _ver

    parser.add_argument("--version", action="version", version=f"%(prog)s {_ver}")
    sub = parser.add_subparsers(dest="command", metavar="COMMAND", required=True)

    build = sub.add_parser(
        "build",
        help="Design, render, write the code and test it in a sandbox",
        description=_ABOUT,
        epilog=_epilog(_BUILD, _BUILD_LOCAL),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _add_common(build)
    build.add_argument(
        "-o",
        "--out",
        required=True,
        metavar="DIR",
        help="Folder the rendered package is written into",
    )
    build.add_argument(
        "--stop-after",
        choices=["render", "build", "audit"],
        default="audit",
        help="How far to run: render the package, write its code, or also test it (default: audit)",
    )
    build.add_argument(
        "--sandbox",
        choices=["podman", "subprocess"],
        default="podman",
        help="Where the generated code is tested: a Podman container, or a plain process on this machine without isolation (default: podman)",
    )

    design = sub.add_parser(
        "design",
        help="Produce the validated design only; writes no files",
        description=_ABOUT_DESIGN,
        epilog=_epilog(_DESIGN),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _add_common(design)
    return parser


def _exit_code(result: ForgeResult) -> int:
    """1 when a stage that ran reported a problem, else 0.

    Design-only runs run no check and return 0. A run that rendered but stopped
    before the audit is judged on the checks it did run.
    """
    if result.report is not None and not result.report.works:
        return 1
    if result.build is not None and result.build.unfillable:
        return 1
    if result.audit is not None and not result.audit.works:
        return 1
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    brief = args.brief
    if brief == "-":
        brief = sys.stdin.read()
    if not brief.strip():
        print(
            "error: the brief is empty; say what the function should do",
            file=sys.stderr,
        )
        return 2
    from andamentum.core.secrets import load_secrets

    load_secrets()  # secrets only (API keys); configuration comes from flags

    if args.command == "design":
        dest, stop_after, sandbox = None, "design", "podman"
    else:
        dest, stop_after, sandbox = Path(args.out), args.stop_after, args.sandbox

    from .errors import BriefRefusedError
    from .graph import run_forge

    reporter = None
    if args.verbose and not args.json:
        # --json owns stdout; a live dashboard would corrupt the JSON stream.
        from rich.console import Console

        from .reporter import RichReporter

        reporter = RichReporter(
            Console(),
            brief=brief,
            model=args.model,
            dest=str(dest) if dest is not None else None,
        )

    code = 0
    result = None
    if reporter is not None:
        reporter.start()
    try:
        result = asyncio.run(
            run_forge(
                brief,
                model=args.model,
                out=dest,
                stop_after=stop_after,
                sandbox_backend=sandbox,
                reporter=reporter,
            )
        )
    except BriefRefusedError as e:
        print(f"error: {e}", file=sys.stderr)
        code = 2
    except KeyboardInterrupt:
        print("error: cancelled", file=sys.stderr)
        code = 4
    except Exception as e:
        print(f"error: {e}", file=sys.stderr)
        code = 4
    finally:
        if reporter is not None:
            reporter.stop()

    if result is None:
        return code

    if args.json:
        print(result.model_dump_json(indent=2))
    else:
        _print_summary(result)
    return _exit_code(result)


if __name__ == "__main__":
    raise SystemExit(main())
