"""Command line for ``andamentum-epistemic``.

Answer a question or check a claim by weighing evidence for and against it.

Commands:

* ``ask QUESTION`` and ``verify CLAIM`` run the pipeline.
* ``stage NAME`` runs one named stage of it, resuming from the run folder.
* ``inspect FOLDER`` and ``confidence FOLDER`` read a finished run.
* ``preflight`` checks the model and the search address before a run.
* ``run AGENT`` runs one agent and ``agents`` lists them; these two and
  ``stage`` are for development.

A run keeps its files only in the folder named by ``--out``. Without
``--out``, ``ask`` and ``verify`` work in a temporary folder that is removed
afterwards and print the result.

Exit codes: 0 success; 1 the run finished and the result is negative (a check
failed); 2 usage error; 3 the input could not be read (no run in the folder);
4 the run failed (model or network error, or a crash).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import traceback
from collections.abc import Sequence
from pathlib import Path

_EXIT_CODES = """\
exit codes:
  0  success
  1  the run finished and the result is negative (a check failed)
  2  usage error: a wrong or missing argument
  3  the input could not be read (no run in the folder)
  4  the run failed: model or network error, or a crash
"""


class _UsageError(Exception):
    """A wrong or missing argument that argparse cannot see (exit 2)."""


class _InputError(Exception):
    """The input could not be read (exit 3)."""


def _add_model(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--model",
        required=True,
        help="Model that does the reasoning, for example anthropic:claude-haiku-4-5 or ollama:gemma4:31b",
    )


def _add_embedding_model(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--embedding-model",
        required=True,
        help="Ollama model that embeds text for routing and passage extraction, for example embeddinggemma-2:latest",
    )


def _add_search(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--searxng-url",
        default=None,
        help="Address of a SearXNG instance for web search, for example http://127.0.0.1:4070",
    )
    parser.add_argument(
        "--browser",
        action="store_true",
        help=(
            "Search the web through an isolated Chrome in a podman container "
            "(started and removed per run) instead of SearXNG. One of "
            "--searxng-url and --browser is required"
        ),
    )


def _add_run_options(parser: argparse.ArgumentParser, *, what: str) -> None:
    _add_model(parser)
    _add_embedding_model(parser)
    _add_search(parser)
    parser.add_argument(
        "-o",
        "--out",
        default=None,
        metavar="DIR",
        help=(
            f"Folder to keep the {what}'s database (epistemic.db) and HTML report "
            "(report.html) in. Without it the run uses a temporary folder that is "
            "removed afterwards"
        ),
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the result as JSON on stdout (progress goes to stderr)",
    )
    parser.add_argument(
        "--provider",
        default="all",
        choices=["all", "web_search"],
        help=(
            "Which evidence sources the run may use: 'all' (the scholarly and "
            "domain sources plus web search) or 'web_search' (web search only). "
            "Default: all"
        ),
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Print progress messages"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="andamentum-epistemic",
        description=(
            "Answer a question or check a claim by weighing evidence for and "
            "against it. Experimental: output and flags may change."
        ),
        epilog=_EXIT_CODES,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    from andamentum import __version__ as _ver

    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {_ver}"
    )
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")

    ask_parser = sub.add_parser(
        "ask",
        help="Answer a research question and show the verdict, claims and evidence",
        description=(
            "Answer a research question by weighing evidence for and against it. "
            "Prints the verdict, the claims behind it and their evidence; with "
            "--out the database and an HTML report are kept in that folder."
        ),
        epilog=_EXIT_CODES,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ask_parser.add_argument("question", help="The research question to investigate")
    _add_run_options(ask_parser, what="run")

    verify_parser = sub.add_parser(
        "verify",
        help="Check one claim: seed exactly one claim from the text and weigh the evidence",
        description=(
            "Check one claim: the text is taken as a single claim (no "
            "decomposition) and the evidence for and against it is weighed."
        ),
        epilog=_EXIT_CODES,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    verify_parser.add_argument("claim", help="The claim to check")
    _add_run_options(verify_parser, what="run")

    run_parser = sub.add_parser(
        "run",
        help="(for development) Run one agent and print its output as JSON",
        epilog=_EXIT_CODES,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    run_parser.add_argument(
        "agent", help="Agent name (list them with the agents command)"
    )
    run_parser.add_argument(
        "kwargs",
        nargs="*",
        help="Inputs for the agent as key=value pairs, for example question='What is X?'",
    )
    _add_model(run_parser)
    run_parser.add_argument(
        "--verbose", action="store_true", help="Print progress messages"
    )

    sub.add_parser(
        "agents",
        help="(for development) List the agents and what each returns",
        epilog=_EXIT_CODES,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    pf_parser = sub.add_parser(
        "preflight",
        help="Check the model, the search address and the providers before a run",
        epilog=_EXIT_CODES,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _add_model(pf_parser)
    pf_parser.add_argument(
        "--providers",
        default=None,
        help=(
            "Also check the evidence sources of this set: 'biomedical' (the "
            "scholarly and biomedical sources, without web search). Leave out "
            "to check only the model and the web search address"
        ),
    )
    pf_parser.add_argument(
        "--searxng-url",
        default=None,
        help="Address of a SearXNG instance to check for web search",
    )
    pf_parser.add_argument(
        "--browser",
        action="store_true",
        help="Check podman (for the isolated Chrome) instead of SearXNG",
    )
    pf_parser.add_argument(
        "--verbose", action="store_true", help="Print extra details"
    )

    from .graph.stages import stage_names

    stage_parser = sub.add_parser(
        "stage",
        help="(for development) Run one stage of the pipeline (preplanning, scrutiny_and_investigation, ...)",
        epilog=_EXIT_CODES,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    stage_parser.add_argument("name", choices=stage_names(), help="Stage to run")
    stage_parser.add_argument(
        "-o",
        "--out",
        required=True,
        metavar="DIR",
        help=(
            "Run folder: the database (epistemic.db) and the stage's files "
            "(run.jsonl, diff.json, timing.txt) go here. A folder with a database "
            "from an earlier stage is resumed"
        ),
    )
    stage_parser.add_argument(
        "--question",
        default=None,
        help="Research question (needed when the folder has no database yet)",
    )
    _add_model(stage_parser)
    _add_embedding_model(stage_parser)
    _add_search(stage_parser)
    stage_parser.add_argument(
        "--verify",
        action="store_true",
        help="Treat --question as a single claim and skip decomposition (default: research question)",
    )

    inspect_parser = sub.add_parser(
        "inspect",
        help="Print the objective, verdict, claims and evidence of a finished run",
        epilog=_EXIT_CODES,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    inspect_parser.add_argument(
        "folder", help="Run folder (the --out folder of an earlier run)"
    )

    conf_parser = sub.add_parser(
        "confidence",
        help="Compute the posterior confidence of a finished run",
        epilog=_EXIT_CODES,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    conf_parser.add_argument(
        "folder", help="Run folder (the --out folder of an earlier run)"
    )
    conf_parser.add_argument(
        "--objective",
        default=None,
        help="Objective id (found automatically when the run has only one)",
    )
    conf_parser.add_argument(
        "--verbose", action="store_true", help="Show per-claim detail"
    )

    return parser


def _parse_kwargs(raw: list[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for item in raw:
        if "=" not in item:
            raise _UsageError(f"invalid argument '{item}': expected key=value")
        key, _, value = item.partition("=")
        result[key] = value
    return result


def _require_search(args: argparse.Namespace) -> None:
    if not (args.searxng_url or args.browser):
        raise _UsageError(
            "web search needs an address: pass --searxng-url (a SearXNG "
            "instance) or --browser (the isolated Chrome)"
        )


def _existing_database(folder: str) -> Path:
    from .databases import database_path

    db = database_path(folder)
    if not db.is_file():
        raise _InputError(f"no run in {folder}: {db.name} not found")
    return db


def _list_agents() -> None:
    from .agents import AGENT_REGISTRY

    if not AGENT_REGISTRY:
        print("No agents registered.")
        return

    for name, defn in sorted(AGENT_REGISTRY.items()):
        model_name = defn.output_model.__name__ if defn.output_model else "None"
        print(f"  {name:<40s}  output={model_name}")


async def _preflight(args: argparse.Namespace) -> int:
    from .preflight import preflight

    providers = None
    if args.providers == "biomedical":
        from .providers import get_biomedical_providers

        providers = get_biomedical_providers()

    result = await preflight(
        model=args.model,
        providers=providers,
        verbose=args.verbose,
        searxng_url=args.searxng_url,
        browser=args.browser,
    )

    for check in result.checks:
        icon = {"pass": "OK", "fail": "FAIL", "skip": "SKIP"}[check.status]
        line = f"  [{icon:>4s}] {check.name:<25s} {check.message}"
        if args.verbose:
            line += f"  ({check.elapsed_ms:.0f}ms)"
        print(line)

    if result.ok:
        print(f"\nAll {len(result.checks)} checks passed.")
        return 0
    failed = [c for c in result.checks if c.status == "fail"]
    print(f"\n{len(failed)} of {len(result.checks)} checks FAILED.")
    return 1


async def _confidence(args: argparse.Namespace) -> int:
    from .confidence import compute_posterior
    from .repository import EpistemicRepository

    db = _existing_database(args.folder)
    repo = await EpistemicRepository.for_database(db)
    try:
        objective_id = args.objective
        if objective_id is None:
            objectives = await repo.query("objective")
            if not objectives:
                raise _InputError(f"no objective in {args.folder}")
            objective_id = objectives[0].entity_id

        posterior = await compute_posterior(repo, objective_id)
        if posterior is not None:
            print(f"Posterior confidence: {posterior.posterior:.2%}")
            print(
                f"  {posterior.supporting_count} supporting, {posterior.contradicting_count} contradicting"
            )
            print(f"  {posterior.explanation}")
        else:
            objective = await repo.get("objective", objective_id)
            qt = getattr(objective, "question_type", None) or "unclassified"
            print(f"Posterior not computed (question type: {qt})")

        if args.verbose:
            print()
            print(f"Objective: {objective_id}")
    finally:
        await repo.close()
    return 0


async def _ask(args: argparse.Namespace) -> int:
    """Run ``ask`` (research mode) or ``verify`` (verify mode)."""
    from .cli_handlers import handle_ask

    _require_search(args)

    # The verify command uses ``args.claim``; ask uses ``args.question``.
    text = getattr(args, "claim", None) or args.question
    mode = "verify" if args.command == "verify" else "research"

    out = Path(args.out) if args.out else None
    database = report = None
    if out is not None:
        from .databases import database_path

        out.mkdir(parents=True, exist_ok=True)
        database = database_path(out)
        report = out / "report.html"

    if args.json:
        # The result is the only thing on stdout; progress goes to stderr.
        from . import cli_handlers

        cli_handlers.console.file = sys.stderr

    result = await handle_ask(
        text,
        model=args.model,
        embedding_model=args.embedding_model,
        database=database,
        report=report,
        verbose=args.verbose,
        mode=mode,
        provider=args.provider,
        searxng_url=args.searxng_url,
        browser=args.browser,
    )

    if args.json:
        print(result.model_dump_json(indent=2))
    if not result.success:
        print("error: no operation of the run completed", file=sys.stderr)
        return 4
    return 0


async def _run(args: argparse.Namespace) -> int:
    from .runner import DefaultAgentRunner

    kwargs = _parse_kwargs(args.kwargs)

    if args.verbose:
        print(f"Agent: {args.agent}")
        print(f"Model: {args.model}  |  Args: {kwargs}")

    runner = DefaultAgentRunner(model=args.model)
    result = await runner.run(args.agent, **kwargs)

    if hasattr(result, "model_dump"):
        print(json.dumps(result.model_dump(), indent=2, default=str))
    else:
        print(result)
    return 0


async def _stage(args: argparse.Namespace) -> int:
    """Run a single named pipeline stage. The database is the checkpoint:
    chain stages by running again with the same ``--out`` folder."""
    from .databases import database_path
    from .graph import run_epistemic_graph
    from .graph.stages import get_stage

    _require_search(args)
    stage = get_stage(args.name)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # Stage runners read the mode from the saved database's Objective on
    # resume; on a fresh run --verify mints the Objective in verify mode.
    mode = "verify" if args.verify else "research"
    result = await run_epistemic_graph(
        question=args.question or "(resumed)",
        database=database_path(out),
        model=args.model,
        embedding_model=args.embedding_model,
        mode=mode,
        start_at=stage.entry,
        stop_after=stage.exit_after,
        output_dir=out,
        searxng_url=args.searxng_url,
        browser=args.browser,
    )

    print(f"Stage {stage.name!r} complete.")
    print(f"  status:       {result.status}")
    print(f"  objective_id: {result.objective_id}")
    print(f"  files:        {out}/")
    return 0


async def _inspect(args: argparse.Namespace) -> int:
    """Print a structured snapshot of a saved DB. No LLM, no graph
    work — just read the entities and summarise."""
    from .databases import open_store
    from .repository import EpistemicRepository

    db = _existing_database(args.folder)
    store = await open_store(db)
    try:
        repo = EpistemicRepository(store)

        objs = await repo.query("objective")
        claims = await repo.query("claim")
        evidence = await repo.query("evidence")

        if not objs:
            print(f"{args.folder}: no objectives. Empty database.")
            return 0

        for obj in objs:
            print(f"Objective {obj.entity_id}")
            print(f"  question_type: {obj.question_type}")
            print(f"  description:   {(obj.description or '')[:120]}")
            decomp = obj.decomposition
            if decomp is not None:
                print(
                    f"  decomposition: {len(decomp.sub_investigations)} sub-investigations "
                    f"(rule={decomp.combination_rule})"
                )
            else:
                print("  decomposition: <none>")

            # The run's verdict lives on the frozen snapshot, not on the
            # decomposition — so verify-mode runs (which have no
            # decomposition at all) print one too.
            snapshot = (
                await repo.get("snapshot", obj.snapshot_id) if obj.snapshot_id else None
            )
            verdict = getattr(snapshot, "verdict", None) if snapshot else None
            if verdict is not None:
                post = (
                    f"{verdict.posterior:.3f}"
                    if verdict.posterior is not None
                    else "n/a"
                )
                print(
                    f"  verdict:       {verdict.direction} posterior={post} "
                    f"n_no_verdict={verdict.provenance.n_no_verdict}"
                )
                # Judgment reliability, printed only when a measurement
                # actually raised something — so a run that measured nothing
                # prints exactly what it printed before this line existed.
                # ``n_doubt_unmeasured_claims`` travels with its companion so
                # a ``0`` in the first can never read as "all sure".
                prov = verdict.provenance
                if (
                    prov.n_high_doubt_claims
                    or prov.n_doubt_unmeasured_claims
                    or prov.judgment_doubt_escalated
                ):
                    print(
                        f"  reliability:   n_high_doubt_claims="
                        f"{prov.n_high_doubt_claims} "
                        f"n_doubt_unmeasured_claims="
                        f"{prov.n_doubt_unmeasured_claims} "
                        f"judgment_doubt_escalated="
                        f"{prov.judgment_doubt_escalated}"
                    )
            else:
                print("  verdict:       <none>")

        print(f"\n{len(claims)} claims:")
        for c in sorted(claims, key=lambda x: x.sub_investigation_id or ""):
            stage = c.stage.value if hasattr(c.stage, "value") else c.stage
            ia = c.integrated_assessment
            ic = (
                f"{c.integrated_confidence:.3f}"
                if c.integrated_confidence is not None
                else "—"
            )
            print(
                f"  [{c.sub_investigation_id or '-'}] stage={stage} "
                f"verdict={ia} conf={ic} "
                f"cycle_capped={c.cycle_capped} abandoned={c.abandoned} "
                f"ev={c.evidence_count}"
            )
            print(f"    {(c.statement or '')[:140]}")

        real = sum(
            1
            for e in evidence
            if (e.extracted_content or "") and len(e.extracted_content) > 200
        )
        by_provider: dict[str, int] = {}
        for e in evidence:
            by_provider[e.source_type] = by_provider.get(e.source_type, 0) + 1
        print(f"\n{len(evidence)} evidence items ({real} with content >200 chars)")
        for prov, n in sorted(by_provider.items()):
            print(f"  {prov}: {n}")
    finally:
        await store.close()
    return 0


def _experimental_banner() -> str:
    """The experimental notice; coloured only when stderr is a terminal."""
    label = "EXPERIMENTAL:"
    if sys.stderr.isatty():
        label = f"\033[33m{label}\033[0m"
    return (
        f"{label} andamentum.epistemic is under active development. Output "
        "shape, agent names and flags may change without notice.\n"
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help(sys.stderr)
        return 2

    # The notice goes to stderr for the commands that run the pipeline, after
    # the arguments are accepted and before any work.
    if args.command in ("ask", "verify", "stage"):
        try:
            _require_search(args)
        except _UsageError as e:
            print(f"error: {e}", file=sys.stderr)
            return 2
        print(_experimental_banner(), file=sys.stderr)

    if args.command == "agents":
        _list_agents()
        return 0

    handlers = {
        "ask": _ask,
        "verify": _ask,
        "run": _run,
        "stage": _stage,
        "inspect": _inspect,
        "confidence": _confidence,
        "preflight": _preflight,
    }
    try:
        return asyncio.run(handlers[args.command](args))
    except KeyboardInterrupt:
        print("\nCancelled.", file=sys.stderr)
        return 4
    except _UsageError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    except _InputError as e:
        print(f"error: {e}", file=sys.stderr)
        return 3
    except Exception as e:
        from .graph import DegenerateQuestionError

        print(f"error: {e}", file=sys.stderr)
        if getattr(args, "verbose", False):
            traceback.print_exc()
        return 2 if isinstance(e, DegenerateQuestionError) else 4


if __name__ == "__main__":
    sys.exit(main())
