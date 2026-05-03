"""CLI entry point for andamentum-epistemic.

Usage::

    andamentum-epistemic run <agent_name> [key=value ...] [--model MODEL] [--verbose]
    andamentum-epistemic agents
"""

import argparse
import asyncio
import json
import sys
import traceback


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="andamentum-epistemic",
        description="Formal epistemology for AI research — evidence-based claims with traceability.",
    )
    sub = parser.add_subparsers(dest="command")

    # Subcommand: ask (primary interface)
    ask_parser = sub.add_parser(
        "ask", help="Ask a research question and get validated findings"
    )
    ask_parser.add_argument("question", help="Research question to investigate")
    ask_parser.add_argument(
        "--model",
        default=None,
        help="LLM model (e.g. bedrock:claude-haiku-4-5, openai:gpt-4o) — or set $ANDAMENTUM_MAIN_LLM_MODEL",
    )
    ask_parser.add_argument(
        "--embedding-model",
        default=None,
        help=(
            "Ollama embedding model for semantic provider routing and passage "
            "extraction (default: embeddinggemma:latest, or set "
            "$ANDAMENTUM_EMBEDDING_MODEL)"
        ),
    )
    ask_parser.add_argument(
        "--keep", action="store_true", help="Keep project database after completion"
    )
    ask_parser.add_argument(
        "--name", default=None, help="Project name (auto-generated if not specified)"
    )
    ask_parser.add_argument(
        "--provider",
        default="all",
        choices=["all", "web_search"],
        help="Evidence provider (default: all)",
    )
    ask_parser.add_argument(
        "--quick",
        action="store_true",
        help="Skip preplanning (clarification + conceptual analysis)",
    )
    ask_parser.add_argument(
        "--decompose",
        action="store_true",
        help=(
            "Run top-down decomposition (multi-seed-claim mode): the "
            "system splits the question into N load-bearing sub-claims, "
            "verifies each as a Claim on one Objective, and combines "
            "the per-claim verdicts via the decomposition's "
            "combination_rule (AND / OR / WEIGHTED_AND / UNION). Without "
            "this flag, the system runs the open-research path "
            "(claims emerge from evidence via ProposeClaims)."
        ),
    )
    ask_parser.add_argument(
        "--trace",
        default="timeline",
        choices=["timeline", "flow", "claims", "all", "none"],
        help="Trace visualization mode",
    )
    ask_parser.add_argument("--output", default=None, help="Path to save HTML report")
    ask_parser.add_argument(
        "--verbose", action="store_true", help="Print progress messages"
    )

    # Subcommand: run
    run_parser = sub.add_parser("run", help="Run an epistemic agent")
    run_parser.add_argument(
        "agent", help="Agent name (e.g. epistemic_clarify_question)"
    )
    run_parser.add_argument(
        "kwargs",
        nargs="*",
        help="Key=value pairs for the agent (e.g. question='What is X?')",
    )
    run_parser.add_argument(
        "--model",
        default=None,
        help="LLM model (e.g. bedrock:claude-haiku-4-5, openai:gpt-4o) — or set $ANDAMENTUM_MAIN_LLM_MODEL",
    )
    run_parser.add_argument(
        "--verbose", action="store_true", help="Print progress messages"
    )

    # Subcommand: agents
    sub.add_parser("agents", help="List registered agents and their output models")

    # Subcommand: preflight
    pf_parser = sub.add_parser(
        "preflight", help="Check LLM, SearXNG, and provider connectivity before a run"
    )
    pf_parser.add_argument(
        "--model",
        default=None,
        help="LLM model (e.g. bedrock:claude-haiku-4-5, openai:gpt-4o) — or set $ANDAMENTUM_MAIN_LLM_MODEL",
    )
    pf_parser.add_argument(
        "--providers", default=None, help="Provider set: 'biomedical' or omit for none"
    )
    pf_parser.add_argument("--verbose", action="store_true", help="Print extra details")

    # Subcommand: stage — run a single named pipeline stage
    from .graph.stages import stage_names

    stage_parser = sub.add_parser(
        "stage",
        help="Run a single pipeline stage (preplanning, scrutiny_and_investigation, ...)",
    )
    stage_parser.add_argument("name", choices=stage_names(), help="Stage to run")
    stage_parser.add_argument(
        "--question",
        default=None,
        help="Research question (required if --from-db has no Objective)",
    )
    stage_parser.add_argument(
        "--from-db",
        default=None,
        help="Resume from a saved database (read by db_dir lookup)",
    )
    stage_parser.add_argument(
        "--db",
        default="stage_run",
        help="Database name (default: 'stage_run')",
    )
    stage_parser.add_argument(
        "--db-dir",
        default=None,
        help="Database directory (default: ~/.local/share/document-store/)",
    )
    stage_parser.add_argument(
        "--model",
        default=None,
        help="LLM model — or set $ANDAMENTUM_MAIN_LLM_MODEL",
    )
    stage_parser.add_argument(
        "--embedding-model",
        default=None,
        help="Ollama embedding model (default: embeddinggemma:latest)",
    )
    stage_parser.add_argument(
        "--decompose",
        action="store_true",
        help="Multi-seed-claim decomposition mode",
    )
    stage_parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for run.jsonl / diff.json / timing.txt artifacts",
    )

    # Subcommand: inspect — print structured state of a saved DB
    inspect_parser = sub.add_parser(
        "inspect",
        help="Print structured state of a saved epistemic DB",
    )
    inspect_parser.add_argument(
        "db",
        help="Database name (looked up under --db-dir or default location)",
    )
    inspect_parser.add_argument(
        "--db-dir",
        default=None,
        help="Database directory (default: ~/.local/share/document-store/)",
    )

    # Subcommand: confidence
    conf_parser = sub.add_parser(
        "confidence", help="Compute post-hoc confidence for a completed epistemic run"
    )
    conf_parser.add_argument("--db", required=True, help="Database name")
    conf_parser.add_argument(
        "--db-dir",
        default=None,
        help="Custom database directory (default: ~/.config/andamentum/databases/)",
    )
    conf_parser.add_argument(
        "--objective",
        default=None,
        help="Objective ID (auto-detected if only one exists)",
    )
    conf_parser.add_argument(
        "--verbose", action="store_true", help="Show per-claim detail"
    )

    return parser


def _resolve_model(args: argparse.Namespace) -> str:
    from andamentum.core.models import resolve_model_from_args

    return resolve_model_from_args(args.model)


def _parse_kwargs(raw: list[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for item in raw:
        if "=" not in item:
            print(
                f"Error: invalid argument '{item}' — expected key=value format",
                file=sys.stderr,
            )
            sys.exit(1)
        key, _, value = item.partition("=")
        result[key] = value
    return result


def _list_agents() -> None:
    from .agents import AGENT_REGISTRY

    if not AGENT_REGISTRY:
        print("No agents registered.")
        return

    for name, defn in sorted(AGENT_REGISTRY.items()):
        model_name = defn.output_model.__name__ if defn.output_model else "None"
        print(f"  {name:<40s}  output={model_name}")


async def _preflight(args: argparse.Namespace) -> None:
    from .preflight import preflight

    model = _resolve_model(args)
    providers = None
    if args.providers == "biomedical":
        from .providers import get_biomedical_providers

        providers = get_biomedical_providers()

    result = await preflight(model=model, providers=providers, verbose=args.verbose)

    for check in result.checks:
        icon = {"pass": "OK", "fail": "FAIL", "skip": "SKIP"}[check.status]
        line = f"  [{icon:>4s}] {check.name:<25s} {check.message}"
        if args.verbose:
            line += f"  ({check.elapsed_ms:.0f}ms)"
        print(line)

    if result.ok:
        print(f"\nAll {len(result.checks)} checks passed.")
    else:
        failed = [c for c in result.checks if c.status == "fail"]
        print(f"\n{len(failed)} of {len(result.checks)} checks FAILED.")
        sys.exit(1)


async def _confidence(args: argparse.Namespace) -> None:
    from pathlib import Path

    from .confidence import compute_posterior
    from .repository import EpistemicRepository

    db_dir = Path(args.db_dir) if args.db_dir else None
    repo = await EpistemicRepository.for_database(args.db, db_dir=db_dir)

    objective_id = args.objective
    if objective_id is None:
        objectives = await repo.query("objective")
        if not objectives:
            print("Error: no objectives found in database.", file=sys.stderr)
            sys.exit(1)
        objective_id = objectives[0].entity_id

    # Posterior confidence (evidential direction)
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


async def _ask(args: argparse.Namespace) -> None:
    try:
        from .cli_handlers import handle_ask
    except ImportError:
        print(
            "Error: pydantic-ai and rich are required. Install with: pip install andamentum",
            file=sys.stderr,
        )
        sys.exit(1)

    from andamentum.core.models import resolve_embedding_model_from_args

    model = _resolve_model(args)
    embedding_model = resolve_embedding_model_from_args(args.embedding_model)
    result = await handle_ask(
        question=args.question,
        name=args.name,
        model=model,
        embedding_model=embedding_model,
        keep=args.keep,
        verbose=args.verbose,
        trace=args.trace,
        force_quick=args.quick,
        decompose=args.decompose,
        provider=args.provider,
        output_path=args.output,
    )

    if not result.success and result.error:
        sys.exit(1)


async def _run(args: argparse.Namespace) -> None:
    try:
        from .runner import DefaultAgentRunner
    except ImportError:
        print(
            "Error: pydantic-ai is required. Install with: pip install andamentum",
            file=sys.stderr,
        )
        sys.exit(1)

    model = _resolve_model(args)
    kwargs = _parse_kwargs(args.kwargs)

    if args.verbose:
        print(f"Agent: {args.agent}")
        print(f"Model: {model}  |  Args: {kwargs}")

    runner = DefaultAgentRunner(model=model)
    result = await runner.run(args.agent, **kwargs)

    if hasattr(result, "model_dump"):
        print(json.dumps(result.model_dump(), indent=2, default=str))
    else:
        print(result)


async def _stage(args: argparse.Namespace) -> None:
    """Run a single named pipeline stage. The DB is the checkpoint —
    chain stages by re-running with the same --db on later
    invocations."""
    from pathlib import Path

    from .graph import run_epistemic_graph
    from .graph.stages import get_stage

    stage = get_stage(args.name)
    model = _resolve_model(args)
    output_dir = Path(args.output_dir) if args.output_dir else None

    db_name = args.from_db or args.db
    result = await run_epistemic_graph(
        question=args.question or "(resumed)",
        database_name=db_name,
        db_dir=args.db_dir,
        model=model,
        embedding_model=args.embedding_model,
        decompose=args.decompose,
        start_at=stage.entry,
        stop_after=stage.exit_after,
        output_dir=output_dir,
    )

    print(f"Stage {stage.name!r} complete.")
    print(f"  status:       {result.status}")
    print(f"  objective_id: {result.objective_id}")
    if output_dir is not None:
        print(f"  artifacts:    {output_dir}/")


async def _inspect(args: argparse.Namespace) -> None:
    """Print a structured snapshot of a saved DB. No LLM, no graph
    work — just read the entities and summarise."""
    from andamentum.document_store import DocumentStore

    from .repository import EpistemicRepository

    db_dir = args.db_dir
    store = DocumentStore.for_database(args.db, db_dir=db_dir)
    await store.initialize()
    repo = EpistemicRepository(store)

    objs = await repo.query("objective")
    claims = await repo.query("claim")
    evidence = await repo.query("evidence")

    if not objs:
        print(f"DB {args.db!r}: no objectives. Empty database.")
        return

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
            cv = decomp.combined_verdict
            if cv is not None:
                post = (
                    f"{cv.posterior:.3f}" if cv.posterior is not None else "n/a"
                )
                print(
                    f"  combined:      verdict={cv.verdict} posterior={post} "
                    f"n_no_verdict={cv.n_no_verdict}"
                )
            else:
                print("  combined:      <none>")
        else:
            print("  decomposition: <none>")

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


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.command == "agents":
        _list_agents()
        return

    if args.command == "ask":
        try:
            asyncio.run(_ask(args))
        except KeyboardInterrupt:
            print("\nCancelled.")
            sys.exit(0)
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            if "--verbose" in sys.argv:
                traceback.print_exc()
            sys.exit(1)
        return

    if args.command == "preflight":
        try:
            asyncio.run(_preflight(args))
        except KeyboardInterrupt:
            print("\nCancelled.")
            sys.exit(0)
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            if "--verbose" in sys.argv:
                traceback.print_exc()
            sys.exit(1)
        return

    if args.command == "confidence":
        try:
            asyncio.run(_confidence(args))
        except KeyboardInterrupt:
            print("\nCancelled.")
            sys.exit(0)
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            if "--verbose" in sys.argv:
                traceback.print_exc()
            sys.exit(1)
        return

    if args.command == "stage":
        try:
            asyncio.run(_stage(args))
        except KeyboardInterrupt:
            print("\nCancelled.")
            sys.exit(0)
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            if "--verbose" in sys.argv:
                traceback.print_exc()
            sys.exit(1)
        return

    if args.command == "inspect":
        try:
            asyncio.run(_inspect(args))
        except KeyboardInterrupt:
            print("\nCancelled.")
            sys.exit(0)
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
        return

    if args.command == "run":
        try:
            asyncio.run(_run(args))
        except KeyboardInterrupt:
            print("\nQuery cancelled.")
            sys.exit(0)
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            if "--verbose" in sys.argv:
                traceback.print_exc()
            sys.exit(1)
        return

    parser.print_help()
    sys.exit(1)


if __name__ == "__main__":
    main()
