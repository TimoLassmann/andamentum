"""CLI entry point for andamentum-research.

Usage::

    andamentum-research "What is quantum computing?" [--model MODEL] [--max-iterations N] [--verbose]
    andamentum-research agents
"""

import argparse
import asyncio
import sys
import traceback
from typing import Any


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="andamentum-research",
        description="Deep research with iterative search, verification, and synthesis.",
    )
    from andamentum import __version__ as _ver

    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s (andamentum {_ver})",
    )
    parser.add_argument("query", help="Research question")
    parser.add_argument(
        "--model",
        default=None,
        help="LLM model (e.g. anthropic:claude-haiku-4-5, openai:gpt-4o) — or set $ANDAMENTUM_MAIN_LLM_MODEL",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=3,
        help="Max research iterations (default: 3)",
    )
    parser.add_argument(
        "--searxng-url",
        default="http://127.0.0.1:4070",
        help="SearXNG instance URL (default: http://127.0.0.1:4070)",
    )
    parser.add_argument(
        "--max-results",
        type=int,
        default=10,
        help="Max search results per query (default: 10)",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=5,
        help="Max pages to fetch per iteration (default: 5)",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Print progress messages"
    )
    parser.add_argument(
        "--json", action="store_true", dest="json_output", help="Output result as JSON"
    )
    parser.add_argument(
        "--tdm-host",
        action="append",
        default=[],
        metavar="HOST",
        help=(
            "Hostname for which you attest to holding a text-and-data-mining "
            "licence. Disarms the paywalled-publisher tripwire for this host. "
            "Repeatable. Example: --tdm-host nature.com --tdm-host sciencedirect.com"
        ),
    )
    return parser


def _resolve_model(args: argparse.Namespace) -> str:
    from andamentum.core.models import resolve_model_from_args

    return resolve_model_from_args(args.model)


def _list_agents() -> None:
    from .agents import AGENT_REGISTRY

    if not AGENT_REGISTRY:
        print("No agents registered.")
        return

    for name, defn in sorted(AGENT_REGISTRY.items()):
        model_name = defn.output_model.__name__ if defn.output_model else "None"
        tools = " [has tools]" if defn.has_tools else ""
        print(f"  {name:<30s}  output={model_name}{tools}")


def _display_results(result: Any, verbose: bool) -> None:
    """Display research results as structured plain text."""
    report = result.output
    verification = result.verification

    # Evidence summary
    print("\n=== Evidence Summary ===\n")
    print(report.evidence_summary)

    # Key findings
    if report.key_findings:
        print("\n=== Key Findings ===\n")
        for i, finding in enumerate(report.key_findings, 1):
            print(f"  {i}. {finding}")

    # Sources
    if report.sources:
        print(f"\n=== Sources ({len(report.sources)}) ===\n")
        for i, source in enumerate(report.sources, 1):
            print(f"  {i}. {source}")

    # Verification summary
    if verification:
        rate = verification.get("verification_rate", 0.0)
        verified = verification.get("verified_count", 0)
        total = verification.get("total_cited", 0)
        print("\n=== Verification ===\n")
        print(f"  Rate: {rate:.0%} ({verified}/{total} sources verified)")

    # Stats
    print(
        f"\n--- Stats: {result.iterations} iterations, {result.searches} searches, "
        f"{result.pages_fetched} pages fetched ---"
    )

    # Errors
    if result.errors.search_errors or result.errors.fetch_errors:
        print(
            f"\n--- Errors: {result.errors.search_errors} search, {result.errors.fetch_errors} fetch ---"
        )


async def _run(args: argparse.Namespace) -> None:
    try:
        from .orchestrator import run_research
    except ImportError:
        print(
            "Error: pydantic-ai is required. Install with: pip install andamentum",
            file=sys.stderr,
        )
        sys.exit(1)

    model = _resolve_model(args)

    reporter: Any = None
    if args.verbose:
        from rich.console import Console

        from .reporter import RichReporter

        console = Console()
        console.print(
            f"\n[bold]🔬 Researching:[/bold] {args.query}\n"
            f"[dim]Model: {model}  |  max_iterations={args.max_iterations}[/dim]"
        )
        reporter = RichReporter(console)

    tdm_hosts = frozenset(h.strip().lower() for h in args.tdm_host if h.strip())

    result = await run_research(
        args.query,
        model=model,
        max_iterations=args.max_iterations,
        searxng_url=args.searxng_url,
        max_results=args.max_results,
        max_pages=args.max_pages,
        verbose=args.verbose,
        reporter=reporter,
        tdm_allowed_hosts=tdm_hosts,
    )

    if args.json_output:
        print(result.model_dump_json(indent=2))
    else:
        _display_results(result, args.verbose)


def main() -> None:
    # Handle "agents" subcommand before argparse (avoids positional arg conflict)
    if len(sys.argv) > 1 and sys.argv[1] == "agents":
        _list_agents()
        return

    parser = _build_parser()
    args = parser.parse_args()
    try:
        asyncio.run(_run(args))
    except KeyboardInterrupt:
        print("\nResearch cancelled.")
        sys.exit(0)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        if "--verbose" in sys.argv:
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
