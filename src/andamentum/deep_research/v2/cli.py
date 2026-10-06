"""``andamentum-research v2``: the evidence-first engine from the command line.

    andamentum-research v2 "QUESTION" --mode both --model M
        [--preset baseline|fast|criteria]
        [--decision-model NAME] [--store PATH] [--cache PATH] [--searxng-url URL]
        [--sources openalex,europepmc,arxiv] [--openalex-search keyword]
        [--tdm-host HOST ...] [--results-per-query N]
        [--max-rounds N ... every Budget field] [--json | --md] [--out FILE]

Every model and address is a flag; nothing has a hidden default. ``--preset``
names the run's ``ResearchConfig`` (``config.PRESETS``; ``baseline`` unless
another is named). Without
``--store`` nothing persists. Secrets are read here only, from the environment
after ``load_dotenv()``: ``OPENALEX_API_KEY``, ``OPENALEX_EMAIL`` (the polite
contact address for OpenAlex and Crossref) and ``SEMANTIC_SCHOLAR_API_KEY``.

Exit status: 0 for a clean run, 4 for a run that finished degraded (a source
or model failed; the result is still written and the reasons are on stderr),
1 for an error, 2 for a usage error.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path
from typing import Literal, get_args

from dotenv import load_dotenv

from .config import DEFAULT_PRESET, PRESETS
from .graph import research
from .sources import SourceConfig
from .store import ResearchStore
from .types import Budget, Mode, ResearchResult

EXIT_DEGRADED = 4
ScholarlyName = Literal["openalex", "europepmc", "arxiv", "semantic_scholar"]
_SCHOLARLY: tuple[ScholarlyName, ...] = get_args(ScholarlyName)


def _flag(field: str) -> str:
    return "--" + field.replace("_", "-")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="andamentum-research v2",
        description="Research a question over the web and scholarly sources; "
        "the report cites verified findings by id.",
    )
    parser.add_argument("question", help="the research question")
    parser.add_argument(
        "--mode",
        required=True,
        choices=get_args(Mode),
        help="web (SearxNG), scholarly (OpenAlex, Europe PMC, arXiv) or both",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="the language model, a pydantic-ai id (e.g. ollama:gemma4:31b-cloud)",
    )
    parser.add_argument(
        "--preset",
        choices=sorted(PRESETS),
        default=DEFAULT_PRESET,
        help="what the run does at every swappable step: baseline (one generic "
        "relevance question per candidate), fast (the baseline with a lexical "
        "pre-filter and code-first claim-support checks) or criteria "
        "(LLM-compiled criteria); default: %(default)s",
    )
    parser.add_argument(
        "--decision-model",
        default=None,
        metavar="NAME",
        help="a local Ollama decision model for screening and checks; without "
        "it the language model answers those questions",
    )
    parser.add_argument(
        "--store",
        default=None,
        metavar="PATH",
        help="the research store (a SQLite file; blobs go beside it). Without "
        "it nothing persists",
    )
    parser.add_argument(
        "--cache",
        default=None,
        metavar="PATH",
        help="a shared HTTP cache (a SQLite file; blobs go beside it) every "
        "source reads first and writes into, kept apart from --store; the "
        "run's own store holds a copy of each entry the run used",
    )
    parser.add_argument(
        "--searxng-url", default=None, help="the SearxNG instance (web and both modes)"
    )
    parser.add_argument(
        "--sources",
        type=scholarly_names,
        default=",".join(SourceConfig().scholarly),
        help=f"scholarly sources, comma-separated, from {', '.join(_SCHOLARLY)} "
        "(default: %(default)s)",
    )
    parser.add_argument(
        "--openalex-search",
        choices=("keyword", "semantic", "both"),
        default="keyword",
        help="OpenAlex endpoint (default: %(default)s)",
    )
    parser.add_argument(
        "--tdm-host",
        action="append",
        default=[],
        metavar="HOST",
        help="a publisher host whose text you are licensed to mine (repeatable)",
    )
    parser.add_argument(
        "--results-per-query",
        type=int,
        default=SourceConfig().results_per_query,
        help="results asked of each source per query (default: %(default)s)",
    )
    limits = parser.add_argument_group("budget (every bound of the run)")
    for name, info in Budget.model_fields.items():
        kind = float if info.annotation is float else int
        limits.add_argument(
            _flag(name), type=kind, default=info.default, help="default: %(default)s"
        )
    output = parser.add_mutually_exclusive_group()
    output.add_argument(
        "--json", action="store_true", help="print the whole result as JSON"
    )
    output.add_argument(
        "--md", action="store_true", help="print the report as Markdown (default)"
    )
    parser.add_argument(
        "--out", default=None, metavar="FILE", help="write the output here, not stdout"
    )
    parser.add_argument(
        "--quiet", action="store_true", help="no progress lines on stderr"
    )
    return parser


def scholarly_names(text: str) -> tuple[ScholarlyName, ...]:
    """The scholarly sources named in *text* (comma-separated), in its order."""
    names: list[ScholarlyName] = []
    for raw in text.split(","):
        name = raw.strip()
        known: list[ScholarlyName] = [s for s in _SCHOLARLY if s == name]
        if not known:
            raise argparse.ArgumentTypeError(
                f"unknown source {name!r}; known: {', '.join(_SCHOLARLY)}"
            )
        names += known
    return tuple(dict.fromkeys(names))


def _source_config(args: argparse.Namespace) -> SourceConfig:
    return SourceConfig(
        mailto=os.environ.get("OPENALEX_EMAIL") or None,
        openalex_api_key=os.environ.get("OPENALEX_API_KEY") or None,
        semantic_scholar_api_key=os.environ.get("SEMANTIC_SCHOLAR_API_KEY") or None,
        searxng_url=args.searxng_url,
        tdm_allowed_hosts=frozenset(args.tdm_host),
        scholarly=args.sources,
        openalex_search=args.openalex_search,
        results_per_query=args.results_per_query,
    )


def _render(result: ResearchResult, *, as_json: bool) -> str:
    if as_json:
        return result.model_dump_json(indent=2) + "\n"
    return result.report_markdown


async def _run(args: argparse.Namespace) -> ResearchResult:
    budget = Budget(**{name: getattr(args, name) for name in Budget.model_fields})

    def progress(line: str) -> None:
        if not args.quiet:
            print(line, file=sys.stderr, flush=True)

    cache = ResearchStore(Path(args.cache)) if args.cache else None
    try:
        return await research(
            args.question,
            mode=args.mode,
            model=args.model,
            decision_model=args.decision_model,
            store=Path(args.store) if args.store else None,
            budget=budget,
            sources=_source_config(args),
            config=PRESETS[args.preset],
            http_cache=cache,
            progress=progress,
        )
    finally:
        if cache is not None:
            cache.close()


def main(argv: list[str]) -> None:
    """Run one research question; *argv* is what follows ``v2``."""
    load_dotenv()  # secrets only (API keys, contact mail); configuration is flags
    args = build_parser().parse_args(argv)
    try:
        result = asyncio.run(_run(args))
    except KeyboardInterrupt:
        print("\nResearch cancelled.", file=sys.stderr)
        sys.exit(1)
    except Exception as error:  # reported, then a failing exit status
        print(f"Error: {type(error).__name__}: {error}", file=sys.stderr)
        sys.exit(1)
    text = _render(result, as_json=args.json)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text)
    print(
        f"preset: {result.config_name} ({result.config_hash[:12]}); "
        f"stop: {result.stop_reason}; {len(result.findings)} findings from "
        f"{len(result.works)} works; {len(result.verification)} verification flags",
        file=sys.stderr,
    )
    if result.notes:
        print(f"notes ({len(result.notes)}):", file=sys.stderr)
        for note in result.notes:
            print(f"  - {note}", file=sys.stderr)
    if result.degraded:
        print(f"degraded ({len(result.degraded)}):", file=sys.stderr)
        for reason in result.degraded:
            print(f"  - {reason}", file=sys.stderr)
        sys.exit(EXIT_DEGRADED)
