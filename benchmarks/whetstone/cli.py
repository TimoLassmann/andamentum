"""Whetstone evaluation CLI.

Subcommands (run in order, or `all` to chain fetch→…→html):

  discover    find candidate preprints (later published) → candidates.json
              to curate into a committed seeds.json (one-time helper)
  fetch       download v1 PDFs for the COMMITTED seeds → harvest once
  arms        run Arm A (whetstone) and Arm B (whole-doc) on the SAME --model
  adjudicate  judge-model alignment + blinded human worksheets (--judge-model)
  report      aggregate runs/ into the decision-grade readout
  html        self-contained side-by-side HTML visualiser over all papers
  all         fetch → arms → adjudicate → report → html

The corpus is FIXED and reproducible: `fetch` reads a committed seeds.json
(curate it from `discover`), never auto-discovers, so every run uses the same
papers. Only the open v1 preprint is ever downloaded — never the paywalled
published version (the journal is metadata, used only as a selection signal).

Model strings resolve through the shared core infrastructure, so any
pydantic-ai id works (ollama:… / openai:… / bedrock:…). Corpus and outputs
default to the gitignored corpus/ and runs/ dirs; override with --corpus-dir /
--out-dir to keep data fully outside the repo.

Run as:  uv run python -m benchmarks.whetstone.cli <subcommand> [opts]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

from .types import PaperResult

logger = logging.getLogger("whetstone.bench")

_HERE = Path(__file__).resolve().parent
_DEFAULT_CORPUS = _HERE / "corpus"
_DEFAULT_RUNS = _HERE / "runs"


# ── fetch ─────────────────────────────────────────────────────────────────


_DEFAULT_SEEDS = _HERE / "seeds.json"


def cmd_discover(args: argparse.Namespace) -> int:
    """Find candidate preprints (later published) for you to curate into seeds."""
    import json as _json

    from .loader import discover_biorxiv_published

    cands = discover_biorxiv_published(n=args.n)
    out = Path(args.corpus_dir) / "candidates.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        _json.dumps([c.model_dump() for c in cands], indent=2), encoding="utf-8"
    )
    for c in cands:
        logger.info(
            "  %s — %s [%s] → %s",
            c.id,
            (c.title or "")[:70],
            c.subfield,
            c.published_journal or "?",
        )
    logger.info(
        "[discover] wrote %d candidate(s) to %s — curate into seeds.json",
        len(cands),
        out,
    )
    return 0


def cmd_fetch(args: argparse.Namespace) -> int:
    """Download v1 PDFs for the COMMITTED seed list and harvest them once.

    Seeds are required — the corpus must be fixed and reproducible. Run
    `discover` to find candidates, then pin the ones you want into seeds.json.
    """
    from .loader import fetch_pdf, harvest_paper, load_seeds, save_manifest

    seeds_path = Path(args.seeds) if args.seeds else _DEFAULT_SEEDS
    if not seeds_path.exists():
        logger.error(
            "[fetch] no seed list at %s. Run `discover` and curate a seeds.json "
            "(reproducible corpus); auto-discovery is intentionally not a fetch "
            "default.",
            seeds_path,
        )
        return 1
    refs = load_seeds(seeds_path)
    logger.info("[fetch] %d seed paper(s) from %s", len(refs), seeds_path)

    corpus = Path(args.corpus_dir)
    corpus.mkdir(parents=True, exist_ok=True)
    fetched = []
    for ref in refs:
        try:
            ref = fetch_pdf(ref, corpus)
            ref = asyncio.run(harvest_paper(ref, corpus))
            fetched.append(ref)
        except Exception as exc:  # keep going; one bad paper shouldn't abort
            logger.warning("[fetch] %s failed: %s", ref.slug, exc)
    save_manifest(fetched, corpus / "manifest.json")
    logger.info("[fetch] %d/%d papers ready", len(fetched), len(refs))
    return 0 if fetched else 1


def cmd_html(args: argparse.Namespace) -> int:
    """Build the self-contained side-by-side HTML visualiser over all papers."""
    from .visualize import write_report

    runs = Path(args.out_dir)
    adj_files = sorted(runs.glob("*.adj.json"))
    if not adj_files:
        logger.error("[html] no *.adj.json — run `adjudicate` first")
        return 1
    results = [PaperResult.model_validate_json(p.read_text("utf-8")) for p in adj_files]
    out = write_report(results, runs / "report.html")
    logger.info("[html] wrote %s (%d papers)", out, len(results))
    return 0


# ── arms ──────────────────────────────────────────────────────────────────


def cmd_arms(args: argparse.Namespace) -> int:
    from .arms import run_arm_a, run_arm_b
    from .loader import load_manifest

    corpus = Path(args.corpus_dir)
    runs = Path(args.out_dir)
    runs.mkdir(parents=True, exist_ok=True)
    refs = load_manifest(corpus / "manifest.json")

    async def one(ref):
        a = await run_arm_a(ref, model=args.model)
        b = await run_arm_b(ref, model=args.model)
        result = PaperResult(paper=ref, arm_a=a, arm_b=b)
        (runs / f"{ref.slug}.arms.json").write_text(
            result.model_dump_json(indent=2), encoding="utf-8"
        )

    for ref in refs:
        try:
            asyncio.run(one(ref))
            logger.info("[arms] %s done", ref.slug)
        except Exception as exc:
            logger.warning("[arms] %s failed: %s", ref.slug, exc)
    return 0


# ── adjudicate ──────────────────────────────────────────────────────────────


def cmd_adjudicate(args: argparse.Namespace) -> int:
    from .adjudicate import adjudicate, build_worksheet, judge_verdict_match

    runs = Path(args.out_dir)
    arms_files = sorted(runs.glob("*.arms.json"))
    if not arms_files:
        logger.error("[adjudicate] no *.arms.json in %s — run `arms` first", runs)
        return 1

    keys: dict[str, dict[str, str]] = {}
    worksheets: list[str] = []
    for path in arms_files:
        result = PaperResult.model_validate_json(path.read_text("utf-8"))

        async def go():
            adj = await adjudicate(result.arm_a, result.arm_b, model=args.judge_model)
            match = await judge_verdict_match(
                result.arm_a.verdict, result.arm_b.verdict, model=args.judge_model
            )
            return adj, match

        adj, match = asyncio.run(go())
        result.adjudications = adj
        result.verdict_match = match
        path.with_suffix(".adj.json").write_text(
            result.model_dump_json(indent=2), encoding="utf-8"
        )
        sheet, key = build_worksheet(result, seed=args.seed)
        worksheets.append(sheet)
        keys[result.paper.slug] = key
        logger.info("[adjudicate] %s done", result.paper.slug)

    (runs / "worksheet.md").write_text("\n".join(worksheets), encoding="utf-8")
    (runs / "worksheet_key.json").write_text(json.dumps(keys, indent=2), "utf-8")
    logger.info("[adjudicate] wrote blinded worksheet.md + key")
    return 0


# ── report ────────────────────────────────────────────────────────────────


def cmd_report(args: argparse.Namespace) -> int:
    from .report import aggregate, render_markdown

    runs = Path(args.out_dir)
    adj_files = sorted(runs.glob("*.adj.json"))
    if not adj_files:
        logger.error("[report] no *.adj.json — run `adjudicate` first")
        return 1
    results = [PaperResult.model_validate_json(p.read_text("utf-8")) for p in adj_files]
    readout = aggregate(results)
    md = render_markdown(readout)
    (runs / "readout.md").write_text(md, encoding="utf-8")
    (runs / "readout.json").write_text(readout.model_dump_json(indent=2), "utf-8")
    print(md)
    return 0


def cmd_all(args: argparse.Namespace) -> int:
    for step in (cmd_fetch, cmd_arms, cmd_adjudicate, cmd_report, cmd_html):
        rc = step(args)
        if rc != 0:
            return rc
    return 0


# ── argparse ────────────────────────────────────────────────────────────────


def _resolve(model_arg: str | None) -> str:
    from andamentum.core.models import resolve_model_from_args

    return resolve_model_from_args(model_arg)


def build_parser() -> argparse.ArgumentParser:
    # Shared options live on a parent parser so they're accepted AFTER the
    # subcommand (e.g. `discover --n 15`), which is what people type.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--corpus-dir", default=str(_DEFAULT_CORPUS))
    common.add_argument("--out-dir", default=str(_DEFAULT_RUNS))
    common.add_argument("--n", type=int, default=5, help="papers (pilot default 5)")
    common.add_argument(
        "--seeds", default="", help="curated seed JSON (else seeds.json)"
    )
    common.add_argument(
        "--model", default=None, help="model for BOTH arms (ollama:/openai:/…)"
    )
    common.add_argument("--judge-model", default=None, help="model for adjudication")
    common.add_argument("--seed", type=int, default=1, help="worksheet blinding seed")

    p = argparse.ArgumentParser(prog="whetstone-bench", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    for name in ("discover", "fetch", "arms", "adjudicate", "report", "html", "all"):
        sub.add_parser(name, parents=[common])
    return p


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    # API keys live in .env, not the shell — load them before any model call.
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass
    args = build_parser().parse_args(argv)

    # Resolve models only for the commands that need them (so `fetch`/`report`
    # don't demand a --model).
    if args.command in ("arms", "all"):
        args.model = _resolve(args.model)
    if args.command in ("adjudicate", "all"):
        args.judge_model = _resolve(args.judge_model)

    dispatch = {
        "discover": cmd_discover,
        "fetch": cmd_fetch,
        "arms": cmd_arms,
        "adjudicate": cmd_adjudicate,
        "report": cmd_report,
        "html": cmd_html,
        "all": cmd_all,
    }
    return dispatch[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
