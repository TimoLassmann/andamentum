"""Command-line entry point: ``andamentum-proofread``.

Accepts URLs, PDFs, .docx, .html, .pptx, .md, plain text files — anything
``andamentum.harvest`` can convert to markdown — and runs ``analyze()``
on the extracted text. ``-`` reads raw text from stdin (no harvest pass).

This is the ONE place inside ``andamentum.proofread`` that imports
``andamentum.harvest`` (lazily, so ``--help`` stays light); the library proper
remains a leaf service. With ``--slides`` it reads the deck through
``slides_io`` (python-pptx) instead, because harvest's markdown loses slide
boundaries, notes and font sizes.

Exit codes:
    0 — success (the check is advisory: findings do not change the code)
    2 — usage error: a wrong or missing argument
    3 — the input could not be read (missing file, unreachable URL,
        unsupported format, extraction failed)
    4 — the run failed (crash)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import NoReturn, Sequence

from .api import analyze
from .errors import ProofreadError
from .models import ProofreadResult

_EXIT_USAGE = 2
_EXIT_INPUT = 3
_EXIT_FAILED = 4

_EPILOG = """\
exit codes:
  0  success (findings do not change the exit code)
  2  usage error
  3  the input could not be read
  4  the run failed
"""


def build_parser() -> argparse.ArgumentParser:
    from andamentum import __version__

    parser = argparse.ArgumentParser(
        prog="andamentum-proofread",
        description=(
            "Deterministic readability + style check; no model. Accepts URLs "
            "and any document type the harvest module can read (PDF / HTML / "
            "DOCX / PPTX / Markdown / plain). Use '-' to read raw text from "
            "stdin."
        ),
        epilog=_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    parser.add_argument(
        "source",
        help=(
            "URL, local file path, or '-' for stdin (raw text, no extraction). "
            "Document files are converted to markdown via andamentum.harvest "
            "before analysis."
        ),
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the result as JSON instead of a readable report.",
    )
    parser.add_argument(
        "-o",
        "--out",
        type=Path,
        default=None,
        metavar="FILE",
        help="Write the result to this file instead of stdout.",
    )
    parser.add_argument(
        "--slides",
        action="store_true",
        help=(
            "Treat SOURCE as a .pptx deck and run the per-slide checks "
            "(headline length, deixis, banned words, unexpanded acronyms, one "
            "entity under two names, body restating the headline) instead of "
            "the prose checks."
        ),
    )
    parser.add_argument(
        "--allow",
        default=None,
        metavar="ACRONYMS",
        help=(
            "With --slides: comma-separated acronyms the audience knows, "
            "which need no expansion on the slide (e.g. 'MEK,RAS,FDA')."
        ),
    )
    return parser


def _die(code: int, message: str) -> NoReturn:
    print(f"error: {message}", file=sys.stderr)
    sys.exit(code)


async def _load_text(source: str) -> str:
    if source == "-":
        return sys.stdin.read()
    # Deferred: harvest is only needed for a file or URL, and --help must not
    # import it.
    from andamentum.harvest import HarvestError, extract

    try:
        return await extract(source)
    except HarvestError as exc:
        _die(_EXIT_INPUT, f"could not read {source!r}: {exc}")
    except OSError as exc:
        _die(_EXIT_INPUT, f"could not read {source!r}: {exc}")


def _format_human(result: ProofreadResult) -> str:
    """Compact human-readable report. A readability block with the six
    scores, then one section per finding type (omitted when empty)."""
    r = result.readability
    lines = [
        result.summary,
        "",
        "Readability:",
        f"  SMOG index             {r.smog_index:6.2f}  (grade level)",
        f"  Flesch-Kincaid grade   {r.flesch_kincaid_grade:6.2f}",
        f"  Flesch reading ease    {r.flesch_reading_ease:6.2f}  (0=hard, 100=easy)",
        f"  Gunning Fog            {r.gunning_fog:6.2f}",
        f"  Coleman-Liau           {r.coleman_liau_index:6.2f}",
        f"  ARI                    {r.automated_readability_index:6.2f}",
        f"  Words / Sentences      {r.word_count} / {r.sentence_count}",
        f"  Avg sentence length    {r.avg_sentence_length:6.2f} words",
        f"  Avg syllables/word     {r.avg_syllables_per_word:6.2f}",
    ]

    def _section(title: str, items: list[str]) -> None:
        if not items:
            return
        lines.append("")
        lines.append(f"{title} ({len(items)}):")
        for it in items:
            lines.append(f"  - {it}")

    _section(
        "Weasel words",
        [f"'{f.word}' (sentence {f.sentence_index + 1})" for f in result.weasel_words],
    )
    _section(
        "Passive voice",
        [
            f"'{f.matched_text}' (sentence {f.sentence_index + 1})"
            for f in result.passive_voice
        ],
    )
    _section(
        "Duplicate words",
        [
            f"'{f.word} {f.word}' (sentence {f.sentence_index + 1})"
            for f in result.duplicate_words
        ],
    )
    _section(
        "Weak openers",
        [
            f"'{f.matched_text}' (sentence {f.sentence_index + 1})"
            for f in result.weak_openers
        ],
    )
    if result.adverbs.adverb_count:
        lines.append("")
        lines.append(
            f"Adverbs (-ly): {result.adverbs.adverb_count} "
            f"({result.adverbs.adverb_density:.1%} of words)"
        )
    return "\n".join(lines)


def _run_slides(args: argparse.Namespace) -> int:
    from .slides import DEFAULT_ALLOWED_ACRONYMS, analyze_slides
    from .slides_io import read_pptx

    path = Path(args.source)
    if path.suffix.lower() != ".pptx":
        _die(_EXIT_USAGE, f"--slides needs a .pptx file: {args.source!r}")
    if not path.is_file():
        _die(_EXIT_INPUT, f"not a file: {args.source!r}")
    try:
        slides = read_pptx(path)
    except ProofreadError as exc:
        _die(_EXIT_INPUT, str(exc))

    allowed = DEFAULT_ALLOWED_ACRONYMS | {
        a.strip() for a in (args.allow or "").split(",") if a.strip()
    }
    result = analyze_slides(slides, allowed_acronyms=frozenset(allowed))
    if args.json:
        output = json.dumps(result.model_dump(), indent=2)
    else:
        lines = [result.summary]
        for f in result.findings:
            lines.append(f"  slide {f.slide:>2}  {f.check:<16} {f.detail}")
        output = "\n".join(lines)
    _write(output, args.out)
    return 0


def _write(output: str, target: Path | None) -> None:
    if target is None:
        sys.stdout.write(output)
        if not output.endswith("\n"):
            sys.stdout.write("\n")
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(output, encoding="utf-8")


async def _run(args: argparse.Namespace) -> int:
    text = await _load_text(args.source)
    result = analyze(text)
    output = (
        json.dumps(result.model_dump(), indent=2)
        if args.json
        else _format_human(result)
    )
    _write(output, args.out)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
        if args.allow is not None and not args.slides:
            parser.error("--allow only applies with --slides")
        if args.slides:
            return _run_slides(args)
        return asyncio.run(_run(args))
    except KeyboardInterrupt:
        print("error: interrupted", file=sys.stderr)
        return _EXIT_FAILED
    except SystemExit as exc:
        if exc.code is None:
            return 0
        return exc.code if isinstance(exc.code, int) else _EXIT_USAGE


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
