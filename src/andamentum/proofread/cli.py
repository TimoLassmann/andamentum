"""Command-line entry point: ``andamentum-proofread``.

Accepts URLs, PDFs, .docx, .html, .pptx, .md, plain text files — anything
``andamentum.harvest`` can convert to markdown — and runs ``analyze()``
on the extracted text. ``-`` reads raw text from stdin (no harvest pass).

This is the ONE place inside ``andamentum.proofread`` that imports
``andamentum.harvest``; the library proper remains a leaf service.

Exit codes:
    0 — success
    1 — argument / IO error
    2 — fetch / unsupported-format error
    3 — extraction failed
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import NoReturn, Sequence

from ..harvest import (
    ExtractionError,
    FetchError,
    HarvestError,
    UnsupportedFormatError,
    extract,
)
from .api import analyze
from .models import ProofreadResult


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="andamentum-proofread",
        description=(
            "Deterministic readability + style check. Accepts URLs and any "
            "document type the harvest module can read (PDF / HTML / DOCX / "
            "PPTX / Markdown / plain). Use '-' to read raw text from stdin."
        ),
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
        "-f",
        "--format",
        choices=("human", "json"),
        default="human",
        help="Output format. Default: human.",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="-",
        metavar="FILE",
        help="Output file path. Default: '-' (stdout).",
    )
    parser.add_argument(
        "--raw",
        action="store_true",
        help=(
            "Treat <source> as a plain text file path — skip harvest "
            "extraction. Useful for already-clean .txt or .md inputs."
        ),
    )
    return parser


def _die(code: int, message: str) -> NoReturn:
    print(f"error: {message}", file=sys.stderr)
    sys.exit(code)


async def _load_text(source: str, raw: bool) -> str:
    if source == "-":
        return sys.stdin.read()
    if raw:
        path = Path(source)
        if not path.is_file():
            _die(1, f"not a file: {source!r}")
        return path.read_text(encoding="utf-8")
    return await extract(source)


def _format_human(result: ProofreadResult) -> str:
    """Compact human-readable report. Five-line readability block, then one
    section per finding type (omitted when empty)."""
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


async def _run(args: argparse.Namespace) -> int:
    try:
        text = await _load_text(args.source, args.raw)
    except FetchError as exc:
        _die(2, f"could not fetch {args.source!r}: {exc}")
    except UnsupportedFormatError as exc:
        _die(2, f"unsupported format for {args.source!r}: {exc}")
    except ExtractionError as exc:
        _die(3, f"extraction failed for {args.source!r}: {exc}")
    except HarvestError as exc:
        _die(3, f"harvest failed for {args.source!r}: {exc}")
    except OSError as exc:
        _die(1, f"could not read {args.source!r}: {exc}")

    result = analyze(text)
    output = (
        json.dumps(result.model_dump(), indent=2)
        if args.format == "json"
        else _format_human(result)
    )

    if args.output == "-":
        sys.stdout.write(output)
        if not output.endswith("\n"):
            sys.stdout.write("\n")
    else:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(output, encoding="utf-8")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        return asyncio.run(_run(args))
    except KeyboardInterrupt:
        _die(1, "interrupted")


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
