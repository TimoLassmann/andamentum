"""CLI for whetstone v2 — ``andamentum-whetstone``.

Single positional argument (the input document, anything ``harvest`` can
read), one or more ``--out`` files (format inferred from the extension),
and a small set of well-named options. ``--help`` is exhaustive on
purpose: this is the surface most users will touch.

Exit codes:
    0 — success
    1 — configuration / argument error
    2 — input could not be loaded (harvest failure, file not found)
    3 — review pipeline failed
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path
from typing import NoReturn, Sequence


_HELP_DESCRIPTION = """\
Sharpen your draft with structured review.

INPUT
    Path or URL to a document. Supported: .pdf, .docx, .html, .md, .txt
    (and any URL; HTML article-vs-listing is auto-detected).

OUTPUTS
    --out FILE         An output file. Repeat for multiple formats.
                       Format inferred from the extension:
                         .md   → markdown report
                         .html → HTML report (via typeset)
                         .docx → Word with track-changes + comments
                       For .docx: if INPUT is .docx we use it directly;
                       otherwise a clean .docx is generated from the
                       harvested text and patches are applied to that.

REVIEW OPTIONS
    --model MODEL              pydantic-ai model id, e.g.
                               openai:gpt-5.4-nano,
                               ollama:gemma4:31b-nvfp4.
                               Required unless --no-llm is given.
    --editor                   Also generate concrete edits (rewrites).
                               Default off — adds one LLM call per section.
    --editor-criteria LIST     Comma-separated. Default:
                               clarity,concision,grammar
    --no-challenge             Skip the refutation phase
                               (faster, slightly less reliable findings).
    --perspectives LIST        Comma-separated personas.
                               Default: rigorous
                               Examples: rigorous,statistician,writer
    --budget N                 Max LLM-investigated hypotheses.
                               Caps cost per review. Default: 30
    --no-llm                   Run only the deterministic structural pass
                               (citations, terms, numerics, cross-refs).
                               No --model required. Free, instant.
    -v, --verbose              Print progress to stderr.
"""


_HELP_EXAMPLES = """\
EXAMPLES
    # Quick deterministic-only check (no LLM, no cost)
    andamentum-whetstone paper.pdf --no-llm --out review.md

    # Full review of a PDF, output in three formats
    andamentum-whetstone paper.pdf \\
        --model openai:gpt-5.4-nano \\
        --out review.md --out review.html --out paper.reviewed.docx

    # Edit-mode: get concrete rewrites in a tracked-change Word doc
    andamentum-whetstone draft.docx \\
        --model openai:gpt-5.4-nano --editor \\
        --out draft.reviewed.docx

    # Multi-perspective panel on an arXiv paper
    andamentum-whetstone https://arxiv.org/pdf/1901.01753 \\
        --model openai:gpt-5.4-nano \\
        --perspectives rigorous,statistician,writer \\
        --out panel-review.html

    # Local model, generous budget, all outputs
    andamentum-whetstone manuscript.pdf \\
        --model ollama:gemma4:31b-nvfp4 --budget 60 --editor \\
        --out review.md --out review.html --out manuscript.reviewed.docx

EXIT CODES
    0  success
    1  configuration / argument error
    2  input could not be loaded (harvest failure, file not found)
    3  review pipeline failed
"""


_SUPPORTED_OUT_EXTENSIONS = {".md", ".html", ".docx"}


def _build_parser() -> argparse.ArgumentParser:
    """Build the argparse parser with the full help text."""
    parser = argparse.ArgumentParser(
        prog="andamentum-whetstone",
        description=_HELP_DESCRIPTION,
        epilog=_HELP_EXAMPLES,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "input",
        help="Path or URL to the document to review.",
    )
    parser.add_argument(
        "--out",
        action="append",
        type=Path,
        required=True,
        metavar="FILE",
        help="Output file. Repeat for multiple formats. "
        "Format inferred from extension (.md / .html / .docx).",
    )
    parser.add_argument(
        "--model",
        metavar="MODEL",
        help="pydantic-ai model id (e.g. openai:gpt-5.4-nano, "
        "ollama:gemma4:31b-nvfp4). Required unless --no-llm.",
    )
    parser.add_argument(
        "--editor",
        action="store_true",
        help="Also generate concrete edits (rewrites). Adds one LLM call per section.",
    )
    parser.add_argument(
        "--editor-criteria",
        default="clarity,concision,grammar",
        metavar="LIST",
        help="Comma-separated editor criteria. Default: clarity,concision,grammar",
    )
    parser.add_argument(
        "--no-challenge",
        action="store_true",
        help="Skip the refutation phase (faster, slightly less reliable findings).",
    )
    parser.add_argument(
        "--perspectives",
        default="rigorous",
        metavar="LIST",
        help="Comma-separated reviewer personas. Default: rigorous",
    )
    parser.add_argument(
        "--budget",
        type=int,
        default=30,
        metavar="N",
        help="Max LLM-investigated hypotheses. Caps cost per review. Default: 30",
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Run only the deterministic structural pass. No --model required.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print progress to stderr.",
    )
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    """Argparse-time validation that argparse can't express directly.

    Raises ``SystemExit(1)`` with a friendly message on any violation.
    """
    if not args.no_llm and not args.model:
        _die(
            1,
            "--model is required unless you pass --no-llm.\n"
            "Examples:\n"
            "  --model openai:gpt-5.4-nano\n"
            "  --model ollama:gemma4:31b-nvfp4",
        )
    if args.no_llm and args.editor:
        _die(
            1,
            "--editor requires --model (the editor agent is an LLM call). "
            "Drop --editor or --no-llm.",
        )
    for out in args.out:
        ext = out.suffix.lower()
        if ext not in _SUPPORTED_OUT_EXTENSIONS:
            _die(
                1,
                f"Unsupported output extension {ext!r} for {out}. "
                f"Supported: {', '.join(sorted(_SUPPORTED_OUT_EXTENSIONS))}",
            )


def _die(code: int, message: str) -> NoReturn:
    """Print to stderr and exit with the given code."""
    print(f"error: {message}", file=sys.stderr)
    sys.exit(code)


async def _run(args: argparse.Namespace) -> None:
    """Execute the review and write all requested outputs."""
    from .api import review_document
    from .renderers import render_docx, render_html, render_markdown

    perspectives = tuple(p.strip() for p in args.perspectives.split(",") if p.strip())
    editor_criteria = tuple(
        c.strip() for c in args.editor_criteria.split(",") if c.strip()
    )

    if args.verbose:
        print(f"reviewing: {args.input}", file=sys.stderr)
        print(f"  model:        {args.model or '(deterministic only)'}", file=sys.stderr)
        print(f"  perspectives: {', '.join(perspectives)}", file=sys.stderr)
        print(f"  editor:       {'on' if args.editor else 'off'}", file=sys.stderr)
        print(f"  challenge:    {'off' if args.no_challenge else 'on'}", file=sys.stderr)
        print(f"  budget:       {args.budget}", file=sys.stderr)
        print(f"  outputs:      {', '.join(str(o) for o in args.out)}", file=sys.stderr)

    try:
        result = await review_document(
            args.input,
            model=None if args.no_llm else args.model,
            perspectives=perspectives,
            hypothesis_budget=args.budget,
            challenge=not args.no_challenge,
            editor=args.editor,
            editor_criteria=editor_criteria,
        )
    except Exception as exc:
        # Distinguish "couldn't load the input" from "review crashed"
        # so users know whether to fix their input or their config.
        msg = str(exc)
        if any(t in msg.lower() for t in ("fetch", "harvest", "no such file", "not found")):
            _die(2, f"could not load input {args.input!r}: {exc}")
        else:
            _die(3, f"review pipeline failed: {exc}")

    if args.verbose:
        m = result.metrics
        print(
            f"  → {m.deterministic_findings_count} deterministic + "
            f"{m.investigated_findings_count} investigated finding(s), "
            f"{m.edits_count} edit(s); "
            f"{m.llm_calls} LLM call(s); "
            f"{m.wall_seconds:.1f}s",
            file=sys.stderr,
        )

    # If any output is .docx, prepare a baseline .docx ONCE upfront
    # (rather than per-output) so we don't re-harvest if the user asked
    # for multiple .docx outputs.
    baseline_docx: Path | None = None
    baseline_is_temp = False
    if any(o.suffix.lower() == ".docx" for o in args.out):
        baseline_docx, baseline_is_temp = await _ensure_docx_source(
            args.input, args.verbose
        )

    try:
        # ── Write each requested output ──────────────────────────────
        for out in args.out:
            ext = out.suffix.lower()
            out.parent.mkdir(parents=True, exist_ok=True)
            if ext == ".md":
                render_markdown(result, out)
            elif ext == ".html":
                render_html(result, out)
            elif ext == ".docx":
                assert baseline_docx is not None  # validated above
                render_docx(result, source_docx_path=baseline_docx, output_path=out)
            if args.verbose:
                print(f"  wrote {out}", file=sys.stderr)
    finally:
        # Clean up the auto-generated baseline docx (if we created one).
        if baseline_is_temp and baseline_docx is not None:
            try:
                baseline_docx.unlink()
            except OSError:
                pass


async def _ensure_docx_source(
    input_arg: str, verbose: bool
) -> tuple[Path, bool]:
    """Return ``(path, is_temp)`` for a .docx suitable as track-changes baseline.

    If the input is already a local .docx, return its Path with
    ``is_temp=False``. Otherwise harvest the input into markdown and
    write a clean baseline .docx via python-docx; return that path with
    ``is_temp=True`` so the caller knows to clean it up.
    """
    input_path = Path(input_arg) if not _is_url(input_arg) else None
    if input_path is not None and input_path.suffix.lower() == ".docx":
        return input_path, False

    if verbose:
        print(
            "  baseline docx: input is not .docx — generating from harvested text",
            file=sys.stderr,
        )

    from andamentum.harvest import extract as harvest_extract

    markdown = await harvest_extract(input_arg)
    return _markdown_to_baseline_docx(markdown), True


def _is_url(s: str) -> bool:
    return s.startswith(("http://", "https://", "file://"))


def _markdown_to_baseline_docx(markdown: str) -> Path:
    """Generate a clean baseline .docx from markdown text.

    The output is intentionally minimal — just headings and paragraphs —
    because the goal is to give the track-changes machinery something to
    write into. Anchor matching against this plain prose works because
    ``find_anchor`` is whitespace-tolerant.
    """
    import tempfile

    from docx import Document  # python-docx

    doc = Document()
    for block in markdown.split("\n\n"):
        block = block.strip()
        if not block:
            continue
        if block.startswith("#"):
            level = 0
            while level < 9 and level < len(block) and block[level] == "#":
                level += 1
            title = block[level:].strip()
            doc.add_heading(title, level=min(level, 9))
        else:
            doc.add_paragraph(block)

    fd, path = tempfile.mkstemp(suffix=".docx", prefix="whetstone-baseline-")
    import os

    os.close(fd)
    out = Path(path)
    doc.save(str(out))
    return out


def main(argv: Sequence[str] | None = None) -> None:
    """Entry point — invoked by the ``andamentum-whetstone`` script."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    _validate_args(args)
    try:
        asyncio.run(_run(args))
    except KeyboardInterrupt:
        _die(1, "interrupted")


if __name__ == "__main__":  # pragma: no cover
    main()
