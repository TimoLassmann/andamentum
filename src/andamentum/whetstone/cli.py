"""CLI for whetstone — ``andamentum-whetstone``.

Six subcommands:

    andamentum-whetstone [review] INPUT --model M --out OUT [...]
        Criterion-cascade review (the default; verb is optional).

    andamentum-whetstone panel INPUT --model M --out OUT [...]
        Multi-expert simulated peer review.

    andamentum-whetstone checklist INPUT --model M --out OUT [...]
        Quota-driven defect extraction against a probe bank, summarised
        by a deterministic coverage table. Same outputs as review.

    andamentum-whetstone proofread INPUT [...]
        Deterministic style + readability check (no LLM).

    andamentum-whetstone apply-patches INPUT --patches P.json --out O.docx
        Apply a pre-built JSON patch list to a .docx (no LLM).

    andamentum-whetstone verify-provenance INPUT.docx [--format human|json]
        Read-only: report which whetstone AI-provenance markers a .docx
        contains. Exit 0 if any marker present, 2 if none, 1 on read
        error.

Exit codes:
    0 — success (or, for verify-provenance, "markers found")
    1 — configuration / argument error
    2 — input could not be loaded (harvest failure, file not found, or
        — for verify-provenance — readable but no markers present)
    3 — review pipeline failed
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
import time
from pathlib import Path
from typing import NoReturn, Sequence

from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.table import Table

_LOGGER_NAME = "andamentum.whetstone"


_HELP_DESCRIPTION = """\
Sharpen YOUR OWN DRAFT with structured review.

Whetstone is NOT a peer-review tool. Do not run it on manuscripts,
grants, or any other documents shared with you in confidence (as a
journal reviewer, grant panel member, examiner, or editor). Most
publishers and funders currently prohibit sharing such documents
with AI tools, including cloud LLMs. See RESPONSIBLE_USE.md.

Which one do I want? (first positional, optional — defaults to
`review`):

  review     "help me improve this draft" — criterion cascade,
             connected findings, synthesised prose. Mid-draft.
             (default)
  checklist  "run the standard check" — fixed probe bank, same shape
             every run, deterministic coverage table. Pre-submission,
             draft-vs-draft, CI. The output centres on that table;
             findings hang off it rather than off synthesised prose.
  panel      simulated multi-expert peer review of your own draft
  proofread  readability + style only, no LLM

Two further verbs, neither of them a review:

  apply-patches     apply a pre-built JSON patch list to a .docx (no LLM)
  verify-provenance read-only: report AI-provenance markers in a .docx

INPUT is a path or URL to a document. Supported: .pdf, .docx, .html,
.md, .txt; URLs auto-detect article vs index page. OUTPUTS: pass
--out FILE one or more times; format is inferred from the extension
(.md / .html / .docx).

For .docx output, if INPUT is already .docx whetstone uses it as the
track-changes baseline; otherwise it harvests INPUT to markdown and
generates a clean baseline .docx to patch.
"""


_HELP_EXAMPLES = """\
EXAMPLES

    # Default review (criterion cascade + gap loop)
    andamentum-whetstone draft.md --model openai:gpt-5.4-nano \\
        --out review.md

    # Same review, multiple output formats in one pass
    andamentum-whetstone paper.pdf --model openai:gpt-5.4-nano \\
        --out review.md --out review.html --out paper.reviewed.docx

    # Review with concrete edits as tracked-changes in Word
    andamentum-whetstone draft.docx --model openai:gpt-5.4-nano \\
        --editor --out draft.reviewed.docx

    # Review with caller-supplied criteria
    andamentum-whetstone draft.md --model openai:gpt-5.4-nano \\
        --criteria "originality; depth of literature; clarity" \\
        --out review.md

    # Review with free-text guidelines (one extractor call + cascade)
    andamentum-whetstone draft.md --model openai:gpt-5.4-nano \\
        --guidelines @/path/to/author-guidelines.md --out review.md

    # Multi-expert panel review (different shape — N expert reviews + synthesis)
    andamentum-whetstone panel draft.md --model openai:gpt-5.4-nano \\
        --n-experts 4 --i-am-the-author --out panel-review.md

    # Checklist: one quota-driven probe per defect class + coverage table
    andamentum-whetstone checklist draft.md --model openai:gpt-5.4-nano \\
        --out checklist.md

    # Checklist against the non-academic bank, straight into Word
    andamentum-whetstone checklist memo.docx --model openai:gpt-5.4-nano \\
        --document-type external_communication --out memo.reviewed.docx

    # Deterministic style + readability check (no LLM)
    andamentum-whetstone proofread draft.md --format json

    # Apply a pre-built JSON patch list to a .docx (no LLM)
    andamentum-whetstone apply-patches draft.docx \\
        --patches patches.json --out draft.reviewed.docx \\
        --patch-author "Claude" --patch-report review.md

    # Read-only: report AI-provenance markers in a .docx
    andamentum-whetstone verify-provenance suspicious.docx
    andamentum-whetstone verify-provenance suspicious.docx --format json

EXIT CODES
    0  success
    1  configuration / argument error
    2  input could not be loaded (harvest failure, file not found)
    3  review pipeline failed
"""


_SUPPORTED_OUT_EXTENSIONS = {".md", ".html", ".docx"}


# Defaults for the review-only flags that carry a non-sentinel value. They are
# named once because _validate_args rejects these flags under `checklist` by
# comparing the parsed value against its default — sound only while there is a
# single definition of each. A second literal in the parser would silently turn
# the rejection off for that flag.
_DEFAULT_EDITOR_CRITERIA = "clarity,concision,grammar"
_DEFAULT_NOVELTY_SEARCH_DEPTH = 2
_DEFAULT_ROUNDS = 3

# (namespace attribute, user-facing flag, value meaning "not set").
_REVIEW_ONLY_FLAGS: tuple[tuple[str, str, object], ...] = (
    ("criteria", "--criteria", None),
    ("guidelines", "--guidelines", ""),
    ("editor", "--editor", False),
    ("editor_criteria", "--editor-criteria", _DEFAULT_EDITOR_CRITERIA),
    ("check_novelty", "--check-novelty", False),
    ("novelty_search_depth", "--novelty-search-depth", _DEFAULT_NOVELTY_SEARCH_DEPTH),
    ("rounds", "--rounds", _DEFAULT_ROUNDS),
)


def _build_parser() -> argparse.ArgumentParser:
    """Build the (single) flat argparse parser. The subcommand front-end
    in :func:`main` rewrites verb-style invocations into flat argv before
    parsing, so this single parser serves the four subcommands that reach
    it — review, panel, checklist and apply-patches. ``proofread`` and
    ``verify-provenance`` short-circuit in :func:`main` and never get
    here."""
    parser = argparse.ArgumentParser(
        prog="andamentum-whetstone",
        description=_HELP_DESCRIPTION,
        epilog=_HELP_EXAMPLES,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    from andamentum import __version__ as _ver

    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s (andamentum {_ver})",
    )

    # ── Common: required positional + outputs + universal options ───────
    parser.add_argument("input", help="Path or URL to the document.")
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
        "ollama:gemma4:31b-nvfp4). Required for the review, panel and "
        "checklist subcommands.",
    )
    # Shared by review and checklist: the same six types route to a criterion
    # set in one mode and to a probe bank in the other. Deliberately ONE flag —
    # a second `--document-type` in the checklist group would be an argparse
    # conflict, and a differently-named twin would make the same question look
    # like two questions.
    parser.add_argument(
        "--document-type",
        choices=(
            "auto",
            "academic",
            "external_communication",
            "essay",
            "tutorial",
            "creative",
            "general",
        ),
        default="auto",
        help=(
            "Document type. In review mode it routes to one of six criterion "
            "sets; in checklist mode to the matching probe bank. 'auto' "
            "(default) runs a one-shot classifier using --model; explicit "
            "values skip the classifier. In checklist mode 'auto' therefore "
            "costs one extra LLM call on top of the probe calls — the CLI "
            "still defaults to it because detection is what a command-line "
            "run is expected to do, while the Python API "
            "(run_checklist_v3) defaults to 'general' so its default path "
            "stays classifier-free. Pass an explicit type to match the API."
        ),
    )
    parser.add_argument(
        "--confirm-own-draft",
        action="store_true",
        help=(
            "Bypass the confidentiality-marker tripwire (which refuses "
            "to run on text containing 'Manuscript ID:', 'Reviewer "
            "Instructions', 'Editorial Office', etc.). Pass this only "
            "when the matched marker is a false positive in your own draft."
        ),
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print phase-by-phase progress to stderr.",
    )

    # Watermark flags (mutually exclusive)
    visible_wm = parser.add_mutually_exclusive_group()
    visible_wm.add_argument(
        "--visible-watermark",
        dest="visible_watermark",
        action="store_true",
        default=True,
        help="Include a visible 'AI-generated review content' banner "
        "in the review report (default).",
    )
    visible_wm.add_argument(
        "--no-visible-watermark",
        dest="visible_watermark",
        action="store_false",
        help="Suppress the visible banner. Invisible provenance metadata "
        "(docx core properties, HTML <meta>) is written regardless.",
    )

    # --mode is internal: set by the subcommand rewriter (`panel` verb →
    # --mode panel, `checklist` verb → --mode checklist). Users invoke the
    # subcommands instead.
    parser.add_argument(
        "--mode",
        choices=("review", "panel", "checklist"),
        default="review",
        help=argparse.SUPPRESS,
    )

    # ── Review options ─────────────────────────────────────────────────
    review_grp = parser.add_argument_group(
        "review options (andamentum-whetstone [review] INPUT ...)"
    )
    review_grp.add_argument(
        "--criteria",
        action="append",
        default=None,
        metavar="LIST",
        help="Caller-supplied criteria. Semicolon-separated, or repeat the "
        "flag. Overrides the document-type default set.",
    )
    review_grp.add_argument(
        "--guidelines",
        default="",
        metavar="TEXT",
        help="Free-text reviewer brief / journal guidelines. Use "
        "``@path/to/file`` to read from a file. One LLM call decomposes "
        "the prose into a criterion list. Mutually exclusive with --criteria.",
    )
    review_grp.add_argument(
        "--editor",
        action="store_true",
        help="Generate concrete edits (rewrites). Adds one LLM call per "
        "section. The .docx renderer applies them as tracked changes.",
    )
    review_grp.add_argument(
        "--editor-criteria",
        default=_DEFAULT_EDITOR_CRITERIA,
        metavar="LIST",
        help="Comma-separated editor criteria. Default: clarity,concision,grammar.",
    )
    review_grp.add_argument(
        "--check-novelty",
        action="store_true",
        help="Run the 3-node novelty check against deep_research. Adds 1 "
        "extractor + N searches + N judge calls. Disabled by default.",
    )
    review_grp.add_argument(
        "--novelty-search-depth",
        type=int,
        default=_DEFAULT_NOVELTY_SEARCH_DEPTH,
        metavar="N",
        help="Search depth for --check-novelty (1=quick, 2=balanced, "
        "3=thorough). Default: 2.",
    )
    review_grp.add_argument(
        "--rounds",
        type=int,
        default=_DEFAULT_ROUNDS,
        metavar="N",
        help="Hard cap on the gap-loop re-examination rounds. Default: 3. "
        "The loop typically exits earlier when no demands remain.",
    )

    # ── Panel options ──────────────────────────────────────────────────
    panel_grp = parser.add_argument_group(
        "panel options (andamentum-whetstone panel INPUT ...)"
    )
    panel_grp.add_argument(
        "--n-experts",
        type=int,
        default=4,
        metavar="N",
        help="Number of fictional expert reviewers to generate. Default: 4.",
    )
    panel_grp.add_argument(
        "--panel-disciplines",
        default="",
        metavar="LIST",
        help="Comma-separated explicit discipline list. When supplied, "
        "skips the keyword-extraction LLM call.",
    )
    panel_grp.add_argument(
        "--i-am-the-author",
        action="store_true",
        help="Required for the panel subcommand. Affirms that INPUT is "
        "your own draft, not a manuscript shared with you confidentially. "
        "Set ANDAMENTUM_PANEL_OWN_AUTHOR=1 to pre-affirm.",
    )

    # ── Checklist options ──────────────────────────────────────────────
    #
    # Checklist mode has no flags of its own. It is configured by
    # --document-type (which bank) and --model, both declared above and both
    # shared with review mode, and it writes the same --out formats. That is
    # not an oversight: the mode's whole proposition is that the machinery is
    # fixed and only the probe bank varies, so a knob here would be a knob
    # that makes two runs less comparable. A caller who wants a different
    # bank passes ``probes=[...]`` to ``run_checklist_v3`` in Python rather
    # than assembling one from flags.

    # ── Apply-patches options ──────────────────────────────────────────
    patches_grp = parser.add_argument_group(
        "apply-patches options (andamentum-whetstone apply-patches INPUT.docx --patches P.json ...)"
    )
    patches_grp.add_argument(
        "--apply-patches",
        type=Path,
        default=None,
        metavar="PATH",
        help="Skip the review pipeline and apply a pre-built JSON patch "
        "list to INPUT. INPUT must be a local .docx; --out must be one "
        ".docx file. PATH is a JSON array of DocumentPatch objects.",
    )
    patches_grp.add_argument(
        "--patch-author",
        default=None,
        metavar="NAME",
        help="Track-changes author attribution. Default: 'andamentum-"
        "whetstone (AI)'. Overriding requires --allow-author-override.",
    )
    patches_grp.add_argument(
        "--allow-author-override",
        action="store_true",
        help="Explicitly authorise a non-default --patch-author value. "
        "Misrepresenting AI-generated edits as a human reviewer's may "
        "constitute research misconduct.",
    )
    patches_grp.add_argument(
        "--patch-report",
        type=Path,
        default=None,
        metavar="PATH",
        help="Optional markdown file prepended as a review-summary section.",
    )

    return parser


def _validate_args(args: argparse.Namespace) -> None:
    """Argparse-time validation that argparse can't express directly.

    Raises ``SystemExit(1)`` with a friendly message on any violation.
    """
    if args.apply_patches is not None:
        _validate_apply_patches_args(args)
        return

    if args.patch_report is not None:
        _die(1, "--patch-report is only valid with --apply-patches.")
    if args.patch_author is not None:
        _die(1, "--patch-author is only valid with --apply-patches.")
    if args.allow_author_override:
        _die(1, "--allow-author-override is only valid with --apply-patches.")

    if not args.model:
        _die(
            1,
            "--model is required.\nExamples:\n"
            "  --model openai:gpt-5.4-nano\n"
            "  --model ollama:gemma4:31b-nvfp4",
        )
    if args.mode == "checklist":
        _reject_review_only_flags(args)
    if args.mode == "panel" and not (
        args.i_am_the_author or os.environ.get("ANDAMENTUM_PANEL_OWN_AUTHOR") == "1"
    ):
        _die(
            1,
            "panel subcommand requires --i-am-the-author. Panel-mode output "
            "is shaped exactly like a journal peer-review report (3-5 "
            "fictional reviewer biosketches + Accept/Reject recommendation). "
            "Affirm that the document being reviewed is your own draft, not "
            "a manuscript shared with you confidentially. Pass "
            "--i-am-the-author, or set ANDAMENTUM_PANEL_OWN_AUTHOR=1.",
        )
    # --criteria and --guidelines are mutually exclusive — both route to
    # the unified criterion-input surface (Phase C); the API itself
    # enforces this with a ValueError, but we surface a friendlier
    # message earlier here.
    if args.criteria and args.guidelines:
        _die(
            1,
            "--criteria and --guidelines are mutually exclusive. Pass one "
            "or the other — both are routes to the same active-criteria slot.",
        )
    if args.rounds < 1:
        _die(1, "--rounds must be at least 1.")
    if args.n_experts < 1:
        _die(1, "--n-experts must be at least 1.")
    for out in args.out:
        ext = out.suffix.lower()
        if ext not in _SUPPORTED_OUT_EXTENSIONS:
            _die(
                1,
                f"Unsupported output extension {ext!r} for {out}. "
                f"Supported: {', '.join(sorted(_SUPPORTED_OUT_EXTENSIONS))}",
            )


def _reject_review_only_flags(args: argparse.Namespace) -> None:
    """Refuse the review-mode flags under ``checklist``, naming the offender.

    The checklist branch of :func:`_run` passes none of these on: the mode has
    no criterion cascade, no editor pass, no novelty check and no gap loop for
    them to configure. Accepting one and running anyway would report success on
    a run that did not do what was asked — a silent failure, which this project
    treats as worse than a loud one.

    Detection is by comparison against the flag's default (see
    ``_REVIEW_ONLY_FLAGS``), so passing a flag its own default value is
    tolerated: that run genuinely is the run the user asked for.
    """
    for dest, flag, unset_value in _REVIEW_ONLY_FLAGS:
        if getattr(args, dest) != unset_value:
            _die(
                1,
                f"{flag} is a review-mode flag and has no effect under "
                "`checklist`, so the run is refused rather than run without "
                "it. Checklist mode takes only --model, --document-type and "
                "--out: the probe bank is fixed, which is what keeps two runs "
                "comparable. Drop the flag, or use "
                "`andamentum-whetstone review` instead.",
            )


def _validate_apply_patches_args(args: argparse.Namespace) -> None:
    """Validate the patch-only path. ``--apply-patches`` is set."""
    input_path = Path(args.input)
    if _is_url(args.input) or not input_path.exists():
        _die(
            1,
            "--apply-patches requires INPUT to be a local .docx file "
            f"(got {args.input!r}).",
        )
    if input_path.suffix.lower() != ".docx":
        _die(
            1,
            "--apply-patches requires INPUT to be a .docx file "
            f"(got {input_path.suffix!r}).",
        )
    if len(args.out) != 1 or args.out[0].suffix.lower() != ".docx":
        _die(
            1,
            "--apply-patches requires exactly one --out FILE ending in .docx.",
        )
    if not args.apply_patches.exists():
        _die(1, f"patches file not found: {args.apply_patches}")
    if args.patch_report is not None and not args.patch_report.exists():
        _die(1, f"patch report file not found: {args.patch_report}")
    if args.patch_author is not None and not args.allow_author_override:
        _die(
            1,
            "--patch-author overrides the AI-attribution default — "
            "pass --allow-author-override to confirm you intend this. "
            "Attributing AI-generated edits to a human name may "
            "constitute research misconduct under most institutional codes.",
        )
    if args.allow_author_override and args.patch_author is None:
        _die(1, "--allow-author-override requires --patch-author NAME.")


def _die(code: int, message: str) -> NoReturn:
    """Print to stderr and exit with the given code."""
    print(f"error: {message}", file=sys.stderr)
    sys.exit(code)


def _resolve_guidelines(value: str) -> str:
    """Resolve --guidelines value, expanding ``@path`` to file contents.

    A leading ``@`` selects file mode; everything else is treated as a
    literal guidelines string.

    Raises
    ------
    SystemExit
        With exit code 2 if the file cannot be read.
    """
    if value.startswith("@"):
        path = Path(value[1:])
        try:
            return path.read_text(encoding="utf-8")
        except OSError as exc:
            _die(2, f"could not read guidelines file {path}: {exc}")
    return value


def _parse_criteria(values: list[str]) -> list[str]:
    """Parse ``--criteria`` into a flat list of criterion strings.

    Each ``--criteria`` invocation may be a semicolon-separated string;
    multiple invocations stack. Empty / whitespace-only entries are
    silently dropped.
    """
    out: list[str] = []
    for raw in values:
        for chunk in raw.split(";"):
            cleaned = chunk.strip()
            if cleaned:
                out.append(cleaned)
    return out


async def _run(args: argparse.Namespace, console: Console) -> None:
    """Execute the review and write all requested outputs."""
    from .renderers import render_docx, render_html, render_markdown
    from .v3 import Criterion, review_document_v3
    from .v3.checklist import checklist_document_v3
    from .v3.panel import run_panel_v3

    editor_criteria = tuple(
        c.strip() for c in args.editor_criteria.split(",") if c.strip()
    )
    panel_disciplines = tuple(
        d.strip() for d in args.panel_disciplines.split(",") if d.strip()
    )
    guidelines_text = _resolve_guidelines(args.guidelines) if args.guidelines else ""

    # Always-on one-line scope banner — short, unmissable, not gated on
    # --verbose. Goes to stderr so it doesn't pollute piped output.
    print(
        "Note: whetstone is for your own drafts. Not a peer-review tool. "
        "See RESPONSIBLE_USE.md.",
        file=sys.stderr,
    )

    logger = logging.getLogger(_LOGGER_NAME)
    if args.verbose:
        _log_run_config(console, args, editor_criteria)

    try:
        _t0 = time.perf_counter()
        if args.mode == "panel":
            # Panel mode runs its own graph (different shape — multi-
            # expert fan-out + synthesis); criterion-cascade kwargs do
            # not apply.
            from .v3.graph import _harvest_or_treat_as_markdown

            md = await _harvest_or_treat_as_markdown(args.input)
            result = await run_panel_v3(
                md,
                model=args.model,
                n_experts=args.n_experts,
                panel_disciplines=list(panel_disciplines) or None,
                # The panel subcommand already required --i-am-the-author
                # (gated above) — that affirmation is what disarms the
                # confidentiality tripwire, mirroring review's
                # --confirm-own-draft. Library callers of run_panel_v3 that
                # omit confirm_own_draft still get the marker scan.
                confirm_own_draft=True,
            )
        elif args.mode == "checklist":
            # Checklist mode runs its own graph (one quota-driven probe per
            # defect class + a deterministic coverage table); the
            # criterion-cascade kwargs do not apply. The confidentiality
            # tripwire does — --confirm-own-draft is threaded through exactly
            # as it is for review, and unlike panel there is no mandatory
            # authorship affirmation, because checklist output is a defect
            # table rather than a simulated peer-review report.
            result = await checklist_document_v3(
                args.input,
                model=args.model,
                document_type=args.document_type,
                confirm_own_draft=args.confirm_own_draft,
            )
        else:
            # Default review path — Phase C unified criteria/guidelines
            # input is mutually exclusive (CLI validator enforces).
            v3_criteria: list[Criterion] | None = None
            v3_guidelines: str | None = None
            if args.criteria:
                parsed = _parse_criteria(args.criteria)
                v3_criteria = [Criterion(name=s, questions=[s]) for s in parsed]
            if args.guidelines:
                v3_guidelines = guidelines_text

            result = await review_document_v3(
                args.input,
                model=args.model,
                cap=args.rounds,
                document_type=args.document_type,
                confirm_own_draft=args.confirm_own_draft,
                criteria=v3_criteria,
                guidelines_text=v3_guidelines,
                editor=args.editor,
                editor_criteria=list(editor_criteria) if editor_criteria else None,
                check_novelty=args.check_novelty,
                novelty_search_depth=args.novelty_search_depth,
            )
        result.metrics.wall_seconds = time.perf_counter() - _t0
    except Exception as exc:
        # Confidentiality tripwire — its own error class so we can give a
        # focused error code (1: configuration) instead of pipeline-crash (3).
        from ._confidentiality import ConfidentialityMarkerError

        if isinstance(exc, ConfidentialityMarkerError):
            _die(1, str(exc))
        # Distinguish "couldn't load the input" from "review crashed"
        # so users know whether to fix their input or their config.
        msg = str(exc)
        if any(
            t in msg.lower() for t in ("fetch", "harvest", "no such file", "not found")
        ):
            _die(2, f"could not load input {args.input!r}: {exc}")
        else:
            _die(3, f"review pipeline failed: {exc}")

    # If any output is .docx, prepare a baseline .docx ONCE upfront
    # (rather than per-output) so we don't re-harvest if the user asked
    # for multiple .docx outputs.
    baseline_docx: Path | None = None
    baseline_is_temp = False
    if any(o.suffix.lower() == ".docx" for o in args.out):
        baseline_docx, baseline_is_temp = await _ensure_docx_source(args.input)

    written: list[Path] = []
    try:
        # ── Write each requested output ──────────────────────────────
        for out in args.out:
            ext = out.suffix.lower()
            out.parent.mkdir(parents=True, exist_ok=True)
            model_id = args.model
            if ext == ".md":
                render_markdown(
                    result,
                    out,
                    model=model_id,
                    visible_watermark=args.visible_watermark,
                )
            elif ext == ".html":
                render_html(
                    result,
                    out,
                    model=model_id,
                    visible_watermark=args.visible_watermark,
                )
            elif ext == ".docx":
                assert baseline_docx is not None  # validated above
                patch_result = render_docx(
                    result,
                    source_docx_path=baseline_docx,
                    output_path=out,
                    model=model_id,
                    visible_watermark=args.visible_watermark,
                )
                _warn_on_unplaced_annotations(console, patch_result)
            written.append(out)
            logger.info("[output] wrote %s", out)
    finally:
        # Clean up the auto-generated baseline docx (if we created one).
        if baseline_is_temp and baseline_docx is not None:
            try:
                baseline_docx.unlink()
            except OSError:
                pass

    _print_summary(console, result, written)

    # Disclosure reminder — always shown at end of run. AI assistance must
    # be disclosed in submitted artifacts per most journal / funder rules.
    from ._watermark import DISCLOSURE_REMINDER

    print(f"\nNote: {DISCLOSURE_REMINDER}", file=sys.stderr)


def _log_run_config(
    console: Console,
    args: argparse.Namespace,
    editor_criteria: tuple[str, ...],
) -> None:
    """Render the input/options panel before the run starts."""
    table = Table.grid(padding=(0, 1))
    table.add_column(style="bold cyan", justify="right")
    table.add_column()
    table.add_row("input:", str(args.input))
    table.add_row("model:", args.model or "")
    table.add_row("document type:", args.document_type)
    if args.mode == "panel":
        table.add_row("pipeline:", "panel (multi-expert simulated peer review)")
        table.add_row("n_experts:", str(args.n_experts))
        if args.panel_disciplines:
            table.add_row("disciplines:", args.panel_disciplines)
    elif args.mode == "checklist":
        table.add_row("pipeline:", "checklist (probe bank + coverage table)")
    else:
        table.add_row("pipeline:", "review (criterion cascade + gap loop)")
        if args.criteria:
            criteria_preview = "; ".join(args.criteria)
            table.add_row("criteria:", criteria_preview)
        elif args.guidelines:
            guideline_preview = (
                f"@{args.guidelines[1:]}"
                if args.guidelines.startswith("@")
                else f"<{len(args.guidelines)} chars inline>"
            )
            table.add_row("guidelines:", guideline_preview)
        table.add_row("editor:", "on" if args.editor else "off")
        if args.editor:
            table.add_row("editor criteria:", ", ".join(editor_criteria))
        table.add_row("novelty check:", "on" if args.check_novelty else "off")
        table.add_row("gap-loop cap:", str(args.rounds))
    table.add_row("outputs:", ", ".join(str(o) for o in args.out))
    console.print(Panel(table, title="whetstone", border_style="cyan"))


def _print_summary(console: Console, result, written: list[Path]) -> None:
    """Render a Rich summary table after the run completes."""
    m = result.metrics
    table = Table(show_header=False, box=None, padding=(0, 1))
    table.add_column(style="bold cyan", justify="right")
    table.add_column()
    table.add_row("wall time:", f"{m.wall_seconds:.1f}s")
    table.add_row("LLM calls:", str(m.llm_calls))
    table.add_row("sections:", str(m.sections_processed))
    # v3 reports criterion-cascade findings as "investigated"; there is
    # no separate deterministic / challenged column anymore.
    table.add_row("findings:", str(m.investigated_findings_count))
    table.add_row("edits:", str(m.edits_count))
    table.add_row("author questions:", str(len(result.author_questions)))
    # Gap-loop rounds are stored on the same metric field v2 used for
    # the reflection loop — the underlying mechanism is similar
    # ("re-examine until no new demands or cap hit").
    if m.reflection_rounds_used:
        table.add_row("gap-loop rounds:", str(m.reflection_rounds_used))
    table.add_row("outputs:", "\n".join(str(p) for p in written))
    console.print(Panel(table, title="review complete", border_style="green"))


async def _ensure_docx_source(input_arg: str) -> tuple[Path, bool]:
    """Return ``(path, is_temp)`` for a .docx suitable as track-changes baseline.

    If the input is already a local .docx, return its Path with
    ``is_temp=False``. Otherwise harvest the input into markdown and
    write a clean baseline .docx via python-docx; return that path with
    ``is_temp=True`` so the caller knows to clean it up.
    """
    input_path = Path(input_arg) if not _is_url(input_arg) else None
    if input_path is not None and input_path.suffix.lower() == ".docx":
        return input_path, False

    logger = logging.getLogger(_LOGGER_NAME)
    logger.info("[baseline] input is not .docx — generating from harvested text")

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


# ── Subcommand front-end (Phase F) ──────────────────────────────────────────
#
# The CLI is internally a single flat argparse parser (the canonical surface
# tests build against), with a thin subcommand alias layer in front of it.
# Recognised verbs:
#
#   review         — default; the bare positional invocation also routes here
#                    (`andamentum-whetstone draft.md --model X --out r.md`)
#   panel          — equivalent to `--mode panel`
#   checklist      — equivalent to `--mode checklist`
#   proofread      — shells out to `andamentum.proofread` (no whetstone review
#                    pipeline at all; the two are intentionally disjoint)
#   apply-patches  — equivalent to `--apply-patches PATH`; INPUT and PATCHES
#                    become positional args in the subcommand form
#
# The flat-flag form keeps working for every existing test + script. Users
# who want the subcommand UX get it. v2's mode flag stays callable until
# Phase I deletes v2.

_KNOWN_SUBCOMMANDS: frozenset[str] = frozenset(
    {"review", "panel", "checklist", "proofread", "apply-patches"}
)
# ``proofread`` and ``verify-provenance`` short-circuit the main flat-argparse
# pipeline entirely (see ``main()`` below); they have their own argv handlers
# and never reach _build_parser(). All other subcommands go through
# _rewrite_subcommand().


def _rewrite_subcommand(argv: list[str]) -> list[str]:
    """Translate a subcommand-style invocation into the underlying flat-CLI
    argv. Returns ``argv`` unchanged when no subcommand verb is present
    (back-compat for the existing bare-positional form)."""
    if not argv:
        return argv
    head = argv[0]
    rest = argv[1:]
    if head == "review":
        # `review draft.md --model X --out r.md`  →  `draft.md --model X --out r.md`
        return rest
    if head == "panel":
        # `panel draft.md ...`  →  `draft.md --mode panel ...`
        if not rest or rest[0].startswith("-"):
            return ["--mode", "panel", *rest]
        return [rest[0], "--mode", "panel", *rest[1:]]
    if head == "checklist":
        # `checklist draft.md ...`  →  `draft.md --mode checklist ...`
        # Same shape as the panel branch, including the leading-flag case
        # (`checklist --model X draft.md`), where there is no positional to
        # keep in front and argparse sorts the order out itself.
        if not rest or rest[0].startswith("-"):
            return ["--mode", "checklist", *rest]
        return [rest[0], "--mode", "checklist", *rest[1:]]
    if head == "apply-patches":
        # `apply-patches draft.docx --patches p.json --out r.docx`  →
        # `draft.docx --apply-patches p.json --out r.docx`
        # The subcommand requires the patches as a positional or via --patches.
        if not rest or rest[0].startswith("-"):
            _die(
                1,
                "apply-patches subcommand: positional INPUT required.\n"
                "Usage: andamentum-whetstone apply-patches INPUT.docx "
                "--patches PATCHES.json --out OUTPUT.docx",
            )
        out: list[str] = [rest[0]]
        skip_next = False
        for i, tok in enumerate(rest[1:], start=1):
            if skip_next:
                skip_next = False
                continue
            if tok == "--patches":
                if i + 1 >= len(rest):
                    _die(1, "apply-patches: --patches needs a value.")
                out.extend(["--apply-patches", rest[i + 1]])
                skip_next = True
            else:
                out.append(tok)
        return out
    return argv


def _run_proofread_subcommand(argv: list[str]) -> int:
    """The proofread subcommand is a thin shell over ``andamentum.proofread.cli``.
    Whetstone's idea-review pipeline and the deterministic style check are
    intentionally separate jobs (see the project memory entry on this).
    Passes through stdin / file / URL handling unchanged."""
    from andamentum.proofread.cli import main as proofread_main

    return proofread_main(argv)


def _run_verify_provenance_subcommand(argv: list[str]) -> int:
    """Report which AI-provenance markers are present in a .docx file.

    Usage:
        andamentum-whetstone verify-provenance INPUT.docx [--format human|json]

    Exit codes:
        0 — at least one provenance marker found
        1 — file unreadable / not a .docx zip
        2 — readable but no markers found
    """
    sub = argparse.ArgumentParser(
        prog="andamentum-whetstone verify-provenance",
        description=(
            "Inspect a .docx and report which whetstone provenance markers "
            "are present. Read-only — never modifies the input file. Useful "
            "for editors and integrity teams checking whether a file was "
            "produced or modified by whetstone."
        ),
    )
    sub.add_argument("input", type=Path, help="Path to a .docx file.")
    sub.add_argument(
        "--format",
        choices=("human", "json"),
        default="human",
        help="Report format. Default: human.",
    )
    parsed = sub.parse_args(argv)

    from ._watermark import read_provenance_markers

    markers = read_provenance_markers(parsed.input)
    customxml = markers.get("customxml_provenance")
    found_any = bool(
        markers.get("core_properties_marker")
        or markers.get("core_properties_author_marker")
        or customxml
    )

    if parsed.format == "json":
        import json as _json

        print(_json.dumps(markers, indent=2, default=str))
    else:
        # Human-readable summary. Plain ASCII so it pipes cleanly.
        print(f"File: {parsed.input}")
        if not markers.get("readable"):
            print("  Readable: NO (not a valid .docx zip)")
            return 1
        print("  Readable: yes")
        print(
            "  Core-properties keywords/comments marker: "
            f"{'present' if markers.get('core_properties_marker') else 'absent'}"
        )
        print(
            "  Core-properties author marker:           "
            f"{'present' if markers.get('core_properties_author_marker') else 'absent'}"
        )
        if isinstance(customxml, dict) and customxml:
            print("  customXml provenance part: present")
            for key in ("generator", "version", "model", "produced-at", "ai-generated"):
                if key in customxml:
                    print(f"    {key}: {customxml[key]}")
        else:
            print("  customXml provenance part: absent")
        print()
        print(
            "Verdict: "
            + (
                "whetstone provenance found"
                if found_any
                else "no whetstone provenance markers found"
            )
        )

    if not markers.get("readable"):
        return 1
    return 0 if found_any else 2


def main(argv: Sequence[str] | None = None) -> None:
    """Entry point — invoked by the ``andamentum-whetstone`` script.

    Accepts either the bare-positional form (``andamentum-whetstone draft.md
    --model X --out r.md``) or the subcommand form (``andamentum-whetstone
    review draft.md ...`` / ``... panel ...`` / ``... checklist ...`` /
    ``... proofread draft.md`` / ``... apply-patches draft.docx --patches
    p.json --out r.docx`` / ``... verify-provenance reviewed.docx``)."""
    raw = list(sys.argv[1:] if argv is None else argv)
    if raw and raw[0] == "proofread":
        # proofread short-circuits the whetstone pipeline entirely
        rc = _run_proofread_subcommand(raw[1:])
        raise SystemExit(rc)
    if raw and raw[0] == "verify-provenance":
        # verify-provenance is a read-only inspection: no LLM, no
        # pipeline, no harvest. Short-circuit the main argparse entirely.
        rc = _run_verify_provenance_subcommand(raw[1:])
        raise SystemExit(rc)

    if raw and raw[0] in _KNOWN_SUBCOMMANDS:
        raw = _rewrite_subcommand(raw)

    parser = _build_parser()
    args = parser.parse_args(raw)
    _validate_args(args)

    console = Console(stderr=True)
    _configure_logging(console, args.verbose)

    try:
        if args.apply_patches is not None:
            _apply_patches_only(args, console)
        else:
            asyncio.run(_run(args, console))
    except KeyboardInterrupt:
        _die(1, "interrupted")


def _apply_patches_only(args: argparse.Namespace, console: Console) -> None:
    """Patch-only path: load JSON patches and apply to INPUT.docx → OUT.docx.

    No LLM, no review pipeline. Reuses ``finalize_reviewed_document`` so
    the output is byte-identical to what the full review pipeline would
    produce given the same patches.
    """
    import json

    from .docx.finalization import finalize_reviewed_document
    from .models import DocumentPatch

    raw = json.loads(args.apply_patches.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        _die(
            1,
            f"patches file must contain a JSON array, got {type(raw).__name__}.",
        )

    try:
        patches = [DocumentPatch(**p) for p in raw]
    except Exception as exc:
        _die(1, f"invalid patch in {args.apply_patches}: {exc}")

    review_summary = ""
    if args.patch_report is not None:
        review_summary = args.patch_report.read_text(encoding="utf-8")

    output_path: Path = args.out[0]
    output_path.parent.mkdir(parents=True, exist_ok=True)

    from .renderers.docx import DEFAULT_AI_AUTHOR

    resolved_author = (
        args.patch_author if args.patch_author is not None else DEFAULT_AI_AUTHOR
    )
    if args.allow_author_override:
        print(
            f"WARNING: attributing AI-generated patches to '{resolved_author}' "
            f"instead of the default '{DEFAULT_AI_AUTHOR}'. Misrepresenting "
            f"AI work as human authorship may constitute research misconduct.",
            file=sys.stderr,
        )

    try:
        _, patch_result = finalize_reviewed_document(
            original_file_path=Path(args.input),
            patches=patches,
            review_summary=review_summary,
            issues_count=len(patches),
            output_path=output_path,
            author=resolved_author,
            use_patch_authors=False,
        )
        # Apply invisible AI-provenance metadata so the modified manuscript
        # carries discoverable AI authorship even without a visible banner.
        from ._watermark import stamp_docx_core_properties

        stamp_docx_core_properties(output_path, model=None)
    except Exception as exc:
        _die(3, f"patch application failed: {exc}")

    _print_apply_patches_summary(console, patch_result, output_path)


def _warn_on_unplaced_annotations(console: Console, result) -> None:
    """Say what a .docx run failed to place, and whether it was lost.

    ``render_docx`` returns the patch-application result and this used to be
    discarded, which made a partial write indistinguishable from a clean one:
    on a real manuscript, 7 findings produced 2 comments and the run reported
    success.

    The two kinds of failure are reported apart because only one of them loses
    anything. A comment carries a finding, and every finding is also listed in
    the prepended report (``renderers.docx._findings_as_issues``), so an
    unplaced comment is a thinner margin and nothing more. An edit exists only
    as its tracked change: the report lists findings, never edits, so a dropped
    edit leaves no trace at all. Reporting both under one "all findings are
    still listed" line told the author nothing was lost when a rewrite was.
    """
    if result is None or not result.failed_patches:
        return

    lost_edits = [p for p in result.failed_patches if p.patch_type == "text_edit"]
    unplaced_comments = [p for p in result.failed_patches if p.patch_type == "comment"]
    conflicts = getattr(result, "conflict_failures", 0)

    if unplaced_comments:
        console.print(
            f"[yellow]note:[/yellow] {len(unplaced_comments)} of "
            f"{result.total_patches} comment(s) could not be anchored in the "
            f"text — their quotes have no exact match in the .docx. Those "
            f"findings are still listed in the review report at the top of the "
            f"document."
        )

    if lost_edits:
        console.print(
            f"[yellow]warning:[/yellow] {len(lost_edits)} suggested edit(s) "
            f"were NOT applied and appear nowhere in the .docx"
            + (f" ({conflicts} dropped as overlapping)" if conflicts else "")
            + ". Edits are not listed in the review report — use a markdown or "
            "HTML output as well to see every proposed rewrite:"
        )
        for patch in lost_edits:
            console.print(f"  • {(patch.text_pattern or '')[:70]}")


def _print_apply_patches_summary(console: Console, result, output_path: Path) -> None:
    """Render a Rich summary table after a patch-only run."""
    table = Table(show_header=False, box=None, padding=(0, 1))
    table.add_column(style="bold cyan", justify="right")
    table.add_column()
    table.add_row("total patches:", str(result.total_patches))
    table.add_row(
        "applied:",
        f"{result.applied_patches} "
        f"({result.applied_edits} edits, {result.applied_comments} comments)",
    )
    table.add_row("failed:", str(len(result.failed_patches)))
    table.add_row("success rate:", f"{result.success_rate:.1f}%")
    table.add_row("output:", str(output_path))
    border = "yellow" if result.failed_patches else "green"
    console.print(Panel(table, title="patches applied", border_style=border))

    if result.failed_patches:
        console.print(
            f"[yellow]warning:[/yellow] {len(result.failed_patches)} "
            f"patch(es) could not be applied:"
        )
        for p in result.failed_patches:
            label = (p.text_pattern or p.patch_type)[:60]
            console.print(f"  • [{p.patch_type}] {label}")


def _configure_logging(console: Console, verbose: bool) -> None:
    """Install a RichHandler on the v2 logger.

    INFO when --verbose, WARNING otherwise so warnings/errors still surface
    even on quiet runs. We attach to our own namespace logger (not root)
    so we don't accidentally hijack pydantic-ai/httpx noise.
    """
    logger = logging.getLogger(_LOGGER_NAME)
    # Avoid duplicate handlers if main() is called twice in one process.
    for h in list(logger.handlers):
        logger.removeHandler(h)
    logger.setLevel(logging.INFO if verbose else logging.WARNING)
    handler = RichHandler(
        console=console,
        show_time=verbose,
        show_path=False,
        show_level=verbose,
        markup=False,
        rich_tracebacks=True,
    )
    handler.setLevel(logging.INFO if verbose else logging.WARNING)
    logger.addHandler(handler)
    logger.propagate = False


if __name__ == "__main__":  # pragma: no cover
    main()
