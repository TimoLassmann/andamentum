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
import tempfile
import time
from pathlib import Path
from typing import Any, NoReturn, Sequence

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

Whetstone reviews the text of the document it annotates. A .docx INPUT
is read directly; anything else is harvested to markdown and written out
as a clean baseline .docx FIRST, and the review then reads that file.
Either way a quote's offsets are already an address in the output.
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

    # A second round over the first round's output. No round flag: the input
    # file carries round one's changes, and the review reads them as accepted.
    # Give it its own name or Word shows both rounds as one reviewer.
    andamentum-whetstone draft.reviewed.docx --model openai:gpt-5.4-nano \\
        --editor --patch-author "Second pass (AI)" --allow-author-override \\
        --out draft.reviewed2.docx

    # Review with caller-supplied criteria
    andamentum-whetstone draft.md --model openai:gpt-5.4-nano \\
        --criteria "originality; depth of literature; clarity" \\
        --out review.md

    # A named reviewer: its own criteria, probes and stance
    andamentum-whetstone draft.md --model openai:gpt-5.4-nano \\
        --reviewer methods --out review.md

    # Which reviewers exist, and which suit this kind of document
    andamentum-whetstone --list-reviewers --document-type academic

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
#
# --editor-criteria is deliberately NOT in this list: its default is None, the
# sentinel meaning "the user did not say". A default of "clarity,concision,
# grammar" here would make the CLI pass an explicit editor-criteria list on
# every run, and an explicit list is an inline override — so a reviewer
# profile's own editor criteria would be silently replaced by a value nobody
# typed. The editor's default list lives in one place, `editor
# .DEFAULT_EDITOR_CRITERIA`, and the run falls through to it.
_DEFAULT_NOVELTY_SEARCH_DEPTH = 2
_DEFAULT_ROUNDS = 3

# One list of document types, read by the main parser and by the
# --list-reviewers short-circuit in main(), which builds a parser of its own.
# Two lists would let a typo pass the second parser and print a fit column
# against a type the first one refuses.
_DOCUMENT_TYPE_CHOICES: tuple[str, ...] = (
    "auto",
    "academic",
    "external_communication",
    "essay",
    "tutorial",
    "creative",
    "general",
)

# (namespace attribute, user-facing flag, value meaning "not set").
_REVIEW_ONLY_FLAGS: tuple[tuple[str, str, object], ...] = (
    ("criteria", "--criteria", None),
    ("guidelines", "--guidelines", ""),
    ("editor", "--editor", False),
    ("editor_criteria", "--editor-criteria", None),
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
        choices=_DOCUMENT_TYPE_CHOICES,
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
    # Shared by review and checklist for the same reason --document-type is:
    # one reviewer selects a criterion set in one mode and a probe bank in the
    # other, and a second flag would make one question look like two. Panel and
    # apply-patches reject it in _validate_args rather than ignoring it.
    #
    # The choices come from the profile registry, so a profile added there is
    # offered here with no second list to update. Validation is
    # _parse_reviewers, not argparse choices, because the value may be a list.
    from .v3.reviewers import DEFAULT_PROFILE_NAME

    parser.add_argument(
        "--reviewer",
        # NOT argparse choices: the value may be a comma-separated list, which
        # choices cannot express. Validated by _parse_reviewers, which reports
        # the unknown name and the full list rather than argparse's one-liner.
        default=DEFAULT_PROFILE_NAME,
        metavar="NAME[,NAME...]",
        help=(
            "Who reads the draft. One name, or several separated by commas to "
            "run them in that order — each reviewer reads the previous one's "
            "edits, their tracked changes stack by author, and a run with more "
            "than one ends with a summary reconciling them. Leave it out for "
            f"'{DEFAULT_PROFILE_NAME}', the ordinary review: the criteria that "
            "suit this kind of document, which is what ran before reviewers "
            f"were named. Put '{DEFAULT_PROFILE_NAME}' in the list to have that "
            "review as well as a specialist. Every value a reviewer supplies is "
            "still overridden by the matching flag. Run --list-reviewers to see "
            "them all."
        ),
    )
    parser.add_argument(
        "--list-reviewers",
        action="store_true",
        help=(
            "List the reviewers, with a fit marker per document type when "
            "--document-type is given, and exit. Needs no input file and no "
            "model."
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
    # Shared by every path that writes a .docx. A second review round over the
    # first round's output is just another run with another name here — the
    # input file already carries round one's changes, so there is no round
    # number to pass and no sequential mode to switch on.
    parser.add_argument(
        "--patch-author",
        default=None,
        metavar="NAME",
        help="Track-changes author attribution. Default: 'andamentum-"
        "whetstone (AI)'. Overriding requires --allow-author-override. "
        "Give a second review round a different name to tell the two "
        "apart in Word's reviewing pane; the same name twice is valid "
        "but indistinguishable there.",
    )
    parser.add_argument(
        "--allow-author-override",
        action="store_true",
        help="Explicitly authorise a non-default --patch-author value. "
        "Misrepresenting AI-generated edits as a human reviewer's may "
        "constitute research misconduct.",
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
        # The HTML renderer writes no provenance meta tag — it never has, and
        # the help text used to say it did. What it does write, banner or not,
        # is the AI-content disclaimer callout.
        help="Suppress the visible banner. Invisible provenance metadata "
        "(docx core properties and customXml) is written regardless, and the "
        "HTML report keeps a short disclaimer callout.",
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
        default=None,
        metavar="LIST",
        help="Comma-separated editor criteria. Default: clarity,concision,"
        "grammar — or the reviewer's own, when --reviewer names one that "
        "sets them.",
    )
    review_grp.add_argument(
        "--check-novelty",
        action="store_true",
        help="VERIFY the novelty claims against the literature, by searching "
        "the web. Every review already LISTS what the draft claims as new — "
        "that costs one call to the model already reading your draft and "
        "discloses nothing further. This flag takes those sentences and sends "
        "them to a search engine, so it is off by default: an unpublished "
        "manuscript should not reach a third party without you deciding it "
        "should. Adds N searches + N judge calls over the claims found.",
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
    if args.list_reviewers:
        # Also handled in main() before parsing, so the listing works without
        # an input file or --out. Reaching it here means the run supplied
        # them anyway: print the listing and stop, rather than reviewing a
        # document the user did not ask to have reviewed yet.
        _print_reviewer_listing(document_type=args.document_type)
        sys.exit(0)

    if args.apply_patches is not None:
        _validate_apply_patches_args(args)
        return

    if args.patch_report is not None:
        _die(1, "--patch-report is only valid with --apply-patches.")
    _validate_author_args(args)

    if not args.model:
        _die(
            1,
            "--model is required.\nExamples:\n"
            "  --model openai:gpt-5.4-nano\n"
            "  --model ollama:gemma4:31b-nvfp4",
        )
    # Parsed here rather than at run time: an unknown reviewer should cost you
    # nothing, and by run time the document has been loaded and, for a URL,
    # fetched. The parsed list is kept so the run does not parse it twice.
    args.reviewers = _parse_reviewers(args.reviewer)
    if args.mode == "checklist":
        _reject_review_only_flags(args)
        _reject_the_line_editor_under_checklist(args)
        if len(args.reviewers) > 1:
            _die(
                1,
                "checklist mode takes one reviewer, not a list. Its output is "
                "a coverage table whose rows come from one probe bank, and two "
                "banks in one run would produce two tables that cannot be "
                "compared with each other or with the last run — which is the "
                "property checklist exists for. Run it once per reviewer, or "
                "use review mode, which stacks them.",
            )
    if args.mode == "review":
        _reject_a_review_that_would_review_nothing(args)
    if args.mode == "panel" and args.reviewer != _default_reviewer_name():
        # Detected by comparison against the default, matching
        # _reject_review_only_flags: `--reviewer default` under panel asks for
        # nothing panel does not already do, so it is tolerated.
        _die(
            1,
            "panel generates its own reviewers; --reviewer applies to review "
            "and checklist.",
        )
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
                "it. Checklist mode takes only --model, --document-type, "
                "--out and the author flags: the probe bank is fixed, which "
                "is what keeps two runs comparable. Drop the flag, or use "
                "`andamentum-whetstone review` instead.",
            )


def _default_reviewer_name() -> str:
    """The name of the reviewer that runs when none is asked for.

    Read from the profile registry rather than written here, for the reason
    the comment above ``_REVIEW_ONLY_FLAGS`` gives: the rejections below detect
    "was this passed" by comparing against the default, which is sound only
    while there is one definition of it.
    """
    from .v3.reviewers import DEFAULT_PROFILE_NAME

    return DEFAULT_PROFILE_NAME


def _reject_the_line_editor_under_checklist(args: argparse.Namespace) -> None:
    """The one reviewer checklist mode cannot run.

    The line editor selects no probes — it is a prose pass, and its whole
    output is tracked changes the editor produces under ``--editor`` in review
    mode. Running it here would produce a coverage table with no rows, which
    reads like a document that passed every check rather than like a run that
    checked nothing.
    """
    from .v3.reviewers import PROFILES

    for name in getattr(args, "reviewers", None) or [args.reviewer]:
        profile = PROFILES[name]
        if profile.probes == ():
            _die(
                1,
                f"the {profile.identity.lower()} runs no checks; use review "
                "mode with --editor.",
            )


def _reject_a_review_that_would_review_nothing(args: argparse.Namespace) -> None:
    """A review with no criteria and no editor makes no judgement at all.

    The mirror of :func:`_reject_the_line_editor_under_checklist`, and refused
    for the same reason: a reviewer that selects no criteria has nothing for
    the criterion cascade to run and nothing for the gap loop to re-ground, so
    without ``--editor`` the run does claim extraction, synthesis and
    self-critique, costs model calls for all three, and produces a report with
    no findings and no edits — which reads like a draft that passed rather
    than like a run that examined nothing.

    Stated as "does this run examine anything" rather than as the name of a
    profile, so a later prose-only reviewer is covered without an edit here.
    Inline ``--criteria`` and ``--guidelines`` both override a profile's
    criteria, so either of them makes the run a real review again.
    """
    from .v3.reviewers import PROFILES

    names = getattr(args, "reviewers", None) or [args.reviewer]
    # A list is a real review as long as ONE of its reviewers examines
    # something: `default,editor` is a review followed by a prose pass, which
    # is a sensible thing to ask for.
    if any(PROFILES[n].criteria != () for n in names):
        return
    if args.criteria or args.guidelines or args.editor:
        return
    named = ", ".join(PROFILES[n].identity.lower() for n in names)
    _die(
        1,
        f"the {named} runs no review criteria, so without --editor this run "
        "would examine nothing and report nothing. Add --editor to get the "
        "tracked changes it exists for, name a reviewer that runs checks, or "
        "pass --criteria of your own.",
    )


def _validate_apply_patches_args(args: argparse.Namespace) -> None:
    """Validate the patch-only path. ``--apply-patches`` is set."""
    if args.reviewer != _default_reviewer_name():
        # apply-patches never calls the review pipeline, so a reviewer here
        # would configure nothing. Refused rather than ignored: a flag that
        # silently does nothing is the failure this project treats as worse
        # than a loud one.
        _die(
            1,
            "--reviewer selects a review configuration; apply-patches applies "
            "patches you supply.",
        )
    input_path = Path(args.input)
    if not input_path.exists():
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
    _validate_author_args(args)


def _validate_author_args(args: argparse.Namespace) -> None:
    """The author flags pair up the same way on every path that writes a .docx.

    Review mode takes ``--patch-author`` too, because a second reviewer needs a
    second name: round two opens round one's output, reads it as Word will show
    it with the changes accepted, and writes its own tracked changes beside
    them. One name for both rounds is legal and produces a valid document — it
    just cannot be told apart in Word's reviewing pane.

    Which is exactly why the misconduct rail stays on both paths: a second name
    is more tempting to make a human one, not less.
    """
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
    if args.patch_author is not None and not any(
        out.suffix.lower() == ".docx" for out in args.out
    ):
        # An author is a property of tracked changes, and only .docx carries
        # them. Accepting the flag and writing nowhere it applies would report
        # success on a run that ignored what was asked for.
        _die(
            1,
            "--patch-author names the author of tracked changes, so it needs "
            "a .docx output. Add --out FILE.docx, or drop the flag.",
        )


def _resolve_author(args: argparse.Namespace) -> str:
    """The tracked-change author, and the warning that overriding it earns.

    A named reviewer signs its own changes — "Methods reviewer
    (andamentum-whetstone, AI)" — which is what lets two reviewers' rounds be
    told apart in Word's reviewing pane. The composition itself lives in
    ``reviewers.author_for``, which the pipeline calls too, so the report
    header and the tracked changes cannot name different authors. A profile
    supplies only the identity half and can never drop the machine-authorship
    marker; the one path that writes an unmarked name is ``--patch-author``,
    still gated behind ``--allow-author-override``.
    """
    from .v3.reviewers import PROFILES, author_for

    # The LAST reviewer in the list signs the final render. Each earlier round
    # already signed its own pass on the way through (_carry_forward), so this
    # is the one still unwritten — and a list whose final entry is the line
    # editor should be signed by the line editor.
    names = getattr(args, "reviewers", None) or _parse_reviewers(args.reviewer)
    default_author = author_for(profile=PROFILES[names[-1]])
    if args.patch_author is None:
        return default_author
    print(
        f"WARNING: attributing AI-generated changes to '{args.patch_author}' "
        f"instead of the default '{default_author}'. Misrepresenting "
        f"AI work as human authorship may constitute research misconduct.",
        file=sys.stderr,
    )
    return args.patch_author


_RECOMMENDED_SEQUENCE = """\
Recommended sequence for a manuscript:

  General first; it is free. Then claims, methods and adversary — the three
  that change what goes out. Fix. Then reader, before anything leaves the
  group. Editor last, because prose that will still change is wasted effort
  to edit. Anything two reviewers raise independently goes to the top of the
  list."""


def _print_reviewer_listing(*, document_type: str = "auto") -> None:
    """Print every reviewer, one line each, and how to sequence them.

    The fit column appears only when the document type was actually stated.
    Under ``--document-type auto`` the type is decided by a classifier during
    the run, and printing a fit marker against a type nobody has determined
    yet would be an answer to a question that has not been asked.
    """
    from .v3.reviewers import DEFAULT_PROFILE_NAME, PROFILES, fits

    known_type = document_type != "auto"
    labels = {
        name: name + (" (default)" if name == DEFAULT_PROFILE_NAME else "")
        for name in PROFILES
    }
    width = max(len(label) for label in labels.values())
    print("Reviewers — pass one to --reviewer NAME:\n")
    if known_type:
        print(f"  {'reviewer':<{width}}  suits {document_type}?  what it does")
    for name, profile in PROFILES.items():
        fit = ""
        if known_type:
            suits = (
                "yes" if fits(profile=profile, document_type=document_type) else "no"
            )
            # Width of the "suits <type>?" heading above, so the column of
            # answers lines up under the question it answers.
            fit = f"  {suits:<{len(document_type) + 7}}"
        print(f"  {labels[name]:<{width}}{fit}  {profile.description}")
    print()
    if known_type:
        print(
            "A reviewer that does not suit this document type still runs when "
            "you ask for it.\nIt warns once in the terminal, and the same line "
            "goes into the report.\n"
        )
    else:
        print(
            "Pass --document-type TYPE as well to see which reviewers suit "
            "that kind of document.\n"
        )
    print(_RECOMMENDED_SEQUENCE)


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


def _reconciliation_markdown(rec: Any) -> str:
    """The reconciliation as the summary prose of the combined result.

    Rendered as markdown rather than as a new atom kind so all three renderers
    carry it without knowing it exists — the summary is prose in every one of
    them already.
    """
    out: list[str] = []
    if rec.verdict:
        out += [rec.verdict, ""]
    if rec.agreed:
        out.append(f"**Raised by more than one reviewer ({len(rec.agreed)})** — "
                   "independent agreement is the strongest signal here, so "
                   "these come first.")
        out.append("")
        for point in rec.agreed:
            who = ", ".join(point.raised_by) if point.raised_by else "several reviewers"
            out.append(f"- {point.issue} *({who})*")
            if point.quote:
                # Models often return the passage already quoted; wrapping it
                # again gives ""like this"".
                bare = point.quote.strip().strip('"').strip("\u201c\u201d")
                out.append(f'  - concerning: "{bare}"')
        out.append("")
    if rec.disagreements:
        out.append(f"**Where the reviewers disagree ({len(rec.disagreements)})** — "
                   "both positions are given because choosing between them is "
                   "your judgement, not the tool's.")
        out.append("")
        for clash in rec.disagreements:
            out.append(f"- {clash.about}")
            for position in clash.positions:
                out.append(f"  - {position}")
        out.append("")
    if rec.raised_by_one:
        out.append(f"**Raised by one reviewer ({len(rec.raised_by_one)})**")
        out.append("")
        out += [f"- {line}" for line in rec.raised_by_one]
        out.append("")
    return "\n".join(out).strip()


def _reconciliation_only(reconciliation: Any, *, like: Any) -> Any:
    """The reconciliation with nothing else, for the final .docx render.

    Every round already wrote its findings, comments and tracked changes into
    the document. All that is left to add is the summary across them, so this
    result carries a summary and no findings, no edits and no refusals —
    rendering those again would put a second copy of every comment in the file.

    It prepends like any other report, which means it lands ABOVE the N reports
    already there: the reconciliation is the first thing you read, and each
    reviewer's own report is underneath in the order they ran.
    """
    return like.model_copy(
        update={
            "summary": _reconciliation_markdown(reconciliation)
            if reconciliation is not None
            else "Several reviewers ran; reconciling them failed. Their "
            "reports are below, unchanged.",
            "findings": [],
            "edits": [],
            "declined_findings": [],
            "novelty_claims": [],
            # The summary is across reviewers, so no single reviewer's block
            # belongs at the top of it.
            "reviewer_header": None,
        }
    )


def _combine_reviews(reviews: list[tuple[str, Any]], *, reconciliation: Any, last: Any) -> Any:
    """Several reviewers' results as one, for the renderers.

    The .docx does not need this — each round already wrote itself into the
    file, so the last document carries every reviewer's comments and tracked
    changes. Markdown and HTML render a single result, though, and rendering
    only the last reviewer's would silently drop the others.

    So findings, edits, refusals and novelty claims are concatenated, each
    finding tagged with the reviewer that raised it through the ``perspective``
    field the panel renderers already read. The summary becomes the
    reconciliation, which is why it is the first thing in every output.

    Offsets across rounds are NOT comparable — round two's are addresses in
    round two's document, not round one's. That is harmless here because
    markdown and HTML display a finding's quote rather than seeking to it, and
    the .docx path never uses this object.
    """
    findings = []
    edits = []
    declined = []
    for identity, res in reviews:
        for finding in res.findings:
            findings.append(finding.model_copy(update={"perspective": identity}))
        edits.extend(res.edits)
        declined.extend(res.declined_findings)

    summary = (
        _reconciliation_markdown(reconciliation)
        if reconciliation is not None
        else (
            "Several reviewers ran, and reconciling them failed — their "
            "reports are below, unchanged."
        )
    )
    combined = last.model_copy(
        update={
            "summary": summary,
            # No single reviewer's block: this result is all of them.
            "reviewer_header": None,
            "findings": findings,
            "edits": edits,
            "declined_findings": declined,
            # Claims come from the first reviewer only; the draft's assertions
            # do not change because a second reviewer read it.
            "novelty_claims": reviews[0][1].novelty_claims,
        }
    )
    combined.metrics.llm_calls = sum(r.metrics.llm_calls for _, r in reviews)
    combined.metrics.edits_count = len(edits)
    return combined


async def _carry_forward(
    result: Any,
    *,
    document: Any,
    profile: Any,
    args: argparse.Namespace,
    workdir: Path,
    index: int,
) -> Any:
    """Write one reviewer's pass to disk and reopen it for the next.

    This is the whole of sequential review, and it needs no round counter: the
    file itself says what has happened to it. Reviewer N+1 reads the projection
    of this .docx, which excludes deletions and includes insertions — the text
    as Word shows it with reviewer N's changes accepted — so N+1 reviews the
    improved draft and, when it edits text N inserted, Word nests the two as
    ``<w:ins N><w:del N+1>``, its own markup for exactly that.

    The interim file is written into a temporary directory, not beside the
    user's output. Only the last round's document is theirs; the ones in
    between are working state, and leaving four numbered .docx files in their
    folder would be four files to explain and delete.
    """
    from .renderers.docx import render_docx
    from .reviewed_document import open_source
    from .v3.reviewers import author_for

    interim = workdir / f"round{index + 1}.docx"
    render_docx(
        result,
        document=document,
        output_path=interim,
        # The reviewer's own identity, so Word's reviewing pane separates the
        # passes without the caller naming each one. --patch-author still wins
        # for a single reviewer; naming one author for several rounds would
        # make them indistinguishable in the pane, which is the one thing
        # sequential review exists to avoid.
        author=author_for(profile=profile),
        model=args.model,
        visible_watermark=args.visible_watermark,
    )
    return await open_source(str(interim))


def _parse_reviewers(value: str) -> list[str]:
    """``--reviewer`` into a list of profile names, in the order given.

    Order is significant and the list is a SEQUENCE, not a set: each reviewer
    reads the previous one's edited text, so ``claims,methods`` and
    ``methods,claims`` do not produce the same document. Duplicates are
    rejected rather than silently collapsed — asking for the same reviewer
    twice is a typo every time, and running it twice would cost a full review.

    Validated here rather than by argparse ``choices``, which cannot express a
    comma-separated list, and which reports an unknown value in a form that
    does not say what the alternatives mean.
    """
    from .v3.reviewers import PROFILES

    names = [chunk.strip() for chunk in value.split(",") if chunk.strip()]
    if not names:
        _die(1, "--reviewer needs at least one name. Run --list-reviewers.")
    unknown = [n for n in names if n not in PROFILES]
    if unknown:
        _die(
            1,
            f"unknown reviewer(s): {', '.join(unknown)}. "
            f"Available: {', '.join(PROFILES)}. "
            f"Run --list-reviewers for what each one asks.",
        )
    seen: set[str] = set()
    repeated = [n for n in names if n in seen or seen.add(n)]  # type: ignore[func-returns-value]
    if repeated:
        _die(
            1,
            f"reviewer named more than once: {', '.join(sorted(set(repeated)))}. "
            f"Each reviewer runs a full review, so a repeat is a typo rather "
            f"than a request.",
        )
    return names


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


def _editor_criteria(args: argparse.Namespace) -> list[str] | None:
    """The editor criteria the user typed, or ``None`` when they typed none.

    ``None`` is not the same as the default list, and passing the default list
    on every run would be a bug rather than a convenience: an explicit list is
    an inline override, so it would replace whatever editor criteria the named
    reviewer carries — the ``claims`` reviewer's single "hedging", say — with
    a value the user never asked for. Only what was actually typed is passed
    on; the fall-through to the reviewer's list and then to the editor's own
    default happens in ``reviewers.resolve``.

    A flag given as an empty or all-blank string parses to ``None`` too. The
    editor has no "run with no criteria" state — its prompt substitutes a
    placeholder for an empty list and it proposes edits anyway — so the honest
    reading of ``--editor-criteria ""`` is the same as not passing it.
    """
    if args.editor_criteria is None:
        return None
    parsed = [c.strip() for c in args.editor_criteria.split(",") if c.strip()]
    return parsed or None


async def _run(args: argparse.Namespace, console: Console) -> None:
    """Resolve the input once, then review it and write every output."""
    editor_criteria = _editor_criteria(args)

    # Always-on one-line scope banner — short, unmissable, not gated on
    # --verbose. Goes to stderr so it doesn't pollute piped output.
    print(
        "Note: whetstone is for your own drafts. Not a peer-review tool. "
        "See RESPONSIBLE_USE.md.",
        file=sys.stderr,
    )

    if args.verbose:
        _log_run_config(console, args, editor_criteria)

    # ONE resolution of the input, before any mode runs and whatever the
    # requested outputs are. It yields the text to review together with the
    # .docx that text is the projection of, so the annotated file is provably
    # the file that was reviewed. The two harvests this replaces could
    # genuinely disagree: for an ambiguous HTML page harvest races two
    # extractors and keeps the better-scoring output.
    from .reviewed_document import open_source

    try:
        document = await open_source(args.input)
    except Exception as exc:
        _die(2, f"could not load input {args.input!r}: {exc}")
    for warning in document.warnings:
        console.print(f"[yellow]note:[/yellow] {warning}")

    # A non-.docx source left a generated baseline on disk, and it is this
    # function's to remove — including when the review crashes, which is when
    # the file is least likely to be noticed.
    try:
        await _review_and_write(args, console, document)
    finally:
        document.cleanup()


async def _review_and_write(
    args: argparse.Namespace, console: Console, document
) -> None:
    """Run the requested pipeline over *document* and write every output."""
    from .renderers import render_docx, render_html, render_markdown
    from .v3 import Criterion, review_document_v3
    from .v3.checklist import checklist_document_v3
    from .v3.panel import run_panel_v3
    from .v3.reviewers import PROFILES

    logger = logging.getLogger(_LOGGER_NAME)
    editor_criteria = _editor_criteria(args)
    panel_disciplines = tuple(
        d.strip() for d in args.panel_disciplines.split(",") if d.strip()
    )
    guidelines_text = _resolve_guidelines(args.guidelines) if args.guidelines else ""

    # Where a multi-reviewer run keeps the document between rounds. Temporary
    # because only the LAST round's file is the user's; the ones in between are
    # working state, and leaving four numbered .docx files beside their output
    # would be four files to explain and delete. Held open for the whole run
    # and cleaned in the finally below, because the final render reads the
    # document the last round produced.
    _interim = tempfile.TemporaryDirectory(prefix="whetstone-rounds-")
    interim_dir = Path(_interim.name)

    # Set only by a multi-reviewer review run. None means "render `result`",
    # which is what every other path wants. Declared out here because the
    # render loop below is shared by review, checklist and panel.
    docx_result: Any = None

    try:
        _t0 = time.perf_counter()
        if args.mode == "panel":
            # Panel mode runs its own graph (different shape — multi-
            # expert fan-out + synthesis); criterion-cascade kwargs do
            # not apply.
            result = await run_panel_v3(
                document.text,
                model=args.model,
                n_experts=args.n_experts,
                sections=document.sections,
                confidentiality_text=document.confidentiality_text,
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
                document,
                model=args.model,
                document_type=args.document_type,
                confirm_own_draft=args.confirm_own_draft,
                reviewer=PROFILES[
                    (getattr(args, "reviewers", None) or [args.reviewer])[0]
                ],
                patch_author=args.patch_author,
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

            # One reviewer or several. Several run IN SEQUENCE, each reading
            # the previous one's edited text — which is the only shape that
            # works on a Word file, because two reviewers rewriting the same
            # original sentence would collide with no right answer. So the
            # loop renders an interim .docx between rounds and reopens it;
            # reviewer N+1's projection is reviewer N's output with the
            # changes applied, which is what Word shows a human opening it.
            # _validate parses this and stores it, so the unknown-name error
            # costs nothing. Derived again here for a caller that built the
            # namespace itself (the library tests do), rather than requiring
            # every such caller to know about a derived attribute.
            reviewer_names = getattr(args, "reviewers", None) or _parse_reviewers(
                args.reviewer
            )
            reviews: list[tuple[str, Any]] = []
            for index, reviewer_name in enumerate(reviewer_names):
                profile = PROFILES[reviewer_name]
                if len(reviewer_names) > 1:
                    console.print(
                        f"[dim]reviewer {index + 1} of {len(reviewer_names)}: "
                        f"{profile.identity}[/dim]"
                    )
                result = await review_document_v3(
                    document,
                    model=args.model,
                    cap=args.rounds,
                    document_type=args.document_type,
                    confirm_own_draft=args.confirm_own_draft,
                    criteria=v3_criteria,
                    guidelines_text=v3_guidelines,
                    editor=args.editor,
                    editor_criteria=editor_criteria,
                    # Only the first reviewer lists the novelty claims. The
                    # draft's assertions do not change because a second
                    # reviewer read it, and paying for the same list once per
                    # reviewer would put four copies of it in one document.
                    check_novelty=args.check_novelty and index == 0,
                    novelty_search_depth=args.novelty_search_depth,
                    # The profile is resolved inside the pipeline, not here:
                    # the document type it resolves against is only known once
                    # the classifier has run, and classifying twice to learn it
                    # early would be an LLM call spent on a flag.
                    reviewer=profile,
                    patch_author=args.patch_author,
                )
                reviews.append((profile.identity, result))
                if len(reviewer_names) > 1:
                    # EVERY round writes its own pass, the last one included.
                    # The final render then adds the reconciliation on top of a
                    # document that already carries all N reports — otherwise
                    # the last reviewer's own report is the one the summary
                    # replaces, and it is the only report that goes missing.
                    document = await _carry_forward(
                        result,
                        document=document,
                        profile=profile,
                        args=args,
                        workdir=interim_dir,
                        index=index,
                    )
            result = reviews[-1][1]

            # Several reviewers means one place that reconciles them. Not a
            # flag: asking for several opinions and then being handed four
            # reports to cross-reference by hand is the problem this solves,
            # not a preference. A failed reconciliation costs the summary and
            # nothing else — every reviewer's report is already in the file.
            if len(reviews) > 1:
                from .v3.reconcile import reconcile_reviewers

                reconciliation = await reconcile_reviewers(
                    reviews, model=args.model
                )
                # Two shapes, because the two output families need different
                # things. The .docx already holds every finding, comment and
                # tracked change from all N rounds, so it needs ONLY the
                # reconciliation — rendering the findings again would duplicate
                # every comment in the file. Markdown and HTML render one
                # result and would otherwise show the last reviewer alone.
                docx_result = _reconciliation_only(reconciliation, like=result)
                result = _combine_reviews(
                    reviews, reconciliation=reconciliation, last=result
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

    _print_reviewer_configuration(console, result)

    written: list[Path] = []
    # Resolved once: it prints the override warning, and a run asking for two
    # .docx outputs should not warn twice.
    author = _resolve_author(args)
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
            report = render_docx(
                # A multi-reviewer run hands the .docx only the reconciliation:
                # the document it is written into already carries every round's
                # findings, comments and tracked changes.
                docx_result if docx_result is not None else result,
                document=document,
                output_path=out,
                author=author,
                model=model_id,
                visible_watermark=args.visible_watermark,
            )
            _warn_on_dropped_annotations(console, report)
        written.append(out)
        logger.info("[output] wrote %s", out)

    _print_summary(console, result, written)

    # Disclosure reminder — always shown at end of run. AI assistance must
    # be disclosed in submitted artifacts per most journal / funder rules.
    from ._watermark import DISCLOSURE_REMINDER

    print(f"\nNote: {DISCLOSURE_REMINDER}", file=sys.stderr)


def _log_run_config(
    console: Console,
    args: argparse.Namespace,
    editor_criteria: list[str] | None,
) -> None:
    """Render the input/options panel before the run starts."""
    table = Table.grid(padding=(0, 1))
    table.add_column(style="bold cyan", justify="right")
    table.add_column()
    table.add_row("input:", str(args.input))
    table.add_row("model:", args.model or "")
    table.add_row("document type:", args.document_type)
    # What the reviewer resolves TO is only known after the run (the document
    # type decides it), and is printed then — see _print_reviewer_configuration.
    # This is the name that was asked for.
    table.add_row("reviewer:", args.reviewer)
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
            # Before the run, the only honest answer when the flag was not
            # given is who decides — the reviewer, then the editor's default.
            # What it resolves to is printed afterwards, with its source, by
            # _print_reviewer_configuration.
            table.add_row(
                "editor criteria:",
                ", ".join(editor_criteria)
                if editor_criteria
                else f"from reviewer {args.reviewer}",
            )
        table.add_row("novelty check:", "on" if args.check_novelty else "off")
        table.add_row("gap-loop cap:", str(args.rounds))
    table.add_row("outputs:", ", ".join(str(o) for o in args.out))
    console.print(Panel(table, title="whetstone", border_style="cyan"))


def _print_reviewer_configuration(console: Console, result) -> None:
    """What the named reviewer was actually set to do, and where each value
    came from.

    Printed after the run because that is when it is known: the document type
    a profile resolves against is decided inside the pipeline. Nothing is
    printed for the default reviewer — it carries no header, and a run that
    named no reviewer has nothing to disclose that the report does not already
    say.
    """
    header = getattr(result, "reviewer_header", None)
    if header is None:
        return
    table = Table.grid(padding=(0, 1))
    table.add_column(style="bold cyan", justify="right")
    table.add_column()
    table.add_column(style="dim")
    table.add_row("reviewer:", header.identity, "")
    for line in header.configuration:
        table.add_row(f"{line.label}:", line.value, f"({line.source})")
    if header.fit_warning:
        table.add_row("fit:", f"warning: {header.fit_warning}", "")
    console.print(Panel(table, title="reviewer", border_style="cyan"))


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

    if "--list-reviewers" in raw:
        # Short-circuited before the main parser, which requires an input file
        # and at least one --out: listing the reviewers is a question about the
        # tool, not about a document. --document-type is the only other flag
        # that changes the answer, so it is the only one read here.
        listing = argparse.ArgumentParser(
            prog="andamentum-whetstone --list-reviewers", add_help=False
        )
        listing.add_argument("--list-reviewers", action="store_true")
        # The same choices the main parser enforces, from the same list. Left
        # open, a mistyped type would reach the fit column as an unknown
        # value, match no profile's `fits` set and print "no" against every
        # reviewer including the default one, which fits all six by
        # construction — a confidently wrong answer where the run itself would
        # have refused the flag.
        listing.add_argument(
            "--document-type", choices=_DOCUMENT_TYPE_CHOICES, default="auto"
        )
        parsed, _unused = listing.parse_known_args(raw)
        _print_reviewer_listing(document_type=parsed.document_type)
        raise SystemExit(0)

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

    resolved_author = _resolve_author(args)

    try:
        _, patch_result = finalize_reviewed_document(
            original_file_path=Path(args.input),
            patches=patches,
            review_summary=review_summary,
            issues_count=len(patches),
            output_path=output_path,
            author=resolved_author,
        )
        # Apply invisible AI-provenance metadata so the modified manuscript
        # carries discoverable AI authorship even without a visible banner.
        from ._watermark import stamp_docx_core_properties

        stamp_docx_core_properties(output_path, model=None)
    except Exception as exc:
        _die(3, f"patch application failed: {exc}")

    _print_apply_patches_summary(console, patch_result, output_path)


def _warn_on_dropped_annotations(console: Console, report) -> None:
    """Say what a .docx run did not write, and whether it was lost.

    The result used to be discarded, which made a partial write
    indistinguishable from a clean one: on a real manuscript, 7 findings
    produced 2 comments and the run reported success.

    Both remaining failures are decisions rather than misses — "the quote
    could not be found" is not a category any more. A comment is dropped when
    a rewrite is about to consume the text it anchors to; a rewrite is dropped
    when two of them want to change the same span. They are reported apart
    because only one of them loses anything: every finding is also listed in
    the prepended report, so an unplaced comment is a thinner margin and
    nothing more, while an edit exists only as its tracked change.
    """
    if report is None:
        return

    if report.moved_notes:
        console.print(
            f"[dim]note:[/dim] {len(report.moved_notes)} comment(s) cover "
            f"text a suggested rewrite revised, so they sit on the passage as "
            f"it now reads rather than as it was written."
        )

    if report.shadowed_notes:
        attempted = len(report.shadowed_notes) + report.applied_notes
        console.print(
            f"[yellow]note:[/yellow] {len(report.shadowed_notes)} of "
            f"{attempted} comment(s) were not placed — the rewrite covering "
            f"each one only removes text, so there is nothing left to point "
            f"at. Those findings are still listed in the review report at the "
            f"top of the document."
        )

    if report.conflicted_rewrites:
        console.print(
            f"[yellow]warning:[/yellow] {len(report.conflicted_rewrites)} "
            f"suggested edit(s) were NOT applied and appear nowhere in the "
            f".docx — each overlaps an edit already accepted over the same "
            f"span. Edits are not listed in the review report — use a "
            f"markdown or HTML output as well to see every proposed rewrite:"
        )
        for rewrite in report.conflicted_rewrites:
            console.print(f"  • {(rewrite.label or rewrite.new_text)[:70]}")


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
