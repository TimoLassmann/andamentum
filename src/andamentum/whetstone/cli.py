"""CLI entry point for andamentum-whetstone.

Usage::

    andamentum-whetstone document.txt --task edit -o reviewed.docx
    andamentum-whetstone document.docx --task review -o report.html
    andamentum-whetstone document.md --task panel --num-experts 3
    andamentum-whetstone agents

Requires the whetstone package with LLM support.
"""

import argparse
import asyncio
import sys
import traceback
from pathlib import Path


def _read_document(path: Path) -> str:
    """Read document content, handling both text and binary formats."""
    suffix = path.suffix.lower()
    if suffix == ".docx":
        try:
            from docx import Document

            doc = Document(str(path))
            return "\n\n".join(p.text for p in doc.paragraphs if p.text.strip())
        except Exception as e:
            raise ValueError(f"Failed to read .docx file {path}: {e}") from e
    if suffix == ".pdf":
        raise ValueError(
            "PDF reading requires docling. Convert to .docx or .txt first, or install docling: pip install docling"
        )
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError as e:
        raise ValueError(f"Cannot read {path} as UTF-8 text: {e}") from e


def _build_parser() -> argparse.ArgumentParser:
    """Build the argparse parser for the andamentum-whetstone CLI.

    Defines the positional ``file`` argument and all option flags:
    ``--task``, ``--num-experts``, ``--criteria``, ``--model``,
    ``--verbose``, and ``-o/--output``. The ``agents`` subcommand is
    handled separately in :func:`main` before argparse runs.
    """
    parser = argparse.ArgumentParser(
        prog="andamentum-whetstone",
        description=(
            "Sharpen your own drafts — structured feedback with track changes, "
            "multi-specialist review, or expert panel. "
            "This tool is for your own work. It is NOT a peer-review tool — do "
            "not use it on manuscripts you have been asked to review in confidence."
        ),
    )
    parser.add_argument("file", type=Path, help="Path to document to review")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output file (.docx, .html, or .md)",
    )
    parser.add_argument(
        "--task",
        choices=["edit", "review", "panel", "consistency", "checklist"],
        default="review",
        help="Task (default: review)",
    )
    parser.add_argument(
        "--num-experts",
        type=int,
        default=3,
        help="Number of expert reviewers for panel task (default: 3)",
    )
    parser.add_argument(
        "--criteria",
        type=str,
        default=None,
        help="Custom review criteria (text or @filepath)",
    )
    parser.add_argument(
        "--guidelines",
        type=str,
        default=None,
        help="Journal author guidelines (text or @filepath). Only valid with --task checklist.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="LLM model (default: $ANDAMENTUM_MAIN_LLM_MODEL or openai:gpt-4o)",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Print progress messages"
    )
    return parser


def _resolve_model(args: argparse.Namespace) -> str:
    """Resolve the LLM model string to use for agent execution.

    Precedence: ``--model`` flag > ``ANDAMENTUM_MAIN_LLM_MODEL`` environment
    variable > default ``openai:gpt-4o``. Returns a pydantic-ai model
    string (e.g. ``openai:gpt-4o``, ``anthropic:claude-sonnet-4-5``).
    """
    import os

    return args.model or os.environ.get("ANDAMENTUM_MAIN_LLM_MODEL") or "openai:gpt-4o"


def _resolve_criteria(raw: str | None) -> str | None:
    """Resolve criteria from string or @filepath."""
    if not raw or not raw.strip():
        return None
    if raw.startswith("@"):
        criteria_path = Path(raw[1:])
        if not criteria_path.exists():
            print(f"Error: criteria file not found: {criteria_path}", file=sys.stderr)
            sys.exit(1)
        try:
            return criteria_path.read_text(encoding="utf-8")
        except UnicodeDecodeError as e:
            print(f"Error: cannot read criteria file as UTF-8: {e}", file=sys.stderr)
            sys.exit(1)
    return raw


def _resolve_guidelines(raw: str | None) -> str | None:
    """Resolve guidelines from string or @filepath (mirrors _resolve_criteria)."""
    if not raw or not raw.strip():
        return None
    if raw.startswith("@"):
        p = Path(raw[1:])
        if not p.exists():
            print(f"Error: guidelines file not found: {p}", file=sys.stderr)
            sys.exit(1)
        try:
            return p.read_text(encoding="utf-8")
        except UnicodeDecodeError as e:
            print(f"Error: cannot read guidelines file as UTF-8: {e}", file=sys.stderr)
            sys.exit(1)
    return raw


def _list_agents() -> None:
    """Print all registered agents to stdout.

    Handles the ``andamentum-whetstone agents`` subcommand. Each line shows
    the agent name and its output model class, or ``dynamic`` for
    agents like ``custom_document_reviewer`` that use runtime schemas.
    """
    from .agents import AGENT_REGISTRY

    if not AGENT_REGISTRY:
        print("No agents registered.")
        return

    for name, defn in sorted(AGENT_REGISTRY.items()):
        model_name = defn.output_model.__name__ if defn.output_model else "dynamic"
        print(f"  {name:<35s}  output={model_name}")


def _render_output(
    result: object, content: str, input_path: Path, output_path: Path | None
) -> None:
    """Render the ReviewResult to the appropriate output format."""
    from .renderers import render_diff

    if output_path is None:
        # Default: markdown diff to stdout
        patches = getattr(result, "patches", [])
        issues = getattr(result, "issues", [])
        synthesis = getattr(result, "synthesis", None)
        synthesis_text = (
            getattr(synthesis, "review_summary", None) if synthesis else None
        )

        checklist = getattr(result, "checklist", None) or None
        diff_output = render_diff(
            patches=patches,
            issues=issues,
            original_content=content,
            synthesis_text=synthesis_text,
            checklist=checklist,
        )
        print(diff_output)
        return

    suffix = output_path.suffix.lower()

    if suffix == ".docx":
        if input_path.suffix.lower() != ".docx":
            print("Error: .docx output requires .docx input file", file=sys.stderr)
            sys.exit(1)

        from .renderers import render_docx

        patches = getattr(result, "patches", [])
        synthesis = getattr(result, "synthesis", None)
        review_summary = getattr(synthesis, "review_summary", "") if synthesis else ""
        critical_issues = getattr(synthesis, "critical_issues", []) if synthesis else []
        expert_reviews = getattr(result, "expert_reviews", [])
        expert_profiles = getattr(result, "expert_profiles", [])

        patch_result = render_docx(
            input_path=input_path,
            output_path=output_path,
            patches=patches,
            review_summary=review_summary,
            critical_issues=critical_issues,
            expert_reviews=list(expert_reviews) if expert_reviews else None,
            generated_experts=list(expert_profiles) if expert_profiles else None,
        )
        applied = getattr(patch_result, "applied_patches", 0)
        total = getattr(patch_result, "total_patches", 0)
        if total > 0:
            print(f"Patches: {applied}/{total} applied")
        print(f"Output: {output_path}")

    elif suffix == ".html":
        from .renderers import render_html

        html_output = render_html(result=result, original_content=content)
        output_path.write_text(html_output, encoding="utf-8")
        print(f"Output: {output_path}")

    elif suffix == ".md":
        from .renderers import render_diff

        patches = getattr(result, "patches", [])
        issues = getattr(result, "issues", [])
        synthesis = getattr(result, "synthesis", None)
        synthesis_text = (
            getattr(synthesis, "review_summary", None) if synthesis else None
        )

        checklist = getattr(result, "checklist", None) or None
        diff_output = render_diff(
            patches=patches,
            issues=issues,
            original_content=content,
            synthesis_text=synthesis_text,
            checklist=checklist,
        )
        output_path.write_text(diff_output, encoding="utf-8")
        print(f"Output: {output_path}")

    else:
        print(
            f"Error: unsupported output format '{suffix}'. Use .docx, .html, or .md",
            file=sys.stderr,
        )
        sys.exit(1)


async def _run(args: argparse.Namespace) -> None:
    """Execute a review pipeline from parsed CLI arguments.

    Reads the input document, resolves the model and criteria, runs
    :func:`sharpen_document`, then dispatches to the appropriate
    renderer via :func:`_render_output`. Exits with a non-zero status
    if the input file is missing or if the LLM extra is not installed.
    """
    try:
        from .orchestrator import sharpen_document
    except ImportError as exc:
        print(
            f"Error: pydantic-ai required but failed to import: {exc}",
            file=sys.stderr,
        )
        sys.exit(1)

    if not args.file.exists():
        print(f"Error: file not found: {args.file}", file=sys.stderr)
        sys.exit(1)

    content = _read_document(args.file)
    model = _resolve_model(args)
    criteria = _resolve_criteria(args.criteria)
    guidelines = _resolve_guidelines(args.guidelines)
    if guidelines is not None and args.task != "checklist":
        print(
            "Error: --guidelines is only valid with --task checklist", file=sys.stderr
        )
        sys.exit(2)

    # Always show basic progress on stderr
    parts = [f"Task: {args.task}"]
    if criteria:
        parts.append("Criteria: custom")
    if guidelines:
        parts.append("Guidelines: provided")
    if args.task == "panel":
        parts.append(f"Experts: {args.num_experts}")
    parts.append(f"Model: {model}")
    print(f"Reviewing: {args.file}", file=sys.stderr)
    print("  |  ".join(parts), file=sys.stderr)
    print(
        f"Document: {len(content)} chars, ~{len(content.split())} words",
        file=sys.stderr,
    )

    result = await sharpen_document(
        content,
        task=args.task,
        num_experts=args.num_experts,
        criteria=criteria,
        guidelines=guidelines,
        model=model,
        verbose=args.verbose,
    )

    _render_output(result, content, args.file, args.output)


def main() -> None:
    """Entry point for the ``andamentum-whetstone`` console script.

    Handles the ``agents`` subcommand before argparse (so ``agents``
    isn't parsed as a file path). Runs :func:`_run` in an event loop
    for all other invocations. Translates ``KeyboardInterrupt`` into
    a clean exit and prints tracebacks only when ``--verbose`` is set.
    """
    if len(sys.argv) > 1 and sys.argv[1] == "agents":
        _list_agents()
        return

    parser = _build_parser()
    args = parser.parse_args()

    # Configure logging so andamentum.core and pydantic-ai messages are visible
    import logging

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )
    # Always show andamentum.core runner messages (fallback, retries)
    logging.getLogger("andamentum.core").setLevel(logging.INFO)
    # Show pydantic-ai retry/validation messages with --verbose
    if args.verbose:
        logging.getLogger("pydantic_ai").setLevel(logging.DEBUG)

    try:
        asyncio.run(_run(args))
    except KeyboardInterrupt:
        print("\nReview cancelled.")
        sys.exit(0)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
