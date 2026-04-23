"""Whetstone orchestrator — run structured feedback over your own drafts.

Public entry point: ``sharpen_document(content, *, task, ...)``.

Three tasks:

- ``edit``: a unified editor (or multiple parallel editors) producing
  DocumentPatch objects for grammar, style, and polish.
- ``review``: four specialist reviewers (clarity, scientific merit,
  methodology, results interpretation) plus a synthesizer that consolidates
  their findings into a prioritised report.
- ``panel``: generate N fictional expert biosketches matched to the
  document's disciplines, have each write a scored review, then synthesize
  a panel assessment.

Custom criteria override the standard review path with a schema generated
at runtime from free-text instructions.

**Scope note:** this is a tool for improving your own drafts before
submission. It is not for peer-reviewing manuscripts other authors have
sent you — that would violate journal confidentiality policy.
"""

from __future__ import annotations

import asyncio
import logging
import sys
from typing import Any, Optional

from pydantic import BaseModel, Field

from andamentum.core.agents import AgentDefinition, AgentRunner

from .agents import AGENT_REGISTRY
from .agents.output_models import (
    CriticalIssue,  # noqa: F401 — needed for pydantic union resolution
    DocumentReviewSynthesisOutput,
    ExpertProfile,
    ExpertReviewOutput,
    FormatterOutput,
    PanelSynthesisOutput,
    SynthesisCriticalIssue,  # noqa: F401 — needed for pydantic union resolution
)
from .issues import DocumentIssue
from .models import DocumentPatch

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result model
# ---------------------------------------------------------------------------


class ReviewResult(BaseModel):
    """Complete result from sharpen_document().

    Pure structured data — no file paths, no rendered output. Pass this into
    render_docx, render_html, render_diff, or apply_patches.
    """

    task: str = Field(description="Task that was run: 'edit', 'review', or 'panel'")

    patches: list[DocumentPatch] = Field(
        default_factory=list,
        description="Edits and comments from editing agents",
    )
    issues: list[DocumentIssue] = Field(
        default_factory=list,
        description="Issues from review agents",
    )
    synthesis: Optional[DocumentReviewSynthesisOutput | PanelSynthesisOutput | FormatterOutput] = Field(
        default=None, description="Consolidated review report"
    )

    disciplines: list[str] = Field(default_factory=list, description="Extracted disciplines (panel)")
    expert_profiles: list[ExpertProfile] = Field(default_factory=list, description="Generated expert profiles (panel)")
    expert_reviews: list[ExpertReviewOutput] = Field(
        default_factory=list, description="Individual expert reviews (panel)"
    )


# ---------------------------------------------------------------------------
# Agent execution helpers
# ---------------------------------------------------------------------------


async def _run_agents(phase_name: str, *coros: Any) -> Any:
    """Gather coroutines, re-raising with phase context on failure."""
    try:
        return await asyncio.gather(*coros)
    except Exception as exc:
        raise RuntimeError(f"Agent failure during {phase_name}: {exc}") from exc


async def _run_one(runner: AgentRunner, agent_name: str, **kwargs: Any) -> Any:
    """Run a registered whetstone agent via the core AgentRunner."""
    defn = AGENT_REGISTRY.get(agent_name)
    if defn is None:
        raise ValueError(f"Unknown whetstone agent: {agent_name}. Available: {sorted(AGENT_REGISTRY)}")
    if defn.output_model is None:
        raise ValueError(
            f"Agent {agent_name} uses a dynamic output model. Use _run_one_dynamic() with an explicit output_type."
        )
    return await runner.run(defn, **kwargs)


async def _run_one_dynamic(
    runner: AgentRunner,
    agent_name: str,
    *,
    output_type: type[Any],
    **kwargs: Any,
) -> Any:
    """Run a dynamic-schema agent with an explicit runtime output type."""
    defn = AGENT_REGISTRY.get(agent_name)
    if defn is None:
        raise ValueError(f"Unknown whetstone agent: {agent_name}. Available: {sorted(AGENT_REGISTRY)}")

    runtime_defn = AgentDefinition(
        name=f"{defn.name}__dynamic",
        prompt=defn.prompt,
        output_model=output_type,
        retries=defn.retries,
        output_retries=defn.output_retries,
    )
    return await runner.run(runtime_defn, **kwargs)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


async def sharpen_document(
    content: str,
    *,
    task: str = "review",
    num_experts: int = 3,
    criteria: Optional[str] = None,
    editors: Optional[list[str]] = None,
    model: str = "openai:gpt-4o",
    verbose: bool = False,
) -> ReviewResult:
    """Run structured feedback over a draft you wrote yourself.

    Args:
        content: Full text of the draft.
        task: "edit" (grammar/style/polish), "review" (4 specialists +
            synthesis), or "panel" (N expert reviews + panel synthesis).
        num_experts: Number of fictional experts for the panel task.
        criteria: Custom free-text criteria. When provided, replaces the
            standard review path with a schema generated at runtime.
        editors: List of editing instructions. When omitted, runs one unified
            editor. When provided, runs one editor per instruction in parallel.
        model: pydantic-ai model string (e.g. "openai:gpt-4o",
            "anthropic:claude-haiku-4-5").
        verbose: Print progress messages to stderr.

    Returns:
        ReviewResult with structured data. Pass to a renderer for output.

    Raises:
        ValueError: If `task` is not one of "edit", "review", "panel".
        RuntimeError: If any agent phase fails.
    """
    if task not in ("edit", "review", "panel"):
        raise ValueError(f"Invalid task '{task}'. Must be 'edit', 'review', or 'panel'.")

    runner = AgentRunner(model=model)
    result = ReviewResult(task=task)

    if task == "edit":
        await _run_edit(runner, result, content, editors, verbose)
    elif task == "review":
        if criteria is not None:
            await _run_custom_review(runner, result, content, criteria, verbose)
        else:
            await _run_standard_review(runner, result, content, verbose)
    elif task == "panel":
        await _run_panel_review(runner, result, content, num_experts, verbose)

    return result


# ---------------------------------------------------------------------------
# Task: Edit
# ---------------------------------------------------------------------------


async def _run_edit(
    runner: AgentRunner,
    result: ReviewResult,
    content: str,
    editors: Optional[list[str]],
    verbose: bool,
) -> None:
    if editors is None:
        print("Running unified editor...", file=sys.stderr)
        output = await _run_one(runner, "unified_editor", document=content)
        result.patches = getattr(output, "patches", [])
    else:
        print(f"Running {len(editors)} editors...", file=sys.stderr)
        coros = [_run_one(runner, "unified_editor", document=content, editing_instructions=inst) for inst in editors]
        outputs = await _run_agents("multi-editor", *coros)
        for output in outputs:
            result.patches.extend(getattr(output, "patches", []))


# ---------------------------------------------------------------------------
# Task: Standard review
# ---------------------------------------------------------------------------


async def _run_standard_review(
    runner: AgentRunner,
    result: ReviewResult,
    content: str,
    verbose: bool,
) -> None:
    print(
        "Running review agents (clarity, merit, methodology, results)...",
        file=sys.stderr,
    )
    clarity, merit, methodology, results_review = await _run_agents(
        "standard review",
        _run_one(runner, "clarity_accessibility_reviewer", document=content),
        _run_one(runner, "core_scientific_merit_reviewer", document=content),
        _run_one(runner, "methodology_reviewer", document=content),
        _run_one(runner, "results_interpretation_reviewer", document=content),
    )

    for review_output in (clarity, merit, methodology, results_review):
        result.issues.extend(getattr(review_output, "issues", []))

    print("Synthesizing reviews...", file=sys.stderr)
    review_data = _format_standard_reviews(clarity, merit, methodology, results_review)
    result.synthesis = await _run_one(
        runner,
        "document_review_synthesizer",
        reviews=review_data,
        document=content,
    )


# ---------------------------------------------------------------------------
# Task: Custom review
# ---------------------------------------------------------------------------


async def _run_custom_review(
    runner: AgentRunner,
    result: ReviewResult,
    content: str,
    criteria: str,
    verbose: bool,
) -> None:
    from .dynamic_models import convert_fields_to_schema, create_output_model

    print("Generating custom review schema...", file=sys.stderr)
    schema_output = await _run_one(runner, "schema_generator", criteria=criteria)
    spec = convert_fields_to_schema(schema_output.fields)
    dynamic_model = create_output_model("custom_review", spec)

    field_names = [f.name for f in schema_output.fields]
    print(f"  Schema fields: {', '.join(field_names)}", file=sys.stderr)
    print("Running custom document reviewer...", file=sys.stderr)

    custom_result = await _run_one_dynamic(
        runner,
        "custom_document_reviewer",
        output_type=dynamic_model,
        document=content,
        review_criteria=criteria,
    )

    print("Formatting results...", file=sys.stderr)
    custom_data = custom_result.model_dump() if hasattr(custom_result, "model_dump") else str(custom_result)
    result.synthesis = await _run_one(
        runner,
        "results_formatter",
        review_results=str(custom_data),
        review_criteria=criteria,
    )


# ---------------------------------------------------------------------------
# Task: Panel review
# ---------------------------------------------------------------------------


async def _run_panel_review(
    runner: AgentRunner,
    result: ReviewResult,
    content: str,
    num_experts: int,
    verbose: bool,
) -> None:
    print(f"Running multi-expert review ({num_experts} experts)...", file=sys.stderr)

    kw_result = await _run_one(runner, "keyword_extractor", document=content)
    disciplines = kw_result.disciplines[:num_experts]
    if not disciplines:
        raise RuntimeError("keyword_extractor returned no disciplines.")
    result.disciplines = disciplines
    print(f"  Disciplines: {', '.join(disciplines)}", file=sys.stderr)

    profiles = await _run_agents(
        "expert profile generation",
        *[_run_one(runner, "expert_generator", discipline=d) for d in disciplines],
    )
    result.expert_profiles = list(profiles)

    expert_reviews = await _run_agents(
        "expert review",
        *[
            _run_one(
                runner,
                "expert_reviewer",
                document=content,
                expert_biosketch=_format_biosketch(profile),
                discipline=profile.discipline,
            )
            for profile in profiles
        ],
    )
    result.expert_reviews = list(expert_reviews)

    for er in expert_reviews:
        print(
            f"  {er.expert_name}: {er.overall_score}/10 — {er.recommendation}",
            file=sys.stderr,
        )

    panel_data = _format_expert_reviews(expert_reviews)
    result.synthesis = await _run_one(
        runner,
        "review_synthesizer",
        reviews=panel_data,
        document=content,
    )


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def _format_standard_reviews(*reviews: Any) -> str:
    parts: list[str] = []
    names = [
        "Clarity & Accessibility",
        "Scientific Merit",
        "Methodology",
        "Results Interpretation",
    ]
    for name, review in zip(names, reviews):
        issues = getattr(review, "issues", [])
        issue_text = "\n".join(
            f"  - [{getattr(i, 'issue_type', 'issue')}] {getattr(i, 'title', str(i))}: {getattr(i, 'description', '')}"
            for i in issues
        )
        parts.append(f"## {name} Review\n{issue_text or '  No issues identified.'}")
    return "\n\n".join(parts)


def _format_biosketch(profile: Any) -> str:
    return (
        f"Name: {profile.name}\n"
        f"Position: {profile.position}\n"
        f"Education: {profile.education}\n"
        f"Contributions: {profile.contributions}\n"
        f"Research: {profile.research}\n"
        f"Discipline: {profile.discipline}"
    )


def _format_expert_reviews(reviews: list[Any]) -> str:
    parts: list[str] = []
    for review in reviews:
        parts.append(
            f"## {review.expert_name} ({review.discipline})\n"
            f"Overall: {review.overall_score}/10\n"
            f"Scientific Rigor: {review.scientific_rigor_score}/10 — "
            f"{review.scientific_rigor_justification}\n"
            f"Methodology: {review.methodology_score}/10 — "
            f"{review.methodology_justification}\n"
            f"Novelty: {review.novelty_score}/10 — {review.novelty_justification}\n"
            f"Clarity: {review.clarity_score}/10 — {review.clarity_justification}\n"
            f"Strengths: {', '.join(review.strengths)}\n"
            f"Weaknesses: {', '.join(review.weaknesses)}\n"
            f"Recommendation: {review.recommendation}\n"
            f"Justification: {review.recommendation_justification}"
        )
    return "\n\n".join(parts)
