"""Shared state passed across whetstone v2 graph nodes.

Mirrors deep_research's ``ResearchState`` pattern: every field has a
default; nodes mutate the state; nothing is required at construction time
beyond the input ``source`` and a small set of run-level knobs.

The flow is:

  HarvestSource  → ChunkAndScan  → CriticalRead  → ReflectAndInvestigate
                                                  ↳ EditSections (optional)
                                                  ↳ Challenge
                                                  ↳ AuthorQuestions
                                                  ↳ Synthesise

Phase 1 (HarvestSource + ChunkAndScan) populates ``markdown``, ``sections``,
``structural_facts``, ``document_map``, ``deterministic_findings``.

Phase 2 (CriticalRead) runs each lens × each section in parallel and
appends Findings to the shared ``findings`` pool.

Phase 3 (ReflectAndInvestigate) is a bounded loop (at most
``reflection_round_cap`` rounds): one open-ended reflection call proposes
investigation tasks; each task is one investigator call that re-reads
named sections from source and decides keep/refine/drop/raise. Every
quote is anchor-verified against actual section text.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from .schemas import (
    AuthorQuestion,
    CheckableItem,
    CustomEvaluation,
    Edit,
    ExpertProfile,
    ExpertReview,
    Finding,
    GuidelineEvaluation,
    PanelSynthesis,
    SectionCard,
)
from .structural.types import SectionRef, StructuralFacts


@dataclass
class FailedTask:
    """One reflection-task investigation that crashed (kept for diagnostics)."""

    description: str
    error: str


@dataclass
class ReviewState:
    """All mutable state for one review run."""

    # ── Input ──────────────────────────────────────────────────────────
    source: str | Path
    perspectives: list[str] = field(default_factory=lambda: ["rigorous"])

    # ── Deterministic substrate (Phase 1) ──────────────────────────────
    markdown: str = ""  # populated by HarvestSource
    sections: list[SectionRef] = field(default_factory=list)
    structural_facts: StructuralFacts = field(default_factory=StructuralFacts)
    document_map: list[SectionCard] = field(default_factory=list)
    deterministic_findings: list[Finding] = field(default_factory=list)

    # ── Critical-review pool (lens reads + reflection loop) ────────────
    findings: list[Finding] = field(default_factory=list)
    challenged_findings: list[Finding] = field(default_factory=list)
    edits: list[Edit] = field(default_factory=list)  # from EditSections (optional)
    author_questions: list[AuthorQuestion] = field(default_factory=list)
    summary: str = ""

    # ── Reflection loop control ────────────────────────────────────────
    reflection_round_cap: int = 3
    reflection_round: int = 0
    prior_task_descriptions: list[str] = field(default_factory=list)

    # ── Editor / Challenge knobs ───────────────────────────────────────
    challenge_enabled: bool = True
    editor_enabled: bool = False  # opt-in: edits are extra LLM cost
    editor_criteria: list[str] = field(
        default_factory=lambda: ["clarity", "concision", "grammar"]
    )

    # ── Telemetry ──────────────────────────────────────────────────────
    llm_calls: int = 0

    # ── Errors (accumulated, not raised) ───────────────────────────────
    failed_tasks: list[FailedTask] = field(default_factory=list)

    # ── Panel mode (mode="panel") ──────────────────────────────────────
    mode: Literal["review", "panel", "guidelines", "custom"] = "review"
    n_experts: int = 4
    panel_disciplines: list[str] = field(default_factory=list)  # provided OR extracted
    disciplines: list[str] = field(default_factory=list)  # extracted by ExtractKeywords
    expert_profiles: list[ExpertProfile] = field(default_factory=list)
    expert_reviews: list[ExpertReview] = field(default_factory=list)
    panel_synthesis: PanelSynthesis | None = None

    # ── Guidelines mode (mode="guidelines") ────────────────────────────
    guidelines_text: str = ""  # free-text journal author guidelines
    checkable_items: list[CheckableItem] = field(default_factory=list)
    guideline_evaluations: list[GuidelineEvaluation] = field(default_factory=list)

    # ── Custom-criteria mode (mode="custom") ───────────────────────────
    custom_criteria: list[str] = field(default_factory=list)
    custom_evaluations: list[CustomEvaluation] = field(default_factory=list)

    # ── Novelty check (orthogonal to mode; opt-in) ─────────────────────
    check_novelty: bool = False
    novelty_search_depth: int = 2  # 1=quick, 2=balanced, 3=thorough
    novelty_cache_dir: Path | None = None  # None → ~/.cache/whetstone/novelty

    # ── Flow control ───────────────────────────────────────────────────
    current_phase: Literal[
        "harvest",
        "scan",
        "critical_read",
        "reflect_investigate",
        "edit",
        "challenge",
        "author_questions",
        "synthesise",
        "extract_keywords",
        "generate_panel",
        "expert_review",
        "panel_synthesise",
        "extract_checkable_items",
        "evaluate_guideline_items",
        "custom_review",
        "novelty_check",
        "done",
    ] = "harvest"
