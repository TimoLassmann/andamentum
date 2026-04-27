"""Shared state passed across whetstone v2 graph nodes.

Mirrors deep_research's ``ResearchState`` pattern: every field has a
default; nodes mutate the state; nothing is required at construction time
beyond the input ``source`` and a small set of run-level knobs.

In Phase 1 only the deterministic-substrate fields are populated.
Hypothesis / investigation / synthesis fields are present so the type
surface is stable from day one — they remain empty until later phases
fill them in.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from .schemas import (
    AuthorQuestion,
    Edit,
    Finding,
    Hypothesis,
    SectionCard,
)
from .structural.types import SectionRef, StructuralFacts


@dataclass
class FailedInvestigation:
    """One investigation that crashed (kept for diagnostics, not raised)."""

    hypothesis: Hypothesis
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

    # ── LLM-driven (Phase 2+) ──────────────────────────────────────────
    hypotheses: list[Hypothesis] = field(default_factory=list)
    findings: list[Finding] = field(default_factory=list)
    challenged_findings: list[Finding] = field(default_factory=list)
    edits: list[Edit] = field(default_factory=list)  # from EditSections (optional)
    author_questions: list[AuthorQuestion] = field(default_factory=list)
    summary: str = ""

    # ── Budget / control ───────────────────────────────────────────────
    hypothesis_budget: int = 30
    investigations_done: int = 0
    challenge_enabled: bool = True
    editor_enabled: bool = False  # opt-in: edits are extra LLM cost
    editor_criteria: list[str] = field(
        default_factory=lambda: ["clarity", "concision", "grammar"]
    )

    # ── Telemetry ──────────────────────────────────────────────────────
    llm_calls: int = 0

    # ── Errors (accumulated, not raised) ───────────────────────────────
    failed_investigations: list[FailedInvestigation] = field(default_factory=list)

    # ── Flow control ───────────────────────────────────────────────────
    current_phase: Literal[
        "harvest",
        "scan",
        "skim",
        "investigate",
        "edit",
        "challenge",
        "author_questions",
        "synthesise",
        "done",
    ] = "harvest"
