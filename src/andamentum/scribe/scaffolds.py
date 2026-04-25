# src/andamentum/scribe/scaffolds.py
"""Built-in document scaffolds.

Each scaffold is a list of (section_name, guide_text) tuples. When a
document is created with `scaffold=<name>`, scribe inserts one level-1
heading per section and one placeholder paragraph carrying the guide
text in `metadata.guide` for downstream agents to consume.

Guide text is sourced from the section structures in
manuscript-tools/section-guides.md and is intentionally short.
"""

from __future__ import annotations

ARTICLE: list[tuple[str, str]] = [
    (
        "Abstract",
        "Background → gap → approach → key results → significance. 150-300 words.",
    ),
    (
        "Introduction",
        "Funnel: broad context → narrowing to gap → contribution. 500-1000 words.",
    ),
    (
        "Methods",
        "Reproducibility goal. Specific tools, versions, parameters. Logical order, not chronological.",
    ),
    (
        "Results",
        "Lead each paragraph with the finding, then the evidence. Reference every figure and table.",
    ),
    (
        "Discussion",
        "Restate finding in context. Compare with prior work. Limitations honestly. Concrete future directions.",
    ),
    ("References", ""),
]

GRANT: list[tuple[str, str]] = [
    (
        "Specific Aims",
        "One-page overview. State the long-term goal, the specific aims, and why the work matters.",
    ),
    (
        "Background and Significance",
        "Establish the problem, cite key prior work, identify the gap your work fills.",
    ),
    (
        "Innovation",
        "What is conceptually or methodologically new. Distinguish from incremental work.",
    ),
    (
        "Approach",
        "For each aim: rationale, methods, expected outcomes, alternative strategies, pitfalls.",
    ),
    (
        "Timeline and Milestones",
        "Project schedule with measurable deliverables per period.",
    ),
    ("References", ""),
]

SCAFFOLDS: dict[str, list[tuple[str, str]]] = {
    "article": ARTICLE,
    "grant": GRANT,
}
