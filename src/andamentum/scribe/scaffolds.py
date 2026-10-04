"""Built-in bundle scaffolds.

A scaffold is a list of (section title, guide text) tuples. ``Bundle.create``
with ``scaffold=<name>`` adds one empty ``planned`` section per entry. The
guide text is kept for callers that draft into the scaffold; it is not stored
in the bundle. References are not a section: they render from
``references.yaml`` at the end of the document.
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
]

SCAFFOLDS: dict[str, list[tuple[str, str]]] = {
    "article": ARTICLE,
}
