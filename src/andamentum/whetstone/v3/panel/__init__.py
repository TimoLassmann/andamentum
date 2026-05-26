"""Panel mode for v3 — multi-expert simulated peer review.

A different review *shape* from the criterion cascade (`run_review_v3`):
N fictional expert biosketches × N independent reviews + 1 synthesis.
Lives in its own graph (this package) because it cannot be expressed
as a criterion set.

Single public entry: ``run_panel_v3(markdown, *, model, n_experts,
panel_disciplines) -> ReviewResult``. Populates ``ReviewResult``'s
panel-specific fields (``expert_profiles``, ``expert_reviews``,
``panel_synthesis``); the criterion-cascade fields (``findings``,
``edits``, ``author_questions``) stay empty.

The three existing renderers (`render_markdown` / `render_html` /
`render_docx`) consume panel output unchanged via the same
``ReviewResult`` contract.
"""

from .graph import run_panel_v3

__all__ = ["run_panel_v3"]
