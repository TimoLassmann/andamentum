"""Pre-submission checklist agents and baseline definition.

- BASELINE_CHECKS: the authoritative list of baseline items. Edit here
  to add/remove/modify checks.
- checklist_item_evaluator: evaluates one check against a document.
- journal_guidelines_extractor: converts free-form journal author
  guidelines into a list of checkable item names.
"""

from __future__ import annotations

from . import AgentDefinition, register_agent
from .output_models import ExtractedChecklistNames
from ..models import BaselineCheck, ChecklistItem


# ---------------------------------------------------------------------------
# Baseline list — the single source of truth for journal-agnostic checks.
# ---------------------------------------------------------------------------

BASELINE_CHECKS: list[BaselineCheck] = [
    # Abstract (LLM)
    BaselineCheck(
        name="Abstract has clear structured sections",
        category="abstract",
        kind="llm",
        prompt_hint="Look for implicit or explicit sections: background, methods, results, conclusion.",
    ),
    BaselineCheck(
        name="Abstract stays within a reasonable word count",
        category="abstract",
        kind="llm",
        prompt_hint="Most journals require 150-300 words. Count words in the abstract and flag if >400.",
    ),
    BaselineCheck(
        name="Abstract defines any abbreviations it uses",
        category="abstract",
        kind="llm",
        prompt_hint="The abstract must stand alone. Any non-standard acronym should be expanded on first use.",
    ),
    # Figures & tables (deterministic)
    BaselineCheck(
        name="All figures are referenced in the text",
        category="figures",
        kind="deterministic",
        scanner="check_all_figures_referenced",
    ),
    BaselineCheck(
        name="All tables are referenced in the text",
        category="figures",
        kind="deterministic",
        scanner="check_all_tables_referenced",
    ),
    BaselineCheck(
        name="Figure numbering is sequential",
        category="figures",
        kind="deterministic",
        scanner="check_figure_numbering_sequential",
    ),
    BaselineCheck(
        name="Table numbering is sequential",
        category="figures",
        kind="deterministic",
        scanner="check_table_numbering_sequential",
    ),
    # References (mixed)
    BaselineCheck(
        name="All in-text citations resolve to reference entries",
        category="references",
        kind="deterministic",
        scanner="check_citations_resolve",
    ),
    BaselineCheck(
        name="Reference list is formatted consistently",
        category="references",
        kind="llm",
        prompt_hint=(
            "Check for consistent formatting of authors, titles, journal names, "
            "years, volumes, and page numbers across entries."
        ),
    ),
    # Required statements (deterministic presence checks)
    BaselineCheck(
        name="Conflict-of-interest / competing-interests statement present",
        category="statements",
        kind="deterministic",
        scanner="check_coi_statement",
    ),
    BaselineCheck(
        name="Data availability statement present",
        category="statements",
        kind="deterministic",
        scanner="check_data_availability_statement",
    ),
    BaselineCheck(
        name="Ethics statement present if human/animal work is involved",
        category="statements",
        kind="deterministic",
        scanner="check_ethics_statement",
    ),
    BaselineCheck(
        name="Funding / acknowledgements statement present",
        category="statements",
        kind="deterministic",
        scanner="check_funding_statement",
    ),
    # Manuscript hygiene
    BaselineCheck(
        name="Keywords section present",
        category="hygiene",
        kind="deterministic",
        scanner="check_keywords_section",
    ),
    BaselineCheck(
        name="Authors and affiliations listed",
        category="hygiene",
        kind="deterministic",
        scanner="check_authors_listed",
    ),
    BaselineCheck(
        name="Title is meaningful and specific",
        category="hygiene",
        kind="llm",
        prompt_hint=(
            "A meaningful title describes what the paper is about. Avoid "
            "generic ('A study of…') or vague titles."
        ),
    ),
]


# ---------------------------------------------------------------------------
# checklist_item_evaluator
# ---------------------------------------------------------------------------

_CHECKLIST_ITEM_EVALUATOR_PROMPT = """\
# Single pre-submission check evaluator

You are evaluating ONE pre-submission check against a manuscript.

Return a ChecklistItem with:

- `name`: copy the check name exactly as given in the user message.
- `status`: "pass" (the check is clearly met), "fail" (clearly not met),
  or "unclear" (ambiguous, or the check does not apply to this document).
- `notes`: one or two sentences.
    - For "pass": briefly cite the evidence (quote a phrase or name the section).
    - For "fail": say what is missing and what the author should add.
    - For "unclear": say why it's unclear or not applicable.

Leave `category` and `source` with their defaults — the orchestrator
sets them.

Keep notes concise. Do not pad. Do not hedge.
"""


register_agent(
    AgentDefinition(
        name="checklist_item_evaluator",
        prompt=_CHECKLIST_ITEM_EVALUATOR_PROMPT,
        output_model=ChecklistItem,
        retries=2,
    )
)


# ---------------------------------------------------------------------------
# journal_guidelines_extractor
# ---------------------------------------------------------------------------

_JOURNAL_EXTRACTOR_PROMPT = """\
# Journal guidelines extractor

Read the journal author guidelines below and extract every rule an
author should verify before submission, one rule per item.

Rules:
- 10–30 items total
- Skip general editorial prose ("We welcome submissions...")
- Keep only actionable, checkable rules
- Phrase each item as a short declarative requirement (e.g.
  "Abstract ≤ 250 words", "Figures in vector format",
  "Data availability statement present", "Author contributions section included")

Return a list of item name strings.
"""


register_agent(
    AgentDefinition(
        name="journal_guidelines_extractor",
        prompt=_JOURNAL_EXTRACTOR_PROMPT,
        output_model=ExtractedChecklistNames,
        retries=2,
    )
)
