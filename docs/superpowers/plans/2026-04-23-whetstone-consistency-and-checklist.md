# Whetstone: consistency + pre-submission checklist — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add two new whetstone tasks — `consistency` (internal-consistency review via deterministic scanners + one LLM reviewer) and `checklist` (per-item baseline + optional journal-specific pre-submission checklist).

**Architecture:** Follows whetstone's existing task pattern (orchestrator helper function + agents registered in `AGENT_REGISTRY`). Countable/regex-detectable work lives in pure-Python scanners per Constitution Rule 4. Reading-comprehension work lives in single-purpose agents. New `ChecklistItem` model is flat (3 LLM-visible fields) to stay tractable for small local models.

**Tech Stack:** Python 3.13, pydantic, pytest (asyncio_mode=auto), pydantic-ai (via `andamentum.core.agents`), `re` for scanners.

**Spec:** `docs/superpowers/specs/2026-04-23-whetstone-consistency-and-checklist-design.md`

---

## File structure

**Create:**
- `src/andamentum/whetstone/consistency_scanners.py` — deterministic scanners emitting `DocumentIssue`s
- `src/andamentum/whetstone/checklist_scanners.py` — deterministic scanners returning `(status, notes)` tuples
- `src/andamentum/whetstone/agents/consistency.py` — one agent (`consistency_reviewer`)
- `src/andamentum/whetstone/agents/checklist.py` — two agents + `BASELINE_CHECKS` list
- `src/andamentum/whetstone/tests/test_consistency_scanners.py`
- `src/andamentum/whetstone/tests/test_checklist_scanners.py`
- `src/andamentum/whetstone/tests/test_checklist_models.py`
- `src/andamentum/whetstone/tests/test_consistency_orchestrator.py`
- `src/andamentum/whetstone/tests/test_checklist_orchestrator.py`

**Modify:**
- `src/andamentum/whetstone/models.py` — add `ChecklistItem`, `BaselineCheck`
- `src/andamentum/whetstone/agents/output_models.py` — add `ConsistencyReviewOutput`, `ExtractedChecklistNames`
- `src/andamentum/whetstone/agents/__init__.py` — import new agent modules
- `src/andamentum/whetstone/orchestrator.py` — extend `ReviewResult`, `sharpen_document`, add `_run_consistency`, `_run_checklist`
- `src/andamentum/whetstone/__init__.py` — export `ChecklistItem`
- `src/andamentum/whetstone/cli.py` — new task choices, `--guidelines` flag
- `src/andamentum/whetstone/renderers/diff.py` — render checklist markdown
- `src/andamentum/whetstone/renderers/html.py` — render checklist section
- `src/andamentum/whetstone/renderers/docx.py` + `docx/finalization.py` — prepend checklist
- `src/andamentum/whetstone/tests/test_orchestrator_smoke.py` — assert new defaults
- `src/andamentum/whetstone/tests/test_agent_registry.py` — assert new agents present
- `src/andamentum/whetstone/tests/test_public_api.py` — assert `ChecklistItem` exported
- `src/andamentum/whetstone/tests/test_cli.py` — new task choices, `--guidelines`
- `src/andamentum/whetstone/README.md` — document new tasks

---

### Task 1: Add `ChecklistItem` and `BaselineCheck` to `models.py`

**Files:**
- Modify: `src/andamentum/whetstone/models.py`
- Create: `src/andamentum/whetstone/tests/test_checklist_models.py`

- [ ] **Step 1: Write the failing tests**

Create `src/andamentum/whetstone/tests/test_checklist_models.py`:

```python
"""Tests for ChecklistItem and BaselineCheck models."""

import pytest
from pydantic import ValidationError

from andamentum.whetstone.models import BaselineCheck, ChecklistItem


def test_checklist_item_minimal():
    item = ChecklistItem(name="Abstract word count", status="pass", notes="240 words on page 1")
    assert item.name == "Abstract word count"
    assert item.status == "pass"
    assert item.notes == "240 words on page 1"
    assert item.category == ""
    assert item.source == "baseline"


def test_checklist_item_all_fields():
    item = ChecklistItem(
        name="Keywords present", status="fail", notes="No keywords section",
        category="hygiene", source="baseline",
    )
    assert item.category == "hygiene"
    assert item.source == "baseline"


def test_checklist_item_rejects_bad_status():
    with pytest.raises(ValidationError):
        ChecklistItem(name="x", status="maybe", notes="")


def test_checklist_item_rejects_bad_source():
    with pytest.raises(ValidationError):
        ChecklistItem(name="x", status="pass", notes="y", source="other")


def test_baseline_check_deterministic():
    c = BaselineCheck(
        name="All figures referenced", category="figures",
        kind="deterministic", scanner="check_all_figures_referenced",
    )
    assert c.kind == "deterministic"
    assert c.scanner == "check_all_figures_referenced"
    assert c.prompt_hint is None


def test_baseline_check_llm():
    c = BaselineCheck(
        name="Abstract structured", category="abstract",
        kind="llm", prompt_hint="Look for background/methods/results/conclusion.",
    )
    assert c.kind == "llm"
    assert c.prompt_hint is not None
    assert c.scanner is None


def test_baseline_check_rejects_bad_kind():
    with pytest.raises(ValidationError):
        BaselineCheck(name="x", category="y", kind="guess")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest src/andamentum/whetstone/tests/test_checklist_models.py -v`
Expected: `ImportError` for `ChecklistItem`/`BaselineCheck`.

- [ ] **Step 3: Add the models**

Append to `src/andamentum/whetstone/models.py`:

```python
class ChecklistItem(BaseModel):
    """One pre-submission check, evaluated against a draft.

    Three fields are LLM-visible (name, status, notes). Two are set by
    the orchestrator (category, source). The LLM is not asked to re-emit
    contextual metadata already known at dispatch time — this keeps the
    output schema small and tractable for local models.
    """

    name: str = Field(..., description="The check, e.g. 'Abstract word count declared'")
    status: Literal["pass", "fail", "unclear"] = Field(..., description="Outcome of the check")
    notes: str = Field("", description="Evidence (quote/location) and, if status=fail, what to fix")
    category: str = Field("", description="Category — set by the orchestrator, not the LLM")
    source: Literal["baseline", "journal"] = Field(
        "baseline",
        description="Which source produced this check — set by the orchestrator, not the LLM",
    )


class BaselineCheck(BaseModel):
    """An entry in the pre-submission baseline checklist.

    `kind` selects between a deterministic scanner (pure Python) and an
    LLM evaluator. `scanner` names a function in checklist_scanners;
    `prompt_hint` carries extra guidance for the LLM. The two are
    mutually exclusive in practice.
    """

    name: str
    category: str
    kind: Literal["deterministic", "llm"]
    scanner: Optional[str] = Field(None, description="Function name in checklist_scanners (kind='deterministic')")
    prompt_hint: Optional[str] = Field(None, description="Extra guidance (kind='llm')")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest src/andamentum/whetstone/tests/test_checklist_models.py -v`
Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
git add src/andamentum/whetstone/models.py src/andamentum/whetstone/tests/test_checklist_models.py
git commit -m "feat(whetstone): add ChecklistItem and BaselineCheck models"
```

---

### Task 2: Extend `ReviewResult` with `checklist` field

**Files:**
- Modify: `src/andamentum/whetstone/orchestrator.py`
- Modify: `src/andamentum/whetstone/tests/test_orchestrator_smoke.py`

- [ ] **Step 1: Write the failing test**

Append to `src/andamentum/whetstone/tests/test_orchestrator_smoke.py`:

```python
def test_review_result_checklist_default():
    r = ReviewResult(task="checklist")
    assert r.checklist == []


def test_review_result_checklist_accepts_items():
    from andamentum.whetstone.models import ChecklistItem
    items = [ChecklistItem(name="x", status="pass", notes="")]
    r = ReviewResult(task="checklist", checklist=items)
    assert len(r.checklist) == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest src/andamentum/whetstone/tests/test_orchestrator_smoke.py -v`
Expected: 2 new tests fail with `ReviewResult` having no `checklist` field.

- [ ] **Step 3: Add the field**

In `src/andamentum/whetstone/orchestrator.py`, update the imports near the top:

```python
from .models import DocumentPatch, ChecklistItem
```

Add to the `ReviewResult` class (after the existing fields, before class end):

```python
    checklist: list[ChecklistItem] = Field(
        default_factory=list,
        description="Checklist items from 'checklist' task",
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest src/andamentum/whetstone/tests/test_orchestrator_smoke.py -v`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/andamentum/whetstone/orchestrator.py src/andamentum/whetstone/tests/test_orchestrator_smoke.py
git commit -m "feat(whetstone): add ReviewResult.checklist field"
```

---

### Task 3: Add agent output models

**Files:**
- Modify: `src/andamentum/whetstone/agents/output_models.py`
- Modify: `src/andamentum/whetstone/tests/test_checklist_models.py`

- [ ] **Step 1: Write the failing tests**

Append to `src/andamentum/whetstone/tests/test_checklist_models.py`:

```python
def test_consistency_review_output_defaults():
    from andamentum.whetstone.agents.output_models import ConsistencyReviewOutput
    o = ConsistencyReviewOutput()
    assert o.issues == []


def test_extracted_checklist_names_defaults():
    from andamentum.whetstone.agents.output_models import ExtractedChecklistNames
    o = ExtractedChecklistNames()
    assert o.items == []


def test_extracted_checklist_names_accepts_list():
    from andamentum.whetstone.agents.output_models import ExtractedChecklistNames
    o = ExtractedChecklistNames(items=["check one", "check two"])
    assert len(o.items) == 2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest src/andamentum/whetstone/tests/test_checklist_models.py -v`
Expected: 3 new tests fail with `ImportError`.

- [ ] **Step 3: Add the output models**

Append to `src/andamentum/whetstone/agents/output_models.py`:

```python
# ---------------------------------------------------------------------------
# Consistency-reviewer output
# ---------------------------------------------------------------------------


class ConsistencyReviewOutput(BaseModel):
    """Output from the consistency_reviewer agent."""

    issues: list[DocumentIssue] = Field(
        default_factory=list,
        description="Internal-consistency issues found by reading comprehension",
    )


# ---------------------------------------------------------------------------
# Journal-guidelines-extractor output
# ---------------------------------------------------------------------------


class ExtractedChecklistNames(BaseModel):
    """Output from the journal_guidelines_extractor agent.

    A flat list of short, checkable item names. Each name becomes one
    call to checklist_item_evaluator downstream.
    """

    items: list[str] = Field(
        default_factory=list,
        description="Short checkable items extracted from journal author guidelines",
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest src/andamentum/whetstone/tests/test_checklist_models.py -v`
Expected: 10 passed.

- [ ] **Step 5: Commit**

```bash
git add src/andamentum/whetstone/agents/output_models.py src/andamentum/whetstone/tests/test_checklist_models.py
git commit -m "feat(whetstone): add ConsistencyReviewOutput and ExtractedChecklistNames"
```

---

### Task 4: Consistency scanners (`consistency_scanners.py`)

**Files:**
- Create: `src/andamentum/whetstone/consistency_scanners.py`
- Create: `src/andamentum/whetstone/tests/test_consistency_scanners.py`

- [ ] **Step 1: Write the failing tests**

Create `src/andamentum/whetstone/tests/test_consistency_scanners.py`:

```python
"""Tests for consistency_scanners — pure functions, no LLM."""

from andamentum.whetstone.consistency_scanners import (
    check_acronym_first_use,
    check_citation_resolution,
    check_figure_order,
    run_all,
)


# ---- check_figure_order ----------------------------------------------------

def test_figure_order_clean():
    text = "See Figure 1 for overview. Figure 2 shows the breakdown. Figure 3 summarises."
    assert check_figure_order(text) == []


def test_figure_order_out_of_order():
    text = "First, see Figure 2 for the overview. Later, Figure 1 gives the background."
    issues = check_figure_order(text)
    assert len(issues) == 1
    assert "Figure 2" in issues[0].title
    assert issues[0].agent_type == "scanner:figure_order"


def test_figure_order_no_figures():
    assert check_figure_order("Plain text with no figures.") == []


def test_figure_order_handles_fig_abbreviation():
    text = "We cite Fig. 2 first, then Fig. 1."
    issues = check_figure_order(text)
    assert len(issues) == 1


# ---- check_acronym_first_use -----------------------------------------------

def test_acronym_defined_on_first_use():
    text = "We used random forests (RF) for training. The RF classifier outperformed baselines."
    assert check_acronym_first_use(text) == []


def test_acronym_used_before_definition():
    text = "The RF classifier outperformed baselines. We used random forests (RF) for training."
    issues = check_acronym_first_use(text)
    assert any(i.agent_type == "scanner:acronym_first_use" for i in issues)
    assert any("RF" in i.title for i in issues)


def test_acronym_common_skipped():
    text = "We examined DNA extracted from patient samples."
    assert check_acronym_first_use(text) == []


def test_acronym_no_acronyms():
    assert check_acronym_first_use("Plain prose with no acronyms.") == []


# ---- check_citation_resolution ---------------------------------------------

def test_citation_resolution_all_present():
    text = "As shown [1], and also [2].\n\nReferences\n[1] First paper.\n[2] Second paper.\n"
    assert check_citation_resolution(text) == []


def test_citation_resolution_missing_entry():
    text = "As shown [1], and [3].\n\nReferences\n[1] First paper.\n[2] Second paper.\n"
    issues = check_citation_resolution(text)
    assert len(issues) == 1
    assert "[3]" in issues[0].title


def test_citation_resolution_no_references_section():
    text = "As shown [1], and [2]. Discussion follows."
    assert check_citation_resolution(text) == []


def test_citation_resolution_handles_ranges():
    text = (
        "See [1-3] and [5].\n\n"
        "References\n[1] A.\n[2] B.\n[3] C.\n[5] E.\n"
    )
    assert check_citation_resolution(text) == []


def test_citation_resolution_handles_comma_list():
    text = (
        "See [1, 2, 4].\n\n"
        "References\n[1] A.\n[2] B.\n[3] C.\n"
    )
    issues = check_citation_resolution(text)
    assert len(issues) == 1
    assert "[4]" in issues[0].title


# ---- run_all ---------------------------------------------------------------

def test_run_all_combines_scanner_output():
    text = "Figure 2 first. Figure 1 after. See [1].\n\nReferences\n"
    issues = run_all(text)
    # Figure order issue + no references entries → citation resolution returns []
    assert any(i.agent_type == "scanner:figure_order" for i in issues)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest src/andamentum/whetstone/tests/test_consistency_scanners.py -v`
Expected: all 13 fail with `ImportError`.

- [ ] **Step 3: Create the scanner module**

Create `src/andamentum/whetstone/consistency_scanners.py`:

```python
"""Deterministic scanners for internal consistency.

Pure functions. No LLM, no async, no IO. Each scanner examines
document text and emits DocumentIssue objects for problems it can
verify without reading comprehension.

Constitution Rule 4: countable work lives here; reading-comprehension
work lives in the consistency_reviewer agent.
"""

from __future__ import annotations

import re

from .issues import DocumentIssue

_FIGURE_PATTERN = re.compile(r"\b(?:Figure|Fig\.?)\s+(\d+)\b")
_ACRONYM_PATTERN = re.compile(r"\b([A-Z]{2,6})\b")
_CITATION_NUMERIC = re.compile(r"\[(\d+(?:\s*[,-]\s*\d+)*)\]")
_REFERENCES_HEADER = re.compile(r"^\s*(?:References|Bibliography)\s*$", re.MULTILINE | re.IGNORECASE)

# Common acronyms skipped by check_acronym_first_use — universally recognised,
# not worth flagging. Add to this set rather than changing the check logic.
_COMMON_ACRONYMS: frozenset[str] = frozenset({
    "DNA", "RNA", "PCR", "HIV", "USA", "UK", "EU", "CI", "SD", "SEM",
    "FDA", "NIH", "NASA", "NSF", "PDF", "HTML", "URL", "API", "CPU",
    "GPU", "RAM", "USB", "AI", "ML", "ATP", "GDP", "OECD", "CO2", "H2O",
    "mRNA", "tRNA", "rRNA", "PCA", "SVM", "MHz", "GHz", "MB", "GB",
})


def check_figure_order(text: str) -> list[DocumentIssue]:
    """Find figures first-referenced out of ascending order.

    Emits one issue per out-of-order first-reference.
    """
    seen: set[int] = set()
    expected_next = 1
    issues: list[DocumentIssue] = []

    for match in _FIGURE_PATTERN.finditer(text):
        n = int(match.group(1))
        if n in seen:
            continue
        seen.add(n)
        if n != expected_next:
            issues.append(
                DocumentIssue(
                    issue_type="minor",
                    category="structure",
                    title=f"Figure {n} introduced out of order",
                    description=(
                        f"Figure {n} is first referenced before Figure {expected_next}. "
                        f"Figures should be introduced in ascending numerical order."
                    ),
                    recommendation=f"Move the first mention of Figure {n} after Figure {expected_next}.",
                    location=f"Offset {match.start()}",
                    agent_type="scanner:figure_order",
                    confidence=1.0,
                    priority="medium",
                )
            )
        expected_next = max(expected_next, n + 1)

    return issues


def check_acronym_first_use(text: str) -> list[DocumentIssue]:
    """Find acronyms whose first use is not accompanied by a parenthesised definition.

    Definition pattern recognised: capitalised words immediately followed
    by ' (ACRONYM)'. Acronyms in _COMMON_ACRONYMS are skipped.
    """
    issues: list[DocumentIssue] = []
    seen: set[str] = set()

    for match in _ACRONYM_PATTERN.finditer(text):
        acronym = match.group(1)
        if acronym in seen or acronym in _COMMON_ACRONYMS:
            continue
        seen.add(acronym)

        # Look backward for '(ACRONYM)' definition pattern within 200 chars.
        start = max(0, match.start() - 200)
        window = text[start : match.end() + 1]
        if re.search(rf"\(\s*{re.escape(acronym)}\s*\)", window):
            continue  # defined earlier in the window

        # Or it's defined *at* this occurrence: "Phrase (ACR)" — look backward
        # from the opening paren.
        before = text[max(0, match.start() - 100) : match.start()]
        if re.search(r"[A-Za-z][^.\n]*\s+\(\s*$", before):
            continue  # preceded by a phrase and opening paren — first-use definition

        issues.append(
            DocumentIssue(
                issue_type="minor",
                category="structure",
                title=f"Acronym '{acronym}' used before being defined",
                description=(
                    f"The acronym '{acronym}' appears without a parenthesised "
                    f"definition before or at its first use."
                ),
                recommendation=f"Expand on first use: 'Full Phrase ({acronym})'.",
                location=f"Offset {match.start()}",
                agent_type="scanner:acronym_first_use",
                confidence=0.75,
                priority="low",
            )
        )

    return issues


def check_citation_resolution(text: str) -> list[DocumentIssue]:
    """Verify every [N]-style in-text citation has a matching reference entry.

    Returns [] if no References section is found (out of scope for this
    scanner) or if no numbered entries are detected in the references.
    """
    ref_match = _REFERENCES_HEADER.search(text)
    if not ref_match:
        return []

    body = text[: ref_match.start()]
    refs_text = text[ref_match.end() :]

    ref_nums: set[int] = set()
    for m in re.finditer(r"^\s*(?:\[(\d+)\]|(\d+)\.)\s", refs_text, re.MULTILINE):
        ref_nums.add(int(m.group(1) or m.group(2)))
    if not ref_nums:
        return []

    issues: list[DocumentIssue] = []
    seen: set[int] = set()
    for m in _CITATION_NUMERIC.finditer(body):
        raw = m.group(1)
        nums: list[int] = []
        for part in re.split(r"\s*,\s*", raw):
            if "-" in part:
                lo, hi = part.split("-")
                nums.extend(range(int(lo), int(hi) + 1))
            else:
                nums.append(int(part))
        for n in nums:
            if n in seen:
                continue
            seen.add(n)
            if n not in ref_nums:
                issues.append(
                    DocumentIssue(
                        issue_type="major",
                        category="references",
                        title=f"Citation [{n}] has no matching reference entry",
                        description=(
                            f"In-text citation [{n}] found, but no reference [{n}] in References section."
                        ),
                        recommendation=f"Add a reference entry [{n}] or renumber the citation.",
                        location=f"Offset {m.start()}",
                        agent_type="scanner:citation_resolution",
                        confidence=1.0,
                        priority="high",
                    )
                )
    return issues


def run_all(text: str) -> list[DocumentIssue]:
    """Run all consistency scanners and return the merged issue list."""
    return (
        check_figure_order(text)
        + check_acronym_first_use(text)
        + check_citation_resolution(text)
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest src/andamentum/whetstone/tests/test_consistency_scanners.py -v`
Expected: 13 passed.

- [ ] **Step 5: Commit**

```bash
git add src/andamentum/whetstone/consistency_scanners.py src/andamentum/whetstone/tests/test_consistency_scanners.py
git commit -m "feat(whetstone): deterministic scanners for internal consistency"
```

---

### Task 5: Checklist scanners (`checklist_scanners.py`)

**Files:**
- Create: `src/andamentum/whetstone/checklist_scanners.py`
- Create: `src/andamentum/whetstone/tests/test_checklist_scanners.py`

- [ ] **Step 1: Write the failing tests**

Create `src/andamentum/whetstone/tests/test_checklist_scanners.py`:

```python
"""Tests for checklist_scanners — deterministic baseline checks."""

from andamentum.whetstone.checklist_scanners import (
    check_all_figures_referenced,
    check_all_tables_referenced,
    check_authors_listed,
    check_citations_resolve,
    check_coi_statement,
    check_data_availability_statement,
    check_ethics_statement,
    check_figure_numbering_sequential,
    check_funding_statement,
    check_keywords_section,
    check_table_numbering_sequential,
)


# ---- figures & tables ------------------------------------------------------

def test_all_figures_referenced_pass():
    text = "Figure 1: The plot.\n\nBody sees Figure 1 twice: Figure 1 again."
    status, _ = check_all_figures_referenced(text)
    assert status == "pass"


def test_all_figures_referenced_fail():
    text = "Figure 1: The plot.\nFigure 2: Second plot.\n\nBody sees Figure 1 only."
    status, notes = check_all_figures_referenced(text)
    assert status == "fail"
    assert "2" in notes


def test_all_figures_referenced_unclear_no_figs():
    status, _ = check_all_figures_referenced("No figures here.")
    assert status == "unclear"


def test_figure_numbering_sequential_pass():
    text = "Figure 1: A.\nFigure 2: B.\nFigure 3: C."
    status, _ = check_figure_numbering_sequential(text)
    assert status == "pass"


def test_figure_numbering_sequential_fail():
    text = "Figure 1: A.\nFigure 3: C."
    status, _ = check_figure_numbering_sequential(text)
    assert status == "fail"


def test_all_tables_referenced_pass():
    text = "Table 1: Data.\n\nBody sees Table 1 clearly."
    status, _ = check_all_tables_referenced(text)
    assert status == "pass"


def test_all_tables_referenced_unclear_no_tables():
    status, _ = check_all_tables_referenced("No tables.")
    assert status == "unclear"


def test_table_numbering_sequential_pass():
    text = "Table 1: A.\nTable 2: B."
    status, _ = check_table_numbering_sequential(text)
    assert status == "pass"


def test_table_numbering_sequential_fail():
    text = "Table 1: A.\nTable 3: C."
    status, _ = check_table_numbering_sequential(text)
    assert status == "fail"


# ---- citations -------------------------------------------------------------

def test_citations_resolve_pass():
    text = "See [1] and [2].\n\nReferences\n[1] A.\n[2] B.\n"
    status, _ = check_citations_resolve(text)
    assert status == "pass"


def test_citations_resolve_fail():
    text = "See [1] and [3].\n\nReferences\n[1] A.\n[2] B.\n"
    status, notes = check_citations_resolve(text)
    assert status == "fail"
    assert "3" in notes


def test_citations_resolve_no_refs_section():
    status, _ = check_citations_resolve("See [1] and [2].")
    assert status == "unclear"


# ---- required statements ---------------------------------------------------

def test_coi_present():
    status, _ = check_coi_statement("We declare no conflict of interest.")
    assert status == "pass"


def test_coi_absent():
    status, _ = check_coi_statement("No declarations here.")
    assert status == "fail"


def test_coi_competing_interests():
    status, _ = check_coi_statement("Competing interests: none.")
    assert status == "pass"


def test_data_availability_present():
    status, _ = check_data_availability_statement("Data availability: on request.")
    assert status == "pass"


def test_data_availability_absent():
    status, _ = check_data_availability_statement("No mention of data.")
    assert status == "fail"


def test_ethics_unclear_no_subjects():
    status, _ = check_ethics_statement("A theoretical analysis of algorithms.")
    assert status == "unclear"


def test_ethics_pass():
    text = "We recruited 50 participants. IRB approval was obtained from the institutional review board."
    status, _ = check_ethics_statement(text)
    assert status == "pass"


def test_ethics_fail():
    text = "We recruited 50 participants. They completed the survey."
    status, _ = check_ethics_statement(text)
    assert status == "fail"


def test_funding_present():
    status, _ = check_funding_statement("This work was supported by NIH grant R01-12345.")
    assert status == "pass"


def test_funding_absent():
    status, _ = check_funding_statement("Just body text.")
    assert status == "fail"


# ---- hygiene ---------------------------------------------------------------

def test_keywords_present():
    status, _ = check_keywords_section("Keywords: reproducibility, methodology.\n\nAbstract: ...")
    assert status == "pass"


def test_keywords_absent():
    status, _ = check_keywords_section("Just a title.\n\nAbstract: ...")
    assert status == "fail"


def test_authors_pass():
    text = "Jane Doe\nDepartment of Computer Science, University of Somewhere\n\nAbstract: ..."
    status, _ = check_authors_listed(text)
    assert status == "pass"


def test_authors_unclear():
    status, _ = check_authors_listed("No affiliation keywords here.")
    assert status == "unclear"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest src/andamentum/whetstone/tests/test_checklist_scanners.py -v`
Expected: all fail with `ImportError`.

- [ ] **Step 3: Create the scanner module**

Create `src/andamentum/whetstone/checklist_scanners.py`:

```python
"""Deterministic scanners for baseline checklist items.

Each function takes document text and returns a `(status, notes)` tuple
that the orchestrator wraps into a `ChecklistItem`.

Constitution Rule 4: items that can be verified by regex/string-search
go here. Items requiring reading comprehension go to the LLM path.
"""

from __future__ import annotations

import re
from typing import Literal

Status = Literal["pass", "fail", "unclear"]


# ---------------------------------------------------------------------------
# Figures & tables
# ---------------------------------------------------------------------------

_FIGURE_CAPTION = re.compile(r"^\s*(?:Figure|Fig\.?)\s+(\d+)[\.:]", re.MULTILINE)
_FIGURE_REF = re.compile(r"\b(?:Figure|Fig\.?)\s+(\d+)\b")
_TABLE_CAPTION = re.compile(r"^\s*Table\s+(\d+)[\.:]", re.MULTILINE)
_TABLE_REF = re.compile(r"\bTable\s+(\d+)\b")


def check_all_figures_referenced(text: str) -> tuple[Status, str]:
    captions = {int(m.group(1)) for m in _FIGURE_CAPTION.finditer(text)}
    if not captions:
        return ("unclear", "No figure captions found in document.")
    refs = {int(m.group(1)) for m in _FIGURE_REF.finditer(text)}
    missing = captions - refs
    if missing:
        return ("fail", f"Figures without in-text references: {sorted(missing)}")
    return ("pass", f"All {len(captions)} figure captions referenced in text.")


def check_figure_numbering_sequential(text: str) -> tuple[Status, str]:
    nums = [int(m.group(1)) for m in _FIGURE_CAPTION.finditer(text)]
    if not nums:
        return ("unclear", "No figure captions found.")
    expected = list(range(1, len(nums) + 1))
    if sorted(nums) != expected:
        return ("fail", f"Figure captions numbered {sorted(nums)}; expected 1..{len(nums)}.")
    return ("pass", f"Figure captions numbered sequentially 1..{len(nums)}.")


def check_all_tables_referenced(text: str) -> tuple[Status, str]:
    captions = {int(m.group(1)) for m in _TABLE_CAPTION.finditer(text)}
    if not captions:
        return ("unclear", "No table captions found in document.")
    refs = {int(m.group(1)) for m in _TABLE_REF.finditer(text)}
    missing = captions - refs
    if missing:
        return ("fail", f"Tables without in-text references: {sorted(missing)}")
    return ("pass", f"All {len(captions)} table captions referenced in text.")


def check_table_numbering_sequential(text: str) -> tuple[Status, str]:
    nums = [int(m.group(1)) for m in _TABLE_CAPTION.finditer(text)]
    if not nums:
        return ("unclear", "No table captions found.")
    expected = list(range(1, len(nums) + 1))
    if sorted(nums) != expected:
        return ("fail", f"Table captions numbered {sorted(nums)}; expected 1..{len(nums)}.")
    return ("pass", f"Table captions numbered sequentially 1..{len(nums)}.")


# ---------------------------------------------------------------------------
# Citations
# ---------------------------------------------------------------------------

_REFERENCES_HEADER = re.compile(r"^\s*(?:References|Bibliography)\s*$", re.MULTILINE | re.IGNORECASE)


def check_citations_resolve(text: str) -> tuple[Status, str]:
    ref_match = _REFERENCES_HEADER.search(text)
    if not ref_match:
        return ("unclear", "No References section found.")
    body = text[: ref_match.start()]
    refs_text = text[ref_match.end() :]
    ref_nums: set[int] = set()
    for m in re.finditer(r"^\s*(?:\[(\d+)\]|(\d+)\.)\s", refs_text, re.MULTILINE):
        ref_nums.add(int(m.group(1) or m.group(2)))
    if not ref_nums:
        return ("unclear", "References section found but no numbered entries detected.")
    cit_nums: set[int] = set()
    for m in re.finditer(r"\[(\d+(?:\s*[,-]\s*\d+)*)\]", body):
        for part in re.split(r"\s*,\s*", m.group(1)):
            if "-" in part:
                lo, hi = part.split("-")
                cit_nums.update(range(int(lo), int(hi) + 1))
            else:
                cit_nums.add(int(part))
    unresolved = cit_nums - ref_nums
    if unresolved:
        return ("fail", f"Citations with no matching reference: {sorted(unresolved)}")
    return ("pass", f"All {len(cit_nums)} citations resolve to reference entries.")


# ---------------------------------------------------------------------------
# Required statements
# ---------------------------------------------------------------------------


def check_coi_statement(text: str) -> tuple[Status, str]:
    pattern = re.compile(
        r"(?:conflicts?\s+of\s+interests?|competing\s+(?:financial\s+)?interests?|declarations?\s+of\s+interest)",
        re.IGNORECASE,
    )
    if pattern.search(text):
        return ("pass", "Conflict-of-interest / competing-interests statement found.")
    return ("fail", "No conflict-of-interest or competing-interests statement found.")


def check_data_availability_statement(text: str) -> tuple[Status, str]:
    pattern = re.compile(r"data\s+(?:availability|accessibility|sharing)", re.IGNORECASE)
    if pattern.search(text):
        return ("pass", "Data availability statement found.")
    return ("fail", "No data availability statement found.")


def check_ethics_statement(text: str) -> tuple[Status, str]:
    subjects = bool(re.search(
        r"\b(?:human\s+subjects?|participants?|patients?|volunteers?|animals?|mice|rats|murine|primates?)\b",
        text,
        re.IGNORECASE,
    ))
    if not subjects:
        return ("unclear", "No human/animal subjects mentioned — ethics statement may not apply.")
    has_ethics = bool(re.search(
        r"(?:ethics\s+(?:approval|committee|statement|review\s+board)|IRB|IACUC|institutional\s+review)",
        text,
        re.IGNORECASE,
    ))
    if has_ethics:
        return ("pass", "Ethics / IRB / IACUC statement found.")
    return ("fail", "Human/animal subjects mentioned but no ethics statement found.")


def check_funding_statement(text: str) -> tuple[Status, str]:
    pattern = re.compile(
        r"(?:funding|supported\s+by|grant\s+(?:number|no\.?)|acknowledg(?:e)?ments?)",
        re.IGNORECASE,
    )
    if pattern.search(text):
        return ("pass", "Funding / acknowledgements statement found.")
    return ("fail", "No funding or acknowledgements statement found.")


# ---------------------------------------------------------------------------
# Hygiene
# ---------------------------------------------------------------------------


def check_keywords_section(text: str) -> tuple[Status, str]:
    pattern = re.compile(r"^\s*Key\s*words?\s*[:\s]", re.MULTILINE | re.IGNORECASE)
    if pattern.search(text):
        return ("pass", "Keywords section found.")
    return ("fail", "No keywords section found.")


def check_authors_listed(text: str) -> tuple[Status, str]:
    head = text[:2000]
    pattern = re.compile(
        r"\b(?:Department|Institute|School|University|Laboratory|Center|Centre|Faculty|Hospital)\b"
    )
    if pattern.search(head):
        return ("pass", "Affiliation markers found near document head.")
    return ("unclear", "No standard affiliation keywords found in the first 2000 characters.")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest src/andamentum/whetstone/tests/test_checklist_scanners.py -v`
Expected: all passed.

- [ ] **Step 5: Commit**

```bash
git add src/andamentum/whetstone/checklist_scanners.py src/andamentum/whetstone/tests/test_checklist_scanners.py
git commit -m "feat(whetstone): deterministic scanners for baseline checklist items"
```

---

### Task 6: `consistency_reviewer` agent

**Files:**
- Create: `src/andamentum/whetstone/agents/consistency.py`
- Modify: `src/andamentum/whetstone/tests/test_agent_registry.py`

- [ ] **Step 1: Write the failing test**

Append to `src/andamentum/whetstone/tests/test_agent_registry.py`:

```python
def test_consistency_reviewer_registered():
    from andamentum.whetstone.agents import AGENT_REGISTRY
    from andamentum.whetstone.agents.output_models import ConsistencyReviewOutput

    assert "consistency_reviewer" in AGENT_REGISTRY
    defn = AGENT_REGISTRY["consistency_reviewer"]
    assert defn.output_model is ConsistencyReviewOutput
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest src/andamentum/whetstone/tests/test_agent_registry.py::test_consistency_reviewer_registered -v`
Expected: `AssertionError` — agent not in registry.

- [ ] **Step 3: Create the agent module**

Create `src/andamentum/whetstone/agents/consistency.py`:

```python
"""Consistency-reviewer agent — reading-comprehension pass over a draft.

Only handles inconsistencies that require reading comprehension:
numbers disagreeing across sections, terminology drift, claim emphasis
shifting. Mechanical checks (figure order, acronym first-use,
citation resolution) live in consistency_scanners.
"""

from __future__ import annotations

from . import AgentDefinition, register_agent
from .output_models import ConsistencyReviewOutput

_CONSISTENCY_PROMPT = """\
# Internal consistency reviewer

You are reviewing a draft the author wrote themselves, to catch internal
inconsistencies before submission. This is NOT peer review.

## Focus on reading-comprehension issues

- Numbers or statistics that disagree between abstract, results, and
  conclusions (e.g. "n=50" in abstract, "n=48" in results)
- Terminology drift — the same concept called different names across
  sections (e.g. "cohort" and "sample" used interchangeably)
- Claims emphasized differently across sections (abstract headlines
  finding A, conclusion headlines finding B)
- Tense, voice, or person shifts across sections
- Contradicting statements about methods, scope, or population

## Do NOT comment on

- Figure numbering order (handled by a scanner)
- Reference-list completeness or formatting (handled elsewhere)
- Acronym first-use definition (handled by a scanner)
- Grammar, typos, style, or word choice (handled by the edit task)

## Output

For each issue you find, emit a DocumentIssue with:
- `issue_type`: "major" for real contradictions; "minor" for drift;
  "suggestion" for minor polish
- `category`: "consistency"
- `title`: brief, specific
- `description`: what the inconsistency is and where — quote the
  excerpts if possible
- `recommendation`: concrete fix
- `location`: section names where the inconsistency occurs
- `agent_type`: "consistency_reviewer"
- `confidence`: 0.0–1.0

Quality over quantity. Emit 0–8 issues. Only flag things you are
confident about.
"""


register_agent(
    AgentDefinition(
        name="consistency_reviewer",
        prompt=_CONSISTENCY_PROMPT,
        output_model=ConsistencyReviewOutput,
        retries=2,
    )
)
```

- [ ] **Step 4: Wire into `agents/__init__.py`**

Edit `src/andamentum/whetstone/agents/__init__.py` to add the import at the bottom, next to the other domain-module imports. Find this block:

```python
# Import domain modules to populate the registry on first access.
from . import editing as _editing  # noqa: E402, F401
from . import review as _review  # noqa: E402, F401
from . import synthesis as _synthesis  # noqa: E402, F401
from . import multi_expert as _multi_expert  # noqa: E402, F401
from . import custom as _custom  # noqa: E402, F401
```

Add one line:

```python
from . import consistency as _consistency  # noqa: E402, F401
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest src/andamentum/whetstone/tests/test_agent_registry.py -v`
Expected: all pass, including new one.

- [ ] **Step 6: Commit**

```bash
git add src/andamentum/whetstone/agents/consistency.py src/andamentum/whetstone/agents/__init__.py src/andamentum/whetstone/tests/test_agent_registry.py
git commit -m "feat(whetstone): register consistency_reviewer agent"
```

---

### Task 7: `checklist` agents + `BASELINE_CHECKS`

**Files:**
- Create: `src/andamentum/whetstone/agents/checklist.py`
- Modify: `src/andamentum/whetstone/agents/__init__.py`
- Modify: `src/andamentum/whetstone/tests/test_agent_registry.py`

- [ ] **Step 1: Write the failing tests**

Append to `src/andamentum/whetstone/tests/test_agent_registry.py`:

```python
def test_checklist_agents_registered():
    from andamentum.whetstone.agents import AGENT_REGISTRY
    assert "checklist_item_evaluator" in AGENT_REGISTRY
    assert "journal_guidelines_extractor" in AGENT_REGISTRY


def test_baseline_checks_shape():
    from andamentum.whetstone.agents.checklist import BASELINE_CHECKS
    assert len(BASELINE_CHECKS) >= 10
    for check in BASELINE_CHECKS:
        assert check.name
        assert check.category
        if check.kind == "deterministic":
            assert check.scanner is not None
            assert check.prompt_hint is None
        else:
            assert check.prompt_hint is not None
            assert check.scanner is None


def test_baseline_scanners_exist():
    """Every deterministic BASELINE_CHECK must point to a real scanner function."""
    from andamentum.whetstone.agents.checklist import BASELINE_CHECKS
    from andamentum.whetstone import checklist_scanners

    for check in BASELINE_CHECKS:
        if check.kind == "deterministic":
            assert hasattr(checklist_scanners, check.scanner), (
                f"BASELINE_CHECK '{check.name}' references missing scanner '{check.scanner}'"
            )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest src/andamentum/whetstone/tests/test_agent_registry.py -v`
Expected: 3 new tests fail.

- [ ] **Step 3: Create the agent module**

Create `src/andamentum/whetstone/agents/checklist.py`:

```python
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
```

- [ ] **Step 4: Wire into `agents/__init__.py`**

Edit `src/andamentum/whetstone/agents/__init__.py`, adding to the import block:

```python
from . import checklist as _checklist  # noqa: E402, F401
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest src/andamentum/whetstone/tests/test_agent_registry.py -v`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add src/andamentum/whetstone/agents/checklist.py src/andamentum/whetstone/agents/__init__.py src/andamentum/whetstone/tests/test_agent_registry.py
git commit -m "feat(whetstone): baseline checklist + evaluator/extractor agents"
```

---

### Task 8: Orchestrator — `_run_consistency`

**Files:**
- Modify: `src/andamentum/whetstone/orchestrator.py`
- Create: `src/andamentum/whetstone/tests/test_consistency_orchestrator.py`

- [ ] **Step 1: Write the failing tests**

Create `src/andamentum/whetstone/tests/test_consistency_orchestrator.py`:

```python
"""Tests for the consistency-task orchestrator path."""

from dataclasses import dataclass
from typing import Any

import pytest

from andamentum.whetstone import orchestrator
from andamentum.whetstone.agents import AGENT_REGISTRY
from andamentum.whetstone.agents.output_models import ConsistencyReviewOutput
from andamentum.whetstone.issues import DocumentIssue
from andamentum.whetstone.orchestrator import ReviewResult


@dataclass
class _FakeRunner:
    """Minimal AgentRunner stand-in.

    `returns` maps agent name → output object. Raises KeyError for
    unexpected calls so tests catch unintended dispatch.
    """

    returns: dict[str, Any]

    async def run(self, defn, **kwargs):  # noqa: ANN001
        return self.returns[defn.name]


async def test_consistency_merges_scanner_and_llm(monkeypatch):
    # LLM returns a tense-drift issue
    llm_out = ConsistencyReviewOutput(
        issues=[
            DocumentIssue(
                issue_type="minor", category="consistency",
                title="Tense drift between methods and results",
                description="Methods use past tense; results use present.",
                agent_type="consistency_reviewer",
            ),
        ]
    )
    runner = _FakeRunner(returns={"consistency_reviewer": llm_out})

    result = ReviewResult(task="consistency")
    # Document has Figure 2 before Figure 1 — scanner should flag it
    doc = "First see Figure 2. Later Figure 1 explains."
    await orchestrator._run_consistency(runner, result, doc, verbose=False)

    assert any(i.agent_type == "scanner:figure_order" for i in result.issues)
    assert any(i.agent_type == "consistency_reviewer" for i in result.issues)
    assert len(result.issues) == 2


async def test_consistency_no_scanner_findings():
    """Clean doc → only LLM issues in result."""
    llm_out = ConsistencyReviewOutput(issues=[])
    runner = _FakeRunner(returns={"consistency_reviewer": llm_out})
    result = ReviewResult(task="consistency")
    await orchestrator._run_consistency(runner, result, "Clean text with no problems.", verbose=False)
    assert result.issues == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest src/andamentum/whetstone/tests/test_consistency_orchestrator.py -v`
Expected: `AttributeError: module 'andamentum.whetstone.orchestrator' has no attribute '_run_consistency'`.

- [ ] **Step 3: Add the orchestrator helper**

In `src/andamentum/whetstone/orchestrator.py`, add a new import at the top (with the other local imports):

```python
from . import consistency_scanners
```

Then add a new function, placed after `_run_edit` and before `_run_standard_review`:

```python
# ---------------------------------------------------------------------------
# Task: Consistency
# ---------------------------------------------------------------------------


async def _run_consistency(
    runner: AgentRunner,
    result: ReviewResult,
    content: str,
    verbose: bool,
) -> None:
    """Run deterministic scanners + the consistency_reviewer LLM agent."""
    print("Running consistency scanners...", file=sys.stderr)
    scanner_issues = consistency_scanners.run_all(content)
    logger.debug("consistency scanners produced %d issues", len(scanner_issues))

    print("Running consistency_reviewer agent...", file=sys.stderr)
    llm_output = await _run_one(runner, "consistency_reviewer", document=content)
    llm_issues = getattr(llm_output, "issues", [])

    result.issues = [*scanner_issues, *llm_issues]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest src/andamentum/whetstone/tests/test_consistency_orchestrator.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/andamentum/whetstone/orchestrator.py src/andamentum/whetstone/tests/test_consistency_orchestrator.py
git commit -m "feat(whetstone): _run_consistency orchestrator path"
```

---

### Task 9: Orchestrator — `_run_checklist`

**Files:**
- Modify: `src/andamentum/whetstone/orchestrator.py`
- Create: `src/andamentum/whetstone/tests/test_checklist_orchestrator.py`

- [ ] **Step 1: Write the failing tests**

Create `src/andamentum/whetstone/tests/test_checklist_orchestrator.py`:

```python
"""Tests for the checklist-task orchestrator path."""

from dataclasses import dataclass, field
from typing import Any

import pytest

from andamentum.whetstone import orchestrator
from andamentum.whetstone.agents.checklist import BASELINE_CHECKS
from andamentum.whetstone.agents.output_models import ExtractedChecklistNames
from andamentum.whetstone.models import ChecklistItem
from andamentum.whetstone.orchestrator import ReviewResult


@dataclass
class _FakeRunner:
    calls: list = field(default_factory=list)
    extractor_items: list[str] | None = None
    evaluator_factory: Any = None  # callable(check_name) -> ChecklistItem or raises

    async def run(self, defn, **kwargs):  # noqa: ANN001
        self.calls.append((defn.name, kwargs))
        if defn.name == "journal_guidelines_extractor":
            return ExtractedChecklistNames(items=self.extractor_items or [])
        if defn.name == "checklist_item_evaluator":
            return self.evaluator_factory(kwargs["check_name"])
        raise AssertionError(f"Unexpected agent call: {defn.name}")


def _ok_evaluator(check_name: str) -> ChecklistItem:
    return ChecklistItem(name=check_name, status="pass", notes="looks fine")


# Sample document that satisfies the deterministic baseline checks
_GOOD_DOC = """\
Jane Doe
Department of Computer Science, University of Somewhere

Keywords: reproducibility, methodology

Abstract: short abstract.

Figure 1: The setup.
Figure 2: The result.

Body references Figure 1 and Figure 2 in turn.

Table 1: Data.

Body references Table 1.

We had 50 participants. IRB approval obtained.
Conflict of interest: none.
Data availability: on request.
This work was supported by NIH grant X.

As shown [1] and [2].

References
[1] First.
[2] Second.
"""


async def test_checklist_baseline_only():
    runner = _FakeRunner(evaluator_factory=_ok_evaluator)
    result = ReviewResult(task="checklist")
    await orchestrator._run_checklist(runner, result, _GOOD_DOC, guidelines=None, verbose=False)

    assert len(result.checklist) == len(BASELINE_CHECKS)
    # All items tagged as baseline
    assert all(item.source == "baseline" for item in result.checklist)
    # Category tagging flowed through
    categories = {item.category for item in result.checklist}
    assert "abstract" in categories
    assert "figures" in categories
    # No journal-extractor calls
    assert not any(name == "journal_guidelines_extractor" for name, _ in runner.calls)


async def test_checklist_baseline_plus_journal():
    runner = _FakeRunner(
        extractor_items=["Funding disclosure complete", "Preprint policy respected"],
        evaluator_factory=_ok_evaluator,
    )
    result = ReviewResult(task="checklist")
    guidelines = "Short guidelines text."
    await orchestrator._run_checklist(runner, result, _GOOD_DOC, guidelines=guidelines, verbose=False)

    baseline_count = len(BASELINE_CHECKS)
    assert len(result.checklist) == baseline_count + 2
    journal_items = [i for i in result.checklist if i.source == "journal"]
    assert len(journal_items) == 2
    assert all(i.category == "journal" for i in journal_items)
    # Journal item names are what the extractor returned
    journal_names = {i.name for i in journal_items}
    assert "Funding disclosure complete" in journal_names


async def test_checklist_journal_item_failure_becomes_unclear():
    # First journal item fails, second succeeds
    def flaky(check_name: str) -> ChecklistItem:
        if check_name == "Will fail":
            raise RuntimeError("model timeout")
        return _ok_evaluator(check_name)

    runner = _FakeRunner(
        extractor_items=["Will fail", "Will succeed"],
        evaluator_factory=flaky,
    )
    result = ReviewResult(task="checklist")
    await orchestrator._run_checklist(runner, result, _GOOD_DOC, guidelines="x", verbose=False)

    journal_items = [i for i in result.checklist if i.source == "journal"]
    assert len(journal_items) == 2
    failed = next(i for i in journal_items if i.name == "Will fail")
    assert failed.status == "unclear"
    assert "model timeout" in failed.notes


async def test_checklist_baseline_evaluator_failure_raises():
    def always_fail(check_name: str) -> ChecklistItem:
        raise RuntimeError("hard failure")

    runner = _FakeRunner(evaluator_factory=always_fail)
    result = ReviewResult(task="checklist")
    with pytest.raises(RuntimeError):
        await orchestrator._run_checklist(runner, result, _GOOD_DOC, guidelines=None, verbose=False)


async def test_checklist_llm_item_name_is_authoritative():
    """Orchestrator overwrites whatever name the LLM returned."""
    def drifted(check_name: str) -> ChecklistItem:
        return ChecklistItem(name="DRIFTED NAME", status="pass", notes="")

    runner = _FakeRunner(evaluator_factory=drifted)
    result = ReviewResult(task="checklist")
    await orchestrator._run_checklist(runner, result, _GOOD_DOC, guidelines=None, verbose=False)

    for item in result.checklist:
        assert item.name != "DRIFTED NAME"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest src/andamentum/whetstone/tests/test_checklist_orchestrator.py -v`
Expected: all fail with `AttributeError` on `_run_checklist`.

- [ ] **Step 3: Add the orchestrator helper**

In `src/andamentum/whetstone/orchestrator.py`, add imports near the top (with the other local imports):

```python
from . import checklist_scanners
from .agents.checklist import BASELINE_CHECKS
from .models import BaselineCheck, ChecklistItem
```

Add the new function after `_run_panel_review`:

```python
# ---------------------------------------------------------------------------
# Task: Checklist
# ---------------------------------------------------------------------------


async def _evaluate_baseline_item(
    runner: AgentRunner,
    check: BaselineCheck,
    content: str,
) -> ChecklistItem:
    """Dispatch a single baseline check to its scanner or LLM evaluator."""
    if check.kind == "deterministic":
        assert check.scanner is not None
        func = getattr(checklist_scanners, check.scanner)
        status, notes = func(content)
        return ChecklistItem(
            name=check.name, status=status, notes=notes,
            category=check.category, source="baseline",
        )

    item = await _run_one(
        runner,
        "checklist_item_evaluator",
        document=content,
        check_name=check.name,
        prompt_hint=check.prompt_hint or "",
    )
    # Overwrite LLM-drifted metadata with authoritative values
    item.name = check.name
    item.category = check.category
    item.source = "baseline"
    return item


async def _evaluate_journal_item(
    runner: AgentRunner,
    check_name: str,
    content: str,
) -> ChecklistItem:
    """Evaluate one journal-extracted item. Failures become 'unclear'.

    Per Constitution Rule 5, baseline-item failures are hard errors, but
    journal-extracted items come from fuzzy extractor output, so a
    single failure is an acceptable soft failure.
    """
    try:
        item = await _run_one(
            runner, "checklist_item_evaluator",
            document=content, check_name=check_name, prompt_hint="",
        )
        item.name = check_name
        item.category = "journal"
        item.source = "journal"
        return item
    except Exception as exc:
        logger.warning("journal item %r evaluation failed: %s", check_name, exc)
        return ChecklistItem(
            name=check_name, status="unclear",
            notes=f"Evaluation failed: {exc}",
            category="journal", source="journal",
        )


async def _run_checklist(
    runner: AgentRunner,
    result: ReviewResult,
    content: str,
    guidelines: Optional[str],
    verbose: bool,
) -> None:
    """Run the baseline checklist and, if guidelines are provided, the journal layer."""
    print(f"Running baseline checklist ({len(BASELINE_CHECKS)} items)...", file=sys.stderr)

    baseline_items = await asyncio.gather(
        *[_evaluate_baseline_item(runner, check, content) for check in BASELINE_CHECKS]
    )
    result.checklist.extend(baseline_items)

    if guidelines is None:
        return

    print("Extracting journal-specific items...", file=sys.stderr)
    extracted = await _run_one(runner, "journal_guidelines_extractor", guidelines=guidelines)
    journal_names = list(getattr(extracted, "items", []))
    logger.info("journal extractor produced %d items", len(journal_names))

    if not journal_names:
        return

    print(f"Evaluating {len(journal_names)} journal items...", file=sys.stderr)
    journal_items = await asyncio.gather(
        *[_evaluate_journal_item(runner, name, content) for name in journal_names]
    )
    result.checklist.extend(journal_items)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest src/andamentum/whetstone/tests/test_checklist_orchestrator.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add src/andamentum/whetstone/orchestrator.py src/andamentum/whetstone/tests/test_checklist_orchestrator.py
git commit -m "feat(whetstone): _run_checklist orchestrator path"
```

---

### Task 10: `sharpen_document` dispatch + validation

**Files:**
- Modify: `src/andamentum/whetstone/orchestrator.py`
- Modify: `src/andamentum/whetstone/tests/test_orchestrator_smoke.py`

- [ ] **Step 1: Write the failing tests**

Append to `src/andamentum/whetstone/tests/test_orchestrator_smoke.py`:

```python
async def test_sharpen_document_accepts_consistency_task():
    """consistency task is in the valid-tasks tuple; validation passes before runner call."""
    # Use a model string that parses but never connects — the FakeRunner is unused
    # because we only care that the ValueError from task validation is NOT raised.
    from unittest.mock import patch

    async def noop(*a, **kw):
        return None

    with patch.object(orchestrator, "_run_consistency", noop):
        r = await sharpen_document("text", task="consistency", model="anthropic:claude-haiku-4-5")
        assert r.task == "consistency"


async def test_sharpen_document_accepts_checklist_task():
    from unittest.mock import patch

    async def noop(*a, **kw):
        return None

    with patch.object(orchestrator, "_run_checklist", noop):
        r = await sharpen_document("text", task="checklist", model="anthropic:claude-haiku-4-5")
        assert r.task == "checklist"


async def test_sharpen_document_guidelines_with_wrong_task_raises():
    with pytest.raises(ValueError, match="guidelines"):
        await sharpen_document("text", task="review", guidelines="X", model="anthropic:claude-haiku-4-5")
```

Add at the top of the test file (if not already present):

```python
from andamentum.whetstone import orchestrator
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest src/andamentum/whetstone/tests/test_orchestrator_smoke.py -v`
Expected: 3 new tests fail — `ValueError: Invalid task` or missing kwarg.

- [ ] **Step 3: Update `sharpen_document`**

In `src/andamentum/whetstone/orchestrator.py`, replace the `sharpen_document` body. The key changes: add `guidelines` kwarg, extend the valid-task tuple, add the guidelines-with-wrong-task guard, dispatch to the new helpers.

Find:

```python
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
```

Replace with:

```python
async def sharpen_document(
    content: str,
    *,
    task: str = "review",
    num_experts: int = 3,
    criteria: Optional[str] = None,
    editors: Optional[list[str]] = None,
    guidelines: Optional[str] = None,
    model: str = "openai:gpt-4o",
    verbose: bool = False,
) -> ReviewResult:
```

Update the docstring `Args` section by adding, after the `editors:` entry:

```
        guidelines: Journal author guidelines (free text). Only valid
            for task="checklist"; raises ValueError otherwise.
```

Replace the task-validation line:

```python
    if task not in ("edit", "review", "panel"):
        raise ValueError(f"Invalid task '{task}'. Must be 'edit', 'review', or 'panel'.")
```

with:

```python
    valid_tasks = ("edit", "review", "panel", "consistency", "checklist")
    if task not in valid_tasks:
        raise ValueError(f"Invalid task '{task}'. Must be one of {valid_tasks}.")
    if guidelines is not None and task != "checklist":
        raise ValueError(
            f"guidelines is only valid with task='checklist'; got task='{task}'."
        )
```

Extend the dispatch block. Find:

```python
    elif task == "panel":
        await _run_panel_review(runner, result, content, num_experts, verbose)
```

Add after:

```python
    elif task == "consistency":
        await _run_consistency(runner, result, content, verbose)
    elif task == "checklist":
        await _run_checklist(runner, result, content, guidelines, verbose)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest src/andamentum/whetstone/tests/test_orchestrator_smoke.py -v`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/andamentum/whetstone/orchestrator.py src/andamentum/whetstone/tests/test_orchestrator_smoke.py
git commit -m "feat(whetstone): dispatch consistency/checklist from sharpen_document"
```

---

### Task 11: Export `ChecklistItem` from the package

**Files:**
- Modify: `src/andamentum/whetstone/__init__.py`
- Modify: `src/andamentum/whetstone/tests/test_public_api.py`

- [ ] **Step 1: Write the failing test**

Append to `src/andamentum/whetstone/tests/test_public_api.py`:

```python
def test_checklist_item_exported():
    from andamentum.whetstone import ChecklistItem
    item = ChecklistItem(name="x", status="pass", notes="y")
    assert item.name == "x"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest src/andamentum/whetstone/tests/test_public_api.py::test_checklist_item_exported -v`
Expected: `ImportError: cannot import name 'ChecklistItem'`.

- [ ] **Step 3: Add the export**

Edit `src/andamentum/whetstone/__init__.py`. Change the models-import line:

```python
from .models import DocumentPatch, PatchApplicationResult
```

to:

```python
from .models import ChecklistItem, DocumentPatch, PatchApplicationResult
```

Update `__all__`, inserting `"ChecklistItem"` in the Data models group:

```python
__all__ = [
    # Public entry point
    "sharpen_document",
    "ReviewResult",
    # Data models
    "DocumentPatch",
    "DocumentIssue",
    "ChecklistItem",
    "PatchApplicationResult",
    # Renderers
    "render_docx",
    "render_html",
    "render_diff",
    "apply_patches",
    # Agents (for introspection / extension)
    "AgentDefinition",
    "AGENT_REGISTRY",
    # Dynamic schema helpers
    "convert_fields_to_schema",
    "create_output_model",
]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest src/andamentum/whetstone/tests/test_public_api.py -v`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/andamentum/whetstone/__init__.py src/andamentum/whetstone/tests/test_public_api.py
git commit -m "feat(whetstone): export ChecklistItem from package __init__"
```

---

### Task 12: CLI — task choices + `--guidelines`

**Files:**
- Modify: `src/andamentum/whetstone/cli.py`
- Modify: `src/andamentum/whetstone/tests/test_cli.py`

- [ ] **Step 1: Write the failing tests**

Append to `src/andamentum/whetstone/tests/test_cli.py` (imports at top if not already there):

```python
def test_cli_accepts_consistency_task():
    from andamentum.whetstone.cli import _build_parser
    parser = _build_parser()
    args = parser.parse_args(["doc.md", "--task", "consistency"])
    assert args.task == "consistency"


def test_cli_accepts_checklist_task():
    from andamentum.whetstone.cli import _build_parser
    parser = _build_parser()
    args = parser.parse_args(["doc.md", "--task", "checklist"])
    assert args.task == "checklist"


def test_cli_guidelines_flag_parses():
    from andamentum.whetstone.cli import _build_parser
    parser = _build_parser()
    args = parser.parse_args(["doc.md", "--task", "checklist", "--guidelines", "hello"])
    assert args.guidelines == "hello"


def test_cli_resolve_guidelines_inline(tmp_path):
    from andamentum.whetstone.cli import _resolve_guidelines
    assert _resolve_guidelines("some text") == "some text"
    assert _resolve_guidelines(None) is None
    assert _resolve_guidelines("") is None


def test_cli_resolve_guidelines_from_file(tmp_path):
    from andamentum.whetstone.cli import _resolve_guidelines
    p = tmp_path / "guidelines.txt"
    p.write_text("rules here", encoding="utf-8")
    assert _resolve_guidelines(f"@{p}") == "rules here"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest src/andamentum/whetstone/tests/test_cli.py -v`
Expected: new tests fail — `--task choices` rejects new values; `--guidelines` unknown; `_resolve_guidelines` missing.

- [ ] **Step 3: Update the CLI**

In `src/andamentum/whetstone/cli.py`:

Find the `--task` argument definition:

```python
    parser.add_argument("--task", choices=["edit", "review", "panel"], default="review", help="Task (default: review)")
```

Replace with:

```python
    parser.add_argument(
        "--task",
        choices=["edit", "review", "panel", "consistency", "checklist"],
        default="review",
        help="Task (default: review)",
    )
```

Add a new argument (adjacent to `--criteria`):

```python
    parser.add_argument(
        "--guidelines",
        type=str,
        default=None,
        help="Journal author guidelines (text or @filepath). Only valid with --task checklist.",
    )
```

Add a `_resolve_guidelines` helper next to `_resolve_criteria`:

```python
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
```

Update `_run()` to resolve and validate `--guidelines` and pass it through. Find:

```python
    content = _read_document(args.file)
    model = _resolve_model(args)
    criteria = _resolve_criteria(args.criteria)
```

Insert after:

```python
    guidelines = _resolve_guidelines(args.guidelines)
    if guidelines is not None and args.task != "checklist":
        print("Error: --guidelines is only valid with --task checklist", file=sys.stderr)
        sys.exit(2)
```

Find the `sharpen_document` call. Update it to pass `guidelines`:

```python
    result = await sharpen_document(
        content,
        task=args.task,
        num_experts=args.num_experts,
        criteria=criteria,
        guidelines=guidelines,
        model=model,
        verbose=args.verbose,
    )
```

Update the progress-banner line. Find:

```python
    parts = [f"Task: {args.task}"]
    if criteria:
        parts.append("Criteria: custom")
    if args.task == "panel":
        parts.append(f"Experts: {args.num_experts}")
```

Insert after the `criteria` branch:

```python
    if guidelines:
        parts.append("Guidelines: provided")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest src/andamentum/whetstone/tests/test_cli.py -v`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/andamentum/whetstone/cli.py src/andamentum/whetstone/tests/test_cli.py
git commit -m "feat(whetstone): CLI support for consistency and checklist tasks"
```

---

### Task 13: `render_diff` — checklist markdown output

**Files:**
- Modify: `src/andamentum/whetstone/renderers/diff.py`
- Modify: `src/andamentum/whetstone/tests/test_diff_renderer.py`

- [ ] **Step 1: Inspect current `render_diff` signature**

Run: `sed -n '1,40p' src/andamentum/whetstone/renderers/diff.py`

The existing signature accepts `patches`, `issues`, and optional `synthesis_text`. We're adding optional `checklist` parameter — keyword-only to avoid breaking callers.

- [ ] **Step 2: Write the failing test**

Append to `src/andamentum/whetstone/tests/test_diff_renderer.py`:

```python
def test_render_diff_checklist_block():
    from andamentum.whetstone import ChecklistItem
    from andamentum.whetstone.renderers import render_diff

    items = [
        ChecklistItem(name="Abstract words", status="pass", notes="240 on p.1", category="abstract"),
        ChecklistItem(name="Ethics statement", status="fail", notes="Missing IRB block", category="statements"),
        ChecklistItem(name="Keywords", status="unclear", notes="Found but inline", category="hygiene"),
    ]
    output = render_diff(patches=[], issues=[], original_content="", checklist=items)
    assert "Abstract words" in output
    assert "Ethics statement" in output
    assert "PASS" in output or "✓" in output
    assert "FAIL" in output or "✗" in output


def test_render_diff_no_checklist_keeps_old_output():
    from andamentum.whetstone.renderers import render_diff
    out_no = render_diff(patches=[], issues=[], original_content="")
    out_empty_cl = render_diff(patches=[], issues=[], original_content="", checklist=[])
    assert out_no == out_empty_cl
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest src/andamentum/whetstone/tests/test_diff_renderer.py -v`
Expected: new tests fail with `TypeError: unexpected keyword 'checklist'`.

- [ ] **Step 4: Extend `render_diff`**

Edit `src/andamentum/whetstone/renderers/diff.py`. Update the signature to accept `checklist`:

```python
def render_diff(
    *,
    patches: list,
    issues: list,
    original_content: str,
    synthesis_text: str | None = None,
    checklist: list | None = None,
) -> str:
```

Add a small helper near the top of the module (after existing imports):

```python
_STATUS_MARKER = {"pass": "✓ PASS", "fail": "✗ FAIL", "unclear": "? UNCLEAR"}


def _render_checklist_markdown(items: list) -> str:
    if not items:
        return ""
    from collections import defaultdict
    by_cat: dict[str, list] = defaultdict(list)
    for it in items:
        by_cat[it.category or "other"].append(it)

    passes = sum(1 for i in items if i.status == "pass")
    fails = sum(1 for i in items if i.status == "fail")
    unclears = sum(1 for i in items if i.status == "unclear")

    lines = [
        "## Pre-submission checklist",
        "",
        f"**Summary:** {passes} pass · {fails} fail · {unclears} unclear (of {len(items)} items)",
        "",
    ]
    for cat in sorted(by_cat):
        lines.append(f"### {cat.title()}")
        lines.append("")
        for it in by_cat[cat]:
            marker = _STATUS_MARKER.get(it.status, "?")
            lines.append(f"- **{marker}** {it.name}")
            if it.notes:
                lines.append(f"  - {it.notes}")
        lines.append("")
    return "\n".join(lines)
```

Inside `render_diff` body, append the checklist section to the output. Find where the return happens, and insert just before it (or wherever the function assembles its output). The simplest addition: after the existing body is built and just before returning, add:

```python
    if checklist:
        output = _render_checklist_markdown(checklist) + "\n\n" + output
```

(Adjust to the actual shape of the existing function — the principle is: prepend the checklist block when present.)

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest src/andamentum/whetstone/tests/test_diff_renderer.py -v`
Expected: all pass.

- [ ] **Step 6: Wire into the CLI output path**

Edit `src/andamentum/whetstone/cli.py` `_render_output()` — every call to `render_diff(...)` needs to pass the checklist. Find both call sites (the stdout default and the `.md` output branch). Change:

```python
        diff_output = render_diff(
            patches=patches,
            issues=issues,
            original_content=content,
            synthesis_text=synthesis_text,
        )
```

to:

```python
        checklist = getattr(result, "checklist", None) or None
        diff_output = render_diff(
            patches=patches,
            issues=issues,
            original_content=content,
            synthesis_text=synthesis_text,
            checklist=checklist,
        )
```

in both locations.

- [ ] **Step 7: Commit**

```bash
git add src/andamentum/whetstone/renderers/diff.py src/andamentum/whetstone/cli.py src/andamentum/whetstone/tests/test_diff_renderer.py
git commit -m "feat(whetstone): render_diff supports checklist output"
```

---

### Task 14: `render_html` — checklist section

**Files:**
- Modify: `src/andamentum/whetstone/renderers/html.py`
- Modify: `src/andamentum/whetstone/tests/test_html_renderer.py`

- [ ] **Step 1: Write the failing test**

Append to `src/andamentum/whetstone/tests/test_html_renderer.py`:

```python
def test_html_renders_checklist_items():
    from andamentum.whetstone import ChecklistItem, ReviewResult
    from andamentum.whetstone.renderers import render_html

    result = ReviewResult(
        task="checklist",
        checklist=[
            ChecklistItem(name="Abstract wordcount", status="pass", notes="240 words", category="abstract"),
            ChecklistItem(name="Ethics statement", status="fail", notes="Missing", category="statements"),
        ],
    )
    html = render_html(result=result, original_content="")
    assert "Abstract wordcount" in html
    assert "Ethics statement" in html
    assert "fail" in html.lower() or "✗" in html
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest src/andamentum/whetstone/tests/test_html_renderer.py::test_html_renders_checklist_items -v`
Expected: fail — content missing.

- [ ] **Step 3: Inspect the existing html.py to find where to add the checklist atoms**

Run: `sed -n '1,40p' src/andamentum/whetstone/renderers/html.py`

The existing renderer walks a `ReviewResult` into typeset atoms. We add a section for `result.checklist` when non-empty.

Locate `_TASK_TITLES` and the main `render_html` function. Add a new task title and a new atom-building helper.

In `_TASK_TITLES`, add entries:

```python
    "consistency": "Whetstone — Internal Consistency",
    "checklist": "Whetstone — Pre-Submission Checklist",
```

Add this helper function (next to existing atom-builders):

```python
def _checklist_atoms(items: list) -> list:
    """Build typeset atoms for a list of ChecklistItem."""
    from andamentum.typeset import heading, items as items_atom, callout

    if not items:
        return []

    passes = sum(1 for i in items if i.status == "pass")
    fails = sum(1 for i in items if i.status == "fail")
    unclears = sum(1 for i in items if i.status == "unclear")

    atoms = [
        callout(
            kind="info",
            body=(
                f"<strong>{passes}</strong> pass · "
                f"<strong>{fails}</strong> fail · "
                f"<strong>{unclears}</strong> unclear "
                f"(of {len(items)} checks)"
            ),
        ),
    ]

    from collections import defaultdict
    by_cat: dict[str, list] = defaultdict(list)
    for it in items:
        by_cat[it.category or "other"].append(it)

    status_icon = {"pass": "✓", "fail": "✗", "unclear": "?"}
    for cat in sorted(by_cat):
        atoms.append(heading(level=2, text=cat.title()))
        entries = []
        for it in by_cat[cat]:
            icon = status_icon.get(it.status, "?")
            body = f"<strong>{icon} {it.status.upper()} — {it.name}</strong>"
            if it.notes:
                body += f"<br/><em>{it.notes}</em>"
            entries.append(body)
        atoms.append(items_atom(entries))
    return atoms
```

Call it from the `render_html` body. Find where atoms are assembled for each task branch and add a `checklist` branch that calls `_checklist_atoms(result.checklist)`. The exact line depends on the current structure — read `html.py:291-320` first, then extend the atom-assembly chain to include checklist atoms when `result.task == "checklist"` or when `result.checklist` is non-empty.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest src/andamentum/whetstone/tests/test_html_renderer.py -v`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/andamentum/whetstone/renderers/html.py src/andamentum/whetstone/tests/test_html_renderer.py
git commit -m "feat(whetstone): HTML renderer supports checklist section"
```

---

### Task 15: `render_docx` — checklist prepended to report header

**Files:**
- Modify: `src/andamentum/whetstone/renderers/docx.py`
- Modify: `src/andamentum/whetstone/docx/finalization.py`
- Modify: `src/andamentum/whetstone/tests/test_docx_smoke.py` (if it covers this path; otherwise add a targeted test)

- [ ] **Step 1: Inspect `finalize_reviewed_document` signature**

Run: `grep -n "def finalize_reviewed_document" src/andamentum/whetstone/docx/finalization.py`

The current signature accepts `review_summary`, `critical_issues`, `expert_reviews`, etc. We'll add a new optional `checklist_items` parameter.

- [ ] **Step 2: Write the failing test**

Append to `src/andamentum/whetstone/tests/test_docx_smoke.py`:

```python
def test_render_docx_accepts_checklist(tmp_path):
    """Smoke test: render_docx accepts a checklist parameter without error."""
    from andamentum.whetstone import ChecklistItem
    from andamentum.whetstone.renderers import render_docx

    src = tmp_path / "in.docx"
    dst = tmp_path / "out.docx"
    # Create a minimal .docx for input
    from docx import Document
    doc = Document()
    doc.add_paragraph("Hello.")
    doc.save(str(src))

    items = [ChecklistItem(name="x", status="pass", notes="y", category="abstract")]
    # Should accept checklist kwarg and not raise
    render_docx(
        input_path=src,
        output_path=dst,
        patches=[],
        checklist_items=items,
    )
    assert dst.exists()
```

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run pytest src/andamentum/whetstone/tests/test_docx_smoke.py::test_render_docx_accepts_checklist -v`
Expected: `TypeError: unexpected keyword 'checklist_items'`.

- [ ] **Step 4: Add `checklist_items` to the public renderer**

Edit `src/andamentum/whetstone/renderers/docx.py`. Add to `render_docx` signature, after `generated_experts`:

```python
    checklist_items: Optional[list] = None,
```

Update the docstring's `Args:` to include it:

```
        checklist_items: Optional list of ChecklistItem objects to prepend
            to the review report (checklist task).
```

Pass it through to `finalize_reviewed_document`:

```python
    _, patch_result = finalize_reviewed_document(
        original_file_path=input_path,
        patches=patches,
        review_summary=review_summary,
        issues_count=len(critical_issues) if critical_issues else 0,
        output_path=output_path,
        author=author,
        critical_issues=critical_issues,
        expert_reviews=expert_reviews,
        generated_experts=generated_experts,
        novelty_findings=novelty_findings,
        checklist_items=checklist_items,
    )
```

- [ ] **Step 5: Extend `finalize_reviewed_document`**

Edit `src/andamentum/whetstone/docx/finalization.py`. Add `checklist_items` to the function signature (matching the call above).

Inside the function, generate a markdown block for the checklist and prepend it to the existing `review_summary` / report header. Find the spot where the markdown report is assembled, and insert:

```python
    if checklist_items:
        from collections import defaultdict

        passes = sum(1 for i in checklist_items if i.status == "pass")
        fails = sum(1 for i in checklist_items if i.status == "fail")
        unclears = sum(1 for i in checklist_items if i.status == "unclear")

        by_cat = defaultdict(list)
        for it in checklist_items:
            by_cat[it.category or "other"].append(it)

        cl_lines = [
            "## Pre-submission Checklist",
            "",
            f"*{passes} pass · {fails} fail · {unclears} unclear (of {len(checklist_items)} checks)*",
            "",
        ]
        status_marker = {"pass": "✓ PASS", "fail": "✗ FAIL", "unclear": "? UNCLEAR"}
        for cat in sorted(by_cat):
            cl_lines.append(f"### {cat.title()}")
            cl_lines.append("")
            for it in by_cat[cat]:
                cl_lines.append(f"- **{status_marker.get(it.status, '?')}** {it.name}")
                if it.notes:
                    cl_lines.append(f"    - {it.notes}")
            cl_lines.append("")
        checklist_markdown = "\n".join(cl_lines) + "\n\n---\n\n"
        # Prepend to whatever markdown report is being built
        review_summary = checklist_markdown + (review_summary or "")
```

(Place this near the start of the function, before the existing review-summary assembly.)

- [ ] **Step 6: Wire into the CLI**

Edit `src/andamentum/whetstone/cli.py` `_render_output()`. Find the `.docx` branch (the `render_docx(...)` call) and add `checklist_items`:

```python
        checklist_items = getattr(result, "checklist", None) or None
        patch_result = render_docx(
            input_path=input_path,
            output_path=output_path,
            patches=patches,
            review_summary=review_summary,
            critical_issues=critical_issues,
            expert_reviews=list(expert_reviews) if expert_reviews else None,
            generated_experts=list(expert_profiles) if expert_profiles else None,
            checklist_items=checklist_items,
        )
```

- [ ] **Step 7: Run tests to verify they pass**

Run: `uv run pytest src/andamentum/whetstone/tests/test_docx_smoke.py -v`
Expected: all pass.

- [ ] **Step 8: Commit**

```bash
git add src/andamentum/whetstone/renderers/docx.py src/andamentum/whetstone/docx/finalization.py src/andamentum/whetstone/cli.py src/andamentum/whetstone/tests/test_docx_smoke.py
git commit -m "feat(whetstone): DOCX renderer prepends checklist section"
```

---

### Task 16: Update README

**Files:**
- Modify: `src/andamentum/whetstone/README.md`

- [ ] **Step 1: Read the current README structure**

Run: `sed -n '1,80p' src/andamentum/whetstone/README.md`

Find the section describing the three existing tasks (`edit`, `review`, `panel`). That's where to add documentation for the two new tasks.

- [ ] **Step 2: Add the new task docs**

Extend the tasks section to include:

```markdown
### `consistency` — internal consistency check

Flags internal consistency problems in your draft. Combines deterministic
scanners (figure ordering, acronym first-use, citation resolution) with
an LLM pass that looks for reading-comprehension issues (numbers that
disagree across sections, terminology drift, claim emphasis shifts).

Output: `DocumentIssue`s on `ReviewResult.issues`. Renders through the
existing `review` rendering path.

```python
result = await sharpen_document(text, task="consistency", model="anthropic:claude-haiku-4-5")
```

CLI:

```bash
andamentum-whetstone draft.docx --task consistency -o issues.html
```

### `checklist` — pre-submission checklist

Evaluates your draft against a baseline pre-submission checklist and,
optionally, a journal's author guidelines. Each check returns
`pass` / `fail` / `unclear` with evidence.

The baseline (16 items) covers abstract hygiene, figures/tables,
references, required statements (COI, data availability, ethics,
funding), and manuscript hygiene. Journal-specific items are extracted
on-the-fly from free-form guideline text you provide.

Output: `ChecklistItem`s on `ReviewResult.checklist`.

```python
result = await sharpen_document(
    text,
    task="checklist",
    guidelines=open("journal_guidelines.txt").read(),  # optional
    model="anthropic:claude-haiku-4-5",
)
```

CLI:

```bash
# Baseline only
andamentum-whetstone draft.docx --task checklist -o report.md

# With journal guidelines
andamentum-whetstone draft.docx --task checklist \\
    --guidelines @guidelines.txt -o report.html
```
```

- [ ] **Step 3: Commit**

```bash
git add src/andamentum/whetstone/README.md
git commit -m "docs(whetstone): document consistency and checklist tasks"
```

---

### Task 17: Final verification — full suite green

**Files:** none modified; verification only.

- [ ] **Step 1: Run the full test suite**

Run: `uv run pytest`
Expected: all tests pass. Baseline was 814; expect ~860–880 after this work. Report the exact count.

- [ ] **Step 2: Run type checker**

Run: `uv run pyright`
Expected: 0 errors.

- [ ] **Step 3: Run linter**

Run: `uv run ruff check`
Expected: clean output (no violations).

- [ ] **Step 4: Run formatter check**

Run: `uv run ruff format --check`
Expected: clean.

- [ ] **Step 5: Sanity-check the CLI help**

Run: `uv run andamentum-whetstone --help`
Expected: `--task` choices include `consistency` and `checklist`; `--guidelines` present.

Run: `uv run andamentum-whetstone agents`
Expected: output includes `consistency_reviewer`, `checklist_item_evaluator`, `journal_guidelines_extractor`.

- [ ] **Step 6: Manual smoke — no commit**

Create a tiny test document and run each new task against it to confirm end-to-end wiring (requires a real model or Ollama). This is a manual verification step, not an automated test. Report what you saw.

- [ ] **Step 7: Final commit if verification surfaces fixes**

If any of steps 1–5 surface issues, fix them, then commit with a clear message. If everything is green, no commit is needed for this task.

---

## Self-review

**Spec coverage:**

- Scope — two new tasks, CLI + Python entry point: Task 10 (Python), Task 12 (CLI).
- `guidelines` kwarg — Task 10 (signature + validation), Task 12 (CLI flag).
- `ChecklistItem` + `BaselineCheck` models — Task 1; exported Task 11.
- `ConsistencyReviewOutput` + `ExtractedChecklistNames` — Task 3.
- `consistency_scanners.py` — Task 4.
- `checklist_scanners.py` — Task 5.
- `consistency_reviewer` agent — Task 6.
- `checklist_item_evaluator`, `journal_guidelines_extractor`, `BASELINE_CHECKS` — Task 7.
- `_run_consistency` — Task 8.
- `_run_checklist` (incl. baseline fan-out, journal layer, failure handling) — Task 9.
- `sharpen_document` dispatch + validation — Task 10.
- `ReviewResult.checklist` — Task 2.
- Renderer impact: `render_diff` — Task 13; `render_html` — Task 14; `render_docx` — Task 15.
- README — Task 16.
- Green-state verification — Task 17.

All spec sections mapped to tasks. No gaps found.

**Placeholder scan:** searched for "TBD", "TODO", "implement later", "add appropriate". None found outside of quoted prose.

**Type consistency:**
- `ChecklistItem` fields (`name`, `status`, `notes`, `category`, `source`) are consistent across Task 1 (definition), Task 7 (LLM output model), Task 9 (orchestrator construction), Task 13/14/15 (renderers).
- `BaselineCheck.scanner` is the function name; used as `getattr(checklist_scanners, check.scanner)` in Task 9 — and verified to exist by Task 7 test `test_baseline_scanners_exist`.
- `_run_consistency` / `_run_checklist` signatures consistent across definition (Task 8/9) and dispatch (Task 10).
- `render_diff(checklist=...)` kwarg name consistent between Task 13 (renderer) and CLI wiring in Task 13 step 6.
- `render_docx(checklist_items=...)` kwarg name consistent between Task 15 step 4 (renderer) and Task 15 step 6 (CLI wiring) and `finalize_reviewed_document(checklist_items=...)` (Task 15 step 5).

No inconsistencies found.

**Scope check:** focused on two tasks as spec'd. Renderer work is present but tightly scoped to each format's minimal integration point. No cross-module work, no epistemic/deep_research changes, no typeset module changes.
