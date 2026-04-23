# Whetstone: internal consistency + pre-submission checklist

**Date:** 2026-04-23
**Module:** `andamentum.whetstone`
**Status:** Design approved, ready for implementation plan.

## Motivation

Whetstone today sharpens drafts through three tasks: `edit` (grammar/style
patches), `review` (four specialist reviewers + synthesis), and `panel` (N
fictional experts). It covers the "catch the obvious mistakes" and "get
critical feedback" stages of a draft's life.

Two gaps remain in the "between finishing the draft and hitting submit"
window:

1. **Internal consistency.** Numbers that disagree between abstract and
   results. Figures referenced out of order. Acronyms used before
   definition. Terminology that drifts across sections. A human author
   won't spot most of these on a re-read of their own work.
2. **Pre-submission checklist.** Journals publish author guidelines with
   twenty-odd specific requirements (word counts, required statements,
   figure formats, data-availability language). Missing one is a
   desk-rejection risk.

Both fit whetstone's shape: take a draft, run agents, produce a structured
report. Neither requires companion documents (rebuttal, cover letter),
which would belong in a different module.

## Scope

Two new tasks added to whetstone:

- `--task consistency` — flags internal consistency problems as
  `DocumentIssue`s.
- `--task checklist` — evaluates a draft against a baseline
  pre-submission checklist and, optionally, a journal's author guidelines.

## Non-goals

- No companion-document drafting (rebuttals, cover letters, lay
  summaries). Different tool shape.
- No auto-patching of consistency problems. Fixes often require judgment
  the tool can't supply.
- No journal-guidelines database. Users paste or point to a file; we do
  not fetch from the web or ship a library of guidelines.
- No UI changes beyond the existing HTML/DOCX renderer output paths.

## Public surface

### Python entry point

`sharpen_document` gains support for two new `task` values and one new
kwarg:

```python
async def sharpen_document(
    content: str,
    *,
    task: str = "review",
    # ... existing kwargs ...
    guidelines: Optional[str] = None,  # NEW — only valid with task="checklist"
) -> ReviewResult:
```

- `task` accepts `"edit"`, `"review"`, `"panel"`, `"consistency"`,
  `"checklist"`.
- `guidelines`: text of the journal's author guidelines. Ignored unless
  `task == "checklist"`.
- Any `task != "checklist"` with non-`None` `guidelines` raises
  `ValueError`.

### CLI

`andamentum-whetstone`:

- `--task` choices extended to the five task names above.
- New `--guidelines TEXT_OR_@PATH` flag (same `@path` convention as
  `--criteria`).
- Guard: `--guidelines` with `--task != checklist` → parser error.

### `ReviewResult` changes

One new optional field:

```python
checklist: list[ChecklistItem] = Field(default_factory=list)
```

`consistency` uses the existing `issues` field. No other `ReviewResult`
changes.

### New public types

Exported from `andamentum.whetstone`:

- `ChecklistItem` — one item's outcome.

Internal types (not in `__all__`):

- `BaselineCheck` — an entry in `BASELINE_CHECKS`.
- `ConsistencyReviewOutput` — wraps `issues: list[DocumentIssue]`.
- `ExtractedChecklistNames` — wraps `items: list[str]`.

## Data models

```python
# In whetstone/models.py

class ChecklistItem(BaseModel):
    """One pre-submission check, evaluated against a draft."""
    name: str
    status: Literal["pass", "fail", "unclear"]
    notes: str
    category: str = ""       # tagged by orchestrator, not by the LLM
    source: Literal["baseline", "journal"] = "baseline"  # tagged by orchestrator

class BaselineCheck(BaseModel):
    """An entry in the pre-submission baseline list."""
    name: str
    category: str
    kind: Literal["deterministic", "llm"]
    scanner: Optional[str] = None     # function name, when kind="deterministic"
    prompt_hint: Optional[str] = None  # extra guidance, when kind="llm"
```

Design notes:

- `ChecklistItem` has three LLM-visible fields (`name`, `status`, `notes`)
  plus two Python-tagged fields (`category`, `source`). The LLM is not
  asked to emit `category`/`source`; the orchestrator sets them from the
  `BaselineCheck` that originated the call or from the journal-specific
  path. This keeps the agent's output schema flat for small local models.
- `status` is three values only. "Not applicable" collapses into
  `unclear` with a descriptive `notes` string. Small local models confuse
  N/A and unclear anyway; collapsing avoids the distinction.
- `scanner` and `prompt_hint` are mutually exclusive based on `kind`, but
  expressing this with a discriminated union costs more than a runtime
  assertion in the orchestrator. Keep as plain optional fields.

## Module layout

```
src/andamentum/whetstone/
├── agents/
│   ├── consistency.py          NEW — consistency_reviewer agent
│   ├── checklist.py            NEW — checklist_item_evaluator,
│   │                                  journal_guidelines_extractor,
│   │                                  BASELINE_CHECKS list
│   ├── output_models.py        MODIFIED — adds ConsistencyReviewOutput,
│   │                                       ExtractedChecklistNames
│   └── __init__.py             MODIFIED — imports new modules
├── consistency_scanners.py     NEW — deterministic scanners for consistency
├── checklist_scanners.py       NEW — deterministic scanners for baseline items
├── models.py                   MODIFIED — adds ChecklistItem, BaselineCheck
├── orchestrator.py             MODIFIED — _run_consistency, _run_checklist
├── cli.py                      MODIFIED — adds --guidelines, task choices
└── __init__.py                 MODIFIED — exports ChecklistItem
```

Placement rationale:

- `ChecklistItem` goes in `models.py` because that file already holds
  user-facing structured output types (`DocumentPatch`,
  `PatchApplicationResult`); `issues.py` is for the `DocumentIssue`
  ecosystem specifically. `ChecklistItem` is a peer of `DocumentPatch`,
  not of `DocumentIssue`.
- Scanners are module-level (`whetstone/consistency_scanners.py`,
  `whetstone/checklist_scanners.py`), not nested under `agents/`, because
  they are **not agents**. They are pure Python functions with zero LLM
  involvement — Constitution Rule 4 requires deterministic work to live
  in deterministic code paths.
- Agent modules follow the existing pattern: one module per task family
  (`editing.py`, `review.py`, `synthesis.py`, `multi_expert.py`,
  `custom.py`). Adding `consistency.py` and `checklist.py` extends the
  pattern without restructuring it.
- All agents register themselves into
  `andamentum.whetstone.agents.AGENT_REGISTRY` at import time via
  `register_agent(...)`, identical to existing modules.

## Agents

### `consistency_reviewer`

- **Input kwargs**: `document: str`.
- **Output model**: `ConsistencyReviewOutput` — one field,
  `issues: list[DocumentIssue]`.
- **Prompt focus**: only the things that need reading comprehension —
  numbers disagreeing between sections, terminology drift, claim
  emphasis shifting, tense/voice shifts. Prompt explicitly tells the
  agent NOT to comment on figure numbering, reference-list
  completeness, or acronym first-use — those are handled by scanners.
- **Retries**: 2 (matches existing whetstone agents).

### `checklist_item_evaluator`

- **Input kwargs**: `document: str`, `check_name: str`,
  `prompt_hint: Optional[str]`.
- **Output model**: `ChecklistItem` directly (no wrapper — flattest
  schema for small local models). The three LLM-emitted fields are
  `name`, `status`, `notes`. `category` and `source` default to their
  empty/`"baseline"` values and are overwritten by the orchestrator.
- **Post-processing**: orchestrator overwrites the returned `name` with
  the authoritative `check_name` (belt-and-braces against small-model
  drift). Orchestrator sets `category` and `source`.
- **Retries**: 2.

### `journal_guidelines_extractor`

- **Input kwargs**: `guidelines: str`.
- **Output model**: `ExtractedChecklistNames` — one field,
  `items: list[str]`.
- **Prompt focus**: turn free-form guideline text into 10–30 short,
  checkable questions. Skip editorial prose; keep only actionable
  rules.
- **Retries**: 2.

## Scanners

### `consistency_scanners.py`

Pure functions, no async, no external dependencies beyond `re`. Each
returns `list[DocumentIssue]` with `agent_type = "scanner:<name>"` for
renderer-side grouping.

- `check_figure_order(text)` — scan for "Figure N", "Fig. N", and verify
  the first in-text reference to each figure number is in ascending
  order. Emits one issue per out-of-order reference.
- `check_acronym_first_use(text)` — scan for capitalised acronyms of
  length 2–6. Verify each is expanded on first use (expansion pattern:
  `Full Phrase (ACRONYM)`). Emits one issue per acronym used before
  expansion.
- `check_citation_resolution(text)` — identify in-text citations
  (`[N]`, `[N, M]`, `(Author, Year)`) and, if a References section is
  present, verify each citation resolves to an entry. Emits one issue
  per unresolved citation. If no References section is found, returns
  `[]` — a missing references section is out of scope for this scanner.
- `run_all(text)` — dispatches to the above and returns the flattened
  list.

### `checklist_scanners.py`

One function per deterministic baseline item. Signature:

```python
def <name>(text: str) -> tuple[Literal["pass", "fail", "unclear"], str]:
```

Returns the `status` and `notes` fields. The orchestrator wraps the
tuple into a `ChecklistItem` with `name`, `category`, `source` filled
from the originating `BaselineCheck`.

## BASELINE_CHECKS contents (initial)

Lives in `whetstone/agents/checklist.py` as a module-level list. This
is the authoritative source; additions go through normal PR review.

Initial list, 15–20 items grouped by category. Each entry declares
whether it is deterministic (with a named scanner) or LLM-driven (with
an optional prompt hint).

**Abstract** (LLM): structured sections present; word-count stated;
no undefined abbreviations.

**Figures & tables** (deterministic): all figures cited in text; all
tables cited in text; figure numbering sequential; table numbering
sequential.

**References** (mixed): all in-text citations resolve (deterministic);
reference list formatted consistently (LLM).

**Required statements** (deterministic for presence, LLM for adequacy):
conflict-of-interest statement present; data-availability statement
present; ethics statement present if human/animal work is mentioned;
funding statement present.

**Manuscript hygiene** (mixed): keywords section present
(deterministic); title meaningful and specific (LLM); authors listed
(deterministic).

The initial list is not exhaustive — it is a starting point. Growing it
is expected and requires only editing the Python list.

## Orchestration

### `_run_consistency(runner, result, content, verbose)`

1. `scanner_issues = consistency_scanners.run_all(content)` — synchronous.
2. `llm_output = await _run_one(runner, "consistency_reviewer", document=content)`.
3. `result.issues = scanner_issues + llm_output.issues`.
4. No synthesis step. Renderers reuse the existing `review` rendering
   for issues.

### `_run_checklist(runner, result, content, guidelines, verbose)`

1. **Baseline evaluation** (parallel where possible):
   - Deterministic items: call the named scanner function, wrap the
     `(status, notes)` tuple in a `ChecklistItem` with
     `name`/`category`/`source="baseline"` set.
   - LLM items: `await` one `checklist_item_evaluator` call per item,
     dispatched via `_run_agents` for parallelism. Post-process: set
     `name = check.name`, `category = check.category`,
     `source = "baseline"`.
2. **Journal evaluation** (only if `guidelines is not None`):
   - `extracted = await _run_one(runner, "journal_guidelines_extractor", guidelines=guidelines)`.
   - Fan out `checklist_item_evaluator` calls for each `extracted.items`
     entry in parallel. Set `category = "journal"`, `source = "journal"`.
3. `result.checklist = baseline_items + journal_items`.
4. `result.issues` stays empty.

## Error handling

Follows Constitution Rule 2 (fail fast, fail loud) and Rule 5 (no
silent fallbacks), plus the whetstone precedent that `_run_agents`
re-raises with phase context.

**Invalid inputs**:
- Bad `task` → `ValueError` (extend existing check).
- `guidelines` with non-checklist task → `ValueError`.
- `guidelines=""` (empty after strip) → treated as `None`. Not an error.
- `@path` for `--guidelines` where file is missing → `FileNotFoundError`.

**Agent failures**:
- Extractor failure, consistency-reviewer failure, or any
  baseline-item evaluator failure → `RuntimeError` via `_run_agents`.
  No partial results.
- Per-item evaluator failure on a **journal-extracted** item → log a
  warning, emit a `ChecklistItem(status="unclear", notes="Evaluation failed: <exc>")`.
  This is the single exception Rule 5 allows (service reachable but
  bad data for one item after retries), justified by the extractor's
  output being fuzzy by nature.

**Scanner edge cases**:
- `consistency_scanners`: document with no figures / no acronyms / no
  references section → scanner returns `[]` silently. The consistency
  task is a problem-finder; silence means nothing broken.
- `checklist_scanners`: for a check like "all figures referenced" on a
  document with no figures → return `("unclear", "No figures found in document")`.
  The checklist is an affirmative report; silence there would lose
  information.

**Output validation**:
- `ChecklistItem.name` from the LLM is overwritten with the
  authoritative `check_name`.
- `ChecklistItem.status` is enforced by the Pydantic `Literal`.
  Out-of-range values trigger the agent's configured retry.

**Observability**:
- Each phase logs start/end at INFO (matches existing orchestrator
  style for non-new code).
- New code uses the module `logger`, not `print`.
- Scanner issues logged at DEBUG.
- Extractor output logged at INFO with the item count.

## Renderer impact

No renderer changes required for `consistency` — the existing review
rendering path handles `ReviewResult.issues` already.

For `checklist`, renderers need to render `ReviewResult.checklist`:

- **HTML**: add a new typeset atom composition (group by category,
  pass-count summary at the top, status icon per item). Implementation
  detail for the implementation plan; the typeset module already
  supports the building blocks (`heading`, `items`, `callout`).
- **DOCX**: prepend a checklist section to the review header. Plain
  markdown list with status markers; the existing `DocumentIssueCollection.format_as_markdown`
  style is the template to follow.
- **Diff**: not meaningful for checklist output. When `task="checklist"`
  and the user asks for `.md` output, emit a markdown checklist, not a
  diff.

## Testing

Tests live next to code in `src/andamentum/whetstone/tests/`. Follows
existing whetstone conventions (no separate top-level `tests/` dir).

**New test files**:
- `test_consistency_scanners.py` — fully deterministic, no LLM. ≥3
  cases per scanner: finds the planted problem, no false positive on a
  clean doc, handles the edge case (no figures, no acronyms, etc.).
- `test_checklist_scanners.py` — same pattern for each deterministic
  baseline item.
- `test_checklist_models.py` — `ChecklistItem` and `BaselineCheck`
  construction, invalid status rejected.
- `test_consistency_orchestrator.py` — stub `AgentRunner.run` to
  return a canned `ConsistencyReviewOutput`; verify scanner + LLM
  issues are merged into `ReviewResult.issues`; verify `agent_type`
  tagging.
- `test_checklist_orchestrator.py` — verify baseline fan-out,
  per-item tagging (source/category), `guidelines=None` path (baseline
  only), `guidelines=<str>` path (baseline + journal), per-item
  journal-evaluator failure becomes `status="unclear"`, per-item
  baseline-evaluator failure raises.
- `test_checklist_cli.py` — `--guidelines` flag wiring, `@path`
  resolution, rejection when combined with a non-`checklist` task.

**Fixture data**: tiny paragraph-sized fixtures in
`src/andamentum/whetstone/tests/fixtures/`. No large real manuscripts.

**Not tested**:
- Per-item LLM quality (prompt-engineering drift, not logic we own).
- Contents of `BASELINE_CHECKS` beyond shape validation.
- End-to-end against a real model (`ollama`-marked tests only if added,
  deselected by default — matches existing pattern).

**Green state before complete**: `uv run pytest` clean, `uv run pyright`
zero errors, `uv run ruff check` clean. Existing baseline: 814 tests
passing. Expected delta: +40–60 tests, no regressions.

## Constitution alignment summary

- **Rule 1 (Simplicity first)**: minimum viable. No JSON/YAML for
  baseline checks, no journal-guidelines database, no auto-patching,
  no companion documents.
- **Rule 2 (Think before coding)**: captured in this spec. Two tasks,
  not one. Scanners, not LLM, for countable checks.
- **Rule 3 (Surgical changes)**: does not touch existing tasks'
  orchestrator paths, does not modify existing agents, does not change
  `ReviewResult` for tasks other than `checklist`.
- **Rule 4 (Deterministic vs intelligent)**: scanners for countable /
  regex-detectable work; LLM only where reading comprehension is
  required.
- **Rule 5 (Fail fast)**: `ValueError` for bad inputs, `RuntimeError`
  for phase failures, one documented soft-failure path for journal
  per-item evaluation.
- **Principle 3 (Single source of truth)**: `BASELINE_CHECKS` is the
  one authoritative list; `ChecklistItem` defined once in `models.py`.
- **Principle 5 (Dependencies flow one way)**: `models.py` imports no
  agents; `agents/` imports from `models.py` and `core.agents`;
  `orchestrator.py` imports both. No cycles.
