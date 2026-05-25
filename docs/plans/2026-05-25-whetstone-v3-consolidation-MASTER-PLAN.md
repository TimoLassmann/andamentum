# Whetstone v3 consolidation — MASTER PLAN (autonomous execution)

**Status:** Active. Auto-executing on branch `whetstone-v3-consolidation`.
**Companion:** `2026-05-25-whetstone-v3-consolidation-prd.md` (the design decisions).
**Tasks:** #2…#11 in this session's task list.

This document is the runtime playbook. The PRD is the design. The plan
revises the PRD's 10-step sequence into **9 actionable phases (A–I) + a
benchmark phase (J)** based on findings from 10 structural research
agents (see §1 below).

## 0. Why the resequencing

The PRD §5 listed 10 sequential steps. Risk-synthesis agent (general-
purpose) flagged five real problems:

1. **Step 1 alone is a partial fix.** The auto-routing patch claims to
   stop "forced findings on non-academic input", but `criterion_set_for`
   already falls back to GENERAL for unknown types — the actual fix only
   lands when step 2's expanded criterion sets ship. **→ Merged into
   Phase A.**
2. **Steps 3, 4, 5, 6 each edit `review_document_v3`'s signature.** The
   PRD treated them as independent; in practice each adds a kwarg to the
   same call site. **→ Ordered so the API surface grows monotonically:
   B (confirm_own_draft) → C (criteria/guidelines) → D (editor) → E
   (novelty).**
3. **CLI subcommand refactor is an implicit prerequisite** of steps 7
   (panel) and 8 (proofread). The current CLI is flag-based, not
   subcommand-based. **→ New Phase F before Phases G and H.**
4. **Steps 9 + 10 cannot leave a working module in between** — module
   rename and v2 deletion must be atomic. **→ Merged into Phase I.**
5. **Step 7 (panel) carries the most risk** — different graph topology,
   needs 4 agent ports, panel-specific renderer fields. **→ Phase G
   starts with a 10-minute mini-PRD; if the design surfaces are wider
   than expected, halt and ask.**

Two false trivials called out:
- **Step 1 (auto-routing)** isn't 3 lines. `classify()` needs the same
  `model` arg as the run. **Decision:** when `model=None`, `auto` falls
  back to `general` to preserve the `--no-llm` path.
- **Step 8 (proofread separation)** isn't just "remove wiring" — v3 has
  no wiring to remove. The subcommand is the real work.

Hidden v2-located **shared** code that must survive deletion: `renderers/`,
`schemas.py`, `models.py`, `_confidentiality.py`, `_watermark.py`,
`_document_type.py`, `docx/`, `anchoring.py`. **B1 risk:** the PRD's
"delete v2 renderers/" phrasing was wrong — they're shared.

## 1. Structural research summary

| # | Agent | Key finding (one line) |
|---|---|---|
| 1 | Explore v2 callers | Only 2 external callers: `cli.py:680`, `__init__.py:28-29` re-exports. ~29 v2 tests, ~47 node+agent .py files. |
| 2 | Explore v3 callers | Only 2 external callers: `cli.py:668-677` (under `--v3`) and `benchmarks/whetstone/arms.py:110`. Public API is `review_document_v3` / `run_review_v3`. |
| 3 | Test surface | ~152 v2-specific tests (delete); 84 v3 tests (keep); 169 shared tests (verify after deletion); 56 strunk-lens tests (decide). |
| 4 | CLI surface | 24 argparse flags, ~12 validation rules, 4 codepaths (v2-review / v2-modes / v3 / patch-only). **No subparsers today.** |
| 5 | DocumentType classifier | 8 hard-coded 3-type assumptions: `cli.py:282 choices`, `_document_type.py:25-30`, `state.py:130-132`, `api.py:50-52`, `v3/criteria.py:172-176`, `nodes/chunk_and_scan.py:109`, `nodes/synthesise.py:214-226 _DOC_TYPE_VOCAB`, `tests/test_document_type.py:217`. All must be touched in Phase A. |
| 6 | Edit/Finding → DocumentPatch | `Edit` (schemas.py:86-106), `DocumentPatch` (models.py:26-170), adapter at `renderers/docx.py:172-205`. v3 has no docx renderer; reuses v2's. **Editor node port is purely additive — no adapter changes.** |
| 7 | Panel mode internals | 6 nodes, 4 agents, 3 ReviewState fields, 3 ReviewResult fields. Renderer-specific code in `renderers/_panel_layout.py` (must survive deletion). |
| 8 | Novelty internals | Currently 1 node (`nodes/novelty_check.py`) running 1 LLM extraction + N parallel deep_research calls + N adapter calls. **Public deep_research API: `run_research(query, *, model, max_iterations, verbose) → ResearchResult` with `sources: list[str]`.** |
| 9 | Editor phase internals | 1 node, 1 agent, per-section call (max 5 concurrent). Schema (`EditorOutput`, `EditProposal`, `Edit`) is stable. **Port is mechanical.** |
| 10 | Risk synthesis | (Drove the resequencing above.) Biggest single risk: Phase G (panel) needs runtime criterion-set construction not in v3's spec. |

## 2. Per-phase execution protocol

Each phase follows the same loop. The autonomous-execution machinery is
just: do these steps in order, halt on the listed conditions.

```
For phase X:
  1. Spawn ≥5 targeted research agents (per-phase questions in §3 below)
  2. Synthesize → action items (Edit list, new files, test additions)
  3. Implement (Edit/Write tools in worktree)
  4. Validate:
       a. uv run pyright            — no NEW errors beyond baseline (23)
       b. uv run pytest <changed>   — all green
       c. uv run ruff check         — clean
       d. uv run ruff format        — clean
       e. Diff vs plan — every planned file edited; no surprise files
  5. If any validation fails twice → HALT (write STATUS.md, mark task in_progress, surface to user)
  6. git commit (with Co-Authored-By trailer + detailed message)
  7. TaskUpdate to completed
  8. Move to next phase
```

### Universal halt conditions

Stop and surface to user when:
- Validation fails after one fix attempt (don't sink time into a hole)
- An action item not on the phase plan needs to be added (scope creep —
  user should know)
- A design decision pops up that wasn't pre-decided (e.g. Phase G
  surfacing the "runtime criterion-set construction" issue)
- Test count drops by more than the v2-test count attributed to the
  phase (Phase I only — others should be additive)
- Any new pyright error appears (CLAUDE.md baseline is 23; that's the
  ceiling)
- `git status` shows files modified outside the worktree

### What "complete" means per phase

For each phase, *definition of done* is:
- All listed action items shipped
- New tests written and green
- Existing tests still green
- pyright/ruff/format clean
- Commit landed on branch with descriptive message
- Task #N marked completed

## 3. Per-phase research protocols (5+ agents each)

Spawn each batch when starting the phase. Each agent prompt should be
self-contained, ≤500 words output, with file:line citations.

### Phase A — Auto-routing + 6 criterion types

1. **Explore**: enumerate every reference to `criterion_set_for`, `SPECS`, `EXTERNAL_COMMS`, `GENERAL` across the codebase
2. **Explore**: enumerate every `if document_type == X` / `match document_type:` branch
3. **Explore**: read `nodes/synthesise.py` `_DOC_TYPE_VOCAB` and identify what semantic each vocab entry carries
4. **Explore**: read `tests/test_document_type.py` end-to-end; list every assertion that locks the 3-type set
5. **general-purpose**: for each of the 3 new types (essay/tutorial/creative), write 5-question criterion blocks following the EXTERNAL_COMMS shape — these become the actual content of the new criterion sets

### Phase B — Confidentiality tripwire

1. **Explore**: read `_confidentiality.py` end-to-end; document API + exception type
2. **Explore**: read v2 `cli.py:705-720` try/except dispatcher; replicate the error code structure
3. **Explore**: find every v2 caller of `_confidentiality` to understand current wiring
4. **Explore**: find any v3 test that already plants confidentiality markers (probably none)
5. **general-purpose**: write the test plan: 3 tests covering tripwire fires, confirm_own_draft bypasses, own-draft input passes through

### Phase C — Collapsed criteria input

1. **Explore**: read v2 `agents/extract_checkable_items.py` (or equivalent) — the prose-to-criteria extractor we want to port
2. **Explore**: read v2 `agents/custom_reviewer.py` — understand current custom-criteria evaluator
3. **Explore**: examine `v3/criteria.py:Criterion` model — confirm it can absorb both list-of-strings and prose-extractor output
4. **Explore**: find every test that mocks `extract_checkable_items` (porting opportunities)
5. **general-purpose**: design the unified API surface (criteria= vs guidelines_text= mutual exclusion; extractor signature)

### Phase D — Editor node

1. **Explore**: read v2 `nodes/edit_sections.py` end-to-end + `agents/editor.py`
2. **Explore**: find every reference to `EDITOR_AGENT`, `EditorOutput`, `EditProposal`
3. **Explore**: identify how Edit objects get anchored to source spans (v2 uses `find_anchor`)
4. **Explore**: check v3's existing `locate.py` shim — confirm it provides the anchoring v3 needs
5. **general-purpose**: design v3/editor.py node + tests; confirm Edit objects flow correctly into `ReviewResult.edits`

### Phase E — Novelty 3-node pipeline

1. **Explore**: read `deep_research/orchestrator.py:run_research` signature + ResearchResult shape
2. **Explore**: read v2 `nodes/novelty_check.py` + `agents/novelty_claim_extractor.py` to understand the extraction logic we're keeping
3. **Explore**: examine the v2 cache mechanism (`~/.cache/whetstone/novelty/`) — decide whether to port (probably not, per `feedback_avoid_hidden_home_dirs`)
4. **Explore**: find every reference to `NoveltyReport`, `NoveltyClaim`, `SimilarWork` types
5. **general-purpose**: design the three node contracts (FlagNoveltyTargets I/O, RunNoveltySearches I/O, JudgeNovelty I/O) and where they insert in the graph

### Phase F — CLI subcommand refactor

1. **Explore**: read `cli.py` end-to-end (we have a partial map from research agent #4)
2. **Explore**: find every argparse-related test
3. **WebFetch** or read: standard argparse subcommand patterns (back-compat strategies for default subcommand)
4. **Explore**: find any external doc / README / shell script referencing the current flat CLI invocation
5. **general-purpose**: design the new subcommand structure with explicit back-compat for bare `andamentum-whetstone <input>`

### Phase G — Panel mode (mini-PRD first)

**Pre-research:** before the 5 agents, write a 1-page mini-PRD covering:
- Graph topology: shared substrate vs forked
- Where the 4 v2 agents live in v3 (v3/agents/ or v3/panel/agents/)
- Runtime criterion-set generation (the issue surfaced in risk synthesis)
- panel_synthesis on ReviewResult

If the mini-PRD surfaces unresolved design issues, **HALT and surface to user.**

Otherwise:
1. **Explore**: read each of v2's 4 panel agents end-to-end
2. **Explore**: read `nodes/extract_keywords.py`, `nodes/generate_expert_panel.py`, `nodes/expert_review.py`, `nodes/panel_synthesise.py`
3. **Explore**: read `renderers/_panel_layout.py` — understand panel-specific rendering
4. **Explore**: identify which v3 primitives (digest, document_model, sectionize) the panel graph can reuse
5. **general-purpose**: write panel test plan covering all 3 ReviewResult.* output fields

### Phase H — Proofread subcommand

1. **Explore**: read `andamentum.proofread` public API (`analyze()`, `ProofreadResult`)
2. **Explore**: read `andamentum.proofread.cli` to understand the existing standalone CLI shape
3. **Explore**: read `harvest.extract` async API to confirm it works for the same input types
4. **Explore**: find any test pattern for argparse subcommands inside whetstone
5. **general-purpose**: write the smallest possible wrapper that doesn't duplicate proofread logic

### Phase I — Triage + delete v2 + rename (atomic)

1. **Explore**: enumerate EVERY .py file under `src/andamentum/whetstone/` (not in `v3/` and not in `docx/`); bucket each as v2-only / shared / unknown
2. **Explore**: enumerate every test file the same way; bucket
3. **Explore**: enumerate every reference to v2-named symbols (`review_document` without _v3, `ReviewState`, `ReviewDeps`, `HarvestSource`, etc.) in non-whetstone code (benchmarks, docs, README)
4. **Explore**: confirm the model.py / models.py collision plan (keep `models.py` for `DocumentPatch`; rename `v3/model.py` → `whetstone/document_model.py` post-move)
5. **Explore**: read CLAUDE.md's whetstone section; mark every line that needs editing
6. **general-purpose**: dry-run the move — list every git mv, every file deletion, every Edit. Estimate the commit diff size.

(Phase I gets 6 research agents because it's the highest-blast-radius
phase.)

### Phase J — Benchmark + PR summary

1. **Explore**: read `benchmarks/whetstone/` to confirm post-rename arms.py works
2. Run the benchmark suite (long-running; use Bash run_in_background)
3. Compare metrics against the baseline noted in `5828f4c`
4. Write end-of-branch PR summary

## 4. Progress tracker

| Phase | Task ID | Status | Commit | Notes |
|---|---|---|---|---|
| A | #2 | pending | — | merged from PRD steps 1+2 |
| B | #3 | pending | — | |
| C | #4 | pending | — | |
| D | #5 | pending | — | |
| E | #6 | pending | — | |
| F | #7 | pending | — | new (CLI subcommands) |
| G | #8 | pending | — | mini-PRD first; elevated risk |
| H | #9 | pending | — | |
| I | #10 | pending | — | merged from PRD steps 9+10; atomic |
| J | #11 | pending | — | benchmark only |

## 5. Working agreements for autonomous execution

- **Worktree only.** All edits inside `.worktrees/whetstone-v3-consolidation/`. Never touch files in the primary checkout.
- **Branch only.** Every commit lands on `whetstone-v3-consolidation`. Never push.
- **No backwards-compat shims during the port.** Per `feedback_no_backcompat_during_dev`. v2 callers either get updated or get deleted.
- **No new hidden-home-dir writes.** Per `feedback_avoid_hidden_home_dirs`. Novelty cache is OUT.
- **No env vars in new code.** Per `feedback_no_env_vars`. CLI → kwargs.
- **Test models for any LLM-call demo:** `openai:gpt-5.4-nano` for cloud, `ollama:gemma4:31b-nvfp4` / `ollama:gemma4:26b-nvfp4` / `ollama:gpt-oss:20b` for local.
- **Memory of what we've done lives in git log + this plan's progress tracker.** No status files except on halt.
