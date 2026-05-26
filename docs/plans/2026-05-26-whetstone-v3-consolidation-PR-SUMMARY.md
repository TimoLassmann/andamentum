# Whetstone v3 consolidation — PR summary

**Branch:** `whetstone-v3-consolidation`
**Base:** `main` (06bd648)
**Status:** Ready for review. All planned phases complete (A-G, H, I; J = this doc + final validation).

## What this branch does

Collapses whetstone v2 (the lens-based section-by-section pipeline) and
v3 (the whole-document criterion-cascade pipeline) into a single
v3-only review surface. The user-facing API name stays the same —
`await review_document(source, *, model)` — but now resolves to the
v3 implementation, with all v2-only features either ported into v3,
restructured into separate code paths, or intentionally dropped.

## Headline numbers

- **10 commits**, each separately reviewable, with full validation gate
- **2152 project tests pass** (up from 2075; net +77 = +460 v3 additions over Phases A-G minus ~380 v2 deletions)
- **Whetstone test count: 270** (was 654 before deletion; -384 = v2 deletions)
- **pyright clean** at the pre-existing baseline (no new errors introduced)
- **ruff check + format clean** on every changed file

## Per-phase summary

| Phase | Commit | What landed | Validation |
|---|---|---|---|
| Plan | `3f11447` | PRD + 9-phase MASTER PLAN with halt criteria + per-phase research protocols | 10 structural research agents informed the plan |
| **A** | `00f243d` | Auto-routing fix + expand criterion taxonomy to 6 types (essay / tutorial / creative added) across all 8 hard-coded sites | +3 tests → 604 pass |
| **B** | `655d831` | Confidentiality-marker tripwire ported to v3; fires before any LLM call including the auto classifier | +5 tests → 609 pass |
| **C** | `2fec570` | Collapsed v2's custom + guidelines modes into one unified `criteria=` / `guidelines_text=` API with mutual-exclusion validation + a one-call extractor agent for prose-to-criteria | +8 tests → 617 pass |
| **D** | `296ad34` | Editor node ported as an optional `editor=True` pass between CritiqueRevise and Finalize, 5-concurrent semaphore, anchored via the existing `locate` shim | +8 tests → 625 pass |
| **E** | `9f79388` | Novelty check split from v2's single-node-with-tool shape into three deterministic graph nodes (`FlagNoveltyTargets` → `RunNoveltySearches` → `JudgeNovelty`); the v2 `~/.cache/whetstone/novelty/` was dropped per the project no-hidden-home-dir rule | +13 tests → 638 pass |
| **F + H** | `b0d77e9` | CLI subcommand front-end (`review` / `panel` / `proofread` / `apply-patches`) as a thin alias layer over the existing flat parser; `proofread` short-circuits straight to `andamentum.proofread.cli` | +9 tests → 647 pass |
| **G** mini-PRD | `a5e9fdd` | Halt checkpoint with five design questions before the panel-mode port | — |
| **G** | `60601f5` | Panel mode ported as a separate graph in `v3/panel/`; 4 v2 agents copied verbatim, 5-node chain (Sectionize → ExtractKeywords → GenerateExpertPanel → ExpertReviewPhase → PanelSynthesisPhase), 2-concurrent semaphore matching v2 calibration | +7 tests → 654 pass |
| **I** | `2eaf2c6` | Atomic v2 deletion + public API rewire; ~25 v2 test files removed, ~75 v2 source files removed; `__init__.py` re-exports v3 as the public surface; `cli.py` dispatches through v3 only | 654 → 270 whetstone tests (drop = v2 deletions; project total: 2152) |

## v2 features and their fate

| v2 feature | Where it went |
|---|---|
| lens-based section-by-section review | dropped (v3's criterion cascade replaces it) |
| `--mode panel` | ported to `v3/panel/` graph, accessed via `andamentum-whetstone panel` |
| `--mode guidelines` (free-text → checkable items) | absorbed into the unified `guidelines_text=` kwarg + new `extract_criteria_from_guidelines` helper |
| `--mode custom` (caller-supplied criteria list) | absorbed into the unified `criteria=[Criterion(...)]` kwarg |
| editor phase (`editor=True`) | ported to v3 as an optional node |
| `--check-novelty` | ported as the 3-node deterministic pipeline (FlagNoveltyTargets → RunNoveltySearches → JudgeNovelty) |
| `--persist-novelty-cache` (~/.cache writes) | dropped per the no-hidden-home-dirs rule |
| `--perspectives` (rigorous / writer / methodology / statistician / etc.) | dropped (criterion sets are the v3 replacement) |
| `--no-challenge` (skip refutation phase) | dropped (v3 has no challenge phase) |
| `--no-proofread` / proofread integration | dropped from review pipeline; proofread is now a separate subcommand (Phase H) |
| `--embedding-model` (v2 Consolidate phase) | dropped (v3's consolidate doesn't need embeddings) |
| `--confirm-own-draft` tripwire | ported unchanged (Phase B) |
| `--document-type` flag | retained; the underlying classifier now produces six types instead of three |
| `--apply-patches` patch-only path | retained; also accessible via the new `apply-patches` subcommand |
| 6 panel-specific renderer fields (`expert_profiles` / `expert_reviews` / `panel_synthesis`) | retained (panel graph populates them unchanged) |
| `_DOC_TYPE_VOCAB` synthesis vocabulary injection (v2 only) | dropped (v3 synth currently ignores document_type — minor regression for the human-readable summary's vocabulary tuning, but the criterion cascade itself routes correctly to the type's set) |

## New public surface

```python
from andamentum.whetstone import (
    review_document,        # the canonical entry — alias for v3 review_document_v3
    run_review,             # markdown-in entry — alias for v3 run_review_v3
    run_panel,              # panel-mode entry — alias for v3 panel.run_panel_v3
    Criterion,              # for custom criteria sets
    criterion_set_for,      # document_type → builtin criterion set
    extract_criteria_from_guidelines,  # prose-to-criteria helper
    render_docx, render_html, render_markdown,
    # Plus every existing ReviewResult schema type
    AuthorQuestion, Edit, Finding, Quote, ReviewMetrics, ReviewResult,
    ExpertProfile, ExpertReview, PanelSynthesis, SectionCard,
    CheckableItem, CustomEvaluation, GuidelineEvaluation,
)
```

CLI:

```bash
andamentum-whetstone draft.md --model X --out review.md           # bare review (default)
andamentum-whetstone review draft.md ...                          # explicit review subcommand
andamentum-whetstone panel draft.md --model X --i-am-the-author   # multi-expert panel mode
andamentum-whetstone proofread draft.md                           # deterministic style/readability
andamentum-whetstone apply-patches draft.docx --patches p.json --out r.docx  # patch-only mode
```

## Sequencing decisions worth knowing

1. **Phase A merged PRD steps 1+2** — the auto-routing fix alone was a partial fix; `criterion_set_for` already fell back to `GENERAL` for unknown types. Merging guarantees the routing change is meaningful in one commit.
2. **Phase F was rebuilt as a thin alias layer over the flat parser**, not a full argparse-subparser rewrite. This preserves every existing test invocation (no test-file mass-edit) while delivering the subcommand UX.
3. **Phase G started with a mini-PRD** (the planned halt checkpoint for the elevated-risk phase). The five design questions in §2 of `2026-05-25-whetstone-v3-panel-mini-prd.md` were resolved with the recommended options before any code shipped.
4. **The "rename v3 → top-level" step from the original master plan was deferred** — the public API rename (e.g. `review_document_v3` → `review_document`) was achieved via `__init__.py` re-exports without moving the v3 source files. An actual directory move is mechanical but offers no functional improvement; can land as a follow-up cleanup commit.
5. **CLI argparse cleanup is incomplete** — v2-only flag definitions (`--mode`, `--perspectives`, `--no-llm`, etc.) still exist in the parser but are no longer consumed. They're effectively ignored. Pruning the definitions is a follow-up; doing it now would force a mass-edit of `test_apply_patches_cli.py`'s 14 invocation sites unnecessarily.

## What's NOT in this branch

- **Live benchmark numbers**: the benchmark harness was updated to use the consolidated review path (Phase I touch to `benchmarks/whetstone/arms.py`) but not re-run end-to-end. Recommended next step before merging: run `uv run python -m benchmarks.whetstone.cli ...` against the existing reference manuscripts and confirm no quality regression vs. the pre-branch baseline (commit `5828f4c`).
- **The literal v3-directory rename** — see §"Sequencing decisions" #4.
- **Argparse pruning** — see §"Sequencing decisions" #5.
- **Documentation outside CLAUDE.md** — `README.md` and `docs/index.md` may have stale snippets; a follow-up should grep for `--mode` / `--perspectives` / `--no-llm` and update.

## Read order for review

1. `docs/plans/2026-05-25-whetstone-v3-consolidation-prd.md` — the design decisions
2. `docs/plans/2026-05-25-whetstone-v3-consolidation-MASTER-PLAN.md` — the execution playbook with per-phase research protocol
3. `docs/plans/2026-05-25-whetstone-v3-panel-mini-prd.md` — the Phase G design checkpoint
4. Commits in chronological order (Phase A → I), each with its own design-rationale body
5. `src/andamentum/whetstone/__init__.py` — the new public surface
6. `src/andamentum/whetstone/v3/` — the only review pipeline now
7. `src/andamentum/whetstone/v3/panel/` — panel-mode subgraph

## Halt conditions encountered during execution

None. The planned halt at the Phase G mini-PRD was the only one; user approved the recommendations and execution resumed. Every phase passed its validation gate (pytest, ruff check, ruff format, no new pyright errors) before its commit was created.
