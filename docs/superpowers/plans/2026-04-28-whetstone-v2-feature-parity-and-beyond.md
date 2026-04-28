# Whetstone v2 Feature Parity (and Beyond) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bring v2 to feature parity with v1 (steps 1–6) and add capabilities that neither version has but that a research team leader actually uses every week (steps 7–13). When complete, v1 (`src/andamentum/whetstone/`, top level) is deleted and v2 is renamed flat into its place.

**Non-goal:** Keeping v1 alive. Once a v2 capability matches or exceeds the v1 equivalent, the v1 module/test/CLI is removed in the same step's PR. No long-running parallel codebase.

**Architecture continuity:** Every new capability extends the existing v2 design — pydantic-graph state machine, lens-agent pattern, anchored quotes via `chunker.validation.find_anchor`, deterministic structural substrate in `v2/structural/`, single-job LLM agents. Nothing in this plan introduces a new architectural primitive. New behaviour comes from new lens prompts, new structural extractors, and new graph nodes that compose with the existing reflection loop.

**Tech stack:** No new dependencies. Step 8 (statistical self-consistency) uses `scipy.stats`, already in the dependency tree. Step 10 (novelty check) reaches into `andamentum.deep_research`. Everything else is prompt and graph-node work.

---

## Background — why this plan exists

The 2026-04-28 audit found v2 is genuinely better than v1 on architecture (anchored quotes, deterministic substrate, reflection loop, challenge phase) but worse on substance: required-statements checklist is missing, panel mode is missing, prompts are thinner, MUST/SHOULD/CONSIDER bucketing is gone, prose-level consistency LLM check is absent, custom criteria are gone, journal-guidelines extractor is gone. Issuing reviews from v2 today loses real signal compared with v1.

The user explicitly does not want v1 and v2 living in parallel as a long-term state. This plan therefore covers (a) every v1 capability that hasn't yet been ported — in priority order, smallest to largest — and (b) seven additions that aren't in either version but that close real gaps for a research team leader's weekly workflow.

The 2026-04-27-whetstone-v2-critical-review plan describes the existing v2 design; this plan extends it. Read that plan first.

---

## Step 1 — Required-statements checklist as deterministic structural findings

**Why first:** The single most-asked-for capability v2 lacks. Catches what kills a real submission — missing COI, no data-availability statement, no ethics approval, missing funding line. The v1 implementation is already mostly deterministic, transfers nearly line-for-line into `v2/structural/`, and needs zero LLM design work.

**Scope:**
- Create `src/andamentum/whetstone/v2/structural/checklist.py` that emits `Finding` objects for each missing required-statement.
- Required statements covered (ported from v1 `checklist_scanners.py` and `agents/checklist.py`):
  - COI / Conflict of Interest declaration present
  - Data-availability / data-sharing statement present
  - Ethics approval / IRB / IACUC / animal-care statement present
  - Funding / acknowledgements statement present
  - Author affiliations block present
  - Keywords list present, between 3 and 8 items
  - Title present, length within journal-typical range (10–25 words)
  - Abstract present, within wordcount range (configurable, default 150–300)
  - Abstract has IMRAD-ish structure (Background/Aim → Methods → Results → Conclusion cues)
  - Figures referenced from the prose are all present and numbered consecutively
  - Tables similarly
  - Every reference in the bibliography is cited at least once in the prose (already partially in `v2/structural/citations.py` — extend rather than duplicate)
- Detection is regex/structural over the section tree, not LLM. Each missing item becomes a Finding with severity `major`, category `compliance`, source `deterministic-checklist`, and an explicit fix suggestion.
- Wire into `v2/graph.py` as a new node `RunChecklistChecks` running in parallel with the existing structural extractors.

**Tasks:**
- [ ] 1.1 — Create `v2/structural/checklist.py` with one extractor function per required-statement, each returning `list[Finding]`.
- [ ] 1.2 — Extend `v2/structural/deterministic_findings.py` (or its successor) to call the new extractors.
- [ ] 1.3 — Port v1's `checklist_scanners.py` tests into `v2/tests/test_checklist_structural.py`, adjusted to assert against `Finding` not `DocumentIssue`.
- [ ] 1.4 — Update v2 CLI: when `--checklist-statements` flag is passed (default on for `--mode review`), surface checklist findings prominently in the docx renderer.
- [ ] 1.5 — Delete `src/andamentum/whetstone/checklist_scanners.py` and the v1 checklist agent. Remove the v1 CLI's `--task checklist` route.
- [ ] 1.6 — Verify pyright/ruff/pytest clean.

---

## Step 2 — Enrich v2 lens prompts with v1's enumerated failure modes

**Why second:** The biggest cause of "v2 finds less than v1" is prompt thinness, not graph structure. The v2 agents already exist; we are upgrading what they are told to look for. Zero structural change.

**Scope:**
- For each existing v2 lens prompt (`v2/agents/lens_prompts.py`: `_RIGOROUS_PROMPT`, `_WRITER_PROMPT`, `_METHODOLOGY_PROMPT`, `_STATISTICIAN_PROMPT`), lift the enumerated failure-mode lists from the v1 equivalents:
  - `agents/review.py` `_METHODOLOGY_PROMPT` (~50 lines of categories + 10–15 issue quota)
  - `agents/review.py` `_CORE_SCIENTIFIC_MERIT_PROMPT`, `_CLARITY_ACCESSIBILITY_PROMPT`, `_RESULTS_INTERPRETATION_PROMPT`
  - `agents/editing.py` `_UNIFIED_EDITOR_PROMPT` (subject-verb agreement, parallel structure, dangling modifiers, run-ons, comma splices, weak verbs, "data is plural" etc.)
- Keep the v2 single-section, 0–3-issues-per-section calibration. The reflection loop is what recovers cross-section signal — don't break that contract by reintroducing whole-document quotas.
- The category vocabulary stays bounded (current 7-word vocab) but each lens prompt expands the "things that fall under this category" list.
- Calibrate each upgraded prompt against the synthetic-corpus method used in commit 552ff96 (`test(deep_research): calibrate page_summarizer prompt against synthetic corpus`). Build `v2/tests/calibration_corpus.py` with hand-labelled section excerpts → expected findings. Run each lens against the corpus; expect ≥85% precision and ≥70% recall before merging.

**Tasks:**
- [ ] 2.1 — Build `v2/tests/calibration_corpus.py` with ~30 hand-labelled section excerpts × expected findings per lens.
- [ ] 2.2 — Upgrade `_RIGOROUS_PROMPT`, lift content from v1 `_CORE_SCIENTIFIC_MERIT_PROMPT` + `_RESULTS_INTERPRETATION_PROMPT`. Run calibration; iterate.
- [ ] 2.3 — Upgrade `_WRITER_PROMPT`, lift content from v1 `_CLARITY_ACCESSIBILITY_PROMPT`. Calibrate.
- [ ] 2.4 — Upgrade `_METHODOLOGY_PROMPT`, lift content from v1 methodology prompt. Calibrate.
- [ ] 2.5 — Upgrade `_STATISTICIAN_PROMPT`, lift content from v1 results-interpretation. Calibrate.
- [ ] 2.6 — Upgrade `EDITOR_PROMPT` (`v2/agents/editor.py`), lift enumerated failure modes from v1 unified editor. Calibrate against an editing corpus.
- [ ] 2.7 — Update v2 lens-prompt tests to lock in the new behaviour.

---

## Step 3 — MUST-FIX / SHOULD-FIX / CONSIDER bucketing in synthesis

**Why third:** Single prompt edit + one schema field. Recovers v1's prioritisation, the thing that makes a review actionable.

**Scope:**
- Add `priority: Literal["must_fix", "should_fix", "consider"]` to v2 `Finding` (`v2/schemas.py:33-87`).
- Each lens infers initial priority from severity (`major` → `must_fix`, `moderate` → `should_fix`, `minor` → `consider`) with override possible during reflection.
- The `synthesise` agent (`v2/agents/synthesise.py`) prompt is rewritten to:
  - Open with a 3-paragraph executive summary (kept from current v2).
  - Then enumerate findings in three explicit buckets — MUST FIX BEFORE SUBMISSION / SHOULD FIX / CONSIDER.
  - Each bucketed item links back to the Finding by id.
- Renderers (`render_markdown`, `render_html`, `render_docx`) display the buckets prominently. The docx renderer prepends a one-page summary section before track-changes.

**Tasks:**
- [ ] 3.1 — Add `priority` field to `Finding` schema.
- [ ] 3.2 — Update lens prompts to set initial priority from severity.
- [ ] 3.3 — Rewrite `synthesise` prompt for bucketed output.
- [ ] 3.4 — Update three renderers to surface buckets.
- [ ] 3.5 — Tests in `v2/tests/test_synthesise_buckets.py`.

---

## Step 4 — Panel mode (expert-generator → expert-reviewer → panel-synthesiser)

**Why fourth:** First step that needs new graph topology. Larger but self-contained. Implements as an alternate entry-mode (`mode="panel"`) so it doesn't disturb the standard review pipeline.

**Scope:**
- New v2 graph entry point: `review_document(source, *, model, mode="panel", n_experts=4, disciplines=None)`.
- Three new graph nodes:
  - `ExtractKeywords` — single LLM call producing 5–10 discipline-tagging keywords from the document. Port v1 `agents/multi_expert.py:_KEYWORD_EXTRACTOR_PROMPT`.
  - `GenerateExpertPanel` — single LLM call producing N fictional expert biosketches matched to the keywords, returning `list[ExpertProfile]` (port v1 schema). Each biosketch includes name, position, education, contributions, research focus, discipline.
  - `ExpertReview` — N parallel LLM calls, one per expert, each producing `ExpertReviewOutput` with rigor / methodology / novelty / clarity scores 1–10 + justifications + strengths + weaknesses + recommendation (Accept / Minor Revisions / Major Revisions / Reject) + recommendation justification.
  - `PanelSynthesise` — single LLM call surfacing consensus opinions, divergent opinions, and an aggregated recommendation.
- Re-use v1 prompts from `agents/multi_expert.py` directly — they are already well-tuned.
- Output schema: `ReviewResult.expert_profiles: list[ExpertProfile]`, `expert_reviews: list[ExpertReviewOutput]`, `panel_synthesis: PanelSynthesisOutput`.
- Renderers must handle panel output (the v1 docx/html already has the layouts — port them).

**Tasks:**
- [ ] 4.1 — Port `ExpertProfile`, `ExpertReviewOutput`, `PanelSynthesisOutput` schemas from v1 into `v2/schemas.py`.
- [ ] 4.2 — Implement `ExtractKeywords` node + agent.
- [ ] 4.3 — Implement `GenerateExpertPanel` node + agent.
- [ ] 4.4 — Implement `ExpertReview` node (N parallel calls) + agent.
- [ ] 4.5 — Implement `PanelSynthesise` node + agent.
- [ ] 4.6 — Wire mode dispatch in `v2/graph.py` and CLI flag `--mode panel`.
- [ ] 4.7 — Renderers handle panel output (port v1 layouts).
- [ ] 4.8 — Tests with mocked agents for panel-mode happy path + edge cases.
- [ ] 4.9 — Delete v1 `agents/multi_expert.py` and the v1 panel CLI route.

---

## Step 5 — Prose-consistency lens (terminology drift, tense shifts, contradicting claims)

**Why fifth:** v2's structural substrate catches the *mechanical* consistency subset (acronym redefinitions, inconsistent N values). The *semantic* subset (terminology drift, tense/voice/person shifts, prose claims that contradict each other across sections) needs an LLM. New lens, but it reuses existing infrastructure.

**Scope:**
- New lens `_CONSISTENCY_PROMPT` in `v2/agents/lens_prompts.py` ported from v1 `agents/consistency.py:_CONSISTENCY_REVIEWER_PROMPT`.
- The consistency lens is an exception to the "one section at a time" rule: it sees the full document map plus the verbatim text of all sections it is asked to compare. Add a `multi_section: bool` flag to the lens-agent contract to support this.
- Detection categories (from v1): cross-section number disagreement, terminology drift (same concept named differently in different sections), claim-emphasis shift (same finding flagged variably as central/peripheral), tense/voice/person shifts, methods/scope contradictions in prose.
- Wire into the existing `CriticalRead` node as an additional lens, configurable via `--perspectives consistency`.

**Tasks:**
- [ ] 5.1 — Add `multi_section` bit to the lens-agent contract.
- [ ] 5.2 — Port `_CONSISTENCY_REVIEWER_PROMPT`.
- [ ] 5.3 — Wire into `CriticalRead`.
- [ ] 5.4 — Tests over corpus with seeded inconsistencies.
- [ ] 5.5 — Delete v1 `agents/consistency.py`.

---

## Step 6 — Journal-guidelines and custom-criteria modes

**Why sixth:** The most design-heavy parity step because v2 needs a runtime-schema mechanism (v1 had `dynamic_models.py`).

**Scope:**
- **Journal-guidelines mode:** Take a free-text journal author-guidelines file (`--guidelines @nature.txt`). New `ExtractCheckableItems` node turns the free text into 10–30 structured check items (each: name, description, expected output type, pass/fail criteria). Each item is then evaluated either by a deterministic check (if recognisable — wordcount, presence of section, etc.) or by a `GuidelinesEvaluator` lens (one LLM call per item).
- **Custom-criteria mode:** `--criteria "criterion1; criterion2; ..."` produces a runtime schema and a single `CustomReviewer` lens evaluating each criterion. Port v1 `dynamic_models.py:create_output_model` into `v2/dynamic_schemas.py`.
- Both modes emit `Finding`s into the standard pool; downstream pipeline (reflection, challenge, synthesis) treats them like any other finding.

**Tasks:**
- [ ] 6.1 — Port `dynamic_models.py` → `v2/dynamic_schemas.py`. Strip ad-hoc dictionary contracts; use Pydantic create_model with field-level validators.
- [ ] 6.2 — Implement `ExtractCheckableItems` node + agent.
- [ ] 6.3 — Implement `GuidelinesEvaluator` lens (per-item LLM call).
- [ ] 6.4 — Implement `CustomReviewer` lens.
- [ ] 6.5 — CLI flags `--guidelines @file` and `--criteria "..."`.
- [ ] 6.6 — Tests.
- [ ] 6.7 — Delete v1 dynamic_models, custom-criteria, and guidelines code paths.

---

## Step 7 — Decommission v1

**Why seventh:** Once steps 1–6 land, v1 has no unique surface. Delete it.

**Scope:**
- Delete `src/andamentum/whetstone/agents/`, `src/andamentum/whetstone/consistency_scanners.py`, `src/andamentum/whetstone/checklist_scanners.py`, `src/andamentum/whetstone/dynamic_models.py`, `src/andamentum/whetstone/orchestrator.py`, `src/andamentum/whetstone/issues.py`, `src/andamentum/whetstone/models.py`, `src/andamentum/whetstone/cli.py`, `src/andamentum/whetstone/renderers/`, `src/andamentum/whetstone/tests/`.
- Move `src/andamentum/whetstone/v2/*` to `src/andamentum/whetstone/`.
- Update `pyproject.toml` `[project.scripts]`: `andamentum-whetstone` stays (now points at the only whetstone), `andamentum-whetstone-v1` is deleted.
- Update CLAUDE.md whetstone section to drop the v1/v2 distinction.
- Update top-level `src/andamentum/whetstone/__init__.py` `__all__` to expose the (former-v2) entry points.

**Tasks:**
- [ ] 7.1 — Verify no imports of `andamentum.whetstone` (outside `whetstone.v2`) remain elsewhere in the repo.
- [ ] 7.2 — Delete v1 files.
- [ ] 7.3 — Move v2 → top level.
- [ ] 7.4 — Update pyproject scripts, CLAUDE.md, and `__init__.py`.
- [ ] 7.5 — Verify pyright/ruff/pytest clean.

---

# Beyond v1 — capabilities that should exist but currently do not

The following steps add capabilities that neither v1 nor v2 has. They are independent of each other and can be sequenced in any order after step 7. Roughly priority-ordered.

---

## Step 8 — Statistical self-consistency check (statcheck-equivalent)

**Why this matters:** Published psychology has documented ~10–15% reporting errors detectable by mechanical recomputation; biomedical research is not far behind. This is one of the highest signal-per-effort additions we can make. Lives entirely in deterministic territory — no LLM, no flaky calibration.

**Scope:**
- New `v2/structural/stat_consistency.py`. The existing `v2/structural/numerics.py` already extracts p-values, percentages, and statistic strings. Extend the regex set to also extract: t-statistics + df, F-statistics + df1,df2, chi-square + df, z-statistics, r (correlation) + n, sample size N.
- For each (test_statistic, df, reported_p) triple, recompute the implied p-value from the test statistic (using `scipy.stats` already in the dependency tree). If reported and recomputed differ by more than the rounding tolerance, emit a `Finding` with severity `major` and category `statistics`. Distinguish between:
  - "decision-changing inconsistency" (reported p<0.05 but recomputed p≥0.05, or vice versa) → severity `major`
  - "non-decision-changing inconsistency" (both same side of 0.05 but more than rounding apart) → severity `moderate`
- Each finding includes the verbatim quote, the reported value, the recomputed value, and a one-line explanation of which test was assumed.

**Tasks:**
- [ ] 8.1 — Extend `numerics.py` extractors for t/F/chi-square/z/r.
- [ ] 8.2 — Implement `recompute_p_value(stat, df, kind)` using `scipy.stats`.
- [ ] 8.3 — Wire findings into `RunChecklistChecks` (or alongside).
- [ ] 8.4 — Test corpus of seeded inconsistencies + correct cases.

---

## Step 9 — Claim → evidence anchoring lens

**Why this matters:** This is the central senior-PI review move and currently both v1 and v2 do it softly. A claim like "X was significantly elevated" should anchor to a specific figure/table number AND a supporting statistic. The lens makes that anchoring explicit and flags claims that have neither.

**Scope:**
- New lens `_CLAIM_EVIDENCE_PROMPT`. Reads Results and Discussion sections (plus Abstract claims). For each claim it identifies, it asks whether the surrounding prose anchors the claim to (a) a figure/table reference and (b) a quantitative value (or qualitative one tied to a clearly-presented data source).
- Output: `Finding` per unanchored claim with severity `major` (if the claim is in the abstract or main results) or `moderate` (elsewhere). Categories: `claim_anchoring`.
- Reuse the existing v2 `chunker.validation.find_anchor` infrastructure to verify the anchored figure/table actually exists in the document.
- This lens is opinionated about which sections to read — it should not be applied to Methods or Background. Add a `target_sections: list[SectionKind]` field to the lens contract.

**Tasks:**
- [ ] 9.1 — Add `target_sections` to lens contract.
- [ ] 9.2 — Section-kind classifier (lightweight: heading-text → kind enum). Already partial in v2 chunker.
- [ ] 9.3 — `_CLAIM_EVIDENCE_PROMPT` + agent.
- [ ] 9.4 — Calibrate against the corpus from step 2 augmented with claim-anchoring labels.

---

## Step 10 — Novelty / prior-work check via deep_research

**Why this matters:** Closes the question every senior PI asks manually before signing off on a student's draft: "are we actually first / are we missing the obvious overlapping paper?" Connects two andamentum modules — this kind of integration is exactly what justifies them being in one package. No plugin equivalent; this lives in andamentum and only andamentum.

**Scope:**
- New v2 graph node `NoveltyCheck`. Runs after lens-reading, in parallel with the reflection loop. Steps:
  1. Extract 3–5 "novelty claims" from the document — sentences that explicitly claim novelty/firstness/lack-of-prior-work. Use a small LLM call (`_NOVELTY_CLAIM_EXTRACTOR_PROMPT`).
  2. For each novelty claim, call `andamentum.deep_research.run_research` with a goal derived from the claim ("Find published work that contradicts or pre-dates the claim that …"). Cap at one deep_research run per claim, with a small budget.
  3. Surface results as `Finding`s. If deep_research returns supporting evidence for prior work that contradicts the claim → severity `major`, category `novelty`. If it returns clearly-related but distinct work → severity `moderate`, category `prior_work_engagement`. If it returns nothing → no finding.
- Disabled by default. Enable with `--check-novelty` flag because deep_research runs cost real time/tokens.
- Cache deep_research results per claim hash so re-running on the same draft is cheap.

**Tasks:**
- [ ] 10.1 — Verify `deep_research.run_research` has a stable programmatic entry point (might need light surfacing).
- [ ] 10.2 — `_NOVELTY_CLAIM_EXTRACTOR_PROMPT` + agent.
- [ ] 10.3 — `NoveltyCheck` graph node with deep_research fan-out + result-to-Finding adapter.
- [ ] 10.4 — Per-claim disk cache.
- [ ] 10.5 — CLI flag, integration test using a recorded deep_research result.

---

## Step 11 — Overclaim / "reviewer 2 bait" lens

**Why this matters:** Tiny lens, weekly value when supervising students. Catches the words and patterns that draw aggressive Reviewer 2 comments before submission.

**Scope:**
- New lens `_OVERCLAIM_PROMPT`. Reads abstract + introduction + discussion. Detects:
  - Unsupported novelty assertions: "first", "novel", "unprecedented", "landmark", "groundbreaking", "paradigm-shifting" without citation
  - Mechanism-implied-where-correlation-shown: "X causes Y" claims when only correlational data was shown
  - Generalisation-beyond-data: claims about humans from a 4-mouse study, claims about populations from a 12-cell-line experiment
  - Miracle-effect-size language: "dramatic", "robust", "remarkable" without quantification
- Output: `Finding`s with category `overclaim` and severity tuned to where the claim sits (abstract/discussion → major; methods/results → moderate).
- Combine deterministic word-list detection with a per-hit LLM verification step (does the cited evidence actually support the strong language?). Word list alone produces too many false positives.

**Tasks:**
- [ ] 11.1 — Word lists in `v2/structural/overclaim_lexicon.py`.
- [ ] 11.2 — `_OVERCLAIM_PROMPT` + agent.
- [ ] 11.3 — Deterministic-then-LLM-verify pipeline.
- [ ] 11.4 — Calibration corpus.

---

## Sequencing summary

```
Parity track (deletes v1 by step 7):
  1 — Checklist (deterministic)         small   — afternoon
  2 — Lens-prompt enrichment            small   — 1 day
  3 — Synthesis bucketing               tiny    — half day
  4 — Panel mode                        medium  — 2–3 days
  5 — Prose-consistency lens            small   — 1 day
  6 — Journal-guidelines + custom       medium  — 2 days
  7 — Decommission v1                   small   — half day

Beyond-v1 track (independent of each other; pick by current pain):
  8  — Statistical self-consistency     small   — 1 day
  9  — Claim→evidence anchoring         small   — 1 day
  10 — Novelty check via deep_research  medium  — 2 days
  11 — Overclaim lens                   small   — 1 day
```

The four beyond-v1 additions all share a design philosophy: each is a *general mechanism* (a deterministic recomputation, a verifier of claim-to-evidence linkage, a domain-agnostic literature-overlap check, a generic over-strength-language detector) rather than an encoding of a specific journal or agency's compliance rules. Reporting-standard checklists (CONSORT/ARRIVE/PRISMA…), grant-specific scoring lenses (NHMRC/ARC/NIH…), and link/URL verification were considered and explicitly excluded — all three either over-fit andamentum to specific institutional rules or sit outside the system's epistemic-reasoning character. Whetstone stays a general document-quality engine; agency-specific compliance work, if needed, is the user's responsibility outside this tool.

After parity (step 7), the highest-leverage beyond-v1 additions are **Step 8 (statcheck)** and **Step 9 (claim→evidence anchoring)** — both substrate-extending and useful on every draft. **Step 10 (novelty via deep_research)** is the killer cross-module integration.

---

## Cross-cutting requirements

Every step:
- Maintains pyright 0 errors, ruff clean.
- Adds tests at the same level of granularity as existing v2 tests (per-node where applicable; per-prompt where the value is in calibration).
- Updates CLAUDE.md whetstone section if the public surface or sub-module structure changes.
- Lands as a single PR. No multi-step PRs that leave the tree in an intermediate state.
- Once a step ships, the corresponding v1 surface (if any) is removed in the same PR. No long-lived parallel codepaths.

Every renderer change:
- Word/HTML/markdown all carry the same information density. The docx track-changes file is the user's primary artifact and gets first-class treatment.

Every new LLM lens:
- Calibrated against a hand-labelled corpus before merge. Precision-first; the cost of a false-positive Finding (researcher loses trust in the tool) is higher than the cost of a false-negative.
- Output schema constrained at the Pydantic level so small local models can fill it reliably (per memory `project_small_local_models.md`).

---

## Open questions to resolve before starting

1. **Does v2's reflection loop need to run after the new lenses (steps 5, 9, 12) or do they bypass it?** Default: yes, all lenses feed into the same pool and the reflection loop sees everything. But the consistency lens (step 5) is multi-section by construction and may produce findings the reflection loop would only damage. Decide per-lens.
2. **Where does the section-kind classifier live?** Step 9 needs it; chunker has partial support. Either extend chunker or build a small whetstone-local classifier.
3. **Deep_research integration shape (step 10)** — is `run_research` programmatic-enough, or does it need a small wrapper for non-CLI callers? Audit before starting step 10.
4. **Calibration-corpus location** — under `v2/tests/calibration/` (test-time) or `v2/calibration/` (shipped, runnable as a benchmark)? Decide based on whether we want to expose calibration runs as part of the public CI.
