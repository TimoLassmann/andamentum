# Changelog

## 0.3.0rc1 — 2026-05-13

Pre-release tag. This is the first version of `andamentum` with a
unified routing layer across initial gather and follow-up
investigation rounds, plus an explicit stage invariant that prevents
the class of silent-calibration-regression bug that the architectural
work surfaced and fixed.

### Highlights

- **Description-driven dispatch is the only evidence-gathering path.**
  Provider knowledge lives on each provider class as four self-
  description attributes (`description`, `query_guidance`,
  `query_examples`, `output_kind`). A single generic dispatch agent
  reads those attributes at runtime to commit native-syntax queries
  or abstain. Adding a new provider is a class-attribute + HTTP-
  wrapper task with no agent design.
- **Investigation rounds use the same dispatch path as initial
  gather.** `epistemic_investigate_claim` is now a pure gap-analysis
  agent: it generates 0–3 methodological *intents* (named angles
  along method / population / temporal frame / control / level-of-
  analysis dimensions) and the dispatch agent handles routing. The
  intent layer carries per-round yield-annotated memory and may
  return zero intents to rationally suspend judgment when the search
  space is exhausted.
- **Source-agnostic judging contract, enforced as a stage
  invariant.** `ExtractNewEvidence` judges by predicate (any
  claim-linked, content-bearing Evidence with no `support_judgment`
  is judged) regardless of which path created it. The
  `scrutiny_and_investigation` stage exit invariant
  (`_all_active_claim_evidence_judged`) refuses to proceed if any
  such Evidence remains unjudged — a future creation path that
  bypasses judging fails loudly at the stage boundary rather than
  silently degrading calibration.
- **Filtered resolved uncertainties from the investigation agent's
  input.** Resolved uncertainties no longer leak into
  `scrutiny_issues`, so the agent doesn't re-target gaps that have
  already been closed.

### Benchmarks (dev30, n=20 calibratable)

- **Epistemic (gpt-5.4-nano)**: AUC 0.89, Brier 0.166, ECE 0.167.
- **vs. baseline_frontier (gpt-5.4 alone)**: tied within CIs.
- **vs. rag_replay on the same evidence pool**: Brier reduced from
  0.294 to 0.166 (40% reduction). This is the cleanest
  architecture-attributable claim from the run.

See `docs/results/dev30_v9.md` for the full table, methodology, and
honest discussion of what the run does and does not support.

### Removed

- The legacy three-agent evidence-gathering chain
  (`epistemic_select_provider`, `epistemic_rank_providers` for
  initial gather, `epistemic_formulate_query`).
- `PlanTaskOperation` and the `_run_provider_tournament` helper.
- The `dispatch_mode` toggle on `run_epistemic_graph` (single
  routing path now; no switch needed).
- The `PROVIDER_QUERY_GUIDANCE` and `PROVIDER_DESCRIPTIONS`
  module-level registry dicts. Provider data lives on the provider
  class.
- `get_source_catalogue` helper (orphaned).
- The orphaned `should_flag_for_review` helper in
  `adversarial_balance.py`.
- Three obsolete test files (`test_provider_tournament.py`,
  `test_phase2_lazy_planning.py`, `routing_benchmark_queries.py`).

### Added

- `andamentum.epistemic.dispatch` module — description-driven
  dispatch implementation (`DispatchResult`,
  `formulate_provider_query`, `gather_evidence_new`).
- `andamentum.epistemic.operations.dispatch_gather` —
  `DispatchGatherOperation` (initial gather) and the shared
  `dispatch_and_persist_for_text` helper used by both initial gather
  and investigation rounds.
- `andamentum.epistemic.entities.intent_record` — `IntentRecord`
  Pydantic model carrying intent text + per-intent yield count.
- `Claim.investigation_intents: list[IntentRecord]` — per-claim
  memory of follow-up search angles tried in prior rounds.
- `epistemic_dispatch_provider` agent (one per-provider call;
  receives `claim` + optional `angle`).
- `epistemic_investigate_claim` agent — rewritten to output intents
  with the dimension-shift discipline, replacing the prior
  query-generation output.
- Stage invariant `_all_active_claim_evidence_judged` enforced at
  the `scrutiny_and_investigation` boundary.
- New dispatch-quality benchmark harness at
  `benchmarks/epistemic/dispatch_quality/` (Tier 1 triage accuracy
  per provider with mocked-LLM unit tests).
- Smoke-test artefacts at `docs/results/dev30_v9.md`.
- Architecture summary at `docs/architecture.md`.

### Compatibility

This release is a behaviour change relative to `0.2.0`. Callers of
`run_epistemic_graph(...)` should:

- **Remove** any `dispatch_mode=` kwarg (the parameter no longer
  exists; the behaviour the old `"new"` value selected is now the
  default and only path).
- **Be aware** that `Claim.investigation_intents` is now
  `list[IntentRecord]` rather than `list[str]`; old database
  representations load via `IntentRecord.from_dict` with
  `evidence_count` defaulting to 0 for legacy rows.

No other public API changes.

### Canonical green state

- pyright: 23 errors (pre-existing test-only typing noise —
  pydantic-graph generic variance in `test_topology.py`,
  `Decomposition` dict-form fixtures in several test files).
- ruff: clean.
- pytest: 2075 passing, 2 skipped, 25 deselected.
