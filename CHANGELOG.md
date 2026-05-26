# Changelog

## 0.3.0 — 2026-05-26

First stable release. Headline change is the whetstone v3 consolidation:
v2's lens-based section-by-section pipeline is gone; v3's criterion-
cascade is the only review path. The rest of this entry covers
follow-up polish and release-prep work since `0.3.0rc3`.

### Whetstone v3 consolidation

  - Six pluggable criterion sets routed by document type (academic /
    external_communication / essay / tutorial / creative / general),
    auto-detected by a one-shot classifier or set explicitly via
    `--document-type`. Essays, tutorials, and creative writing get
    criteria that fit the form instead of being forced through SPECS.
  - `--criteria` / `--guidelines` collapsed into one unified
    criterion-input surface. Pass either pre-decomposed `criteria=`
    or free-text `guidelines_text=`; an extractor decomposes the
    latter into a criterion list in one LLM call.
  - Confidentiality tripwire (`_confidentiality.py`) ported to v3 and
    runs as the very first step of `run_review_v3`, before any LLM
    call including the auto-classifier.
  - Editor phase ported as an optional node between Synthesise and
    Finalize; concrete `Edit` objects feed the docx track-changes
    renderer when `--editor` is passed.
  - Novelty check restructured from v2's single-node-with-tool shape
    into a deterministic 3-node pipeline (FlagNoveltyTargets →
    RunNoveltySearches → JudgeNovelty). Reproducible flag-set,
    bounded search budget, on-disk novelty cache dropped per the
    no-hidden-home-dirs rule.
  - Panel mode now lives in its own graph (`whetstone.v3.panel`).
    Invoked via the new `andamentum-whetstone panel` subcommand;
    keeps the existing N-expert biosketches + per-expert reviews +
    panel synthesis shape, with the in-code authorship gate
    (`--i-am-the-author`) preserved.

### CLI: four subcommands

  `andamentum-whetstone` is now organised around verbs:
    review (default — criterion-cascade), panel, proofread, apply-patches.
  Bare positional invocations still route to `review` so existing
  scripts don't break. 7 dead v2 flags pruned (`--v3`,
  `--embedding-model`, `--no-challenge`, `--perspectives`,
  `--no-llm`, `--no-proofread`, `--persist-novelty-cache`).
  Remaining flags organised into argparse argument groups so the
  `--help` output reads cleanly.

### Output layout

  - HTML + markdown renderers rewritten to an editorial-annotation
    layout: per-finding section header → quoted passage in
    `tone-quote` callout → comment in `tone-warning` / `tone-note`
    callout with severity / confidence chips. Composes existing
    typeset atoms — no CSS edits, no new visual primitives.
  - Document map moves from the bottom of the report to the top so
    section ids in finding headers are interpretable on first read.
  - The two stacked top banners (note + AI-watermark) merge into
    one combined warning.
  - No more collapsed `<details>` cards; comment body is visible
    at first read.

### Metrics + reliability

  - Real LLM-call + gap-round counters wired throughout v3 (was
    always `0` after the consolidation regression). Uses a
    `ContextVar`-held counter so `asyncio.gather`'d tasks aggregate
    into one total. Reads pydantic-ai's `result.usage().requests`
    to capture tool-call expansion.
  - Loud-fail on missing input path: a path-shaped string that
    doesn't exist now raises `FileNotFoundError` with a clear
    message; previously the path string was silently treated as raw
    markdown content (the LLM "reviewed" the literal string
    `/tmp/draft.md`).

### Release-prep

  - **Supply-chain hardening.** Added a `[tool.uv]` block with a
    rolling 28-day cooldown — at resolve time `uv` ignores any
    package version uploaded in the last 28 days, blocking
    freshly-published malicious releases from being pulled before
    the community has time to flag them. The lockfile remains the
    actual pin; CI / containers should run `uv sync --locked` (or
    `UV_LOCKED=1`) to refuse silent drift.
  - **Dependency upper bounds.** Every direct dependency now has an
    explicit upper bound (e.g. `pydantic>=2,<3`, `docling<3`,
    `matplotlib<4`, `sqlite-vec<0.2`) so a future major bump can't
    silently break the install.
  - **License hygiene.** `trafilatura` (GPL-3.0) moved from the
    default `dependencies` to a `[project.optional-dependencies]`
    extra (`pip install andamentum[html-articles]`); the default
    install is now MIT-clean. The harvest backend falls back to
    `docling` automatically when trafilatura isn't installed.
  - **PyPI metadata.** `[project.urls]` block added (Homepage,
    Repository, Documentation, Changelog, Issues) so PyPI's package
    page links back to the source.
  - **Plans + smoke files relocated.** 16 internal design docs
    moved from `docs/plans/` to `docs/.internal/plans/`. Smoke-run
    review outputs at the repo root moved to `docs/results/smoke/`.
    Both directories carry a README labelling them maintainer-only
    so first-time users don't mistake them for documentation.
  - `andamentum-epistemic` prints an explicit `⚠ EXPERIMENTAL`
    banner to stderr on every invocation. The module ships
    installed and is callable, but output shape / agent names /
    flags may still change without notice.

### Other

  - Documentation refreshed for the v3-only surface (README,
    docs/index.md, docs/architecture.md, both RESPONSIBLE_USE.md
    files). `docs/whetstone/lenses/strunk.md` removed — the Strunk
    lens was a v2-only feature deleted in the consolidation.
  - `<your-github-handle>` placeholder in `CITATION.cff` and the
    `harvest` / `deep_research` User-Agent string replaced with
    `TimoLassmann`.

## 0.3.0rc3 — 2026-05-13

Adds a **second report layout** alongside the classic one — Cochrane-
style audit report. The classic layout (``typeset_report.py``) is
unchanged; the new audit layout lives in ``audit_report.py``. Choose
between them via ``--report-style {classic,audit,both}``. ``both``
writes two files so they can be opened side-by-side for comparison.

### Why a parallel layout

The classic report is prose-led and reads cleanly when the answer is
self-evident. The audit report is structured for cases where the
*reasoning trail* is the value — clinical decision support,
regulatory submissions, anywhere "show your work" matters. The two
layouts are not redundant; they target different reader needs.

### What the audit layout shows

1. **Headline panel** — claim, verdict badge (Supported / Refuted /
   Inconclusive / Insufficient evidence), posterior pill.
2. **Summary of findings** — small Cochrane-style table immediately
   under the headline: directional split (supports / contradicts /
   no bearing) with counts and percentages.
3. **Plain-language summary** — the existing answer prose, separated
   from the evidence breakdown.
4. **Key evidence per claim** — claim card (with collapsible details
   carrying the audit trail) plus the top 3-5 supporting items and
   the strongest counter-evidence rendered inline as a markdown list
   with **clickable source links** (DOI / PMID / NCT auto-link). The
   classic layout's inline reference-number list (which on a 98-item
   claim rendered as ``1, 2, 3, …, 98``) is gone.
5. **Audit trail per claim** — investigation rounds as a real list
   (each round its own bullet, properly rendered), IBE chain
   candidates as a markdown table with loveliness / likeliness
   scores, and the adversarial probe explicitly framed. All folded
   into the claim card's ``<details>`` so the reader opts in.
6. **Caveats & Limitations** — same data as classic, visually
   separated.
7. **Appendix** — a single collapsible card containing the full
   evidence trail (every retrieved item with its one-sentence
   judgement and clickable source), grouped by direction. The full
   list is available for verification but not imposed on the scanner.

### Source-ref clickability

DOI, PMID, and NCT identifiers are now auto-converted to clickable
URLs:

- ``doi:10.1234/abc`` → ``https://doi.org/10.1234/abc``
- ``PMID:12345678`` → ``https://pubmed.ncbi.nlm.nih.gov/12345678/``
- ``NCT04501978`` → ``https://clinicaltrials.gov/study/NCT04501978``
- Bare ``https://`` URLs are unchanged.

Used by the audit layout for both inline highlights and the
appendix's full list. The classic layout is unchanged.

### Run modes

The audit layout handles both run modes from the same code path:

- **Verify mode**: one claim seeded from ``claim_to_verify`` — single
  ``Key evidence`` section with the claim card and audit trail.
- **Research mode**: question decomposed into sub-investigations —
  ``Sub-investigations`` section with each sub-claim numbered
  (#1, #2, …), its own card, key-evidence breakdown, and audit
  trail. The combined verdict shows in the headline.

### Files

- ``src/andamentum/epistemic/audit_report.py`` (new) — the Cochrane-
  style renderer. ~500 LOC. Uses only the 7 built-in typeset atoms
  (heading, prose, callout, items, aside, card, reference) — no
  custom atoms added.
- ``src/andamentum/epistemic/report_generator.py`` — ``generate_html``
  and ``save_html`` now take a ``style`` parameter
  (``"classic"`` (default) | ``"audit"``).
- ``src/andamentum/epistemic/cli.py`` — ``--report-style`` flag added
  to ``ask`` and ``verify`` subcommands. ``both`` writes two files
  with ``-classic.html`` and ``-audit.html`` suffixes.
- ``src/andamentum/epistemic/cli_handlers.py`` — ``handle_ask``
  threads ``report_style`` through to ``save_html``.

### Tests

- 22 new tests in ``test_audit_report.py``. Covers source-URL
  conversion, verdict callouts (Supported / Refuted / Inconclusive /
  Insufficient), Summary-of-Findings table, single-claim rendering
  (no inline number list, evidence counts in details, clickable
  source links, audit trail in card details), research-mode rendering
  (sub-investigations section, numbered sub-claims), and appendix
  presence/absence.

### Verification

- pyright: 23 errors (unchanged baseline).
- ruff: clean.
- pytest: 2105 passing (+22 audit-report tests), 2 skipped,
  25 deselected.

The classic report is bit-for-bit unchanged; existing databases
re-render identically under ``--report-style classic``. The audit
layout consumes the same ``ReportData`` so existing dev30 v9
databases render audit-style with zero LLM tokens.

## 0.3.0rc2 — 2026-05-13

Report-rendering update. Surfaces the audit trail of investigative
work that was already stored in entity state but previously invisible
to readers of the rendered report. **No behaviour change** — the
pipeline, operations, agents, and scoring are unchanged. Pure
rendering and report-data extraction. Existing databases re-render
into the new shape; no re-run required.

### New report sections

- **How this claim was investigated** (per claim) — every follow-up
  intent the gap-analysis agent proposed across investigation rounds,
  with the routing-yield count per intent. Empty when the claim
  reached a verdict on initial gather alone. Reads from
  `Claim.investigation_intents`.
- **Inference to the best explanation** (per claim) — every IBE
  candidate the integration step enumerated, with its loveliness /
  likeliness scores, the chosen candidate marked, the integrated
  assessment shown. Reads from `Claim.integration_candidates`.
- **Adversarial probe** (per claim) — counterarguments are now
  prefaced with an explicit "the system searched for evidence that
  would contradict this claim" intro, so the reader sees the probe,
  not just the result. Same underlying data; reframed presentation.
- **Evidence judgement breakdown** (top-of-Sources) — total support /
  contradict / no_bearing counts with percentages. The reader can see
  at a glance the audit-trail view of how each retrieved item was
  categorised.

### Schema additions to `report_data.py`

- `InvestigationRound` dataclass — text + evidence_count per round.
- `IBECandidate` dataclass — candidate description, loveliness,
  likeliness, chosen / runner_up flags, gap scores.
- `ClaimSummary` extended with `investigation_rounds`,
  `ibe_candidates`, `integrated_assessment`, `integrated_confidence`.
- `InvestigationStats` extended with `evidence_supports`,
  `evidence_contradicts`, `evidence_no_bearing`, `evidence_invalidated`,
  `investigation_rounds_total`.

### Tests

- 8 new tests in `test_report_audit_trail.py` covering: investigation
  rounds rendering (present, empty, singular/plural yield), IBE
  candidates rendering (chosen/runner-up/rejected), adversarial probe
  intro, evidence judgement breakdown.

### Verification

- pyright: 23 errors (unchanged baseline).
- ruff: clean.
- pytest: 2083 passing (+8 from new audit-trail tests), 2 skipped,
  25 deselected.

Re-renders cleanly against existing databases. Smoke-tested against
`test2_hcq` (100 evidence items, 9 investigation intents across 3
rounds, verdict: fail — correctly identified) and `test3_statins`
(33 items, 5 IBE candidates, chosen=B "supports_refined", verdict:
pass).

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
