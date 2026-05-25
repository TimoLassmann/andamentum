# Whetstone v3 consolidation — one version, full feature coverage

**Status:** Design. User review required before build.
**Author:** Claude + user discussion 2026-05-25.
**Supersedes:** all of v2 (`whetstone/api.py`, `whetstone/orchestrator.py`,
v2 graph + nodes, v2 agents/, v2 renderers/, v2-only CLI branches).
**Builds on:** `docs/plans/2026-05-22-whetstone-v3-whole-doc-specs.md` (the
original v3 design) and the issue-1…9 increments shipped 2026-05-22→24.

---

## 1. Why now

Two versions of whetstone are live:

- **v2** — the lens-based, section-by-section pipeline. Has the full feature
  surface (panel mode, guidelines mode, custom mode, editor phase, novelty
  check, perspectives, proofread integration, confidentiality tripwire).
- **v3** — the whole-document, digest-focused, criterion-cascade pipeline.
  Cleaner architecture; the head-to-head benchmark arm (`5828f4c`) shows it
  is competitive. But it lacks most of v2's optional features and has at
  least one regression (the document-type classifier is bypassed in `auto`
  mode — see §4.1).

Carrying both is confusing for users and a maintenance tax for us. The
decision is: **v3 becomes the only version.** This PRD plans the port of the
v2 features we want to keep, redesigns the ones that need rethinking, drops
the rest, and ends with deleting v2.

## 2. Non-goals

- Re-litigating the v3 architecture (digest, criterion cascade, gap loop,
  consolidate, gate, synthesise, critique/revise). Those land as-is.
- Touching `whetstone.docx` track-changes machinery — only the
  Edit/Finding→DocumentPatch adapter changes.
- Building anything for grant-application drafting (see
  `feedback_no_grant_writing_tools`).
- Adding journal- or funder-specific checklists (see
  `feedback_prefer_general_mechanisms`).

## 3. Decisions (from triage)

| v2 feature | Decision | Notes |
|---|---|---|
| Panel mode | **Port** | Different review *shape*, not just different criteria. Stays as a separate top-level entry point. |
| Guidelines mode | **Port, collapsed with custom** | Becomes one criterion-list path; optional extractor LLM step for free-text input. |
| Custom mode | **Port, collapsed with guidelines** | Same evaluator. See §4.2. |
| Editor phase | **Port** | Load-bearing for track-changes .docx. One extra node, off by default. |
| Novelty check | **Port, redesigned** | Three deterministic graph nodes (flag → search → judge), not agent-with-tool. See §4.3 and [[novelty-check-deterministic-pipeline]]. |
| Perspectives / lenses | **Drop** | Subsumed by criterion sets — a "statistician" lens is really a different criterion set. Keeping both is redundant. |
| Proofread integration | **Separate** | Don't intermingle with idea review. See [[separate-style-from-ideas]] and §4.4. |
| confirm-own-draft tripwire | **Port** | Safety. Runs before any LLM call. |

Plus two pre-existing v3 gaps:

- **Auto-routing regression** — `cli.py:671-673` hard-codes `"academic"` when
  `--document-type auto`. The classifier in `whetstone/_document_type.py` is
  never called. See §4.1.
- **Essay / non-academic criterion thinness** — `GENERAL` is a grab-bag, no
  essay-specific criteria, no creative-writing surface. See §4.5.

## 4. Designs for the non-trivial items

### 4.1 Auto-routing fix

Trivial code change, but a clean place to start the branch:

```python
# v3/graph.py — review_document_v3
if document_type == "auto":
    sections = sectionize(md)
    titles = [s.title for s in sections]
    document_type = await classify(model=model, section_titles=titles, markdown=md)
```

CLI strips its own `auto → academic` workaround. Classifier output is
logged. No silent fallback — if classification fails the classifier already
returns `"general"`, which routes correctly.

### 4.2 Collapsed custom/guidelines path

One concept in v3: `criteria: list[Criterion]`. Three ways to populate it:

1. **Document-type default** — `criterion_set_for(document_type)` (the
   existing path; SPECS / EXTERNAL_COMMS / GENERAL / new essay sets).
2. **Caller-supplied list of strings** — what v2's `custom` mode took. We
   wrap each string in a `Criterion(name=string, questions=[string])`.
3. **Caller-supplied free-text prose** — what v2's `guidelines` mode took.
   An extractor LLM call (one call) decomposes prose into a criterion list;
   that list then enters path #2.

CLI:
- `--criteria "name1: question; name2: question"` → path #2
- `--guidelines @file.md` → path #3 → path #2
- Neither → path #1

API: `review_document_v3(..., criteria: list[Criterion] | None = None,
guidelines_text: str | None = None)`. Mutually exclusive with each other;
either overrides the document-type default. No `mode=` enum.

### 4.3 Novelty check as a three-node pipeline

Opt-in (`--check-novelty`). Inserts between `Consolidate` and `Gate`:

```
... → Consolidate → [FlagNoveltyTargets → RunNoveltySearches → JudgeNovelty] → Gate → ...
```

- **`FlagNoveltyTargets`** (one LLM call, optional): walks current findings
  + claims, picks the ones where novelty is at stake (significance/novelty
  criterion hits; claims tagged as the paper's main contribution). Emits
  `list[NoveltyTarget(claim_id, search_brief, justification)]`. Hard cap:
  `≤8` targets per run by default, configurable via `--novelty-target-cap`.
- **`RunNoveltySearches`** (N deep_research runs, parallel-bounded): for
  each target, calls `andamentum.deep_research` with `search_brief`. Emits
  `list[NoveltyEvidence(target_id, hits[{title, url, excerpt, year}])]`.
  Budget: `--novelty-search-depth` (already exists in v2) passed through.
- **`JudgeNovelty`** (one LLM call per target, parallel): given the claim,
  its evidence, and the search hits, emits a verdict
  `{target_id, verdict: novel|partial_overlap|prior_work_exists, summary,
  refs[…]}`. Verdicts become new `Finding`s under the `Significance`
  criterion (or `Argument` for non-academic).

Why three nodes, not one agent-with-tool: reproducibility, bounded budget,
benchmarkability. Same audit trail every run. See
[[novelty-check-deterministic-pipeline]].

### 4.4 Proofread as a separate path

Remove all proofread wiring from v3. `andamentum-proofread` already exists
as its own CLI; users who want both run them separately. No `--proofread` /
`--no-proofread` flag on whetstone.

**Decision:** ship a convenience subcommand
`andamentum-whetstone proofread <source>` that shells out to
`andamentum.proofread.analyze`. One CLI to discover, but the pipelines stay
disjoint — proofread never enters the review graph.

### 4.5 Essay / non-academic criterion taxonomy

Current three categories (`academic` / `external_communication` / `general`)
are too coarse. `GENERAL` claims to cover "notes, drafts, books, technical
docs, internal writeups" — a book chapter and a Slack writeup have nothing
in common.

Proposed expansion to **six** types (still kept narrow — see
[[prefer-general-mechanisms]]):

| Type | Examples | Sketch criteria |
|---|---|---|
| `academic` | manuscripts, theses, papers | SPECS (unchanged) |
| `external_communication` | blog posts, op-eds, press releases | Hook / Argument / Evidence / Voice / Clarity (unchanged) |
| `essay` | personal essays, narrative essays, opinion essays | Thesis / Narrative arc / Specificity / Voice / Fresh observation |
| `tutorial` | how-tos, technical walkthroughs, cookbooks | Goal / Prerequisites / Step ordering / Correctness / Completeness |
| `creative` | short fiction, memoir, narrative non-fiction | Premise / Character & voice / Scene & sensory grounding / Tension / Prose craft |
| `general` | notes, drafts, internal writeups | Purpose / Structure / Completeness / Clarity (unchanged — now a true catch-all, not the dumping ground) |

Classifier (`whetstone/_document_type.py`) grows to discriminate six types;
prompt and `DocumentType` Literal change in lockstep with `_SETS`. The
classifier remains one LLM call — same model, same shape.

Open question — six is a guess. Want a different cut (e.g. add `journalism`
for long-form feature writing; merge `essay` and `creative`)? See §6.

### 4.6 Panel mode

Panel mode is a different review *shape* — N expert biosketches generated
from extracted keywords, each expert reviews independently, a synthesis
merges them. It cannot be expressed as a criterion set.

Port as a separate graph (`v3/panel_graph.py`) that reuses v3's digest +
document model + criterion-cascade primitives. CLI: subcommand
`andamentum-whetstone panel <source> ...` (cleaner than a `--mode` flag,
and the panel surface has different required args anyway —
`--n-experts`, `--panel-disciplines`). Same `n_experts` /
`panel_disciplines` knobs as v2.

### 4.7 Editor phase

One additional node between `Synthesise` and `Finalize`, gated on
`editor=True`. Per-section LLM call emits `Edit`s (already a type in
`whetstone/schemas.py`); the DOCX renderer's Edit→DocumentPatch adapter
already exists. Off by default to keep the cost predictable.

### 4.8 Confidentiality tripwire

Port the marker scanner unchanged. Runs as a pre-flight check in
`review_document_v3` before any LLM call. `--confirm-own-draft` is the
attestation override. Same `ConfidentialityMarkerError` class, same error
code surface (exit 1).

## 5. Sequencing

**Note 2026-05-25:** the original 10-step sequence was revised after
deep structural research (10 agents) surfaced five real risks. The
runnable sequence now lives in
[`2026-05-25-whetstone-v3-consolidation-MASTER-PLAN.md`](./2026-05-25-whetstone-v3-consolidation-MASTER-PLAN.md)
as 9 actionable phases (A–I) + a benchmark phase (J):

| Phase | Maps to PRD steps | Why combined / moved |
|---|---|---|
| A | 1 + 2 | Step 1 alone is a partial fix — `criterion_set_for` already falls back to GENERAL |
| B | 3 | unchanged |
| C | 5 | moved up to consolidate the API surface before D/E touch it |
| D | 4 | unchanged |
| E | 6 | unchanged |
| F | (new) | CLI subcommand refactor — implicit prerequisite of G + H |
| G | 7 | starts with mini-PRD; elevated risk per risk synthesis |
| H | 8 | becomes mostly subcommand work (proofread is already a separate CLI) |
| I | 9 + 10 | must be atomic — module cannot exist in a half-renamed state |
| J | (new) | benchmark + PR summary |

Each phase is a separately reviewable commit. Phases A–H are additive
and could individually ship even if later phases stall; Phase I is the
atomic v2-deletion + rename that finishes the consolidation.

## 6. Open questions

- **Six criterion categories — right cut?** *(awaiting decision)* Want
  `journalism` as its own type? Merge `essay` + `creative`? See §4.5.
- **`review_document` API breakage** — when we rename `v3.review_document_v3`
  → `whetstone.review_document`, any downstream code in the repo importing
  v2's `review_document` breaks. Worth grepping for callers before step 10.
- **Benchmark gating on v2 deletion** — *(deferred to step 9)* — decide
  then whether to re-run the v2/v3 head-to-head and gate deletion on v3
  winning, or delete on architectural grounds and record the new baseline.
  See [[run-benchmark-before-done]].

### Settled

- **Convenience subcommand** — `andamentum-whetstone proofread <source>`
  ships. Pipelines stay disjoint. (§4.4)
- **Novelty target cap** — default `≤8`, `--novelty-target-cap` override.
  (§4.3)
- **Panel mode CLI shape** — subcommand
  `andamentum-whetstone panel <source> ...`, no `--mode` flag. (§4.6)

## 7. Success criteria

- One CLI entry (`andamentum-whetstone`) with no `--v3` flag.
- One Python entry (`andamentum.whetstone.review_document`) with no version
  qualifier.
- All v2 features in the "Port" rows of §3 reach feature parity in v3.
- Auto-routing regression closed.
- Essay / tutorial / creative inputs produce coherent, non-forced findings
  on smoke runs.
- Project test suite stays green throughout (each commit), with the
  v2-test-deletion commit explicitly justifying the line-count drop.
- Benchmark suite re-run on the consolidated v3 — no quality regression vs.
  v2 on the academic arm; new arms added for essay / tutorial / creative.
