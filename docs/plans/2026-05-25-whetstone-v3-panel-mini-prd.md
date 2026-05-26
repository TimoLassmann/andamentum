# Whetstone v3 panel mode — mini-PRD (Phase G design checkpoint)

**Status:** Design. User review required before implementation.
**Author:** Claude, 2026-05-26.
**Parent plan:** `2026-05-25-whetstone-v3-consolidation-MASTER-PLAN.md`.
**Position in sequence:** Phase G — flagged as elevated-risk in the master
plan, "starts with a 10-minute mini-PRD; if the design surfaces
unresolved issues, halt and ask."

This document is the halt. Five design questions need your call before I
implement. The implementation itself, once decided, is straightforward.

---

## 1. What panel mode does (status quo, v2)

Panel mode simulates a journal peer-review panel:

```
HarvestSource → ChunkAndScan → ExtractKeywords → GenerateExpertPanel
              → ExpertReview → PanelSynthesise → End[ReviewResult]
```

- **ExtractKeywords** — one LLM call, returns 3-5 disciplines from the
  document content (or skipped if `--panel-disciplines` is supplied)
- **GenerateExpertPanel** — N parallel LLM calls (default N=4), each
  generates a fictional senior-expert biosketch (NIH-format) for one
  discipline
- **ExpertReview** — N parallel LLM calls, each expert reviews the
  document in role; emits per-expert scores (rigor / methodology /
  novelty / clarity, 1-10) + strengths + weaknesses + recommendation
  (Accept / Minor / Major / Reject)
- **PanelSynthesise** — one LLM call, aggregates the N reviews into a
  panel synthesis with averaged scores, consensus strengths/weaknesses,
  divergent opinions, key decision factors, 5-7 paragraph prose

Total cost: `2N + 2` LLM calls (default 10 at N=4).

Output populates three `ReviewResult` fields not used by review mode:
`expert_profiles`, `expert_reviews`, `panel_synthesis` (the
findings / edits / author_questions fields stay empty).

Renderer path: `renderers/_panel_layout.py` (markdown + HTML; docx
renderer prepends the panel synthesis to the patched output).

## 2. Five design questions

### Q1. Graph topology: separate file or shared graph?

**Option A (recommended): separate file** `v3/panel/graph.py` with its
own `panel_graph_v3 = Graph(nodes=[...])` and a public
`run_panel_v3(markdown, *, model, n_experts, panel_disciplines) ->
ReviewResult`. The main `run_review_v3` graph is untouched.

**Option B: shared graph with `mode="panel"` branching.** The first
node inspects state and dispatches to a panel sub-graph vs the
existing review chain. Adds complexity to the main graph for one
opt-in feature.

**Recommendation: A.** Two graphs with shared digest / sectionize
primitives, no `mode` parameter on `V3Deps`, no branching in the main
review chain. Panel mode is a different *shape* (multi-expert fan-out
+ synthesis) not a different *content* (different criterion set). Per
the project's "one code path" preference, the cleanest separation is at
the graph-entry level, not via a runtime branch.

### Q2. The 4 panel agents — copy or import from v2?

The 4 agents (`extract_keywords`, `expert_generator`, `expert_reviewer`,
`panel_synthesise`) live in `whetstone/agents/`. Their prompts and
output schemas are stable.

**Option A (recommended): copy verbatim into `v3/panel/agents/`.**
Phase I can then delete `whetstone/agents/` cleanly. ~200 lines of
near-mechanical copy + import-path updates.

**Option B: import from v2 path.** Blocks Phase I's `whetstone/agents/`
deletion — the import surface survives v2's other code.

**Recommendation: A.** Copy. Matches the choice made for the editor
agent (`v3/editor.py` copied v2's EDITOR_PROMPT verbatim) — same
rationale, same reliability.

### Q3. ReviewResult panel fields — keep or rework?

`ReviewResult.expert_profiles`, `expert_reviews`, `panel_synthesis` are
the three panel-specific fields. Renderers (`_panel_layout.py`) read
them directly.

**Option A (recommended): keep unchanged.** These fields already work,
the renderers work, the schemas are stable. Phase G's v3 panel graph
populates exactly the same fields, so renderer code is reused unchanged.

**Option B: introduce v3-native `PanelResult` type.** Cleaner separation
but requires forking the renderer path.

**Recommendation: A.** Less work, more reliable. The panel fields
become part of the v3 public surface — same approach v3 already takes
for `Finding`, `Edit`, `Quote`, `ReviewMetrics`.

### Q4. CLI subcommand wiring (post-Phase I)

Phase F shipped `panel <input>` as an alias that rewrites to
`<input> --mode panel`. That works *today* because v2's `--mode` flag
still exists. After Phase I deletes v2, `--mode` is gone and the
rewriter must instead route to `run_panel_v3(...)`.

The fix is mechanical (replace the rewrite with a direct dispatch into
the panel handler in `cli.py`). Worth flagging because it means the
Phase F rewrite shape changes during Phase I.

**No question here — just a coordination note.**

### Q5. Sequencing: do G before or after I?

This is the call I most want your input on.

**Option A (recommended): Phase G before Phase I.**
- G ports panel to v3 fully (separate graph + agents + tests)
- I then deletes v2 cleanly, knowing every feature has a v3 home
- I is the "victory lap" commit — pure deletion, no porting

**Option B: Phase I before Phase G — drop panel for now.**
- I deletes v2 including the panel mode path
- Panel mode is unavailable until G ships in a follow-up branch
- v3-only release ships sooner; panel mode is a known gap

**Option C: Skip Phase G entirely — drop panel mode permanently.**
- Panel mode goes away; users wanting multi-expert review use the
  custom-criteria path (Phase C) with manually-curated expert criterion
  sets
- Significant feature regression but matches the project's "general
  mechanisms over hard-coded domain rules" preference

**Recommendation: A.** Panel mode is non-trivial output (~10 LLM calls
producing a structured peer-review report — distinct enough that custom
criteria don't substitute for it). The port is mechanical (4 agents +
6 graph nodes; the agents already work, the renderers already work).
Roughly 2-3 hours of focused implementation + tests.

## 3. The implementation itself, if you approve Option A

Concretely (under "Option A" for all five questions):

1. **New: `v3/panel/__init__.py`** — empty package marker
2. **New: `v3/panel/agents.py`** — verbatim copies of the 4 v2 panel
   agents (prompts + Pydantic output schemas)
3. **New: `v3/panel/graph.py`** — new graph with 6 nodes mirroring
   v2's panel chain, but reusing `v3/sectionize.py` for the harvest +
   chunking substrate
4. **New: `v3/panel/__init__.py` re-exports** `run_panel_v3` (the
   public entry) so the CLI can `from .v3.panel import run_panel_v3`
5. **New: `v3/panel/tests/test_panel.py`** — 6-8 tests covering each
   node + an end-to-end smoke run with mocked agents
6. **Edit: `cli.py`** — the `panel` subcommand rewrites to direct
   dispatch into `run_panel_v3(...)` instead of `--mode panel`
   (this edit lands during Phase I, not Phase G)

Total file footprint: ~600 lines new code + ~200 lines tests.
Estimated runtime to implement + validate: 2-3 focused hours.

## 4. The halt — what I need from you

Please confirm or override:

- [ ] Q1 — graph topology: separate file (recommended) or shared graph?
- [ ] Q2 — panel agents: copy into v3/panel/agents.py (recommended) or
  import from v2?
- [ ] Q3 — ReviewResult fields: keep `expert_profiles` /
  `expert_reviews` / `panel_synthesis` unchanged (recommended) or
  introduce v3-native `PanelResult`?
- [ ] Q5 — sequencing: G before I (recommended), I before G with
  panel as a known gap, or drop panel mode entirely?

Once you confirm, I'll implement Phase G then Phase I. If you want me
to start either or both immediately based on the recommendations, say
so and I'll proceed with the recommended answers.
