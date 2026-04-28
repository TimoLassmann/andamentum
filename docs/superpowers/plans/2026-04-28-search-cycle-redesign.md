# Search Cycle Redesign — per-slot generate / verify / search

## Motivation

The current `SearchPhase` node in `deep_research/nodes.py` does three jobs in
one: it generates 2–3 search queries via the `search_planner` agent, runs a
**regex-based** topic-drift guard (`guard_query_against_goal` in
`deep_research/text_utils.py`) on the queries, then issues the searches in
parallel.

Two problems with that:

1. The regex guard is fragile — it tokenises with `[A-Za-z0-9]+` (which
   splits `half-life` into `half` and `life`), filters via a stopword list
   that drops legitimate research vocabulary (`effect`, `cause`, `result`),
   and gets fooled by synonyms/paraphrases. When it triggers a "repair" the
   warning is silently dropped because the call site at `nodes.py:123`
   doesn't pass a logger. So drifted-query repairs corrupt search behaviour
   *without leaving evidence in logs*.
2. The whole batch is generated in one LLM call. If only one of three
   queries is bad, fixing it requires regenerating the entire batch via
   `ModelRetry`, which loses the good queries to cosmetic instability.

We want to replace this with a per-slot generate-and-verify loop:

- One LLM call generates one query at a time, given the goal and any
  already-validated queries.
- A separate LLM verifier judges each query against the goal.
- Rejected queries trigger a bounded retry within the slot, with verifier
  feedback fed back to the generator.
- When the slot retry budget exhausts, we skip the slot and tighten the
  target count rather than spinning forever.

## Out of scope

- Replacing the `gap_analyzer`, `page_fetcher`, `page_summarizer`, or
  `lead_agent` agents.
- Touching the outer research-cycle loop (`SearchPhase → FetchPhase →
  SummarizePages → AnalyzeGaps → RefineSearch → SearchPhase`) — only the
  internals of the search-query-production phase change.
- Changing the public `run_research()` entry point or its signature.
- Anything in `epistemic`, `whetstone`, etc. — this is a `deep_research`
  internal refactor.

## Architecture

### Node graph after redesign

```
PlanResearch ─┐
              ├──→ PrepareSearchCycle  (pure Python — sets state.cycle.*)
RefineSearch ─┘             │
                            ↓
                       GenerateOne  ←──┐
                            │          │
                            ↓          │ (slot retry: feedback set)
                         Verify  ──────┤
                            │          │ (slot accepted: next slot)
                            │          │
                            ↓
                      ParallelSearch  (asyncio.gather, no LLM)
                            │
                            ↓
                        FetchPhase ─→ ...
```

Replaces the monolithic `SearchPhase`. Same external boundary
(PlanResearch/RefineSearch enter, FetchPhase exits). The cycle internals are
fully self-contained.

### State additions

Wrap cycle-scoped state in a small dataclass on `ResearchState`:

```python
# deep_research/state.py
@dataclass
class SearchCycleState:
    mode: Literal["initial", "gap"] = "initial"
    target_count: int = 0
    gaps: list[str] = field(default_factory=list)
    validated_queries: list[str] = field(default_factory=list)
    slot_attempts: int = 0

# new field on ResearchState
cycle: SearchCycleState = field(default_factory=SearchCycleState)
```

`PrepareSearchCycle` reinitialises `state.cycle` on entry so every search
cycle starts clean. Module-level constant `MAX_SLOT_RETRIES = 3` lives
alongside the node definitions.

### Bounding contract

Three layered guards — none new beyond `slot_attempts`:

| Where | Bounds | Cap |
|---|---|---|
| `state.cycle.slot_attempts` | retries on a single slot | `MAX_SLOT_RETRIES = 3` |
| `state.cycle.target_count` | slots in one cycle | starts at 3 (initial) or 2 (gap), decrements on slot exhaustion |
| `state.iteration_count` / `state.max_iterations` | whole-research cycles | already in code |

Worst case: `MAX_SLOT_RETRIES × target_count` LLM calls per cycle (18 for
initial mode), bounded by `max_iterations` for the whole run.

### Skip-and-tighten on slot exhaustion

When `slot_attempts >= MAX_SLOT_RETRIES` for a slot:

1. Reset `slot_attempts` to 0.
2. Decrement `target_count` by 1 (accept fewer queries this cycle).
3. If `len(validated_queries) >= target_count` → ParallelSearch.
4. Otherwise → next slot, fresh attempt.

When `target_count` hits 0 with `validated_queries` empty, ParallelSearch
runs with no queries, AnalyzeGaps gets no new evidence, and the outer
`max_iterations` cap takes over. Log a `WARNING` so degraded behaviour is
visible.

## File-by-file changes

### New files

1. **`deep_research/agents/query_generator.py`** — single-query generator
   agent definition. Output schema: `GeneratorOutput { query: str,
   rationale: str }`. Prompt accepts goal, validated queries so far, gaps
   (if any), feedback (if any), and is asked to produce ONE query.

2. **`deep_research/agents/topic_verifier.py`** — verifier agent
   definition. Output schema: `VerifierOutput { on_topic: bool, reason: str
   }`. Prompt accepts goal + query, returns yes/no plus a one-sentence
   reason.

3. **`deep_research/tests/test_search_loop_plumbing.py`** — Surface 1
   tests: deterministic stubs, no LLM, exercises every loop edge.

4. **`deep_research/tests/test_generator_diversity.py`** — Surface 2 tests
   (cloud-marked): real generator + always-reject verifier, captures
   generated queries to verify the generator explores instead of looping.

5. **`deep_research/tests/test_verifier_calibration.py`** — Surface 3 tests
   (cloud-marked): stub generator emitting curated corpus, real verifier,
   produces a confusion matrix per query category.

### Modified files

6. **`deep_research/state.py`** — add `SearchCycleState` dataclass and
   `ResearchState.cycle` field.

7. **`deep_research/nodes.py`** — delete `SearchPhase` (and its top-of-file
   `from .text_utils import guard_queries_against_drift` import). Add
   `PrepareSearchCycle`, `GenerateOne`, `Verify`, `ParallelSearch`. Wire
   `PlanResearch` and `RefineSearch` to return `PrepareSearchCycle()`
   instead of `SearchPhase()`.

8. **`deep_research/graph.py`** — replace `SearchPhase` in `nodes=[...]`
   with the four new nodes.

9. **`deep_research/agents/__init__.py`** — register the two new agents.
   Either delete the `search_planner` registration (clean break) or leave
   it dormant for one transition window. Plan: clean break — its callers
   are gone.

10. **`deep_research/agents/search.py`** — delete the legacy
    `search_planner` agent definition (and `SearchPlan` output model if
    only used here). Verify nothing else imports from this file before
    deletion; if so, leave the file with only the still-needed exports.

11. **`deep_research/text_utils.py`** — delete `guard_query_against_goal`,
    `guard_queries_against_drift`, `extract_anchor_terms`, `STOP_WORDS`,
    and the `__all__` entries for them. Keep the SSRF re-exports added in
    a previous commit.

12. **`deep_research/nodes.py`** — also support agent-injection for
    testing: extend `NodeDeps` with optional
    `agent_overrides: dict[str, Callable] | None = None`, and update
    `_build_agent` to honour it before going to the registry.

13. **`CLAUDE.md`** — update the deep_research description to reflect the
    new search-cycle architecture (one paragraph).

## Test strategy

Three test surfaces, each isolating a different failure mode:

### Surface 1 — Plumbing (no LLM, no network)

`test_search_loop_plumbing.py`. Deterministic stubs for both generator and
verifier. Drives the loop with scripted behaviour. Asserts state
transitions are correct. Covers:

- happy path: 3 queries pass first attempt
- one retry on slot 2, then accept
- slot exhaustion triggers skip-and-tighten
- total collapse: target tightens to 0, ParallelSearch runs with empty list
- gap mode (target_count=2) vs initial mode (target_count=3)

Runs in CI on every commit. Sub-second.

### Surface 2 — Generator diversity (real generator only)

`test_generator_diversity.py`, `@pytest.mark.cloud`. Real
`query_generator` agent, stub verifier. Run with `MAX_SLOT_RETRIES=10` and
always-reject. Captures every query, asserts:

- ≥6 distinct queries before falling back (no looping)
- bigram repetition rate < 30% (anti-cluster)
- when feedback specifies a missing element, query 2 onward addresses it

Reveals whether the generator loops, ignores feedback, or drifts.

### Surface 3 — Verifier calibration (real verifier only)

`test_verifier_calibration.py`, `@pytest.mark.cloud`. Curated corpus of
~30 (goal, query, ground_truth, category) tuples. Stub generator emits
queries from corpus; real verifier judges each. Aggregates into a
confusion matrix per category.

Categories covered: synonym substitution, mechanism-adjacent, wrong-drug,
shared-noun-but-off-topic, specialist jargon, vocabulary shift, multi-hop,
foreign-language (small subset).

Hard assertion: false-reject rate < 20% per category, false-accept rate
< 20% per category. Below either threshold → verifier prompt needs tuning.

### Stub-injection mechanism

Add `agent_overrides: dict[str, Callable] | None = None` to `NodeDeps`.
`_build_agent` checks the override map first:

```python
def _build_agent(name: str, model: Any, overrides: dict | None = None):
    if overrides and name in overrides:
        return overrides[name]
    from andamentum.core.agents import build_pydantic_ai_agent
    return build_pydantic_ai_agent(get_agent(name), model)
```

Tests pass `agent_overrides={"query_generator": stub, "topic_verifier":
stub}`. Production code never sets it. Zero test-only branches in
production paths.

## Acceptance criteria

- All existing deep_research tests still pass.
- Surface 1 plumbing tests cover the four scenarios above and pass.
- Surface 2 + Surface 3 tests are written and runnable; pass thresholds
  documented in the test file (failures on first run are diagnostic, not
  blocking — they tell us the prompts need tuning).
- A real `--quick` end-to-end run via `andamentum-research --query "..."`
  produces the same shape of output as today (an EvidenceReport).
- `pyright` and `ruff` stay at baseline (no new errors).
- `text_utils.py` no longer contains `guard_query_against_goal` or
  related regex code.
- CLAUDE.md reflects the new architecture in the deep_research bullet.

## Implementation phases (sub-tasks)

### Phase A — Plumbing (no LLM, no network)

1. Add `SearchCycleState` to `state.py`.
2. Add `agent_overrides` field to `NodeDeps` and update `_build_agent` in
   `nodes.py` to honour it.
3. Add `PrepareSearchCycle` node.
4. Add `GenerateOne` node — stubbed prompt that just returns a hardcoded
   query for now (real prompt comes in Phase B).
5. Add `Verify` node — stubbed prompt with hardcoded accept/reject.
6. Add `ParallelSearch` node by lifting the search-only logic out of the
   old `SearchPhase`.
7. Delete `SearchPhase`. Update `PlanResearch` and `RefineSearch` to
   return `PrepareSearchCycle()`.
8. Update `graph.py` node list.
9. Write Surface 1 plumbing tests using stubs. All pass.
10. Commit Phase A.

### Phase B — Real agents

11. Author `query_generator` agent with full prompt + `GeneratorOutput`
    schema. Register it.
12. Author `topic_verifier` agent with full prompt + `VerifierOutput`
    schema. Register it.
13. Wire real agents into `GenerateOne` and `Verify` (replace stub calls).
14. Delete `search_planner` agent + `SearchPlan` model.
15. Run a real `--quick` end-to-end query to confirm the pipeline works
    against real LLMs and produces a sensible EvidenceReport.
16. Commit Phase B.

### Phase C — Cognitive tests

17. Write Surface 2 generator-diversity tests with a real cloud model.
18. Write Surface 3 verifier-calibration tests with curated corpus.
19. Run both; capture confusion matrix; tune prompts if obvious bias
    appears (low-effort tuning only — major prompt engineering is a
    follow-up).
20. Commit Phase C.

### Phase D — Cleanup

21. Delete `guard_query_against_goal`, `guard_queries_against_drift`,
    `extract_anchor_terms`, `STOP_WORDS` from `text_utils.py`. Strip
    `__all__` entries.
22. Update CLAUDE.md.
23. Run full test suite + pyright + ruff. Verify baseline.
24. Commit Phase D.

## Open questions / decisions locked

- **Target count per mode**: 3 for initial, 2 for gap. Locked.
- **Skip-and-tighten on slot exhaustion**: yes (vs hard-fail). Locked.
- **Empty-validated-queries fallback**: ParallelSearch runs with empty
  list, log WARNING, let outer `max_iterations` clean up. Locked.
- **Verifier output schema**: `{on_topic, reason}` minimal. No
  `suggested_repair` — generator does the rewriting. Locked.
- **Verifier model**: cloud (`openai:gpt-5.4-nano` or equivalent), not
  local. Calibration matters more than latency for this role. Locked.
- **`max_round_turns` knob**: `MAX_SLOT_RETRIES = 3`. Module constant for
  now, exposed via state if needed later. Locked.
- **Transition strategy for `search_planner`**: clean break, deleted in
  same commit as new agents. Locked.
