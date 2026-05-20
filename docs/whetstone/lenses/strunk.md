# Strunk lens

A whetstone lens that applies rules from William Strunk's *Elements of
Style* to your draft. Unlike the seven persona lenses (`rigorous`,
`writer`, `methodology`, `statistician`, `consistency`, `overclaim`,
`claim_evidence`), the Strunk lens is **rule-based**: each Strunk
rule is implemented by its own narrow agent — or, where the rule has
a mechanical test, by a deterministic regex / dictionary check.

The lens is built as a pydantic-graph sub-graph rather than a single
prompt. **One LLM call per rule per section** — not per sentence.
Whetstone already chunks the document into 2–10 KB sections via
`andamentum.chunker`, and that is the natural unit for each rule.
Each agent sees the section in context and returns its full list of
violations in one go.

## Phase A — what's implemented

| Rule | Source | NodeKind | What it checks |
|---|---|---|---|
| **R2** Series comma | Strunk Ch II §2 | `deterministic` | In a series of 3+ items with a single conjunction, use a comma after each term except the last. Regex over `section.text`. |
| **R11** Active voice | Strunk Ch III §11 | `agent` | Returns a list of passive-voice violations in the section with active-voice rewrites. |
| **R13** Omit needless words | Strunk Ch III §13 | `agent` | Returns a list of needless-words violations classified into a closed-set category (throat-clearing, redundancy, weak-qualifier, filler-prepositional, other). |

Each agent's output is a list (`ActiveVoiceReport.violations`,
`OmitNeedlessWordsReport.violations`). The empty list IS the "no
violations found" answer — no per-violation yes/no/unsure verdict.

Phase B/C add the remaining 12 numbered rules plus Chapter V (`Words
and Expressions Commonly Misused`, ~180-entry dictionary) and Chapter
VI (common misspellings).

## Graph topology

```
DeterministicScreen  [deterministic]
   │   reads:  section
   │   writes: findings              (Phase A: R2 only)
   ▼
R11ActiveVoice  [agent]              one LLM call, returns list of violations
   │   reads:  section
   │   writes: findings, demands
   ▼
R13OmitNeedlessWords  [agent]        one LLM call, returns list of violations
   │   reads:  section
   │   writes: findings, demands
   ▼
ResolveDemands  [control]            Phase A: no-op (pass-through)
   │   reads:  demands                Phase 4: re-runs ambiguous cases
   │   writes: findings                       on a stronger model
   ▼
Aggregate  [control]
   │   reads:  findings, section
   │   writes: (returns to caller)
   ▼
End[list[Finding]]
```

Concurrency at the whetstone level is provided by `CriticalRead`,
which already fans out sections × lenses in parallel (bounded by
`_MAX_CONCURRENT = 4`). Each section runs the full sub-graph
sequentially; cross-section parallelism is what makes the whole
review fast.

The topology is introspectable as a Python value:
`andamentum.whetstone.lenses.strunk.topology.topology()` returns a
`dict[str, dict[str, Any]]` with one entry per node. The structural
tests in `lenses/strunk/tests/test_topology.py` assert reachability,
the linear chain, no orphan successors, and that exactly one node
(`Aggregate`) returns `End`.

## NodeKind discipline

Three kinds, declared as `ClassVar[NodeKind]` on every node:

* **`deterministic`** — pure function. No LLM call, no I/O. Output is
  a function of input.
* **`agent`** — invokes an LLM through the `AgentExecutor` in
  `StrunkLensDeps`. Declares `model`, `output_model`, `rule_number`,
  `rule_source` ClassVars.
* **`control`** — graph plumbing (aggregate, demand routing).

A static AST test (`test_node_kinds.py::test_deterministic_nodes_do_not_import_agent_machinery`)
walks the source of every `DETERMINISTIC` node and asserts none of
them import `andamentum.core.agents`, `AgentRunner`, `AgentDefinition`,
or `build_pydantic_ai_agent`. A complementary test asserts every
`AGENT` node *does* import `AgentDefinition`.

## Anchoring

Each LLM-returned violation carries a `span` field — the verbatim
substring of the section text that exhibits the violation. Before a
finding is emitted, that span is matched back against `section.text`
via `andamentum.chunker.validation.find_anchor` (the same matcher
the rest of whetstone uses). Unanchorable spans — paraphrases,
hallucinations, broken quotes — are dropped silently. The persisted
`StrunkFinding.span_text` is always the *source* slice, not the
model's input.

## Demand-routed escalation (Phase 4)

An agent node may emit a `StrunkDemand` if its LLM call:

| Trigger | `StrunkDemand.reason` |
|---|---|
| Pydantic validation fails / returns wrong type | `"schema_validation_failed"` |
| Executor raises (network error, model crash) | `"executor_exception"` |

Phase A records demands but does not consume them — `ResolveDemands`
is a no-op pass-through. Phase 4 will re-run flagged sections on a
stronger model.

## Public API

```python
from andamentum.whetstone.lenses.strunk import run_strunk_lens
from andamentum.whetstone.lenses.strunk.state import StrunkLensDeps
from andamentum.core.agents import AgentRunner

runner = AgentRunner(model="ollama:gemma3:4b-it-q4_K_M")
deps = StrunkLensDeps(executor=runner)
findings = await run_strunk_lens(section, deps=deps)
```

The lens is also wired into the whetstone main pipeline. The CLI
accepts it via `--perspectives strunk` (or in combination, e.g.
`--perspectives rigorous,strunk`).

## Tests

55 tests in `lenses/strunk/tests/`, all run in the default suite
(none require a live LLM):

| Layer | File | Count | Purpose |
|---|---|---|---|
| R2 deterministic | `test_deterministic_screen.py` | 10 | Regex hits + Oxford insertion + multi-violation |
| Aggregate | `test_aggregate.py` | 7 | Conversion to `whetstone.Finding` |
| Node-kind discipline | `test_node_kinds.py` | 6 | AST-level static enforcement |
| Topology | `test_topology.py` | 8 | Reachability + linear chain + agent metadata |
| R11 agent | `test_r11_active_voice.py` | 9 | Anchor → finding, mock-LLM, exception handling |
| R13 agent | `test_r13_omit_needless.py` | 9 | Anchor → finding (each category), mock-LLM, exception handling |
| End-to-end | `test_integration.py` | 6 | Full sub-graph through `run_strunk_lens` — pins "one LLM call per rule per section" |

A live `@pytest.mark.ollama` smoke test
(`test_integration_ollama.py`) runs the same fixture against a real
local model. It's deselected by default — run with
`uv run pytest -m ollama src/andamentum/whetstone/lenses/strunk/`.

## Performance

On an N-section document with 2 LLM-backed rules, the lens makes
**2N LLM calls** — independent of how many sentences each section
contains. For typical research drafts (5–10 sections) that's 10–20
calls; for the full *Elements of Style* book (18 sections) it's 36
calls.

If a section is small enough that 2N is still slow on a heavy local
model, consider:

* Using a smaller model — `gemma3:4b-it-q4_K_M` is the Phase A
  default and handles these flat list schemas well.
* Per-rule overrides via `StrunkLensDeps.model_for_rule[rule_num]`.
* `ANDAMENTUM_LLM_CONCURRENCY=4` to let Ollama serve concurrent
  requests (bounded by GPU memory).

## Roadmap beyond Phase A

* **Phase 1** — calibration: per-rule fixtures (section, expected
  violations list), sweep `(rule, model)` pairs, pin a model per rule.
* **Phase 2** — deterministic Chapter V dictionary (~180 misused-word
  entries) + R1 (possessive `'s`) + R5 (comma splice) regex screens.
* **Phase 3** — remaining LLM rules: R7, R9, R10, R12, R14, R15,
  R16, R17, R18 + their fixtures.
* **Phase 4** — wire `ResolveDemands` to escalate
  schema-failed / executor-failed cases on a stronger model.
* **Phase 5** — `pytest -m benchmark` precision/recall on a held-out
  fixture set.
