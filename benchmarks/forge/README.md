# Forge benchmark

Does a natural-language brief become the **right shape** of agentic system? This benchmark
drives `andamentum.forge` over a small corpus of briefs and scores each on whether forge
reaches the expected verdict — `build` or `refuse` — and, for buildable briefs, whether the
designed `SystemSpec` exhibits the control-flow grammar the brief calls for.

The corpus is eight cases (`cases.py`): five buildable briefs, one per grammar (sequence,
branch, loop, fan-out, stateful), plus three out-of-scope briefs forge must refuse at the
fitness gate (an app, an agent, a service). Refusable cases carry a `note` describing the
rung-1/2 function hiding inside them.

## The two tiers

- **Tier 1 (default)** — *design-only*. `run_forge(brief, model=..., dest=None)` runs
  understand → frame → decompose → compile → review and returns a `ForgeResult` whose
  `.spec` is inspected (`shape.detect_features`, purely structural — no model). A
  `ValueError` is forge refusing at the fitness gate (or failing the coherence loop), which
  the benchmark scores as a refusal. Fast, no sandbox, no generated-code execution.
- **Tier 2 (`--full`)** — *end-to-end build + sandbox audit*. A documented hook only; this
  pass does **not** implement it (it currently behaves like Tier 1, see the TODO in
  `runner.py`). When implemented it will render, agent-author, and sandbox-audit the package
  and score on whether it actually works.

## Running

Forge requires an explicit model — there is **no default and no env-var fallback**.

```bash
# Pytest mode — loose per-case floor (pass_rate >= 0.5) gates the run.
# The `benchmark` marker is deselected by default; pass --forge-bench-model to run live.
uv run pytest benchmarks/forge -m benchmark --forge-bench-model <model-id> -v

# Standalone CLI — prints a markdown report (and writes one with --output).
uv run python -m benchmarks.forge.cli --model <model-id>
uv run python -m benchmarks.forge.cli --model <model-id> --runs 5 --output report.md

# One subset of cases (substring match on the brief).
uv run python -m benchmarks.forge.cli --model <model-id> --case loop --case branch

# Tier-2 hook (currently == Tier 1).
uv run python -m benchmarks.forge.cli --model <model-id> --full
```

### Offline self-tests (no model, no network)

The harness is proven run-ready by self-tests that drive the *real* forge graph with a stub
`AgentSink` — zero model calls. They are **not** marked `benchmark`, so they run on an
explicit path:

```bash
uv run pytest benchmarks/forge -m "not benchmark" -q
```

`test_shape.py` builds real `SystemSpec`s (via `compile_spec` over hand-built `DesignPlan`s)
and asserts `detect_features`; `test_runner_offline.py` drives the full `run_case` +
scoring path with a scripted sink for both a build and a refuse case.

## Adding a case

Append a `Case` to `CASES` in `cases.py`:

```python
Case(
    brief="…",                # the natural-language request
    expected="build",          # "build" | "refuse"
    grammar="branch",          # "sequence" | "branch" | "loop" | "fanout" | "stateful" | "none"
    note="…",                  # optional: the reshape hint for a refusable brief
)
```

The grammar maps to a structural feature the design must show (`shape._GRAMMAR_FEATURE`):
`sequence`/`none` → no structural feature; `branch` → a decision or multi-way successor;
`loop` → a declared loop cap; `fanout` → one written State field read by ≥2 nodes;
`stateful` → a declared entity. For a `refuse` case only the verdict is checked. New cases
are picked up automatically by both the CLI and the live pytest run.

## Results are gitignored

Generated reports (`*.md` written by `--output`, anything under `results/`) are ignored —
see `.gitignore`. This `README.md` is tracked.
