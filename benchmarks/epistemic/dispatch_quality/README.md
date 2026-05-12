# Dispatch-quality benchmark

Phase 3 of the description-driven-dispatch refactor
(`docs/superpowers/plans/2026-05-12-description-driven-provider-dispatch.md`).

This benchmark measures **how well the new dispatch agent triages and
constructs queries** per provider, against curated in-domain and
out-of-domain claims that live on each provider class as `query_examples`.

## What it measures

For each provider in the registry, the harness:

1. Iterates over `provider.query_examples` — a list of `(claim, native_query)` pairs
   where `query` is `None` for claims the provider should abstain on.
2. Runs the new dispatch agent on each claim.
3. Records whether the agent's triage decision (commit vs abstain) matches
   the expected (in-domain → commit, out-of-domain → abstain).
4. Aggregates three per-provider metrics:

   | metric | definition |
   |---|---|
   | **hit rate** | fraction of in-domain claims where the agent committed at least one query |
   | **abstention accuracy** | fraction of out-of-domain claims where the agent correctly returned empty |
   | **triage accuracy** | combined: (correct commits + correct abstains) / total |

The harness does **NOT** call `provider.gather()` and does **NOT** score
relevance via the judge agent. Those are out of scope for Tier 1 — they
belong to Tier 2 (5-claim full-pipeline sanity check) which is run once
at the end of Phase 3.

## Why this is the right benchmark for Tier 1

The thing we're trying to validate is *retrieval routing*: does the
dispatch agent send the right claims to the right providers? Tier 1
measures exactly that, with per-provider attribution and at low cost
(one LLM call per claim per provider, no HTTP calls, no judge calls).

Per-provider attribution matters because the iteration loop is
"description → dispatch outcome → tweak description → re-run." If a
provider's triage is off, you iterate on *that provider's* description
and `query_examples`, then re-run *that provider's* benchmark slice
(seconds, not minutes).

## How to run

```bash
# Default: every registered provider, model from $ANDAMENTUM_MAIN_LLM_MODEL
uv run python -m benchmarks.epistemic.dispatch_quality.run

# Specific model
uv run python -m benchmarks.epistemic.dispatch_quality.run \
    --model openai:gpt-5.4-nano

# Subset of providers (iteration on a single provider's prompt or examples)
uv run python -m benchmarks.epistemic.dispatch_quality.run \
    --providers pubmed,arxiv,clinicaltrials

# Held-out evaluation: hide each tested example from the agent's
# in-context teaching during its own evaluation. Stricter generalisation test.
uv run python -m benchmarks.epistemic.dispatch_quality.run --held-out

# Custom triage-accuracy threshold (default 0.80 per PRD §6 Phase 3)
uv run python -m benchmarks.epistemic.dispatch_quality.run --threshold 0.90
```

## Output

Each run writes to `benchmarks/epistemic/results/dispatch_quality/<timestamp>/`:

- `tier_one_summary.md` — per-provider metrics table
- `tier_one_failures.md` — detailed per-claim diagnostics for failed triage decisions
- `tier_one_results.json` — full per-claim outcomes (diff-able across runs)

The CLI also prints the summary to stdout and exits with code 1 if any
provider falls below the triage-accuracy threshold — so it's CI-suitable.

## Cost estimate

For 10 providers × ~9 examples each ≈ **90 dispatch agent calls per run**.
At a small-model price tier (e.g., `openai:gpt-5.4-nano`), that's
fractional cents per run and runs in under a minute. The benchmark is
designed to be the fast iteration loop — running it dozens of times
during prompt + description tuning is the intended workflow.

## Acceptance per PRD §6 Phase 3

> For each provider, the new dispatch matches or beats legacy on
> relevance rate and hit rate on in-domain claims. Abstention accuracy
> on out-of-domain claims is ≥ 80%. If a provider fails on relevance
> rate, fix the description/prompt and re-run that provider's benchmark
> slice.

This harness covers the **hit rate** and **abstention accuracy** parts of
that criterion. Relevance rate requires calling `provider.gather()` and
scoring returned evidence with the judge agent — that's a separate
extension to add when we want it.

## Iteration workflow

1. Run the benchmark on all providers.
2. Open `tier_one_failures.md` — it lists every miss with the agent's
   reasoning per claim. Each miss points at one of:
   - **Description gap**: the agent didn't read the provider's scope
     correctly. → tweak `provider.description`.
   - **Example gap**: the agent guessed at syntax not demonstrated by
     `query_examples`. → add a representative example covering that style.
   - **Borderline example**: the claim is genuinely ambiguous. → split
     it into clearer in-domain or out-of-domain.
3. Edit `src/andamentum/epistemic/providers/<name>.py`.
4. Re-run for that provider: `--providers <name>`.
5. Loop until triage accuracy clears 0.80.

## What's intentionally not here

- **Tier 1.5** (abstention-pattern stability vs legacy across the dev30
  corpus): adds value once we want to confirm new dispatch matches
  legacy's "I returned nothing" pattern on real claims. Not part of
  this harness because it needs read access to a v5-style result set.
- **Tier 2** (5-claim full-pipeline sanity): standalone fixture, run
  once at the end of Phase 3. Lives elsewhere in the benchmark tree.
- **Live `gather()` + judge-relevance scoring**: would extend Tier 1
  with HTTP calls and judge calls. Useful but expensive and slow —
  not the right shape for the fast iteration loop.

Each of these is a natural extension to this harness if the per-provider
hit-rate + abstention-accuracy metrics turn out to be insufficient.
