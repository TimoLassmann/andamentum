# Integration Tests — Philosophical Pathway Verification

These are **not unit tests**. They run the full epistemic pipeline with real LLM calls
and web search, then verify that specific epistemological traditions actually fire.

They are excluded from pytest (`testpaths = ["tests"]` in pyproject.toml).

## Usage

```bash
# Run all integration tests (costs money, takes ~15 min)
uv run python packages/epistemic/integration_tests/run.py --model openai:gpt-5.4-mini

# Run a specific tradition only
uv run python packages/epistemic/integration_tests/run.py --tradition doyle --model openai:gpt-5.4-mini

# List available traditions
uv run python packages/epistemic/integration_tests/run.py --list

# Keep databases for manual inspection after run
uv run python packages/epistemic/integration_tests/run.py --model openai:gpt-5.4-mini --keep
```

## What each test verifies

| Test | Tradition | Key check |
|------|-----------|-----------|
| `doyle` | Doyle TMS | `revalidate_claim` operation fired, claim demoted |
| `peirce` | Peirce inquiry cycling | `investigate_claim` operation fired, scrutiny re-ran |
| `tetlock` | Tetlock predictions | `generate_prediction` fired (requires ROBUST stage) |
| `lipton` | Lipton contrastive | `contrastive_evaluation` fired (explanatory question) |
| `kahneman` | Kahneman independence | Per-evidence scrutiny produced diverse issues |
| `agm` | AGM minimal change | Demotion preserved evidence links |

## Adding a new test

Add an entry to `PATHWAY_TESTS` in `questions.py`. Each test defines:
- `question`: the research question
- `tradition`: which philosophical tradition it targets
- `expected_operations`: operations that MUST appear in the execution trace
- `db_checks`: functions that verify database state after the run
