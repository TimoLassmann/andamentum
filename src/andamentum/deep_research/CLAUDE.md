# andamentum.deep_research — notes for Claude

Moved from the root `CLAUDE.md`, word for word. Claude Code loads this file when it reads files in this directory.

## Overview

- `andamentum.deep_research` — web research pipeline (search → fetch → extract → verify → synthesize) built on `pydantic-graph`
- `andamentum.deep_research.v2` — the evidence-first engine (web and scholarly sources, a research store, a baseline that screens with one generic decision-model relevance question per candidate, with query-dependent screening criteria kept as the `criteria` preset, frozen-input replays for experiments, findings with code-checked quotes, a verified report): `await v2.research(question, *, mode, model, config=PRESETS['baseline'], frozen=None, ...)`, `andamentum-research v2 ...`, `andamentum-research mcp ...`. Its notes are in `v2/CLAUDE.md`. `v2.run_novelty_check_v2` is the v2 novelty check (scholarly sources, a decision model judging passages; whetstone `--novelty-engine v2`). The legacy pipeline below stays for `epistemic`.

## Design

**Deep research pipeline** is a `pydantic-graph` state machine. `state.py` holds `ResearchState`; `graph.py` / `nodes.py` define the nodes; `orchestrator.py` / `runner.py` drive execution. `searxng.py` manages the local SearxNG instance and `circuit_breaker.py` wraps it. Content extraction (`content_extractor.py`) uses `trafilatura` for HTML and `docling` for PDF. Search-query production runs through a per-slot generate→verify loop (`PrepareSearchCycle` → `GenerateOne` ⇄ `Verify` → `ParallelSearch`): the `query_generator` agent emits one query at a time, the `topic_verifier` agent judges it against the goal, rejected queries get up to `MAX_SLOT_RETRIES = 3` retries with verifier feedback before skip-and-tighten kicks in. The previous regex+stopword "topic guard" in `text_utils.py` has been retired.

## Topic verifier with a decision model

`run_research(..., decision_model=<opened DecisionModel>)` and the CLI flag `--decision-model NAME` replace the `topic_verifier` agent call in `verify_query` with one yes/no question (`verify_query.QUESTION`, noul, p >= 0.5 accepts); a rejection gives the generator the fixed `REJECT_FEEDBACK`. Without it the agent call is unchanged (pinned by `tests/test_topic_gate.py`); there is no calibration, threshold or escalation.

## Reliability rules (R1)

These hold for the legacy pipeline that `epistemic` still calls; pinned by `tests/test_reliability_*.py`.

- **A failed search raises.** `HttpxSearchBackend.search` raises `SearchBackendError` (transport, HTTP status, unreadable reply) or `CircuitOpenError`; `[]` only ever means "the search ran and found nothing". `run_searches` records the error and the stamp in `synthesize_report` marks the run degraded.
- **The circuit breaker belongs to one backend instance.** There is no process-global breaker; `run_research` builds a backend per run, so one run's failures never make the next skip its searches.
- **A failed summary is a `SummaryError`, never a placeholder summary.** `summarize_pages` returns a `SummarizeOutcome(summaries, errors)`; the errors land in `state.summary_errors`, count toward the degraded stamp, and appear in `ResearchErrors.summary_errors`. `SummarizePages` sends only pages not yet summarised or failed, so a run does not re-summarise its whole pool each cycle.
- **Novelty never guesses.** A failed assessment, a run with no summarised page, and a "no prior work" verdict from a degraded run all give `is_novel=None`. A "prior work exists" verdict from a degraded run still stands (it is positive evidence).
- **Ambient state is out of the library.** `load_dotenv()` runs only in `cli.main`. `searxng.py` reads no environment variables; `SearxngManager` takes a required `state_dir` and creates nothing until asked. `run_research` never starts a container: `andamentum-research --start-searxng --searxng-state-dir DIR` does, and fails loudly when it cannot.
