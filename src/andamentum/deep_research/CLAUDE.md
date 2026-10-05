# andamentum.deep_research — notes for Claude

Moved from the root `CLAUDE.md`, word for word. Claude Code loads this file when it reads files in this directory.

## Overview

- `andamentum.deep_research` — web research pipeline (search → fetch → extract → verify → synthesize) built on `pydantic-graph`

## Design

**Deep research pipeline** is a `pydantic-graph` state machine. `state.py` holds `ResearchState`; `graph.py` / `nodes.py` define the nodes; `orchestrator.py` / `runner.py` drive execution. `searxng.py` manages the local SearxNG instance and `circuit_breaker.py` wraps it. Content extraction (`content_extractor.py`) uses `trafilatura` for HTML and `docling` for PDF. Search-query production runs through a per-slot generate→verify loop (`PrepareSearchCycle` → `GenerateOne` ⇄ `Verify` → `ParallelSearch`): the `query_generator` agent emits one query at a time, the `topic_verifier` agent judges it against the goal, rejected queries get up to `MAX_SLOT_RETRIES = 3` retries with verifier feedback before skip-and-tighten kicks in. The previous regex+stopword "topic guard" in `text_utils.py` has been retired.

## Topic verifier with a decision model

`run_research(..., decision_model=<opened DecisionModel>)` and the CLI flag `--decision-model NAME` replace the `topic_verifier` agent call in `verify_query` with one yes/no question (`verify_query.QUESTION`, noul, p >= 0.5 accepts); a rejection gives the generator the fixed `REJECT_FEEDBACK`. Without it the agent call is unchanged (pinned by `tests/test_topic_gate.py`); there is no calibration, threshold or escalation.
