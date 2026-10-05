# andamentum.core — notes for Claude

Moved from the root `CLAUDE.md`, word for word. Claude Code loads this file when it reads files in this directory.

## Overview

- `andamentum.core` — shared model-resolution, `AgentRunner`, and embedding infrastructure (`core.embeddings`) used by all sub-modules

## Decision models (`core.decisions`)

- `decide(state, questions, *, model="ollama:<name>")` calls Ollama's `POST /v1/systemone` (Ollama 0.35+, Jev-style decision models). Question types: `choice` (options with probabilities), `noul` (probability a condition holds), `score` (value over ordered options). The wire shape is confined to `_request_body` and `_parse_answers`; the real endpoint differs from the early spec sketch in three ways: `choice` criteria is an object option -> description, `noul` answers carry only `{noul: p}` (so `DecisionAnswer` derives yes/no probabilities and `confidence = max(p, 1-p)`), and `score` answers carry `score` and `legend` (we keep `score`).
- A malformed response raises `ValueError`; there is no fallback. One request per call, awaited; never gather local-model calls.
- Decision models order and escalate work, they never certify. `calibrate(pairs, model=, digest=, task=)` builds a `CalibrationRecord` (reliability bins, ECE, precision/recall at thresholds); `threshold_for(precision=)` returns None when no threshold qualifies (default `min_support=5`), and callers must then stay advisory. `model_digest` reads `digest` from `/api/tags` (verified against Ollama 0.35.1: `/api/show` carries none) and raises if the model is not listed.
- Synthetic calibration sets live in `compose/harness/bench/calibration/`.
- Live tests use the `ollama` marker (deselected by default).
