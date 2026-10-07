# andamentum.chunker — notes for Claude

Moved from the root `CLAUDE.md`, word for word. Claude Code loads this file when it reads files in this directory.

## Overview

- `andamentum.chunker` — structural-first semantic chunking of long markdown into 2k–10k char self-contained units. Stage 1: split at markdown headings. Stage 2: embedding-based split for over-budget sections. Stage 3 (optional): small-LLM judge for grey-zone boundaries. The LLM is never the primary segmenter.

## Layering

- `chunker` depends only on `core` (for `AgentRunner`, model resolution, `core.decisions`) and `rapidfuzz` (for tiered anchor matching). MUST NOT depend on `epistemic`, `deep_research`, `document_store`, `whetstone`, `compose`, `figures`, or `typeset`. Other modules MAY depend on `chunker` (e.g. whetstone for section-by-section review on huge documents, document_store for embedding-quality chunks).

## Design

**Chunker module** (`andamentum.chunker`) — structural-first semantic chunker. Single entry point: `extract_units(source, *, target_min_chars=2000, target_max_chars=10000, embedding_fn=None, judge_executor=None, domain="general") -> ChunkingResult`. Pipeline (literature consensus, 2026): **Stage 1** — split at markdown headings (`structural.py`); **Stage 2** — for sections >`target_max_chars`, semantic-split at paragraph boundaries chosen by largest cosine drops between adjacent paragraph embeddings (`semantic_split.py`, defaults to local Ollama `embeddinggemma-2:latest` via `embeddings.py`); **Stage 3** (optional) — small LLM judge (`judge.py`) for cuts whose drop-percentile is in the grey zone (60–90th by default), answers `keep | merge`. The LLM is never the primary segmenter. Each unit's `text` is byte-identical to a source span. The legacy agentic `windowing.py`/`refinement.py`/`NextUnitResult` machinery has been removed; legacy kwargs (`primary_executor`, `window_size`, `lookahead`, `extension_chars`, `max_iterations`) are accepted on `extract_units` for backward compat but ignored. Stage 3 runs only when `judge_executor` is passed explicitly; a legacy `primary_executor` is not promoted to the judge. The CLI follows the same rule: `andamentum-chunker` makes no LLM call unless `--judge-model MODEL` is given (there is no `--model` and no `$ANDAMENTUM_MAIN_LLM_MODEL` fallback); embeddings for stage 2 are a local Ollama call either way.

## Stage-3 decision model

`extract_units(..., decision_model=None)` and `andamentum-chunker --decision-model NAME` replace the stage-3 LLM judge with a local decision model (`core.decisions`, Ollama only). It is asked `judge.MERGE_QUESTION` (noul, "do LEFT and RIGHT continue the same topic?") about the same two 400-character windows, and `p >= 0.5` merges; a failed call keeps the cut. With `decision_model=None` stage 3 is exactly the LLM judge, and `ChunkingResult.decision_calls` counts the cuts the decision model decided.
