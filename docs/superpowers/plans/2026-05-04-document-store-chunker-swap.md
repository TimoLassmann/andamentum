# Document Store — Swap to `andamentum.chunker`

**Date:** 2026-05-04
**Author:** session with Claude
**Status:** plan, not yet started

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax. Each phase has a verification gate (pyright + ruff + pytest); do not advance to the next phase if any gate fails. The plan is deliberately conservative — adapter and feature-flag first, default flip only after a behaviour-bench passes.

**Goal:** Replace `document_store/chunking.py` with `andamentum.chunker.extract_units` so the package has one canonical chunking implementation, with measurably better boundaries on long documents and lower per-ingest LLM cost.

**Constraint:** Reuse existing pieces. No schema changes. No changes to `andamentum.chunker` itself. Backwards compatible at the storage level (existing chunks stay searchable; new chunks may have different sizes; both coexist in the same DB without trouble). Ollama remains the only external runtime requirement (already true for chunk embeddings).

**Tech stack:** Python 3.13, asyncio, Ollama with `embeddinggemma:latest`. The `andamentum.chunker` and `andamentum.core.embeddings` modules already in tree. No new dependencies.

**Non-goals:**

- Stage-3 LLM judge inside the chunker (we won't enable it; ingest is already LLM-heavy)
- Adding overlap to `andamentum.chunker` (different mitigation if needed: `prev_chunk_id` / `next_chunk_id` columns; separate follow-up plan)
- Modifying `andamentum.chunker` or its public API
- Re-chunking existing documents in user databases (heterogeneous chunk sizes per DB are fine; new chunks come on next ingest or `repair()`)
- Changing `extract_chunk_metadata` agent or chunk-level metadata schema
- Changing the four-signal RRF search behaviour
- Promoting `chunks.py` / `chunks_search.py` from internal to public API (they stay internal)

---

## The principle in one sentence

**One chunker per package — the structural-first one we already invested in — sized for document_store's metadata-extraction use case, with `section_path` recovered through the chunker's own structural API.**

---

## Why swap

Three things drive this, in order of weight:

1. **Two chunkers in the package, one of them duplicating shared infrastructure.** `core/embeddings.py:chunk_text` already encodes the `2000-char / 200-overlap` convention with the explicit comment *"Aligned with document_store's 500-token / 4-chars-per-token chunking"*. `document_store/chunking.py` is essentially a markdown-aware re-implementation of that helper. The convention is centralised; the implementation has drifted.

2. **Better boundaries on long documents.** The simple chunker windows long sections at arbitrary char offsets (`chunk_markdown` defaults to `max_tokens=500, overlap_tokens=50`). The structural-first chunker splits at the largest cosine drops between paragraph embeddings — a semantic boundary, not a byte boundary. For papers, articles, and any prose longer than 2k chars, this directly improves retrieval quality because chunks no longer cut mid-argument.

3. **Existing in-package adopter de-risks the move.** `whetstone/nodes/chunk_and_scan.py:25` already imports and uses `extract_units` with the same defaults this plan calls for (modulo `target_max_chars`). This is the second adoption, not the first.

Secondary motivation: per-ingest cost. The simple chunker emits ~15 chunks for a 30k-char paper (sliding-window with overlap inside long sections); the structural-first chunker emits ~6 (one per top-level section, semantic split for any section >`target_max`). Each chunk costs one embed call plus one `extract_chunk_metadata` LLM call, so dropping ~9 chunks per paper drops ~9 LLM calls. For short notes (the majority of personal-KB inputs), both produce 1 chunk and the cost difference is zero.

---

## The five load-bearing decisions

### D1 — Use `target_max_chars=4000`, not the chunker's default 10000

`extract_chunk_metadata` asks for `topics: 2-3 specific tags`, `has_decision: bool`, `has_action_item: bool`. These were tuned for ~2k-char chunks. At 10k, a chunk can span 3-4 subtopics, which dilutes `topics` extraction and forces the boolean flags into a meaningless OR over the whole region.

`4000` is the compromise: chunks remain "paragraph-of-paragraphs" sized so metadata stays signal-rich, while long documents still get fewer chunks than the simple chunker's ~2000-char windowing produced. `target_min_chars=1500` matches.

This is *the document_store's* sizing choice. Whetstone keeps its own (`10000`) because review use-case wants larger sections. Different consumers, different sizing — that is what the parameter is for.

### D2 — Recover `section_path` via the chunker's structural API

The simple chunker tracks the markdown heading breadcrumb (`"Methods > ODE Solver"`) per chunk and stores it on `ChunkMetadataFields.section_path`. The andamentum chunker's `Unit.title` is just the immediate heading — losing parent context is a real regression for retrieval display.

The fix uses the chunker's already-public structural API:

```python
from andamentum.chunker.structural import find_headings, build_section_tree
```

For each unit, walk the section tree to find the deepest section whose `(start, end)` span contains `unit.source_start`, then build the path by walking parents up. ~30 LOC adapter, deterministic, zero new dependencies, no changes to the chunker itself.

### D3 — No stage-3 LLM judge

`_run_phase2` already issues N+1 LLM calls per ingest (N chunk-metadata extractions plus one document-metadata extraction). The judge would add 1-2 more LLM calls for long documents in exchange for boundary-quality improvements that are already marginal once stage-2 has run. Not worth the wall-clock cost for the personal-KB use case. Stage 3 stays available for future use; we just don't pass `judge_executor` from document_store.

### D4 — One embedder instance per ingest call

Currently `_run_phase2` constructs an `EmbeddingService` for the chunk-embedding loop. The chunker also needs an embedder for stage-2 paragraph embeddings. Construct one via `core.embeddings.make_ollama_embedder()` at the top of `_run_phase2` and pass it to both `extract_units` and (where compatible) the chunk-loop. If interface incompatibility makes one shared embedder awkward, two short-lived ones are fine — they hit the same Ollama endpoint.

### D5 — Delete `chunk_markdown` and `Chunk` from the public API

Repo-wide grep confirms zero external consumers. Removing them tightens the public surface and eliminates the second-implementation maintenance burden. `document_store/chunking.py` goes too.

If at any point the swap is reverted, restoration is easy (single revert of the deletion commit). Keeping a fallback path "just in case" would add a permanent if/else for a contingency that doesn't have a real trigger.

---

## What stays exactly as-is

- The chunks table schema (`chunks_schema.py`) and the chunk-embeddings vec0 table — no change.
- The `ChunkMetadataFields` model, including `section_path`, `chunk_index`, `parent_doc_id`.
- The `_run_phase2` two-phase ingest pattern (document registered first, chunks/embeddings second, idempotent under `repair()`).
- The four-signal RRF search (`search.py:search_unified`) — chunk-size agnostic.
- `find_duplicates` (uses doc-level embeddings, not chunks).
- The `repair()` semantics — it'll just produce new-shape chunks the next time it runs on a previously-incomplete or never-completed document.

---

## Phased implementation

### Phase 0 — Behaviour bench (BEFORE any code change)

Goal: lock down what "no regression" means concretely so later phases have something to measure against.

- [ ] Pick three representative inputs: (a) a short note (~500 chars, no headings), (b) a medium meeting log (~3k chars, two H2 sections), (c) a paper-shaped markdown (~30k chars, ~8 H2 sections, one section >5k chars).
- [ ] Run the *current* `chunk_markdown` against each. Record: chunk count, list of `(start_char, end_char, section_path)` triples, total content coverage.
- [ ] Save these as a fixture under `src/andamentum/document_store/tests/fixtures/chunker_baseline/` (markdown source + expected JSON).
- [ ] Add a single regression test `test_chunker_swap_shape_compatibility.py` that loads each fixture, runs the *new* chunker through the adapter, and asserts: (i) section_path is recovered for every chunk that has a heading ancestor; (ii) chunk count is ≤ baseline; (iii) total covered chars ≥ 95% of source. The test is allowed to fail until Phase 2; it pins the contract.

**Verification gate:** the test exists and runs (failing is OK at this point).

### Phase 1 — The adapter (no production code path changes yet)

Goal: build and unit-test the `Unit → Chunk` adapter in isolation.

- [ ] Create `src/andamentum/document_store/chunker_adapter.py` with two functions:
  - `_compute_section_paths(content: str, units: Iterable[Unit]) -> list[str]` — uses `chunker.structural.find_headings` and `build_section_tree`, walks the tree to find the deepest section containing `unit.source_start`, returns ` > `-joined heading path. Empty string for units with no heading ancestor (e.g. preamble).
  - `units_to_chunks(content: str, units: list[Unit]) -> list[Chunk]` — emits `Chunk` records (text, section_path, chunk_index from enumerate, start_char from `source_start`, end_char from `source_end`).
- [ ] Unit tests under `src/andamentum/document_store/tests/test_chunker_adapter.py`:
  - Single-section document → one chunk, section_path matches the H2 title
  - Nested headings document → section_path is the full breadcrumb (`"Methods > ODE Solver"`)
  - Preamble before first heading → section_path is empty
  - Source with no headings at all → section_path is empty for every unit
  - `start_char` / `end_char` round-trip: `content[c.start_char:c.end_char] == c.text` for every chunk
- [ ] All adapter tests pass.

**Verification gate:** `uv run pytest src/andamentum/document_store/tests/test_chunker_adapter.py` green; `uv run pyright src/andamentum/document_store/` 0 errors; `uv run ruff check src/andamentum/document_store/` clean.

### Phase 2 — Wire into `_run_phase2`

Goal: production code path now uses the new chunker, but the old `chunk_markdown` and `Chunk` exports stay live for one more phase as a safety net.

- [ ] In `document_store/public.py:_run_phase2`, replace `chunks = chunk_markdown(content, max_tokens=500, overlap_tokens=50)` with:
  ```python
  from andamentum.chunker import extract_units
  from .chunker_adapter import units_to_chunks
  from andamentum.core.embeddings import make_ollama_embedder

  embedder = make_ollama_embedder(model=embedding_model)
  result = await extract_units(
      content,
      target_min_chars=1500,
      target_max_chars=4000,
      embedding_fn=embedder,
  )
  chunks = units_to_chunks(content, result.units)
  ```
- [ ] Inspect: can the same `embedder` be reused for the chunk-level `embed_text` loop, or do we keep `EmbeddingService` for that path? If reuse is mechanical, do it; if not, accept two short-lived embedders (they hit the same endpoint).
- [ ] Empty-result branch (`if not chunks: chunks = [Chunk(text=content, ...)]`) survives — same fallback as before for fully-empty extract output, which now happens only if `content.strip()` is empty.
- [ ] Run the Phase 0 regression test — it should now pass for all three fixtures.
- [ ] Run a real end-to-end ingest against a fresh test database with the medium and paper-shaped fixtures. Confirm by hand:
  - `find_by_metadata({"has_decision": True})` returns expected results
  - `search()` on a known query returns the expected top hit
  - The `topics` field on a sample chunk is specific (not generic like `"general"` or `"discussion"`)
- [ ] Repair test: ingest a doc, manually delete its chunks (simulate phase-2 crash), call `repair()`, confirm chunks are recreated with new shape.

**Verification gate:** full `uv run pytest` green (test count = previous + Phase 1 adapter tests + Phase 0 regression test, exactly — no other test should change behaviour); pyright 0 errors in document_store; ruff clean. End-to-end smoke checks above all pass.

### Phase 3 — Remove the simple chunker

Goal: delete the old code path.

- [ ] Confirm zero external consumers of `chunk_markdown` and `Chunk`:
  ```bash
  grep -rn "chunk_markdown\|from .*document_store.*Chunk\b" --include="*.py" . | grep -v "src/andamentum/document_store/chunking\|test_chunker"
  ```
  Expected output: empty.
- [ ] Delete `src/andamentum/document_store/chunking.py`.
- [ ] In `document_store/__init__.py`: remove the `chunk_markdown` and `Chunk` imports and remove them from `__all__`.
- [ ] In `document_store/public.py`: the `from .chunking import Chunk` inside `_run_phase2` is now stale — replace with `from .chunker_adapter import units_to_chunks` if any leftover reference, or remove the empty-result branch's `Chunk(...)` synthesis (the new path goes through the adapter, which can yield a one-element list directly if needed; or fall back to skipping store_chunk altogether if `result.units` is empty and content is whitespace).
- [ ] Delete the Phase 0 fixture's *old-chunker* baseline JSON (it's a snapshot of behaviour we no longer have). The regression test should now assert against a fixed *new-chunker* baseline JSON, regenerated once and committed. (This stops the test from being a freeze of dead behaviour.)

**Verification gate:** full `uv run pytest` green; pyright 0 errors in document_store; ruff clean; the grep above returns empty.

### Phase 4 — Update CLAUDE.md

Goal: make sure the next developer (human or model) reads accurate documentation.

- [ ] Update the `## Project` section's `andamentum.document_store` description in `CLAUDE.md` to mention that document_store uses `andamentum.chunker` for chunking.
- [ ] Update the layering rules under `## Architectural conventions` to add `chunker` as a permitted upstream dependency for `document_store`.

**Verification gate:** rendered CLAUDE.md reads coherently; no contradictions with the actual code.

---

## Risks and mitigations

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Bigger chunks dilute `topics` / `has_decision` | Medium | Medium | `target_max_chars=4000` (D1). If observed in Phase 2 smoke checks, drop to 3000. |
| Boundary-straddling fact lost without overlap | Low–Medium | Medium | Semantic boundaries are typically *better* than overlap windows. If empirically a problem after Phase 2, follow-up plan adds `prev_chunk_id` / `next_chunk_id` columns and retrieval-time neighbour expansion. Out of scope here. |
| `section_path` adapter has a subtle off-by-one | Low | Low–Medium | Phase 1 unit tests cover preamble, nested headings, no-heading inputs, and the byte-identical round-trip. |
| Ollama unreachable → stage-2 fails → ingest fails | Existing | Same as today | Document_store already requires Ollama for chunk embeddings. Stage-2 doesn't add a new failure mode. |
| Heterogeneous chunk sizes in the same DB after upgrade | Certain | None | Chunks are doc-scoped; embeddings are dimension-aligned; RRF and BM25 are size-agnostic. This is fine. |
| Stage-2 latency on long documents | Low | Low | Bounded — only fires for sections >`target_max`. ~20 paragraph embeds for a 12k section, all batched. Embed calls are ~20× cheaper than the LLM-metadata calls we're saving. Net wall-clock improvement on long docs. |
| Repair() produces different chunks than original ingest | Yes (post-swap) | Low | Already true within a DB during normal use (chunks change if the chunker improves). Search behaviour is unaffected. |

---

## Open questions for the user

These are not blockers for starting Phase 0, but should be answered before Phase 2 default-on:

1. **Should `target_max_chars` be a parameter of `ingest()` or hardcoded to 4000?** Recommendation: hardcode in `_run_phase2`. Document_store has a specific use-case; adding a knob invites tuning in the wrong direction. If we ever need different sizing for sub-domains, add it then.
2. **Worktree?** Per the previous turn, the chunker swap could be done on a fresh `chunker-swap` branch in `.worktrees/chunker-swap/` so the rag/ cleanup commits ship independently. Default plan if no answer: same branch as the rag/ cleanup.
3. **Does Phase 4 (CLAUDE.md update) belong in the same commit as Phase 3, or its own commit?** Recommendation: same commit as Phase 3. Doc-and-code together, one atom.

---

## Out of scope (for later, not now)

- `prev_chunk_id` / `next_chunk_id` columns + neighbour expansion at retrieval time. Only worth doing if Phase 2 smoke checks reveal real recall regressions, which is unlikely given semantic boundaries.
- Per-ingest stats / observability for chunk counts and stage-2 firing rate. Useful but not gating.
- Replacing `EmbeddingService` (in `document_store/embeddings.py`) with `core.embeddings.make_ollama_embedder` everywhere. Adjacent cleanup, separate commit.
- An MCP wrapper for `ingest` / `search` / `find_by_metadata` / `update_metadata` / `delete` / `find_duplicates`. Big leverage move flagged in the prior architectural review; deserves its own plan.

---

## Acceptance criteria (the whole plan, all phases)

- [ ] `document_store/chunking.py` deleted; `chunk_markdown` and `Chunk` no longer in `__all__`.
- [ ] `document_store/chunker_adapter.py` exists with `units_to_chunks` and `_compute_section_paths`, fully unit-tested.
- [ ] `_run_phase2` calls `andamentum.chunker.extract_units` with `target_max_chars=4000` and an Ollama embedder.
- [ ] `section_path` continues to be populated correctly with full heading breadcrumbs for documents with nested headings.
- [ ] All existing pytest green (`pytest --no-header -q` shows the same passing count as before, plus the new adapter tests, plus the regression test).
- [ ] pyright 0 errors in `src/andamentum/document_store/`.
- [ ] ruff clean on `src/andamentum/document_store/`.
- [ ] CLAUDE.md mentions document_store's chunker dependency.
- [ ] One end-to-end ingest of a paper-shaped fixture confirmed by manual inspection of `topics`, `has_decision`, `search` results.
