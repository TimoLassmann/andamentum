# andamentum.document_store

Personal knowledge base with 4-signal search. Store anything — thoughts, papers, meeting notes, decisions, tasks — and find it again with natural language.

## What it provides

Seven async functions that form the entire public surface:

| Function | Purpose |
|----------|---------|
| `ingest(database, content)` | Store content with auto-chunking, embedding, and metadata extraction |
| `search(database, query)` | Natural language search with LLM query planning |
| `find_by_metadata(database, filters)` | Structured query by exact metadata fields |
| `update_metadata(database, doc_id, metadata)` | Update fields on a document |
| `delete(database, doc_id)` | Remove a document and all its chunks |
| `repair(database)` | Fix incomplete ingestions after crashes |
| `find_duplicates(database)` | Detect near-duplicate documents via embeddings |

## How search works

`search()` fuses four signals via Reciprocal Rank Fusion (RRF):

1. **FTS5 keyword matching** — fast, available immediately after ingest
2. **Chunk-level semantic search** — embedding similarity on individual chunks
3. **Document-level semantic search** — embedding similarity on whole documents
4. **DHP temporal clustering** — recently active and relevant topics rank higher

An LLM decomposes the query into a search plan (semantic query + optional metadata filter) before running the four signals.

## How ingest works

Two-phase design:
- **Phase 1** (atomic): document stored in SQLite, FTS5 keyword-searchable immediately
- **Phase 2** (repairable): content chunked, each chunk embedded, LLM extracts metadata; if interrupted, `repair()` re-runs it

## Installation

```bash
pip install andamentum
```

## Quick start

```python
from andamentum.document_store import ingest, search

doc_id = await ingest("research", "MAP-Elites could work for antibody optimization")
results = await search("research", "What do I know about quality-diversity optimization?")
for r in results:
    print(r.title, r.score)
```

Databases are stored at `~/.config/andamentum/databases/{name}.db`. Override with `$DOCUMENT_STORE_DIR` or `$ANDAMENTUM_DATABASES_DIR`.
