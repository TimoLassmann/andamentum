# andamentum.document_store

Personal knowledge base with 4-signal search. Store anything — thoughts, papers, meeting notes, decisions, tasks — and find it again with natural language.

## What it provides

The core async functions that form the public surface:

| Function | Purpose |
|----------|---------|
| `ingest(database, content)` | Store content with auto-chunking, embedding, and metadata extraction |
| `search(database, query)` | Natural language search with LLM query planning |
| `find_by_metadata(database, filters)` | Structured query by metadata fields — exact match or set-membership |
| `describe_metadata(database)` | Discover the metadata schema — fields present + value distributions |
| `update_metadata(database, doc_id, metadata)` | Update fields on a document |
| `delete(database, doc_id)` | Remove a document and all its chunks |
| `repair(database)` | Fix incomplete ingestions after crashes |
| `find_duplicates(database)` | Detect near-duplicate documents via embeddings |

### The three read functions

The store is **schema-less** — metadata is arbitrary JSON, and consumers define
their own vocabulary (`record_type`, `status`, …). Three uniform functions cover
reading, identically usable by humans and agents:

- **`search`** — unstructured recall over *content* (NL query, RRF ranking). It
  does **not** rank or filter on arbitrary metadata; only the LLM query planner's
  small built-in field whitelist narrows by metadata.
- **`find_by_metadata`** — deterministic structured query over *any* metadata
  field. A scalar value matches by equality; a **list value matches by
  set-membership** (`field IN (...)`), so "any of several statuses" is one call.
- **`describe_metadata`** — reports the vocabulary actually present, so a caller
  can fill `find_by_metadata` filters without prior knowledge of the schema.

Domain-specific querying (a task layer's "open" / "overdue" / "high priority")
belongs to the **consumer**, built on top of these primitives — the store does
not grow a function per consumer.

## Querying metadata

```python
from andamentum.document_store import describe_metadata, find_by_metadata

# 1. Discover what exists — no prior knowledge of the schema.
schema = await describe_metadata("brain")
#   {"record_type": FieldProfile(present_in=62, distinct=3,
#                                values={"task": 42, "idea": 13, "decision": 7}),
#    "title":       FieldProfile(present_in=62, distinct=62, values=None)}  # high-cardinality

# 2. Drill into a subset to see its fields.
await describe_metadata("brain", filters={"record_type": "task"})
#   {"status": FieldProfile(present_in=42, distinct=4,
#                           values={"todo": 20, "in_progress": 5, "blocked": 3, "done": 14}), ...}

# 3. Now the filter is grounded. Set-membership in a single query:
open_tasks = await find_by_metadata("brain", {
    "record_type": "task",
    "status": ["todo", "in_progress", "blocked"],
})

# Cheap overview — metadata only, no per-document content reads:
rows = await find_by_metadata("brain", {"record_type": "task"},
                              limit=10_000, include_content=False)
```

A `FieldProfile` carries `present_in` (documents with the field), `distinct`
(number of distinct values), and `values` (the value→count breakdown, populated
only for low-cardinality fields so output stays bounded — `None` otherwise; tune
the cut-off with `describe_metadata(..., max_values=...)`).

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
