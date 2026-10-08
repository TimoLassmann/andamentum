# andamentum.document_store — notes for Claude

Claude Code loads this file when it reads files in this directory. Read it before you change the module.

## Overview

One SQLite file of documents, searchable. The design rule is **fixed data, flexible index, fixed API**:

- **Source tables** (`db.py`): `documents` (ULID id, optional key, created/deleted time), `revisions` (every revision's full text, media type, metadata JSON, optional link to the original bytes), `blobs` (original bytes by sha256), `changes` (the log every write appends to), `meta` (`schema_version`). They never change shape. If they ever must, bump `SCHEMA_VERSION`; `connect` refuses a file of another version.
- **Indexes** (`index.py`): disposable tables `ix_<name>_*` built from the source tables. `Store.sync` hands each index the documents changed since its `last_seq`. An index whose `config()` differs from the recorded one is dropped and rebuilt, so a new chunk size or embedding model is a rebuild, never a migration.
- **API** (`store.py`, all async): `Store.open(path, *, indexes=(), searcher=None)`, `put`, `get`, `original`, `resolve`, `update`, `delete`, `restore`, `purge`, `list`, `count`, `search`, `read`, `sync`, `close`. Downstream code (epistemic, the bench, the CLI) uses only these.
- **Searchers** (`searchers.py`, `agent.py`) are chosen at open. **Enrichers** (`enrich/`) sit outside the store and write metadata through `update`.

## CLI

| Script | What it does | LLM? |
|---|---|---|
| `andamentum-docstore` | `ingest DB SOURCE...` (harvest → markdown, original bytes kept), `sync DB --embedding-model M`, `search DB QUERY --embedding-model M --reranker-model R`, `get`, `list`, `delete`, `enrich DB NAME` | `ingest`, `get`, `list`, `delete` and the identifiers/facts enrichers need none; `sync` embeds; `search` embeds and reranks; `enrich extract` needs `--model`, `enrich classify` a decision model, `--labels` and `--max-chars`. `--json` on every command that prints a result; exit 1 when `sync` leaves documents unindexed, 2 usage, 3 missing store/source/document, 4 model or network error. `ingest --mode accurate` needs `--figure-model` |

## Rules the code relies on

- **`put` never deduplicates.** Equal texts are separate documents (epistemic stores equal execution-step messages).
- **A key names at most one live document.** The unique index is partial (`WHERE deleted_at IS NULL`): a deleted document gives its key up, and `restore` raises `KeyExists` if a live document took it meanwhile.
- **`update` writes a new revision; metadata replaces the whole object.** Old revisions stay until `purge`. Callers that change one field do get → change → update.
- **`list` has no default limit.** `limit=None` means all. The old store's silent `limit=100` truncated epistemic queries without a warning; that is why the parameter has no default.
- **`where` is three forms only:** equality, `None` (missing or null), list/tuple/set (one of; empty matches nothing). AND of entries, dotted names for nested fields. Equality checks the JSON type too, so `True`, `1` and `"1"` are three different values (SQLite alone stores `true` as 1). Field names are bound as JSON paths (non-ASCII kept unescaped, because SQLite paths do not decode `\uXXXX`), never interpolated. Range and "contains" filters stay in the caller.
- **`list` orders by the newest revision's rowid, descending,** so `limit=1` is the most recently written document. `DocRef.cursor` is that rowid; paging walks the order as it stood, so a document updated between pages moves to the front and is not returned again.
- **Nothing in `store.py` calls a model.** `sync` and `search` do so only through the index and searcher the caller passed. A store opened with neither (epistemic) never embeds.
- **WAL and a 30 s busy timeout;** every write is one `BEGIN IMMEDIATE` transaction, so several processes can write one file.
- **No default location, no environment variables.** The caller names the path.

## The hybrid index

`HybridIndex(*, embedding_model, max_chars=2000, name="hybrid", embed_fn=None)`. `embedding_model=None` builds a lexical-only index. `embed_fn` replaces the Ollama client (tests, the bench's embedding cache); the model name is still recorded in the config.

- Chunks come from `andamentum.chunker.chunk(text, max_chars=...)`: headings, then paragraph breaks, then sentence ends; fenced code and tables are never cut. Each chunk keeps its character offsets and heading path.
- Each chunk gets a context prefix `"<title> > <heading path>"`; the title is `metadata["title"]`, else the first heading. The embedder and the reranker see prefix + chunk; BM25 indexes both.
- Tables: `docs` (doc → rev, content hash), `chunks`, a porter FTS5 index, a trigram FTS5 index (for `grep` and identifier matching) and a sqlite-vec table created at the first embedding.
- A metadata-only update with the same title and text moves the revision number and embeds nothing.
- **Failures are never absorbed.** A document the index cannot process goes into `ix_<name>_failures` and into `SyncReport.failures`, and is retried on the next sync; `search` refuses to run while a failure stands, because it would silently miss that document. An embedding error (service down) is raised and the index keeps its position.

## Searchers

**Default: `HybridRerankSearcher(*, index, reranker_model, depth=50)`.** BM25 (OR-joined terms) and vector nearest neighbours over chunks (with a `where`, both legs rank only the allowed documents' chunks), fused by RRF (k=60) to the top 50, then reranked by the cross-encoder. This is the configuration measured as `v2_rerank` (MuSiQue dev full_support@10 0.490, HotpotQA dev 0.955; v1 with the rerank was 0.505 and 0.945). The trigram leg is not in the fusion: it was off in that measurement. `reranker_model` has no default; `SUGGESTED_RERANKER_MODEL` (`BAAI/bge-reranker-v2-m3`) is the measured one.

**Opt-in: `AgentSearcher(*, model, inner, budget=AgentBudget(), model_settings=None)`.** Round 0 is the inner searcher's candidates reranked against the request; one structured call decides whether they are enough; otherwise a pydantic-ai tool loop (`list`, `glob`, `grep`, `search`, `outline`, `read`, `note`) runs until the model returns a `Selection` of passages it was shown. The `search` tool is the inner searcher, reranked against the tool's own query. The result order is selected, then seen, then the rest of round 0; every score is the reranker score against the original request. Measured as `v2_agent`: MuSiQue dev 0.625. Use `search_with_run` when several searches run concurrently; `last_run` is per instance.

`group_by_document(hits)` groups passages in the order of each document's first passage (the ranking) and scores a document by its best passage.

## Python entry points

`Store` is the main entry. `await Store.open_with_search(path, *, embedding_model, reranker_model, max_chars=2000)` builds the default searcher (`HybridIndex` + `HybridRerankSearcher`) in one call; neither model has a default. `ingest` (from `pipeline.py`, exported) turns a file or URL into a document (`mode="accurate"` needs `figure_model=`). The enrichers stay under `andamentum.document_store.enrich`. Errors derive from `DocumentStoreError`: `SearchUnavailable` (no searcher, or some documents could not be indexed; also a `RuntimeError`), `SchemaMismatch` (also a `RuntimeError`), `DocumentNotFound`, `KeyExists`. `__all__` holds the store, searchers, result types, `ingest` and the errors; `rerank`, `select_relevant`, `rrf` and `reset_reranker_cache` stay importable from the package but are not exported.

## Reranking (`reranking.py`)

**`search` always reranks.** The cross-encoder stays resident for the life of the process, a memory budget nothing in `ollama ps` shows, plus a few seconds per search; sentence-transformers is a core dependency so results never depend on extras. The model is loaded with `transformers` directly and only ever sees **fixed input shapes** (length buckets × fixed batch size), because PyTorch's Apple GPU backend keeps a compiled graph per distinct shape and never frees it (pytorch/pytorch#181213). `min_relevance` and `reranker_model` have no defaults. `select_relevant` needs both the elbow cut and the absolute floor; the elbow errs generous by up to one item, and the floor removes it on evidence — do not tune that away.

## Ingest (`pipeline.py`)

`ingest(store, source, *, key=None, metadata=None, mode="fast")` converts a path or URL with `andamentum.harvest` (the only place harvest is imported, lazily), keeps the original bytes and records `metadata["source"] = {"location", "format", "media_type"}`. It calls no model besides harvest's converter; indexing happens in `sync`.

## Layering

`document_store` may import `core`, `chunker`, `references` (enrichers) and `harvest` (`pipeline.py` only). It must not import `epistemic`, `deep_research`, `compose` or `whetstone`. `epistemic` depends on it.

## The benchmark (`bench/`)

`python -m andamentum.document_store.bench run --suite musique --tier dev --arm default ...`. Arms: `oracle`, `shuffled`, `bm25`, `fused`, `default`, `lexical`, `closed_book`, `rag`, `agent`, `agent_lexical`. One store file per corpus holds both a `hybrid` and a `lexical` index. A built store is reused only when its layout, corpus hash, embedding model and digest match; bump `STORE_LAYOUT` when the build changes. Answering (`bench/answering.py`) lives here, not in the store: search never writes prose.
