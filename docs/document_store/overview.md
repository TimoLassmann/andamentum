# andamentum.document_store

One SQLite file of documents that you can search. The design keeps three things apart:

- **Fixed data.** The source tables keep every revision of every document in full, plus the original bytes (a PDF, say) when you give them. Their shape never changes.
- **Flexible index.** Indexes are disposable tables built from the source tables. A new chunk size, embedding model or search method rebuilds the index; the data stays as it is.
- **Fixed API.** A small set of async calls on `Store`. Searchers and enrichers change behind it.

## Store and read documents

```python
from andamentum.document_store import Store

async with await Store.open("notes.db") as store:
    doc_id = await store.put(
        "# Apollo\n\nThe SaltLog service ships logs.",
        key="notes/apollo",                  # optional, unique among live documents
        metadata={"title": "Apollo notes", "kind": "note"},
    )
    doc = await store.get(doc_id)            # full text + metadata of the newest revision
    await store.update(doc_id, metadata={**doc.metadata, "reviewed": True})  # revision 2
    old = await store.get(doc_id, rev=1)     # every revision stays readable
    same = await store.resolve("notes/apollo")
    notes = await store.list(where={"kind": "note"}, limit=None)   # None = all
    n = await store.count(where={"reviewed": True})
    await store.delete(doc_id)               # soft delete; restore() undoes it
```

| Call | What it does |
|---|---|
| `put(text, *, key, media_type, metadata, original)` | New document. Never deduplicates. A key held by a live document raises `KeyExists`. |
| `get(doc_id, *, rev=None)` | The newest (or a given) revision with its full text, or `None`. |
| `original(doc_id, *, rev=None)` | The original bytes stored with a revision, or `None`. |
| `resolve(key)` | The id of the live document with this key, or `None`. |
| `update(doc_id, *, text, metadata, media_type, original)` | A new revision; what you leave out is copied. `metadata` replaces the whole object. |
| `delete` / `restore` / `purge` | Soft delete, undo it, remove for good (with its revisions and unshared original bytes). |
| `list(*, where, since, until, limit, cursor)` | Live documents, newest first, without text; each `DocRef` has `created_at` and `updated_at`. `limit` has no default. |
| `count(*, where, since, until)` | How many live documents match. |
| `read(doc_id, *, rev, start, end)` | Characters `start..end` of a revision. |
| `search(query, *, limit, where, since, until)` | The best passages, best first (needs a searcher). |
| `sync()` | Brings every index up to date. |

`where` has three forms: `{"field": value}` (equal, including the JSON type: `True` is not `1`), `{"field": None}` (missing or null), `{"field": [a, b]}` (one of; an empty list matches nothing). For the next page of `list`, pass the last result's `cursor`. All entries must hold; dotted names reach nested fields (`{"bib.year": 2024}`).

`since` and `until` keep to the documents first stored in a time window: `since` is included, `until` is not, so two windows that meet never share a document. They compare the time of the first `put`, not of the newest revision, so an update (an enricher's metadata write, say) does not make an old document new. Both must be `datetime`s with a time zone; a naive one is a `ValueError`, because the store does not guess a zone:

```python
from datetime import datetime, timedelta, timezone

since = datetime.now(timezone.utc) - timedelta(days=1)
notes = await store.list(where={"type": "note"}, since=since, limit=None)
```

## Search

Open the store with an index and a searcher:

```python
from andamentum.document_store import (
    HybridIndex, HybridRerankSearcher, Store, SUGGESTED_RERANKER_MODEL, group_by_document,
)

index = HybridIndex(embedding_model="embeddinggemma-2:latest")   # an Ollama model
searcher = HybridRerankSearcher(index=index, reranker_model=SUGGESTED_RERANKER_MODEL)
store = await Store.open("notes.db", indexes=[index], searcher=searcher)

hits = await store.search("what limits the method?", limit=10)
for h in hits:
    print(h.score, h.doc_id, h.heading_path, h.text[:80])
docs = group_by_document(hits)    # documents in rank order, each scored by its best passage
```

`search` calls `sync` first, so the results cover every document. Each `Hit` holds `doc_id`, `rev`, `start`, `end` (character offsets in that revision), `text`, `score` and `heading_path`; `store.read(h.doc_id, rev=h.rev, start=h.start, end=h.end) == h.text`. To return the whole document, call `store.get(h.doc_id)`.

**The default searcher** splits each document into chunks of at most 2,000 characters with `andamentum.chunker` (at headings, then paragraph breaks, then sentence ends; code blocks and tables are never cut). It ranks chunks by BM25 and by vector similarity, fuses the two rankings (reciprocal rank fusion, top 50), and reranks those 50 with a cross-encoder. Each chunk is scored with its context, `"<title> > <heading path>"`. On the retrieval benchmark (MuSiQue dev, 200 multi-hop questions) this configuration found all supporting passages in the top 10 for 0.490 of questions.

**The search agent** is opt-in:

```python
from andamentum.document_store import AgentSearcher

searcher = AgentSearcher(
    model="ollama:gemma4:31b-cloud",
    inner=HybridRerankSearcher(index=index, reranker_model=SUGGESTED_RERANKER_MODEL),
)
```

It runs the default search first, asks the model whether the passages are enough, and if not, lets the model use tools (`search`, `grep`, `outline`, `read`, `list`, `glob`, `note`) until it selects the passages the request needs. Its `search` tool is the default searcher. It returns passages in the same form, with every score from the reranker against your request. On the same benchmark it reached 0.625, at the cost of several model calls per search.

**Keyword search, with no model at all.** `KeywordSearcher` ranks the chunks of an index built without embeddings by BM25 and returns them in that order: no embedding, no rerank, no Ollama, and no model held in memory.

```python
store = await Store.open_with_keyword_search("notes.db")   # an index named "keyword"
hits = await store.search("salt records", limit=10)
```

It finds passages by their words only: a passage with more of the query's words usually ranks higher, and one that says the same thing in other words is not found. The plain BM25 arm scored below the default search on the benchmark above. `Hit.score` is the BM25 score, higher is better; on a store of a few documents it is very small, because a word that occurs in half of the documents or more gets almost no weight. The `keyword` index sits beside a `hybrid` index in the same file, and neither rebuilds the other.

A store opened with no index and no searcher (as epistemic opens it) stores and reads documents and never calls a model.

## Indexing

`sync()` processes the documents changed since the last sync. It is safe to stop and run again: documents already embedded are not embedded again, and a metadata change that leaves the title and text alone embeds nothing. If a document cannot be indexed, `sync` reports it in `SyncReport.failures` and retries it next time; `search` refuses to run until it is fixed, because it would silently miss that document. If the embedding service is down, `sync` raises.

Changing `embedding_model` or `max_chars` rebuilds the index from scratch on the next sync.

## Ingest files and URLs

```python
from andamentum.document_store.pipeline import ingest

doc_id = await ingest(store, "paper.pdf", metadata={"kind": "paper"})
```

`ingest` converts a path or URL to markdown with `andamentum.harvest`, keeps the original bytes and records `metadata["source"]` (location, format, media type).

## Enrichers

Enrichers live outside the store (`andamentum.document_store.enrich`). Each reads a document and writes one top-level metadata key through `update`, keeping the other keys: identifiers (DOI, PMID, PMCID, arXiv, ISBN, URLs), bibliographic records (Crossref by DOI, PubMed by PMID), basic facts, LLM-extracted fields (needs `model=`) and labels from a decision model with labels you define (needs `decision_model=`). `enrich_all` runs one enricher over every document a `where` matches, one at a time.

## Command line

Python: `store = await Store.open_with_search(path, embedding_model=..., reranker_model=...)` opens a store that can search (`Store.open_with_keyword_search(path)` one that searches with no model); `await ingest(store, source)` stores a file or URL. Every command takes `--json` where it prints a result. Exit codes: 0 success, 1 `sync` could not index some documents, 2 usage, 3 an input could not be read, 4 the run failed. `--since` and `--until` take an ISO 8601 time with its zone; a bare date such as `2026-11-19` is a usage error, because midnight UTC is 08:00 in Perth. `--index keyword` on `sync` and `search` needs no model and refuses `--embedding-model` and `--reranker-model`. `delete --purge` removes a document for good, and also takes the id of a document deleted earlier.

```bash
andamentum-docstore ingest notes.db paper.pdf https://example.org/page
andamentum-docstore sync notes.db --embedding-model embeddinggemma-2:latest
andamentum-docstore search notes.db "what limits the method?" \
    --embedding-model embeddinggemma-2:latest --reranker-model BAAI/bge-reranker-v2-m3
andamentum-docstore list notes.db --where '{"kind": "paper"}' --limit 20
andamentum-docstore list notes.db --since 2026-11-19T00:00:00+08:00
andamentum-docstore sync notes.db --index keyword
andamentum-docstore search notes.db "salt records" --index keyword
andamentum-docstore get notes.db notes/apollo
andamentum-docstore delete notes.db notes/apollo
andamentum-docstore delete notes.db notes/apollo --purge
andamentum-docstore enrich notes.db identifiers
andamentum-docstore enrich notes.db extract --model ollama:gemma4:31b-cloud
```

## Reranking cost

The cross-encoder stays loaded in the process after the first search: a 0.6B model's memory, which `ollama ps` does not show, plus a few seconds per search. `rerank(query, texts, reranker_model=...)` and `select_relevant(scored, min_relevance=..., buffer=...)` are importable from `andamentum.document_store.reranking` for callers who want to score their own passages and decide how many to keep.

## Files

The store is one SQLite file (WAL mode, so several processes can write it) plus its `-wal` and `-shm` files. There is no default location and no environment variable; you name the path. Files written by the earlier v1 and v2 stores are not read.
