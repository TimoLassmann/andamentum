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
| `list(*, where, limit, cursor)` | Live documents, newest first, without text. `limit` has no default. |
| `count(*, where)` | How many live documents match. |
| `read(doc_id, *, rev, start, end)` | Characters `start..end` of a revision. |
| `search(query, *, limit, where)` | The best passages, best first (needs a searcher). |
| `sync()` | Brings every index up to date. |

`where` has three forms: `{"field": value}` (equal, including the JSON type: `True` is not `1`), `{"field": None}` (missing or null), `{"field": [a, b]}` (one of; an empty list matches nothing). For the next page of `list`, pass the last result's `cursor`. All entries must hold; dotted names reach nested fields (`{"bib.year": 2024}`).

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

```bash
andamentum-docstore ingest notes.db paper.pdf https://example.org/page
andamentum-docstore sync notes.db --embedding-model embeddinggemma-2:latest
andamentum-docstore search notes.db "what limits the method?" \
    --embedding-model embeddinggemma-2:latest --reranker-model BAAI/bge-reranker-v2-m3
andamentum-docstore list notes.db --where '{"kind": "paper"}' --limit 20
andamentum-docstore get notes.db notes/apollo
andamentum-docstore delete notes.db notes/apollo
andamentum-docstore enrich notes.db identifiers
andamentum-docstore enrich notes.db extract --model ollama:gemma4:31b-cloud
```

## Reranking cost

The cross-encoder stays loaded in the process after the first search: a 0.6B model's memory, which `ollama ps` does not show, plus a few seconds per search. `rerank(query, texts, reranker_model=...)` and `select_relevant(scored, min_relevance=..., buffer=...)` are exported for callers who want to score their own passages and decide how many to keep.

## Files

The store is one SQLite file (WAL mode, so several processes can write it) plus its `-wal` and `-shm` files. There is no default location and no environment variable; you name the path. Files written by the earlier v1 and v2 stores are not read.
