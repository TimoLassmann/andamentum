# andamentum.serve — notes for Claude

## Overview

`andamentum.serve` is a small HTTP server for another program on the same machine (the first caller is a home service that forwards calls for agents). It is a thin layer over the Python functions, with no logic of its own: every route calls one library function. Public surface: `create_app` and `ServeError` (`__all__`, served by a lazy `__getattr__` so `import andamentum.serve` works without the `serve` extra and the command can report it missing). Command: `andamentum-serve -o DIR [--port 8765] [--store PATH --embedding-model M --reranker-model R] [--searxng-url URL | --browser]`. Needs the `serve` extra (FastAPI, uvicorn).

## Routes (`app.py`)

- `GET /health` → version, whether a library is open, and the queue state.
- `POST /library/search` `{query, limit}` → hits with `doc_id`, `key`, `title`, `rev`, `start`, `end`, `heading_path`, `text`, `score`. `POST /library/read` `{doc_id, start, end}` → the stored markdown, or a span of it. Both answer 503 when the server was started without `--store`.
- `POST /procedure/review_document` `{note, model, reviewer}` → `ReviewResult` as JSON; every edit carries its `id`.
- `POST /procedure/accept_edits` `{note, edits}` → `AcceptedEdits` (`text`, `applied`, `not_applied`). No model.
- `POST /procedure/research` `{question, mode, depth, model}` → the research result. Web modes are refused (422) unless the server was started with `--searxng-url` or `--browser`.

Rules every route keeps:

- **Bodies refuse an unknown field** (`extra="forbid"`), so a misspelt field is a 422, not a silent default.
- **Every error is `{"error": "<one sentence>"}`**, including validation errors and an unknown route (404 "no such route"); the caller passes it on unchanged. A failed run is a 500 naming the procedure and the exception.
- **The model is named in each request**; the server has no default model.
- **No route takes a path, a store or a URL to read.** A note is reviewed through `ReviewedDocument.from_markdown`, never `open_source`, so a note that reads like a path is reviewed as those words. Search hits leave out the document's source location (a local path) on purpose.
- **The confidentiality tripwire runs** as on the command line (`ConfidentialityMarkerError` → 422). There is no bypass field.

## Model queue (`queue.py`)

A local Ollama model (`ollama:NAME` without `-cloud`) serves one request at a time and holds gigabytes while it does, so every call that uses one holds one `asyncio.Lock` for its whole run. A call on any other model takes a slot of a `Semaphore(4)`. `/library/search` always holds the local lock, because the store embeds the query with a local model. `accept_edits` takes no slot. The rule matches the user's standing rule: never two local model calls at once; cloud calls may overlap, about four at a time.

## Records

Every procedure call writes `out/<UTC time>-<6 hex>/request.json` (procedure, body, and the caller headers `X-Agent`, `X-Run-Session`, `X-Scope` when present) before it runs, then `result.json` or `error.json`. A result therefore outlives a client that stopped waiting. Library calls are not recorded.

## Layering

`serve` depends on `core`, `whetstone`, `deep_research` and `document_store`, all imported inside the functions that use them (`_open_store`, `_research`, `_review`, `_accept`), so starting the server loads no model library. Nothing depends on `serve`; `tests/test_serve.py::test_nothing_imports_serve` enforces it.

## Tests

`tests/test_serve.py` runs the app in process (`httpx.ASGITransport`) with the module-level helpers (`_open_store`, `_research`, `_review`, `_accept`) replaced by stubs: body validation, the error shape, the records, the queue (two local calls run one after the other, two cloud calls overlap), the library routes over a stub store, and the command's exit codes. No test calls a model or a real store.

## Open questions (owner's decision)

- `/library/read` returns the stored markdown only, never the original bytes (a PDF).
- A research call can run longer than the caller's own timeout (30 minutes for the first caller). The caller then reports a 502, but the server finishes the run and writes `result.json` under `--out`.
