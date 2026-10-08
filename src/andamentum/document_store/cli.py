"""``andamentum-docstore`` — store documents now, index them later.

The first argument of every command is the store file. ``ingest`` and the
read commands never call a model. ``sync`` builds the index (it embeds with
the named Ollama model); ``search`` syncs first, then searches.

    andamentum-docstore ingest notes.db paper.pdf https://example.org/page
    andamentum-docstore sync notes.db --embedding-model embeddinggemma-2:latest
    andamentum-docstore search notes.db "what limits the method?" \\
        --embedding-model embeddinggemma-2:latest \\
        --reranker-model BAAI/bge-reranker-v2-m3
    andamentum-docstore list notes.db --where '{"kind": "paper"}' --limit 20
    andamentum-docstore get notes.db DOC_ID_OR_KEY

``sync`` can be stopped with Ctrl-C at any time; running it again continues,
and documents already embedded are not embedded again. The index settings
(``--embedding-model``, ``--max-chars``) must stay the same between runs: a
change rebuilds the index from scratch.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

from .index import HybridIndex
from .searchers import HybridRerankSearcher
from .store import Store
from .types import line_range


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="andamentum-docstore",
        description="Store documents in one SQLite file; index and search them.",
    )
    sub = p.add_subparsers(dest="command", required=True)

    ing = sub.add_parser("ingest", help="convert sources to markdown and store them (no model)")
    ing.add_argument("db", type=Path)
    ing.add_argument("sources", nargs="+", help="file paths or URLs; '-' reads markdown from stdin")
    ing.add_argument("--key", help="key for the document (only with one source)")
    ing.add_argument("--metadata", help="JSON object stored as the document's metadata")
    ing.add_argument("--mode", choices=("fast", "accurate"), default="fast")

    def index_args(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--embedding-model", required=True, help="Ollama embedding model")
        sp.add_argument("--max-chars", type=int, default=2000, help="chunk size limit")

    sy = sub.add_parser("sync", help="bring the index up to date (embeds new documents)")
    sy.add_argument("db", type=Path)
    index_args(sy)

    se = sub.add_parser("search", help="search passages (syncs first)")
    se.add_argument("db", type=Path)
    se.add_argument("query")
    index_args(se)
    se.add_argument("--reranker-model", required=True, help="cross-encoder for the rerank")
    se.add_argument("--limit", type=int, default=10)
    se.add_argument("--where", help="JSON metadata filter")
    se.add_argument("--json", action="store_true", help="print JSON")

    gt = sub.add_parser("get", help="print a document's text")
    gt.add_argument("db", type=Path)
    gt.add_argument("doc", help="document id or key")
    gt.add_argument("--rev", type=int)
    gt.add_argument("--json", action="store_true", help="print JSON with metadata")

    ls = sub.add_parser("list", help="list documents, newest first")
    ls.add_argument("db", type=Path)
    ls.add_argument("--where", help="JSON metadata filter")
    ls.add_argument("--limit", type=int, default=None, help="default: all")

    de = sub.add_parser("delete", help="soft-delete a document")
    de.add_argument("db", type=Path)
    de.add_argument("doc", help="document id or key")
    return p


def _json_arg(value: str | None, *, name: str) -> Any:
    if value is None:
        return None
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"{name} is not valid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise SystemExit(f"{name} must be a JSON object")
    return parsed


async def _doc_id(store: Store, ref: str) -> str:
    found = await store.resolve(ref)
    if found is None and await store.get(ref) is not None:
        found = ref
    if found is None:
        raise SystemExit(f"no document {ref!r}")
    return found


async def _run(args: argparse.Namespace) -> int:
    if args.command in ("sync", "search"):
        ix = HybridIndex(embedding_model=args.embedding_model, max_chars=args.max_chars)
        searcher = (
            HybridRerankSearcher(index=ix, reranker_model=args.reranker_model)
            if args.command == "search"
            else None
        )
        store = await Store.open(args.db, indexes=[ix], searcher=searcher)
    else:
        store = await Store.open(args.db)
    try:
        if args.command == "ingest":
            if args.key and len(args.sources) > 1:
                raise SystemExit("--key needs exactly one source")
            metadata = _json_arg(args.metadata, name="--metadata")
            for source in args.sources:
                if source == "-":
                    doc_id = await store.put(sys.stdin.read(), key=args.key, metadata=metadata)
                else:
                    from .pipeline import ingest

                    doc_id = await ingest(
                        store, source, key=args.key, metadata=metadata, mode=args.mode
                    )
                print(f"{doc_id}\t{source}")
        elif args.command == "sync":
            report = await store.sync()
            for name, n in report.processed.items():
                print(f"{name}: {n} changes applied")
            for name, failures in report.failures.items():
                for doc_id, error in failures:
                    print(f"{name}: FAILED {doc_id}: {error}", file=sys.stderr)
            return 0 if report.ok else 1
        elif args.command == "search":
            hits = await store.search(
                args.query, limit=args.limit, where=_json_arg(args.where, name="--where")
            )
            rows = []
            for h in hits:
                doc = await store.get(h.doc_id, rev=h.rev)
                assert doc is not None
                a, b = line_range(doc.text, start=h.start, end=h.end)
                rows.append(
                    {
                        "doc": doc.key or doc.doc_id,
                        "lines": [a, b],
                        "score": round(h.score, 4),
                        "heading_path": list(h.heading_path),
                        "text": h.text,
                    }
                )
            if args.json:
                print(json.dumps(rows, ensure_ascii=False, indent=1))
            else:
                for r in rows:
                    snippet = " ".join(r["text"].split())[:200]
                    print(f"{r['score']:.3f}  {r['doc']}:{r['lines'][0]}-{r['lines'][1]}  {snippet}")
        elif args.command == "get":
            doc = await store.get(await _doc_id(store, args.doc), rev=args.rev)
            if doc is None:
                raise SystemExit(f"no revision {args.rev} of {args.doc!r}")
            if args.json:
                print(json.dumps(doc.__dict__, ensure_ascii=False, indent=1))
            else:
                print(doc.text)
        elif args.command == "list":
            refs = await store.list(where=_json_arg(args.where, name="--where"), limit=args.limit)
            for r in refs:
                title = r.metadata.get("title", "")
                print(f"{r.doc_id}\t{r.key or ''}\trev {r.rev}\t{r.updated_at}\t{title}")
        elif args.command == "delete":
            await store.delete(await _doc_id(store, args.doc))
    finally:
        await store.close()
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        return asyncio.run(_run(args))
    except KeyboardInterrupt:
        print("stopped; run the same command again to continue", file=sys.stderr)
        return 130


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
