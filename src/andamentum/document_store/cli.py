"""``andamentum-docstore``: store documents now, index them later.

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
    andamentum-docstore enrich notes.db identifiers
    andamentum-docstore enrich notes.db classify --decision-model tev1:4b \\
        --labels '{"trial": "a clinical trial report", "review": "a review"}' \\
        --max-text-chars 8000

``sync`` can be stopped with Ctrl-C at any time; running it again continues,
and documents already embedded are not embedded again. The index settings
(``--embedding-model``, ``--max-chars``) must stay the same on every ``sync``
and ``search`` of one store: a change rebuilds the index from scratch.

Every command that prints a result takes ``--json``.

Exit status: 0 success; 1 ``sync`` finished but some documents could not be
indexed (the rest are indexed); 2 usage error; 3 an input could not be read
(the store file or a source file does not exist, an unreachable URL, an
unsupported format, no such document); 4 the run failed (model or network
error, or a crash).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .store import Store
from .types import DocumentNotFound, SearchUnavailable, line_range

EXIT_NEGATIVE = 1
EXIT_USAGE = 2
EXIT_INPUT = 3
EXIT_FAILED = 4

_EPILOG = """\
exit status:
  0  success
  1  sync finished, but some documents could not be indexed
  2  usage error: a wrong or missing argument
  3  an input could not be read (store file or source file missing, URL
     unreachable, unsupported format, no such document)
  4  the run failed (model or network error, or a crash)
"""

_ENRICH_EPILOG = """\
what each enricher needs:
  identifiers     nothing (DOIs, PMIDs, arXiv ids found in the text)
  facts           nothing (counts and dates found in the text)
  bibliographic   network; --mailto is optional (looks the references up);
                  NCBI_API_KEY from the environment is used when set
  extract         --model
  classify        --decision-model, --labels and --max-text-chars (all three)
  --multi applies to classify only.
"""

_ENRICHERS = ("identifiers", "facts", "bibliographic", "extract", "classify")


class _Usage(Exception):
    """A wrong argument found after parsing."""


class _Unreadable(Exception):
    """An input the command needs cannot be read."""


def _formatter() -> type[argparse.HelpFormatter]:
    return argparse.RawDescriptionHelpFormatter


def build_parser() -> argparse.ArgumentParser:
    from andamentum import __version__

    p = argparse.ArgumentParser(
        prog="andamentum-docstore",
        description="Store documents in one SQLite file; index and search them.",
        epilog=_EPILOG,
        formatter_class=_formatter(),
    )
    p.add_argument(
        "--version", action="version", version=f"andamentum-docstore {__version__}"
    )
    sub = p.add_subparsers(dest="command", required=True, metavar="COMMAND")

    def add(name: str, help: str, *, epilog: str = _EPILOG) -> argparse.ArgumentParser:
        sp = sub.add_parser(
            name,
            help=help,
            description=help,
            epilog=epilog,
            formatter_class=_formatter(),
        )
        sp.add_argument(
            "db", type=Path, metavar="STORE", help="the store file (SQLite)"
        )
        return sp

    def json_flag(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--json", action="store_true", help="print the result as JSON")

    ing = add("ingest", "convert sources to markdown and store them (no model needed)")
    ing.add_argument(
        "sources", nargs="+", help="file paths or URLs; '-' reads markdown from stdin"
    )
    ing.add_argument("--key", help="a name for the document (only with one source)")
    ing.add_argument(
        "--metadata", help="a JSON object stored as the document's metadata"
    )
    ing.add_argument(
        "--mode",
        choices=("fast", "accurate"),
        default="fast",
        help="fast (default) or accurate (PDF only; needs --figure-model)",
    )
    ing.add_argument(
        "--figure-model",
        default=None,
        help="the model that describes figures; required with --mode accurate",
    )
    json_flag(ing)

    def index_args(sp: argparse.ArgumentParser) -> None:
        sp.add_argument(
            "--embedding-model", required=True, help="the Ollama embedding model"
        )
        sp.add_argument(
            "--max-chars",
            type=int,
            default=2000,
            help="chunk size limit in characters (default: %(default)s); must be the "
            "same value on every sync and search of one store: a change "
            "rebuilds the index",
        )

    sy = add(
        "sync",
        "bring the index up to date (embeds new documents; --max-chars must be "
        "the same on every sync and search of one store)",
    )
    index_args(sy)
    json_flag(sy)

    se = add(
        "search",
        "search passages; syncs the index first (which embeds any new "
        "documents), so --embedding-model and --max-chars must match your "
        "earlier syncs",
    )
    se.add_argument("query", help="what to look for")
    index_args(se)
    se.add_argument(
        "--reranker-model", required=True, help="the cross-encoder that reranks hits"
    )
    se.add_argument(
        "--limit",
        type=int,
        default=10,
        help="most passages to return (default: %(default)s)",
    )
    se.add_argument("--where", help="a JSON metadata filter")
    json_flag(se)

    gt = add("get", "print a document's text")
    gt.add_argument("doc", help="document id or key")
    gt.add_argument("--rev", type=int, help="a revision number (default: the newest)")
    gt.add_argument("--json", action="store_true", help="print JSON with the metadata")

    ls = add("list", "list documents, newest first")
    ls.add_argument("--where", help="a JSON metadata filter")
    ls.add_argument(
        "--limit", type=int, default=None, help="most documents to list (default: all)"
    )
    json_flag(ls)

    de = add(
        "delete",
        "delete a document (it can be restored from Python); --json prints "
        '{"deleted": DOC_ID}',
    )
    de.add_argument("doc", help="document id or key")
    json_flag(de)

    en = add(
        "enrich",
        "add one metadata key to every matching document",
        epilog=_ENRICH_EPILOG + "\n" + _EPILOG,
    )
    en.add_argument(
        "enricher", choices=_ENRICHERS, help="which enricher to run (see below)"
    )
    en.add_argument("--where", help="a JSON metadata filter (default: every document)")
    en.add_argument("--model", help="extract: the language model")
    en.add_argument("--mailto", help="bibliographic: a contact address for Crossref")
    en.add_argument("--decision-model", help="classify: the Ollama decision model")
    en.add_argument("--labels", help="classify: a JSON object of label -> description")
    en.add_argument(
        "--multi", action="store_true", help="classify: allow any number of labels"
    )
    en.add_argument(
        "--max-text-chars",
        type=int,
        help="classify: how many characters of each document are sent to the "
        "decision model",
    )
    json_flag(en)
    return p


def _json_arg(value: str | None, *, name: str) -> Any:
    if value is None:
        return None
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise _Usage(f"{name} is not valid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise _Usage(f"{name} must be a JSON object")
    return parsed


def _check_arguments(args: argparse.Namespace) -> None:
    """Everything argparse cannot see: JSON values and flags that go together."""
    if args.command == "ingest":
        if args.key and len(args.sources) > 1:
            raise _Usage("--key needs exactly one source")
        if args.mode == "accurate" and not args.figure_model:
            raise _Usage("--mode accurate needs --figure-model")
        _json_arg(args.metadata, name="--metadata")
    if args.command in ("search", "list", "enrich"):
        _json_arg(args.where, name="--where")
    if args.command == "search" and args.limit < 1:
        raise _Usage("--limit must be at least 1")
    if args.command == "enrich":
        if args.enricher == "extract" and not args.model:
            raise _Usage("enrich extract needs --model")
        if args.enricher == "classify":
            if not (args.decision_model and args.labels and args.max_text_chars):
                raise _Usage(
                    "enrich classify needs --decision-model, --labels and --max-text-chars"
                )
            _json_arg(args.labels, name="--labels")


async def _doc_id(store: Store, ref: str) -> str:
    found = await store.resolve(ref)
    if found is None and await store.get(ref) is not None:
        found = ref
    if found is None:
        raise _Unreadable(f"no document {ref!r}")
    return found


def _print_json(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=1))


async def _open(args: argparse.Namespace) -> Store:
    if args.command != "ingest" and not args.db.is_file():
        raise _Unreadable(f"no store at {args.db}; create one with 'ingest'")
    if args.command == "search":
        return await Store.open_with_search(
            args.db,
            embedding_model=args.embedding_model,
            reranker_model=args.reranker_model,
            max_chars=args.max_chars,
        )
    if args.command == "sync":
        from .index import HybridIndex

        index = HybridIndex(
            embedding_model=args.embedding_model, max_chars=args.max_chars
        )
        return await Store.open(args.db, indexes=[index])
    return await Store.open(args.db)


async def _ingest(store: Store, args: argparse.Namespace) -> int:
    from .pipeline import ingest

    metadata = _json_arg(args.metadata, name="--metadata")
    for source in args.sources:
        if (
            source != "-"
            and "://" not in source
            and not Path(source).expanduser().exists()
        ):
            raise _Unreadable(f"no such file: {source}")
    rows = []
    for source in args.sources:
        if source == "-":
            doc_id = await store.put(sys.stdin.read(), key=args.key, metadata=metadata)
        else:
            doc_id = await ingest(
                store,
                source,
                key=args.key,
                metadata=metadata,
                mode=args.mode,
                figure_model=args.figure_model,
            )
        rows.append({"doc_id": doc_id, "source": source})
        if not args.json:
            print(f"{doc_id}\t{source}")
    if args.json:
        _print_json(rows)
    return 0


async def _sync(store: Store, args: argparse.Namespace) -> int:
    report = await store.sync()
    for name, failures in report.failures.items():
        for doc_id, error in failures:
            print(f"{name}: FAILED {doc_id}: {error}", file=sys.stderr)
    if args.json:
        _print_json(
            {
                "ok": report.ok,
                "processed": report.processed,
                "failures": {
                    n: [list(f) for f in fs] for n, fs in report.failures.items()
                },
            }
        )
    else:
        for name, n in report.processed.items():
            print(f"{name}: {n} changes applied")
    return 0 if report.ok else EXIT_NEGATIVE


async def _search(store: Store, args: argparse.Namespace) -> int:
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
        _print_json(rows)
    else:
        for r in rows:
            snippet = " ".join(r["text"].split())[:200]
            print(
                f"{r['score']:.3f}  {r['doc']}:{r['lines'][0]}-{r['lines'][1]}  {snippet}"
            )
    return 0


async def _get(store: Store, args: argparse.Namespace) -> int:
    doc = await store.get(await _doc_id(store, args.doc), rev=args.rev)
    if doc is None:
        raise _Unreadable(f"no revision {args.rev} of {args.doc!r}")
    if args.json:
        _print_json(doc.__dict__)
    else:
        print(doc.text)
    return 0


async def _list(store: Store, args: argparse.Namespace) -> int:
    refs = await store.list(
        where=_json_arg(args.where, name="--where"), limit=args.limit
    )
    if args.json:
        _print_json(
            [
                {
                    "doc_id": r.doc_id,
                    "key": r.key,
                    "rev": r.rev,
                    "updated_at": r.updated_at,
                    "media_type": r.media_type,
                    "metadata": r.metadata,
                }
                for r in refs
            ]
        )
    else:
        for r in refs:
            title = r.metadata.get("title", "")
            print(f"{r.doc_id}\t{r.key or ''}\trev {r.rev}\t{r.updated_at}\t{title}")
    return 0


async def _delete(store: Store, args: argparse.Namespace) -> int:
    doc_id = await _doc_id(store, args.doc)
    await store.delete(doc_id)
    if args.json:
        _print_json({"deleted": doc_id})
    else:
        print(f"deleted {doc_id}")
    return 0


async def _enrich(store: Store, args: argparse.Namespace) -> int:
    from . import enrich

    where = _json_arg(args.where, name="--where")
    name = args.enricher
    if name == "identifiers":
        results = await enrich.enrich_all(store, enrich.enrich_identifiers, where=where)
    elif name == "facts":
        results = await enrich.enrich_all(store, enrich.enrich_facts, where=where)
    elif name == "extract":
        results = await enrich.enrich_all(
            store, enrich.enrich_extract, where=where, model=args.model
        )
    elif name == "bibliographic":
        import os

        from ..references import ReferenceHttp

        http = ReferenceHttp(mailto=args.mailto)
        try:
            results = await enrich.enrich_all(
                store,
                enrich.enrich_bibliographic,
                where=where,
                http=http,
                # A credential, read from the environment as the PubMed
                # providers do; never a command-line flag.
                api_key=os.environ.get("NCBI_API_KEY"),
            )
        finally:
            await http.aclose()
    else:  # classify
        from ..core.decisions import DecisionModel

        labels = _json_arg(args.labels, name="--labels")
        async with await DecisionModel.open(args.decision_model) as dm:
            results = await enrich.enrich_all(
                store,
                enrich.enrich_classify,
                where=where,
                decision_model=dm,
                labels=labels,
                multi=args.multi,
                max_chars=args.max_text_chars,
            )
    done = sum(r.skipped is None for r in results)
    if args.json:
        _print_json(
            {
                "enricher": name,
                "enriched": done,
                "skipped": len(results) - done,
                "results": [
                    {
                        "doc_id": r.doc_id,
                        "key": r.key,
                        "rev": r.rev,
                        "skipped": r.skipped,
                    }
                    for r in results
                ],
            }
        )
    else:
        for r in results:
            if r.skipped:
                print(f"{r.doc_id}\t{r.skipped}")
        print(f"{name}: {done} documents enriched, {len(results) - done} skipped")
    return 0


_COMMANDS = {
    "ingest": _ingest,
    "sync": _sync,
    "search": _search,
    "get": _get,
    "list": _list,
    "delete": _delete,
    "enrich": _enrich,
}


async def _run(args: argparse.Namespace) -> int:
    store = await _open(args)
    try:
        return await _COMMANDS[args.command](store, args)
    finally:
        await store.close()


def _error(message: str) -> None:
    print(f"error: {message}", file=sys.stderr)


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(sys.argv[1:] if argv is None else list(argv))
    try:
        _check_arguments(args)
        return asyncio.run(_run(args))
    except _Usage as exc:
        parser.error(str(exc))
    except (_Unreadable, DocumentNotFound, FileNotFoundError) as exc:
        _error(str(exc))
        return EXIT_INPUT
    except KeyboardInterrupt:
        _error("stopped; run the same command again to continue")
        return EXIT_FAILED
    except Exception as exc:
        from andamentum.harvest import (
            FetchError,
            FigureModelRequired,
            UnsupportedFormatError,
        )

        if isinstance(exc, FigureModelRequired):
            parser.error(str(exc))
        if isinstance(exc, (FetchError, UnsupportedFormatError)):
            _error(str(exc))
            return EXIT_INPUT
        if isinstance(exc, SearchUnavailable):
            _error(str(exc))
            return EXIT_FAILED
        _error(f"{type(exc).__name__}: {exc}")
        return EXIT_FAILED
    return EXIT_FAILED  # unreachable: parser.error exits


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
