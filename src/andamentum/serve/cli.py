"""Command-line entry point: andamentum-serve.

Usage (the shared secret comes from the environment, never a flag)::

    export ANDAMENTUM_SERVE_SECRET="$(openssl rand -hex 32)"
    andamentum-serve --out records/
    andamentum-serve --out records/ --store library.db \\
        --embedding-model nomic-embed-text --reranker-model BAAI/bge-reranker-v2-m3

Runs until it is stopped (Ctrl-C). Exit codes: 0 stopped; 2 usage error, no or
a too short secret, or the ``serve`` extra is missing; 3 the store could not be opened; 4 the server failed.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Sequence
from pathlib import Path

#: The port when --port is not given.
DEFAULT_PORT = 8765

#: The environment variable the shared secret is read from. A secret is a
#: credential, so it is never a flag: a flag shows in ``ps`` and shell history.
SECRET_ENV = "ANDAMENTUM_SERVE_SECRET"

_EPILOG = """\
examples:
  andamentum-serve --out records/
  andamentum-serve --out records/ --port 8800 --searxng-url http://127.0.0.1:8888
  andamentum-serve --out records/ --store library.db \\
      --embedding-model nomic-embed-text --reranker-model BAAI/bge-reranker-v2-m3

secret:
  The server needs ANDAMENTUM_SERVE_SECRET in its environment (at least 32
  characters, for example from `openssl rand -hex 32`). Every request must send
  "Authorization: Bearer <secret>"; any other request is answered 401.

routes (127.0.0.1 only; JSON bodies; errors are {"error": "..."}):
  POST /library/search            {"query", "limit"}
  POST /library/read              {"doc_id", "start", "end"}
  POST /procedure/research        {"question", "mode", "depth", "model"}
  POST /procedure/review_document {"note", "model", "reviewer"}
  POST /procedure/accept_edits    {"note", "edits"}
  GET  /health

exit codes:
  0  stopped
  2  usage error, no or a too short secret, or the serve extra is not installed
  3  the store could not be opened
  4  the server failed
"""


def build_parser() -> argparse.ArgumentParser:
    from andamentum import __version__

    parser = argparse.ArgumentParser(
        prog="andamentum-serve",
        description="Serve the document library, document review and research over HTTP "
        "on this machine only (127.0.0.1), to a caller that sends the shared secret "
        f"from {SECRET_ENV}. Each procedure call names its model in the "
        "request; calls on a local Ollama model run one at a time. Every procedure call "
        "is written to the --out folder.",
        epilog=_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"andamentum-serve {__version__}")
    parser.add_argument(
        "-o",
        "--out",
        required=True,
        metavar="DIR",
        help="folder for the record of every procedure call (request and result)",
    )
    parser.add_argument(
        "--port", type=int, default=DEFAULT_PORT, help=f"port on 127.0.0.1 (default {DEFAULT_PORT})"
    )
    library = parser.add_argument_group("library (the /library routes; without --store they answer 503)")
    library.add_argument("--store", metavar="PATH", help="the document store to search and read")
    library.add_argument(
        "--embedding-model", help="the Ollama embedding model the store's index was built with"
    )
    library.add_argument("--reranker-model", help="the reranker for library search")
    web = parser.add_argument_group("web research (needed for research with mode web or both)")
    choice = web.add_mutually_exclusive_group()
    choice.add_argument("--searxng-url", metavar="URL", help="a SearxNG instance to search the web")
    choice.add_argument(
        "--browser",
        action="store_true",
        help="search the web with an isolated Chrome in podman, started per research call",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.store and not (args.embedding_model and args.reranker_model):
        parser.error("--store needs --embedding-model and --reranker-model")
    if not args.store and (args.embedding_model or args.reranker_model):
        parser.error("--embedding-model and --reranker-model are for --store")
    secret = os.environ.get(SECRET_ENV, "")
    if not secret:
        print(f"error: set {SECRET_ENV} to the secret the caller sends", file=sys.stderr)
        return 2
    if args.store and not Path(args.store).is_file():
        print(f"error: no store at {args.store}", file=sys.stderr)
        return 3
    try:
        import uvicorn

        from .app import ServeError, create_app
    except ImportError:
        print("error: install andamentum[serve] (FastAPI and uvicorn)", file=sys.stderr)
        return 2
    try:
        app = create_app(
            out=args.out,
            secret=secret,
            store=args.store,
            embedding_model=args.embedding_model,
            reranker_model=args.reranker_model,
            searxng_url=args.searxng_url,
            browser=args.browser,
        )
    except ServeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"andamentum-serve on http://127.0.0.1:{args.port} (records in {args.out})", file=sys.stderr)
    try:
        uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")
    except Exception as exc:  # noqa: BLE001 — a crash of the server is exit 4
        print(f"error: the server failed: {exc}", file=sys.stderr)
        return 4
    return 0


if __name__ == "__main__":
    sys.exit(main())
