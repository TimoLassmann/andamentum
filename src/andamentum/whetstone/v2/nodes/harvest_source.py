"""Node 1: HarvestSource — bring the document into memory as markdown.

Delegates to ``andamentum.harvest.extract`` for URLs and files. Zero LLM
calls. Source dispatch is explicit (no silent fallbacks per the
constitution): URL prefix → harvest; Path object → harvest; existing
file path string → harvest; everything else → treated as raw markdown
text.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic_graph import BaseNode, GraphRunContext

from andamentum.harvest import extract

from ..deps import ReviewDeps
from ..schemas import ReviewResult
from ..state import ReviewState

if TYPE_CHECKING:
    from .chunk_and_scan import ChunkAndScan


logger = logging.getLogger("andamentum.whetstone.v2")

_URL_PREFIXES = ("http://", "https://", "file://")


@dataclass
class HarvestSource(BaseNode[ReviewState, ReviewDeps, ReviewResult]):
    """Resolve state.source to markdown, store on state.markdown."""

    async def run(
        self, ctx: GraphRunContext[ReviewState, ReviewDeps]
    ) -> "ChunkAndScan":
        ctx.state.current_phase = "harvest"
        source = ctx.state.source
        logger.info("[harvest] loading source: %s", _describe_source(source))

        if isinstance(source, Path):
            # Path object → always go through harvest (extracts via PDF/DOCX/etc.)
            ctx.state.markdown = await extract(source)
        elif isinstance(source, str):
            if source.startswith(_URL_PREFIXES):
                ctx.state.markdown = await extract(source)
            elif _looks_like_existing_file(source):
                ctx.state.markdown = await extract(source)
            else:
                # Treat as raw markdown text. This is the explicit, intentional
                # path — not a fallback from a failed harvest call.
                ctx.state.markdown = source
        else:
            raise TypeError(
                f"source must be str or pathlib.Path, got {type(source).__name__}"
            )

        logger.info(
            "[harvest] done — %d chars of markdown", len(ctx.state.markdown)
        )

        # Defer the import to avoid a circular ChunkAndScan ↔ this file at module load.
        from .chunk_and_scan import ChunkAndScan

        return ChunkAndScan()


def _describe_source(source: object) -> str:
    """One-line description of the source for logs (truncate raw markdown)."""
    if isinstance(source, Path):
        return str(source)
    if isinstance(source, str):
        if source.startswith(_URL_PREFIXES) or len(source) <= 120:
            return source
        return f"<raw text, {len(source)} chars>"
    return repr(source)


def _looks_like_existing_file(source: str) -> bool:
    """True iff ``source`` resolves to an existing regular file on disk.

    Bounded check to avoid surprising filesystem touches on huge strings:
    if the string is longer than 4 KiB, it cannot be a sensible filesystem
    path and we treat it as raw markdown without touching the disk.
    """
    if len(source) > 4096:
        return False
    if "\n" in source:
        return False
    try:
        p = Path(source)
        return p.is_file()
    except (OSError, ValueError):
        return False
