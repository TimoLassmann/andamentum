"""The single public entry point: ``extract``.

Dispatches by detected format. For HTML uses metadata to fast-path
article-vs-not, falling back to a race between trafilatura and Docling
when the page's metadata is ambiguous.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from typing import Literal

from .backends import (
    extract_passthrough,
    extract_with_docling,
    extract_with_trafilatura,
)
from .errors import ExtractionError, HarvestError, UnsupportedFormatError
from .fetch import Fetched, resolve
from .metadata import sniff_html_metadata
from .scoring import score_markdown

Format = Literal["pdf", "html", "docx", "pptx", "markdown", "plain"]

logger = logging.getLogger(__name__)


async def extract(
    source: str | Path,
    *,
    tdm_allowed_hosts: frozenset[str] = frozenset(),
) -> str:
    """Convert any document source to clean markdown.

    Parameters
    ----------
    source:
        A URL string ("http://..." / "https://...") OR a local file path
        (str or pathlib.Path). "file://" URIs are also accepted.
    tdm_allowed_hosts:
        Hostnames the caller attests they hold a text-and-data-mining
        licence for. Disarms the paywalled-publisher tripwire for those
        hosts. Default empty: tripwire fully active.

    Returns
    -------
    str
        Best-quality markdown extraction. For HTML, "best" is determined
        empirically — when page metadata is ambiguous we run multiple
        backends and pick the one whose output has more structure.

    Raises
    ------
    FetchError
        URL unreachable, file missing, SSRF block, robots.txt disallow,
        or paywall tripwire (host on the curated paywalled-publisher list
        and not in ``tdm_allowed_hosts``).
    UnsupportedFormatError
        Source format detected but no backend can handle it.
    ExtractionError
        Every applicable backend failed (or returned empty content).
    """
    fetched = await resolve(source, tdm_allowed_hosts=tdm_allowed_hosts)
    return await _dispatch(fetched)


async def extract_from_bytes(
    data: bytes,
    *,
    format: Format,
    source_url: str = "",
) -> str:
    """Run the same extraction pipeline as `extract`, but on bytes you already have.

    Useful when the caller has already fetched the URL (e.g. ``deep_research``
    bringing pages in via its own HTTP backend, the editor's ``/api/fetch``
    endpoint). Skips the resolve/fetch step entirely.

    Parameters
    ----------
    data:
        Raw bytes of the document.
    format:
        One of ``"pdf"``, ``"html"``, ``"docx"``, ``"pptx"``, ``"markdown"``,
        ``"plain"``. Caller is responsible for picking the right format —
        for a stronger guarantee use ``extract`` instead.
    source_url:
        Optional URL or filename for diagnostics + link resolution
        (trafilatura uses it to resolve relative links in HTML).
    """
    fetched = Fetched(data=data, format=format, source_url=source_url)
    return await _dispatch(fetched)


async def _dispatch(fetched: Fetched) -> str:
    """Format-specific routing — shared by ``extract`` and ``extract_from_bytes``."""
    if fetched.format == "html":
        return await _extract_html(fetched)
    if fetched.format == "pdf":
        return await extract_with_docling(fetched.data, fetched.source_url, fmt="pdf")
    if fetched.format == "docx":
        return await extract_with_docling(fetched.data, fetched.source_url, fmt="docx")
    if fetched.format == "pptx":
        return await extract_with_docling(fetched.data, fetched.source_url, fmt="pptx")
    if fetched.format in ("markdown", "plain"):
        return await extract_passthrough(fetched.data, fetched.source_url)

    # Should not be reachable — fetch.resolve() raises UnsupportedFormatError
    # when it can't detect a format. Defensive only.
    raise UnsupportedFormatError(f"no backend for format {fetched.format!r}")


async def _extract_html(fetched: Fetched) -> str:
    """HTML-specific dispatch: sniff metadata, then route or race."""
    meta = sniff_html_metadata(fetched.data)
    logger.debug(
        "html metadata sniff: verdict=%s reason=%s og_type=%s ld_type=%s",
        meta.verdict,
        meta.reason,
        meta.og_type,
        meta.ld_json_type,
    )

    if meta.verdict == "article":
        # Article-like → trafilatura. If it fails (rare for real articles),
        # fall back to docling rather than giving up.
        try:
            return await extract_with_trafilatura(fetched.data, fetched.source_url)
        except HarvestError as exc:
            logger.warning(
                "trafilatura failed on article-tagged page; falling back: %s", exc
            )
            return await extract_with_docling(
                fetched.data, fetched.source_url, fmt="html"
            )

    if meta.verdict == "not_article":
        # Index/listing/homepage → docling preserves card structure.
        return await extract_with_docling(fetched.data, fetched.source_url, fmt="html")

    # Ambiguous → race both, score, pick winner.
    return await _race_html_backends(fetched)


async def _race_html_backends(fetched: Fetched) -> str:
    """Run trafilatura and docling concurrently, score, return the better one."""
    traf_task = asyncio.create_task(
        _safe_call(extract_with_trafilatura, fetched.data, fetched.source_url)
    )
    docl_task = asyncio.create_task(
        _safe_call(extract_with_docling, fetched.data, fetched.source_url, fmt="html")
    )
    traf_md, traf_err = await traf_task
    docl_md, docl_err = await docl_task

    candidates: list[tuple[str, float, str]] = []  # (name, score, markdown)
    diagnostics: dict[str, str] = {}
    if traf_md is not None:
        s = score_markdown(traf_md)
        candidates.append(("trafilatura", s, traf_md))
        diagnostics["trafilatura"] = f"{len(traf_md)} chars, score={s:.1f}"
    else:
        diagnostics["trafilatura"] = f"failed: {traf_err}"
    if docl_md is not None:
        s = score_markdown(docl_md)
        candidates.append(("docling", s, docl_md))
        diagnostics["docling"] = f"{len(docl_md)} chars, score={s:.1f}"
    else:
        diagnostics["docling"] = f"failed: {docl_err}"

    if not candidates:
        raise ExtractionError(
            f"all HTML backends failed for {fetched.source_url}",
            attempted=["trafilatura", "docling"],
            diagnostics=diagnostics,
        )

    candidates.sort(key=lambda c: -c[1])
    winner = candidates[0]
    logger.info(
        "html race winner: %s (score=%.1f) for %s; diagnostics=%s",
        winner[0],
        winner[1],
        fetched.source_url,
        diagnostics,
    )
    return winner[2]


async def _safe_call(fn, *args, **kwargs) -> tuple[str | None, Exception | None]:
    """Call `fn`; return (result, None) on success, (None, exc) on failure."""
    try:
        return await fn(*args, **kwargs), None
    except Exception as exc:
        return None, exc
