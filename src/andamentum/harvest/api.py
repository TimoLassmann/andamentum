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
from .errors import (
    ExtractionError,
    FigureModelRequired,
    HarvestError,
    UnsupportedFormatError,
)
from .fetch import Fetched, resolve
from .metadata import sniff_html_metadata
from .scoring import score_markdown

Format = Literal["pdf", "html", "docx", "pptx", "markdown", "plain"]
ExtractionMode = Literal["fast", "accurate"]

logger = logging.getLogger(__name__)


async def extract(
    source: str | Path,
    *,
    mode: ExtractionMode = "fast",
    ocr: bool = True,
    figure_model: str | None = None,
    tdm_allowed_hosts: frozenset[str] = frozenset(),
    preserve_superscripts: bool = False,
) -> str:
    """Convert any document source to clean markdown.

    Parameters
    ----------
    source:
        A URL string ("http://..." / "https://...") OR a local file path
        (str or pathlib.Path). "file://" URIs are also accepted.
    mode:
        ``"fast"`` (default) runs docling's standard pipeline on its stock
        settings, so it behaves as calling docling with no options did.
        Use ``ocr=False`` for the born-digital speed path.

        ``"accurate"`` runs docling's VLM pipeline: one model replaces the
        whole layout + OCR + table stack, and figures get a written
        description marked as such in the output. PDF only — every other
        format runs the standard pipeline either way. It needs the optional
        ``vlm`` extra (``uv sync --extra vlm``) and is roughly an order of
        magnitude slower per page.
    ocr:
        Fast mode only — the accurate pipeline has no OCR stage to switch.
        Defaults to True, which is docling's own default, so an unchanged
        call behaves exactly as it did before this parameter existed.

        Pass False for the born-digital speed path: a PDF that already has
        a text layer gains nothing from OCR and OCR is the expensive stage.
        A scan read with ``ocr=False`` produces no text and raises
        `ExtractionError` naming the remedy.
    figure_model:
        Which model writes the figure descriptions in accurate mode. No
        default: ``mode="accurate"`` without it raises `FigureModelRequired`.
        Ignored by ``mode="fast"``, which describes nothing.

        A docling picture-description preset id — ``"smolvlm"`` (256M), ``"granite_vision"`` (2B), ``"qwen"`` (3B),
        ``"pixtral"`` (12B) — downloads weights and runs them in-process.

        ``"ollama:<model>"`` instead reaches a vision model already served
        by a local Ollama, downloading nothing and loading nothing into this
        process (e.g. ``"ollama:minicpm-v4.6:1b"``). The model must report
        the ``vision`` capability; ``ollama show <model>`` lists it.

        Every option is asked the same question, so changing this compares
        models rather than prompts.
    tdm_allowed_hosts:
        Hostnames the caller attests they hold a text-and-data-mining
        licence for. Disarms the paywalled-publisher tripwire for those
        hosts. Default empty: tripwire fully active.
    preserve_superscripts:
        PDF only. Docling's PDF pipeline drops script information, so a
        superscript citation "patients¹." arrives as "patients 1 ." and a
        page's footnotes are merged into the body. With this on, the PDF's
        own character sizes and positions are read (``pdf_script``): raised
        runs are written as ``<sup>…</sup>``, and a raised number with a
        footnote of that number on its page as a Markdown footnote
        (``[^page-n]`` at the mark, ``[^page-n]: text`` at the end). Off by
        default, so every existing caller gets exactly the markdown it got
        before; other formats ignore it.

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
    FigureModelRequired
        ``mode="accurate"`` without ``figure_model``.
    ImportError
        ``mode="accurate"`` without the optional ``vlm`` extra installed.
    """
    _require_figure_model(mode, figure_model)
    fetched = await resolve(source, tdm_allowed_hosts=tdm_allowed_hosts)
    return await _dispatch(
        fetched,
        mode=mode,
        ocr=ocr,
        figure_model=figure_model,
        preserve_superscripts=preserve_superscripts,
    )


async def extract_from_bytes(
    data: bytes,
    *,
    format: Format,
    source_url: str = "",
    mode: ExtractionMode = "fast",
    ocr: bool = True,
    figure_model: str | None = None,
    preserve_superscripts: bool = False,
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
    mode:
        ``"fast"`` or ``"accurate"`` — see `extract`.
    ocr:
        Whether fast mode runs OCR — see `extract`.
    figure_model:
        Which model describes figures in accurate mode — see `extract`.
    preserve_superscripts:
        Mark a PDF's raised text and page footnotes — see `extract`.
    """
    _require_figure_model(mode, figure_model)
    fetched = Fetched(data=data, format=format, source_url=source_url)
    return await _dispatch(
        fetched,
        mode=mode,
        ocr=ocr,
        figure_model=figure_model,
        preserve_superscripts=preserve_superscripts,
    )


def _require_figure_model(mode: ExtractionMode, figure_model: str | None) -> None:
    if mode == "accurate" and figure_model is None:
        raise FigureModelRequired(
            'mode="accurate" needs figure_model: name the model that describes '
            "figures (CLI: --figure-model), e.g. 'smolvlm' or 'ollama:<model>'"
        )


async def _dispatch(
    fetched: Fetched,
    *,
    mode: ExtractionMode,
    ocr: bool,
    figure_model: str | None,
    preserve_superscripts: bool = False,
) -> str:
    """Format-specific routing — shared by ``extract`` and ``extract_from_bytes``."""
    if fetched.format == "pdf" and preserve_superscripts:
        markdown = await extract_with_docling(
            fetched.data,
            source_url=fetched.source_url,
            fmt="pdf",
            mode=mode,
            ocr=ocr,
            figure_model=figure_model,
        )
        # Docling's PDF pipeline drops script information; the PDF's own
        # characters still carry it. See pdf_script.
        from .pdf_script import analyse_pdf, mark_markdown

        marked, stats = mark_markdown(markdown, analyse_pdf(fetched.data))
        logger.info(
            "marked %d raised run(s) and %d footnote(s) from the PDF (%d unplaced)",
            stats.marked,
            stats.notes,
            stats.unplaced,
        )
        return marked
    if fetched.format == "html":
        return await _extract_html(
            fetched, mode=mode, ocr=ocr, figure_model=figure_model
        )
    if fetched.format in ("pdf", "docx", "pptx"):
        return await extract_with_docling(
            fetched.data,
            source_url=fetched.source_url,
            fmt=fetched.format,
            mode=mode,
            ocr=ocr,
            figure_model=figure_model,
        )
    if fetched.format in ("markdown", "plain"):
        return await extract_passthrough(fetched.data, fetched.source_url)

    # Should not be reachable — fetch.resolve() raises UnsupportedFormatError
    # when it can't detect a format. Defensive only.
    raise UnsupportedFormatError(f"no backend for format {fetched.format!r}")


async def _extract_html(
    fetched: Fetched, *, mode: ExtractionMode, ocr: bool, figure_model: str | None
) -> str:
    """HTML-specific dispatch: sniff metadata, then route or race.

    ``mode`` is carried through for completeness, but docling's HTML backend
    is not paginated, so the VLM pipeline has no page images to read and both
    modes run the same extraction.
    """
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
                fetched.data,
                source_url=fetched.source_url,
                fmt="html",
                mode=mode,
                ocr=ocr,
                figure_model=figure_model,
            )

    if meta.verdict == "not_article":
        # Index/listing/homepage → docling preserves card structure.
        return await extract_with_docling(
            fetched.data,
            source_url=fetched.source_url,
            fmt="html",
            mode=mode,
            ocr=ocr,
            figure_model=figure_model,
        )

    # Ambiguous → race both, score, pick winner.
    return await _race_html_backends(
        fetched, mode=mode, ocr=ocr, figure_model=figure_model
    )


async def _race_html_backends(
    fetched: Fetched, *, mode: ExtractionMode, ocr: bool, figure_model: str | None
) -> str:
    """Run trafilatura and docling concurrently, score, return the better one."""
    traf_task = asyncio.create_task(
        _safe_call(extract_with_trafilatura, fetched.data, fetched.source_url)
    )
    docl_task = asyncio.create_task(
        _safe_call(
            extract_with_docling,
            fetched.data,
            source_url=fetched.source_url,
            fmt="html",
            mode=mode,
            ocr=ocr,
            figure_model=figure_model,
        )
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
