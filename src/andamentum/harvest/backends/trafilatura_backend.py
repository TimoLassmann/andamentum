"""HTML → markdown via trafilatura.

Article-optimised: trafilatura strips nav/ads/cookie banners aggressively
and emits clean prose with `##` headings preserved. Excellent on real
articles; near-useless on link-card homepages (returns one structureless
run of text).

**Optional dependency.** Trafilatura is GPL-3.0; to keep the default
``andamentum`` install MIT-clean it's only pulled in by
``pip install andamentum[html-articles]``. When the package isn't
installed this backend raises ``ExtractionError`` (a ``HarvestError``
subclass) which the dispatcher in ``harvest/api.py`` catches and falls
back to docling.
"""

from __future__ import annotations

import logging

from ..errors import ExtractionError


async def extract(data: bytes, source_url: str) -> str:
    """Run trafilatura on raw HTML bytes; return markdown.

    Raises ExtractionError if trafilatura isn't installed, or if it
    returned no content (typical for JavaScript-rendered or paywalled
    pages).
    """
    # Trafilatura is sync; the cost is small enough we don't bother with
    # to_thread. Logger noise from missing-link-attribute warnings on every
    # anchor is suppressed per request.
    try:
        import trafilatura
    except ImportError as exc:
        raise ExtractionError(
            "trafilatura is not installed",
            attempted=["trafilatura"],
            diagnostics={
                "trafilatura": "missing; install with `pip install andamentum[html-articles]` to enable. Falling back to docling.",
            },
        ) from exc

    logging.getLogger("trafilatura").setLevel(logging.ERROR)

    text = trafilatura.extract(
        data,
        url=source_url,
        include_links=True,        # let scoring see link density
        include_tables=True,
        include_images=False,
        favor_recall=False,
        output_format="markdown",
    )
    if not text:
        # Try once more with looser settings before giving up
        text = trafilatura.extract(
            data,
            url=source_url,
            include_links=True,
            include_tables=True,
            include_images=False,
            favor_recall=True,
            output_format="markdown",
        )
    if not text:
        raise ExtractionError(
            "trafilatura returned no content",
            attempted=["trafilatura"],
            diagnostics={"trafilatura": "empty output (likely JS-rendered or paywalled)"},
        )
    return text
