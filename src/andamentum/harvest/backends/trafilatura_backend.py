"""HTML → markdown via trafilatura.

Article-optimised: trafilatura strips nav/ads/cookie banners aggressively
and emits clean prose with `##` headings preserved. Excellent on real
articles; near-useless on link-card homepages (returns one structureless
run of text).
"""

from __future__ import annotations

import logging

from ..errors import ExtractionError


async def extract(data: bytes, source_url: str) -> str:
    """Run trafilatura on raw HTML bytes; return markdown.

    Raises ExtractionError if trafilatura returned no content (typical for
    JavaScript-rendered or paywalled pages).
    """
    # Trafilatura is sync; the cost is small enough we don't bother with
    # to_thread. Logger noise from missing-link-attribute warnings on every
    # anchor is suppressed per request.
    import trafilatura

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
