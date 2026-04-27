"""Sniff HTML page metadata to decide if a page is an article or not.

Cheap heuristic based on signals every major news/blog site emits:
  - <meta property="og:type" content="article">
  - <script type="application/ld+json"> with "@type": "Article"|"NewsArticle"|"BlogPosting"
  - <article> semantic tag

Also recognises explicit non-article markers (CollectionPage, ItemList,
WebPage) so the orchestrator can route a homepage straight to Docling
without bothering trafilatura.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Literal

ArticleVerdict = Literal["article", "not_article", "ambiguous"]

# Schema.org @types that identify article-like content.
_ARTICLE_TYPES = {"Article", "NewsArticle", "BlogPosting", "ScholarlyArticle", "Report"}
# Schema.org @types that identify clearly non-article content (lists, indexes).
_NON_ARTICLE_TYPES = {
    "WebPage",
    "CollectionPage",
    "ItemList",
    "SearchResultsPage",
    "ProfilePage",
}

_OG_TYPE_RE = re.compile(
    rb'<meta\s+[^>]*property=["\']og:type["\'][^>]*content=["\']([^"\']+)["\']',
    re.IGNORECASE,
)
_LD_JSON_RE = re.compile(
    rb'<script[^>]*type=["\']application/ld\+json["\'][^>]*>(.*?)</script>',
    re.IGNORECASE | re.DOTALL,
)
_ARTICLE_TAG_RE = re.compile(rb"<article[\s>]", re.IGNORECASE)


@dataclass
class PageMetadata:
    """Result of metadata sniffing on raw HTML bytes."""

    verdict: ArticleVerdict
    og_type: str | None = None
    ld_json_type: str | None = None
    has_article_tag: bool = False
    reason: str = ""


def sniff_html_metadata(html: bytes) -> PageMetadata:
    """Decide if a page looks like an article based on its HTML metadata.

    Returns a PageMetadata; the `verdict` field is the actionable answer:
      - "article"      → route to trafilatura (article-optimised extractor)
      - "not_article"  → route to Docling (preserves layout for indexes)
      - "ambiguous"    → race both extractors and pick the highest-scoring output
    """
    og_type = _find_og_type(html)
    ld_type = _find_first_ld_json_type(html)
    has_article_tag = bool(_ARTICLE_TAG_RE.search(html))

    # Strongest signal first: og:type.
    if og_type:
        if og_type.lower() == "article":
            return PageMetadata(
                verdict="article",
                og_type=og_type,
                ld_json_type=ld_type,
                has_article_tag=has_article_tag,
                reason="og:type=article",
            )
        # og:type=website is the typical homepage marker — but not strong enough
        # alone to declare "not_article" if other article signals exist.

    # Then JSON-LD @type.
    if ld_type:
        if ld_type in _ARTICLE_TYPES:
            return PageMetadata(
                verdict="article",
                og_type=og_type,
                ld_json_type=ld_type,
                has_article_tag=has_article_tag,
                reason=f"JSON-LD @type={ld_type}",
            )
        if ld_type in _NON_ARTICLE_TYPES:
            return PageMetadata(
                verdict="not_article",
                og_type=og_type,
                ld_json_type=ld_type,
                has_article_tag=has_article_tag,
                reason=f"JSON-LD @type={ld_type}",
            )

    # The <article> tag is HTML5's semantic marker for article content. When
    # present without any contradicting metadata, treat it as a positive
    # signal — a page that explicitly declares <article> almost always wants
    # the article-optimised extractor.
    if has_article_tag and not og_type and not ld_type:
        return PageMetadata(
            verdict="article",
            og_type=og_type,
            ld_json_type=ld_type,
            has_article_tag=True,
            reason="<article> tag present (HTML5 semantic marker)",
        )

    # No signal either way.
    return PageMetadata(
        verdict="ambiguous",
        og_type=og_type,
        ld_json_type=ld_type,
        has_article_tag=has_article_tag,
        reason="no decisive metadata",
    )


def _find_og_type(html: bytes) -> str | None:
    m = _OG_TYPE_RE.search(html)
    return m.group(1).decode("utf-8", errors="replace") if m else None


def _find_first_ld_json_type(html: bytes) -> str | None:
    """Walk JSON-LD blocks and return the first @type that's a known marker.

    A page may have multiple JSON-LD blocks (Organization, BreadcrumbList,
    Article…). We scan all of them and return the first @type we recognise
    as either article-like or non-article-like — preferring informative
    types over generic ones like Organization.
    """
    for m in _LD_JSON_RE.finditer(html):
        try:
            data = json.loads(m.group(1))
        except (ValueError, json.JSONDecodeError):
            continue
        for type_name in _walk_types(data):
            if type_name in _ARTICLE_TYPES or type_name in _NON_ARTICLE_TYPES:
                return type_name
    return None


def _walk_types(node: object) -> list[str]:
    """Pull every @type value out of a (possibly nested) JSON-LD structure."""
    out: list[str] = []
    if isinstance(node, dict):
        t = node.get("@type")
        if isinstance(t, str):
            out.append(t)
        elif isinstance(t, list):
            out.extend(x for x in t if isinstance(x, str))
        # Recurse into all values
        for v in node.values():
            out.extend(_walk_types(v))
    elif isinstance(node, list):
        for item in node:
            out.extend(_walk_types(item))
    return out
