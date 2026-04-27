"""Text utilities for deep research — topic anchoring + SSRF re-exports.

The SSRF / URL-safety helpers (``is_safe_url``, ``is_internal_ip``,
``SEARXNG_WHITELIST``, ``CLOUD_METADATA_HOSTS``, ``BLOCKED_SCHEMES``,
``ALLOWED_SCHEMES``) now live in ``andamentum.harvest.url_safety``.
They are re-exported here so existing import paths and the runtime
``text_utils.is_safe_url`` monkey-patch in tests keep working.
"""

import logging
import re

from andamentum.harvest.url_safety import (
    ALLOWED_SCHEMES,
    BLOCKED_SCHEMES,
    CLOUD_METADATA_HOSTS,
    SEARXNG_WHITELIST,
    is_internal_ip,
    is_safe_url,
)

__all__ = [
    "ALLOWED_SCHEMES",
    "BLOCKED_SCHEMES",
    "CLOUD_METADATA_HOSTS",
    "SEARXNG_WHITELIST",
    "is_internal_ip",
    "is_safe_url",
    "extract_anchor_terms",
    "guard_query_against_goal",
    "guard_queries_against_drift",
    "STOP_WORDS",
]

# ── Topic Anchoring ─────────────────────────────────────────────────────

STOP_WORDS = {
    "about",
    "after",
    "also",
    "been",
    "being",
    "between",
    "both",
    "could",
    "does",
    "doing",
    "during",
    "each",
    "even",
    "from",
    "have",
    "having",
    "here",
    "into",
    "just",
    "like",
    "make",
    "many",
    "more",
    "most",
    "much",
    "only",
    "other",
    "over",
    "said",
    "same",
    "should",
    "some",
    "such",
    "than",
    "that",
    "their",
    "them",
    "then",
    "there",
    "these",
    "they",
    "this",
    "through",
    "under",
    "very",
    "want",
    "well",
    "were",
    "what",
    "when",
    "where",
    "which",
    "while",
    "will",
    "with",
    "would",
    "your",
    "keep",
    "need",
    "going",
    "know",
    "think",
    "look",
    "looking",
    "help",
    "helps",
    "helping",
    "work",
    "working",
    "result",
    "results",
    "cause",
    "causes",
    "effect",
    "effects",
}


def extract_anchor_terms(text: str) -> list[str]:
    """Extract anchor terms from text for topic validation.

    Returns terms in order of appearance (deterministic). Includes words 4+
    characters, ALL-CAPS acronyms, and gene names with numbers.
    """
    tokens = re.findall(r"\b[A-Za-z0-9]+\b", text)
    seen: set[str] = set()
    anchors: list[str] = []

    for token in tokens:
        is_gene_or_acronym = (token.isupper() and len(token) >= 2) or (
            any(c.isdigit() for c in token) and any(c.isupper() for c in token)
        )
        is_significant = len(token) >= 4 and token.lower() not in STOP_WORDS

        if is_gene_or_acronym:
            key = token.upper()
            if key not in seen:
                seen.add(key)
                anchors.append(token)
        elif is_significant:
            key = token.lower()
            if key not in seen:
                seen.add(key)
                anchors.append(token.lower())

    return anchors


def guard_query_against_goal(
    query: str,
    goal: str,
    logger: logging.Logger | None = None,
) -> str:
    """Validate a search query stays on-topic relative to the research goal.

    # TODO(theme-7): replace with LLM check or remove if planner agent is
    # trustworthy enough. Current regex+stopword approach does fuzzy semantic
    # work deterministically — fragile and likely to produce false positives.
    """
    anchors = extract_anchor_terms(goal)
    if not anchors:
        return query

    core_anchors = anchors[:5]
    core_set = {a.lower() for a in core_anchors}
    query_tokens = re.findall(r"\b[A-Za-z0-9]+\b", query.lower())
    query_words = set(query_tokens)

    if query_words & core_set:
        return query
    else:
        topic_prefix = " ".join(core_anchors[:3])
        repaired = f"{topic_prefix}: {query}"
        if logger:
            logger.warning(
                f"TopicGuard: Repaired drifted query: '{query}' → '{repaired}'"
            )
        return repaired


def guard_queries_against_drift(
    queries: list[str],
    objective: str,
    logger: logging.Logger | None = None,
) -> list[str]:
    """Validate multiple search queries stay on-topic relative to the objective."""
    return [guard_query_against_goal(q, objective, logger) for q in queries]
