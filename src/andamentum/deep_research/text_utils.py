"""Text utilities for deep research — SSRF protection and topic anchoring.

Standalone copies of utilities that were previously in src/utilities/.
Kept small and independent so the package has no dependency on mosaic.
"""

import ipaddress
import logging
import re
from urllib.parse import urlparse

# ── SSRF Protection ─────────────────────────────────────────────────────

SEARXNG_WHITELIST = frozenset({"localhost:4070", "127.0.0.1:4070"})

CLOUD_METADATA_HOSTS = frozenset({
    "169.254.169.254",
    "metadata.google.internal",
    "metadata.google.com",
    "metadata",
    "169.254.0.1",
})

BLOCKED_SCHEMES = frozenset({"file", "ftp", "gopher", "data", "javascript"})
ALLOWED_SCHEMES = frozenset({"http", "https"})


def is_internal_ip(ip_str: str) -> bool:
    """Check if an IP address is internal/private."""
    try:
        ip = ipaddress.ip_address(ip_str)
        return ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast
    except ValueError:
        return False


def is_safe_url(url: str, allow_searxng: bool = False) -> tuple[bool, str]:
    """Validate URL for SSRF protection.

    Args:
        url: URL to validate
        allow_searxng: If True, allow whitelisted SearXNG endpoints

    Returns:
        Tuple of (is_safe, reason)
    """
    if not url:
        return False, "Empty URL"

    try:
        parsed = urlparse(url)
    except Exception as e:
        return False, f"Invalid URL format: {e}"

    scheme = parsed.scheme.lower()
    if not scheme:
        return False, "Missing URL scheme"
    if scheme in BLOCKED_SCHEMES:
        return False, f"Blocked scheme: {scheme}"
    if scheme not in ALLOWED_SCHEMES:
        return False, f"Unsupported scheme: {scheme}"

    host = parsed.hostname or ""
    port = parsed.port
    host_with_port = f"{host}:{port}" if port else host

    if allow_searxng and host_with_port in SEARXNG_WHITELIST:
        return True, ""

    if host.lower() in CLOUD_METADATA_HOSTS:
        return False, f"Blocked cloud metadata endpoint: {host}"

    if is_internal_ip(host):
        return False, f"Blocked internal/private IP: {host}"

    if re.match(r"^(localhost|127\.\d+\.\d+\.\d+|::1|\[::1\])$", host, re.IGNORECASE):
        return False, f"Blocked localhost: {host}"

    internal_patterns = [
        r"^internal\.", r"^private\.", r"^local\.",
        r"\.internal$", r"\.local$", r"\.localhost$",
    ]
    for pattern in internal_patterns:
        if re.search(pattern, host, re.IGNORECASE):
            return False, f"Blocked internal hostname pattern: {host}"

    return True, ""


# ── Topic Anchoring ─────────────────────────────────────────────────────

STOP_WORDS = {
    "about", "after", "also", "been", "being", "between", "both", "could",
    "does", "doing", "during", "each", "even", "from", "have", "having",
    "here", "into", "just", "like", "make", "many", "more", "most", "much",
    "only", "other", "over", "said", "same", "should", "some", "such",
    "than", "that", "their", "them", "then", "there", "these", "they",
    "this", "through", "under", "very", "want", "well", "were", "what",
    "when", "where", "which", "while", "will", "with", "would", "your",
    "keep", "need", "going", "know", "think", "look", "looking", "help",
    "helps", "helping", "work", "working", "result", "results", "cause",
    "causes", "effect", "effects",
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
    """Validate a search query stays on-topic relative to the research goal."""
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
            logger.warning(f"TopicGuard: Repaired drifted query: '{query}' → '{repaired}'")
        return repaired


def guard_queries_against_drift(
    queries: list[str],
    objective: str,
    logger: logging.Logger | None = None,
) -> list[str]:
    """Validate multiple search queries stay on-topic relative to the objective."""
    return [guard_query_against_goal(q, objective, logger) for q in queries]
