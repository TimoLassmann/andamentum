"""Text utilities for deep research — SSRF re-exports.

The SSRF / URL-safety helpers (``is_safe_url``, ``is_internal_ip``,
``SEARXNG_WHITELIST``, ``CLOUD_METADATA_HOSTS``, ``BLOCKED_SCHEMES``,
``ALLOWED_SCHEMES``) live in ``andamentum.core.url_safety``. They are
re-exported here so existing import paths and the runtime
``text_utils.is_safe_url`` monkey-patch in tests keep working.

The previous regex+stopword "topic guard"
(``guard_query_against_goal``/``guard_queries_against_drift``/
``extract_anchor_terms``/``STOP_WORDS``) has been retired. Topic checking
is now an LLM call (``topic_verifier`` agent, see
``agents/topic_verifier.py``) wired into the per-slot generate→verify
loop in ``nodes.py``.
"""

from andamentum.core.url_safety import (
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
]
