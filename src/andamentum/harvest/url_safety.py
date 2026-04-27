"""Single home for SSRF / URL-safety checks used across andamentum.

Combines the strengths of the two pre-consolidation implementations:
- DNS resolution + private/loopback/reserved/multicast IP check (was harvest)
- Cloud metadata blocklist, hostname-pattern blocklist, SearXNG whitelist
  (was deep_research)

Why it lives in ``harvest``: harvest is the leaf service responsible for
fetching bytes from URLs, so the safety check naturally belongs alongside
the fetcher. Other modules (deep_research) MAY depend on harvest per the
layering rule.
"""

from __future__ import annotations

import ipaddress
import re
import socket
from urllib.parse import urlparse

# Default-allowed local endpoints when callers opt in via ``allow_searxng=True``.
SEARXNG_WHITELIST = frozenset({"localhost:4070", "127.0.0.1:4070"})

# Cloud-provider metadata endpoints — never safe to fetch from a server-side
# fetcher; they expose IAM tokens, instance secrets, etc.
CLOUD_METADATA_HOSTS = frozenset(
    {
        "169.254.169.254",
        "metadata.google.internal",
        "metadata.google.com",
        "metadata",
        "169.254.0.1",
    }
)

BLOCKED_SCHEMES = frozenset({"file", "ftp", "gopher", "data", "javascript"})
ALLOWED_SCHEMES = frozenset({"http", "https"})

_LOCALHOST_RE = re.compile(
    r"^(localhost|127\.\d+\.\d+\.\d+|::1|\[::1\])$", re.IGNORECASE
)
_INTERNAL_HOST_PATTERNS = (
    re.compile(r"^internal\.", re.IGNORECASE),
    re.compile(r"^private\.", re.IGNORECASE),
    re.compile(r"^local\.", re.IGNORECASE),
    re.compile(r"\.internal$", re.IGNORECASE),
    re.compile(r"\.local$", re.IGNORECASE),
    re.compile(r"\.localhost$", re.IGNORECASE),
)


def is_internal_ip(ip_str: str) -> bool:
    """True if ``ip_str`` parses to a private / loopback / link-local / reserved /
    multicast IP. Returns False for non-IP strings (i.e. hostnames)."""
    try:
        ip = ipaddress.ip_address(ip_str)
    except ValueError:
        return False
    return (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_reserved
        or ip.is_multicast
    )


def is_safe_url(url: str, *, allow_searxng: bool = False) -> tuple[bool, str]:
    """Validate a URL for SSRF protection.

    Performs (in order, cheapest-first):
      1. Scheme allow/blocklist (must be http or https)
      2. SearXNG whitelist (only when ``allow_searxng=True``)
      3. Cloud-metadata hostname blocklist
      4. Localhost / internal hostname-pattern blocklist
      5. Direct private-IP check on the host (catches literal IPs)
      6. DNS resolution → private-IP check on the resolved address

    Args:
        url: URL to validate.
        allow_searxng: When True, allow whitelisted local SearXNG endpoints.
            Used by deep_research's search backend; harvest never sets this.

    Returns:
        ``(is_safe, reason)``. ``reason`` is empty on success and a short
        diagnostic string on failure.
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
    if not host:
        return False, "URL has no hostname"

    port = parsed.port
    host_with_port = f"{host}:{port}" if port else host

    if allow_searxng and host_with_port in SEARXNG_WHITELIST:
        return True, ""

    if host.lower() in CLOUD_METADATA_HOSTS:
        return False, f"Blocked cloud metadata endpoint: {host}"

    if _LOCALHOST_RE.match(host):
        return False, f"Blocked localhost: {host}"

    for pattern in _INTERNAL_HOST_PATTERNS:
        if pattern.search(host):
            return False, f"Blocked internal hostname pattern: {host}"

    # Catch literal-IP hosts before paying for DNS.
    if is_internal_ip(host):
        return False, f"Blocked internal/private IP: {host}"

    # Last line of defence: a public-looking hostname that DNS-resolves
    # to a private IP (e.g. attacker-controlled DNS rebinding).
    try:
        resolved = socket.gethostbyname(host)
    except socket.gaierror as exc:
        return False, f"DNS resolution failed: {exc}"
    if is_internal_ip(resolved):
        return False, f"hostname resolves to non-public IP {resolved}"

    return True, ""
