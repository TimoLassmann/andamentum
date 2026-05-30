"""Resolve a URL or path into raw bytes + a detected format.

Two responsibilities only:
  1. Bring the bytes into memory (HTTP fetch with SSRF protection, OR file read).
  2. Decide what format the bytes are (PDF / HTML / DOCX / PPTX / Markdown / Plain).

Format detection is intentionally three-tiered:
  a) Path/URL extension (cheap)
  b) HTTP Content-Type header (when available)
  c) Magic-byte sniff on the bytes themselves (last resort)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse

import httpx

from andamentum.core.fetch_gate import (
    PaywallBlocked,
    RobotsBlocked,
    check_fetch_allowed,
    user_agent_for,
)

from andamentum.core.url_safety import (
    ResponseTooLarge,
    SsrfBlocked,
    fetch_with_safe_redirects,
    is_safe_url,
)

from .errors import FetchError, UnsupportedFormatError

# Identifies andamentum-harvest to remote hosts so abuse-desks can contact the
# project rather than block the netblock. Per RFC 9110 §10.1.5.
_USER_AGENT = user_agent_for("harvest")

Format = Literal["pdf", "html", "docx", "pptx", "markdown", "plain"]

# Known file extensions → format. Lower-case, no dot.
_EXT_TO_FORMAT: dict[str, Format] = {
    "pdf": "pdf",
    "html": "html",
    "htm": "html",
    "xhtml": "html",
    "docx": "docx",
    "pptx": "pptx",
    "md": "markdown",
    "markdown": "markdown",
    "txt": "plain",
    "rst": "plain",
}

# HTTP Content-Type prefixes → format. Order matters (more specific first).
_MIME_TO_FORMAT: list[tuple[str, Format]] = [
    ("application/pdf", "pdf"),
    ("application/vnd.openxmlformats-officedocument.wordprocessingml.document", "docx"),
    (
        "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        "pptx",
    ),
    ("application/msword", "docx"),
    ("application/vnd.ms-powerpoint", "pptx"),
    ("text/html", "html"),
    ("application/xhtml+xml", "html"),
    ("text/markdown", "markdown"),
    ("text/x-markdown", "markdown"),
    ("text/plain", "plain"),
]


@dataclass
class Fetched:
    """Raw bytes + detected format + provenance string for diagnostics."""

    data: bytes
    format: Format
    source_url: str  # the URL or "file://..." that produced these bytes


# ----------------------------------------------------------------------------
# SSRF protection — thin alias preserving the in-module name used by tests
# and the fetch path. Implementation lives in ``core.url_safety``.
# ----------------------------------------------------------------------------


def _is_safe_url(url: str) -> tuple[bool, str]:
    """Block URLs that resolve to private/loopback IPs (basic SSRF protection).

    Delegates to ``core.url_safety.is_safe_url`` so harvest and
    deep_research share one implementation.
    """
    return is_safe_url(url)


# ----------------------------------------------------------------------------
# Public entry: resolve()
# ----------------------------------------------------------------------------


async def resolve(
    source: str | Path,
    *,
    tdm_allowed_hosts: frozenset[str] = frozenset(),
) -> Fetched:
    """Bring `source` into memory and detect its format.

    Accepts:
      - http(s):// URL  → httpx GET (SSRF-protected, robots.txt + paywall gated)
      - file:// URI     → local file read
      - filesystem path → local file read (str or pathlib.Path)

    Parameters
    ----------
    source:
        URL or file path.
    tdm_allowed_hosts:
        Hostnames the caller attests they hold a text-and-data-mining
        licence for. A match here disarms the paywall tripwire for that
        host. Default empty: tripwire fully active.
    """
    if isinstance(source, Path):
        return _read_file(source)

    if not isinstance(source, str):
        raise FetchError(f"unsupported source type: {type(source).__name__}")

    if source.startswith(("http://", "https://")):
        return await _fetch_url(source, tdm_allowed_hosts=tdm_allowed_hosts)
    if source.startswith("file://"):
        # Strip "file://" prefix and treat as a path
        return _read_file(Path(source[len("file://") :]))

    # No scheme → assume filesystem path
    return _read_file(Path(source))


# ----------------------------------------------------------------------------
# URL fetch
# ----------------------------------------------------------------------------


async def _fetch_url(
    url: str,
    *,
    tdm_allowed_hosts: frozenset[str] = frozenset(),
) -> Fetched:
    safe, reason = _is_safe_url(url)
    if not safe:
        raise FetchError(f"URL blocked: {reason} ({url})")

    try:
        # follow_redirects is False at the client level: the main GET drives
        # the redirect chain through fetch_with_safe_redirects (which
        # re-validates every hop), and the robots.txt fetch that
        # check_fetch_allowed issues on this same client therefore does not
        # silently chase a cross-host redirect either.
        async with httpx.AsyncClient(
            timeout=30.0,
            follow_redirects=False,
            headers={"User-Agent": _USER_AGENT},
        ) as client:
            try:
                await check_fetch_allowed(
                    url,
                    user_agent=_USER_AGENT,
                    tdm_allowed_hosts=tdm_allowed_hosts,
                    client=client,
                )
            except (PaywallBlocked, RobotsBlocked) as exc:
                raise FetchError(str(exc)) from exc

            resp = await fetch_with_safe_redirects(client, url)
            resp.raise_for_status()
    except (SsrfBlocked, ResponseTooLarge) as exc:
        raise FetchError(str(exc)) from exc
    except httpx.HTTPError as exc:
        raise FetchError(f"HTTP fetch failed for {url}: {exc}") from exc

    data = resp.content
    fmt = _detect_format(
        url=str(resp.url),
        content_type=resp.headers.get("content-type", ""),
        data=data,
    )
    return Fetched(data=data, format=fmt, source_url=str(resp.url))


# ----------------------------------------------------------------------------
# Local file read
# ----------------------------------------------------------------------------


def _read_file(path: Path) -> Fetched:
    if not path.exists():
        raise FetchError(f"file not found: {path}")
    if not path.is_file():
        raise FetchError(f"not a regular file: {path}")
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise FetchError(f"could not read file {path}: {exc}") from exc
    fmt = _detect_format(
        url=path.resolve().as_uri(),
        content_type="",
        data=data,
        path=path,
    )
    return Fetched(data=data, format=fmt, source_url=path.resolve().as_uri())


# ----------------------------------------------------------------------------
# Format detection
# ----------------------------------------------------------------------------


def _detect_format(
    *,
    url: str,
    content_type: str,
    data: bytes,
    path: Path | None = None,
) -> Format:
    """Detect format using extension → mime → magic-byte sniff, in that order."""
    # 1) Extension on the path or URL path
    ext_source = path.suffix if path is not None else urlparse(url).path
    ext = ext_source.rsplit(".", 1)[-1].lower() if "." in ext_source else ""
    if ext in _EXT_TO_FORMAT:
        return _EXT_TO_FORMAT[ext]

    # 2) MIME type
    if content_type:
        mime = content_type.split(";", 1)[0].strip().lower()
        for prefix, fmt in _MIME_TO_FORMAT:
            if mime == prefix:
                return fmt

    # 3) Magic-byte sniff
    sniff = _sniff_magic(data)
    if sniff is not None:
        return sniff

    # Nothing matched — explicit error so the caller doesn't get a wrong-format
    # extraction silently.
    raise UnsupportedFormatError(
        f"could not detect format for {url} "
        f"(extension={ext!r}, content_type={content_type!r}, first_bytes={data[:16]!r})"
    )


def _sniff_magic(data: bytes) -> Format | None:
    """Best-effort magic-byte sniff. Returns None if nothing matches."""
    if not data:
        return None
    head = data[:512]
    if head.startswith(b"%PDF"):
        return "pdf"
    if head[:4] == b"PK\x03\x04":
        # Generic ZIP container — could be DOCX or PPTX. Look further inside.
        if b"word/" in data[:4096]:
            return "docx"
        if b"ppt/" in data[:4096]:
            return "pptx"
        # Unknown ZIP — don't guess.
        return None
    head_lower = head.lstrip().lower()
    if head_lower.startswith((b"<!doctype html", b"<html", b"<head>", b"<body>")):
        return "html"
    # Heuristic: if it's mostly printable ASCII, treat as plain text.
    try:
        decoded = head.decode("utf-8")
    except UnicodeDecodeError:
        return None
    printable = sum(1 for c in decoded if c.isprintable() or c in "\n\t\r")
    if printable / max(len(decoded), 1) > 0.95:
        # Distinguish markdown vs plain by looking for markdown signals
        if any(
            line.startswith(("#", "- ", "* ", "1.")) for line in decoded.splitlines()
        ):
            return "markdown"
        return "plain"
    return None
