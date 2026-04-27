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

import ipaddress
import socket
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse

import httpx

from .errors import FetchError, UnsupportedFormatError

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
    ("application/vnd.openxmlformats-officedocument.presentationml.presentation", "pptx"),
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
# SSRF protection
# ----------------------------------------------------------------------------


def _is_safe_url(url: str) -> tuple[bool, str]:
    """Block URLs that resolve to private/loopback IPs (basic SSRF protection).

    Mirrors the contract used by deep_research's text_utils.is_safe_url so
    behaviour is consistent across modules — but reimplemented here so we
    don't import from deep_research (layering rule).
    """
    try:
        parsed = urlparse(url)
    except Exception as exc:
        return False, f"URL parse error: {exc}"
    if parsed.scheme not in ("http", "https"):
        return False, f"unsupported scheme: {parsed.scheme!r}"
    if not parsed.hostname:
        return False, "URL has no hostname"
    try:
        ip = socket.gethostbyname(parsed.hostname)
    except socket.gaierror as exc:
        return False, f"DNS resolution failed: {exc}"
    try:
        addr = ipaddress.ip_address(ip)
    except ValueError as exc:
        return False, f"invalid IP from DNS: {exc}"
    if (
        addr.is_private
        or addr.is_loopback
        or addr.is_link_local
        or addr.is_reserved
        or addr.is_multicast
    ):
        return False, f"hostname resolves to non-public IP {ip}"
    return True, "ok"


# ----------------------------------------------------------------------------
# Public entry: resolve()
# ----------------------------------------------------------------------------


async def resolve(source: str | Path) -> Fetched:
    """Bring `source` into memory and detect its format.

    Accepts:
      - http(s):// URL  → httpx GET (SSRF-protected)
      - file:// URI     → local file read
      - filesystem path → local file read (str or pathlib.Path)
    """
    if isinstance(source, Path):
        return _read_file(source)

    if not isinstance(source, str):
        raise FetchError(f"unsupported source type: {type(source).__name__}")

    if source.startswith(("http://", "https://")):
        return await _fetch_url(source)
    if source.startswith("file://"):
        # Strip "file://" prefix and treat as a path
        return _read_file(Path(source[len("file://") :]))

    # No scheme → assume filesystem path
    return _read_file(Path(source))


# ----------------------------------------------------------------------------
# URL fetch
# ----------------------------------------------------------------------------


async def _fetch_url(url: str) -> Fetched:
    safe, reason = _is_safe_url(url)
    if not safe:
        raise FetchError(f"URL blocked: {reason} ({url})")

    try:
        async with httpx.AsyncClient(
            timeout=30.0,
            follow_redirects=True,
            headers={"User-Agent": "andamentum-harvest/0.1"},
        ) as client:
            resp = await client.get(url)
            resp.raise_for_status()
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
        if any(line.startswith(("#", "- ", "* ", "1.")) for line in decoded.splitlines()):
            return "markdown"
        return "plain"
    return None
