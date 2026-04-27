"""Tests for fetch.resolve and the format-detection chain."""

from pathlib import Path

import pytest

from andamentum.harvest.errors import FetchError, UnsupportedFormatError
from andamentum.harvest.fetch import _detect_format, _is_safe_url, resolve


# ---------- format detection -----------------------------------------------


def test_detect_format_via_extension_pdf():
    assert _detect_format(url="https://x.com/paper.pdf", content_type="", data=b"") == "pdf"


def test_detect_format_via_extension_html():
    assert _detect_format(url="https://x.com/article.html", content_type="", data=b"") == "html"


def test_detect_format_via_extension_markdown():
    assert _detect_format(url="https://x.com/notes.md", content_type="", data=b"") == "markdown"


def test_detect_format_via_mime_pdf():
    assert (
        _detect_format(url="https://x.com/r", content_type="application/pdf", data=b"")
        == "pdf"
    )


def test_detect_format_via_mime_html_with_charset():
    """Content-Type often has `; charset=utf-8` — must still match."""
    assert (
        _detect_format(
            url="https://x.com/r", content_type="text/html; charset=utf-8", data=b""
        )
        == "html"
    )


def test_detect_format_via_magic_bytes_pdf():
    assert _detect_format(url="https://x.com/r", content_type="", data=b"%PDF-1.4 ...") == "pdf"


def test_detect_format_via_magic_bytes_html():
    assert (
        _detect_format(
            url="https://x.com/r", content_type="", data=b"<!DOCTYPE html><html>"
        )
        == "html"
    )


def test_detect_format_unknown_raises():
    with pytest.raises(UnsupportedFormatError):
        _detect_format(url="https://x.com/r", content_type="", data=b"\x01\x02\x03\x04")


# ---------- SSRF protection ------------------------------------------------


def test_is_safe_url_blocks_localhost():
    ok, reason = _is_safe_url("http://127.0.0.1/")
    assert not ok
    assert "127" in reason or "loopback" in reason.lower() or "non-public" in reason.lower()


def test_is_safe_url_blocks_private_range():
    ok, reason = _is_safe_url("http://10.0.0.1/")
    assert not ok


def test_is_safe_url_blocks_unsupported_scheme():
    ok, reason = _is_safe_url("ftp://example.com/file")
    assert not ok
    assert "scheme" in reason.lower()


def test_is_safe_url_passes_public_host():
    """example.com is reserved-looking but is_global per RFC 6890; ensure we accept it.

    We use a hostname that always resolves to a public IP. Skip if DNS down.
    """
    ok, reason = _is_safe_url("https://example.com/")
    # Either it's safe OR DNS is unavailable — both are OK in this test.
    assert ok or "DNS" in reason


# ---------- resolve() with file paths --------------------------------------


async def test_resolve_local_file_md(tmp_path: Path):
    p = tmp_path / "notes.md"
    p.write_text("# Title\n\nbody\n")
    fetched = await resolve(p)
    assert fetched.format == "markdown"
    assert fetched.data == p.read_bytes()
    assert fetched.source_url.startswith("file://")


async def test_resolve_local_file_via_string_path(tmp_path: Path):
    p = tmp_path / "doc.pdf"
    p.write_bytes(b"%PDF-1.4 stub")
    fetched = await resolve(str(p))
    assert fetched.format == "pdf"


async def test_resolve_missing_file_raises(tmp_path: Path):
    with pytest.raises(FetchError):
        await resolve(tmp_path / "does-not-exist.pdf")


async def test_resolve_directory_raises(tmp_path: Path):
    with pytest.raises(FetchError):
        await resolve(tmp_path)
