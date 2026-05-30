"""Tests for fetch.resolve and the format-detection chain."""

from pathlib import Path

import httpx
import pytest

from andamentum.harvest.errors import FetchError, UnsupportedFormatError
from andamentum.harvest.fetch import _detect_format, _is_safe_url, resolve
from andamentum.harvest.url_safety import SsrfBlocked, fetch_with_safe_redirects


# ---------- format detection -----------------------------------------------


def test_detect_format_via_extension_pdf():
    assert (
        _detect_format(url="https://x.com/paper.pdf", content_type="", data=b"")
        == "pdf"
    )


def test_detect_format_via_extension_html():
    assert (
        _detect_format(url="https://x.com/article.html", content_type="", data=b"")
        == "html"
    )


def test_detect_format_via_extension_markdown():
    assert (
        _detect_format(url="https://x.com/notes.md", content_type="", data=b"")
        == "markdown"
    )


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
    assert (
        _detect_format(url="https://x.com/r", content_type="", data=b"%PDF-1.4 ...")
        == "pdf"
    )


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
    assert (
        "127" in reason
        or "loopback" in reason.lower()
        or "non-public" in reason.lower()
    )


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


# ---------- SSRF: redirect re-validation -----------------------------------
#
# is_safe_url validates the INITIAL url. The regression these tests guard is
# a public-looking URL that 3xx-redirects to a private/loopback/cloud-metadata
# address: fetch_with_safe_redirects must re-check every hop and refuse it.


def _redirect_client(location: str) -> httpx.AsyncClient:
    """A client whose first GET (to the public start IP) 302s to *location*,
    and which serves 200 OK for anything else."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "1.1.1.1":
            return httpx.Response(302, headers={"location": location})
        return httpx.Response(200, text="final content")

    # follow_redirects=False mirrors how harvest/deep_research configure their
    # real clients; the helper drives the hop chain itself regardless.
    return httpx.AsyncClient(
        transport=httpx.MockTransport(handler), follow_redirects=False
    )


@pytest.mark.parametrize(
    "blocked_target",
    [
        "http://169.254.169.254/latest/meta-data/",  # cloud metadata
        "http://127.0.0.1/admin",  # loopback
        "http://10.0.0.5/internal",  # RFC-1918 private
    ],
)
async def test_redirect_to_blocked_address_is_refused(blocked_target: str):
    """A 302 from a public host to a private/metadata host must raise."""
    async with _redirect_client(blocked_target) as client:
        with pytest.raises(SsrfBlocked):
            await fetch_with_safe_redirects(client, "http://1.1.1.1/start")


async def test_redirect_to_public_address_is_followed():
    """A 302 between two public hosts is followed to the final response."""
    async with _redirect_client("http://8.8.8.8/ok") as client:
        resp = await fetch_with_safe_redirects(client, "http://1.1.1.1/start")
    assert resp.status_code == 200
    assert resp.text == "final content"


async def test_redirect_loop_is_bounded():
    """A self-redirecting public host is capped, not followed forever."""

    def handler(request: httpx.Request) -> httpx.Response:
        # Always redirect to another public IP → never terminates on its own.
        return httpx.Response(302, headers={"location": "http://8.8.8.8/loop"})

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler), follow_redirects=False
    ) as client:
        with pytest.raises(SsrfBlocked):
            await fetch_with_safe_redirects(
                client, "http://8.8.8.8/loop", max_redirects=3
            )
