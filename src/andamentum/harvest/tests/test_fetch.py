"""Tests for fetch.resolve and the format-detection chain."""

import gzip
from pathlib import Path

import httpx
import pytest

from andamentum.core.url_safety import (
    ResponseTooLarge,
    SsrfBlocked,
    fetch_with_safe_redirects,
)
from andamentum.harvest.errors import FetchError, UnsupportedFormatError
from andamentum.harvest.fetch import _detect_format, _is_safe_url, resolve


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


# ---------- SSRF: response size cap ----------------------------------------


async def test_response_over_size_cap_is_refused():
    """A terminal body larger than max_bytes raises ResponseTooLarge."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"x" * 5000)

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler), follow_redirects=False
    ) as client:
        with pytest.raises(ResponseTooLarge):
            await fetch_with_safe_redirects(
                client, "http://8.8.8.8/big", max_bytes=1000
            )


async def test_response_within_size_cap_is_returned():
    """A body within max_bytes is returned intact."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"x" * 500)

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler), follow_redirects=False
    ) as client:
        resp = await fetch_with_safe_redirects(
            client, "http://8.8.8.8/ok", max_bytes=1000
        )
    assert resp.status_code == 200
    assert resp.content == b"x" * 500


# ---------- content-encoding: the body must be decoded exactly once ---------
#
# fetch_with_safe_redirects streams the body itself to enforce the size cap.
# ``aiter_bytes`` already runs httpx's content decoder, so the bytes it yields
# are plain text. A rebuilt ``httpx.Response(headers=..., content=body)`` would
# decode them a SECOND time against the surviving ``Content-Encoding: gzip``
# header and raise ``DecodingError: Error -3 while decompressing data:
# incorrect header check`` — which is what every gzip page on the web hit.


def _gzip_client(payload: bytes, *, encoding: str = "gzip") -> httpx.AsyncClient:
    """A client that serves *payload* compressed, exactly as a real server
    does: a streamed body plus a ``Content-Encoding`` header."""
    body = gzip.compress(payload)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers=[
                ("content-encoding", encoding),
                ("content-length", str(len(body))),
                ("content-type", "text/html; charset=utf-8"),
            ],
            stream=httpx.ByteStream(body),
        )

    return httpx.AsyncClient(
        transport=httpx.MockTransport(handler), follow_redirects=False
    )


async def test_gzip_body_is_decoded_once_under_the_size_cap():
    """The capped read path returns the decoded body, not a decode error."""
    payload = b"<html><body>hello gzip</body></html>" * 50
    async with _gzip_client(payload) as client:
        resp = await fetch_with_safe_redirects(
            client, "http://8.8.8.8/page", max_bytes=1_000_000
        )
    assert resp.content == payload
    assert resp.text == payload.decode()


async def test_gzip_body_is_decoded_once_without_a_size_cap():
    """The uncapped read path (max_bytes=None) behaves the same way."""
    payload = b"<html><body>hello gzip</body></html>" * 50
    async with _gzip_client(payload) as client:
        resp = await fetch_with_safe_redirects(
            client, "http://8.8.8.8/page", max_bytes=None
        )
    assert resp.content == payload


async def test_gzip_response_keeps_the_headers_the_server_sent():
    """The returned response reports what the server sent, exactly as a plain
    ``client.get`` does — decoded ``.content`` alongside the original
    ``Content-Encoding``."""
    payload = b"<html><body>hello gzip</body></html>" * 50
    async with _gzip_client(payload) as client:
        resp = await fetch_with_safe_redirects(client, "http://8.8.8.8/page")
    assert resp.headers["content-encoding"] == "gzip"
    assert resp.headers["content-type"] == "text/html; charset=utf-8"
    assert str(resp.url) == "http://8.8.8.8/page"
    assert resp.content == payload


async def test_size_cap_counts_decoded_bytes_not_compressed_bytes():
    """A small compressed body that expands past the cap is refused.

    The cap bounds the memory the body occupies. Capping the compressed size
    instead would let a decompression bomb through the check and blow up
    afterwards.
    """
    payload = b"\x00" * 5_000_000  # compresses to a few kilobytes
    async with _gzip_client(payload) as client:
        with pytest.raises(ResponseTooLarge):
            await fetch_with_safe_redirects(
                client, "http://8.8.8.8/bomb", max_bytes=100_000
            )


async def test_undecodable_content_encoding_is_left_alone():
    """httpx does not decode an encoding it has no decoder for. The body stays
    encoded and the header saying so must survive to describe it."""
    payload = b"<html>brotli</html>"
    async with _gzip_client(payload, encoding="br-unsupported") as client:
        resp = await fetch_with_safe_redirects(client, "http://8.8.8.8/page")
    assert resp.headers["content-encoding"] == "br-unsupported"
    assert resp.content == gzip.compress(payload)
