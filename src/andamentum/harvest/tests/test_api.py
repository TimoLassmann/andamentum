"""End-to-end tests for harvest.extract — orchestration with stubbed backends."""

from pathlib import Path

import pytest

from andamentum.harvest import api as api_mod
from andamentum.harvest import extract
from andamentum.harvest.errors import ExtractionError
from andamentum.harvest.fetch import Fetched


# ---------- helpers --------------------------------------------------------


def _make_fetched(
    *, fmt: str, data: bytes = b"", url: str = "https://example.com"
) -> Fetched:
    return Fetched(data=data, format=fmt, source_url=url)  # type: ignore[arg-type]


@pytest.fixture
def patched_resolve(monkeypatch):
    """Patch fetch.resolve so tests don't need network or filesystem access."""

    holder: dict[str, Fetched] = {}

    async def fake_resolve(source, **_kwargs):
        return holder["fetched"]

    monkeypatch.setattr(api_mod, "resolve", fake_resolve)

    def _set(fetched: Fetched) -> Fetched:
        holder["fetched"] = fetched
        return fetched

    return _set


@pytest.fixture
def patched_backends(monkeypatch):
    """Patch the three backend functions; capture call sites + return canned MD."""

    state = {
        "trafilatura_md": "## Article\n\nbody\n",
        "trafilatura_calls": 0,
        "trafilatura_raises": None,
        "docling_md": "## Layout\n\nbody\n",
        "docling_calls": 0,
        "docling_raises": None,
        "plain_md": "plain text",
        "plain_calls": 0,
    }

    async def fake_traf(data, source_url):
        state["trafilatura_calls"] += 1
        if state["trafilatura_raises"]:
            raise state["trafilatura_raises"]
        return state["trafilatura_md"]

    async def fake_docl(data, source_url, fmt="html"):
        state["docling_calls"] += 1
        if state["docling_raises"]:
            raise state["docling_raises"]
        return state["docling_md"]

    async def fake_plain(data, source_url):
        state["plain_calls"] += 1
        return state["plain_md"]

    monkeypatch.setattr(api_mod, "extract_with_trafilatura", fake_traf)
    monkeypatch.setattr(api_mod, "extract_with_docling", fake_docl)
    monkeypatch.setattr(api_mod, "extract_passthrough", fake_plain)

    return state


# ---------- format dispatch ------------------------------------------------


async def test_pdf_routes_to_docling_only(patched_resolve, patched_backends):
    patched_resolve(_make_fetched(fmt="pdf", data=b"%PDF-1.4"))
    md = await extract("https://x.com/p.pdf")
    assert md == "## Layout\n\nbody\n"
    assert patched_backends["docling_calls"] == 1
    assert patched_backends["trafilatura_calls"] == 0


async def test_docx_routes_to_docling(patched_resolve, patched_backends):
    patched_resolve(_make_fetched(fmt="docx"))
    await extract("https://x.com/p.docx")
    assert patched_backends["docling_calls"] == 1


async def test_markdown_routes_to_passthrough(patched_resolve, patched_backends):
    patched_resolve(_make_fetched(fmt="markdown", data=b"## hi"))
    md = await extract("https://x.com/p.md")
    assert md == "plain text"
    assert patched_backends["plain_calls"] == 1
    assert patched_backends["docling_calls"] == 0


# ---------- HTML metadata-driven dispatch ----------------------------------


async def test_html_with_article_metadata_uses_trafilatura_only(
    patched_resolve, patched_backends
):
    html = b'<html><head><meta property="og:type" content="article"></head></html>'
    patched_resolve(_make_fetched(fmt="html", data=html))
    md = await extract("https://x.com/article")
    assert md == "## Article\n\nbody\n"
    assert patched_backends["trafilatura_calls"] == 1
    assert patched_backends["docling_calls"] == 0


async def test_html_with_webpage_metadata_uses_docling_only(
    patched_resolve, patched_backends
):
    html = b'<html><head><script type="application/ld+json">{"@type":"WebPage"}</script></head></html>'
    patched_resolve(_make_fetched(fmt="html", data=html))
    await extract("https://x.com/index")
    assert patched_backends["docling_calls"] == 1
    assert patched_backends["trafilatura_calls"] == 0


async def test_html_ambiguous_races_both_and_picks_winner(
    patched_resolve, patched_backends
):
    """No metadata → race. Docling output has more headings, so docling wins."""
    html = b"<html><body><p>no metadata</p></body></html>"
    patched_resolve(_make_fetched(fmt="html", data=html))
    patched_backends["trafilatura_md"] = "flat text no headings " * 100
    patched_backends["docling_md"] = "## A\n\nx\n\n## B\n\ny\n\n## C\n\nz\n"
    md = await extract("https://x.com/?")
    assert md == patched_backends["docling_md"]
    assert patched_backends["trafilatura_calls"] == 1
    assert patched_backends["docling_calls"] == 1


async def test_html_ambiguous_picks_trafilatura_when_it_wins(
    patched_resolve, patched_backends
):
    html = b"<html><body><p>no metadata</p></body></html>"
    patched_resolve(_make_fetched(fmt="html", data=html))
    patched_backends["trafilatura_md"] = (
        "## A\n\nbody\n\n## B\n\nbody\n\n## C\n\nbody\n"
    )
    # Make docling output structureless so it gets disqualified
    patched_backends["docling_md"] = "x" * 5000  # no headings, no \n\n
    md = await extract("https://x.com/?")
    assert md == patched_backends["trafilatura_md"]


async def test_html_race_raises_when_both_fail(patched_resolve, patched_backends):
    html = b"<html><body><p>no metadata</p></body></html>"
    patched_resolve(_make_fetched(fmt="html", data=html))
    patched_backends["trafilatura_raises"] = ExtractionError("traf failed")
    patched_backends["docling_raises"] = ExtractionError("docl failed")
    with pytest.raises(ExtractionError) as exc_info:
        await extract("https://x.com/?")
    err = exc_info.value
    # Diagnostic should mention both backends
    assert "trafilatura" in str(err).lower()
    assert "docling" in str(err).lower()


async def test_html_article_falls_back_to_docling_when_trafilatura_fails(
    patched_resolve, patched_backends
):
    """If trafilatura blows up on a tagged-as-article page, docling is the safety net."""
    html = b'<html><head><meta property="og:type" content="article"></head></html>'
    patched_resolve(_make_fetched(fmt="html", data=html))
    patched_backends["trafilatura_raises"] = ExtractionError("traf gave up")
    md = await extract("https://x.com/article")
    assert md == "## Layout\n\nbody\n"
    assert patched_backends["docling_calls"] == 1


# ---------- file path --------------------------------------------------


async def test_extract_local_pdf_file(patched_backends, tmp_path: Path):
    """Real fetch via tmp file (no resolve patching) to confirm Path support end-to-end."""
    p = tmp_path / "doc.pdf"
    p.write_bytes(b"%PDF-1.4 stub")
    md = await extract(p)
    assert md == "## Layout\n\nbody\n"
    assert patched_backends["docling_calls"] == 1


# ---------- extract_from_bytes -----------------------------------------


async def test_extract_from_bytes_pdf(patched_backends):
    """Caller already has bytes — skip the fetch step."""
    from andamentum.harvest import extract_from_bytes

    md = await extract_from_bytes(b"%PDF-1.4 stub", format="pdf", source_url="x.pdf")
    assert md == "## Layout\n\nbody\n"
    assert patched_backends["docling_calls"] == 1


async def test_extract_from_bytes_html_uses_metadata_routing(patched_backends):
    """Same metadata-driven dispatch as extract(), just with caller-supplied bytes."""
    from andamentum.harvest import extract_from_bytes

    html = b'<html><head><meta property="og:type" content="article"></head></html>'
    md = await extract_from_bytes(html, format="html", source_url="https://x.com")
    assert md == "## Article\n\nbody\n"
    assert patched_backends["trafilatura_calls"] == 1
    assert patched_backends["docling_calls"] == 0
