"""Fast PDF extraction (the text layer), the extractor record and the frozen-text reason.

A PDF is built by hand in the test (a few lines of text on one page, or an empty
page for a scan), so nothing is downloaded and no document is real.
"""

from __future__ import annotations

import httpx
import pytest

from andamentum.core import fetch_gate, url_safety
from andamentum.deep_research.content_extractor import ExtractionError
from andamentum.deep_research.v2.sources import SourceClient, SourceConfig, SourceError
from andamentum.deep_research.v2.sources import extraction
from andamentum.deep_research.v2.sources.base import FROZEN_TEXT_REASON
from andamentum.deep_research.v2.sources.extraction import (
    extract_body,
    extraction_method,
    pdf_text_layer,
)
from andamentum.deep_research.v2.store import ResearchStore


def make_pdf(lines: list[str]) -> bytes:
    """A one-page PDF whose text layer holds *lines* (Helvetica, 14 pt apart)."""
    stream = "BT /F1 11 Tf 40 760 Td 14 TL\n"
    for line in lines:
        escaped = line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        stream += f"({escaped}) Tj T*\n"
    stream += "ET"
    return assemble(stream)


def assemble(stream: str) -> bytes:
    objects = [
        "<< /Type /Catalog /Pages 2 0 R >>",
        "<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        "/Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>",
        f"<< /Length {len(stream)} >>\nstream\n{stream}\nendstream",
        "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out = "%PDF-1.4\n"
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{number} 0 obj\n{body}\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n"
    for offset in offsets:
        out += f"{offset:010d} 00000 n \n"
    out += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n"
    return out.encode("latin-1")


PAPER = [
    "Invented Methods for Synthetic Widgets",
    "We study how invented widgets behave under synthetic load across many",
    "trials, and report that the median latency of the widget stays below the",
    "stated bound in every condition we tried, including the unusually long-",
    "running ones. It holds for all sizes.",
    "A second paragraph begins here and describes the apparatus used for the",
    "measurements in some detail, so that the setting can be reproduced by a",
    "reader without any further information from the authors of the study.",
]


def test_the_text_layer_is_read_as_paragraphs_and_hyphens_are_joined() -> None:
    text = pdf_text_layer(make_pdf(PAPER * 3))
    assert "unusually longrunning ones" in text  # the line-end hyphen is joined
    assert "\x02" not in text
    assert "median latency of the widget" in text
    assert "\n" not in text.replace("\n\n", "")  # no stray single line breaks


def test_a_short_last_line_ends_a_paragraph() -> None:
    long = "a line of invented prose that runs to the full width of the column"
    text = pdf_text_layer(
        make_pdf([long, long, long, "That ends it.", long, long, long, "Done."] * 3)
    )
    paragraphs = text.split("\n\n")
    assert len(paragraphs) > 3
    assert paragraphs[0].endswith("That ends it.")


def test_a_scan_has_no_text_layer_and_is_refused_not_ocred() -> None:
    with pytest.raises(ExtractionError, match="no usable text layer"):
        pdf_text_layer(assemble("q 100 0 0 100 10 10 cm Q"))


def test_garbage_and_empty_bytes_are_extraction_errors() -> None:
    with pytest.raises(ExtractionError):
        pdf_text_layer(b"this is not a pdf")
    with pytest.raises(ExtractionError, match="Empty"):
        pdf_text_layer(b"")


def test_the_method_named_follows_the_content_type_and_the_knob() -> None:
    assert extraction_method("application/pdf", pdf_extraction="fast") == "pypdfium2"
    assert extraction_method("application/pdf", pdf_extraction="docling") == "docling"
    assert extraction_method("text/html; charset=utf-8", pdf_extraction="fast") == (
        "harvest-html"
    )
    assert extraction_method("text/plain", pdf_extraction="fast") == "text"


async def test_only_pdfs_change_with_the_knob(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[str] = []

    async def stand_in(data: bytes, content_type: str, url: str) -> str:
        seen.append(content_type)
        return "from extract_content"

    monkeypatch.setattr(extraction, "extract_content", stand_in)
    pdf = make_pdf(PAPER * 3)
    assert "median latency" in await extract_body(
        pdf, "application/pdf", "https://e.org/a.pdf", pdf_extraction="fast"
    )
    assert seen == []  # the fast path did not touch docling
    assert (
        await extract_body(
            pdf, "application/pdf", "https://e.org/a.pdf", pdf_extraction="docling"
        )
        == "from extract_content"
    )
    assert (
        await extract_body(
            b"<p>x</p>", "text/html", "https://e.org/a", pdf_extraction="fast"
        )
        == "from extract_content"
    )
    assert seen == ["application/pdf", "text/html"]


def test_the_library_default_is_unchanged() -> None:
    assert SourceConfig().pdf_extraction == "docling"
    with ResearchStore.in_memory() as store:
        assert SourceClient(store).pdf_extraction == "docling"
        with pytest.raises(ValueError, match="unknown pdf_extraction"):
            SourceClient(store, pdf_extraction="ocr")  # type: ignore[arg-type]


@pytest.fixture
def public_names(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(url_safety, "is_safe_url", lambda url, **_kw: (True, ""))
    monkeypatch.setattr(
        "andamentum.references.http.is_safe_url", lambda url, **_kw: (True, "")
    )
    fetch_gate._robots_cache_clear()


async def test_the_client_records_the_extractor_per_fetched_url(
    public_names: None,
) -> None:
    pdf = make_pdf(PAPER * 3)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/robots.txt":
            return httpx.Response(404)
        if request.url.path.endswith(".pdf"):
            return httpx.Response(
                200, content=pdf, headers={"content-type": "application/pdf"}
            )
        return httpx.Response(
            200,
            content=(
                "<html><body><p>"
                + "a page of invented text " * 20
                + "</p></body></html>"
            ).encode(),
            headers={"content-type": "text/plain"},
        )

    with ResearchStore.in_memory() as store:
        client = SourceClient(
            store,
            client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
            pdf_extraction="fast",
            intervals={"fetch:e.org": 0.0},
        )
        pdf_text = await client.fetch_document("web", "https://e.org/paper.pdf")
        await client.fetch_document("web", "https://e.org/page")
    assert "median latency" in pdf_text.text
    assert client.extractions == {
        "https://e.org/paper.pdf": "pypdfium2",
        "https://e.org/page": "text",
    }


async def test_a_text_not_in_the_frozen_set_is_named_as_abstract_only() -> None:
    with ResearchStore.in_memory() as store:
        client = SourceClient(store, offline=True)
        with pytest.raises(SourceError) as raised:
            await client.fetch_document("web", "https://e.org/never-prefetched")
    assert raised.value.kind == "offline"
    assert raised.value.reason == FROZEN_TEXT_REASON
    assert "abstract only" in raised.value.reason
    assert client.offline_misses["web"] == 1
