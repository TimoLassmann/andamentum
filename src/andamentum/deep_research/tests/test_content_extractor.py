"""Tests for content_extractor module."""

import pytest

from unittest.mock import patch

from ..content_extractor import extract_content, extract_html, extract_pdf, ExtractionError


class TestExtractHtml:
    def test_extracts_article_content(self):
        """Trafilatura should extract the main article, not navigation."""
        html = """
        <html>
        <head><title>Test Article</title></head>
        <body>
            <nav><a href="/">Home</a><a href="/about">About</a></nav>
            <aside>
                <h3>Sidebar</h3>
                <ul><li>Link 1</li><li>Link 2</li></ul>
            </aside>
            <article>
                <h1>Important Research Finding</h1>
                <p>Scientists have discovered a new method for analyzing data
                that significantly improves accuracy. The study, published in
                the Journal of Example Research, demonstrates a 40% improvement
                over previous approaches. This finding has major implications
                for the field of computational biology.</p>
                <p>The research team, led by Dr. Example, conducted experiments
                over a period of three years involving more than 500 participants
                from diverse backgrounds across multiple institutions.</p>
            </article>
            <footer>Copyright 2024 Example Corp</footer>
        </body>
        </html>
        """
        result = extract_html(html, url="https://example.com/article")

        assert isinstance(result, str)
        assert len(result) > 50
        # Article content should be present
        assert "Important Research Finding" in result or "analyzing data" in result
        # Navigation junk should be stripped
        assert "Sidebar" not in result
        assert "Copyright 2024" not in result

    def test_empty_html_raises(self):
        """Non-extractable HTML should raise ExtractionError."""
        html = "<html><body></body></html>"
        with pytest.raises(ExtractionError):
            extract_html(html, url="https://example.com")

    def test_returns_string(self):
        """Output should be a plain string, not bytes."""
        html = """
        <html><body>
            <article>
                <p>This is a substantial article with enough content for
                trafilatura to extract. It contains multiple sentences and
                meaningful information about a research topic that should
                be preserved in the output.</p>
            </article>
        </body></html>
        """
        result = extract_html(html, url="https://example.com")
        assert isinstance(result, str)


class TestExtractPdf:
    def test_extracts_markdown_from_pdf_bytes(self):
        """Docling should convert valid PDF bytes to markdown."""
        from io import BytesIO

        from reportlab.lib.pagesizes import letter
        from reportlab.pdfgen import canvas

        buf = BytesIO()
        c = canvas.Canvas(buf, pagesize=letter)
        c.drawString(100, 700, "Hello World from test PDF")
        c.save()
        pdf_bytes = buf.getvalue()

        result = extract_pdf(pdf_bytes, source_name="test.pdf")
        assert isinstance(result, str)
        assert len(result.strip()) > 0

    def test_empty_bytes_raises(self):
        """Empty/corrupt bytes should raise ExtractionError."""
        with pytest.raises(ExtractionError):
            extract_pdf(b"", source_name="empty.pdf")

    def test_garbage_bytes_raises(self):
        """Non-PDF bytes should raise ExtractionError."""
        with pytest.raises(ExtractionError):
            extract_pdf(b"this is not a pdf", source_name="garbage.pdf")


class TestExtractContent:
    def test_routes_html_by_content_type(self):
        """Content-type containing 'html' should route to extract_html."""
        html_bytes = b"""
        <html><body>
            <article>
                <p>This is a substantial article with enough content for
                trafilatura to extract. It contains multiple sentences and
                meaningful information about a research topic that should
                be preserved in the output for downstream processing.</p>
            </article>
        </body></html>
        """
        result = extract_content(html_bytes, "text/html; charset=utf-8", "https://example.com")
        assert isinstance(result, str)
        assert len(result) > 20

    def test_routes_pdf_by_content_type(self):
        """Content-type containing 'pdf' should route to extract_pdf."""
        with patch("andamentum.deep_research.content_extractor.extract_pdf", return_value="# Mocked PDF output") as mock:
            result = extract_content(b"%PDF-1.4 data", "application/pdf", "https://example.com/paper.pdf")
            mock.assert_called_once_with(b"%PDF-1.4 data", "paper.pdf")
            assert result == "# Mocked PDF output"

    def test_unknown_type_falls_back_to_text(self):
        """Unknown content-types should decode bytes as UTF-8 text."""
        raw = "Plain text content here"
        result = extract_content(raw.encode("utf-8"), "text/plain", "https://example.com/file.txt")
        assert result == raw

    def test_extracts_source_name_from_url(self):
        """Should extract filename from URL path for PDF source_name."""
        with patch("andamentum.deep_research.content_extractor.extract_pdf", return_value="markdown") as mock:
            extract_content(b"%PDF", "application/pdf", "https://example.com/papers/study-2024.pdf")
            mock.assert_called_once_with(b"%PDF", "study-2024.pdf")

    def test_html_charset_respected(self):
        """Should decode HTML bytes using charset from content-type header."""
        text = "Caf\u00e9 au lait"
        html = f"<html><body><article><p>{text} is a popular drink that many researchers enjoy during long experiments in the laboratory.</p></article></body></html>"
        html_bytes = html.encode("latin-1")
        result = extract_content(html_bytes, "text/html; charset=iso-8859-1", "https://example.com")
        assert "Caf" in result
