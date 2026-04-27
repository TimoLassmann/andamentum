"""Tests for HTML metadata sniffing."""

from andamentum.harvest.metadata import sniff_html_metadata


def test_og_type_article_wins():
    html = b'<html><head><meta property="og:type" content="article"></head></html>'
    m = sniff_html_metadata(html)
    assert m.verdict == "article"
    assert m.og_type == "article"


def test_ld_json_news_article_wins():
    html = b"""<html><head>
    <script type="application/ld+json">
    {"@type": "NewsArticle", "headline": "x"}
    </script>
    </head></html>"""
    m = sniff_html_metadata(html)
    assert m.verdict == "article"
    assert m.ld_json_type == "NewsArticle"


def test_ld_json_webpage_marks_not_article():
    """The BBC homepage symptom — JSON-LD says WebPage, not Article."""
    html = b"""<html><head>
    <script type="application/ld+json">
    {"@type": "WebPage", "name": "BBC News"}
    </script>
    </head></html>"""
    m = sniff_html_metadata(html)
    assert m.verdict == "not_article"
    assert m.ld_json_type == "WebPage"


def test_ld_json_collection_page_marks_not_article():
    html = b"""<html><head>
    <script type="application/ld+json">
    {"@type": "CollectionPage"}
    </script>
    </head></html>"""
    m = sniff_html_metadata(html)
    assert m.verdict == "not_article"


def test_no_signal_is_ambiguous():
    html = b"<html><body><p>just some prose</p></body></html>"
    m = sniff_html_metadata(html)
    assert m.verdict == "ambiguous"


def test_article_tag_alone_marks_article():
    """An <article> tag is HTML5's semantic marker — strong enough to route."""
    html = b"<html><body><article>some text</article></body></html>"
    m = sniff_html_metadata(html)
    assert m.verdict == "article"
    assert m.has_article_tag


def test_nested_ld_json_is_walked():
    """JSON-LD often wraps Article inside a @graph array."""
    html = b"""<html><head>
    <script type="application/ld+json">
    {"@graph": [
      {"@type": "Organization", "name": "x"},
      {"@type": "Article", "headline": "y"}
    ]}
    </script>
    </head></html>"""
    m = sniff_html_metadata(html)
    assert m.verdict == "article"
    assert m.ld_json_type == "Article"


def test_invalid_ld_json_is_ignored():
    """A broken JSON-LD block must not crash the sniffer."""
    html = b"""<html><head>
    <script type="application/ld+json">{not valid json</script>
    <meta property="og:type" content="article">
    </head></html>"""
    m = sniff_html_metadata(html)
    assert m.verdict == "article"
