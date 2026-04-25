"""Pin the public surface of andamentum.scribe."""

import andamentum.scribe as scribe


def test_public_all():
    expected = {
        "Document",
        "Heading",
        "Paragraph",
        "Figure",
        "Table",
        "Block",
        "Reference",
        "Revision",
        "ValidationIssue",
        "StaleRevisionError",
    }
    assert set(scribe.__all__) == expected


def test_public_imports_are_resolvable():
    for name in scribe.__all__:
        assert hasattr(scribe, name), f"scribe.{name} is missing"
