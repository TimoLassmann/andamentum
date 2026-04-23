"""The public API of andamentum.whetstone must be stable — every symbol
listed in __all__ must import cleanly from the top-level package."""

from andamentum import whetstone


def test_public_symbols_resolve():
    expected = {
        "sharpen_document",
        "ReviewResult",
        "DocumentPatch",
        "DocumentIssue",
        "PatchApplicationResult",
        "render_docx",
        "render_html",
        "render_diff",
        "apply_patches",
        "AgentDefinition",
        "AGENT_REGISTRY",
        "convert_fields_to_schema",
        "create_output_model",
    }
    assert expected.issubset(set(whetstone.__all__))
    for name in expected:
        assert hasattr(whetstone, name), f"missing public symbol: {name}"


def test_version_exported():
    assert hasattr(whetstone, "__version__")
    assert whetstone.__version__  # non-empty
