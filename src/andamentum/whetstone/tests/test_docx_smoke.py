"""Import-level smoke tests for the docx subpackage.

Verifies every symbol exported by whetstone.docx.__init__ imports without
error. DOCX round-trip tests live alongside the renderer tests.
"""


def test_docx_subpackage_imports():
    from andamentum.whetstone.docx import (
        ChangeAttributionTracker,
        DocxEditor,
        FinalizationError,
        FuzzyMatchResult,
        LocationResult,
        LocationStatus,
        ParagraphContext,
        ParagraphData,
        PatchDocxEditor,
        PatchValidator,
        ProcessingResult,
        SimilarityResult,
        TextProcessor,
        TokenAttribution,
        TokenData,
        TokenIterator,
        TokenProcessor,
        ValidationResult,
        ValidationStatus,
        XMLElementBuilder,
        XMLPatternMatcher,
        combine_validation_results,
        finalize_reviewed_document,
    )

    # Touch each symbol so unused-import linting stays honest.
    assert all(
        [
            ChangeAttributionTracker,
            DocxEditor,
            FinalizationError,
            FuzzyMatchResult,
            LocationResult,
            LocationStatus,
            ParagraphContext,
            ParagraphData,
            PatchDocxEditor,
            PatchValidator,
            ProcessingResult,
            SimilarityResult,
            TextProcessor,
            TokenAttribution,
            TokenData,
            TokenIterator,
            TokenProcessor,
            ValidationResult,
            ValidationStatus,
            XMLElementBuilder,
            XMLPatternMatcher,
            combine_validation_results,
            finalize_reviewed_document,
        ]
    )


def test_render_docx_accepts_checklist(tmp_path):
    """Smoke test: render_docx accepts a checklist parameter without error."""
    from andamentum.whetstone import ChecklistItem
    from andamentum.whetstone.renderers import render_docx

    src = tmp_path / "in.docx"
    dst = tmp_path / "out.docx"
    # Create a minimal .docx for input
    from docx import Document

    doc = Document()
    doc.add_paragraph("Hello.")
    doc.save(str(src))

    items = [ChecklistItem(name="x", status="pass", notes="y", category="abstract")]
    # Should accept checklist_items kwarg and not raise
    render_docx(
        input_path=src,
        output_path=dst,
        patches=[],
        checklist_items=items,
    )
    assert dst.exists()
