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
