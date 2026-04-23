"""Smoke tests for whetstone.models (DocumentPatch, PatchApplicationResult)."""

import pytest
from pydantic import ValidationError

from andamentum.whetstone.models import DocumentPatch, PatchApplicationResult


def test_text_edit_patch_requires_new_text():
    with pytest.raises(ValidationError):
        DocumentPatch(
            patch_type="text_edit",
            text_pattern="old",
            new_text="",
            explanation="fix typo",
        )


def test_text_edit_patch_valid():
    p = DocumentPatch(
        patch_type="text_edit",
        text_pattern="teh",
        new_text="the",
        explanation="typo",
        confidence=0.95,
    )
    assert p.patch_type == "text_edit"
    assert p.patch_id  # auto-generated 8-char id
    assert len(p.patch_id) == 8


def test_comment_patch_requires_comment_text():
    with pytest.raises(ValidationError):
        DocumentPatch(
            patch_type="comment",
            text_pattern="something",
            comment_text="",
            explanation="x",
        )


def test_document_analysis_requires_analysis_text():
    with pytest.raises(ValidationError):
        DocumentPatch(
            patch_type="document_analysis",
            analysis_text="",
            explanation="x",
        )


def test_application_result_success_rate():
    r = PatchApplicationResult(total_patches=10, applied_patches=7, processing_time=0.1)
    assert r.success_rate == 70.0


def test_application_result_zero_patches():
    r = PatchApplicationResult(total_patches=0, applied_patches=0, processing_time=0.0)
    assert r.success_rate == 100.0
