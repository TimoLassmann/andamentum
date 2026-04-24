"""Tests for the markdown diff renderer and apply_patches utility."""

from andamentum.whetstone.issues import DocumentIssue
from andamentum.whetstone.models import DocumentPatch
from andamentum.whetstone.renderers import apply_patches, render_diff


def _edit(pattern: str, new: str, conf: float = 0.9) -> DocumentPatch:
    return DocumentPatch(
        patch_type="text_edit",
        text_pattern=pattern,
        new_text=new,
        explanation="test",
        confidence=conf,
    )


def test_apply_patches_simple():
    content = "The data is showing a clear effect."
    patches = [_edit("The data is", "The data are")]
    assert apply_patches(content, patches) == "The data are showing a clear effect."


def test_apply_patches_skips_missing_pattern():
    content = "Hello world"
    patches = [_edit("goodbye", "farewell")]
    assert apply_patches(content, patches) == "Hello world"


def test_apply_patches_highest_confidence_first():
    content = "cat"
    patches = [
        _edit("cat", "dog", conf=0.5),
        _edit("cat", "fish", conf=0.9),
    ]
    # Highest-confidence wins; second replacement's pattern no longer matches.
    assert apply_patches(content, patches) == "fish"


def test_render_diff_empty():
    assert (
        render_diff(patches=[], issues=[], original_content="x")
        == "No edits or issues found."
    )


def test_render_diff_with_edit_and_issue():
    patches = [_edit("teh", "the")]
    issues = [
        DocumentIssue(
            issue_type="major",
            category="methodology",
            title="Missing control",
            description="No control group.",
            agent_type="methodology",
        )
    ]
    out = render_diff(patches=patches, issues=issues, original_content="teh dog")

    assert "## Edits" in out
    assert "- teh" in out
    assert "+ the" in out
    assert "## Issues" in out
    assert "### Major Issues" in out
    assert "Missing control" in out


def test_render_diff_with_synthesis():
    out = render_diff(
        patches=[],
        issues=[],
        original_content="",
        synthesis_text="Overall this is a solid draft.",
    )
    assert "## Synthesis" in out
    assert "Overall this is a solid draft." in out


def test_render_diff_checklist_block():
    from andamentum.whetstone import ChecklistItem
    from andamentum.whetstone.renderers import render_diff

    items = [
        ChecklistItem(
            name="Abstract words",
            status="pass",
            notes="240 on p.1",
            category="abstract",
        ),
        ChecklistItem(
            name="Ethics statement",
            status="fail",
            notes="Missing IRB block",
            category="statements",
        ),
        ChecklistItem(
            name="Keywords",
            status="unclear",
            notes="Found but inline",
            category="hygiene",
        ),
    ]
    output = render_diff(patches=[], issues=[], original_content="", checklist=items)
    assert "Abstract words" in output
    assert "Ethics statement" in output
    # Some kind of status marker is present
    assert "PASS" in output or "✓" in output
    assert "FAIL" in output or "✗" in output


def test_render_diff_no_checklist_keeps_old_output():
    from andamentum.whetstone.renderers import render_diff

    out_no = render_diff(patches=[], issues=[], original_content="")
    out_empty_cl = render_diff(patches=[], issues=[], original_content="", checklist=[])
    assert out_no == out_empty_cl
