"""Tests for the markdown / HTML / docx renderers.

Markdown and HTML are exercised on a sample ReviewResult — they're pure
functions, no external dependencies. The docx renderer test is lighter:
it only verifies the v2→v1 DocumentPatch adapter (since the actual
docx-writing machinery is v1's, well-tested elsewhere).
"""

from pathlib import Path


from andamentum.whetstone.v2 import (
    AuthorQuestion,
    Edit,
    Finding,
    Quote,
    ReviewMetrics,
    ReviewResult,
    SectionCard,
    render_html,
    render_markdown,
)


def _sample_result() -> ReviewResult:
    """A small ReviewResult exercising every section a renderer might emit."""
    return ReviewResult(
        summary="## Executive Summary\n\nThe paper has solid methodology but two unresolved citation issues.\n\n## Major Findings\n\nOne unresolved citation reference.",
        findings=[
            Finding(
                title="Unsupported claim about novelty",
                severity="major",
                confidence="high",
                rationale="The introduction claims the method is novel, but [12] reports an equivalent approach.",
                quotes=[
                    Quote(
                        section_id="sec_001",
                        char_start=10,
                        char_end=40,
                        text="our novel approach to alignment",
                    )
                ],
                sections_involved=["sec_001"],
                source="challenged",
                perspective="rigorous",
            ),
            Finding(
                title="Style: long sentences in §2",
                severity="minor",
                confidence="medium",
                rationale="Several sentences exceed 40 words in the methods section.",
                quotes=[],
                sections_involved=["sec_002"],
                source="investigate",
            ),
        ],
        deterministic_findings=[
            Finding(
                title="Citation [42] used but not in references",
                severity="major",
                confidence="high",
                rationale="[42] is cited 3 times but has no entry in the references section.",
                quotes=[
                    Quote(
                        section_id="sec_001",
                        char_start=100,
                        char_end=104,
                        text="[42]",
                    )
                ],
                sections_involved=["sec_001"],
                source="deterministic",
            ),
        ],
        edits=[
            Edit(
                title="Tighten opening sentence",
                severity="minor",
                confidence="high",
                rationale="The opening sentence is wordier than it needs to be.",
                section_id="sec_001",
                char_start=0,
                char_end=80,
                original_text="It is generally the case that approaches to this problem have been varied",
                new_text="Approaches to this problem vary widely",
            ),
        ],
        author_questions=[
            AuthorQuestion(
                question="Is the sample size in §3 really N=50, or N=48 as §5 implies?",
                why="The two values appear in different sections without reconciliation.",
                sections_involved=["sec_003", "sec_005"],
            )
        ],
        document_map=[
            SectionCard(section_id="sec_001", title="Introduction", one_line_gist="Frames the problem"),
            SectionCard(section_id="sec_002", title="Methods", one_line_gist="Describes the algorithm"),
        ],
        metrics=ReviewMetrics(
            llm_calls=12, deterministic_findings_count=1, edits_count=1
        ),
    )


# ── Markdown ───────────────────────────────────────────────────────────


def test_markdown_renders_every_section():
    md = render_markdown(_sample_result())
    assert "# Whetstone Review" in md
    assert "Executive Summary" in md
    assert "Author questions" in md
    assert "Edits (1)" in md
    assert "Findings (LLM-investigated)" in md
    assert "Deterministic findings" in md
    assert "Document map" in md
    # Edit shows as diff block
    assert "```diff" in md
    assert "- It is generally the case" in md
    assert "+ Approaches to this problem vary" in md


def test_markdown_writes_to_file_when_path_given(tmp_path: Path):
    out = tmp_path / "review.md"
    md = render_markdown(_sample_result(), output_path=out)
    assert out.exists()
    assert out.read_text() == md


def test_markdown_clean_document_says_so():
    """An empty result emits a clean message, not a wall of empty headings."""
    md = render_markdown(ReviewResult())
    assert "looks clean" in md.lower()


def test_markdown_findings_grouped_by_priority():
    md = render_markdown(_sample_result())
    # MUST FIX appears before CONSIDER in the LLM findings section
    findings_section = md.split("## Findings (LLM-investigated)")[1].split("---")[0]
    assert findings_section.index("### MUST FIX") < findings_section.index(
        "### CONSIDER"
    )


def test_markdown_persona_shown_for_panel_findings():
    md = render_markdown(_sample_result())
    assert "_rigorous_" in md  # the perspective tag


# ── HTML ───────────────────────────────────────────────────────────────


def test_html_renders_to_self_contained_document():
    html = render_html(_sample_result())
    assert "<html" in html.lower()
    assert "Whetstone Review" in html
    # Edits section present
    assert "Tighten opening sentence" in html
    # Findings present
    assert "Unsupported claim" in html
    # Citation deterministic finding present
    assert "[42]" in html
    # Author questions present
    assert "N=50" in html or "N&#x3D;50" in html  # may be HTML-escaped


def test_html_writes_to_file_when_path_given(tmp_path: Path):
    out = tmp_path / "review.html"
    html = render_html(_sample_result(), output_path=out)
    assert out.exists()
    assert out.read_text() == html


def test_html_clean_document_emits_success_callout():
    html = render_html(ReviewResult())
    assert "looks clean" in html.lower()


# ── DOCX adapter (v2 → v1 DocumentPatch) ───────────────────────────────


def test_docx_adapter_converts_edits_to_text_edit_patches():
    from andamentum.whetstone.models import DocumentPatch
    from andamentum.whetstone.v2.renderers.docx import _to_document_patches

    patches = _to_document_patches(_sample_result(), DocumentPatch)
    text_edits = [p for p in patches if p.patch_type == "text_edit"]
    assert len(text_edits) == 1
    assert text_edits[0].text_pattern == "It is generally the case that approaches to this problem have been varied"
    assert text_edits[0].new_text == "Approaches to this problem vary widely"
    assert "wordier" in text_edits[0].explanation.lower()


def test_docx_adapter_converts_findings_with_quotes_to_comments():
    from andamentum.whetstone.models import DocumentPatch
    from andamentum.whetstone.v2.renderers.docx import _to_document_patches

    patches = _to_document_patches(_sample_result(), DocumentPatch)
    comments = [p for p in patches if p.patch_type == "comment"]
    # Two findings have quotes (one LLM, one deterministic). The other LLM
    # finding has no quotes → must be skipped (no anchor).
    assert len(comments) == 2
    titles_in_comments = " ".join(p.comment_text for p in comments)
    assert "Unsupported claim" in titles_in_comments
    assert "[42]" in titles_in_comments


def test_docx_adapter_skips_findings_without_quotes():
    """A finding without quotes can't be anchored to a Word comment."""
    from andamentum.whetstone.models import DocumentPatch
    from andamentum.whetstone.v2.renderers.docx import _to_document_patches

    result = ReviewResult(
        findings=[
            Finding(
                title="No quote",
                severity="moderate",
                confidence="high",
                rationale="floating finding",
                quotes=[],  # no anchor
                sections_involved=["sec_001"],
                source="investigate",
            ),
        ],
    )
    patches = _to_document_patches(result, DocumentPatch)
    assert patches == []


def test_docx_confidence_levels_map_to_floats():
    """v2 uses low/medium/high; v1's DocumentPatch wants a 0..1 float."""
    from andamentum.whetstone.v2.renderers.docx import _confidence_to_float

    assert 0.0 < _confidence_to_float("low") < _confidence_to_float("medium")
    assert _confidence_to_float("medium") < _confidence_to_float("high") <= 1.0
