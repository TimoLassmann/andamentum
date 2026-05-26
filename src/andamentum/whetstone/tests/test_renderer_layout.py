"""Editorial-annotation layout (Proposal A).

These tests pin the new finding-rendering shape:

  per-finding section header (`Methods · s1`)
    → quoted passage (typeset-callout tone-quote in HTML,
                      ``> "…"`` blockquote in markdown)
    → comment block (typeset-callout tone-warning/note in HTML,
                     paragraph with chip line in markdown)

Plus the prelude changes:

  - one combined AI / not-peer-review banner (was two stacked callouts)
  - document map at the TOP (was at the bottom)
  - no collapsed ``<details>`` cards (everything visible at first read)
"""

from __future__ import annotations

from andamentum.whetstone.renderers import render_html, render_markdown
from andamentum.whetstone.schemas import (
    Finding,
    Quote,
    ReviewResult,
    SectionCard,
)


def _result_with_one_finding() -> ReviewResult:
    """Minimal ReviewResult: one section, one finding with a quote and
    rationale. Covers every Proposal-A render branch."""
    return ReviewResult(
        summary="## Executive Summary\n\nA tight summary.",
        findings=[
            Finding(
                title="Clarity of methods",
                severity="major",
                confidence="medium",
                rationale="The Methods section is too generic.",
                quotes=[
                    Quote(
                        text="We trained a small transformer on synthetic data.",
                        section_id="s1",
                        char_start=0,
                        char_end=51,
                    )
                ],
                sections_involved=["s1"],
                source="investigate",
                category="clarity",
                priority="must_fix",
            )
        ],
        document_map=[
            SectionCard(
                section_id="s1",
                title="Methods",
                one_line_gist="Training setup.",
            ),
            SectionCard(
                section_id="s2",
                title="Results",
                one_line_gist="Reported accuracy.",
            ),
        ],
    )


# ── HTML ───────────────────────────────────────────────────────────────────


def test_html_emits_single_combined_top_banner() -> None:
    """Pre-Proposal-A, two banners stacked: a 'note' disclaimer and a
    'warning' AI watermark. Now: one warning callout covers both."""
    html = render_html(_result_with_one_finding())
    # Exactly one warning-tone callout in the prelude. (The finding's
    # comment block is also tone-warning for major severity — that's
    # not in the prelude.)
    prelude = html.split("Executive Summary")[0]
    assert prelude.count("typeset-callout tone-warning") == 1
    # The combined banner mentions both the AI-generated nature and the
    # not-a-peer-review-tool scope statement.
    assert "AI-generated" in prelude
    assert "peer-review" in prelude.lower()


def test_html_document_map_appears_before_findings() -> None:
    """Orientation before findings — the map tells the reader what
    section ids like `s1` mean before they see them in a finding header."""
    html = render_html(_result_with_one_finding())
    map_pos = html.find("Document map")
    findings_pos = html.find("Findings")
    assert 0 < map_pos < findings_pos


def test_html_finding_emits_quote_callout_followed_by_comment_callout() -> None:
    """Proposal A's load-bearing shape: a tone-quote callout (verbatim
    passage, serif italic, left rule) immediately followed by a
    tone-warning callout (the comment, sans-serif, accent bar)."""
    html = render_html(_result_with_one_finding())
    # The quoted passage uses the existing tone-quote atom.
    assert "typeset-callout tone-quote" in html
    assert "We trained a small transformer on synthetic data." in html
    # The comment immediately follows. tone-warning for major-severity
    # findings (tone-note for moderate / minor).
    quote_pos = html.find("typeset-callout tone-quote")
    comment_pos = html.find("typeset-callout tone-warning", quote_pos)
    assert comment_pos > quote_pos
    # Severity + confidence render as typeset-badge chips.
    assert '<span class="typeset-badge">major</span>' in html
    assert '<span class="typeset-badge">medium confidence</span>' in html


def test_html_finding_has_no_collapsed_details_block() -> None:
    """Pre-Proposal-A, each finding was a typeset-card with a collapsed
    ``<details>``. Now: comment body is rendered inline in the callout
    — first read shows everything, no click required."""
    html = render_html(_result_with_one_finding())
    # The CSS still defines .typeset-card-details (it's a generic
    # typeset class), but no body element uses it. Checking for the
    # actual <details> HTML element is the load-bearing assertion.
    body = html.split("<body>")[1] if "<body>" in html else html
    assert "<details" not in body
    # Rationale must be visible in the rendered HTML (not hidden).
    assert "Methods section is too generic" in html


def test_html_per_finding_header_includes_section_title_not_just_id() -> None:
    """The per-finding header must surface the section TITLE from the
    document map, not just the section_id. `s1` alone is opaque."""
    html = render_html(_result_with_one_finding())
    # The section title shows up as a header near the finding.
    assert "Methods" in html
    # And the section_id appears alongside in <code> for navigation.
    assert "<code>s1</code>" in html


def test_html_minor_severity_uses_tone_note_not_warning() -> None:
    """Visual weight should match severity: tone-warning for major,
    tone-note for moderate / minor."""
    r = _result_with_one_finding()
    r.findings[0].severity = "minor"
    html = render_html(r)
    # The finding comment is tone-note (since severity is minor).
    # The single prelude banner remains tone-warning.
    assert html.count("typeset-callout tone-warning") == 1
    assert html.count("typeset-callout tone-note") >= 1


# ── Markdown ───────────────────────────────────────────────────────────────


def test_markdown_document_map_appears_before_findings() -> None:
    md = render_markdown(_result_with_one_finding())
    map_pos = md.find("Document map")
    findings_pos = md.find("## Findings")
    assert 0 < map_pos < findings_pos


def test_markdown_finding_emits_blockquote_passage_then_paragraph_comment() -> None:
    """Markdown equivalent of the HTML quote-then-comment pair: the
    quoted passage as a `>` blockquote, the comment as a paragraph
    immediately below with a title-and-chips line."""
    md = render_markdown(_result_with_one_finding())
    # Per-finding header pulls in the section title from the doc map.
    assert "#### Methods" in md
    # The passage renders as a blockquote.
    assert "> We trained a small transformer on synthetic data." in md
    # Title row uses chips bracketed for plain-markdown legibility.
    assert "**Clarity of methods**" in md
    assert "[major]" in md
    assert "[medium confidence]" in md


def test_markdown_finding_renders_rationale_inline() -> None:
    """No more collapsed details — rationale flows in body prose."""
    md = render_markdown(_result_with_one_finding())
    assert "Methods section is too generic" in md
