"""What the draft stakes as new reaches the reader, and reads as unverified.

The claims are found on every run and checked on none of them unless
``--check-novelty`` was passed. The one way this section can do harm is by
being read as a verification, so every renderer states that it is not one, and
these tests pin the sentence rather than merely the presence of the list.
"""

from __future__ import annotations

from andamentum.whetstone.renderers.docx import _collect_novelty_claims
from andamentum.whetstone.renderers.html import render_html
from andamentum.whetstone.renderers.markdown import render_markdown
from andamentum.whetstone.schemas import NoveltyClaim, ReviewResult


def _claims(n: int = 2) -> list[NoveltyClaim]:
    return [
        NoveltyClaim(
            claim_text=f"We present the first system that does thing {i}.",
            short_summary=f"first system for thing {i}",
            why_load_bearing=f"contribution statement {i}",
            section_id="s1",
        )
        for i in range(1, n + 1)
    ]


def test_markdown_lists_the_claims_and_says_nothing_checked_them() -> None:
    out = render_markdown(ReviewResult(summary="s", novelty_claims=_claims(2)))
    assert "Claims you are staking as novel (2)" in out
    assert "checked against nothing" in out
    assert "We present the first system that does thing 1." in out
    assert "contribution statement 1" in out
    assert "--check-novelty" in out  # the way to verify, where it is relevant


def test_html_lists_the_claims_and_says_nothing_checked_them() -> None:
    out = render_html(ReviewResult(summary="s", novelty_claims=_claims(1)))
    assert "Claims you are staking as novel (1)" in out
    assert "checked against nothing" in out


def test_the_docx_block_warns_before_it_lists() -> None:
    block = _collect_novelty_claims(ReviewResult(summary="s", novelty_claims=_claims(1)))
    assert block.startswith("*These are checked against nothing")
    # Asterisks, not underscores: the markdown-to-Word renderer leaves
    # _italic_ as literal characters. Same trap as the declined-findings block.
    assert "_These are" not in block


def test_silence_when_the_draft_stakes_no_claim() -> None:
    """A tutorial or a short story asserts nothing novel, and gets no section.

    An explicit "none found" line would put a reassuring notice on every
    document that never made such a claim, which is noise on most of them.
    """
    empty = ReviewResult(summary="s")
    assert "staking as novel" not in render_markdown(empty)
    assert "staking as novel" not in render_html(empty)
    assert _collect_novelty_claims(empty) == ""
