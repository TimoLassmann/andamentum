"""Tests for scoring.score_markdown."""

from andamentum.harvest.scoring import score_markdown


def test_empty_string_is_disqualified():
    assert score_markdown("") < -10_000


def test_structureless_soup_is_disqualified():
    """The BBC-homepage symptom: lots of text, zero `##`, zero `\\n\\n`."""
    soup = "x" * 5000  # no headings, no paragraph breaks
    assert score_markdown(soup) < -10_000


def test_headings_dominate_the_score():
    """A markdown with 5 headings should beat one with 0 headings of similar length."""
    structured = "## A\n\nbody\n\n## B\n\nbody\n\n## C\n\nbody\n\n## D\n\nbody\n\n## E\n\nbody\n"
    flat = "body body body body body body body body " * 10
    assert score_markdown(structured) > score_markdown(flat)


def test_paragraph_breaks_break_ties():
    """Same chars, no headings: more paragraph breaks → higher score."""
    one_para = "x" * 1000
    many_paras = "x\n\n" * 200
    # one_para is disqualified (no \n\n, no headings), many_paras has \n\n
    assert score_markdown(many_paras) > score_markdown(one_para)


def test_link_spam_is_penalised():
    """Heavy link decoration (every word a markdown link) tanks the score."""
    clean = "## Title\n\nbody " * 50
    spammy = "## Title\n\n" + "[word](url) " * 200
    # Both have headings + paragraph breaks; spammy gets penalised
    assert score_markdown(clean) > score_markdown(spammy)
