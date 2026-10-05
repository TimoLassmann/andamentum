"""respond: the pieces that need no document and no model."""

from __future__ import annotations

from andamentum.compose.harness.kernel import (
    Delete,
    Insert,
    Replace,
    Comment,
)
from andamentum.compose.harness.respond.collab import changed_paragraph_ids
from andamentum.compose.harness.respond.edits import EditRefused, span_ops
from andamentum.compose.harness.respond.items import (
    comment_items,
    is_harness_author,
)
from andamentum.compose.harness.respond.quotes import (
    find_in_paragraph,
    find_in_view,
    occurs_in,
)
from andamentum.compose.harness.respond.run import new_errors
from andamentum.compose.harness.kernel import Diagnostic, Severity

import pytest

from ._helpers import field, make_view


def para(text: str):
    return make_view((text, None)).paragraphs[0]


def apply_ops(text: str, ops) -> str:
    """Apply span ops right to left, as the writer's view of them."""
    out = text
    for op in sorted(
        ops, key=lambda o: getattr(o, "start", getattr(o, "at", 0)), reverse=True
    ):
        if isinstance(op, Replace):
            assert out[op.start : op.end] == op.expect
            out = out[: op.start] + op.text + out[op.end :]
        elif isinstance(op, Delete):
            assert out[op.start : op.end] == op.expect
            out = out[: op.start] + out[op.end :]
        else:
            assert isinstance(op, Insert)
            out = out[: op.at] + op.text + out[op.at :]
    return out


def test_one_word_change_is_one_replace():
    p = para("Participants logged every meal on paper for four weeks.")
    new = "Participants logged every meal in an app for four weeks."
    ops = span_ops(p, new, fields=[])
    assert apply_ops(p.text, ops) == new
    assert all(len(o.expect) < 12 for o in ops if isinstance(o, Replace | Delete))


def test_insertion_and_deletion_round_trip():
    p = para("The pilot enrolled 48 participants.")
    for new in (
        "The pilot enrolled 48 adult participants.",
        "The pilot enrolled participants.",
        "The pilot enrolled 52 participants at three clinics.",
    ):
        assert apply_ops(p.text, span_ops(p, new, fields=[])) == new


@pytest.mark.parametrize(
    ("new", "why"),
    [
        ("One.\nTwo.", "more than one paragraph"),
        ("", "empty"),
        ("The pilot enrolled 48 participants.", "changes nothing"),
    ],
)
def test_refused_rewrites(new: str, why: str):
    p = para("The pilot enrolled 48 participants.")
    with pytest.raises(EditRefused, match=why):
        span_ops(p, new, fields=[])


def test_a_rewrite_that_changes_a_citation_field_is_refused():
    p = para("Sodium intake is high (Doe, 2020) in adults.")
    token = field("citation", "zotero", display="(Doe, 2020)", pid="p0")
    at = p.text.index("(Doe, 2020)")
    token = token.model_copy(update={"start": at, "end": at + 11})
    with pytest.raises(EditRefused, match="field"):
        span_ops(p, "Sodium intake is high (Roe, 2021) in adults.", fields=[token])
    # a change elsewhere in the paragraph is fine
    ops = span_ops(
        p, "Sodium intake is very high (Doe, 2020) in adults.", fields=[token]
    )
    assert apply_ops(p.text, ops) == "Sodium intake is very high (Doe, 2020) in adults."


@pytest.mark.parametrize(
    "added",
    [
        "(Smith, 2019)",
        "(Smith et al., 2019; Doe 2020)",
        "[7]",
        "[2, 5]",
        "Smith et al. (2019)",
    ],
)
def test_a_rewrite_that_adds_a_citation_is_refused(added: str):
    p = para("Sodium intake is high in adults.")
    with pytest.raises(EditRefused, match="adds a citation"):
        span_ops(p, f"Sodium intake is high in adults {added}.", fields=[])


def test_a_citation_that_was_already_there_may_stay():
    p = para("Sodium intake is high (Smith, 2019) in adults [3].")
    ops = span_ops(
        p, "Sodium intake is very high (Smith, 2019) in adults [3].", fields=[]
    )
    assert apply_ops(p.text, ops).endswith("(Smith, 2019) in adults [3].")
    # a second copy of the same citation is an addition
    with pytest.raises(EditRefused, match="adds a citation"):
        span_ops(
            p,
            "Sodium intake is high (Smith, 2019) (Smith, 2019) in adults [3].",
            fields=[],
        )


def test_empty_paragraph_is_refused():
    with pytest.raises(EditRefused, match="no text"):
        span_ops(para(""), "Something.", fields=[])


def test_quotes_ignore_whitespace_and_typography():
    p = para("The “pilot”  enrolled\n48 participants.")
    assert find_in_paragraph(p, 'pilot" enrolled 48') is not None
    assert find_in_paragraph(p, "enrolled 49") is None
    assert not occurs_in("abc", "")
    view = make_view(("Alpha beta.", None), ("Gamma delta.", None))
    found = find_in_view(view, "gamma delta")
    assert found is None  # case matters
    found = find_in_view(view, "Gamma delta")
    assert found is not None and found[0].id == "p1" and found[1:] == (0, 11)


def comment(
    cid: str,
    author: str,
    text: str = "x",
    *,
    parent_id: str | None = None,
    resolved: bool = False,
) -> Comment:
    return Comment(
        id=cid,
        author=author,
        date=None,
        text=text,
        paragraph_id="p0",
        start=0,
        end=3,
        parent_id=parent_id,
        resolved=resolved,
    )


def test_comment_items_skip_resolved_harness_and_answered_threads():
    view = make_view(("Alpha beta gamma.", None)).model_copy(
        update={
            "comments": [
                comment("0", "Jane Doe", "open question"),
                comment("1", "Jane Doe", "resolved one", resolved=True),
                comment("2", "andamentum check", "a check finding"),
                comment("3", "Jane Doe", "answered one"),
                comment("4", "andamentum respond", "we replied", parent_id="3"),
                comment("5", "Roe", "needs more"),
                comment("6", "Jane Doe", "thanks, but", parent_id="5"),
            ]
        }
    )
    items, skipped = comment_items(view)
    assert [i.comment_id for i in items] == ["0", "5"]
    assert items[1].text == "Roe: needs more\nJane Doe: thanks, but"
    assert items[0].anchor_text == "Alp"
    assert len(skipped) == 2 and any("resolved" in s for s in skipped)
    assert any("already answered" in s for s in skipped)
    assert is_harness_author("Andamentum Compose") and not is_harness_author("Jane")


def test_changed_paragraphs_are_found_by_text():
    previous = ["Title", "One.", "Two.", "Three."]
    view = make_view(
        ("Title", None),
        ("One.", None),
        ("Two changed.", None),
        ("Three.", None),
        ("Four.", None),
    )
    assert changed_paragraph_ids(view, previous) == ["p2", "p4"]
    assert changed_paragraph_ids(make_view(("Title", None)), ["Title"]) == []


def diag(
    check: str, pid: str, kind: str = "", severity: Severity = "error"
) -> Diagnostic:
    return Diagnostic(
        check=check,
        tier=0,
        severity=severity,
        paragraph_id=pid,
        start=0,
        end=1,
        message="m",
        evidence=[],
        data={"kind": kind} if kind else {},
    )


def test_new_errors_only_counts_what_the_original_did_not_have():
    before = [diag("statistics", "p1"), diag("crossrefs", "p2", "dangling_target")]
    after = [
        diag("statistics", "p1"),  # already there
        diag("statistics", "p1"),  # a second one: new
        diag("crossrefs", "p2", "dangling_target"),
        diag("statistics", "p3", severity="warning"),  # not an error
        diag("statistics", "p9"),  # outside the changed paragraphs
    ]
    found = new_errors(before, after, paragraph_ids={"p1", "p2", "p3"})
    assert list(found) == ["p1"] and len(found["p1"]) == 1
