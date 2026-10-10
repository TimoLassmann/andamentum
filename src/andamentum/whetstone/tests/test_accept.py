"""accept_edits: the accepted edits of a review, applied to the markdown note."""

from __future__ import annotations

import pytest

from andamentum.whetstone import AcceptedEdits, Edit, accept_edits

NOTE = (
    "# Apollo\n\n"
    "The rover moved slow across the the crater.\n\n"
    "It sent **three** images back.\n"
)


def _edit(original: str, new: str, *, at: int = 0, id: str | None = None) -> Edit:
    edit = Edit(
        title="t",
        rationale="r",
        section_id="s1",
        char_start=at,
        char_end=at + len(original),
        original_text=original,
        new_text=new,
    )
    return edit if id is None else edit.model_copy(update={"id": id})


def test_every_edit_gets_its_own_id_which_survives_json() -> None:
    a, b = _edit("x", "y"), _edit("x", "y")
    assert a.id and a.id != b.id
    assert Edit.model_validate_json(a.model_dump_json()).id == a.id


def test_accepted_edits_are_applied_to_the_markdown() -> None:
    slow = _edit("moved slow", "moved slowly", at=10)
    twice = _edit("the the crater", "the crater", at=30)
    result = accept_edits(NOTE, [slow, twice])
    assert result.text == (
        "# Apollo\n\nThe rover moved slowly across the crater.\n\n"
        "It sent **three** images back.\n"
    )
    assert result.applied == [slow.id, twice.id]
    assert result.not_applied == []


def test_markup_inside_the_span_is_kept_as_written() -> None:
    edit = _edit("sent **three** images", "sent **four** images", at=50)
    assert "It sent **four** images back." in accept_edits(NOTE, [edit]).text


def test_text_not_in_the_note_is_reported_not_dropped() -> None:
    missing = _edit("Apollo The rover", "x")  # crosses the heading mark
    result = accept_edits(NOTE, [missing])
    assert result.text == NOTE
    assert result.applied == []
    assert [n.id for n in result.not_applied] == [missing.id]
    assert "not in the note" in result.not_applied[0].reason


def test_of_several_occurrences_the_nearest_one_is_replaced() -> None:
    text = "the cat sat. the cat ran."
    edit = _edit("the cat", "a dog", at=13)
    assert accept_edits(text, [edit]).text == "the cat sat. a dog ran."


def test_overlapping_edits_apply_the_first_and_report_the_second() -> None:
    first = _edit("moved slow", "moved slowly", at=10)
    second = _edit("slow across", "slowly over", at=16)
    result = accept_edits(NOTE, [first, second])
    assert result.applied == [first.id]
    assert result.not_applied[0].id == second.id
    assert first.id in result.not_applied[0].reason


def test_every_input_id_is_in_exactly_one_list() -> None:
    edits = [
        _edit("moved slow", "moved slowly", at=10),
        _edit("nowhere", "x"),
        _edit("", "inserted"),
        _edit("slow across", "y", at=16),
    ]
    result = accept_edits(NOTE, edits)
    named = result.applied + [n.id for n in result.not_applied]
    assert sorted(named) == sorted(e.id for e in edits)
    assert isinstance(result, AcceptedEdits)


def test_two_edits_with_one_id_are_refused() -> None:
    with pytest.raises(ValueError, match="share an id"):
        accept_edits(NOTE, [_edit("a", "b", id="e1"), _edit("c", "d", id="e1")])
