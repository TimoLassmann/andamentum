"""The web question set: the answer check and the drift check."""

from __future__ import annotations

import pytest

from andamentum.deep_research.v2.bench.web_set import (
    WEB_QUESTIONS,
    AnswerSpec,
    answer_correct,
    contains_phrase,
    normalise_for_match,
    page_text,
    verify_sources,
)


def test_there_are_fifteen_distinct_questions_with_https_sources() -> None:
    assert len(WEB_QUESTIONS) == 15
    assert len({q.question_id for q in WEB_QUESTIONS}) == 15
    assert len({q.question for q in WEB_QUESTIONS}) == 15
    assert all(q.source_url.startswith("https://") for q in WEB_QUESTIONS)


def test_each_expected_answer_passes_its_own_check() -> None:
    # the "expected" wording of every question satisfies the question's spec,
    # so a perfect answer can never score wrong
    for question in WEB_QUESTIONS:
        assert answer_correct(question.expected, question.answer), question.question_id


def test_a_question_is_not_answered_by_its_own_text() -> None:
    for question in WEB_QUESTIONS:
        assert not answer_correct(question.question, question.answer), (
            question.question_id
        )


def test_normalising_folds_case_accents_punctuation_and_thousands() -> None:
    assert normalise_for_match("Éclair, 1,000 – GPL-2.0!") == "eclair 1000 gpl 2 0"


def test_phrases_match_whole_words_only() -> None:
    assert contains_phrase("The answer is 88 days.", "88")
    assert not contains_phrase("It takes 1988 days", "88")
    assert contains_phrase("licensed under GPL-2.0-only", "GPL-2.0")


def test_a_phrase_without_words_is_a_programming_error() -> None:
    with pytest.raises(ValueError, match="no letters"):
        contains_phrase("text", "!!!")


def test_every_group_is_needed_and_any_phrase_of_a_group_will_do() -> None:
    spec = AnswerSpec(all_of=[["Kilimanjaro", "Kibo"], ["Tanzania"]])
    assert answer_correct("Mount Kibo in Tanzania", spec)
    assert not answer_correct("Mount Kilimanjaro in Kenya", spec)


def test_page_text_drops_markup_scripts_and_reads_entities() -> None:
    markup = "<p>Lake&nbsp;Burley <b>Griffin</b></p><script>var x = 1;</script>"
    assert "Lake\xa0Burley" in page_text(markup)
    assert "var x" not in page_text(markup)
    assert contains_phrase(page_text(markup), "Lake Burley Griffin")


async def test_verify_sources_flags_a_page_that_lost_the_answer_and_a_failed_fetch() -> (
    None
):
    first, second, third = WEB_QUESTIONS[:3]

    async def fetch(url: str) -> str:
        if url == first.source_url:
            return f"A page stating {first.expected}."
        if url == second.source_url:
            return "A page about something else."
        raise OSError("connection refused")

    checks = await verify_sources(fetch, (first, second, third))
    assert [c.states_answer for c in checks] == [True, False, False]
    assert checks[1].error == ""
    assert checks[2].error == "OSError: connection refused"
