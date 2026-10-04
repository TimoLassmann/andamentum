from __future__ import annotations

from ._stub import with_usage
from andamentum.compose.checks import reader
from andamentum.compose.checks.reader import ReaderAnswers, reader_test
from andamentum.compose.log import ActionLog
from andamentum.compose.schemas import Brief, ReaderQuestion


def brief(questions: list[ReaderQuestion]) -> Brief:
    return Brief(
        task="SECRET TASK WORDS",
        genre="memo",
        title="Apollo",
        audience="the board",
        purpose="p",
        key_messages=["SECRET KEY MESSAGE"],
        must_include=[],
        constraints=[],
        word_range=(1, 99),
        reader_questions=questions,
    )


QS = [
    ReaderQuestion(question="When is the launch?", expected_answer="May"),
    ReaderQuestion(question="What does it cost?", expected_answer="12 thousand"),
]


class FakeScore:
    def __init__(self, overall: str):
        self.overall = overall


async def test_reader_sees_only_audience_and_text_and_judges_each_answer(
    tmp_path, monkeypatch
):
    seen: dict = {}

    async def fake_agent(model, *, instructions, output_type, user_message, **kw):
        seen["model"], seen["message"] = model, user_message
        assert output_type is ReaderAnswers
        return ReaderAnswers(answers=["In May.", "not stated"])

    judged: list[tuple] = []

    async def fake_judge(output, *, criteria, context, model):
        judged.append((output, context, model, len(criteria)))
        return FakeScore("meets" if output == "In May." else "fails")

    monkeypatch.setattr(reader, "run_agent_with_usage", with_usage(fake_agent))
    monkeypatch.setattr(reader, "judge_score", fake_judge)
    res = await reader_test(
        "Apollo launches in May.",
        brief=brief(QS),
        reader_model="r:m",
        judge_model="j:m",
        log=ActionLog(tmp_path),
    )
    assert seen["model"] == "r:m"
    assert (
        "the board" in seen["message"] and "Apollo launches in May." in seen["message"]
    )
    assert "SECRET" not in seen["message"] and "12 thousand" not in seen["message"]
    assert [j[2] for j in judged] == ["j:m", "j:m"] and all(j[3] == 1 for j in judged)
    assert "Expected answer: 12 thousand" in judged[1][1]
    assert (res.correct, res.total) == (1, 2)
    assert res.check.passed is False
    assert res.misses == [QS[1]]
    assert "What does it cost?" in res.check.details[0]
    roles = [r["role"] for r in ActionLog(tmp_path).records()]
    assert roles == ["reader", "judge", "judge"]


async def test_missing_answers_are_padded_as_not_stated(tmp_path, monkeypatch):
    async def fake_agent(model, **kw):
        return ReaderAnswers(answers=["In May."])

    async def fake_judge(output, *, criteria, context, model):
        return FakeScore("fails" if output == "not stated" else "meets")

    monkeypatch.setattr(reader, "run_agent_with_usage", with_usage(fake_agent))
    monkeypatch.setattr(reader, "judge_score", fake_judge)
    res = await reader_test(
        "t", brief=brief(QS), reader_model="r", judge_model="j", log=ActionLog(tmp_path)
    )
    assert res.misses == [QS[1]]


async def test_all_correct_passes(tmp_path, monkeypatch):
    async def fake_agent(model, **kw):
        return ReaderAnswers(answers=["May", "12 thousand"])

    async def fake_judge(output, *, criteria, context, model):
        return FakeScore("meets")

    monkeypatch.setattr(reader, "run_agent_with_usage", with_usage(fake_agent))
    monkeypatch.setattr(reader, "judge_score", fake_judge)
    res = await reader_test(
        "t", brief=brief(QS), reader_model="r", judge_model="j", log=ActionLog(tmp_path)
    )
    assert res.check.passed and res.misses == []


async def test_no_questions_makes_no_calls(tmp_path):
    res = await reader_test(
        "t", brief=brief([]), reader_model="r", judge_model="j", log=ActionLog(tmp_path)
    )
    assert res.check.passed and res.total == 0
