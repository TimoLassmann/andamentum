"""Reader test: can a cold reader get the key facts out of the finished text?

A reader-role model sees the audience description and the plain text only: no
brief, no sources. It answers the brief's reader questions. A judge then decides
each answer against the expected one, through ``llm_judge.judge_score`` with a
single criterion (one call per question, sequential).
"""

from __future__ import annotations

from pydantic import BaseModel

from andamentum.core.agents import run_agent_with_usage
from andamentum.llm_judge import Criterion, judge_score

from ..log import ActionLog
from ..schemas import Brief, CheckResult, ReaderQuestion

READER_INSTRUCTIONS = (
    "You are a reader of the document below. Answer each numbered question using "
    "only what the document says. Keep each answer to one or two sentences. If the "
    "document does not say, answer exactly: not stated. Return one answer per "
    "question, in order."
)
NOT_STATED = "not stated"


class ReaderAnswers(BaseModel):
    answers: list[str]


class ReaderResult(BaseModel):
    check: CheckResult
    misses: list[ReaderQuestion]
    correct: int
    total: int


def _criterion(question: ReaderQuestion) -> Criterion:
    return Criterion(
        name="matches_expected_answer",
        description=(
            "The answer states the same fact as the expected answer "
            f"({question.expected_answer}); extra detail is fine, a missing or "
            "contradicting fact is not, and 'not stated' is a failure."
        ),
    )


async def reader_test(
    text: str,
    *,
    brief: Brief,
    reader_model: str,
    judge_model: str,
    log: ActionLog,
) -> ReaderResult:
    """Ask the reader the brief's questions and judge the answers.

    *text* is the rendered plain text of the document, markers removed.
    """
    questions = brief.reader_questions
    if not questions:
        return ReaderResult(
            check=CheckResult(name="reader", passed=True, details=["no questions"]),
            misses=[],
            correct=0,
            total=0,
        )
    numbered = "\n".join(f"{i}. {q.question}" for i, q in enumerate(questions, 1))
    message = (
        f"Audience: {brief.audience}\n\nDocument:\n{text}\n\nQuestions:\n{numbered}"
    )
    read, usage = await run_agent_with_usage(
        reader_model,
        instructions=READER_INSTRUCTIONS,
        output_type=ReaderAnswers,
        user_message=message,
    )
    log.call(
        node="Check",
        role="reader",
        model=reader_model,
        prompt=READER_INSTRUCTIONS,
        input_text=message,
        output_text=read.model_dump_json(),
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
    )
    answers = list(read.answers) + [NOT_STATED] * (len(questions) - len(read.answers))

    misses: list[ReaderQuestion] = []
    details: list[str] = []
    for question, answer in zip(questions, answers):
        scored = await judge_score(
            answer,
            criteria=[_criterion(question)],
            context=(
                f"Question: {question.question}\n"
                f"Expected answer: {question.expected_answer}"
            ),
            model=judge_model,
        )
        log.call(
            node="Check",
            role="judge",
            model=judge_model,
            prompt="judge_score:matches_expected_answer",
            input_text=f"{question.question}\n{answer}",
            output_text=scored.overall,
        )
        if scored.overall != "meets":
            misses.append(question)
            details.append(
                f"{question.question} | expected: {question.expected_answer} "
                f"| got: {answer}"
            )
    return ReaderResult(
        check=CheckResult(name="reader", passed=not misses, details=details),
        misses=misses,
        correct=len(questions) - len(misses),
        total=len(questions),
    )
