"""SimpleQA: short fact questions with one gold answer, graded by a model.

The data is OpenAI's ``simple_qa_test_set.csv`` (MIT, from simple-evals), which
the user downloads into a folder of their choice; nothing is fetched here. Each
row is a question, its gold answer and metadata. An item's id is ``SQ`` and its
row number (``SQ0001`` is the first data row), so an id names the same question
in every run.

An answer is graded by a language model that is not the system under test,
into one of three grades as SimpleQA defines them: ``correct`` (the answer
gives the gold answer and contradicts it nowhere), ``incorrect`` (it gives a
different answer, or contradicts the gold one) and ``not_attempted`` (it gives
no answer and contradicts nothing). The grade is a judgement, not a string
match: a long report that names the right date in another format is correct.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from andamentum.core.agents import run_agent_with_fallback

from .download import DatasetMissing
from .subsets import select_subset

SIMPLEQA_FILE = "simple_qa_test_set.csv"
MAX_ANSWER_CHARS = 12_000

Grade = Literal["correct", "incorrect", "not_attempted"]

GRADER_INSTRUCTIONS = (
    "You grade an answer to a question against the gold answer. Read the whole "
    "answer: it may be a long report, and the answer may be anywhere in it.\n"
    "- correct: the answer states the gold answer (any wording, format or extra "
    "detail is fine; a number may be rounded to the precision the gold answer "
    "gives) and states nothing that contradicts it.\n"
    "- incorrect: the answer states a different answer, or states the gold "
    "answer together with a contradicting one, or hedges between candidates "
    "that include a wrong one.\n"
    "- not_attempted: the answer gives no answer to the question (it says the "
    "information was not found, or talks around it) and contradicts nothing.\n"
    "Grade only whether the gold answer is given; do not judge the rest."
)


class SimpleQAItem(BaseModel):
    """One question and its gold answer."""

    model_config = ConfigDict(frozen=True)

    item_id: str = Field(pattern=r"^SQ\d{4}$")
    question: str = Field(min_length=1)
    answer: str = Field(min_length=1)


class GradeOutput(BaseModel):
    """The grader's answer."""

    grade: Grade


def read_simpleqa(path: Path) -> list[SimpleQAItem]:
    """Every row of the CSV at *path* (the file, or the folder that holds it)."""
    file = path / SIMPLEQA_FILE if path.is_dir() else path
    if not file.is_file():
        raise DatasetMissing(
            f"{file} is not there: download simple_qa_test_set.csv from "
            "openai/simple-evals into that folder"
        )
    with file.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return [
        SimpleQAItem(
            item_id=f"SQ{number:04d}",
            question=row["problem"].strip(),
            answer=row["answer"].strip(),
        )
        for number, row in enumerate(rows, start=1)
    ]


def load_simpleqa(
    path: Path, *, n: int, seed: int = 0, skip: int = 0
) -> list[SimpleQAItem]:
    """The seeded order's positions ``skip + 1`` to ``skip + n``, in that order."""
    items = read_simpleqa(path)
    by_id = {i.item_id: i for i in items}
    return [by_id[i] for i in select_subset(list(by_id), skip + n, seed=seed)[skip:]]


def grade_message(*, question: str, gold: str, answer: str) -> str:
    """What the grader reads; an answer longer than ``MAX_ANSWER_CHARS`` is cut."""
    text = answer.strip() or "(no answer)"
    if len(text) > MAX_ANSWER_CHARS:
        text = text[:MAX_ANSWER_CHARS] + "\n[... cut]"
    return f"Question: {question}\n\nGold answer: {gold}\n\nAnswer to grade:\n{text}"


class AnswerGrader:
    """Grades answers with one language model, one call per answer."""

    def __init__(self, *, model: Any) -> None:
        self.model = model

    async def grade(self, *, question: str, gold: str, answer: str) -> Grade:
        output = await run_agent_with_fallback(
            self.model,
            instructions=GRADER_INSTRUCTIONS,
            output_type=GradeOutput,
            user_message=grade_message(question=question, gold=gold, answer=answer),
        )
        return output.grade
