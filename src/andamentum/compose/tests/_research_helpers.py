"""A synthetic research result and a stand-in for ``deep_research.research``.

Invented works only (Okafor's SaltLog study, a DOI-less Apollo blog post).
"""

from __future__ import annotations

from typing import Any

from andamentum.deep_research import (
    Finding,
    Passage,
    ResearchResult,
    RunStats,
    SubQuestion,
)
from andamentum.references import Work

STUDY_ID = "w1"
BLOG_ID = "w2"
STUDY_TEXT = (
    "Structured logging of reading sessions improves fidelity of delivery in "
    "school interventions. The effect held across all three schools."
)
BLOG_TEXT = "Turning on the Apollo cache cut median latency from 120 ms to 50 ms."
STUDY_QUOTE = "Structured logging of reading sessions improves fidelity of delivery"
BLOG_QUOTE = "cut median latency from 120 ms to 50 ms"


def study() -> Work:
    return Work(
        work_id=STUDY_ID,
        doi="10.5555/saltlog.2020.1",
        title="Structured session logs and the fidelity of school interventions",
        authors=["Ada Okafor", "Ravi Rowe", "Lin Quay", "Kim Lee"],
        year=2020,
        venue="Journal of Invented Studies",
    )


def blog() -> Work:
    return Work(
        work_id=BLOG_ID,
        url="https://apollo-docs.example/latency",
        title="Apollo cache latency guide",
    )


def make_result(
    *,
    degraded: list[str] | None = None,
    with_blog: bool = True,
    blog_verified: bool = True,
) -> ResearchResult:
    study_passage = Passage.make(work_id=STUDY_ID, start=40, text=STUDY_TEXT)
    blog_passage = Passage.make(work_id=BLOG_ID, start=0, text=BLOG_TEXT)
    findings = [
        Finding(
            finding_id="F1",
            subquestion_id="Q1",
            claim="Structured logging improves delivery fidelity.",
            quote=STUDY_QUOTE,
            quote_verified=True,
            passage_id=study_passage.passage_id,
            work_id=STUDY_ID,
        )
    ]
    works = [study()]
    passages = [study_passage]
    if with_blog:
        findings.append(
            Finding(
                finding_id="F2",
                subquestion_id="Q1",
                claim="The Apollo cache halves median latency.",
                quote=BLOG_QUOTE,
                quote_verified=blog_verified,
                passage_id=blog_passage.passage_id,
                work_id=BLOG_ID,
            )
        )
        works.append(blog())
        passages.append(blog_passage)
    return ResearchResult(
        run_id="r_test",
        question="q",
        mode="scholarly",
        subquestions=[SubQuestion(id="Q1", text="What is known?")],
        works=works,
        passages=passages,
        findings=findings,
        stop_reason="answered",
        degraded=degraded or [],
        stats=RunStats(llm_calls=3, input_tokens=1000, output_tokens=200),
    )


class FakeResearch:
    """Replaces ``deep_research.research``; records how it was called."""

    def __init__(
        self, result: ResearchResult | None = None, *, error: Exception | None = None
    ):
        self.result = result or make_result()
        self.error = error
        self.calls: list[dict[str, Any]] = []

    async def __call__(self, question: str, **kwargs: Any) -> ResearchResult:
        self.calls.append({"question": question, **kwargs})
        if self.error is not None:
            raise self.error
        return self.result.model_copy(update={"mode": kwargs["mode"]})
