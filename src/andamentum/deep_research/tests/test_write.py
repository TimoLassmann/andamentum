"""WRITE: one call over all sub-questions, every sentence cited by construction."""

from __future__ import annotations

from pathlib import Path

import pytest

from andamentum.references import Work

from ..ask import UNTRUSTED
from ..types import Budget, Finding, Passage, SubQuestion, SubQuestionResult
from ..write import (
    CODE_LINE_PREFIXES,
    MAX_FINDINGS_PER_SECTION,
    NO_EVIDENCE_LINE,
    SOURCES_HEADING,
    WRITE_INSTRUCTIONS,
    ReportWriterError,
    write_report,
    write_reserve,
)
from .llm_stub import ScriptedLLM, make_asker

Q1 = SubQuestion(id="Q1", text="Does Apollo cut latency?")
Q2 = SubQuestion(id="Q2", text="What does SaltLog cost?")
Q3 = SubQuestion(id="Q3", text="Is Apollo free to use?")

P1 = Passage.make(
    work_id="W1", start=0, text="Apollo cut median latency from 120 ms to 45 ms."
)
P2 = Passage.make(
    work_id="W2", start=0, text="Apollo raised median latency from 45 ms to 80 ms."
)
P3 = Passage.make(
    work_id="W3", start=0, text="SaltLog verbose mode adds roughly 3 ms per request."
)


def _finding(
    number: int,
    passage: Passage,
    claim: str,
    *,
    subq: str = "Q1",
    verified: bool = True,
) -> Finding:
    return Finding(
        finding_id=f"F{number}",
        subquestion_id=subq,
        claim=claim,
        quote=passage.text,
        quote_verified=verified,
        passage_id=passage.passage_id,
        work_id=passage.work_id,
    )


F1 = _finding(1, P1, "Apollo cut median latency.")
F2 = _finding(2, P2, "Apollo raised median latency.", verified=False)
F3 = _finding(3, P3, "SaltLog adds 3 ms per request.", subq="Q2")
WORKS = {
    "W1": Work(
        work_id="W1",
        title="Apollo caching",
        authors=["Jane Doe"],
        year=2024,
        venue="Journal of Caches",
        doi="10.5555/apollo.1",
    ),
    "W2": Work(
        work_id="W2", title="A blog on Apollo", url="https://blog.example.org/apollo"
    ),
    "W3": Work(
        work_id="W3",
        title="SaltLog costs",
        authors=["Max Muster"],
        year=2023,
        url="https://example.net/saltlog",
    ),
    "W4": Work(
        work_id="W4",
        title="A retracted study",
        authors=["Ann Roe"],
        retracted="retracted",
    ),
}
ANSWERS = {
    "Q1": SubQuestionResult(subquestion_id="Q1", answer="Apollo likely cuts latency."),
}


def _writer(seen: list[str] | None = None):
    """A model that writes one sentence per section, citing its first finding."""

    def respond(instructions: str, prompt: str) -> dict[str, object]:
        if seen is not None:
            seen.append(prompt)
        sections = []
        for sub_id in ("Q1", "Q2", "Q3"):
            marker = f"SUB-QUESTION {sub_id}:"
            if marker not in prompt:
                continue
            block = prompt.split(marker)[1].split("SUB-QUESTION")[0]
            first = next(
                line.split()[0]
                for line in block.splitlines()
                if line.startswith("F") and line[1:2].isdigit()
            )
            sections.append(
                {
                    "subquestion": sub_id,
                    "sentences": [
                        {"text": f"Statement for {sub_id}.", "finding_ids": [first]}
                    ],
                }
            )
        return {"sections": sections}

    return respond


async def test_one_call_over_all_sections_with_cited_sentences() -> None:
    llm = ScriptedLLM(_writer())
    asker = make_asker(model=llm)
    report = await write_report(
        "Is Apollo worth it?",
        subquestions=[Q1, Q2],
        answers=ANSWERS,
        findings=[F1, F2, F3],
        works=WORKS,
        asker=asker,
    )
    assert len(llm.calls) == 1
    assert asker.stats.llm_calls == 1
    assert asker.stats.step_llm_calls == {"Write": 1}
    md = report.markdown
    assert md.startswith("# Is Apollo worth it?\n")
    assert md.index("## Does Apollo cut latency?") < md.index("## What does SaltLog")
    assert "Statement for Q1 [F1]." in md
    assert "Statement for Q2 [F3]." in md
    assert report.cited_finding_ids == ["F1", "F3"]
    assert report.excluded_finding_ids == []


async def test_the_prompt_carries_rules_origins_markers_and_the_draft_answer() -> None:
    seen: list[str] = []
    llm = ScriptedLLM(_writer(seen))
    await write_report(
        "Is Apollo worth it?",
        subquestions=[Q1, Q2],
        answers=ANSWERS,
        findings=[F1, F2, F3],
        works=WORKS,
        asker=make_asker(model=llm),
    )
    instructions, prompt = llm.calls[0]
    for rule in (
        "best-supported answer",
        "primary, official and encyclopedic",
        "independent origins",
        "time scope",
        "disagreement",
        "quote not verified",
        "Never state anything the findings do not say",
        "how sure",
    ):
        assert rule in instructions
    assert instructions == WRITE_INSTRUCTIONS
    assert UNTRUSTED in instructions
    # Origin labels and metadata of each finding, fenced.
    assert "origin: doe" in prompt
    assert "origin: blog.example.org" not in prompt  # the registrable domain
    assert "origin: example.org" in prompt
    assert "origin: muster" in prompt
    assert "year: 2024" in prompt and "venue: Journal of Caches" in prompt
    assert "<<<" in prompt and ">>>" in prompt
    # The unverified marker sits on F2 only.
    assert prompt.count("(quote not verified)") == 1
    assert "F2 (quote not verified)" in prompt
    # The research's draft answer is labelled as unverified.
    assert "Apollo likely cuts latency." in prompt
    assert "unverified" in prompt
    assert "Independent origins among the findings below: 2" in prompt


async def test_sources_section_lists_the_cited_works() -> None:
    report = await write_report(
        "Is Apollo worth it?",
        subquestions=[Q1, Q2],
        answers={},
        findings=[F1, F2, F3],
        works=WORKS,
        asker=make_asker(model=ScriptedLLM(_writer())),
    )
    head, sources = report.markdown.split(SOURCES_HEADING)
    assert "https://doi.org/10.5555/apollo.1" in sources
    assert "[F1] Jane Doe (2024). Apollo caching. Journal of Caches." in sources
    assert "https://example.net/saltlog" in sources
    assert "blog.example.org" not in sources  # F2 was not cited
    assert "[F3]" in sources


async def test_ids_written_into_the_text_are_stripped() -> None:
    def respond(instructions: str, prompt: str) -> dict[str, object]:
        return {
            "sections": [
                {
                    "subquestion": "Q1",
                    "sentences": [
                        {"text": "Apollo helps [F2], they say.", "finding_ids": ["F1"]}
                    ],
                }
            ]
        }

    report = await write_report(
        "q",
        subquestions=[Q1],
        answers={},
        findings=[F1],
        works=WORKS,
        asker=make_asker(model=ScriptedLLM(respond)),
    )
    assert "Apollo helps, they say [F1]." in report.markdown
    assert "[F2]" not in report.markdown


async def test_the_validator_rejects_unknown_ids_no_ids_and_a_missing_section() -> None:
    attempts: list[dict[str, object]] = [
        # Unknown id.
        {
            "sections": [
                {
                    "subquestion": "Q1",
                    "sentences": [{"text": "A.", "finding_ids": ["F99"]}],
                },
                {
                    "subquestion": "Q2",
                    "sentences": [{"text": "B.", "finding_ids": ["F3"]}],
                },
            ]
        },
        # A sentence without ids.
        {
            "sections": [
                {"subquestion": "Q1", "sentences": [{"text": "A.", "finding_ids": []}]},
                {
                    "subquestion": "Q2",
                    "sentences": [{"text": "B.", "finding_ids": ["F3"]}],
                },
            ]
        },
        # A missing section (Q2).
        {
            "sections": [
                {
                    "subquestion": "Q1",
                    "sentences": [{"text": "A.", "finding_ids": ["F1"]}],
                }
            ]
        },
        # Good.
        {
            "sections": [
                {
                    "subquestion": "Q1",
                    "sentences": [{"text": "A.", "finding_ids": ["F1"]}],
                },
                {
                    "subquestion": "Q2: What does SaltLog cost?",
                    "sentences": [{"text": "B.", "finding_ids": ["F3"]}],
                },
            ]
        },
    ]
    queue = list(attempts)
    llm = ScriptedLLM(lambda *_: queue.pop(0))
    report = await write_report(
        "q",
        subquestions=[Q1, Q2],
        answers={},
        findings=[F1, F3],
        works=WORKS,
        asker=make_asker(model=llm),
    )
    # The repairs are retries of one call, not extra calls of the run.
    assert len(llm.calls) == 4
    assert queue == []
    assert "A. [F1]." not in report.markdown  # the sentence ends "A [F1]."
    assert "A [F1]." in report.markdown and "B [F3]." in report.markdown


def _sections(*pairs: tuple[str, list[str]]) -> dict[str, object]:
    """A model answer: (sub-question id, sentence texts) per section; each
    sentence cites the first finding of the sub-question's block."""
    return {
        "sections": [
            {
                "subquestion": sub_id,
                "sentences": [
                    {"text": text, "finding_ids": [_first_id(sub_id)]} for text in texts
                ],
            }
            for sub_id, texts in pairs
        ]
    }


_FIRST = {"Q1": "F1", "Q2": "F3"}


def _first_id(sub_id: str) -> str:
    return _FIRST[sub_id]


def _asker(llm: ScriptedLLM, *, calls: int = 100):
    return make_asker(model=llm, budget=Budget(max_llm_calls=calls))


async def _write(llm: ScriptedLLM, *, asker=None):
    return await write_report(
        "q",
        subquestions=[Q1, Q2],
        answers={},
        findings=[F1, F2, F3],
        works=WORKS,
        asker=asker or _asker(llm),
    )


def _alone(prompt: str) -> bool:
    return "SUB-QUESTION Q1:" not in prompt or "SUB-QUESTION Q2:" not in prompt


async def test_an_empty_section_is_rejected_and_repaired_without_a_degraded_note() -> (
    None
):
    answers = iter(
        [
            _sections(("Q1", []), ("Q2", ["SaltLog is cheap."])),
            _sections(("Q1", ["Apollo cuts latency."]), ("Q2", ["SaltLog is cheap."])),
        ]
    )
    llm = ScriptedLLM(lambda _i, _p: next(answers))
    report = await _write(llm)
    assert len(llm.calls) == 2  # the same joint call, repaired once
    assert "Apollo cuts latency [F1]." in report.markdown
    assert "SaltLog is cheap [F3]." in report.markdown
    assert not hasattr(report, "degraded")


async def test_a_section_of_only_citations_or_blanks_is_retried_by_name() -> None:
    answers = iter(
        [
            _sections(("Q1", ["[F1]"]), ("Q2", ["SaltLog is cheap."])),
            _sections(("Q1", ["  "]), ("Q2", ["SaltLog is cheap."])),
            _sections(("Q1", ["Apollo cuts latency."]), ("Q2", ["SaltLog is cheap."])),
        ]
    )
    prompts: list[str] = []
    llm = ScriptedLLM(lambda _i, p: (prompts.append(p), next(answers))[1])
    report = await _write(llm)
    assert len(llm.calls) == 3
    assert "Apollo cuts latency [F1]." in report.markdown
    assert "[F1]." not in report.markdown.split("Apollo cuts latency")[0].split("##")[1]


async def test_the_schema_requires_a_non_empty_sentence_list() -> None:
    import pytest
    from pydantic import ValidationError

    from ..write import ReportDraft, SectionDraft

    with pytest.raises(ValidationError):
        SectionDraft(subquestion="Q1")  # type: ignore[call-arg]
    with pytest.raises(ValidationError):
        SectionDraft(subquestion="Q1", sentences=[])
    with pytest.raises(ValidationError):
        ReportDraft(sections=[])


async def test_a_joint_call_that_stays_empty_is_replaced_by_one_call_per_section() -> (
    None
):
    def respond(_instructions: str, prompt: str) -> dict[str, object]:
        if _alone(prompt):
            sub_id = "Q1" if "SUB-QUESTION Q1:" in prompt else "Q2"
            return _sections((sub_id, [f"Written alone for {sub_id}."]))
        return _sections(("Q1", []), ("Q2", []))  # every joint attempt is empty

    llm = ScriptedLLM(respond)
    asker = _asker(llm)
    report = await _write(llm, asker=asker)
    assert "Written alone for Q1 [F1]." in report.markdown
    assert "Written alone for Q2 [F3]." in report.markdown
    alone = [p for _, p in llm.calls if _alone(p)]
    assert len(alone) == 2
    assert all(
        ("SUB-QUESTION Q2:" not in p) or ("SUB-QUESTION Q1:" not in p) for p in alone
    )
    # one counted joint call, one counted call per section
    assert asker.stats.step_llm_calls["Write"] == 3


async def test_a_section_the_model_cannot_write_raises_naming_the_subquestion() -> None:
    def respond(_instructions: str, prompt: str) -> dict[str, object]:
        if _alone(prompt) and "SUB-QUESTION Q2:" in prompt:
            return _sections(("Q2", []))  # never writes Q2
        if _alone(prompt):
            return _sections(("Q1", ["Fine."]))
        return _sections(("Q1", []), ("Q2", []))

    with pytest.raises(ReportWriterError) as caught:
        await _write(ScriptedLLM(respond))
    assert caught.value.subquestion_id == "Q2"
    assert "Q2" in str(caught.value) and "What does SaltLog cost?" in str(caught.value)
    assert caught.value.reason


async def test_a_failing_model_raises_instead_of_listing_claims() -> None:
    def respond(_instructions: str, _prompt: str) -> dict[str, object]:
        return {"sections": []}  # always rejected

    with pytest.raises(ReportWriterError) as caught:
        await _write(ScriptedLLM(respond))
    assert caught.value.subquestion_id == "Q1"
    assert "the model failed" in caught.value.reason


async def test_a_spent_budget_raises_instead_of_listing_claims() -> None:
    llm = ScriptedLLM(_writer())
    asker = make_asker(model=llm, budget=Budget(max_llm_calls=1))
    asker.stats.llm_calls = 1
    with pytest.raises(ReportWriterError) as caught:
        await write_report(
            "q",
            subquestions=[Q1, Q2],
            answers={},
            findings=[F1, F3],
            works=WORKS,
            asker=asker,
        )
    assert llm.calls == []
    assert "budget:max_llm_calls" in caught.value.reason


def test_the_reserve_covers_the_joint_call_and_one_call_per_subquestion() -> None:
    assert write_reserve(1) == 2
    assert write_reserve(4) == 5


def test_there_is_no_plain_claims_fallback() -> None:
    import andamentum.deep_research.write as module

    assert not hasattr(module, "_claims")
    assert "lists its findings' claims" not in Path(module.__file__).read_text()


async def test_a_question_without_findings_gets_the_fixed_line_and_no_call() -> None:
    llm = ScriptedLLM(_writer())
    report = await write_report(
        "q",
        subquestions=[Q1, Q3],
        answers={},
        findings=[F1],
        works=WORKS,
        asker=make_asker(model=llm),
    )
    assert len(llm.calls) == 1
    assert "SUB-QUESTION Q3" not in llm.calls[0][1]
    assert report.markdown.count(NO_EVIDENCE_LINE) == 1
    assert f"## {Q3.text}\n\n{NO_EVIDENCE_LINE}" in report.markdown
    assert NO_EVIDENCE_LINE in CODE_LINE_PREFIXES


async def test_no_findings_at_all_means_no_model_call() -> None:
    llm = ScriptedLLM(_writer())
    asker = make_asker(model=llm)
    report = await write_report(
        "q",
        subquestions=[Q1, Q2],
        answers={},
        findings=[],
        works=WORKS,
        asker=asker,
    )
    assert llm.calls == [] and asker.stats.llm_calls == 0
    assert report.markdown.count(NO_EVIDENCE_LINE) == 2
    assert SOURCES_HEADING not in report.markdown
    assert report.cited_finding_ids == []


async def test_findings_of_a_retracted_work_are_never_offered() -> None:
    retracted_passage = Passage.make(work_id="W4", start=0, text="Apollo is useless.")
    bad = _finding(4, retracted_passage, "Apollo is useless.")
    llm = ScriptedLLM(_writer())
    report = await write_report(
        "q",
        subquestions=[Q1],
        answers={},
        findings=[F1, bad],
        works=WORKS,
        asker=make_asker(model=llm),
    )
    assert "Apollo is useless" not in llm.calls[0][1]
    assert report.excluded_finding_ids == ["F4"]
    assert "F4" not in report.cited_finding_ids

    only_bad = await write_report(
        "q",
        subquestions=[Q1],
        answers={},
        findings=[bad],
        works=WORKS,
        asker=make_asker(model=ScriptedLLM(_writer())),
    )
    assert NO_EVIDENCE_LINE in only_bad.markdown
    assert only_bad.excluded_finding_ids == ["F4"]


async def test_at_most_the_cap_is_offered_one_origin_at_a_time() -> None:
    works = {
        f"W{n}": Work(work_id=f"W{n}", title=f"T{n}", authors=[f"Author{n}"])
        for n in range(1, 4)
    }
    findings: list[Finding] = []
    for n in range(1, 41):  # W1 holds 40 findings, W2 and W3 one each
        passage = Passage.make(work_id="W1", start=n, text=f"Fact number {n}.")
        findings.append(_finding(n, passage, f"claim {n}"))
    for n, work_id in ((41, "W2"), (42, "W3")):
        passage = Passage.make(work_id=work_id, start=0, text=f"Other {n}.")
        findings.append(_finding(n, passage, f"claim {n}"))
    seen: list[str] = []
    await write_report(
        "q",
        subquestions=[Q1],
        answers={},
        findings=findings,
        works=works,
        asker=make_asker(model=ScriptedLLM(_writer(seen))),
    )
    prompt = seen[0]
    offered = [
        line
        for line in prompt.splitlines()
        if line.startswith("F") and line[1:2].isdigit()
    ]
    assert len(offered) == MAX_FINDINGS_PER_SECTION
    assert "F41" in prompt and "F42" in prompt
