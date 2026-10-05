"""WRITE: a report built from findings, with every sentence cited."""

from __future__ import annotations

import re

import httpx

from andamentum.references import Work

from ..types import Budget, Conflict, Coverage, Criterion, Finding, Passage, SubQuestion
from ..verify import verify_report
from ..write import (
    DISAGREEMENTS_HEADING,
    LIMITED_EVIDENCE_PREFIX,
    MAX_FINDINGS_PER_SECTION,
    NO_EVIDENCE_LINE,
    SOURCES_HEADING,
    write_report,
)
from .llm_stub import ScriptedLLM, decisions, http, make_asker

Q1 = SubQuestion(id="Q1", text="Does Apollo cut latency?")
Q2 = SubQuestion(id="Q2", text="What does SaltLog cost?")
Q3 = SubQuestion(id="Q3", text="Is Apollo free to use?")
CRITERIA = {
    "Q1": [
        Criterion(id="Q1-C1", text="Is it measured?", kind="must", applies_to="passage")
    ]
}

P1 = Passage.make(
    work_id="W1", start=0, text="Apollo cut median latency from 120 ms to 45 ms."
)
P2 = Passage.make(
    work_id="W2", start=0, text="Apollo raised median latency from 45 ms to 80 ms."
)
P3 = Passage.make(
    work_id="W3", start=0, text="SaltLog verbose mode adds roughly 3 ms per request."
)
PASSAGES = {p.passage_id: p for p in (P1, P2, P3)}


def _finding(
    number: int, passage: Passage, quote: str, claim: str, subq: str = "Q1"
) -> Finding:
    return Finding(
        finding_id=f"F{number}",
        subquestion_id=subq,
        claim=claim,
        quote=quote,
        passage_id=passage.passage_id,
        work_id=passage.work_id,
        criterion_ids=["Q1-C1"] if subq == "Q1" else [],
    )


F1 = _finding(
    1, P1, "cut median latency from 120 ms to 45 ms", "Apollo cut median latency."
)
F2 = _finding(
    2, P2, "raised median latency from 45 ms to 80 ms", "Apollo raised median latency."
)
F3 = _finding(3, P3, "adds roughly 3 ms per request", "SaltLog adds 3 ms.", subq="Q2")
WORKS = {
    "W1": Work(
        work_id="W1",
        doi="10.5555/apollo",
        title="Apollo caching",
        authors=["Jane Doe"],
        year=2024,
        venue="Journal of Caches",
    ),
    "W2": Work(
        work_id="W2",
        url="https://example.org/apollo-review",
        title="Apollo review",
        authors=["Max Muster", "Ann Roe", "A B", "C D"],
        year=2023,
    ),
    "W3": Work(
        work_id="W3",
        url="https://example.net/saltlog",
        title="SaltLog costs",
        year=2022,
    ),
}


def _respond(instructions: str, prompt: str) -> dict[str, object]:
    if "QUESTION: Does Apollo" in prompt:
        return {
            "sentences": [
                {
                    "text": "Apollo cut median latency in one benchmark [F9].",
                    "finding_ids": ["F1"],
                },
                {
                    "text": "Another benchmark found it raised latency.",
                    "finding_ids": ["F2"],
                },
            ]
        }
    return {
        "sentences": [
            {"text": "SaltLog adds about 3 ms per request.", "finding_ids": ["F3"]}
        ]
    }


def _coverage(subq: str, sources: int) -> Coverage:
    return Coverage(
        subquestion_id=subq, independent_sources=sources, covered=sources >= 2
    )


async def _report(llm: ScriptedLLM, **overrides: object):
    asker = make_asker(model=llm, budget=overrides.pop("budget", None))  # type: ignore[arg-type]
    arguments: dict[str, object] = {
        "subquestions": [Q1, Q2, Q3],
        "criteria": CRITERIA,
        "findings": [F1, F2, F3],
        "conflicts": [
            Conflict(
                subquestion_id="Q1", finding_ids=["F1", "F2"], note="language model"
            )
        ],
        "coverage": [_coverage("Q1", 2), _coverage("Q2", 1), _coverage("Q3", 0)],
        "works": WORKS,
        "asker": asker,
    }
    arguments.update(overrides)
    return await write_report("Does Apollo help?", **arguments)  # type: ignore[arg-type]


async def test_every_sentence_ends_with_its_finding_ids() -> None:
    llm = ScriptedLLM(_respond)
    report = await _report(llm)
    md = report.markdown

    assert md.startswith("# Does Apollo help?\n")
    assert "## Does Apollo cut latency?" in md
    # The stray id the model wrote into the text is gone; the code's ids stand.
    assert "Apollo cut median latency in one benchmark [F1]." in md
    assert "Another benchmark found it raised latency [F2]." in md
    assert "SaltLog adds about 3 ms per request [F3]." in md
    assert "[F9]" not in md
    # The sub-question without findings says so, in a fixed line.
    assert f"## Is Apollo free to use?\n\n{NO_EVIDENCE_LINE}" in md
    # An uncovered one states how thin the evidence is.
    assert f"{LIMITED_EVIDENCE_PREFIX} 1 independent source found" in md
    assert report.cited_finding_ids == ["F1", "F2", "F3"]
    assert llm.calls and len(llm.calls) == 2  # Q3 has nothing to write


async def test_conflicts_are_stated_from_the_two_claims() -> None:
    report = await _report(ScriptedLLM(_respond))
    section = report.markdown.split(DISAGREEMENTS_HEADING)[1].split(SOURCES_HEADING)[0]
    assert (
        "- Sources disagree. One reports: Apollo cut median latency [F1]; "
        "another reports: Apollo raised median latency [F2]." in section
    )


async def test_sources_lists_each_cited_work_with_its_link() -> None:
    report = await _report(ScriptedLLM(_respond))
    sources = report.markdown.split(SOURCES_HEADING)[1]
    assert (
        "- [F1] Jane Doe (2024). Apollo caching. Journal of Caches. https://doi.org/10.5555/apollo"
        in sources
    )
    assert (
        "- [F2] Max Muster, Ann Roe, A B et al. (2023). Apollo review. https://example.org/apollo-review"
        in sources
    )
    assert "- [F3] (2022). SaltLog costs. https://example.net/saltlog" in sources


async def test_the_model_sees_the_findings_with_their_quotes_and_the_criteria() -> None:
    llm = ScriptedLLM(_respond)
    await _report(llm)
    prompt = llm.calls[0][1]
    assert "F1 [criteria: Q1-C1]\n<<<\nsource: Apollo caching, 2024\n" in prompt
    assert 'quote: "cut median latency from 120 ms to 45 ms"' in prompt
    assert "- Q1-C1: Is it measured?" in prompt
    assert "SaltLog" not in prompt  # the other sub-question's findings stay out


async def test_an_id_that_is_not_a_finding_of_the_section_is_sent_back() -> None:
    attempts = 0

    def respond(instructions: str, prompt: str) -> dict[str, object]:
        nonlocal attempts
        attempts += 1
        ids = ["F7"] if attempts == 1 else ["F1"]
        return {
            "sentences": [{"text": "Apollo cut median latency.", "finding_ids": ids}]
        }

    llm = ScriptedLLM(respond)
    report = await _report(
        llm, subquestions=[Q1], findings=[F1], conflicts=[], coverage=[]
    )
    assert attempts == 2
    assert "Apollo cut median latency [F1]." in report.markdown


async def test_a_retracted_works_findings_are_not_used() -> None:
    retracted = {
        **WORKS,
        "W2": WORKS["W2"].model_copy(update={"retracted": "retracted"}),
    }

    # With W2 retracted, F2 is not offered to the model at all.
    def respond(instructions: str, prompt: str) -> dict[str, object]:
        assert "F2" not in prompt
        if "SaltLog" in prompt:
            return {
                "sentences": [{"text": "SaltLog adds 3 ms.", "finding_ids": ["F3"]}]
            }
        return {
            "sentences": [{"text": "Apollo cut median latency.", "finding_ids": ["F1"]}]
        }

    llm = ScriptedLLM(respond)
    report = await _report(llm, works=retracted)
    assert report.excluded_finding_ids == ["F2"]
    assert "F2" not in report.markdown
    # A conflict that involves an excluded finding is not stated.
    assert DISAGREEMENTS_HEADING not in report.markdown


async def test_one_source_cannot_fill_the_prompt() -> None:
    many = [
        _finding(n, P1, "cut median latency from 120 ms to 45 ms", f"Apollo claim {n}.")
        for n in range(1, 41)
    ]
    other = _finding(
        41, P2, "raised median latency from 45 ms to 80 ms", "Apollo claim 41."
    )
    seen: list[str] = []

    def respond(instructions: str, prompt: str) -> dict[str, object]:
        seen.extend(re.findall(r"^F\d+", prompt, flags=re.MULTILINE))
        return {"sentences": [{"text": "Claim.", "finding_ids": ["F41"]}]}

    await _report(
        ScriptedLLM(respond),
        subquestions=[Q1],
        findings=[*many, other],
        conflicts=[],
        coverage=[],
    )
    assert len(seen) == MAX_FINDINGS_PER_SECTION
    assert "F41" in seen  # the second origin is not crowded out


async def test_a_spent_budget_lists_the_claims_of_the_sections_left() -> None:
    report = await _report(ScriptedLLM(_respond), budget=Budget(max_llm_calls=1))
    assert report.degraded and all("budget:max_llm_calls" in d for d in report.degraded)
    for finding_id in report.cited_finding_ids:
        assert (
            f"[{finding_id}" in report.markdown or f", {finding_id}" in report.markdown
        )


async def test_a_failing_writer_lists_the_claims_and_says_so() -> None:
    def broken(instructions: str, prompt: str) -> dict[str, object]:
        return {"sentences": "not a list"}

    report = await _report(ScriptedLLM(broken))
    assert report.degraded and "the writer failed" in report.degraded[0]
    assert F1.claim.rstrip(".") in report.markdown


async def test_a_written_report_passes_verification_structurally() -> None:
    """Every sentence the writer produces is one VERIFY recognises, with ids."""
    report = await _report(ScriptedLLM(_respond))
    decision_model, _ = decisions(lambda *_: 0.9)
    asker = make_asker(decision_model=decision_model)

    def ok(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"message": {}})

    outcome = await verify_report(
        report.markdown,
        findings=[F1, F2, F3],
        passages=PASSAGES,
        works=WORKS,
        asker=asker,
        http=http(ok),
    )
    kinds = {f.kind for f in outcome.flags}
    assert kinds <= {"single_source"}
    assert outcome.sentences_checked == 5  # 3 section sentences + 2 disagreement halves
    assert "[unsupported]" not in outcome.report_markdown
