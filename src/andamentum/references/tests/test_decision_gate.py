"""A decision model replaces the language-model call for open candidates."""

from __future__ import annotations

from typing import Any, cast

import pytest

from andamentum.core.decisions import Answer, DecisionModel
from andamentum.references import check_references, llm

DOC = """Introduction
Sodium intake varied between sites1. Sleep was shorter in the high group2. Readings were taken at day3 of the trial. Earlier work3 agrees. Sodium intake varied again1. Later reads at week2 were excluded.

References
1. Marlow A, Voss L (2015) Sodium intake across six clinics. J Apollo Stud 12:1-9
2. Okafor L (2013) Sleep duration in a pilot study. SaltLog Rev 3:4-8
3. Hollis B (2012) Salt sensitivity in a registry study. SaltLog Rev 9:2-7
"""


class FakeDecisionModel:
    """Stands in for an opened DecisionModel: yes only for ``work3``."""

    def __init__(self) -> None:
        self.calls: list[tuple[Any, dict[str, Any]]] = []

    async def ask_all(self, state: Any, questions: dict[str, Any]) -> dict[str, Answer]:
        self.calls.append((state, questions))
        out = {}
        for key in questions:
            marked = state["candidates"][key]["marked_item_in_context"]
            p = 0.9 if "work⟦3⟧" in marked else 0.1
            out[key] = Answer(
                choice="yes" if p >= 0.5 else "no",
                probabilities={"yes": p, "no": 1 - p},
                confidence=max(p, 1 - p),
                p_true=p,
            )
        return out


class FailingDecisionModel:
    async def ask_all(self, state: Any, questions: dict[str, Any]) -> dict[str, Answer]:
        raise RuntimeError("server gone")


def summary(report) -> list[tuple[int, bool, str]]:
    return [(j.candidate.id, j.is_citation, j.reason) for j in report.judged]


async def test_without_a_decision_model_behaviour_is_unchanged() -> None:
    report = await check_references(DOC, lookup=False)
    assert [(i, c) for i, c, _ in summary(report)] == [
        (0, True),
        (1, True),
        (2, True),
        (3, False),
        (4, False),
        (5, False),
    ]
    assert all(r.startswith("not judged") for _, c, r in summary(report) if not c)


async def test_the_decision_model_replaces_the_language_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def refuse(*_a: object, **_k: object):
        raise AssertionError("the language model was called")

    monkeypatch.setattr(llm, "ask_citations", refuse)
    fake = FakeDecisionModel()
    report = await check_references(
        DOC,
        lookup=False,
        model="stub:model",
        decision_model=cast(DecisionModel, fake),
    )
    by_id = {j.candidate.id: j for j in report.judged}
    assert [by_id[i].is_citation for i in (3, 4, 5)] == [True, False, False]
    assert by_id[3].reason.startswith("decision model")
    assert len(fake.calls) == 1
    assert sorted(fake.calls[0][1]) == ["3", "4", "5"]
    # The rule-settled candidates were never asked.
    assert "0" not in fake.calls[0][1]


async def test_a_failing_decision_model_leaves_candidates_not_judged() -> None:
    report = await check_references(
        DOC, lookup=False, decision_model=cast(DecisionModel, FailingDecisionModel())
    )
    by_id = {j.candidate.id: j for j in report.judged}
    assert all(
        by_id[i].reason.startswith("not judged: the decision model call failed")
        for i in (3, 4, 5)
    )
