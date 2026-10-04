from __future__ import annotations

from ._stub import with_usage
from andamentum.compose.checks import grounding
from andamentum.compose.checks.grounding import (
    GroundingVerdict,
    Unsupported,
    check_grounding,
    cited_sentences,
)
from andamentum.compose.log import ActionLog
from andamentum.compose.schemas import Fact
from andamentum.compose.bundle import Section

FACTS = [
    Fact(
        id="f1",
        source_id="a",
        statement="Apollo cut costs 12%",
        quote="costs fell by 12%",
        start=0,
        end=5,
    ),
    Fact(
        id="f2",
        source_id="a",
        statement="Launch in May",
        quote="launch is set for May",
        start=6,
        end=9,
    ),
]


def sec(i: int, body: str) -> Section:
    return Section(
        id=f"s{i:02d}",
        title=f"T{i}",
        order=i * 10,
        revision=1,
        status="drafted",
        body=body,
    )


def test_marker_attaches_to_preceding_sentence():
    pairs = cited_sentences(
        "Costs fell. [[f1]] Launch is in May [[f2]]. Nothing cited here."
    )
    assert pairs == [("Costs fell.", ["f1"]), ("Launch is in May.", ["f2"])]


def test_sentences_without_markers_are_ignored():
    assert cited_sentences("Plain sentence. Another one.") == []


async def test_one_call_per_changed_section_and_cache(tmp_path, monkeypatch):
    calls: list[str] = []

    async def fake(model, *, instructions, output_type, user_message, **kw):
        calls.append(user_message)
        assert model == "judge:m"
        assert output_type is GroundingVerdict
        if "Launch" in user_message:
            return GroundingVerdict(
                unsupported=[
                    Unsupported(
                        marker="f2",
                        sentence="Launch is in June.",
                        reason="quote says May",
                    )
                ]
            )
        return GroundingVerdict(unsupported=[])

    monkeypatch.setattr(grounding, "run_agent_with_usage", with_usage(fake))
    log = ActionLog(tmp_path)
    cache: dict = {}
    sections = [
        sec(1, "Costs fell 12%. [[f1]]"),
        sec(2, "Launch is in June. [[f2]]"),
        sec(3, "No citations at all."),
    ]
    results = await check_grounding(
        sections, facts=FACTS, model="judge:m", log=log, cache=cache
    )
    assert len(calls) == 2
    assert [(r.section_id, r.passed) for r in results] == [
        ("s01", True),
        ("s02", False),
    ]
    assert "f2" in results[1].details[0]
    assert "quote says May" in calls[1] or "launch is set for May" in calls[1]

    again = await check_grounding(
        sections, facts=FACTS, model="judge:m", log=log, cache=cache
    )
    assert len(calls) == 2  # all cached
    assert [r.passed for r in again] == [True, False]

    changed = [sec(1, "Costs fell 12%. [[f1]]"), sec(2, "Launch is in May. [[f2]]")]
    await check_grounding(changed, facts=FACTS, model="judge:m", log=log, cache=cache)
    assert len(calls) == 3  # only the changed section was re-judged
    assert all(r["role"] == "judge" for r in log.records())


async def test_unknown_fact_ids_are_left_to_deterministic_check(tmp_path, monkeypatch):
    async def boom(*a, **k):
        raise AssertionError("no model call expected")

    monkeypatch.setattr(grounding, "run_agent_with_usage", boom)
    results = await check_grounding(
        [sec(1, "Claim. [[f99]]")],
        facts=FACTS,
        model="m",
        log=ActionLog(tmp_path),
        cache={},
    )
    assert results == []
