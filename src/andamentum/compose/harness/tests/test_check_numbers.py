"""The numbers check: a stubbed model extracts, code compares (no model, no network)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic_ai.usage import RunUsage

from andamentum.compose.harness.checks import numbers
from andamentum.compose.harness.checks.numbers import (
    NumbersCheck,
    NumericClaim,
    NumericClaims,
    load_results,
)
from andamentum.compose.harness.kernel import CheckContext, ProjectStore, select

from ._helpers import make_view

PARAGRAPH = "In the intervention arm osmolality fell by 23.4% and 159 people took part."


class StubModel:
    def __init__(self, claims: dict[str, list[NumericClaim]]) -> None:
        self.claims = claims
        self.calls: list[str] = []

    async def __call__(self, model, *, user_message, **_kw):
        self.calls.append(user_message)
        return (
            NumericClaims(claims=self.claims.get(user_message, [])),
            RunUsage(input_tokens=10, output_tokens=4),
        )


def claim(value: str, *, sentence: str = PARAGRAPH, refers: str = "a result"):
    """A claim; the default sentence is the whole paragraph."""
    return NumericClaim(sentence=sentence, value=value, refers_to=refers)


@pytest.fixture
def results(tmp_path: Path) -> Path:
    folder = tmp_path / "results"
    folder.mkdir()
    (folder / "summary.csv").write_text(
        "arm,effect,n\nintervention,0.234,159\ncontrol,0.05,160\n", "utf-8"
    )
    (folder / "extra.json").write_text(
        json.dumps({"models": [{"auc": 0.81}], "flag": True}), "utf-8"
    )
    (folder / "notes.txt").write_text("99999", "utf-8")  # not a result file
    return folder


def ctx(results: Path, store: ProjectStore | None = None) -> CheckContext:
    return CheckContext(
        store=store, offline=True, models={"model": "stub:model"}, results_dir=results
    )


def install(monkeypatch: pytest.MonkeyPatch, stub: StubModel) -> StubModel:
    monkeypatch.setattr(numbers, "run_agent_with_usage", stub)
    return stub


def test_registered_tier_one_and_needs_results_and_model():
    (check,) = select(ids=["numbers"])
    assert check.spec.tier == 1
    assert set(check.spec.needs) == {"results", "model"}
    assert select(needs=["network"], ids=["numbers"]) == []


def test_load_results_reads_csv_and_json_only(results: Path):
    found, digests = load_results(results)
    values = sorted(r.value for r in found)
    assert 0.234 in values and 159.0 in values and 0.81 in values
    assert 99999.0 not in values
    assert True not in values and 1.0 not in values  # the JSON boolean is not a number
    assert len(digests) == 2
    assert any("summary.csv row 2 column 'effect'" in r.where for r in found)


def test_load_results_needs_a_folder(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_results(tmp_path / "missing")


async def test_matched_claims_are_silent(monkeypatch, results):
    stub = install(
        monkeypatch,
        StubModel({PARAGRAPH: [claim("23.4%"), claim("159", refers="participants")]}),
    )
    out = await NumbersCheck().run(make_view((PARAGRAPH, None)), ctx(results))
    assert out == []
    assert stub.calls == [PARAGRAPH]


async def test_percent_matches_a_fraction_at_the_written_precision(
    monkeypatch, results
):
    text = "Osmolality fell by 23% in one analysis and by 23.4% in the other."
    install(
        monkeypatch,
        StubModel({text: [claim("23%", sentence=text), claim("23.4%", sentence=text)]}),
    )
    assert await NumbersCheck().run(make_view((text, None)), ctx(results)) == []


async def test_unmatched_claim_warns_with_nearest_candidates(monkeypatch, results):
    text = "In the intervention arm osmolality fell by 25.1% over twelve weeks."
    install(
        monkeypatch,
        StubModel({text: [claim("25.1%", sentence=text, refers="the drop")]}),
    )
    (d,) = await NumbersCheck().run(make_view((text, None)), ctx(results))
    assert d.severity == "warning"
    assert d.tier == 1
    assert text[d.start : d.end] == "25.1%"
    assert d.paragraph_id == "p0"
    assert "the drop" in d.message
    nearest = [e for e in d.evidence if e.startswith("nearest:")]
    assert 1 <= len(nearest) <= 3
    assert "summary.csv" in nearest[0]


async def test_precision_sets_the_tolerance(monkeypatch, results):
    text = "The effect size was 0.24 and the second was 0.5 for the groups."
    install(
        monkeypatch,
        StubModel({text: [claim("0.24", sentence=text), claim("0.5", sentence=text)]}),
    )
    found = await NumbersCheck().run(make_view((text, None)), ctx(results))
    # 0.24 vs stored 0.234: rounds to 0.23, so it does not match; 0.5 vs 0.05: no.
    assert [d.data["value"] for d in found] == ["0.24", "0.5"]


async def test_a_claim_not_in_the_text_is_dropped_and_counted(monkeypatch, results):
    install(
        monkeypatch,
        StubModel(
            {
                PARAGRAPH: [
                    claim("77%", sentence="A sentence the paragraph does not hold."),
                    claim("88%"),
                ]
            }
        ),
    )
    found = await NumbersCheck().run(make_view((PARAGRAPH, None)), ctx(results))
    assert [d.severity for d in found] == ["info"]
    assert "2 extracted numeric claim" in found[0].message
    assert found[0].paragraph_id is None


async def test_references_are_not_examined(monkeypatch, results):
    stub = install(monkeypatch, StubModel({}))
    view = make_view(
        (PARAGRAPH, None),
        ("References", 1),
        ("Smith, J. (2019). Something with 12 numbers inside it here.", None),
    )
    await NumbersCheck().run(view, ctx(results))
    assert stub.calls == [PARAGRAPH]


async def test_the_paragraph_limit_is_reported(monkeypatch, results):
    monkeypatch.setattr(numbers, "MAX_PARAGRAPHS", 2)
    stub = install(monkeypatch, StubModel({}))
    view = make_view(
        *[(f"Paragraph number {i} reports 5 things here.", None) for i in range(4)]
    )
    found = await NumbersCheck().run(view, ctx(results))
    assert len(stub.calls) == 2
    assert [d.severity for d in found] == ["info"]
    assert "2 paragraph(s)" in found[0].message


async def test_extractions_are_cached_in_the_project(monkeypatch, results, tmp_path):
    store = ProjectStore(tmp_path / "p.harness")
    text = "In the intervention arm osmolality fell by 25.1% over twelve weeks."
    stub = install(monkeypatch, StubModel({text: [claim("25.1%", sentence=text)]}))
    view = make_view((text, None))
    first = await NumbersCheck().run(view, ctx(results, store))
    again = await NumbersCheck().run(view, ctx(results, store))
    assert first == again and len(first) == 1
    assert len(stub.calls) == 1
    # changing a result file changes the key, so the paragraph is asked again
    (results / "summary.csv").write_text("arm,effect\nx,0.251\n", "utf-8")
    after = await NumbersCheck().run(view, ctx(results, store))
    assert len(stub.calls) == 2
    assert after == []


async def test_tokens_are_counted(monkeypatch, results):
    install(monkeypatch, StubModel({}))
    context = ctx(results)
    await NumbersCheck().run(make_view((PARAGRAPH, None)), context)
    assert context.stats.model_calls == 1
    assert (context.stats.input_tokens, context.stats.output_tokens) == (10, 4)


async def test_needs_results_and_a_model(results):
    with pytest.raises(ValueError, match="--results"):
        await NumbersCheck().run(
            make_view((PARAGRAPH, None)), CheckContext(store=None, offline=True)
        )
    empty = results.parent / "empty"
    empty.mkdir()
    (empty / "a.csv").write_text("a,b\nx,y\n", "utf-8")
    with pytest.raises(ValueError, match="no numbers"):
        await NumbersCheck().run(make_view((PARAGRAPH, None)), ctx(empty))
