"""The offline benchmark: harness arm on the planted set, with measured floors.

Measured (offline, fixtures; seeds 0, 1, 2, 3 and 12345 gave identical totals):
50 of 50 planted defects found (every kind 100%), 0 false positives on the 4
clean files and 0 on the defective ones. The floors below sit under that so a
regression fails the test but a one-defect wobble does not:

- overall recall >= 0.9 and every kind >= 0.8,
- no false positive on a clean file,
- at most 2 false positives on defective files.

The controls prove the scoring can fail: with a check left out, the kinds it
reports drop to zero recall.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic_ai.usage import RunUsage

from andamentum.compose.harness.bench import run as bench
from andamentum.compose.harness.bench.planted import KINDS, generate
from andamentum.compose.harness.check import CheckParams, run_check
from andamentum.compose.harness.docx.reader import open_docx
from andamentum.compose.harness.kernel import Diagnostic

from andamentum.compose.harness.bench.fixture_http import (
    fixture_transport,
    scheme_only_url_check,
)

RECALL_FLOOR = 0.9
KIND_FLOOR = 0.8
FALSE_POSITIVES_ELSEWHERE_CEILING = 2


@pytest.fixture(scope="module")
def planted(tmp_path_factory: pytest.TempPathFactory) -> Path:
    folder = tmp_path_factory.mktemp("planted")
    generate(folder, seed=0)
    return folder


@pytest.fixture(scope="module")
def result(
    planted: Path, tmp_path_factory: pytest.TempPathFactory
) -> bench.BenchResult:
    import asyncio

    return asyncio.run(
        bench.run_benchmark(
            bench.load_set(planted),
            work=tmp_path_factory.mktemp("work"),
            offline=True,
            seed=0,
            baseline_model=None,
        )
    )


def test_set_shape(result: bench.BenchResult):
    assert result.files >= 12 and result.clean_files >= 4 and result.defects >= 40
    assert all(result.harness.per_kind[k].planted >= 5 for k in KINDS)


def test_recall_floors(result: bench.BenchResult):
    harness = result.harness
    assert harness.recall >= RECALL_FLOOR, bench.table(result)
    for kind in KINDS:
        recall = harness.per_kind[kind].recall
        assert recall is not None and recall >= KIND_FLOOR, (kind, bench.table(result))


def test_false_positive_floors(result: bench.BenchResult):
    assert result.harness.false_positives_clean == 0, bench.table(result)
    assert (
        result.harness.false_positives_defective <= FALSE_POSITIVES_ELSEWHERE_CEILING
    ), bench.table(result)


def test_the_offline_harness_arm_makes_no_model_calls_and_uses_fixtures(
    result: bench.BenchResult,
):
    assert result.harness.input_tokens == 0 and result.harness.output_tokens == 0
    assert result.harness.network_calls > 0  # served by the fixture transport


def test_table_has_every_kind(result: bench.BenchResult):
    text = bench.table(result)
    for kind in KINDS:
        assert kind in text
    assert "baseline" not in text


# ---------------------------------------------------------------- controls


async def test_leaving_a_check_out_drops_its_kinds_to_zero(
    planted: Path, tmp_path: Path
):
    files = bench.load_set(planted)
    import shutil

    score = bench._empty_score("crossrefs only")
    for docx, truth in files:
        copy = tmp_path / docx.name
        shutil.copy(docx, copy)
        run = await run_check(
            CheckParams(file=copy, checks=["crossrefs"]),
            transport=fixture_transport(),
            url_check=scheme_only_url_check,
        )
        bench.score_file(
            truth,
            bench.harness_findings(run.report.diagnostics, open_docx(copy)),
            score,
        )
    for kind in (
        "fabricated_doi",
        "retracted_doi",
        "broken_url",
        "inconsistent_statistic",
    ):
        assert score.per_kind[kind].found == 0
    assert score.per_kind["dangling_crossref"].recall == 1.0
    assert score.per_kind["missing_figure"].recall == 1.0
    assert score.false_positives == 0


def test_a_finding_at_the_wrong_place_or_kind_is_not_a_hit():
    truth = bench.Truth(
        file="x.docx",
        clean=False,
        defects=[
            bench.TruthDefect(
                id="d1",
                kind="broken_url",
                locations=[bench.TruthLocation(paragraph_index=3)],
            )
        ],
    )
    score = bench._empty_score("t")
    bench.score_file(
        truth,
        [
            bench.Finding(kind="broken_url", paragraph=4, severity="warning"),
            bench.Finding(kind="fabricated_doi", paragraph=3, severity="warning"),
            bench.Finding(kind=None, paragraph=None, severity="warning"),
        ],
        score,
    )
    assert score.per_kind["broken_url"].found == 0
    assert score.false_positives_defective == 3
    bench.score_file(
        truth,
        [bench.Finding(kind="broken_url", paragraph=3, severity="warning")],
        score,
    )
    assert score.per_kind["broken_url"].found == 1
    assert score.per_kind["broken_url"].planted == 2
    assert score.false_positives_defective == 3


def test_every_finding_on_a_clean_file_is_a_false_positive():
    truth = bench.Truth(file="c.docx", clean=True, defects=[])
    score = bench._empty_score("t")
    bench.score_file(
        truth, [bench.Finding(kind="broken_url", paragraph=1, severity="error")], score
    )
    assert score.false_positives_clean == 1


def test_info_diagnostics_are_not_findings():
    view = open_docx_stub()
    found = bench.harness_findings(
        [
            Diagnostic(
                check="needs_citation",
                tier=2,
                severity="info",
                paragraph_id="p0",
                start=0,
                end=1,
                message="m",
                evidence=[],
            )
        ],
        view,
    )
    assert found == []


def open_docx_stub():
    from ._helpers import make_view

    return make_view(("text", None))


# ------------------------------------------------------------ baseline arm


async def test_baseline_arm_scores_the_models_labelled_findings(
    planted: Path, monkeypatch: pytest.MonkeyPatch
):
    files = bench.load_set(planted)[:2]
    seen: list[str] = []

    async def fake(model, *, instructions, output_type, user_message, **_kw):
        seen.append(user_message)
        assert model == "stub:strong"
        assert user_message.startswith("[0] ")
        return (
            bench.BaselineFindings(
                findings=[
                    bench.BaselineFinding(
                        paragraph=0, category="other", quote="q", problem="p"
                    )
                ]
            ),
            RunUsage(input_tokens=100, output_tokens=20),
        )

    monkeypatch.setattr(bench, "run_agent_with_usage", fake)
    score = await bench.baseline_arm(files, model="stub:strong")
    assert len(seen) == 2
    assert (score.input_tokens, score.output_tokens) == (200, 40)
    assert score.found == 0 and score.findings == 2
    assert score.false_positives >= 2  # an 'other' finding maps to no kind


async def test_baseline_arm_counts_a_correct_labelled_finding(
    planted: Path, monkeypatch: pytest.MonkeyPatch
):
    docx, truth = next((d, t) for d, t in bench.load_set(planted) if t.defects)
    defect = truth.defects[0]

    async def fake(model, **_kw):
        return (
            bench.BaselineFindings(
                findings=[
                    bench.BaselineFinding(
                        paragraph=defect.locations[0].paragraph_index,
                        category=defect.kind,  # type: ignore[arg-type]
                        quote="q",
                        problem="p",
                    )
                ]
            ),
            RunUsage(input_tokens=1, output_tokens=1),
        )

    monkeypatch.setattr(bench, "run_agent_with_usage", fake)
    score = await bench.baseline_arm([(docx, truth)], model="stub:strong")
    assert score.per_kind[defect.kind].found == 1


async def test_table_with_a_baseline_column(
    planted: Path, result: bench.BenchResult, monkeypatch: pytest.MonkeyPatch
):
    base = bench._empty_score("baseline")
    with_base = result.model_copy(update={"baseline": base})
    text = bench.table(with_base)
    assert "baseline found" in text and "baseline recall" in text


def test_cli_prints_the_table_and_writes_json(
    planted: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    out = tmp_path / "result.json"
    code = bench.main(
        [
            "--offline",
            "--dir",
            str(planted),
            "--work",
            str(tmp_path / "w"),
            "--json",
            str(out),
        ]
    )
    assert code == 0
    printed = capsys.readouterr().out
    assert "| **all** | 50 | 50 | 100% |" in printed
    assert json.loads(out.read_text("utf-8"))["mode"] == "offline"


def test_cli_needs_a_mode():
    with pytest.raises(SystemExit):
        bench.main([])
