"""Claim analysis: sentence kinds, number binding, the ledger file."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from andamentum.compose.harness.bindings.results import Catalogue, ResultEntry
from andamentum.compose.harness.draft.body import parse_body
from andamentum.compose.harness.draft.claims import (
    ClaimLabel,
    ClaimLabels,
    SourceInfo,
    analyse_section,
    claim_entries,
    raw_sentences,
    record_claims,
)
from andamentum.compose.harness.draft.works import CitableWork
from andamentum.compose.harness.kernel import ProjectStore
from andamentum.compose.log import ActionLog
from andamentum.compose.schemas import Fact, Source

from ._draft_helpers import WORK_ABSTRACT, WORK_CSL, Script, install_script


def source(number: int, label: str) -> Source:
    return Source(
        id=f"src{number}",
        path=label,
        title=label,
        sha256="0" * 64,
        chars=10,
        kind="file",
        label=label,
    )


def fact(number: int, source_id: str, quote: str = "a quote") -> Fact:
    return Fact(
        id=f"f{number}",
        source_id=source_id,
        statement=quote,
        quote=quote,
        start=0,
        end=len(quote),
    )


FACTS = {
    "f1": fact(1, "src1", "The pilot enrolled 48 children."),
    "f2": fact(2, "src2", "treatment,41.2,53.8"),
    "f3": fact(3, "src3", WORK_ABSTRACT),
}
INFOS = {
    "src1": SourceInfo(source=source(1, "notes.md")),
    "src2": SourceInfo(source=source(2, "scores.csv"), results_file="scores.csv"),
    "src3": SourceInfo(
        source=source(3, "A work"),
        work=CitableWork(csl=WORK_CSL, abstract=WORK_ABSTRACT),
    ),
}
CATALOGUE = Catalogue(
    entries=[
        ResultEntry(
            file="scores.csv", key="scores.csv row 2 column 'baseline'", value=41.2
        ),
        ResultEntry(
            file="scores.csv", key="scores.csv row 2 column 'followup'", value=53.8
        ),
        ResultEntry(
            file="scores.csv", key="scores.csv row 3 column 'followup'", value=43.1
        ),
        ResultEntry(file="other.csv", key="other.csv row 2 column 'x'", value=12.0),
    ],
    digest="d",
)


def test_markers_belong_to_the_sentence_they_stand_in() -> None:
    paragraphs = parse_body("First claim [[f1]]. Second claim. Third [[f2]][[f9]].")
    sentences, unknown = raw_sentences(paragraphs, known=FACTS)
    assert [(s.text, s.fact_ids) for s in sentences] == [
        ("First claim.", ("f1",)),
        ("Second claim.", ()),
        ("Third.", ("f2",)),
    ]
    assert unknown == ["f9"]


def test_table_rows_and_placeholders_are_not_claims() -> None:
    paragraphs = parse_body(
        "| a | b |\n|---|---|\n| 1 | 2 |\n\n[TO FILL: the sample size]"
    )
    sentences, _ = raw_sentences(paragraphs, known=FACTS)
    assert sentences == []


async def analyse(body: str, script: Script | None = None, **extra: object):
    return await analyse_section(
        parse_body(body),
        facts=FACTS,
        infos=INFOS,
        catalogue=extra.pop("catalogue", CATALOGUE),  # type: ignore[arg-type]
        taken_anchors=set(),
        model="stub:judge",
        cache={},
        log=None,
    )


async def test_markers_settle_result_and_background_without_the_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    script = install_script(monkeypatch)
    analysis = await analyse("Scores rose [[f2]]. Logging helps fidelity [[f3]].")
    assert [s.kind for s in analysis.sentences] == ["result", "background"]
    assert script.count(ClaimLabels) == 0


async def test_other_sentences_are_labelled_in_one_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def labels(message: str) -> ClaimLabels:
        assert message.splitlines() == [
            "1: We enrolled children.",
            "2: This suggests a causal route.",
        ]
        return ClaimLabels(
            labels=[
                ClaimLabel(number=1, kind="method"),
                ClaimLabel(number=2, kind="interpretation"),
            ]
        )

    script = install_script(monkeypatch, {ClaimLabels: labels})
    analysis = await analyse("We enrolled children. This suggests a causal route.")
    assert [s.kind for s in analysis.sentences] == ["method", "interpretation"]
    assert script.count(ClaimLabels) == 1


async def test_an_unlabelled_sentence_is_treated_as_interpretation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    install_script(monkeypatch, {ClaimLabels: lambda _: ClaimLabels(labels=[])})
    log = ActionLog(tmp_path)
    analysis = await analyse_section(
        parse_body("Something unlabelled."),
        facts=FACTS,
        infos=INFOS,
        catalogue=None,
        taken_anchors=set(),
        model="stub:judge",
        cache={},
        log=log,
    )
    assert analysis.sentences[0].kind == "interpretation"
    assert any(r.get("event") == "label_missing" for r in log.records())


async def test_the_label_call_is_cached_by_the_sentences(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    script = install_script(
        monkeypatch,
        {
            ClaimLabels: lambda _: ClaimLabels(
                labels=[ClaimLabel(number=1, kind="method")]
            )
        },
    )
    cache: dict = {}
    for _ in range(2):
        await analyse_section(
            parse_body("We enrolled children."),
            facts=FACTS,
            infos=INFOS,
            catalogue=None,
            taken_anchors=set(),
            model="stub:judge",
            cache=cache,
            log=None,
        )
    assert script.count(ClaimLabels) == 1


async def test_numbers_bind_to_the_file_the_sentence_cites() -> None:
    analysis = await analyse("Scores rose from 41.2 to 53.8 [[f2]].")
    (sentence,) = analysis.sentences
    keys = [n.binding.key for n in sentence.numbers]
    assert keys == [
        "scores.csv row 2 column 'baseline'",
        "scores.csv row 2 column 'followup'",
    ]
    first = sentence.numbers[0]
    assert sentence.text[first.start : first.end] == "41.2"
    assert first.binding.scale == 1.0
    assert first.binding.anchor_id.startswith("n-")


async def test_a_number_not_in_the_cited_file_is_left_alone() -> None:
    analysis = await analyse("Scores rose by 12 points [[f2]].")
    assert analysis.number_spots() == []


async def test_a_number_in_a_sentence_without_result_facts_is_not_bound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_script(
        monkeypatch,
        {
            ClaimLabels: lambda _: ClaimLabels(
                labels=[ClaimLabel(number=1, kind="result")]
            )
        },
    )
    analysis = await analyse("Scores rose from 41.2 to 53.8 [[f1]].")
    assert analysis.number_spots() == []


async def test_percent_binds_a_stored_fraction() -> None:
    catalogue = Catalogue(
        entries=[
            ResultEntry(
                file="scores.csv", key="scores.csv row 2 column 'rate'", value=0.538
            )
        ],
        digest="d",
    )
    analysis = await analyse("The rate was 53.8% [[f2]].", catalogue=catalogue)
    (spot,) = analysis.number_spots()
    assert spot.binding.scale == 100.0
    assert spot.binding.written == "53.8%"


async def test_an_ambiguous_number_is_reported_not_guessed() -> None:
    catalogue = Catalogue(
        entries=[
            ResultEntry(
                file="scores.csv", key="scores.csv row 2 column 'a'", value=7.0
            ),
            ResultEntry(
                file="scores.csv", key="scores.csv row 3 column 'b'", value=7.0
            ),
        ],
        digest="d",
    )
    analysis = await analyse("There were 7 events [[f2]].", catalogue=catalogue)
    assert analysis.number_spots() == []
    assert len(analysis.ambiguous_numbers) == 1
    assert "not bound" in analysis.ambiguous_numbers[0]


async def test_a_number_inside_a_placeholder_is_not_bound() -> None:
    analysis = await analyse("It rose [[f2]]. [TO FILL: was it 53.8 or more?]")
    assert analysis.number_spots() == []


async def test_anchor_ids_are_unique_for_a_repeated_number() -> None:
    analysis = await analyse("From 53.8 and again 53.8 [[f2]].")
    ids = [n.binding.anchor_id for n in analysis.number_spots()]
    assert len(ids) == 2 and len(set(ids)) == 2


async def test_ledger_entries_carry_kind_evidence_and_status() -> None:
    analysis = await analyse("Scores rose from 41.2 to 53.8 [[f2]].")
    bound = {n.binding.anchor_id for n in analysis.number_spots()}
    (entry,) = claim_entries("Results", analysis, facts=FACTS, infos=INFOS, bound=bound)
    assert entry["kind"] == "result"
    assert entry["status"] == "supported"
    assert entry["approved_by"] is None
    assert entry["strength"] == "quoted source, number bound to result file"
    assert sorted(entry["anchors"]) == sorted(bound)  # type: ignore[arg-type]
    (pointer,) = entry["evidence"]  # type: ignore[misc]
    assert pointer["file"] == "scores.csv" and pointer["fact_id"] == "f2"


async def test_a_refused_number_is_not_listed_as_bound() -> None:
    analysis = await analyse("Scores rose from 41.2 to 53.8 [[f2]].")
    (entry,) = claim_entries("Results", analysis, facts=FACTS, infos=INFOS, bound=set())
    assert entry["anchors"] == []
    assert entry["strength"] == "quoted source"


async def test_interpretations_need_the_author(monkeypatch: pytest.MonkeyPatch) -> None:
    install_script(
        monkeypatch,
        {
            ClaimLabels: lambda _: ClaimLabels(
                labels=[ClaimLabel(number=1, kind="interpretation")]
            )
        },
    )
    analysis = await analyse("This suggests a mechanism.")
    (entry,) = claim_entries(
        "Discussion", analysis, facts=FACTS, infos=INFOS, bound=set()
    )
    assert entry["status"] == "needs-author"


def test_record_claims_appends_and_keeps_foreign_entries(tmp_path: Path) -> None:
    store = ProjectStore(tmp_path / "p.harness")
    store.path.mkdir()
    ledger = store.path / "claims.yaml"
    foreign = {
        "id": "n-abc",
        "text": "t",
        "kind": "result",
        "binding": {"anchor_id": "n-abc"},
    }
    ledger.write_text(
        yaml.safe_dump({"claims": [foreign, {"id": "d4", "text": "old"}]})
    )
    written = record_claims(store, [{"text": "one"}, {"text": "two"}])
    assert written == 2
    claims = yaml.safe_load(ledger.read_text())["claims"]
    assert [c["id"] for c in claims] == ["n-abc", "d4", "d5", "d6"]
    assert claims[0] == foreign


def test_record_claims_creates_the_file(tmp_path: Path) -> None:
    store = ProjectStore(tmp_path / "p.harness")
    record_claims(store, [{"text": "one"}])
    claims = yaml.safe_load((store.path / "claims.yaml").read_text())["claims"]
    assert claims == [{"id": "d1", "text": "one"}]


def test_a_malformed_ledger_raises(tmp_path: Path) -> None:
    store = ProjectStore(tmp_path / "p.harness")
    store.path.mkdir()
    (store.path / "claims.yaml").write_text("claims: nope\n")
    with pytest.raises(ValueError, match="claims"):
        record_claims(store, [{"text": "one"}])


def test_a_comma_or_full_stop_after_a_number_is_not_part_of_it():
    """ "by week 4, the mean" gave the token "4," and bound it with its comma."""
    from andamentum.compose.harness.draft.claims import NUMBER_TOKEN

    found = [
        m.group(0)
        for m in NUMBER_TOKEN.finditer("By week 4, 1,234 of 52. Then .05, 3.10.")
    ]
    assert found == ["4", "1,234", "52", ".05", "3.10"]
