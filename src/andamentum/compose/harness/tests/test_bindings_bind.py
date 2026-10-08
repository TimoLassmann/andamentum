"""bind-numbers: a stubbed model proposes, code verifies, the copy carries the controls."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import pytest
from pydantic_ai.usage import RunUsage

from andamentum.word import cli
from andamentum.compose.harness.bindings import bind as bind_module
from andamentum.compose.harness.bindings.bind import (
    BindParams,
    Proposal,
    Proposals,
    run_bind,
)
from andamentum.compose.harness.bindings.common import UsageError
from andamentum.compose.harness.bindings.ledger import Ledger
from andamentum.compose.harness.docx.reader import open_docx
from andamentum.compose.harness.kernel import ProjectStore

from ._helpers import copy_of
from ._bindings_helpers import (
    EFFECT,
    SAMPLE,
    make_manuscript,
    make_results,
)


class StubModel:
    """Chooses candidates by the key text the test names; records every call."""

    def __init__(self, picks: dict[str, list[tuple[str, str]]]) -> None:
        # sentence -> [(value as written, substring of the candidate line or "")]
        self.picks = picks
        self.calls: list[str] = []

    async def __call__(self, model, *, user_message, output_type, **_kw):
        self.calls.append(user_message)
        paragraph, _, listed = user_message.partition("\n\nCandidates:\n")
        lines = dict(re.findall(r"^(c\d+): (.*)$", listed, flags=re.MULTILINE))
        proposals = []
        for sentence, wanted in self.picks.items():
            if sentence not in paragraph:
                continue
            for value, needle in wanted:
                chosen = next(
                    (cid for cid, text in lines.items() if needle and needle in text),
                    "",
                )
                proposals.append(
                    Proposal(sentence=sentence, value=value, candidate=chosen)
                )
        return Proposals(proposals=proposals), RunUsage(input_tokens=7, output_tokens=3)


@pytest.fixture
def manuscript(tmp_path: Path) -> Path:
    return make_manuscript(tmp_path / "paper.docx")


@pytest.fixture
def results(tmp_path: Path) -> Path:
    return make_results(tmp_path / "results")


def install(monkeypatch: pytest.MonkeyPatch, stub: StubModel) -> StubModel:
    monkeypatch.setattr(bind_module, "run_agent_with_usage", stub)
    return stub


PICKS = {
    EFFECT: [("23.4%", "effect")],
    SAMPLE: [("1,234", "meals"), ("159", "participants"), ("twelve", "")],
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


async def test_binds_verified_numbers_and_writes_ledger(
    monkeypatch, manuscript: Path, results: Path
):
    install(monkeypatch, StubModel(PICKS))
    before = sha(manuscript)
    run = await run_bind(BindParams(out=copy_of(manuscript, "bound"), file=manuscript, results=results, model="stub:m"))
    assert sha(manuscript) == before  # the input is never modified
    assert run.output == manuscript.with_name("paper.bound.docx")

    copy = open_docx(run.output)
    bound = {a.id: a for a in copy.anchors}
    assert len(bound) == 3 and {a.kind for a in bound.values()} == {"number"}
    texts = {p.id: p.text for p in copy.paragraphs}
    shown = sorted(texts[a.paragraph_id][a.start : a.end] for a in bound.values())
    assert shown == ["1,234", "159", "23.4%"]
    # the document text is unchanged by binding
    assert [p.text for p in copy.paragraphs] == [
        p.text for p in open_docx(manuscript).paragraphs
    ]
    assert copy.project_id == run.report.project_id

    ledger = Ledger(ProjectStore(Path(run.report.project_dir))).bindings()
    assert set(ledger) == set(bound)
    effect = next(b for b in ledger.values() if b.written == "23.4%")
    assert (effect.file, effect.value, effect.scale) == ("summary.csv", 0.234, 100.0)
    assert (effect.key, effect.context) == (
        "summary.csv row 2 column 'value'",
        "effect",
    )
    assert effect.sentence == EFFECT

    report = run.report
    assert [i.written for i in report.bound if i.paragraph == 1] == ["23.4%"]
    # 'twelve' was proposed with no candidate: listed with the reason
    proposed = [u for u in report.unmatched if "not proposed" not in u.reason]
    assert [(u.written, u.reason) for u in proposed] == [
        ("twelve", "no result number is the same quantity")
    ]
    # numbers the model never named are listed by code, bound ones are not
    left_over = {u.written for u in report.unmatched if "not proposed" in u.reason}
    assert {"157", "3.10", ".03", "1.40", "1.10", "1.80"} <= left_over
    assert not left_over & {"23.4%", "1,234", "159"}
    assert report.model_calls == 4 and report.input_tokens == 28
    assert run.report_md.is_file() and run.report_json.is_file()
    assert "Bound" in run.report_md.read_text("utf-8")


async def test_a_value_that_does_not_match_is_not_bound(
    monkeypatch, manuscript: Path, tmp_path: Path
):
    # The file says 0.251; the text says 23.4%: the model picks it, code refuses.
    results = make_results(tmp_path / "other", effect="0.251")
    install(monkeypatch, StubModel({EFFECT: [("23.4%", "effect")]}))
    run = await run_bind(BindParams(out=copy_of(manuscript, "bound"), file=manuscript, results=results, model="stub:m"))
    assert run.report.bound == []
    assert open_docx(run.output).anchors == []
    (item,) = [u for u in run.report.unmatched if u.written == "23.4%"]
    # no candidate was close enough at the written precision, so the model had none
    assert "same quantity" in item.reason


async def test_a_candidate_the_model_made_up_is_refused(
    monkeypatch, manuscript: Path, results: Path
):
    class Inventor:
        async def __call__(self, model, *, user_message, **_kw):
            if EFFECT not in user_message:
                return Proposals(proposals=[]), RunUsage()
            return (
                Proposals(
                    proposals=[
                        Proposal(sentence=EFFECT, value="23.4%", candidate="c99")
                    ]
                ),
                RunUsage(),
            )

    monkeypatch.setattr(bind_module, "run_agent_with_usage", Inventor())
    run = await run_bind(BindParams(out=copy_of(manuscript, "bound"), file=manuscript, results=results, model="stub:m"))
    assert run.report.bound == []
    assert "not one of the candidates" in run.report.unmatched[0].reason


async def test_a_value_the_text_does_not_hold_is_ignored_and_counted(
    monkeypatch, manuscript: Path, results: Path
):
    install(monkeypatch, StubModel({EFFECT: [("77%", "effect")]}))
    run = await run_bind(BindParams(out=copy_of(manuscript, "bound"), file=manuscript, results=results, model="stub:m"))
    assert run.report.bound == []
    # the ignored proposal is not a row of its own; the number is still listed
    assert [u.reason for u in run.report.unmatched if u.written == "23.4%"] == [
        "not proposed by the model; not bound"
    ]
    assert any("did not occur in the text" in n for n in run.report.notes)


async def test_binding_twice_binds_nothing_new(
    monkeypatch, manuscript: Path, results: Path, tmp_path: Path
):
    install(monkeypatch, StubModel(PICKS))
    first = await run_bind(BindParams(out=copy_of(manuscript, "bound"), file=manuscript, results=results, model="stub:m"))
    second = await run_bind(
        BindParams(
            file=first.output,
            results=results,
            model="stub:m",
            out=tmp_path / "again.docx",
            project=Path(first.report.project_dir),
        )
    )
    assert second.report.bound == []
    reasons = {u.written: u.reason for u in second.report.unmatched}
    assert reasons["23.4%"] == reasons["1,234"] == reasons["159"] == "already bound"
    assert len(open_docx(second.output).anchors) == 3
    assert len(Ledger(ProjectStore(Path(first.report.project_dir))).bindings()) == 3


async def test_other_ledger_entries_survive(
    monkeypatch, manuscript: Path, results: Path
):
    store_dir = manuscript.with_name("state")
    store_dir.mkdir()
    (store_dir / "claims.yaml").write_text(
        "claims:\n- id: c-other\n  text: kept\n  kind: method\n", "utf-8"
    )
    install(monkeypatch, StubModel(PICKS))
    await run_bind(BindParams(out=copy_of(manuscript, "bound"), file=manuscript, results=results, model="stub:m"))
    text = (store_dir / "claims.yaml").read_text("utf-8")
    assert "c-other" in text and "kind: result" in text


async def test_needs_a_model_and_a_results_folder(manuscript: Path, results: Path):
    with pytest.raises(UsageError, match="--model"):
        await run_bind(BindParams(out=copy_of(manuscript, "bound"), file=manuscript, results=results))
    with pytest.raises(UsageError, match="--results"):
        await run_bind(
            BindParams(out=copy_of(manuscript, "bound"), file=manuscript, results=results / "nope", model="stub:m")
        )
    with pytest.raises(UsageError, match="never edited"):
        await run_bind(
            BindParams(file=manuscript, results=results, model="m", out=manuscript)
        )


async def test_answers_are_cached_in_the_project(
    monkeypatch, manuscript: Path, results: Path, tmp_path: Path
):
    stub = install(monkeypatch, StubModel(PICKS))
    project = tmp_path / "proj"
    await run_bind(
        BindParams(out=copy_of(manuscript, "bound"), file=manuscript, results=results, model="m", project=project)
    )
    calls = len(stub.calls)
    await run_bind(
        BindParams(
            file=manuscript,
            results=results,
            model="m",
            project=project,
            out=tmp_path / "second.docx",
        )
    )
    assert len(stub.calls) == calls  # every paragraph answered from the cache


def test_the_verbs_are_registered_and_in_the_cli(tmp_path: Path, capsys):
    from andamentum.compose.harness.kernel.verbs import all_verbs, load_builtin_verbs

    load_builtin_verbs()
    names = {v.name for v in all_verbs()}
    assert {"bind-numbers", "refresh"} <= names
    with pytest.raises(SystemExit) as exit_info:
        cli.main(["bind-numbers", "--help"])
    assert exit_info.value.code == 0
    assert "bind-numbers collaborator.docx --results" in " ".join(
        capsys.readouterr().out.split()
    )


async def test_a_number_the_model_leaves_out_is_still_listed(
    monkeypatch, manuscript: Path, results: Path
):
    # The model proposes nothing for the sample paragraph; code lists its numbers.
    install(monkeypatch, StubModel({EFFECT: [("23.4%", "effect")]}))
    run = await run_bind(BindParams(out=copy_of(manuscript, "bound"), file=manuscript, results=results, model="stub:m"))
    listed = {(u.written, u.reason) for u in run.report.unmatched if u.paragraph == 2}
    assert listed == {
        ("1,234", "not proposed by the model; not bound"),
        ("159", "not proposed by the model; not bound"),
    }
    assert "1,234" in run.report_md.read_text("utf-8")


def test_table_figure_and_citation_numbers_are_not_listed():
    from andamentum.compose.harness.bindings.bind import _REFERENCE

    text = "As Table 3 and Figures 2-4 show [12, 14], 52 people stayed."
    assert [m.group(0) for m in _REFERENCE.finditer(text)] == [
        "Table 3",
        "Figures 2-4",
        "[12, 14]",
    ]


UNIT_SENTENCES = [
    "The pilot logged a median of 412 mg of sodium per meal.",
    "Total intake reached 3,310 mg in the control arm.",
    "Adherence was 87.3% across the pilot weeks.",
    "The effect was significant (p = .003) in the pilot.",
    "The change in systolic pressure was -4.2 mmHg for the arm.",
]
UNIT_PICKS = {
    UNIT_SENTENCES[0]: [("412 mg", "median_sodium")],
    UNIT_SENTENCES[1]: [("3,310 mg", "control_total")],
    UNIT_SENTENCES[2]: [("87.3%", "adherence")],
    UNIT_SENTENCES[3]: [("p = .003", "p_value")],
    UNIT_SENTENCES[4]: [("-4.2 mmHg", "systolic_change")],
}


async def test_a_number_with_a_unit_or_label_binds_only_the_number(
    monkeypatch, tmp_path: Path
):
    manuscript = make_manuscript(
        tmp_path / "units.docx", [(s, None) for s in UNIT_SENTENCES]
    )
    folder = tmp_path / "res"
    folder.mkdir()
    (folder / "summary.csv").write_text(
        "measure,value\nmedian_sodium,412\ncontrol_total,3310\nadherence,0.873\n"
        "p_value,0.003\nsystolic_change,-4.2\n",
        "utf-8",
    )
    install(monkeypatch, StubModel(UNIT_PICKS))
    run = await run_bind(BindParams(out=copy_of(manuscript, "bound"), file=manuscript, results=folder, model="stub:m"))

    copy = open_docx(run.output)
    texts = {p.id: p.text for p in copy.paragraphs}
    shown = sorted(texts[a.paragraph_id][a.start : a.end] for a in copy.anchors)
    assert shown == ["-4.2", ".003", "3,310", "412", "87.3%"]
    assert [i.written for i in run.report.bound].count("412") == 1
    rejected = [u for u in run.report.unmatched if "not proposed" not in u.reason]
    assert rejected == []
    ledger = Ledger(ProjectStore(Path(run.report.project_dir))).bindings()
    assert sorted(b.written for b in ledger.values()) == sorted(shown)
