"""refresh: bound numbers recomputed from the result files as tracked changes."""

from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path

import pytest

from andamentum.compose.harness.bindings import bind as bind_module
from andamentum.compose.harness.bindings.bind import BindParams, run_bind
from andamentum.compose.harness.bindings.ledger import Ledger
from andamentum.compose.harness.bindings.refresh import (
    AUTHOR,
    RefreshParams,
    run_refresh,
)
from andamentum.compose.harness.bindings.common import UsageError
from andamentum.compose.harness.docx.reader import open_docx
from andamentum.compose.harness.kernel import ProjectStore

from ._helpers import copy_of
from ._bindings_helpers import (
    EFFECT,
    INTERVAL,
    PVALUE,
    SAMPLE,
    make_manuscript,
    make_results,
)
from .test_bindings_bind import StubModel

PICKS = {
    EFFECT: [("23.4%", "effect")],
    SAMPLE: [("1,234", "meals"), ("159", "participants")],
    PVALUE: [(".03", "t.p ")],
    INTERVAL: [("1.40", "odds_ratio"), ("1.10", "ci_low"), ("1.80", "ci_high")],
}


@pytest.fixture
def bound(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """A manuscript with every number bound; returns (bound docx, results dir, project)."""
    monkeypatch.setattr(bind_module, "run_agent_with_usage", StubModel(PICKS))
    manuscript = make_manuscript(tmp_path / "paper.docx")
    results = make_results(tmp_path / "results")
    import asyncio

    run = asyncio.run(
        run_bind(BindParams(out=copy_of(manuscript, "bound"), file=manuscript, results=results, model="stub:m"))
    )
    assert len(run.report.bound) == 7, run.report.unmatched
    return run.output, results, Path(run.report.project_dir)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def refresh(bound, **kw):
    docx, results, project = bound
    kw.setdefault("out", copy_of(docx, "refreshed"))
    return run_refresh(RefreshParams(file=docx, results=results, project=project, **kw))


def ours(path: Path):
    return [c for c in open_docx(path).comments if c.author == AUTHOR]


def test_nothing_changed_nothing_written(bound):
    run = refresh(bound)
    report = run.report
    assert report.changes == [] and report.unchanged == 7 and report.errors == 0
    assert ours(run.output) == []
    assert report.exit_code == 0


def test_a_number_with_a_comment_opening_inside_its_control_is_still_replaced(
    bound, tmp_path
):
    """A check comment on the bare number opens its range inside the control."""
    from andamentum.compose.harness.docx.writer import DocxAdapter
    from andamentum.compose.harness.kernel import Transaction
    from andamentum.compose.harness.kernel.ops import CommentOp

    docx, results, project = bound
    adapter = DocxAdapter()
    view = adapter.open(docx)
    texts = {p.id: p.text for p in view.paragraphs}
    anchor = next(
        a for a in view.anchors if texts[a.paragraph_id][a.start : a.end] == "23.4%"
    )
    commented = tmp_path / "commented.docx"
    adapter.apply(
        view,
        Transaction(
            ops=[
                CommentOp(
                    paragraph_id=anchor.paragraph_id,
                    start=anchor.start,
                    end=anchor.end,
                    text="check this",
                )
            ],
            author="a reviewer",
            reason="test",
            source="test",
        ),
        output=commented,
    )
    make_results(results, effect="0.2511")
    run = run_refresh(RefreshParams(out=copy_of(commented, "refreshed"), file=commented, results=results, project=project))
    (change,) = run.report.changes
    assert change.applied
    copy = open_docx(run.output)
    moved = next(a for a in copy.anchors if a.id == change.anchor_id)
    paragraph = next(p for p in copy.paragraphs if p.id == moved.paragraph_id)
    assert paragraph.text[moved.start : moved.end] == "25.1%"


def test_a_changed_value_is_a_tracked_replacement_and_a_comment(bound):
    docx, results, project = bound
    make_results(results, effect="0.2511")  # 25.1%
    before = sha(docx)
    run = refresh(bound)
    assert sha(docx) == before  # the input is never modified

    (change,) = run.report.changes
    assert (change.old, change.new, change.applied) == ("23.4%", "25.1%", True)
    copy = open_docx(run.output)
    effect_paragraph = next(p for p in copy.paragraphs if "sodium intake" in p.text)
    assert "25.1%" in effect_paragraph.text and "23.4%" not in effect_paragraph.text
    # the control survives the replacement and still wraps the new number
    anchor = next(a for a in copy.anchors if a.id == change.anchor_id)
    assert effect_paragraph.text[anchor.start : anchor.end] == "25.1%"
    # tracked: the old text is a deletion, the new one an insertion
    with zipfile.ZipFile(run.output) as archive:
        xml = archive.read("word/document.xml").decode("utf-8")
    assert "<w:del " in xml and "<w:ins " in xml and AUTHOR in xml

    (comment,) = ours(run.output)
    assert "23.4% → 25.1%" in comment.text
    assert "check the claim still holds" in comment.text
    assert comment.text.startswith("[warning]")
    sentence = effect_paragraph.text[comment.start : comment.end]
    assert sentence.startswith("In the SaltLog pilot") and "25.1%" in sentence
    assert run.report.exit_code == 0

    # the ledger now holds the new text and value; a second refresh of the
    # refreshed copy changes nothing
    binding = Ledger(ProjectStore(project)).bindings()[change.anchor_id]
    assert (binding.written, binding.value) == ("25.1%", 0.2511)
    again = run_refresh(
        RefreshParams(
            file=run.output,
            results=results,
            project=project,
            out=run.output.with_name("again.docx"),
        )
    )
    assert again.report.changes == []


def test_precision_units_and_separators_are_kept(bound):
    docx, results, _ = bound
    (results / "summary.csv").write_text(
        "measure,value\neffect,0.234\nmeals,12345\nparticipants,159.4\n", "utf-8"
    )
    run = refresh(bound)
    new = {c.old: c.new for c in run.report.changes}
    assert new == {"1,234": "12,345"}  # 159.4 still rounds to 159
    text = " ".join(p.text for p in open_docx(run.output).paragraphs)
    assert "12,345 meals" in text and "159 participants" in text


def test_a_value_that_rounds_the_same_is_left_alone(bound):
    _, results, _ = bound
    make_results(results, effect="0.2338")  # still 23.4%
    run = refresh(bound)
    assert run.report.changes == [] and ours(run.output) == []


def test_a_p_value_crossing_alpha_is_an_error(bound):
    _, results, _ = bound
    make_results(results, p="0.071")
    run = refresh(bound)
    (change,) = run.report.changes
    assert (change.old, change.new) == (".03", ".07")  # leading zero stays off
    (flip,) = run.report.flips
    assert flip.kind == "p" and "0.05" in flip.message
    assert run.report.exit_code == 1
    (comment,) = ours(run.output)
    assert comment.text.startswith("[error]")
    assert "Significance flip" in comment.text and ".03 → .07" in comment.text


def test_a_p_value_staying_significant_is_only_a_warning(bound):
    _, results, _ = bound
    make_results(results, p="0.04")
    run = refresh(bound)
    assert run.report.flips == [] and run.report.exit_code == 0
    assert ours(run.output)[0].text.startswith("[warning]")


def test_a_confidence_interval_crossing_the_null_is_an_error(bound):
    docx, results, _ = bound
    (results / "model.json").write_text(
        '{"t": {"p": 0.03, "odds_ratio": 1.4, "ci_low": 0.9, "ci_high": 1.8}}', "utf-8"
    )
    run = refresh(bound)
    (flip,) = run.report.flips
    assert flip.kind == "ci" and "now contains 1" in flip.message
    (comment,) = ours(run.output)
    assert comment.text.startswith("[error]") and "1.10 → 0.90" in comment.text


def test_two_changed_numbers_in_one_sentence_share_a_comment(bound):
    _, results, _ = bound
    (results / "summary.csv").write_text(
        "measure,value\neffect,0.234\nmeals,1300\nparticipants,170\n", "utf-8"
    )
    run = refresh(bound)
    assert len(run.report.changes) == 2
    (comment,) = ours(run.output)
    assert "values changed" in comment.text
    assert "1,234 → 1,300" in comment.text and "159 → 170" in comment.text


def test_a_key_gone_from_the_results_is_reported_and_not_replaced(bound):
    docx, results, _ = bound
    (results / "summary.csv").write_text("measure,value\neffect,0.5\n", "utf-8")
    run = refresh(bound)
    errors = [p for p in run.report.problems if p.severity == "error"]
    assert len(errors) == 2 and all(
        "not in the result files" in p.message for p in errors
    )
    assert run.report.exit_code == 1
    text = " ".join(p.text for p in open_docx(run.output).paragraphs)
    assert "1,234 meals" in text and "159 participants" in text  # untouched


def test_a_reordered_csv_is_caught_by_the_row_label(bound):
    _, results, _ = bound
    (results / "summary.csv").write_text(
        "measure,value\nparticipants,159\neffect,0.234\nmeals,1234\n", "utf-8"
    )
    run = refresh(bound)
    problems = [p for p in run.report.problems if "reordered" in p.message]
    assert problems and run.report.exit_code == 1
    assert run.report.changes == []


def test_a_hand_edited_number_is_replaced_and_the_comment_says_so(bound, tmp_path):
    from andamentum.compose.harness.docx.writer import DocxAdapter
    from andamentum.compose.harness.kernel import Replace, Transaction

    docx, results, project = bound
    adapter = DocxAdapter()
    view = adapter.open(docx)
    anchor = next(
        a
        for a in view.anchors
        if next(p for p in view.paragraphs if p.id == a.paragraph_id).text[
            a.start : a.end
        ]
        == "159"
    )
    edited = tmp_path / "edited.docx"
    adapter.apply(
        view,
        Transaction(
            ops=[
                Replace(
                    paragraph_id=anchor.paragraph_id,
                    start=anchor.start,
                    end=anchor.end,
                    text="169",
                    expect="159",
                )
            ],
            author="Collaborator",
            reason="hand edit",
            source="test",
        ),
        output=edited,
    )
    run = run_refresh(
        RefreshParams(
            file=edited, results=results, project=project, out=tmp_path / "out.docx"
        )
    )
    (change,) = run.report.changes
    assert (change.old, change.new, change.edited_by_hand) == ("169", "159", True)
    (comment,) = ours(run.output)
    assert "169 → 159" in comment.text and "edited since it was bound" in comment.text


def test_a_deleted_number_and_a_lost_control_are_warnings(bound, tmp_path):
    from andamentum.compose.harness.docx.writer import DocxAdapter
    from andamentum.compose.harness.kernel import Delete, Transaction

    docx, results, project = bound
    adapter = DocxAdapter()
    view = adapter.open(docx)
    paragraph = next(p for p in view.paragraphs if "SaltLog pilot the mean" in p.text)
    start = paragraph.text.index("In the SaltLog pilot")
    removed = tmp_path / "removed.docx"
    adapter.apply(
        view,
        Transaction(
            ops=[
                Delete(
                    paragraph_id=paragraph.id,
                    start=start,
                    end=len(paragraph.text),
                    expect=paragraph.text[start:],
                )
            ],
            author="Collaborator",
            reason="cut",
            source="test",
        ),
        output=removed,
    )
    make_results(results, effect="0.3")
    run = run_refresh(
        RefreshParams(
            file=removed, results=results, project=project, out=tmp_path / "o.docx"
        )
    )
    (deleted,) = [p for p in run.report.problems if "was deleted" in p.message]
    assert deleted.severity == "warning" and run.report.changes == []

    # the original manuscript has no controls at all: every binding is reported
    # lost and nothing is touched
    plain = make_manuscript(tmp_path / "plain.docx")
    lost = run_refresh(
        RefreshParams(
            file=plain, results=results, project=project, out=tmp_path / "p.docx"
        )
    )
    assert len(lost.report.problems) == 7
    assert all("no longer in the document" in p.message for p in lost.report.problems)
    assert lost.report.changes == [] and lost.report.exit_code == 0


def test_numbers_that_were_never_bound_are_untouched(bound, tmp_path):
    docx, results, project = bound
    make_results(results, effect="0.3")
    # an extra control the ledger knows nothing about
    from andamentum.compose.harness.docx.writer import DocxAdapter
    from andamentum.compose.harness.kernel import Bind, Transaction

    adapter = DocxAdapter()
    view = adapter.open(docx)
    paragraph = next(p for p in view.paragraphs if "twelve weeks" in p.text)
    start = paragraph.text.index("twelve")
    extra = tmp_path / "extra.docx"
    adapter.apply(
        view,
        Transaction(
            ops=[
                Bind(
                    paragraph_id=paragraph.id,
                    start=start,
                    end=start + 6,
                    anchor_id="by-hand",
                    kind="number",
                )
            ],
            author="Collaborator",
            reason="bind",
            source="test",
        ),
        output=extra,
    )
    run = run_refresh(
        RefreshParams(
            file=extra, results=results, project=project, out=tmp_path / "e.docx"
        )
    )
    assert run.report.unbound_anchors == ["by-hand"]
    assert [c.new for c in run.report.changes] == ["30.0%"]


def test_usage_errors(bound, tmp_path):
    docx, results, project = bound
    with pytest.raises(UsageError, match="never edited"):
        run_refresh(RefreshParams(file=docx, results=results, out=docx))
    with pytest.raises(UsageError, match="--results"):
        run_refresh(RefreshParams(out=copy_of(docx, "refreshed"), file=docx, results=tmp_path / "nope"))
    with pytest.raises(UsageError, match=".docx"):
        run_refresh(RefreshParams(out=copy_of(tmp_path / "x.txt", "refreshed"), file=tmp_path / "x.txt", results=results))


def test_the_verb_exit_codes(bound, capsys, tmp_path):
    from andamentum.word import cli

    docx, results, project = bound
    folder = docx.parent  # the folder bind-numbers wrote into, with its state/
    args = ["refresh", str(docx), "--results", str(results), "--out", str(folder)]
    assert cli.main(args) == 0
    make_results(results, p="0.9")
    assert cli.main(args) == 1
    missing = ["refresh", str(docx), "--results", str(results / "x"), "--out", str(folder)]
    assert cli.main(missing) == 3
    assert "error:" in capsys.readouterr().err


def test_a_second_run_into_the_same_folder_finds_the_ledger(bound):
    docx, results, project = bound
    make_results(results, effect="0.3")
    run = run_refresh(RefreshParams(out=copy_of(docx, "refreshed"), file=docx, results=results))  # no project: state/ beside the output
    assert Path(run.report.project_dir) == project
    assert len(run.report.changes) == 1


def test_a_binding_without_a_row_label_is_not_checked_for_reordering(bound):
    """Bindings made by the draft (no context recorded) still refresh."""
    docx, results, project = bound
    ledger = Ledger(ProjectStore(project))
    ledger.record(
        [b.model_copy(update={"context": ""}) for b in ledger.bindings().values()]
    )
    make_results(results, effect="0.3")
    run = refresh(bound)
    assert [c.new for c in run.report.changes] == ["30.0%"]
    assert run.report.problems == []
