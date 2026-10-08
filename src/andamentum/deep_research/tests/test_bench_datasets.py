"""The data side of the bench, on invented data: deterministic subsets, the
download manifest and the SimpleQA loader and grader."""

from __future__ import annotations

from pathlib import Path

import pytest

from andamentum.deep_research.bench import simpleqa
from andamentum.deep_research.bench.download import (
    DatasetCard,
    DatasetMissing,
    read_manifest,
    save_file,
)
from andamentum.deep_research.bench.simpleqa import (
    AnswerGrader,
    GradeOutput,
    grade_message,
    load_simpleqa,
    read_simpleqa,
)
from andamentum.deep_research.bench.subsets import select_subset

from .test_bench_experiment_run import GOLD_BY_QUESTION, N_ROWS, write_simpleqa

# -- subsets -----------------------------------------------------------------------


def test_a_subset_does_not_depend_on_input_order_and_extends_a_smaller_one() -> None:
    ids = [f"Q{i}" for i in range(50)]
    forward = select_subset(ids, 10, seed=0)
    assert forward == select_subset(list(reversed(ids)), 10, seed=0)
    assert select_subset(ids, 5, seed=0) == forward[:5]
    assert select_subset(ids, 10, seed=1) != forward
    assert len(select_subset(ids, 500, seed=0)) == 50


def test_a_subset_refuses_duplicates_and_a_negative_size() -> None:
    with pytest.raises(ValueError, match="unique"):
        select_subset(["a", "a"], 1, seed=0)
    with pytest.raises(ValueError, match="negative"):
        select_subset(["a"], -1, seed=0)


# -- the manifest ------------------------------------------------------------------------


def card(key: str, licence: str) -> DatasetCard:
    return DatasetCard(
        key=key,
        title=f"An invented dataset {key}",
        source=f"https://example.org/{key}",
        licence=licence,
        licence_evidence="invented for the test",
        terms="none",
        citation="Doe, J. (2026). Invented.",
    )


def test_save_file_records_checksum_licence_and_replaces_by_path(
    tmp_path: Path,
) -> None:
    first, second = card("apollo", "CC0-1.0"), card("saltlog", "MIT")
    save_file(tmp_path, "a/x.txt", b"one", url="https://e.org/x", card=first)
    save_file(tmp_path, "a/x.txt", b"two!", url="https://e.org/x", card=first)
    save_file(tmp_path, "b/y.txt", b"y", url="https://e.org/y", card=second)
    manifest = read_manifest(tmp_path)
    assert (tmp_path / "a" / "x.txt").read_bytes() == b"two!"
    assert [f.path for f in manifest.files] == ["a/x.txt", "b/y.txt"]
    assert manifest.files[0].bytes == 4
    assert {c.key: c.licence for c in manifest.cards} == {
        "apollo": "CC0-1.0",
        "saltlog": "MIT",
    }
    assert not list(tmp_path.rglob("*.part"))


# -- SimpleQA ----------------------------------------------------------------------------


def test_simpleqa_items_are_numbered_by_row_and_read_from_the_file_or_folder(
    tmp_path: Path,
) -> None:
    root = write_simpleqa(tmp_path)
    from_folder = read_simpleqa(root)
    from_file = read_simpleqa(root / "simple_qa_test_set.csv")
    assert from_folder == from_file and len(from_folder) == N_ROWS
    assert from_folder[0].item_id == "SQ0001"
    assert GOLD_BY_QUESTION[from_folder[0].question] == from_folder[0].answer


def test_a_missing_simpleqa_file_names_what_to_download(tmp_path: Path) -> None:
    with pytest.raises(DatasetMissing, match="simple_qa_test_set.csv"):
        read_simpleqa(tmp_path)


def test_simpleqa_positions_are_the_seeded_order_after_the_skipped_ones(
    tmp_path: Path,
) -> None:
    root = write_simpleqa(tmp_path)
    ids = [f"SQ{i:04d}" for i in range(1, N_ROWS + 1)]
    order = select_subset(ids, N_ROWS, seed=1)
    first = [i.item_id for i in load_simpleqa(root, n=4, seed=1)]
    later = [i.item_id for i in load_simpleqa(root, n=4, seed=1, skip=4)]
    assert first == order[:4] and later == order[4:8]
    assert not set(first) & set(later)


def test_the_grading_message_shows_the_question_the_gold_and_a_bounded_answer() -> None:
    message = grade_message(question="Q?", gold="teal1", answer="  ")
    assert "Question: Q?" in message and "Gold answer: teal1" in message
    assert "(no answer)" in message
    long = grade_message(question="Q?", gold="g", answer="x" * 20_000)
    assert long.endswith("[... cut]") and len(long) < 13_000


async def test_the_grader_asks_its_model_once_and_returns_the_grade(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[dict[str, object]] = []

    async def fake(model: object, **kwargs: object) -> GradeOutput:
        seen.append({"model": model, **kwargs})
        return GradeOutput(grade="correct")

    monkeypatch.setattr(simpleqa, "run_agent_with_fallback", fake)
    grade = await AnswerGrader(model="ollama:invented-grader").grade(
        question="Q?", gold="teal1", answer="It is teal1."
    )
    assert grade == "correct" and len(seen) == 1
    assert seen[0]["model"] == "ollama:invented-grader"
    assert seen[0]["output_type"] is GradeOutput
