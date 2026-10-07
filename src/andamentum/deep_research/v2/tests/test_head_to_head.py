"""The SimpleQA set, the live LitSearch set and the item-by-item cell order.

No model, no network: the CSV is invented, the grader is a stand-in that reads
the answer for the gold text, and the arms answer from a script.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from andamentum.deep_research.v2.bench.arms import ArmOutput
from andamentum.deep_research.v2.bench.experiment import Experiment
from andamentum.deep_research.v2.bench.experiment_run import (
    DataDirs,
    Services,
    load_items,
    run_experiment,
    set_mode,
)
from andamentum.deep_research.v2.bench.metrics import Cost
from andamentum.deep_research.v2.bench.registry import (
    RunRecord,
    read_records,
    registry_path,
)
from andamentum.deep_research.v2.bench.simpleqa import (
    MAX_ANSWER_CHARS,
    AnswerGrader,
    Grade,
    grade_message,
    load_simpleqa,
    read_simpleqa,
)

ROWS = [
    ("In which year did the invented Apollo Ferry first sail?", "1911"),
    ("Who founded the invented SaltLog Society?", "Jane Doe"),
    ("How many towers does the invented Castle Vellum have?", "7"),
    ("What colour is the invented Moonfish?", "teal"),
]


def write_csv(folder: Path) -> Path:
    lines = ["metadata,problem,answer"]
    for question, answer in ROWS:
        lines.append(f"\"{{'topic': 'Invented'}}\",\"{question}\",{answer}")
    path = folder / "simple_qa_test_set.csv"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


class StubGrader(AnswerGrader):
    """Correct when the gold text is in the answer; not attempted when empty."""

    def __init__(self) -> None:
        super().__init__(model="stub")

    async def grade(self, *, question: str, gold: str, answer: str) -> Grade:
        if not answer:
            return "not_attempted"
        return "correct" if gold in answer else "incorrect"


def experiment(arms: list[dict], *, grader: str | None = "judge-model") -> Experiment:
    return Experiment.model_validate(
        {
            "name": "h2h",
            "registered": "2026-10-07",
            "grader": grader,
            "sets": [{"name": "simpleqa", "n": 3, "seed": 1}],
            "arms": arms,
            "baselines": {"simpleqa": arms[0]["name"]},
        }
    )


TWO_ARMS = [
    {"name": "old", "system": "legacy", "model": "m"},
    {"name": "new", "system": "v2", "preset": "hybrid", "model": "m", "frozen": False},
]


class ScriptedArm:
    returns_pages = False

    def __init__(self, name: str, answers: dict[str, str], order: list[str]) -> None:
        self.name = name
        self.answers = answers
        self.order = order

    async def run(
        self, question: str, *, frozen: object = None, store: object = None
    ) -> ArmOutput:
        self.order.append(f"{self.name}:{question[:20]}")
        return ArmOutput(
            answer_text=self.answers.get(question, ""), cost=Cost(wall_s=1.0)
        )


def test_rows_are_numbered_from_one_and_read_from_the_folder(tmp_path: Path) -> None:
    write_csv(tmp_path)
    items = read_simpleqa(tmp_path)
    assert [i.item_id for i in items] == ["SQ0001", "SQ0002", "SQ0003", "SQ0004"]
    assert items[1].answer == "Jane Doe"


def test_a_skipped_subset_holds_the_items_the_smaller_one_did_not(
    tmp_path: Path,
) -> None:
    write_csv(tmp_path)
    first = load_simpleqa(tmp_path, n=2, seed=1)
    rest = load_simpleqa(tmp_path, n=2, seed=1, skip=2)
    assert {i.item_id for i in first} | {i.item_id for i in rest} == {
        "SQ0001",
        "SQ0002",
        "SQ0003",
        "SQ0004",
    }
    assert [i.item_id for i in load_simpleqa(tmp_path, n=4, seed=1)][:2] == [
        i.item_id for i in first
    ]


def test_a_missing_file_names_where_to_get_it(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="simple-evals"):
        read_simpleqa(tmp_path)


def test_a_long_answer_is_cut_for_the_grader() -> None:
    message = grade_message(
        question="q", gold="g", answer="x" * (MAX_ANSWER_CHARS + 50)
    )
    assert message.endswith("[... cut]")
    assert "(no answer)" in grade_message(question="q", gold="g", answer="  ")


def test_the_simpleqa_set_needs_a_grader_that_is_no_arm_model() -> None:
    with pytest.raises(ValueError, match="needs a grader"):
        experiment(TWO_ARMS, grader=None)
    with pytest.raises(ValueError, match="must not grade itself"):
        experiment(TWO_ARMS, grader="m")


def test_simpleqa_and_live_litsearch_modes() -> None:
    assert set_mode("simpleqa") == "web"
    assert set_mode("litsearch-web") == "scholarly"


async def test_cells_run_item_by_item_and_are_graded(tmp_path: Path) -> None:
    write_csv(tmp_path)
    exp = experiment(TWO_ARMS)
    items = load_items(exp.set("simpleqa"), DataDirs(simpleqa=tmp_path))
    gold = {q: a for q, a in ROWS}
    order: list[str] = []
    arms = {
        # old answers every question right, new leaves the first one empty
        "old": ScriptedArm("old", {i.question: gold[i.question] for i in items}, order),
        "new": ScriptedArm(
            "new",
            {i.question: f"It is {gold[i.question]}." for i in items[1:]},
            order,
        ),
    }
    written = await run_experiment(
        exp,
        out_dir=tmp_path / "out",
        dirs=DataDirs(simpleqa=tmp_path),
        services=Services(
            answer_arm=lambda spec, _mode, _decision: arms[spec.name],
            grader=StubGrader(),
        ),
    )
    assert [o.split(":")[0] for o in order] == ["old", "new"] * 3
    assert order[0].split(":")[1] == order[1].split(":")[1]
    runs = [
        r
        for r in read_records(registry_path(tmp_path / "out"))
        if isinstance(r, RunRecord)
    ]
    assert len(runs) == len(written) == 6
    new = {r.item_id: r.metrics for r in runs if r.arm == "new"}
    assert new[items[0].item_id]["correct"] == 0.0
    assert new[items[0].item_id]["attempted"] == 0.0
    assert all(new[i.item_id]["correct"] == 1.0 for i in items[1:])
    assert all(r.metrics["correct"] == 1.0 for r in runs if r.arm == "old")


def test_live_litsearch_reads_the_local_download_items(tmp_path: Path) -> None:
    exp = Experiment.model_validate(
        {
            "name": "h",
            "registered": "2026-10-07",
            "sets": [{"name": "litsearch-web", "n": 2}],
            "arms": [TWO_ARMS[0]],
            "baselines": {"litsearch-web": "old"},
        }
    )
    with pytest.raises(ValueError, match="--litsearch-local-dir"):
        load_items(exp.set("litsearch-web"), DataDirs())
