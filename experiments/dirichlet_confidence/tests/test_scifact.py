"""Offline tests for the SciFact loader using tiny synthetic fixtures."""

from __future__ import annotations

import json
from pathlib import Path

from experiments.dirichlet_confidence.scifact import (
    balanced_subset,
    label_counts,
    load_scifact,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")


def test_canonical_directory_loading(tmp_path: Path):
    _write_jsonl(
        tmp_path / "corpus.jsonl",
        [
            {"doc_id": 1, "title": "T1", "abstract": ["a", "b"]},
            {"doc_id": 2, "title": "T2", "abstract": ["c", "d"]},
            {"doc_id": 3, "title": "T3", "abstract": ["e"]},
        ],
    )
    _write_jsonl(
        tmp_path / "claims_dev.jsonl",
        [
            {
                "id": 10,
                "claim": "C supports",
                "evidence": {"1": [{"sentences": [0], "label": "SUPPORT"}]},
                "cited_doc_ids": [1],
            },
            {
                "id": 11,
                "claim": "C contradicts",
                "evidence": {"2": [{"sentences": [0], "label": "CONTRADICT"}]},
                "cited_doc_ids": [2],
            },
            {"id": 12, "claim": "C noinfo", "evidence": {}, "cited_doc_ids": [3]},
        ],
    )
    examples = load_scifact(tmp_path)
    assert label_counts(examples) == {"SUPPORT": 1, "CONTRADICT": 1, "NOINFO": 1}
    support = next(e for e in examples if e.label_name == "SUPPORT")
    assert "T1" in support.evidence and "a b" in support.evidence


def test_canonical_excludes_noinfo_when_requested(tmp_path: Path):
    _write_jsonl(
        tmp_path / "corpus.jsonl", [{"doc_id": 3, "title": "T", "abstract": ["e"]}]
    )
    _write_jsonl(
        tmp_path / "claims.jsonl",
        [{"id": 12, "claim": "c", "evidence": {}, "cited_doc_ids": [3]}],
    )
    assert load_scifact(tmp_path, include_noinfo=False) == []


def test_flat_jsonl_loading(tmp_path: Path):
    path = tmp_path / "flat.jsonl"
    _write_jsonl(
        path,
        [
            {"id": "a", "claim": "x", "evidence": "ev", "label": "SUPPORT"},
            {"id": "b", "claim": "y", "evidence": "ev2", "label": 1},
        ],
    )
    examples = load_scifact(path)
    assert examples[0].label == 0
    assert examples[1].label == 1


def test_balanced_subset_is_balanced(tmp_path: Path):
    path = tmp_path / "flat.jsonl"
    rows = (
        [
            {"id": f"s{i}", "claim": "c", "evidence": "e", "label": "SUPPORT"}
            for i in range(20)
        ]
        + [
            {"id": f"c{i}", "claim": "c", "evidence": "e", "label": "CONTRADICT"}
            for i in range(20)
        ]
        + [
            {"id": f"n{i}", "claim": "c", "evidence": "e", "label": "NOINFO"}
            for i in range(20)
        ]
    )
    _write_jsonl(path, rows)
    examples = load_scifact(path)
    subset = balanced_subset(examples, 9, seed=1)
    counts = label_counts(subset)
    assert counts["SUPPORT"] == counts["CONTRADICT"] == counts["NOINFO"] == 3
