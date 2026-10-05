"""Tiny synthetic datasets for the benchmark tests. Everything here is invented
(Apollo, SaltLog, Jane Doe): no real record, query or review is reproduced."""

from __future__ import annotations

import csv
import io
import json
import zipfile
from pathlib import Path

REVIEW_KEY = "Invented_2020"
REVIEW_TITLE = "Does SaltLog reduce brine loss in Apollo kilns? A systematic review"
N_INCLUDED = 4
N_EXCLUDED = 16


def inverted(text: str) -> dict[str, list[int]]:
    index: dict[str, list[int]] = {}
    for position, word in enumerate(text.split()):
        index.setdefault(word, []).append(position)
    return index


def invented_work(number: int, *, included: bool) -> dict[str, object]:
    subject = "SaltLog" if included else "Unrelated"
    return {
        "id": f"https://openalex.org/W{1000 + number}",
        "doi": f"https://doi.org/10.5555/invented.{number}",
        "title": f"{subject} kiln study number {number}",
        "publication_year": 2000 + number,
        "primary_location": {"source": {"display_name": "Journal of Invented Results"}},
        "abstract_inverted_index": (
            inverted(f"{subject} was measured in kiln {number} of the Apollo plant")
            if number % 7
            else None
        ),
    }


def make_synergy_dir(root: Path, *, key: str = REVIEW_KEY, parts: int = 2) -> Path:
    """``root/synergy/<key>/`` with labels, publication record and works zips."""
    folder = root / "synergy" / key
    folder.mkdir(parents=True)
    numbers = list(range(N_INCLUDED + N_EXCLUDED))
    rows = io.StringIO()
    writer = csv.writer(rows)
    writer.writerow(["openalex_id", "doi", "pmid", "label_included"])
    works = []
    for number in numbers:
        included = number < N_INCLUDED
        work = invented_work(number, included=included)
        works.append(work)
        writer.writerow([work["id"], work["doi"], "", "1" if included else "0"])
    (folder / "labels.csv").write_text(rows.getvalue(), encoding="utf-8")
    (folder / "metadata_publication.json").write_text(
        json.dumps({"title": REVIEW_TITLE, "doi": "https://doi.org/10.5555/review"}),
        encoding="utf-8",
    )
    chunk = -(-len(works) // parts)
    for part in range(parts):
        with zipfile.ZipFile(folder / f"works_{part + 1}.zip", "w") as archive:
            archive.writestr(
                f"works_{part * chunk}_{(part + 1) * chunk}.json",
                json.dumps(works[part * chunk : (part + 1) * chunk]),
            )
    return folder


def litsearch_rows() -> list[dict[str, object]]:
    return [
        {
            "row_idx": i,
            "row": {
                "query_set": "inline_acl",
                "query": f"Which paper studies invented topic {i}?",
                "specificity": i % 2,
                "quality": 2,
                "corpusids": [9000 + i, 9100 + i] if i == 0 else [9000 + i],
            },
            "truncated_cells": [],
        }
        for i in range(5)
    ]


def s2_paper(corpus_id: int) -> dict[str, object] | None:
    if corpus_id == 9003:  # one gold paper Semantic Scholar does not know
        return None
    return {
        "paperId": f"{corpus_id:040x}",
        "title": f"An invented paper about topic {corpus_id}",
        "year": 2021,
        "externalIds": {
            "CorpusId": corpus_id,
            "DOI": f"10.5555/gold.{corpus_id}",
            **({"ArXiv": f"2101.{corpus_id:05d}"} if corpus_id % 2 else {}),
        },
    }
