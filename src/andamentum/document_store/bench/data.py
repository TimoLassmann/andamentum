"""Benchmark datasets: download, normalise, sample once, pool.

Each suite is a public multi-hop QA dev set whose questions come with their
own candidate passages, some marked as supporting. The benchmark never uses
the answer text: a question is scored only on whether the store ranks its
supporting passages near the top of a POOLED corpus — every passage supplied
with every sampled question, so each question's passages compete with the
distractors of all the others.

The sample is drawn once (``freeze``) and written to ``suites/<name>.json``,
which is committed. Every later run reads the same question ids and checks the
pooled corpus against the recorded sha256, so a changed download fails loudly
instead of quietly moving the numbers.
"""

from __future__ import annotations

import hashlib
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import httpx

SUITES_DIR = Path(__file__).parent / "suites"

#: Tier name -> number of questions. ``smoke`` is the first 20 of the frozen
#: ``standard`` sample, so a smoke run is a strict subset of a standard run.
TIERS: dict[str, int] = {"smoke": 20, "standard": 200}

_SEED = 17

# HotpotQA distractor dev, served as JSON by the Hugging Face dataset viewer
# (the parquet files need pyarrow, which is not a dependency).
_HOTPOT_ROWS_URL = "https://datasets-server.huggingface.co/rows"
_HOTPOT_DATASET = "hotpotqa/hotpot_qa"
_HOTPOT_PAGE = 100

# MuSiQue answerable dev, pinned to one commit of a public mirror of the
# official release (the official copy sits behind a Google Drive link).
_MUSIQUE_URL = (
    "https://huggingface.co/datasets/bdsaglam/musique/resolve/"
    "22873a405dd809893b22ada0b499299fb612d2df/musique_ans_v1.0_dev.jsonl"
)


@dataclass(frozen=True)
class Passage:
    pid: str
    title: str
    text: str


@dataclass(frozen=True)
class Question:
    qid: str
    text: str
    group: str
    gold: frozenset[str]  # pids of the supporting passages


@dataclass(frozen=True)
class Corpus:
    suite: str
    tier: str
    questions: list[Question]
    passages: list[Passage]  # pooled, deduplicated, sorted by pid
    sha256: str


def passage_id(*, title: str, text: str) -> str:
    """Content address of a passage: identical title + text is one passage."""
    return hashlib.sha256(f"{title}\n{text}".encode()).hexdigest()[:16]


def corpus_sha256(passages: Iterable[Passage]) -> str:
    h = hashlib.sha256()
    for p in sorted(passages, key=lambda p: p.pid):
        h.update(f"{p.pid}\t{p.title}\t{p.text}\n".encode())
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Normalising one raw record -> (Question, its passages)
# ---------------------------------------------------------------------------


def parse_hotpotqa(row: dict) -> tuple[Question, list[Passage]]:
    support_titles = set(row["supporting_facts"]["title"])
    passages: list[Passage] = []
    gold: set[str] = set()
    for title, sentences in zip(
        row["context"]["title"], row["context"]["sentences"], strict=True
    ):
        text = "".join(sentences).strip()
        p = Passage(pid=passage_id(title=title, text=text), title=title, text=text)
        passages.append(p)
        if title in support_titles:
            gold.add(p.pid)
    q = Question(
        qid=row["id"], text=row["question"], group=row["type"], gold=frozenset(gold)
    )
    return q, passages


def parse_musique(row: dict) -> tuple[Question, list[Passage]]:
    passages: list[Passage] = []
    gold: set[str] = set()
    for para in row["paragraphs"]:
        title, text = para["title"], para["paragraph_text"].strip()
        p = Passage(pid=passage_id(title=title, text=text), title=title, text=text)
        passages.append(p)
        if para["is_supporting"]:
            gold.add(p.pid)
    # Ids look like "2hop__153573_109006" or "3hop1__...": the prefix is the
    # hop structure, which is the natural reporting group.
    group = row["id"].split("__", 1)[0]
    q = Question(qid=row["id"], text=row["question"], group=group, gold=frozenset(gold))
    return q, passages


# ---------------------------------------------------------------------------
# Download (once, into the caller's workdir)
# ---------------------------------------------------------------------------


def _fetch_hotpotqa(dest: Path) -> None:
    rows: list[dict] = []
    with httpx.Client(timeout=60.0) as client:
        offset, total = 0, None
        while total is None or offset < total:
            params = {
                "dataset": _HOTPOT_DATASET,
                "config": "distractor",
                "split": "validation",
                "offset": offset,
                "length": _HOTPOT_PAGE,
            }
            r = client.get(_HOTPOT_ROWS_URL, params=params)
            for wait in (5, 15, 30, 60, 120):
                # The viewer rate-limits after a few dozen pages.
                if r.status_code != 429:
                    break
                time.sleep(float(r.headers.get("retry-after", wait)))
                r = client.get(_HOTPOT_ROWS_URL, params=params)
            r.raise_for_status()
            page = r.json()
            total = page["num_rows_total"]
            rows.extend(item["row"] for item in page["rows"])
            offset += _HOTPOT_PAGE
    if len(rows) != total:
        raise RuntimeError(f"HotpotQA download incomplete: {len(rows)} of {total} rows")
    _write_jsonl(dest, rows)


def _fetch_musique(dest: Path) -> None:
    with httpx.Client(timeout=120.0, follow_redirects=True) as client:
        r = client.get(_MUSIQUE_URL)
        r.raise_for_status()
    tmp = dest.with_suffix(".part")
    tmp.write_bytes(r.content)
    tmp.replace(dest)


def _write_jsonl(dest: Path, rows: list[dict]) -> None:
    tmp = dest.with_suffix(".part")
    with tmp.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(dest)


@dataclass(frozen=True)
class SuiteSource:
    name: str
    filename: str
    fetch: Callable[[Path], None]
    parse: Callable[[dict], tuple[Question, list[Passage]]]


SOURCES: dict[str, SuiteSource] = {
    "hotpotqa": SuiteSource(
        "hotpotqa", "hotpotqa_distractor_dev.jsonl", _fetch_hotpotqa, parse_hotpotqa
    ),
    "musique": SuiteSource(
        "musique", "musique_ans_dev.jsonl", _fetch_musique, parse_musique
    ),
}


def dataset_path(suite: str, *, workdir: Path) -> Path:
    return workdir / "datasets" / SOURCES[suite].filename


def fetch(suite: str, *, workdir: Path) -> Path:
    """Download a suite's raw dev set into ``workdir/datasets`` unless present."""
    dest = dataset_path(suite, workdir=workdir)
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        SOURCES[suite].fetch(dest)
    return dest


def load_records(
    suite: str, *, workdir: Path
) -> dict[str, tuple[Question, list[Passage]]]:
    path = dataset_path(suite, workdir=workdir)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing; run `python -m andamentum.document_store.bench "
            f"fetch --workdir {workdir}` first"
        )
    parse = SOURCES[suite].parse
    out: dict[str, tuple[Question, list[Passage]]] = {}
    with path.open() as f:
        for line in f:
            q, ps = parse(json.loads(line))
            out[q.qid] = (q, ps)
    return out


# ---------------------------------------------------------------------------
# Freeze the sample once; build pooled corpora from it
# ---------------------------------------------------------------------------


def suite_file(suite: str) -> Path:
    return SUITES_DIR / f"{suite}.json"


def pool(
    suite: str,
    tier: str,
    records: dict[str, tuple[Question, list[Passage]]],
    qids: list[str],
) -> Corpus:
    questions: list[Question] = []
    by_pid: dict[str, Passage] = {}
    for qid in qids:
        q, ps = records[qid]
        questions.append(q)
        for p in ps:
            by_pid.setdefault(p.pid, p)
    passages = sorted(by_pid.values(), key=lambda p: p.pid)
    return Corpus(
        suite=suite,
        tier=tier,
        questions=questions,
        passages=passages,
        sha256=corpus_sha256(passages),
    )


def freeze(suite: str, *, workdir: Path) -> Path:
    """Draw the sample and record it. Refuses to redraw an existing sample."""
    path = suite_file(suite)
    if path.exists():
        raise FileExistsError(
            f"{path} already exists; a frozen sample is never redrawn. "
            "Bump the suite version deliberately by deleting it in a reviewed commit."
        )
    records = load_records(suite, workdir=workdir)
    # Questions without a supporting passage cannot be scored on retrieval.
    eligible = sorted(qid for qid, (q, _) in records.items() if q.gold)
    sample = random.Random(_SEED).sample(eligible, TIERS["standard"])
    tiers = {}
    for tier, n in TIERS.items():
        corpus = pool(suite, tier, records, sample[:n])
        tiers[tier] = {
            "n_questions": n,
            "n_passages": len(corpus.passages),
            "corpus_sha256": corpus.sha256,
        }
    spec = {
        "suite": suite,
        "version": 1,
        "source": SOURCES[suite].filename,
        "seed": _SEED,
        "eligible_questions": len(eligible),
        "tiers": tiers,
        "sample_ids": sample,
    }
    SUITES_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(spec, indent=1) + "\n")
    return path


def load_corpus(suite: str, tier: str, *, workdir: Path) -> tuple[Corpus, dict]:
    """The frozen sample's pooled corpus for one tier, checked against its hash."""
    spec = json.loads(suite_file(suite).read_text())
    n = spec["tiers"][tier]["n_questions"]
    records = load_records(suite, workdir=workdir)
    missing = [q for q in spec["sample_ids"][:n] if q not in records]
    if missing:
        raise ValueError(
            f"{suite}: {len(missing)} frozen question ids not in the dataset"
        )
    corpus = pool(suite, tier, records, spec["sample_ids"][:n])
    expected = spec["tiers"][tier]["corpus_sha256"]
    if corpus.sha256 != expected:
        raise ValueError(
            f"{suite}/{tier}: pooled corpus sha256 {corpus.sha256} does not match "
            f"the frozen {expected}; the downloaded data changed"
        )
    return corpus, spec
