"""Benchmark datasets: download, normalise, sample once, pool.

Each suite is a public multi-hop QA dev set whose questions come with their
own candidate passages, some marked as supporting. The benchmark never uses
the answer text: a question is scored only on whether the store ranks its
supporting passages near the top of a POOLED corpus — every passage supplied
with every sampled question, so each question's passages compete with the
distractors of all the others.

The sample is drawn once (``freeze``) and written to ``suites/<name>.json``,
which is committed. Every later run reads the same question ids and checks the
raw file and the pooled corpus against the recorded sha256, so a changed
download fails loudly instead of quietly moving the numbers.

Dev and test are split by linked group, not by question: two questions are
linked when they share a supporting passage, and a whole group goes to one
side. Dev and test therefore share no question and no supporting passage. A
test question's supporting passage can still appear in the dev pool as a
distractor: linking on every shared passage puts most of MuSiQue into one
group, so that stricter split cannot be drawn.
"""

from __future__ import annotations

import hashlib
import json
import random
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Iterable

import httpx

SUITES_DIR = Path(__file__).parent / "suites"

#: Split name -> number of questions. Design choices are made on ``dev``;
#: ``test`` only reports. No question that shapes the store also scores it.
SPLITS: dict[str, int] = {"dev": 200, "test": 300}

#: ``smoke`` is the first 20 dev questions: it checks that an arm runs, and its
#: numbers are never reported.
SMOKE_SIZE = 20
RUN_TIERS = ("smoke", *SPLITS)

#: Version of the frozen-sample format; a run records it.
SUITE_VERSION = 2

_SEED = 1234

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
    answers: tuple[
        str, ...
    ] = ()  # accepted answer strings (never used by retrieval arms)
    cluster: str = ""  # linked group; the resampling unit of the bootstrap


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
        qid=row["id"],
        text=row["question"],
        group=row["type"],
        gold=frozenset(gold),
        answers=(row["answer"],),
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
    answers = (row["answer"], *row.get("answer_aliases", ()))
    q = Question(
        qid=row["id"],
        text=row["question"],
        group=group,
        gold=frozenset(gold),
        answers=tuple(answers),
    )
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


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


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


def suite_file(suite: str, *, suites_dir: Path = SUITES_DIR) -> Path:
    return suites_dir / f"{suite}.json"


def pool(
    suite: str,
    tier: str,
    records: dict[str, tuple[Question, list[Passage]]],
    qids: list[str],
    clusters: dict[str, str] | None = None,
) -> Corpus:
    questions: list[Question] = []
    by_pid: dict[str, Passage] = {}
    for qid in qids:
        q, ps = records[qid]
        if clusters is not None:
            q = replace(q, cluster=clusters[qid])
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


def link_groups(questions: Iterable[Question]) -> dict[str, str]:
    """Question id -> linked-group id; questions sharing a supporting passage link.

    The group id is the smallest question id in the group, so it does not
    depend on the order of the input.
    """
    qs = sorted(questions, key=lambda q: q.qid)
    parent = {q.qid: q.qid for q in qs}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    first_with: dict[str, str] = {}
    for q in qs:
        for pid in sorted(q.gold):
            if pid in first_with:
                a, b = find(first_with[pid]), find(q.qid)
                if a != b:
                    parent[max(a, b)] = min(a, b)
            else:
                first_with[pid] = q.qid
    return {q.qid: find(q.qid) for q in qs}


def draw_splits(
    questions: Iterable[Question], *, seed: int = _SEED
) -> tuple[list[str], list[str], dict[str, str]]:
    """Draw (dev ids, test ids, groups); a linked group never spans both splits.

    Groups are shuffled and dealt to the test side until it holds half the
    eligible questions, the rest to the dev side; each split's sample is then
    drawn from its own side.
    """
    eligible = [q for q in questions if q.gold]
    groups = link_groups(eligible)
    members: dict[str, list[str]] = {}
    for qid, g in sorted(groups.items()):
        members.setdefault(g, []).append(qid)
    order = sorted(members)
    rng = random.Random(seed)
    rng.shuffle(order)
    test_side: list[str] = []
    dev_side: list[str] = []
    for g in order:
        side = test_side if len(test_side) < len(eligible) / 2 else dev_side
        side.extend(members[g])
    for name, side in (("dev", dev_side), ("test", test_side)):
        if len(side) < SPLITS[name]:
            raise ValueError(
                f"only {len(side)} eligible questions on the {name} side; "
                f"{SPLITS[name]} needed"
            )
    dev = rng.sample(sorted(dev_side), SPLITS["dev"])
    test = rng.sample(sorted(test_side), SPLITS["test"])
    return dev, test, {qid: groups[qid] for qid in dev + test}


def check_splits(
    spec: dict, records: dict[str, tuple[Question, list[Passage]]]
) -> None:
    """Raise unless dev and test share no question, group or supporting passage."""
    dev, test = spec["dev_ids"], spec["test_ids"]
    shared = set(dev) & set(test)
    if shared:
        raise ValueError(f"{len(shared)} questions are in both dev and test")
    groups = spec["groups"]
    shared_groups = {groups[q] for q in dev} & {groups[q] for q in test}
    if shared_groups:
        raise ValueError(f"{len(shared_groups)} linked groups span dev and test")
    gold_dev = set().union(*(records[q][0].gold for q in dev))
    gold_test = set().union(*(records[q][0].gold for q in test))
    if gold_dev & gold_test:
        raise ValueError(
            f"{len(gold_dev & gold_test)} supporting passages are in dev and test"
        )


def _tier_ids(spec: dict, tier: str) -> list[str]:
    if tier == "smoke":
        return spec["dev_ids"][:SMOKE_SIZE]
    return spec[f"{tier}_ids"]


def freeze(suite: str, *, workdir: Path, suites_dir: Path = SUITES_DIR) -> Path:
    """Draw dev and test and record them. Refuses to redraw an existing sample."""
    path = suite_file(suite, suites_dir=suites_dir)
    if path.exists():
        raise FileExistsError(
            f"{path} already exists; a frozen sample is never redrawn. "
            "Bump the suite version deliberately by deleting it in a reviewed commit."
        )
    records = load_records(suite, workdir=workdir)
    dev, test, groups = draw_splits(q for q, _ in records.values())
    spec: dict = {
        "suite": suite,
        "version": SUITE_VERSION,
        "source": SOURCES[suite].filename,
        "source_sha256": file_sha256(dataset_path(suite, workdir=workdir)),
        "seed": _SEED,
        "eligible_questions": sum(1 for q, _ in records.values() if q.gold),
        "split_rule": "linked groups (shared supporting passage) never span dev and test",
        "dev_ids": dev,
        "test_ids": test,
        "groups": groups,
    }
    check_splits(spec, records)
    spec["tiers"] = {}
    for tier in RUN_TIERS:
        corpus = pool(suite, tier, records, _tier_ids(spec, tier))
        spec["tiers"][tier] = {
            "n_questions": len(corpus.questions),
            "n_passages": len(corpus.passages),
            "corpus_sha256": corpus.sha256,
        }
    suites_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(spec, indent=1) + "\n")
    return path


def load_spec(suite: str, *, suites_dir: Path = SUITES_DIR) -> dict:
    spec = json.loads(suite_file(suite, suites_dir=suites_dir).read_text())
    if spec.get("version") != SUITE_VERSION:
        raise ValueError(
            f"{suite}: frozen sample version {spec.get('version')} is not "
            f"{SUITE_VERSION}; freeze it again"
        )
    return spec


def load_corpus(
    suite: str, tier: str, *, workdir: Path, suites_dir: Path = SUITES_DIR
) -> tuple[Corpus, dict]:
    """The frozen sample's pooled corpus for one tier, checked against its hashes."""
    spec = load_spec(suite, suites_dir=suites_dir)
    raw = dataset_path(suite, workdir=workdir)
    if raw.exists() and file_sha256(raw) != spec["source_sha256"]:
        raise ValueError(
            f"{raw} sha256 does not match the frozen {spec['source_sha256']}; "
            "the downloaded data changed"
        )
    records = load_records(suite, workdir=workdir)
    check_splits(spec, records)
    ids = _tier_ids(spec, tier)
    missing = [q for q in ids if q not in records]
    if missing:
        raise ValueError(
            f"{suite}: {len(missing)} frozen question ids not in the dataset"
        )
    corpus = pool(suite, tier, records, ids, spec["groups"])
    expected = spec["tiers"][tier]["corpus_sha256"]
    if corpus.sha256 != expected:
        raise ValueError(
            f"{suite}/{tier}: pooled corpus sha256 {corpus.sha256} does not match "
            f"the frozen {expected}; the downloaded data changed"
        )
    return corpus, spec
