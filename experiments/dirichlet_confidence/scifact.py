"""SciFact loader → flat ``(claim, evidence, label)`` examples.

Handles two input shapes (auto-detected):

1. **Canonical AllenAI SciFact**: a directory containing ``corpus.jsonl`` and a
   claims file (``claims_dev.jsonl`` / ``claims_train.jsonl`` / ``claims.jsonl``).
   - ``corpus.jsonl`` rows: ``{"doc_id", "title", "abstract": [sentences], ...}``
   - claims rows: ``{"id", "claim", "evidence": {doc_id: [{"sentences",
     "label"}]}, "cited_doc_ids": [...]}``.
   For each claim with evidence we emit one example per cited evidence doc with
   that doc's label (SUPPORT/CONTRADICT) and the doc abstract as the evidence
   passage. Claims with no evidence but with cited docs become NOINFO examples
   against the first cited abstract.

2. **Pre-flattened JSONL**: a ``.jsonl`` file whose rows already carry
   ``claim``, ``evidence`` (text) and ``label`` (string or 0/1/2). Used as-is.

If your files differ, point this at them and the detection / field names can be
adjusted in one place.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

LABEL2ID = {"SUPPORT": 0, "CONTRADICT": 1, "NOINFO": 2, "NEI": 2}
ID2LABEL = {0: "SUPPORT", 1: "CONTRADICT", 2: "NOINFO"}

_CLAIMS_CANDIDATES = ("claims_dev.jsonl", "claims_train.jsonl", "claims.jsonl")
_CORPUS_CANDIDATES = ("corpus.jsonl",)


@dataclass
class Example:
    claim_id: str
    claim: str
    evidence: str
    label: int  # 0=SUPPORT, 1=CONTRADICT, 2=NOINFO
    label_name: str


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _normalise_label(value) -> int:
    if isinstance(value, int):
        return value
    return LABEL2ID[str(value).strip().upper()]


def _abstract_text(corpus_row: dict) -> str:
    abstract = corpus_row.get("abstract", "")
    if isinstance(abstract, list):
        abstract = " ".join(abstract)
    title = corpus_row.get("title", "")
    return f"{title}\n\n{abstract}".strip() if title else str(abstract).strip()


def _load_flat(path: Path) -> list[Example]:
    out: list[Example] = []
    for i, row in enumerate(_read_jsonl(path)):
        label = _normalise_label(row["label"])
        out.append(
            Example(
                claim_id=str(row.get("id", row.get("claim_id", i))),
                claim=row["claim"],
                evidence=row["evidence"],
                label=label,
                label_name=ID2LABEL[label],
            )
        )
    return out


def _find(directory: Path, candidates: tuple[str, ...]) -> Path | None:
    for name in candidates:
        p = directory / name
        if p.exists():
            return p
    return None


def _load_canonical(
    directory: Path, claims_file: Path, include_noinfo: bool
) -> list[Example]:
    corpus_path = _find(directory, _CORPUS_CANDIDATES)
    if corpus_path is None:
        raise FileNotFoundError(
            f"corpus.jsonl not found in {directory} (needed to resolve evidence text)"
        )
    corpus = {str(r["doc_id"]): _abstract_text(r) for r in _read_jsonl(corpus_path)}

    out: list[Example] = []
    for row in _read_jsonl(claims_file):
        claim_id = str(row["id"])
        claim = row["claim"]
        evidence_map = row.get("evidence") or {}
        if evidence_map:
            for doc_id, entries in evidence_map.items():
                abstract = corpus.get(str(doc_id))
                if not abstract or not entries:
                    continue
                label_name = str(entries[0]["label"]).upper()
                label = LABEL2ID[label_name]
                out.append(
                    Example(
                        claim_id=f"{claim_id}:{doc_id}",
                        claim=claim,
                        evidence=abstract,
                        label=label,
                        label_name=ID2LABEL[label],
                    )
                )
        elif include_noinfo:
            cited = row.get("cited_doc_ids") or []
            for doc_id in cited[:1]:  # one NOINFO example per claim, keeps balance
                abstract = corpus.get(str(doc_id))
                if not abstract:
                    continue
                out.append(
                    Example(
                        claim_id=f"{claim_id}:{doc_id}",
                        claim=claim,
                        evidence=abstract,
                        label=2,
                        label_name="NOINFO",
                    )
                )
    return out


def load_scifact(path: str | Path, include_noinfo: bool = True) -> list[Example]:
    """Load SciFact examples from a directory or a flat JSONL file."""
    p = Path(path)
    if p.is_dir():
        claims_file = _find(p, _CLAIMS_CANDIDATES)
        if claims_file is None:
            raise FileNotFoundError(
                f"no claims file ({' / '.join(_CLAIMS_CANDIDATES)}) found in {p}"
            )
        return _load_canonical(p, claims_file, include_noinfo)
    if p.suffix == ".jsonl":
        return _load_flat(p)
    raise ValueError(
        f"unrecognised SciFact path {p!r}: expected a directory or a .jsonl file"
    )


def balanced_subset(examples: list[Example], n: int, seed: int = 42) -> list[Example]:
    """Take up to ``n`` examples, balanced across the three labels."""
    import numpy as np

    rng = np.random.default_rng(seed)
    by_label: dict[int, list[Example]] = {0: [], 1: [], 2: []}
    for ex in examples:
        by_label[ex.label].append(ex)

    # Distribute n across the three labels as evenly as possible (the first
    # ``n % 3`` labels get one extra), so n=8 -> 3/3/2 rather than 2/2/2.
    base, rem = divmod(n, 3)
    quota = {label: base + (1 if i < rem else 0) for i, label in enumerate((0, 1, 2))}
    picked: list[Example] = []
    for label in (0, 1, 2):
        pool = by_label[label]
        if not pool:
            continue
        idx = rng.permutation(len(pool))[: quota[label]]
        picked.extend(pool[i] for i in idx)
    rng.shuffle(picked)
    return picked[:n]


def label_counts(examples: list[Example]) -> dict[str, int]:
    counts = {"SUPPORT": 0, "CONTRADICT": 0, "NOINFO": 0}
    for ex in examples:
        counts[ex.label_name] += 1
    return counts
