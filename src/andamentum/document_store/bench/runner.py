"""Run one arm on one suite tier, record it, and compare two recorded runs.

A run is a directory under ``workdir/runs`` holding ``manifest.json`` (what was
run, on what), ``questions.jsonl`` (ranked passage ids and metrics per
question) and ``summary.json``. Two runs are compared only when their suite,
tier, question ids and corpus hash match; every other manifest difference is
printed next to the result.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import platform
import sqlite3
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from .arms import ARMS, DEPTH, ArmResult, build_current_store, build_v2_store
from .data import load_corpus
from .embedding_cache import EmbeddingCache
from .metrics import (
    CUTOFFS,
    answer_metrics,
    mean_metrics,
    paired_bootstrap,
    question_metrics,
)

#: Must match for two runs to be compared at all.
_COMPARABLE = ("suite", "suite_version", "tier", "question_ids_sha256", "corpus_sha256")


def _git_state() -> dict[str, str | bool]:
    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args], capture_output=True, text=True, cwd=Path(__file__).parent
        ).stdout.strip()

    return {
        "commit": git("rev-parse", "HEAD"),
        "dirty": bool(git("status", "--porcelain")),
    }


def _agent_means(rows: list[dict]) -> dict[str, float]:
    """Mean em / f1 / llm_calls / tokens / seconds over the rows that carry each."""
    out: dict[str, float] = {}
    for key in ("em", "f1", "llm_calls", "tokens_in", "tokens_out", "seconds"):
        vals = [r[key] for r in rows if r.get(key) is not None]
        if vals:
            out[key] = sum(vals) / len(vals)
    return out


async def run(
    *,
    suite: str,
    tier: str,
    arm_name: str,
    workdir: Path,
    embedding_model: str | None,
    driver_model: str | None = None,
    concurrency: int = 1,
) -> Path:
    if concurrency < 1:
        raise ValueError("concurrency must be at least 1")
    arm = ARMS[arm_name]
    if arm.needs_embeddings and embedding_model is None:
        raise ValueError(f"arm {arm_name!r} needs --embedding-model")
    corpus, spec = load_corpus(suite, tier, workdir=workdir)
    emb = (
        EmbeddingCache.for_ollama(
            workdir / "cache" / "embeddings.sqlite", model=embedding_model
        )
        if arm.needs_embeddings and embedding_model
        else None
    )

    t0 = time.perf_counter()
    builder = build_v2_store if arm.store_kind == "v2" else build_current_store
    built = await builder(corpus, workdir=workdir, emb=emb)
    build_seconds = time.perf_counter() - t0

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_name = f"{stamp}_{suite}_{tier}_{arm_name}"
    semaphore = asyncio.Semaphore(concurrency)

    async def one(q) -> dict:
        async with semaphore:
            # A store query that SQLite rejects is the store's behaviour on that
            # question, not a benchmark crash: it scores as an empty ranking and
            # the error is recorded and counted, never dropped.
            error = None
            result: ArmResult | None = None
            t_q = time.perf_counter()
            try:
                out = await arm.rank(
                    built,
                    q,
                    emb,
                    driver_model=driver_model,
                    run_id=f"{run_name}:{q.qid}",
                )
                result = out if isinstance(out, ArmResult) else ArmResult(ranked=out)
                ranked = result.ranked
            except sqlite3.OperationalError as e:
                ranked, error = [], f"{type(e).__name__}: {e}"
            seconds = time.perf_counter() - t_q
        row: dict = {
            "qid": q.qid,
            "group": q.group,
            "n_gold": len(q.gold),
            "ranked": ranked,
            "error": error,
            "metrics": question_metrics(ranked, q.gold),
        }
        if result is not None and (
            result.answer is not None or result.llm_calls or result.trace
        ):
            row.update(
                answer=result.answer,
                llm_calls=result.llm_calls,
                tokens_in=result.tokens_in,
                tokens_out=result.tokens_out,
                seconds=round(seconds, 3),
                entered_loop=result.entered_loop,
                trace=result.trace,
            )
            if result.answer is not None:
                row.update(answer_metrics(result.answer, q.answers))
        return row

    t0 = time.perf_counter()
    rows = list(await asyncio.gather(*(one(q) for q in corpus.questions)))
    search_seconds = time.perf_counter() - t0

    run_dir = workdir / "runs" / run_name
    run_dir.mkdir(parents=True)
    manifest = {
        "run": run_dir.name,
        "created": stamp,
        "suite": suite,
        "suite_version": spec["version"],
        "tier": tier,
        "n_questions": len(corpus.questions),
        "n_passages": len(corpus.passages),
        "question_ids_sha256": hashlib.sha256(
            "\n".join(q.qid for q in corpus.questions).encode()
        ).hexdigest(),
        "corpus_sha256": corpus.sha256,
        "arm": arm_name,
        "store": arm.store_kind,
        "reranker": getattr(arm, "reranker", None),
        "v2_exact": getattr(arm, "exact", None),
        "v2_vector_weight": getattr(arm, "vector_weight", None),
        "depth": DEPTH,
        "driver_model": driver_model,
        "concurrency": concurrency,
        **{
            f"mean_{k}": round(v, 4)
            for k, v in _agent_means(rows).items()
            if k in ("llm_calls", "tokens_in", "tokens_out", "seconds")
        },
        "embedding_model": emb.model if emb else None,
        "embedding_digest": emb.digest if emb else None,
        "embeddings_computed": emb.misses if emb else 0,
        "build_seconds": round(build_seconds, 2),
        "search_seconds": round(search_seconds, 2),
        "query_errors": sum(r["error"] is not None for r in rows),
        "empty_rankings": sum(not r["ranked"] for r in rows),
        "git": _git_state(),
        "python": platform.python_version(),
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    with (run_dir / "questions.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    groups = sorted({r["group"] for r in rows})
    summary = {
        "all": {**mean_metrics([r["metrics"] for r in rows]), **_agent_means(rows)},
        "by_group": {
            g: {
                "n": sum(r["group"] == g for r in rows),
                **mean_metrics([r["metrics"] for r in rows if r["group"] == g]),
                **_agent_means([r for r in rows if r["group"] == g]),
            }
            for g in groups
        },
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    return run_dir


def _load(run_dir: Path) -> tuple[dict, list[dict]]:
    manifest = json.loads((run_dir / "manifest.json").read_text())
    rows = [json.loads(line) for line in (run_dir / "questions.jsonl").open()]
    return manifest, rows


HEADLINE = (
    "full_support@10",
    "full_support@20",
    "recall@5",
    "recall@10",
    "mrr",
    "ndcg@10",
)


def format_summary(run_dir: Path) -> str:
    manifest, rows = _load(run_dir)
    summary = json.loads((run_dir / "summary.json").read_text())
    lines = [
        f"{manifest['run']}  ({manifest['n_questions']} questions, "
        f"{manifest['n_passages']} pooled passages; {manifest['query_errors']} query "
        f"errors, {manifest['empty_rankings']} empty rankings)",
        "  " + "  ".join(f"{k:>16}" for k in ("group", "n", *HEADLINE)),
    ]
    blocks = [("all", {"n": len(rows), **summary["all"]}), *summary["by_group"].items()]
    for name, m in blocks:
        lines.append(
            "  "
            + f"{name:>16}  {m['n']:>16}  "
            + "  ".join(f"{m[k]:>16.3f}" for k in HEADLINE)
        )
    allm = summary["all"]
    if "em" in allm:
        lines.append(
            f"  answers: em {allm['em']:.3f}  f1 {allm['f1']:.3f}  "
            f"llm_calls {allm.get('llm_calls', 0):.2f}  "
            f"tokens_in {allm.get('tokens_in', 0):.0f}  "
            f"tokens_out {allm.get('tokens_out', 0):.0f}  "
            f"seconds {allm.get('seconds', 0):.1f} per question"
        )
    return "\n".join(lines)


def compare(run_a: Path, run_b: Path) -> str:
    """Per-metric paired-bootstrap difference (b - a) over shared questions."""
    ma, ra = _load(run_a)
    mb, rb = _load(run_b)
    mismatched = [k for k in _COMPARABLE if ma[k] != mb[k]]
    if mismatched:
        raise ValueError(
            f"runs are not comparable; they differ in {', '.join(mismatched)}"
        )
    other = sorted(
        k
        for k in ma.keys() & mb.keys()
        if k not in _COMPARABLE
        and k
        not in (
            "run",
            "created",
            "build_seconds",
            "search_seconds",
            "embeddings_computed",
            "mean_llm_calls",
            "mean_tokens_in",
            "mean_tokens_out",
            "mean_seconds",
            "query_errors",
            "empty_rankings",
        )
        and ma[k] != mb[k]
    )
    by_q = {r["qid"]: r for r in rb}
    pairs = [(r, by_q[r["qid"]]) for r in ra]
    metrics = (
        [f"full_support@{k}" for k in CUTOFFS]
        + [f"recall@{k}" for k in CUTOFFS]
        + ["mrr", "ndcg@10"]
    )
    answered = all("em" in r for r in ra) and all("em" in r for r in rb)
    if answered:
        metrics += ["em", "f1"]
    lines = [
        f"A = {ma['run']}",
        f"B = {mb['run']}",
        f"{len(pairs)} paired questions; difference is B - A, 95% paired-bootstrap interval",
        f"  {'metric':>16}  {'A':>7}  {'B':>7}  {'B - A':>7}  {'interval':>17}",
    ]
    for m in metrics:
        if m in ("em", "f1"):
            a = [x[m] for x, _ in pairs]
            b = [y[m] for _, y in pairs]
        else:
            a = [x["metrics"][m] for x, _ in pairs]
            b = [y["metrics"][m] for _, y in pairs]
        diff, lo, hi = paired_bootstrap(a, b)
        flag = "" if lo <= 0 <= hi else "  *"
        lines.append(
            f"  {m:>16}  {sum(a) / len(a):7.3f}  {sum(b) / len(b):7.3f}  "
            f"{diff:+7.3f}  [{lo:+.3f}, {hi:+.3f}]{flag}"
        )
    lines.append("  * interval excludes zero")
    if "mean_llm_calls" in ma and "mean_llm_calls" in mb:
        for label, m_ in (("A", ma), ("B", mb)):
            lines.append(
                f"  cost {label}: llm_calls {m_['mean_llm_calls']:.2f}  "
                f"seconds {m_['mean_seconds']:.1f} per question"
            )
    if other:
        lines.append(
            "Manifest differences: "
            + "; ".join(f"{k}: {ma[k]} -> {mb[k]}" for k in other)
        )
    return "\n".join(lines)
