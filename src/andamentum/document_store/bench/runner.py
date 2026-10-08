"""Run one arm on one suite tier, record it, and compare two recorded runs.

A run writes into a directory the caller names: ``manifest.json`` (what was
run, on what, and how it ended), ``questions.jsonl`` (ranked passage ids,
metrics and answers per question) and ``summary.json``. The manifest is
written first with ``status: running``, so a run that fails still leaves a
record; any error in any question stops the whole run, marks the manifest
``failed`` and re-raises, so the caller sees a nonzero exit. Nothing is scored
from a broken run.

Two runs are compared only when their suite, tier, question ids, corpus hash
and code commit match, and their embedding models match when both used one.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import platform
import time
import traceback
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .arms import (
    ARMS,
    DEPTH,
    DRIVER_SETTINGS,
    ArmResult,
    build_pool,
    build_store,
)
from .data import load_corpus
from .embedding_cache import EmbeddingCache
from .metrics import (
    CUTOFFS,
    answer_metrics,
    mean_metrics,
    paired_bootstrap_test,
    question_metrics,
)
from .provenance import git_state, uv_lock_sha256

#: Must match for two runs to be compared at all.
_COMPARABLE = ("suite", "suite_version", "tier", "question_ids_sha256", "corpus_sha256")

_BUILDERS = {
    "store": build_store,
    "none": build_pool,
}

_ANSWER_KEYS = (
    "em",
    "f1",
    "unknown",
    "llm_calls",
    "tokens_in",
    "tokens_out",
    "seconds",
)

RETRIEVAL_METRICS = (
    *(f"full_support@{k}" for k in CUTOFFS),
    *(f"recall@{k}" for k in CUTOFFS),
    "mrr",
    "ndcg@10",
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _agent_means(rows: list[dict]) -> dict[str, float]:
    """Mean of each answer and cost key; a run's rows carry the same keys."""
    out: dict[str, float] = {}
    for key in _ANSWER_KEYS:
        vals = [r[key] for r in rows if key in r]
        if vals:
            if len(vals) != len(rows):
                raise ValueError(f"{key} is missing from {len(rows) - len(vals)} rows")
            out[key] = sum(vals) / len(vals)
    return out


def _reranker_revision(name: str | None) -> str | None:
    """The Hugging Face snapshot the reranker loads, read from the local cache."""
    if name is None:
        return None
    from huggingface_hub import try_to_load_from_cache

    path = try_to_load_from_cache(name, "config.json")
    if not isinstance(path, str):
        raise FileNotFoundError(f"reranker {name} is not in the local model cache")
    return Path(path).parent.name


def _write_json(path: Path, data: Any) -> None:
    tmp = path.with_suffix(".part")
    tmp.write_text(json.dumps(data, indent=1) + "\n")
    tmp.replace(path)


async def run(
    *,
    suite: str,
    tier: str,
    arm_name: str,
    workdir: Path,
    out_dir: Path,
    embedding_model: str | None,
    driver_model: str | None = None,
    concurrency: int = 1,
    repeat: int = 1,
    argv: list[str] | None = None,
) -> Path:
    if concurrency < 1:
        raise ValueError("concurrency must be at least 1")
    if (out_dir / "manifest.json").exists():
        raise FileExistsError(
            f"{out_dir} already holds a run; a run is never overwritten"
        )
    arm = ARMS[arm_name]
    if arm.needs_embeddings and embedding_model is None:
        raise ValueError(f"arm {arm_name!r} needs --embedding-model")
    out_dir.mkdir(parents=True, exist_ok=True)
    answering = bool(getattr(arm, "answers", False))
    # A model-driven arm calls the driver model: it answers, or an agent runs
    # its search. Either way its costs and trace are recorded per question.
    llm_driven = answering or bool(getattr(arm, "llm_driven", False))
    manifest: dict[str, Any] = {
        "run": out_dir.name,
        "status": "running",
        "started": _now(),
        "argv": argv,
        "suite": suite,
        "tier": tier,
        "arm": arm_name,
        "repeat": repeat,
        "store": arm.store_kind,
        "reranker": getattr(arm, "reranker", None),
        "depth": DEPTH,
        "driver_model": driver_model,
        "driver_settings": DRIVER_SETTINGS if llm_driven else None,
        "concurrency": concurrency,
        "git": git_state(),
        "uv_lock_sha256": uv_lock_sha256(),
        "python": platform.python_version(),
    }
    manifest_path = out_dir / "manifest.json"
    _write_json(manifest_path, manifest)
    try:
        await _run_questions(
            manifest,
            out_dir=out_dir,
            arm=arm,
            workdir=workdir,
            embedding_model=embedding_model,
            driver_model=driver_model,
            concurrency=concurrency,
            answering=answering,
            llm_driven=llm_driven,
        )
    except BaseException as exc:
        manifest.update(
            status="failed",
            ended=_now(),
            error=f"{type(exc).__name__}: {exc}",
            traceback=traceback.format_exc(),
        )
        _write_json(manifest_path, manifest)
        raise
    manifest.update(status="complete", ended=_now())
    _write_json(manifest_path, manifest)
    return out_dir


async def _run_questions(
    manifest: dict[str, Any],
    *,
    out_dir: Path,
    arm: Any,
    workdir: Path,
    embedding_model: str | None,
    driver_model: str | None,
    concurrency: int,
    answering: bool,
    llm_driven: bool,
) -> None:
    corpus, spec = load_corpus(manifest["suite"], manifest["tier"], workdir=workdir)
    emb = (
        EmbeddingCache.for_ollama(
            workdir / "cache" / "embeddings.sqlite", model=embedding_model
        )
        if arm.needs_embeddings and embedding_model
        else None
    )
    manifest.update(
        suite_version=spec["version"],
        n_questions=len(corpus.questions),
        n_passages=len(corpus.passages),
        question_ids_sha256=hashlib.sha256(
            "\n".join(q.qid for q in corpus.questions).encode()
        ).hexdigest(),
        corpus_sha256=corpus.sha256,
        embedding_model=emb.model if emb else None,
        embedding_digest=emb.digest if emb else None,
        reranker_revision=_reranker_revision(manifest["reranker"]),
    )
    if answering:
        from .answering import prompts_sha256

        manifest["prompts_sha256"] = prompts_sha256()
    elif llm_driven:
        from ..agent import prompts_sha256

        manifest["prompts_sha256"] = prompts_sha256()

    t0 = time.perf_counter()
    built = await _BUILDERS[arm.store_kind](corpus, workdir=workdir, emb=emb)
    manifest["build_seconds"] = round(time.perf_counter() - t0, 2)
    info = built.info
    manifest["store_file"] = (
        None
        if info is None
        else {
            "path": Path(info.path).name,
            "layout": info.layout,
            "reused": info.reused,
            "built": info.build.get("built"),
            "build_git": info.build.get("git"),
        }
    )

    semaphore = asyncio.Semaphore(concurrency)
    run_name = out_dir.name

    async def one(q) -> dict:
        async with semaphore:
            t_q = time.perf_counter()
            out = await arm.rank(
                built, q, emb, driver_model=driver_model, run_id=f"{run_name}:{q.qid}"
            )
            seconds = time.perf_counter() - t_q
        result = out if isinstance(out, ArmResult) else ArmResult(ranked=out)
        ranked = result.ranked
        if len(ranked) != len(set(ranked)) or len(ranked) > DEPTH:
            raise ValueError(
                f"{q.qid}: ranking has duplicates or exceeds depth {DEPTH}"
            )
        unknown = set(ranked) - {p.pid for p in corpus.passages}
        if unknown:
            raise ValueError(f"{q.qid}: {len(unknown)} ranked ids are not in the pool")
        row: dict = {
            "qid": q.qid,
            "group": q.group,
            "cluster": q.cluster,
            "n_gold": len(q.gold),
            "ranked": ranked,
            "metrics": question_metrics(ranked, q.gold),
        }
        if llm_driven:
            row.update(
                llm_calls=result.llm_calls,
                tokens_in=result.tokens_in,
                tokens_out=result.tokens_out,
                seconds=round(seconds, 3),
                entered_loop=result.entered_loop,
                trace=result.trace,
            )
        if answering:
            if result.answer is None:
                raise ValueError(f"{q.qid}: answering arm returned no answer")
            row.update(answer=result.answer, **answer_metrics(result.answer, q.answers))
        return row

    t0 = time.perf_counter()
    try:
        async with asyncio.TaskGroup() as tg:  # the first error cancels the rest
            tasks = [tg.create_task(one(q)) for q in corpus.questions]
    except ExceptionGroup as group:
        raise group.exceptions[0] from group  # the caller sees the real error
    finally:
        store = getattr(built, "store", None)
        if store is not None:
            await store.close()
    rows = [t.result() for t in tasks]
    manifest.update(
        search_seconds=round(time.perf_counter() - t0, 2),
        embeddings_computed=emb.misses if emb else 0,
        empty_rankings=sum(not r["ranked"] for r in rows),
        # How many questions the agent answered by a fallback, by kind, so a
        # gain or loss that comes from the fallback path is visible.
        fallbacks=dict(
            sorted(
                Counter(
                    step.get("kind", "")
                    for r in rows
                    for step in r.get("trace", [])
                    if step.get("kind", "").endswith("_fallback")
                ).items()
            )
        ),
        **{
            f"mean_{k}": round(v, 4)
            for k, v in _agent_means(rows).items()
            if k in ("llm_calls", "tokens_in", "tokens_out", "seconds")
        },
    )
    with (out_dir / "questions.jsonl").open("w") as f:
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
    _write_json(out_dir / "summary.json", summary)


def load_run(run_dir: Path) -> tuple[dict, list[dict]]:
    """A complete run's manifest and rows; refuses a running or failed run."""
    manifest = json.loads((run_dir / "manifest.json").read_text())
    if manifest.get("status") != "complete":
        raise ValueError(
            f"{run_dir} did not complete (status {manifest.get('status')})"
        )
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
    manifest, rows = load_run(run_dir)
    summary = json.loads((run_dir / "summary.json").read_text())
    lines = [
        f"{manifest['run']}  ({manifest['n_questions']} questions, "
        f"{manifest['n_passages']} pooled passages; "
        f"{manifest['empty_rankings']} empty rankings)",
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
            f"unknown {allm['unknown']:.3f}  "
            f"llm_calls {allm['llm_calls']:.2f}  "
            f"tokens_in {allm['tokens_in']:.0f}  "
            f"tokens_out {allm['tokens_out']:.0f}  "
            f"seconds {allm['seconds']:.1f} per question"
        )
    elif "llm_calls" in allm:
        lines.append(
            f"  cost: llm_calls {allm['llm_calls']:.2f}  "
            f"tokens_in {allm['tokens_in']:.0f}  "
            f"tokens_out {allm['tokens_out']:.0f}  "
            f"seconds {allm['seconds']:.1f} per question"
        )
    return "\n".join(lines)


def _check_comparable(ma: dict, mb: dict) -> None:
    mismatched = [k for k in _COMPARABLE if ma[k] != mb[k]]
    if ma["git"]["commit"] != mb["git"]["commit"]:
        mismatched.append("git commit")
    if ma["git"]["dirty"] or mb["git"]["dirty"]:
        mismatched.append("uncommitted code")
    for key in ("embedding_model", "embedding_digest"):
        if ma[key] is not None and mb[key] is not None and ma[key] != mb[key]:
            mismatched.append(key)
    if mismatched:
        raise ValueError(
            f"runs are not comparable; they differ in {', '.join(mismatched)}"
        )


def _values(rows: list[dict], metric: str) -> list[float]:
    if metric in ("em", "f1", "unknown"):
        return [r[metric] for r in rows]
    return [r["metrics"][metric] for r in rows]


def contrast(run_a: Path, run_b: Path, metric: str) -> dict[str, Any]:
    """B - A on one metric over paired questions, with a linked-group bootstrap."""
    ma, ra = load_run(run_a)
    mb, rb = load_run(run_b)
    _check_comparable(ma, mb)
    by_q = {r["qid"]: r for r in rb}
    if set(by_q) != {r["qid"] for r in ra}:
        raise ValueError("runs do not cover the same questions")
    rb = [by_q[r["qid"]] for r in ra]
    a, b = _values(ra, metric), _values(rb, metric)
    diff, lo, hi, p = paired_bootstrap_test(a, b, clusters=[r["cluster"] for r in ra])
    return {
        "a": ma["run"],
        "b": mb["run"],
        "metric": metric,
        "n": len(a),
        "n_clusters": len({r["cluster"] for r in ra}),
        "mean_a": sum(a) / len(a),
        "mean_b": sum(b) / len(b),
        "diff": diff,
        "lo": lo,
        "hi": hi,
        "p": p,
        "b_better": sum(y > x for x, y in zip(a, b, strict=True)),
        "a_better": sum(x > y for x, y in zip(a, b, strict=True)),
    }


def compare(run_a: Path, run_b: Path) -> str:
    """Per-metric paired-bootstrap difference (b - a) over shared questions."""
    _, ra = load_run(run_a)
    _, rb = load_run(run_b)
    metrics = list(RETRIEVAL_METRICS)
    if all("em" in r for r in ra) and all("em" in r for r in rb):
        metrics += ["em", "f1", "unknown"]
    results = [contrast(run_a, run_b, m) for m in metrics]
    first = results[0]
    lines = [
        f"A = {first['a']}",
        f"B = {first['b']}",
        f"{first['n']} paired questions in {first['n_clusters']} linked groups; "
        "difference is B - A, 95% linked-group bootstrap interval",
        f"  {'metric':>16}  {'A':>7}  {'B':>7}  {'B - A':>7}  {'interval':>17}",
    ]
    for c in results:
        flag = "" if c["lo"] <= 0 <= c["hi"] else "  *"
        lines.append(
            f"  {c['metric']:>16}  {c['mean_a']:7.3f}  {c['mean_b']:7.3f}  "
            f"{c['diff']:+7.3f}  [{c['lo']:+.3f}, {c['hi']:+.3f}]{flag}"
        )
    lines.append("  * interval excludes zero")
    return "\n".join(lines)


TABLE_COLUMNS = (
    "suite",
    "tier",
    "arm",
    "repeat",
    "n_questions",
    *RETRIEVAL_METRICS,
    *_ANSWER_KEYS,
    "embedding_model",
    "reranker",
    "driver_model",
    "commit",
    "run",
)


def table_row(run_dir: Path) -> dict[str, Any]:
    """One outcome-table row for a complete run; answer columns empty for retrieval arms."""
    manifest, _ = load_run(run_dir)
    summary = json.loads((run_dir / "summary.json").read_text())["all"]
    row: dict[str, Any] = {k: manifest.get(k) for k in TABLE_COLUMNS}
    row.update({k: summary.get(k) for k in (*RETRIEVAL_METRICS, *_ANSWER_KEYS)})
    row["commit"] = manifest["git"]["commit"]
    return row
