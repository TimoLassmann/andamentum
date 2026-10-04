"""Evaluation harness: compose against a single-call baseline, same checks.

For every synthetic task under ``evals/briefs/<genre>/<name>/`` arm A runs the
full ``write_document`` pipeline and arm B has one model call write the whole
document. The brief compiled in arm A is shared with arm B, and both final
documents get the same check pass (``check_bundle``, no revision), so the arms
differ only in how the text was produced. A judge then compares the two
documents in both presentation orders. Everything runs sequentially.

One difference stays by design: arm A's checks also see its own outline (the
section targets it planned); arm B has none.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel

from andamentum.llm_judge import judge_compare
from andamentum.compose.bundle import Bundle

from .. import store
from ..api import check_bundle, write_document
from ..assess import Assessment
from ..evidence import is_tabular, stored_source_path
from ..genres import GENRES, genre_for
from ..log import ActionLog
from ..render import plain_text
from ..schemas import Roles
from ..steps import call_counts
from .baseline import single_call

BRIEFS_DIR = Path(__file__).parent / "briefs"
_WINNERS: dict[str, Literal["compose", "baseline", "tie"]] = {
    "a": "compose",
    "b": "baseline",
    "tie": "tie",
}


class EvalTask(BaseModel):
    genre: str
    name: str
    task: str
    sources: list[Path]


class ArmMetrics(BaseModel):
    checks_passed: list[str]
    checks_failed: list[str]
    reader_correct: int
    reader_total: int
    grounding_unsupported: int
    words: int
    rounds: int
    stop_reason: str
    tokens: dict[str, int]
    calls: dict[str, int]


class TaskEval(BaseModel):
    genre: str
    name: str
    compose: ArmMetrics
    baseline: ArmMetrics
    winner: Literal["compose", "baseline", "tie"]
    judge_confidence: float
    needs_review: bool


def _tokens(m: ArmMetrics) -> str:
    """Total tokens as ``N (drafter N, ...)``; ``not reported`` when none were."""
    total = sum(m.tokens.values())
    if not total:
        return "not reported"
    by_role = ", ".join(f"{role} {n}" for role, n in sorted(m.tokens.items()) if n)
    return f"{total} ({by_role})"


class EvalReport(BaseModel):
    roles: Roles
    tasks: list[TaskEval]

    def markdown(self) -> str:
        head = (
            "| genre | task | arm | checks failed | reader | unsupported | words "
            "| rounds | stop | tokens |\n|---|---|---|---|---|---|---|---|---|---|"
        )
        rows = [head]
        for t in self.tasks:
            for arm, m in (("compose", t.compose), ("baseline", t.baseline)):
                failed = ", ".join(m.checks_failed) or "none"
                rows.append(
                    f"| {t.genre} | {t.name} | {arm} | {failed} "
                    f"| {m.reader_correct}/{m.reader_total} | {m.grounding_unsupported} "
                    f"| {m.words} | {m.rounds} | {m.stop_reason} | {_tokens(m)} |"
                )
        wins = {
            k: sum(1 for t in self.tasks if t.winner == k)
            for k in ("compose", "baseline", "tie")
        }
        rows += [
            "",
            f"Judge (both orders): compose {wins['compose']}, baseline {wins['baseline']}, "
            f"tie {wins['tie']} of {len(self.tasks)} tasks.",
        ]
        for t in self.tasks:
            rows.append(
                f"- {t.genre}/{t.name}: {t.winner} (confidence {t.judge_confidence:.2f}"
                + (", needs review)" if t.needs_review else ")")
            )
        return "\n".join(rows)


def load_tasks(
    *, genres: list[str] | None = None, briefs_dir: Path | None = None
) -> list[EvalTask]:
    """The synthetic tasks, optionally limited to some genres, in a stable order."""
    root = briefs_dir or BRIEFS_DIR
    wanted = genres if genres is not None else sorted(GENRES)
    for name in wanted:
        if name not in GENRES:
            raise ValueError(
                f"unknown genre {name!r}; valid: {', '.join(sorted(GENRES))}"
            )
    tasks: list[EvalTask] = []
    for genre in wanted:
        for directory in sorted(p for p in (root / genre).glob("*") if p.is_dir()):
            data = yaml.safe_load((directory / "task.yaml").read_text(encoding="utf-8"))
            tasks.append(
                EvalTask(
                    genre=genre,
                    name=directory.name,
                    task=str(data["task"]).strip(),
                    sources=[
                        directory / "sources" / s for s in data.get("sources", [])
                    ],
                )
            )
    if not tasks:
        raise ValueError(f"no evaluation tasks found under {root} for {wanted}")
    return tasks


def metrics(
    assessment: Assessment,
    *,
    bundle: Bundle,
    rounds: int,
    stop_reason: str,
) -> ArmMetrics:
    """One arm's numbers from a check pass over its final bundle."""
    log = ActionLog(bundle.path)
    return ArmMetrics(
        checks_passed=sorted({c.name for c in assessment.checks if c.passed}),
        checks_failed=sorted({c.name for c in assessment.checks if not c.passed}),
        reader_correct=assessment.reader.correct,
        reader_total=assessment.reader.total,
        grounding_unsupported=assessment.grounding_unsupported,
        words=sum(len(s.body.split()) for s in bundle.sections()),
        rounds=rounds,
        stop_reason=stop_reason,
        tokens=log.usage(),
        calls=call_counts(log),
    )


async def _baseline_arm(
    task: EvalTask,
    *,
    compose_dir: Path,
    out: Path,
    roles: Roles,
    progress: Callable[[str], None],
) -> Bundle:
    """Arm B: one call writes the document into its own bundle, same brief."""
    brief = store.load_brief(compose_dir)
    bundle = Bundle.create(out, title=brief.title)
    store.save_brief(bundle.path, brief)
    store.save_sources(bundle.path, store.load_sources(compose_dir))
    store.save_facts(bundle.path, store.load_facts(compose_dir))
    bundle.set_metadata(genre=task.genre, task=task.task, models=roles.model_dump())
    sources = store.load_sources(compose_dir)
    texts = {
        s.title: stored_source_path(compose_dir, s).read_text(encoding="utf-8")
        for s in sources
        if not is_tabular(s.path)
    }
    progress(f"baseline: {task.genre}/{task.name}")
    document = await single_call(
        task.task,
        texts,
        genre_for(task.genre),
        model=roles.drafter,
        facts=store.load_facts(compose_dir),
        log=ActionLog(bundle.path),
    )
    for section in document.sections:
        bundle.add_section(title=section.title, body=section.body, status="drafted")
    return bundle


async def run_eval(
    *,
    genres: list[str] | None,
    roles: Roles,
    out: str | Path,
    max_rounds: int,
    progress: Callable[[str], None],
    briefs_dir: Path | None = None,
) -> EvalReport:
    """Run both arms over the tasks, write ``eval.json`` and ``eval.md``."""
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    results: list[TaskEval] = []
    for task in load_tasks(genres=genres, briefs_dir=briefs_dir):
        base = out_dir / task.genre / task.name
        progress(f"compose: {task.genre}/{task.name}")
        written = await write_document(
            task.task,
            out=base / "compose",
            roles=roles,
            genre=task.genre,
            sources=[str(p) for p in task.sources],
            max_rounds=max_rounds,
            progress=progress,
        )
        compose_bundle = Bundle.open(written.bundle)
        baseline_bundle = await _baseline_arm(
            task,
            compose_dir=written.bundle,
            out=base / "baseline",
            roles=roles,
            progress=progress,
        )
        arm_a = await check_bundle(compose_bundle.path, roles=roles, progress=progress)
        arm_b = await check_bundle(baseline_bundle.path, roles=roles, progress=progress)
        verdict = await judge_compare(
            plain_text(compose_bundle.sections(), title=compose_bundle.title),
            plain_text(baseline_bundle.sections(), title=baseline_bundle.title),
            context=written.brief.task,
            model=roles.judge,
        )
        results.append(
            TaskEval(
                genre=task.genre,
                name=task.name,
                compose=metrics(
                    arm_a,
                    bundle=compose_bundle,
                    rounds=len(written.rounds),
                    stop_reason=written.stop_reason,
                ),
                baseline=metrics(
                    arm_b, bundle=baseline_bundle, rounds=0, stop_reason="single_call"
                ),
                winner=_WINNERS[verdict.winner],
                judge_confidence=verdict.confidence,
                needs_review=verdict.needs_review,
            )
        )
    report = EvalReport(roles=roles, tasks=results)
    (out_dir / "eval.json").write_text(
        report.model_dump_json(indent=2), encoding="utf-8"
    )
    (out_dir / "eval.md").write_text(report.markdown() + "\n", encoding="utf-8")
    return report
