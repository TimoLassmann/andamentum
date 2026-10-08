"""The experiment report: tables, paired differences, cost, verdicts.

``build_report`` reads the registry of one experiment and writes markdown. It
computes, and decides nothing the experiment file did not register:

- per-arm means of each item metric (an item's value is the mean over its
  repeats);
- paired differences against the set's baseline arm, 95% bootstrap intervals
  over items (``stats.paired_bootstrap_ci``);
- cost, and where a run's time and model requests went, step by step
  (``RunStats.step_*`` of every v2 run);
- how many runs failed, and how many of those met a search outage;
- the clean-timing (solo) runs, reported apart from everything else;
- the verdict of every pre-registered rule, the outcome of each rule group and
  the pre-registered wording of what the groups decide.

Where an answer was lost is not in this report: ``experiment losses`` reads the
run logs for that (``attribution.py``).

Work ids are random per store, so before any two runs are compared the ids are
replaced by the identity key of the work (``stable_key``).
"""

from __future__ import annotations

import statistics
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

from andamentum.references import Work
from andamentum.references.works import normalise_title

from ..types import STEPS, ResearchResult, RunStats
from .experiment import ArmSpec, Experiment, Rule
from .registry import RunRecord, StartRecord, read_records, registry_path
from .tables import NOTHING, fmt, table
from .verdicts import (
    Interval,
    Outcome,
    Verdict,
    judge_rule,
    paired_interval,
)

DEFAULT_REPORT_METRICS: dict[str, list[str]] = {
    "litsearch-web": [
        "recall_at_5",
        "recall_at_10",
        "recall_at_20",
        "recall_at_50",
        "found_any",
        "kept_precision",
        "kept_n",
    ],
    "simpleqa": ["correct", "attempted"],
}
COST_METRICS = (
    "wall_s",
    "llm_calls",
    "decision_calls",
    "input_tokens",
    "output_tokens",
)
# -- grouping -----------------------------------------------------------------------


def stable_key(work: Work) -> str:
    """A run-independent name for a work: its first identifier, else its title."""
    keys = work.identity_keys()
    if keys:
        kind, value = keys[0]
        return f"{kind}:{value}"
    return f"title:{normalise_title(work.title)}"


def canonicalise(result: ResearchResult) -> ResearchResult:
    """The result with every work id replaced by the work's stable key."""
    names = {w.work_id: stable_key(w) for w in result.works}
    return result.model_copy(
        update={
            "works": [
                w.model_copy(update={"work_id": names[w.work_id]}) for w in result.works
            ],
            "findings": [
                f.model_copy(update={"work_id": names.get(f.work_id, f.work_id)})
                for f in result.findings
            ],
        }
    )


class Cells:
    """The run records of one experiment, indexed by (arm, set, item)."""

    def __init__(self, runs: Sequence[RunRecord]) -> None:
        self.runs = list(runs)
        self._by: dict[tuple[str, str, str], list[RunRecord]] = defaultdict(list)
        for run in sorted(runs, key=lambda r: r.repeat):
            self._by[(run.arm, run.set, run.item_id)].append(run)

    def repeats(self, arm: str, set_name: str, item_id: str) -> list[RunRecord]:
        return self._by.get((arm, set_name, item_id), [])

    def items(self, arm: str, set_name: str) -> list[str]:
        seen: dict[str, None] = {}
        for a, s, i in self._by:
            if a == arm and s == set_name:
                seen[i] = None
        return sorted(seen)

    def n_errors(self, arm: str, set_name: str) -> int:
        return sum(
            1 for r in self.runs if r.arm == arm and r.set == set_name and r.error
        )

    def n_outages(self, arm: str, set_name: str) -> int:
        return sum(
            1 for r in self.runs if r.arm == arm and r.set == set_name and r.outage
        )


def _mean(values: Sequence[float | None]) -> float | None:
    known = [v for v in values if v is not None]
    return sum(known) / len(known) if known else None


def item_values(
    cells: Cells, arm: str, set_name: str, metric: str
) -> dict[str, float | None]:
    """One value per item: the mean of the metric over the item's repeats."""
    return {
        item: _mean([r.metrics.get(metric) for r in cells.repeats(arm, set_name, item)])
        for item in cells.items(arm, set_name)
    }


# -- formatting ---------------------------------------------------------------------


def fmt_interval(interval: Interval | None) -> str:
    if interval is None:
        return NOTHING
    return f"{interval.estimate:+.3f} [{interval.low:+.3f}, {interval.high:+.3f}]"


def _metrics_for(exp: Experiment, set_name: str) -> list[str]:
    return exp.report_metrics.get(set_name) or DEFAULT_REPORT_METRICS[set_name]


# -- sections -----------------------------------------------------------------------


def rule_values(cells: Cells, arm: str, rule: Rule) -> dict[str, float | None]:
    """The item values a rule reads for *arm*: its set's, or, for a rule that
    pools sets, every set's with the items named ``<set>/<item>``."""
    if not rule.also_sets:
        return item_values(cells, arm, rule.set, rule.metric)
    return {
        f"{set_name}/{item}": value
        for set_name in rule.sets
        for item, value in item_values(cells, arm, set_name, rule.metric).items()
    }


def _arms_section(exp: Experiment) -> str:
    rows = []
    for arm in exp.arms:
        rows.append(
            [
                arm.name,
                arm.system,
                arm.model,
                ("solo: first %d items of each set" % arm.solo if arm.is_solo else "-")
                + (", failed outage cells recorded" if arm.outage == "record" else ""),
                str(exp.repeats_of(arm)),
            ]
        )
    return table(["arm", "system", "model", "timing", "repeats"], rows)


def _means_section(exp: Experiment, cells: Cells, set_name: str) -> str:
    metrics = _metrics_for(exp, set_name)
    rows = []
    for arm in exp.main_arms_on(set_name):
        n = len(cells.items(arm.name, set_name))
        row = [arm.name, str(n), str(cells.n_errors(arm.name, set_name))]
        row += [
            fmt(_mean(list(item_values(cells, arm.name, set_name, m).values())))
            for m in metrics
        ]
        rows.append(row)
    return table(["arm", "items", "errors", *metrics], rows)


def _paired_section(exp: Experiment, cells: Cells, set_name: str) -> str:
    baseline = exp.baselines.get(set_name)
    if baseline is None:
        return "_No baseline arm is registered for this set._"
    metrics = _metrics_for(exp, set_name)
    base = {m: item_values(cells, baseline, set_name, m) for m in metrics}
    rows = []
    for arm in exp.main_arms_on(set_name):
        if arm.name == baseline:
            continue
        row = [arm.name]
        for m in metrics:
            interval = paired_interval(
                item_values(cells, arm.name, set_name, m),
                base[m],
                resamples=exp.resamples,
                seed=exp.bootstrap_seed,
            )
            n_items = len(
                set(cells.items(arm.name, set_name))
                | set(cells.items(baseline, set_name))
            )
            k = 0 if interval is None else interval.n_items
            row.append(f"{fmt_interval(interval)} ({k} of {n_items})")
        rows.append(row)
    if not rows:
        return f"_Only the baseline arm {baseline} runs on this set._"
    return (
        f"Difference from the baseline arm **{baseline}** (arm minus baseline), mean over "
        f"items, 95% bootstrap interval ({exp.resamples} resamples over items). "
        "`(k of N)`: the items both arms have a value for, of all items either ran; "
        "an item without a value (no claims to judge, an error) drops out of that cell:\n\n"
        + table(["arm", *metrics], rows)
    )


def _cost_section(exp: Experiment, cells: Cells) -> str:
    rows = []
    for arm in exp.arms:
        if arm.is_solo:
            continue  # clean timing has its own section
        for set_spec in exp.sets:
            if not arm.runs_on(set_spec.name):
                continue
            runs = [
                r for r in cells.runs if r.arm == arm.name and r.set == set_spec.name
            ]
            if not runs:
                continue
            cost = {m: [r.metrics.get(m) for r in runs] for m in COST_METRICS}
            wall = [v for v in cost["wall_s"] if v is not None]
            rows.append(
                [
                    arm.name,
                    set_spec.name,
                    str(len(runs)),
                    fmt(statistics.median(wall), 1) if wall else NOTHING,
                    fmt(sum(wall), 0) if wall else NOTHING,
                    *[fmt(_mean(cost[m]), 1) for m in COST_METRICS[1:]],
                ]
            )
    return (
        table(
            [
                "arm",
                "set",
                "runs",
                "median wall s",
                "total wall s",
                "mean llm calls",
                "mean decision calls",
                "mean input tokens",
                "mean output tokens",
            ],
            rows,
        )
        + "\n\nEvery arm runs live, so wall time is end to end. The legacy "
        "pipeline counts no tokens or calls (n/a)."
    )


def judge_rules(exp: Experiment, cells: Cells) -> list[Verdict]:
    verdicts = []
    for rule in exp.rules:
        arm = rule_values(cells, rule.arm, rule)
        reference = rule_values(cells, rule.reference, rule) if rule.reference else None
        verdict = judge_rule(
            rule,
            arm=arm,
            reference=reference,
            resamples=exp.resamples,
            seed=exp.bootstrap_seed,
        )
        verdicts.append(_require_every_item(rule, verdict, cells))
    return verdicts


def _require_every_item(rule: Rule, verdict: Verdict, cells: Cells) -> Verdict:
    """A gating rule that rests on fewer items than the arms ran is inconclusive:
    an item that dropped out (no claims, an error) must not be quietly ignored."""
    if not rule.gate or verdict.interval is None or rule.test == "no_harm":
        # a guard has no inconclusive outcome; its statement names how many items it read
        return verdict
    ran = {
        (set_name, item)
        for set_name in rule.sets
        for item in cells.items(rule.arm, set_name)
    }
    if rule.reference:
        ran |= {
            (set_name, item)
            for set_name in rule.sets
            for item in cells.items(rule.reference, set_name)
        }
    k = verdict.interval.n_items
    if k >= len(ran) or verdict.outcome == "inconclusive":
        return verdict
    return verdict.model_copy(
        update={
            "outcome": "inconclusive",
            "statement": f"{verdict.statement}; only {k} of {len(ran)} items have a "
            f"value, so the rule is not settled (it said {verdict.outcome} on that subset)",
        }
    )


def overall(verdicts: Sequence[Verdict]) -> str:
    """The gate: accepted only if every gating rule is accepted."""
    gates = [v for v in verdicts if v.gate]
    if not gates:
        return "no gating rule"
    if any(v.outcome == "not accepted" for v in gates):
        return "not accepted"
    if all(v.outcome == "accepted" for v in gates):
        return "accepted"
    return "inconclusive"


def group_outcome(verdicts: Sequence[Verdict]) -> Outcome:
    """Rules that stand or fall together: ``accepted`` when every one is,
    ``not accepted`` when any is not, else ``inconclusive`` (the rules' own
    ``gate`` flags are not read: a group is asked about as a whole)."""
    if any(v.outcome == "not accepted" for v in verdicts):
        return "not accepted"
    if verdicts and all(v.outcome == "accepted" for v in verdicts):
        return "accepted"
    return "inconclusive"


def decision_lines(exp: Experiment, verdicts: Sequence[Verdict]) -> list[str]:
    """What each registered decision says, in the file's own words: the
    ``if_accepted`` wording when every required group is accepted, else the
    ``otherwise`` wording, with each group's outcome beside it."""
    by_rule = {v.rule_id: v for v in verdicts}
    groups = exp.rule_groups()
    lines = []
    for decision in exp.decisions:
        outcomes = {
            g: group_outcome([by_rule[r.id] for r in groups[g]])
            for g in decision.requires
        }
        said = (
            decision.if_accepted
            if all(o == "accepted" for o in outcomes.values())
            else decision.otherwise
        )
        shown = ", ".join(f"{g} {o}" for g, o in outcomes.items())
        lines.append(f"- **{decision.id}** ({shown}): {said}")
    return lines


def _verdict_section(exp: Experiment, cells: Cells) -> str:
    verdicts = judge_rules(exp, cells)
    by_id = {v.rule_id: v for v in verdicts}
    rows = []
    for rule in exp.rules:
        verdict = by_id[rule.id]
        rows.append(
            [
                rule.id,
                rule.group or "-",
                rule.question,
                "gate" if rule.gate else "reported",
                f"**{verdict.outcome}**",
                verdict.statement,
            ]
        )
    text = table(["rule", "group", "question", "role", "verdict", "evidence"], rows)
    text += f"\n\nGating rules together: **{overall(verdicts)}**."
    groups = exp.rule_groups()
    if groups:
        text += "\n\nRule groups (all rules of a group accepted = accepted):\n\n"
        text += table(
            ["group", "rules", "outcome"],
            [
                [
                    group,
                    ", ".join(r.id for r in rules),
                    f"**{group_outcome([by_id[r.id] for r in rules])}**",
                ]
                for group, rules in groups.items()
            ],
        )
    if exp.decisions:
        text += "\n\nWhat the groups decide (wording fixed in the file):\n\n"
        text += "\n".join(decision_lines(exp, verdicts))
    return text


# -- steps, failures and clean timing --------------------------------------------


def _stats(record: RunRecord) -> RunStats | None:
    """The v2 run's stats, when the record holds a v2 result."""
    if record.error or record.output is None or record.output.result is None:
        return None
    return record.output.result.stats


def _median(values: Sequence[float]) -> float | None:
    return statistics.median(values) if values else None


def _v2_runs(
    exp: Experiment, cells: Cells, arm: ArmSpec, set_name: str
) -> list[RunStats]:
    return [
        stats
        for r in cells.runs
        if r.arm == arm.name and r.set == set_name and (stats := _stats(r)) is not None
    ]


def _step_tables(exp: Experiment, cells: Cells, *, solo: bool) -> str | None:
    """Seconds and model requests per step of the v2 runs of the arms that are
    (or are not) clean-timing re-runs. A step a run never reached counts as 0."""
    seconds_rows: list[list[str]] = []
    calls_rows: list[list[str]] = []
    for arm in exp.arms:
        if arm.system != "v2" or arm.is_solo != solo:
            continue
        for set_spec in exp.sets:
            if not arm.runs_on(set_spec.name):
                continue
            runs = _v2_runs(exp, cells, arm, set_spec.name)
            if not runs:
                continue
            head = [arm.name, set_spec.name, str(len(runs))]
            seconds_rows.append(
                [
                    *head,
                    *[
                        fmt(_median([s.step_seconds.get(step, 0.0) for s in runs]), 1)
                        for step in STEPS
                    ],
                    fmt(_median([sum(s.step_seconds.values()) for s in runs]), 1),
                    fmt(_median([s.wall_s for s in runs]), 1),
                ]
            )
            calls_rows.append(
                [
                    *head,
                    *[
                        fmt(
                            _mean([float(s.step_llm_calls.get(step, 0)) for s in runs]),
                            1,
                        )
                        for step in STEPS
                    ],
                ]
            )
    if not seconds_rows:
        return None
    return (
        "Seconds per step, median over runs (a step a run did not reach counts as 0; "
        "tasks that ran in parallel add up, so `steps` can exceed `wall`; `steps` is "
        "the median of each run's sum, `wall` of each run's total):\n\n"
        + table(["arm", "set", "runs", *STEPS, "steps", "wall"], seconds_rows)
        + "\n\nLanguage-model requests per step, mean per run:\n\n"
        + table(["arm", "set", "runs", *STEPS], calls_rows)
    )


def _failures_section(exp: Experiment, cells: Cells) -> str | None:
    rows = []
    for arm in exp.arms:
        for set_spec in exp.sets:
            if not arm.runs_on(set_spec.name):
                continue
            runs = [
                r for r in cells.runs if r.arm == arm.name and r.set == set_spec.name
            ]
            failed = sum(1 for r in runs if r.error)
            outages = sum(1 for r in runs if r.outage)
            if failed or arm.outage == "record":
                rows.append(
                    [
                        arm.name,
                        set_spec.name,
                        str(len(runs)),
                        str(failed),
                        str(outages),
                        str(failed - outages),
                    ]
                )
    if not rows:
        return None
    return (
        table(["arm", "set", "runs", "failed", "search outage", "other errors"], rows)
        + "\n\nA run that met a search outage is recorded as failed with no metric "
        "values: it is not a zero (no recall is claimed or denied) and it drops out "
        "of every paired comparison. Another error of a record arm is handled the "
        "same way; an arm that does not record failures counts an error as a miss "
        "on the outcome metrics."
    )


def _solo_section(exp: Experiment, cells: Cells) -> str | None:
    solo = exp.solo_arms
    if not solo:
        return None
    rows = []
    for arm in solo:
        per_set = []
        pooled: list[float] = []
        for set_spec in exp.sets:
            if not arm.runs_on(set_spec.name):
                continue
            walls = [
                w
                for r in cells.runs
                if r.arm == arm.name and r.set == set_spec.name and not r.error
                if (w := r.metrics.get("wall_s")) is not None
            ]
            pooled += walls
            per_set.append(
                f"{set_spec.name} {fmt(_median(walls), 1)} s ({len(walls)} runs)"
            )
        rows.append(
            [
                arm.name,
                "; ".join(per_set),
                f"{fmt(_median(pooled), 1)} s ({len(pooled)} runs)",
            ]
        )
    text = (
        "These are the only wall-time numbers to use for speed claims: each cell "
        "ran alone (concurrency 1, after every other arm). Median wall time per "
        "run:\n\n" + table(["arm", "per set", "pooled median"], rows)
    )
    steps = _step_tables(exp, cells, solo=True)
    if steps:
        text += "\n\n" + steps
    return text


# -- the report ---------------------------------------------------------------------


def _live_note(exp: Experiment) -> str:
    return (
        "Every arm runs Plan and Search itself, live: any web or source drift "
        "between runs is part of the difference, and a comparison with the "
        "legacy pipeline is end to end."
    )


def build_report(exp: Experiment, records: Sequence[StartRecord | RunRecord]) -> str:
    """The markdown report of *exp* from the registry's *records*."""
    experiment_hash = exp.config_hash()
    starts = [
        r
        for r in records
        if isinstance(r, StartRecord) and r.experiment_hash == experiment_hash
    ]
    runs = [
        r
        for r in records
        if isinstance(r, RunRecord) and r.experiment_hash == experiment_hash
    ]
    cells = Cells(runs)
    lines = [
        f"# Experiment {exp.name}",
        "",
        exp.description,
        "",
        f"Registered {exp.registered}; experiment hash `{experiment_hash[:12]}`; "
        f"{len(runs)} run records, {sum(1 for r in runs if r.error)} with errors "
        "(an errored run counts as a miss on the outcome metrics, never as absent).",
    ]
    used = sorted({s.concurrency for s in starts}) or [1]
    lines += [
        "",
        "Concurrency used: "
        + ", ".join(str(n) for n in used)
        + " cell(s) at a time"
        + (" (across invocations)" if len(used) > 1 else "")
        + ". Wall times are measured under that load.",
        "",
        _live_note(exp),
    ]
    for text in exp.notes:
        lines += ["", text]
    lines += ["", "## Arms", "", _arms_section(exp)]
    for set_spec in exp.sets:
        name = set_spec.name
        lines += ["", f"## Set {name}", ""]
        items = max(
            (len(cells.items(a.name, name)) for a in exp.main_arms_on(name)), default=0
        )
        lines += [f"{items} items, seed {set_spec.seed}.", ""]
        lines += [
            _means_section(exp, cells, name),
            "",
            _paired_section(exp, cells, name),
        ]
    lines += ["", "## Cost", "", _cost_section(exp, cells)]
    steps = _step_tables(exp, cells, solo=False)
    if steps:
        lines += ["", "## Where the time and the requests went", "", steps]
    failures = _failures_section(exp, cells)
    if failures:
        lines += ["", "## Failed runs", "", failures]
    solo = _solo_section(exp, cells)
    if solo:
        lines += ["", "## Clean timing (solo runs)", "", solo]
    lines += [
        "",
        "## Verdicts (pre-registered rules)",
        "",
        _verdict_section(exp, cells),
        "",
    ]
    return "\n".join(lines)


def report_from_registry(exp: Experiment, out_dir: Path) -> str:
    """Read ``experiments.jsonl`` in *out_dir* and build the report."""
    path = registry_path(out_dir)
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist: run the experiment first")
    return build_report(exp, list(read_records(path)))


__all__ = [
    "build_report",
    "canonicalise",
    "item_values",
    "judge_rules",
    "overall",
    "report_from_registry",
    "stable_key",
]
