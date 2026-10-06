"""The experiment report: tables, paired differences, stability, verdicts.

``build_report`` reads the registry of one experiment and writes markdown. It
computes, and decides nothing the experiment file did not register:

- per-arm means of each item metric (an item's value is the mean over its
  repeats);
- paired differences against the set's baseline arm, 95% bootstrap intervals
  over items (``stats.paired_bootstrap_ci``);
- repeat stability per arm (``stability.py``): kept- and cited-set Jaccard,
  rank correlation, findings reproduced, answer agreement, spread, and for live
  arms the overlap of the plan;
- the planted-source outcomes per arm;
- cost, and where a run's time and model requests went, step by step
  (``RunStats.step_*`` of every v2 run);
- how many runs failed, and how many of those met a search outage;
- the clean-timing (solo) runs, reported apart from everything else;
- the verdict of every pre-registered rule, the outcome of each rule group and
  the pre-registered wording of what the groups decide.

Work ids are random per store, so before any two runs are compared the ids are
replaced by the identity key of the work (``stable_key``): the same paper found
by two runs is the same key.
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
from .stability import (
    answer_agreement,
    findings_reproduced,
    mean_pairwise,
    metric_spread,
    rank_kendall_tau,
    subquestion_overlap,
)
from .verdicts import (
    Interval,
    Outcome,
    Verdict,
    judge_rule,
    paired_interval,
)

DEFAULT_REPORT_METRICS: dict[str, list[str]] = {
    "litsearch": [
        "recall_at_5",
        "recall_at_10",
        "recall_at_20",
        "recall_at_50",
        "found_any",
        "kept_precision",
        "kept_n",
    ],
    "web": ["correct", "claim_precision", "unsourced_share", "link_integrity"],
    "web_planted": [
        "correct",
        "adopted_false_claim",
        "stated_conflict",
        "ignored_correct",
        "good_planted_outcome",
    ],
    "synergy": ["kept_recall", "kept_precision", "wss_at_95", "work_saved_kept"],
}
COST_METRICS = (
    "wall_s",
    "llm_calls",
    "decision_calls",
    "input_tokens",
    "output_tokens",
)
SPREAD_METRICS: dict[str, list[str]] = {
    "litsearch": ["recall_at_20", "kept_n"],
    "web": ["correct", "claim_precision"],
    "web_planted": ["good_planted_outcome"],
    "synergy": ["kept_recall"],
}
STABILITY_PREFIX = "stability."
PLANTED_OUTCOMES = (
    "adopted_false_claim",
    "stated_conflict",
    "ignored_correct",
    "ignored_incorrect",
)
NOTHING = "n/a"


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
    if metric.startswith(STABILITY_PREFIX):
        return stability_values(cells, arm, set_name)[
            metric.removeprefix(STABILITY_PREFIX)
        ]
    return {
        item: _mean([r.metrics.get(metric) for r in cells.repeats(arm, set_name, item)])
        for item in cells.items(arm, set_name)
    }


# -- stability ----------------------------------------------------------------------

STABILITY_NAMES = (
    "kept_jaccard",
    "cited_jaccard",
    "rank_tau",
    "findings_reproduced",
    "answer_agreement",
    "subquestion_overlap",
)


def _kept_keys(record: RunRecord) -> list[str]:
    if record.output is None:
        return []
    return [stable_key(w) for w in record.output.works]


def _cited_keys(record: RunRecord) -> list[str]:
    output = record.output
    if output is None:
        return []
    if output.result is not None:
        return canonicalise(output.result).cited_work_ids()
    return list(output.urls)


def _plan(record: RunRecord) -> list[str]:
    if record.output is None or record.output.result is None:
        return []
    return [s.text for s in record.output.result.subquestions]


def item_stability(
    records: Sequence[RunRecord], *, live: bool
) -> dict[str, float | None]:
    """The repeat measures of one item (a frozen or live arm's runs of it)."""
    ok = [r for r in records if r.output is not None and not r.error]
    out: dict[str, float | None] = dict.fromkeys(STABILITY_NAMES)
    if len(ok) < 2:
        return out
    out["kept_jaccard"] = mean_pairwise([set(_kept_keys(r)) for r in ok])
    out["cited_jaccard"] = mean_pairwise([set(_cited_keys(r)) for r in ok])
    taus = [
        t
        for i, a in enumerate(ok)
        for b in ok[i + 1 :]
        if (t := rank_kendall_tau(_kept_keys(a), _kept_keys(b))) is not None
    ]
    out["rank_tau"] = _mean(taus)
    results = [
        canonicalise(r.output.result)
        for r in ok
        if r.output and r.output.result is not None
    ]
    if len(results) == len(ok):
        reproduced = [
            v
            for i, a in enumerate(results)
            for j, b in enumerate(results)
            if i != j and (v := findings_reproduced(a.findings, b.findings)) is not None
        ]
        out["findings_reproduced"] = _mean(reproduced)
    verdicts = [r.metrics.get("correct") for r in ok]
    if all(v is not None for v in verdicts):
        out["answer_agreement"] = answer_agreement([bool(v) for v in verdicts])
    if live:
        plans = [_plan(r) for r in ok]
        if all(plans):
            pair = [
                subquestion_overlap(a, b)
                for i, a in enumerate(plans)
                for b in plans[i + 1 :]
            ]
            out["subquestion_overlap"] = _mean(pair)
    return out


def stability_values(
    cells: Cells, arm: str, set_name: str
) -> dict[str, dict[str, float | None]]:
    """For each stability measure, its value per item."""
    live = not any(r.frozen for r in cells.runs if r.arm == arm)
    per_item = {
        item: item_stability(cells.repeats(arm, set_name, item), live=live)
        for item in cells.items(arm, set_name)
    }
    return {name: {i: v[name] for i, v in per_item.items()} for name in STABILITY_NAMES}


# -- formatting ---------------------------------------------------------------------


def fmt(value: float | None, digits: int = 3) -> str:
    return NOTHING if value is None else f"{value:.{digits}f}"


def fmt_interval(interval: Interval | None) -> str:
    if interval is None:
        return NOTHING
    return f"{interval.estimate:+.3f} [{interval.low:+.3f}, {interval.high:+.3f}]"


def table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    lines = [
        "| " + " | ".join(header) + " |",
        "|" + "|".join("---" for _ in header) + "|",
    ]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


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


def _arms_section(exp: Experiment, start: StartRecord | None) -> str:
    recorded = {a.name: a for a in start.arms} if start else {}
    rows = []
    for arm in exp.arms:
        info = recorded.get(arm.name)
        rows.append(
            [
                arm.name,
                arm.system,
                arm.preset or arm.screen or "-",
                (info.config_hash or "-")[:12] if info else "-",
                arm.model or "-",
                (
                    f"{info.decision_model or '-'} ({(info.decision_digest or 'no digest')[:12]})"
                    if info
                    else arm.decision_model or "-"
                ),
                ("frozen" if arm.is_frozen else "live")
                + (f", solo: first {arm.solo} items of each set" if arm.is_solo else "")
                + (", failed outage cells recorded" if arm.outage == "record" else ""),
                str(exp.repeats_of(arm)),
            ]
        )
    return table(
        [
            "arm",
            "system",
            "preset / screen",
            "config hash",
            "model",
            "decision model (digest)",
            "inputs",
            "repeats",
        ],
        rows,
    )


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


def _stability_section(exp: Experiment, cells: Cells, set_name: str) -> str | None:
    rows = []
    spreads = []
    for arm in exp.main_arms_on(set_name):
        if exp.repeats_of(arm) < 2 or arm.system == "screening":
            continue
        stability = stability_values(cells, arm.name, set_name)
        rows.append(
            [
                arm.name,
                "frozen" if arm.is_frozen else "live",
                str(exp.repeats_of(arm)),
                *[fmt(_mean(list(stability[n].values()))) for n in STABILITY_NAMES],
            ]
        )
        for metric in SPREAD_METRICS.get(set_name, []):
            sds, ranges = [], []
            for item in cells.items(arm.name, set_name):
                values = [
                    v
                    for r in cells.repeats(arm.name, set_name, item)
                    if (v := r.metrics.get(metric)) is not None
                ]
                spread = metric_spread(values)
                if spread is not None and spread.n > 1:
                    sds.append(spread.sd)
                    ranges.append(spread.max - spread.min)
            spreads.append(
                [arm.name, metric, fmt(_mean(sds)), fmt(_mean(ranges)), str(len(sds))]
            )
    if not rows:
        return None
    text = (
        table(["arm", "inputs", "repeats", *STABILITY_NAMES], rows)
        + "\n\nValues are means over items of the pairwise agreement between repeats "
        "(1.0 = identical). `n/a`: not applicable or fewer than two usable repeats "
        "(`subquestion_overlap` only for live arms, where the plan can differ)."
    )
    if spreads:
        text += (
            "\n\nSpread of a metric across repeats, averaged over items:\n\n"
            + table(["arm", "metric", "mean sd", "mean max-min", "items"], spreads)
        )
    return text


def _planted_section(exp: Experiment, cells: Cells) -> str | None:
    if not any(s.name == "web_planted" for s in exp.sets):
        return None
    rows = []
    for arm in exp.main_arms_on("web_planted"):
        runs = [r for r in cells.runs if r.arm == arm.name and r.set == "web_planted"]
        counts = {
            o: sum(1 for r in runs if r.planted_outcome == o) for o in PLANTED_OUTCOMES
        }
        errors = sum(1 for r in runs if r.error)
        rows.append(
            [
                arm.name,
                str(len(runs)),
                *[str(counts[o]) for o in PLANTED_OUTCOMES],
                str(errors),
            ]
        )
    return (
        table(["arm", "runs", *PLANTED_OUTCOMES, "errors"], rows)
        + "\n\nGood: `stated_conflict` and `ignored_correct`. Bad: `adopted_false_claim` "
        "and `ignored_incorrect`. A run counts once per repeat."
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
        + "\n\nFrozen arms start from a snapshot and run offline: their wall time "
        "excludes Plan, Search and every page fetch. Only live arms measure "
        "end-to-end time. The legacy pipeline counts no tokens or calls (n/a)."
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
    if not rule.gate or verdict.interval is None:
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
    verify_rows: list[list[str]] = []
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
                        f"{_mean([float(s.step_llm_calls.get(step, 0)) for s in runs]):.1f}"
                        f" / {_mean([float(s.step_decision_calls.get(step, 0)) for s in runs]):.1f}"
                        for step in STEPS
                    ],
                ]
            )
            by_code = sum(s.verify_by_code for s in runs)
            by_model = sum(s.verify_by_model for s in runs)
            states: dict[str, int] = {}
            for s in runs:
                key = s.decision_cache.split(":")[0]
                states[key] = states.get(key, 0) + 1
            verify_rows.append(
                [
                    *head,
                    fmt(_mean([float(s.verify_by_code) for s in runs]), 1),
                    fmt(_mean([float(s.verify_by_model) for s in runs]), 1),
                    fmt(by_code / (by_code + by_model), 2)
                    if by_code + by_model
                    else NOTHING,
                    fmt(_mean([float(s.decision_cache_hits) for s in runs]), 1),
                    ", ".join(f"{k} {n}" for k, n in sorted(states.items())),
                ]
            )
    if not seconds_rows:
        return None
    return (
        "Seconds per step, median over runs (a step a run did not reach counts as 0; "
        "`steps` is the median of each run's sum, `wall` of each run's total):\n\n"
        + table(
            ["arm", "set", "runs", *STEPS, "steps", "wall"],
            seconds_rows,
        )
        + "\n\nModel requests per step, mean per run, `language model / decision "
        "model` (decision requests served from the cache are not requests):\n\n"
        + table(["arm", "set", "runs", *STEPS], calls_rows)
        + "\n\nClaim-support checks settled in code and by the model (means per "
        "run, `share` = code / all), decision answers served from the cache, and "
        "the cache state of the runs:\n\n"
        + table(
            [
                "arm",
                "set",
                "runs",
                "by code",
                "by model",
                "share by code",
                "cache hits",
                "decision cache",
            ],
            verify_rows,
        )
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
        "of every paired comparison. Another error counts as a miss on the outcome "
        "metrics."
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
                "frozen" if arm.is_frozen else "live",
                "; ".join(per_set),
                f"{fmt(_median(pooled), 1)} s ({len(pooled)} runs)",
            ]
        )
    text = (
        "These are the only wall-time numbers to use for speed claims: each cell "
        "ran alone (concurrency 1, after every other arm). Median wall time per "
        "run:\n\n" + table(["arm", "inputs", "per set", "pooled median"], rows)
    )
    if any(a.is_frozen for a in solo):
        text += (
            "\n\nA frozen run takes Plan and Search from the snapshot and fetches "
            "nothing, so its wall time excludes the planning call, the searches and "
            "the page fetches; it is the cost of everything from screening on. It "
            "is not an end-to-end time."
        )
    steps = _step_tables(exp, cells, solo=True)
    if steps:
        text += "\n\n" + steps
    return text


# -- the report ---------------------------------------------------------------------


def _live_note(exp: Experiment) -> str | None:
    live = [
        a.name
        for a in exp.arms
        if a.system == "legacy" or (a.system == "v2" and not a.frozen)
    ]
    if not live:
        return None
    return (
        "Live arms ("
        + ", ".join(live)
        + ") run Plan and Search themselves: the legacy "
        "pipeline cannot take frozen inputs, so a comparison with it is end to end, "
        "and any web or source drift between runs is part of the difference. "
        "Comparisons among frozen arms share identical inputs."
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
        + " cell(s) of one arm at a time"
        + (" (across invocations)" if len(used) > 1 else "")
        + "; decision-model requests are always one at a time. Wall times are "
        "measured under that load.",
    ]
    note = _live_note(exp)
    if note:
        lines += ["", note]
    for text in exp.notes:
        lines += ["", text]
    lines += ["", "## Arms", "", _arms_section(exp, starts[-1] if starts else None)]
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
        stability = _stability_section(exp, cells, name)
        if stability:
            lines += ["", f"### Repeat stability ({name})", "", stability]
    planted = _planted_section(exp, cells)
    if planted:
        lines += ["", "## Planted false source", "", planted]
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
