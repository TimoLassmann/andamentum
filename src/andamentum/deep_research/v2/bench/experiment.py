"""The experiment file: what is compared, on which items, judged by which rule.

An experiment is a YAML file, committed with the code, that fixes before any
run: the sets and their subset sizes and seeds, the arms, how often each arm is
repeated, and the **pre-registered rules** that decide the verdict. A rule
names its metric, the arm under test, the reference arm, the kind of test and
the margin; ``verdicts.py`` applies it to the results and nothing else decides
the outcome, so a threshold cannot be tuned after the numbers are seen (a
tuned threshold is a new experiment with its own file).

The file names data sets, never directories: where the downloaded data and the
snapshots live is given on the command line.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

SetName = Literal["litsearch", "web", "web_planted", "synergy"]
System = Literal["legacy", "v2", "screening"]
Test = Literal["noninferior", "superior", "reduced_by", "at_least", "at_most"]
Aggregate = Literal["mean", "median", "max"]
Outage = Literal["stop", "record"]

# The sets an arm of each system can be run on.
SYSTEM_SETS: dict[str, frozenset[str]] = {
    "legacy": frozenset({"litsearch", "web"}),
    "v2": frozenset({"litsearch", "web", "web_planted"}),
    "screening": frozenset({"synergy"}),
}
DEFAULT_RESAMPLES = 10_000
DEFAULT_BOOTSTRAP_SEED = 0


class SetSpec(BaseModel):
    """One data set and the deterministic subset of it that is used."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: SetName
    n: int | None = Field(
        default=None, ge=1, description="Items drawn (None: the whole set)."
    )
    seed: int = 0
    reviews: list[str] | None = Field(
        default=None, description="synergy only: the review keys."
    )
    records: int | None = Field(
        default=None, ge=1, description="synergy only: records sampled per review."
    )

    @model_validator(mode="after")
    def _synergy_only(self) -> SetSpec:
        if self.name != "synergy" and (self.reviews or self.records):
            raise ValueError(f"{self.name}: reviews and records belong to synergy")
        if self.name == "synergy" and self.n is not None:
            raise ValueError("synergy: give reviews and records, not n")
        return self


class ArmSpec(BaseModel):
    """One system under test: a system, a strategy preset, models, how it is run."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._+-]*$")
    system: System
    preset: str | None = Field(
        default=None, description="v2: the ResearchConfig preset."
    )
    screen: str | None = Field(
        default=None,
        description="screening: '<baseline|generic|criteria>+<decision|llm>'.",
    )
    model: str | None = None
    decision_model: str | None = None
    frozen: bool = Field(
        default=True,
        description="v2 only: take Plan and Search from the snapshot (False: live).",
    )
    repeats: int | None = Field(default=None, ge=1)
    sets: list[SetName] | None = Field(
        default=None, description="Restrict the arm to these sets."
    )
    solo: int | None = Field(
        default=None,
        ge=1,
        description=(
            "Clean timing: run only the first N items of each set, after every "
            "other arm, one cell at a time (concurrency 1), so the wall time is "
            "not shared with another cell. Answer arms only."
        ),
    )
    outage: Outage = Field(
        default="stop",
        description=(
            "A live cell that meets a search outage: 'stop' ends the run without "
            "recording the cell (a resume runs it again); 'record' records it as "
            "failed, with no metric values (never as zero), and goes on."
        ),
    )

    @model_validator(mode="after")
    def _consistent(self) -> ArmSpec:
        if self.solo is not None and self.system == "screening":
            raise ValueError(f"arm {self.name}: solo timing is for answer arms")
        if self.outage == "record" and not (
            self.system == "legacy" or (self.system == "v2" and not self.frozen)
        ):
            raise ValueError(
                f"arm {self.name}: outage applies to live arms (legacy, or v2 with "
                "frozen: false)"
            )
        if self.system == "v2" and not self.preset:
            raise ValueError(f"arm {self.name}: a v2 arm names its preset")
        if self.system == "screening":
            if not self.screen:
                raise ValueError(f"arm {self.name}: a screening arm names its screen")
            if self.preset:
                raise ValueError(f"arm {self.name}: a screening arm has no preset")
        elif self.screen:
            raise ValueError(f"arm {self.name}: screen belongs to screening arms")
        if self.system != "v2" and "frozen" in self.model_fields_set:
            raise ValueError(f"arm {self.name}: frozen belongs to v2 arms")
        if self.system != "screening" and not self.model:
            raise ValueError(f"arm {self.name}: {self.system} needs a model")
        allowed = SYSTEM_SETS[self.system]
        if self.sets is not None:
            bad = [s for s in self.sets if s not in allowed]
            if bad:
                raise ValueError(
                    f"arm {self.name}: a {self.system} arm cannot run on {bad}"
                )
        return self

    @property
    def is_frozen(self) -> bool:
        return self.system == "v2" and self.frozen

    @property
    def is_solo(self) -> bool:
        return self.solo is not None

    def runs_on(self, set_name: str) -> bool:
        if self.sets is not None:
            return set_name in self.sets
        return set_name in SYSTEM_SETS[self.system]


class Rule(BaseModel):
    """One pre-registered acceptance rule.

    ``noninferior``: *arm* is accepted when its paired difference from
    *reference* (arm minus reference, per item) is above ``-margin`` with the
    whole interval; ``superior``: above ``margin``. Both read the sign through
    ``higher_is_better``. ``at_least`` / ``at_most``: the aggregate of *arm*'s
    own item values against ``threshold``; no reference. The verdict is
    ``accepted`` when the whole interval is on the right side, ``not accepted``
    when the whole interval is on the wrong side, else ``inconclusive``.
    ``reduced_by``: per item, ``1 - arm / reference`` (items whose reference is
    0 or that lack a value drop out), aggregated (``aggregate``, median by
    convention) and compared with ``threshold`` as a share: accepted when the
    whole interval is at or above it. ``aggregate: max`` gives the largest item
    value with no interval (``at_most: 0`` then reads "no item at all").
    ``also_sets`` pools more sets into the rule's values (items are named
    ``<set>/<item>``). ``group`` names the rules that stand or fall together
    (``Experiment.decisions`` read groups, not single rules).
    ``gate=False`` marks a rule that is reported, not a condition.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    question: str
    set: SetName
    metric: str = Field(
        description="An item metric, or 'stability.<name>' for a repeat measure."
    )
    arm: str
    reference: str | None = None
    test: Test
    margin: float = Field(default=0.0, ge=0.0)
    threshold: float | None = None
    aggregate: Aggregate = "mean"
    higher_is_better: bool = True
    gate: bool = True
    also_sets: list[SetName] = Field(default_factory=list)
    group: str | None = None

    @model_validator(mode="after")
    def _consistent(self) -> Rule:
        shift = self.test in ("noninferior", "superior")
        paired = shift or self.test == "reduced_by"
        if paired and not self.reference:
            raise ValueError(f"rule {self.id}: {self.test} needs a reference arm")
        if self.test == "reduced_by":
            if self.threshold is None or not 0.0 < self.threshold <= 1.0:
                raise ValueError(
                    f"rule {self.id}: reduced_by needs a threshold share in (0, 1]"
                )
            if self.margin:
                raise ValueError(f"rule {self.id}: reduced_by takes a threshold only")
            if self.aggregate == "max":
                raise ValueError(
                    f"rule {self.id}: reduced_by aggregates by mean or median"
                )
        elif not paired and (self.reference or self.threshold is None):
            raise ValueError(
                f"rule {self.id}: {self.test} needs a threshold, no reference"
            )
        if shift and self.threshold is not None:
            raise ValueError(
                f"rule {self.id}: {self.test} takes a margin, not a threshold"
            )
        if shift and self.aggregate != "mean":
            raise ValueError(f"rule {self.id}: paired rules compare means")
        if self.set in self.also_sets or len(set(self.also_sets)) != len(
            self.also_sets
        ):
            raise ValueError(f"rule {self.id}: also_sets repeats a set")
        return self

    @property
    def sets(self) -> list[str]:
        """Every set the rule reads, its own first."""
        return [self.set, *self.also_sets]


class JudgeSpec(BaseModel):
    """Who judges whether a claim is supported by its evidence.

    Part of the pre-registration: the trust metrics (claim precision, unsourced
    share) mean different things under different judges, so the judge is named
    here, hashed with the file and recorded on every run. Exactly one of
    ``model`` (an LLM) and ``decision_model`` (a local decision model).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    model: str | None = None
    decision_model: str | None = None

    @model_validator(mode="after")
    def _one(self) -> JudgeSpec:
        if (self.model is None) == (self.decision_model is None):
            raise ValueError("a judge names exactly one of model and decision_model")
        return self

    @property
    def label(self) -> str:
        return self.decision_model or self.model or ""


class ExperimentDecision(BaseModel):
    """What the registered rules mean for the project, in words fixed before the
    run: ``if_accepted`` is the verdict when every rule group in ``requires`` is
    accepted, ``otherwise`` when any is not (rejected or inconclusive; the
    report states each group's outcome beside it)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    requires: list[str] = Field(min_length=1)
    if_accepted: str
    otherwise: str


# Item metrics that exist only when claims are judged.
JUDGED_METRICS = frozenset({"claim_precision", "unsourced_share", "n_unsourced"})


class Experiment(BaseModel):
    """The whole file."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
    description: str = ""
    registered: str = Field(
        description="Date the rules were fixed (YYYY-MM-DD), before any run."
    )
    repeats: int = Field(default=1, ge=1, description="Default repeats per item.")
    judge: JudgeSpec | None = Field(
        default=None, description="Judges claim support for the trust metrics."
    )
    sets: list[SetSpec] = Field(min_length=1)
    arms: list[ArmSpec] = Field(min_length=1)
    baselines: dict[str, str] = Field(
        description="Per set, the arm every other arm is compared with in the tables."
    )
    rules: list[Rule] = Field(default_factory=list)
    decisions: list[ExperimentDecision] = Field(default_factory=list)
    report_metrics: dict[str, list[str]] = Field(
        default_factory=dict,
        description="Metrics of the paired tables per set (default: the set's own list).",
    )
    notes: list[str] = Field(default_factory=list)
    resamples: int = Field(default=DEFAULT_RESAMPLES, ge=1)
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED

    @model_validator(mode="after")
    def _consistent(self) -> Experiment:
        arm_names = [a.name for a in self.arms]
        if len(set(arm_names)) != len(arm_names):
            raise ValueError("arm names must be unique")
        set_names = [s.name for s in self.sets]
        if len(set(set_names)) != len(set_names):
            raise ValueError("a set may be listed once")
        known = set(arm_names)
        for set_name in self.report_metrics:
            if set_name not in set_names:
                raise ValueError(
                    f"report_metrics for {set_name}, which is not a set here"
                )
        for set_name, arm in self.baselines.items():
            if set_name not in set_names:
                raise ValueError(f"baseline for {set_name}, which is not a set here")
            if arm not in known:
                raise ValueError(f"baseline {arm!r} is not an arm")
            if not self._arm(arm).runs_on(set_name):
                raise ValueError(f"baseline {arm!r} does not run on {set_name}")
        for rule in self.rules:
            for name in (rule.arm, rule.reference):
                if name is not None and name not in known:
                    raise ValueError(f"rule {rule.id}: {name!r} is not an arm")
            if rule.set not in set_names:
                raise ValueError(f"rule {rule.id}: {rule.set} is not a set here")
            for set_name in rule.sets:
                if set_name not in set_names:
                    raise ValueError(f"rule {rule.id}: {set_name} is not a set here")
                for name in (rule.arm, rule.reference):
                    if name is not None and not self._arm(name).runs_on(set_name):
                        raise ValueError(
                            f"rule {rule.id}: {name!r} does not run on {set_name}"
                        )
        ids = [r.id for r in self.rules]
        if len(set(ids)) != len(ids):
            raise ValueError("rule ids must be unique")
        groups = {r.group for r in self.rules if r.group}
        decision_ids = [d.id for d in self.decisions]
        if len(set(decision_ids)) != len(decision_ids):
            raise ValueError("decision ids must be unique")
        for decision in self.decisions:
            unknown = [g for g in decision.requires if g not in groups]
            if unknown:
                raise ValueError(
                    f"decision {decision.id}: no rule is in group {', '.join(unknown)}"
                )
        if self.judge is None and self.needs_judge():
            raise ValueError(
                "a rule or report metric reads claim support, so the file must "
                "name its judge (judge: {decision_model: ...} or {model: ...})"
            )
        for arm in self.arms:
            if arm.system == "v2" and not any(arm.runs_on(s) for s in set_names):
                raise ValueError(f"arm {arm.name} runs on none of the sets")
        return self

    def needs_judge(self) -> bool:
        """Does any rule or report metric need claims judged?"""
        wanted = {r.metric for r in self.rules}
        for metrics in self.report_metrics.values():
            wanted |= set(metrics)
        return bool(wanted & JUDGED_METRICS)

    def _arm(self, name: str) -> ArmSpec:
        return next(a for a in self.arms if a.name == name)

    def arm(self, name: str) -> ArmSpec:
        return self._arm(name)

    def arms_on(self, set_name: str) -> list[ArmSpec]:
        return [a for a in self.arms if a.runs_on(set_name)]

    def main_arms_on(self, set_name: str) -> list[ArmSpec]:
        """The arms on a set that are not clean-timing (solo) re-runs."""
        return [a for a in self.arms_on(set_name) if not a.is_solo]

    @property
    def solo_arms(self) -> list[ArmSpec]:
        return [a for a in self.arms if a.is_solo]

    def rule_groups(self) -> dict[str, list[Rule]]:
        """The rules by group, in file order."""
        groups: dict[str, list[Rule]] = {}
        for rule in self.rules:
            if rule.group:
                groups.setdefault(rule.group, []).append(rule)
        return groups

    def repeats_of(self, arm: ArmSpec) -> int:
        return arm.repeats if arm.repeats is not None else self.repeats

    def set(self, name: str) -> SetSpec:
        return next(s for s in self.sets if s.name == name)

    def config_hash(self) -> str:
        """sha256 of the canonical JSON of the file: the experiment's identity."""
        return canonical_hash(self.model_dump(mode="json"))


def canonical_hash(data: object) -> str:
    """sha256 of *data* as sorted-key compact JSON."""
    text = json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_experiment(path: Path) -> Experiment:
    """Read and validate an experiment file; a bad file raises ``ValueError``."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: an experiment file is a mapping")
    return Experiment.model_validate(raw)


__all__ = [
    "ArmSpec",
    "Experiment",
    "ExperimentDecision",
    "JUDGED_METRICS",
    "JudgeSpec",
    "Rule",
    "SetSpec",
    "canonical_hash",
    "load_experiment",
]
