"""The experiment registry: ``experiments.jsonl``, one JSON object per line.

Append-only. A ``start`` line is written each time ``experiment run`` begins
(the experiment's name and hash, the arms with their config hashes and model
digests); a ``run`` line is written after every (arm, set, item, repeat) with
its metrics and its output, so a run that dies keeps what it finished and the
next invocation resumes after the last finished cell. A registry that holds
runs of a different version of the experiment file is refused: the file is
the pre-registration, and results of two versions must not be mixed.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, Field, TypeAdapter

from .arms import ArmOutput
from .metrics import ScreeningScore

REGISTRY_FILE = "experiments.jsonl"


class ArmRecord(BaseModel):
    """One arm as it ran: what identifies the system under test."""

    name: str
    system: str
    preset: str | None = None
    screen: str | None = None
    config_hash: str | None = Field(
        default=None, description="sha256 of the ResearchConfig (v2 arms)."
    )
    model: str | None = None
    decision_model: str | None = None
    decision_digest: str | None = None
    embedding_model: str | None = Field(
        default=None, description="v2 arms that rank by dense similarity."
    )
    frozen: bool = False
    repeats: int = 1
    solo: int | None = None
    outage: str = "stop"
    inputs: dict[str, str] = Field(
        default_factory=dict, description="Per set, the inputs profile it ran on."
    )


class StartRecord(BaseModel):
    kind: Literal["start"] = "start"
    experiment: str
    experiment_hash: str
    started_at: float
    arms: list[ArmRecord]
    judge: str | None = Field(default=None, description="Who judged claim support.")
    judge_digest: str | None = None
    notes: list[str] = Field(default_factory=list)
    concurrency: int = Field(
        default=1, ge=1, description="Cells of one arm in flight at once."
    )
    keyless: bool = Field(
        default=False, description="No API key was read or used (the file says so)."
    )
    inputs: dict[str, str] = Field(
        default_factory=dict,
        description="Each inputs profile's name and canonical-JSON hash.",
    )


class RunRecord(BaseModel):
    kind: Literal["run"] = "run"
    experiment: str
    experiment_hash: str
    arm: str
    set: str
    item_id: str
    repeat: int = Field(ge=1)
    started_at: float
    config_hash: str | None = None
    model: str | None = None
    decision_model: str | None = None
    decision_digest: str | None = None
    judge: str | None = None
    judge_digest: str | None = None
    frozen: bool = False
    metrics: dict[str, float | None] = Field(default_factory=dict)
    planted_outcome: str | None = None
    output: ArmOutput | None = None
    screening: ScreeningScore | None = None
    error: str | None = None
    outage: bool = Field(
        default=False,
        description=(
            "The cell met a search outage and was recorded as failed (the arm's "
            "'record' outage mode): its metrics are absent, not zero."
        ),
    )

    @property
    def cell(self) -> tuple[str, str, str, int]:
        return (self.arm, self.set, self.item_id, self.repeat)


Record = Annotated[StartRecord | RunRecord, Field(discriminator="kind")]
_RECORD = TypeAdapter(Record)


class RegistryMismatch(RuntimeError):
    """The registry holds runs of another version of the experiment file."""


def registry_path(out_dir: Path) -> Path:
    return out_dir / REGISTRY_FILE


def append(path: Path, record: StartRecord | RunRecord) -> None:
    """Append one record and flush it to disk before returning."""
    path.parent.mkdir(parents=True, exist_ok=True)
    line = record.model_dump_json() + "\n"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line)
        handle.flush()
        os.fsync(handle.fileno())


def read_records(path: Path) -> Iterator[StartRecord | RunRecord]:
    """Every record of the registry; a missing file has none, a bad line raises."""
    if not path.exists():
        return
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                yield _RECORD.validate_json(line)
            except ValueError as error:
                raise ValueError(f"{path}:{number}: {error}") from error


def runs_of(path: Path, *, experiment_hash: str | None = None) -> list[RunRecord]:
    """The run records, optionally only those of one version of the experiment."""
    return [
        r
        for r in read_records(path)
        if isinstance(r, RunRecord)
        and (experiment_hash is None or r.experiment_hash == experiment_hash)
    ]


def check_same_experiment(path: Path, *, experiment: str, experiment_hash: str) -> None:
    """Raise unless every record in the registry belongs to this exact file."""
    for record in read_records(path):
        if record.experiment != experiment or record.experiment_hash != experiment_hash:
            raise RegistryMismatch(
                f"{path} holds records of experiment {record.experiment!r} "
                f"({record.experiment_hash[:12]}), not {experiment!r} "
                f"({experiment_hash[:12]}); the file is the pre-registration, so "
                "use a new output directory for a changed file"
            )
