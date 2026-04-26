"""Pydantic-free dataclasses for the chunker benchmark."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class TruthUnit:
    """One ground-truth unit, anchor-based (as written by humans)."""

    title: str
    start_anchor: str
    end_anchor: str


@dataclass
class ResolvedTruthUnit:
    """One ground-truth unit after anchor resolution to char offsets."""

    title: str
    start_offset: int  # inclusive
    end_offset: int  # exclusive

    @property
    def length(self) -> int:
        return self.end_offset - self.start_offset


@dataclass
class ResolvedTruth:
    """Ground truth after anchor resolution."""

    convention: str
    units: list[ResolvedTruthUnit]


@dataclass
class BenchmarkCase:
    """A single benchmark case: source text + resolved ground truth + thresholds."""

    name: str
    source: str
    domain: str  # academic | web | code | transcript | general
    expected_f1_floor: float
    boundary_tolerance_chars: int
    truth: ResolvedTruth


@dataclass
class Metrics:
    """Per-case metrics produced by the runner."""

    boundary_f1: float
    boundary_precision: float
    boundary_recall: float
    coverage: float
    gap_fraction: float
    granularity_ratio: float  # predicted_count / truth_count
    unit_count_predicted: int
    unit_count_truth: int
    fragmentation_rate: float  # fraction of units with complete=False
    anchor_method_exact: int
    anchor_method_normalised: int
    anchor_method_fuzzy: int
    wall_clock_seconds: float
    model_calls: int


@dataclass
class CaseRun:
    """Outcome of running one case against a model."""

    case: BenchmarkCase
    metrics: Optional[Metrics]
    model: str
    error: Optional[str]  # str of the ChunkingFailedError, if any

    @property
    def passed_floor(self) -> bool:
        if self.error is not None or self.metrics is None:
            return False
        return self.metrics.boundary_f1 >= self.case.expected_f1_floor
