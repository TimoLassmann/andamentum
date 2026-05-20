"""Strunk lens — Elements of Style review as a pydantic-graph sub-graph.

Public surface:

    from andamentum.whetstone.lenses.strunk import run_strunk_lens

    findings = await run_strunk_lens(section, deps=StrunkLensDeps(...))

Where ``section`` is a ``whetstone.structural.types.SectionRef`` and the
return value is a ``list[whetstone.schemas.Finding]`` ready to merge
into the main review pool.

Phase A implements three rules:

* **R2** — Series comma (deterministic regex, scans the whole section).
* **R11** — Use the active voice (one agent call per section returning
  a list of violations).
* **R13** — Omit needless words (one agent call per section returning
  a list of violations).

The sub-graph topology, in declaration order:

    DeterministicScreen → R11ActiveVoice → R13OmitNeedlessWords
                       → ResolveDemands → Aggregate → End[list[Finding]]

Each node declares its ``NodeKind`` so the deterministic / agent /
control split is visible at grep time and asserted in tests.
"""

from __future__ import annotations

from .api import run_strunk_lens
from .kinds import NodeKind
from .models import (
    ActiveVoiceReport,
    ActiveVoiceViolation,
    OmitNeedlessWordsReport,
    OmitNeedlessWordsViolation,
    StrunkDemand,
    StrunkFinding,
)
from .state import (
    AgentExecutor,
    StrunkLensDeps,
    StrunkLensState,
)

__all__ = [
    "run_strunk_lens",
    "NodeKind",
    "ActiveVoiceReport",
    "ActiveVoiceViolation",
    "OmitNeedlessWordsReport",
    "OmitNeedlessWordsViolation",
    "StrunkDemand",
    "StrunkFinding",
    "AgentExecutor",
    "StrunkLensDeps",
    "StrunkLensState",
]
