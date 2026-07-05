"""andamentum.forge — an agentic system that builds agentic systems.

Forge turns the Agent Graph Recipe / agentic dialect on itself: a natural-language
brief becomes a typed, recipe-validated ``SystemSpec`` (Input / Entities / State /
Nodes / Agents), which a deterministic renderer assembles into a runnable,
dialect-conforming package.

The meta-pipeline is itself a dialect-conforming ``pydantic-graph`` system (see
``graph.py``): thin orchestrator steps over engine-free workers, agents as data, a
single operator-trusted branch, bounded fan-out, a typed ``ForgeResult``.

Public surface:

- ``run_forge(brief, *, model, dest=None, ...)`` — design (and optionally render +
  verify) a system from a brief; returns a ``ForgeResult``.
- ``render(spec, dest)`` — deterministic ``SystemSpec`` → package (no LLM).
- ``compile_spec(plan)`` — deterministic design board → validated ``SystemSpec``.
- ``verify_package(spec, dest)`` — deterministic checks over a rendered package.
- ``SystemSpec`` — the recipe made executable (validating it is checking the recipe).
"""

from __future__ import annotations

from .audit import audit_system
from .build import build_system
from .compile_spec import compile_spec
from .graph import ForgeDeps, ForgeState, graph, run_forge
from .render import render
from .sandbox import SandboxPort, SandboxUnavailableError, make_sandbox
from .schemas import (
    AuditIssue,
    AuditReport,
    BuildReport,
    CheckResult,
    CriticVerdict,
    DesignPlan,
    DesignReport,
    Fitness,
    ForgeResult,
    NodeDraft,
    NodeFinding,
    RequirementsVerdict,
    VerificationReport,
)
from .spec import SystemSpec
from .verify import verify_package

__all__ = [
    "run_forge",
    "render",
    "compile_spec",
    "build_system",
    "audit_system",
    "verify_package",
    "make_sandbox",
    "SandboxPort",
    "SandboxUnavailableError",
    "SystemSpec",
    "ForgeResult",
    "BuildReport",
    "AuditReport",
    "VerificationReport",
    "CheckResult",
    "DesignPlan",
    "NodeDraft",
    "ForgeDeps",
    "ForgeState",
    "graph",
    # Types reachable from a returned ForgeResult — exported so a caller can annotate /
    # isinstance-check the result tree without reaching into andamentum.forge.schemas.
    "AuditIssue",
    "CriticVerdict",
    "NodeFinding",
    "Fitness",
    "DesignReport",
    "RequirementsVerdict",
]
