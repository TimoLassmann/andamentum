"""andamentum.forge — an agentic system that builds agentic systems.

Forge turns the Agent Graph Recipe / agentic dialect on itself: a natural-language
brief becomes a typed, recipe-validated ``SystemSpec`` (Input / Entities / State /
Nodes / Agents), which a deterministic renderer assembles into a runnable,
dialect-conforming package.

The meta-pipeline is itself a dialect-conforming ``pydantic-graph`` system (see
``graph.py``): thin orchestrator steps over engine-free workers, agents as data, a
single operator-trusted branch, bounded fan-out, a typed ``ForgeResult``.

Public surface (``__all__``): ``run_forge(brief, *, model, out=None, stop_after=...)``
designs (and optionally renders, builds and audits) a small function package from a
brief and returns a ``ForgeResult``; ``SystemSpec`` is the validated design; the errors
derive from ``ForgeError``. The workers (``render``, ``compile_spec``, ``verify_package``,
``build_system``, ``audit_system``, ``make_sandbox``, ...) stay importable from this
package but are internals.
"""

# ruff: noqa: F401  (Names imported here but not in __all__ stay importable from the package for older callers.)

from __future__ import annotations

from .audit import audit_system
from .build import build_system
from .compile_spec import compile_spec
from .errors import BriefRefusedError, DesignError, ForgeError
from .graph import ForgeDeps, ForgeState, StopAfter, graph, run_forge
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
    # Entry point and what it takes / returns
    "run_forge",
    "StopAfter",
    "ForgeResult",
    "BuildReport",
    "AuditReport",
    "VerificationReport",
    "CheckResult",
    "Fitness",
    "SystemSpec",
    # Errors
    "ForgeError",
    "BriefRefusedError",
    "DesignError",
    "SandboxUnavailableError",
]
