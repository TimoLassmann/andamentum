"""State and deps for the Strunk sub-graph.

Mirrors the deep_research / epistemic convention: a single mutable
state dataclass passed across nodes, plus an immutable deps dataclass
for per-run configuration and injected services. The ``AgentExecutor``
protocol is the seam tests use to inject a stub in place of a real
``AgentRunner`` — production code passes the real runner, tests pass
a fake that returns canned reports.

The state carries the input section and the running list of findings
and demands. There is no sentence- or paragraph-level structure: each
rule node sees the whole section's text and returns its full list of
violations in one LLM call (anchoring uses the chunker's verbatim
matcher to recover char offsets per violation).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from ...structural.types import SectionRef


# ── Agent executor protocol (the test seam) ─────────────────────────────


class AgentExecutor(Protocol):
    """Minimal interface the Strunk lens needs from an agent runner.

    Matches ``andamentum.core.agents.AgentRunner.run``: pass an
    ``AgentDefinition`` and keyword args that become the user message.
    Returns the structured output instance.
    """

    async def run(self, defn: Any, /, **kwargs: Any) -> Any:  # noqa: D401
        ...


# ── Sub-graph state and deps ────────────────────────────────────────────


@dataclass
class StrunkLensState:
    """Mutable state for one run of the Strunk sub-graph on one section."""

    section: SectionRef
    # Raw rule-level findings; Aggregate converts to whetstone.Finding.
    findings: list[Any] = field(default_factory=list)  # list[StrunkFinding]
    demands: list[Any] = field(default_factory=list)   # list[StrunkDemand]


@dataclass
class StrunkLensDeps:
    """Per-run config + the agent executor.

    ``executor`` is None when no LLM is required (e.g. running only
    deterministic nodes for testing). Agent nodes must guard against
    a missing executor explicitly rather than crashing in the runner.
    """

    executor: AgentExecutor | None = None
    model_default: str = "ollama:gemma3:4b-it-q4_K_M"
    model_for_rule: dict[int, str] = field(default_factory=dict)
    enable_demands: bool = False             # Phase 4 flips this on

    def model_for(self, rule_number: int) -> str:
        """Resolve the model for a given rule (default unless overridden)."""
        return self.model_for_rule.get(rule_number, self.model_default)
