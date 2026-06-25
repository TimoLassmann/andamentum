"""Boundary schemas for the forge meta-pipeline — and the ``ForgeResult``.

Two families live here:

- **Agent outputs** (``ForgeWhy``, ``ForgeAreas``, ``JobList``, ``NodeTyping``) — the
  small, flat, enum-guarded models the design heads fill. Each obeys the dialect's
  "small heads": ≤6 flat fields, closed vocabularies as enums. These are *not* the
  generated system's agents — they are forge's own.
- **The board + the result** (``NodeDraft``, ``DesignPlan``, ``CheckResult``,
  ``VerificationReport``, ``ForgeResult``) — the typed values that ride between steps
  and the final ``End[ForgeResult]`` payload.

Leaf worker file: ``pydantic`` + the sibling ``spec`` enums only; no graph engine.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field

from .spec import NodeControl, NodeKind, SystemSpec

# Tokens a node may name to mean "the graph input" (resolved to the input primary field).
INPUT_TOKENS = frozenset({"input", "brief", "request"})


class DataKind(str, Enum):
    """Whether a datum a node produces is a transient signal or a durable entity.

    A *signal* is a run-scoped value handed toward the next step (lives in State).
    An *entity* persists and is reasoned about across the run (lives in a store).
    """

    SIGNAL = "signal"
    ENTITY = "entity"


# --- design agent outputs (small, flat, enum-guarded) ---------------------------


class ForgeWhy(BaseModel):
    """The understand head: purpose + boundary, in plain language."""

    purpose: str = Field(
        description="One or two plain sentences: what the system is for and the value it delivers"
    )
    boundary_in: str = Field(
        description="What the system takes in — the single natural-language input at the door"
    )
    boundary_out: str = Field(description="What the system produces — its final output")


class ForgeAreas(BaseModel):
    """The frame head: the 2–4 big concerns the system must get right."""

    areas: list[str] = Field(
        description="The distinct big concerns — one or more (usually exactly one for a simple task), each a short phrase"
    )


class JobList(BaseModel):
    """Decompose stage 1: an area's atomic steps as plain sentences (no types yet).

    A list of strings is the one list shape small models handle reliably — the typed
    fields are filled later, one node at a time (``NodeTyping``).
    """

    jobs: list[str] = Field(
        description="The atomic steps for this area, each a short job sentence (12 words or fewer)"
    )


class NodeTyping(BaseModel):
    """Decompose stage 2: the typed fields for ONE already-named node.

    One object, never an array — the model fills this for a single job it is handed,
    while seeing the whole plan as context.
    """

    kind: NodeKind = Field(
        default=NodeKind.SPINE,
        description="spine = code-computable (math, lookup, regex, API call, routing); head = LLM judgment over text",
    )
    consumes: list[str] = Field(
        default_factory=list,
        description="Exact data names this node reads — reuse a name from the plan, or 'input'",
    )
    produces: list[str] = Field(
        default_factory=list,
        description="Exactly one NEW data name this node writes (a short noun phrase)",
    )
    produces_kind: DataKind = Field(
        default=DataKind.SIGNAL,
        description="signal = run-scoped value handed to the next step; entity = a stored database record",
    )
    control: NodeControl = Field(
        default=NodeControl.NONE,
        description="none | checkpoint (loop control) | decision (branch on output) | consequential (human approval)",
    )
    network: bool = Field(
        default=False,
        description="True if the node reaches an external service over the internet",
    )


# --- the board (internal boundary value) ----------------------------------------


class NodeDraft(BaseModel):
    """One typed leaf job on the design board — the compile worker's input unit."""

    id: str
    area: str = ""
    job: str = ""
    kind: NodeKind = NodeKind.SPINE
    consumes: list[str] = Field(default_factory=list)
    produces: list[str] = Field(default_factory=list)
    produces_kind: DataKind = DataKind.SIGNAL
    control: NodeControl = NodeControl.NONE
    network: bool = False
    checkpoint_cap: int | None = None


class DesignPlan(BaseModel):
    """The decomposed design: the why plus the fully-typed node board."""

    why: ForgeWhy
    nodes: list[NodeDraft]


# --- verification + result ------------------------------------------------------


class CheckResult(BaseModel):
    """One deterministic verification check over a rendered package."""

    name: str
    passed: bool
    detail: str = ""


class VerificationReport(BaseModel):
    """Does the rendered package work? Deterministic checks (imports, assembles, recipe)."""

    works: bool = Field(description="True when every required check passed")
    score: float = Field(ge=0.0, le=1.0, description="Fraction of checks that passed")
    checks: list[CheckResult] = Field(default_factory=list)


# --- sandbox + build + audit ----------------------------------------------------


class SandboxResult(BaseModel):
    """The typed verdict of one out-of-process sandbox run."""

    stdout: str = ""
    stderr: str = ""
    exit_code: int = 0
    timed_out: bool = False

    @property
    def ok(self) -> bool:
        return self.exit_code == 0 and not self.timed_out


class PieceOut(BaseModel):
    """A draft/repair head's output: the complete function body (the only thing the
    model writes; everything else is deterministic)."""

    body: str = Field(
        description="The function body only — the lines inside the method, no def line, no fences"
    )


class FilledNode(BaseModel):
    """A node whose hole was drafted and passed every static gate."""

    node: str
    attempts: int


class UnfillableNode(BaseModel):
    """A node that exhausted its attempts; its NotImplementedError is restored."""

    node: str
    last_error: str
    attempts: int


class BuildReport(BaseModel):
    """Summary of the per-node build stage."""

    filled: list[FilledNode] = Field(default_factory=list)
    unfillable: list[UnfillableNode] = Field(default_factory=list)

    @property
    def all_filled(self) -> bool:
        return len(self.unfillable) == 0

    @property
    def remaining_holes(self) -> list[str]:
        return [u.node for u in self.unfillable]


class RequirementsVerdict(BaseModel):
    """The requirements head: does the built system serve the brief?"""

    meets_brief: bool = Field(
        description="True if the system, as built, addresses the brief"
    )
    gaps: list[str] = Field(
        default_factory=list,
        description="Concrete unmet requirements (empty if it meets)",
    )


class CriticVerdict(BaseModel):
    """The adversarial critic head: what is missing, wrong, or faked?"""

    issues: list[str] = Field(
        default_factory=list, description="Concrete problems found (empty if none)"
    )


class AuditIssue(BaseModel):
    """One whole-system problem found during the audit."""

    source: str = Field(
        description="assemble | smoke | dialect | requirements | critic"
    )
    detail: str


class AuditReport(BaseModel):
    """The verdict on the assembled system (sandboxed execution + agent review)."""

    works: bool = Field(
        description="True when the system assembles, smoke-runs, and stays dialect-clean"
    )
    rounds: int = 0
    checks: list[CheckResult] = Field(default_factory=list)
    requirements: RequirementsVerdict | None = None
    critic: CriticVerdict | None = None
    remaining_holes: list[str] = Field(default_factory=list)
    issues: list[AuditIssue] = Field(default_factory=list)


# --- the result -----------------------------------------------------------------


class ForgeResult(BaseModel):
    """The End[T] payload: the designed spec, what was rendered/built, and the verdicts."""

    spec: SystemSpec
    stage_reached: str = Field(
        default="design",
        description="design | render | build | audit — how far the run went",
    )
    rendered_files: list[str] = Field(default_factory=list)
    report: VerificationReport | None = Field(
        default=None, description="Cheap deterministic render-stage verdict"
    )
    build: BuildReport | None = None
    audit: AuditReport | None = None
    notes: list[str] = Field(
        default_factory=list,
        description="Advisory notes: caps hit, truncations, fallbacks",
    )

    @property
    def design_only(self) -> bool:
        return self.stage_reached == "design"

    @property
    def works(self) -> bool:
        if self.audit is not None:
            return self.audit.works
        return self.report.works if self.report is not None else False
