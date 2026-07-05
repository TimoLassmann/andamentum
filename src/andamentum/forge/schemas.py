"""Boundary schemas for the forge meta-pipeline — and the ``ForgeResult``.

Two families live here:

- **Agent outputs** (``ForgeWhy``, ``ForgeAreas``, ``JobList``, ``NodeDeclaration``,
  ``ConsumeSelection``) — the
  small, flat, enum-guarded models the design heads fill. Each obeys the dialect's
  "small heads": ≤6 flat fields, closed vocabularies as enums. These are *not* the
  generated system's agents — they are forge's own.
- **The board + the result** (``NodeDraft``, ``DesignPlan``, ``CheckResult``,
  ``VerificationReport``, ``ForgeResult``) — the typed values that ride between steps
  and the final ``End[ForgeResult]`` payload.

Leaf worker file: ``pydantic``, the sibling ``spec`` enums, the dialect's leaf
``Violation`` model (retained structured on ``CheckResult``), and the sibling
``runtime.EnvelopeTolerantModel`` base (agent-output models inherit it so a small model's
schema-envelope reply is unwrapped deterministically); ``runtime`` is itself engine-free,
so no graph engine enters here.
"""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field

from andamentum.agentic_dialect import Violation

from .runtime import EnvelopeTolerantModel

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


class ForgeWhy(EnvelopeTolerantModel):
    """The understand head: purpose + boundary, in plain language."""

    purpose: str = Field(
        description="One or two plain sentences: what the system is for and the value it delivers"
    )
    boundary_in: str = Field(
        description="What the system takes in — the single natural-language input at the door"
    )
    boundary_out: str = Field(description="What the system produces — its final output")


class Fitness(EnvelopeTolerantModel):
    """The fitness-gate head (dialect L9): is the brief realisable as a function?

    Judges SHAPE — who owns the control loop — never vocabulary. A function is one input
    at the door, one output at the end, one run, control owned inside. If an external
    driver decides what happens next (an operation-chooser, a session, an event source)
    the brief is an app / agent / service and is out of forge's scope.
    """

    realizable_as_function: bool = Field(
        description="True if the brief is a function or stateful_function; false if it is an app, agent, or service"
    )
    rung: Literal["function", "stateful_function", "app", "agent", "service"] = Field(
        description=(
            "The system class by who owns the control loop: function = one input → one "
            "output → one run; stateful_function = same, plus durable memory across runs; "
            "app = caller chooses among operations; agent = caller drives a session; "
            "service = the world emits triggering events"
        )
    )
    reason: str = Field(
        description=(
            "One or two plain sentences naming who owns the control loop — for an app/agent/"
            "service, which external driver decides what happens next; for a function, the "
            "interpretation adopted if anything was ambiguous"
        )
    )
    suggested_reshape: str = Field(
        default="",
        description=(
            "When not realizable, the rung-1/2 function hiding inside the request, phrased as "
            "a brief the user could resubmit; empty when realizable"
        ),
    )


class ForgeAreas(EnvelopeTolerantModel):
    """The frame head: the 2–4 big concerns the system must get right."""

    areas: list[str] = Field(
        description="The distinct big concerns — one or more (usually exactly one for a simple task), each a short phrase"
    )


class JobList(EnvelopeTolerantModel):
    """Decompose stage 1: an area's atomic steps as plain sentences (no types yet).

    A list of strings is the one list shape small models handle reliably — the typed
    fields are filled later, one node at a time (``NodeDeclaration`` + ``ConsumeSelection``).
    """

    jobs: list[str] = Field(
        description="The atomic steps for this area, each a short job sentence (12 words or fewer)"
    )


class NodeDeclaration(EnvelopeTolerantModel):
    """Decompose pass 2a (DECLARE): the typed fields for ONE already-named node — its
    kind and the SINGLE output it produces. It declares NO inputs here.

    Inputs are chosen in a separate pass (2b) as ORDINALS into a closed list, never as
    free-text names, so a consume can never reference a name no step produces. Producing
    one name per node, deduped deterministically afterwards, makes the produced-name set
    globally unique by construction — which is what removes the small-model wiring thrash.
    """

    kind: NodeKind = Field(
        default=NodeKind.SPINE,
        description="spine = code-computable (math, lookup, regex, API call, routing); head = LLM judgment over text",
    )
    produces: str = Field(
        default="",
        description="ONE NEW data name this node writes — a short noun phrase (e.g. 'ranked candidates')",
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


class ConsumeSelection(EnvelopeTolerantModel):
    """Decompose pass 2b (SELECT): which inputs this node reads, chosen BY NUMBER from a
    closed, numbered list of available data (the graph input plus every produced name).

    The model returns ordinals, never name strings — so a selected input is always a real,
    already-produced datum. An out-of-range number is dropped and recorded (never silently
    kept as a phantom read).
    """

    consume_indices: list[int] = Field(
        default_factory=list,
        description="The NUMBERS of the inputs this step reads, taken from the numbered list shown (usually the output of an earlier step)",
    )


# --- design diagnostics (assemble → diagnose → repair) --------------------------


class FindingKind(str, Enum):
    """The catalogue of structural problems the deterministic diagnoser detects.

    Every kind is blocking — this is a fail-loud pipeline; there are no advisory-only
    findings. Each describes one way a declared node board fails to form a coherent
    data DAG.
    """

    DANGLING_READ = (
        "dangling_read"  # consumes a name nobody produces (and not an input)
    )
    ORPHAN_OUTPUT = "orphan_output"  # produces a name nobody reads (and not the output)
    NEAR_MISS = (
        "near_miss"  # a produce/consume pair almost matches — likely one variable
    )
    DUPLICATE_PRODUCER = "duplicate_producer"  # >1 node produces the same name
    MULTIPLE_SINKS = "multiple_sinks"  # more than one terminal output node
    NO_OUTPUT = "no_output"  # no terminal output node at all
    UNREACHABLE = "unreachable"  # node not reachable from the input
    DEAD_END = "dead_end"  # node whose work reaches no output
    DISCONNECTED = "disconnected"  # a component with no path to the input→output flow
    UNINTENDED_CYCLE = (
        "unintended_cycle"  # a cycle not gated by a bounded-loop checkpoint
    )
    UNCOVERED_AREA = (
        "uncovered_area"  # a framed concern that decomposed to zero node jobs
    )


class DesignFinding(BaseModel):
    """One structural problem with a concrete, agent-readable suggested fix."""

    kind: FindingKind
    node: str = Field(default="", description="The node id involved, if any")
    variable: str = Field(default="", description="The data name involved, if any")
    detail: str = Field(description="What is wrong, in plain language")
    suggestion: str = Field(
        default="", description="The concrete fix to apply (e.g. a near-miss rename)"
    )


class DesignReport(BaseModel):
    """The deterministic diagnosis of a declared node board.

    All findings are blocking. ``clean`` is the success predicate the refine loop
    checks; ``summary()`` renders the compact text the repair agent reads back.
    """

    findings: list[DesignFinding] = Field(default_factory=list)

    @property
    def clean(self) -> bool:
        """True when the board has no structural problems."""
        return len(self.findings) == 0

    def summary(self) -> str:
        """A compact, line-per-finding text for the repair agent and for fail-loud messages."""
        if not self.findings:
            return "no findings — the design is clean"
        lines: list[str] = []
        for f in self.findings:
            parts = [f"[{f.kind.value}]"]
            if f.node:
                parts.append(f"node {f.node}")
            if f.variable:
                parts.append(f"variable {f.variable!r}")
            head = " ".join(parts)
            line = f"{head}: {f.detail}"
            if f.suggestion:
                line += f"  — suggestion: {f.suggestion}"
            lines.append(line)
        return "\n".join(lines)


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


class DesignPlan(BaseModel):
    """The decomposed design: the why plus the fully-typed node board."""

    why: ForgeWhy
    nodes: list[NodeDraft]


# --- verification + result ------------------------------------------------------


class CheckResult(BaseModel):
    """One deterministic verification check over a rendered package.

    ``detail`` is the human-readable one-liner. The remaining fields retain the
    structured evidence attribution consumes (§4.4/§4.5): for the ``tests`` check,
    ``raw_output`` is the untruncated pytest stdout (whose ``nodes.py:line`` frames
    map failures to nodes) and ``tests_passed``/``tests_failed`` are the parsed
    summary counts (the regression guard's total order); for the ``dialect`` check,
    ``violations`` is the untruncated ``list[Violation]`` (each carries
    ``.file``/``.line``/``.law``), never re-parsed from ``detail``."""

    name: str
    passed: bool
    detail: str = ""
    raw_output: str = ""
    tests_passed: int = 0
    tests_failed: int = 0
    tests_errored: int = 0  # pytest collection/import ERRORS (≠ assertion failures)
    violations: list[Violation] = Field(default_factory=list)


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


class PieceOut(EnvelopeTolerantModel):
    """A draft/repair head's output: the complete function body (the only thing the
    model writes; everything else is deterministic)."""

    body: str = Field(
        description="The function body only — the lines inside the method, no def line, no fences"
    )


class FilledNode(BaseModel):
    """A node whose hole was drafted and passed every static gate."""

    node: str
    attempts: int
    body: str


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


class RequirementsVerdict(EnvelopeTolerantModel):
    """The requirements head: does the built system serve the brief?"""

    meets_brief: bool = Field(
        description="True if the system, as built, addresses the brief"
    )
    gaps: list[str] = Field(
        default_factory=list,
        description="Concrete unmet requirements (empty if it meets)",
    )


class NodeFinding(BaseModel):
    """One adversarial-critic problem, attributed to the node that carries it.

    ``node`` is the model's best guess at the offending node's name; it is a free
    string (node names are per-spec, so it cannot be a ``Literal``) reconciled to a
    real spec node name via rapidfuzz where the critic result is consumed (§4.4
    signal 4). Kept flat — two string fields — so small models fill it reliably."""

    node: str = Field(
        description="The name of the node whose body carries this problem"
    )
    issue: str = Field(description="The concrete problem found in that node's body")


class CriticVerdict(EnvelopeTolerantModel):
    """The adversarial critic head: what is missing, wrong, or faked?"""

    issues: list[NodeFinding] = Field(
        default_factory=list,
        description="Concrete problems found, each attributed to a node (empty if none)",
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
    fitness: Fitness | None = Field(
        default=None,
        description="The accepted front-gate verdict (rung, reason); None only if the gate did not run",
    )
    rendered_files: list[str] = Field(default_factory=list)
    design_report: DesignReport | None = Field(
        default=None,
        description="The deterministic structural diagnosis of the node board (clean after repair)",
    )
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
