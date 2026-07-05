"""The design heads, as data — forge's own LLM calls (recipe R5, dialect "agents").

Four tiny heads drive the brief → spec design. Each is an ``AgentDefinition`` (a
name, a prompt, a flat output schema); the workers call them through an
``AgentSink`` Port held in Deps, so the whole pipeline runs against a stub with no
live model (dialect: "stub an agent by swapping a fake into Deps").

Prompts ported from the ``forge`` exploratory dump's design roles. Leaf worker
file: ``pydantic`` + ``andamentum.core`` only, no graph engine.
"""

from __future__ import annotations

from typing import Any, Protocol, cast

from pydantic import BaseModel

from pydantic_ai.exceptions import ModelHTTPError, UnexpectedModelBehavior

from andamentum.agentic_dialect import law
from andamentum.core import AgentDefinition

from .schemas import (
    ConsumeSelection,
    CriticVerdict,
    Fitness,
    ForgeAreas,
    ForgeWhy,
    JobList,
    NodeDeclaration,
    PieceOut,
    RequirementsVerdict,
)

# A hard LLM-output failure: the model could not produce schema-valid output after the
# runner's own retries (``UnexpectedModelBehavior`` — e.g. a small model returning the JSON
# schema envelope instead of an instance), or a transient provider/HTTP error
# (``ModelHTTPError``). forge catches this at each ``sink.run`` call site and degrades —
# one node becomes unfillable, an advisory head is skipped, a design stage fails loud and
# legible — never an uncaught pydantic-ai traceback that discards the whole run.
MODEL_OUTPUT_ERRORS = (UnexpectedModelBehavior, ModelHTTPError)


class AgentSink(Protocol):
    """The one capability the design workers need: run an agent, get its output.

    Structurally satisfied by ``andamentum.core.AgentRunner``; a test stub keyed by
    agent name satisfies it too. This is the dialect's agent test seam as a Port.
    """

    async def run(self, defn: AgentDefinition, **kwargs: object) -> BaseModel: ...


class CoreAgentSink:
    """The production ``AgentSink`` — wraps ``core.AgentRunner`` (tool-calling output
    with the small-model PromptedOutput fallback). Built once per run from the model."""

    def __init__(self, model: str) -> None:
        from andamentum.core import AgentRunner

        self._runner = AgentRunner(model=model)

    async def run(self, defn: AgentDefinition, **kwargs: object) -> BaseModel:
        # AgentRunner.run takes content kwargs plus an optional `validators=`; cast so the
        # `object`-typed content kwargs forward cleanly (we never pass validators here).
        out = await self._runner.run(defn, **cast("dict[str, Any]", kwargs))
        assert isinstance(out, BaseModel)
        return out


UNDERSTAND_PROMPT = (
    "Restate the user's brief as a problem, not a solution. Give the purpose in one "
    "honest paragraph, what the system takes in (one natural-language input) and what "
    "it produces. Do not mention graphs or nodes."
)

FITNESS_PROMPT = (
    "You are a fitness gate. Before any design work, judge ONE thing about this brief: can "
    "the system it asks for be built as a FUNCTION — one input at the door, one output at "
    "the end, computed in a single run, with the system itself deciding what happens next "
    "from start to finish?\n\n"
    "Ask exactly this question and nothing else: to do what this brief asks, does anything "
    "OUTSIDE the system have to decide what happens next — a user choosing which operation "
    "to run, a session continuing a conversation across turns, or an event firing on its "
    "own?\n\n"
    "- If NO external driver decides what happens next, it is a function. Set rung to "
    "'function' if the output is computed purely from this run's input, or 'stateful_function' "
    "if the output must depend on what EARLIER RUNS produced (a durable memory loaded at the "
    "start of the run and saved at the end).\n"
    "- If something external DOES decide what happens next, name which of the three structural "
    "drivers it is: a caller choosing among several operations is an 'app'; a caller driving a "
    "multi-turn session is an 'agent'; the world emitting triggering events is a 'service'.\n\n"
    "Judge SHAPE, never words. Do not key off any particular verb in the brief. A brief is a "
    "function or not because of who owns the control loop, not because it used a word like "
    "save, track, manage, or watch. A loop INSIDE one run (generate, check, retry until a "
    "bounded condition, all within the single call) is internal control flow, NOT an external "
    "driver — it stays a function.\n\n"
    "When the brief is genuinely ambiguous, PREFER a function: set realizable_as_function true, "
    "choose the function rung, and STATE the interpretation you adopted in reason rather than "
    "refusing. A terse but valid request (e.g. 'summarise this') is a function — never block it "
    "for lack of detail.\n\n"
    "Fill the fields. realizable_as_function: true for a function or stateful_function, false "
    "for an app/agent/service. rung: the single best-fit class. reason: who owns the loop, and "
    "which external driver (if any) decides what happens next. suggested_reshape: if not "
    "realizable, the function hiding inside the request, phrased as a brief the user could "
    "resubmit (e.g. for 'manage my reading list' → 'given my current reading list and a new "
    "message, return the updated list'); empty when realizable."
)

FRAME_PROMPT = (
    "List the DISTINCT big concerns this system must get right — the fundamentally different "
    "things it must handle, each needing its own kind of work. Most simple tasks have exactly "
    "ONE concern. Add a second or third ONLY when a concern is genuinely separable: a different "
    "input, a different kind of judgment, or a stage that can succeed or fail on its own. Never "
    "split one job into phases of itself ('identify X' then 'extract X' then 'refine X' are ONE "
    "concern, not three). Each is a short phrase. Plain language; no nodes yet."
)

LIST_JOBS_PROMPT = (
    "List the FEWEST steps that do this area's job — and a whole area is OFTEN A SINGLE STEP. "
    "A step earns its place only if it does one thing no other step should do. Break a step in "
    "two ONLY when: one call would have to make two unrelated judgments (e.g. extract a value, "
    "THEN decide whether to search again), or a deterministic transform sits between two "
    "judgments, or a step loops back to repeat. NEVER split one judgment across phases "
    "('identify', then 'extract', then 'refine' the same thing is ONE step). Each step does "
    "exactly one thing. One sentence each, 12 words or fewer, no semicolons. Do NOT specify "
    "types, inputs, or outputs yet — only the sentences. "
    "\n\nA LOOP IS WRITTEN ONCE, NEVER UNROLLED. If the area repeats until a condition is met "
    "(search until the evidence is enough, retry until it succeeds), write the body ONE time "
    "plus ONE check step that repeats it — that check step's whole job is to run the earlier "
    "steps again. Do NOT write the search/gather/synthesize step more than once, do NOT add a "
    "second or third check step, and do NOT write out separate 'first search', 'fill the gaps', "
    "'search again' phases: they are the SAME one step, repeated by the loop. Two or three "
    "steps is right for a loop; eight is unrolling. "
    "\n\nGood single-step areas: 'Summarize the text into three bullet points.'  "
    "'Translate the paragraph into French.' "
    "\n\nGood multi-step area (a genuine loop, EXACTLY these two steps): 'Search the web with "
    "the query terms.'  'Check if another search round is needed.'"
)

DECLARE_NODE_PROMPT = (
    "You are declaring the OUTPUT and kind of ONE step in a larger plan (shown to you, the step "
    "marked >>> with what every step declares it produces). Return just its fields. You declare "
    "what this step PRODUCES — you do NOT choose its inputs here (that is a separate step). "
    "\n\n(1) kind — SPINE if the answer is computable from the inputs by a function you could "
    "name: math, regex, a lookup, a sort/filter on a field that ALREADY exists, an API call, or "
    "a branch on a value an earlier step produced. HEAD if producing the output needs READING "
    "natural-language text and judging its meaning — the ranking key, selection criterion, "
    "label, score, or condensed content is NOT a field, it must be derived from the prose. "
    "Ranking, selecting, scoring, classifying, extracting, or condensing OPEN TEXT is ALWAYS a "
    "head, never code. "
    "(2) produces — name EXACTLY ONE new variable this step writes — a short noun phrase "
    "describing its output — that a later step can read. (Don't worry about clashing with another "
    "step's name; identical names are made unique automatically.) "
    "(3) produces_kind — signal (run-scoped value handed onward) or entity (a database record "
    "stored and retrieved by id, like a User or Ticket — only for records that persist BEYOND "
    "the run and are queried later; never right for intermediate processing results). "
    "(4) control — EXACTLY one of: none (default); checkpoint (loop control: 'run this area "
    "again, or move on?' — when an area repeats until a condition is met); decision (routes to "
    "DIFFERENT downstream pipelines — distinct paths to different outcomes, NOT loop control); "
    "consequential (requires human approval before proceeding). "
    "(5) network — true ONLY if the step reaches an external service over the internet (web/HTTP "
    "API, fetches a page, queries a remote database). A network step is ALWAYS spine. Default false. "
    "\n\nKEY RULES: 'should I run this area again?' → checkpoint. 'which of several different "
    "paths?' → decision. A search/harvest loop that repeats until stall is ALWAYS a checkpoint."
)

SELECT_CONSUMES_PROMPT = (
    "You are choosing the INPUTS of ONE step (its job is shown). Below is a NUMBERED list of every "
    "datum available to read: number 0 is `input` (the raw original text the whole system was "
    "given), and each later number is a value an EARLIER step produces, annotated with that step's "
    "job. Return `consume_indices` — the NUMBERS of the inputs this step reads. Choose numbers only "
    "from the list; do NOT invent names. "
    "\n\nSteps form a CHAIN: this step should almost always READ the output of an earlier step "
    "(most often the one right before it) and build on it — pick that number. Pick number 0 "
    "(`input`) ONLY when this step genuinely needs the unprocessed original text, NEVER when an "
    "earlier step already turned it into something this step should refine. A step may read one or "
    "more inputs (a fan-in); pick every number it truly needs, and no others. "
    "\n\nIf the message includes feedback about a structural problem, fix EXACTLY what it names by "
    "re-choosing the numbers — most often picking the number of the earlier step whose output this "
    "step should read."
)


UNDERSTAND = AgentDefinition(
    name="understand", prompt=UNDERSTAND_PROMPT, output_model=ForgeWhy
)
FITNESS = AgentDefinition(name="fitness", prompt=FITNESS_PROMPT, output_model=Fitness)
FRAME = AgentDefinition(name="frame", prompt=FRAME_PROMPT, output_model=ForgeAreas)
LIST_JOBS = AgentDefinition(
    name="list_jobs", prompt=LIST_JOBS_PROMPT, output_model=JobList
)
DECLARE_NODE = AgentDefinition(
    name="declare_node", prompt=DECLARE_NODE_PROMPT, output_model=NodeDeclaration
)
SELECT_CONSUMES = AgentDefinition(
    name="select_consumes", prompt=SELECT_CONSUMES_PROMPT, output_model=ConsumeSelection
)


# --- the code-authoring heads (stage 3 build) -----------------------------------
#
# These are the agents that WRITE code. They are grounded in the agentic dialect: the
# laws a node body must honour are pulled from the canon (`law(...)`) and pushed into
# the prompt, so the authors can't drift from the spec the rest of the system enforces.


def _dialect_body_grounding() -> str:
    """The dialect laws a node body must honour, drawn from the canonical module so
    forge's authoring agents and the dialect cannot diverge."""
    ids = ("L4", "L5", "L6", "L7", "L8")
    lines = [
        "This node body belongs to an andamentum-dialect agentic system. Honour these laws (the canon):"
    ]
    for i in ids:
        lw = law(i)
        lines.append(f"- {lw.id} {lw.name}: {lw.statement}")
    return "\n".join(lines)


_BODY_GROUNDING = _dialect_body_grounding()

DRAFT_PROMPT = (
    _BODY_GROUNDING + "\n\n"
    "You implement ONE function body in a pydantic-graph node. `ctx.state` is the working "
    "memory; `ctx.deps` holds injected resources. Return ONLY the lines inside the function — "
    "no def line, no class, no markdown fences, 4-space base indentation. Read and write ONLY "
    "the ctx.state fields listed in the context, and USE them: read EVERY input field listed (your "
    "logic must actually depend on its inputs) and set EVERY output field listed (the node must "
    "produce its output). Access fields one at a time as `ctx.state.<field>` — never `model_dump`, "
    "`getattr`, or any bulk/dynamic access. Return "
    "one of the listed successors (`return NodeName()`) or `return End(<str>)`. Keep any PREAMBLE "
    "lines first, unchanged. Write a REAL implementation — never a hardcoded value, a bare pass, "
    "or `raise NotImplementedError`. No clock, randomness, process control, raw files, or sockets. "
    "FAIL LOUD — never a silent fallback: do NOT wrap logic in a broad `except` that swallows the "
    "error, do NOT default or `continue` when state is missing or wrong, do NOT paper over a value "
    "with `or <default>`. If something required is absent or unexpected, let it raise. A node that "
    "runs but produces the wrong thing is worse than one that stops."
)

REPAIR_PROMPT = (
    _BODY_GROUNDING + "\n\n"
    "You fix a function body a gate rejected. Same rules as drafting. Read the rejection reason "
    "and fix EXACTLY what it names — most often a state field or successor not in the allowed list, "
    "or a forbidden import. Return only the corrected body."
)

REQUIREMENTS_PROMPT = (
    "You audit whether a built agentic system serves the user's brief. You are shown the brief and "
    "a summary of the system (purpose, nodes, what each does). Decide meets_brief (true/false) and "
    "list concrete gaps — requirements from the brief the system does not address. Be specific; an "
    "empty gaps list means it fully meets the brief."
)

CRITIC_PROMPT = (
    "You are an adversarial reviewer of a built agentic system. You are shown its node bodies and "
    "the list of node names. Find what is "
    "missing, wrong, or faked — a hardcoded value standing in for real logic, a node that drops its "
    "input, a TODO left behind. Also flag SILENT FALLBACKS that hide failures: a default value "
    "substituted when state is missing, an `or <default>` / `.get(key, default)` papering over an "
    "absent value, or any error quietly absorbed so the run continues on wrong data. For EACH issue, "
    "name the offending node — copy its name EXACTLY from the node-names list — in `node`, and the "
    "concrete problem in `issue`. List concrete issues; an empty list means you found none."
)

DRAFT = AgentDefinition(name="build_draft", prompt=DRAFT_PROMPT, output_model=PieceOut)
REPAIR = AgentDefinition(
    name="build_repair", prompt=REPAIR_PROMPT, output_model=PieceOut
)
REQUIREMENTS = AgentDefinition(
    name="requirements", prompt=REQUIREMENTS_PROMPT, output_model=RequirementsVerdict
)
CRITIC = AgentDefinition(
    name="critic", prompt=CRITIC_PROMPT, output_model=CriticVerdict
)
