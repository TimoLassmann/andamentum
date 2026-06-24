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

from andamentum.core import AgentDefinition

from .schemas import ForgeAreas, ForgeWhy, JobList, NodeTyping


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

FRAME_PROMPT = (
    "Break the problem into its 2–4 big concerns — the fundamentally different things "
    "the system must get right. Each is a short phrase. Plain language; no nodes yet."
)

LIST_JOBS_PROMPT = (
    "List the atomic STEPS for THIS area — just the steps, as short job sentences. "
    "Each step does exactly one thing; anything needing 'and then' or a list of sub-steps "
    "is more than one step. One sentence each, 12 words or fewer, no semicolons. Do NOT "
    "specify types, inputs, or outputs yet — only the sentences. "
    "\n\nGood steps: 'Extract disease synonyms from the query.'  'Search PubMed with query "
    "terms.'  'Check if another search round is needed.'  'Score each gene by evidence strength.' "
    "\n\nKeep this area to 2–4 steps; the whole system should stay well under 20 steps."
)

TYPE_NODE_PROMPT = (
    "You are specifying the fields of ONE step in a larger plan (shown to you). Type only "
    "the step marked >>>; return just its fields. "
    "\n\n(1) kind — spine if the answer is code-computable (math, API call, regex, structured "
    "lookup, routing); head if it needs reading text and making a judgment (extraction, "
    "synthesis, scoring, classification). "
    "(2) consumes — the EXACT data names this step reads, copied from the plan above; the graph "
    "input is always named `input`. "
    "(3) produces — EXACTLY ONE new datum name, a short noun phrase a later step can consume. "
    "(4) produces_kind — signal (run-scoped value handed onward) or entity (a database record "
    "stored and retrieved by id, like a User or Ticket — only for records that persist BEYOND "
    "the run and are queried later; never right for intermediate processing results). "
    "(5) control — EXACTLY one of: none (default); checkpoint (loop control: 'run this area "
    "again, or move on?' — when an area repeats until a condition is met); decision (routes to "
    "DIFFERENT downstream pipelines — distinct paths to different outcomes, NOT loop control); "
    "consequential (requires human approval before proceeding). "
    "(6) network — true ONLY if the step reaches an external service over the internet (web/HTTP "
    "API, fetches a page, queries a remote database). A network step is ALWAYS spine. Default false. "
    "\n\nKEY RULES: 'should I run this area again?' → checkpoint. 'which of several different "
    "paths?' → decision. A search/harvest loop that repeats until stall is ALWAYS a checkpoint."
)


UNDERSTAND = AgentDefinition(
    name="understand", prompt=UNDERSTAND_PROMPT, output_model=ForgeWhy
)
FRAME = AgentDefinition(name="frame", prompt=FRAME_PROMPT, output_model=ForgeAreas)
LIST_JOBS = AgentDefinition(
    name="list_jobs", prompt=LIST_JOBS_PROMPT, output_model=JobList
)
TYPE_NODE = AgentDefinition(
    name="type_node", prompt=TYPE_NODE_PROMPT, output_model=NodeTyping
)
