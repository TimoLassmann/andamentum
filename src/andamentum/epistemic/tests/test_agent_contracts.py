"""The contract every registered agent shares, written once.

Two properties held by all agents, previously copied as
``test_agent_registered`` and ``test_output_model_fields`` into a class
per agent across three files:

1. The agent is registered under exactly the name callers pass to
   ``get_agent``. A mismatch is invisible until a pipeline asks for an
   agent that is not there.
2. The registered agent carries the output model this table names. A
   registration wired to the wrong model type-checks and passes every
   field test, then produces the wrong shape at run time.
3. Its output model declares exactly the expected fields — no more, no
   fewer. Small local models fill these schemas, so a field silently
   added or renamed changes what the model is asked to produce.

Adding an agent is a new row in ``AGENT_CONTRACTS``, not a new copied
class, so a new agent cannot be added without the same three checks.

Behaviour specific to one agent — adapter normalisation, Literal
vocabularies, whitespace stripping — stays in the file that owns that
agent (`test_decomposed_agents.py`, `test_new_tracks.py`,
`test_classify_question.py`). This file holds only what is shared.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from ..agents import get_agent
from ..agents.output_models import (
    CheckPairwiseIndependenceOutput,
    ClassifyQuestionOutput,
    ContrastiveEvaluationOutput,
    CrossClaimConsistencyOutput,
    DefineFalsificationOutput,
    DraftClaimOutput,
    ExtractAssertionOutput,
    GenerateCounterqueryOutput,
    IdentifyTestableAspectOutput,
    SpecifyPredictionOutput,
)


@dataclass(frozen=True)
class AgentContract:
    """One agent's registration name and its output model's field set."""

    name: str
    output_model: type
    fields: frozenset[str]


AGENT_CONTRACTS: list[AgentContract] = [
    AgentContract(
        "epistemic_generate_counterquery",
        GenerateCounterqueryOutput,
        frozenset({"query", "framing"}),
    ),
    AgentContract(
        "epistemic_check_pairwise_independence",
        CheckPairwiseIndependenceOutput,
        frozenset({"independent", "rationale"}),
    ),
    AgentContract(
        "epistemic_identify_testable_aspect",
        IdentifyTestableAspectOutput,
        frozenset({"testable_dimension", "observation_type"}),
    ),
    AgentContract(
        "epistemic_specify_prediction",
        SpecifyPredictionOutput,
        frozenset(
            {"expected_observation", "conditions", "timeframe", "measurability"}
        ),
    ),
    AgentContract(
        "epistemic_define_falsification",
        DefineFalsificationOutput,
        frozenset({"falsification_criterion"}),
    ),
    AgentContract(
        "epistemic_extract_assertion",
        ExtractAssertionOutput,
        frozenset({"assertion"}),
    ),
    AgentContract(
        "epistemic_draft_claim",
        DraftClaimOutput,
        frozenset({"statement", "scope", "direction"}),
    ),
    AgentContract(
        "epistemic_contrastive_evaluation",
        ContrastiveEvaluationOutput,
        frozenset({"better_claim", "distinguishing_observation", "confidence"}),
    ),
    AgentContract(
        "epistemic_cross_claim_consistency",
        CrossClaimConsistencyOutput,
        frozenset({"conflicts", "tension_point"}),
    ),
    AgentContract(
        "epistemic_classify_question",
        ClassifyQuestionOutput,
        frozenset({"question_type", "reasoning"}),
    ),
]


@pytest.mark.parametrize("contract", AGENT_CONTRACTS, ids=lambda c: c.name)
def test_agent_registered(contract: AgentContract) -> None:
    agent = get_agent(contract.name)
    assert agent.name == contract.name


@pytest.mark.parametrize("contract", AGENT_CONTRACTS, ids=lambda c: c.name)
def test_agent_carries_its_declared_output_model(contract: AgentContract) -> None:
    """The registry entry points at the model this table names.

    Only `epistemic_classify_question` checked this before; the table
    already pairs every agent with its model, so the other nine cost
    nothing. A mismatch here is invisible to the field test — a wrong
    model with the right field names would pass that one.
    """
    agent = get_agent(contract.name)
    assert agent.output_model is contract.output_model


@pytest.mark.parametrize("contract", AGENT_CONTRACTS, ids=lambda c: c.name)
def test_output_model_fields(contract: AgentContract) -> None:
    """Exactly the declared fields.

    The per-agent version of this asserted membership plus a length; a
    set comparison is the same check and names the offending field when
    it fails.
    """
    assert set(contract.output_model.model_fields) == set(contract.fields)
