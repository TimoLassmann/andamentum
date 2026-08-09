"""Shared fixtures for epistemic package tests."""

import pytest
from types import SimpleNamespace
from typing import Any

from andamentum.epistemic.entities.verdict import (
    ClaimVerdict,
    EpistemicVerdict,
    PosteriorReport,
    VerdictProvenance,
)
from andamentum.epistemic.repository import EpistemicRepository


def make_verdict(
    objective_id: str = "obj-1",
    *,
    posterior: float | None = None,
    n_active: int = 1,
    n_adjudicated: int = 1,
    claim_ids: tuple[str, ...] = (),
) -> EpistemicVerdict:
    """A minimal ``EpistemicVerdict`` for tests that build a Snapshot directly.

    Both synthesis operations RENDER the run's verdict and raise when it
    is absent (a missing verdict at synthesis is a routing bug —
    ``FreezeSnapshotOperation`` stamps it from graph state). Tests that
    construct a Snapshot by hand therefore have to stamp one too; this
    factory keeps that one line rather than eight.
    """
    report = (
        PosteriorReport(
            posterior=posterior,
            log_odds=0,
            supporting_count=0.0,
            contradicting_count=0.0,
            counting_posterior=posterior,
            objective_id=objective_id,
            question_type="verificatory",
            explanation="test fixture",
        )
        if posterior is not None
        else None
    )
    return EpistemicVerdict(
        objective_id=objective_id,
        posterior_report=report,
        claims=[
            ClaimVerdict(claim_id=cid, posterior=posterior) for cid in claim_ids
        ],
        provenance=VerdictProvenance(
            n_claims_total=n_active,
            n_active_claims=n_active,
            n_adjudicated=n_adjudicated,
            n_no_verdict=max(0, n_active - n_adjudicated),
        ),
    )


def _to_namespace(obj: Any) -> Any:
    """Recursively convert dicts to SimpleNamespace for attribute access.

    The adapters use attribute access (raw.field) on agent outputs,
    so test stubs must return objects, not plain dicts.
    """
    if isinstance(obj, dict):
        return SimpleNamespace(**{k: _to_namespace(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_to_namespace(item) for item in obj]
    return obj


class MockAgentRunner:
    """Mock agent runner returning canned responses per agent name."""

    def __init__(self, responses: dict[str, Any] | None = None):
        self.responses = responses or {}
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.model = "test-model"

    async def run(self, agent_name: str, **kwargs: Any) -> Any:
        self.calls.append((agent_name, kwargs))
        raw = self.responses.get(agent_name, {})
        return _to_namespace(raw)


class FakeAgentRunner:
    """Richer stub with default canned responses for common agents.

    Response shapes match the actual agent manifest output_model fields
    so adapters in epistemic.adapters can process them correctly.
    """

    def __init__(self, overrides: dict[str, Any] | None = None):
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self._overrides = overrides or {}
        self.model = "test-model"

    async def run(self, agent_name: str, **kwargs: Any) -> Any:
        self.calls.append((agent_name, kwargs))
        if agent_name in self._overrides:
            return _to_namespace(self._overrides[agent_name])
        raw = _FAKE_DEFAULTS.get(agent_name, {})
        return _to_namespace(raw)


class FailingRepo(EpistemicRepository):
    """Repository that fails on specific entity lookups.

    Used to test error handling when repo operations throw.
    """

    def __init__(
        self,
        store,
        fail_on: set[str] | None = None,
        fail_on_query: set[str] | None = None,
    ):
        super().__init__(store)
        self.fail_on = fail_on or set()
        self.fail_on_query = fail_on_query or set()
        self.failure_log: list[str] = []

    async def get(self, entity_type: str, entity_id: str) -> Any:  # type: ignore[override]
        if entity_id in self.fail_on:
            self.failure_log.append(f"get:{entity_type}:{entity_id}")
            raise RuntimeError(f"Simulated repo failure for {entity_id}")
        return await super().get(entity_type, entity_id)

    async def query(self, entity_type: str, **filters: Any) -> Any:  # type: ignore[override]
        if entity_type in self.fail_on_query:
            self.failure_log.append(f"query:{entity_type}")
            raise RuntimeError(f"Simulated query failure for {entity_type}")
        return await super().query(entity_type, **filters)


class PartiallyFailingRunner:
    """Agent runner that fails on specific agents.

    Used to test error handling when specific agent calls throw.
    """

    def __init__(
        self, fail_on: set[str], fallback_runner: FakeAgentRunner | None = None
    ):
        self.fail_on = fail_on
        self.fallback = fallback_runner or FakeAgentRunner()
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def run(self, agent_name: str, **kwargs: Any) -> Any:
        self.calls.append((agent_name, kwargs))
        if agent_name in self.fail_on:
            raise RuntimeError(f"Simulated agent failure: {agent_name}")
        return await self.fallback.run(agent_name, **kwargs)


class MalformedOutputRunner:
    """Agent runner that returns outputs with missing or wrong-typed fields.

    Used to test adapter resilience to unexpected agent outputs.
    """

    def __init__(self, overrides: dict[str, dict]):
        self.overrides = overrides
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def run(self, agent_name: str, **kwargs: Any) -> Any:
        self.calls.append((agent_name, kwargs))
        raw = self.overrides.get(agent_name, {})
        return _to_namespace(raw)


# Default canned responses matching actual agent manifest output_model fields.
# The adapters use attribute access, so field names must be exact.
_FAKE_DEFAULTS: dict[str, dict[str, Any]] = {
    "epistemic_check_synthesis_demand": {
        # Phase 1 of lazy-escalation: default to needs_more=False so
        # tests that incidentally exercise this agent get a satisfied
        # demand without further setup. Tests that want to exercise the
        # needs_more=True path can override per-test via fake_runner.
        "needs_more": False,
        "justification": "Default fake_runner output: satisfied.",
        "target_hint": "",
    },
    "epistemic_clarify_question": {
        "clarified_question": "What is spaced repetition and does it work?",
        "key_terms": ["spaced repetition", "memory"],
        "reasoning": "Clarified scope to focus on effectiveness evidence",
    },
    "epistemic_classify_question": {
        "question_type": "verificatory",
        "reasoning": "Default classification for testing",
    },
    "epistemic_conceptual_analysis": {
        "terms": ["spaced repetition"],
        "definitions": ["A learning technique using increasing intervals"],
        "assumptions": ["Learner has access to material"],
        "context_summary": "Educational psychology domain",
    },
    "epistemic_extract_evidence": {
        "relevant_quotes": ["Spaced repetition improves retention by 40%"],
        "limitations": ["Single study"],
        "experimental_context": "Educational research",
    },
    "epistemic_identify_single_issue": {
        "has_issue": False,
        "description": "",
        "issue_type": "",
        "reversal_test": False,
    },
    "epistemic_deductive_validation": {
        "passes_deductive_validation": True,
        "issues_found": [],
        "issue_types": [],
    },
    "epistemic_verify_computationally": {
        "verification_code": "assert True",
        "packages_required": [],
        "expected_behavior": "Consistent",
        "test_description": "Basic consistency check",
    },
    "epistemic_resolve_uncertainty": {
        "resolution": "Resolved through additional evidence",
        "can_resolve": True,
        "remaining_concerns": [],
    },
    # No "verdict" key: WriteAnswerOutput has no such field. The
    # machine-readable headline is rendered from the run's
    # EpistemicVerdict, not authored by the model.
    "epistemic_write_answer": {
        "title": "Research Summary",
        "answer": "Spaced repetition is effective for long-term retention.",
    },
    "epistemic_validate_answer": {
        "approved": True,
        "feedback": [],
    },
    "epistemic_analyze_argument": {
        "premises": ["Evidence supports spacing effect"],
        "conclusion": "Spaced repetition works",
        "validity": "valid",
        "soundness": "sound",
        "fallacies": [],
    },
    "epistemic_investigate_claim": {
        "intents": [
            "follow-up evidence-search angle for spaced repetition effectiveness"
        ],
        "reasoning": "Default fake_runner output: one intent.",
    },
    "epistemic_record_decision": {
        "statement": "Spaced repetition is effective",
        "justification": "Supported by evidence from multiple domains",
    },
    "epistemic_assess_evidence_quality": {
        "source_credibility": 0.7,
        "relevance": 0.8,
        "specificity": 0.6,
        "recency_appropriate": 0.7,
        "justification": "Moderate quality source with relevant content",
    },
    "epistemic_generate_counterquery": {
        "query": "failed replication of test claim",
        "framing": "replication_failures",
    },
    "epistemic_evaluate_counterargument": {
        "relevance": 0.6,
        "specificity": 0.5,
        "evidence_backed": 0.5,
        "source_credibility": 0.5,
        "category": "interpretation",
        "justification": "Moderate counterargument quality",
    },
    "epistemic_classify_evidence_domain": {
        "method_type": "observational",
        "data_source": "primary",
        "temporal_approach": "cross_sectional",
        "causal_role": "phenomenological",
        "confidence": 0.7,
        "justification": "Classified based on evidence content",
    },
    "epistemic_check_pairwise_independence": {
        "independent": True,
        "rationale": "Different research groups and methodologies",
    },
    "epistemic_classify_prediction": {
        "prediction_type": "empirical",
        "specificity": 0.6,
        "success_criteria": "Measurable improvement observed",
        "failure_criteria": "No significant change detected",
        "time_horizon": "6 months",
        "justification": "Empirical prediction based on claim scope",
    },
    "epistemic_identify_testable_aspect": {
        "testable_dimension": "Blood pressure should decrease by 5-10mmHg",
        "observation_type": "quantitative",
    },
    "epistemic_specify_prediction": {
        "expected_observation": "Systolic BP decreases by 5-10mmHg",
        "conditions": "In adults with mild hypertension",
        "timeframe": "Within 3 months of regular consumption",
        "measurability": "quantitative",
    },
    "epistemic_define_falsification": {
        "falsification_criterion": "No change in blood pressure after 6 months of regular consumption",
    },
    "epistemic_contrastive_evaluation": {
        "better_claim": "A",
        "distinguishing_observation": "Testing distinguishing observation",
        "confidence": 0.7,
    },
    "epistemic_cross_claim_consistency": {
        "conflicts": False,
        "tension_point": "",
    },
    "epistemic_integrate_evidence": {
        "verdict": "supports",
        "confidence": 0.75,
        "reasoning": "Evidence collectively supports the claim through convergent indirect evidence",
    },
    # IBE pipeline (4-stage decomposition replacing epistemic_integrate_evidence
    # in active runs). Defaults are calibrated so a default run picks
    # candidate A (supports) over B (contradicts) with moderate confidence.
    "epistemic_propose_one_candidate": {
        "done": True,  # default: trigger fallback to default 3-candidate set
        "verdict": None,
        "description": None,
    },
    "epistemic_score_candidate_loveliness": {
        "loveliness": 0.7,
        "reasoning": "Default mock: clean mechanism, scope match.",
    },
    "epistemic_score_candidate_likeliness": {
        "likeliness": 0.7,
        "reasoning": "Default mock: fits supporting evidence.",
    },
    "epistemic_select_best_explanation": {
        "chosen_candidate_id": "A",
        "runner_up_candidate_id": "B",
        "confidence": 0.75,
        "reasoning": "Default mock: A dominates B on combined loveliness × likeliness.",
    },
    "epistemic_screen_relevance": {
        "is_relevant": True,
        "reason": "Evidence is relevant to the research question",
    },
    "epistemic_extract_assertion": {
        "assertion": "Test assertion from evidence",
    },
    "epistemic_draft_claim": {
        "statement": "Test claim from assertions",
        "scope": "General",
        "direction": "supports",
    },
    "epistemic_judge_evidence": {
        "verdict": "supports",
        "reasoning": "Test judgment",
        "distribution": [0.8, 0.15, 0.05],
    },
    # Phase 1 of top-down decomposition: returns a verificatory-style
    # decomposition with 3 sub-investigations combined via AND. Tests
    # exercising specific decomposition shapes should pass overrides.
    "epistemic_decompose_question": {
        "sub_investigations": [
            {
                "id": "A",
                "seed_claim": "There is a plausible mechanism for the claim.",
                "rationale": "Mechanism is load-bearing for the claim.",
            },
            {
                "id": "B",
                "seed_claim": "Direct empirical observation is consistent with the claim.",
                "rationale": "Existence proof — required for the claim to hold.",
            },
            {
                "id": "C",
                "seed_claim": "The observed effect is not a methodological artifact.",
                "rationale": "Methodological soundness check.",
            },
        ],
        "combination_rule": "AND",
        "rationale": "Mechanism + observation + artifact-control jointly settle the question.",
    },
}


@pytest.fixture
async def store(tmp_path):
    """Fresh DocumentStore backed by a temp directory."""
    from andamentum.document_store import DocumentStore

    s = DocumentStore.for_database("test", db_dir=tmp_path)
    await s.initialize()
    return s


@pytest.fixture
async def repo(store):
    """EpistemicRepository backed by a real DocumentStore."""
    return EpistemicRepository(store)


@pytest.fixture
def mock_runner():
    """MockAgentRunner with empty responses."""
    return MockAgentRunner()


@pytest.fixture
def fake_runner():
    """FakeAgentRunner with default canned responses."""
    return FakeAgentRunner()


@pytest.fixture
async def failing_repo(store):
    """FailingRepo backed by a real DocumentStore."""
    return FailingRepo(store)


@pytest.fixture
def partially_failing_runner():
    """PartiallyFailingRunner factory — call with fail_on set."""

    def _factory(fail_on: set[str], fallback: FakeAgentRunner | None = None):
        return PartiallyFailingRunner(fail_on=fail_on, fallback_runner=fallback)

    return _factory


class _StubbedJudge:
    """Recorder for a monkeypatched ``llm_judge`` elicitation."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []


@pytest.fixture
def stub_llm_judge(monkeypatch):
    """Monkeypatch ``llm_judge``'s single model-call seam, offline.

    Returns a factory: call it with a ``(meets, partial, fails)`` belief
    triple (or an exception instance to raise) and it installs the stub,
    returning a recorder whose ``.calls`` holds one dict per elicitation —
    ``model``, ``temperature``, ``criterion``, ``output``, ``context``.

    The patch is applied to ``andamentum.llm_judge.elicit`` — the MODULE
    object — because ``panel`` calls it as ``elicit.elicit_criterion_score``,
    so patching a name imported into ``panel`` would not be seen. This fixture
    lives in the shared conftest so no epistemic test can accidentally reach a
    live model on the second-opinion path.
    """
    from andamentum.llm_judge import elicit as _elicit

    def _install(
        triple: tuple[int, int, int] = (5, 5, 90),
        *,
        reasoning: str = "stub judgment",
        raises: BaseException | None = None,
    ) -> _StubbedJudge:
        recorder = _StubbedJudge()

        async def _fake(output, criterion, context, *, model, temperature):
            recorder.calls.append(
                {
                    "model": model,
                    "temperature": temperature,
                    "criterion": criterion.name,
                    "output": output,
                    "context": context,
                }
            )
            if raises is not None:
                raise raises
            meets, partial, fails = triple
            return SimpleNamespace(
                reasoning=reasoning, meets=meets, partial=partial, fails=fails
            )

        monkeypatch.setattr(_elicit, "elicit_criterion_score", _fake)
        return recorder

    return _install
