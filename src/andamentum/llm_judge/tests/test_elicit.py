"""Model-call-layer tests — offline. No real provider is ever contacted; the
pydantic-ai ``Agent.run`` is monkeypatched and ``resolve_model`` is counted.

This layer had no tests at all, which is how an unbounded cache and a
retry-on-everything fallback went unnoticed.
"""

from __future__ import annotations

import pytest
from pydantic_ai.exceptions import ModelHTTPError, UnexpectedModelBehavior

from andamentum.llm_judge import elicit
from andamentum.llm_judge.schemas import Criterion

CRITERION = Criterion(name="correctness", description="Is it correct?")


@pytest.fixture(autouse=True)
def clear_caches():
    elicit._AGENT_CACHE.clear()
    elicit._MODEL_CACHE.clear()
    yield
    elicit._AGENT_CACHE.clear()
    elicit._MODEL_CACHE.clear()


class _Result:
    def __init__(self, output):
        self.output = output


def _good_dist():
    return elicit._CriterionDist(reasoning="r", meets=80, partial=15, fails=5)


class _Runs(list):
    """Recorder for Agent.run calls; ``outcomes`` scripts what each returns
    (or raises). Exhausting ``outcomes`` falls back to a well-formed answer."""

    def __init__(self):
        super().__init__()
        self.outcomes: list[object] = []


@pytest.fixture
def runs(monkeypatch):
    calls = _Runs()

    async def fake_run(self, prompt, *, model_settings=None, **kwargs):
        calls.append(
            {
                "prompt": prompt,
                "temperature": model_settings["temperature"]
                if model_settings
                else None,
            }
        )
        outcome = calls.outcomes.pop(0) if calls.outcomes else _good_dist()
        if isinstance(outcome, Exception):
            raise outcome
        return _Result(outcome)

    monkeypatch.setattr("pydantic_ai.Agent.run", fake_run)
    return calls


# ── the resolved model owns the transport: resolve it ONCE per model id ──


async def test_the_resolved_model_is_cached_so_no_client_is_minted_per_call(
    runs, monkeypatch
):
    """`core.resolve_model` builds a NEW OllamaModel + OllamaProvider (and a
    new httpx client) on every call, and a new boto3 client for bedrock. It is
    not itself cached. Resolving per agent-cache-miss therefore leaks one
    never-closed connection pool per miss — on exactly the local-Ollama path
    this module targets. It must be resolved once per model id."""
    resolved: list[str] = []

    def counting_resolve(model: str):
        resolved.append(model)
        return "openai:gpt-5.4-nano"

    monkeypatch.setattr(elicit, "resolve_model", counting_resolve)

    # 20 calls, one model, DISTINCT criteria each -> 20 distinct agent keys.
    for i in range(20):
        await elicit.elicit_criterion_score(
            "out",
            Criterion(name=f"c{i}", description=f"d{i}"),
            None,
            model="ollama:gemma4:31b-nvfp4",
            temperature=0.0,
        )

    assert resolved == ["ollama:gemma4:31b-nvfp4"], (
        f"resolved the model {len(resolved)} times — each one is a fresh, "
        "never-closed HTTP client"
    )


async def test_distinct_model_ids_are_each_resolved_once(runs, monkeypatch):
    resolved: list[str] = []
    monkeypatch.setattr(
        elicit,
        "resolve_model",
        lambda m: (resolved.append(m), "openai:gpt-5.4-nano")[1],
    )

    for model in ["m1", "m2", "m1", "m2", "m1"]:
        await elicit.elicit_criterion_score(
            "out", CRITERION, None, model=model, temperature=0.0
        )

    assert sorted(resolved) == ["m1", "m2"]


# ── the agent cache must be bounded ──────────────────────────────────────


async def test_the_agent_cache_cannot_grow_without_bound(runs, monkeypatch):
    """`build_compare_instructions` folds the criterion NAMES into the
    instructions string, which is part of the cache key — so a long-running
    scorer that sees many criteria sets used to add an entry per set, forever."""
    monkeypatch.setattr(elicit, "resolve_model", lambda m: "openai:gpt-5.4-nano")

    for i in range(elicit._AGENT_CACHE_MAXSIZE * 3):
        await elicit.elicit_pairwise(
            "a",
            "b",
            None,
            [Criterion(name=f"criterion_{i}", description="d")],
            model="m",
            order="AB",
            temperature=0.0,
        )

    assert len(elicit._AGENT_CACHE) <= elicit._AGENT_CACHE_MAXSIZE


async def test_the_agent_cache_still_hits_for_the_common_access_pattern(
    runs, monkeypatch
):
    """The bound must not destroy the hit rate for what callers actually do:
    the same criteria set, reused across criteria and across panel repeats."""
    built = 0
    real_agent_init = elicit.Agent.__init__

    def counting_init(self, *args, **kwargs):
        nonlocal built
        built += 1
        return real_agent_init(self, *args, **kwargs)

    monkeypatch.setattr(elicit, "resolve_model", lambda m: "openai:gpt-5.4-nano")
    monkeypatch.setattr(elicit.Agent, "__init__", counting_init)

    for _ in range(10):
        await elicit.elicit_criterion_score(
            "out", CRITERION, None, model="m", temperature=0.7
        )

    assert built == 1, "the same (model, instructions, schema) must reuse its agent"


# ── the PromptedOutput fallback must not fire on unfixable failures ──────


async def test_a_malformed_output_falls_back_to_prompted_output(runs, monkeypatch):
    monkeypatch.setattr(elicit, "resolve_model", lambda m: "openai:gpt-5.4-nano")
    runs.outcomes.extend([UnexpectedModelBehavior("bad tool call"), _good_dist()])

    result = await elicit.elicit_criterion_score(
        "out", CRITERION, None, model="m", temperature=0.0
    )
    assert result.meets == 80
    assert len(runs) == 2, "expected the tool call then the prompted retry"


async def test_a_400_falls_back_because_a_rejected_schema_is_plausible(
    runs, monkeypatch
):
    monkeypatch.setattr(elicit, "resolve_model", lambda m: "openai:gpt-5.4-nano")
    runs.outcomes.extend(
        [ModelHTTPError(status_code=400, model_name="m", body=None), _good_dist()]
    )

    result = await elicit.elicit_criterion_score(
        "out", CRITERION, None, model="m", temperature=0.0
    )
    assert result.meets == 80
    assert len(runs) == 2


@pytest.mark.parametrize(
    ("status", "what"),
    [(401, "auth"), (429, "rate limit"), (500, "server error"), (503, "unavailable")],
)
async def test_an_unfixable_http_error_is_not_retried(runs, monkeypatch, status, what):
    """Retrying these with PromptedOutput cannot help — output mode has nothing
    to do with auth, rate limiting, or a server fault. The old code retried
    every ModelHTTPError, doubling the billed requests on every hard failure
    and answering a 429 by immediately knocking again."""
    monkeypatch.setattr(elicit, "resolve_model", lambda m: "openai:gpt-5.4-nano")
    runs.outcomes.append(ModelHTTPError(status_code=status, model_name="m", body=None))

    with pytest.raises(ModelHTTPError) as exc:
        await elicit.elicit_criterion_score(
            "out", CRITERION, None, model="m", temperature=0.0
        )

    assert exc.value.status_code == status
    assert len(runs) == 1, f"a {status} ({what}) must not be retried"


async def test_a_failing_fallback_propagates_and_is_not_swallowed(runs, monkeypatch):
    monkeypatch.setattr(elicit, "resolve_model", lambda m: "openai:gpt-5.4-nano")
    runs.outcomes.extend(
        [UnexpectedModelBehavior("bad"), UnexpectedModelBehavior("still bad")]
    )

    with pytest.raises(UnexpectedModelBehavior):
        await elicit.elicit_criterion_score(
            "out", CRITERION, None, model="m", temperature=0.0
        )


# ── temperature is the mode axis: it must actually reach the model ───────


async def test_the_requested_temperature_reaches_the_model(runs, monkeypatch):
    monkeypatch.setattr(elicit, "resolve_model", lambda m: "openai:gpt-5.4-nano")

    for temp in (0.0, 0.7, 0.0, 0.7):
        await elicit.elicit_criterion_score(
            "out", CRITERION, None, model="m", temperature=temp
        )

    assert [c["temperature"] for c in runs] == [0.0, 0.7, 0.0, 0.7], (
        "a CACHED agent must not pin the temperature of its first call — the "
        "panel's whole agreement gate depends on sampling at 0.7"
    )


async def test_the_fallback_path_also_carries_the_temperature(runs, monkeypatch):
    monkeypatch.setattr(elicit, "resolve_model", lambda m: "openai:gpt-5.4-nano")
    runs.outcomes.extend([UnexpectedModelBehavior("bad"), _good_dist()])

    await elicit.elicit_criterion_score(
        "out", CRITERION, None, model="m", temperature=0.7
    )
    assert [c["temperature"] for c in runs] == [0.7, 0.7]


# ── order handling ───────────────────────────────────────────────────────


async def test_elicit_pairwise_swaps_the_shown_responses_for_the_ba_order(
    runs, monkeypatch
):
    monkeypatch.setattr(elicit, "resolve_model", lambda m: "openai:gpt-5.4-nano")
    runs.outcomes.extend(
        [
            elicit._PairwiseDist(
                reasoning="r", response_1_better=60, tie=20, response_2_better=20
            ),
            elicit._PairwiseDist(
                reasoning="r", response_1_better=60, tie=20, response_2_better=20
            ),
        ]
    )

    await elicit.elicit_pairwise(
        "AAA", "BBB", None, [CRITERION], model="m", order="AB", temperature=0.0
    )
    await elicit.elicit_pairwise(
        "AAA", "BBB", None, [CRITERION], model="m", order="BA", temperature=0.0
    )

    ab_prompt, ba_prompt = runs[0]["prompt"], runs[1]["prompt"]
    assert ab_prompt.index("AAA") < ab_prompt.index("BBB")
    assert ba_prompt.index("BBB") < ba_prompt.index("AAA")


async def test_elicit_pairwise_rejects_an_unknown_order(runs, monkeypatch):
    monkeypatch.setattr(elicit, "resolve_model", lambda m: "openai:gpt-5.4-nano")
    with pytest.raises(ValueError, match="order must be"):
        await elicit.elicit_pairwise(
            "a",
            "b",
            None,
            [CRITERION],
            model="m",
            order="XY",  # type: ignore[arg-type]  # the runtime guard is the point
            temperature=0.0,
        )
