"""DecisionModel: the open() checks, the shared lock, limits, chunking, usage, logging."""

from __future__ import annotations

import asyncio
import warnings

import httpx
import pytest

from andamentum.core.decisions import (
    DecisionModel,
    DecisionModelUnavailable,
    DecisionRequestRejected,
    DecisionRequestTooLarge,
    Question,
    make_decision_model,
)
from andamentum.core.decisions.testing import (
    STUB_DIGEST,
    STUB_NAME,
    StubDecisions,
    noul_answer,
)

CLAIM = Question(type="noul", instructions="Is it a claim?")
URL = "http://stub.invalid"


def stub(p: float = 0.8, **kw) -> StubDecisions:
    return StubDecisions(
        lambda state, questions: {k: noul_answer(p) for k in questions}, **kw
    )


async def open_model(server: StubDecisions, **kw) -> DecisionModel:
    return await DecisionModel.open(
        server.name, base_url=URL, transport=server.transport(), **kw
    )


async def test_open_reads_the_digest_and_accepts_both_name_forms() -> None:
    server = stub()
    async with await open_model(server) as model:
        assert model.name == STUB_NAME and model.id == f"ollama:{STUB_NAME}"
        assert model.digest == STUB_DIGEST
        assert model.capabilities == ["completion", "decision"]
    async with await DecisionModel.open(
        f"ollama:{STUB_NAME}", base_url=URL, transport=server.transport()
    ) as again:
        assert again.name == STUB_NAME
    handle = await make_decision_model(
        STUB_NAME, base_url=URL, transport=server.transport()
    )
    assert handle.digest == STUB_DIGEST
    await handle.aclose()


async def test_old_servers_are_refused() -> None:
    with pytest.raises(DecisionModelUnavailable, match="0.35 or later"):
        await open_model(stub(version="0.34.9"))
    assert (await open_model(stub(version="0.35.0-rc1"))).digest == STUB_DIGEST


async def test_a_missing_model_says_what_to_pull() -> None:
    server = stub()
    with pytest.raises(DecisionModelUnavailable, match="ollama pull other:1b"):
        await DecisionModel.open("other:1b", base_url=URL, transport=server.transport())


async def test_the_decision_capability_is_required() -> None:
    server = stub(capabilities=["completion", "embedding"])
    with pytest.raises(DecisionModelUnavailable, match="'decision' capability"):
        await open_model(server)
    model = await open_model(server, require_capability=False)
    assert model.digest == STUB_DIGEST


async def test_absent_capabilities_are_accepted_with_a_warning() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/show":
            return httpx.Response(200, json={"details": {}})
        if request.url.path == "/api/version":
            return httpx.Response(200, json={"version": "0.35.0"})
        return httpx.Response(
            200, json={"models": [{"name": STUB_NAME, "digest": "d1"}]}
        )

    with pytest.warns(UserWarning, match="reports no capabilities"):
        model = await DecisionModel.open(
            STUB_NAME, base_url=URL, transport=httpx.MockTransport(handler)
        )
    assert model.capabilities is None and model.digest == "d1"


async def test_an_unreachable_server_is_a_decision_error() -> None:
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    with pytest.raises(DecisionModelUnavailable, match="ollama serve"):
        await DecisionModel.open(
            STUB_NAME, base_url=URL, transport=httpx.MockTransport(refuse)
        )
    model = DecisionModel(
        STUB_NAME, base_url=URL, transport=httpx.MockTransport(refuse)
    )
    with pytest.raises(DecisionModelUnavailable, match="cannot reach"):
        await model.decide("s", {"q": CLAIM})


async def test_decide_usage_log_and_totals() -> None:
    records: list[dict[str, object]] = []
    server = stub(0.7)
    model = await open_model(server, log=records.append)
    result = await model.decide_with_usage("a sentence", {"a": CLAIM, "b": CLAIM})
    assert result.answers["a"].p_true == 0.7
    assert (result.usage.input_tokens, result.usage.output_tokens) == (10, 2)
    await model.decide("again", {"a": CLAIM})
    assert model.calls == 2 and model.usage.input_tokens == 20
    first = records[0]
    assert first["model"] == STUB_NAME and first["digest"] == STUB_DIGEST
    assert first["questions"] == 2 and first["input_tokens"] == 10
    assert isinstance(first["ms"], float)


async def test_a_failed_request_is_logged_and_raised() -> None:
    records: list[dict[str, object]] = []
    transport = httpx.MockTransport(lambda r: httpx.Response(400, text="too long"))
    model = DecisionModel(
        STUB_NAME, base_url=URL, transport=transport, log=records.append
    )
    with pytest.raises(DecisionRequestRejected):
        await model.decide("s", {"q": CLAIM})
    assert "too long" in str(records[0]["error"]) and model.calls == 0


async def test_the_first_call_gets_the_load_timeout() -> None:
    seen: list[float | None] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.extensions["timeout"]["read"])
        return httpx.Response(
            200, json={"answers": {"q": {"type": "noul", "noul": 0.5}}}
        )

    model = DecisionModel(
        STUB_NAME,
        base_url=URL,
        timeout=5.0,
        load_timeout=77.0,
        transport=httpx.MockTransport(handler),
    )
    await model.decide("s", {"q": CLAIM})
    await model.decide("s", {"q": CLAIM})
    assert seen == [77.0, 5.0]


async def test_concurrent_callers_are_serialised_across_handles() -> None:
    active = 0
    peak = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0.01)
        active -= 1
        return httpx.Response(
            200, json={"answers": {"q": {"type": "noul", "noul": 0.5}}}
        )

    transport = httpx.MockTransport(handler)
    handles = [
        DecisionModel(STUB_NAME, base_url=URL, transport=transport) for _ in range(3)
    ]
    await asyncio.gather(
        *(h.decide("s", {"q": CLAIM}) for h in handles for _ in range(4))
    )
    assert peak == 1
    assert sum(h.calls for h in handles) == 12


async def test_servers_do_not_share_a_lock() -> None:
    active = 0
    peak = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0.01)
        active -= 1
        return httpx.Response(
            200, json={"answers": {"q": {"type": "noul", "noul": 0.5}}}
        )

    transport = httpx.MockTransport(handler)
    one = DecisionModel(STUB_NAME, base_url="http://one.invalid", transport=transport)
    two = DecisionModel(STUB_NAME, base_url="http://two.invalid", transport=transport)
    await asyncio.gather(one.decide("s", {"q": CLAIM}), two.decide("s", {"q": CLAIM}))
    assert peak == 2


async def test_decide_many_asks_one_state_at_a_time_in_order() -> None:
    server = StubDecisions(
        lambda state, questions: {
            k: noul_answer(0.9 if "yes" in str(state) else 0.1) for k in questions
        }
    )
    model = server.model()
    answers = await model.decide_many(["yes 1", "no 2", "yes 3"], CLAIM)
    assert [a.p_true for a in answers] == [0.9, 0.1, 0.9]
    assert [state for state, _ in server.calls] == ["yes 1", "no 2", "yes 3"]
    assert all(len(questions) == 1 for _, questions in server.calls)


async def test_ask_all_chunks_at_64_questions() -> None:
    server = stub()
    model = server.model()
    questions = {f"q{i}": CLAIM for i in range(150)}
    answers = await model.ask_all("one state", questions)
    assert list(answers) == list(questions)
    assert [len(q) for _, q in server.calls] == [64, 64, 22]


async def test_ask_all_chunks_by_body_size() -> None:
    server = stub()
    model = server.model()
    long = Question(type="noul", instructions="x" * 20_000)
    answers = await model.ask_all("s", {f"q{i}": long for i in range(7)})
    assert len(answers) == 7
    assert len(server.calls) == 3  # three of 20 KB fit in 64 KiB
    with pytest.raises(DecisionRequestTooLarge):
        await model.ask_all("s" * 70_000, {"q": CLAIM})


async def test_decide_validates_before_sending() -> None:
    server = stub()
    model = server.model()
    with pytest.raises(ValueError, match="at least one"):
        await model.decide("s", {})
    with pytest.raises(DecisionRequestTooLarge):
        await model.decide("s", {f"q{i}": CLAIM for i in range(65)})
    with pytest.raises(DecisionRequestTooLarge, match="bytes"):
        await model.decide("x" * 70_000, {"q": CLAIM})
    assert server.calls == []


async def test_no_capability_warning_when_declared() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        await open_model(stub())
