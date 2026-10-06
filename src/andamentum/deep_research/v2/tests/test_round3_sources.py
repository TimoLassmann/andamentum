"""Patient retries for a rate-limited source (a freeze asks keyless Semantic
Scholar), on a fake clock: success after some 429s, the wait a server asks for,
giving up after three requests in a row ran out of retries, and every other
source and every live run keeping the default schedule."""

from __future__ import annotations

import httpx
import pytest

from andamentum.deep_research.v2.sources.base import (
    DEFAULT_RETRIES,
    PATIENT_GIVE_UP_AFTER,
    PATIENT_MAX_TOTAL_WAIT_S,
    PATIENT_RETRY_WAITS_S,
    SourceClient,
    SourceError,
)
from andamentum.deep_research.v2.store import TTL_SEARCH_S

from .source_stub import SourceKit

SOURCES = ("openalex", "crossref", "europepmc", "arxiv", "semantic_scholar", "web")


class Server:
    """A mock server that answers 429 until told otherwise."""

    def __init__(self) -> None:
        self.limited = True
        self.retry_after: str | None = None
        self.limited_after = 10**9  # 200s before this many requests are made

    def handler(self, request: httpx.Request) -> httpx.Response:
        if not self.limited:
            return httpx.Response(200, content=b'{"data": []}')
        headers = {"retry-after": self.retry_after} if self.retry_after else {}
        return httpx.Response(429, headers=headers, content=b"slow down")


def clients(
    *, patient: tuple[str, ...] = ("semantic_scholar",), server: Server | None = None
) -> tuple[SourceKit, SourceClient, Server]:
    server = server or Server()
    kit = SourceKit(server.handler)
    client = SourceClient(
        kit.store,
        client=kit.http,
        intervals=dict.fromkeys(SOURCES, 0.0),
        clock=kit.time.clock,
        sleep=kit.time.sleep,
        patient=patient,
    )
    return kit, client, server


async def search(client: SourceClient, source: str, n: int) -> object:
    return await client.get_json(
        source,
        f"https://api.example/{source}",
        params={"q": str(n)},
        ttl_s=TTL_SEARCH_S,
    )


def test_the_schedule_is_the_registered_one() -> None:
    assert PATIENT_RETRY_WAITS_S == (5, 10, 20, 40, 60, 60, 60, 60)
    assert PATIENT_GIVE_UP_AFTER == 3
    assert DEFAULT_RETRIES == 2  # live runs and other sources keep this


async def test_a_patient_source_succeeds_after_some_429s_with_the_scheduled_waits() -> (
    None
):
    kit, client, server = clients()
    answers = iter([429, 429, 429, 200])

    def handler(request: httpx.Request) -> httpx.Response:
        return (
            httpx.Response(200, content=b'{"data": [1]}')
            if next(answers) == 200
            else httpx.Response(429, content=b"slow down")
        )

    kit._handler = handler  # noqa: SLF001
    body = await search(client, "semantic_scholar", 1)
    assert body == {"data": [1]}
    assert len(kit.requests) == 4
    assert kit.time.sleeps == [5, 10, 20]


async def test_a_retry_after_up_to_two_minutes_is_honoured() -> None:
    kit, client, server = clients()
    server.retry_after = "90"
    with pytest.raises(SourceError):
        await search(client, "semantic_scholar", 1)
    # 90 s beats the 5, 10, 20, 40, 60 schedule; the later 60 s waits stay 90
    assert kit.time.sleeps[:3] == [90, 90, 90]
    assert max(kit.time.sleeps) == 90

    kit, client, server = clients()
    server.retry_after = "120"
    with pytest.raises(SourceError):
        await search(client, "semantic_scholar", 1)
    assert set(kit.time.sleeps) == {120}


async def test_a_retry_after_beyond_two_minutes_is_not_asked_earlier() -> None:
    kit, client, server = clients()
    server.retry_after = "500"
    with pytest.raises(SourceError) as error:
        await search(client, "semantic_scholar", 1)
    assert error.value.retry_after_s == 500
    assert kit.time.sleeps == []  # never sleeps less than the server said
    assert len(kit.requests) == 1


async def test_the_total_wait_cap_refuses_a_slow_pool_that_never_trips_the_row_rule() -> (
    None
):
    kit, client, server = clients()
    answered = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        answered["n"] += 1
        if answered["n"] % 5 == 0:  # about one answer in five
            return httpx.Response(200, content=b'{"data": []}')
        return httpx.Response(429, content=b"slow down")

    kit._handler = handler  # noqa: SLF001
    refused = False
    for n in range(200):
        try:
            await search(client, "semantic_scholar", n)
        except SourceError as exc:
            if "refused for the rest of this process" in str(exc):
                refused = True
                break
    assert refused
    assert sum(kit.time.sleeps) <= PATIENT_MAX_TOTAL_WAIT_S
    sent = len(kit.requests)
    with pytest.raises(SourceError, match="refused for the rest of this process"):
        await search(client, "semantic_scholar", 999)
    assert len(kit.requests) == sent


async def test_a_request_that_runs_out_of_retries_raises_and_counts() -> None:
    kit, client, _server = clients()
    with pytest.raises(SourceError) as error:
        await search(client, "semantic_scholar", 1)
    assert error.value.kind == "rate_limited" and error.value.status == 429
    assert len(kit.requests) == len(PATIENT_RETRY_WAITS_S) + 1  # 8 retries
    assert kit.time.sleeps == list(PATIENT_RETRY_WAITS_S)


async def test_after_three_exhausted_requests_in_a_row_the_source_is_refused() -> None:
    kit, client, _server = clients()
    for n in range(PATIENT_GIVE_UP_AFTER):
        with pytest.raises(SourceError):
            await search(client, "semantic_scholar", n)
    sent = len(kit.requests)
    slept = len(kit.time.sleeps)
    with pytest.raises(SourceError, match="refused for the rest of this process") as e:
        await search(client, "semantic_scholar", 99)
    assert e.value.kind == "rate_limited"
    assert len(kit.requests) == sent and len(kit.time.sleeps) == slept  # nothing sent


async def test_a_success_resets_the_count() -> None:
    kit, client, server = clients()
    for n in range(PATIENT_GIVE_UP_AFTER - 1):
        with pytest.raises(SourceError):
            await search(client, "semantic_scholar", n)
    server.limited = False
    await search(client, "semantic_scholar", 50)
    server.limited = True
    for n in range(PATIENT_GIVE_UP_AFTER - 1):
        with pytest.raises(SourceError, match="429"):
            await search(client, "semantic_scholar", 100 + n)
    # two more failed in a row since the success: still asked
    sent = len(kit.requests)
    with pytest.raises(SourceError, match="429"):
        await search(client, "semantic_scholar", 200)
    assert len(kit.requests) > sent


async def test_a_cached_answer_is_served_after_the_source_is_given_up() -> None:
    kit, client, server = clients()
    server.limited = False
    first = await search(client, "semantic_scholar", 1)
    server.limited = True
    for n in range(10, 10 + PATIENT_GIVE_UP_AFTER):
        with pytest.raises(SourceError):
            await search(client, "semantic_scholar", n)
    again = await search(client, "semantic_scholar", 1)  # the cached query
    assert again == first


async def test_other_sources_keep_the_default_schedule_and_are_not_refused() -> None:
    kit, client, _server = clients()
    for n in range(PATIENT_GIVE_UP_AFTER):
        with pytest.raises(SourceError):
            await search(client, "semantic_scholar", n)
    sent = len(kit.requests)
    kit.time.sleeps.clear()
    with pytest.raises(SourceError):
        await search(client, "openalex", 1)  # not patient: its own two retries
    assert len(kit.requests) - sent == DEFAULT_RETRIES + 1
    assert kit.time.sleeps == [1.0, 2.0]
    # and the count of a source that is not patient is never kept
    for n in range(2, 6):
        with pytest.raises(SourceError, match="429"):
            await search(client, "openalex", n)


async def test_a_client_without_patient_sources_never_waits_long() -> None:
    kit, client, _server = clients(patient=())
    with pytest.raises(SourceError):
        await search(client, "semantic_scholar", 1)
    assert len(kit.requests) == DEFAULT_RETRIES + 1
    assert kit.time.sleeps == [1.0, 2.0]
    for n in range(2, 8):  # never given up
        with pytest.raises(SourceError, match="429"):
            await search(client, "semantic_scholar", n)


async def test_a_server_error_exhausts_too_and_a_404_does_not_count() -> None:
    kit, client, server = clients()
    server.limited = False

    def gone(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, content=b"no such paper")

    kit._handler = gone  # noqa: SLF001
    for n in range(PATIENT_GIVE_UP_AFTER + 2):
        with pytest.raises(SourceError, match="404"):
            await search(client, "semantic_scholar", n)
    assert len(kit.requests) == PATIENT_GIVE_UP_AFTER + 2  # one each, never retried

    kit._handler = lambda request: httpx.Response(503, content=b"down")  # noqa: SLF001
    for n in range(100, 100 + PATIENT_GIVE_UP_AFTER):
        with pytest.raises(SourceError):
            await search(client, "semantic_scholar", n)
    with pytest.raises(SourceError, match="refused for the rest of this process"):
        await search(client, "semantic_scholar", 999)
