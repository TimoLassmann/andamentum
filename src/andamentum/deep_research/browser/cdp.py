"""One isolated page of a Chrome over the Chrome DevTools Protocol.

``isolated_page`` is the only way the package opens a page in a Chrome: it
creates a new browser context (its own cookies, storage and cache), denies
downloads in it, opens one tab in that context and, when the block ends, closes
the tab and disposes the context. Nothing one page sets (a cookie, local storage)
is visible to the next page, whatever Chrome it talks to.
"""

from __future__ import annotations

import asyncio
import itertools
import json
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

import httpx

if TYPE_CHECKING:
    from websockets.asyncio.client import ClientConnection

CONNECT_TIMEOUT_S = 10.0
CLEANUP_TIMEOUT_S = 10.0


class CdpError(Exception):
    """The Chrome did not answer, or answered with a protocol error."""


class IsolatedPage:
    """A tab in its own browser context. Commands go to the tab's session."""

    def __init__(
        self,
        socket: ClientConnection,
        session_id: str,
        ids: itertools.count[int],
        *,
        context_id: str,
    ):
        self._socket = socket
        self._session_id = session_id
        self.context_id = context_id
        self._ids = ids
        self._loaded = asyncio.Event()

    async def call(
        self, method: str, *, session: bool = True, **params: object
    ) -> dict[str, object]:
        return await _call(
            self._socket,
            self._ids,
            method,
            params,
            session_id=self._session_id if session else None,
            on_event=self._note,
        )

    def _note(self, message: dict[str, object]) -> None:
        if (
            message.get("method") == "Page.loadEventFired"
            and message.get("sessionId") == self._session_id
        ):
            self._loaded.set()

    async def wait_loaded(self, *, timeout: float | None = None) -> bool:
        """Wait for the tab's load event; False when *timeout* passes first."""
        try:
            async with asyncio.timeout(timeout):
                while not self._loaded.is_set():
                    message = json.loads(await self._socket.recv())
                    self._note(message)
        except TimeoutError:
            return False
        return True

    async def evaluate(self, script: str) -> object:
        answer = await self.call(
            "Runtime.evaluate", expression=script, returnByValue=True
        )
        result = answer.get("result")
        return result.get("value") if isinstance(result, dict) else None


async def _call(
    socket: ClientConnection,
    ids: itertools.count[int],
    method: str,
    params: dict[str, object],
    *,
    session_id: str | None,
    on_event: Callable[[dict[str, object]], None] | None = None,
) -> dict[str, object]:
    wanted = next(ids)
    request: dict[str, object] = {"id": wanted, "method": method, "params": params}
    if session_id is not None:
        request["sessionId"] = session_id
    await socket.send(json.dumps(request))
    while True:
        message = json.loads(await socket.recv())
        if message.get("id") == wanted:
            if "error" in message:
                raise CdpError(f"{method}: {message['error']}")
            return message.get("result") or {}
        if on_event is not None:
            on_event(message)


@asynccontextmanager
async def isolated_page(cdp_url: str) -> AsyncIterator[IsolatedPage]:
    """A new tab in a new browser context of the Chrome at *cdp_url*.

    Raises ``CdpError`` when no Chrome answers at the address (other transport
    failures are ``OSError``, ``TimeoutError`` or a websockets exception).
    """
    import websockets

    base = cdp_url.rstrip("/")
    try:
        async with httpx.AsyncClient(timeout=CONNECT_TIMEOUT_S) as http:
            version = (await http.get(f"{base}/json/version")).json()
        endpoint = version["webSocketDebuggerUrl"]
    except (httpx.HTTPError, ValueError, KeyError, TypeError) as exc:
        raise CdpError(
            f"no Chrome answers at {cdp_url} ({type(exc).__name__})"
        ) from exc
    ids = itertools.count(1)
    async with websockets.connect(endpoint, max_size=None) as socket:
        context_id: str | None = None
        target_id: str | None = None
        try:
            made = await _call(
                socket, ids, "Target.createBrowserContext", {}, session_id=None
            )
            context_id = str(made["browserContextId"])
            await _call(
                socket,
                ids,
                "Browser.setDownloadBehavior",
                {
                    "behavior": "deny",
                    "browserContextId": context_id,
                    "eventsEnabled": False,
                },
                session_id=None,
            )
            created = await _call(
                socket,
                ids,
                "Target.createTarget",
                {"url": "about:blank", "browserContextId": context_id},
                session_id=None,
            )
            target_id = str(created["targetId"])
            attached = await _call(
                socket,
                ids,
                "Target.attachToTarget",
                {"targetId": target_id, "flatten": True},
                session_id=None,
            )
            page = IsolatedPage(
                socket, str(attached["sessionId"]), ids, context_id=context_id
            )
            yield page
        finally:
            await _dispose(socket, ids, target_id=target_id, context_id=context_id)


async def _dispose(
    socket: ClientConnection,
    ids: itertools.count[int],
    *,
    target_id: str | None,
    context_id: str | None,
) -> None:
    """Close the tab and dispose the context. A Chrome that has gone away takes
    its contexts with it, so a transport failure here changes nothing."""
    import websockets

    steps: list[tuple[str, dict[str, object]]] = []
    if target_id is not None:
        steps.append(("Target.closeTarget", {"targetId": target_id}))
    if context_id is not None:
        steps.append(("Target.disposeBrowserContext", {"browserContextId": context_id}))
    for method, params in steps:
        try:
            async with asyncio.timeout(CLEANUP_TIMEOUT_S):
                await _call(socket, ids, method, params, session_id=None)
        except (OSError, TimeoutError, CdpError, websockets.WebSocketException):
            return
