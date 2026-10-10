"""Which calls may run at the same time.

A local Ollama model serves one request at a time and holds gigabytes while it
does, so two calls that both use one are run one after the other: a call holds
``local`` for its whole run. A cloud model (``-cloud`` on Ollama, or any other
provider) runs beside it, at most ``cloud_width`` at once.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Iterable
from contextlib import asynccontextmanager


def is_local(model: str) -> bool:
    """True for a model this machine runs: ``ollama:NAME`` without ``-cloud``."""
    return model.startswith("ollama:") and not model.endswith("-cloud")


class ModelQueue:
    """The lock for local models and the limit for cloud models."""

    def __init__(self, *, cloud_width: int = 4) -> None:
        self._local = asyncio.Lock()
        self._cloud = asyncio.Semaphore(cloud_width)
        self.local_waiting = 0
        self.cloud_running = 0

    @property
    def local_running(self) -> bool:
        return self._local.locked()

    @asynccontextmanager
    async def hold(self, *, local: bool) -> AsyncIterator[None]:
        """Run the body under the local lock (``local``) or a cloud slot."""
        if local:
            self.local_waiting += 1
            try:
                await self._local.acquire()
            finally:
                self.local_waiting -= 1
            try:
                yield
            finally:
                self._local.release()
            return
        async with self._cloud:
            self.cloud_running += 1
            try:
                yield
            finally:
                self.cloud_running -= 1

    def hold_for(self, models: Iterable[str]):
        """:meth:`hold` for a call that uses *models*: local if any of them is."""
        return self.hold(local=any(is_local(m) for m in models))

    def state(self) -> dict[str, object]:
        return {
            "local_running": self.local_running,
            "local_waiting": self.local_waiting,
            "cloud_running": self.cloud_running,
        }
