"""The isolated Chrome: a podman container started for a research session.

``browser_session()`` builds the image when it is missing (the Containerfile
ships in ``container/``; the image is tagged by a hash of its files, so a changed
file builds a new image), starts one container, waits until Chrome answers on its
debugging port, yields a :class:`BrowserSession` and removes the container in
``finally`` (also on cancellation and ``KeyboardInterrupt``: the removal is a
blocking ``podman rm -f`` that does not depend on the event loop).

The container is the isolation (see ``container/Containerfile`` and
``container/entrypoint.sh``): a read-only root filesystem, tmpfs for the profile
(a fresh profile per container, gone with it), every capability dropped but the
three the entrypoint needs to install its egress firewall and then drops, an
unprivileged user, and a firewall that rejects every private, loopback and
link-local range, so a page cannot reach the host's 127.0.0.1, the host's other
services or the local network. The debugging port is published on 127.0.0.1 on a
random host port.

One container per ``research()`` run: the cold start measured 0.55 to 0.76 s with
the image built, small against a run's minutes. A caller that runs many
researches and wants one container opens ``browser_session()`` itself and passes
it to ``research(browser_session=...)``.
"""

from __future__ import annotations

import asyncio
import hashlib
import shutil
import subprocess
import sys
import uuid
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path

import httpx

CONTAINER_DIR = Path(__file__).parent / "container"
IMAGE_REPOSITORY = "andamentum-chrome"
# Every container we start carries this label, so a test (or a person) can find
# containers of ours: ``podman ps -a --filter label=andamentum.browser=deep_research``.
LABEL_KEY = "andamentum.browser"
LABEL_VALUE = "deep_research"
LABEL = f"{LABEL_KEY}={LABEL_VALUE}"
DEBUG_PORT = 9222
READY_TIMEOUT_S = 60.0
READY_POLL_S = 0.1
BUILD_TIMEOUT_S = 1800.0
STOP_TIMEOUT_S = 30.0
_FILES = ("Containerfile", "entrypoint.sh", "browser.sh")
_build_lock = asyncio.Lock()


class BrowserUnavailable(RuntimeError):
    """The isolated Chrome cannot be started (podman missing or not running, the
    image cannot be built, Chrome did not come up)."""


@dataclass(frozen=True)
class BrowserSession:
    """A running isolated Chrome: its debugging address on this machine and the
    name of its container."""

    cdp_url: str
    container: str


def image_tag() -> str:
    """The image's name: a hash of the files that make it."""
    digest = hashlib.sha256()
    for name in _FILES:
        digest.update(name.encode())
        digest.update((CONTAINER_DIR / name).read_bytes())
    return f"{IMAGE_REPOSITORY}:{digest.hexdigest()[:12]}"


def require_podman() -> str:
    """The path of podman, or ``BrowserUnavailable`` naming it."""
    found = shutil.which("podman")
    if found is None:
        raise BrowserUnavailable(
            "the isolated browser needs podman, and no `podman` is on the PATH "
            "(install podman; on a Mac, `brew install podman` and "
            "`podman machine init && podman machine start`)"
        )
    return found


async def _run(*args: str, timeout: float = 120.0) -> tuple[int, str, str]:
    """Run a command; (return code, stdout, stderr). A command that outlives
    *timeout* is killed and raises ``BrowserUnavailable``."""
    process = await asyncio.create_subprocess_exec(
        *args,
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        out, err = await asyncio.wait_for(process.communicate(), timeout)
    except TimeoutError as slow:
        process.kill()
        await process.wait()
        raise BrowserUnavailable(
            f"`{' '.join(args[:3])}` did not finish within {timeout:.0f} s"
        ) from slow
    except BaseException:
        process.kill()
        raise
    return (
        process.returncode if process.returncode is not None else -1,
        out.decode(errors="replace"),
        err.decode(errors="replace"),
    )


async def ensure_image(*, progress: Callable[[str], None] | None = None) -> str:
    """The tag of the image, built first when podman does not hold it."""
    podman = require_podman()
    tag = image_tag()
    async with _build_lock:
        code, _out, err = await _run(podman, "image", "exists", tag)
        if code == 0:
            return tag
        say = progress or (lambda line: print(line, file=sys.stderr, flush=True))
        say(
            f"building the isolated browser image {tag} (Debian + chromium; about "
            "two minutes, once)"
        )
        code, out, err = await _run(
            podman,
            "build",
            "--tag",
            tag,
            "--file",
            str(CONTAINER_DIR / "Containerfile"),
            str(CONTAINER_DIR),
            timeout=BUILD_TIMEOUT_S,
        )
        if code != 0:
            raise BrowserUnavailable(
                f"podman could not build the browser image (exit {code}):\n"
                f"{(err or out).strip()[-2000:]}"
            )
        say(f"built {tag}")
    return tag


def run_arguments(*, name: str, tag: str) -> list[str]:
    """The ``podman run`` arguments of one browser container."""
    return [
        "run",
        "--detach",
        "--rm",
        "--name",
        name,
        "--label",
        LABEL,
        "--read-only",
        "--user",
        "0",
        "--tmpfs",
        "/profile:rw,size=256m,mode=1777",
        "--tmpfs",
        "/tmp:rw,size=128m,mode=1777",
        "--tmpfs",
        "/home/browser:rw,size=64m,mode=1777",
        "--cap-drop=ALL",
        "--cap-add=NET_ADMIN",
        "--cap-add=SETUID",
        "--cap-add=SETGID",
        "--security-opt",
        "no-new-privileges",
        "--shm-size=256m",
        "--publish",
        f"127.0.0.1::{DEBUG_PORT}",
        tag,
    ]


def _remove(podman: str, name: str) -> None:
    """Remove the container, blocking: it must run even when the event loop is
    being torn down by a cancellation."""
    try:
        subprocess.run(
            [podman, "rm", "--force", "--time", "1", "--ignore", name],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            timeout=STOP_TIMEOUT_S,
            check=False,
        )
    except subprocess.TimeoutExpired as slow:
        raise BrowserUnavailable(
            f"podman did not remove the browser container {name} within "
            f"{STOP_TIMEOUT_S:.0f} s; remove it with `podman rm -f {name}`"
        ) from slow


async def _host_port(podman: str, name: str) -> int:
    code, out, err = await _run(podman, "port", name, str(DEBUG_PORT))
    line = out.strip().splitlines()[0] if out.strip() else ""
    if code != 0 or ":" not in line:
        raise BrowserUnavailable(
            f"podman did not report the debugging port of {name}: {(err or out).strip()}"
        )
    return int(line.rsplit(":", 1)[1])


async def _wait_ready(url: str, *, podman: str, name: str) -> None:
    deadline = asyncio.get_running_loop().time() + READY_TIMEOUT_S
    async with httpx.AsyncClient(timeout=2.0) as http:
        while True:
            try:
                response = await http.get(f"{url}/json/version")
                if response.status_code == 200:
                    return
            except httpx.HTTPError:
                pass  # not up yet; the deadline below ends the wait
            if asyncio.get_running_loop().time() >= deadline:
                _code, logs, _err = await _run(podman, "logs", "--tail", "20", name)
                raise BrowserUnavailable(
                    f"Chrome in {name} did not answer within {READY_TIMEOUT_S:.0f} s; "
                    f"its log:\n{logs.strip()}"
                )
            await asyncio.sleep(READY_POLL_S)


@asynccontextmanager
async def browser_session(
    *, progress: Callable[[str], None] | None = None
) -> AsyncIterator[BrowserSession]:
    """Start the isolated Chrome, yield its session, remove the container after.

    Raises ``BrowserUnavailable`` when podman is missing or not running, the image
    cannot be built, or Chrome does not come up (the container is removed then too).
    """
    podman = require_podman()
    tag = await ensure_image(progress=progress)
    name = f"andamentum-chrome-{uuid.uuid4().hex[:10]}"
    try:
        code, out, err = await _run(podman, *run_arguments(name=name, tag=tag))
        if code != 0:
            raise BrowserUnavailable(
                f"podman could not start the browser container (exit {code}): "
                f"{(err or out).strip()[-2000:]}"
            )
        url = f"http://127.0.0.1:{await _host_port(podman, name)}"
        await _wait_ready(url, podman=podman, name=name)
        yield BrowserSession(cdp_url=url, container=name)
    finally:
        _remove(podman, name)
