"""The isolated Chrome's container: image, run arguments, lifecycle. Podman is a
stub here; the live proof (a real container, the isolation) is
``test_browser_container_live.py`` under the ``browser_container`` marker."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import pytest

from .. import engine
from ..browser import container
from ..browser.container import (
    BrowserSession,
    BrowserUnavailable,
    browser_session,
    image_tag,
    require_podman,
    run_arguments,
)
from ..engine import research
from ..sources import SourceConfig, SourceError, build_sources
from ..store import ResearchStore


def test_the_containerfile_ships_with_its_scripts() -> None:
    for name in ("Containerfile", "entrypoint.sh", "browser.sh"):
        assert (container.CONTAINER_DIR / name).is_file(), name
    text = (container.CONTAINER_DIR / "Containerfile").read_text()
    assert "@sha256:" in text and "chromium" in text


def test_the_image_tag_follows_the_files(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    first = image_tag()
    assert first == image_tag() and first.startswith("andamentum-chrome:")
    for name in ("Containerfile", "entrypoint.sh", "browser.sh"):
        (tmp_path / name).write_bytes((container.CONTAINER_DIR / name).read_bytes())
    monkeypatch.setattr(container, "CONTAINER_DIR", tmp_path)
    assert image_tag() == first
    (tmp_path / "entrypoint.sh").write_text("# changed\n")
    assert image_tag() != first


def test_the_run_arguments_isolate_the_container() -> None:
    args = run_arguments(name="n", tag="t:1")
    joined = " ".join(args)
    assert args[-1] == "t:1" and "--rm" in args and "--read-only" in args
    assert f"--label {container.LABEL}" in joined
    assert "127.0.0.1::9222" in args  # loopback only, random host port
    assert "--cap-drop=ALL" in args and "no-new-privileges" in args
    assert not any(a.startswith("--network") for a in args)
    assert sum(a.startswith("--tmpfs") for a in args) == 3


def test_a_missing_podman_is_an_error_that_names_podman(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(container.shutil, "which", lambda _name: None)
    with pytest.raises(BrowserUnavailable, match="podman"):
        require_podman()


class FakePodman:
    """Stands in for the subprocess calls of ``container``."""

    def __init__(self, *, start_code: int = 0, ready: bool = True) -> None:
        self.start_code = start_code
        self.ready = ready
        self.removed: list[str] = []
        self.started: list[str] = []

    async def run(self, *args: str, timeout: float = 120.0) -> tuple[int, str, str]:
        verb = args[1]
        if verb == "image":
            return 0, "", ""
        if verb == "run":
            self.started.append(args[args.index("--name") + 1])
            return self.start_code, "id", "no such machine" if self.start_code else ""
        if verb == "port":
            return 0, "127.0.0.1:43210\n", ""
        return 0, "", ""

    def remove(self, _podman: str, name: str) -> None:
        self.removed.append(name)


@pytest.fixture
def podman(monkeypatch: pytest.MonkeyPatch) -> FakePodman:
    fake = FakePodman()
    monkeypatch.setattr(container, "require_podman", lambda: "podman")
    monkeypatch.setattr(container, "_run", fake.run)
    monkeypatch.setattr(container, "_remove", fake.remove)

    async def ready(url: str, *, podman: str, name: str) -> None:
        if not fake.ready:
            raise BrowserUnavailable("Chrome did not answer")

    monkeypatch.setattr(container, "_wait_ready", ready)
    return fake


async def test_the_session_names_the_published_port_and_removes_the_container(
    podman: FakePodman,
) -> None:
    async with browser_session() as session:
        assert session.cdp_url == "http://127.0.0.1:43210"
        assert session.container == podman.started[0]
        assert podman.removed == []
    assert podman.removed == podman.started


async def test_the_container_is_removed_when_the_block_raises(
    podman: FakePodman,
) -> None:
    with pytest.raises(RuntimeError, match="boom"):
        async with browser_session():
            raise RuntimeError("boom")
    assert podman.removed == podman.started


async def test_the_container_is_removed_when_the_task_is_cancelled(
    podman: FakePodman,
) -> None:
    inside = asyncio.Event()

    async def work() -> None:
        async with browser_session():
            inside.set()
            await asyncio.sleep(60)

    task = asyncio.create_task(work())
    await inside.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert podman.removed == podman.started and len(podman.removed) == 1


async def test_a_chrome_that_never_answers_leaves_no_container(
    podman: FakePodman,
) -> None:
    podman.ready = False
    with pytest.raises(BrowserUnavailable, match="did not answer"):
        async with browser_session():
            pytest.fail("the session must not open")
    assert podman.removed == podman.started


async def test_a_failed_start_says_why_and_removes_the_name(podman: FakePodman) -> None:
    podman.start_code = 125
    with pytest.raises(BrowserUnavailable, match="no such machine"):
        async with browser_session():
            pytest.fail("the session must not open")
    assert podman.removed == podman.started


# -- the engine and the source set ------------------------------------------------


def test_the_browser_option_without_a_running_browser_is_a_config_error() -> None:
    with pytest.raises(SourceError) as caught:
        build_sources(
            SourceConfig(browser=True, searxng_url="http://searx.test"),
            store=ResearchStore.in_memory(),
            mode="web",
        )
    assert caught.value.kind == "config" and "browser" in caught.value.reason


class EngineProbe:
    def __init__(self) -> None:
        self.events: list[str] = []
        self.browsers: list[BrowserSession | None] = []

    @asynccontextmanager
    async def session(self, **_kw: object) -> AsyncIterator[BrowserSession]:
        self.events.append("started")
        try:
            yield BrowserSession(cdp_url="http://127.0.0.1:1", container="c")
        finally:
            self.events.append("stopped")

    def build(self, config, *, store, mode, client, browser):  # noqa: ANN001
        self.browsers.append(browser)
        self.events.append("built")
        raise RuntimeError("stop here")


@pytest.mark.parametrize(
    ("mode", "browser", "starts"),
    [
        ("web", True, True),
        ("both", True, True),
        ("scholarly", True, False),
        ("web", False, False),
    ],
)
async def test_research_starts_and_stops_the_browser_only_for_web_work(
    monkeypatch: pytest.MonkeyPatch, mode: str, browser: bool, starts: bool
) -> None:
    probe = EngineProbe()
    monkeypatch.setattr(engine, "browser_session", probe.session)
    monkeypatch.setattr(engine, "build_sources", probe.build)
    with pytest.raises(RuntimeError, match="stop here"):
        await research(
            "q",
            mode=mode,  # type: ignore[arg-type]
            model="test",
            sources=SourceConfig(browser=browser, searxng_url="http://s.test"),
        )
    if starts:
        assert probe.events == ["started", "built", "stopped"]
        assert probe.browsers[0] is not None
    else:
        assert probe.events == ["built"] and probe.browsers == [None]


async def test_a_session_passed_in_is_used_and_left_running(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    probe = EngineProbe()
    monkeypatch.setattr(engine, "browser_session", probe.session)
    monkeypatch.setattr(engine, "build_sources", probe.build)
    mine = BrowserSession(cdp_url="http://127.0.0.1:2", container="mine")
    with pytest.raises(RuntimeError, match="stop here"):
        await research(
            "q",
            mode="web",
            model="test",
            sources=SourceConfig(browser=True),
            browser=mine,
        )
    assert probe.events == ["built"] and probe.browsers == [mine]
