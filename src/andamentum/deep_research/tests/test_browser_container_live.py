"""The isolated Chrome for real: podman, the browser image, the network.

Deselected by default (marker ``browser_container``); run with
``uv run pytest -m browser_container``. The first run builds the image (about two
minutes). Needs podman running and internet access (example.com, a search engine).

What it proves (WORK.md deliverable 2): search and page read work through the
container; no container of ours is left after the session; a page in the
container cannot reach a server on the Mac's 127.0.0.1 (navigation and fetch),
nor the Mac's LAN address; a cookie set in read A is absent in read B; a download
leaves no file.
"""

from __future__ import annotations

import socket
import subprocess
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from ..browser import LABEL, browser_session, isolated_page
from ..sources import SourceClient, WebSource
from ..sources.browser_fetch import browser_page_text
from ..store import ResearchStore

pytestmark = pytest.mark.browser_container

SECRET = "hello-from-the-mac"
PUBLIC = "http://example.com/"


def containers_of_ours() -> list[str]:
    out = subprocess.run(
        ["podman", "ps", "-a", "--filter", f"label={LABEL}", "--format", "{{.Names}}"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return out.split()


class _Secret(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802
        body = SECRET.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/plain")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        return None


@pytest.fixture
def mac_server() -> Iterator[int]:
    """A server on every interface of this machine (so on 127.0.0.1 and on the
    LAN address), serving a secret."""
    server = ThreadingHTTPServer(("0.0.0.0", 0), _Secret)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server.server_address[1]
    server.shutdown()
    thread.join(timeout=5)


def lan_address() -> str | None:
    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        probe.connect(("192.0.2.1", 9))  # no packet is sent; picks the route's address
        address = probe.getsockname()[0]
    except OSError:
        return None
    finally:
        probe.close()
    return address if address and not address.startswith("127.") else None


async def test_search_and_read_work_and_no_container_is_left() -> None:
    before = containers_of_ours()
    async with browser_session() as session:
        assert len(containers_of_ours()) == len(before) + 1
        page = await browser_page_text(session.cdp_url, PUBLIC)
        assert page.title == "Example Domain" and "documentation examples" in page.text
        client = SourceClient(ResearchStore.in_memory())
        try:
            source = WebSource(browser=session, client=client)
            found = await source.search("Mount Kilimanjaro height", limit=5)
        finally:
            await client.aclose()
        print("search results:", len(found), source.take_warnings())
        assert found and all(c.work.url for c in found)
    assert containers_of_ours() == before


async def test_a_page_cannot_reach_a_server_on_the_macs_loopback(
    mac_server: int,
) -> None:
    targets = [
        f"http://host.containers.internal:{mac_server}/",
        f"http://192.168.127.254:{mac_server}/",
        f"http://127.0.0.1:{mac_server}/",
    ]
    async with browser_session() as session:
        for target in targets:
            async with isolated_page(session.cdp_url) as page:
                await page.call("Page.enable")
                await page.call("Page.navigate", url=target)
                await page.wait_loaded(timeout=15)
                shown = await page.evaluate(
                    "location.href + ' | ' + (document.body ? document.body.innerText : '')"
                )
                assert SECRET not in str(shown), target
                assert str(shown).startswith("chrome-error:"), (target, shown)
        async with isolated_page(session.cdp_url) as page:
            await page.call("Page.enable")
            await page.call("Page.navigate", url=PUBLIC)
            await page.wait_loaded(timeout=15)
            for target in targets:
                reached = await page.call(
                    "Runtime.evaluate",
                    expression=f"fetch({target!r}).then(r => r.text()).then("
                    "t => 'REACHED:' + t, e => 'blocked')",
                    awaitPromise=True,
                    returnByValue=True,
                )
                value = (reached.get("result") or {}).get("value")  # type: ignore[union-attr]
                assert value == "blocked", (target, value)


async def test_the_local_network_is_blocked_too(mac_server: int) -> None:
    lan = lan_address()
    if lan is None:
        pytest.skip("this machine has no LAN address")
    async with browser_session() as session:
        async with isolated_page(session.cdp_url) as page:
            await page.call("Page.enable")
            await page.call("Page.navigate", url=f"http://{lan}:{mac_server}/")
            await page.wait_loaded(timeout=15)
            shown = await page.evaluate(
                "location.href + ' | ' + (document.body ? document.body.innerText : '')"
            )
    print(f"LAN address {lan}: navigation showed {str(shown)[:80]!r}")
    assert SECRET not in str(shown) and str(shown).startswith("chrome-error:")


async def test_a_cookie_set_in_one_read_is_absent_in_the_next() -> None:
    async with browser_session() as session:
        async with isolated_page(session.cdp_url) as first:
            await first.call("Page.enable")
            await first.call("Page.navigate", url=PUBLIC)
            await first.wait_loaded(timeout=15)
            await first.evaluate(
                "document.cookie = 'k=secret-a; path=/'; "
                "localStorage.setItem('k', 'secret-a'); 1"
            )
            own = await first.evaluate(
                "document.cookie + '|' + localStorage.getItem('k')"
            )
            assert own == "k=secret-a|secret-a"
            # the next read starts while the first is still open, and after
            async with isolated_page(session.cdp_url) as second:
                await second.call("Page.enable")
                await second.call("Page.navigate", url=PUBLIC)
                await second.wait_loaded(timeout=15)
                during = await second.evaluate(
                    "document.cookie + '|' + localStorage.getItem('k')"
                )
        async with isolated_page(session.cdp_url) as third:
            await third.call("Page.enable")
            await third.call("Page.navigate", url=PUBLIC)
            await third.wait_loaded(timeout=15)
            after = await third.evaluate(
                "document.cookie + '|' + localStorage.getItem('k')"
            )
    assert during == "|null" and after == "|null"


_DOWNLOAD = (
    "(() => { const a = document.createElement('a'); "
    "a.href = URL.createObjectURL(new Blob(['payload'])); a.download = 'dl-blob.bin'; "
    "document.body.appendChild(a); a.click(); return 1; })()"
)


def downloaded_files(container: str) -> list[str]:
    out = subprocess.run(
        [
            "podman",
            "exec",
            "--user",
            "10001",
            container,
            "find",
            "/tmp",
            "/profile",
            "/home/browser",
            "-name",
            "dl-*",
        ],
        capture_output=True,
        text=True,
        check=False,  # find exits 1 on a directory root may not read; the list is what counts
    ).stdout
    return out.split()


async def test_a_download_leaves_no_file_and_the_control_proves_it_could() -> None:
    async with browser_session() as session:
        async with isolated_page(session.cdp_url) as page:
            await page.call("Page.enable")
            await page.call("Page.navigate", url=PUBLIC)
            await page.wait_loaded(timeout=15)
            await page.evaluate(_DOWNLOAD)
            await page.evaluate("new Promise(r => setTimeout(() => r(1), 2000))")
        assert downloaded_files(session.container) == []
        # control: the same click with downloads allowed does write a file
        async with isolated_page(session.cdp_url) as page:
            await page.call(
                "Browser.setDownloadBehavior",
                session=False,
                behavior="allow",
                downloadPath="/tmp/dltest",
                browserContextId=page.context_id,
            )
            await page.call("Page.enable")
            await page.call("Page.navigate", url=PUBLIC)
            await page.wait_loaded(timeout=15)
            await page.evaluate(_DOWNLOAD)
            await page.evaluate("new Promise(r => setTimeout(() => r(1), 2000))")
        assert any(
            f.endswith("dl-blob.bin") for f in downloaded_files(session.container)
        )
