"""SearXNG container management and health checking.

Manages the lifecycle of a Podman-based SearXNG container for web search.
Handles Podman machine startup on macOS/Windows, container creation,
health checks, and graceful shutdown.

Usage::

    from andamentum.deep_research.searxng import SearxngManager, check_health

    manager = SearxngManager()
    manager.ensure_running()

    health = await check_health("http://127.0.0.1:4070")
    if health.healthy:
        print(f"SearXNG ready ({health.response_time_ms:.0f}ms)")
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from pydantic import BaseModel

# ── Defaults ─────────────────────────────────────────────────────────────────

DEFAULT_HOST_PORT = int(os.getenv("SEARXNG_HOST_PORT", "4070"))
DEFAULT_SEARXNG_URL = os.getenv("SEARXNG_URL", f"http://127.0.0.1:{DEFAULT_HOST_PORT}")
CONTAINER_NAME = os.getenv("SEARXNG_CONTAINER", "mcp-searxng")
SEARXNG_IMAGE = os.getenv("SEARXNG_IMAGE", "docker.io/searxng/searxng:latest")
INTERNAL_PORT = 8080


class HealthCheck(BaseModel):
    """Result of a SearXNG health probe (L7: typed, not a dict grab-bag).

    ``container_running`` is only meaningful for the synchronous
    :meth:`SearxngManager.health_check` (which inspects the Podman
    container); the async :func:`check_health` probes the HTTP endpoint
    directly and leaves it at the default.
    """

    healthy: bool = False
    container_running: bool = False
    response_time_ms: float = 0.0
    error: str | None = None
    url: str = ""


class SearxngManager:
    """Manages a Podman-based SearXNG container.

    Handles the full lifecycle: Podman machine startup (macOS/Windows),
    container creation, health checking, and shutdown.

    Args:
        container: Container name.
        image: SearXNG image to use.
        host_port: Port to expose on the host.
        internal_port: Port inside the container.
        bind_host: Host to bind to (empty string for all interfaces on macOS).
    """

    def __init__(
        self,
        container: str = CONTAINER_NAME,
        image: str = SEARXNG_IMAGE,
        host_port: int = DEFAULT_HOST_PORT,
        internal_port: int = INTERNAL_PORT,
        bind_host: str | None = None,
    ) -> None:
        self.container = container
        self.image = image
        self.host_port = host_port
        self.internal_port = internal_port
        # Linux binds the published port to loopback explicitly. On macOS the
        # empty bind is required for Podman's gvproxy port-forwarding to work;
        # it is NOT an open relay — gvproxy still publishes the port on the
        # host's 127.0.0.1, so the container is not reachable from the network.
        default_bind = "" if sys.platform == "darwin" else "127.0.0.1"
        self.bind_host = (
            bind_host
            if bind_host is not None
            else os.getenv("SEARXNG_BIND_HOST", default_bind)
        )
        self.state_dir = Path(
            os.getenv("SEARXNG_STATE_DIR", Path.home() / ".cache" / "mcp-searxng")
        )
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.settings_path = self.state_dir / "settings.yml"

    # ── Low-level helpers ────────────────────────────────────────────────

    def _run(self, cmd: list[str], timeout: int = 60) -> tuple[int, str, str]:
        try:
            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
            )
        except FileNotFoundError as exc:
            return 127, "", str(exc)
        try:
            out, err = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            out, err = proc.communicate()
            return 124, out.strip(), err.strip() or "command timed out"
        return proc.returncode, out.strip(), err.strip()

    def _has_podman(self) -> bool:
        return shutil.which("podman") is not None

    def _is_podman_machine_running(self) -> bool:
        """Check if Podman machine is running (macOS/Windows only)."""
        if sys.platform not in ("darwin", "win32"):
            return True  # Linux runs natively
        code, out, _ = self._run(
            ["podman", "machine", "list", "--format", "{{.Running}}"]
        )
        return code == 0 and "true" in out.lower()

    def _start_podman_machine(self) -> None:
        """Start the Podman machine if needed (macOS/Windows only)."""
        if sys.platform not in ("darwin", "win32"):
            return

        print("Starting Podman machine (this may take a few minutes)...")
        code, out, err = self._run(["podman", "machine", "start"], timeout=180)
        if code != 0:
            if "no machine" in err.lower() or "no vm" in err.lower():
                print("   Initializing new Podman machine...")
                init_code, _, init_err = self._run(
                    ["podman", "machine", "init"], timeout=180
                )
                if init_code != 0:
                    raise RuntimeError(
                        f"Failed to initialize Podman machine: {init_err}"
                    )
                code, out, err = self._run(["podman", "machine", "start"], timeout=180)
                if code != 0:
                    raise RuntimeError(f"Failed to start Podman machine: {err or out}")
            else:
                raise RuntimeError(f"Failed to start Podman machine: {err or out}")
        print("Podman machine started")

    def _ensure_podman_machine(self) -> None:
        if not self._is_podman_machine_running():
            self._start_podman_machine()

    # ── Public API ───────────────────────────────────────────────────────

    def write_minimal_settings(self) -> None:
        """Write a minimal SearXNG settings.yml to the state directory."""
        secret = os.urandom(32).hex()
        content = (
            "use_default_settings: true\n"
            "search:\n"
            "  formats:\n"
            "    - html\n"
            "    - json\n"
            "server:\n"
            f'  secret_key: "{secret}"\n'
            "  port: 8080\n"
            "  method: GET\n"
            "ui:\n"
            "  query_in_title: true\n"
        )
        self.settings_path.write_text(content, encoding="utf-8")

    def start(self) -> None:
        """Start the SearXNG container."""
        if not self._has_podman():
            raise RuntimeError("podman is not installed or not in PATH")

        self._ensure_podman_machine()
        self.write_minimal_settings()
        self._run(["podman", "pull", self.image])

        port_spec = f"{self.host_port}:{self.internal_port}"
        if self.bind_host:
            port_spec = f"{self.bind_host}:{self.host_port}:{self.internal_port}"
        cmd = [
            "podman",
            "run",
            "--name",
            self.container,
            "--replace",
            "-d",
            "-p",
            port_spec,
            "-v",
            f"{self.settings_path}:/etc/searxng/settings.yml:ro",
            self.image,
        ]
        code, out, err = self._run(cmd)
        if code != 0:
            raise RuntimeError(f"Failed to start SearXNG: {err or out}")

    def stop(self) -> None:
        """Stop and remove the SearXNG container."""
        if not self._has_podman():
            raise RuntimeError("podman is not installed or not in PATH")
        code, out, err = self._run(["podman", "stop", self.container])
        if code != 0 and "no such container" not in (err or "").lower():
            raise RuntimeError(
                f"Failed to stop container {self.container}: {err or out}"
            )
        code, out, err = self._run(["podman", "rm", self.container])
        if code != 0 and "no such container" not in (err or "").lower():
            raise RuntimeError(
                f"Failed to remove container {self.container}: {err or out}"
            )

    def is_running(self) -> bool:
        """Check if the SearXNG container is running."""
        if not self._has_podman():
            raise RuntimeError("podman is not installed or not in PATH")
        if not self._is_podman_machine_running():
            return False
        code, out, _ = self._run(
            [
                "podman",
                "ps",
                "--filter",
                f"name={self.container}",
                "--format",
                "{{.Names}}\t{{.Status}}",
            ]
        )
        return code == 0 and out.startswith(self.container)

    def status(self) -> str:
        """Get human-readable status string."""
        if not self._has_podman():
            return "podman not found"
        machine_status = (
            "running" if self._is_podman_machine_running() else "not running"
        )
        code, out, err = self._run(
            [
                "podman",
                "ps",
                "-a",
                "--filter",
                f"name={self.container}",
                "--format",
                "{{.Names}}\t{{.Status}}",
            ]
        )
        container_status = out or err or "not found"
        return f"Podman machine: {machine_status} | Container: {container_status}"

    def ensure_running(self) -> None:
        """Start the container if it's not already running."""
        if not self.is_running():
            self.start()

    def logs(self, tail: int = 200) -> str:
        """Get container logs."""
        if not self._has_podman():
            return "podman not found"
        code, out, err = self._run(
            ["podman", "logs", self.container, "--tail", str(tail)]
        )
        return out if code == 0 else (err or out or "no logs available")

    def health_check(self, timeout: float = 5.0) -> HealthCheck:
        """Synchronous health check using urllib (no async dependencies)."""
        result = HealthCheck(url=f"http://127.0.0.1:{self.host_port}")

        try:
            result.container_running = self.is_running()
        except RuntimeError as e:
            result.error = str(e)
            return result

        if not result.container_running:
            result.error = "SearXNG container is not running"
            return result

        health_url = f"http://127.0.0.1:{self.host_port}/healthz"
        search_url = f"http://127.0.0.1:{self.host_port}/search?q=test&format=json"

        start = time.time()
        for url in (health_url, search_url):
            try:
                req = urllib.request.Request(url, method="GET")
                with urllib.request.urlopen(req, timeout=timeout) as response:
                    if response.status == 200:
                        result.healthy = True
                        result.response_time_ms = (time.time() - start) * 1000
                        return result
            except (urllib.error.URLError, urllib.error.HTTPError):
                continue

        result.error = f"SearXNG not responding at port {self.host_port}"
        return result


async def check_health(url: str | None = None, timeout: float = 5.0) -> HealthCheck:
    """Async health check for a SearXNG instance.

    Args:
        url: SearXNG URL (defaults to SEARXNG_URL env or localhost:4070).
        timeout: Request timeout in seconds.

    Returns:
        :class:`HealthCheck` with healthy, response_time_ms, error, url.
    """
    if url is None:
        url = DEFAULT_SEARXNG_URL

    result = HealthCheck(url=url)

    start = time.time()
    try:
        import httpx

        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(
                f"{url}/search", params={"q": "test", "format": "json"}
            )
            if response.status_code == 200:
                result.healthy = True
                result.response_time_ms = (time.time() - start) * 1000
            else:
                result.error = f"SearXNG returned HTTP {response.status_code}"
    except ImportError:
        # httpx not installed — fall back to urllib
        try:
            req = urllib.request.Request(
                f"{url}/search?q=test&format=json", method="GET"
            )
            with urllib.request.urlopen(req, timeout=timeout) as response:
                if response.status == 200:
                    result.healthy = True
                    result.response_time_ms = (time.time() - start) * 1000
        except Exception as e:
            result.error = f"Health check failed: {e}"
    except Exception as e:
        result.error = f"Health check failed: {e}"

    return result
