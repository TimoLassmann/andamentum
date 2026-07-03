"""The one seam through which LLM-written code is executed — never in the forge process.

A ``SandboxPort`` runs a command out-of-process and returns a typed ``SandboxResult``.
Two real impls, plus a stub for tests:

- ``PodmanSandbox`` (the default): the command runs inside an ephemeral container
  (memory/pids caps, host filesystem mounted read-only, non-root user). A pure run gets
  ``--network none``; a node that declares a network effect runs with the network on but
  every other isolation intact. The *host* cannot be touched. If podman is unavailable, a
  *pure* run degrades to a subprocess with a loud warning; a *network* run refuses (there
  is no safe backend) — never a silent unprotected run.
- ``SubprocessSandbox``: a child process with a scrubbed env (no host secrets), POSIX
  rlimits, and a hard timeout that SIGKILLs the process group. Out-of-process, but NOT
  host-isolated, so it refuses network execution.

The backend is chosen by an explicit argument (``make_sandbox("podman")``), threaded from
the entry point — never an environment variable. Leaf worker file: ``stdlib`` + the
sibling ``schemas`` only; no graph engine.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

from .schemas import SandboxResult

#: Env vars a child needs to run Python; nothing else (no API keys, tokens, configs).
_KEEP_ENV = (
    "PATH",
    "HOME",
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "TMPDIR",
    "SystemRoot",
    "VIRTUAL_ENV",
)

_DEFAULT_IMAGE = "andamentum-forge-sandbox"


class SandboxUnavailableError(RuntimeError):
    """Network execution was requested but no host-isolating backend (container) exists.

    Raised instead of running network-capable code in a bare subprocess (not host-
    isolated). The caller turns this into an honest, bounded failure pointing at podman.
    """


class SandboxPort(Protocol):
    """Run ``argv`` out-of-process; return a typed verdict, never raise on child failure."""

    def run(
        self,
        argv: Sequence[str],
        *,
        cwd: Path | None = None,
        extra_path: Path | None = None,
        timeout: int = 30,
        mem_mb: int = 512,
        allow_network: bool = False,
    ) -> SandboxResult: ...


def _scrubbed_env(extra_path: Path | None) -> dict[str, str]:
    env = {k: os.environ[k] for k in _KEEP_ENV if k in os.environ}
    env["PYTHONHASHSEED"] = "0"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    if extra_path is not None:
        env["PYTHONPATH"] = str(extra_path)
    return env


class SubprocessSandbox:
    """Out-of-process via a child process with a scrubbed env + rlimits. Not host-isolated."""

    def run(
        self,
        argv: Sequence[str],
        *,
        cwd: Path | None = None,
        extra_path: Path | None = None,
        timeout: int = 30,
        mem_mb: int = 512,
        allow_network: bool = False,
    ) -> SandboxResult:
        if allow_network:
            raise SandboxUnavailableError(
                "this node reaches the network, which may only run behind the container sandbox (host-isolated). "
                "Use the podman backend; a bare subprocess is not isolated enough to run network code safely."
            )
        env = _scrubbed_env(extra_path)

        def _apply_limits() -> (
            None
        ):  # runs in the child after fork, before exec (POSIX)
            try:
                import resource

                cpu = max(1, int(timeout) + 1)
                for which, val in (
                    (resource.RLIMIT_CPU, cpu),
                    (resource.RLIMIT_AS, mem_mb * 1024 * 1024),
                    (resource.RLIMIT_FSIZE, 64 * 1024 * 1024),
                ):
                    try:
                        resource.setrlimit(which, (val, val))
                    except (ValueError, OSError):
                        pass
                if hasattr(resource, "RLIMIT_NPROC"):
                    try:
                        resource.setrlimit(resource.RLIMIT_NPROC, (256, 256))
                    except (ValueError, OSError):
                        pass
            except Exception:
                pass

        proc = subprocess.Popen(
            list(argv),
            cwd=str(cwd) if cwd is not None else None,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
            preexec_fn=_apply_limits if os.name == "posix" else None,
        )
        try:
            out, err = proc.communicate(timeout=timeout)
            return SandboxResult(
                stdout=out or "",
                stderr=err or "",
                exit_code=proc.returncode,
                timed_out=False,
            )
        except subprocess.TimeoutExpired:
            _kill_group(proc)
            out, err = proc.communicate()
            return SandboxResult(
                stdout=out or "", stderr=err or "", exit_code=-9, timed_out=True
            )


def _kill_group(proc: subprocess.Popen[str]) -> None:
    import signal

    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        proc.kill()


class PodmanSandbox:
    """Out-of-process inside an ephemeral, host-isolated container. The strong tier."""

    def __init__(self, image: str = _DEFAULT_IMAGE) -> None:
        self.image = image

    def run(
        self,
        argv: Sequence[str],
        *,
        cwd: Path | None = None,
        extra_path: Path | None = None,
        timeout: int = 30,
        mem_mb: int = 512,
        allow_network: bool = False,
    ) -> SandboxResult:
        if shutil.which("podman") is None:
            # Fail loud, never a silent downgrade. Silently running LLM-authored code in an
            # unisolated subprocess when the caller asked for the host-isolated default is a
            # security downgrade with no signal — exactly the pattern the project forbids.
            # If the caller genuinely accepts no host isolation, they pass --sandbox subprocess.
            raise SandboxUnavailableError(
                "podman is not on PATH, so LLM-authored code cannot run host-isolated. forge "
                "refuses to silently downgrade to an unisolated subprocess. Install podman and build "
                "the image (`podman build -t andamentum-forge-sandbox -f "
                "src/andamentum/forge/Containerfile .`), or pass `--sandbox subprocess` to explicitly "
                "accept running without host isolation."
            )

        raw_mount = extra_path or cwd
        if raw_mount is None:
            raise SandboxUnavailableError(
                "podman execution needs a mount point (cwd or extra_path); none was given."
            )

        # Podman bind mounts require ABSOLUTE paths; resolve so a caller may pass a
        # relative dest, and the same absolute path is used inside and out.
        mount = raw_mount.resolve()
        workdir = cwd.resolve() if cwd is not None else mount
        pypath = extra_path.resolve() if extra_path is not None else mount
        network = [] if allow_network else ["--network", "none"]
        inner = (
            ["python", *list(argv)[1:]]
            if argv and argv[0] == sys.executable
            else list(argv)
        )
        cmd = [
            "podman", "run", "--rm",
            *network,
            "--cap-drop=all",  # drop all Linux capabilities — untrusted code needs none
            "--security-opt=no-new-privileges",  # no setuid/setgid escalation
            f"--memory={mem_mb}m",
            "--pids-limit=256",
            "--cpus=2",  # bound CPU so a runaway body can't peg the host
            "-v", f"{mount}:{mount}:ro",
            "-w", str(workdir),
            "-e", f"PYTHONPATH={pypath}",
            "-e", "PYTHONDONTWRITEBYTECODE=1",
            self.image,
            *inner,
        ]  # fmt: skip
        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True, timeout=timeout + 10
            )
            return SandboxResult(
                stdout=proc.stdout or "",
                stderr=proc.stderr or "",
                exit_code=proc.returncode,
            )
        except subprocess.TimeoutExpired:
            return SandboxResult(
                stderr="podman run exceeded the time limit",
                exit_code=-9,
                timed_out=True,
            )


def make_sandbox(
    backend: str = "podman", *, image: str = _DEFAULT_IMAGE
) -> SandboxPort:
    """Build a ``SandboxPort`` for the named backend (default ``podman``)."""
    if backend == "podman":
        return PodmanSandbox(image=image)
    if backend == "subprocess":
        return SubprocessSandbox()
    raise ValueError(
        f"unknown sandbox backend {backend!r}; expected 'podman' or 'subprocess'"
    )
