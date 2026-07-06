"""The one seam through which LLM-written code is executed — never in the forge process.

A ``SandboxPort`` runs a command out-of-process and returns a typed ``SandboxResult``.
Two real impls, plus a stub for tests:

- ``PodmanSandbox`` (the default): the command runs inside an ephemeral container
  (memory/pids caps, non-root user). The package is **copied in** over stdin (a tar stream
  extracted inside the container) rather than bind-mounted — so the container needs no host
  path shared into the VM, and the host filesystem is never exposed at all (strictly more
  isolated than a read-only mount, and portable across any podman machine config). A pure
  run gets ``--network none``; a node that declares a network effect runs with the network
  on but every other isolation intact. The *host* cannot be touched. If podman is
  unavailable, a *pure* run degrades to a subprocess with a loud warning; a *network* run
  refuses (there is no safe backend) — never a silent unprotected run.
- ``SubprocessSandbox``: a child process with a scrubbed env (no host secrets), POSIX
  rlimits, and a hard timeout that SIGKILLs the process group. Out-of-process, but NOT
  host-isolated, so it refuses network execution.

The backend is chosen by an explicit argument (``make_sandbox("podman")``), threaded from
the entry point — never an environment variable. Leaf worker file: ``stdlib`` + the
sibling ``schemas`` only; no graph engine.
"""

from __future__ import annotations

import io
import os
import shlex
import shutil
import subprocess
import sys
import tarfile
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

#: Where the copied-in package tree is unpacked inside the container. Must be writable by
#: the image's non-root ``sandbox`` user (uid 10001) — its home is the safe choice, since
#: ``/`` is root-owned. This path mirrors the host *mount root*: a host path ``{mount}/x``
#: becomes ``{_SANDBOX_ROOT}/x`` inside, and every command path is rewritten accordingly.
_SANDBOX_ROOT = "/home/sandbox/pkg"


def _tar_stream(root: Path) -> bytes:
    """Serialise ``root``'s tree into an in-memory tar (members rooted at ``.``).

    Extracted with ``tar -x -C {_SANDBOX_ROOT}`` inside the container, so a host file at
    ``root/pkg/tests`` lands at ``{_SANDBOX_ROOT}/pkg/tests``.
    """
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        tar.add(str(root), arcname=".")
    return buf.getvalue()


def _to_container_path(value: str, mount: str) -> str:
    """Rewrite a host path under ``mount`` to its in-container location.

    Anything not under ``mount`` (a bare ``python``, a pytest flag) is returned unchanged.
    """
    if value == mount:
        return _SANDBOX_ROOT
    if value.startswith(mount + os.sep):
        return _SANDBOX_ROOT + value[len(mount) :]
    return value


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
                "podman execution needs a source root (cwd or extra_path); none was given."
            )

        # The package tree is COPIED IN, not bind-mounted: `mount` is the host root we tar
        # up, `_SANDBOX_ROOT` is where it lands inside. Resolve so a caller may pass a
        # relative dest; every command path under `mount` is rewritten to the in-container
        # location. No host path is shared with the container.
        mount = raw_mount.resolve()
        mount_s = str(mount)
        workdir = str(cwd.resolve()) if cwd is not None else mount_s
        pypath = str(extra_path.resolve()) if extra_path is not None else mount_s
        workdir_c = _to_container_path(workdir, mount_s)
        pypath_c = _to_container_path(pypath, mount_s)
        network = [] if allow_network else ["--network", "none"]
        argv_c = [
            _to_container_path(a, mount_s)
            for a in (
                ["python", *list(argv)[1:]]
                if argv and argv[0] == sys.executable
                else list(argv)
            )
        ]
        # Unpack the tar from stdin, cd into the (now-existing) workdir, then hand off to
        # the real command via `exec "$@"` (so its exit code/signals propagate cleanly).
        shell = (
            f"mkdir -p {shlex.quote(_SANDBOX_ROOT)} "
            f"&& tar -xf - -C {shlex.quote(_SANDBOX_ROOT)} "
            f"&& cd {shlex.quote(workdir_c)} "
            f'&& exec "$@"'
        )
        cmd = [
            "podman", "run", "--rm", "-i",  # -i: the tar stream arrives on stdin
            *network,
            "--cap-drop=all",  # drop all Linux capabilities — untrusted code needs none
            "--security-opt=no-new-privileges",  # no setuid/setgid escalation
            f"--memory={mem_mb}m",
            "--pids-limit=256",
            "--cpus=2",  # bound CPU so a runaway body can't peg the host
            "-e", f"PYTHONPATH={pypath_c}",
            "-e", "PYTHONDONTWRITEBYTECODE=1",
            self.image,
            "sh", "-c", shell, "sh", *argv_c,  # $0=sh, $1.. = the rewritten argv
        ]  # fmt: skip
        try:
            proc = subprocess.run(
                cmd,
                input=_tar_stream(mount),
                capture_output=True,
                timeout=timeout + 10,
            )
            return SandboxResult(
                stdout=proc.stdout.decode("utf-8", "replace"),
                stderr=proc.stderr.decode("utf-8", "replace"),
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
