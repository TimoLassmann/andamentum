"""The Podman sandbox's copy-in transport.

Two layers:

- ``_to_container_path`` is pure and always tested — it maps a host path under the copied
  root to its in-container location and leaves everything else alone.
- The live copy-in test needs a real ``podman`` + the built ``andamentum-forge-sandbox``
  image, so it SKIPS when either is absent (CI has neither; the maintainer's box has both).
  It proves the exact regression this transport fixes: a generated package under an
  *unshared* host path (``tmp_path`` lives under ``/private/var/folders`` on macOS, which
  the podman VM does not mount) still has its tests collected and run inside the container.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from andamentum.forge.sandbox import (
    _SANDBOX_ROOT,
    PodmanSandbox,
    _to_container_path,
)


# --- pure: host path -> container path rewrite -----------------------------------------


def test_rewrite_maps_the_mount_root() -> None:
    assert _to_container_path("/home/u/work", "/home/u/work") == _SANDBOX_ROOT


def test_rewrite_maps_a_path_under_the_root() -> None:
    assert (
        _to_container_path("/home/u/work/pkg/tests", "/home/u/work")
        == f"{_SANDBOX_ROOT}/pkg/tests"
    )


def test_rewrite_leaves_unrelated_values_untouched() -> None:
    # A bare interpreter name and a pytest flag are not under the mount — unchanged.
    assert _to_container_path("python", "/home/u/work") == "python"
    assert _to_container_path("--tb=short", "/home/u/work") == "--tb=short"
    # A sibling path that merely shares a prefix string is NOT under the root.
    assert (
        _to_container_path("/home/u/workshop/x", "/home/u/work")
        == "/home/u/workshop/x"
    )


# --- live: real container, unshared host path ------------------------------------------

_IMAGE = "andamentum-forge-sandbox"


def _image_present() -> bool:
    if shutil.which("podman") is None:
        return False
    proc = subprocess.run(
        ["podman", "image", "exists", _IMAGE], capture_output=True
    )
    return proc.returncode == 0


_needs_podman = pytest.mark.skipif(
    not _image_present(),
    reason="needs podman + the andamentum-forge-sandbox image",
)


def _write_pkg(dest: Path) -> Path:
    pkg = dest / "widget"
    tests = pkg / "tests"
    tests.mkdir(parents=True)
    (pkg / "__init__.py").write_text("")
    (pkg / "core.py").write_text("def double(x):\n    return x * 2\n")
    (tests / "__init__.py").write_text("")
    (tests / "test_core.py").write_text(
        "from widget.core import double\n\n"
        "def test_double():\n    assert double(3) == 6\n"
    )
    return pkg


@_needs_podman
def test_copyin_collects_and_runs_tests_from_an_unshared_path(tmp_path: Path) -> None:
    pkg = _write_pkg(tmp_path)
    res = PodmanSandbox().run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", str(pkg / "tests")],
        cwd=tmp_path,
        extra_path=tmp_path,
        timeout=60,
        allow_network=False,
    )
    assert res.exit_code == 0, f"{res.stdout}\n{res.stderr}"
    assert "1 passed" in res.stdout


@_needs_podman
def test_pure_run_has_no_network(tmp_path: Path) -> None:
    (tmp_path / "n.py").write_text(
        "import socket\n"
        "socket.setdefaulttimeout(4)\n"
        "try:\n"
        "    socket.create_connection(('1.1.1.1', 53)); print('NET_OK')\n"
        "except OSError:\n    print('NET_BLOCKED')\n"
    )
    res = PodmanSandbox().run(
        [sys.executable, str(tmp_path / "n.py")],
        cwd=tmp_path,
        extra_path=tmp_path,
        timeout=30,
        allow_network=False,
    )
    assert "NET_BLOCKED" in res.stdout, f"{res.stdout}\n{res.stderr}"


@_needs_podman
def test_network_run_reaches_the_net(tmp_path: Path) -> None:
    (tmp_path / "n.py").write_text(
        "import socket\n"
        "socket.setdefaulttimeout(4)\n"
        "socket.create_connection(('1.1.1.1', 53)); print('NET_OK')\n"
    )
    res = PodmanSandbox().run(
        [sys.executable, str(tmp_path / "n.py")],
        cwd=tmp_path,
        extra_path=tmp_path,
        timeout=30,
        allow_network=True,
    )
    assert "NET_OK" in res.stdout, f"{res.stdout}\n{res.stderr}"
