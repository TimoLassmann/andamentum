"""Provenance: everything needed to say what, exactly, was measured.

The load-bearing pin is the MODEL DIGEST. ``gemma4:26b-nvfp4`` is a mutable tag;
``c8656f50...`` is not. A run that cannot name the digest of the weights it
questioned has not recorded its instrument.

Deliberately NOT recorded: the full environment. Only an allowlisted slice
(DOCUMENT_STORE_DIR, OLLAMA_BASE_URL, TZ, LANG) is captured, and any key matching
KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL is never read at all.

Run:  uv run python -m experiments.docstore_deferred_ingestion.scripts.provenance
"""

from __future__ import annotations

import json
import os
import platform
import re
import subprocess
import sys
from importlib import metadata as importlib_metadata
from typing import Any

import httpx

from . import _common as C

SCHEMA = "andamentum.experiment.provenance/1"

#: Only these environment variables are ever read into the record.
ENV_ALLOWLIST = ("DOCUMENT_STORE_DIR", "OLLAMA_BASE_URL", "TZ", "LANG")

#: Never touched, not even for a presence check.
SECRET_PATTERN = re.compile(r"KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL", re.IGNORECASE)

PACKAGES = (
    "andamentum",
    "docling",
    "trafilatura",
    "rapidocr-onnxruntime",
    "aiosqlite",
    "sqlite-vec",
    "pydantic",
    "pydantic-ai",
    "pydantic-graph",
    "httpx",
    "snakemake",
    "numpy",
    "matplotlib",
)


def _run(cmd: list[str], *, timeout: float = 20.0) -> str:
    """Run a command, returning stdout (stripped) or '' — never raises."""
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.SubprocessError):
        return ""
    return out.stdout.strip()


WORKING_TREE_PATCH = C.RESULTS / "working_tree.patch"

#: Files whose sha256 identifies the measuring apparatus itself.
#: ``captions/*.rst`` is in here because those files are not decoration: they are
#: the prose the Snakemake HTML report puts in front of a reader beside each
#: number. Text that explains a result is part of the apparatus that publishes
#: it, and it should be identifiable by the same one digest.
HARNESS_GLOBS = ("Snakefile", "config.yaml", "scripts/*.py", "captions/*.rst")


def git_facts() -> dict[str, Any]:
    """HEAD, branch, dirty flag, porcelain — AND the diff itself when dirty.

    A recorded sha with ``dirty: true`` and no diff on disk names code that
    cannot be recovered. That is not cosmetic: the previous edition of this run
    was measured against an UNCOMMITTED one-line change to
    ``document_store/fts_query.py``, without which a stranger checking out the
    recorded commit gets a crash in claim (e) rather than the published answer,
    while the run_id embeds the sha and reads as if it identified the code.

    So when the tree is dirty the full ``git diff HEAD`` is written to
    ``results/working_tree.patch``, hashed into this record, and REQUIRED by
    ``validate_results``. The run then carries its own instrument.
    """
    porcelain = _run(["git", "-C", str(C.REPO_ROOT), "status", "--porcelain"])
    entries = [line for line in porcelain.splitlines() if line.strip()]
    facts: dict[str, Any] = {
        "head": _run(["git", "-C", str(C.REPO_ROOT), "rev-parse", "HEAD"]),
        "head_short": C.git_short_sha(),
        "branch": _run(["git", "-C", str(C.REPO_ROOT), "rev-parse", "--abbrev-ref", "HEAD"]),
        "dirty": bool(entries),
        "status_porcelain": entries,
        "working_tree_patch": None,
        "working_tree_patch_sha256": None,
        "working_tree_patch_stat": None,
    }
    if not entries:
        if WORKING_TREE_PATCH.exists():
            WORKING_TREE_PATCH.unlink()
        return facts

    diff = _run(["git", "-C", str(C.REPO_ROOT), "diff", "HEAD"], timeout=60.0)
    WORKING_TREE_PATCH.parent.mkdir(parents=True, exist_ok=True)
    WORKING_TREE_PATCH.write_text(diff + "\n" if diff else "")
    facts["working_tree_patch"] = str(WORKING_TREE_PATCH.relative_to(C.EXP_DIR))
    facts["working_tree_patch_sha256"] = C.sha256_file(WORKING_TREE_PATCH)
    facts["working_tree_patch_stat"] = _run(
        ["git", "-C", str(C.REPO_ROOT), "diff", "HEAD", "--stat"], timeout=60.0
    ).splitlines()
    facts["working_tree_patch_note"] = (
        "tracked-file modifications only — `git diff HEAD` does not carry untracked "
        "files, which the porcelain list above names with '??'"
    )
    return facts


def harness_facts() -> dict[str, Any]:
    """One sha256 identifying the Snakefile + every harness script.

    MANIFEST.json hashes the artefacts; nothing hashed the code that produced
    them, so there was no route from a published number back to the apparatus.
    """
    files: list[dict[str, str]] = []
    for pattern in HARNESS_GLOBS:
        for path in sorted(C.EXP_DIR.glob(pattern)):
            if not path.is_file() or "__pycache__" in path.parts:
                continue
            files.append(
                {
                    "path": str(path.relative_to(C.EXP_DIR)),
                    "sha256": C.sha256_file(path),
                }
            )
    aggregate = C.sha256_text(
        "\n".join(f"{f['path']}:{f['sha256']}" for f in files)
    )
    return {
        "n_files": len(files),
        "files": files,
        "harness_sha256": aggregate,
        "meaning": (
            "sha256 over the sorted per-file digests of the Snakefile, config.yaml and "
            "every scripts/*.py — one value that identifies the measuring apparatus"
        ),
    }


def machine_facts() -> dict[str, Any]:
    """Host identity — enough to know a re-run happened somewhere else."""
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python_version": sys.version,
        "python_executable": sys.executable,
        "cpu_count": os.cpu_count(),
        "load_average": list(os.getloadavg()),
    }


def sqlite_facts() -> dict[str, Any]:
    """sqlite version plus an explicit FTS5 probe (the store cannot work without it)."""
    import sqlite3

    conn = sqlite3.connect(":memory:")
    try:
        conn.execute("CREATE VIRTUAL TABLE probe USING fts5(body)")
        fts5 = True
    except sqlite3.OperationalError:
        fts5 = False
    finally:
        conn.close()
    return {
        "sqlite_version": sqlite3.sqlite_version,
        # sqlite3.version is deprecated (3.12) and removed (3.14) — read it only
        # if it exists rather than pinning the record to a doomed attribute.
        "pysqlite_version": getattr(sqlite3, "version", None),
        "fts5_available": fts5,
    }


def package_versions() -> dict[str, str | None]:
    """Installed version of every package whose behaviour this run depends on."""
    versions: dict[str, str | None] = {}
    for name in PACKAGES:
        try:
            versions[name] = importlib_metadata.version(name)
        except importlib_metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def tool_versions() -> dict[str, str]:
    return {
        "uv": _run(["uv", "--version"]),
        "snakemake": _run(["uv", "run", "snakemake", "--version"], timeout=90.0),
    }


def ollama_facts(root_url: str, wanted: list[str]) -> dict[str, Any]:
    """Ollama version and the pinned models' digests. FAILS LOUD if one is absent.

    An experiment that silently runs against a model it did not intend has no
    measurement at all, so a missing pin is fatal here rather than a warning.
    """
    with httpx.Client(timeout=20.0) as client:
        version = client.get(f"{root_url}/api/version").json()
        tags = client.get(f"{root_url}/api/tags").json()

    by_name = {m["name"]: m for m in tags.get("models", [])}
    pinned: dict[str, Any] = {}
    missing: list[str] = []
    for name in wanted:
        model = by_name.get(name)
        if model is None:
            missing.append(name)
            continue
        details = model.get("details", {}) or {}
        pinned[name] = {
            "name": model.get("name"),
            "digest": model.get("digest"),
            "size_bytes": model.get("size"),
            "quantization_level": details.get("quantization_level"),
            "parameter_size": details.get("parameter_size"),
            "family": details.get("family"),
            "modified_at": model.get("modified_at"),
        }
    if missing:
        raise RuntimeError(
            f"Pinned model(s) not present in ollama /api/tags: {missing}. "
            f"Available: {sorted(by_name)}. Refusing to record provenance for models "
            "this host cannot run."
        )
    return {"version": version, "pinned_models": pinned}


def ollama_ps(root_url: str) -> Any:
    """Currently-loaded models — recorded around model rules so a cold reload is
    not silently charged to whichever arm happened to run second."""
    try:
        with httpx.Client(timeout=10.0) as client:
            return client.get(f"{root_url}/api/ps").json()
    except Exception as exc:  # noqa: BLE001 — advisory only, must never abort a rule
        return {"error": f"{type(exc).__name__}: {exc}"}


def env_slice() -> dict[str, str | None]:
    """The allowlisted environment slice. Secret-shaped names are never read."""
    out: dict[str, str | None] = {}
    for key in ENV_ALLOWLIST:
        if SECRET_PATTERN.search(key):
            continue
        out[key] = os.environ.get(key)
    return out


def build() -> dict[str, Any]:
    cfg = C.CONFIG
    root_url = cfg["models"]["ollama_root_url"]
    wanted = [
        cfg["models"]["llm"].removeprefix("ollama:"),
        cfg["models"]["embedding"],
    ]
    git = git_facts()
    lock = C.REPO_ROOT / "uv.lock"
    return {
        "run_id": f"{C.utc_now()}-{git['head_short']}",
        "experiment": "docstore_deferred_ingestion",
        "git": git,
        "harness": harness_facts(),
        "machine": machine_facts(),
        "sqlite": sqlite_facts(),
        "packages": package_versions(),
        "tools": tool_versions(),
        "uv_lock": {
            "path": str(lock),
            "sha256": C.sha256_file(lock) if lock.exists() else None,
        },
        "ollama": ollama_facts(root_url, wanted),
        "ollama_ps_at_start": ollama_ps(root_url),
        "environment_allowlisted": env_slice(),
        "config": cfg,
    }


def main() -> int:
    payload = build()
    # Written WITHOUT the standard envelope's provenance_ref self-reference —
    # this file *is* the provenance record, so pointing at itself would be
    # circular and the hash would be unstable. `written_at` is still carried:
    # only the self-reference is exempt, and validate_results requires
    # schema + written_at on EVERY artefact, this one included.
    C.PROVENANCE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with C.PROVENANCE_PATH.open("w") as fh:
        json.dump(
            {
                "schema": SCHEMA,
                "schema_version": C.SCHEMA_VERSION,
                "written_at": C.utc_now(),
                **payload,
            },
            fh,
            indent=2,
            default=str,
        )
        fh.write("\n")
    print(f"wrote {C.PROVENANCE_PATH}")
    print(f"  run_id : {payload['run_id']}")
    print(f"  harness: {payload['harness']['harness_sha256'][:16]} "
          f"({payload['harness']['n_files']} files)")
    if payload["git"]["dirty"]:
        print(
            f"  DIRTY  : working tree diff written to {WORKING_TREE_PATCH.name} "
            f"(sha256 {str(payload['git']['working_tree_patch_sha256'])[:16]})"
        )
    for name, info in payload["ollama"]["pinned_models"].items():
        print(f"  model  : {name} digest={str(info['digest'])[:16]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
