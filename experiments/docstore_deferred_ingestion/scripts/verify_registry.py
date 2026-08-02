"""Re-hash every registered PDF and EXIT 1 on drift.

Every measurement rule takes this rule's output as an input, so Snakemake — not
operator discipline — is the gate that stops drifted inputs reaching a
measurement.

Run:  uv run python -m experiments.docstore_deferred_ingestion.scripts.verify_registry
"""

from __future__ import annotations

from typing import Any

from . import _common as C

SCHEMA = "andamentum.experiment.docstore_deferred.data_integrity/1"


def main() -> int:
    registry = C.read_json(C.REGISTRY_PATH)
    checks: list[dict[str, Any]] = []
    for paper in registry["papers"]:
        path = C.EXP_DIR / paper["path"]
        if not path.exists():
            checks.append(
                {
                    "arxiv_id": paper["arxiv_id"],
                    "ok": False,
                    "reason": f"missing file {path}",
                    "expected_sha256": paper["sha256"],
                    "actual_sha256": None,
                    "expected_bytes": paper["bytes"],
                    "actual_bytes": None,
                }
            )
            continue
        actual_sha = C.sha256_file(path)
        actual_bytes = path.stat().st_size
        ok = actual_sha == paper["sha256"] and actual_bytes == paper["bytes"]
        checks.append(
            {
                "arxiv_id": paper["arxiv_id"],
                "ok": ok,
                "reason": "" if ok else "sha256 or size differs from the registry",
                "expected_sha256": paper["sha256"],
                "actual_sha256": actual_sha,
                "expected_bytes": paper["bytes"],
                "actual_bytes": actual_bytes,
            }
        )

    failures = [c for c in checks if not c["ok"]]
    C.write_json(
        C.RESULTS / "data_integrity.json",
        {
            "verdict": "PASS" if not failures else "FAIL",
            "registry_version": registry.get("registry_version"),
            "n_checked": len(checks),
            "n_failed": len(failures),
            "checks": checks,
        },
        schema=SCHEMA,
    )
    for check in checks:
        print(f"{'ok  ' if check['ok'] else 'FAIL'} {check['arxiv_id']} {check['reason']}")
    if failures:
        print(
            "\nInput data drifted from the registry. Refusing to let a measurement run "
            "against inputs that are not the registered ones.",
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
