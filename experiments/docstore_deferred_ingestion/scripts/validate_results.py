"""Validate every results artefact against SCHEMAS.json, plus referential integrity.

EXITS 1 on any violation. Because ``rule validate_results`` sits in the DAG,
Snakemake — not discipline — is what stops an invalid results directory being
reported as a finished run.

Three classes of check:
  1. schema     — the declared schema id matches, required keys are present
  2. envelope   — every artefact carries schema/written_at/provenance_ref
  3. referential — every arxiv_id in a result exists in the registry, every doc_id
                   in a claim exists in some fingerprint, and every
                   provenance_ref.sha256 matches the actual provenance.json

Run:  uv run python -m experiments.docstore_deferred_ingestion.scripts.validate_results
"""

from __future__ import annotations

from typing import Any

from . import _common as C
from .schemas import ARTEFACT_GLOBS, ARTEFACTS

SCHEMA = "andamentum.experiment.docstore_deferred.validation/1"

#: Artefacts produced AT or AFTER this rule. Their absence during this rule is
#: expected, not a violation — flagging them would make the validator always fail
#: on a first run and train the reader to ignore it.
PRODUCED_LATER = frozenset(
    {
        "results/validation.json",  # this rule's own output
        "results/MANIFEST.json",  # written by `manifest`, downstream of here
    }
)


def collect_doc_ids(payload: Any, sink: set[str]) -> None:
    """Recursively harvest every doc_uuid / doc_id string in a payload."""
    if isinstance(payload, dict):
        for key, value in payload.items():
            if key in ("doc_uuid", "doc_id", "target_doc_id", "good_doc_id") and isinstance(
                value, str
            ):
                sink.add(value)
            else:
                collect_doc_ids(value, sink)
    elif isinstance(payload, list):
        for item in payload:
            collect_doc_ids(item, sink)


def collect_fingerprint_doc_ids(payload: Any, sink: set[str]) -> None:
    """Harvest doc_uuids that appear inside a snapshot's document row dump."""
    if isinstance(payload, dict):
        if "documents" in payload and isinstance(payload["documents"], list):
            for row in payload["documents"]:
                if isinstance(row, dict) and isinstance(row.get("doc_uuid"), str):
                    sink.add(row["doc_uuid"])
        for value in payload.values():
            collect_fingerprint_doc_ids(value, sink)
    elif isinstance(payload, list):
        for item in payload:
            collect_fingerprint_doc_ids(item, sink)


def main() -> int:
    violations: list[str] = []
    checked: list[str] = []

    provenance_sha = (
        C.sha256_file(C.PROVENANCE_PATH) if C.PROVENANCE_PATH.exists() else None
    )
    if provenance_sha is None:
        violations.append("results/provenance.json is missing — the run has no identity")

    registry_ids: set[str] = set()
    if C.REGISTRY_PATH.exists():
        registry_ids = {p["arxiv_id"] for p in C.read_json(C.REGISTRY_PATH)["papers"]}
    else:
        violations.append("data/REGISTRY.json is missing")

    known_doc_ids: set[str] = set()
    referenced_doc_ids: set[str] = set()
    referenced_arxiv_ids: set[str] = set()

    for relative, spec in ARTEFACTS.items():
        path = C.EXP_DIR / relative
        if not path.exists():
            if relative in PRODUCED_LATER:
                continue
            violations.append(f"{relative}: absent")
            continue
        if path.suffix != ".json":
            checked.append(relative)
            continue

        payload = C.read_json(path)
        checked.append(relative)

        if payload.get("schema") != spec["schema"]:
            violations.append(
                f"{relative}: schema {payload.get('schema')!r} != {spec['schema']!r}"
            )
        for key in ("schema", "written_at"):
            if key not in payload:
                violations.append(f"{relative}: envelope key {key!r} missing")
        for key in spec["required"]:
            if key not in payload:
                violations.append(f"{relative}: required key {key!r} missing")

        ref = payload.get("provenance_ref")
        if relative != "results/provenance.json":
            if not isinstance(ref, dict):
                violations.append(f"{relative}: provenance_ref missing")
            elif ref.get("sha256") not in (None, provenance_sha):
                violations.append(
                    f"{relative}: provenance_ref.sha256 {ref.get('sha256')} does not "
                    f"match results/provenance.json ({provenance_sha}) — the artefact "
                    "was written against a different provenance record"
                )

        collect_doc_ids(payload, referenced_doc_ids)
        collect_fingerprint_doc_ids(payload, known_doc_ids)
        for key in ("arxiv_id", "target_arxiv_id", "source_arxiv_id"):
            value = payload.get(key)
            if isinstance(value, str):
                referenced_arxiv_ids.add(value)

    # --- glob-keyed artefacts ---------------------------------------------
    # Previously unchecked entirely: the ledgers, snapshots and event logs the
    # README calls load-bearing carried `schema: null` in the manifest and never
    # passed through this validator at all. JSON ones get the same envelope and
    # provenance_ref treatment as everything else; JSONL ones are checked for
    # parseability, since a corrupt line silently truncates a count.
    from .instrument import read_jsonl

    glob_checked: dict[str, int] = {}
    for pattern, spec in ARTEFACT_GLOBS.items():
        matches = sorted(C.EXP_DIR.glob(pattern))
        glob_checked[pattern] = len(matches)
        for path in matches:
            relative = str(path.relative_to(C.EXP_DIR))
            if path.suffix == ".jsonl":
                try:
                    read_jsonl(path)
                except ValueError as exc:
                    violations.append(f"{relative}: {exc}")
                continue
            if path.suffix != ".json":
                continue
            payload = C.read_json(path)
            checked.append(relative)
            if payload.get("schema") != spec["schema"]:
                violations.append(
                    f"{relative}: schema {payload.get('schema')!r} != {spec['schema']!r}"
                )
            for key in ("schema", "written_at"):
                if key not in payload:
                    violations.append(f"{relative}: envelope key {key!r} missing")
            for key in spec["required"]:
                if key not in payload:
                    violations.append(f"{relative}: required key {key!r} missing")
            ref = payload.get("provenance_ref")
            if not isinstance(ref, dict):
                violations.append(f"{relative}: provenance_ref missing")
            elif ref.get("sha256") not in (None, provenance_sha):
                violations.append(
                    f"{relative}: provenance_ref.sha256 does not match provenance.json"
                )
            collect_fingerprint_doc_ids(payload, known_doc_ids)

    # A DIRTY TREE MUST CARRY ITS DIFF. A recorded sha with dirty=true and no
    # patch on disk names code nobody can recover, which is the reproducibility
    # leg of FAIR failing quietly rather than loudly.
    if C.PROVENANCE_PATH.exists():
        git = C.read_json(C.PROVENANCE_PATH).get("git") or {}
        if git.get("dirty"):
            patch = C.RESULTS / "working_tree.patch"
            if not patch.exists():
                violations.append(
                    "provenance records git.dirty=true but results/working_tree.patch "
                    "is absent — the measured code cannot be reconstructed"
                )
            elif C.sha256_file(patch) != git.get("working_tree_patch_sha256"):
                violations.append(
                    "results/working_tree.patch does not match the sha256 recorded in "
                    "provenance.json"
                )

    unknown_arxiv = sorted(referenced_arxiv_ids - registry_ids)
    if unknown_arxiv:
        violations.append(
            f"arxiv ids referenced but not in the registry: {unknown_arxiv}"
        )

    orphan_docs = sorted(referenced_doc_ids - known_doc_ids)
    if orphan_docs:
        violations.append(
            f"{len(orphan_docs)} doc_id(s) referenced but absent from every fingerprint: "
            f"{orphan_docs[:5]}"
        )

    payload = {
        "verdict": "PASS" if not violations else "FAIL",
        "n_artefacts_checked": len(checked),
        "artefacts_checked": checked,
        "glob_artefacts_checked": glob_checked,
        "n_violations": len(violations),
        "violations": violations,
        "referential": {
            "registry_arxiv_ids": sorted(registry_ids),
            "referenced_arxiv_ids": sorted(referenced_arxiv_ids),
            "n_doc_ids_in_fingerprints": len(known_doc_ids),
            "n_doc_ids_referenced": len(referenced_doc_ids),
        },
    }
    C.write_json(C.RESULTS / "validation.json", payload, schema=SCHEMA)

    print(f"validated {len(checked)} artefacts; {len(violations)} violation(s)")
    for violation in violations:
        print(f"  ! {violation}")
    return 1 if violations else 0


if __name__ == "__main__":
    raise SystemExit(main())
