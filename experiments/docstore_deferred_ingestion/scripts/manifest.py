"""results/MANIFEST.json — every artefact with its sha256, producing rule and schema.

Findable and Accessible in the FAIR sense: a directory copied elsewhere still
says what each file is, which rule made it, and whether it has been altered.

``--archive`` additionally copies results/ figures/ bench/ and report.* verbatim
into ``runs/<run_id>/``. results/ is flat (house style), so a re-run overwrites —
archiving is how history survives.

Run:  uv run python -m experiments.docstore_deferred_ingestion.scripts.manifest
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Any

from . import _common as C
from .schemas import ARTEFACT_GLOBS, ARTEFACTS

SCHEMA = "andamentum.experiment.docstore_deferred.manifest/1"

#: Directories walked in full, with the rule that owns them.
#:
#: ``scripts`` IS IN HERE ON PURPOSE. The manifest hashed 122 artefacts and none
#: of the code that produced them: no Snakefile, no scripts/*.py. Combined with an
#: experiment that was gitignored repo-wide, there was no route at all from a
#: published number back to the apparatus that produced it — which is exactly the
#: Reproducible leg of FAIR.
TREES = {
    "results": "various",
    "figures": "figures",
    "data": "fetch_pdfs",
    "logs": "claim_c_kill / claim_d_pause",
    "bench": "snakemake benchmark:",
    "scripts": "harness (hand-written; hashed so the apparatus is pinned too)",
    "tests": "harness (offline unit tests for the instruments)",
}

#: Never hashed into the manifest: databases are large, machine-local, and
#: deliberately disposable (dbs/ is the clean-slate target).
SKIP_DIRS = {"dbs", "pdfs", "__pycache__", ".snakemake"}

#: Files at the experiment root, hashed by name.
ROOT_FILES = ("report.md", "report.html", "Snakefile", "config.yaml", "README.md", ".gitignore")


def _spec_for(relative: str) -> dict[str, Any]:
    """Exact-path spec, falling back to a glob match.

    Without the fallback, 101 of 122 entries carried ``schema: null`` — including
    every ledger, snapshot and event log the README calls load-bearing.
    """
    import fnmatch

    spec = ARTEFACTS.get(relative)
    if spec is not None:
        return spec
    for pattern, glob_spec in ARTEFACT_GLOBS.items():
        if fnmatch.fnmatch(relative, pattern):
            return glob_spec
    return {}


def describe(path: Path) -> dict[str, Any]:
    relative = str(path.relative_to(C.EXP_DIR))
    spec = _spec_for(relative)
    return {
        "path": relative,
        "bytes": path.stat().st_size,
        "sha256": C.sha256_file(path),
        "producing_rule": spec.get("produced_by"),
        "schema": spec.get("schema"),
    }


#: Files this rule cannot honestly hash, because they are not final until after
#: it has finished. `MANIFEST.json` would have to contain its own hash; Snakemake
#: writes `bench/manifest.tsv` only once the rule body returns. Left in, each
#: records the PREVIOUS run's bytes and `--verify` reports a permanent false
#: discrepancy — on the very file the check exists to make trustworthy.
SELF_EXCLUDED = frozenset({"results/MANIFEST.json", "bench/manifest.tsv"})


def verify(manifest: dict[str, Any]) -> list[str]:
    """Re-hash every recorded artefact. Returns the discrepancies.

    The manifest is presented as the integrity record, and nothing could check
    it. ``--verify`` is that check.
    """
    problems: list[str] = []
    for entry in manifest["artefacts"]:
        path = C.EXP_DIR / entry["path"]
        if not path.exists():
            problems.append(f"{entry['path']}: recorded in the manifest but absent")
            continue
        actual = C.sha256_file(path)
        if actual != entry["sha256"]:
            problems.append(
                f"{entry['path']}: sha256 {actual[:16]} != recorded {entry['sha256'][:16]}"
            )
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", action="store_true", help="also copy into runs/<run_id>/")
    parser.add_argument(
        "--verify",
        action="store_true",
        help="re-hash every recorded artefact against MANIFEST.json; exit 1 on drift",
    )
    args = parser.parse_args()

    if args.verify:
        problems = verify(C.read_json(C.RESULTS / "MANIFEST.json"))
        for problem in problems:
            print(f"  ! {problem}")
        print(f"verify: {len(problems)} discrepanc{'y' if len(problems) == 1 else 'ies'}")
        return 1 if problems else 0

    artefacts: list[dict[str, Any]] = []
    for tree in TREES:
        root = C.EXP_DIR / tree
        if not root.exists():
            continue
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            if any(part in SKIP_DIRS for part in path.relative_to(C.EXP_DIR).parts):
                continue
            if str(path.relative_to(C.EXP_DIR)) in SELF_EXCLUDED:
                continue
            artefacts.append(describe(path))

    for name in ROOT_FILES:
        path = C.EXP_DIR / name
        if path.exists():
            artefacts.append(describe(path))

    harness = [a for a in artefacts if a["path"].startswith(("scripts/", "Snakefile"))]
    payload = {
        "run_id": C.run_id(),
        "n_artefacts": len(artefacts),
        "n_harness_files": len(harness),
        "harness_sha256": C.sha256_text(
            "\n".join(f"{a['path']}:{a['sha256']}" for a in sorted(harness, key=lambda a: a["path"]))
        ),
        "harness_note": (
            "one digest over the Snakefile and every harness script. Cross-check it "
            "against provenance.json's git.harness_sha256 — they are computed "
            "independently over the same file set"
        ),
        "n_unschemad": len([a for a in artefacts if a["schema"] is None]),
        "skipped": sorted(SKIP_DIRS),
        "self_excluded": sorted(SELF_EXCLUDED),
        "self_excluded_why": (
            "neither is final when this rule hashes: MANIFEST.json would have to "
            "contain its own hash, and Snakemake writes bench/manifest.tsv only "
            "after the rule body returns. Including them would record the PREVIOUS "
            "run's bytes and make --verify report a permanent false discrepancy"
        ),
        "skipped_note": (
            "databases and downloaded PDFs are not hashed into the manifest: they are "
            "large, machine-local and disposable. data/REGISTRY.json carries the PDF "
            "hashes instead."
        ),
        "artefacts": artefacts,
    }
    C.write_json(C.RESULTS / "MANIFEST.json", payload, schema=SCHEMA)
    print(
        f"wrote {C.RESULTS / 'MANIFEST.json'} ({len(artefacts)} artefacts, "
        f"{payload['n_harness_files']} of them harness code, "
        f"{payload['n_unschemad']} without a documented schema)"
    )

    if args.archive:
        run_dir = C.RUNS / C.run_id().replace(":", "").replace("+", "_")
        run_dir.mkdir(parents=True, exist_ok=True)
        for tree in ("results", "figures", "bench"):
            source = C.EXP_DIR / tree
            if source.exists():
                shutil.copytree(source, run_dir / tree, dirs_exist_ok=True)
        for name in ("report.md", "report.html", "config.yaml"):
            path = C.EXP_DIR / name
            if path.exists():
                shutil.copy2(path, run_dir / name)
        print(f"archived to {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
