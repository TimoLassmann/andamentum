"""Set the isolation environment BEFORE any harness module is imported.

``scripts/_common.py`` raises at import unless DOCUMENT_STORE_DIR is set and
resolves inside the experiment directory, and unless OLLAMA_BASE_URL is set. That
guard is deliberate (it is what makes it impossible to run a rule against the
user's real store), so the tests satisfy it rather than weakening it.

These tests are OFFLINE: no Ollama, no network, no document store. They test the
HARNESS — the ledger, the fingerprint, the recall metric and the prereg/analyze
contract — because an instrument you have not tested is not evidence.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

EXP_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = EXP_DIR.parent.parent

# The package lives outside src/ and outside the default testpaths, so the repo
# root has to be on sys.path for `experiments.docstore_deferred_ingestion` to
# resolve (same pattern as experiments/dirichlet_confidence/tests/conftest.py).
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("DOCUMENT_STORE_DIR", str(EXP_DIR / "dbs"))
os.environ.setdefault("OLLAMA_BASE_URL", "http://localhost:11434/v1")
