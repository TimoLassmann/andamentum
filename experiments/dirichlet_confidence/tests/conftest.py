"""Make ``experiments.dirichlet_confidence`` importable when pytest is run
from the repo root (the package lives outside ``src/`` and outside the default
``testpaths``)."""

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
