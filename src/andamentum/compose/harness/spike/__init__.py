"""M0 spike: can anchors, citation fields and comment threads survive Word?"""

from .make import make_spike_docx
from .probe import probe
from .survival import run_survival

__all__ = ["make_spike_docx", "probe", "run_survival"]
