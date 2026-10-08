"""Andamentum — composable agentic systems for scientific automation.

Every tool has a command (``andamentum-<tool>``) and one main Python
function. ``TOOLS`` lists them; ``andamentum list`` prints the same list.
The shared rules for commands and functions are in ``docs/interface.md``.
"""

from .tools import TOOLS, Tool

__version__ = "0.3.0"

__all__ = ["TOOLS", "Tool", "__version__"]
