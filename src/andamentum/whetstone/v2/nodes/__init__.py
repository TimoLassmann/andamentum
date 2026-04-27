"""Graph nodes for whetstone v2.

Each node is a small dataclass with a single ``async def run(ctx) -> NextNode``
method, in the deep_research style. The graph file just lists them.

Pipeline:
  HarvestSource → ChunkAndScan → CriticalRead → ReflectAndInvestigate
                                              → EditSections (optional)
                                              → Challenge
                                              → AuthorQuestions
                                              → Synthesise
"""

from .author_questions import AuthorQuestions
from .challenge import Challenge
from .chunk_and_scan import ChunkAndScan
from .critical_read import CriticalRead
from .edit_sections import EditSections
from .harvest_source import HarvestSource
from .reflect_and_investigate import ReflectAndInvestigate
from .synthesise import Synthesise

__all__ = [
    "AuthorQuestions",
    "Challenge",
    "ChunkAndScan",
    "CriticalRead",
    "EditSections",
    "HarvestSource",
    "ReflectAndInvestigate",
    "Synthesise",
]
