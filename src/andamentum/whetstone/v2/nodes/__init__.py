"""Graph nodes for whetstone v2.

Each node is a small dataclass with a single ``async def run(ctx) -> NextNode``
method, in the deep_research style. The graph file just lists them.

Phase 1: HarvestSource, ChunkAndScan
Phase 2: Skim, InvestigateLoop
Phase 3: Challenge
Phase 4: Synthesise
Phase 6: AuthorQuestions
"""

from .author_questions import AuthorQuestions
from .challenge import Challenge
from .chunk_and_scan import ChunkAndScan
from .edit_sections import EditSections
from .harvest_source import HarvestSource
from .investigate import InvestigateLoop
from .skim import Skim
from .synthesise import Synthesise

__all__ = [
    "AuthorQuestions",
    "Challenge",
    "ChunkAndScan",
    "EditSections",
    "HarvestSource",
    "InvestigateLoop",
    "Skim",
    "Synthesise",
]
