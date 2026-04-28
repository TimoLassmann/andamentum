"""Graph nodes for whetstone v2.

Each node is a small dataclass with a single ``async def run(ctx) -> NextNode``
method, in the deep_research style. The graph file just lists them.

Review pipeline (mode="review"):
  HarvestSource → ChunkAndScan → CriticalRead → ReflectAndInvestigate
                                              → EditSections (optional)
                                              → Challenge
                                              → AuthorQuestions
                                              → Synthesise

Panel pipeline (mode="panel"):
  HarvestSource → ChunkAndScan → ExtractKeywords → GenerateExpertPanel
                                                 → ExpertReview
                                                 → PanelSynthesise

Guidelines pipeline (mode="guidelines"):
  HarvestSource → ChunkAndScan → ExtractCheckableItems
                                              → EvaluateGuidelineItems

Custom-criteria pipeline (mode="custom"):
  HarvestSource → ChunkAndScan → CustomReviewer
"""

from .author_questions import AuthorQuestions
from .challenge import Challenge
from .chunk_and_scan import ChunkAndScan
from .critical_read import CriticalRead
from .custom_reviewer import CustomReviewer
from .edit_sections import EditSections
from .evaluate_guideline_items import EvaluateGuidelineItems
from .expert_review import ExpertReview
from .extract_checkable_items import ExtractCheckableItems
from .extract_keywords import ExtractKeywords
from .generate_expert_panel import GenerateExpertPanel
from .harvest_source import HarvestSource
from .novelty_check import NoveltyCheck
from .panel_synthesise import PanelSynthesise
from .reflect_and_investigate import ReflectAndInvestigate
from .synthesise import Synthesise

__all__ = [
    "AuthorQuestions",
    "Challenge",
    "ChunkAndScan",
    "CriticalRead",
    "CustomReviewer",
    "EditSections",
    "EvaluateGuidelineItems",
    "ExpertReview",
    "ExtractCheckableItems",
    "ExtractKeywords",
    "GenerateExpertPanel",
    "HarvestSource",
    "NoveltyCheck",
    "PanelSynthesise",
    "ReflectAndInvestigate",
    "Synthesise",
]
