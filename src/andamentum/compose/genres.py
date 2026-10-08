"""Genre packs: what each kind of document needs, as data.

One code path serves every genre. The orchestrator reads a ``Genre``; nothing
branches on a genre name. A genre is a skeleton (sections with purposes and word
ranges), a word range, a review mode and a block of style notes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, model_validator

OutputFormat = Literal["docx", "md", "html", "txt"]
ResearchMode = Literal["web", "scholarly", "both"]
WHETSTONE_DOCUMENT_TYPES = (
    "academic",
    "external_communication",
    "essay",
    "tutorial",
    "creative",
    "general",
)


class SectionSpec(BaseModel):
    title: str
    purpose: str
    required: bool = True
    words: tuple[int, int]


class Genre(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    description: str
    sections: list[SectionSpec]
    flexible_outline: bool
    word_range: tuple[int, int]
    cites: bool
    figures: bool
    review: Literal["checklist", "review"]
    whetstone_document_type: str
    reviewer: str | None
    max_grade_level: float | None
    reader_question_hints: list[str]
    style_notes: str
    outputs: list[OutputFormat]
    provenance_required: bool = False
    # Where ``--research`` looks. Unset: the web for a genre that does not
    # cite, web and scholarly sources together for one that does.
    research_mode: ResearchMode | None = None

    @property
    def research(self) -> ResearchMode:
        """The research mode this genre asks for (see ``research_mode``)."""
        if self.research_mode is not None:
            return self.research_mode
        return "both" if self.cites else "web"

    @model_validator(mode="after")
    def _consistent(self) -> Genre:
        if not self.sections:
            raise ValueError(f"genre {self.name!r}: needs at least one section")
        if self.whetstone_document_type not in WHETSTONE_DOCUMENT_TYPES:
            raise ValueError(
                f"genre {self.name!r}: whetstone_document_type "
                f"{self.whetstone_document_type!r} is not one of "
                f"{', '.join(WHETSTONE_DOCUMENT_TYPES)}"
            )
        if not self.outputs:
            raise ValueError(f"genre {self.name!r}: needs at least one output format")
        lo, hi = self.word_range
        if not 0 < lo <= hi:
            raise ValueError(f"genre {self.name!r}: bad word_range {self.word_range}")
        titles = [s.title.casefold() for s in self.sections]
        if len(set(titles)) != len(titles):
            raise ValueError(f"genre {self.name!r}: duplicate section titles")
        for s in self.sections:
            if not 0 < s.words[0] <= s.words[1]:
                raise ValueError(
                    f"genre {self.name!r}: section {s.title!r} has bad words {s.words}"
                )
        return self


_NO_INVENTION = (
    "Use only what the facts support. Never invent a number, name, date or "
    "claim; where a specific is needed and no fact gives it, write "
    "[TO FILL: what is needed]."
)

EMAIL = Genre(
    name="email",
    description="A short email that gets one thing done.",
    sections=[
        SectionSpec(
            title="Body",
            purpose="Open with the ask or the news, give the one reason that matters, "
            "state the deadline if there is one, and end on the next step.",
            words=(50, 300),
        )
    ],
    flexible_outline=False,
    word_range=(50, 300),
    cites=False,
    figures=False,
    review="checklist",
    whetstone_document_type="general",
    reviewer=None,
    max_grade_level=10.0,
    reader_question_hints=[
        "What is being asked of the reader?",
        "By when must the reader act?",
        "What happens next?",
    ],
    style_notes="Plain, direct, courteous. Short paragraphs. No greeting or sign-off "
    "boilerplate beyond one line each. No headings, no bullet walls. " + _NO_INVENTION,
    outputs=["txt", "md"],
    research_mode="web",
)

MEMO = Genre(
    name="memo",
    description="A decision memo: what is going on, the options, what to do.",
    sections=[
        SectionSpec(
            title="Summary",
            purpose="The decision asked for and the recommended answer, in a few sentences.",
            words=(40, 150),
        ),
        SectionSpec(
            title="Background",
            purpose="Only the context a reader needs to follow the options.",
            words=(60, 300),
        ),
        SectionSpec(
            title="Options or Analysis",
            purpose="The realistic options and what each costs and gains, compared fairly.",
            words=(100, 450),
        ),
        SectionSpec(
            title="Recommendation",
            purpose="The chosen option and the reasons it wins over the others.",
            words=(50, 200),
        ),
        SectionSpec(
            title="Next steps",
            purpose="Who does what by when if the recommendation is accepted.",
            words=(30, 150),
        ),
    ],
    flexible_outline=False,
    word_range=(300, 1200),
    cites=False,
    figures=False,
    review="review",
    whetstone_document_type="external_communication",
    reviewer="reader",
    max_grade_level=None,
    reader_question_hints=[
        "What decision is being asked for?",
        "What is the recommendation?",
        "What are the next steps and who owns them?",
    ],
    style_notes="Busy readers. Lead with the answer. Concrete over abstract; every "
    "option names its cost. " + _NO_INVENTION,
    outputs=["docx", "md"],
    research_mode="web",
)

REPORT = Genre(
    name="report",
    description="A findings report for readers who need to act on the evidence.",
    sections=[
        SectionSpec(
            title="Executive summary",
            purpose="The question, the main findings and the recommendations, standalone.",
            words=(150, 450),
        ),
        SectionSpec(
            title="Introduction",
            purpose="Why the report exists, its scope, and how it is organised.",
            words=(150, 600),
        ),
        SectionSpec(
            title="Findings",
            purpose="What the evidence shows, grouped by theme, each claim tied to a source.",
            words=(600, 3000),
        ),
        SectionSpec(
            title="Discussion",
            purpose="What the findings mean, their limits, and how firmly they hold.",
            words=(300, 1500),
        ),
        SectionSpec(
            title="Recommendations",
            purpose="Specific actions that follow from the findings, in priority order.",
            words=(150, 800),
        ),
    ],
    flexible_outline=True,
    word_range=(1500, 6000),
    cites=True,
    figures=True,
    review="review",
    whetstone_document_type="general",
    reviewer=None,
    max_grade_level=None,
    reader_question_hints=[
        "What is the main finding?",
        "What is the most important recommendation?",
        "What is the main limitation of the evidence?",
    ],
    style_notes="Neutral, evidence-led. Every factual claim carries its evidence marker. "
    "Separate what the sources show from what you infer. " + _NO_INVENTION,
    outputs=["docx", "html", "md"],
    research_mode="scholarly",
)

ARTICLE = Genre(
    name="article",
    description="A research article in the IMRaD shape.",
    sections=[
        SectionSpec(
            title="Abstract",
            purpose="Background, aim, method, main result and conclusion in one paragraph.",
            words=(120, 300),
        ),
        SectionSpec(
            title="Introduction",
            purpose="The gap, why it matters, and the aim of this work.",
            words=(300, 1200),
        ),
        SectionSpec(
            title="Methods",
            purpose="What was done, in enough detail that a reader could repeat it.",
            words=(400, 2000),
        ),
        SectionSpec(
            title="Results",
            purpose="What was found, with the numbers, in the order of the aims. No interpretation.",
            words=(500, 2500),
        ),
        SectionSpec(
            title="Discussion",
            purpose="What the results mean, how they sit with prior work, limits and next steps.",
            words=(500, 2000),
        ),
    ],
    flexible_outline=True,
    word_range=(2500, 8000),
    cites=True,
    figures=True,
    review="review",
    whetstone_document_type="academic",
    reviewer=None,
    max_grade_level=None,
    reader_question_hints=[
        "What question does the work address?",
        "What is the main result?",
        "What is the main limitation?",
    ],
    style_notes="Scholarly and precise. Hedge exactly as far as the evidence allows and "
    "no further. Methods in past tense. Results report; Discussion interprets. "
    + _NO_INVENTION,
    outputs=["docx", "md"],
    research_mode="scholarly",
)

HOWTO = Genre(
    name="howto",
    description="A how-to guide that gets a reader from nothing to a working result.",
    sections=[
        SectionSpec(
            title="Goal",
            purpose="What the reader will have at the end, and who this is for.",
            words=(30, 150),
        ),
        SectionSpec(
            title="Prerequisites",
            purpose="Exactly what the reader needs before step one: tools, access, knowledge.",
            words=(30, 250),
        ),
        SectionSpec(
            title="Steps",
            purpose="Numbered steps in the order they are done, one action each, with expected outcome.",
            words=(120, 1600),
        ),
        SectionSpec(
            title="Verification",
            purpose="How the reader confirms it worked.",
            words=(30, 250),
        ),
        SectionSpec(
            title="Troubleshooting",
            purpose="The failures a reader will most likely hit and the fix for each.",
            words=(50, 600),
        ),
    ],
    flexible_outline=False,
    word_range=(300, 3000),
    cites=False,
    figures=False,
    review="review",
    whetstone_document_type="tutorial",
    reviewer=None,
    max_grade_level=None,
    reader_question_hints=[
        "What does the reader need before starting?",
        "What is the first step?",
        "How does the reader know it worked?",
    ],
    style_notes="Imperative mood, second person. One action per step. Name every command, "
    "path and setting exactly as the sources give it. " + _NO_INVENTION,
    outputs=["md", "docx"],
    research_mode="web",
)

PROPOSAL = Genre(
    name="proposal",
    description="A business proposal that makes the case for a solution and its cost.",
    sections=[
        SectionSpec(
            title="Summary",
            purpose="The problem, the proposed solution, the cost and the ask, in one page.",
            words=(100, 350),
        ),
        SectionSpec(
            title="Problem",
            purpose="The client's problem in their terms, and what it costs them today.",
            words=(120, 600),
        ),
        SectionSpec(
            title="Proposed solution",
            purpose="What will be delivered and how it removes the problem.",
            words=(200, 1000),
        ),
        SectionSpec(
            title="Benefits",
            purpose="What the client gains, tied to evidence, not adjectives.",
            words=(100, 500),
        ),
        SectionSpec(
            title="Cost and timeline",
            purpose="What it costs, what is included, and when each stage lands.",
            words=(100, 500),
        ),
        SectionSpec(
            title="Risks",
            purpose="What could go wrong, how likely, and the mitigation for each.",
            words=(100, 500),
        ),
        SectionSpec(
            title="Next steps",
            purpose="What the client does to accept, and what happens after.",
            words=(40, 200),
        ),
    ],
    flexible_outline=False,
    word_range=(800, 4000),
    cites=True,
    figures=False,
    review="review",
    whetstone_document_type="external_communication",
    reviewer="adversary",
    max_grade_level=None,
    reader_question_hints=[
        "What problem does this solve for the client?",
        "What does it cost and how long does it take?",
        "What is the main risk?",
    ],
    style_notes="Client-centred and concrete. Promise only what the facts support; a risk "
    "named plainly builds more trust than one avoided. " + _NO_INVENTION,
    outputs=["docx", "md"],
    research_mode="both",
)

GRANT = Genre(
    name="grant",
    description="A generic research grant application. The call text, passed as a source "
    "or as guidelines, decides section names, order and limits.",
    sections=[
        SectionSpec(
            title="Summary",
            purpose="The problem, the aims, the approach and the expected impact, standalone.",
            words=(150, 400),
        ),
        SectionSpec(
            title="Aims",
            purpose="Two to four specific, testable aims that together answer the central question.",
            words=(150, 600),
        ),
        SectionSpec(
            title="Significance",
            purpose="Why the problem matters and what is missing today, with evidence.",
            words=(300, 1500),
        ),
        SectionSpec(
            title="Approach",
            purpose="How each aim will be met: design, methods, analysis, and what if it fails.",
            words=(500, 3500),
        ),
        SectionSpec(
            title="Team and environment",
            purpose="Who does the work, why they are placed to do it, and what resources exist.",
            words=(150, 800),
        ),
        SectionSpec(
            title="Timeline and milestones",
            purpose="Stages and the checkable milestone that ends each.",
            words=(100, 600),
        ),
        SectionSpec(
            title="Budget justification",
            purpose="Each cost line and why the work needs it.",
            words=(100, 800),
        ),
        SectionSpec(
            title="Expected outcomes and impact",
            purpose="What exists at the end and who benefits.",
            words=(150, 800),
        ),
    ],
    flexible_outline=True,
    word_range=(1500, 10000),
    cites=True,
    figures=True,
    review="review",
    whetstone_document_type="academic",
    reviewer="adversary",
    max_grade_level=None,
    reader_question_hints=[
        "What is the central question and why does it matter?",
        "What will be done, in what order?",
        "What exists at the end of the funding period?",
    ],
    style_notes="Follow the call text where it names sections, limits or criteria. State "
    "only what the sources support. Budget figures, names, dates and track-record "
    "claims that the sources do not give are marked [TO FILL: what is needed], "
    "never invented. " + _NO_INVENTION,
    outputs=["docx", "md"],
    research_mode="scholarly",
    provenance_required=True,
)

GENRES: dict[str, Genre] = {
    g.name: g for g in (EMAIL, MEMO, REPORT, ARTICLE, HOWTO, PROPOSAL, GRANT)
}


def genre_for(name: str) -> Genre:
    """Resolve a built-in pack by name, or load a genre from a YAML file path."""
    if name in GENRES:
        return GENRES[name]
    path = Path(name)
    if path.suffix.lower() in {".yaml", ".yml"}:
        if not path.is_file():
            raise ValueError(f"genre file not found: {name}")
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError(f"genre file {name} must hold a YAML mapping")
        return Genre.model_validate(data)
    raise ValueError(
        f"unknown genre {name!r}; valid names: {', '.join(GENRES)} "
        "(or a path to a .yaml genre file)"
    )
