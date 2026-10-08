"""The list of andamentum tools: one entry per command.

This is the single place that names every tool. ``andamentum list`` prints
it, ``andamentum <tool>`` forwards to the tool's own command, and
``tests/test_interface.py`` checks it against ``pyproject.toml``. Importing
this module imports no tool.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

#: Whether a tool calls a language model: never, always, or for some
#: subcommands or options only.
NeedsModel = Literal["no", "yes", "some"]


@dataclass(frozen=True)
class Tool:
    name: str  # what follows ``andamentum``; the command is ``andamentum-<name>``
    summary: str  # one plain sentence: what the tool does
    module: str  # the Python module that holds the tool
    main: str  # the main Python function, as ``module.function``
    needs_model: NeedsModel

    @property
    def command(self) -> str:
        return f"andamentum-{self.name}"

    @property
    def cli(self) -> str:
        """The module whose ``main(argv)`` runs the command."""
        return f"{self.module}.cli"


TOOLS: tuple[Tool, ...] = (
    Tool(
        "harvest",
        "Convert a PDF, Word, PowerPoint, HTML page, URL or text file to markdown.",
        "andamentum.harvest",
        "andamentum.harvest.extract",
        "some",
    ),
    Tool(
        "chunker",
        "Split markdown into chunks at headings, paragraphs and sentences.",
        "andamentum.chunker",
        "andamentum.chunker.chunk",
        "no",
    ),
    Tool(
        "docstore",
        "Store documents in one SQLite file, index them and search them.",
        "andamentum.document_store",
        "andamentum.document_store.Store",
        "some",
    ),
    Tool(
        "research",
        "Research a question on the web and in scholarly sources; the report cites what it found.",
        "andamentum.deep_research",
        "andamentum.deep_research.research",
        "yes",
    ),
    Tool(
        "epistemic",
        "Answer a question or check a claim by weighing evidence for and against it.",
        "andamentum.epistemic",
        "andamentum.epistemic.ask",
        "yes",
    ),
    Tool(
        "references",
        "Check the citations and reference list of a draft, and look the references up.",
        "andamentum.references",
        "andamentum.references.check_references",
        "some",
    ),
    Tool(
        "whetstone",
        "Review your own draft and write the review as markdown, HTML or tracked changes in Word.",
        "andamentum.whetstone",
        "andamentum.whetstone.review_document",
        "yes",
    ),
    Tool(
        "proofread",
        "Check readability and style of a text; no model.",
        "andamentum.proofread",
        "andamentum.proofread.analyze",
        "no",
    ),
    Tool(
        "compose",
        "Write a document (email, memo, report, article, proposal, grant) from a task into a folder.",
        "andamentum.compose",
        "andamentum.compose.write_document",
        "yes",
    ),
    Tool(
        "word",
        "Work on an existing Word file: check it, answer reviewer comments, refresh numbers, insert citations.",
        "andamentum.word",
        "andamentum.word.check_document",
        "some",
    ),
    Tool(
        "llm-judge",
        "Score a text, or compare two texts, against criteria with one model or a panel.",
        "andamentum.llm_judge",
        "andamentum.llm_judge.judge_score",
        "yes",
    ),
    Tool(
        "figures",
        "Draw a publication-quality chart from a CSV or JSON table.",
        "andamentum.figures",
        "andamentum.figures.figure",
        "no",
    ),
    Tool(
        "vision-critique",
        "Check a rendered figure for overlapping labels, legibility and layout with a vision model.",
        "andamentum.vision_critique",
        "andamentum.vision_critique.critique_figure",
        "yes",
    ),
    Tool(
        "typeset",
        "Render markdown to a styled HTML page or PDF.",
        "andamentum.typeset",
        "andamentum.typeset.render",
        "no",
    ),
    Tool(
        "agentic-dialect",
        "Show the house style for agentic systems and check code against it.",
        "andamentum.agentic_dialect",
        "andamentum.agentic_dialect.check_code",
        "no",
    ),
    Tool(
        "forge",
        "Build a small runnable function package from a plain-language brief, tested in a sandbox.",
        "andamentum.forge",
        "andamentum.forge.run_forge",
        "yes",
    ),
)


def get_tool(name: str) -> Tool:
    """The tool called ``name`` (with or without the ``andamentum-`` prefix)."""
    bare = name.removeprefix("andamentum-")
    for t in TOOLS:
        if t.name == bare:
            return t
    raise KeyError(f"no tool {name!r}; run 'andamentum list'")


__all__ = ["TOOLS", "NeedsModel", "Tool", "get_tool"]
