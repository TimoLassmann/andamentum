"""The forge benchmark corpus — eight briefs spanning shape and scope.

Five buildable briefs, one per control-flow grammar (sequence, branch, loop, fan-out,
stateful), and three out-of-scope briefs forge must refuse at the fitness gate (an app,
an agent, a service) — each carrying the rung-1/2 function hiding inside it as a note.
"""

from __future__ import annotations

from .types import Case

CASES: list[Case] = [
    Case(
        brief="Summarise the document into three bullet points.",
        expected="build",
        grammar="sequence",
    ),
    Case(
        brief="Classify a support ticket's urgency and route it to the matching team.",
        expected="build",
        grammar="branch",
    ),
    Case(
        brief="Research a question with web search, looping until the evidence is sufficient.",
        expected="build",
        grammar="loop",
    ),
    Case(
        brief="Given a list of article URLs, summarise each and combine them into one digest.",
        expected="build",
        grammar="fanout",
    ),
    Case(
        brief="Given my current reading list and a new chat message, return the updated list.",
        expected="build",
        grammar="stateful",
    ),
    Case(
        brief="Manage my personal reading list.",
        expected="refuse",
        grammar="none",
        note="reshape to #5: one current list + one message → the updated list",
    ),
    Case(
        brief="A chatbot that answers questions about my documents.",
        expected="refuse",
        grammar="none",
        note="reshape to: answer ONE question about the documents",
    ),
    Case(
        brief="Watch my inbox and file incoming mail into folders.",
        expected="refuse",
        grammar="none",
        note="reshape to: classify ONE email into a folder",
    ),
]
