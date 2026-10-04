"""Compile the user's task into a frozen ``Brief``. One model call."""

from __future__ import annotations

from pydantic import BaseModel

from ._call import call_model
from .genres import GENRES, Genre, genre_for
from .log import ActionLog
from .schemas import Brief, ReaderQuestion, Source

MAX_KEY_MESSAGES = 6
MAX_READER_QUESTIONS = 5
MIN_READER_QUESTIONS = 2
SOURCE_PREVIEW_CHARS = 4000

BRIEF_INSTRUCTIONS = """\
You turn a writing task into a brief that a writer and an independent reviewer \
will both work from. Be faithful to the task: do not add requirements the \
writer did not ask for.

Fill the fields like this.
- genre: one of the allowed genre names. If the user message fixes the genre, \
repeat it.
- title: a short working title for the document.
- audience: who will read it and what they already know.
- purpose: what the document must make the reader do, decide or understand.
- key_messages: 1 to 6 short statements the reader must take away.
- must_include: short phrases (a name, date, figure, decision, item) that must \
literally appear in the document because the task or the sources demand them. \
Use the exact wording from the task or sources. Leave empty if none.
- constraints: tone, deadline, length or format requirements stated in the task.
- word_min, word_max: only if the task states a length; otherwise 0 and 0.
- reader_questions: 2 to 5 questions a reader who sees only the finished \
document must be able to answer, each with the expected answer in a few words. \
Cover the key messages.
- guidelines: if a source is a call for proposals, house style or similar \
rules, summarise what the document must follow; otherwise empty.
- email_subject: a subject line when the genre is email; otherwise empty.
"""


class BriefDraft(BaseModel):
    genre: str
    title: str
    audience: str
    purpose: str
    key_messages: list[str]
    must_include: list[str]
    constraints: list[str]
    word_min: int = 0
    word_max: int = 0
    reader_questions: list[ReaderQuestion]
    guidelines: str = ""
    email_subject: str = ""


def _user_message(
    task: str,
    *,
    genre: Genre | None,
    sources: list[Source],
    source_text: dict[str, str],
) -> str:
    lines = [f"TASK:\n{task}\n"]
    if genre is None:
        lines.append("GENRE: choose one of: " + ", ".join(GENRES) + ".\n")
    else:
        lines.append(f"GENRE: {genre.name} ({genre.description})")
        lines.append(
            f"Default length: {genre.word_range[0]}-{genre.word_range[1]} words."
        )
        lines.append("Questions the reader should be able to answer, as hints:")
        lines.extend(f"- {hint}" for hint in genre.reader_question_hints)
        lines.append("")
    if sources:
        lines.append("SOURCES (previews):")
        for src in sources:
            text = source_text.get(src.id, "")
            preview = text[:SOURCE_PREVIEW_CHARS]
            cut = " [preview truncated]" if len(text) > SOURCE_PREVIEW_CHARS else ""
            lines.append(f"--- {src.id}: {src.title}{cut}\n{preview}")
    else:
        lines.append("SOURCES: none.")
    return "\n".join(lines)


async def compile_brief(
    task: str,
    *,
    genre: Genre | None,
    sources: list[Source],
    source_text: dict[str, str],
    model: str,
    log: ActionLog | None = None,
) -> Brief:
    """Compile ``task`` into a frozen ``Brief``.

    With ``genre=None`` the same call picks one of the seven genre names; an
    unknown name raises. A reply with no key message or fewer than two reader
    questions raises rather than producing a brief nothing can be checked against.
    """
    draft = await call_model(
        model=model,
        instructions=BRIEF_INSTRUCTIONS,
        output_type=BriefDraft,
        user_message=_user_message(
            task, genre=genre, sources=sources, source_text=source_text
        ),
        node="brief",
        log=log,
    )
    chosen = genre if genre is not None else genre_for(draft.genre.strip().lower())
    key_messages = [m for m in draft.key_messages if m.strip()][:MAX_KEY_MESSAGES]
    questions = [
        q
        for q in draft.reader_questions
        if q.question.strip() and q.expected_answer.strip()
    ][:MAX_READER_QUESTIONS]
    if not key_messages:
        raise ValueError("compile_brief: the model returned no key messages")
    if len(questions) < MIN_READER_QUESTIONS:
        raise ValueError(
            f"compile_brief: the model returned {len(questions)} usable reader "
            f"questions; at least {MIN_READER_QUESTIONS} are needed"
        )
    word_range = chosen.word_range
    if 0 < draft.word_min <= draft.word_max:
        word_range = (draft.word_min, draft.word_max)
    return Brief(
        task=task,
        genre=chosen.name,
        title=draft.title.strip(),
        audience=draft.audience.strip(),
        purpose=draft.purpose.strip(),
        key_messages=key_messages,
        must_include=[m.strip() for m in draft.must_include if m.strip()],
        constraints=[c.strip() for c in draft.constraints if c.strip()],
        word_range=word_range,
        reader_questions=questions,
        guidelines=draft.guidelines.strip(),
        email_subject=draft.email_subject.strip() if chosen.name == "email" else "",
    )
