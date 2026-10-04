"""Route: failed checks and findings become work items, grouped for one pass.

Deterministic. A section gets at most one revise call per round, carrying all of
its items; items about the outline as a whole go to ``amend_outline``.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from pydantic import BaseModel

from andamentum.compose.bundle import Section

from .checks.deterministic import is_error, word_count
from .render import strip_markers
from .schemas import Brief, CheckResult, Outline, ReaderQuestion, SectionPlan, WorkItem

_WORD = re.compile(r"[a-z0-9]+")
_STOP = frozenset(
    "a an and are as at be by for from in is it of on or that the to was were with".split()
)
_OVERLAP = 0.5
MIN_WORD_ADJUSTMENT = 10  # a smaller length change is not worth a revise call


class Routed(BaseModel):
    """The work for one round."""

    by_section: dict[str, list[WorkItem]]
    outline: list[WorkItem]

    @property
    def items(self) -> list[WorkItem]:
        flat = [i for group in self.by_section.values() for i in group]
        return flat + self.outline


def _content_words(text: str) -> set[str]:
    return {w for w in _WORD.findall(text.casefold()) if w not in _STOP}


def plan_for_text(text: str, *, outline: Outline | None) -> SectionPlan | None:
    """The planned section whose key points carry *text*: substring either way,
    else most of its content words."""
    if outline is None or not text.strip():
        return None
    needle = text.casefold().strip()
    for plan in outline.sections:
        for point in plan.key_points:
            p = point.casefold().strip()
            if p and (needle in p or p in needle):
                return plan
    words = _content_words(text)
    if not words:
        return None
    best: tuple[float, SectionPlan] | None = None
    for plan in outline.sections:
        for point in plan.key_points:
            share = len(words & _content_words(point)) / len(words)
            if share >= _OVERLAP and (best is None or share > best[0]):
                best = (share, plan)
    return best[1] if best else None


def _section_id_for_plan(
    plan: SectionPlan, *, sections: Sequence[Section]
) -> str | None:
    for s in sections:
        if s.title.casefold() == plan.title.casefold():
            return s.id
    return None


def _item_for_text(
    text: str,
    *,
    reason: str,
    source: str,
    sections: Sequence[Section],
    outline: Outline | None,
) -> WorkItem:
    plan = plan_for_text(text, outline=outline)
    section_id = _section_id_for_plan(plan, sections=sections) if plan else None
    if section_id is None:
        return WorkItem(level="outline", section_id=None, reason=reason, source=source)
    return WorkItem(
        level="section", section_id=section_id, reason=reason, source=source
    )


def _length_items(
    check: CheckResult, *, sections: Sequence[Section], brief: Brief
) -> list[WorkItem]:
    """Section-level items that bring the whole document into the word range.

    Every section is asked to grow or shrink in proportion, aiming for the
    middle of the range, because only a section revision changes the text (an
    outline amendment alone leaves drafted sections as they are). Sections whose
    change would be under ``MIN_WORD_ADJUSTMENT`` words are left alone.
    """
    counts = {s.id: word_count(strip_markers(s.body)) for s in sections}
    total = sum(counts.values())
    if total == 0:
        return []
    lo, hi = brief.word_range
    ratio = ((lo + hi) / 2) / total
    verb = "lengthen" if ratio > 1 else "cut"
    items: list[WorkItem] = []
    for section in sections:
        now = counts[section.id]
        target = round(now * ratio)
        if abs(target - now) < MIN_WORD_ADJUSTMENT:
            continue
        items.append(
            WorkItem(
                level="section",
                section_id=section.id,
                reason=(
                    f"{check.name}: the document is {total} words, outside "
                    f"{lo}-{hi}; {verb} this section to about {target} words "
                    f"(now {now}), keeping what the plan needs"
                ),
                source=check.name,
            )
        )
    return items


def work_from_checks(
    checks: Sequence[CheckResult],
    *,
    sections: Sequence[Section],
    outline: Outline | None,
    brief: Brief,
) -> list[WorkItem]:
    """Work items for every failed error-level check except ``reader`` and ``critic``."""
    items: list[WorkItem] = []
    for check in checks:
        if not is_error(check) or check.name in ("reader", "critic"):
            continue
        if check.section_id is not None:
            for detail in check.details or [check.name]:
                items.append(
                    WorkItem(
                        level="section",
                        section_id=check.section_id,
                        reason=f"{check.name}: {detail}",
                        source=check.name,
                    )
                )
        elif check.name == "must_include":
            for detail in check.details:
                missing = detail.removeprefix("missing:").strip()
                items.append(
                    _item_for_text(
                        missing,
                        reason=f"must_include: the text must mention {missing!r}",
                        source=check.name,
                        sections=sections,
                        outline=outline,
                    )
                )
        elif check.name == "total_words" and (
            length := _length_items(check, sections=sections, brief=brief)
        ):
            items += length
        else:
            for detail in check.details or [check.name]:
                items.append(
                    WorkItem(
                        level="outline",
                        section_id=None,
                        reason=f"{check.name}: {detail}",
                        source=check.name,
                    )
                )
    return items


def work_from_reader(
    misses: Sequence[ReaderQuestion],
    *,
    sections: Sequence[Section],
    outline: Outline | None,
) -> list[WorkItem]:
    """A wrong reader answer goes to the section planned to carry that answer."""
    return [
        _item_for_text(
            miss.expected_answer,
            reason=(
                f"reader: a cold reader could not answer {miss.question!r}; "
                f"the text must state: {miss.expected_answer}"
            ),
            source="reader",
            sections=sections,
            outline=outline,
        )
        for miss in misses
    ]


def route(
    checks: Sequence[CheckResult],
    *,
    sections: Sequence[Section],
    outline: Outline | None,
    brief: Brief,
    reader_misses: Sequence[ReaderQuestion] = (),
    critic_items: Sequence[WorkItem] = (),
) -> Routed:
    """Group every work item of a round: per section (one revise call each) or outline-level."""
    items = work_from_checks(checks, sections=sections, outline=outline, brief=brief)
    items += work_from_reader(reader_misses, sections=sections, outline=outline)
    items += list(critic_items)

    seen: set[tuple[str | None, str]] = set()
    by_section: dict[str, list[WorkItem]] = {}
    outline_items: list[WorkItem] = []
    for item in items:
        key = (item.section_id, item.reason)
        if key in seen:
            continue
        seen.add(key)
        if item.level == "section" and item.section_id is not None:
            by_section.setdefault(item.section_id, []).append(item)
        else:
            outline_items.append(item)
    return Routed(by_section=by_section, outline=outline_items)
