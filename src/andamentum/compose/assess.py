"""One check pass over a bundle: every check the goal depends on, then routing.

The orchestrator's Check node, the ``check`` command and the evaluation harness
all call this one function, so the arms of an evaluation are judged identically.
Pure with respect to the bundle's text: it writes only review artefacts under
``reviews/`` and the action log. Model calls are sequential.
"""

from __future__ import annotations

from pydantic import BaseModel

from andamentum.compose.bundle import Bundle

from .checks.critic import CriticResult, run_critic
from .checks.deterministic import is_error, run_deterministic_checks
from .checks.grounding import Unsupported, check_grounding
from .checks.reader import ReaderResult, reader_test
from .genres import Genre
from .log import ActionLog
from .render import plain_text
from .route import Routed, route
from .schemas import Brief, CheckResult, Fact, Outline, Roles, Source


class Assessment(BaseModel):
    round: int
    checks: list[CheckResult]
    reader: ReaderResult
    critic: CriticResult
    routed: Routed
    goal_met: bool
    grounding_unsupported: int


async def assess(
    bundle: Bundle,
    *,
    brief: Brief,
    genre: Genre,
    outline: Outline | None,
    facts: list[Fact],
    sources: list[Source],
    roles: Roles,
    log: ActionLog,
    round_no: int,
    grounding_cache: dict[str, list[Unsupported]],
) -> Assessment:
    """Run deterministic, grounding, reader and critic checks and route the findings.

    Grounding is skipped when there are no facts to ground against. The goal
    holds when no error-level check failed (warnings never count), the reader got
    every question right and the critic reported no major finding.
    """
    sections = bundle.sections()
    checks = run_deterministic_checks(
        sections, brief=brief, genre=genre, outline=outline, facts=facts
    )
    if facts:
        checks += await check_grounding(
            sections,
            facts=facts,
            model=roles.judge,
            log=log,
            cache=grounding_cache,
        )
    reader = await reader_test(
        plain_text(sections, title=bundle.title),
        brief=brief,
        reader_model=roles.reader,
        judge_model=roles.judge,
        log=log,
    )
    checks.append(reader.check)
    critic = await run_critic(
        bundle,
        genre=genre,
        model=roles.critic,
        round_no=round_no,
        log=log,
        facts=facts,
        sources=sources,
    )
    checks.append(critic.check)
    routed = route(
        checks,
        sections=sections,
        outline=outline,
        brief=brief,
        reader_misses=reader.misses,
        critic_items=critic.items,
    )
    return Assessment(
        round=round_no,
        checks=checks,
        reader=reader,
        critic=critic,
        routed=routed,
        goal_met=not any(is_error(c) for c in checks),
        grounding_unsupported=sum(
            len(c.details) for c in checks if c.name == "grounding" and not c.passed
        ),
    )
