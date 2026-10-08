"""The coverage table — checklist mode's summary, computed rather than written.

Review mode ends with an LLM writing prose about what it found. Checklist mode
deliberately does not: the value of this mode is that two runs of the same bank
over two drafts are *diffable*, and a stochastic summariser destroys exactly
that. So the summary is a table with one row per probe, emitted in bank order,
whose shape depends only on the bank — never on the model, the findings, or the
run.

**The invariant this module exists to protect:** nothing that failed to check
is rendered as a bare ``0``. ``0`` from a probe that ran and verified is a
*result* — the strongest one an unquota'd probe can produce, since it says the
reviewer looked for that defect and it is not there. Reporting absence of
checking as absence of defects is the one way a coverage report can actively
mislead the author it is meant to serve, and there are two ways to arrive
there:

- the probe **crashed** — rendered ``not run``, in every cell. Triple-signalled
  (findings cell, severity cell, and a dedicated column) so a careless read of
  any one cell still lands on the truth;
- the probe **ran and reported instances that verification could not check**
  (a paraphrased quote instead of a verbatim one is the commonest small-model
  failure, and the gate exists precisely because of it) — rendered
  ``0 (n unverifiable)``, and named in the prose, so the reader can see that
  the row is empty because the findings were discarded rather than because the
  document is clean.
"""

from __future__ import annotations

import logging
from collections import Counter

from .probes import Probe, ProbeFinding

logger = logging.getLogger("andamentum.whetstone.v3")

# Fixed presentation order for the severity breakdown. Sorting by count would
# make the cell's word order depend on the findings, which is one more thing
# that differs between two runs the reader is trying to diff.
_SEVERITY_ORDER = ("major", "moderate", "minor")

_NOT_RUN = "not run"
_EMPTY_CELL = "—"

# Table header. Findings and Severity both carry the not-run state, and "Ran"
# states it a third time — see the module docstring on why the redundancy is
# the point.
_HEADER = "| Check | Findings | Severity | Ran |\n| --- | --- | --- | --- |"


def _escape_cell(text: str) -> str:
    """Make arbitrary probe text safe inside a markdown table cell.

    Probe titles are bank data, so they can contain anything a bank author
    typed. An unescaped pipe silently splits a row into extra columns, which
    corrupts every row after it in some renderers — a formatting bug that
    reads as missing coverage.
    """
    return " ".join(text.replace("|", r"\|").split())


def _severity_spread(findings: list[ProbeFinding]) -> str:
    """The severity breakdown for one probe's findings, or the empty marker."""
    if not findings:
        return _EMPTY_CELL
    counts = Counter(f.severity for f in findings)
    return ", ".join(f"{counts[s]} {s}" for s in _SEVERITY_ORDER if counts[s])


def _quoted_titles(titles: list[str]) -> str:
    """Join probe titles into readable prose, quoted so commas cannot confuse.

    Probe titles are sentences ("Information needed to act, and absent"), so a
    plain comma-joined list reads as one run-on with no way to tell where a
    title ends. Quoting each one makes the boundaries unambiguous in every
    renderer without needing a separator the titles could also contain.
    """
    quoted = [f'"{title}"' for title in titles]
    if len(quoted) <= 1:
        return "".join(quoted)
    return f"{', '.join(quoted[:-1])} and {quoted[-1]}"


def _plural(count: int, singular: str, plural: str) -> str:
    return singular if count == 1 else plural


def _findings_cell(*, kept: int, quota: int | None, unverifiable: int) -> str:
    """The findings cell for a probe that ran.

    All three inputs are ints that mean different things, so they are
    keyword-only: transposing two of them would silently render a plausible
    but wrong cell rather than raise.

    ``kept`` alone is ambiguous in two directions this function closes:

    - against a quota, ``3`` is a *saturated* row — the probe was asked for at
      most three and returned three, so there may well be more it could not
      report. ``3 of 3`` says so; a bare ``3`` reads as "there are three".
    - a ``0`` next to discarded findings is not the informative zero. The
      ``(n unverifiable)`` suffix keeps the count honest without inventing a
      column, and it is shown whenever anything was discarded — a partial loss
      (``2 of 3 (1 unverifiable)``) matters to the reader for the same reason
      a total one does.
    """
    cell = f"{kept} of {quota}" if quota is not None else str(kept)
    if unverifiable:
        return f"{cell} ({unverifiable} unverifiable)"
    return cell


def coverage_table(
    probes: list[Probe],
    findings: list[ProbeFinding],
    *,
    failed_probe_ids: list[str],
    unverifiable_counts: dict[str, int],
) -> str:
    """Render what the checklist covered as prose + a markdown table.

    One row per probe in *probes*, in bank order — not in findings order, not
    sorted by severity. Bank order is what makes two runs comparable line by
    line; any other ordering makes the table's shape a function of the results,
    which is the property this mode is built to avoid.

    Four row states, and collapsing any of the last three into the first is the
    bug this function is written to prevent:

    - the probe ran, verified, and found nothing → ``0``, a real and often
      informative result (for an unquota'd probe it is the whole point of
      running it);
    - the probe ran and found instances → the count plus the severity spread,
      shown against the probe's quota when it has one so a saturated row is
      visible as saturated;
    - the probe ran and everything it reported was discarded by verification
      (its count in *unverifiable_counts*) → ``0 (n unverifiable)``, plus a
      sentence in the opening prose. Its ``0`` means nothing usable came back,
      NOT that the defect is absent;
    - the probe is in *failed_probe_ids* → ``not run`` in every cell, because
      nothing was checked. It is never rendered as ``0``.

    *failed_probe_ids* is authoritative on whether checking happened: a probe
    listed there is reported as not run even if findings carry its id, since a
    partial result from a crashed probe is not coverage. Any finding that
    cannot be placed in a row (unknown probe id, or an id recorded as crashed)
    is logged and named in the prose rather than dropped — an untabulated
    finding that nobody mentions is the same silent loss in the other
    direction, and a drop count with no row to sit in is logged for the same
    reason.

    *unverifiable_counts* maps probe id → how many of that probe's findings
    :func:`~.verify.verify_probe_findings` discarded; a probe that lost none is
    simply absent from it.
    """
    bank_ids = [p.id for p in probes]
    bank_id_set = set(bank_ids)
    if len(bank_id_set) != len(bank_ids):
        duplicates = sorted({i for i in bank_ids if bank_ids.count(i) > 1})
        logger.warning(
            "[v3.checklist.summary] bank has duplicate probe ids %s — "
            "each will get its own row and their findings will be counted twice",
            ", ".join(duplicates),
        )

    failed = set(failed_probe_ids)
    unknown_failed = sorted(failed - bank_id_set)
    if unknown_failed:
        logger.warning(
            "[v3.checklist.summary] %d crashed probe id(s) are not in this bank "
            "and get no row: %s",
            len(unknown_failed),
            ", ".join(unknown_failed),
        )

    by_probe: dict[str, list[ProbeFinding]] = {}
    for finding in findings:
        by_probe.setdefault(finding.probe_id, []).append(finding)

    # Findings whose probe has no row to sit in: either the id is not in the
    # bank at all, or the probe is recorded as crashed (in which case the row
    # says "not run" and must keep saying it).
    unplaceable = sorted(
        probe_id
        for probe_id in by_probe
        if probe_id not in bank_id_set or probe_id in failed
    )
    unplaceable_count = sum(len(by_probe[probe_id]) for probe_id in unplaceable)
    if unplaceable:
        logger.warning(
            "[v3.checklist.summary] %d finding(s) cannot be placed in the "
            "coverage table (probe id absent from the bank, or recorded as "
            "crashed): %s",
            unplaceable_count,
            ", ".join(unplaceable),
        )

    # A drop count for a probe with no row (or a "not run" one) has nowhere to
    # be shown. Losing it silently is the same failure this module exists to
    # prevent, one level up, so it is reported to the log.
    orphan_drops = sorted(
        probe_id
        for probe_id, count in unverifiable_counts.items()
        if count and (probe_id not in bank_id_set or probe_id in failed)
    )
    if orphan_drops:
        logger.warning(
            "[v3.checklist.summary] discarded-finding counts for %d probe(s) "
            "have no row to appear in (probe id absent from the bank, or "
            "recorded as crashed): %s",
            len(orphan_drops),
            ", ".join(orphan_drops),
        )

    rows: list[str] = []
    ran_count = 0
    tabulated = 0
    probes_with_findings = 0
    # Probes that ran, reported something, and kept none of it — the rows whose
    # 0 must not be read as a clean bill of health.
    silenced: list[str] = []
    any_saturated = False
    for probe in probes:
        title = _escape_cell(probe.title)
        if probe.id in failed:
            rows.append(f"| {title} | {_NOT_RUN} | {_NOT_RUN} | no |")
            continue
        ran_count += 1
        probe_findings = by_probe.get(probe.id, [])
        unverifiable = unverifiable_counts.get(probe.id, 0)
        tabulated += len(probe_findings)
        if probe_findings:
            probes_with_findings += 1
        elif unverifiable:
            silenced.append(probe.title)
        if probe.quota is not None and len(probe_findings) >= probe.quota:
            any_saturated = True
        cell = _findings_cell(
            kept=len(probe_findings), quota=probe.quota, unverifiable=unverifiable
        )
        rows.append(f"| {title} | {cell} | {_severity_spread(probe_findings)} | yes |")

    lines: list[str] = ["## Checklist coverage", ""]

    total = len(probes)
    opening = [
        f"This checklist ran {ran_count} of {total} "
        f"{_plural(total, 'check', 'checks')} over the document and reported "
        f"{tabulated} {_plural(tabulated, 'finding', 'findings')} across "
        f"{probes_with_findings} of them."
    ]
    if silenced:
        # The unqualified "looked for and is not there" sentence is exactly the
        # wrong thing to tell an author when a check's findings were all
        # thrown away, so it is never emitted alone in that case.
        opening.append(
            f"A check that ran and found nothing reports 0, which is usually a "
            f"result: that defect was looked for and is not there. Not so for "
            f"{len(silenced)} of them — "
            f"{_quoted_titles(silenced)} reported instances that could not be "
            f"verified against the document and were discarded, so the 0 there "
            f"means nothing usable came back, not that the document is clean."
        )
    else:
        opening.append(
            "A check that ran and found nothing reports 0, which is a result: "
            "that defect was looked for and is not there."
        )
    if any_saturated:
        opening.append(
            'A count written as "n of q" means the check was asked for at most '
            "q instances; a check that returned its full q may be hiding more."
        )
    lines.append(" ".join(opening))

    missing = [p.title for p in probes if p.id in failed]
    if missing:
        lines.append("")
        lines.append(
            f"**Coverage is incomplete.** {len(missing)} "
            f"{_plural(len(missing), 'check', 'checks')} did not run: "
            f'{", ".join(missing)}. Those rows read "not run" — the '
            f"corresponding defects were never looked for, so nothing here "
            f"says the document is clean of them."
        )

    if unplaceable:
        lines.append("")
        lines.append(
            f"{unplaceable_count} "
            f"{_plural(unplaceable_count, 'finding', 'findings')} could not be "
            f"placed in the table because the check that produced "
            f"{_plural(unplaceable_count, 'it', 'them')} is not in this bank "
            f"or is recorded as crashed: {', '.join(unplaceable)}."
        )

    if rows:
        lines.extend(["", _HEADER])
        lines.extend(rows)
    else:
        # An empty bank is a configuration error, not a clean document. Say so
        # instead of emitting a header with nothing under it, which reads as a
        # document that passed every check.
        logger.warning(
            "[v3.checklist.summary] no probes in the bank — nothing was checked"
        )
        lines.extend(["", "**No checks were configured, so nothing was checked.**"])

    # Co-location lives on the findings, where a reader of the summary alone
    # would never see it. Stated here rather than as a column so the table's
    # shape stays identical between a run with co-located findings and one
    # without.
    #
    # Worded as "the same passage", not as agreement: the checks have different
    # instructions, so the ordinary case is one passage carrying two different
    # defects. That is a reason to look at the passage, not evidence that either
    # finding is more likely correct — see :mod:`.corroborate`.
    corroborated = [f for f in findings if f.corroborated_by]
    if corroborated:
        involved = sorted(
            {f.probe_id for f in corroborated}
            | {other for f in corroborated for other in f.corroborated_by}
        )
        lines.extend(
            [
                "",
                f"{len(corroborated)} "
                f"{_plural(len(corroborated), 'finding sits', 'findings sit')} "
                f"on a passage that more than one check flagged — worth reading "
                f"first, though the checks need not be making the same point. "
                f"Checks involved: {', '.join(involved)}.",
            ]
        )

    return "\n".join(lines)
