"""Where a work comes from, for independence and for showing a site.

Two findings are independent when their works have different origins: the
family name of the first author, else the registrable domain of the URL, else
the work itself. A preprint and its journal version, or two papers by one
first author, count once; two pages of one site count once. Used by the chooser
(the site shown beside a title), the writer (the origin label on each finding
and the agreement count it weighs) and the reflect step.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from urllib.parse import urlsplit

from andamentum.references import Work

from .types import Finding

# Second-level labels under which a country code is the registrable part's parent
# (example.co.uk, example.com.au): not a public-suffix list, a cheap approximation.
_SECOND_LEVEL = frozenset({"co", "com", "org", "net", "ac", "gov", "edu", "ne", "or"})
_FAMILY_SKIP = frozenset({"jr", "sr", "ii", "iii", "iv"})


def registrable_domain(url: str) -> str:
    """The site a URL belongs to: ``www.news.example.com`` becomes ``example.com``."""
    host = (urlsplit(url).hostname or "").lower().rstrip(".")
    labels = [label for label in host.split(".") if label]
    if len(labels) <= 2:
        return ".".join(labels)
    if len(labels[-1]) == 2 and labels[-2] in _SECOND_LEVEL:
        return ".".join(labels[-3:])
    return ".".join(labels[-2:])


def _family_name(author: str) -> str:
    """The family name of an author string, 'Family, Given' or 'Given Family'."""
    author = author.strip()
    if "," in author:
        return author.split(",", 1)[0].strip().casefold()
    parts = [p for p in author.split() if p.casefold().rstrip(".") not in _FAMILY_SKIP]
    return parts[-1].casefold() if parts else ""


def origin_key(work: Work) -> str:
    """The independence key of a work: first author's family name, else the
    registrable domain of its URL, else the work itself."""
    if work.authors:
        family = _family_name(work.authors[0])
        if family:
            return f"author:{family}"
    if work.url:
        domain = registrable_domain(work.url)
        if domain:
            return f"site:{domain}"
    return f"work:{work.work_id}"


def origin_label(work: Work) -> str:
    """A short human label for the origin: the site's domain for a web page, the
    first author's family name for a paper, else the venue or ``unknown``."""
    key = origin_key(work)
    if key.startswith(("site:", "author:")):
        return key.split(":", 1)[1]
    return work.venue or "unknown"


def independent_sources(findings: Sequence[Finding], works: Mapping[str, Work]) -> int:
    """How many distinct origins stand behind *findings*."""
    return len({_origin_of(f, works) for f in findings})


def _origin_of(finding: Finding, works: Mapping[str, Work]) -> str:
    work = works.get(finding.work_id)
    return origin_key(work) if work is not None else f"work:{finding.work_id}"
