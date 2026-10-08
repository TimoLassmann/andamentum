"""The scholarly work record, and the keys that say two records are one work.

``Work`` is the shared currency for anything that cites or finds literature:
the identifiers a work is known by, its bibliographic fields, whether we can
read it, whether it has been retracted. A search source fills in whatever it
has; :func:`merge_works` combines two records of the same work without losing
what either knew.

Identity is by key. :meth:`Work.identity_keys` lists the keys in a fixed order
— DOI, arXiv id, PMID, PMCID, OpenAlex id, Semantic Scholar id, local-corpus id,
canonical URL, then normalised title plus year. Every key but the last is *hard*: sharing one
means the same work. The title key is a fallback, and two records are merged on
it only when :func:`works_compatible` finds no identifier on which they
disagree (a preprint and its published version carry different DOIs and stay
two works).

Identifiers are normalised when a ``Work`` is built, so a stored or compared
value is always the canonical spelling. A non-empty value that is not an
identifier of its kind raises: a caller that holds a doubtful value normalises
it first with the ``normalise_*`` function (``None`` means "not one").

Imports nothing from andamentum.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any, Literal
from urllib.parse import parse_qsl, unquote, urlencode, urlsplit

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    ValidationInfo,
    field_serializer,
    SerializerFunctionWrapHandler,
    field_validator,
    model_serializer,
    model_validator,
)

WorkAccess = Literal["full_text", "abstract", "metadata", "unreadable"]
WorkRetraction = Literal["no", "retracted", "correction", "concern", "unknown"]

# A licence tag no export or report may carry the text of (Semantic Scholar).
LICENCE_NO_REDISTRIBUTE = "no_redistribute"

# How much a state tells us. A merge keeps the higher one.
_ACCESS_RANK: dict[str, int] = {
    "unreadable": 0,
    "metadata": 1,
    "abstract": 2,
    "full_text": 3,
}
_RETRACTION_RANK: dict[str, int] = {
    "unknown": 0,
    "no": 1,
    "correction": 2,
    "concern": 3,
    "retracted": 4,
}

# A normalised title shorter than this is too generic ("Introduction") to
# identify a work with its year.
_MIN_TITLE_KEY_CHARS = 15

_DOI = re.compile(r"10\.\d{4,9}/\S+")
_DOI_PREFIX = re.compile(r"^(?:https?://)?(?:dx\.)?doi\.org/|^doi:\s*", re.IGNORECASE)
_ARXIV_NEW = re.compile(r"\d{4}\.\d{4,5}")
_ARXIV_OLD = re.compile(r"[a-z][a-z\-]*(?:\.[a-z]{2})?/\d{7}")
_ARXIV_DOI_PREFIX = "10.48550/arxiv."
_PMCID = re.compile(r"pmc\d+", re.IGNORECASE)
_OPENALEX = re.compile(r"w\d+", re.IGNORECASE)
_S2 = re.compile(r"[0-9a-f]{40}")
_LOCAL = re.compile(r"local:[A-Za-z0-9][A-Za-z0-9._-]*")

_TRACKING_PARAMS = frozenset(
    {
        "fbclid",
        "gclid",
        "dclid",
        "msclkid",
        "mc_cid",
        "mc_eid",
        "igshid",
        "yclid",
        "_ga",
        "_gl",
    }
)
_DEFAULT_PORTS = {"http": 80, "https": 443}


def normalise_doi(value: str | None) -> str | None:
    """The DOI in *value* (bare, ``doi:`` or a doi.org URL), lower-cased; else None."""
    if value is None:
        return None
    text = value.strip()
    if "%" in text:
        text = unquote(text)
    text = _DOI_PREFIX.sub("", text).strip()
    text = text.rstrip(".,;:")
    # A closing bracket belongs to the DOI only when it closes one inside it.
    for close, opened in ((")", "("), ("]", "[")):
        while text.endswith(close) and text.count(close) > text.count(opened):
            text = text[:-1].rstrip(".,;:")
    if not _DOI.fullmatch(text):
        return None
    return text.lower()


def normalise_arxiv_id(value: str | None) -> str | None:
    """The arXiv id in *value*, version stripped and lower-cased; else None.

    Reads a bare id (``2101.00001v3``, ``hep-th/9901001``), ``arXiv:`` forms,
    arxiv.org abs/pdf/html URLs and the DataCite DOI ``10.48550/arXiv.<id>``.
    """
    if value is None:
        return None
    text = value.strip().lower()
    if text.startswith(("http://", "https://")):
        parts = urlsplit(text)
        host = (parts.hostname or "").removeprefix("www.")
        if host != "arxiv.org":
            return None
        for marker in ("/abs/", "/pdf/", "/html/"):
            if parts.path.startswith(marker):
                text = parts.path[len(marker) :]
                break
        else:
            return None
    elif text.startswith("arxiv:"):
        text = text[len("arxiv:") :]
    elif text.startswith(_ARXIV_DOI_PREFIX):
        text = text[len(_ARXIV_DOI_PREFIX) :]
    text = text.strip().removesuffix(".pdf")
    text = re.sub(r"v\d+$", "", text)
    if _ARXIV_NEW.fullmatch(text) or _ARXIV_OLD.fullmatch(text):
        return text
    return None


def normalise_pmid(value: str | int | None) -> str | None:
    """The PubMed id in *value* as digits without leading zeros; else None."""
    if value is None:
        return None
    text = re.sub(r"^pmid:?\s*", "", str(value).strip(), flags=re.IGNORECASE)
    if not re.fullmatch(r"\d{1,9}", text):
        return None
    text = text.lstrip("0")
    return text or None


def normalise_pmcid(value: str | None) -> str | None:
    """A PubMed Central id as ``PMC<digits>``; digits alone are accepted."""
    if value is None:
        return None
    text = re.sub(r"^pmc:\s*", "", value.strip(), flags=re.IGNORECASE)
    if text.isdigit():
        text = f"PMC{text}"
    if not _PMCID.fullmatch(text):
        return None
    return "PMC" + text[3:]


def normalise_openalex_id(value: str | None) -> str | None:
    """An OpenAlex work id as ``W<digits>`` (the openalex.org URL is accepted)."""
    if value is None:
        return None
    text = value.strip().rstrip("/").rsplit("/", 1)[-1]
    if not _OPENALEX.fullmatch(text):
        return None
    return text.upper()


def normalise_s2_id(value: str | None) -> str | None:
    """A Semantic Scholar paper id (40 hex characters, or a paper URL ending in one)."""
    if value is None:
        return None
    text = value.strip().rstrip("/").rsplit("/", 1)[-1].lower()
    return text if _S2.fullmatch(text) else None


def normalise_local_id(value: str | None) -> str | None:
    """An identifier in a local corpus, as ``local:<id>``; else None.

    A local corpus is a folder of records a run searches without any network;
    its records have no public identifier, so the corpus' own id names them.
    The ``local:`` prefix is required (a bare id is not accepted: it would be
    mistaken for any other kind of id).
    """
    if value is None:
        return None
    text = value.strip()
    return text if _LOCAL.fullmatch(text) else None


def canonical_url(value: str | None) -> str | None:
    """The one spelling of an http(s) URL that two links to a page share.

    Scheme and host lower-cased, ``www.`` and a default port dropped, no
    credentials, no fragment, ``utm_*`` and click-tracking parameters removed,
    remaining parameters sorted, no trailing slash. The scheme is kept as
    written (http and https are not assumed to be one page). ``None`` when
    *value* is not an http(s) URL with a host.
    """
    if value is None:
        return None
    try:
        parts = urlsplit(value.strip())
        host = parts.hostname
        port = parts.port
    except ValueError:
        return None
    scheme = parts.scheme.lower()
    if scheme not in _DEFAULT_PORTS or not host:
        return None
    host = host.lower().removeprefix("www.")
    netloc = host if port in (None, _DEFAULT_PORTS[scheme]) else f"{host}:{port}"
    path = parts.path.rstrip("/")
    query = urlencode(
        sorted(
            (k, v)
            for k, v in parse_qsl(parts.query, keep_blank_values=True)
            if not k.lower().startswith("utm_") and k.lower() not in _TRACKING_PARAMS
        )
    )
    return f"{scheme}://{netloc}{path}" + (f"?{query}" if query else "")


def normalise_title(title: str) -> str:
    """Accent-folded, case-folded title words joined by single spaces."""
    folded = unicodedata.normalize("NFKD", title)
    folded = "".join(c for c in folded if not unicodedata.combining(c))
    return " ".join(re.sub(r"[^0-9a-z]+", " ", folded.casefold()).split())


def title_key(title: str, year: int | None) -> str | None:
    """``<normalised title>|<year>``, or None when either is too weak to identify."""
    words = normalise_title(title)
    if year is None or len(words) < _MIN_TITLE_KEY_CHARS:
        return None
    return f"{words}|{year}"


def identifiers_in_url(url: str) -> dict[str, str]:
    """The identifiers a scholarly URL names: doi.org, arXiv, PubMed, PMC, OpenAlex."""
    try:
        parts = urlsplit(url.strip())
    except ValueError:
        return {}
    host = (parts.hostname or "").lower().removeprefix("www.")
    path = parts.path
    found: dict[str, str | None] = {}
    if host in ("doi.org", "dx.doi.org"):
        found["doi"] = normalise_doi(url)
    elif host == "arxiv.org":
        found["arxiv_id"] = normalise_arxiv_id(url)
    elif host == "pubmed.ncbi.nlm.nih.gov":
        found["pmid"] = normalise_pmid(path.strip("/").split("/")[0])
    elif host in ("ncbi.nlm.nih.gov", "pmc.ncbi.nlm.nih.gov"):
        match = re.search(r"/articles/(PMC\d+)", path, re.IGNORECASE)
        found["pmcid"] = normalise_pmcid(match.group(1)) if match else None
    elif host == "openalex.org":
        found["openalex_id"] = normalise_openalex_id(path)
    return {k: v for k, v in found.items() if v}


_NORMALISERS = {
    "doi": normalise_doi,
    "arxiv_id": normalise_arxiv_id,
    "pmid": normalise_pmid,
    "pmcid": normalise_pmcid,
    "openalex_id": normalise_openalex_id,
    "s2_id": normalise_s2_id,
    "local_id": normalise_local_id,
    "url": canonical_url,
}

# Identifier fields, in identity-key order (the URL and title keys follow).
ID_FIELDS: tuple[str, ...] = (
    "doi",
    "arxiv_id",
    "pmid",
    "pmcid",
    "openalex_id",
    "s2_id",
    "local_id",
)
_KEY_KIND = {
    "doi": "doi",
    "arxiv_id": "arxiv",
    "pmid": "pmid",
    "pmcid": "pmcid",
    "openalex_id": "openalex",
    "s2_id": "s2",
    "local_id": "local",
    "url": "url",
}


class Work(BaseModel):
    """One scholarly or web work, as far as we know it.

    ``work_id`` is empty until a store assigns it. ``access`` is what we hold
    (full text, an abstract, bare metadata) or ``unreadable`` after a failed
    attempt; ``retracted`` is ``unknown`` until checked.
    """

    model_config = ConfigDict(extra="forbid")

    work_id: str = ""
    doi: str | None = None
    arxiv_id: str | None = None
    pmid: str | None = None
    pmcid: str | None = None
    openalex_id: str | None = None
    s2_id: str | None = None
    local_id: str | None = None
    url: str | None = None
    title: str = ""
    authors: list[str] = Field(default_factory=list)
    year: int | None = None
    venue: str = ""
    abstract: str = ""
    oa_url: str | None = None
    access: WorkAccess = "metadata"
    retracted: WorkRetraction = "unknown"
    licence: str = ""

    @field_validator(*_NORMALISERS, mode="before")
    @classmethod
    def _normalise_identifier(cls, value: Any, info: ValidationInfo) -> str | None:
        if value is None or (isinstance(value, str) and not value.strip()):
            return None
        field = str(info.field_name)
        normalised = _NORMALISERS[field](value if field == "pmid" else str(value))
        if normalised is None:
            raise ValueError(f"not a valid {field}: {value!r}")
        return normalised

    @field_validator("oa_url", mode="before")
    @classmethod
    def _empty_oa_url(cls, value: Any) -> Any:
        return None if isinstance(value, str) and not value.strip() else value

    @model_validator(mode="after")
    def _identifiers_from_url(self) -> Work:
        """A doi.org, arXiv or PubMed URL names an identifier: use it if none is set."""
        if self.url:
            for field, value in identifiers_in_url(self.url).items():
                if getattr(self, field) is None:
                    setattr(self, field, value)
        return self

    @field_serializer("abstract")
    def _licensed_abstract(self, value: str) -> str:
        """Every dump follows the licence: a ``no_redistribute`` abstract is never exported.

        The store keeps the text in its own column, not through a dump.
        """
        return value if self.redistributable else ""

    @model_serializer(mode="wrap")
    def _local_id_only_when_set(
        self, handler: SerializerFunctionWrapHandler
    ) -> dict[str, Any]:
        """A dump has no ``local_id`` key unless the work has one: only records of a
        local corpus carry it, and every other work's dump (and any digest of it)
        stays what it was before the field existed."""
        data: dict[str, Any] = handler(self)
        if data.get("local_id") is None:
            data.pop("local_id", None)
        return data

    @property
    def redistributable(self) -> bool:
        """False when the record's licence forbids exporting its text."""
        return self.licence != LICENCE_NO_REDISTRIBUTE

    def identity_keys(self) -> list[tuple[str, str]]:
        """``(kind, value)`` keys in merge order; ``title`` is last, and soft."""
        keys: list[tuple[str, str]] = []
        arxiv_id = self.arxiv_id
        if arxiv_id is None and self.doi and self.doi.startswith(_ARXIV_DOI_PREFIX):
            # DataCite's DOI for a preprint is the arXiv id in another form.
            arxiv_id = normalise_arxiv_id(self.doi)
        values = {**{f: getattr(self, f) for f in ID_FIELDS}, "arxiv_id": arxiv_id}
        for field in (*ID_FIELDS, "url"):
            value = values.get(field, getattr(self, field))
            if value:
                keys.append((_KEY_KIND[field], value))
        weak = title_key(self.title, self.year)
        if weak:
            keys.append(("title", weak))
        return keys

    def hard_keys(self) -> list[tuple[str, str]]:
        """The keys that decide identity alone (every key but the title one)."""
        return [key for key in self.identity_keys() if key[0] != "title"]

    def to_csl(self, *, csl_id: str | None = None) -> dict[str, JsonValue]:
        """The record as one CSL-JSON item (what Zotero, Pandoc and citeproc read)."""
        item: dict[str, JsonValue] = {
            "id": csl_id or self.work_id or self.doi or self.url or self.title,
            "type": _csl_type(self),
            "title": self.title,
        }
        if self.authors:
            names: list[JsonValue] = [_csl_name(name) for name in self.authors]
            item["author"] = names
        if self.year is not None:
            item["issued"] = {"date-parts": [[self.year]]}
        if self.venue:
            item["container-title"] = self.venue
        if self.doi:
            item["DOI"] = self.doi
        if self.pmid:
            item["PMID"] = self.pmid
        if self.pmcid:
            item["PMCID"] = self.pmcid
        link = self.url or self.oa_url
        if link:
            item["URL"] = link
        if self.abstract and self.redistributable:
            item["abstract"] = self.abstract
        return item


def _csl_type(work: Work) -> str:
    if work.arxiv_id and not work.venue:
        return "article"
    if work.doi or work.venue:
        return "article-journal"
    return "webpage" if work.url else "article"


def _csl_name(name: str) -> dict[str, JsonValue]:
    name = " ".join(name.split())
    if "," in name:
        family, _, given = name.partition(",")
        return {"family": family.strip(), "given": given.strip()}
    if " " not in name:
        return {"literal": name}
    given, _, family = name.rpartition(" ")
    return {"family": family, "given": given}


def works_compatible(a: Work, b: Work) -> bool:
    """True when no identifier both records hold has a different value."""
    return all(
        getattr(a, field) is None
        or getattr(b, field) is None
        or getattr(a, field) == getattr(b, field)
        for field in ID_FIELDS
    )


def merge_works(first: Work, second: Work) -> Work:
    """One record from two of the same work; *first* wins where both have a value.

    Empty fields are filled from the other record. The longer author list is
    kept. ``access`` and ``retracted`` keep the more informative (or more
    severe) state.

    The licence describes the abstract, the one piece of text a record
    carries, so the two travel together. When only one record is
    ``no_redistribute`` and the other holds an abstract, the redistributable
    abstract and its licence are kept, even when the restricted one is
    longer: a Semantic Scholar copy must not take away text that OpenAlex or
    Europe PMC gives freely. Otherwise the longer abstract is kept, and a
    ``no_redistribute`` tag on either record stays (the restrictive tag wins).
    """
    merged = first.model_copy(deep=True)
    for field in (*ID_FIELDS, "url", "oa_url", "year", "title", "venue", "work_id"):
        if not getattr(merged, field) and getattr(second, field):
            setattr(merged, field, getattr(second, field))
    if len(second.authors) > len(merged.authors):
        merged.authors = list(second.authors)
    if _ACCESS_RANK[second.access] > _ACCESS_RANK[merged.access]:
        merged.access = second.access
    if _RETRACTION_RANK[second.retracted] > _RETRACTION_RANK[merged.retracted]:
        merged.retracted = second.retracted
    free_text = [w for w in (first, second) if w.redistributable and w.abstract.strip()]
    if not (first.redistributable and second.redistributable) and free_text:
        merged.abstract = free_text[0].abstract
        merged.licence = free_text[0].licence
        return merged
    if len(second.abstract) > len(merged.abstract):
        merged.abstract = second.abstract
    if LICENCE_NO_REDISTRIBUTE in (first.licence, second.licence):
        merged.licence = LICENCE_NO_REDISTRIBUTE
    elif not merged.licence:
        merged.licence = second.licence
    return merged
