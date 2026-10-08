"""Enrichers: functions that add one metadata key to a stored document.

They sit outside the store core. Each reads a document, computes a value and
writes it through ``Store.update`` under its own top-level metadata key,
keeping every other key and recording what produced the value. Model-using
enrichers (``extract``, ``classify``) are opt-in and take their model as a
required keyword.

===============  ==============  =========================================
enricher         metadata key    needs
===============  ==============  =========================================
identifiers      ``identifiers``  nothing
facts            ``facts``        nothing
bibliographic    ``bib``          a ``ReferenceHttp`` (network)
extract          ``extracted``    an LLM ``model``
classify         ``labels``       a ``DecisionModel`` and your labels
===============  ==============  =========================================
"""

from ._base import EnrichError, EnrichResult
from .bibliographic import enrich_bibliographic
from .classify import enrich_classify
from .extract import enrich_extract
from .facts import enrich_facts
from .identifiers import enrich_identifiers
from .run import enrich_all

__all__ = [
    "EnrichError",
    "EnrichResult",
    "enrich_all",
    "enrich_bibliographic",
    "enrich_classify",
    "enrich_extract",
    "enrich_facts",
    "enrich_identifiers",
]
