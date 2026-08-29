"""Mixture-of-Dirichlets confidence aggregation (standalone experiment).

Developed deliberately *outside* ``andamentum.epistemic``: the question of
whether this method produces trustworthy confidence is open, so it lives here
and is validated independently before any consideration of lifting it into the
epistemic package.

Public surface:
- :class:`mixture.MixtureOfDirichlets`, :func:`mixture.build_mixture`,
  :func:`mixture.repair_row`, :func:`mixture.fit_concentration_s`
- metrics in :mod:`metrics`
- elicitation + SciFact loading + the A/B/C/D benchmark are in
  :mod:`elicitation`, :mod:`scifact`, :mod:`benchmark` (these touch models / disk).
"""

from .mixture import (
    MixtureOfDirichlets,
    build_mixture,
    fit_concentration_s,
    repair_row,
)

__all__ = [
    "MixtureOfDirichlets",
    "build_mixture",
    "fit_concentration_s",
    "repair_row",
]
