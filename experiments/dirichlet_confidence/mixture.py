"""Mixture-of-Dirichlets aggregation for verbalized LLM confidence.

This is the pure-maths core of the method described in
``confidence_method_walkthrough.md``. It has no I/O, no model calls, and no
dependency on the rest of ``andamentum`` — numpy + scipy only — so it can be
unit-tested in isolation and (if it proves itself) lifted elsewhere wholesale.

Pipeline per query:

1. Elicit ``M`` models x ``K`` phrasings of a categorical distribution over
   ``C`` classes (verbalized integers, repaired + smoothed by :func:`repair_row`).
2. Build a *mixture* of ``M*K`` Dirichlets — one component per elicitation, each
   ``Dir(s * p_km)`` with mean ``p_km`` and shared concentration ``s``
   (:class:`MixtureOfDirichlets`). A mixture, not an average: the spread between
   elicitations is preserved and propagated into downstream sampling.
3. Monte-Carlo sample: pick a component, sample ``p`` from its Dirichlet, sample
   a category from ``p``.
4. Decompose total variance into named parts — within-component (elicitation
   noise, controlled by ``s``), within-model (phrasing brittleness), and
   between-model (model-family disagreement).

``s`` is fit on labelled data via held-out *sampled*-NLL minimisation
(:func:`fit_concentration_s`); the predictive mean alone is insensitive to ``s``,
so it must be fit against samples, not the mean.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize_scalar

# Floor applied to component means so no Dirichlet alpha is ever exactly 0.
_EPS = 1e-6
# Floor applied inside logs to avoid -inf when a class probability is ~0.
_LOG_FLOOR = 1e-12


# ---------------------------------------------------------------------------
# Elicitation repair: parse + smooth one verbalized row
# ---------------------------------------------------------------------------


def repair_row(v: np.ndarray, alpha0: float = 0.5) -> np.ndarray | None:
    """Convert one raw verbalized row to a smoothed probability vector.

    - Clips negatives to 0.
    - Adds a weak symmetric Dirichlet prior ``alpha0`` so a category the model
      scored 0 lands at a small non-zero probability instead of being
      hard-pruned (which would make Monte-Carlo sample it exactly never and the
      downstream Dirichlet machinery undefined).
    - Returns ``None`` if the row is unrecoverable (all zeros / parse failure),
      so the caller can drop that elicitation from the mixture.

    With ``alpha0 == 0`` this is plain normalisation (used by the regression
    test that recovers single-prompt behaviour exactly).
    """
    v = np.clip(np.asarray(v, dtype=float), 0.0, None)
    total = v.sum()
    if total <= 0:
        return None
    return (v + alpha0) / (total + alpha0 * len(v))


# ---------------------------------------------------------------------------
# Mixture-of-Dirichlets container
# ---------------------------------------------------------------------------


@dataclass
class MixtureOfDirichlets:
    """A mixture of ``N`` Dirichlet components over ``C`` classes.

    Each component ``i`` is ``Dir(s * means[i])`` — mean ``means[i]``, shared
    concentration ``s``. ``model_ids`` records which model produced each
    component so variance can be decomposed by model.

    Attributes:
        means: ``(N, C)`` per-elicitation point estimates ``p_km``.
        model_ids: ``(N,)`` model label per component.
        s: concentration / trust parameter (pseudo-count). Large ``s`` => each
            elicitation is treated as nearly exact; small ``s`` => wide noise
            around each point estimate.
        weights: ``(N,)`` mixture weights; an empty array (the default) means
            uniform.
    """

    means: np.ndarray
    model_ids: np.ndarray
    s: float = 50.0
    weights: np.ndarray = field(default_factory=lambda: np.empty(0))

    def __post_init__(self) -> None:
        self.means = np.asarray(self.means, dtype=float)
        self.model_ids = np.asarray(self.model_ids)
        if self.means.ndim != 2:
            raise ValueError(f"means must be 2-D (N, C); got shape {self.means.shape}")
        if self.means.shape[0] != self.model_ids.shape[0]:
            raise ValueError("means and model_ids disagree on component count")
        if self.means.shape[0] == 0:
            raise ValueError("mixture needs at least one component")
        # Empty weights array is the sentinel for "uniform".
        self.weights = np.asarray(self.weights, dtype=float)
        if self.weights.size == 0:
            n = self.means.shape[0]
            self.weights = np.ones(n) / n
        elif self.weights.shape[0] != self.means.shape[0]:
            raise ValueError("weights and means disagree on component count")
        # Renormalise means defensively (callers pass repaired rows, but a
        # floored renormalise keeps every Dirichlet alpha strictly positive).
        self.means = np.clip(self.means, _EPS, 1.0)
        self.means = self.means / self.means.sum(axis=1, keepdims=True)

    @property
    def n_components(self) -> int:
        return self.means.shape[0]

    @property
    def n_classes(self) -> int:
        return self.means.shape[1]

    def with_s(self, s: float) -> "MixtureOfDirichlets":
        """Return a copy of this mixture with a different concentration ``s``.

        Used pervasively when fitting/scoring ``s`` without rebuilding the
        component means each time.
        """
        return MixtureOfDirichlets(
            means=self.means, model_ids=self.model_ids, s=s, weights=self.weights
        )

    # ----- sampling -----

    def sample_distributions(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Sample ``n`` full probability vectors from the mixture. Shape ``(n, C)``.

        Two-step per draw: pick a component by weight, then draw ``p`` from that
        component's ``Dir(s * mean)``.
        """
        comps = rng.choice(self.n_components, size=n, p=self.weights)
        alphas = self.s * self.means[comps]  # (n, C)
        return _vec_dirichlet(alphas, rng)

    def sample_categories(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Sample ``n`` category indices (component -> p -> category)."""
        ps = self.sample_distributions(n, rng)
        cumulative = ps.cumsum(axis=1)
        u = rng.random(n)[:, None]
        return (u < cumulative).argmax(axis=1)

    # ----- closed-form moments -----

    def pooled_mean(self) -> np.ndarray:
        """``E[p]`` under the mixture = weight-averaged component means.

        Note this is invariant to ``s`` — which is exactly why ``s`` cannot be
        fit from the predictive mean alone (see :func:`fit_concentration_s`).
        """
        return (self.weights[:, None] * self.means).sum(axis=0)

    def variance_decomposition(self) -> dict[str, np.ndarray]:
        """Decompose ``Var[p_c]`` per class into named, interpretable parts.

        Returns a dict of ``(C,)`` arrays:

        - ``pooled_mean``: ``E[p]``.
        - ``within_component``: average Dirichlet variance inside each component
          (elicitation noise; controlled by ``s``).
        - ``between_component``: variance of the component means around the pool
          (= ``within_model + between_model``).
        - ``within_model``: spread of phrasings around each model's own mean
          (phrasing brittleness).
        - ``between_model``: spread of per-model means around the pool
          (model-family disagreement).
        - ``total``: ``within_component + between_component`` — the full variance
          used downstream.
        """
        C = self.n_classes
        w = self.weights

        # Within-component (Dirichlet) variance: m_c (1 - m_c) / (s + 1).
        within_comp = self.means * (1 - self.means) / (self.s + 1.0)
        within_component_avg = (w[:, None] * within_comp).sum(axis=0)

        pooled = self.pooled_mean()
        between_component = (w[:, None] * (self.means - pooled) ** 2).sum(axis=0)

        # Model-level split: group components by model_id.
        unique_models = np.unique(self.model_ids)
        model_means = np.zeros((len(unique_models), C))
        model_weights = np.zeros(len(unique_models))
        within_model = np.zeros(C)

        for i, mid in enumerate(unique_models):
            mask = self.model_ids == mid
            wm = w[mask]
            wm_sum = wm.sum()
            model_weights[i] = wm_sum
            mm = (wm[:, None] * self.means[mask]).sum(axis=0) / wm_sum
            model_means[i] = mm
            within_model += (wm[:, None] * (self.means[mask] - mm) ** 2).sum(axis=0)

        between_model = (model_weights[:, None] * (model_means - pooled) ** 2).sum(
            axis=0
        )

        return {
            "pooled_mean": pooled,
            "within_component": within_component_avg,
            "between_component": between_component,
            "within_model": within_model,
            "between_model": between_model,
            "total": within_component_avg + between_component,
        }

    # ----- likelihood -----

    def log_likelihood_of_label(self, y: int) -> float:
        """Marginal ``log p(y | mixture)`` integrating out ``p``.

        Under ``Dir(s*m)``, ``E[p_y] = m_y``, so this reduces to
        ``log(sum_k w_k m_{k,y})`` and is independent of ``s``. Use
        :meth:`sampled_nll` for an ``s``-sensitive score.
        """
        py = (self.weights * self.means[:, y]).sum()
        return float(np.log(max(py, _LOG_FLOOR)))

    def sampled_nll(self, y: int, n: int, rng: np.random.Generator) -> float:
        """``s``-sensitive negative log-likelihood of label ``y``.

        Draws ``p ~ mixture(s)`` and averages ``-log p_y``. Unlike the marginal
        likelihood this *does* depend on ``s``: too-large ``s`` over-penalises
        wrong labels (samples collapse to the mean), too-small ``s`` smears
        everything. This is the metric config D is meant to win on.
        """
        ps = self.sample_distributions(n, rng)
        return float(-np.log(np.clip(ps[:, y], _LOG_FLOOR, 1.0)).mean())


def _vec_dirichlet(alphas: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Vectorised Dirichlet sampling via the gamma trick. ``(n, C) -> (n, C)``."""
    gammas = rng.standard_gamma(alphas)
    return gammas / gammas.sum(axis=1, keepdims=True)


# ---------------------------------------------------------------------------
# Building a mixture from raw elicitations
# ---------------------------------------------------------------------------


def build_mixture(
    elicitations: list[dict],
    s: float = 50.0,
    alpha0: float = 0.5,
    model_weights: dict[str, float] | None = None,
) -> MixtureOfDirichlets | None:
    """Build a mixture from a list of raw elicitations.

    Args:
        elicitations: each ``{"model": str, "phrasing_id": int,
            "row": array-like of length C}``. Rows that fail
            :func:`repair_row` (all-zero / unparseable) are dropped.
        s: initial concentration.
        alpha0: smoothing prior passed to :func:`repair_row`.
        model_weights: optional per-model trust. If given, components from a
            model with weight ``w_m`` each get ``w_m / K_m`` so a model's total
            weight is independent of how many phrasings it contributed.

    Returns:
        The mixture, or ``None`` if every elicitation was dropped.
    """
    means: list[np.ndarray] = []
    model_ids: list[str] = []
    for e in elicitations:
        p = repair_row(e["row"], alpha0=alpha0)
        if p is None:
            continue
        means.append(p)
        model_ids.append(e["model"])
    if not means:
        return None

    means_arr = np.stack(means)
    model_ids_arr = np.array(model_ids)

    if model_weights is None:
        w = np.ones(len(means_arr)) / len(means_arr)
    else:
        w = np.zeros(len(means_arr))
        for mid in np.unique(model_ids_arr):
            mask = model_ids_arr == mid
            w[mask] = model_weights.get(mid, 1.0) / mask.sum()
        w = w / w.sum()

    return MixtureOfDirichlets(means=means_arr, model_ids=model_ids_arr, s=s, weights=w)


# ---------------------------------------------------------------------------
# Fitting s on labelled data
# ---------------------------------------------------------------------------


def fit_concentration_s(
    mixtures_and_labels: list[tuple[MixtureOfDirichlets, int]],
    s_bounds: tuple[float, float] = (1.0, 1000.0),
    n_samples: int = 400,
    seed: int = 0,
) -> float:
    """Fit one global concentration ``s`` by minimising summed sampled-NLL.

    The marginal label likelihood is independent of ``s`` (see
    :meth:`MixtureOfDirichlets.log_likelihood_of_label`), so we instead minimise
    the *expected* ``-log p_y`` under predictive sampling, which rewards
    calibrated sharpness. A single RNG seeded once gives common random numbers
    across ``s`` evaluations, so the objective is smooth in ``s``.

    If the optimum sits at the upper bound, the point mixture is already as sharp
    as the data warrants and Dirichlet smearing is not helping — a real finding,
    not a failure.
    """
    rng = np.random.default_rng(seed)

    def neg_expected_ll(s: float) -> float:
        total = 0.0
        for mix, y in mixtures_and_labels:
            total += mix.with_s(s).sampled_nll(y, n_samples, rng)
        return total

    res = minimize_scalar(
        neg_expected_ll, bounds=s_bounds, method="bounded", options={"xatol": 1.0}
    )
    return float(res.x)
