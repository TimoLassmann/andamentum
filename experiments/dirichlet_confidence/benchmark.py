"""SciFact A/B/C/D benchmark for the mixture-of-Dirichlets confidence method.

Compares, on a held-out test split:

- **per-model**: each model asked directly (single phrasing) and its
  multi-phrasing mean — "how good is one model's own verbalized confidence".
- **A single**: first model, first phrasing (the baseline).
- **B multi-phrasing mean**: first model, K phrasings averaged.
- **C full point mixture**: M x K elicitations, pooled mean.
- **D Dirichlet mixture**: M x K elicitations, Dirichlet noise at fitted ``s``.

C and D share a predictive mean, so they tie on accuracy / ECE / Brier; D's win
(if any) shows up in **sampled-NLL** and the variance decomposition.

Elicitations are cached to a JSONL keyed by (claim_id, model, phrasing_id) so a
rerun re-pays for nothing already gathered — important with paid models.
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

import numpy as np

from . import metrics
from .elicitation import Elicitor, n_phrasings
from .mixture import MixtureOfDirichlets, build_mixture, fit_concentration_s
from .scifact import Example, label_counts

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Elicitation cache
# ---------------------------------------------------------------------------


def _cache_key(claim_id: str, model: str, phrasing_id: int) -> str:
    return f"{claim_id}\x1f{model}\x1f{phrasing_id}"


def _load_cache(cache_path: Path) -> dict[str, list[float] | None]:
    cache: dict[str, list[float] | None] = {}
    if cache_path.exists():
        with cache_path.open() as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                cache[_cache_key(rec["claim_id"], rec["model"], rec["phrasing_id"])] = (
                    rec["row"]
                )
    return cache


def _append_cache(
    cache_path: Path, claim_id: str, model: str, phrasing_id: int, row
) -> None:
    rec = {
        "claim_id": claim_id,
        "model": model,
        "phrasing_id": phrasing_id,
        "row": (None if row is None else [float(x) for x in row]),
    }
    with cache_path.open("a") as fh:
        fh.write(json.dumps(rec) + "\n")


async def gather_elicitations(
    examples: list[Example],
    elicitor: Elicitor,
    models: list[str],
    cache_path: Path,
    concurrency: int = 6,
) -> dict[str, list[float] | None]:
    """Elicit every (example, model, phrasing), using and updating the cache.

    Returns a dict keyed by ``_cache_key`` mapping to a row (list) or ``None``
    (dropped elicitation). The cache file is appended to as results arrive, so an
    interrupted run resumes cheaply.
    """
    cache = _load_cache(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    sem = asyncio.Semaphore(concurrency)
    write_lock = asyncio.Lock()

    todo: list[tuple[Example, str, int]] = []
    for ex in examples:
        for model in models:
            for k in range(n_phrasings()):
                if _cache_key(ex.claim_id, model, k) not in cache:
                    todo.append((ex, model, k))

    logger.info(
        "elicitations: %d cached, %d to fetch (%d examples x %d models x %d phrasings)",
        len(cache),
        len(todo),
        len(examples),
        len(models),
        n_phrasings(),
    )

    async def one(ex: Example, model: str, k: int) -> None:
        async with sem:
            row = await elicitor.elicit(ex.claim, ex.evidence, model, k)
        async with write_lock:
            _append_cache(cache_path, ex.claim_id, model, k, row)
            cache[_cache_key(ex.claim_id, model, k)] = (
                None if row is None else [float(x) for x in row]
            )

    # Chunk so progress is logged and the cache flushes periodically.
    done = 0
    for i in range(0, len(todo), concurrency * 4):
        batch = todo[i : i + concurrency * 4]
        await asyncio.gather(*(one(ex, m, k) for ex, m, k in batch))
        done += len(batch)
        if todo:
            logger.info("  elicited %d/%d", done, len(todo))
    return cache


# ---------------------------------------------------------------------------
# Mixture assembly
# ---------------------------------------------------------------------------


def build_all_mixtures(
    examples: list[Example],
    cache: dict[str, list[float] | None],
    models: list[str],
    s_initial: float,
    alpha0: float,
) -> list[MixtureOfDirichlets | None]:
    """Build one mixture per example from cached elicitations (model order kept)."""
    mixtures: list[MixtureOfDirichlets | None] = []
    for ex in examples:
        elicitations = []
        for model in models:
            for k in range(n_phrasings()):
                row = cache.get(_cache_key(ex.claim_id, model, k))
                if row is None:
                    continue
                elicitations.append({"model": model, "phrasing_id": k, "row": row})
        mixtures.append(build_mixture(elicitations, s=s_initial, alpha0=alpha0))
    return mixtures


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def _score_probs(probs: np.ndarray, labels: np.ndarray) -> dict:
    preds = probs.argmax(axis=1)
    top_conf = probs.max(axis=1)
    return {
        "accuracy": metrics.accuracy(probs, labels),
        "ece": metrics.expected_calibration_error(top_conf, preds == labels),
        "brier": metrics.brier_score(probs, labels),
        "nll": metrics.negative_log_likelihood(probs, labels),
        "n": int(len(labels)),
    }


def _collect(
    test_idx,
    mixtures: list[MixtureOfDirichlets | None],
    examples: list[Example],
    predict_fn,
) -> dict:
    probs, labels = [], []
    for i in test_idx:
        mix = mixtures[i]
        if mix is None:
            continue
        p = predict_fn(mix)
        if p is None:
            continue
        probs.append(p)
        labels.append(examples[i].label)
    if not probs:
        return {"n": 0}
    return _score_probs(np.array(probs), np.array(labels))


def evaluate(
    examples: list[Example],
    mixtures: list[MixtureOfDirichlets | None],
    models: list[str],
    dev_frac: float,
    n_mc_samples: int,
    seed: int,
) -> dict:
    """Fit ``s`` on dev, score every configuration on test."""
    rng = np.random.default_rng(seed)
    n = len(examples)
    idx = rng.permutation(n)
    n_dev = int(n * dev_frac)
    dev_idx, test_idx = idx[:n_dev], idx[n_dev:]

    dev_pairs = [
        (mixtures[i], examples[i].label) for i in dev_idx if mixtures[i] is not None
    ]
    s_fit = fit_concentration_s(dev_pairs) if dev_pairs else 50.0

    results: dict = {"s_fit": float(s_fit), "n_dev": len(dev_pairs)}

    # --- per-model baselines (directly ask each model) ---
    per_model = {}
    for model in models:

        def predict_single(mix, _model=model):
            mask = mix.model_ids == _model
            if not mask.any():
                return None
            # first phrasing of this model = the single direct ask
            return mix.means[mask][0]

        def predict_mean(mix, _model=model):
            mask = mix.model_ids == _model
            if not mask.any():
                return None
            return mix.means[mask].mean(axis=0)

        per_model[model] = {
            "single_prompt": _collect(test_idx, mixtures, examples, predict_single),
            "multi_phrasing_mean": _collect(test_idx, mixtures, examples, predict_mean),
        }
    results["per_model"] = per_model

    # --- A/B/C/D ---
    def predict_A(mix):
        return mix.means[0]

    def predict_B(mix):
        first_model = mix.model_ids[0]
        mask = mix.model_ids == first_model
        return mix.means[mask].mean(axis=0)

    def predict_C(mix):
        return mix.pooled_mean()

    def predict_D(mix):
        return mix.with_s(s_fit).pooled_mean()

    results["A_single"] = _collect(test_idx, mixtures, examples, predict_A)
    results["B_multi_phrasing_mean"] = _collect(test_idx, mixtures, examples, predict_B)
    results["C_full_point_mixture"] = _collect(test_idx, mixtures, examples, predict_C)
    results["D_dirichlet_mixture"] = _collect(test_idx, mixtures, examples, predict_D)

    # --- sampled-NLL: the metric where D should beat C ---
    test_pairs = [
        (mixtures[i], examples[i].label) for i in test_idx if mixtures[i] is not None
    ]
    # C is the point mixture (very large s => samples collapse to the mean).
    results["C_full_point_mixture"]["sampled_nll"] = metrics.mean_sampled_nll(
        test_pairs, s=1e6, n_samples=n_mc_samples, rng=rng
    )
    results["D_dirichlet_mixture"]["sampled_nll"] = metrics.mean_sampled_nll(
        test_pairs, s=s_fit, n_samples=n_mc_samples, rng=rng
    )

    # --- variance decomposition (test mean, at fitted s) ---
    decomps = {"within_component": [], "within_model": [], "between_model": []}
    for mix, _ in test_pairs:
        d = mix.with_s(s_fit).variance_decomposition()
        for key in decomps:
            decomps[key].append(float(d[key].sum()))
    results["mean_variance_decomposition"] = {
        k: (float(np.mean(v)) if v else float("nan")) for k, v in decomps.items()
    }

    # --- highest-disagreement examples (for the Lever-7 sanity read) ---
    diss = []
    for i in test_idx:
        mix = mixtures[i]
        if mix is None:
            continue
        d = mix.with_s(s_fit).variance_decomposition()
        diss.append(
            {
                "claim_id": examples[i].claim_id,
                "claim": examples[i].claim,
                "label": examples[i].label_name,
                "within_model": float(d["within_model"].sum()),
                "between_model": float(d["between_model"].sum()),
                "pooled_mean": [round(x, 3) for x in d["pooled_mean"]],
            }
        )
    diss.sort(key=lambda r: r["between_model"] + r["within_model"], reverse=True)
    results["top_disagreement"] = diss[:10]

    return results


# ---------------------------------------------------------------------------
# Top-level run
# ---------------------------------------------------------------------------


async def run_benchmark(
    examples: list[Example],
    models: list[str],
    cache_path: Path,
    *,
    temperature: float = 0.0,
    s_initial: float = 50.0,
    alpha0: float = 0.5,
    dev_frac: float = 0.3,
    n_mc_samples: int = 2000,
    concurrency: int = 6,
    seed: int = 42,
) -> dict:
    """Elicit (cached), build mixtures, fit ``s``, score, and return results."""
    logger.info(
        "dataset: %d examples; labels=%s", len(examples), label_counts(examples)
    )
    elicitor = Elicitor(models, temperature=temperature)
    cache = await gather_elicitations(
        examples, elicitor, models, cache_path, concurrency
    )
    mixtures = build_all_mixtures(examples, cache, models, s_initial, alpha0)
    n_built = sum(m is not None for m in mixtures)
    logger.info("mixtures built: %d/%d", n_built, len(examples))
    results = evaluate(examples, mixtures, models, dev_frac, n_mc_samples, seed)
    results["dataset"] = {
        "n_examples": len(examples),
        "n_mixtures_built": n_built,
        "labels": label_counts(examples),
        "models": models,
    }
    return results
