"""Judge-scoring-method experiment: does a *continuous* score buy anything over
the module's *argmax* verdict, and are token logprobs even alive on our models?

Motivated by a paper (logit-expectation judging) an agent surfaced. Two distinct
probability sources must NOT be conflated:

  * LOGIT route (Test 0 / Test 2): read the model's token-logit distribution over
    score tokens and take its expectation. This is the paper's actual method. It
    needs raw logprobs, which Ollama exposes on /api/generate but historically
    DROPS on the OpenAI-compatible /v1/chat/completions endpoint — the one
    pydantic-ai (and so andamentum.llm_judge.elicit) talks to. So Test 0 probes
    /api/generate directly; it does not go through the module.

  * VERBALIZED route (Test 1): the module ALREADY makes the judge write a belief
    distribution (meets/partial/fails, or response_1/tie/response_2 summing to
    100). Test 1 reuses that exact elicitation and asks a cheaper question: does
    reducing that distribution by its EXPECTATION beat reducing it by ARGMAX?
    No logprobs, works on every model, shippable today. It is a PROXY for the
    paper's idea, not the paper's method — reported as such.

Test 3 (repeated evaluation): average the continuous score over K temperature-0.7
draws; orthogonal to the above, model-agnostic.

This is a research harness, not shipped code. It imports the module's private
elicitation seam (`andamentum.llm_judge.elicit`) deliberately, to measure the
real thing rather than a re-implementation.

Subcommands
-----------
    uv run python experiments/judge_scoring/run.py peek    --model ollama:gemma4:31b-cloud [--limit N]
    uv run python experiments/judge_scoring/run.py dataset  --n 150         # expand examples for a powered run
    uv run python experiments/judge_scoring/run.py test1    --model M [--model M ...] [--limit N] [--data FILE]
    uv run python experiments/judge_scoring/run.py test3    --model M --k 8 [--limit N]

Every model call is cached to results/cache.jsonl keyed by (model, kind, id,
order, temperature, draw) so a killed run resumes for free and reductions are
recomputed offline without re-calling. Ollama calls run strictly sequentially
(the repo rule: never two local-model calls at once).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import sys
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[1]
sys.path.insert(0, str(_REPO / "src"))

from andamentum.llm_judge import elicit, signals  # noqa: E402
from andamentum.llm_judge.schemas import Criterion  # noqa: E402

# The benchmark's single factual-accuracy criterion, verbatim, so score results
# are directly comparable to benchmarks/llm_judge.
FACTUAL_ACCURACY = Criterion(
    name="factual accuracy",
    description=(
        "The statement is factually correct and supported by the provided evidence: "
        "'meets' if the evidence supports it, 'fails' if the evidence contradicts it, "
        "'partial' if the evidence is insufficient to decide."
    ),
)

_EXAMPLES = _REPO / "benchmarks" / "llm_judge" / "examples.json"
_CACHE = _HERE / "results" / "cache.jsonl"
_OLLAMA_GENERATE = "http://localhost:11434/api/generate"


# ── cache ────────────────────────────────────────────────────────────────


def _load_cache() -> dict[str, Any]:
    if not _CACHE.exists():
        return {}
    out: dict[str, Any] = {}
    for line in _CACHE.read_text().splitlines():
        if line.strip():
            rec = json.loads(line)
            out[rec["key"]] = rec["value"]
    return out


def _append_cache(key: str, value: Any) -> None:
    _CACHE.parent.mkdir(parents=True, exist_ok=True)
    with _CACHE.open("a") as fh:
        fh.write(json.dumps({"key": key, "value": value}) + "\n")


_CACHE_MEM = _load_cache()


# ── Test 0: logprob peek (LOGIT route, ollama /api/generate only) ─────────


async def _peek_one(model_name: str, claim: str, evidence: str) -> dict[str, Any]:
    """One raw scoring call; read the logit distribution over the score token."""
    prompt = (
        "You are grading how well a piece of evidence supports a claim.\n\n"
        f"CLAIM: {claim}\n\nEVIDENCE: {evidence}\n\n"
        "On an integer scale from 0 (evidence fully contradicts the claim) to "
        "9 (evidence fully supports the claim), output ONLY the single digit:\n"
        "SCORE: "
    )
    async with httpx.AsyncClient(timeout=600) as client:
        resp = await client.post(
            _OLLAMA_GENERATE,
            json={
                "model": model_name,
                "prompt": prompt,
                "options": {"temperature": 0},
                "logprobs": True,
                "top_logprobs": 20,
                "stream": False,
            },
        )
    resp.raise_for_status()
    body = resp.json()
    # Ollama returns per-generated-token logprobs; take the first token that is
    # a digit (the score), fall back to the very first token.
    toks = body.get("logprobs") or []
    if not toks:
        return {"available": False, "raw_response_keys": list(body)}
    chosen = None
    for t in toks:
        if t.get("token", "").strip().isdigit():
            chosen = t
            break
    chosen = chosen or toks[0]
    tops = chosen.get("top_logprobs") or []
    probs = sorted((math.exp(e["logprob"]) for e in tops), reverse=True)
    total = sum(probs) or 1.0
    norm = [p / total for p in probs]
    entropy = -sum(p * math.log(p) for p in norm if p > 0)
    return {
        "available": True,
        "token": chosen.get("token"),
        "top_prob": norm[0] if norm else None,
        "n_tokens_over_1pct": sum(1 for p in norm if p >= 0.01),
        "entropy_nats": entropy,
        "top5": [
            {"tok": e["token"], "p": round(math.exp(e["logprob"]) / total, 4)}
            for e in tops[:5]
        ],
    }


async def cmd_peek(args: argparse.Namespace) -> None:
    model_name = args.model[0].split(":", 1)[1] if args.model[0].startswith("ollama:") else args.model[0]
    data = json.loads(_EXAMPLES.read_text())["score"][: args.limit]
    print(f"Test 0 — logprob peek on {model_name} ({len(data)} claims)\n")
    results = []
    for ex in data:  # strictly sequential
        r = await _peek_one(model_name, ex["claim"], ex["evidence"])
        results.append(r)
        if not r["available"]:
            print(f"  logprobs NOT returned. keys={r.get('raw_response_keys')}")
            print("  -> the logit route is unavailable on this call path; "
                  "argmax/verbalized only.")
            break
        print(f"  id={ex['id'][:16]:16} tok={r['token']!r} "
              f"top_p={r['top_prob']:.3f} entropy={r['entropy_nats']:.3f} "
              f">1%: {r['n_tokens_over_1pct']}  {r['top5']}")
    avail = [r for r in results if r.get("available")]
    if avail:
        mean_top = sum(r["top_prob"] for r in avail) / len(avail)
        mean_h = sum(r["entropy_nats"] for r in avail) / len(avail)
        print(f"\n  mean top-token prob: {mean_top:.3f}   mean entropy: {mean_h:.3f} nats")
        print("  VERDICT:", "one-hot — logits buy nothing here"
              if mean_top > 0.9 else "mass is spread — logit route worth pursuing")
    _dump("peek", model_name, results)


# ── elicitation with cache (VERBALIZED route) ─────────────────────────────


async def _score_dist(model: str, ex: dict, *, temperature: float, draw: int) -> list[float]:
    key = f"score|{model}|{ex['id']}|-|{temperature}|{draw}"
    if key in _CACHE_MEM:
        return _CACHE_MEM[key]
    output = f"CLAIM: {ex['claim']}\n\nEVIDENCE: {ex['evidence']}"
    d = await elicit.elicit_criterion_score(
        output, FACTUAL_ACCURACY, None, model=model, temperature=temperature
    )
    dist = signals.normalize_three(d.meets, d.partial, d.fails)
    _CACHE_MEM[key] = dist
    _append_cache(key, dist)
    return dist


async def _compare_avg(model: str, ex: dict, *, temperature: float, draw: int) -> list[float]:
    """Order-averaged canonical [pa, ptie, pb] — mirrors panel._judge_compare_once."""
    key = f"compare|{model}|{ex['id']}|avg|{temperature}|{draw}"
    if key in _CACHE_MEM:
        return _CACHE_MEM[key]
    ctx = ex["question"]
    ab = await elicit.elicit_pairwise(
        ex["response_a"], ex["response_b"], ctx, [FACTUAL_ACCURACY],
        model=model, order="AB", temperature=temperature,
    )
    ba = await elicit.elicit_pairwise(
        ex["response_a"], ex["response_b"], ctx, [FACTUAL_ACCURACY],
        model=model, order="BA", temperature=temperature,
    )
    canon_ab = signals.canonicalize(signals.normalize_three(*ab.to_row()), "AB")
    canon_ba = signals.canonicalize(signals.normalize_three(*ba.to_row()), "BA")
    avg = signals.order_average(canon_ab, canon_ba)
    _CACHE_MEM[key] = avg
    _append_cache(key, avg)
    return avg


# ── Test 1: argmax vs expectation ─────────────────────────────────────────


def _auroc(pos: list[float], neg: list[float]) -> float | None:
    if not pos or not neg:
        return None
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


async def cmd_test1(args: argparse.Namespace) -> None:
    data = json.loads((Path(args.data) if args.data else _EXAMPLES).read_text())
    compare = data["compare"][: args.limit]
    score = data["score"][: args.limit]

    for model in args.model:
        print(f"\n{'='*70}\n{model}\n{'='*70}")

        # -- compare: gold is 'a' or 'b' (never tie) --
        argmax_correct = argmax_tie = exp_correct = 0
        margins = []
        for ex in compare:
            avg = await _compare_avg(model, ex, temperature=0.0, draw=0)
            verdict = signals.compare_verdict(avg)          # module's argmax reduction
            expected = avg[0] + 0.5 * avg[1]                # E[preference for a] in [0,1]
            gold = ex["gold"].lower()
            if verdict == "tie":
                argmax_tie += 1
            elif verdict == gold:
                argmax_correct += 1
            exp_pred = "a" if expected > 0.5 else "b"       # forced decision, no abstain
            if exp_pred == gold:
                exp_correct += 1
            margins.append(expected if gold == "a" else 1 - expected)
        n = len(compare)
        decided = n - argmax_tie
        print(f"\ncompare (n={n}, gold has no ties):")
        print(f"  argmax   : acc {argmax_correct/n:.0%} (tie=wrong) | "
              f"tie-rate {argmax_tie/n:.0%} | acc-on-decided "
              f"{(argmax_correct/decided) if decided else float('nan'):.0%}")
        print(f"  expected : acc {exp_correct/n:.0%} (E>0.5, never abstains) | "
              f"mean margin-toward-gold {sum(margins)/n:.3f}")

        # -- score: SUPPORT/CONTRADICT/NOINFO; measure separation, not just argmax --
        s_correct = 0
        quality = {"SUPPORT": [], "CONTRADICT": [], "NOINFO": []}
        for ex in score:
            dist = await _score_dist(model, ex, temperature=0.0, draw=0)
            verdict = signals.argmax_label(dist, signals.SCORE_LABELS, signals.SCORE_TIEBREAK)
            if verdict == ex["expected_verdict"]:
                s_correct += 1
            q = dist[0] * 1.0 + dist[1] * 0.5             # continuous quality in [0,1]
            quality[ex["gold_label"]].append(q)
        ns = len(score)
        auroc = _auroc(quality["SUPPORT"], quality["CONTRADICT"])
        print(f"\nscore (n={ns}):")
        print(f"  argmax   : acc vs expected_verdict {s_correct/ns:.0%}")
        print(f"  expected : AUROC(quality; SUPPORT vs CONTRADICT) "
              f"{auroc:.3f}" if auroc is not None else "  expected : AUROC n/a")

        _dump("test1", model, {
            "compare": {"n": n, "argmax_acc": argmax_correct / n,
                        "argmax_tie_rate": argmax_tie / n,
                        "expected_acc": exp_correct / n},
            "score": {"n": ns, "argmax_acc": s_correct / ns, "auroc_support_vs_contradict": auroc},
        })


# ── Test 2: LOGIT-EXPECTATION (the paper's actual method, OpenAI raw) ─────
#
# Reads the model's token-logit distribution over a numeric score token and
# takes its expectation. Needs a raw OpenAI call — the module's structured
# output hides logprobs. These models cap top_logprobs at 5 and require
# max_completion_tokens (not max_tokens).

_OAI_CLIENT: Any = None


def _oai():
    global _OAI_CLIENT
    if _OAI_CLIENT is None:
        from openai import AsyncOpenAI

        _OAI_CLIENT = AsyncOpenAI()
    return _OAI_CLIENT


def _bare(model: str) -> str:
    return model.split(":", 1)[1] if model.startswith("openai:") else model


async def _logit_dist_over_digits(
    model: str, prompt: str, top_g: int, *, marker: str | None = None, budget: int = 24
) -> dict[int, float]:
    """Return {digit: prob} over the score token's top-5 logprobs.

    ``marker``: when the model is allowed to REASON before scoring, a stray digit
    in the reasoning ("study 2 showed...") would otherwise be mistaken for the
    score. So only accept the first digit token appearing AFTER a token
    containing the marker (e.g. "SCORE").
    """
    r = await _oai().chat.completions.create(
        model=_bare(model),
        messages=[{"role": "user", "content": prompt}],
        logprobs=True,
        top_logprobs=5,
        max_completion_tokens=budget,
    )
    content = r.choices[0].logprobs.content if r.choices[0].logprobs else None
    if not content:
        return {}

    def _ascii_digit(s: str) -> bool:
        # `str.isdigit()` is True for Unicode digits like '②', which int() then
        # rejects. The model really does put such tokens in its top-5.
        return s.isascii() and s.isdigit()

    def _is_score(t) -> bool:
        s = t.token.strip()
        return _ascii_digit(s) and 0 <= int(s) <= top_g

    if marker is None:
        tok = next((t for t in content if _is_score(t)), None)
    else:
        # The marker is NOT reliably one token — "SCORE" tokenizes as 'S'+'CORE'.
        # So accumulate the emitted text and arm on the marker appearing in it,
        # then take the first in-range digit token after that point.
        seen = False
        acc = ""
        tok = None
        for t in content:
            if not seen:
                acc += t.token
                if marker in acc.upper():
                    seen = True
                continue
            if _is_score(t):
                tok = t
                break
    if tok is None:
        return {}
    out: dict[int, float] = {}
    for e in tok.top_logprobs:
        s = e.token.strip()
        if _ascii_digit(s) and 0 <= int(s) <= top_g:
            out[int(s)] = out.get(int(s), 0.0) + math.exp(e.logprob)
    total = sum(out.values())
    return {k: v / total for k, v in out.items()} if total > 0 else {}


async def _logit_score(model: str, ex: dict, g: int, *, reason: bool = False) -> float | None:
    """Expected support score in [0,1] from the logit distribution (0..g).

    ``reason=True`` lets the judge reason BEFORE emitting the score token — the
    fair comparison against the verbalized route, which always reasons first
    (derive-then-judge). Without it we would be testing reasoning-vs-none, not
    logits-vs-verbalized.
    """
    tag = "r" if reason else "0"
    key = f"logit-score|{model}|{ex['id']}|g{g}|{tag}|0"
    if key in _CACHE_MEM:
        return _CACHE_MEM[key]
    head = (
        "Grade how well the evidence supports the claim.\n\n"
        f"CLAIM: {ex['claim']}\n\nEVIDENCE: {ex['evidence']}\n\n"
        f"Use an integer scale from 0 (evidence fully contradicts) to {g} "
        f"(evidence fully supports).\n"
    )
    if reason:
        prompt = head + (
            "First reason in one or two short sentences. Then, on a new line, "
            "give your final answer in exactly this form:\nSCORE: <integer>"
        )
        dist = await _logit_dist_over_digits(model, prompt, g, marker="SCORE", budget=300)
    else:
        prompt = head + "Reply with ONLY the single integer:"
        dist = await _logit_dist_over_digits(model, prompt, g)
    val = None if not dist else sum(k * p for k, p in dist.items()) / g
    _CACHE_MEM[key] = val
    _append_cache(key, val)
    return val


async def _logit_compare_one(model: str, ex: dict, order: str, *, reason: bool) -> float | None:
    """p(Response 1 is better) from the logit of the '1' vs '2' token."""
    r1, r2 = (ex["response_a"], ex["response_b"]) if order == "AB" else (ex["response_b"], ex["response_a"])
    head = (
        f"QUESTION: {ex['question']}\n\n--- RESPONSE 1 ---\n{r1}\n\n"
        f"--- RESPONSE 2 ---\n{r2}\n\nWhich response better answers the question?\n"
    )
    if reason:
        prompt = head + (
            "First reason in one or two short sentences. Then, on a new line, "
            "give your final answer in exactly this form:\nANSWER: <1 or 2>"
        )
        dist = await _logit_dist_over_digits(model, prompt, 2, marker="ANSWER", budget=300)
    else:
        prompt = head + "Reply with ONLY the single digit 1 or 2:"
        dist = await _logit_dist_over_digits(model, prompt, 2)
    p1, p2 = dist.get(1, 0.0), dist.get(2, 0.0)
    if p1 + p2 == 0:
        return None
    return p1 / (p1 + p2)  # p(shown-first is better)


async def _logit_compare(model: str, ex: dict, *, reason: bool = False) -> float | None:
    """Order-averaged p(response_a is better) in [0,1]."""
    tag = "r" if reason else "0"
    key = f"logit-compare|{model}|{ex['id']}|avg|{tag}|0"
    if key in _CACHE_MEM:
        return _CACHE_MEM[key]
    ab = await _logit_compare_one(model, ex, "AB", reason=reason)  # p(a better)
    ba = await _logit_compare_one(model, ex, "BA", reason=reason)  # p(b better) -> 1-that
    vals = [v for v in (ab, (1 - ba) if ba is not None else None) if v is not None]
    val = sum(vals) / len(vals) if vals else None
    _CACHE_MEM[key] = val
    _append_cache(key, val)
    return val


async def cmd_logit(args: argparse.Namespace) -> None:
    data = json.loads((Path(args.data) if args.data else _EXAMPLES).read_text())
    compare = data["compare"][: args.limit]
    score = data["score"][: args.limit]
    granularities = [args.g] if args.g else [2, 5, 9]
    reason = args.reason

    for model in args.model:
        mode = "REASON-THEN-SCORE" if reason else "SCORE-ONLY (no reasoning)"
        print(f"\n{'='*70}\n{model}  —  LOGIT-EXPECTATION [{mode}]\n{'='*70}")

        # compare: continuous logit preference, order-averaged
        correct = abstain = 0
        for ex in compare:
            pa = await _logit_compare(model, ex, reason=reason)
            if pa is None:
                abstain += 1
                continue
            correct += ("a" if pa > 0.5 else "b") == ex["gold"].lower()
        n = len(compare)
        print(f"\ncompare (n={n}): logit-expectation acc "
              f"{correct/n:.0%}  (abstain {abstain})")

        # score: AUROC per granularity — does finer granularity separate better?
        print(f"\nscore (n={len(score)}): AUROC(logit-score; SUPPORT vs CONTRADICT)")
        for g in granularities:
            q = {"SUPPORT": [], "CONTRADICT": [], "NOINFO": []}
            miss = 0
            for ex in score:
                v = await _logit_score(model, ex, g, reason=reason)
                if v is None:
                    miss += 1
                    continue
                q[ex["gold_label"]].append(v)
            auroc = _auroc(q["SUPPORT"], q["CONTRADICT"])
            snr = _snr(q["SUPPORT"], q["CONTRADICT"])
            print(f"  G={g:2d} (0..{g}): AUROC {auroc if auroc is None else round(auroc,3)}"
                  f"   SNR {snr if snr is None else round(snr,2)}   (missing {miss})")

        _dump("logit", model, {"note": "logit-expectation; see stdout"})


def _snr(pos: list[float], neg: list[float]) -> float | None:
    """Signal-to-noise of the score separation (paper's lens)."""
    if len(pos) < 2 or len(neg) < 2:
        return None
    import statistics as st

    diff = st.mean(pos) - st.mean(neg)
    spread = math.sqrt(st.pvariance(pos) + st.pvariance(neg)) or 1e-9
    return diff / spread


# ── Test 3: repeated evaluation ───────────────────────────────────────────


async def cmd_test3(args: argparse.Namespace) -> None:
    data = json.loads(_EXAMPLES.read_text())["compare"][: args.limit]
    for model in args.model:
        print(f"\n{model}: repeated-evaluation, K={args.k}, temp=0.7")
        for k in (1, 4, args.k):
            correct = 0
            spreads = []
            for ex in data:
                draws = [
                    (await _compare_avg(model, ex, temperature=0.7, draw=i))[0]
                    + 0.5 * (await _compare_avg(model, ex, temperature=0.7, draw=i))[1]
                    for i in range(k)
                ]
                mean_e = sum(draws) / k
                spreads.append(max(draws) - min(draws))
                pred = "a" if mean_e > 0.5 else "b"
                correct += pred == ex["gold"].lower()
            print(f"  K={k:2d}: acc {correct/len(data):.0%}  "
                  f"mean draw-spread {sum(spreads)/len(spreads):.3f}")


# ── dataset expansion (for a powered run; not called in the cheap slice) ──


def cmd_dataset(args: argparse.Namespace) -> None:
    sys.path.insert(0, str(_REPO / "experiments" / "pairwise_judge"))
    sys.path.insert(0, str(_REPO / "experiments" / "dirichlet_confidence"))
    import judgebench  # type: ignore
    import scifact  # type: ignore

    n = args.n

    def _balanced(items, key, want_per_class):
        """Round-robin across classes so AUROC/accuracy aren't dominated by one label."""
        buckets: dict[Any, list] = {}
        for it in items:
            buckets.setdefault(key(it), []).append(it)
        out = []
        for i in range(want_per_class):
            for cls in sorted(buckets):
                if i < len(buckets[cls]):
                    out.append(buckets[cls][i])
        return out

    jb = judgebench.load_judgebench()
    jb_bal = _balanced(jb, lambda e: e.gold, n)  # balance gold a/b
    compare = [
        {"id": str(e.pair_id), "question": e.question,
         "response_a": e.response_a, "response_b": e.response_b,
         "gold": "a" if e.gold == 0 else "b", "source": "judgebench"}
        for e in jb_bal[: 2 * n]
    ]
    sf = scifact.load_scifact(str(_REPO / "experiments/dirichlet_confidence/data/scifact"))
    _EXP = {"SUPPORT": "meets", "CONTRADICT": "fails", "NOINFO": "partial"}
    sf_bal = _balanced(sf, lambda e: e.label_name, n)  # balance SUPPORT/CONTRADICT/NOINFO
    score = [
        {"id": str(e.claim_id), "claim": e.claim, "evidence": e.evidence,
         "gold_label": e.label_name, "expected_verdict": _EXP[e.label_name]}
        for e in sf_bal[: 3 * n]
    ]
    out = _HERE / "data_powered.json"
    out.write_text(json.dumps({"compare": compare, "score": score}, indent=2))
    print(f"wrote {len(compare)} compare + {len(score)} score -> {out}")


def _dump(test: str, model: str, payload: Any) -> None:
    safe = model.replace(":", "_").replace("/", "_")
    (_HERE / "results" / f"{test}_{safe}.json").write_text(json.dumps(payload, indent=2))


def main() -> None:
    load_dotenv()
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("peek", "test1", "logit", "test3", "dataset"):
        sp = sub.add_parser(name)
        if name != "dataset":
            sp.add_argument("--model", action="append", required=True)
            sp.add_argument("--limit", type=int, default=None)
        if name in ("test1", "logit"):
            sp.add_argument("--data", default=None)
        if name == "logit":
            sp.add_argument("--reason", action="store_true",
                            help="let the judge reason BEFORE the score token "
                                 "(fair comparison vs the verbalized route)")
            sp.add_argument("--g", type=int, default=None,
                            help="single granularity instead of the 2/5/9 sweep")
        if name == "test3":
            sp.add_argument("--k", type=int, default=8)
        if name == "dataset":
            sp.add_argument("--n", type=int, default=150)
    args = p.parse_args()
    fn = {"peek": cmd_peek, "test1": cmd_test1, "logit": cmd_logit, "test3": cmd_test3}.get(args.cmd)
    if args.cmd == "dataset":
        cmd_dataset(args)
    else:
        asyncio.run(fn(args))


if __name__ == "__main__":
    main()
