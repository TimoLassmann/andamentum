# Judge scoring methods — argmax vs continuous scoring

**Question.** The shipped `andamentum.llm_judge` reduces a judge's verdict to an
**argmax** over a discrete label set (`meets`/`partial`/`fails`, `a`/`b`/`tie`).
A paper an agent surfaced argues for a **continuous** score instead — the
*expectation* of the model's score distribution — claiming fewer ties and better
ranking. Does that buy us anything on the models we actually run?

**Two probability sources, kept strictly separate.** They are not the same
method and must not be conflated:

| Route | Distribution read | Needs | Where it works |
|---|---|---|---|
| **Logit-expectation** (the paper's method) | the model's **token logprobs** over a score token | a raw API call exposing logprobs | see Test 0 |
| **Verbalized-expectation** (cheap proxy) | the belief points the judge **writes out** (already elicited by the module) | nothing extra | every model |

`run.py` implements both plus the argmax baseline, over two tasks: **compare**
(JudgeBench pairs, objective which-is-better) and **score** (SciFact claims,
SUPPORT/CONTRADICT/NOINFO under one factual-accuracy criterion — the same
criterion `benchmarks/llm_judge` uses).

## Models

Per the request: `gpt-5.4-nano` and `gpt-5.4-mini`. Ollama **cloud** models were
dropped — see Test 0. All calls cached to `results/cache.jsonl` (keyed by
model/task/id/order/temperature/draw), so runs resume for free and every
reduction is recomputed offline without re-calling.

## The tests (a decision tree — cheapest, most decisive first)

- **Test 0 — is the logit route even alive?** One raw call per model; inspect
  the score token's logprob spread. Gates everything logit-related.
- **Test 1 — verbalized argmax vs expectation.** The shippable-today question:
  does reducing the *written* belief distribution by its expectation beat the
  argmax? No logprobs. `run.py test1`.
- **Test 2 — logit-expectation (the paper's method).** Read token logprobs,
  take the expectation, sweep granularity G ∈ {2, 5, 9}. `run.py logit`.
- **Test 3 — repeated evaluation.** Average the continuous score over K temp-0.7
  draws. Orthogonal, model-agnostic. `run.py test3`.

## Test 0 result — the logit route on each backend

Ollama is **0.32.0** (the surfacing agent's "v0.12.11" was wrong). Findings:

| Backend | Logprobs on a score token? | Consequence |
|---|---|---|
| `gemma4:31b-cloud` (Ollama cloud) | **No** — identical `/api/generate` request that works locally returns no `logprobs` field | logit route unavailable; **dropped from the study** |
| local Ollama (`gemma4:12b-mxfp8`) | Yes | but **near one-hot** (mean top-token prob 0.94, entropy 0.16 nats) — expectation collapses to argmax |
| `gpt-5.4-nano` / `-mini` | **Yes**, via a raw call (`logprobs=True`, `top_logprobs≤5`, `max_completion_tokens`) — *not* through the module's structured-output path | the only place the logit route is worth running |

So the paper's method is only testable on the OpenAI models, which is where the
powered run lives.

## Reproduce

```bash
# gate
uv run python benchmarks/judge_scoring/run.py peek  --model ollama:gemma4:12b-mxfp8 --limit 6

# powered dataset (balanced: 50/class compare, 50/class score)
uv run python benchmarks/judge_scoring/run.py dataset --n 50

# the two methods, both models
D=benchmarks/judge_scoring/data_powered.json
uv run python benchmarks/judge_scoring/run.py logit --data $D --model openai:gpt-5.4-nano --model openai:gpt-5.4-mini
uv run python benchmarks/judge_scoring/run.py test1 --data $D --model openai:gpt-5.4-nano --model openai:gpt-5.4-mini
```

## Findings

See **[RESULTS.md](RESULTS.md)**. Headline: the paper's logit-expectation method
loses to the belief points the module already elicits — and the free win is exposing
the *expectation* of that distribution instead of collapsing it to an argmax.
