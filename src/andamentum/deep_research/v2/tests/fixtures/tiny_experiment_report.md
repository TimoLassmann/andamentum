# Experiment tiny



Registered 2026-10-06; experiment hash `f2bbf73defa6`; 16 run records, 0 with errors (an errored run counts as a miss on the outcome metrics, never as absent).

Concurrency used: 1 cell(s) of one arm at a time; decision-model requests are always one at a time. Wall times are measured under that load.

Live arms (base, other) run Plan and Search themselves: the legacy pipeline cannot take frozen inputs, so a comparison with it is end to end, and any web or source drift between runs is part of the difference. Comparisons among frozen arms share identical inputs.

## Arms

| arm | system | preset / screen | config hash | model | decision model (digest) | inputs | repeats |
|---|---|---|---|---|---|---|---|
| base | v2 | baseline | 9621c6dae218 | m | - (no digest) | live | 2 |
| other | v2 | baseline | 9621c6dae218 | m | - (no digest) | live | 2 |

## Set web

4 items, seed 0.

| arm | items | errors | correct | claim_precision | unsourced_share | link_integrity |
|---|---|---|---|---|---|---|
| base | 4 | 0 | 1.000 | n/a | n/a | n/a |
| other | 4 | 0 | 1.000 | n/a | n/a | n/a |

Difference from the baseline arm **base** (arm minus baseline), mean over items, 95% bootstrap interval (200 resamples over items). `(k of N)`: the items both arms have a value for, of all items either ran; an item without a value (no claims to judge, an error) drops out of that cell:

| arm | correct | claim_precision | unsourced_share | link_integrity |
|---|---|---|---|---|
| other | +0.000 [+0.000, +0.000] (4 of 4) | n/a (0 of 4) | n/a (0 of 4) | n/a (0 of 4) |

### Repeat stability (web)

| arm | inputs | repeats | kept_jaccard | cited_jaccard | rank_tau | findings_reproduced | answer_agreement | subquestion_overlap |
|---|---|---|---|---|---|---|---|---|
| base | live | 2 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| other | live | 2 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

Values are means over items of the pairwise agreement between repeats (1.0 = identical). `n/a`: not applicable or fewer than two usable repeats (`subquestion_overlap` only for live arms, where the plan can differ).

Spread of a metric across repeats, averaged over items:

| arm | metric | mean sd | mean max-min | items |
|---|---|---|---|---|
| base | correct | 0.000 | 0.000 | 4 |
| base | claim_precision | n/a | n/a | 0 |
| other | correct | 0.000 | 0.000 | 4 |
| other | claim_precision | n/a | n/a | 0 |

## Cost

| arm | set | runs | median wall s | total wall s | mean llm calls | mean decision calls | mean input tokens | mean output tokens |
|---|---|---|---|---|---|---|---|---|
| base | web | 8 | 1.5 | 12 | 2.0 | 3.0 | n/a | n/a |
| other | web | 8 | 1.5 | 12 | 2.0 | 3.0 | n/a | n/a |

Frozen arms start from a snapshot and run offline: their wall time excludes Plan, Search and every page fetch. Only live arms measure end-to-end time. The legacy pipeline counts no tokens or calls (n/a).

## Where the time and the requests went

Seconds per step, median over runs (a step a run did not reach counts as 0; `steps` is the median of each run's sum, `wall` of each run's total):

| arm | set | runs | Plan | Search | Screen | Expand | Read | Coverage | Write | Verify | steps | wall |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| base | web | 8 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| other | web | 8 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |

Model requests per step, mean per run, `language model / decision model` (decision requests served from the cache are not requests):

| arm | set | runs | Plan | Search | Screen | Expand | Read | Coverage | Write | Verify |
|---|---|---|---|---|---|---|---|---|---|---|
| base | web | 8 | 0.0 / 0.0 | 0.0 / 0.0 | 0.0 / 0.0 | 0.0 / 0.0 | 0.0 / 0.0 | 0.0 / 0.0 | 0.0 / 0.0 | 0.0 / 0.0 |
| other | web | 8 | 0.0 / 0.0 | 0.0 / 0.0 | 0.0 / 0.0 | 0.0 / 0.0 | 0.0 / 0.0 | 0.0 / 0.0 | 0.0 / 0.0 | 0.0 / 0.0 |

Claim-support checks settled in code and by the model (means per run, `share` = code / all), decision answers served from the cache, and the cache state of the runs:

| arm | set | runs | by code | by model | share by code | cache hits | decision cache |
|---|---|---|---|---|---|---|---|
| base | web | 8 | 0.0 | 0.0 | n/a | 0.0 | off 8 |
| other | web | 8 | 0.0 | 0.0 | n/a | 0.0 | off 8 |

## Verdicts (pre-registered rules)

| rule | group | question | role | verdict | evidence |
|---|---|---|---|---|---|
| R-correct | - | other is not worse than base on correctness | gate | **accepted** | other minus base, oriented so that higher is better: +0.000 (95% interval +0.000 to +0.000, 4 items); bar: not worse than -0.1 |
| R-agree | - | base answers agree across repeats | reported | **accepted** | base mean stability.answer_agreement: 1.000 (95% interval 1.000 to 1.000, 4 items); bar: >= 0.9 |

Gating rules together: **accepted**.
