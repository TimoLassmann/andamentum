"""results/deviations.json — every change made AFTER a number was seen.

A pre-registration that can be edited after the fact, with no amendment trail,
provides the appearance of the guarantee rather than the guarantee. This file is
that trail: one row per post-hoc change, each naming what moved, why, and which
hypothesis ids it could have shifted.

It is a DECLARED ARTEFACT (registered in ``schemas.ARTEFACTS``), so
``validate_results`` fails the run if it goes missing — the trail cannot be
quietly dropped the way an untracked note could.

Rows are added by hand, at the moment the change is made. Nothing here is
derived from the results, on purpose: a generated deviation log would only ever
record the deviations someone remembered to generate.

Run:  uv run python -m experiments.docstore_deferred_ingestion.scripts.deviations
"""

from __future__ import annotations

from typing import Any

from . import _common as C

SCHEMA = "andamentum.experiment.docstore_deferred.deviations/1"


def deviation(
    *,
    when: str,
    what: str,
    why: str,
    affects: list[str],
    direction: str,
) -> dict[str, Any]:
    """One amendment. ``direction`` states honestly which way it could have cut."""
    return {
        "when": when,
        "what": what,
        "why": why,
        "affects": affects,
        "could_have_moved_the_verdict": direction,
    }


#: EDITION 2 — every change made after edition 1's numbers were on disk and had
#: been read by independent reviewers. Edition 1's own artefacts were not
#: archived (``rule archive`` was never invoked), which is itself recorded below.
DEVIATIONS: list[dict[str, Any]] = [
    deviation(
        when="after edition 1, before edition 2's measurements",
        what=(
            "H-f2's registered metric renamed from `stage_test_converter_calls` to "
            "`stage_test_converter_calls_good_doc`, and the falsifier narrowed from "
            "'any converter re-invocation' to 'any re-invocation on the "
            "markdown-bearing document'. The unscoped total is now its own "
            "OBSERVATION row under `stage_test_converter_calls_total`."
        ),
        why=(
            "In edition 1 the registered metric measured 3 against an expected 0, and "
            "the analyzer was then edited to read a DIFFERENT artefact field under the "
            "registered name. The redefinition is defensible on the merits — "
            "retry_failed requeues every failed row, and the three still-broken "
            "sources hold no markdown, so re-converting them is the same rule applied "
            "correctly — but it was made after the metric failed and was presented as "
            "a clean pre-registered pass, with `stage_test_converter_calls` reading 0 "
            "in claims.json and 3 in claim_f_fail.json."
        ),
        affects=["H-f2"],
        direction=(
            "FAIL -> PASS under the original wording. The narrowed hypothesis is a "
            "weaker claim than the one first registered, and is labelled as such."
        ),
    ),
    deviation(
        when="after edition 1, before edition 2's measurements",
        what=(
            "H-d1's secondary threshold changed from `pause_discarded_seconds == 0` "
            "to `pause_discarded_document_fraction < 0.5`, and the primary metric "
            "`pause_midstage_documents` re-derived from EACH ARM'S OWN post-drain "
            "snapshot instead of the snapshot after the final unrestricted drain."
        ),
        why=(
            "Neither edition-1 metric could fail. `pause_discarded_seconds` was a "
            "hard-coded 0.0 literal compared against 0 — an assertion no behaviour of "
            "process_pending could falsify. `pause_midstage_documents` was computed "
            "after a drain that repairs exactly the state it was hunting for, so a "
            "genuinely stranded document would have been completed before the check "
            "ran. Both are now measurements."
        ),
        affects=["H-d1"],
        direction=(
            "Strictly harder to pass. Edition 1's PASS rows carried no measurement; "
            "these can fail."
        ),
    ),
    deviation(
        when="after edition 1, before edition 2's measurements",
        what=(
            "H-d5 added: the should_continue arm must report stopped_early=True with "
            "remaining >= 1. The arm's queue is topped up to 2 pending documents "
            "before it runs."
        ),
        why=(
            "In edition 1 arms 1 and 2 consumed 4 of the 5 queued documents, so the "
            "should_continue arm drained the single remaining one and the loop ended "
            "naturally: `stopped_early: false, remaining: 0`. should_continue never "
            "caused a stop, and the assertion (`processed <= 1`) passed on the "
            "exhausted queue. One of the three library pause mechanisms named in the "
            "claim was therefore not validated at all, while the README and report "
            "stated all four cooperative stops had been exercised."
        ),
        affects=["H-d1", "H-d5"],
        direction="Adds a claim that edition 1 asserted without evidence.",
    ),
    deviation(
        when="after edition 1, before edition 2's measurements",
        what=(
            "H-d4's metric renamed `kill_repeated_stages` -> "
            "`documents_pending_enrich_at_resume`, tightened from `<= 1` to `== 1`, "
            "and a second threshold added on "
            "`kill_repeated_enrichment_over_conversion < 1.0`."
        ),
        why=(
            "The old name claimed to count repeated stages while counting rows in a "
            "state. The in-script companion assertion (`saved_documents <= 1`) could "
            "not fail in the direction it named: a completely broken checkpoint that "
            "re-converted all three sources gives saved_documents == 0, and `0 <= 1` "
            "passes. And the registered statement's seconds clause ('strictly less "
            "than its measured conversion time') had no metric attached at all, so it "
            "went unscored while the row read PASS."
        ),
        affects=["H-d4"],
        direction="Strictly harder to pass; adds the previously unscored clause.",
    ),
    deviation(
        when="after edition 1, before edition 2's measurements",
        what=(
            "H-e1's primary metric changed from `post_chunk_recall_at_3` to "
            "`post_chunk_recall_at_1`; the secondary control changed from "
            "`pre_chunk_recall_at_3 == 0` to the unified-RRF pre/post pair; chance "
            "baselines and the candidate-pool size are now published beside the score."
        ),
        why=(
            "`pre_chunk_recall_at_3 == 0` is guaranteed by H-a2's own assertion that "
            "deferred documents have zero chunks — semantic_search cannot return a "
            "row that does not exist — so the control added no information. And over "
            "a 5-document candidate pool chance recall@3 is 3/5, which recall@3 == "
            "1.0 barely beats. The unified stack scores above zero before the drain "
            "(FTS5 already answers some probes) and rising is the non-trivial "
            "contrast."
        ),
        affects=["H-e1"],
        direction=(
            "Harder to pass on the primary; the demoted control was unfalsifiable. "
            "The residual limitation (no hard negatives, 4 topically disjoint papers) "
            "is stated in the hypothesis rather than fixed."
        ),
    ),
    deviation(
        when="after edition 1, before edition 2's measurements",
        what=(
            "The INCONCLUSIVE / noise-band rule moved out of analyze.py into "
            "prereg.SCORING_RULES, and its band restricted to /v1/chat/completions "
            "latencies read from the raw per-request ledgers. "
            "`defer_median_seconds` removed from the softenable set; "
            "`concurrency_speedup` added and judged against 1.0 using the replicate "
            "spread."
        ),
        why=(
            "The rule was the only mechanism that could override a threshold "
            "comparison and it appeared nowhere in the pre-registration; it was also "
            "rewritten in edition 1 AFTER observing that it scored H-a3 INCONCLUSIVE. "
            "Its band pooled 25 chat latencies with 12 order statistics of "
            "mixed-endpoint summaries, so the published 'relative spread of per-call "
            "LLM latency' (1.413) was the CV of a bimodal mixture. And "
            "`defer_median_seconds` is an absolute duration of a path this run proves "
            "makes zero LLM calls."
        ),
        affects=["H-a3", "H-m1", "all ratio metrics"],
        direction=(
            "Narrower band, so fewer softenings; but concurrency_speedup can now be "
            "reported INCONCLUSIVE where edition 1 published it as a measured effect."
        ),
    ),
    deviation(
        when="after edition 1, before edition 2's measurements",
        what=(
            "H-x's statement kept at six ProcessReport fields, and reconcile() "
            "extended from four to six (documents_skipped, stopped_early); "
            "claim_c_resume's subprocess drain now reconciled too."
        ),
        why=(
            "The registered statement named six fields; the computation checked four "
            "and skipped the kill lineage entirely. A hypothesis whose statement "
            "outruns its metric turns a PASS into a claim nobody tested."
        ),
        affects=["H-x"],
        direction="Strictly more to fail on.",
    ),
    deviation(
        when="after edition 1, before edition 2's measurements",
        what=(
            "H-m1 registered (concurrency_speedup <= 1.3), and the ratio re-attributed "
            "by doc_uuid instead of by title prefix, refusing tick index 0."
        ),
        why=(
            "dfr_main holds the target paper under TWO rows with identical titles — "
            "the markdown ingest and claim (a) arm 2's source probe — so edition 1's "
            "title match landed on the source row at tick 0, whose interval starts at "
            "the drain's t0 and therefore contained the preflight tax and a full "
            "Docling conversion. The published 1.067 was the wrong row with two extra "
            "stages in its denominator; the correct figure was ~1.010. The metric also "
            "had no registered threshold, so it was never scored, yet reached the "
            "report as an unqualified fact on n=1."
        ),
        affects=["H-m1"],
        direction="Adds a scored row where edition 1 published an unscored assertion.",
    ),
    deviation(
        when="after edition 1, before edition 2's measurements",
        what=(
            "claim (d) gained arm 5: a second CLI SIGTERM measured with an "
            "UNTRUNCATED document in flight. `sigterm_to_exit_seconds` now comes from "
            "that arm; the truncated figure is retained as "
            "`sigterm_to_exit_seconds_truncated`."
        ),
        why=(
            "Edition 1 measured 67.57 s on 6,000-char truncations (~2 chunks) and "
            "published it as guidance for sizing a launchd/cron shutdown grace period, "
            "with no truncation caveat, while the real corpus is 39k-70k chars (25-43 "
            "chunks) at 240-438 s per document. The shape of the bound transferred; "
            "the number did not."
        ),
        affects=["operational guidance, not a scored hypothesis"],
        direction="Replaces an understated constant with a measured one.",
    ),
    deviation(
        when="after edition 1, before edition 2's measurements",
        what=(
            "`checkpoint_savings_seconds` re-sourced from the kill lineage's own "
            "converter-ledger entry, and convert_reference now converts one PDF twice "
            "so `docling_init_seconds` is a measured constant rather than folded in."
        ),
        why=(
            "Edition 1 published `conversion_baseline`'s FIRST call (14.976 s) as the "
            "per-document conversion cost. Those four numbers fall monotonically in "
            "call order and are uncorrelated with size, i.e. one-time Docling/RapidOCR "
            "initialisation dominates the first — and the resume process pays that "
            "initialisation regardless, so the marginal saving was overstated by "
            "roughly 50%."
        ),
        affects=["H-d4", "cross-cutting checkpoint economics"],
        direction="Reduces a headline number; does not move a verdict.",
    ),
    deviation(
        when="after edition 1, before edition 2's measurements",
        what=(
            "`defer_now_ratio` computed on the control document itself rather than as "
            "the median over four documents divided by the single control run."
        ),
        why=(
            "The numerator and denominator were different quantities. The "
            "same-document ratio was available for free; the median remains published "
            "as H-a3's separate absolute-latency metric."
        ),
        affects=["H-a3"],
        direction="No verdict change — four orders of magnitude below the threshold.",
    ),
    deviation(
        when="edition 2, DURING the run, after claim_d_pause failed loud",
        what=(
            "`top_up_queue` now stamps a per-CALL counter into each top-up "
            "document's text, and RAISES if the queue is still short afterwards. "
            "Consequence for provenance: `provenance.json`'s `harness.harness_sha256` "
            "is the harness as it stood at run START, while `MANIFEST.json`'s "
            "`harness_sha256` covers the FINAL harness. They differ by exactly this "
            "handful of files. By the end of the run those were "
            "`scripts/claim_d_pause.py` (this fix), `scripts/drain.py` (the "
            "post-commit clock origin), `scripts/manifest.py` (self-exclusion) and "
            "`scripts/deviations.py` (these entries). MANIFEST.json's "
            "`harness_sha256` covers the final state and is the one to trust; the "
            "measurements themselves were produced by the final versions, because "
            "every rule that consumes those files was re-run after they changed."
        ),
        why=(
            "Making the should_continue arm real (H-d5) meant calling top_up_queue "
            "TWICE — once before arm 3, once before arm 4. `ingest` deduplicates on a "
            "content hash, and the top-up text was keyed only on the document index, "
            "so call 2's document 0 was byte-identical to call 1's and silently "
            "collapsed onto it. The CLI arm then ran against a 2-document queue and "
            "drained it to completion: '[1/2] ... remaining: 0', no 'Paused' line. "
            "That is the SAME defect this edition set out to remove, reintroduced one "
            "arm to the left — and it was caught only because H-d/arm4_message asserts "
            "the operator-visible pause message rather than assuming it. The rule "
            "exited non-zero and Snakemake removed the artefact; nothing was published "
            "from the broken arm. `rule provenance` was deliberately NOT re-run: it "
            "would mint a new run_id and every existing artefact's provenance_ref "
            "would then fail validation, forcing a full re-measurement for a "
            "one-file change that only claim_d_pause consumes."
        ),
        affects=["H-d1", "H-d5"],
        direction=(
            "The broken arm FAILED and blocked the DAG. The fix makes the arm "
            "measurable at all; it cannot turn a fail into a pass, because the arm "
            "produced no publishable measurement before it."
        ),
    ),
    deviation(
        when="edition 2, after H-m1 was first scored",
        what=(
            "report.md now states plainly that concurrency_speedup has **n=1 and no "
            "variance estimate** in this run, instead of citing a replicate spread "
            "that came back null."
        ),
        why=(
            "H-m1's pre-registered rationale says the run 'contains a free replicate ... "
            "and the softening rule below uses it'. The mechanism is wired and "
            "pre-registered, but this run's geometry did not feed it: the second "
            "enrichment of identical content settled at a drain's FIRST progress tick, "
            "whose interval starts at the drain's t0 and so contains the preflight tax "
            "and a Docling conversion. `attribute_document_seconds` refuses index 0 by "
            "design — attributing it is precisely the error that produced edition 1's "
            "1.067 — so only one clean instance exists and the spread is null. Saying "
            "'n=1, no variance estimate' is more honest than manufacturing a band, and "
            "the softening rule remains registered for a run whose geometry supplies "
            "two attributable instances."
        ),
        affects=["H-m1"],
        direction=(
            "Weakens the published claim: 1.052 is a single observation consistent "
            "with 1.0, not a measured 5% effect. The registered threshold (<= 1.3) "
            "still passes."
        ),
    ),
    deviation(
        when="edition 2, after the first `--verify` run",
        what=(
            "`results/MANIFEST.json` is EXCLUDED from its own artefact list "
            "(`manifest.SELF`)."
        ),
        why=(
            "A file cannot contain its own hash. Including it recorded whatever the "
            "PREVIOUS run had written, so `manifest --verify` reported a permanent "
            "false discrepancy on exactly the file the integrity check exists to make "
            "trustworthy. Observed: 'results/MANIFEST.json: sha256 ce399f96 != "
            "recorded 948d5f64'."
        ),
        affects=["integrity checking, no hypothesis"],
        direction="Removes a false positive; every other artefact is still hashed.",
    ),
    deviation(
        when="edition 2, after the first complete scoreboard was read",
        what=(
            "`post_commit_seconds` now subtracts the last progress tick from a "
            "wall-clock measured against the SAME origin (`t0`) instead of against "
            "`started`."
        ),
        why=(
            "The two clocks are set a few milliseconds apart — `t0` before the "
            "initial settled-row poll, `started` after it — so the subtraction "
            "under-counted by that gap and returned a NEGATIVE duration whenever the "
            "true post-commit time was ~0. It published as "
            "`pause_discarded_seconds = -0.0187 s` and would have drawn negative bars "
            "in fig3, a figure whose whole point this edition is that every bar is a "
            "real measurement. The magnitude (~6 ms per arm against a 55 s document) "
            "never threatened H-d1's scored threshold, which read 0.0 either way."
        ),
        affects=["H-d1"],
        direction=(
            "Neutral on the verdict; removes a physically impossible value from a "
            "reported observation and a published figure."
        ),
    ),
    deviation(
        when="edition 1, not recorded at the time",
        what=(
            "Eight earlier runs were executed and discarded during edition 1 "
            "(logs/run_cycle1{,b,c,d,e,f,g,h}.log), and `rule archive` was never "
            "invoked, so no runs/<run_id>/ tree preserves any of them."
        ),
        why=(
            "Recorded here because it was not recorded anywhere else. Edition 2 "
            "invokes `rule archive` as part of the documented run command, so its own "
            "artefacts survive a subsequent re-run."
        ),
        affects=["reproducibility of edition 1"],
        direction="No verdict; a gap in the record, stated rather than concealed.",
    ),
    deviation(
        when="edition 1, during the run",
        what=(
            "A genuine bug was found and fixed in shipped code: "
            "`document_store/fts_query.py` upper-cased the query before testing for "
            "deliberate FTS5 power-query syntax, so prose containing a lowercase "
            "'and' was returned unescaped and a hyphenated token in the same query "
            "reached FTS5 raw (OperationalError: no such column). Edition 1 ran with "
            "that fix UNCOMMITTED, so its recorded git sha did not name the code that "
            "produced its numbers."
        ),
        why=(
            "FTS5 honours AND/OR/NOT/NEAR only in upper case. Edition 2 runs against "
            "the committed fix, and provenance now writes results/working_tree.patch "
            "whenever the tree is dirty."
        ),
        affects=["H-e1", "claim (e) probes"],
        direction=(
            "Edition 1's published claim (e) numbers are correct; the run was simply "
            "not reproducible from its own recorded commit."
        ),
    ),
]


def main() -> int:
    C.write_json(
        C.RESULTS / "deviations.json",
        {
            "n_deviations": len(DEVIATIONS),
            "policy": (
                "Every change to a hypothesis, metric, threshold or scoring rule made "
                "after a measurement was observed appears here, with the direction it "
                "could have moved a verdict. Written by hand at the moment of the "
                "change; a generated log would only record the deviations someone "
                "remembered to generate."
            ),
            "deviations": DEVIATIONS,
        },
        schema=SCHEMA,
    )
    print(f"wrote {C.RESULTS / 'deviations.json'} ({len(DEVIATIONS)} deviations)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
