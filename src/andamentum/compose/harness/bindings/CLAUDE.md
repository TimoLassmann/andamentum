# andamentum.compose.harness.bindings — notes for Claude

Claude Code loads this file when it reads files in this directory. Two verbs, `bind-numbers` and `refresh`, registered in `kernel/verbs.py` `VERB_MODULES`. Nothing here imports `pydantic_graph`.

## What a binding is

A bound number is a hidden content control (`Bind` op, `kind="number"`, tag `am:<anchor id>`) around the number as the text writes it, plus an entry in `<project>/claims.yaml` (`ledger.py`): the sentence, `kind: result`, an evidence pointer (file, key, stored value) and a `binding` block (`NumberBinding`: anchor id, sentence, file, key, stored value, row label `context`, `written` text, `scale`). `refresh` reads only entries with a `binding` block and rewrites only those; every other entry in the file (the draft's `d<N>` claims) is kept as it is. `ledger.record` replaces entries by anchor id.

- **Key** = the label `checks.numbers.load_results` gives a number (`file row 2 column 'value'`, `file path[0].x`). It is positional. `results.py` adds the row's first cell when it is text as `context`; the prompt shows it and `refresh` refuses to refresh a binding whose key now sits in a row with a different label (error, nothing replaced). A binding with an empty `context` (made by code that did not record one) is not checked. A csv that is edited so that the same row label keeps a different meaning cannot be told from a changed value.
- **Scale**: 100 when the file stores a fraction and the text shows a percentage (`0.234` stored, `23.4%` written), else 1. Decided at bind time by `written.scale_for`, kept in the binding.

## bind-numbers

Per paragraph (`checks.numbers._examined`, at most 80, sequential): code lists the result numbers that equal one of the paragraph's numbers at its written precision (`candidates_for`, at most 40, the surplus noted in the report); the model sees the paragraph and the numbered candidates and answers, per result claim, with the sentence, the value as written and a candidate id or nothing. It never produces a file or a key. Code verifies (sentence and value occur, span not already bound, candidate exists, value equal within the written precision) and only then builds the `Bind` op and ledger entry. Anything proposed and not bound is listed with its reason; numbers the model did not propose are not listed. `--model` is required (the verb exits 2 without it): without a model there is no proposal step, and a deterministic fallback would be a second code path. Answers are cached in the project (`cache/bind_numbers/`) by model, paragraph and candidates.

## refresh

For each ledger binding: control present? key present (and row label unchanged)? text still a number? Then `format_like` writes the stored value in the shape of the text as it stands now (decimals, `%`, thousands separator, minus sign kind, no leading zero for `.03`; rounding is half up on the decimal repr). Same text → untouched. Otherwise a `Replace` over the control's range and one comment per sentence holding changed numbers (`[warning] refresh: value changed 48 → 52; check the claim still holds.`). A number whose text differs from the ledger's `written` was edited by hand: replaced from the results all the same, and the comment says so. `significance.py` upgrades the comment to `[error]` and the verb exits 1 for a p-value crossing .05 (`p =`, `p <`, `P >` before the number) or a bracketed `CI [lo, hi]` / `(lo to hi)` crossing its null (0, or 1 when OR/HR/RR/"ratio" precedes). Not recognised: intervals without brackets, p-values in tables, other thresholds; no flag means nothing recognised. After the write the copy is re-read: a change counts as applied only when the control reads the new text, and only then is the ledger updated.

Report: `<out>.md` and `.json` (changes, flips, problems, unchanged, unbound controls). Exit 0 / 1 (flip or a missing result key) / 2 usage.

## Writer interaction

`DocxAdapter` places a replacement's insertion after the character before the range, which for a whole-control replacement is outside the control; `docx/writer.py::_adopt_replaced` moves it into the control (first in its content, with the deleted run's properties) so the anchor still wraps the live text. Without it the control holds only deleted text and reads empty.

## Pitfalls

- The default project folder is `<stem>.harness`; `bind-numbers paper.docx` writes `paper.bound.docx`, so `harness/project.py:project_dir` (used by every verb) finds the folder from the project id the copy carries. Pass `--project` to be explicit.
- Tests stub `bindings.bind.run_agent_with_usage`; synthetic data only (`tests/_bindings_helpers.py`).

## Catalogue exclusions and zero (found in the first live run)

A numeric first column of a csv or tsv with several columns and rows (week, dose, id) labels its rows and is not in the catalogue, so "week 4" is never bound to it. A written zero never matches a stored non-zero value (`scale_for`), so "week 0" is never bound to a p-value of .003. Number tokens take a comma only as a thousands separator and never a trailing full stop.
