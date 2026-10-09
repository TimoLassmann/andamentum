# The andamentum interface

Every tool in andamentum has the same shape, on the command line and in Python, so that a person or an agent who has used one tool can use the next without reading its source.

## Finding a tool

```bash
andamentum list            # every tool: name, what it does, whether it needs a model
andamentum list --json     # the same, for programs and agents
andamentum <tool> ...      # runs that tool; identical to andamentum-<tool> ...
```

`andamentum` only lists and forwards. Each tool keeps its own small command, `andamentum-<tool>`, with its own `--help`.

In Python, `andamentum.TOOLS` holds the same list: for each tool its name, summary, command, main function and whether it needs a model.

## The command line

1. **Input.** The thing the tool works on is a positional argument: a file path, a URL, a text, a question. Where the input is text, `-` reads it from stdin.
2. **Output.**
   - A tool that produces one result prints it to stdout. `-o/--out FILE` writes it to a file instead.
   - A tool that produces several files needs `--out DIR` and writes every file into that folder. There is no other location.
   - No tool writes into the current folder, the home folder or anywhere else on its own.
3. **Machine-readable output.** `--json` prints the result as JSON. A tool whose only output is JSON (or a document such as a PDF) does not need the flag.
4. **Models.**
   - A tool that calls a language model needs `--model MODEL` on the command line. There is no environment variable and no default.
   - A second model has a name that says its job: `--embedding-model`, `--reranker-model`, `--decision-model`, `--figure-model`. None has a default.
5. **Exit codes**, the same in every tool:

   | Code | Meaning |
   |---|---|
   | 0 | success |
   | 1 | the tool ran and the result is negative: problems found, goal not met, a check failed; any output is still written |
   | 2 | usage error: a wrong or missing argument |
   | 3 | the input could not be read: missing file, unreachable URL, unsupported format |
   | 4 | the run failed: model or network error, or a crash |

6. **Errors** go to stderr as one line that starts with `error:` and says what to do.
7. **Help.** `--help` explains every flag in plain words and ends with one to three examples that a user can copy, the most common job first. `--version` prints the andamentum version.
8. **Subcommands** only when a tool really does several different jobs. A tool with one job has no subcommand.
9. **Plain words.** Flags and help text use words a user already knows (file, folder, output, model, references). No internal names.
10. **Credentials** (API keys) come from the environment, as each provider documents. They are never flags and never configuration.
11. **Presets, not knobs.** The command line offers a few named presets and as few flags as possible; the Python functions keep every setting.
    - Tuning numbers (rounds, limits, sizes, concurrency) are not flags. Choose a good default, or give one named preset such as `--depth quick|standard|deep`.
    - A mode is one flag with named values (`--research auto|web|scholarly|both`), never a switch plus a second flag for the kind.
    - One idea has one flag name in every tool: `--offline` means "make no network call besides the model".
    - A flag earns its place when a user needs it for a real job. Something only a benchmark or an experiment changes stays a Python argument.

## Python

1. **One main function per tool**, exported from the module, with a name that says what it does: `extract`, `chunk`, `research`, `review_document`, `judge_score`.
2. **The first argument is the input; everything else is keyword-only.** A function that calls a model takes `model=` as a required keyword.
3. **Async when the function does input/output or calls a model; plain functions when it is pure computation.** There is one code path, not an async and a sync copy.
4. **Typed results** (pydantic models or dataclasses), never bare dicts.
5. **Typed errors.** Each module has one base error class; specific errors derive from it.
6. **A small `__all__`:** the main function(s), their result types and the module's errors. Internals are not exported.
7. **No environment variables, no default locations, no files written that the caller did not name.**
8. **No version suffixes** (`_v2`, `_v3`) in names and no leftover folders from older versions.

## Checks

`src/andamentum/tests/test_interface.py` holds every tool to the command-line rules it can check mechanically: each console script is registered in `andamentum.TOOLS` and the reverse; each command has `--help` and `--version`; output flags are `-o/--out`; a model tool requires `--model`; every command and subcommand shows an example in its `--help`, and every example there and in [guide.md](./guide.md) is accepted by the command; no command module reads configuration from the environment.
