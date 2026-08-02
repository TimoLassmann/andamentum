**The preflight tax.** Three **fresh processes**, each draining an empty queue.

Fresh is mandatory: ``_stores`` and ``_preflight_done`` are module-level,
per-process caches, so repeating this inside one process would measure zero and
report a comfortable falsehood.

Measured: 1.42 / 1.81 / 2.16 s over 2 requests. This is what a cron- or
launchd-driven drain pays on **every wake-up**, before doing any work at all —
the number to weigh against a polling interval. All three ran back-to-back
against an already-warm Ollama, so none is a true cold start; the coldest
(2.16 s) is the one a wake-up most resembles.
