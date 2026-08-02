**The CLI surface.** Until this rule existed, exactly **one** of the six
``andamentum-docstore`` subcommands had ever run against a real database.

Every subcommand is now exercised through the venv console script — ``ingest
--defer``, ``ingest-source``, ``status``, ``retry-failed``, ``process-pending
--max-docs``, ``process-pending --max-seconds`` — asserting exit codes, stdout
shape, and that what ``status`` *prints* reconciles with what sqlite *holds*.
A status command that reports from a different source of truth than the drain
is a class of bug no library-level test can see.
