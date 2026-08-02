**The checksum gate.** Every registered PDF is re-hashed and the rule exits 1 on
drift. Every measurement rule takes this output as an input, so Snakemake — not
operator discipline — is what stops a run against a corpus that changed under
it.
