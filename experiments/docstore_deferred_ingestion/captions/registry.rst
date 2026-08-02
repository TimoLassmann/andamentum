**The input registry**: for each of the four arXiv papers, the **versioned**
PDF URL, the sha256, the ETag and the byte count. Re-fetching compares hashes —
a match skips the download, a mismatch raises ``RegistryDriftError`` rather
than silently measuring a different corpus.

The arXiv *API* is blocked on this host and is never used; the PDFs are fetched
by direct versioned URL, strictly sequentially with a polite delay regardless
of ``--cores``. ``data/pdfs/`` itself is deliberately not committed — it is
large and exactly re-derivable from this file.
