**Standalone conversion cost per PDF**, measured through ``harvest.extract``
with no LLM and no store involved: 3.9 / 10.1 / 13.2 / 16.2 s for the four
papers.

It also converts **one PDF twice** to separate one-time Docling initialisation
(4.70 s) from the marginal warm conversion cost. That split matters: the
checkpoint-savings headline is a cold conversion, so quoting it as a
per-document saving overstates the marginal case by roughly 50%, and the
artefact carries both numbers with a flag saying which is which.
