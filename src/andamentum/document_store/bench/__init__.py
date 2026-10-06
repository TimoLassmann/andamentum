"""Retrieval benchmark for the document store.

Measures whether a store ranks the supporting passages of public multi-hop
questions (HotpotQA, MuSiQue) near the top of a pooled corpus. Retrieval only:
no answer generation, no judge, no LLM call. Run it with
``python -m andamentum.document_store.bench --help``; see ``cli.py``.
"""
