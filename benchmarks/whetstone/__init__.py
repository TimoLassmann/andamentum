"""Whetstone evaluation harness.

Compares whetstone's chunked review pipeline (Arm A) against a single
whole-document read (Arm B) by the SAME model, to measure whether the
chunked architecture misses critical, cross-section issues.

See docs/plans/2026-05-21-whetstone-benchmark-prd.md for the full design.
"""
