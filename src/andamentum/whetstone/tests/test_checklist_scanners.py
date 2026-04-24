"""Tests for checklist_scanners — deterministic baseline checks."""

from andamentum.whetstone.checklist_scanners import (
    check_all_figures_referenced,
    check_all_tables_referenced,
    check_authors_listed,
    check_citations_resolve,
    check_coi_statement,
    check_data_availability_statement,
    check_ethics_statement,
    check_figure_numbering_sequential,
    check_funding_statement,
    check_keywords_section,
    check_table_numbering_sequential,
)


# ---- figures & tables ------------------------------------------------------


def test_all_figures_referenced_pass():
    text = "Figure 1: The plot.\n\nBody sees Figure 1 twice: Figure 1 again."
    status, _ = check_all_figures_referenced(text)
    assert status == "pass"


def test_all_figures_referenced_fail():
    text = "Figure 1: The plot.\nFigure 2: Second plot.\n\nBody sees Figure 1 only."
    status, notes = check_all_figures_referenced(text)
    assert status == "fail"
    assert "2" in notes


def test_all_figures_referenced_unclear_no_figs():
    status, _ = check_all_figures_referenced("No figures here.")
    assert status == "unclear"


def test_figure_numbering_sequential_pass():
    text = "Figure 1: A.\nFigure 2: B.\nFigure 3: C."
    status, _ = check_figure_numbering_sequential(text)
    assert status == "pass"


def test_figure_numbering_sequential_fail():
    text = "Figure 1: A.\nFigure 3: C."
    status, _ = check_figure_numbering_sequential(text)
    assert status == "fail"


def test_all_tables_referenced_pass():
    text = "Table 1: Data.\n\nBody sees Table 1 clearly."
    status, _ = check_all_tables_referenced(text)
    assert status == "pass"


def test_all_tables_referenced_fail():
    text = "Table 1: Data.\nTable 2: More data.\n\nBody sees Table 1 only."
    status, notes = check_all_tables_referenced(text)
    assert status == "fail"
    assert "2" in notes


def test_all_tables_referenced_unclear_no_tables():
    status, _ = check_all_tables_referenced("No tables.")
    assert status == "unclear"


def test_table_numbering_sequential_pass():
    text = "Table 1: A.\nTable 2: B."
    status, _ = check_table_numbering_sequential(text)
    assert status == "pass"


def test_table_numbering_sequential_fail():
    text = "Table 1: A.\nTable 3: C."
    status, _ = check_table_numbering_sequential(text)
    assert status == "fail"


# ---- citations -------------------------------------------------------------


def test_citations_resolve_pass():
    text = "See [1] and [2].\n\nReferences\n[1] A.\n[2] B.\n"
    status, _ = check_citations_resolve(text)
    assert status == "pass"


def test_citations_resolve_fail():
    text = "See [1] and [3].\n\nReferences\n[1] A.\n[2] B.\n"
    status, notes = check_citations_resolve(text)
    assert status == "fail"
    assert "3" in notes


def test_citations_resolve_no_refs_section():
    status, _ = check_citations_resolve("See [1] and [2].")
    assert status == "unclear"


def test_citations_resolve_handles_spaced_range():
    # Spaced-dash ranges like [1 - 3] must be parsed, not silently dropped.
    text = "See [1 - 3].\n\nReferences\n[1] A.\n[2] B.\n[3] C.\n"
    status, _ = check_citations_resolve(text)
    assert status == "pass"


# ---- required statements ---------------------------------------------------


def test_coi_present():
    status, _ = check_coi_statement("We declare no conflict of interest.")
    assert status == "pass"


def test_coi_absent():
    status, _ = check_coi_statement("No declarations here.")
    assert status == "fail"


def test_coi_competing_interests():
    status, _ = check_coi_statement("Competing interests: none.")
    assert status == "pass"


def test_data_availability_present():
    status, _ = check_data_availability_statement("Data availability: on request.")
    assert status == "pass"


def test_data_availability_absent():
    status, _ = check_data_availability_statement("No mention of data.")
    assert status == "fail"


def test_ethics_unclear_no_subjects():
    status, _ = check_ethics_statement("A theoretical analysis of algorithms.")
    assert status == "unclear"


def test_ethics_pass():
    text = "We recruited 50 participants. IRB approval was obtained from the institutional review board."
    status, _ = check_ethics_statement(text)
    assert status == "pass"


def test_ethics_fail():
    text = "We recruited 50 participants. They completed the survey."
    status, _ = check_ethics_statement(text)
    assert status == "fail"


def test_funding_present():
    status, _ = check_funding_statement(
        "This work was supported by NIH grant R01-12345."
    )
    assert status == "pass"


def test_funding_absent():
    status, _ = check_funding_statement("Just body text.")
    assert status == "fail"


# ---- hygiene ---------------------------------------------------------------


def test_keywords_present():
    status, _ = check_keywords_section(
        "Keywords: reproducibility, methodology.\n\nAbstract: ..."
    )
    assert status == "pass"


def test_keywords_absent():
    status, _ = check_keywords_section("Just a title.\n\nAbstract: ...")
    assert status == "fail"


def test_authors_pass():
    text = "Jane Doe\nDepartment of Computer Science, University of Somewhere\n\nAbstract: ..."
    status, _ = check_authors_listed(text)
    assert status == "pass"


def test_authors_unclear():
    status, _ = check_authors_listed("No affiliation keywords here.")
    assert status == "unclear"
