"""Tests for consistency_scanners — pure functions, no LLM."""

from andamentum.whetstone.consistency_scanners import (
    check_acronym_first_use,
    check_citation_resolution,
    check_figure_order,
    run_all,
)


# ---- check_figure_order ----------------------------------------------------


def test_figure_order_clean():
    text = (
        "See Figure 1 for overview. Figure 2 shows the breakdown. Figure 3 summarises."
    )
    assert check_figure_order(text) == []


def test_figure_order_out_of_order():
    text = "First, see Figure 2 for the overview. Later, Figure 1 gives the background."
    issues = check_figure_order(text)
    assert len(issues) == 1
    assert "Figure 2" in issues[0].title
    assert issues[0].agent_type == "scanner:figure_order"


def test_figure_order_no_figures():
    assert check_figure_order("Plain text with no figures.") == []


def test_figure_order_handles_fig_abbreviation():
    text = "We cite Fig. 2 first, then Fig. 1."
    issues = check_figure_order(text)
    assert len(issues) == 1


def test_figure_order_handles_fig_no_space_after_period():
    # Some journals format "Fig.1" without a space after the period.
    text = "We cite Fig.2 first, then Fig.1."
    issues = check_figure_order(text)
    assert len(issues) == 1


# ---- check_acronym_first_use -----------------------------------------------


def test_acronym_defined_on_first_use():
    text = "We used random forests (RF) for training. The RF classifier outperformed baselines."
    assert check_acronym_first_use(text) == []


def test_acronym_used_before_definition():
    text = "The RF classifier outperformed baselines. We used random forests (RF) for training."
    issues = check_acronym_first_use(text)
    assert any(i.agent_type == "scanner:acronym_first_use" for i in issues)
    assert any("RF" in i.title for i in issues)


def test_acronym_common_skipped():
    text = "We examined DNA extracted from patient samples."
    assert check_acronym_first_use(text) == []


def test_acronym_no_acronyms():
    assert check_acronym_first_use("Plain prose with no acronyms.") == []


# ---- check_citation_resolution ---------------------------------------------


def test_citation_resolution_all_present():
    text = "As shown [1], and also [2].\n\nReferences\n[1] First paper.\n[2] Second paper.\n"
    assert check_citation_resolution(text) == []


def test_citation_resolution_missing_entry():
    text = "As shown [1], and [3].\n\nReferences\n[1] First paper.\n[2] Second paper.\n"
    issues = check_citation_resolution(text)
    assert len(issues) == 1
    assert "[3]" in issues[0].title


def test_citation_resolution_no_references_section():
    text = "As shown [1], and [2]. Discussion follows."
    assert check_citation_resolution(text) == []


def test_citation_resolution_handles_ranges():
    text = "See [1-3] and [5].\n\nReferences\n[1] A.\n[2] B.\n[3] C.\n[5] E.\n"
    assert check_citation_resolution(text) == []


def test_citation_resolution_handles_comma_list():
    text = "See [1, 2, 4].\n\nReferences\n[1] A.\n[2] B.\n[3] C.\n"
    issues = check_citation_resolution(text)
    assert len(issues) == 1
    assert "[4]" in issues[0].title


def test_citation_resolution_malformed_range_skipped():
    # [1-3-5] is malformed; scanner should silently skip it without crashing.
    text = "See [1-3-5].\n\nReferences\n[1] A.\n[2] B.\n[3] C.\n"
    assert check_citation_resolution(text) == []


# ---- run_all ---------------------------------------------------------------


def test_run_all_combines_scanner_output():
    text = "Figure 2 first. Figure 1 after. See [1].\n\nReferences\n"
    issues = run_all(text)
    # Figure order issue + no references entries → citation resolution returns []
    assert any(i.agent_type == "scanner:figure_order" for i in issues)
