"""Bundle validation."""

from andamentum.compose.bundle import Bundle


def _issues(b):
    return [(i.severity, i.message) for i in b.validate()]


def test_clean_bundle_has_only_info_for_empty(tmp_path):
    b = Bundle.create(tmp_path / "d", title="T")
    b.add_section(title="A", body="Fine.")
    assert b.validate() == []


def test_duplicate_titles_error(tmp_path):
    b = Bundle.create(tmp_path / "d", title="T")
    b.add_section(title="A", body="x")
    b.add_section(title="A", body="y")
    assert any(s == "error" and "'A'" in m for s, m in _issues(b))


def test_heading_in_body_error(tmp_path):
    b = Bundle.create(tmp_path / "d", title="T")
    b.add_section(title="A", body="text\n\n## Sub\n\nmore")
    errors = [m for s, m in _issues(b) if s == "error"]
    assert len(errors) == 1 and "line 3" in errors[0]


def test_missing_image_error_and_present_image_ok(tmp_path):
    b = Bundle.create(tmp_path / "d", title="T")
    b.add_section(title="A", body="![c](figures/missing.png){#fig:m}")
    assert any(s == "error" and "figures/missing.png" in m for s, m in _issues(b))
    (tmp_path / "d" / "figures" / "ok.png").write_bytes(b"x")
    b.write_section("s01", body="![c](figures/ok.png)", expected_revision=1)
    assert b.validate() == []


def test_citations_and_markers_warn(tmp_path):
    b = Bundle.create(tmp_path / "d", title="T")
    b.add_reference(key="known", text="Known.")
    b.add_section(
        title="A",
        body="See [@known] and [@missing]. Also [verify] and [citation needed].",
    )
    warnings = [m for s, m in _issues(b) if s == "warning"]
    assert len(warnings) == 3
    assert any("[@missing]" in m for m in warnings)
    assert not any("[@known]" in m for m in warnings)


def test_empty_body_is_info(tmp_path):
    b = Bundle.create(tmp_path / "d", title="T")
    b.add_section(title="A")
    assert [s for s, _ in _issues(b)] == ["info"]
