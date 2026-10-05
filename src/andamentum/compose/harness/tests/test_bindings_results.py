"""The result catalogue and the ledger file."""

from __future__ import annotations

from pathlib import Path

from andamentum.compose.harness.bindings.ledger import Ledger, NumberBinding
from andamentum.compose.harness.bindings.results import load_catalogue
from andamentum.compose.harness.kernel import ProjectStore


def test_catalogue_splits_file_and_row_label(tmp_path: Path):
    folder = tmp_path / "res"
    (folder / "sub dir").mkdir(parents=True)
    (folder / "sub dir" / "a b.csv").write_text("name,v\nalpha,1.5\n3,2.5\n", "utf-8")
    (folder / "x.json").write_text('{"k": [0.1]}', "utf-8")
    catalogue = load_catalogue(folder)
    by_key = {e.key: e for e in catalogue.entries}
    entry = by_key["sub dir/a b.csv row 2 column 'v'"]
    assert (entry.file, entry.context, entry.value) == ("sub dir/a b.csv", "alpha", 1.5)
    # a numeric first cell labels its row (week, dose, id): it is not a result
    assert "sub dir/a b.csv row 3 column 'name'" not in by_key
    assert by_key["sub dir/a b.csv row 3 column 'v'"].context == ""
    assert by_key["x.json k[0]"].file == "x.json"
    assert catalogue.get("missing") is None


def test_a_numeric_first_column_is_not_bound_but_a_single_number_file_is(
    tmp_path: Path,
):
    """ "week 0" bound to the week column of a live example's intake table."""
    (tmp_path / "weeks.csv").write_text("week,mean\n0,3310\n1,3205\n", "utf-8")
    (tmp_path / "one.csv").write_text("n\n48\n", "utf-8")
    keys = {e.key for e in load_catalogue(tmp_path).entries}
    assert keys == {
        "weeks.csv row 2 column 'mean'",
        "weeks.csv row 3 column 'mean'",
        "one.csv row 2 column 'n'",
    }


def test_catalogue_digest_follows_the_files(tmp_path: Path):
    (tmp_path / "a.csv").write_text("a\n1\n", "utf-8")
    first = load_catalogue(tmp_path).digest
    (tmp_path / "a.csv").write_text("a\n2\n", "utf-8")
    assert load_catalogue(tmp_path).digest != first


def binding(anchor: str, written: str = "5") -> NumberBinding:
    return NumberBinding(
        anchor_id=anchor,
        sentence="It is 5.",
        file="a.csv",
        key="a.csv row 2 column 'a'",
        value=5.0,
        written=written,
        scale=1.0,
    )


def test_ledger_round_trip_and_replace(tmp_path: Path):
    ledger = Ledger(ProjectStore(tmp_path / "p.harness"))
    assert ledger.bindings() == {}
    ledger.record([binding("n1"), binding("n2")])
    ledger.record([binding("n1", written="6")])
    found = ledger.bindings()
    assert set(found) == {"n1", "n2"} and found["n1"].written == "6"
    text = ledger.path.read_text("utf-8")
    assert text.count("- id: n1") == 1
    assert "approved_by: null" in text and "kind: result" in text
