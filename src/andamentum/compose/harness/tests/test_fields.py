"""Field instruction classification (synthetic instructions only)."""

from __future__ import annotations

import base64
import json

import pytest

from andamentum.compose.harness.docx.fields import classify_field

CSL = {
    "citationID": "T1",
    "citationItems": [
        {"id": "x", "itemData": {"DOI": "10.9999/t.1", "title": "Invented"}}
    ],
}


def test_zotero_item_parses_csl_json():
    kind, manager, data = classify_field(
        f"ADDIN ZOTERO_ITEM CSL_CITATION {json.dumps(CSL)} "
    )
    assert (kind, manager) == ("citation", "zotero")
    assert data["citationID"] == "T1"
    assert "parse_error" not in data


def test_zotero_item_malformed_json_is_reported_not_dropped():
    kind, manager, data = classify_field(
        'ADDIN ZOTERO_ITEM CSL_CITATION {"citationItems": ['
    )
    assert (kind, manager) == ("citation", "zotero")
    assert "invalid JSON" in str(data["parse_error"])
    assert str(data["raw"]).startswith("ADDIN ZOTERO_ITEM")


def test_json_that_is_not_an_object_is_a_parse_error():
    _, _, data = classify_field("ADDIN ZOTERO_ITEM CSL_CITATION [1, 2]")
    assert data["parse_error"] == "JSON payload is not an object"


def test_zotero_bibliography():
    kind, manager, data = classify_field(
        'ADDIN ZOTERO_BIBL {"uncited":[],"omitted":[],"custom":[]} CSL_BIBLIOGRAPHY'
    )
    assert (kind, manager) == ("bibliography", "zotero")
    assert data == {"uncited": [], "omitted": [], "custom": []}


def test_mendeley_citation_and_bibliography():
    kind, manager, data = classify_field(f"ADDIN CSL_CITATION {json.dumps(CSL)}")
    assert (kind, manager) == ("citation", "mendeley")
    assert data["citationID"] == "T1"
    kind, manager, _ = classify_field("ADDIN Mendeley Bibliography CSL_BIBLIOGRAPHY")
    assert (kind, manager) == ("bibliography", "mendeley")


ENDNOTE = (
    "<EndNote><Cite><Author>Roe</Author><Year>2019</Year><RecNum>7</RecNum>"
    "<DisplayText>(Roe, 2019)</DisplayText><record><rec-number>7</rec-number>"
    "<titles><title><style>Invented <style>diaries</style></style></title></titles>"
    "<contributors><authors><author>Roe, R.</author></authors></contributors>"
    "<dates><year>2019</year></dates>"
    "<electronic-resource-num>10.9999/t.2</electronic-resource-num></record></Cite></EndNote>"
)


def test_endnote_cite_reads_the_record():
    kind, manager, data = classify_field(f" ADDIN EN.CITE {ENDNOTE}")
    assert (kind, manager) == ("citation", "endnote")
    cites = data["cites"]
    assert isinstance(cites, list) and len(cites) == 1
    cite = cites[0]
    assert cite["author"] == "Roe" and cite["doi"] == "10.9999/t.2"
    assert cite["title"] == "Invented diaries"
    assert cite["authors"] == ["Roe, R."]


def test_endnote_cite_with_broken_xml_is_reported():
    _, manager, data = classify_field("ADDIN EN.CITE <EndNote><Cite>")
    assert manager == "endnote"
    assert "invalid EndNote XML" in str(data["parse_error"])


def test_endnote_cite_without_payload_is_reported():
    _, _, data = classify_field("ADDIN EN.CITE")
    assert "no <EndNote> payload" in str(data["parse_error"])


def test_endnote_cite_data_is_not_a_second_citation():
    encoded = base64.b64encode(ENDNOTE.encode()).decode()
    kind, manager, data = classify_field(f"ADDIN EN.CITE.DATA {encoded}")
    assert (kind, manager) == ("other", "endnote")
    assert data["encoded"] is True
    kind, manager, data = classify_field("ADDIN EN.CITE.DATA")
    assert (kind, manager, data) == ("other", "endnote", {"encoded": False})


def test_endnote_reference_list():
    assert classify_field("ADDIN EN.REFLIST")[:2] == ("bibliography", "endnote")


@pytest.mark.parametrize("name", ["REF", "PAGEREF", "NOTEREF"])
def test_word_cross_references(name):
    kind, manager, data = classify_field(f" {name} _Ref123 \\h \\* MERGEFORMAT ")
    assert (kind, manager) == ("crossref", "word")
    assert data["field"] == name and data["target"] == "_Ref123"
    assert data["switches"] == ["\\h", "\\*"]


def test_quoted_reference_target():
    _, _, data = classify_field('REF "my mark" \\h')
    assert data["target"] == "my mark"


def test_seq_names_its_series():
    kind, manager, data = classify_field("SEQ Figure \\* ARABIC")
    assert (kind, manager) == ("crossref", "word")
    assert data["field"] == "SEQ" and data["target"] == "Figure"


def test_word_native_citation_and_bibliography():
    assert classify_field("CITATION Doe2020 \\l 1033")[:2] == ("citation", "word")
    assert classify_field("BIBLIOGRAPHY \\l 1033")[:2] == ("bibliography", "word")


def test_other_fields():
    kind, manager, data = classify_field("PAGE")
    assert (kind, manager, data["field"]) == ("other", "word", "PAGE")
    kind, manager, data = classify_field("ADDIN SOMETHING.ELSE payload")
    assert (kind, manager) == ("other", "none")
    assert classify_field("   ")[:2] == ("other", "none")
