"""PubMed XML parsing and the efetch call, on a synthetic article."""

from __future__ import annotations

import httpx
import pytest

from andamentum.references import fetch_pubmed, parse_pubmed_xml

ARTICLE = """<?xml version="1.0"?>
<PubmedArticleSet>
<PubmedArticle>
  <MedlineCitation>
    <PMID Version="1">9000001</PMID>
    <Article>
      <Journal>
        <JournalIssue><Volume>12</Volume><Issue>3</Issue>
          <PubDate><Year>2021</Year></PubDate></JournalIssue>
        <Title>Journal of Invented Salt Studies</Title>
      </Journal>
      <ArticleTitle>Effects of <i>Salinus apollo</i> on daily salt logging</ArticleTitle>
      <Pagination><MedlinePgn>101-109</MedlinePgn></Pagination>
      <Abstract>
        <AbstractText Label="BACKGROUND">Salt logging is <i>invented</i> here.</AbstractText>
        <AbstractText Label="RESULTS">Logging improved.</AbstractText>
      </Abstract>
      <AuthorList>
        <Author><LastName>Doe</LastName><Initials>J</Initials></Author>
        <Author><LastName>Roe</LastName><Initials>RA</Initials></Author>
        <Author><CollectiveName>The Invented Group</CollectiveName></Author>
      </AuthorList>
      <PublicationTypeList>
        <PublicationType>Randomized Controlled Trial</PublicationType>
      </PublicationTypeList>
    </Article>
    <MeshHeadingList>
      <MeshHeading><DescriptorName>Sodium Chloride</DescriptorName></MeshHeading>
      <MeshHeading><DescriptorName>Logging</DescriptorName></MeshHeading>
    </MeshHeadingList>
  </MedlineCitation>
  <PubmedData>
    <ArticleIdList>
      <ArticleId IdType="pubmed">9000001</ArticleId>
      <ArticleId IdType="doi">10.5555/saltlog.001</ArticleId>
      <ArticleId IdType="pmc">PMC7000001</ArticleId>
    </ArticleIdList>
  </PubmedData>
</PubmedArticle>
<PubmedArticle>
  <MedlineCitation>
    <PMID>9000002</PMID>
    <Article><ArticleTitle>A bare record</ArticleTitle></Article>
  </MedlineCitation>
</PubmedArticle>
</PubmedArticleSet>
"""


def test_parse_full_record() -> None:
    first = parse_pubmed_xml(ARTICLE)[0]
    assert first.pmid == "9000001"
    assert first.title == "Effects of Salinus apollo on daily salt logging"
    assert first.abstract == (
        "BACKGROUND: Salt logging is invented here. RESULTS: Logging improved."
    )
    assert first.abstract_sections[1] == ("RESULTS", "Logging improved.")
    assert first.authors == ["Doe J", "Roe RA"]
    assert first.journal == "Journal of Invented Salt Studies"
    assert (first.year, first.volume, first.issue, first.pages) == (
        "2021",
        "12",
        "3",
        "101-109",
    )
    assert first.doi == "10.5555/saltlog.001"
    assert first.pmcid == "PMC7000001"
    assert first.mesh_terms == ["Sodium Chloride", "Logging"]
    assert first.publication_types == ["Randomized Controlled Trial"]


def test_parse_sparse_record_has_empty_fields() -> None:
    second = parse_pubmed_xml(ARTICLE)[1]
    assert second.pmid == "9000002" and second.title == "A bare record"
    assert second.abstract == "" and second.authors == [] and second.doi == ""


def test_parse_rejects_non_xml() -> None:
    with pytest.raises(ValueError):
        parse_pubmed_xml("<not closed")


async def test_fetch_sends_ids_and_key_and_parses() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, text=ARTICLE)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        records = await fetch_pubmed(["9000001", "9000002"], client=client, api_key="k")
    assert [r.pmid for r in records] == ["9000001", "9000002"]
    params = seen[0].url.params
    assert params["id"] == "9000001,9000002" and params["retmode"] == "xml"
    assert params["api_key"] == "k"


async def test_fetch_raises_on_http_error() -> None:
    transport = httpx.MockTransport(lambda request: httpx.Response(500))
    async with httpx.AsyncClient(transport=transport) as client:
        with pytest.raises(httpx.HTTPStatusError):
            await fetch_pubmed(["1"], client=client)


async def test_fetch_without_ids_makes_no_request() -> None:
    assert await fetch_pubmed([]) == []
