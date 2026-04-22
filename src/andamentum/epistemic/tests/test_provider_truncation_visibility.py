"""Tests asserting that provider truncation is visible — not silently dropped.

Covers:
- ChEMBL: _get_activities returns all API results (no [:5] cut)
- PubMed: structured_data["authors"] has full list; content shows "(et al, N total)"
- ClinicalTrials: conditions content shows "(and N more)" suffix when truncated
"""

from __future__ import annotations

from typing import Any

import httpx

# ──────────────────────────────────────────────────────────────────────────────
# Mock transport helpers (same pattern as test_providers.py)
# ──────────────────────────────────────────────────────────────────────────────

_RealAsyncClient = httpx.AsyncClient


class _UrlDispatchTransport(httpx.AsyncBaseTransport):
    """Dispatches responses by URL path."""

    def __init__(self, responses: dict[str, Any], status_code: int = 200):
        self.requests: list[httpx.Request] = []
        self._responses = responses
        self._status_code = status_code

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        url_path = request.url.path
        body = self._responses.get(url_path)
        if body is None:
            return httpx.Response(404, request=request)
        if isinstance(body, str):
            return httpx.Response(
                self._status_code,
                text=body,
                request=request,
                headers={"content-type": "text/xml"},
            )
        return httpx.Response(self._status_code, json=body, request=request)


def _patched_client(transport: httpx.AsyncBaseTransport):
    class _Client:
        def __init__(self, **kwargs: Any):
            self._c = _RealAsyncClient(transport=transport, timeout=30.0)

        async def __aenter__(self):
            return self._c

        async def __aexit__(self, *args: Any):
            await self._c.aclose()

    return _Client


# ──────────────────────────────────────────────────────────────────────────────
# ChEMBL: all activities returned, no [:5] cap
# ──────────────────────────────────────────────────────────────────────────────


class TestChEMBLReturnsAllActivities:
    """_get_activities must return all activities from the API (up to limit=25),
    not silently truncate to 5."""

    async def test_returns_all_activities(self, monkeypatch):
        """Stub returns 15 activities; structured_data['activities'] must have all 15."""
        # Build 15 fake activity records
        activities_data = [
            {
                "pchembl_value": str(9.0 - i * 0.1),
                "standard_type": "IC50",
                "standard_value": 10 + i,
                "standard_units": "nM",
                "target_pref_name": f"Target {i}",
                "target_chembl_id": f"CHEMBL{1000 + i}",
            }
            for i in range(15)
        ]

        transport = _UrlDispatchTransport(
            {
                "/chembl/api/data/molecule/search.json": {
                    "molecules": [
                        {
                            "molecule_chembl_id": "CHEMBL941",
                            "pref_name": "Aspirin",
                            "max_phase": 4,
                            "molecule_type": "Small molecule",
                            "first_approval": 1899,
                        }
                    ]
                },
                "/chembl/api/data/mechanism.json": {"mechanisms": []},
                "/chembl/api/data/activity.json": {"activities": activities_data},
                "/chembl/api/data/molecule/CHEMBL941.json": {
                    "molecule_structures": {
                        "canonical_smiles": "CC(=O)Oc1ccccc1C(=O)O"
                    },
                    "molecule_properties": {},
                },
            }
        )

        monkeypatch.setattr("httpx.AsyncClient", _patched_client(transport))

        from ..providers.chembl import ChEMBLProvider

        provider = ChEMBLProvider(max_results=5)
        results = await provider.gather("aspirin")

        assert len(results) == 1
        r = results[0]
        # All 15 activities must be in structured_data
        assert len(r.structured_data["activities"]) == 15, (
            f"Expected 15 activities, got {len(r.structured_data['activities'])}"
        )
        # Content shows "top 3 of 15"
        assert "showing top 3 of 15" in r.content

    async def test_no_suffix_when_three_or_fewer_activities(self, monkeypatch):
        """When 3 or fewer activities, no count suffix in content."""
        activities_data = [
            {
                "pchembl_value": "8.5",
                "standard_type": "IC50",
                "standard_value": 3,
                "standard_units": "nM",
                "target_pref_name": "Target A",
                "target_chembl_id": "CHEMBL1001",
            }
        ]

        transport = _UrlDispatchTransport(
            {
                "/chembl/api/data/molecule/search.json": {
                    "molecules": [
                        {
                            "molecule_chembl_id": "CHEMBL941",
                            "pref_name": "Aspirin",
                            "max_phase": 4,
                            "molecule_type": "Small molecule",
                            "first_approval": 1899,
                        }
                    ]
                },
                "/chembl/api/data/mechanism.json": {"mechanisms": []},
                "/chembl/api/data/activity.json": {"activities": activities_data},
                "/chembl/api/data/molecule/CHEMBL941.json": {
                    "molecule_structures": None,
                    "molecule_properties": {},
                },
            }
        )

        monkeypatch.setattr("httpx.AsyncClient", _patched_client(transport))

        from ..providers.chembl import ChEMBLProvider

        provider = ChEMBLProvider(max_results=5)
        results = await provider.gather("aspirin")

        assert len(results) == 1
        assert "showing top 3 of" not in results[0].content


# ──────────────────────────────────────────────────────────────────────────────
# PubMed: full author list in structured_data; content shows "et al, N authors total"
# ──────────────────────────────────────────────────────────────────────────────

_PUBMED_XML_MANY_AUTHORS = """<?xml version="1.0"?>
<PubmedArticleSet>
  <PubmedArticle>
    <MedlineCitation>
      <PMID>99999999</PMID>
      <Article>
        <ArticleTitle>A study with twelve authors</ArticleTitle>
        <AuthorList>
          <Author><LastName>Alpha</LastName><Initials>A</Initials></Author>
          <Author><LastName>Beta</LastName><Initials>B</Initials></Author>
          <Author><LastName>Gamma</LastName><Initials>C</Initials></Author>
          <Author><LastName>Delta</LastName><Initials>D</Initials></Author>
          <Author><LastName>Epsilon</LastName><Initials>E</Initials></Author>
          <Author><LastName>Zeta</LastName><Initials>F</Initials></Author>
          <Author><LastName>Eta</LastName><Initials>G</Initials></Author>
          <Author><LastName>Theta</LastName><Initials>H</Initials></Author>
          <Author><LastName>Iota</LastName><Initials>I</Initials></Author>
          <Author><LastName>Kappa</LastName><Initials>J</Initials></Author>
          <Author><LastName>Lambda</LastName><Initials>K</Initials></Author>
          <Author><LastName>Mu</LastName><Initials>L</Initials></Author>
        </AuthorList>
        <Abstract>
          <AbstractText>Abstract text here.</AbstractText>
        </Abstract>
      </Article>
    </MedlineCitation>
    <PubmedData>
      <ArticleIdList>
        <ArticleId IdType="doi">10.1234/twelve.authors</ArticleId>
      </ArticleIdList>
    </PubmedData>
  </PubmedArticle>
</PubmedArticleSet>"""


class PubMedMockTransport(httpx.AsyncBaseTransport):
    """Returns JSON for esearch, XML for efetch."""

    def __init__(self, xml: str = _PUBMED_XML_MANY_AUTHORS):
        self._xml = xml
        self.requests: list[httpx.Request] = []

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if "esearch" in request.url.path:
            return httpx.Response(
                200,
                json={"esearchresult": {"idlist": ["99999999"]}},
                request=request,
            )
        elif "efetch" in request.url.path:
            return httpx.Response(
                200,
                text=self._xml,
                request=request,
                headers={"content-type": "text/xml"},
            )
        return httpx.Response(404, request=request)


class TestPubMedAuthorsFullInStructuredData:
    """structured_data['authors'] must have all authors; content must show
    '(et al, N authors total)' when more than 5."""

    async def test_twelve_authors_full_in_structured_data(self, monkeypatch):
        """12 authors → structured_data has all 12, content has 'et al, 12 authors total'."""
        transport = PubMedMockTransport()
        monkeypatch.setattr("httpx.AsyncClient", _patched_client(transport))

        from ..providers.pubmed import PubMedProvider

        provider = PubMedProvider(max_results=5)
        results = await provider.gather("twelve authors study")

        assert len(results) == 1
        r = results[0]

        # Full author list in structured_data
        assert len(r.structured_data["authors"]) == 12, (
            f"Expected 12 authors, got {len(r.structured_data['authors'])}"
        )

        # Content shows et al notice
        assert "et al, 12 authors total" in r.content, (
            f"Expected 'et al, 12 authors total' in content. Got:\n{r.content}"
        )

        # First 5 authors present
        assert "Alpha A" in r.content
        assert "Epsilon E" in r.content

    async def test_five_authors_no_et_al(self, monkeypatch):
        """5 authors → no 'et al' suffix."""
        xml = """<?xml version="1.0"?>
<PubmedArticleSet>
  <PubmedArticle>
    <MedlineCitation>
      <PMID>11111111</PMID>
      <Article>
        <ArticleTitle>Five author paper</ArticleTitle>
        <AuthorList>
          <Author><LastName>One</LastName><Initials>A</Initials></Author>
          <Author><LastName>Two</LastName><Initials>B</Initials></Author>
          <Author><LastName>Three</LastName><Initials>C</Initials></Author>
          <Author><LastName>Four</LastName><Initials>D</Initials></Author>
          <Author><LastName>Five</LastName><Initials>E</Initials></Author>
        </AuthorList>
      </Article>
    </MedlineCitation>
  </PubmedArticle>
</PubmedArticleSet>"""
        transport = PubMedMockTransport(xml=xml)
        monkeypatch.setattr("httpx.AsyncClient", _patched_client(transport))

        from ..providers.pubmed import PubMedProvider

        provider = PubMedProvider(max_results=5)
        results = await provider.gather("five authors")

        assert len(results) == 1
        r = results[0]
        assert "et al" not in r.content
        assert len(r.structured_data["authors"]) == 5


# ──────────────────────────────────────────────────────────────────────────────
# ClinicalTrials: conditions count suffix when truncated
# ──────────────────────────────────────────────────────────────────────────────


class TestClinicalTrialsConditionsCountSuffix:
    """Content must show '(and N more)' when conditions list exceeds 3."""

    async def test_seven_conditions_shows_and_4_more(self, monkeypatch):
        """7 conditions → content includes '(and 4 more)'."""
        conditions = [f"Condition {i}" for i in range(7)]
        study = {
            "protocolSection": {
                "identificationModule": {
                    "nctId": "NCT12345678",
                    "officialTitle": "Test Study",
                },
                "statusModule": {"overallStatus": "RECRUITING"},
                "designModule": {
                    "phases": ["PHASE2"],
                    "enrollmentInfo": {"count": 100, "type": "ESTIMATED"},
                },
                "descriptionModule": {"briefSummary": "A test study."},
                "conditionsModule": {"conditions": conditions},
                "armsInterventionsModule": {"interventions": []},
                "outcomesModule": {
                    "primaryOutcomes": [],
                    "secondaryOutcomes": [],
                    "otherOutcomes": [],
                },
                "sponsorCollaboratorsModule": {"leadSponsor": {"name": "Test Sponsor"}},
            },
            "hasResults": False,
        }

        transport = _UrlDispatchTransport({"/api/v2/studies": {"studies": [study]}})
        monkeypatch.setattr("httpx.AsyncClient", _patched_client(transport))

        from ..providers.clinicaltrials import ClinicalTrialsProvider

        provider = ClinicalTrialsProvider(max_results=5)
        results = await provider.gather("test condition")

        assert len(results) == 1
        r = results[0]

        # structured_data has ALL 7 conditions
        assert len(r.structured_data["conditions"]) == 7

        # Content shows top 3 + count suffix
        assert "(and 4 more)" in r.content, (
            f"Expected '(and 4 more)' in content. Got:\n{r.content}"
        )
        assert "Condition 0" in r.content
        assert "Condition 2" in r.content
        # Condition 3 onwards not individually listed in content
        assert "Condition 3," not in r.content

    async def test_three_conditions_no_suffix(self, monkeypatch):
        """3 conditions → no count suffix."""
        study = {
            "protocolSection": {
                "identificationModule": {
                    "nctId": "NCT11111111",
                    "officialTitle": "Short Study",
                },
                "statusModule": {"overallStatus": "COMPLETED"},
                "designModule": {
                    "phases": ["PHASE3"],
                    "enrollmentInfo": {"count": 50, "type": "ACTUAL"},
                },
                "descriptionModule": {"briefSummary": ""},
                "conditionsModule": {"conditions": ["Cond A", "Cond B", "Cond C"]},
                "armsInterventionsModule": {"interventions": []},
                "outcomesModule": {
                    "primaryOutcomes": [],
                    "secondaryOutcomes": [],
                    "otherOutcomes": [],
                },
                "sponsorCollaboratorsModule": {"leadSponsor": {"name": "Sponsor"}},
            },
            "hasResults": True,
        }

        transport = _UrlDispatchTransport({"/api/v2/studies": {"studies": [study]}})
        monkeypatch.setattr("httpx.AsyncClient", _patched_client(transport))

        from ..providers.clinicaltrials import ClinicalTrialsProvider

        provider = ClinicalTrialsProvider(max_results=5)
        results = await provider.gather("short study")

        assert len(results) == 1
        assert "(and" not in results[0].content
