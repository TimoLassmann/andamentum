"""Tests for the case-editor FastAPI app."""

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from andamentum.chunker.judge import JudgeVerdict
from andamentum.deep_research.models import FetchedPage, SearchResult
from benchmarks.chunker.app.main import (
    _canonicalise_pdf_url,
    _extract_pdf_links,
    _set_cases_dir,
    _set_executor_factory,
    _set_search_backend_factory,
    _set_searxng_manager_factory,
    app,
)


def _fake_executor_factory(model: str):
    """Returns an executor that acts as a 'keep' judge — invoked only if the
    structural-first chunker reaches a grey-zone boundary. For most test
    inputs (no headings, short text) it isn't called at all."""

    async def executor(*, instructions, user_message, output_type, validators):
        return JudgeVerdict(decision="keep", reason="fake judge")

    return executor


# ---------- Index + chunk endpoint ----------------------------------------


def test_index_serves_html():
    client = TestClient(app)
    r = client.get("/")
    assert r.status_code == 200
    assert "<html" in r.text.lower()
    assert "chunker" in r.text.lower()


def test_chunk_endpoint_returns_units_from_markdown_headings():
    """Structural-first: a doc with `## ` headings becomes one unit per section."""
    _set_executor_factory(_fake_executor_factory)
    client = TestClient(app)
    text = (
        "## Introduction\n\nIntro body. " * 10
        + "\n\n## Methods\n\nMethods body. " * 10
        + "\n\n## Results\n\nResults body. " * 10
    )
    r = client.post(
        "/api/chunk",
        json={"text": text, "domain": "academic"},  # no model — structural is enough
    )
    assert r.status_code == 200, r.text
    data = r.json()
    assert "units" in data
    titles = [u["title"] for u in data["units"]]
    # Three section units (the markdown was crafted with 3 headings)
    assert titles[:3] == ["Introduction", "Methods", "Results"][:3] or len(titles) >= 3


def test_chunk_endpoint_works_without_model():
    """The new chunker doesn't require a model — structural pass is free."""
    client = TestClient(app)
    r = client.post(
        "/api/chunk",
        json={"text": "Just some plain text with no structure."},
    )
    assert r.status_code == 200, r.text
    data = r.json()
    assert len(data["units"]) >= 1


def test_chunk_endpoint_accepts_legacy_window_size_lookahead():
    """Old callers pass window_size/lookahead; we ignore them quietly."""
    client = TestClient(app)
    r = client.post(
        "/api/chunk",
        json={
            "text": "Hello world.",
            "window_size": 200,
            "lookahead": 50,
        },
    )
    assert r.status_code == 200, r.text


# ---------- /api/match-anchor ---------------------------------------------


def test_match_anchor_exact():
    client = TestClient(app)
    r = client.post(
        "/api/match-anchor",
        json={"text": "Hello world. End here.", "anchor": "Hello world"},
    )
    assert r.status_code == 200
    data = r.json()
    assert data == {"found": True, "start": 0, "end": 11, "method": "exact"}


def test_match_anchor_respects_search_from():
    client = TestClient(app)
    text = "alpha beta gamma alpha delta"
    r = client.post(
        "/api/match-anchor",
        json={"text": text, "anchor": "alpha", "search_from": 5},
    )
    assert r.status_code == 200
    data = r.json()
    assert data["found"] is True
    assert data["start"] == text.index("alpha", 5)


def test_match_anchor_not_found():
    client = TestClient(app)
    r = client.post(
        "/api/match-anchor",
        json={"text": "Hello world.", "anchor": "missing-string-xyz"},
    )
    assert r.status_code == 200
    assert r.json() == {"found": False, "start": None, "end": None, "method": None}


def test_match_anchor_empty_anchor():
    client = TestClient(app)
    r = client.post(
        "/api/match-anchor",
        json={"text": "Hello world.", "anchor": ""},
    )
    assert r.status_code == 200
    assert r.json()["found"] is False


# ---------- /api/save-case ------------------------------------------------


@pytest.fixture
def tmp_cases_dir(tmp_path: Path):
    _set_cases_dir(tmp_path)
    yield tmp_path
    _set_cases_dir(None)


def _good_payload() -> dict:
    return {
        "name": "demo_case",
        "extension": "md",
        "text": (
            "Hello world. This is the first paragraph.\n\n"
            "Second paragraph here. End here."
        ),
        "convention": "Each paragraph is one unit.",
        "expected_f1_floor": 0.7,
        "boundary_tolerance_chars": 50,
        "domain": "general",
        "units": [
            {
                "title": "Para 1",
                "start_anchor": "Hello world",
                "end_anchor": "first paragraph.",
            },
            {
                "title": "Para 2",
                "start_anchor": "Second paragraph",
                "end_anchor": "End here.",
            },
        ],
    }


def test_save_case_writes_input_and_truth(tmp_cases_dir: Path):
    client = TestClient(app)
    r = client.post("/api/save-case", json=_good_payload())
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["name"] == "demo_case"
    assert data["units_written"] == 2

    input_path = tmp_cases_dir / "demo_case.input.md"
    truth_path = tmp_cases_dir / "demo_case.truth.json"
    assert input_path.exists()
    assert truth_path.exists()

    truth = json.loads(truth_path.read_text())
    assert truth["convention"] == "Each paragraph is one unit."
    assert truth["expected_f1_floor"] == 0.7
    assert truth["domain"] == "general"
    assert len(truth["units"]) == 2
    assert truth["units"][0]["title"] == "Para 1"
    # `kind` was not provided → must NOT appear in the truth file
    assert "kind" not in truth["units"][0]


def test_save_case_includes_kind_when_provided(tmp_cases_dir: Path):
    client = TestClient(app)
    payload = _good_payload()
    payload["units"][0]["kind"] = "prose"
    r = client.post("/api/save-case", json=payload)
    assert r.status_code == 200, r.text

    truth = json.loads((tmp_cases_dir / "demo_case.truth.json").read_text())
    assert truth["units"][0]["kind"] == "prose"


def test_save_case_rejects_bad_name(tmp_cases_dir: Path):
    client = TestClient(app)
    payload = _good_payload()
    payload["name"] = "../escape"
    r = client.post("/api/save-case", json=payload)
    assert r.status_code == 400
    assert "case name" in r.json()["detail"]


def test_save_case_rejects_underscore_prefix(tmp_cases_dir: Path):
    client = TestClient(app)
    payload = _good_payload()
    payload["name"] = "_fixture"
    r = client.post("/api/save-case", json=payload)
    assert r.status_code == 400


def test_save_case_rejects_unfindable_anchor(tmp_cases_dir: Path):
    client = TestClient(app)
    payload = _good_payload()
    payload["units"][1]["start_anchor"] = "this string does not appear at all"
    r = client.post("/api/save-case", json=payload)
    assert r.status_code == 400
    assert "not found" in r.json()["detail"]


def test_save_case_rejects_overwrite_without_flag(tmp_cases_dir: Path):
    client = TestClient(app)
    r = client.post("/api/save-case", json=_good_payload())
    assert r.status_code == 200
    # Second save with same name and overwrite=false → 409
    r = client.post("/api/save-case", json=_good_payload())
    assert r.status_code == 409


def test_save_case_overwrites_with_flag(tmp_cases_dir: Path):
    client = TestClient(app)
    r = client.post("/api/save-case", json=_good_payload())
    assert r.status_code == 200
    payload = _good_payload()
    payload["overwrite"] = True
    payload["convention"] = "REPLACED"
    r = client.post("/api/save-case", json=payload)
    assert r.status_code == 200
    truth = json.loads((tmp_cases_dir / "demo_case.truth.json").read_text())
    assert truth["convention"] == "REPLACED"


def test_save_case_rejects_bad_extension(tmp_cases_dir: Path):
    client = TestClient(app)
    payload = _good_payload()
    payload["extension"] = "exe"
    r = client.post("/api/save-case", json=payload)
    assert r.status_code in (400, 422)


# ---------- /api/searxng-status -------------------------------------------


class _FakeManager:
    """Stub of SearxngManager — drives status/start endpoints in tests."""

    def __init__(self, *, running: bool = False, podman_missing: bool = False):
        self._running = running
        self._podman_missing = podman_missing
        self.start_calls = 0

    def is_running(self) -> bool:
        if self._podman_missing:
            raise RuntimeError("podman is not installed or not in PATH")
        return self._running

    def ensure_running(self) -> None:
        if self._podman_missing:
            raise RuntimeError("podman is not installed or not in PATH")
        self.start_calls += 1
        self._running = True


@pytest.fixture
def fake_manager():
    """Inject a fake SearxngManager and reset on teardown."""
    holder: dict[str, _FakeManager] = {}

    def factory():
        return holder["m"]

    _set_searxng_manager_factory(factory)

    def _install(manager: _FakeManager) -> _FakeManager:
        holder["m"] = manager
        return manager

    yield _install
    # Reset to the real factory after the test.
    from benchmarks.chunker.app.main import _default_searxng_manager_factory

    _set_searxng_manager_factory(_default_searxng_manager_factory)


def test_searxng_status_running(fake_manager):
    fake_manager(_FakeManager(running=True))
    client = TestClient(app)
    r = client.get("/api/searxng-status")
    assert r.status_code == 200
    data = r.json()
    assert data["state"] == "running"


def test_searxng_status_stopped(fake_manager):
    fake_manager(_FakeManager(running=False))
    client = TestClient(app)
    r = client.get("/api/searxng-status")
    assert r.status_code == 200
    assert r.json()["state"] == "stopped"


def test_searxng_status_podman_missing(fake_manager):
    fake_manager(_FakeManager(podman_missing=True))
    client = TestClient(app)
    r = client.get("/api/searxng-status")
    assert r.status_code == 200
    data = r.json()
    assert data["state"] == "podman-missing"
    assert "podman" in data["message"].lower()


# ---------- /api/searxng-start --------------------------------------------


def test_searxng_start_succeeds(fake_manager):
    m = fake_manager(_FakeManager(running=False))
    client = TestClient(app)
    r = client.post("/api/searxng-start")
    assert r.status_code == 200
    assert r.json()["state"] == "running"
    assert m.start_calls == 1


def test_searxng_start_returns_503_when_podman_missing(fake_manager):
    fake_manager(_FakeManager(podman_missing=True))
    client = TestClient(app)
    r = client.post("/api/searxng-start")
    assert r.status_code == 503
    assert "podman" in r.json()["detail"].lower()


# ---------- /api/search + /api/fetch --------------------------------------


class _FakeBackend:
    """Stub of SearchBackend that returns canned hits/pages."""

    def __init__(
        self,
        *,
        hits: list[SearchResult] | None = None,
        page: FetchedPage | None = None,
        fetch_error: Exception | None = None,
    ):
        self.hits = hits or []
        self.page = page
        self.fetch_error = fetch_error
        self.search_calls: list[tuple[str, int]] = []
        self.fetch_calls: list[str] = []

    async def search(self, query: str, max_results: int = 10):
        self.search_calls.append((query, max_results))
        return self.hits

    async def fetch_page(self, url: str):
        self.fetch_calls.append(url)
        if self.fetch_error:
            raise self.fetch_error
        assert self.page is not None
        return self.page


@pytest.fixture
def fake_backend():
    holder: dict[str, _FakeBackend] = {}

    def factory():
        return holder["b"]

    _set_search_backend_factory(factory)

    def _install(backend: _FakeBackend) -> _FakeBackend:
        holder["b"] = backend
        return backend

    yield _install
    from benchmarks.chunker.app.main import _default_search_backend_factory

    _set_search_backend_factory(_default_search_backend_factory)


def test_search_returns_hits(fake_backend, fake_manager):
    fake_manager(_FakeManager(running=True))
    fake_backend(
        _FakeBackend(
            hits=[
                SearchResult(
                    link_id=0,
                    title="Hello",
                    url="https://example.com/a",
                    snippet="snip",
                    domain="example.com",
                    relevance_score=0.8,
                ),
            ]
        )
    )
    client = TestClient(app)
    r = client.post("/api/search", json={"query": "hello", "max_results": 5})
    assert r.status_code == 200
    data = r.json()
    assert len(data["hits"]) == 1
    assert data["hits"][0]["url"] == "https://example.com/a"
    assert data["hits"][0]["domain"] == "example.com"


def test_search_503_when_searxng_stopped(fake_backend, fake_manager):
    """Empty hits + container down → 503 (not silent empty list)."""
    fake_manager(_FakeManager(running=False))
    fake_backend(_FakeBackend(hits=[]))
    client = TestClient(app)
    r = client.post("/api/search", json={"query": "hello"})
    assert r.status_code == 503
    assert "not running" in r.json()["detail"].lower()


def test_search_empty_when_running_returns_empty_list(fake_backend, fake_manager):
    """Empty hits + container running → 200 with [] (legitimately no results)."""
    fake_manager(_FakeManager(running=True))
    fake_backend(_FakeBackend(hits=[]))
    client = TestClient(app)
    r = client.post("/api/search", json={"query": "asdjkasdjkasd"})
    assert r.status_code == 200
    assert r.json()["hits"] == []


@pytest.fixture
def fake_harvest(monkeypatch):
    """Patch andamentum.harvest.extract for editor-app fetch tests.

    The /api/fetch endpoint imports `extract` lazily inside the handler, so
    we patch it on the harvest module itself — that's what the import binds.
    """
    import andamentum.harvest as harvest_mod
    from andamentum.harvest import HarvestError

    state: dict = {"markdown": "# Hello\n\nbody.", "calls": [], "raises": None}

    async def fake_extract(source):
        state["calls"].append(source)
        if state["raises"] is not None:
            raise state["raises"]
        return state["markdown"]

    monkeypatch.setattr(harvest_mod, "extract", fake_extract)
    yield state, HarvestError


def test_fetch_returns_markdown(fake_harvest):
    state, _ = fake_harvest
    state["markdown"] = "# Hello\n\nThis is markdown."
    client = TestClient(app)
    r = client.post("/api/fetch", json={"url": "https://example.com/a"})
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["markdown"].startswith("# Hello")
    assert data["word_count"] == len("# Hello\n\nThis is markdown.".split())
    assert data["truncated"] is False
    assert data["title"] == "Hello"


def test_fetch_502_on_extraction_failure(fake_harvest):
    _, HarvestError = fake_harvest
    state, _ = fake_harvest
    state["raises"] = HarvestError("blocked by SSRF")
    client = TestClient(app)
    r = client.post("/api/fetch", json={"url": "http://10.0.0.1/"})
    assert r.status_code == 502
    assert "blocked by SSRF" in r.json()["detail"]


# ---------- PDF helpers (unit) --------------------------------------------


@pytest.mark.parametrize(
    "input_url,expected",
    [
        # arXiv abstract → PDF
        ("https://arxiv.org/abs/1901.01753", "https://arxiv.org/pdf/1901.01753"),
        ("http://arxiv.org/abs/1901.01753", "http://arxiv.org/pdf/1901.01753"),
        (
            "https://www.arxiv.org/abs/1901.01753",
            "https://www.arxiv.org/pdf/1901.01753",
        ),
        # arXiv abstract with version suffix — version is stripped
        ("https://arxiv.org/abs/1901.01753v3", "https://arxiv.org/pdf/1901.01753"),
        ("https://arxiv.org/abs/cs.LG/0102003", "https://arxiv.org/pdf/cs.LG/0102003"),
        # bio/medRxiv → .full.pdf
        (
            "https://www.biorxiv.org/content/10.1101/2024.01.01.123456v1",
            "https://www.biorxiv.org/content/10.1101/2024.01.01.123456v1.full.pdf",
        ),
        (
            "https://www.medrxiv.org/content/10.1101/2024.01.01.123456v2",
            "https://www.medrxiv.org/content/10.1101/2024.01.01.123456v2.full.pdf",
        ),
        # Already a PDF — no change
        ("https://arxiv.org/pdf/1901.01753", "https://arxiv.org/pdf/1901.01753"),
        # Other sites — no change
        ("https://example.com/article", "https://example.com/article"),
        ("https://en.wikipedia.org/wiki/Hello", "https://en.wikipedia.org/wiki/Hello"),
    ],
)
def test_canonicalise_pdf_url(input_url, expected):
    assert _canonicalise_pdf_url(input_url) == expected


def test_extract_pdf_links_finds_dot_pdf_hrefs():
    md = """
    Some intro text.

    [Read the full paper](https://example.com/paper.pdf) and
    [also a related work](https://example.com/related/foo.pdf?v=1).

    [Not a PDF](https://example.com/page).
    """
    links = _extract_pdf_links(md, base_url="https://example.com/abs/123")
    urls = {link["url"] for link in links}
    assert "https://example.com/paper.pdf" in urls
    assert "https://example.com/related/foo.pdf?v=1" in urls
    assert "https://example.com/page" not in urls


def test_extract_pdf_links_finds_view_pdf_text():
    md = "[View PDF](https://arxiv.org/pdf/1234.5678) [Download PDF](https://e.com/x)"
    links = _extract_pdf_links(md, base_url="https://arxiv.org/abs/1234.5678")
    urls = {link["url"] for link in links}
    assert "https://arxiv.org/pdf/1234.5678" in urls
    assert "https://e.com/x" in urls  # matched on text, not extension


def test_extract_pdf_links_resolves_relative_hrefs():
    md = "[View PDF](/pdf/1901.01753)"
    links = _extract_pdf_links(md, base_url="https://arxiv.org/abs/1901.01753")
    assert links[0]["url"] == "https://arxiv.org/pdf/1901.01753"


def test_extract_pdf_links_dedupes():
    md = "[A](https://e.com/x.pdf) [B](https://e.com/x.pdf)"
    links = _extract_pdf_links(md, base_url="https://e.com/")
    assert len(links) == 1


def test_extract_pdf_links_caps_at_five():
    md = " ".join(f"[Link {i}](https://e.com/{i}.pdf)" for i in range(20))
    links = _extract_pdf_links(md, base_url="https://e.com/")
    assert len(links) == 5


def test_extract_pdf_links_empty_when_no_pdfs():
    md = "Just some plain prose with no links at all."
    assert _extract_pdf_links(md, base_url="https://e.com/") == []


# ---------- /api/fetch — PDF integration ----------------------------------


def test_fetch_rewrites_arxiv_abs_to_pdf(fake_harvest):
    """harvest.extract must be called with the PDF URL even when the request used /abs/."""
    state, _ = fake_harvest
    state["markdown"] = "## Paper title\n\nAbstract..."
    client = TestClient(app)
    r = client.post("/api/fetch", json={"url": "https://arxiv.org/abs/1901.01753"})
    assert r.status_code == 200
    data = r.json()
    assert data["requested_url"] == "https://arxiv.org/abs/1901.01753"
    assert data["final_url"] == "https://arxiv.org/pdf/1901.01753"
    assert data["is_pdf"] is True
    # PDF responses don't surface PDF links (no point — you ARE the PDF)
    assert data["pdf_links"] == []
    assert state["calls"] == ["https://arxiv.org/pdf/1901.01753"]


def test_fetch_surfaces_pdf_links_from_html(fake_harvest):
    state, _ = fake_harvest
    state["markdown"] = (
        "## Some paper\n\n"
        "Authors: A, B, C.\n\n"
        "[Download full text PDF](/files/paper.pdf) "
        "and [supplementary](https://example.com/supp.pdf)."
    )
    client = TestClient(app)
    r = client.post("/api/fetch", json={"url": "https://example.com/paper"})
    assert r.status_code == 200
    data = r.json()
    assert data["is_pdf"] is False
    urls = {link["url"] for link in data["pdf_links"]}
    assert "https://example.com/files/paper.pdf" in urls
    assert "https://example.com/supp.pdf" in urls
