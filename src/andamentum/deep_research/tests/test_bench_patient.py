"""The bench asks a rate-limited pool patiently: ``Services.patient`` (``--patient
SOURCE``) reaches every arm's HTTP client alike, and a cell whose patient search
ran out of retries, or whose source was given up for the process, is not
recorded (never a zero), so the same command resumes it in a new process.
A mock transport, a fake sleep and a scripted model; no network."""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from ..bench import arms as bench_arms
from ..bench import cli as bench_cli
from ..bench.arms import FinderArm, ResearchArm, SemanticScholarArm
from ..bench.experiment import ArmSpec, Experiment
from ..bench.experiment_run import (
    DataDirs,
    InfrastructureDown,
    Services,
    build_answer_arm,
    run_experiment,
)
from ..bench.registry import RunRecord, StartRecord, read_records, registry_path
from ..sources.base import (
    PATIENT_RETRY_WAITS_S,
    SharedPacing,
    SourceClient,
    SourceConfig,
)
from ..store import ResearchStore
from .llm_stub import ScriptedLLM
from .test_bench_experiment_run import gold_title, write_litsearch

PATIENT = ("semantic_scholar",)


def test_every_arm_gets_the_runs_patient_sources() -> None:
    services = Services(patient=PATIENT)
    specs = [
        ArmSpec(name="B1", system="research", model="ollama:m"),
        ArmSpec(name="B2", system="s2_search"),
        ArmSpec(name="F1", system="finder", model="ollama:m", follow_citations=True),
    ]
    built = [build_answer_arm(spec, "scholarly", services) for spec in specs]
    assert [type(a) for a in built] == [ResearchArm, SemanticScholarArm, FinderArm]
    assert all(a.patient == PATIENT for a in built)  # type: ignore[attr-defined]
    plain = build_answer_arm(specs[1], "scholarly", Services())
    assert plain.patient == ()  # type: ignore[attr-defined]


def test_the_cli_flag_sets_the_patient_sources() -> None:
    parser = bench_cli.build_parser()
    args = parser.parse_args(
        [
            "experiment",
            "run",
            "f.yaml",
            "--out-dir",
            "o",
            "--patient",
            "semantic_scholar",
        ]
    )
    assert args.patient == ["semantic_scholar"]
    assert (
        parser.parse_args(["experiment", "run", "f.yaml", "--out-dir", "o"]).patient
        is None
    )
    with pytest.raises(SystemExit):
        parser.parse_args(
            ["experiment", "run", "f.yaml", "--out-dir", "o", "--patient", "web"]
        )


# -- a process of the bench on a mock transport ----------------------------------------


class Pool:
    """Semantic Scholar's keyless pool: 429 while ``limited``, else one hit, the
    item's gold paper. OpenAlex and Europe PMC answer with no results."""

    def __init__(self) -> None:
        self.limited = True
        self.s2_requests = 0

    def handle(self, request: httpx.Request) -> httpx.Response:
        host = request.url.host
        if host == "api.semanticscholar.org":
            self.s2_requests += 1
            if self.limited:
                return httpx.Response(429, json={"message": "Too Many Requests"})
            if request.url.path.endswith(("/references", "/citations")):
                return httpx.Response(200, json={"data": []})
            number = int(request.url.params["query"].split()[-1].rstrip("?"))
            return httpx.Response(200, json={"total": 1, "data": [s2_hit(number)]})
        if host == "api.openalex.org":
            return httpx.Response(
                200, json={"meta": {"count": 0, "next_cursor": None}, "results": []}
            )
        if host == "www.ebi.ac.uk":
            return httpx.Response(
                200, json={"hitCount": 0, "resultList": {"result": []}}
            )
        return httpx.Response(404)


def s2_hit(number: int) -> dict[str, object]:
    return {
        "paperId": f"{number:040d}",
        "externalIds": {"DOI": f"10.5555/kiln.{number}"},
        "title": gold_title(number),
        "abstract": "About kilns.",
        "year": 2020,
        "venue": "Invented Kiln Letters",
        "authors": [],
    }


class Process:
    """One bench process: every client shares one pacing (as the process's
    pacing would) and waits on a fake sleep that records the waits."""

    def __init__(self, pool: Pool) -> None:
        self.pool = pool
        self.pacing = SharedPacing()
        self.sleeps: list[float] = []
        self.patient_seen: list[tuple[str, ...]] = []

    async def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)

    def client(self, store: ResearchStore, **kw: object) -> SourceClient:
        patient = tuple(kw.pop("patient", ()))  # type: ignore[arg-type]
        self.patient_seen.append(patient)
        return SourceClient(
            store,
            client=httpx.AsyncClient(transport=httpx.MockTransport(self.pool.handle)),
            intervals=dict.fromkeys(("semantic_scholar", "openalex", "europepmc"), 0.0),
            clock=lambda: 0.0,
            sleep=self.sleep,
            pacing=self.pacing,
            patient=patient,
            **kw,  # type: ignore[arg-type]
        )


def experiment(arm: dict[str, object]) -> Experiment:
    return Experiment.model_validate(
        {
            "name": "patient-tiny",
            "registered": "2026-10-08",
            "sets": [{"name": "litsearch-web", "n": 2, "seed": 0}],
            "arms": [arm],
            "baselines": {"litsearch-web": arm["name"]},
        }
    )


def runs(out: Path) -> list[RunRecord]:
    return [r for r in read_records(registry_path(out)) if isinstance(r, RunRecord)]


def starts(out: Path) -> list[StartRecord]:
    return [r for r in read_records(registry_path(out)) if isinstance(r, StartRecord)]


async def test_a_search_that_ran_out_of_patient_retries_is_not_recorded_and_resumes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dirs = DataDirs(litsearch_local=write_litsearch(tmp_path / "ls", n=4))
    exp = experiment({"name": "B2", "system": "s2_search"})
    out = tmp_path / "out"
    pool = Pool()
    first = Process(pool)
    monkeypatch.setattr(bench_arms, "SourceClient", first.client)
    with pytest.raises(InfrastructureDown, match="not recorded"):
        await run_experiment(
            exp, out_dir=out, dirs=dirs, services=Services(patient=PATIENT)
        )
    # the patient schedule was used: one request, eight retries with its waits
    assert first.patient_seen == [PATIENT]
    assert first.sleeps == list(PATIENT_RETRY_WAITS_S)
    assert pool.s2_requests == len(PATIENT_RETRY_WAITS_S) + 1
    assert first.pacing.rate_limited["semantic_scholar"] == pool.s2_requests
    assert first.pacing.patient_waits["semantic_scholar"] == len(PATIENT_RETRY_WAITS_S)
    # nothing recorded: no zero recall, no error line
    assert runs(out) == []
    assert starts(out)[0].patient == ["semantic_scholar"]

    # the pool answers again; the same command in a new process records the cells
    pool.limited = False
    second = Process(pool)
    monkeypatch.setattr(bench_arms, "SourceClient", second.client)
    written = await run_experiment(
        exp, out_dir=out, dirs=dirs, services=Services(patient=PATIENT)
    )
    assert len(written) == 2 and [r.error for r in written] == [None, None]
    assert [r.metrics["recall_at_20"] for r in runs(out)] == [1.0, 1.0]
    assert second.sleeps == []


async def test_a_source_given_up_for_the_process_is_not_asked_and_nothing_is_recorded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dirs = DataDirs(litsearch_local=write_litsearch(tmp_path / "ls", n=4))
    pool = Pool()
    process = Process(pool)
    process.pacing.gave_up.add("semantic_scholar")
    monkeypatch.setattr(bench_arms, "SourceClient", process.client)
    out = tmp_path / "out"
    with pytest.raises(
        InfrastructureDown, match="refused for the rest of this process"
    ):
        await run_experiment(
            experiment({"name": "B2", "system": "s2_search"}),
            out_dir=out,
            dirs=dirs,
            services=Services(patient=PATIENT),
        )
    assert pool.s2_requests == 0
    assert runs(out) == []


async def test_a_finder_cell_whose_patient_search_gave_up_is_not_recorded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dirs = DataDirs(litsearch_local=write_litsearch(tmp_path / "ls", n=4))
    llm = ScriptedLLM(
        lambda instructions, _prompt: (
            {"queries": ["kiln lids", "kiln lid study"]}
            if "literature search queries" in instructions
            else {"seeds": [1]}
            if "`seeds`" in instructions
            else {"ranked": [1]}
        )
    )
    finder = FinderArm(
        model=llm.model,
        follow_citations=True,
        sources=SourceConfig(),
        patient=PATIENT,
        name="F1",
    )
    pool = Pool()
    process = Process(pool)
    monkeypatch.setattr(bench_arms, "SourceClient", process.client)
    out = tmp_path / "out"
    with pytest.raises(InfrastructureDown, match="semantic_scholar search failed"):
        await run_experiment(
            experiment(
                {
                    "name": "F1",
                    "system": "finder",
                    "model": "ollama:m",
                    "follow_citations": True,
                }
            ),
            out_dir=out,
            dirs=dirs,
            services=Services(patient=PATIENT, answer_arm=lambda _spec, _mode: finder),
        )
    assert process.patient_seen == [PATIENT]
    # both queries ran out of the patient retries on Semantic Scholar
    assert process.pacing.patient_waits["semantic_scholar"] == 2 * len(
        PATIENT_RETRY_WAITS_S
    )
    assert runs(out) == []
