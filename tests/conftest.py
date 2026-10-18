from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

import pytest
import yaml

from medquiz.config import DEFAULT_CONFIG, load_settings
from medquiz.llm import CallStats
from medquiz.schemas import Chunk, GenMeta, Question

FIXTURES = Path(__file__).parent / "fixtures"


class FakeLLM:
    """Returns canned outputs per schema name, in order, and records the prompts."""

    def __init__(self, responses: dict[str, list[dict]]):
        self.responses = {k: list(v) for k, v in responses.items()}
        self.calls: list[tuple[str, str, str]] = []

    def chat_json(self, model, system, user, schema, **kw):
        self.calls.append((model, system, user))
        data = self.responses[schema.__name__].pop(0)
        return schema.model_validate(data), CallStats(
            model=model, prompt_tokens=100, completion_tokens=50, seconds=0.5, attempts=1
        )


@pytest.fixture
def settings():
    return load_settings()


@pytest.fixture
def workdir():
    # not pytest's tmp_path: on Windows without developer mode its "current"
    # symlink takes ~30 s to fail for every test
    d = Path(tempfile.mkdtemp(prefix="medquiz-test-"))
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def tmp_settings(workdir, monkeypatch):
    """Settings whose data, runs, reports and database all live in workdir."""
    cfg = yaml.safe_load(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    for key in ("data", "runs", "reports", "releases"):
        cfg["paths"][key] = str(workdir / key)
    cfg["paths"]["database"] = f"sqlite:///{(workdir / 'test.db').as_posix()}"
    path = workdir / "config.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    monkeypatch.setenv("MEDQUIZ_CONFIG", str(path))
    from medquiz import config

    config.get_settings.cache_clear()
    yield load_settings(path)
    config.get_settings.cache_clear()


@pytest.fixture
def chunk() -> Chunk:
    text = (
        "The sinoatrial (SA) node is the normal pacemaker of the heart. It initiates the sinus rhythm. "
        "Impulses then travel to the atrioventricular (AV) node, where they are delayed for about 100 ms. "
        "The delay lets the atria finish contracting before the ventricles begin."
    )
    return Chunk(
        id="19.2-004",
        chapter=19,
        section="19.2",
        section_title="Cardiac Muscle and Electrical Activity",
        heading="Cardiac Muscle and Electrical Activity > Conduction System of the Heart",
        module_id="m46664",
        para_ids=["fs-1"],
        text=text,
        n_words=len(text.split()),
        url="https://openstax.org/books/anatomy-and-physiology-2e/pages/19-2-cardiac-muscle-and-electrical-activity",
    )


@pytest.fixture
def question(chunk) -> Question:
    return Question(
        id="q_test000001",
        run_id="gen-test",
        chunk_id=chunk.id,
        target_type="recall",
        stem="Which structure is the normal pacemaker of the heart?",
        options=["atrioventricular node", "sinoatrial node", "bundle of His", "Purkinje fibers"],
        answer_index=1,
        explanation="The SA node initiates the sinus rhythm.",
        evidence_quote="The sinoatrial (SA) node is the normal pacemaker of the heart.",
        answer_statement="The sinoatrial node is the normal pacemaker of the heart.",
        bloom_level="remember",
        gen=GenMeta(
            model="fake",
            prompt="generate_v1",
            prompt_hash="abc",
            prompt_tokens=600,
            completion_tokens=200,
            seconds=10.0,
            original_answer_index=0,
        ),
    )


def write_run(
    root: Path,
    run_id: str,
    questions: list[Question],
    checks: dict[str, list[dict]] | None = None,
    meta: dict | None = None,
) -> Path:
    path = root / run_id
    path.mkdir(parents=True, exist_ok=True)
    (path / "run.json").write_text(
        json.dumps(meta or {"id": run_id, "kind": "generate", "created_at": "2026-10-18T00:00:00Z"})
    )
    with (path / "questions.jsonl").open("w", encoding="utf-8") as f:
        for q in questions:
            f.write(q.model_dump_json() + "\n")
    with (path / "checks.jsonl").open("w", encoding="utf-8") as f:
        for qid, rows in (checks or {}).items():
            f.write(json.dumps({"question_id": qid, "stage": "all", "checks": rows}) + "\n")
    return path


def passing_checks(settings) -> list[dict]:
    names = [*settings.checks.hard, *settings.checks.soft]
    return [
        {
            "name": n,
            "checker": "rule",
            "severity": "hard",
            "passed": True,
            "score": None,
            "detail": {},
            "tokens": 10,
            "seconds": 1.0,
        }
        for n in names
    ]
