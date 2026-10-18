import json

import pytest
from fastapi.testclient import TestClient

from medquiz import api
from medquiz.release import build_release, verify_release
from medquiz.report import compute, to_markdown
from medquiz.runs import write_jsonl
from medquiz.store import Store

from .conftest import passing_checks, write_run


@pytest.fixture
def loaded(tmp_settings, chunk, question):
    s = tmp_settings
    data = s.path("data")
    write_jsonl(data / "processed" / "chunks.jsonl", [chunk.model_dump()])
    (data / "processed" / "manifest.json").write_text(
        json.dumps({"pages": [{"section": "19.2", "title": "Cardiac Muscle and Electrical Activity"}]})
    )
    q2 = question.model_copy(update={"id": "q_test000002", "stem": "Which node delays the impulse?", "answer_index": 0})
    bad = passing_checks(s)
    for c in bad:
        if c["name"] == "judge_single_correct":
            c["passed"] = False
    write_run(
        s.path("runs"),
        "gen-test",
        [question, q2],
        {question.id: passing_checks(s), q2.id: bad},
        meta={
            "id": "gen-test",
            "kind": "generate",
            "created_at": "2026-10-18T00:00:00Z",
            "generator": {"model": "fake", "digest": "abc"},
            "stages": {"generate": {"seconds": 20.0, "gpu_joules": 7200.0, "n": 2, "gpu": "Test GPU"}},
        },
    )
    store = Store(s.database_url, review_log_dir=data / "reviews")
    store.load_chunks([chunk])
    store.load_run(s.path("runs") / "gen-test", s.checks)
    api.app.dependency_overrides[api.get_store] = lambda: store
    yield s, store
    api.app.dependency_overrides.clear()


def test_auto_status_is_stored(loaded):
    _, store = loaded
    assert store.question("q_test000001")["auto_status"] == "pass"
    q = store.question("q_test000002")
    assert q["auto_status"] == "fail" and q["auto_failures"] == ["multiple_correct"]
    assert q["chunk"]["cite_url"].endswith("#fs-1")


def test_review_flow_over_http(loaded):
    s, store = loaded
    client = TestClient(api.app)
    assert client.get("/health").json()["questions"] == {"generated": 2}

    nxt = client.get("/review/next", params={"reviewer": "sarthak"}).json()
    assert nxt["done"] is False
    assert "checks" not in nxt["question"] and "auto_status" not in nxt["question"]  # blind review
    qid = nxt["question"]["id"]

    r = client.post(f"/questions/{qid}/reviews", json={"reviewer": "sarthak", "decision": "reject"})
    assert r.status_code == 422  # a rejection needs a reason
    r = client.post(
        f"/questions/{qid}/reviews",
        json={"reviewer": "sarthak", "decision": "reject", "reasons": ["multiple_correct"], "seconds": 12},
    )
    assert r.status_code == 201
    assert client.post("/questions/nope/reviews", json={"reviewer": "x", "decision": "accept"}).status_code == 404

    other = client.get("/review/next", params={"reviewer": "sarthak"}).json()["question"]["id"]
    assert other != qid
    client.post(f"/questions/{other}/reviews", json={"reviewer": "sarthak", "decision": "accept", "level": "recall"})
    assert client.get("/review/next", params={"reviewer": "sarthak"}).json() == {"done": True}

    # every review is also in the append-only log, and the table can be rebuilt from it
    log = (s.path("data") / "reviews" / "sarthak.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(log) == 2
    assert store.replay_reviews(s.path("data") / "reviews") == 2

    m = client.get("/metrics").json()
    assert m["human"]["n_reviewed"] == 2


def test_report_and_release(loaded):
    s, store = loaded
    store.add_review("q_test000001", api.ReviewIn(reviewer="sarthak", decision="accept"))
    store.add_review("q_test000002", api.ReviewIn(reviewer="sarthak", decision="reject", reasons=["multiple_correct"]))
    snap = store.snapshot()
    m = compute(snap, s)
    assert m["agreement"]["overall"]["accuracy"] == 1.0  # checks and reviewer agree on both
    assert m["cost"]["total_gpu_wh"] == 2.0
    md = to_markdown(m)
    assert "Rejected: 50.0% (1/2" in md and "Judge vs human" in md

    out = build_release(snap, s, "0.0.1", "gen-test", m)
    rows = [json.loads(x) for x in (out / "questions.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [r["id"] for r in rows] == ["q_test000001"]
    assert rows[0]["answer"] == "sinoatrial node"
    assert verify_release(out) == []
    with pytest.raises(FileExistsError):
        build_release(snap, s, "0.0.1", "gen-test", m)
    (out / "questions.jsonl").write_text("tampered\n", encoding="utf-8")
    assert verify_release(out) == ["questions.jsonl"]
