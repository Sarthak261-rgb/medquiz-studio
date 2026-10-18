"""HTTP API over the question store.

uvicorn medquiz.api:app --reload
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from fastapi import Depends, FastAPI, HTTPException, Query

from .config import get_settings
from .schemas import FAILURE_LABELS, ReviewIn
from .store import Store

app = FastAPI(
    title="MedQuiz Studio",
    version="0.1.0",
    description="Grounded anatomy MCQs with automatic checks and human review.",
)


@lru_cache(maxsize=1)
def _default_store() -> Store:
    s = get_settings()
    return Store(s.database_url, review_log_dir=s.path("data") / "reviews")


def get_store() -> Store:
    return _default_store()


@app.get("/health")
def health(store: Store = Depends(get_store)) -> dict:
    return {"status": "ok", **store.counts()}


@app.get("/runs")
def runs(store: Store = Depends(get_store)) -> list[dict]:
    return store.runs()


@app.get("/questions")
def list_questions(
    origin: Literal["generated", "book", "mutant"] | None = "generated",
    run_id: str | None = None,
    auto_status: Literal["pass", "fail", "incomplete", "unchecked"] | None = None,
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    store: Store = Depends(get_store),
) -> list[dict]:
    return store.questions(origin=origin, run_id=run_id, auto_status=auto_status, limit=limit, offset=offset)


@app.get("/questions/{question_id}")
def get_question(question_id: str, store: Store = Depends(get_store)) -> dict:
    q = store.question(question_id)
    if q is None:
        raise HTTPException(404, "question not found")
    return q


@app.post("/questions/{question_id}/reviews", status_code=201)
def post_review(question_id: str, review: ReviewIn, store: Store = Depends(get_store)) -> dict:
    try:
        return store.add_review(question_id, review)
    except KeyError:
        raise HTTPException(404, "question not found") from None


@app.get("/review/next")
def next_question(reviewer: str, run_id: str | None = None, store: Store = Depends(get_store)) -> dict:
    qid = store.next_for_review(reviewer, run_id=run_id)
    if qid is None:
        return {"done": True}
    q = store.question(qid)
    # blind review: the reviewer sees the question and the source, not the checks
    q.pop("checks", None)
    q.pop("auto_status", None)
    q.pop("auto_failures", None)
    return {"done": False, "question": q}


@app.get("/chunks/{chunk_id}")
def get_chunk(chunk_id: str, store: Store = Depends(get_store)) -> dict:
    c = store.chunk(chunk_id)
    if c is None:
        raise HTTPException(404, "chunk not found")
    return c


@app.get("/failure-types")
def failure_types() -> dict[str, str]:
    return {k.value: v for k, v in FAILURE_LABELS.items()}


@app.get("/metrics")
def metrics(run_id: str | None = None, store: Store = Depends(get_store)) -> dict:
    from .report import compute

    return compute(store.snapshot(), get_settings(), gen_run=run_id)
