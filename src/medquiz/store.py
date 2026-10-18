"""Reads and writes for the API, the review app and the report."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from sqlalchemy import create_engine, delete, func, select
from sqlalchemy.orm import Session, sessionmaker

from .checks import auto_verdict
from .checks.runner import applicable
from .config import ChecksCfg
from .data import load_checks, load_questions
from .db import Base, CheckRow, ChunkRow, QuestionRow, ReviewRow, RunRow
from .runs import read_jsonl, utcnow
from .schemas import Chunk, ReviewIn


def _order_key(qid: str) -> str:
    # stable pseudo-random order, so the queue does not walk the book section by section
    return hashlib.sha1(qid.encode()).hexdigest()


def reviewer_slug(name: str) -> str:
    return re.sub(r"[^a-z0-9_-]+", "-", name.strip().lower()).strip("-") or "anonymous"


class Store:
    def __init__(self, url: str, review_log_dir: Path | None = None):
        args = {"check_same_thread": False} if url.startswith("sqlite") else {}
        if url.startswith("sqlite:///"):
            Path(url.removeprefix("sqlite:///")).parent.mkdir(parents=True, exist_ok=True)
        self.engine = create_engine(url, connect_args=args)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(self.engine, expire_on_commit=False)
        self.review_log_dir = review_log_dir

    # ------------------------------------------------------------ loading

    def load_chunks(self, chunks: list[Chunk]) -> None:
        with self.Session.begin() as s:
            for c in chunks:
                s.merge(ChunkRow(**c.model_dump()))

    def load_run(self, run_path: Path, checks_cfg: ChecksCfg) -> int:
        meta_file = run_path / "run.json"
        meta = json.loads(meta_file.read_text(encoding="utf-8")) if meta_file.exists() else {}
        questions = load_questions(run_path)
        checks = load_checks(run_path)
        with self.Session.begin() as s:
            s.merge(
                RunRow(
                    id=run_path.name,
                    kind=meta.get("kind", "unknown"),
                    created_at=meta.get("created_at", utcnow()),
                    meta=meta,
                )
            )
            for q in questions:
                results = checks.get(q.id, [])
                verdict = auto_verdict(results, checks_cfg, applicable(q, checks_cfg.hard)) if results else None
                s.merge(
                    QuestionRow(
                        id=q.id,
                        run_id=run_path.name,
                        chunk_id=q.chunk_id,
                        origin=q.origin,
                        mutation=q.mutation,
                        parent_id=q.parent_id,
                        target_type=q.target_type,
                        stem=q.stem,
                        options=q.options,
                        answer_index=q.answer_index,
                        explanation=q.explanation,
                        evidence_quote=q.evidence_quote,
                        answer_statement=q.answer_statement,
                        bloom_level=q.bloom_level,
                        gen=q.gen.model_dump(),
                        auto_status=verdict.status if verdict else "unchecked",
                        auto_failures=[f.value for f in verdict.failures] if verdict else [],
                    )
                )
                s.execute(delete(CheckRow).where(CheckRow.question_id == q.id))
                for r in results:
                    s.add(CheckRow(question_id=q.id, **r.model_dump()))
        return len(questions)

    def replay_reviews(self, log_dir: Path) -> int:
        """Rebuild the reviews table from the append-only logs."""
        n = 0
        with self.Session.begin() as s:
            s.execute(delete(ReviewRow))
            known = set(s.scalars(select(QuestionRow.id)))
            for f in sorted(log_dir.glob("*.jsonl")):
                for row in read_jsonl(f):
                    if row["question_id"] in known:
                        s.add(ReviewRow(**row))
                        n += 1
        return n

    # ------------------------------------------------------------ reviews

    def add_review(self, question_id: str, review: ReviewIn) -> dict:
        row = {
            "question_id": question_id,
            "reviewer": review.reviewer.strip(),
            "decision": review.decision,
            "reasons": [r.value for r in review.reasons],
            "level": review.level,
            "note": review.note.strip(),
            "seconds": round(review.seconds, 1) if review.seconds is not None else None,
            "created_at": utcnow(),
        }
        with self.Session.begin() as s:
            if s.get(QuestionRow, question_id) is None:
                raise KeyError(question_id)
            s.add(ReviewRow(**row))
        if self.review_log_dir is not None:
            self.review_log_dir.mkdir(parents=True, exist_ok=True)
            path = self.review_log_dir / f"{reviewer_slug(review.reviewer)}.jsonl"
            with path.open("a", encoding="utf-8", newline="\n") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        return row

    def latest_reviews(self, reviewer: str | None = None) -> dict[str, dict]:
        """The most recent decision per question (per reviewer if given)."""
        with self.Session() as s:
            q = select(ReviewRow).order_by(ReviewRow.id)
            if reviewer:
                q = q.where(ReviewRow.reviewer == reviewer)
            latest: dict[str, ReviewRow] = {}
            for r in s.scalars(q):
                latest[r.question_id] = r
        return {k: _review_dict(v) for k, v in latest.items()}

    def next_for_review(self, reviewer: str, origin: str = "generated", run_id: str | None = None) -> str | None:
        with self.Session() as s:
            reviewed = select(ReviewRow.question_id).where(ReviewRow.reviewer == reviewer)
            q = select(QuestionRow.id).where(QuestionRow.origin == origin, QuestionRow.id.not_in(reviewed))
            if run_id:
                q = q.where(QuestionRow.run_id == run_id)
            ids = list(s.scalars(q))
        return min(ids, key=_order_key) if ids else None

    # ------------------------------------------------------------ queries

    def question(self, qid: str) -> dict | None:
        with self.Session() as s:
            q = s.get(QuestionRow, qid)
            if q is None:
                return None
            chunk = s.get(ChunkRow, q.chunk_id)
            checks = list(s.scalars(select(CheckRow).where(CheckRow.question_id == qid).order_by(CheckRow.id)))
            reviews = list(s.scalars(select(ReviewRow).where(ReviewRow.question_id == qid).order_by(ReviewRow.id)))
            return {
                **_question_dict(q),
                "chunk": _chunk_dict(chunk) if chunk else None,
                "checks": [_check_dict(c) for c in checks],
                "reviews": [_review_dict(r) for r in reviews],
            }

    def questions(
        self,
        origin: str | None = None,
        run_id: str | None = None,
        auto_status: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[dict]:
        with self.Session() as s:
            q = select(QuestionRow).order_by(QuestionRow.chunk_id, QuestionRow.id)
            if origin:
                q = q.where(QuestionRow.origin == origin)
            if run_id:
                q = q.where(QuestionRow.run_id == run_id)
            if auto_status:
                q = q.where(QuestionRow.auto_status == auto_status)
            return [_question_dict(r) for r in s.scalars(q.limit(limit).offset(offset))]

    def runs(self) -> list[dict]:
        with self.Session() as s:
            counts = dict(s.execute(select(QuestionRow.run_id, func.count()).group_by(QuestionRow.run_id)).all())
            return [
                {
                    "id": r.id,
                    "kind": r.kind,
                    "created_at": r.created_at,
                    "n_questions": counts.get(r.id, 0),
                    "meta": r.meta,
                }
                for r in s.scalars(select(RunRow).order_by(RunRow.created_at))
            ]

    def chunk(self, chunk_id: str) -> dict | None:
        with self.Session() as s:
            c = s.get(ChunkRow, chunk_id)
            return _chunk_dict(c) if c else None

    def counts(self) -> dict:
        with self.Session() as s:
            by_origin = dict(s.execute(select(QuestionRow.origin, func.count()).group_by(QuestionRow.origin)).all())
            reviewed = s.scalar(select(func.count(func.distinct(ReviewRow.question_id))))
            return {"questions": by_origin, "reviewed_questions": reviewed or 0}

    def snapshot(self) -> dict:
        """Everything the metrics need, as plain dicts."""
        with self.Session() as s:
            questions = [_question_dict(q) for q in s.scalars(select(QuestionRow))]
            checks: dict[str, dict[str, dict]] = {}
            for c in s.scalars(select(CheckRow)):
                checks.setdefault(c.question_id, {})[c.name] = _check_dict(c)
            reviews = [_review_dict(r) for r in s.scalars(select(ReviewRow).order_by(ReviewRow.id))]
            runs = {r.id: r.meta for r in s.scalars(select(RunRow))}
            chunks = {c.id: {"chapter": c.chapter, "section": c.section} for c in s.scalars(select(ChunkRow))}
        return {"questions": questions, "checks": checks, "reviews": reviews, "runs": runs, "chunks": chunks}

    def session(self) -> Session:
        return self.Session()


def _question_dict(q: QuestionRow) -> dict:
    return {
        "id": q.id,
        "run_id": q.run_id,
        "chunk_id": q.chunk_id,
        "origin": q.origin,
        "mutation": q.mutation,
        "parent_id": q.parent_id,
        "target_type": q.target_type,
        "stem": q.stem,
        "options": q.options,
        "answer_index": q.answer_index,
        "explanation": q.explanation,
        "evidence_quote": q.evidence_quote,
        "answer_statement": q.answer_statement,
        "bloom_level": q.bloom_level,
        "gen": q.gen,
        "auto_status": q.auto_status,
        "auto_failures": q.auto_failures,
    }


def _chunk_dict(c: ChunkRow) -> dict:
    return {
        "id": c.id,
        "chapter": c.chapter,
        "section": c.section,
        "section_title": c.section_title,
        "heading": c.heading,
        "module_id": c.module_id,
        "url": c.url,
        "para_ids": c.para_ids,
        "text": c.text,
        "n_words": c.n_words,
        "cite_url": f"{c.url}#{c.para_ids[0]}" if c.para_ids else c.url,
    }


def _check_dict(c: CheckRow) -> dict:
    return {
        "name": c.name,
        "checker": c.checker,
        "severity": c.severity,
        "passed": c.passed,
        "score": c.score,
        "detail": c.detail,
        "tokens": c.tokens,
        "seconds": c.seconds,
    }


def _review_dict(r: ReviewRow) -> dict:
    return {
        "question_id": r.question_id,
        "reviewer": r.reviewer,
        "decision": r.decision,
        "reasons": r.reasons,
        "level": r.level,
        "note": r.note,
        "seconds": r.seconds,
        "created_at": r.created_at,
    }
