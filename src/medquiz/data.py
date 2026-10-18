"""Loading the processed corpus and run outputs from disk."""

from __future__ import annotations

from pathlib import Path

from .runs import read_jsonl
from .schemas import BookQuestion, CheckResult, Chunk, Question


def processed_dir(data_dir: Path) -> Path:
    return data_dir / "processed"


def load_chunks(data_dir: Path) -> list[Chunk]:
    rows = read_jsonl(processed_dir(data_dir) / "chunks.jsonl")
    if not rows:
        raise FileNotFoundError("no chunks found; run `medquiz ingest` first")
    return [Chunk(**r) for r in rows]


def load_book_questions(data_dir: Path) -> list[BookQuestion]:
    return [BookQuestion(**r) for r in read_jsonl(processed_dir(data_dir) / "book_questions.jsonl")]


def load_questions(run_path: Path) -> list[Question]:
    return [Question(**r) for r in read_jsonl(run_path / "questions.jsonl")]


def load_checks(run_path: Path) -> dict[str, list[CheckResult]]:
    out: dict[str, list[CheckResult]] = {}
    for r in read_jsonl(run_path / "checks.jsonl"):
        # later lines win, so a re-run of one stage overrides older results
        merged = {c.name: c for c in out.get(r["question_id"], [])}
        merged.update({c["name"]: CheckResult(**c) for c in r["checks"]})
        out[r["question_id"]] = list(merged.values())
    return out
