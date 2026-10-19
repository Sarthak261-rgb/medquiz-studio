"""Human review of generated images. Append-only JSONL, one file per reviewer, like
the question reviews. The queue mixes both arms in a fixed pseudo-random order and
never shows which arm an image came from, so base and LoRA are judged blind."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from pydantic import BaseModel, Field, model_validator

from ..runs import read_jsonl, utcnow
from ..store import reviewer_slug
from . import IMAGE_FAILURES


class ImageReviewIn(BaseModel):
    reviewer: str = Field(min_length=1, max_length=64)
    decision: str = Field(pattern="^(accept|reject)$")
    reasons: list[str] = Field(default_factory=list)
    note: str = ""
    seconds: float | None = None

    @model_validator(mode="after")
    def check(self) -> ImageReviewIn:
        unknown = [r for r in self.reasons if r not in IMAGE_FAILURES]
        if unknown:
            raise ValueError(f"unknown reasons {unknown}")
        if self.decision == "reject" and not self.reasons:
            raise ValueError("a rejection needs at least one reason")
        if self.decision == "accept" and self.reasons:
            raise ValueError("an accepted image cannot have rejection reasons")
        return self


def log_path(log_dir: Path, reviewer: str) -> Path:
    return log_dir / f"images-{reviewer_slug(reviewer)}.jsonl"


def add_image_review(log_dir: Path, image_id: str, review: ImageReviewIn) -> dict:
    row = {
        "image_id": image_id,
        "reviewer": review.reviewer.strip(),
        "decision": review.decision,
        "reasons": list(dict.fromkeys(review.reasons)),
        "note": review.note.strip(),
        "seconds": round(review.seconds, 1) if review.seconds is not None else None,
        "created_at": utcnow(),
    }
    log_dir.mkdir(parents=True, exist_ok=True)
    with log_path(log_dir, review.reviewer).open("a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return row


def latest_image_reviews(log_dir: Path, reviewer: str | None = None) -> dict[str, dict]:
    out: dict[str, dict] = {}
    files = [log_path(log_dir, reviewer)] if reviewer else sorted(log_dir.glob("images-*.jsonl"))
    for f in files:
        for r in read_jsonl(f):
            out[r["image_id"]] = r
    return out


def queue(images: list[dict], reviewed: set[str]) -> list[dict]:
    todo = [r for r in images if r["id"] not in reviewed]
    return sorted(todo, key=lambda r: hashlib.sha1(r["id"].encode()).hexdigest())
