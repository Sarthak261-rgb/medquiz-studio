"""A run is a folder under runs/ holding append-only JSONL files and a run.json
with everything needed to say how the results were produced."""

from __future__ import annotations

import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .config import ROOT


def utcnow() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def git_sha() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "--short=12", "HEAD"], cwd=ROOT, capture_output=True, text=True)
        sha = out.stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--", "src", "prompts", "configs"],
            cwd=ROOT,
            capture_output=True,
            text=True,
        ).stdout.strip()
        return f"{sha}{'-dirty' if dirty else ''}" if sha else "unknown"
    except OSError:
        return "unknown"


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


class RunDir:
    def __init__(self, root: Path, run_id: str):
        self.id = run_id
        self.path = root / run_id
        self.path.mkdir(parents=True, exist_ok=True)

    @property
    def meta_path(self) -> Path:
        return self.path / "run.json"

    def meta(self) -> dict[str, Any]:
        if self.meta_path.exists():
            return json.loads(self.meta_path.read_text(encoding="utf-8"))
        return {"id": self.id, "created_at": utcnow()}

    def update_meta(self, **fields: Any) -> dict[str, Any]:
        meta = self.meta()
        for k, v in fields.items():
            if isinstance(v, dict) and isinstance(meta.get(k), dict):
                meta[k].update(v)
            else:
                meta[k] = v
        self.meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
        return meta

    def append(self, name: str, row: dict) -> None:
        with (self.path / name).open("a", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def read(self, name: str) -> list[dict]:
        return read_jsonl(self.path / name)


def new_run_id(kind: str) -> str:
    return f"{kind}-{datetime.now().strftime('%Y%m%d-%H%M')}"
