"""Settings loaded from configs/*.yaml, with a few environment overrides."""

from __future__ import annotations

import hashlib
import os
from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs" / "default.yaml"
PROMPTS_DIR = ROOT / "prompts"


class Chapter(BaseModel):
    number: int
    title: str


class SourceCfg(BaseModel):
    repo: str
    commit: str
    collection: str
    book_slug: str
    license: str
    chapters: list[Chapter]
    keep_note_classes: list[str]


class ChunkCfg(BaseModel):
    target_words: int = 220
    max_words: int = 320
    min_words: int = 70


class EmbedCfg(BaseModel):
    model: str
    query_prefix: str = ""
    device: str = "cuda"
    batch_size: int = 64


class GeneratorCfg(BaseModel):
    model: str
    prompt: str
    temperature: float
    seed: int
    num_ctx: int = 4096
    questions_per_chunk: int = 1


class JudgeCfg(BaseModel):
    model: str
    temperature: float = 0.0
    seed: int = 11
    num_ctx: int = 4096


class NLICfg(BaseModel):
    model: str
    device: str = "cuda"
    entail_threshold: float = 0.5
    window_sentences: int = 3


class ChecksCfg(BaseModel):
    quote_match_threshold: float = 90
    duplicate_cosine: float = 0.92
    retrieval_top_k: int = 3
    length_cue_ratio: float = 1.6
    hard: list[str]
    soft: list[str]


class PathsCfg(BaseModel):
    data: str = "data"
    runs: str = "runs"
    reports: str = "reports"
    releases: str = "releases"
    database: str = "sqlite:///data/medquiz.db"


class CostCfg(BaseModel):
    electricity_inr_per_kwh: float = 8.0


class Settings(BaseModel):
    source: SourceCfg
    chunking: ChunkCfg
    embedding: EmbedCfg
    generator: GeneratorCfg
    judge: JudgeCfg
    nli: NLICfg
    checks: ChecksCfg
    ollama_url: str
    paths: PathsCfg
    cost: CostCfg
    config_hash: str = ""

    def path(self, name: str) -> Path:
        p = Path(getattr(self.paths, name))
        return p if p.is_absolute() else ROOT / p

    @property
    def database_url(self) -> str:
        url = self.paths.database
        if url.startswith("sqlite:///") and not url.startswith("sqlite:////"):
            rel = url.removeprefix("sqlite:///")
            if not Path(rel).is_absolute():
                return f"sqlite:///{(ROOT / rel).as_posix()}"
        return url


def load_settings(path: str | Path | None = None) -> Settings:
    path = Path(path or os.environ.get("MEDQUIZ_CONFIG") or DEFAULT_CONFIG)
    raw = path.read_bytes()
    data = yaml.safe_load(raw)
    if os.environ.get("OLLAMA_URL"):
        data["ollama_url"] = os.environ["OLLAMA_URL"]
    if os.environ.get("DATABASE_URL"):
        data.setdefault("paths", {})["database"] = os.environ["DATABASE_URL"]
    settings = Settings(**data)
    settings.config_hash = hashlib.sha256(raw).hexdigest()[:12]
    return settings


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return load_settings()


def load_prompt(name: str) -> tuple[str, str]:
    """Return (template text, short hash). Prompts are versioned by file name."""
    text = (PROMPTS_DIR / f"{name}.md").read_text(encoding="utf-8")
    return text, hashlib.sha256(text.encode()).hexdigest()[:12]
