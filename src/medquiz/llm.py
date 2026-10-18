"""Thin Ollama client. Every call asks for JSON that matches a Pydantic schema
(Ollama turns the schema into a grammar), validates it, and records tokens and time."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Protocol, TypeVar

import httpx
from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)


@dataclass
class CallStats:
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    seconds: float = 0.0
    attempts: int = 0
    errors: list[str] = field(default_factory=list)

    @property
    def tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class LLMError(RuntimeError):
    pass


class JSONChat(Protocol):
    def chat_json(
        self,
        model: str,
        system: str,
        user: str,
        schema: type[T],
        *,
        temperature: float = 0.0,
        seed: int | None = None,
        num_ctx: int = 4096,
        max_attempts: int = 3,
    ) -> tuple[T, CallStats]: ...


class OllamaClient:
    def __init__(self, base_url: str = "http://localhost:11434", timeout: float = 600):
        self.base_url = base_url.rstrip("/")
        self.http = httpx.Client(base_url=self.base_url, timeout=timeout)

    def chat_json(
        self,
        model: str,
        system: str,
        user: str,
        schema: type[T],
        *,
        temperature: float = 0.0,
        seed: int | None = None,
        num_ctx: int = 4096,
        max_attempts: int = 3,
    ) -> tuple[T, CallStats]:
        stats = CallStats(model=model)
        for attempt in range(max_attempts):
            stats.attempts += 1
            body = {
                "model": model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "format": schema.model_json_schema(),
                "stream": False,
                "keep_alive": "15m",
                "options": {
                    "temperature": temperature,
                    "num_ctx": num_ctx,
                    # a new seed per retry, otherwise a bad sample repeats exactly
                    **({"seed": seed + attempt} if seed is not None else {}),
                },
            }
            t0 = time.perf_counter()
            resp = self.http.post("/api/chat", json=body)
            stats.seconds += time.perf_counter() - t0
            if resp.status_code != 200:
                stats.errors.append(f"http {resp.status_code}: {resp.text[:200]}")
                continue
            data = resp.json()
            stats.prompt_tokens += data.get("prompt_eval_count", 0)
            stats.completion_tokens += data.get("eval_count", 0)
            content = data.get("message", {}).get("content", "")
            try:
                return schema.model_validate_json(content), stats
            except ValidationError as e:
                stats.errors.append(f"invalid output: {e.errors()[0]['msg']}")
        raise LLMError(f"{model}: no valid output after {max_attempts} attempts: {stats.errors}")

    def digest(self, model: str) -> str:
        """Content digest of the local model, so a run records exactly which weights it used."""
        resp = self.http.get("/api/tags")
        resp.raise_for_status()
        for m in resp.json().get("models", []):
            if m.get("name") == model or m.get("model") == model:
                return m.get("digest", "")[:12]
        return ""

    def unload(self, model: str) -> None:
        """Free VRAM before another model (or torch) needs the GPU."""
        try:
            self.http.post("/api/generate", json={"model": model, "keep_alive": 0}, timeout=60)
        except httpx.HTTPError:
            pass

    def available(self) -> bool:
        try:
            return self.http.get("/api/version", timeout=3).status_code == 200
        except httpx.HTTPError:
            return False
