"""Command line entry point: `medquiz <command>`."""

from __future__ import annotations

import json
from pathlib import Path

import typer

from .config import get_settings
from .runs import RunDir, git_sha, new_run_id, utcnow, write_jsonl

app = typer.Typer(add_completion=False, no_args_is_help=True, help=__doc__)


def _settings():
    return get_settings()


def _accumulate(prev: dict, meter, **extra) -> dict:
    """Stage totals add up across resumed invocations of the same run."""
    joules = prev.get("gpu_joules")
    if meter.joules is not None:
        joules = round((joules or 0) + meter.joules, 1)
    return {
        **extra,
        "finished_at": utcnow(),
        "seconds": round(prev.get("seconds", 0) + meter.seconds, 1),
        "gpu_joules": joules,
        "gpu": meter.gpu,
    }


@app.command()
def ingest(check_urls: bool = typer.Option(False, help="HEAD every cited page on openstax.org")):
    """Fetch the pinned OpenStax chapters, chunk them and extract the book's own MCQs."""
    from .ingest import ingest as run_ingest

    s = _settings()
    data = s.path("data")
    modules, chunks, book_qs = run_ingest(s.source, s.chunking, data / "raw")
    out = data / "processed"
    write_jsonl(out / "chunks.jsonl", [c.model_dump() for c in chunks])
    write_jsonl(out / "book_questions.jsonl", [q.model_dump() for q in book_qs])
    pages = [
        {
            "section": m.section,
            "module_id": m.module_id,
            "title": m.title,
            "objectives": m.objectives,
            "n_blocks": len(m.blocks),
            "n_book_questions": len(m.questions),
        }
        for m in modules
    ]
    manifest = {
        "source_repo": s.source.repo,
        "source_commit": s.source.commit,
        "license": s.source.license,
        "created_at": utcnow(),
        "chunking": s.chunking.model_dump(),
        "n_chunks": len(chunks),
        "n_words": sum(c.n_words for c in chunks),
        "n_book_questions": len(book_qs),
        "pages": pages,
    }
    if check_urls:
        import httpx

        status = {}
        for url in sorted({c.url for c in chunks}):
            status[url] = httpx.head(url, follow_redirects=True, timeout=30).status_code
        manifest["url_status"] = status
        bad = {u: c for u, c in status.items() if c != 200}
        typer.echo(f"checked {len(status)} page URLs, {len(bad)} not OK {bad if bad else ''}")
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    typer.echo(f"{len(modules)} pages -> {len(chunks)} chunks, {len(book_qs)} book MCQs -> {out}")


@app.command()
def generate(
    run_id: str = typer.Option(None, help="Reuse an id to resume a run"),
    limit: int = typer.Option(None, help="Only the first N chunks"),
    chapter: int = typer.Option(None),
):
    """Generate one cited MCQ per chunk with the local generator model."""
    from .data import load_chunks
    from .energy import EnergyMeter
    from .generate import generate_run
    from .llm import OllamaClient

    s = _settings()
    chunks = load_chunks(s.path("data"))
    if chapter:
        chunks = [c for c in chunks if c.chapter == chapter]
    if limit:
        chunks = chunks[:limit]
    run = RunDir(s.path("runs"), run_id or new_run_id("gen"))
    client = OllamaClient(s.ollama_url)
    run.update_meta(
        kind="generate",
        code_version=git_sha(),
        config_hash=s.config_hash,
        source_commit=s.source.commit,
        generator={**s.generator.model_dump(), "digest": client.digest(s.generator.model)},
    )
    typer.echo(f"run {run.id}: {len(chunks)} chunks")
    with EnergyMeter() as meter:
        made = generate_run(client, s.generator, chunks, run, progress=typer.echo)
    prev = run.meta().get("stages", {}).get("generate", {})
    run.update_meta(stages={"generate": _accumulate(prev, meter, n_new=len(made))})
    typer.echo(f"done: {len(made)} new questions in {meter.seconds:.0f}s -> {run.path}")


if __name__ == "__main__":
    app()
