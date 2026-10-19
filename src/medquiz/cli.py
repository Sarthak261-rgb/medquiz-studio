"""Command line entry point: `medquiz <command>`."""

from __future__ import annotations

import json

import typer

from .config import get_settings
from .runs import RunDir, git_sha, new_run_id, utcnow, write_jsonl

app = typer.Typer(add_completion=False, no_args_is_help=True, help=__doc__)


def _settings():
    return get_settings()


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
        import re

        import httpx

        # every page must load, and every paragraph id a chunk cites must exist on
        # that page, so a citation link lands on the exact paragraph
        status, cited, missing = {}, 0, []
        for url in sorted({c.url for c in chunks}):
            resp = httpx.get(url, follow_redirects=True, timeout=60)
            status[url] = resp.status_code
            anchors = set(re.findall(r'id="([^"]+)"', resp.text))
            for c in chunks:
                if c.url == url:
                    cited += len(c.para_ids)
                    missing += [f"{c.id}#{p}" for p in c.para_ids if p not in anchors]
        manifest["url_status"] = status
        manifest["citation_anchors"] = {"cited": cited, "found": cited - len(missing), "missing": missing}
        bad = {u: c for u, c in status.items() if c != 200}
        typer.echo(f"checked {len(status)} page URLs, {len(bad)} not OK {bad if bad else ''}")
        typer.echo(f"citation anchors: {cited - len(missing)} of {cited} paragraph ids found on openstax.org")
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n")
    typer.echo(f"{len(modules)} pages -> {len(chunks)} chunks, {len(book_qs)} book MCQs -> {out}")


@app.command()
def generate(
    run_id: str = typer.Option(None, help="Reuse an id to resume a run"),
    limit: int = typer.Option(None, help="Only the first N chunks"),
    chapter: int = typer.Option(None),
):
    """Generate one cited MCQ per chunk with the local generator model."""
    import time

    from .data import load_chunks
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
    t0 = time.perf_counter()
    made = generate_run(client, s.generator, chunks, run, progress=typer.echo)
    typer.echo(f"done: {len(made)} new questions in {time.perf_counter() - t0:.0f}s -> {run.path}")


def _chunks_by_id(s):
    from .data import load_chunks

    return {c.id: c for c in load_chunks(s.path("data"))}


def _record_checkers(run: RunDir, s) -> None:
    from .llm import OllamaClient

    client = OllamaClient(s.ollama_url)
    run.update_meta(
        judge={**s.judge.model_dump(), "digest": client.digest(s.judge.model)},
        self_judge={**s.self_judge.model_dump(), "digest": client.digest(s.self_judge.model)},
        nli={"model": s.nli.model, "entail_threshold": s.nli.entail_threshold},
        embedding={"model": s.embedding.model},
        checks_config=s.checks.model_dump(),
        checks_config_hash=s.config_hash,
        checks_code_version=git_sha(),
    )


@app.command()
def check(
    run_id: str = typer.Argument(..., help="Folder name under runs/"),
    stages: str = typer.Option("rules,retrieval,nli,judge,selfjudge", help="Comma-separated subset"),
    force: bool = typer.Option(False, help="Re-run these stages for every question (old rows are kept)"),
):
    """Run the automatic checks on a run. Each stage resumes where it stopped."""
    from .checks.runner import run_checks
    from .data import load_questions

    s = _settings()
    run = RunDir(s.path("runs"), run_id)
    questions = load_questions(run.path)
    if not questions:
        raise typer.BadParameter(f"no questions in {run.path}")
    _record_checkers(run, s)
    run_checks(run, questions, _chunks_by_id(s), s, tuple(stages.split(",")), progress=typer.echo, force=force)
    if force:
        run.update_meta(rechecked={st: utcnow() for st in stages.split(",")})


@app.command()
def control(
    run_id: str = typer.Option("book-v1"),
    stages: str = typer.Option("rules,judge"),
):
    """Retrieval eval and the full check suite on the textbook's own review MCQs."""
    from .checks.runner import run_checks
    from .control import book_questions_as_run, retrieval_eval
    from .data import load_book_questions, load_chunks, load_questions
    from .index import Embedder, load_or_build

    s = _settings()
    chunks = load_chunks(s.path("data"))
    book_qs = load_book_questions(s.path("data"))
    run = RunDir(s.path("runs"), run_id)
    if not load_questions(run.path):
        embedder = Embedder(s.embedding)
        index = load_or_build(s.path("data") / "index", chunks, embedder)
        ev = retrieval_eval(book_qs, chunks, index, embedder)
        questions = book_questions_as_run(book_qs, chunks, index, embedder, run.id)
        embedder.close()
        write_jsonl(run.path / "questions.jsonl", [q.model_dump() for q in questions])
        run.update_meta(
            kind="book",
            code_version=git_sha(),
            config_hash=s.config_hash,
            source_commit=s.source.commit,
            retrieval_eval=ev,
        )
        typer.echo(f"retrieval: {json.dumps(ev)}")
    _record_checkers(run, s)
    run_checks(
        run, load_questions(run.path), {c.id: c for c in chunks}, s, tuple(stages.split(",")), progress=typer.echo
    )


@app.command()
def stress(
    from_run: str = typer.Option("gen-v1", help="Generation run whose passing questions get mutated"),
    run_id: str = typer.Option("stress-v1"),
    n: int = typer.Option(30, help="How many parent questions"),
    seed: int = typer.Option(13),
    stages: str = typer.Option("rules,nli,judge"),
):
    """Judge stress test: break passing questions in known ways and re-check them."""
    import random

    from .checks import auto_verdict
    from .checks.runner import applicable, run_checks
    from .data import load_checks, load_chunks, load_questions
    from .index import Embedder, load_or_build
    from .llm import OllamaClient
    from .mutate import make_mutants

    s = _settings()
    chunks = load_chunks(s.path("data"))
    run = RunDir(s.path("runs"), run_id)
    if not load_questions(run.path):
        src = s.path("runs") / from_run
        checks = load_checks(src)
        passing = [
            q
            for q in load_questions(src)
            if auto_verdict(checks.get(q.id, []), s.checks, applicable(q, s.checks.hard)).status == "pass"
        ]
        parents = sorted(random.Random(seed).sample(passing, min(n, len(passing))), key=lambda q: q.chunk_id)
        typer.echo(f"{len(passing)} passing questions in {from_run}; mutating {len(parents)}")
        embedder = Embedder(s.embedding)
        index = load_or_build(s.path("data") / "index", chunks, embedder)
        llm = OllamaClient(s.ollama_url)
        # the paraphrases come from the generator; the judges must not still be holding the GPU
        for model in {s.judge.model, s.self_judge.model} - {s.generator.model}:
            llm.unload(model)
        mutants = make_mutants(parents, chunks, index, embedder, llm, s.generator, run.id, seed=seed)
        embedder.close()
        write_jsonl(run.path / "questions.jsonl", [q.model_dump() for q in mutants])
        run.update_meta(
            kind="stress",
            from_run=from_run,
            n_parents=len(parents),
            seed=seed,
            code_version=git_sha(),
            config_hash=s.config_hash,
        )
    _record_checkers(run, s)
    run_checks(
        run, load_questions(run.path), {c.id: c for c in chunks}, s, tuple(stages.split(",")), progress=typer.echo
    )


img_app = typer.Typer(help="Image track: figures, Stable Diffusion + LoRA, image checks")
app.add_typer(img_app, name="images")
IMAGES_RUN = "images-v1"


def _img_run(s) -> RunDir:
    return RunDir(s.path("runs"), IMAGES_RUN)


def _figures(s):
    from .images.figures import Figure
    from .runs import read_jsonl

    rows = read_jsonl(s.path("data") / "figures" / "figures.jsonl")
    if not rows:
        raise typer.BadParameter("no figures yet; run `medquiz images figures` first")
    return [Figure(**r) for r in rows]


def _free_gpu(s) -> None:
    from .llm import OllamaClient

    client = OllamaClient(s.ollama_url)
    for model in {s.generator.model, s.judge.model, s.self_judge.model, s.images.vlm}:
        client.unload(model)


@img_app.command("figures")
def images_figures():
    """Download the OpenStax figures for the image chapters and pad them to 512 px."""
    from .images.figures import ingest_figures

    s = _settings()
    out = s.path("data") / "figures"
    kept, dropped = ingest_figures(s.source, s.images, s.path("data") / "raw", out)
    write_jsonl(out / "figures.jsonl", [f.model_dump() for f in kept])
    write_jsonl(out / "excluded.jsonl", dropped)
    typer.echo(f"{len(kept)} figures kept, {len(dropped)} excluded -> {out}")


@img_app.command("generate")
def images_generate(arm: str = typer.Option(..., help="base or lora")):
    """Generate the prompt x seed grid for one arm. Resumable."""
    from .images.generate import generate_arm

    s = _settings()
    run = _img_run(s)
    lora_dir = run.path / "lora" if arm == "lora" else None
    if lora_dir is not None and not (lora_dir / "pytorch_lora_weights.safetensors").exists():
        raise typer.BadParameter("train the LoRA first: `medquiz images train`")
    run.update_meta(kind="images", code_version=git_sha(), config_hash=s.config_hash, images=s.images.model_dump())
    _free_gpu(s)
    n = generate_arm(s.images, run, arm, lora_dir, progress=typer.echo)
    typer.echo(f"{arm}: {n} new images in {run.path / 'images'}")


@img_app.command("train")
def images_train():
    """LoRA fine-tune of SD 1.5 on the OpenStax figures."""
    from .images.lora import train_lora

    s = _settings()
    run = _img_run(s)
    out = run.path / "lora"
    if (out / "pytorch_lora_weights.safetensors").exists():
        typer.echo("LoRA already trained; delete runs/images-v1/lora to retrain")
        return
    _free_gpu(s)
    meta = train_lora(s.images, _figures(s), s.path("data") / "figures", run, out, progress=typer.echo)
    run.update_meta(lora=meta, code_version=git_sha())
    typer.echo(f"LoRA saved to {out}")


@img_app.command("check")
def images_check():
    """CLIP score and a vision-language model check for every image. Resumable."""
    import httpx

    from .energy import StageTracker
    from .images.checks import ClipScorer, auto_verdict, vlm_check

    s = _settings()
    run = _img_run(s)
    images = run.read("images.jsonl")
    done = {r["id"] for r in run.read("image_checks.jsonl")}
    todo = [r for r in images if r["id"] not in done]
    if not todo:
        typer.echo("nothing to do")
        return
    _free_gpu(s)
    clip = ClipScorer(s.images.clip_model)
    scores = {r["id"]: clip.score(run.path / r["file"], r["subject"]) for r in todo}
    clip.close()
    client = httpx.Client(base_url=s.ollama_url)
    with StageTracker(run, "check_images") as tracker:
        for n, r in enumerate(todo, 1):
            vlm, st = vlm_check(client, s.images.vlm, run.path / r["file"], r["subject"])
            verdict, reasons = auto_verdict(vlm)
            run.append("image_checks.jsonl", {"id": r["id"], "clip_score": scores[r["id"]], "vlm": vlm,
                                              "auto": verdict, "auto_reasons": reasons, **st})  # fmt: skip
            tracker.tick()
            typer.echo(f"  [{n}/{len(todo)}] {r['id']} clip={scores[r['id']]} {verdict} {reasons}")
    run.update_meta(image_checks={"vlm": s.images.vlm, "clip": s.images.clip_model})


db_app = typer.Typer(help="Database maintenance")
app.add_typer(db_app, name="db")


def _rebuild(s):
    from .data import load_chunks
    from .db import Base
    from .store import Store

    store = Store(s.database_url, review_log_dir=s.path("data") / "reviews")
    Base.metadata.drop_all(store.engine)
    Base.metadata.create_all(store.engine)
    store.load_chunks(load_chunks(s.path("data")))
    n = 0
    runs_dir = s.path("runs")
    if runs_dir.exists():
        for run_path in sorted(p for p in runs_dir.iterdir() if (p / "questions.jsonl").exists()):
            n += store.load_run(run_path, s.checks)
    reviews = store.replay_reviews(s.path("data") / "reviews")
    return store, n, reviews


@db_app.command("rebuild")
def db_rebuild():
    """Recreate the database from runs/ and data/reviews/. Safe: reviews live in the JSONL logs."""
    s = _settings()
    _, n, reviews = _rebuild(s)
    typer.echo(f"loaded {n} questions and {reviews} reviews into {s.database_url}")


@app.command()
def report(
    run_id: str = typer.Option(None, help="Generation run (default: latest)"),
    reviewer: str = typer.Option(None),
):
    """Recompute every reported number into reports/metrics.json and reports/results.md."""
    from .report import compute, write

    s = _settings()
    store, _, _ = _rebuild(s)
    m = compute(store.snapshot(), s, gen_run=run_id, reviewer=reviewer)
    jp, mp = write(m, s.path("reports"))
    try:
        from .figures import make_figures
    except ImportError:
        make_figures = None
    if make_figures:
        for f in make_figures(m, s.path("reports") / "figures", settings=s):
            typer.echo(f"figure: {f}")
    typer.echo(f"wrote {jp} and {mp}")


@app.command()
def release(
    version: str = typer.Argument(..., help="Semantic version, e.g. 0.1.0"),
    run_id: str = typer.Option("gen-v1"),
    reviewer: str = typer.Option(None),
    notes: str = typer.Option(""),
):
    """Write an immutable, hashed dataset release of the human-accepted questions."""
    from .release import build_release
    from .report import compute

    s = _settings()
    store, _, _ = _rebuild(s)
    snap = store.snapshot()
    m = compute(snap, s, gen_run=run_id, reviewer=reviewer)
    out = build_release(snap, s, version, run_id, m, reviewer=reviewer, notes=notes)
    typer.echo(f"release written to {out}")


@app.command("verify-release")
def verify_release_cmd(version: str):
    """Check a release's files against the hashes in its manifest."""
    from .release import verify_release

    folder = _settings().path("releases") / f"v{version}"
    bad = verify_release(folder)
    if bad:
        typer.echo(f"MISMATCH: {bad}")
        raise typer.Exit(1)
    typer.echo(f"{folder}: all hashes match")


@app.command()
def serve(host: str = "127.0.0.1", port: int = 8000):
    """Start the HTTP API."""
    import uvicorn

    uvicorn.run("medquiz.api:app", host=host, port=port)


@app.command()
def review(port: int = 8501):
    """Open the review screen."""
    import subprocess
    import sys

    from .config import ROOT

    app_path = ROOT / "review_app" / "app.py"
    raise typer.Exit(
        subprocess.call([sys.executable, "-m", "streamlit", "run", str(app_path), "--server.port", str(port)])
    )


if __name__ == "__main__":
    app()
