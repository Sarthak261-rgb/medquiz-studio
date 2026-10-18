"""Runs the check stages over a run folder. Each stage is resumable and keeps a
running record of its own time and GPU energy in run.json."""

from __future__ import annotations

from collections.abc import Callable

from ..config import Settings
from ..energy import StageTracker
from ..runs import RunDir
from ..schemas import CheckResult, Chunk, Question
from . import severity

STAGES = ("rules", "retrieval", "nli", "judge", "selfjudge")

NOT_FOR_BOOK = {
    "duplicate",
    "citation_retrieval",
    "book_overlap",
    "evidence_in_source",
    "nli_answer_entailed",
    "nli_distractors_not_entailed",
    "statement_consistent",
}
NOT_FOR_MUTANT = {"duplicate", "citation_retrieval", "book_overlap"}


def applicable(q: Question, names: list[str]) -> set[str]:
    skip = NOT_FOR_BOOK if q.origin == "book" else NOT_FOR_MUTANT if q.origin == "mutant" else set()
    return {n for n in names if n not in skip}


def stages_for(q: Question, stages: tuple[str, ...]) -> tuple[str, ...]:
    """Self-judging only makes sense for the generator's own questions."""
    return tuple(s for s in stages if s != "selfjudge" or q.origin == "generated")


def run_checks(
    run: RunDir,
    questions: list[Question],
    chunks: dict[str, Chunk],
    settings: Settings,
    stages: tuple[str, ...] = STAGES,
    progress: Callable[[str], None] = print,
) -> None:
    done = {(r["question_id"], r["stage"]) for r in run.read("checks.jsonl")}
    ccfg = settings.checks

    def write(q: Question, stage: str, results: list[CheckResult]) -> None:
        allowed = applicable(q, [r.name for r in results])
        kept = []
        for r in results:
            if r.name in allowed:
                r.severity = severity(r.name, ccfg)
                kept.append(r.model_dump())
        run.append("checks.jsonl", {"question_id": q.id, "stage": stage, "checks": kept})

    for stage in stages:
        todo = [q for q in questions if (q.id, stage) not in done and stage in stages_for(q, stages)]
        if not todo:
            progress(f"{stage}: nothing to do")
            continue
        progress(f"{stage}: {len(todo)} questions")
        with StageTracker(run, f"check_{stage}") as tracker:
            if stage == "rules":
                from .rules import run_rules

                for q in todo:
                    write(q, stage, run_rules(q, chunks[q.chunk_id], ccfg.length_cue_ratio, ccfg.quote_match_threshold))
                    tracker.tick()

            elif stage in ("retrieval", "nli"):
                from ..llm import OllamaClient

                # torch and Ollama share 6 GB of VRAM; make room first
                ollama = OllamaClient(settings.ollama_url)
                for model in {settings.generator.model, settings.judge.model, settings.self_judge.model}:
                    ollama.unload(model)
                if stage == "retrieval":
                    from ..data import load_book_questions
                    from ..index import Embedder, load_or_build
                    from .retrieval import retrieval_checks

                    embedder = Embedder(settings.embedding)
                    index = load_or_build(settings.path("data") / "index", list(chunks.values()), embedder)
                    book = load_book_questions(settings.path("data"))
                    # all questions go in, so duplicates are judged against the whole run
                    res = retrieval_checks(
                        questions,
                        index,
                        embedder,
                        ccfg.retrieval_top_k,
                        ccfg.duplicate_cosine,
                        book_texts=[f"{b.stem} {b.options[b.answer_index]}" for b in book],
                        book_ids=[b.id for b in book],
                    )
                    for q in todo:
                        write(q, stage, res.get(q.id, []))
                    tracker.tick(len(todo))
                    embedder.close()
                else:
                    from .nli import NLIScorer, nli_checks

                    scorer = NLIScorer(settings.nli)
                    for q in todo:
                        write(q, stage, nli_checks(q, chunks[q.chunk_id], scorer))
                        tracker.tick()
                    scorer.close()

            elif stage in ("judge", "selfjudge"):
                from ..llm import OllamaClient
                from .judge import judge_checks, self_judge_checks

                cfg, fn = (
                    (settings.judge, judge_checks) if stage == "judge" else (settings.self_judge, self_judge_checks)
                )
                client = OllamaClient(settings.ollama_url)
                for model in {settings.generator.model, settings.judge.model} - {cfg.model}:
                    client.unload(model)
                for n, q in enumerate(todo, 1):
                    results = fn(client, cfg, q, chunks[q.chunk_id].text)
                    write(q, stage, results)
                    tracker.tick()
                    flags = " ".join(r.name.split("_", 1)[1] for r in results if r.passed is False)
                    progress(f"  [{n}/{len(todo)}] {q.id} {sum(r.seconds for r in results):5.1f}s {flags}")
            else:
                raise ValueError(f"unknown stage {stage}")
