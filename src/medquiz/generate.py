"""Stage 2: one passage in, one cited MCQ out."""

from __future__ import annotations

import hashlib
import random
from collections.abc import Callable

from .config import GeneratorCfg
from .energy import StageTracker, gpu_counter, joules_since
from .llm import JSONChat, LLMError
from .prompts import render
from .runs import RunDir, utcnow
from .schemas import Chunk, GenMeta, MCQDraft, Question

GUIDANCE = {
    "recall": "It should test a specific fact, term or definition that the passage states directly.",
    "reasoning": (
        "It should need one step of reasoning: predict a consequence, apply a mechanism to a "
        "short scenario, or compare two structures or processes. The answer must still follow "
        "from the passage."
    ),
}


def question_id(run_id: str, chunk_id: str, k: int) -> str:
    return "q_" + hashlib.sha1(f"{run_id}|{chunk_id}|{k}".encode()).hexdigest()[:10]


def target_for(index: int) -> str:
    """Alternate recall and reasoning so the set is balanced by design."""
    return "recall" if index % 2 == 0 else "reasoning"


def shuffle_options(options: list[str], answer_index: int, key: str) -> tuple[list[str], int]:
    """Models like to put the key in position A or B. A seeded shuffle removes that
    bias and keeps the run reproducible."""
    perm = list(range(len(options)))
    random.Random(key).shuffle(perm)
    return [options[p] for p in perm], perm.index(answer_index)


def generate_question(client: JSONChat, cfg: GeneratorCfg, chunk: Chunk, run_id: str, k: int, target: str) -> Question:
    qid = question_id(run_id, chunk.id, k)
    seed = cfg.seed + int(hashlib.sha1(qid.encode()).hexdigest()[:6], 16) % 10_000
    system, user, phash = render(
        cfg.prompt,
        section_title=chunk.section_title,
        heading=chunk.heading,
        passage=chunk.text,
        target_type=target,
        guidance=GUIDANCE[target],
    )
    draft, stats = client.chat_json(
        cfg.model,
        system,
        user,
        MCQDraft,
        temperature=cfg.temperature,
        seed=seed,
        num_ctx=cfg.num_ctx,
        num_gpu=cfg.num_gpu,
    )
    options, answer = shuffle_options(draft.options, draft.answer_index, qid)
    return Question(
        id=qid,
        run_id=run_id,
        chunk_id=chunk.id,
        origin="generated",
        target_type=target,
        stem=draft.stem.strip(),
        options=options,
        answer_index=answer,
        explanation=draft.explanation.strip(),
        evidence_quote=draft.evidence_quote.strip(),
        answer_statement=draft.answer_statement.strip(),
        bloom_level=draft.bloom_level.value,
        gen=GenMeta(
            model=cfg.model,
            prompt=cfg.prompt,
            prompt_hash=phash,
            seed=seed,
            prompt_tokens=stats.prompt_tokens,
            completion_tokens=stats.completion_tokens,
            seconds=round(stats.seconds, 3),
            attempts=stats.attempts,
            original_answer_index=draft.answer_index,
        ),
    )


def generate_run(
    client: JSONChat,
    cfg: GeneratorCfg,
    chunks: list[Chunk],
    run: RunDir,
    progress: Callable[[str], None] = print,
) -> list[Question]:
    """Resumable: questions already in questions.jsonl are skipped."""
    done = {r["id"] for r in run.read("questions.jsonl")}
    made = []
    tracker = StageTracker(run, "generate")
    for i, chunk in enumerate(chunks):
        for k in range(cfg.questions_per_chunk):
            qid = question_id(run.id, chunk.id, k)
            if qid in done:
                continue
            target = target_for(i + k)
            start = gpu_counter().millijoules()
            try:
                q = generate_question(client, cfg, chunk, run.id, k, target)
            except LLMError as e:
                run.append("errors.jsonl", {"chunk_id": chunk.id, "k": k, "error": str(e), "at": utcnow()})
                progress(f"[{i + 1}/{len(chunks)}] {chunk.id} failed: {e}")
                continue
            q.gen.gpu_joules = joules_since(start)
            run.append("questions.jsonl", q.model_dump())
            tracker.tick()
            made.append(q)
            progress(f"[{i + 1}/{len(chunks)}] {chunk.id} {target:9s} {q.gen.seconds:5.1f}s  {q.stem[:70]}")
    tracker.flush()
    return made
