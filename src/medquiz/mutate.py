"""Judge stress test. Take questions the checks passed, break them on purpose in one
known way, and see whether the checks notice. Unlike human review, the label is
known by construction, so this measures the judge itself.

  swap_key            the marked answer is moved to a distractor
  wrong_passage       question shown with a passage from the other chapter
  near_miss_passage   same chapter, on-topic passage that does not contain the fact
  second_correct      one distractor replaced by a paraphrase of the key
  absurd_distractors  all three distractors replaced by off-topic options
  reorder             options reversed; nothing is wrong, the verdict should not move
"""

from __future__ import annotations

import random
import re

from pydantic import BaseModel

from .config import GeneratorCfg
from .index import ChunkIndex, Embedder
from .llm import JSONChat
from .prompts import render
from .schemas import Chunk, Question

MUTATIONS = [
    "swap_key",
    "wrong_passage",
    "near_miss_passage",
    "second_correct",
    "absurd_distractors",
    "reorder",
]

# Checks that should fail for each mutation. "reorder" expects no change.
EXPECTED = {
    "swap_key": ["judge_blind_solve", "judge_answer_grounded", "nli_answer_entailed"],
    "wrong_passage": ["judge_answer_grounded", "nli_answer_entailed", "evidence_in_source"],
    "near_miss_passage": ["judge_answer_grounded", "nli_answer_entailed", "evidence_in_source"],
    "second_correct": ["judge_single_correct", "judge_blind_solve", "options_distinct", "nli_distractors_not_entailed"],
    "absurd_distractors": ["judge_distractor_quality"],
    "reorder": [],
}

ABSURD = [
    "the boiling point of seawater",
    "a sonnet's rhyme scheme",
    "photosynthesis in maize leaves",
    "the Treaty of Versailles",
    "tectonic plate subduction",
    "a binary search tree",
    "the price elasticity of demand",
    "Saturn's largest moon",
    "the offside rule in football",
    "a minor seventh chord",
    "igneous rock crystallization",
    "the Pythagorean theorem",
    "compound interest on savings",
    "the migration of monarch butterflies",
    "a haiku's syllable count",
    "the speed of sound in steel",
    "Gothic cathedral flying buttresses",
    "the rules of chess castling",
]


class Paraphrase(BaseModel):
    rewritten: str


def _replace_key(statement: str, old: str, new: str) -> str:
    pat = re.compile(re.escape(old.strip().rstrip(".")), re.I)
    return pat.sub(new.strip().rstrip("."), statement, count=1) if pat.search(statement) else ""


def _child(q: Question, mutation: str, run_id: str, **changes) -> Question:
    data = q.model_dump()
    data.update(
        id=f"m_{mutation}_{q.id}",
        run_id=run_id,
        origin="mutant",
        mutation=mutation,
        parent_id=q.id,
        **changes,
    )
    return Question(**data)


def make_mutants(
    parents: list[Question],
    chunks: list[Chunk],
    index: ChunkIndex,
    embedder: Embedder,
    llm: JSONChat,
    gen_cfg: GeneratorCfg,
    run_id: str,
    seed: int = 13,
) -> list[Question]:
    rng = random.Random(seed)
    by_id = {c.id: c for c in chunks}
    qvecs = embedder.encode([f"{q.stem} {q.answer}" for q in parents], query=True)
    hits = index.search_vectors(qvecs, k=10)
    out: list[Question] = []
    for q, qv, hit in zip(parents, qvecs, hits, strict=True):
        home = by_id[q.chunk_id]

        # swap_key
        new_key = (q.answer_index + 1 + rng.randrange(3)) % 4
        out.append(
            _child(
                q,
                "swap_key",
                run_id,
                answer_index=new_key,
                answer_statement=_replace_key(q.answer_statement, q.answer, q.options[new_key]),
            )
        )

        # wrong_passage: any chunk from another chapter
        far = [c for c in chunks if c.chapter != home.chapter]
        out.append(_child(q, "wrong_passage", run_id, chunk_id=rng.choice(far).id))

        # near_miss_passage: most similar same-chapter chunk outside the question's
        # own section and outside its top-10 retrieval hits
        top = {cid for cid, _ in hit}
        near = [c for c in chunks if c.chapter == home.chapter and c.section != home.section and c.id not in top]
        if near:
            best = max(near, key=lambda c: float(index.vector(c.id) @ qv))
            out.append(_child(q, "near_miss_passage", run_id, chunk_id=best.id))

        # second_correct: a paraphrase of the key replaces one distractor
        system, user, _ = render("paraphrase_v1", stem=q.stem, option=q.answer)
        para, _ = llm.chat_json(
            gen_cfg.model, system, user, Paraphrase, temperature=0.3, seed=gen_cfg.seed, num_gpu=gen_cfg.num_gpu
        )
        slot = rng.choice([i for i in range(4) if i != q.answer_index])
        opts = list(q.options)
        opts[slot] = para.rewritten.strip()
        out.append(_child(q, "second_correct", run_id, options=opts))

        # absurd_distractors
        junk = iter(rng.sample(ABSURD, 3))
        opts = [o if i == q.answer_index else next(junk) for i, o in enumerate(q.options)]
        out.append(_child(q, "absurd_distractors", run_id, options=opts))

        # reorder
        out.append(_child(q, "reorder", run_id, options=list(reversed(q.options)), answer_index=3 - q.answer_index))
    return out
