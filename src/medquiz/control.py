"""The control set: the 57 review MCQs the OpenStax authors wrote for these chapters.

They have known keys and were written by people, so they let us
  1. measure retrieval (does the question find its own section?), and
  2. measure how often the automatic checks reject a question that a textbook
     author was happy to publish. That is the checks' false-alarm floor.
"""

from __future__ import annotations

import numpy as np

from .index import ChunkIndex, Embedder
from .schemas import BookQuestion, Chunk, GenMeta, Question


def retrieval_eval(
    book_qs: list[BookQuestion], chunks: list[Chunk], index: ChunkIndex, embedder: Embedder, ks=(1, 3, 5)
) -> dict:
    section = {c.id: c.section for c in chunks}
    out = {}
    for label, texts in {
        "stem_only": [q.stem for q in book_qs],
        "stem_plus_answer": [f"{q.stem} {q.options[q.answer_index]}" for q in book_qs],
    }.items():
        hits = index.search_vectors(embedder.encode(texts, query=True), k=max(ks))
        out[label] = {
            f"section_recall@{k}": round(
                float(
                    np.mean(
                        [any(section[c] == q.section for c, _ in h[:k]) for q, h in zip(book_qs, hits, strict=True)]
                    )
                ),
                4,
            )
            for k in ks
        }
    out["n"] = len(book_qs)
    return out


def book_questions_as_run(
    book_qs: list[BookQuestion], chunks: list[Chunk], index: ChunkIndex, embedder: Embedder, run_id: str
) -> list[Question]:
    """Attach each book question to the best-matching chunk of its own section,
    which is the passage the judge will see."""
    qvecs = embedder.encode([f"{q.stem} {q.options[q.answer_index]}" for q in book_qs], query=True)
    out = []
    for q, v in zip(book_qs, qvecs, strict=True):
        candidates = [c for c in chunks if c.section == q.section] or chunks
        scores = [float(index.vector(c.id) @ v) for c in candidates]
        best = candidates[int(np.argmax(scores))]
        out.append(
            Question(
                id=q.id,
                run_id=run_id,
                chunk_id=best.id,
                origin="book",
                stem=q.stem,
                options=q.options,
                answer_index=q.answer_index,
                gen=GenMeta(model="openstax-authors"),
            )
        )
    return out
