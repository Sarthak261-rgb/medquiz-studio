"""Embedding-based checks: does the question point back at its own chunk, is it a
near copy of an earlier question in the same run, and is it a near copy of one of
the textbook's own review questions?"""

from __future__ import annotations

from ..index import ChunkIndex, Embedder
from ..schemas import CheckResult, Question


def retrieval_checks(
    questions: list[Question],
    index: ChunkIndex,
    embedder: Embedder,
    top_k: int,
    dup_cosine: float,
    book_texts: list[str] | None = None,
    book_ids: list[str] | None = None,
) -> dict[str, list[CheckResult]]:
    if not questions:
        return {}
    texts = [f"{q.stem} {q.answer}" for q in questions]
    hits = index.search_vectors(embedder.encode(texts, query=True), k=max(top_k, 10))
    sym = embedder.encode(texts)
    sims = sym @ sym.T
    book_sims = sym @ embedder.encode(book_texts).T if book_texts else None
    out: dict[str, list[CheckResult]] = {}
    for i, q in enumerate(questions):
        if q.origin != "generated":
            out[q.id] = []
            continue
        ids = [cid for cid, _ in hits[i]]
        rank = ids.index(q.chunk_id) + 1 if q.chunk_id in ids else None
        results = [
            CheckResult(
                name="citation_retrieval",
                checker="retrieval",
                passed=rank is not None and rank <= top_k,
                score=rank,
                detail={"top": ids[:top_k]},
            )
        ]
        # only earlier questions count, so the first of a pair survives
        earlier = [
            (j, float(sims[i, j]))
            for j in range(i)
            if questions[j].origin == "generated" and questions[j].run_id == q.run_id
        ]
        j, best = max(earlier, key=lambda t: t[1]) if earlier else (None, 0.0)
        results.append(
            CheckResult(
                name="duplicate",
                checker="retrieval",
                passed=best < dup_cosine,
                score=round(best, 4),
                detail={"nearest": questions[j].id if j is not None else None},
            )
        )
        if book_sims is not None:
            b = int(book_sims[i].argmax())
            results.append(
                CheckResult(
                    name="book_overlap",
                    checker="retrieval",
                    passed=float(book_sims[i, b]) < dup_cosine,
                    score=round(float(book_sims[i, b]), 4),
                    detail={"nearest_book_question": book_ids[b] if book_ids else b},
                )
            )
        out[q.id] = results
    return out
