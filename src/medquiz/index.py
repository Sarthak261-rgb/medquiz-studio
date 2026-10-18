"""Dense retrieval over chunks: bge-small embeddings on the GPU, exact inner-product
search in FAISS. Used for citation checks, duplicate detection, the retrieval eval
on the book's own questions, and picking passages for the judge stress test."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .config import EmbedCfg
from .schemas import Chunk


class Embedder:
    def __init__(self, cfg: EmbedCfg):
        self.cfg = cfg
        self._model = None

    @property
    def model(self):
        if self._model is None:
            import torch
            from sentence_transformers import SentenceTransformer

            device = self.cfg.device if torch.cuda.is_available() else "cpu"
            self._model = SentenceTransformer(self.cfg.model, device=device)
        return self._model

    def encode(self, texts: list[str], query: bool = False) -> np.ndarray:
        if query and self.cfg.query_prefix:
            texts = [self.cfg.query_prefix + t for t in texts]
        vecs = self.model.encode(
            texts,
            batch_size=self.cfg.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return vecs.astype("float32")

    def close(self) -> None:
        if self._model is not None:
            import torch

            del self._model
            self._model = None
            torch.cuda.empty_cache()


class _NumpyIP:
    def __init__(self, vectors: np.ndarray):
        self.vectors = vectors

    def search(self, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        scores = queries @ self.vectors.T
        k = min(k, self.vectors.shape[0])
        idx = np.argsort(-scores, axis=1)[:, :k]
        return np.take_along_axis(scores, idx, axis=1), idx


def corpus_hash(chunks: list[Chunk]) -> str:
    h = hashlib.sha256()
    for c in chunks:
        h.update(c.id.encode())
        h.update(c.text.encode())
    return h.hexdigest()[:12]


class ChunkIndex:
    def __init__(self, ids: list[str], vectors: np.ndarray, model: str, corpus: str):
        self.ids = ids
        self.model = model
        self.corpus = corpus
        self.vectors = vectors
        self._pos = {cid: i for i, cid in enumerate(ids)}
        try:
            import faiss

            self.index = faiss.IndexFlatIP(vectors.shape[1])
            self.index.add(vectors)
        except ImportError:
            # 151 chunks: brute force in numpy gives the same answer, just without faiss
            self.index = _NumpyIP(vectors)

    @classmethod
    def build(cls, chunks: list[Chunk], embedder: Embedder) -> ChunkIndex:
        passages = [f"{c.heading}\n{c.text}" for c in chunks]
        vecs = embedder.encode(passages)
        return cls([c.id for c in chunks], vecs, embedder.cfg.model, corpus_hash(chunks))

    def save(self, folder: Path) -> None:
        folder.mkdir(parents=True, exist_ok=True)
        np.save(folder / "vectors.npy", self.vectors)
        meta = {"ids": self.ids, "model": self.model, "corpus": self.corpus}
        (folder / "meta.json").write_text(json.dumps(meta), encoding="utf-8", newline="\n")

    @classmethod
    def load(cls, folder: Path) -> ChunkIndex:
        meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
        return cls(meta["ids"], np.load(folder / "vectors.npy"), meta["model"], meta["corpus"])

    def search_vectors(self, queries: np.ndarray, k: int) -> list[list[tuple[str, float]]]:
        scores, idx = self.index.search(queries, k)
        return [
            [(self.ids[j], float(s)) for j, s in zip(row_i, row_s, strict=True) if j >= 0]
            for row_i, row_s in zip(idx, scores, strict=True)
        ]

    def vector(self, chunk_id: str) -> np.ndarray:
        return self.vectors[self._pos[chunk_id]]


def load_or_build(folder: Path, chunks: list[Chunk], embedder: Embedder) -> ChunkIndex:
    """Rebuild whenever the chunks or the embedding model changed."""
    if (folder / "meta.json").exists():
        idx = ChunkIndex.load(folder)
        if idx.corpus == corpus_hash(chunks) and idx.model == embedder.cfg.model:
            return idx
    idx = ChunkIndex.build(chunks, embedder)
    idx.save(folder)
    return idx
