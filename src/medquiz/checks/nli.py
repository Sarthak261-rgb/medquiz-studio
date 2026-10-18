"""A non-LLM grounding check: does the passage entail the answer statement?

The answer statement is one sentence the generator writes alongside the question.
Swapping the key's words for a distractor's gives one statement per distractor,
and none of those should be entailed either."""

from __future__ import annotations

import re

import numpy as np

from ..config import NLICfg
from ..ingest import split_sentences
from ..schemas import CheckResult, Chunk, Question
from .rules import STOPWORDS, norm


class NLIScorer:
    def __init__(self, cfg: NLICfg):
        self.cfg = cfg
        self._model = None
        self._entail = 1

    @property
    def model(self):
        if self._model is None:
            import torch
            from sentence_transformers import CrossEncoder

            device = self.cfg.device if torch.cuda.is_available() else "cpu"
            self._model = CrossEncoder(self.cfg.model, device=device)
            id2label = self._model.model.config.id2label
            self._entail = next(i for i, lab in id2label.items() if lab.lower().startswith("entail"))
        return self._model

    def windows(self, passage: str) -> list[str]:
        """Every run of 1..n consecutive sentences. Short premises matter: given a long
        window, this model calls an exact restatement "neutral" surprisingly often."""
        sents = split_sentences(passage.replace("\n", " "))
        n = self.cfg.window_sentences
        return [" ".join(sents[i : i + k]) for k in range(1, n + 1) for i in range(len(sents) - k + 1)] or [passage]

    def max_entailment(self, passage: str, hypotheses: list[str], must_mention: list[str] | None = None) -> list[float]:
        """Highest entailment probability over passage spans, per hypothesis.

        `must_mention[i]`, if given, restricts hypothesis i to spans that contain at
        least half of that text's content words. NLI models lean on word overlap
        (McCoy et al., 2019): "hematology is the study of the heart" scores as
        entailed by "cardiology is the study of the heart". A span that never
        mentions hematology cannot be evidence for it."""
        wins = self.windows(passage)
        pairs = [(w, h) for h in hypotheses for w in wins]
        probs = self.model.predict(pairs, apply_softmax=True, batch_size=64, show_progress_bar=False)
        ent = np.asarray(probs)[:, self._entail].reshape(len(hypotheses), len(wins))
        if must_mention:
            for i, text in enumerate(must_mention):
                if text:
                    ent[i] *= np.array([mentions(w, text) for w in wins], dtype=float)
        return ent.max(axis=1).tolist()

    def close(self) -> None:
        if self._model is not None:
            import torch

            del self._model
            self._model = None
            torch.cuda.empty_cache()


def mentions(span: str, text: str) -> bool:
    terms = [t for t in re.findall(r"[a-z][a-z-]{3,}", norm(text)) if t not in STOPWORDS]
    if not terms:
        return True
    span = norm(span)
    return sum(t[:6] in span for t in terms) / len(terms) >= 0.5


def distractor_statements(q: Question) -> list[str | None]:
    """Replace the key's wording inside the answer statement with each distractor."""
    out: list[str | None] = []
    pattern = re.compile(re.escape(q.answer.strip().rstrip(".")), re.I)
    for i, opt in enumerate(q.options):
        if i == q.answer_index:
            continue
        if q.answer_statement and pattern.search(q.answer_statement):
            out.append(pattern.sub(opt.strip().rstrip("."), q.answer_statement, count=1))
        else:
            out.append(None)
    return out


def nli_checks(q: Question, chunk: Chunk, scorer: NLIScorer) -> list[CheckResult]:
    thr = scorer.cfg.entail_threshold
    if not q.answer_statement:
        na = {"reason": "no answer statement"}
        return [
            CheckResult(name="nli_answer_entailed", checker="nli", passed=None, detail=na),
            CheckResult(name="nli_distractors_not_entailed", checker="nli", passed=None, detail=na),
        ]
    statements = distractor_statements(q)
    distractors = [o for i, o in enumerate(q.options) if i != q.answer_index]
    usable = [(s, d) for s, d in zip(statements, distractors, strict=True) if s]
    scores = scorer.max_entailment(
        chunk.text, [q.answer_statement, *[s for s, _ in usable]], must_mention=[None, *[d for _, d in usable]]
    )
    key_score, d_scores = scores[0], scores[1:]
    results = [
        CheckResult(
            name="nli_answer_entailed",
            checker="nli",
            passed=key_score >= thr,
            score=round(key_score, 4),
            detail={"hypothesis": q.answer_statement},
        )
    ]
    if usable:
        worst = max(d_scores)
        results.append(
            CheckResult(
                name="nli_distractors_not_entailed",
                checker="nli",
                passed=worst < thr,
                score=round(worst, 4),
                detail={"statements": [s for s, _ in usable], "scores": [round(x, 4) for x in d_scores]},
            )
        )
    else:
        results.append(
            CheckResult(
                name="nli_distractors_not_entailed",
                checker="nli",
                passed=None,
                detail={"reason": "key wording not found in answer statement"},
            )
        )
    return results
