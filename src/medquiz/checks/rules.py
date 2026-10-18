"""Deterministic checks. Cheap, explainable, and they never change between runs."""

from __future__ import annotations

import re
import unicodedata

from rapidfuzz import fuzz

from ..ingest import split_sentences
from ..schemas import CheckResult, Chunk, Question

ALL_NONE = re.compile(
    r"\b(all|none|neither|both) of (the )?(above|these|the options)\b|^\s*(both\s+)?[a-d]\s*(and|&)\s*[a-d]\s*\.?$",
    re.I,
)
TEXT_REF = re.compile(
    r"\b(according to|based on|in|from|as (stated|described|mentioned|shown) in) the "
    r"(passage|text|reading|excerpt|figure|table|diagram|information (provided|given))\b"
    r"|\bthe (passage|excerpt)\b",
    re.I,
)


def norm(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    text = re.sub(r"[‐-―]", "-", text)
    text = re.sub(r"\s+", " ", text.lower()).strip()
    return text.strip(" .;:,")


def _result(name: str, passed: bool | None, score: float | None = None, **detail) -> CheckResult:
    return CheckResult(name=name, checker="rule", passed=passed, score=score, detail=detail)


def _option_tokens(option: str) -> tuple[str, ...]:
    # operators are words here: "CO = HR x SV" and "CO = HR - SV" are different answers
    words = re.findall(r"[a-z0-9]+|[-+−×÷*/=<>≤≥%^]", norm(option))
    return tuple(w.removesuffix("s") if len(w) > 3 else w for w in words if w not in {"the", "a", "an"})


def options_distinct(q: Question) -> CheckResult:
    """Two options are duplicates if they are the same words in the same order once
    articles and plurals are ignored ("sinoatrial node" / "the sinoatrial nodes").

    Fuzzy matching is deliberately not used to decide: the first version flagged
    13 of 151 questions and all 13 were good contrasting options ("increased" /
    "decreased", "from the alveoli into the blood" / "from the blood into the
    alveoli"). Paraphrased duplicates are left to judge_single_correct."""
    toks = [_option_tokens(o) for o in q.options]
    dupes = [[i, j] for i in range(4) for j in range(i + 1, 4) if toks[i] == toks[j]]
    sims = {(i, j): fuzz.ratio(" ".join(toks[i]), " ".join(toks[j])) for i in range(4) for j in range(i + 1, 4)}
    (i, j), closest = max(sims.items(), key=lambda kv: kv[1])
    return _result("options_distinct", not dupes, closest, duplicates=dupes, most_similar=[i, j])


def no_all_none_of_above(q: Question) -> CheckResult:
    bad = [i for i, o in enumerate(q.options) if ALL_NONE.search(o)]
    return _result("no_all_none_of_above", not bad, flagged=bad)


def stem_leak(q: Question) -> CheckResult:
    """The key's text appearing in the stem gives the answer away."""
    key = norm(q.answer)
    leaked = len(key) >= 4 and re.search(rf"\b{re.escape(key)}\b", norm(q.stem)) is not None
    return _result("stem_leak", not leaked)


def text_reference(q: Question) -> CheckResult:
    """A student sees the question, not the passage, so "according to the passage"
    or "the figure" does not belong in the stem."""
    m = TEXT_REF.search(q.stem)
    return _result("text_reference", m is None, phrase=m.group(0) if m else None)


def length_cue(q: Question, ratio: float) -> CheckResult:
    """Test-wise students pick the longest option. Flag a key that is much longer
    than every distractor."""
    key_len = len(q.answer)
    others = [len(o) for i, o in enumerate(q.options) if i != q.answer_index]
    r = key_len / max(max(others), 1)
    return _result("length_cue", r < ratio, round(r, 2))


_PAREN = re.compile(r"\s*\([^)]*\)")
_ELLIPSIS = re.compile(r"\.\.\.|…")


def evidence_in_source(q: Question, chunk: Chunk, threshold: float) -> CheckResult:
    """Every sentence of the evidence quote must appear, near enough, in the cited chunk."""
    if not q.evidence_quote:
        return _result("evidence_in_source", None, reason="no evidence quote")
    # models trim quotes: "..." joins pieces and asides like "(see the figure)" get dropped
    source = norm(_PAREN.sub("", chunk.text))
    pieces = [p for s in split_sentences(q.evidence_quote) or [q.evidence_quote] for p in _ELLIPSIS.split(s)]
    scores = [fuzz.partial_ratio(norm(_PAREN.sub("", p)), source) for p in pieces if len(norm(p)) >= 15]
    if not scores:
        return _result("evidence_in_source", False, 0.0, reason="quote too short")
    worst = min(scores)
    return _result("evidence_in_source", worst >= threshold, worst, sentence_scores=scores)


def statement_consistent(q: Question) -> CheckResult:
    """The NLI check reads answer_statement; it is only meaningful if that sentence
    actually contains the key."""
    if not q.answer_statement:
        return _result("statement_consistent", None)
    score = fuzz.partial_ratio(norm(q.answer), norm(q.answer_statement))
    return _result("statement_consistent", score >= 85, score)


# capitals always count; lowercase only in the "which ... is not" shape, so a
# scenario like "if the valve does not close" is not flagged
NEGATIVE = re.compile(
    r"\b(NOT|EXCEPT)\b|(?i:\bleast likely\b|\b(which|what)\b[^?]*\b(not|except|false|incorrect)\b)",
)
ABSOLUTE = re.compile(r"\b(always|never|only|all|none|must|completely|entirely|every)\b", re.I)
NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")
STOPWORDS = {
    "that", "this", "with", "from", "into", "than", "then", "they", "their", "there", "which", "when",
    "while", "would", "could", "should", "does", "have", "been", "being", "more", "most", "less", "both",
    "each", "other", "such", "only", "also", "will", "about", "between", "during", "through", "because",
}  # fmt: skip


def negative_stem(q: Question) -> CheckResult:
    """Item-writing guides (the NBME manual, for one) advise against negatively worded
    stems: students miss the negation and the item tests reading, not anatomy."""
    m = NEGATIVE.search(q.stem)
    return _result("negative_stem", m is None, word=m.group(0) if m else None)


def absolute_terms(q: Question) -> CheckResult:
    """Distractors that say "always" or "never" are easy to strike out. Flag them when
    the key carries no absolute of its own."""
    flagged = [i for i, o in enumerate(q.options) if i != q.answer_index and ABSOLUTE.search(o)]
    key_has = ABSOLUTE.search(q.answer) is not None
    return _result("absolute_terms", not (flagged and not key_has), flagged=flagged)


def _numbers(text: str) -> set[str]:
    return {n.replace(",", "").rstrip(".") for n in NUMBER.findall(text)}


def numeric_grounding(q: Question, chunk: Chunk) -> CheckResult:
    """If the key is a number, that number has to be in the passage."""
    nums = _numbers(q.answer)
    if not nums:
        return _result("numeric_grounding", None)
    missing = sorted(nums - _numbers(chunk.text))
    return _result("numeric_grounding", not missing, missing=missing)


def key_terms_in_source(q: Question, chunk: Chunk) -> CheckResult:
    """Cheap lexical grounding: at least half of the key's content words (by a
    six-letter prefix, so plurals and inflections still match) occur in the passage."""
    tokens = [t for t in re.findall(r"[a-z][a-z-]{3,}", norm(q.answer)) if t not in STOPWORDS]
    if not tokens:
        return _result("key_terms_in_source", None)
    source = norm(chunk.text)
    found = [t for t in tokens if t[:6] in source]
    frac = len(found) / len(tokens)
    return _result("key_terms_in_source", frac >= 0.5, round(frac, 3), missing=sorted(set(tokens) - set(found)))


def run_rules(q: Question, chunk: Chunk, length_ratio: float, quote_threshold: float) -> list[CheckResult]:
    return [
        options_distinct(q),
        no_all_none_of_above(q),
        stem_leak(q),
        text_reference(q),
        negative_stem(q),
        absolute_terms(q),
        length_cue(q, length_ratio),
        evidence_in_source(q, chunk, quote_threshold),
        statement_consistent(q),
        numeric_grounding(q, chunk),
        key_terms_in_source(q, chunk),
    ]
