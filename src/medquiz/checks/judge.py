"""LLM-as-judge checks.

The main judge is a model from a different family than the generator. Per question:
  blind solve    answer from the passage without seeing the key
  option audit   per-option verdict against the passage, distractor plausibility,
                 recall vs reasoning, stem clarity
  closed book    answer with no passage (info: is the passage even needed?)
  options only   pick an answer from the four options alone, no stem, no passage
                 (info: do the options give the answer away? chance is 25%)

The same blind solve and audit are also run with the generator itself as judge
("self_" checks), to measure how much more lenient a model is with its own output.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from ..config import JudgeCfg
from ..llm import JSONChat, LLMError
from ..prompts import format_options, render
from ..schemas import LETTERS, CheckResult, Question

Letter = Literal["A", "B", "C", "D"]


class SolveOut(BaseModel):
    reason: str
    answer: Letter
    confidence: int = Field(ge=1, le=5)


class OptionVerdict(BaseModel):
    option: Letter
    evidence: str
    verdict: Literal["supported", "contradicted", "not_in_passage"]
    plausibility: Literal["low", "medium", "high", "n/a"]


class AuditOut(BaseModel):
    options: list[OptionVerdict] = Field(min_length=4, max_length=4)
    cognitive_level: Literal["recall", "reasoning"]
    stem_clear: bool


def _chat(client: JSONChat, cfg: JudgeCfg, system: str, user: str, schema):
    return client.chat_json(
        cfg.model,
        system,
        user,
        schema,
        temperature=cfg.temperature,
        seed=cfg.seed,
        num_ctx=cfg.num_ctx,
        num_gpu=cfg.num_gpu,
    )


def _err(names: list[str], e: Exception) -> list[CheckResult]:
    return [CheckResult(name=n, checker="llm", passed=None, detail={"error": str(e)[:300]}) for n in names]


def _solve(client, cfg, q, prompt, name, options, answer_index, **values) -> CheckResult:
    system, user, phash = render(prompt, options=format_options(options), **values)
    try:
        out, st = _chat(client, cfg, system, user, SolveOut)
    except LLMError as e:
        return _err([name], e)[0]
    picked = LETTERS.index(out.answer)
    return CheckResult(
        name=name,
        checker="llm",
        passed=picked == answer_index,
        score=out.confidence,
        detail={"picked": out.answer, "picked_text": options[picked], "reason": out.reason, "prompt_hash": phash},
        tokens=st.tokens,
        seconds=round(st.seconds, 3),
    )


def blind_solve(
    client: JSONChat,
    cfg: JudgeCfg,
    q: Question,
    passage: str,
    *,
    prefix: str = "judge",
    options: list[str] | None = None,
    answer_index: int | None = None,
) -> CheckResult:
    """`options`/`answer_index` allow re-asking with the options in another order."""
    options = options or q.options
    answer_index = q.answer_index if answer_index is None else answer_index
    return _solve(
        client, cfg, q, "judge_solve_v1", f"{prefix}_blind_solve", options, answer_index, passage=passage, stem=q.stem
    )


def closed_book(client: JSONChat, cfg: JudgeCfg, q: Question) -> CheckResult:
    return _solve(client, cfg, q, "judge_closed_book_v1", "closed_book_solve", q.options, q.answer_index, stem=q.stem)


def options_only(client: JSONChat, cfg: JudgeCfg, q: Question) -> CheckResult:
    return _solve(client, cfg, q, "judge_options_only_v1", "options_only_solve", q.options, q.answer_index)


def audit_names(prefix: str) -> list[str]:
    return [f"{prefix}_{n}" for n in ("answer_grounded", "single_correct", "distractor_quality", "stem_clear", "level")]


def audit(client: JSONChat, cfg: JudgeCfg, q: Question, passage: str, *, prefix: str = "judge") -> list[CheckResult]:
    names = audit_names(prefix)
    system, user, phash = render(
        "judge_audit_v1",
        passage=passage,
        stem=q.stem,
        options=format_options(q.options),
        answer_letter=q.answer_letter,
    )
    try:
        out, st = _chat(client, cfg, system, user, AuditOut)
    except LLMError as e:
        return _err(names, e)
    by_letter = {v.option: v for v in out.options}
    if len(by_letter) != 4:
        return _err(names, LLMError(f"audit repeated letters: {[v.option for v in out.options]}"))
    key = by_letter[q.answer_letter]
    distractors = [by_letter[L] for L in LETTERS if L != q.answer_letter]
    also_supported = [v.option for v in distractors if v.verdict == "supported"]
    weak = [v.option for v in distractors if v.plausibility == "low"]
    verdicts = {v.option: v.model_dump(exclude={"option"}) for v in out.options}

    def res(name, passed, score=None, tokens=0, seconds=0.0, **detail):
        return CheckResult(
            name=name,
            checker="llm",
            passed=passed,
            score=score,
            detail={**detail, "prompt_hash": phash},
            tokens=tokens,
            seconds=seconds,
        )

    grounded, single, quality, clear, level = names
    return [
        # the call's cost is booked once, on the first result
        res(
            grounded,
            key.verdict == "supported",
            tokens=st.tokens,
            seconds=round(st.seconds, 3),
            key_verdict=key.verdict,
            key_evidence=key.evidence,
            verdicts=verdicts,
        ),
        res(single, not also_supported, also_supported=also_supported),
        res(quality, len(weak) <= 1, len(weak), weak=weak),
        res(clear, out.stem_clear),
        res(level, None, level=out.cognitive_level),
    ]


def judge_checks(client: JSONChat, cfg: JudgeCfg, q: Question, passage: str) -> list[CheckResult]:
    out = [blind_solve(client, cfg, q, passage), *audit(client, cfg, q, passage)]
    # a planted defect does not change what the judge knows without the passage
    if q.origin != "mutant":
        out += [closed_book(client, cfg, q), options_only(client, cfg, q)]
    return out


def self_judge_checks(client: JSONChat, cfg: JudgeCfg, q: Question, passage: str) -> list[CheckResult]:
    return [blind_solve(client, cfg, q, passage, prefix="self"), *audit(client, cfg, q, passage, prefix="self")]
