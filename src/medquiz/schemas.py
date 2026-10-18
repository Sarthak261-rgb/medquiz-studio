"""Data shapes shared by every stage. Each question has the same JSON shape on disk,
in the database and in the API."""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

LETTERS = "ABCD"


class Block(BaseModel):
    """One paragraph, list or feature box from the book, before chunking."""

    id: str
    module_id: str
    heading: list[str]
    kind: Literal["para", "list", "note"]
    text: str


class Chunk(BaseModel):
    id: str  # e.g. "19.1-004"
    chapter: int
    section: str  # "19.1"; intro pages are "19.0"
    section_title: str
    heading: str
    module_id: str
    para_ids: list[str]
    text: str
    n_words: int
    url: str


class BookQuestion(BaseModel):
    """An end-of-section review question written by the OpenStax authors."""

    id: str
    chapter: int
    section: str
    module_id: str
    stem: str
    options: list[str]
    answer_index: int


class Bloom(StrEnum):
    remember = "remember"
    understand = "understand"
    apply = "apply"
    analyze = "analyze"


class MCQDraft(BaseModel):
    """What the generator model must return."""

    stem: str = Field(min_length=10)
    options: list[str] = Field(min_length=4, max_length=4)
    answer_index: int = Field(ge=0, le=3)
    explanation: str
    evidence_quote: str = Field(description="Exact sentence(s) copied from the passage")
    answer_statement: str = Field(description="One declarative sentence stating the answer")
    bloom_level: Bloom

    @field_validator("options")
    @classmethod
    def strip_options(cls, v: list[str]) -> list[str]:
        out = [o.strip() for o in v]
        if any(not o for o in out):
            raise ValueError("empty option")
        return out


class GenMeta(BaseModel):
    model: str = ""
    prompt: str = ""
    prompt_hash: str = ""
    seed: int | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    seconds: float = 0.0
    attempts: int = 1
    original_answer_index: int | None = None


Origin = Literal["generated", "book", "mutant"]


class Question(BaseModel):
    id: str
    run_id: str
    chunk_id: str
    origin: Origin = "generated"
    target_type: Literal["recall", "reasoning"] | None = None
    mutation: str | None = None
    parent_id: str | None = None
    stem: str
    options: list[str]
    answer_index: int
    explanation: str = ""
    evidence_quote: str = ""
    answer_statement: str = ""
    bloom_level: str | None = None
    gen: GenMeta = Field(default_factory=GenMeta)

    @property
    def answer(self) -> str:
        return self.options[self.answer_index]

    @property
    def answer_letter(self) -> str:
        return LETTERS[self.answer_index]


Severity = Literal["hard", "soft", "info"]
Checker = Literal["rule", "retrieval", "nli", "llm"]


class CheckResult(BaseModel):
    name: str
    checker: Checker
    severity: Severity = "info"
    passed: bool | None  # None means the check could not be applied
    score: float | None = None
    detail: dict = Field(default_factory=dict)
    tokens: int = 0
    seconds: float = 0.0


class FailureType(StrEnum):
    unsupported_answer = "unsupported_answer"
    wrong_key = "wrong_key"
    multiple_correct = "multiple_correct"
    weak_distractors = "weak_distractors"
    wrong_citation = "wrong_citation"
    item_writing_flaw = "item_writing_flaw"
    ambiguous_stem = "ambiguous_stem"
    trivial = "trivial"
    duplicate = "duplicate"
    other = "other"


FAILURE_LABELS: dict[FailureType, str] = {
    FailureType.unsupported_answer: "Answer not supported by the source",
    FailureType.wrong_key: "Marked answer is wrong",
    FailureType.multiple_correct: "More than one correct option",
    FailureType.weak_distractors: "Distractors obviously wrong",
    FailureType.wrong_citation: "Citation or evidence quote does not match",
    FailureType.item_writing_flaw: "Item-writing flaw (cue, all/none of the above)",
    FailureType.ambiguous_stem: "Stem unclear or ambiguous",
    FailureType.trivial: "Trivial, off-topic or about a figure",
    FailureType.duplicate: "Duplicate of another question",
    FailureType.other: "Other",
}

# Order used to pick one "primary" failure when several checks fail.
FAILURE_PRIORITY = list(FailureType)

# Which failure type each automatic check stands for.
CHECK_FAILURE: dict[str, FailureType] = {
    "options_distinct": FailureType.multiple_correct,
    "no_all_none_of_above": FailureType.item_writing_flaw,
    "stem_leak": FailureType.item_writing_flaw,
    "length_cue": FailureType.item_writing_flaw,
    "evidence_in_source": FailureType.wrong_citation,
    "citation_retrieval": FailureType.wrong_citation,
    "duplicate": FailureType.duplicate,
    "nli_answer_entailed": FailureType.unsupported_answer,
    "nli_distractors_not_entailed": FailureType.multiple_correct,
    "judge_blind_solve": FailureType.wrong_key,
    "judge_answer_grounded": FailureType.unsupported_answer,
    "judge_single_correct": FailureType.multiple_correct,
    "judge_distractor_quality": FailureType.weak_distractors,
}


class ReviewIn(BaseModel):
    reviewer: str = Field(min_length=1, max_length=64)
    decision: Literal["accept", "reject"]
    reasons: list[FailureType] = Field(default_factory=list)
    level: Literal["recall", "reasoning"] | None = None
    note: str = ""
    seconds: float | None = None

    @field_validator("reasons")
    @classmethod
    def unique(cls, v: list[FailureType]) -> list[FailureType]:
        return list(dict.fromkeys(v))

    @model_validator(mode="after")
    def reasons_match_decision(self) -> ReviewIn:
        if self.decision == "reject" and not self.reasons:
            raise ValueError("a rejection needs at least one reason")
        if self.decision == "accept" and self.reasons:
            raise ValueError("an accepted question cannot have rejection reasons")
        return self
