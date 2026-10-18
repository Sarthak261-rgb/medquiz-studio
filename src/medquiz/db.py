"""Database tables. SQLite by default. Only portable column types are used, so a
Postgres URL in DATABASE_URL should work too, but only SQLite is tested.

The JSONL files under runs/ and data/reviews/ are the source of truth; the database
is an index over them for the API and the review app."""

from __future__ import annotations

from sqlalchemy import JSON, Boolean, Float, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class ChunkRow(Base):
    __tablename__ = "chunks"
    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    chapter: Mapped[int] = mapped_column(Integer)
    section: Mapped[str] = mapped_column(String(16), index=True)
    section_title: Mapped[str] = mapped_column(String(200))
    heading: Mapped[str] = mapped_column(String(400))
    module_id: Mapped[str] = mapped_column(String(32))
    url: Mapped[str] = mapped_column(String(300))
    para_ids: Mapped[list] = mapped_column(JSON)
    text: Mapped[str] = mapped_column(Text)
    n_words: Mapped[int] = mapped_column(Integer)


class RunRow(Base):
    __tablename__ = "runs"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    kind: Mapped[str] = mapped_column(String(32))
    created_at: Mapped[str] = mapped_column(String(32))
    meta: Mapped[dict] = mapped_column(JSON)


class QuestionRow(Base):
    __tablename__ = "questions"
    id: Mapped[str] = mapped_column(String(80), primary_key=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("runs.id"), index=True)
    chunk_id: Mapped[str] = mapped_column(ForeignKey("chunks.id"))
    origin: Mapped[str] = mapped_column(String(16), index=True)
    mutation: Mapped[str | None] = mapped_column(String(32))
    parent_id: Mapped[str | None] = mapped_column(String(80))
    target_type: Mapped[str | None] = mapped_column(String(16))
    stem: Mapped[str] = mapped_column(Text)
    options: Mapped[list] = mapped_column(JSON)
    answer_index: Mapped[int] = mapped_column(Integer)
    explanation: Mapped[str] = mapped_column(Text, default="")
    evidence_quote: Mapped[str] = mapped_column(Text, default="")
    answer_statement: Mapped[str] = mapped_column(Text, default="")
    bloom_level: Mapped[str | None] = mapped_column(String(16))
    gen: Mapped[dict] = mapped_column(JSON)
    # denormalised so the review queue and filters stay one query
    auto_status: Mapped[str] = mapped_column(String(16), default="unchecked", index=True)
    auto_failures: Mapped[list] = mapped_column(JSON, default=list)


class CheckRow(Base):
    __tablename__ = "checks"
    __table_args__ = (UniqueConstraint("question_id", "name"),)
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    question_id: Mapped[str] = mapped_column(ForeignKey("questions.id"), index=True)
    name: Mapped[str] = mapped_column(String(64))
    checker: Mapped[str] = mapped_column(String(16))
    severity: Mapped[str] = mapped_column(String(8))
    passed: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    score: Mapped[float | None] = mapped_column(Float, nullable=True)
    detail: Mapped[dict] = mapped_column(JSON)
    tokens: Mapped[int] = mapped_column(Integer, default=0)
    seconds: Mapped[float] = mapped_column(Float, default=0.0)


class ReviewRow(Base):
    __tablename__ = "reviews"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    question_id: Mapped[str] = mapped_column(ForeignKey("questions.id"), index=True)
    reviewer: Mapped[str] = mapped_column(String(64), index=True)
    decision: Mapped[str] = mapped_column(String(8))
    reasons: Mapped[list] = mapped_column(JSON)
    level: Mapped[str | None] = mapped_column(String(16))
    note: Mapped[str] = mapped_column(Text, default="")
    seconds: Mapped[float | None] = mapped_column(Float, nullable=True)
    created_at: Mapped[str] = mapped_column(String(32))
