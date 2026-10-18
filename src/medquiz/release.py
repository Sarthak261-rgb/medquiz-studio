"""Versioned dataset releases. A release folder is written once and never edited;
the manifest records hashes of every file plus everything that produced them."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .config import Settings
from .metrics import latest_by_question, primary_reviewer
from .runs import git_sha, utcnow, write_jsonl

ATTRIBUTION = (
    "Questions are derived from Anatomy and Physiology 2e by J. Gordon Betts et al., OpenStax, "
    "https://openstax.org/details/books/anatomy-and-physiology-2e, licensed under CC BY-NC-SA 4.0. "
    "Changes: questions were generated from the text by a language model, checked automatically, "
    "and reviewed by a person. Access the original for free at openstax.org."
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_release(
    snap: dict,
    settings: Settings,
    version: str,
    gen_run: str,
    metrics: dict,
    reviewer: str | None = None,
    notes: str = "",
) -> Path:
    out = settings.path("releases") / f"v{version}"
    if out.exists():
        raise FileExistsError(f"{out} already exists; releases are immutable, bump the version")
    qs = [q for q in snap["questions"] if q["origin"] == "generated" and q["run_id"] == gen_run]
    ids = {q["id"] for q in qs}
    reviews_all = [r for r in snap["reviews"] if r["question_id"] in ids]
    reviewer = reviewer or primary_reviewer(reviews_all)
    reviews = latest_by_question(reviews_all, reviewer)
    if not reviews:
        raise ValueError("no human reviews for this run yet; a release only contains reviewed questions")
    chunks = {}
    for c in json.loads((settings.path("data") / "processed" / "manifest.json").read_text(encoding="utf-8"))["pages"]:
        chunks[c["section"]] = c["title"]

    accepted, candidates = [], []
    for q in sorted(qs, key=lambda q: q["chunk_id"]):
        r = reviews.get(q["id"])
        checks = snap["checks"].get(q["id"], {})
        judge_level = checks.get("judge_level", {}).get("detail", {}).get("level")
        section = q["chunk_id"].split("-")[0]
        record = {
            "id": q["id"],
            "stem": q["stem"],
            "options": q["options"],
            "answer_index": q["answer_index"],
            "answer": q["options"][q["answer_index"]],
            "explanation": q["explanation"],
            "bloom_level": q["bloom_level"],
            "level": {"asked": q["target_type"], "judge": judge_level, "reviewer": r.get("level") if r else None},
            "source": {
                "chunk_id": q["chunk_id"],
                "section": section,
                "section_title": chunks.get(section),
                "evidence_quote": q["evidence_quote"],
            },
            "provenance": {
                "run_id": gen_run,
                "generator": q["gen"].get("model"),
                "prompt_hash": q["gen"].get("prompt_hash"),
            },
        }
        candidates.append(
            {
                **record,
                "auto_status": q["auto_status"],
                "auto_failures": q["auto_failures"],
                "checks": {k: v.get("passed") for k, v in checks.items()},
                "review": {k: r.get(k) for k in ("decision", "reasons", "level", "note")} if r else None,
            }
        )
        if r and r["decision"] == "accept":
            accepted.append({**record, "review": {"reviewer": reviewer, "at": r["created_at"]}})

    out.mkdir(parents=True)
    write_jsonl(out / "questions.jsonl", accepted)
    write_jsonl(out / "candidates.jsonl", candidates)
    (out / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n"
    )
    meta = snap["runs"].get(gen_run, {})
    manifest = {
        "name": "medquiz-anatomy-mcq",
        "version": version,
        "created_at": utcnow(),
        "notes": notes,
        "license": settings.source.license,
        "attribution": ATTRIBUTION,
        "source": {
            "repo": settings.source.repo,
            "commit": settings.source.commit,
            "chapters": [c.model_dump() for c in settings.source.chapters],
        },
        "generation_run": gen_run,
        "generator": meta.get("generator"),
        "judge": meta.get("judge"),
        "code_version": git_sha(),
        "config_hash": settings.config_hash,
        "reviewer": reviewer,
        "counts": {"candidates": len(candidates), "reviewed": len(reviews), "accepted": len(accepted)},
        "files": {p.name: sha256(p) for p in sorted(out.glob("*.jsonl"))}
        | {"metrics.json": sha256(out / "metrics.json")},
        "disclaimer": "Not clinically reviewed. For study and research only.",
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n"
    )
    log = settings.path("releases") / "CHANGELOG.md"
    entry = (
        f"## v{version} ({manifest['created_at'][:10]})\n\n{len(accepted)} accepted of {len(reviews)} reviewed "
        f"(run `{gen_run}`, code `{manifest['code_version']}`). {notes}\n\n"
    )
    old = log.read_text(encoding="utf-8") if log.exists() else "# Dataset releases\n\n"
    head, _, rest = old.partition("\n\n")
    log.write_text(f"{head}\n\n{entry}{rest}", encoding="utf-8", newline="\n")
    return out


def verify_release(folder: Path) -> list[str]:
    """Return the files whose hash no longer matches the manifest."""
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    return [name for name, h in manifest["files"].items() if sha256(folder / name) != h]
