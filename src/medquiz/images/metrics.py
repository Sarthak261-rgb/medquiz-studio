"""Numbers for the image track, computed from the files in runs/images-v1 and
data/reviews/images-*.jsonl."""

from __future__ import annotations

import json
import statistics
from collections import Counter
from pathlib import Path

from ..config import Settings
from ..metrics import confusion, rate
from ..runs import read_jsonl
from .review import latest_image_reviews

RUN = "images-v1"


def _mean(xs: list[float]) -> float | None:
    return round(statistics.fmean(xs), 3) if xs else None


def images_summary(settings: Settings) -> dict | None:
    run = settings.path("runs") / RUN
    images = read_jsonl(run / "images.jsonl")
    if not images:
        return None
    meta = json.loads((run / "run.json").read_text(encoding="utf-8")) if (run / "run.json").exists() else {}
    checks = {r["id"]: r for r in read_jsonl(run / "image_checks.jsonl")}
    blind = {r["id"]: r for r in read_jsonl(run / "image_checks_blind.jsonl") if r.get("auto") != "error"}
    log_dir = settings.path("data") / "reviews"
    all_reviews = latest_image_reviews(log_dir)
    by_reviewer = Counter(r["reviewer"] for r in all_reviews.values())
    reviewer = by_reviewer.most_common(1)[0][0] if by_reviewer else None
    reviews = latest_image_reviews(log_dir, reviewer) if reviewer else {}

    arms = {}
    for arm in ("base", "lora"):
        rows = [r for r in images if r["arm"] == arm]
        if not rows:
            continue
        ids = [r["id"] for r in rows]
        chk = [checks[i] for i in ids if i in checks]
        rev = [reviews[i] for i in ids if i in reviews]
        rejected = [r for r in rev if r["decision"] == "reject"]
        joules = [r["gpu_joules"] for r in rows if r.get("gpu_joules") is not None]
        accepted = len(rev) - len(rejected)
        wh_total = sum(joules) / 3600
        out = {
            "n": len(rows),
            "seconds_per_image": _mean([r["seconds"] for r in rows]),
            "gpu_wh_per_image": round(wh_total / len(joules), 4) if joules else None,
            "clip_score": _mean([c["clip_score"] for c in chk]),
            "vlm": {
                "shows_subject": rate(sum(c["vlm"]["shows_subject"] for c in chk), len(chk)),
                "contains_text": rate(sum(c["vlm"]["contains_text"] for c in chk), len(chk)),
                "garbled_text": rate(
                    sum(c["vlm"]["contains_text"] and not c["vlm"]["text_is_readable"] for c in chk), len(chk)
                ),
                "textbook_style": rate(sum(c["vlm"]["looks_like_textbook_illustration"] for c in chk), len(chk)),
            },
            "auto_reject": rate(sum(c["auto"] == "fail" for c in chk), len(chk)),
            "blind": {
                "auto_reject": rate(
                    sum(blind[i]["auto"] == "fail" for i in ids if i in blind), sum(1 for i in ids if i in blind)
                ),
                "not_subject": rate(
                    sum(not blind[i]["on_subject"] for i in ids if i in blind), sum(1 for i in ids if i in blind)
                ),
                "garbled_labels": rate(
                    sum(blind[i]["garbled_labels"] for i in ids if i in blind), sum(1 for i in ids if i in blind)
                ),
                "has_labels": rate(
                    sum(blind[i]["n_label_words"] > 0 for i in ids if i in blind), sum(1 for i in ids if i in blind)
                ),
            },
            "human": {
                "n_reviewed": len(rev),
                "rejection": rate(len(rejected), len(rev)),
                "primary_reason": {
                    k: rate(v, len(rev)) for k, v in Counter(r["reasons"][0] for r in rejected).most_common()
                },
                "any_reason": {
                    k: rate(v, len(rev)) for k, v in Counter(x for r in rejected for x in r["reasons"]).most_common()
                },
            },
            "accepted": accepted,
            "gpu_wh_per_accepted": round(wh_total / accepted, 4) if accepted and joules else None,
            "seconds_per_accepted": round(sum(r["seconds"] for r in rows) / accepted, 1) if accepted else None,
        }
        pairs = [(checks[i]["auto"] == "fail", reviews[i]["decision"] == "reject") for i in ids
                 if i in checks and i in reviews]  # fmt: skip
        if pairs:
            out["vlm_vs_human"] = confusion([p for p, _ in pairs], [h for _, h in pairs])
        pairs = [(blind[i]["auto"] == "fail", reviews[i]["decision"] == "reject") for i in ids
                 if i in blind and i in reviews]  # fmt: skip
        if pairs:
            out["blind_vs_human"] = confusion([p for p, _ in pairs], [h for _, h in pairs])
        notes = [{"id": r["image_id"], "note": r["note"]} for r in rev if r.get("note")]
        out["doctor_notes"] = notes[:20]
        arms[arm] = out

    lora = meta.get("lora") or {}
    stages = meta.get("stages", {})
    train = stages.get("train_lora", {})
    log = read_jsonl(run / "train_log.jsonl")
    return {
        "run": RUN,
        "reviewer": reviewer,
        "base_model": settings.images.base_model if settings.images else None,
        "subjects": len({r["subject_id"] for r in images}),
        "seeds": sorted({r["seed"] for r in images}),
        "arms": arms,
        "lora": {
            "n_figures": lora.get("n_figures"),
            "trainable_params": lora.get("trainable_params"),
            "rank": lora.get("rank"),
            "steps": lora.get("steps"),
            "peak_vram_gb": lora.get("peak_vram_gb"),
            "train_seconds": train.get("seconds"),
            "train_gpu_wh": round(train["gpu_joules"] / 3600, 2) if train.get("gpu_joules") else None,
            "loss_first": log[0]["loss"] if log else None,
            "loss_last": log[-1]["loss"] if log else None,
        },
        "figures_used": len(read_jsonl(settings.path("data") / "figures" / "figures.jsonl")),
        "figures_excluded": len(read_jsonl(settings.path("data") / "figures" / "excluded.jsonl")),
    }


def contact_sheet(settings: Settings, out: Path, seed: int | None = None, thumb: int = 160) -> Path | None:
    """One row per arm, one column per subject, same seed: the comparison at a glance."""
    from PIL import Image, ImageDraw

    run = settings.path("runs") / RUN
    images = read_jsonl(run / "images.jsonl")
    if not images:
        return None
    seed = seed if seed is not None else min(r["seed"] for r in images)
    subjects = list(dict.fromkeys(r["subject_id"] for r in images))
    arms = [a for a in ("base", "lora") if any(r["arm"] == a for r in images)]
    pad = 70
    sheet = Image.new("RGB", (pad + thumb * len(subjects), 24 + thumb * len(arms)), (252, 252, 251))
    draw = ImageDraw.Draw(sheet)
    for j, sid in enumerate(subjects):
        draw.text((pad + j * thumb + 4, 6), sid, fill=(82, 81, 78))
    for i, arm in enumerate(arms):
        draw.text((6, 24 + i * thumb + thumb // 2 - 6), "SD 1.5" if arm == "base" else "+ LoRA", fill=(11, 11, 11))
        for j, sid in enumerate(subjects):
            row = next((r for r in images if r["arm"] == arm and r["subject_id"] == sid and r["seed"] == seed), None)
            if row:
                im = Image.open(run / row["file"]).convert("RGB").resize((thumb, thumb))
                sheet.paste(im, (pad + j * thumb, 24 + i * thumb))
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out, quality=88)
    return out
