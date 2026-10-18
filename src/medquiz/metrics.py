"""All reported numbers come from here, computed from a store snapshot.

Conventions:
- "reject" is the positive class when comparing automatic checks to human review.
- Rates over small samples come with a Wilson 95% interval.
- A check that could not be applied (passed is None) is left out of its own counts.
"""

from __future__ import annotations

import math
import statistics
from collections import Counter, defaultdict

from .checks import auto_verdict
from .checks.runner import applicable
from .config import Settings
from .mutate import EXPECTED, MUTATIONS
from .schemas import CHECK_FAILURE, LETTERS, CheckResult, Question


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (round(max(0.0, centre - half), 4), round(min(1.0, centre + half), 4))


def rate(k: int, n: int) -> dict:
    return {"k": k, "n": n, "rate": round(k / n, 4) if n else None, "ci95": wilson(k, n)}


def cohen_kappa(a: list, b: list) -> float | None:
    if not a or len(a) != len(b):
        return None
    n = len(a)
    po = sum(x == y for x, y in zip(a, b, strict=True)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[k] * cb[k] for k in set(a) | set(b)) / (n * n)
    return round((po - pe) / (1 - pe), 4) if pe < 1 else None


def confusion(pred_reject: list[bool], human_reject: list[bool]) -> dict:
    tp = sum(p and h for p, h in zip(pred_reject, human_reject, strict=True))
    fp = sum(p and not h for p, h in zip(pred_reject, human_reject, strict=True))
    fn = sum(h and not p for p, h in zip(pred_reject, human_reject, strict=True))
    tn = sum(not p and not h for p, h in zip(pred_reject, human_reject, strict=True))
    n = tp + fp + fn + tn
    return {
        "n": n,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "accuracy": round((tp + tn) / n, 4) if n else None,
        "precision": round(tp / (tp + fp), 4) if tp + fp else None,
        "recall": round(tp / (tp + fn), 4) if tp + fn else None,
        "kappa": cohen_kappa(pred_reject, human_reject),
    }


def _q(d: dict) -> Question:
    return Question(**{k: d[k] for k in Question.model_fields if k in d})


def _results(checks: dict[str, dict]) -> list[CheckResult]:
    return [CheckResult(**c) for c in checks.values()]


def primary_reviewer(reviews: list[dict]) -> str | None:
    c = Counter(r["reviewer"] for r in reviews)
    return c.most_common(1)[0][0] if c else None


def latest_by_question(reviews: list[dict], reviewer: str | None) -> dict[str, dict]:
    out = {}
    for r in reviews:
        if reviewer is None or r["reviewer"] == reviewer:
            out[r["question_id"]] = r
    return out


# --------------------------------------------------------------- sections


def auto_summary(questions: list[dict], checks: dict, settings: Settings) -> dict:
    cfg = settings.checks
    statuses, primaries, any_fail = Counter(), Counter(), Counter()
    per_check = defaultdict(lambda: [0, 0])  # failed, applied
    for q in questions:
        res = _results(checks.get(q["id"], {}))
        v = auto_verdict(res, cfg, applicable(_q(q), cfg.hard))
        statuses[v.status] += 1
        if v.primary:
            primaries[v.primary.value] += 1
        for f in v.failures:
            any_fail[f.value] += 1
        for r in res:
            if r.passed is not None:
                per_check[r.name][1] += 1
                per_check[r.name][0] += r.passed is False
    n = len(questions)
    return {
        "n": n,
        "status": dict(statuses),
        "auto_reject": rate(statuses["fail"], n),
        "primary_failure": {k: rate(v, n) for k, v in primaries.most_common()},
        "any_failure": {k: rate(v, n) for k, v in any_fail.most_common()},
        "checks": {
            name: {
                "failed": f,
                "applied": a,
                "fail_rate": round(f / a, 4) if a else None,
                "severity": "hard" if name in cfg.hard else "soft" if name in cfg.soft else "info",
            }
            for name, (f, a) in sorted(per_check.items())
        },
    }


def human_summary(questions: list[dict], reviews: dict[str, dict]) -> dict:
    reviewed = [q for q in questions if q["id"] in reviews]
    n = len(reviewed)
    rejected = [reviews[q["id"]] for q in reviewed if reviews[q["id"]]["decision"] == "reject"]
    primary = Counter(r["reasons"][0] for r in rejected if r["reasons"])
    mentioned = Counter(x for r in rejected for x in r["reasons"])
    levels = Counter(reviews[q["id"]].get("level") or "unset" for q in reviewed)
    secs = [reviews[q["id"]]["seconds"] for q in reviewed if reviews[q["id"]].get("seconds")]
    by_chapter = defaultdict(lambda: [0, 0])
    for q in reviewed:
        ch = q["chunk_id"].split(".")[0]
        by_chapter[ch][1] += 1
        by_chapter[ch][0] += reviews[q["id"]]["decision"] == "reject"
    by_target = defaultdict(lambda: [0, 0])
    for q in reviewed:
        t = q.get("target_type") or "n/a"
        by_target[t][1] += 1
        by_target[t][0] += reviews[q["id"]]["decision"] == "reject"
    return {
        "n_reviewed": n,
        "n_total": len(questions),
        "rejection": rate(len(rejected), n),
        "primary_reason": {k: rate(v, n) for k, v in primary.most_common()},
        "any_reason": {k: rate(v, n) for k, v in mentioned.most_common()},
        "level": dict(levels),
        "by_chapter": {k: rate(*v) for k, v in sorted(by_chapter.items())},
        "by_target_type": {k: rate(*v) for k, v in sorted(by_target.items())},
        "review_seconds_median": round(statistics.median(secs), 1) if secs else None,
        "review_seconds_total": round(sum(secs), 1) if secs else None,
    }


def agreement(questions: list[dict], reviews: dict[str, dict], checks: dict, settings: Settings) -> dict:
    """How well do the automatic checks predict the human decision?"""
    cfg = settings.checks
    rows = []
    for q in questions:
        r = reviews.get(q["id"])
        if not r:
            continue
        res = _results(checks.get(q["id"], {}))
        v = auto_verdict(res, cfg, applicable(_q(q), cfg.hard))
        if v.status == "incomplete":
            continue
        rows.append((q, r, v, {c.name: c for c in res}))
    if not rows:
        return {"n": 0}
    human = [r["decision"] == "reject" for _, r, _, _ in rows]
    overall = confusion([v.status == "fail" for _, _, v, _ in rows], human)

    per_check = {}
    for name in [*cfg.hard, *cfg.soft]:
        pairs = [
            (c[name].passed is False, h)
            for (_, _, _, c), h in zip(rows, human, strict=True)
            if name in c and c[name].passed is not None
        ]
        if pairs:
            per_check[name] = confusion([p for p, _ in pairs], [h for _, h in pairs])

    # does the check that is meant to catch failure type X fire when the human says X?
    per_type = {}
    for name, ftype in CHECK_FAILURE.items():
        pairs = []
        for _, r, _, c in rows:
            if name in c and c[name].passed is not None:
                pairs.append((c[name].passed is False, ftype.value in r["reasons"]))
        if pairs and any(h for _, h in pairs):
            per_type[name] = {"failure_type": ftype.value, **confusion([p for p, _ in pairs], [h for _, h in pairs])}

    disagreements = [
        {
            "id": q["id"],
            "human": r["decision"],
            "human_reasons": r["reasons"],
            "auto": v.status,
            "auto_failed": v.failed_hard,
            "note": r.get("note", ""),
        }
        for (q, r, v, _) in rows
        if (v.status == "fail") != (r["decision"] == "reject")
    ]
    return {"overall": overall, "per_check": per_check, "per_failure_type": per_type, "disagreements": disagreements}


def levels(questions: list[dict], checks: dict, reviews: dict[str, dict]) -> dict:
    """Recall vs reasoning: what the generator was asked for, what the judge saw,
    and what the reviewer saw."""
    target_vs_judge = Counter()
    target_vs_human = Counter()
    for q in questions:
        t = q.get("target_type")
        j = checks.get(q["id"], {}).get("judge_level", {}).get("detail", {}).get("level")
        if t and j:
            target_vs_judge[f"{t}->{j}"] += 1
        h = reviews.get(q["id"], {}).get("level")
        if t and h:
            target_vs_human[f"{t}->{h}"] += 1
    bloom = Counter(q.get("bloom_level") for q in questions)
    pos = Counter(
        q["gen"].get("original_answer_index") for q in questions if q["gen"].get("original_answer_index") is not None
    )
    return {
        "target_vs_judge": dict(target_vs_judge),
        "target_vs_human": dict(target_vs_human),
        "generator_bloom": dict(bloom),
        "key_position_before_shuffle": {LETTERS[k]: v for k, v in sorted(pos.items())},
        "key_position_after_shuffle": dict(sorted(Counter(LETTERS[q["answer_index"]] for q in questions).items())),
    }


def solve_rate(questions: list[dict], checks: dict, name: str) -> dict:
    vals = [checks.get(q["id"], {}).get(name, {}).get("passed") for q in questions]
    vals = [v for v in vals if v is not None]
    return rate(sum(vals), len(vals))


def closed_book(questions: list[dict], checks: dict) -> dict:
    return solve_rate(questions, checks, "closed_book_solve")


JUDGE_CHECKS = ["blind_solve", "answer_grounded", "single_correct", "distractor_quality", "stem_clear"]


def judges(questions: list[dict], checks: dict, reviews: dict[str, dict], settings: Settings) -> dict | None:
    """The independent judge against the generator grading its own questions.

    Same prompts, same passages, same decision rule; only the model differs. If the
    generator passes its own questions more often, that is self-preference."""
    if not any("self_blind_solve" in checks.get(q["id"], {}) for q in questions):
        return None
    cfg = settings.checks
    self_cfg = cfg.model_copy(update={"hard": [n.replace("judge_", "self_", 1) for n in cfg.hard]})
    per_check = {}
    for name in JUDGE_CHECKS:
        a, b = [], []
        for q in questions:
            c = checks.get(q["id"], {})
            x, y = c.get(f"judge_{name}", {}).get("passed"), c.get(f"self_{name}", {}).get("passed")
            if x is not None and y is not None:
                a.append(x)
                b.append(y)
        per_check[name] = {
            "judge_pass": rate(sum(a), len(a)),
            "self_pass": rate(sum(b), len(b)),
            "agreement": round(sum(x == y for x, y in zip(a, b, strict=True)) / len(a), 4) if a else None,
            "kappa": cohen_kappa(a, b),
        }
    verdicts = []
    for q in questions:
        res = _results(checks.get(q["id"], {}))
        v_j = auto_verdict(res, cfg, applicable(_q(q), cfg.hard)).status
        v_s = auto_verdict(res, self_cfg, applicable(_q(q), self_cfg.hard)).status
        if "incomplete" not in (v_j, v_s):
            verdicts.append((q["id"], v_j == "fail", v_s == "fail"))
    out = {
        "judge_model": settings.judge.model,
        "self_model": settings.self_judge.model,
        "per_check": per_check,
        "auto_reject_with_judge": rate(sum(j for _, j, _ in verdicts), len(verdicts)),
        "auto_reject_with_self": rate(sum(s for _, _, s in verdicts), len(verdicts)),
        "verdict_kappa": cohen_kappa([j for _, j, _ in verdicts], [s for _, _, s in verdicts]),
    }
    reviewed = [(j, s, reviews[qid]["decision"] == "reject") for qid, j, s in verdicts if qid in reviews]
    if reviewed:
        human = [h for _, _, h in reviewed]
        out["vs_human"] = {
            "judge": confusion([j for j, _, _ in reviewed], human),
            "self": confusion([s for _, s, _ in reviewed], human),
        }
    return out


def book_control(book: list[dict], checks: dict, settings: Settings) -> dict:
    s = auto_summary(book, checks, settings)
    solve = [checks.get(q["id"], {}).get("judge_blind_solve", {}).get("passed") for q in book]
    solve = [v for v in solve if v is not None]
    neg = [q for q in book if " not " in f" {q['stem'].lower()} " or "except" in q["stem"].lower()]
    neg_rej = sum(
        auto_verdict(_results(checks.get(q["id"], {})), settings.checks, applicable(_q(q), settings.checks.hard)).status
        == "fail"
        for q in neg
    )
    return {
        **s,
        "blind_solve_accuracy": rate(sum(solve), len(solve)),
        "closed_book_accuracy": closed_book(book, checks),
        "negative_stems": rate(neg_rej, len(neg)),
    }


def stress_test(mutants: list[dict], parents: dict[str, dict], checks: dict, settings: Settings) -> dict:
    cfg = settings.checks
    out = {}
    for m in MUTATIONS:
        group = [q for q in mutants if q.get("mutation") == m]
        if not group:
            continue
        row: dict = {"n": len(group)}
        if m == "reorder":
            flips, verdict_flips, n = 0, 0, 0
            for q in group:
                mine = checks.get(q["id"], {})
                theirs = checks.get(q["parent_id"], {})
                a = mine.get("judge_blind_solve", {}).get("detail", {}).get("picked_text")
                b = theirs.get("judge_blind_solve", {}).get("detail", {}).get("picked_text")
                if a and b:
                    n += 1
                    flips += a != b
                va = auto_verdict(_results(mine), cfg, applicable(_q(q), cfg.hard)).status
                vb = auto_verdict(_results(theirs), cfg, applicable(_q(parents[q["parent_id"]]), cfg.hard)).status
                verdict_flips += va != vb
            row["answer_flip"] = rate(flips, n)
            row["auto_verdict_flip"] = rate(verdict_flips, len(group))
        else:
            row["auto_reject"] = rate(
                sum(
                    auto_verdict(_results(checks.get(q["id"], {})), cfg, applicable(_q(q), cfg.hard)).status == "fail"
                    for q in group
                ),
                len(group),
            )
            row["caught_by"] = {}
            for name in EXPECTED[m]:
                vals = [checks.get(q["id"], {}).get(name, {}).get("passed") for q in group]
                vals = [v for v in vals if v is not None]
                row["caught_by"][name] = rate(sum(v is False for v in vals), len(vals))
        out[m] = row
    return out


EXPERIMENT_STAGES = {"check_selfjudge"}
# judge calls that only feed analysis (they never block a question)
DIAGNOSTIC_CALLS = {"closed_book_solve", "options_only_solve"}


def cost(snapshot: dict, gen_run: str, settings: Settings, accepted: int | None, auto_passed: int) -> dict:
    meta = snapshot["runs"].get(gen_run, {})
    stages = meta.get("stages", {})
    qs = [q for q in snapshot["questions"] if q["run_id"] == gen_run]
    n = len(qs)
    gen_tokens = sum(q["gen"].get("prompt_tokens", 0) + q["gen"].get("completion_tokens", 0) for q in qs)
    gen_secs = sum(q["gen"].get("seconds", 0) for q in qs)
    # the self-judge is an experiment about judges, not part of making questions
    prod = [(n, c) for q in qs for n, c in snapshot["checks"].get(q["id"], {}).items() if not n.startswith("self_")]
    check_tokens = sum(c.get("tokens", 0) for _, c in prod)
    check_secs = sum(c.get("seconds", 0) for _, c in prod)
    diag_tokens = sum(c.get("tokens", 0) for n, c in prod if n in DIAGNOSTIC_CALLS)
    stage_rows = {
        k: {
            "seconds": v.get("seconds"),
            "gpu_wh": round(v["gpu_joules"] / 3600, 2) if v.get("gpu_joules") else None,
            "n": v.get("n"),
        }
        for k, v in stages.items()
    }
    production = {k: v for k, v in stages.items() if k not in EXPERIMENT_STAGES}
    wall = sum(v.get("seconds") or 0 for v in production.values())
    joules = sum(v.get("gpu_joules") or 0 for v in production.values())
    wh = joules / 3600
    inr = wh / 1000 * settings.cost.electricity_inr_per_kwh

    def per(x, d):
        return round(x / d, 3) if d else None

    return {
        "n_generated": n,
        "stages": stage_rows,
        "production_stages": sorted(production),
        "llm_tokens": {
            "generation": gen_tokens,
            "checks": check_tokens,
            "diagnostic_share_of_checks": per(diag_tokens, check_tokens),
        },
        "llm_seconds": {"generation": round(gen_secs, 1), "checks": round(check_secs, 1)},
        "total_wall_seconds": round(wall, 1),
        "total_gpu_wh": round(wh, 2),
        "total_inr": round(inr, 4),
        "inr_per_kwh": settings.cost.electricity_inr_per_kwh,
        "per_generated": {"seconds": per(wall, n), "tokens": per(gen_tokens + check_tokens, n), "gpu_wh": per(wh, n)},
        "per_auto_passed": {"seconds": per(wall, auto_passed), "gpu_wh": per(wh, auto_passed)},
        "per_accepted": {
            "seconds": per(wall, accepted),
            "tokens": per(gen_tokens + check_tokens, accepted),
            "gpu_wh": per(wh, accepted),
            "inr": per(inr, accepted),
        }
        if accepted
        else None,
    }
