"""Builds reports/metrics.json and reports/results.md from the store."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .config import Settings
from .images.metrics import images_summary
from .metrics import (
    ablation,
    agreement,
    auto_summary,
    book_control,
    closed_book,
    cost,
    human_summary,
    judges,
    latest_by_question,
    levels,
    plausibility_by_verdict,
    primary_reviewer,
    solve_rate,
    stress_test,
)
from .schemas import FAILURE_LABELS, FailureType


def data_fingerprint(settings: Settings) -> str:
    """Hash of every input file the numbers depend on. Same fingerprint, same report.
    Line endings are normalised so a Windows checkout and CI agree."""
    root = settings.path("data").parent
    files = [settings.path("data") / "processed" / "chunks.jsonl"]
    files += sorted(settings.path("runs").glob("*/*.json*"))
    files += sorted((settings.path("data") / "reviews").glob("*.jsonl"))
    h = hashlib.sha256()
    for f in files:
        if f.is_file():
            h.update(f.relative_to(root).as_posix().encode())
            h.update(f.read_bytes().replace(b"\r\n", b"\n"))
    return h.hexdigest()[:16]


def runs_overview(snap: dict, settings: Settings) -> list[dict]:
    """One line per generation run, for comparing prompt versions on automatic checks."""
    out = []
    for rid, meta in sorted(snap["runs"].items()):
        if meta.get("kind") != "generate":
            continue
        qs = [q for q in snap["questions"] if q["run_id"] == rid]
        a = auto_summary(qs, snap["checks"], settings)
        out.append(
            {
                "run": rid,
                "prompt": (meta.get("generator") or {}).get("prompt"),
                "n": a["n"],
                "auto_reject": a["auto_reject"],
                "primary_failure": dict(list(a["primary_failure"].items())[:3]),
                "text_reference": a["checks"].get("text_reference", {}).get("fail_rate"),
            }
        )
    return out


def compute(snap: dict, settings: Settings, gen_run: str | None = None, reviewer: str | None = None) -> dict:
    qs = snap["questions"]
    checks = snap["checks"]
    gen_runs = sorted(rid for rid, m in snap["runs"].items() if m.get("kind") == "generate")
    if gen_run is None and settings.primary_run in gen_runs:
        gen_run = settings.primary_run
    gen_run = gen_run or (gen_runs[-1] if gen_runs else None)
    generated = [q for q in qs if q["origin"] == "generated" and q["run_id"] == gen_run]
    gen_ids = {q["id"] for q in generated}
    book = [q for q in qs if q["origin"] == "book"]
    mutants = [q for q in qs if q["origin"] == "mutant" and q.get("parent_id") in gen_ids]
    gen_reviews = [r for r in snap["reviews"] if r["question_id"] in gen_ids]
    reviewer = reviewer or primary_reviewer(gen_reviews)
    reviews = latest_by_question(gen_reviews, reviewer)

    auto = auto_summary(generated, checks, settings)
    human = human_summary(generated, reviews)
    accepted = sum(1 for q in generated if reviews.get(q["id"], {}).get("decision") == "accept")
    book_meta = next((m for m in snap["runs"].values() if m.get("kind") == "book"), {})
    gen_meta = snap["runs"].get(gen_run, {})
    return {
        "data_fingerprint": data_fingerprint(settings),
        "gen_run": gen_run,
        "reviewer": reviewer,
        "setup": {
            "generator": gen_meta.get("generator", {}),
            "judge": gen_meta.get("judge", {}),
            "nli_model": settings.nli.model,
            "embedding_model": settings.embedding.model,
            "source": f"{settings.source.repo}@{settings.source.commit[:12]}",
            "n_chunks": len(snap["chunks"]),
            "gpu": next((s.get("gpu") for s in gen_meta.get("stages", {}).values() if s.get("gpu")), None),
        },
        "auto": auto,
        "human": human,
        "agreement": agreement(generated, reviews, checks, settings),
        "levels": levels(generated, checks, reviews),
        "closed_book": closed_book(generated, checks),
        "options_only": solve_rate(generated, checks, "options_only_solve"),
        "judges": judges(generated, checks, reviews, settings),
        "ablation": ablation(generated, checks, reviews, settings),
        "plausibility_by_verdict": {
            "judge": plausibility_by_verdict(generated, checks, "judge"),
            "self": plausibility_by_verdict(generated, checks, "self"),
        },
        "book_control": book_control(book, checks, settings) if book else None,
        "retrieval": book_meta.get("retrieval_eval"),
        "stress_test": stress_test(mutants, {q["id"]: q for q in generated}, checks, settings) if mutants else None,
        "cost": cost(snap, gen_run, settings, accepted or None, auto["status"].get("pass", 0)) if gen_run else None,
        "runs": runs_overview(snap, settings),
        "images": images_summary(settings),
    }


# ---------------------------------------------------------------- markdown


def pct(r: dict | None, ci: bool = True) -> str:
    if not r or r.get("rate") is None:
        return "n/a"
    s = f"{100 * r['rate']:.1f}% ({r['k']}/{r['n']})"
    if ci:
        lo, hi = r["ci95"]
        s += f", 95% CI {100 * lo:.1f} to {100 * hi:.1f}%"
    return s


def num(x, nd=2) -> str:
    return "n/a" if x is None else f"{x:.{nd}f}" if isinstance(x, float) else str(x)


def label(ftype: str) -> str:
    try:
        return FAILURE_LABELS[FailureType(ftype)]
    except ValueError:
        return ftype


def to_markdown(m: dict) -> str:
    L: list[str] = []
    add = L.append
    s = m["setup"]
    add("# Results\n")
    add(
        f"Run `{m['gen_run']}`, data fingerprint `{m['data_fingerprint']}`. Recomputed by `medquiz report` "
        "from the files in `runs/` and `data/reviews/`. Nothing here is typed in by hand, and CI fails "
        "if a rerun gives a different file.\n"
    )
    add("| | |\n|---|---|")
    g, j = s.get("generator", {}), s.get("judge", {})
    add(
        f"| Generator | `{g.get('model', 'n/a')}` (digest {g.get('digest', '?')}), temperature {g.get('temperature')} |"
    )
    add(f"| Judge | `{j.get('model', 'n/a')}` (digest {j.get('digest', '?')}), temperature {j.get('temperature')} |")
    add(f"| NLI | `{s['nli_model']}` |")
    add(f"| Embeddings | `{s['embedding_model']}` + FAISS |")
    add(f"| Source | `{s['source']}`, {s['n_chunks']} chunks |")
    add(f"| Hardware | {s.get('gpu') or 'n/a'} |\n")

    h = m["human"]
    add("## Human review\n")
    if h["n_reviewed"]:
        add(
            f"Reviewer `{m['reviewer']}` reviewed {h['n_reviewed']} of {h['n_total']} generated questions, "
            "without seeing the automatic checks first.\n"
        )
        add(f"**Rejected: {pct(h['rejection'])}.**\n")
        add("| Primary reason | Share of reviewed questions |\n|---|---|")
        for k, r in h["primary_reason"].items():
            add(f"| {label(k)} | {pct(r, ci=False)} |")
        add("")
        if h["by_target_type"]:
            add(
                "Rejection by what the generator was asked for: "
                + "; ".join(f"{k} {pct(v, ci=False)}" for k, v in h["by_target_type"].items())
                + ".\n"
            )
        if h["review_seconds_median"]:
            add(f"Median review time: {h['review_seconds_median']} s per question.\n")
    else:
        add(f"Not reviewed yet (0 of {h['n_total']}). Run `medquiz review` to start.\n")

    a = m["auto"]
    add("## Automatic checks\n")
    counts = ", ".join(f"{v} {k}" for k, v in sorted(a["status"].items(), key=lambda kv: -kv[1]))
    add(f"Auto-rejected (at least one hard check failed): **{pct(a['auto_reject'])}** ({counts}).\n")
    add("| Primary failure | Share of generated questions |\n|---|---|")
    for k, r in a["primary_failure"].items():
        add(f"| {label(k)} | {pct(r, ci=False)} |")
    add("\n| Check | Kind | Failed | Applied | Fail rate |\n|---|---|---|---|---|")
    for name, c in a["checks"].items():
        add(
            f"| `{name}` | {c['severity']} | {c['failed']} | {c['applied']} | {num(c['fail_rate'] * 100 if c['fail_rate'] is not None else None, 1)}% |"
        )
    add(
        f"\nClosed-book solve (judge answers with no passage): {pct(m['closed_book'], ci=False)} correct. "
        "A high number means many questions test general knowledge rather than the passage.\n"
    )
    oo = m.get("options_only")
    if oo and oo.get("n"):
        lo, hi = oo["ci95"]
        verdict = "above" if lo > 0.25 else "not clearly above"
        add(
            f"Options only (judge sees the four options, no stem and no passage): {pct(oo)} correct, "
            f"{verdict} the 25% a blind guess would get. Anything well above chance means the options "
            "themselves leak the answer.\n"
        )

    jd = m.get("judges")
    if jd:
        add("## Judge vs judge: does the generator grade itself more kindly?\n")
        add(
            f"Same prompts, passages and decision rule; only the judge model changes: `{jd['judge_model']}` "
            f"(independent) against `{jd['self_model']}` (the generator itself).\n"
        )
        add("| Check | Independent judge passes | Self-judge passes | Agreement | Kappa |\n|---|---|---|---|---|")
        for name, r in jd["per_check"].items():
            add(
                f"| {name.replace('_', ' ')} | {pct(r['judge_pass'], ci=False)} | {pct(r['self_pass'], ci=False)} | "
                f"{num(r['agreement'])} | {num(r['kappa'])} |"
            )
        add(
            f"\nAuto-reject rate with the independent judge: {pct(jd['auto_reject_with_judge'])}. "
            f"With the generator as its own judge: {pct(jd['auto_reject_with_self'])}. "
            f"Kappa between the two verdicts: {num(jd['verdict_kappa'])}.\n"
        )
        vh = jd.get("vs_human")
        if vh:
            add("| Judge | Accuracy vs human | Precision | Recall | Kappa |\n|---|---|---|---|---|")
            for k in ("judge", "self"):
                c = vh[k]
                add(f"| {k} | {num(c['accuracy'])} | {num(c['precision'])} | {num(c['recall'])} | {num(c['kappa'])} |")
            add("")

    ag = m["agreement"]
    add("## Judge vs human\n")
    if ag.get("n", 1) and ag.get("overall"):
        o = ag["overall"]
        add("Positive class is *reject*.\n")
        add("| | Human reject | Human accept |\n|---|---|---|")
        add(f"| Auto reject | {o['tp']} | {o['fp']} |")
        add(f"| Auto pass | {o['fn']} | {o['tn']} |\n")
        add(
            f"Accuracy {num(o['accuracy'])}, precision {num(o['precision'])}, recall {num(o['recall'])}, "
            f"Cohen's kappa {num(o['kappa'])} (n={o['n']}).\n"
        )
        add("| Check | Precision | Recall | Kappa | n |\n|---|---|---|---|---|")
        for name, c in ag["per_check"].items():
            add(f"| `{name}` | {num(c['precision'])} | {num(c['recall'])} | {num(c['kappa'])} | {c['n']} |")
        add("")
        if ag.get("per_failure_type"):
            add("Does a check fire when the reviewer names the failure it targets?\n")
            add("| Check | Targets | Precision | Recall | n |\n|---|---|---|---|---|")
            for name, c in ag["per_failure_type"].items():
                add(
                    f"| `{name}` | {label(c['failure_type'])} | {num(c['precision'])} | {num(c['recall'])} | {c['n']} |"
                )
            add("")
        add(f"{len(ag['disagreements'])} disagreements are listed in `reports/metrics.json`.\n")
    else:
        add("Needs human review first.\n")

    ab = m.get("ablation")
    if ab:
        add("## What each hard check costs\n")
        add(
            "The configured gate was fixed before any human review. This table only shows what happens when "
            "one hard check is dropped, or when only the rules or only the judge are used.\n"
        )
        has_h = any("kappa" in r for r in ab)
        head = "| Gate | Auto-rejected |" + (" Accuracy | Precision | Recall | Kappa |" if has_h else "")
        add(head + "\n|---|---|" + ("---|---|---|---|" if has_h else ""))
        for r in ab:
            line = f"| {r['variant']} | {pct(r['auto_reject'], ci=False)} |"
            if has_h:
                line += f" {num(r.get('accuracy'))} | {num(r.get('precision'))} | {num(r.get('recall'))} | {num(r.get('kappa'))} |"
            add(line)
        add("")

    pv = (m.get("plausibility_by_verdict") or {}).get("judge")
    if pv:
        add('### Is "implausible" just "not in the passage"?\n')
        add("Distractor plausibility labels from the independent judge, split by its verdict on the same option.\n")
        add("| Verdict | low | medium | high | Share low |\n|---|---|---|---|---|")
        for verdict, c in pv.items():
            tot = sum(c.get(k, 0) for k in ("low", "medium", "high"))
            share = f"{100 * c.get('low', 0) / tot:.0f}%" if tot else "n/a"
            add(f"| {verdict} | {c.get('low', 0)} | {c.get('medium', 0)} | {c.get('high', 0)} | {share} |")
        add("")

    st = m.get("stress_test")
    if st:
        add("## Judge stress test\n")
        add(
            "Questions that passed every check, broken on purpose in one known way. "
            "The label is known by construction, so this measures the checks, not the generator.\n"
        )
        add("| Mutation | n | Auto-rejected | Caught by |\n|---|---|---|---|")
        for k, r in st.items():
            if k == "reorder":
                continue
            by = "; ".join(f"`{c}` {pct(v, ci=False)}" for c, v in r["caught_by"].items())
            add(f"| {k} | {r['n']} | {pct(r['auto_reject'], ci=False)} | {by} |")
        if "reorder" in st:
            r = st["reorder"]
            add(
                f"\nReversing the option order (nothing else changed) flipped the judge's blind answer in "
                f"{pct(r['answer_flip'], ci=False)} of questions and flipped the overall verdict in "
                f"{pct(r['auto_verdict_flip'], ci=False)}.\n"
            )

    b = m.get("book_control")
    if b:
        add("## Control: the textbook's own questions\n")
        add(
            f"The {b['n']} review MCQs written by the OpenStax authors went through the same checks, each with "
            "the best-matching passage from its own section.\n"
        )
        add(f"- Auto-rejected: {pct(b['auto_reject'])}")
        add(f"- Judge blind-solve accuracy: {pct(b['blind_solve_accuracy'])}")
        add(f"- Judge closed-book accuracy: {pct(b['closed_book_accuracy'], ci=False)}")
        add(f"- Auto-rejected among NOT/EXCEPT stems: {pct(b['negative_stems'], ci=False)}\n")
        add("| Primary failure | Share |\n|---|---|")
        for k, r in b["primary_failure"].items():
            add(f"| {label(k)} | {pct(r, ci=False)} |")
        add("")

    rv = m.get("retrieval")
    if rv:
        add("## Retrieval\n")
        add(f"Does a book question retrieve a chunk from its own section? n={rv['n']}.\n")
        add("| Query | R@1 | R@3 | R@5 |\n|---|---|---|---|")
        for k in ("stem_only", "stem_plus_answer"):
            r = rv[k]
            add(
                f"| {k.replace('_', ' ')} | {r['section_recall@1']:.2f} | {r['section_recall@3']:.2f} | {r['section_recall@5']:.2f} |"
            )
        add("")

    lv = m["levels"]
    add("## Recall vs reasoning\n")
    add("Generator was asked for one type; the judge (and the reviewer, if set) labelled what they saw.\n")
    add("| Asked -> judged | Count |\n|---|---|")
    for k, v in sorted(lv["target_vs_judge"].items()):
        add(f"| {k.replace('->', ' -> ')} | {v} |")
    if lv["target_vs_human"]:
        add("\n| Asked -> reviewer | Count |\n|---|---|")
        for k, v in sorted(lv["target_vs_human"].items()):
            add(f"| {k.replace('->', ' -> ')} | {v} |")
    add(
        f"\nKey position as generated: {lv['key_position_before_shuffle']}. "
        f"After the seeded shuffle: {lv['key_position_after_shuffle']}.\n"
    )

    im = m.get("images")
    if im and im.get("arms"):
        A = im["arms"]
        arms = [a for a in ("base", "lora") if a in A]
        arm_label = {"base": "SD 1.5", "lora": "SD 1.5 + OpenStax LoRA"}
        add("## Images\n")
        add(
            f"{im['subjects']} subjects x {len(im['seeds'])} seeds per arm, same prompts and seeds for both arms. "
            f"The LoRA was trained on {im['figures_used']} OpenStax anatomical drawings "
            f"({im['figures_excluded']} figures excluded: third-party credits, photos, charts, tables). "
            "Every image carries a visible SYNTHETIC banner and EXIF metadata.\n"
        )
        if im.get("reviewer"):
            add(f"Reviewed blind by `{im['reviewer']}`: the review screen never shows which arm made an image.\n")
        add("| | " + " | ".join(arm_label[a] for a in arms) + " |\n|---|" + "---|" * len(arms))

        def row(name, fn):
            add(f"| {name} | " + " | ".join(fn(A[a]) for a in arms) + " |")

        row("Images", lambda x: str(x["n"]))
        row("Human-rejected", lambda x: pct(x["human"]["rejection"]) if x["human"]["n_reviewed"] else "not reviewed")
        row("VLM auto-rejected", lambda x: pct(x["auto_reject"], ci=False))
        row("VLM: shows the subject", lambda x: pct(x["vlm"]["shows_subject"], ci=False))
        row("VLM: garbled text", lambda x: pct(x["vlm"]["garbled_text"], ci=False))
        row("VLM: textbook style", lambda x: pct(x["vlm"]["textbook_style"], ci=False))
        row("CLIP score (mean)", lambda x: num(x["clip_score"]))
        row("Seconds per image", lambda x: num(x["seconds_per_image"], 1))
        row("GPU Wh per image", lambda x: num(x["gpu_wh_per_image"], 4))
        row("GPU Wh per accepted image", lambda x: num(x["gpu_wh_per_accepted"], 4))
        add("")
        for a in arms:
            pr = A[a]["human"]["primary_reason"]
            if pr:
                from .images import IMAGE_FAILURES

                add(
                    f"Why {arm_label[a]} images were rejected (primary reason, share of reviewed): "
                    + "; ".join(f"{IMAGE_FAILURES.get(k, k).lower()} {pct(v, ci=False)}" for k, v in pr.items())
                    + ".\n"
                )
            vh = A[a].get("vlm_vs_human")
            if vh:
                add(
                    f"VLM verdict vs human for {arm_label[a]}: accuracy {num(vh['accuracy'])}, "
                    f"precision {num(vh['precision'])}, recall {num(vh['recall'])}, kappa {num(vh['kappa'])} (n={vh['n']}).\n"
                )
        lo = im["lora"]
        if lo.get("steps"):
            add(
                f"LoRA: rank {lo['rank']}, {lo['trainable_params']:,} trainable parameters, {lo['steps']} steps on "
                f"{lo['n_figures']} figures, {num(lo['train_seconds'], 0)} s and {num(lo['train_gpu_wh'])} Wh on the "
                f"laptop GPU, peak {lo['peak_vram_gb']} GB VRAM. Loss {lo['loss_first']} -> {lo['loss_last']} "
                "(diffusion loss is noisy and says little about anatomy, which is why the review exists).\n"
            )
        add("![contact sheet](figures/images_contact_sheet.jpg)\n")

    c = m.get("cost")
    if c:
        add("## Cost\n")
        add(
            "Everything ran locally, so the cost is time and electricity. GPU energy is read from the "
            "driver's energy counter, not estimated.\n"
        )
        add("| Stage | Items | Wall time (s) | GPU energy (Wh) | Counted in cost |\n|---|---|---|---|---|")
        for k, r in c["stages"].items():
            counted = "yes" if k in c.get("production_stages", []) else "no (experiment)"
            add(f"| {k} | {r.get('n') or ''} | {num(r['seconds'], 0)} | {num(r['gpu_wh'])} | {counted} |")
        share = c["llm_tokens"].get("diagnostic_share_of_checks")
        diag = (
            f" ({100 * share:.0f}% of that is the closed-book and options-only diagnostics, which never block a question)"
            if share
            else ""
        )
        add(
            f"\nLLM tokens: {c['llm_tokens']['generation']:,} for generation and {c['llm_tokens']['checks']:,} for "
            f"checks{diag}. The self-judge experiment is left out of every cost figure.\n"
        )
        add("| Per | Seconds | Tokens | GPU Wh |\n|---|---|---|---|")
        add(
            f"| generated question | {num(c['per_generated']['seconds'], 1)} | {num(c['per_generated']['tokens'], 0)} | {num(c['per_generated']['gpu_wh'], 3)} |"
        )
        add(
            f"| auto-passed question | {num(c['per_auto_passed']['seconds'], 1)} | | {num(c['per_auto_passed']['gpu_wh'], 3)} |"
        )
        if c.get("per_accepted"):
            pa = c["per_accepted"]
            add(
                f"| human-accepted question | {num(pa['seconds'], 1)} | {num(pa['tokens'], 0)} | {num(pa['gpu_wh'], 3)} |"
            )
            add(
                f"\nAt INR {c['inr_per_kwh']}/kWh that is INR {pa['inr']:.4f} of electricity per accepted "
                "question, plus the reviewer's time."
            )
        add("")
    return "\n".join(L) + "\n"


def write(m: dict, out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    jp = out_dir / "metrics.json"
    jp.write_text(json.dumps(m, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    mp = out_dir / "results.md"
    mp.write_text(to_markdown(m), encoding="utf-8", newline="\n")
    return jp, mp
