"""Review screen.

    streamlit run review_app/app.py

The checks stay hidden until you submit a decision, so the human labels are not
anchored on what the judge said. That is what makes the judge-vs-human numbers
worth anything.
"""

from __future__ import annotations

import html
import os
import re
import time

import pandas as pd
import streamlit as st
from pydantic import ValidationError

from medquiz.config import get_settings
from medquiz.images import IMAGE_FAILURES
from medquiz.images.review import ImageReviewIn, add_image_review, latest_image_reviews, queue
from medquiz.report import compute
from medquiz.runs import read_jsonl
from medquiz.schemas import FAILURE_LABELS, LETTERS, FailureType, ReviewIn
from medquiz.store import Store

st.set_page_config(page_title="MedQuiz review", layout="wide")


@st.cache_resource
def store() -> Store:
    s = get_settings()
    return Store(s.database_url, review_log_dir=s.path("data") / "reviews")


def highlight(passage: str, quote: str) -> str:
    """Escape the passage and mark the evidence quote if it appears verbatim."""
    text = html.escape(passage)
    q = html.escape(quote.strip())
    if q and q in text:
        text = text.replace(q, f"<mark>{q}</mark>", 1)
    else:
        # fall back to marking each quoted sentence that does appear
        for sent in re.split(r"(?<=[.!?])\s+", q):
            if len(sent) > 25 and sent in text:
                text = text.replace(sent, f"<mark>{sent}</mark>", 1)
    return text.replace("\n\n", "<br><br>").replace("\n", "<br>")


def show_question(q: dict, reveal_key: bool = True) -> None:
    st.subheader(q["stem"], anchor=False)
    for i, opt in enumerate(q["options"]):
        mark = "  **(key)**" if reveal_key and i == q["answer_index"] else ""
        st.markdown(f"**{LETTERS[i]}.** {opt}{mark}")
    with st.container(border=True):
        st.caption("Generator's explanation")
        st.write(q["explanation"] or "none")
        st.caption(
            f"Asked for: {q.get('target_type') or 'n/a'}  |  generator's Bloom level: {q.get('bloom_level') or 'n/a'}"
        )


def show_source(q: dict) -> None:
    c = q["chunk"]
    st.caption(f"Source {c['id']}: {c['heading']}")
    st.markdown(
        f"<div style='font-size:0.92rem;line-height:1.5'>{highlight(c['text'], q.get('evidence_quote', ''))}</div>",
        unsafe_allow_html=True,
    )
    st.markdown(f"[Open on openstax.org]({c['cite_url']})")


def show_checks(q: dict) -> None:
    rows = [
        {
            "check": c["name"],
            "kind": c["severity"],
            "result": "n/a" if c["passed"] is None else "pass" if c["passed"] else "FAIL",
            "score": c["score"],
            "detail": ", ".join(f"{k}={v}" for k, v in c["detail"].items() if k not in ("verdicts", "prompt_hash"))[
                :160
            ],
        }
        for c in q["checks"]
    ]
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)


# ----------------------------------------------------------------- sidebar

s = get_settings()
db = store()
runs = [r for r in db.runs() if r["kind"] == "generate"]
with st.sidebar:
    st.header("MedQuiz review")
    reviewer = st.text_input("Reviewer", value=os.environ.get("MEDQUIZ_REVIEWER", ""), max_chars=64).strip()
    run_id = st.selectbox("Run", [r["id"] for r in runs]) if runs else None
    page = st.radio("Page", ["Review queue", "Image review", "Browse", "Metrics"], label_visibility="collapsed")
    # off by default: seeing the judge's verdicts as you go slowly trains you to agree with it
    reveal = st.checkbox("Show checks after each decision", value=False)
    if reviewer and run_id:
        total = next(r["n_questions"] for r in runs if r["id"] == run_id)
        mine = db.latest_reviews(reviewer)
        done = sum(1 for q in db.questions(run_id=run_id, limit=10_000) if q["id"] in mine)
        st.progress(done / max(total, 1), text=f"{done} of {total} reviewed")
    st.caption("Not medical advice. Reviews are saved to the database and to data/reviews/<you>.jsonl.")

if not runs:
    st.info("No generation runs in the database. Run `medquiz db rebuild` first.")
    st.stop()

# -------------------------------------------------------------- review page

if page == "Review queue":
    if not reviewer:
        st.info("Type your name in the sidebar to start.")
        st.stop()

    last = st.session_state.get("last")
    if last and reveal:
        prev = db.question(last["id"])
        verdict = prev["auto_status"]
        if verdict in ("pass", "fail"):
            agree = (verdict == "fail") == (last["decision"] == "reject")
            title = f"Previous question: you {last['decision']}ed, checks said {verdict}"
            title += "" if agree else "  (disagreement)"
        else:
            agree = True
            title = f"Previous question: you {last['decision']}ed; automatic checks have not finished for it yet"
        with st.expander(title, expanded=not agree):
            st.write(prev["stem"])
            if prev["auto_failures"]:
                st.write("Failed hard checks map to: " + ", ".join(prev["auto_failures"]))
            if prev["checks"]:
                show_checks(prev)

    qid = db.next_for_review(reviewer, run_id=run_id)
    if qid is None:
        st.success("Queue empty. Every question in this run has your review.")
        st.stop()
    q = db.question(qid)
    shown = st.session_state.setdefault("shown_at", {})
    shown.setdefault(qid, time.time())

    left, right = st.columns([1.1, 1])
    with left:
        st.caption(f"{q['id']}  |  section {q['chunk']['section']}")
        show_question(q)
    with right:
        show_source(q)

    # one form per question, so a new question always starts empty; a failed
    # submit keeps what was typed
    with st.form(key=f"review-{qid}"):
        c1, c2 = st.columns([1, 2])
        # no default: an unthinking click must not count as an accept
        decision = c1.radio("Decision", ["accept", "reject"], horizontal=True, index=None)
        level = c1.radio("Level", ["recall", "reasoning"], horizontal=True, index=None)
        reasons = c2.multiselect(
            "Reasons (first one counts as primary)",
            options=[f.value for f in FailureType],
            format_func=lambda v: FAILURE_LABELS[FailureType(v)],
        )
        note = c2.text_input("Note (optional)")
        submitted = st.form_submit_button("Submit and next", type="primary")
    if submitted:
        try:
            if decision is None:
                raise ValueError("Pick accept or reject.")
            review = ReviewIn(
                reviewer=reviewer,
                decision=decision,
                reasons=reasons if decision == "reject" else [],
                level=level,
                note=note,
                seconds=time.time() - shown.get(qid, time.time()),
            )
        except ValidationError as e:
            st.error(e.errors()[0]["msg"].removeprefix("Value error, "))
        except ValueError as e:
            st.error(str(e))
        else:
            db.add_review(qid, review)
            st.session_state["last"] = {"id": qid, "decision": decision}
            st.rerun()

# -------------------------------------------------------------- browse page

elif page == "Browse":
    status = st.segmented_control("Auto status", ["all", "pass", "fail", "incomplete"], default="all")
    qs = db.questions(run_id=run_id, auto_status=None if status in (None, "all") else status, limit=10_000)
    reviews = db.latest_reviews(reviewer or None)
    table = pd.DataFrame(
        [
            {
                "id": q["id"],
                "chunk": q["chunk_id"],
                "auto": q["auto_status"],
                "auto failures": ", ".join(q["auto_failures"]),
                "human": reviews.get(q["id"], {}).get("decision", ""),
                "stem": q["stem"],
            }
            for q in qs
        ]
    )
    st.caption(f"{len(qs)} questions")
    event = st.dataframe(
        table, hide_index=True, use_container_width=True, on_select="rerun", selection_mode="single-row"
    )
    rows = event.selection.rows if event and event.selection else []
    if rows:
        q = db.question(table.iloc[rows[0]]["id"])
        left, right = st.columns([1.1, 1])
        with left:
            show_question(q)
            show_checks(q)
            for r in q["reviews"]:
                st.write(f"{r['reviewer']}: {r['decision']} {r['reasons']} {r['note']}")
        with right:
            show_source(q)

# --------------------------------------------------------- image review page

elif page == "Image review":
    img_run = s.path("runs") / "images-v1"
    images = read_jsonl(img_run / "images.jsonl")
    log_dir = s.path("data") / "reviews"
    if not images:
        st.info("No generated images yet. Run `bash scripts/run_images.sh`.")
        st.stop()
    if not reviewer:
        st.info("Type your name in the sidebar to start.")
        st.stop()
    mine = latest_image_reviews(log_dir, reviewer)
    st.progress(len(mine) / len(images), text=f"{len(mine)} of {len(images)} images reviewed")
    todo = queue(images, set(mine))
    if not todo:
        st.success("Every image has your review.")
        st.stop()
    img = todo[0]
    shown = st.session_state.setdefault("img_shown_at", {})
    shown.setdefault(img["id"], time.time())
    left, right = st.columns([1.2, 1])
    with left:
        # which arm (base or LoRA) made it is deliberately not shown
        st.image(str(img_run / img["file"]), use_container_width=True)
    with right:
        st.subheader("Should show", anchor=False)
        st.write(img["subject"])
        st.caption(
            "Reject if a student could learn something wrong from it, or could not tell what it shows. "
            "Judge the drawing, not the SYNTHETIC banner."
        )
        with st.form(key=f"img-{img['id']}"):
            decision = st.radio("Decision", ["accept", "reject"], horizontal=True, index=None)
            reasons = st.multiselect(
                "Reasons (first one counts as primary)", list(IMAGE_FAILURES), format_func=IMAGE_FAILURES.get
            )
            note = st.text_input("What would a doctor correct? (optional)")
            ok = st.form_submit_button("Submit and next", type="primary")
        if ok:
            try:
                if decision is None:
                    raise ValueError("Pick accept or reject.")
                review = ImageReviewIn(
                    reviewer=reviewer,
                    decision=decision,
                    reasons=reasons if decision == "reject" else [],
                    note=note,
                    seconds=time.time() - shown.get(img["id"], time.time()),
                )
            except ValidationError as e:
                st.error(e.errors()[0]["msg"].removeprefix("Value error, "))
            except ValueError as e:
                st.error(str(e))
            else:
                add_image_review(log_dir, img["id"], review)
                st.rerun()

# ------------------------------------------------------------- metrics page

else:
    m = compute(db.snapshot(), s, gen_run=run_id, reviewer=reviewer or None)
    h, a = m["human"], m["auto"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Reviewed", f"{h['n_reviewed']} / {h['n_total']}")
    rej = h["rejection"]["rate"]
    c2.metric("Human rejection rate", "n/a" if rej is None else f"{100 * rej:.1f}%")
    c3.metric("Auto-reject rate", f"{100 * a['auto_reject']['rate']:.1f}%")
    breakdown = pd.DataFrame(
        {
            "human": {FAILURE_LABELS[FailureType(k)]: v["rate"] for k, v in h["primary_reason"].items()},
            "automatic": {FAILURE_LABELS[FailureType(k)]: v["rate"] for k, v in a["primary_failure"].items()},
        }
    ).fillna(0)
    st.subheader("Primary failure, share of questions")
    st.bar_chart(breakdown, horizontal=True, stack=False)
    o = m["agreement"].get("overall")
    if o:
        st.subheader("Checks vs you")
        st.write(
            f"accuracy {o['accuracy']}, precision {o['precision']}, recall {o['recall']}, kappa {o['kappa']} (n={o['n']})"
        )
        st.dataframe(pd.DataFrame(m["agreement"]["per_check"]).T, use_container_width=True)
    st.caption("Full numbers: `medquiz report` writes reports/results.md and reports/metrics.json.")
