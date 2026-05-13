"""
Streamlit Human Evaluation App
================================
Run locally:  streamlit run eval_app/app.py
Share via:    ngrok http 8501   (gives a public URL for evaluators)
"""
import sys
from pathlib import Path

import markdown as _md
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))

from eval_app.config import DATASET, MODEL_SLUG, N_RECORDS_PER_EVALUATOR, N_CORE_RECORDS
from eval_app.data_loader import (
    load_eval_records, is_suggestions_node,
    actual_positive_score, extract_section,
)
from eval_app.assignment import get_assignment
from eval_app.storage import save_response, completed_ids_for

def _md_to_html(text: str) -> str:
    """Convert markdown text to HTML for embedding inside custom HTML divs."""
    if not text:
        return text
    return _md.markdown(text, extensions=["nl2br"])


# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Feedback Summary Evaluation",
    page_icon="📋",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.summary-box {
    background: #e8f0fe; border-left: 4px solid #4a90d9;
    padding: 1rem 1.2rem; border-radius: 4px; margin-bottom: 1rem;
    color: #111111 !important;
}
.source-tuple {
    background: #f5f5f5; border: 1px solid #cccccc;
    border-radius: 6px; padding: 0.7rem 1rem; margin-bottom: 0.5rem;
    color: #111111 !important;
}
.reveal-box {
    background: #fef9e7; border: 1px solid #f0ad4e;
    border-radius: 6px; padding: 0.8rem 1.2rem; margin-top: 0.5rem;
    color: #111111 !important;
}
.sentiment-group-pos { border-left: 3px solid #28a745; padding-left: 0.6rem; margin-bottom: 0.4rem; }
.sentiment-group-neg { border-left: 3px solid #dc3545; padding-left: 0.6rem; margin-bottom: 0.4rem; }
.sentiment-group-sug { border-left: 3px solid #007bff; padding-left: 0.6rem; margin-bottom: 0.4rem; }
.tag-positive  { background:#c3e6cb; color:#155724 !important; padding:2px 9px; border-radius:12px; font-size:0.8rem; font-weight:600; }
.tag-negative  { background:#f5c6cb; color:#721c24 !important; padding:2px 9px; border-radius:12px; font-size:0.8rem; font-weight:600; }
.tag-suggestion{ background:#bee5eb; color:#0c5460 !important; padding:2px 9px; border-radius:12px; font-size:0.8rem; font-weight:600; }
/* Force text colour inside all custom HTML blocks */
.summary-box *, .source-tuple *, .reveal-box * { color: #111111 !important; }
</style>
""", unsafe_allow_html=True)

# ── Session state defaults ────────────────────────────────────────────────────
_DEFAULTS = {
    "logged_in": False,
    "evaluator_id": None,
    "evaluator_name": None,
    "assigned_records": [],
    "pending_records": [],   # records not yet submitted
    "guess_submitted": False,
    "current_guess": 5,
    "completed": False,
}
for k, v in _DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ── Helpers ───────────────────────────────────────────────────────────────────
def _tag(sentiment: str) -> str:
    css = {"positive": "tag-positive", "negative": "tag-negative",
           "suggestion": "tag-suggestion"}.get(sentiment, "")
    label = sentiment.capitalize()
    return f'<span class="{css}">{label}</span>'


RATING_OPTIONS = ["1 — Poor", "2 — Below average", "3 — Acceptable",
                  "4 — Good", "5 — Excellent"]
ALIGN_OPTIONS  = ["Yes, fully", "Mostly", "Partially", "No"]
PROP_OPTIONS   = ["1 — Key themes lost", "2 — Mostly lost",
                  "3 — Partially preserved", "4 — Mostly preserved",
                  "5 — Fully preserved"]


def _parse_rating(val: str) -> int:
    return int(val[0])


# ═══════════════════════════════════════════════════════════════════════════════
# LOGIN SCREEN
# ═══════════════════════════════════════════════════════════════════════════════
def show_login() -> None:
    col_l, col_m, col_r = st.columns([1, 2, 1])
    with col_m:
        st.markdown("# 📋 Course Feedback Summary Evaluation")
        st.markdown("---")
        st.markdown("""
### Welcome!

You will evaluate AI-generated summaries of student course feedback.

**What you'll do per topic (~2–3 min each):**
1. **Read** the consolidated summary
2. **Guess** the sentiment balance of the underlying feedback
3. **See** the actual source material (the input to the AI) and compare
4. **Rate** the summary on faithfulness, coverage, and coherence
5. **Compare** how well the consolidated and standalone summaries align
6. **Check** how well a broader topic's summary preserves this topic's content

**Total time: approximately 20 minutes** for ~7 topics.

---
💡 **You can complete this in multiple sittings.**
Enter the **same name** each time and you will automatically continue
from where you left off — already-submitted topics will be skipped.
        """)
        st.markdown("---")
        name = st.text_input(
            "Enter your name or evaluator code:",
            placeholder="e.g., Evaluator_01  or  Priya",
        )
        if st.button("▶  Start Evaluation", type="primary", use_container_width=True):
            if not name.strip():
                st.error("Please enter your name or evaluator code to continue.")
                return
            eid = name.strip().lower().replace(" ", "_")
            try:
                all_records = load_eval_records()
            except FileNotFoundError as e:
                st.error(str(e))
                return

            assigned = get_assignment(eid, all_records, N_RECORDS_PER_EVALUATOR, N_CORE_RECORDS)
            done_ids = completed_ids_for(eid)
            pending  = [r for r in assigned if r["eval_id"] not in done_ids]

            st.session_state.logged_in       = True
            st.session_state.evaluator_id    = eid
            st.session_state.evaluator_name  = name.strip()
            st.session_state.assigned_records = assigned
            st.session_state.pending_records  = pending
            st.session_state.guess_submitted  = False
            st.session_state.current_guess    = 5
            st.session_state.completed        = len(pending) == 0
            st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# COMPLETION SCREEN
# ═══════════════════════════════════════════════════════════════════════════════
def show_completion() -> None:
    st.balloons()
    col_l, col_m, col_r = st.columns([1, 2, 1])
    with col_m:
        st.success("## 🎉 Thank you for completing the evaluation!")
        n_done = len(st.session_state.assigned_records)
        st.markdown(f"""
**{st.session_state.evaluator_name}**, your {n_done} responses have been saved.

You may now close this tab.
        """)


# ═══════════════════════════════════════════════════════════════════════════════
# EVALUATION SCREEN
# ═══════════════════════════════════════════════════════════════════════════════
def show_evaluation() -> None:
    pending = st.session_state.pending_records
    total   = len(st.session_state.assigned_records)
    done    = total - len(pending)

    if not pending:
        st.session_state.completed = True
        st.rerun()
        return

    record = pending[0]
    eid    = record["eval_id"]
    skip_t1 = is_suggestions_node(record["node_path"])

    # ── Progress ──────────────────────────────────────────────────────────────
    st.progress(done / total, text=f"Topic {done + 1} of {total}")

    # ── Node header ───────────────────────────────────────────────────────────
    path_str = " › ".join(record["node_path"])
    ntype = {"regular_leaf": "Leaf Node", "negative_sibling_leaf": "Leaf Node (catch-all)",
             "internal": "Internal Node"}.get(record["node_type"], record["node_type"])

    st.markdown(f"## 📍 {record['node_name']}")
    st.caption(f"Path: {path_str}   |   {ntype}, Level {record['node_level']}")
    st.markdown("---")

    # ── PART 1 — Consolidated Summary ─────────────────────────────────────────
    st.markdown("### 📝 Consolidated Summary")
    st.markdown(
        f'<div class="summary-box">{_md_to_html(record.get("summary_A") or "_No summary available._")}</div>',
        unsafe_allow_html=True,
    )

    # ── TASK 1 — Sentiment Guess (hidden until submitted) ─────────────────────
    if not skip_t1:
        st.markdown("---")
        st.markdown("### Task 1 — Sentiment Guess")
        st.info(
            "Based **only** on the summary above (before seeing the source), "
            "estimate the balance of the feedback: is it mostly positive praise, "
            "mostly criticism, or balanced?"
        )

        if not st.session_state.guess_submitted:
            guess = st.slider(
                "← All Criticism · · · · · · · · · · All Praise →",
                min_value=0, max_value=10, value=5, step=1,
                key=f"guess_slider_{eid}",
                help="0 = entirely negative feedback,  10 = entirely positive feedback",
            )
            st.session_state.current_guess = guess

            col_btn, _ = st.columns([1, 3])
            with col_btn:
                if st.button("Submit Guess & See Source →", type="primary"):
                    st.session_state.guess_submitted = True
                    st.rerun()
            return  # nothing below is shown until guess is submitted

        else:
            # Show guess + actual reveal side by side
            guess_val = st.session_state.current_guess
            actual    = actual_positive_score(record["tuple_counts"])
            tc        = record["tuple_counts"]
            # total may be absent; recompute from parts to be safe
            total_t   = max(tc.get("positive", 0) + tc.get("negative", 0) + tc.get("suggestion", 0), 1)

            col_l, col_r = st.columns(2)
            with col_l:
                filled = "🟦" * guess_val + "⬜" * (10 - guess_val)
                st.markdown(f"**Your guess:** {guess_val}/10\n\n`{filled}`")
            with col_r:
                pos = tc.get("positive", 0)
                neg = tc.get("negative", 0)
                sug = tc.get("suggestion", 0)
                st.markdown(
                    f'<div class="reveal-box">'
                    f"<b>Actual breakdown ({total_t} total feedback items):</b><br>"
                    f"🟢 Positive: <b>{pos}</b> ({round(pos/total_t*100)}%)<br>"
                    f"🔴 Negative: <b>{neg}</b> ({round(neg/total_t*100)}%)<br>"
                    f"🔵 Suggestion: <b>{sug}</b> ({round(sug/total_t*100)}%)"
                    f"</div>",
                    unsafe_allow_html=True,
                )
    else:
        # Suggestions subtree — skip Task 1 silently
        if not st.session_state.guess_submitted:
            st.session_state.guess_submitted = True
            st.session_state.current_guess   = -1

    # ── PART 2 — Source Material ──────────────────────────────────────────────
    st.markdown("---")
    with st.expander("📂 Source Material — the input given to the AI model", expanded=True):
        if record.get("source_tuples"):
            tuples = record["source_tuples"]
            st.info(
                f"Below are the **{len(tuples)} raw student feedback items** that were "
                f"fed as input to the AI summarisation model to produce the summary above. "
                f"Use these as the reference when rating **Faithfulness** "
                f"(does the summary invent anything not here?), **Coverage** "
                f"(does it miss important points?), and **Coherence** (is it well-written?)."
                
            )
            st.warning(
                "⚠️ **Important:** Each item below shows the full student review for context. "
                "However, please evaluate the summary **only based on the Aspect and Opinion fields** "
                "(shown in bold at the top of each item) — these are the specific parts of the review "
                "that were annotated and used as input to the summarisation model. "
                "The full review text is provided for background only."
            )
            st.markdown("Feedback is grouped by sentiment for easy scanning:")
            # Group by sentiment for readability
            groups = {"positive": [], "negative": [], "suggestion": []}
            for t in tuples:
                groups.get(t["sentiment"], groups["suggestion"]).append(t)

            group_cfg = [
                ("positive",   "🟢 Positive Feedback",   "sentiment-group-pos"),
                ("negative",   "🔴 Negative Feedback",   "sentiment-group-neg"),
                ("suggestion", "🔵 Suggestions",          "sentiment-group-sug"),
            ]
            for sentiment, header, css_class in group_cfg:
                items = groups[sentiment]
                if not items:
                    continue
                st.markdown(f"**{header} ({len(items)})**")
                for i, t in enumerate(items, 1):
                    st.markdown(
                        f'<div class="source-tuple {css_class}">'
                        f'<b>{i}.</b> &nbsp;'
                        f'<b>Aspect:</b> {t["aspect"]} &nbsp;|&nbsp; '
                        f'<b>Opinion:</b> {t["opinion"]}<br>'
                        f'<span style="font-size:0.9rem;color:#333333;">'
                        f'&ldquo;{t["source_feedback"]}&rdquo;</span>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                st.markdown("")

        elif record.get("child_summaries_used"):
            st.info(
                "This is an **internal (higher-level) topic**. The AI was given the "
                "summaries of its sub-topics (shown below) as input to produce the "
                "consolidated summary above. Use these child summaries as the reference "
                "when rating Faithfulness, Coverage, and Coherence."
            )
            st.markdown("**Sub-topic summaries used as input:**")
            for child_name, child_data in record["child_summaries_used"].items():
                # Cannot nest st.expander inside st.expander — use styled HTML block instead
                parts = []
                if child_data.get("type2_positive"):
                    parts.append(f"🟢 <b>Positive:</b> {child_data['type2_positive']}")
                if child_data.get("type3_negative"):
                    parts.append(f"🔴 <b>Negative:</b> {child_data['type3_negative']}")
                if child_data.get("type4_suggestion"):
                    parts.append(f"🔵 <b>Suggestions:</b> {child_data['type4_suggestion']}")
                body = "<br><br>".join(parts) if parts else "<i>No summaries available.</i>"
                st.markdown(
                    f'<div class="source-tuple">'
                    f'<b style="font-size:1rem;">📌 {child_name}</b><br><br>'
                    f'{body}'
                    f'</div>',
                    unsafe_allow_html=True,
                )
                st.markdown("")

    # ── TASK 2 — Rate the Consolidated Summary ────────────────────────────────
    st.markdown("---")
    st.markdown("### Task 2 — Rate the Consolidated Summary")
    st.info(
        "Use the source material above as reference. Rate each dimension on a 1–5 scale."
    )

    col_f, col_c, col_h = st.columns(3)
    with col_f:
        st.markdown("**Faithfulness**")
        st.caption("Does it only contain information present in the source? No hallucinated or invented claims?")
        faith = st.radio("faith", RATING_OPTIONS, key=f"faith_{eid}", label_visibility="collapsed")
    with col_c:
        st.markdown("**Coverage**")
        st.caption("Does it capture all the important points from the source? Are any key themes missing?")
        cover = st.radio("cover", RATING_OPTIONS, key=f"cover_{eid}", label_visibility="collapsed")
    with col_h:
        st.markdown("**Coherence**")
        st.caption("Is the summary well-written, clear, and easy to read?")
        coher = st.radio("coher", RATING_OPTIONS, key=f"coher_{eid}", label_visibility="collapsed")

    # ── TASK 3 — Section Comparison ──────────────────────────────────────────
    task3_pos_align = None
    task3_neg_align = None
    has_B = bool(record.get("summary_B"))
    has_C = bool(record.get("summary_C"))

    if has_B or has_C:
        st.markdown("---")
        st.markdown("### Task 3 — Compare Summary Sections")
        st.info(
            "The consolidated summary (A) has separate sections for Appraisals "
            "and Criticisms. Compare each section against the standalone "
            "positive-only and negative-only summaries."
        )

        if has_B:
            appraisals_text = extract_section(record.get("summary_A", ""), "## Appraisals")
            st.markdown("#### Appraisals section (A) vs. Positive-only summary (B)")
            col_l, col_r = st.columns(2)
            with col_l:
                st.markdown("**A — Appraisals section:**")
                st.markdown(
                    f'<div class="summary-box">{_md_to_html(appraisals_text or "_No appraisals section found._")}</div>',
                    unsafe_allow_html=True,
                )
            with col_r:
                st.markdown("**B — Positive-only summary:**")
                st.markdown(
                    f'<div class="summary-box">{_md_to_html(record["summary_B"])}</div>',
                    unsafe_allow_html=True,
                )
            task3_pos_align = st.radio(
                "Do A's Appraisals and summary B convey the same key positive points?",
                ALIGN_OPTIONS, key=f"t3pos_{eid}", horizontal=True,
            )

        if has_C:
            criticisms_text = extract_section(record.get("summary_A", ""), "## Criticisms")
            st.markdown("#### Criticisms section (A) vs. Negative-only summary (C)")
            col_l, col_r = st.columns(2)
            with col_l:
                st.markdown("**A — Criticisms section:**")
                st.markdown(
                    f'<div class="summary-box">{_md_to_html(criticisms_text or "_No criticisms section found._")}</div>',
                    unsafe_allow_html=True,
                )
            with col_r:
                st.markdown("**C — Negative-only summary:**")
                st.markdown(
                    f'<div class="summary-box">{_md_to_html(record["summary_C"])}</div>',
                    unsafe_allow_html=True,
                )
            task3_neg_align = st.radio(
                "Do A's Criticisms and summary C convey the same key negative points?",
                ALIGN_OPTIONS, key=f"t3neg_{eid}", horizontal=True,
            )

    # ── TASK 4 — Propagation Quality ─────────────────────────────────────────
    task4_prop = None
    parent_summary = record.get("_parent_summary")
    parent_name    = record.get("_parent_name")

    if parent_summary and record.get("node_level", 0) > 0:
        st.markdown("---")
        st.markdown("### Task 4 — Faithfulness Up the Tree")
        st.info(
            f"**Summary 1** (left) is for the specific topic **'{record['node_name']}'**. "
            f"**Summary 2** (right) is for its broader parent topic **'{parent_name}'**, "
            f"which was generated by combining Summary 1 with summaries from sibling topics.\n\n"
            f"Since Summary 2 is meant to contain the key information from Summary 1 "
            f"(among other things), check: **do the key points from Summary 1 appear "
            f"in Summary 2?** This measures whether faithfulness is preserved as "
            f"summaries propagate up the topic tree."
        )
        col_l, col_r = st.columns(2)
        with col_l:
            st.markdown(f"**Summary 1 — Specific topic:** {record['node_name']}")
            st.markdown(
                f'<div class="summary-box">'
                f'{_md_to_html(record.get("summary_A") or "_No summary._")}</div>',
                unsafe_allow_html=True,
            )
        with col_r:
            st.markdown(f"**Summary 2 — Broader parent topic:** {parent_name}")
            st.markdown(
                f'<div class="summary-box">{_md_to_html(parent_summary)}</div>',
                unsafe_allow_html=True,
            )
        task4_prop = st.radio(
            "Do the key points from Summary 1 appear in Summary 2?",
            PROP_OPTIONS, key=f"t4_{eid}", horizontal=True,
        )

    # ── Comments + Submit ─────────────────────────────────────────────────────
    st.markdown("---")
    comments = st.text_area(
        "💬 Additional comments (optional):",
        key=f"comments_{eid}",
        placeholder="Any observations about these summaries, or issues you noticed…",
        height=80,
    )

    st.markdown("")
    col_btn, col_info = st.columns([2, 3])
    with col_btn:
        submit = st.button("✅  Submit & Continue →", type="primary", use_container_width=True)
    with col_info:
        remaining = len(pending) - 1
        if remaining > 0:
            st.caption(f"{remaining} topic(s) remaining after this one.")
        else:
            st.caption("This is your last topic!")

    if submit:
        if faith is None or cover is None or coher is None:
            st.error("Please complete all three ratings in Task 2 before submitting.")
        else:
            response = {
                "evaluator_id":      st.session_state.evaluator_id,
                "evaluator_name":    st.session_state.evaluator_name,
                "eval_id":           eid,
                "course":            record["course"],
                "node_name":         record["node_name"],
                "node_type":         record["node_type"],
                "node_level":        record["node_level"],
                "task1_guess":       st.session_state.current_guess,
                "task1_skipped":     skip_t1,
                "task2_faithfulness": _parse_rating(faith),
                "task2_coverage":    _parse_rating(cover),
                "task2_coherence":   _parse_rating(coher),
                "task3_pos_alignment": task3_pos_align,
                "task3_neg_alignment": task3_neg_align,
                "task4_propagation": _parse_rating(task4_prop) if task4_prop else None,
                "comments":          comments,
            }
            save_response(response)

            # Advance to next pending record
            st.session_state.pending_records  = pending[1:]
            st.session_state.guess_submitted  = False
            st.session_state.current_guess    = 5
            st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════
def main() -> None:
    if not st.session_state.logged_in:
        show_login()
    elif st.session_state.completed or not st.session_state.pending_records:
        show_completion()
    else:
        show_evaluation()


main()
