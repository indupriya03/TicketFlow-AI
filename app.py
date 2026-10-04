"""
TicketFlow-AI — Streamlit Demo (three-panel chat UI)
Multi-Agent Customer Support Intelligence Platform

Layout:
  left   (sidebar)   product context, session stats, integrations, new conversation
  middle             the chatbot: clean chat history + input, like a real support bot
  right              pipeline analysis for the selected ticket: live processing
                     steps, confidence, human handoff (Approve / Reject), Jira
                     card, and the expandable Agent Logs

Talks to the FastAPI backend (src/api/main.py) over HTTP; run the API first:
    uvicorn src.api.main:app --port 8000

The pipeline is one blocking call, so the live processing steps are replayed
with a short delay AFTER it returns; only stages that actually ran are listed.
Ticket IDs are created once per submission and kept in the message history.
"""

import html
import json
import os
import time
import traceback
from unittest import result
import uuid
from datetime import datetime
from pathlib import Path
import requests
import pandas as pd
import streamlit as st
from src.response.templates import escalated_text, high_priority_text


ROOT_DIR = Path(__file__).parent
PRODUCTS_PATH = ROOT_DIR / "data/processed/support_tickets_clean.csv"

# Labels the reviewer can pick when correcting the pipeline's classification.
# Keep CATEGORY_LABELS in sync with the classes in category_train.py.
CATEGORY_LABELS = [
    "Account & Login", "App & Website Issue", "Delivery Issue", "Payment Issue",
    "Product Issue", "Refund & Return", "Seller & Product Listing",
]
PRIORITY_LABELS = ["Low", "Medium", "High"]
STEP_DELAY = 0.35        # seconds between replayed processing steps
CHAT_HEIGHT = 500        # px, scrollable chat window
PANEL_HEIGHT = 570       # px, scrollable analysis panel

EXAMPLES = [
    ("📦 Late delivery", "Where is my order? It was supposed to arrive 3 days ago and the tracking hasn't updated."),
    ("💶 Big refund", "I want a refund of 120 euros, the item arrived broken."),
    ("😡 Angry customer", "This is unacceptable! I was charged twice and nobody is helping me."),
]

STAGES = [
    ("Intake", "intake"), ("Triage", "triage"), ("Retrieval", "retrieval"),
    ("Response", "response"), ("Escalation", "escalation"), ("Learning", "learning"),
]

REASON_TEXT = {
    "response_needs_review": "The drafted reply needs a human check (weak or missing evidence)",
    "complexity=High": "The ticket was scored as high complexity",
    "refund_amount_unstated": "Refund request with no amount stated",
    "prior_contact_unresolved": "Customer says they were already ignored or got no reply, so a human should check the draft before it is sent",
    "escalation_logic_error": "The escalation check itself errored, so it failed safe to a human",
}

LOG_COLUMNS = [
    "ticket_id", "logged_at", "ticket_text_clean", "intent", "sentiment",
    "category", "priority", "complexity", "team", "retrieval_confidence",
    "retrieval_top_source", "retrieval_attempts", "retrieval_error",
    "response_text", "response_mode", "response_needs_review",
    "escalation_decision", "escalation_reasons", "jira_issue_key",
    "human_action", "final_sent_text",
    "human_verified_category", "human_verified_priority",
]

API_URL = os.environ.get("TICKETFLOW_API_URL", "http://127.0.0.1:8000")
API_DOWN_MSG = (f"Cannot reach the API at {API_URL}. "
                "Start it with: uvicorn src.api.main:app --port 8000")

# ------------------------------------------------------------------ data helpers

@st.cache_data
def load_product_options():
    df = pd.read_csv(PRODUCTS_PATH, usecols=["product_name", "product_segment"]).dropna()
    segments = sorted(df["product_segment"].unique().tolist())
    by_segment = {
        seg: sorted(df.loc[df["product_segment"] == seg, "product_name"].unique().tolist())
        for seg in segments
    }
    return segments, by_segment

def _api(method, path, **kwargs):
    try:
        return requests.request(method, f"{API_URL}{path}", **kwargs)
    except requests.ConnectionError:
        raise RuntimeError(API_DOWN_MSG)


def api_start_ticket(text, product_name, product_segment):
    r = _api("POST", "/ticket", timeout=120, json={
        "ticket_text": text, "product_name": product_name, "product_segment": product_segment})
    r.raise_for_status()
    return r.json()


def api_resume_ticket(ticket_id, reply):
    r = _api("POST", f"/process/{ticket_id}", timeout=120, json={"reply": reply})
    r.raise_for_status()
    return r.json()


def api_review(ticket_id, action, text, corrected_category, corrected_priority):
    r = _api("POST", f"/ticket/{ticket_id}/review", timeout=30, json={
        "action": action, "final_text": text,
        "corrected_category": corrected_category, "corrected_priority": corrected_priority})
    return r.status_code == 200 and bool(r.json().get("updated"))


def fetch_log_row(ticket_id):
    """The REAL logged row, read through the API's /logs endpoint (SQLite behind it)."""
    try:
        r = requests.get(f"{API_URL}/logs/{ticket_id}", timeout=10)
    except requests.RequestException:
        return None
    return r.json() if r.status_code == 200 else None


def jira_configured():
    return all(os.environ.get(k) for k in
               ("JIRA_URL", "JIRA_EMAIL", "JIRA_API_TOKEN", "JIRA_PROJECT_KEY"))


def jira_link(key):
    base = (os.environ.get("JIRA_URL") or "").rstrip("/")
    return f"{base}/browse/{key}" if base and key else None


def parse_reasons(log_row):
    try:
        return json.loads(log_row.get("escalation_reasons") or "[]")
    except (TypeError, ValueError):
        return []


def explain_reason(r):
    if r in REASON_TEXT:
        return REASON_TEXT[r]
    if r.startswith("sentiment="):
        return f"Customer sentiment is {r.split('=', 1)[1]}"
    if r.startswith("refund_amount="):
        try:
            amount, limit = r.split("=", 1)[1].split(">")
            return f"Refund amount {amount} is over the auto-approve limit of {limit}"
        except ValueError:
            pass
    return r


def v(x):
    return x if x not in (None, "") else "—"


def now():
    return datetime.now().strftime("%H:%M")


# ------------------------------------------------------------------ small UI helpers

def chip(text, color="grey"):
    return f'<span class="chip chip-{color}">{html.escape(str(text))}</span>'


def card():
    """Bordered container; older Streamlit versions without border= get a plain one."""
    try:
        return st.container(border=True)
    except TypeError:
        return st.container()


def confidence_color(conf):
    conf = float(conf)
    return "green" if conf >= 0.8 else "amber" if conf >= 0.5 else "red"


def priority_color(p):
    return {"High": "red", "Medium": "amber", "Low": "green"}.get(p, "grey")


def status_chip(msg, log_row):
    kind = msg["kind"]
    if kind == "escalated" and log_row and log_row.get("human_action"):
        label = "Resolved by agent" if log_row["human_action"] == "approved" else "Corrected by agent"
        return chip(label, "green")
    color = {"resolved": "green", "escalated": "amber"}.get(kind, "grey")
    return chip(msg["status"], color)


def summary_chips(msg, log_row):
    chips = [status_chip(msg, log_row)]
    if log_row:
        if log_row.get("category"):
            chips.append(chip(f"🏷 {log_row['category']}", "blue"))
        conf = msg.get("category_confidence")
        if conf is not None:
            chips.append(chip(f"{float(conf):.0%} confident", confidence_color(conf)))
        if log_row.get("priority"):
            chips.append(chip(f"Priority: {log_row['priority']}", priority_color(log_row["priority"])))
        if log_row.get("team"):
            chips.append(chip(f"Team: {log_row['team']}", "grey"))
    if msg.get("elapsed"):
        chips.append(chip(f"⏱ {msg['elapsed']:.1f}s", "grey"))
    return "".join(chips)


def stage_trace(msg, log_row):
    ran = msg.get("ran") or {}
    done = {
        "intake": True,
        "triage": bool(ran.get("triage")),
        "retrieval": bool(ran.get("retrieval")),
        "response": bool(ran.get("response")),
        "escalation": bool(log_row and log_row.get("escalation_decision")),
        "learning": log_row is not None,
    }
    return "  →  ".join(f"{'✅' if done[key] else '⏭️'} {name}" for name, key in STAGES)


# ------------------------------------------------------------------ middle: chat

def render_bubble(msg):
    if msg["role"] == "user":
        with st.chat_message("user"):
            st.markdown(msg["content"])
            st.caption(msg.get("ts", ""))
        return

    with st.chat_message("assistant", avatar="🤖"):
        st.markdown(msg["content"])
        st.caption(msg.get("ts", ""))

    # Once a human agent has decided, their reply appears in the chat.
    if msg.get("kind") == "escalated":
        row = fetch_log_row(msg["ticket_id"])
        if row and row.get("human_action"):
            with st.chat_message("assistant", avatar="🧑‍💼"):
                st.markdown(row["final_sent_text"] or "")
                st.caption("Support agent")


def render_chat(show_welcome):
    if show_welcome:
        with st.chat_message("assistant", avatar="🤖"):
            st.markdown(
                "Hi! Tell me what's wrong and I'll look into it. If it's something I "
                "shouldn't decide on my own, I'll pass it to a human agent."
            )
            st.caption("Try an example:")
            for label, text in EXAMPLES:
                if st.button(label, key=f"example_{label}"):
                    st.session_state.pending_prompt = text
                    st.rerun()
    for msg in st.session_state.messages:
        render_bubble(msg)


# ------------------------------------------------------------------ right: analysis panel

def render_metrics(msg, log_row):
    conf = msg.get("category_confidence")
    m1, m2, m3 = st.columns(3)
    m1.metric("Category confidence", f"{float(conf):.1%}" if conf is not None else "n/a")
    m2.metric("Priority", v(log_row["priority"]))
    m3.metric("Complexity", v(log_row["complexity"]))
    if conf is not None:
        st.progress(min(max(float(conf), 0.0), 1.0))
    rc = log_row.get("retrieval_confidence")
    if rc:
        st.markdown(
            chip(f"Retrieval: {rc}", "green" if rc == "high" else "amber")
            + chip(f"Source: {v(log_row.get('retrieval_top_source'))}", "grey"),
            unsafe_allow_html=True,
        )


def render_jira(msg, log_row):
    key = log_row.get("jira_issue_key") or msg.get("jira_issue_key")
    reasons = parse_reasons(log_row)
    team = v(log_row.get("team"))
    with card():
        st.markdown("##### 🎫 Jira escalation")
        if key:
            url = jira_link(key)
            link = f"[{key}]({url})" if url else f"`{key}`"
            st.markdown(f"Issue filed: **{link}**  ·  Team: **{team}**")
        else:
            st.warning(
                "No Jira issue was created for this ticket. Either the Jira environment "
                "variables aren't set, or the Jira API call failed — check the terminal "
                "log for 'Jira create_issue failed'."
            )
        if reasons:
            st.markdown("**Why it was escalated**\n"
                        + "\n".join(f"- {explain_reason(r)}" for r in reasons))
        st.caption(
            f"What happened: the Escalation agent's rules fired → a Jira issue (type: Request) "
            f"was created for the **{team}** team with the customer's ticket and the drafted "
            f"reply attached → a human on that team reviews it. Their decision is logged and "
            f"feeds retraining."
        )


def render_handoff(msg, log_row):
    """Pending human review: editable reply, classification correction, Approve / Reject."""
    tid = msg["ticket_id"]
    draft = log_row.get("response_text") or msg.get("draft") or ""
    with card():
        st.markdown("##### 👤 Human handoff — support agent view (simulated)")
        st.caption("In production this lands in the team's Jira queue; here you play the agent.")

        edited = st.text_area(
            "Reply to the customer" if draft else "No draft was generated — write the reply",
            value=draft, key=f"edit_{tid}", height=130,
        )

        st.caption("Classification: change only if the pipeline got it wrong. "
                   "'(no change)' keeps the prediction.")
        c1, c2 = st.columns(2)
        new_cat = c1.selectbox(
            f"Category (predicted: {v(log_row['category'])})",
            ["(no change)"] + CATEGORY_LABELS, key=f"cat_{tid}",
        )
        new_pri = c2.selectbox(
            f"Priority (predicted: {v(log_row['priority'])})",
            ["(no change)"] + PRIORITY_LABELS, key=f"pri_{tid}",
        )
        corrected_category = None if new_cat == "(no change)" else new_cat
        corrected_priority = None if new_pri == "(no change)" else new_pri

        changed = bool(edited.strip()) and edited.strip() != draft.strip()
        b1, b2 = st.columns(2)
        approve = b1.button("✅ Approve & Send", key=f"approve_{tid}",
                            disabled=not draft, type="primary")
        reject = b2.button("❌ Reject & Send Correction", key=f"reject_{tid}",
                           disabled=not changed)
        st.caption("Approve sends the draft as written. To send your own wording, edit the "
                   "reply above and use Reject & Send Correction.")

        action = text = None
        if approve:
            action, text = "approved", draft
        elif reject:
            action, text = "rejected_edited", edited.strip()
        if action:
            if api_review(tid, action, text, corrected_category, corrected_priority):
                st.rerun()
            else:
                st.error("Could not save the decision — see the API terminal log.")

def render_agent_logs(log_row):
    with st.expander("🔍 Agent Logs", expanded=True):
        left, right = st.columns(2)
        left.markdown(
            "**📥 Intake**\n"
            f"- Intent: `{v(log_row['intent'])}`\n"
            f"- Sentiment: `{v(log_row['sentiment'])}`\n\n"
            "**🏷️ Triage**\n"
            f"- Category: `{v(log_row['category'])}`\n"
            f"- Team: `{v(log_row['team'])}`"
        )
        if log_row["retrieval_confidence"]:
            retrieval = (f"- Confidence: `{log_row['retrieval_confidence']}`\n"
                         f"- Top source: `{v(log_row['retrieval_top_source'])}`, "
                         f"attempts: {v(log_row['retrieval_attempts'])}")
        else:
            retrieval = "- Skipped"
        if log_row["response_mode"]:
            response = (f"- Mode: `{log_row['response_mode']}`\n"
                        f"- Needs review: `{bool(log_row['response_needs_review'])}`")
        else:
            response = "- Skipped"
        reasons = ", ".join(parse_reasons(log_row)) or "none"
        right.markdown(
            f"**📚 Retrieval**\n{retrieval}\n\n"
            f"**✍️ Response**\n{response}\n\n"
            "**🚦 Escalation**\n"
            f"- Decision: `{v(log_row['escalation_decision'])}`\n"
            f"- Reasons: `{reasons}`"
            + (f"\n- Jira: `{log_row['jira_issue_key']}`" if log_row["jira_issue_key"] else "")
        )
        if log_row["human_action"]:
            st.markdown(
                "**👤 Human decision**\n"
                f"- Action: `{log_row['human_action']}`\n"
                f"- Labels saved for retraining: category=`{v(log_row['human_verified_category'])}`, "
                f"priority=`{v(log_row['human_verified_priority'])}`"
            )
        st.caption("Raw log record")
        st.json(log_row, expanded=False)


def render_panel():
    st.markdown("#### 🔍 Pipeline analysis")
    tickets = [m for m in st.session_state.messages
               if m["role"] == "assistant" and m.get("kind") in ("resolved", "escalated")]

    if st.session_state.awaiting_thread:
        st.info("Waiting for the customer to answer the follow-up question. "
                "The ticket is classified once there's enough detail.")
    if not tickets:
        if not st.session_state.awaiting_thread:
            st.caption("Submit a ticket to see how each agent handled it: confidence, "
                       "human handoff, Jira and the Agent Logs.")
        return

    ids = [m["ticket_id"] for m in tickets]
    by_id = {m["ticket_id"]: m for m in tickets}
    focus = ids[-1]  # newest ticket unless the user picks another one
    if len(ids) > 1:
        labels = []
        for tid in ids:
            row = fetch_log_row(tid)
            text = (row["ticket_text_clean"] if row else "") or ""
            labels.append(f"{tid[-8:]} · {text[:38]}{'…' if len(text) > 38 else ''}")
        # index depends only on how many tickets exist, so the widget keeps its
        # identity (and the user's pick) until a new ticket arrives, which then
        # re-defaults the selector to the newest one.
        picked = st.selectbox("Ticket", labels, index=len(ids) - 1)
        focus = ids[labels.index(picked)]

    msg = by_id[focus]
    log_row = fetch_log_row(focus)
    if not log_row:
        st.caption("Not logged yet.")
        return
    decided = bool(log_row.get("human_action"))

    st.markdown(summary_chips(msg, log_row), unsafe_allow_html=True)
    st.caption(stage_trace(msg, log_row))
    render_metrics(msg, log_row)
    if msg["kind"] == "escalated":
        if not decided:
            render_handoff(msg, log_row)
        render_jira(msg, log_row)
    render_agent_logs(log_row)


# ------------------------------------------------------------------ pipeline run

def interpret(result):
    """Customer-facing text, message kind and status label for one pipeline result."""
    if result.get("status") == "awaiting_customer":
        return "followup", result.get("message", ""), "Needs more info"
    if result.get("escalation_decision") == "no_action_needed":
        return "resolved", result.get("response_text", ""), "Handled automatically"
    if not result.get("triage_ran"):
        return ("escalated",
                "I couldn't classify your request automatically, so I've routed it to a "
                "human agent who will follow     up.", "Routed to human")
    if result.get("response_ran") and result.get("escalation_decision") == "auto_resolve":
        return "resolved", result.get("response_text", ""), "Auto-resolved"
    if result.get("response_ran"):
        return ("escalated", escalated_text(result.get("sentiment")), "Escalated — pending human review")    
    return ("escalated", high_priority_text(result.get("sentiment")), "Escalated — high priority")

def run_with_steps(call):
    """Runs the pipeline inside a live status box. Steps are replayed after the
    blocking call returns; only stages that actually ran are listed."""
    status = st.status("Processing your ticket…", expanded=True)
    status.write("📥 **Intake** — reading the ticket, detecting intent and sentiment")
    started = time.perf_counter()
    try:
        result = call()
    except Exception as e:  # keep the demo alive on API/key problems
        traceback.print_exc()  # full traceback goes to the terminal running Streamlit
        status.update(label="Pipeline error", state="error", expanded=True)
        return None, 0.0, f"{type(e).__name__}: {e}"
    elapsed = time.perf_counter() - started

    if result.get("status") == "awaiting_customer":
        time.sleep(STEP_DELAY)
        status.write("❓ Not enough detail yet — asking a follow-up question")
        status.update(label=f"Waiting for the customer · {elapsed:.1f}s", state="complete", expanded=False)
        return result, elapsed, None

    steps = [
        ("triage_ran", "🏷️ **Triage** — category, priority, complexity, team"),
        ("retrieval_ran", "📚 **Retrieval** — searching FAQs and past resolutions"),
        ("response_ran", "✍️ **Response** — drafting a grounded reply"),
    ]
    for flag, text in steps:
        if result.get(flag):
            time.sleep(STEP_DELAY)
            status.write(text)
    time.sleep(STEP_DELAY)
    status.write("🚦 **Escalation** — checking whether a human must review this")
    if result.get("escalation_decision") == "escalate_to_team" or not result.get("triage_ran"):
        time.sleep(STEP_DELAY)
        key = result.get("jira_issue_key")
        status.write(f"🎫 Jira issue filed: `{key}`" if key else "🎫 Escalated to the support team")
    time.sleep(STEP_DELAY)
    status.write("🗂️ **Learning** — logging the full trace")
    status.update(label=f"Done in {elapsed:.1f}s", state="complete", expanded=False)
    return result, elapsed, None


def handle_prompt(prompt, product_name, product_segment, chat_box, panel_box):
    ts = now()
    ACKNOWLEDGMENTS = {
        "thanks", "thank you", "thanks!", "thank you!", "ok", "okay", "got it",
        "great", "perfect", "sounds good", "cool", "appreciate it", "awesome", "alright",
    }

    def is_acknowledgment(text):
        return text.strip().lower().strip(".!") in ACKNOWLEDGMENTS
    st.session_state.messages.append({"role": "user", "content": prompt, "ts": ts})
    with chat_box:
        with st.chat_message("user"):
            st.markdown(prompt)
            st.caption(ts)
        with st.chat_message("assistant", avatar="🤖"):
            st.markdown("_Typing…_")
    if is_acknowledgment(prompt):
        with chat_box:
            with st.chat_message("assistant", avatar="🤖"):
                st.markdown("You're welcome! Let us know if there's anything else.")
        st.session_state.messages.append({
            "role": "assistant", "kind": "smalltalk",
            "content": "You're welcome! Let us know if there's anything else.",
            "status": "Acknowledged", "ticket_id": None, "ts": now(),
        })
        st.rerun()
        return
    thread_id = st.session_state.awaiting_thread
    if thread_id:
        ticket_id = st.session_state.current_ticket_id
        call = lambda: api_resume_ticket(thread_id, prompt)
    else:
        ticket_id = f"TICKET-{uuid.uuid4().hex[:8]}"  # placeholder, used only if the call fails
        call = lambda: api_start_ticket(
            prompt,
            None if product_name == "(none)" else product_name,
            None if product_segment == "(none)" else product_segment,
        )

    with panel_box:
        st.markdown("#### 🔍 Pipeline analysis")
        result, elapsed, error = run_with_steps(call)

    if error:
        hint = ""
        if any(w in error for w in ("Connection", "Timeout", "timed out")):
            hint = ("\n\nThis looks like a network problem reaching an external service "
                    "(Groq or Gemini). Check your internet/VPN, then send your message again.")
        # awaiting_thread is left as it was: if this was a follow-up reply, the next
        # message retries it instead of starting a brand-new ticket.
        st.session_state.messages.append({
            "role": "assistant", "kind": "error", "ticket_id": ticket_id, "ts": now(),
            "content": f"⚠️ Something went wrong while processing this ticket: `{error}`{hint}",
            "status": "Error",
        })
        st.rerun()
    ticket_id = result.get("ticket_id") or ticket_id
    kind, content, status_label = interpret(result)
    st.session_state.messages.append({
        "role": "assistant", "kind": kind, "content": content, "status": status_label,
        "ticket_id": ticket_id, "elapsed": elapsed, "ts": now(),
        "category_confidence": result.get("category_confidence"),
        "jira_issue_key": result.get("jira_issue_key"),
        "draft": result.get("response_text"),
        "ran": {"triage": result.get("triage_ran"), "retrieval": result.get("retrieval_ran"),
                "response": result.get("response_ran")},
    })
    st.session_state.awaiting_thread = result.get("thread_id") if kind == "followup" else None
    st.session_state.current_ticket_id = ticket_id if kind == "followup" else None
    st.rerun()


# ------------------------------------------------------------------ page

st.set_page_config(page_title="TicketFlow-AI", page_icon="🎫", layout="wide")

st.markdown("""
<style>
.block-container {max-width: 1500px; padding-top: 1rem;}
.hero {padding: .7rem 1.2rem; border-radius: 12px; margin-bottom: .8rem; color: #fff;
       background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);}
.hero h1 {margin: 0; font-size: 1.35rem; color: #fff; padding: 0;}
.hero p {margin: .15rem 0 0; opacity: .92; font-size: .82rem;}
.chip {display: inline-block; padding: 2px 10px; margin: 6px 6px 2px 0; border-radius: 999px;
       font-size: .75rem; font-weight: 600;}
.chip-green {background: rgba(34,197,94,.16); color: #15803d;}
.chip-amber {background: rgba(245,158,11,.18); color: #b45309;}
.chip-red   {background: rgba(239,68,68,.16); color: #dc2626;}
.chip-blue  {background: rgba(79,70,229,.14); color: #4f46e5;}
.chip-grey  {background: rgba(148,163,184,.22); color: #64748b;}
/* chat bubbles: rounded, and the customer's messages on the right */
[data-testid="stChatMessage"] {border-radius: 16px; padding: .6rem .9rem; margin-bottom: .35rem;}
[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
    flex-direction: row-reverse; background: rgba(79,70,229,.10);}
</style>
""", unsafe_allow_html=True)

segments, products_by_segment = load_product_options()

for key, default in [("messages", []), ("awaiting_thread", None),
                     ("current_ticket_id", None)]:
    if key not in st.session_state:
        st.session_state[key] = default

# ---- Left: sidebar ----
with st.sidebar:
    st.markdown("### 🎫 TicketFlow-AI")
    st.caption("Multi-agent customer support · LangGraph")

    st.markdown("**Optional ticket context**")
    segment = st.selectbox("Product segment", ["(none)"] + segments)
    product_options = products_by_segment.get(segment, []) if segment != "(none)" else []
    product_name = st.selectbox("Product name", ["(none)"] + product_options)

    st.divider()
    assistant_msgs = [m for m in st.session_state.messages
                      if m["role"] == "assistant" and m.get("kind") in ("resolved", "escalated")]
    resolved = sum(m["kind"] == "resolved" for m in assistant_msgs)
    escalated = [m for m in assistant_msgs if m["kind"] == "escalated"]
    pending = 0
    for m in escalated:
        row = fetch_log_row(m["ticket_id"])
        if row and not row.get("human_action"):
            pending += 1
    st.markdown("**This session**")
    s1, s2 = st.columns(2)
    s1.metric("Tickets", len(assistant_msgs))
    s2.metric("Auto-resolved", resolved)
    s3, s4 = st.columns(2)
    s3.metric("Escalated", len(escalated))
    s4.metric("Awaiting review", pending)

    st.divider()
    st.markdown("**Integrations**")
    if jira_configured():
        st.success(f"Jira connected · project {os.environ.get('JIRA_PROJECT_KEY')}")
    else:
        st.warning("Jira not configured — escalations won't create issues")

    if st.button("🆕 New conversation"):
        st.session_state.messages = []
        st.session_state.awaiting_thread = None
        st.session_state.current_ticket_id = None
        st.rerun()

# ---- Header ----
st.markdown(
    '<div class="hero"><h1>🎫 TicketFlow-AI</h1>'
    "<p>Intake → Triage → Retrieval → Response → Escalation → Learning · "
    "a human steps in when the system shouldn't decide alone.</p></div>",
    unsafe_allow_html=True,
)

# ---- Middle: chat, right: analysis ----
chat_col, panel_col = st.columns([3, 2], gap="medium")
with chat_col:
    chat_box = st.container(height=CHAT_HEIGHT)
    placeholder = ("Reply to the assistant's question…" if st.session_state.awaiting_thread
                   else "Type your message…")
    typed = st.chat_input(placeholder)
with panel_col:
    panel_box = st.container(height=PANEL_HEIGHT)

pending_prompt = st.session_state.pop("pending_prompt", None)
prompt = typed or pending_prompt

with chat_box:
    render_chat(show_welcome=not st.session_state.messages and not pending_prompt)

if prompt and prompt.strip():
    handle_prompt(prompt.strip(), product_name, segment, chat_box, panel_box)
else:
    with panel_box:
        render_panel()