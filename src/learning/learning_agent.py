"""
Phase 9 — Learning Agent (Logging + Feedback).

Multi-Agent Customer Support Intelligence Platform

Logs every ticket's full trace to SQLite on every exit path. Human decisions
(approve / edit / reject, plus corrected category and priority) are saved
afterwards with update_human_decision(). The offline scripts
index_from_feedback.py and retrain_from_feedback.py then use that feedback
to re-index the knowledge base and retrain the classifiers, with a metric
gate that keeps or rolls back the new model. They run on demand, not on a
schedule.

    START -> log -> END

Table: tickets_log (created on first run if missing), at data/logs/tickets.db.
Upserts by ticket_id, so re-running the same ticket_id (e.g. a demo re-run)
overwrites rather than duplicates.
"""

import json
import logging
import sqlite3
from pathlib import Path
from typing import Optional, TypedDict

from langgraph.graph import END, START, StateGraph
import csv
log = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).parent.parent.parent
DB_PATH = ROOT_DIR / "data" / "logs" / "tickets.db"
CSV_PATH = ROOT_DIR / "data" / "processed" / "tickets_for_retraining.csv"
CSV_COLUMNS = [
    "ticket_id", "ticket_text_clean", "predicted_category", "predicted_priority",
    "intent", "sentiment", "complexity", "response_text", "escalation_decision",
    "human_action", "human_verified_category", "human_verified_priority",
]
MAX_TEXT_LEN = 2000  # defensive cap; Response's own word-limit guardrail
                      # already bounds response_text, ticket_text_clean has no cap upstream

SCHEMA = """
CREATE TABLE IF NOT EXISTS tickets_log (
    ticket_id TEXT PRIMARY KEY,
    logged_at TEXT DEFAULT CURRENT_TIMESTAMP,
    ticket_text_clean TEXT,
    intent TEXT,
    sentiment TEXT,
    category TEXT,
    priority TEXT,
    complexity TEXT,
    team TEXT,
    retrieval_confidence TEXT,
    retrieval_top_source TEXT,
    retrieval_attempts INTEGER,
    retrieval_error INTEGER,
    response_text TEXT,
    response_mode TEXT,
    response_needs_review INTEGER,
    escalation_decision TEXT,
    escalation_reasons TEXT,
    jira_issue_key TEXT,
    human_action TEXT,
    final_sent_text TEXT,
    human_verified_category TEXT,
    human_verified_priority TEXT
)
"""


class LearningState(TypedDict, total=False):
    ticket_id: str
    ticket_text_clean: str
    intent: str
    sentiment: str
    category: str
    priority: str
    complexity: str
    team: str
    retrieval_confidence: str
    retrieval_top_source: Optional[str]
    retrieval_attempts: int
    retrieval_error: bool
    response_text: str
    response_mode: str
    response_needs_review: bool
    escalation_decision: str
    escalation_reasons: list
    jira_issue_key: Optional[str]
    logged: bool


def _ensure_columns(conn):
    """CREATE TABLE IF NOT EXISTS never adds columns to a table that already
    exists, so a tickets.db created before the reviewer-correction change
    needs the two new columns added in place (existing rows keep NULL)."""
    existing = {r[1] for r in conn.execute("PRAGMA table_info(tickets_log)")}
    for col in ("human_verified_category", "human_verified_priority"):
        if col not in existing:
            conn.execute(f"ALTER TABLE tickets_log ADD COLUMN {col} TEXT")
    conn.commit()


def get_connection():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute(SCHEMA)
    _ensure_columns(conn)
    return conn


def _truncate(text, max_len=MAX_TEXT_LEN):
    if not text or not isinstance(text, str):
        return text
    return text[:max_len]


def log_node(state):
    try:
        conn = get_connection()
        with conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO tickets_log (
                    ticket_id, ticket_text_clean, intent, sentiment, category,
                    priority, complexity, team, retrieval_confidence,
                    retrieval_top_source, retrieval_attempts, retrieval_error,
                    response_text, response_mode, response_needs_review,
                    escalation_decision, escalation_reasons, jira_issue_key
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    state.get("ticket_id"),
                    _truncate(state.get("ticket_text_clean")),
                    state.get("intent"),
                    state.get("sentiment"),
                    state.get("category"),
                    state.get("priority"),
                    state.get("complexity"),
                    state.get("team"),
                    state.get("retrieval_confidence"),
                    state.get("retrieval_top_source"),
                    state.get("retrieval_attempts"),
                    int(bool(state.get("retrieval_error"))),
                    _truncate(state.get("response_text")),
                    state.get("response_mode"),
                    int(bool(state.get("response_needs_review"))),
                    state.get("escalation_decision"),
                    json.dumps(state.get("escalation_reasons") or []),
                    state.get("jira_issue_key"),
                ),
            )
        conn.close()
        if state.get("escalation_decision") == "auto_resolve":
            _write_csv_row(
                (
                    state.get("ticket_id"), state.get("ticket_text_clean"),
                    state.get("category"), state.get("priority"), state.get("intent"),
                    state.get("sentiment"), state.get("complexity"),
                    state.get("response_text"), state.get("escalation_decision"), None,
                ),
            )
        return {"logged": True}
    except Exception as e:
        log.error(f"Learning agent failed to log ticket {state.get('ticket_id')}: {e}")
        return {"logged": False}
    
def update_human_decision(ticket_id, human_action, final_sent_text,
                          corrected_category=None, corrected_priority=None):
    """Phase 2: called once a human (simulated in Streamlit) approves or
    edits an escalated ticket's draft. Updates the existing row in place
    rather than inserting a new one, then writes the CSV row now that
    this ticket's story is finished.

    Verified labels (what retrain_from_feedback.py trains on), decided
    field by field:
      - the reviewer supplied a corrected value  -> that value (a real correction)
      - no correction AND human_action == "approved" -> the pipeline's own
        prediction, now confirmed by a human
      - no correction AND the reply was rejected/edited -> left blank.
        Rejecting the reply doesn't say whether the classification was
        right, so it is never treated as verified (better to lose a row
        than train on an unconfirmed label).
    """
    try:
        conn = get_connection()
        with conn:
            predicted = conn.execute(
                "SELECT category, priority FROM tickets_log WHERE ticket_id = ?",
                (ticket_id,),
            ).fetchone()
            row = None
            if predicted is not None:
                approved = human_action == "approved"
                verified_category = corrected_category or (predicted[0] if approved else None) or ""
                verified_priority = corrected_priority or (predicted[1] if approved else None) or ""
                conn.execute(
                    "UPDATE tickets_log SET human_action = ?, final_sent_text = ?, "
                    "human_verified_category = ?, human_verified_priority = ? "
                    "WHERE ticket_id = ?",
                    (human_action, _truncate(final_sent_text),
                     verified_category or None, verified_priority or None, ticket_id),
                )
                row = conn.execute(
                    "SELECT ticket_id, ticket_text_clean, category, priority, intent, "
                    "sentiment, complexity, response_text, escalation_decision, human_action "
                    "FROM tickets_log WHERE ticket_id = ?",
                    (ticket_id,),
                ).fetchone()
        conn.close()
        if predicted is None:
            log.error(f"No logged row for ticket {ticket_id}; human decision not recorded.")
            return {"updated": False}
        _write_csv_row(row, verified_category, verified_priority)
        return {"updated": True}
    except Exception as e:
        log.error(f"Failed to update human decision for ticket {ticket_id}: {e}")
        return {"updated": False}


def _write_csv_row(row, verified_category="", verified_priority=""):
    """Writes one finished ticket's row to the retraining CSV.
    predicted_* columns always hold the pipeline's own output. The
    human_verified_* columns hold only what a human confirmed or corrected
    (see update_human_decision) and stay blank for auto_resolve rows, since
    that's the pipeline's own prediction, not a human judgment."""
    (ticket_id, ticket_text_clean, category, priority, intent,
     sentiment, complexity, response_text, escalation_decision, human_action) = row

    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    is_new = not CSV_PATH.exists()
    with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if is_new:
            writer.writeheader()
        writer.writerow({
            "ticket_id": ticket_id,
            "ticket_text_clean": ticket_text_clean,
            "predicted_category": category,
            "predicted_priority": priority,
            "intent": intent,
            "sentiment": sentiment,
            "complexity": complexity,
            "response_text": response_text,
            "escalation_decision": escalation_decision,
            "human_action": human_action or "",
            "human_verified_category": verified_category or "",
            "human_verified_priority": verified_priority or "",
        })


def build_learning_graph():
    graph = StateGraph(LearningState)
    graph.add_node("log", log_node)
    graph.add_edge(START, "log")
    graph.add_edge("log", END)
    return graph.compile()


if __name__ == "__main__":
    graph = build_learning_graph()
    result = graph.invoke({
        "ticket_id": "DEMO-LOG-1",
        "ticket_text_clean": "my order hasn't arrived in 10 days",
        "intent": "report_delivery_issue",
        "sentiment": "Slightly Negative",
        "category": "Delivery Issue",
        "priority": "Medium",
        "complexity": "Low",
        "team": "Logistics",
        "retrieval_confidence": "high",
        "retrieval_top_source": "faq",
        "retrieval_attempts": 1,
        "retrieval_error": False,
        "response_text": "sample grounded reply",
        "response_mode": "grounded",
        "response_needs_review": False,
        "escalation_decision": "auto_resolve",
        "escalation_reasons": [],
        "jira_issue_key": None,
    })
    print(result)

    conn = sqlite3.connect(DB_PATH)
    row = conn.execute("SELECT * FROM tickets_log WHERE ticket_id = ?", ("DEMO-LOG-1",)).fetchone()
    print("\nRow in DB:", row)
    conn.close()