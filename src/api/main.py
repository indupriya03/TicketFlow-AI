"""
Phase 10 — FastAPI Backend
Multi-Agent Customer Support Intelligence Platform

Thin HTTP wrapper around orchestrator.py — matches the project spec's
End-to-End Flow (Streamlit UI -> FastAPI Backend -> Agent Orchestrator).
No pipeline logic lives here; every endpoint just calls start_ticket/
resume_ticket and returns the result.

ticket_id generation happens HERE (POST /ticket), not in Streamlit —
one HTTP request = one ticket_id, no rerun-duplication risk the way a
bare Streamlit script would have (see the session_state discussion).

In-memory _latest_results cache exists only to serve GET /ticket/{id}
for tickets still awaiting a customer follow-up (not yet in SQLite,
since Learning only logs a ticket once it reaches a terminal state).
Completed tickets are read from SQLite via Learning's own table instead,
which is durable across restarts; the in-memory cache is not.
"""

import uuid

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.orchestrator import start_ticket, resume_ticket
from src.learning.learning_agent import get_connection, update_human_decision

app = FastAPI(title="Ticket-Flow-AI API")

_latest_results = {}  # ticket_id -> last known result dict (in-memory, see docstring)

LOG_COLUMNS = [
    "ticket_id", "logged_at", "ticket_text_clean", "intent", "sentiment",
    "category", "priority", "complexity", "team", "retrieval_confidence",
    "retrieval_top_source", "retrieval_attempts", "retrieval_error",
    "response_text", "response_mode", "response_needs_review",
    "escalation_decision", "escalation_reasons", "jira_issue_key",
    "human_action", "final_sent_text","human_verified_category", "human_verified_priority",
]


class TicketRequest(BaseModel):
    ticket_text: str
    product_name: str | None = None
    product_segment: str | None = None

class ReviewRequest(BaseModel):
    action: str                       # "approved" | "rejected_edited"
    final_text: str | None = None
    corrected_category: str | None = None
    corrected_priority: str | None = None   


class ReplyRequest(BaseModel):
    reply: str


def _row_to_dict(row):
    return dict(zip(LOG_COLUMNS, row))


def _fetch_log_row(ticket_id):
    conn = get_connection()
    row = conn.execute(
        f"SELECT {', '.join(LOG_COLUMNS)} FROM tickets_log WHERE ticket_id = ?",
        (ticket_id,),
    ).fetchone()
    conn.close()
    return _row_to_dict(row) if row else None


@app.post("/ticket")
def submit_ticket(req: TicketRequest):
    """Create a new ticket and run it through the pipeline. Generates the
    ticket_id here — the one place in the system that should."""
    ticket_id = f"TICKET-{uuid.uuid4().hex[:8]}"
    result = start_ticket(ticket_id, req.ticket_text, req.product_name, req.product_segment)
    result.setdefault("ticket_id", ticket_id)
    _latest_results[ticket_id] = result
    return result


@app.post("/process/{ticket_id}")
def continue_ticket(ticket_id: str, req: ReplyRequest):
    """Continue an existing ticket after a customer follow-up reply."""
    if ticket_id not in _latest_results:
        raise HTTPException(status_code=404, detail="Unknown ticket_id")
    result = resume_ticket(ticket_id, req.reply)
    result.setdefault("ticket_id", ticket_id)
    _latest_results[ticket_id] = result
    return result


@app.get("/ticket/{ticket_id}")
def get_ticket(ticket_id: str):
    """Fetch the most recent known state of a ticket — in-memory cache
    first (covers awaiting_customer, not yet logged), falling back to
    SQLite (covers completed tickets, durable across restarts)."""
    if ticket_id in _latest_results:
        return _latest_results[ticket_id]

    log_row = _fetch_log_row(ticket_id)
    if log_row is None:
        raise HTTPException(status_code=404, detail="Unknown ticket_id")
    return log_row


@app.get("/logs/{ticket_id}")
def get_logs(ticket_id: str):
    """Agent decision logs for one ticket, from the Learning agent's own table."""
    log_row = _fetch_log_row(ticket_id)
    if log_row is None:
        raise HTTPException(status_code=404, detail="No logs for this ticket_id")
    return log_row

@app.post("/ticket/{ticket_id}/review")
def review_ticket(ticket_id: str, req: ReviewRequest):
    """Human agent's decision on an escalated ticket (approve the draft, or
    reject it and send a corrected reply, optionally correcting category/priority)."""
    res = update_human_decision(
        ticket_id, req.action, req.final_text,
        corrected_category=req.corrected_category,
        corrected_priority=req.corrected_priority,
    )
    if not res.get("updated"):
        raise HTTPException(status_code=404, detail="Decision not saved (unknown ticket_id or update failed)")
    return res