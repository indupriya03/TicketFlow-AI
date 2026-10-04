"""
Phase 8 — Escalation Agent (LangGraph)
Multi-Agent Customer Support Intelligence Platform

Decides whether a ticket that reached Response can be auto-resolved, or
must be handed to a human on state["team"] before anything is sent.

    START -> decide -> END

NOT the same "Escalation" as assign_team.py's ESCALATION_TEAM. That check
routes a High-priority ticket to a human team BEFORE Retrieval/Response
ever run, purely from priority. This agent runs AFTER Response, for
tickets that were NOT already routed there, and decides whether the
drafted reply is safe to send automatically.

Escalates (does not auto-resolve) if ANY of:
  - response_needs_review is True (Response's own guardrails/weak-evidence flag)
  - sentiment is in ESCALATION_TRIGGER_SENTIMENTS ("Negative", "Very Negative")
  - complexity == "High" (assign_complexity's own top bucket)
  - REFUND_AUTO_APPROVE_LIMIT rule: intent is refund-related AND either no
    amount was stated, or the highest stated amount exceeds the limit.
    A refund ticket with no amount mentioned is treated as "unknown, so
    escalate" rather than assumed safe — see Known limitations.
  - entity_prior_contact_unresolved is True (customer says they were already
    ignored or got no reply; a human should see the draft before it is sent)

Inputs:  response_needs_review, sentiment, complexity, team, intent,
         entity_amounts
Outputs: escalation_decision ("auto_resolve" | "escalate_to_team"),
         escalation_reasons (list[str], empty when auto-resolved)
"""

from typing import Optional, TypedDict

from langgraph.graph import END, START, StateGraph

from src.intake.intake_sentiment import ESCALATION_TRIGGER_SENTIMENTS
from src.escalation.jira_tools import create_issue
import logging
log = logging.getLogger(__name__)

REFUND_AUTO_APPROVE_LIMIT = 50

REFUND_INTENTS = {
    "request_refund",
    "dispute_charge_or_refund",
}


class EscalationState(TypedDict, total=False):
    # inputs
    ticket_id: str
    ticket_text_clean: str
    response_text: str
    response_needs_review: bool
    sentiment: str
    complexity: str
    team: str
    intent: str
    entity_amounts: list
    entity_prior_contact_unresolved: bool
    # outputs
    escalation_decision: str
    escalation_reasons: list
    jira_issue_key: Optional[str]


def _check_refund_amount(intent, entity_amounts):
    if intent not in REFUND_INTENTS:
        return None
    if not entity_amounts:
        return "refund_amount_unstated"
    highest = max(entity_amounts)
    if highest > REFUND_AUTO_APPROVE_LIMIT:
        return f"refund_amount={highest}>{REFUND_AUTO_APPROVE_LIMIT}"
    return None


def decide_node(state):
    try:
        reasons = []
        if state.get("response_needs_review"):
            reasons.append("response_needs_review")
        if state.get("sentiment") in ESCALATION_TRIGGER_SENTIMENTS:
            reasons.append(f"sentiment={state.get('sentiment')}")
        if state.get("complexity") == "High":
            reasons.append("complexity=High")
        if state.get("entity_prior_contact_unresolved"):
            reasons.append("prior_contact_unresolved")

        refund_reason = _check_refund_amount(state.get("intent"), state.get("entity_amounts"))
        if refund_reason:
            reasons.append(refund_reason)
    except Exception as e:
        log.error(f"Escalation decision logic failed for ticket "
                  f"{state.get('ticket_id')}: {e}. Failing safe to escalate_to_team.")
        reasons = ["escalation_logic_error"]

    if not reasons:
        return {"escalation_decision": "auto_resolve", "escalation_reasons": [], "jira_issue_key": None}

    issue_key = create_issue(
        ticket_id=state.get("ticket_id"),
        team=state.get("team"),
        reasons=reasons,
        ticket_text=state.get("ticket_text_clean"),
        draft_reply=state.get("response_text"),
    )
    return {
        "escalation_decision": "escalate_to_team",
        "escalation_reasons": reasons,
        "jira_issue_key": issue_key,
    }

def build_escalation_graph():
    graph = StateGraph(EscalationState)
    graph.add_node("decide", decide_node)
    graph.add_edge(START, "decide")
    graph.add_edge("decide", END)
    return graph.compile()


if __name__ == "__main__":
    graph = build_escalation_graph()

    test_cases = [
        {"name": "clean, auto-resolve", "input": {
            "ticket_id": "TEST-1", "ticket_text_clean": "sample ticket text",
            "response_text": "sample draft reply",
            "response_needs_review": False, "sentiment": "Neutral",
            "complexity": "Low", "team": "Logistics",
            "intent": "check_return_or_order_status", "entity_amounts": [],
        }},
        {"name": "needs_review flag", "input": {
            "ticket_id": "TEST-1", "ticket_text_clean": "sample ticket text",
            "response_text": "sample draft reply",
            "response_needs_review": True, "sentiment": "Neutral",
            "complexity": "Low", "team": "Logistics",
            "intent": "report_delivery_issue", "entity_amounts": [],
        }},
        {"name": "negative sentiment", "input": {
            "ticket_id": "TEST-1", "ticket_text_clean": "sample ticket text",
            "response_text": "sample draft reply",
            "response_needs_review": False, "sentiment": "Negative",
            "complexity": "Low", "team": "Logistics",
            "intent": "report_delivery_issue", "entity_amounts": [],
        }},
        {"name": "high complexity", "input": {
            "ticket_id": "TEST-1", "ticket_text_clean": "sample ticket text",
            "response_text": "sample draft reply",
            "response_needs_review": False, "sentiment": "Neutral",
            "complexity": "High", "team": "Billing",
            "intent": "dispute_charge_or_refund", "entity_amounts": [30],
        }},
        {"name": "refund under limit, auto-resolve", "input": {
            "ticket_id": "TEST-1", "ticket_text_clean": "sample ticket text",
            "response_text": "sample draft reply",
            "response_needs_review": False, "sentiment": "Neutral",
            "complexity": "Low", "team": "Returns & Refunds",
            "intent": "request_refund", "entity_amounts": [25],
        }},
        {"name": "refund over limit", "input": {
            "ticket_id": "TEST-1", "ticket_text_clean": "sample ticket text",
            "response_text": "sample draft reply",
            "response_needs_review": False, "sentiment": "Neutral",
            "complexity": "Low", "team": "Returns & Refunds",
            "intent": "request_refund", "entity_amounts": [120],
        }},
        {"name": "refund, multiple amounts, highest wins", "input": {
            "ticket_id": "TEST-1", "ticket_text_clean": "sample ticket text",
            "response_text": "sample draft reply",
            "response_needs_review": False, "sentiment": "Neutral",
            "complexity": "Low", "team": "Returns & Refunds",
            "intent": "request_refund", "entity_amounts": [10, 75, 20],
        }},
        {"name": "refund, no amount stated", "input": {
            "ticket_id": "TEST-1", "ticket_text_clean": "sample ticket text",
            "response_text": "sample draft reply",
            "response_needs_review": False, "sentiment": "Neutral",
            "complexity": "Low", "team": "Returns & Refunds",
            "intent": "request_refund", "entity_amounts": [],
        }},
        {"name": "multiple triggers at once", "input": {
            "ticket_id": "TEST-1", "ticket_text_clean": "sample ticket text",
            "response_text": "sample draft reply",
            "response_needs_review": True, "sentiment": "Very Negative",
            "complexity": "High", "team": "Returns & Refunds",
            "intent": "request_refund", "entity_amounts": [200],
        }},
        {"name": "prior contact unresolved", "input": {
            "ticket_id": "TEST-1", "ticket_text_clean": "sample ticket text",
            "response_text": "sample draft reply",
            "response_needs_review": False, "sentiment": "Slightly Negative",
            "complexity": "Low", "team": "Product Quality",
            "intent": "report_defect", "entity_amounts": [],
            "entity_prior_contact_unresolved": True,
        }},
    ]

    for case in test_cases:
        result = graph.invoke(case["input"])
        print(f"{case['name']:<40} -> {result['escalation_decision']:<18} "
              f"{result['escalation_reasons']} | team={result.get('team')} | jira={result.get('jira_issue_key')}")