"""
Phase 4 — Intake Agent: combined node + customer follow-up loop (LangGraph)
Multi-Agent Customer Support Intelligence Platform

One ticket goes through Intake ONCE: entities, intent, sentiment, completeness.
If a required field (currently: order_id) is missing, the graph pauses, asks the
customer, and resumes when they reply. Intent and sentiment are NOT recomputed
on a reply (they describe the original ticket); entities + completeness are.

Flow:
    START -> intake -> (complete)            -> END
                    -> (incomplete, rounds<MAX) -> ask_customer -> intake  (loop)
                    -> (incomplete, rounds>=MAX) -> human_handoff -> END
                    -> (intent failed)       -> human_handoff -> END  (status="intent_failed")
                    -> (flagged injection)   -> human_handoff -> END  (status="flagged_injection")

Customer replies only go through regex entity extraction, never to an LLM,
so the PII scrub and injection check (which protect LLM calls) are not
needed on replies. If a reply is ever sent to an LLM, run it through
src/guardrails.py first.

Adjust the import paths below to match where your files live.
"""

import os
import re
from pathlib import Path
from typing import TypedDict

import joblib
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from src.intake.intake_completeness import check_completeness
from src.intake.intake_entities import extract_entities
from src.preprocessing import light_clean_text, pii_scrub
from src.intake.intake_sentiment import predict_sentiment
from src.intake.intake_intent import classify_intent, get_structured_llm

ROOT_DIR = Path(__file__).parent.parent.parent
SENTIMENT_MODEL_PATH = ROOT_DIR / "models/sentiment_classifier.joblib"
EMBEDDER_NAME = "all-MiniLM-L6-v2"
MAX_FOLLOW_UPS = 2

FIELD_PROMPTS = {"order_id": "your order ID (it looks like #ORD12345)"}


class IntakeState(TypedDict, total=False):
    # inputs
    ticket_id: str
    ticket_text: str
    product_name: str
    product_segment: str
    # conversation
    customer_replies: list
    follow_up_rounds: int
    follow_up_message: str
    # outputs
    ticket_text_clean: str
    intent: str
    intent_reasoning: str
    flagged_injection: bool
    sentiment: str
    entity_order_ids: list
    entity_amounts: list
    entity_product_mentioned: bool
    entity_days_elapsed: list
    entity_prior_contact_unresolved: bool
    is_complete: bool
    missing_fields: list
    status: str  # complete | awaiting_customer | escalate_to_human | intent_failed | flagged_injection
    needs_human_review: bool


def build_follow_up_message(missing_fields):
    asked = " and ".join(FIELD_PROMPTS.get(f, f) for f in missing_fields)
    return (
        f"Thanks for reaching out. To help with this, could you please share {asked}? "
        "You can find it in your order confirmation email."
    )

# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------
def intake_node(state):
    updates = {}
    replies = state.get("customer_replies", [])

    # Once-only steps: describe the ORIGINAL ticket, so skipped on later passes.
    if "ticket_text_clean" not in state:
        clean, _ = pii_scrub(light_clean_text(state["ticket_text"]))
        updates["ticket_text_clean"] = clean
    clean = updates.get("ticket_text_clean", state.get("ticket_text_clean"))

    intent = state.get("intent")
    if intent is None:  # first pass, or a previous attempt failed
        result = classify_intent(get_structured_llm(), clean)
        intent = result["intent"]
        updates["intent"] = intent
        updates["intent_reasoning"] = result["reasoning"]
        updates["flagged_injection"] = result["flagged_injection"]

    if "sentiment" not in state:
        updates["sentiment"] = predict_sentiment(
            clean, state.get("product_name", ""), state.get("product_segment", "")
        )

    # Every pass: entities from the original text PLUS any customer replies.
    raw_text = " ".join([state["ticket_text"], *replies])
    entities = extract_entities(raw_text, state.get("product_name"))
    updates["entity_order_ids"] = entities["order_ids"]
    updates["entity_amounts"] = entities["amounts"]
    updates["entity_product_mentioned"] = entities["product_mentioned"]
    updates["entity_days_elapsed"] = entities["days_elapsed"]
    updates["entity_prior_contact_unresolved"] = entities["prior_contact_unresolved"]
    if intent is None:
        updates.update(
            is_complete=False, 
            missing_fields=[], 
            status="intent_failed",
            needs_human_review=True,
        )
        return updates

    # A flagged ticket's intent is a hardcoded placeholder ("other"), not a
    # real classification — running Category/Priority/Team on top of it would
    # produce triage output that looks legitimate but isn't grounded in
    # anything. Route straight to human review instead, same as intent_failed.
    if updates.get("flagged_injection") or state.get("flagged_injection"):
        updates.update(
            is_complete=False,
            missing_fields=[],
            status="flagged_injection",
            needs_human_review=True,
        )
        return updates

    completeness = check_completeness(intent, entities["order_ids"])
    updates["is_complete"] = completeness["is_complete"]
    updates["missing_fields"] = completeness["missing_fields"]
    updates["status"] = "complete" if completeness["is_complete"] else "awaiting_customer"
    return updates


def ask_customer_node(state):
    message = build_follow_up_message(state["missing_fields"])
    # Pauses the graph here. The value passed to Command(resume=...) comes back as `reply`.
    reply = interrupt({"message": message, "missing_fields": state["missing_fields"]})
    return {
        "customer_replies": state.get("customer_replies", []) + [reply],
        "follow_up_rounds": state.get("follow_up_rounds", 0) + 1,
        "follow_up_message": message,
    }


def human_handoff_node(state):
    # Don't clobber a more specific status intake_node already set
    # (e.g. "intent_failed", "flagged_injection") — only assign the generic
    # reason when this is genuinely the follow-up-exhausted path.
    if state.get("status") in ("intent_failed", "flagged_injection"):
        return {"needs_human_review": True}
    return {"status": "escalate_to_human", "needs_human_review": True}


def route_after_intake(state):
    if state.get("intent") is None or state.get("status") == "flagged_injection":
        return "handoff"
    if state["is_complete"]:
        return "done"
    if state.get("follow_up_rounds", 0) >= MAX_FOLLOW_UPS:
        return "handoff"
    return "ask"


def build_intake_graph(checkpointer=None):
    graph = StateGraph(IntakeState)
    graph.add_node("intake", intake_node)
    graph.add_node("ask_customer", ask_customer_node)
    graph.add_node("human_handoff", human_handoff_node)
    graph.add_edge(START, "intake")
    graph.add_conditional_edges(
        "intake",
        route_after_intake,
        {"done": END, "ask": "ask_customer", "handoff": "human_handoff"},
    )
    graph.add_edge("ask_customer", "intake")
    graph.add_edge("human_handoff", END)
    # A checkpointer is required for interrupt(): it stores the paused state.
    return graph.compile(checkpointer=checkpointer or MemorySaver())


if __name__ == "__main__":
    # Terminal demo: python -m src.intake.intake_agent
    graph = build_intake_graph()
    config = {"configurable": {"thread_id": "demo-1"}}
    text = input("Customer ticket: ")
    result = graph.invoke({"ticket_id": "DEMO-1", "ticket_text": text}, config)
    while "__interrupt__" in result:
        prompt = result["__interrupt__"][0].value["message"]
        reply = input(f"\nAgent: {prompt}\nCustomer: ")
        result = graph.invoke(Command(resume=reply), config)
    print(
        f"\nStatus: {result['status']} | intent: {result.get('intent')} | "
        f"sentiment: {result.get('sentiment')} | order_ids: {result.get('entity_order_ids')}"
    )