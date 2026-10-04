"""
Orchestrator — chains all six agents (Intake, Triage, Retrieval, Response,
Escalation, Learning)
Multi-Agent Customer Support Intelligence Platform

Intake and Triage are separate LangGraph graphs (see the design decision
logged for this project: they are NOT one unified graph). This module is
the glue between them.

Two responsibilities:
1. After Intake finishes, check intake_result["status"] and only invoke the
   Triage graph when there's a valid intent to classify against. Triage and 
   Retrieval must never run on "intent_failed", "flagged_injection", or
   "escalate_to_human" — assign_team() has no valid intent to look up for
   those and will raise ValueError otherwise.
2. Surface Intake's interrupt/resume follow-up loop through two entry
   points instead of one blocking call, since a real caller (e.g. a FastAPI
   endpoint) can't block on input() waiting for the customer's reply — it
   needs to return "awaiting_customer" to its own caller and pick the same
   thread back up later via resume_ticket().

    start_ticket(...)   -> first call for a new ticket
    resume_ticket(...)  -> subsequent call(s), once the customer replies

Both return the same shape:
    {"status": "awaiting_customer", "thread_id": ..., "message": ..., "missing_fields": [...]}
    or the merged pipeline state: Intake fields, plus Triage, Retrieval and Response
    fields when those stages ran, and always escalation_decision,
    escalation_reasons and jira_issue_key. Flags "triage_ran", "retrieval_ran" and
    "response_ran" show which stages executed. Every exit path ends by logging the
    ticket through the Learning agent.
"""

from langgraph.types import Command
from src.intake.intake_agent import build_intake_graph
from src.triage.triage_agent import build_triage_graph
from src.retrieval.retrieval_agent import build_retrieval_graph  
from src.escalation.escalation_agent import build_escalation_graph
from src.response.response_agent import build_response_graph
from src.learning.learning_agent import build_learning_graph
from src.triage.assign_team import ESCALATION_TEAM
from src.escalation.jira_tools import create_issue
from src.response.templates import greeting_text

TRIVIAL_WORD_LIMIT = 3
GREETING_PHRASES = {"hi", "hello", "hey", "hii", "helo", "test", "testing", "thanks", "thank you", "ok", "okay"}


def _is_trivial(ticket_text_clean, intent):
    """A message the model couldn't tie to any real intent AND is too short/
    generic to be a genuine request. Only fires when intent == "other", so a
    real but oddly-worded short request that got a real intent label is
    never caught here. Tune TRIVIAL_WORD_LIMIT / GREETING_PHRASES based on
    what you see misrouted once this runs on real traffic."""
    if intent != "other":
        return False
    cleaned = (ticket_text_clean or "").strip().lower()
    if cleaned in GREETING_PHRASES:
        return True
    return len(cleaned.split()) <= TRIVIAL_WORD_LIMIT
# Statuses where Intake ended without a valid intent to classify against —
# Triage must be skipped for all three (see assign_team's ValueError note).
SKIP_TRIAGE_STATUSES = {"intent_failed", "flagged_injection", "escalate_to_human"}

# Built once per process. intake_graph's MemorySaver checkpoint lives on
# this instance, so start_ticket()/resume_ticket() must share it across
# calls for the same thread_id — do not rebuild the graph per call.
_intake_graph = build_intake_graph()
_triage_graph = build_triage_graph()
_retrieval_graph = build_retrieval_graph()  
_response_graph = build_response_graph()
_escalation_graph = build_escalation_graph()
_learning_graph = build_learning_graph()

def _log_ticket(result):
    """Always called last, for every ticket, on every exit path. Missing
    fields (skipped stages) are logged as None — see learning_agent.py."""
    _learning_graph.invoke({
        "ticket_id": result.get("ticket_id"),
        "ticket_text_clean": result.get("ticket_text_clean"),
        "intent": result.get("intent"),
        "sentiment": result.get("sentiment"),
        "category": result.get("category"),
        "priority": result.get("priority"),
        "complexity": result.get("complexity"),
        "team": result.get("team"),
        "retrieval_confidence": result.get("retrieval_confidence"),
        "retrieval_top_source": result.get("retrieval_top_source"),
        "retrieval_attempts": result.get("retrieval_attempts"),
        "retrieval_error": result.get("retrieval_error"),
        "response_text": result.get("response_text"),
        "response_mode": result.get("response_mode"),
        "response_needs_review": result.get("response_needs_review"),
        "escalation_decision": result.get("escalation_decision"),
        "escalation_reasons": result.get("escalation_reasons"),
        "jira_issue_key": result.get("jira_issue_key"),
    })

def start_ticket(ticket_id, ticket_text, product_name=None, product_segment=None):
    """First call for a new ticket. thread_id defaults to ticket_id."""
    config = {"configurable": {"thread_id": ticket_id}}
    intake_input = {
        "ticket_id": ticket_id,
        "ticket_text": ticket_text,
        "product_name": product_name,
        "product_segment": product_segment,
    }
    intake_result = _intake_graph.invoke(intake_input, config)
    return _finish(intake_result, config)


def resume_ticket(thread_id, reply):
    """Subsequent call, once the customer answers a follow-up question."""
    config = {"configurable": {"thread_id": thread_id}}
    intake_result = _intake_graph.invoke(Command(resume=reply), config)
    return _finish(intake_result, config)


def _finish(intake_result, config):
    if "__interrupt__" in intake_result:
        payload = intake_result["__interrupt__"][0].value
        return {
            "status": "awaiting_customer",
            "thread_id": config["configurable"]["thread_id"],
            "message": payload["message"],
            "missing_fields": payload["missing_fields"],
        }

    result = dict(intake_result)

    if intake_result["status"] in SKIP_TRIAGE_STATUSES:
        result["triage_ran"] = False
        result["retrieval_ran"] = False
        result["response_ran"] = False
        result["escalation_decision"] = "escalate_to_team"
        result["escalation_reasons"] = [f"intake_status={intake_result['status']}"]
        result["jira_issue_key"] = create_issue(
            ticket_id=result.get("ticket_id"),
            team="General Support",
            reasons=result["escalation_reasons"],
            ticket_text=result.get("ticket_text_clean", ""),
            draft_reply=None,
        )
        _log_ticket(result)
        return result

    if _is_trivial(intake_result["ticket_text_clean"], intake_result["intent"]):
        result["triage_ran"] = False
        result["retrieval_ran"] = False
        result["response_ran"] = True
        result["category"] = None
        result["priority"] = None
        result["complexity"] = None
        result["team"] = None
        result["response_text"] = greeting_text()
        result["response_mode"] = "greeting"
        result["response_needs_review"] = False
        result["escalation_decision"] = "no_action_needed"
        result["escalation_reasons"] = []
        result["jira_issue_key"] = None
        _log_ticket(result)
        return result
    triage_result = _triage_graph.invoke({
        "ticket_text_clean": intake_result["ticket_text_clean"],
        "sentiment": intake_result["sentiment"],
        "intent": intake_result["intent"],
        "follow_up_rounds": intake_result.get("follow_up_rounds", 0),  
        "entity_days_elapsed": intake_result.get("entity_days_elapsed"),  
        "entity_order_ids": intake_result.get("entity_order_ids"),        
        "entity_amounts": intake_result.get("entity_amounts"),            
    })
    result.update(triage_result)
    result["triage_ran"] = True
    if result["team"] == ESCALATION_TEAM:
        result["retrieval_ran"] = False
        result["response_ran"] = False
        result["escalation_decision"] = "escalate_to_team"
        result["escalation_reasons"] = ["priority=High"]
        result["jira_issue_key"] = create_issue(
            ticket_id=result.get("ticket_id"),
            team=result["team"],
            reasons=result["escalation_reasons"],
            ticket_text=result.get("ticket_text_clean", ""),
            draft_reply=None,
        )
        _log_ticket(result)
        return result

    retrieval_result = _retrieval_graph.invoke({
        "ticket_text_clean": result["ticket_text_clean"],
        "intent": result["intent"],
        "category": result.get("category", ""),
    })
    result.update(retrieval_result)
    result["retrieval_ran"] = True

    response_result = _response_graph.invoke({
        "ticket_text_clean": result["ticket_text_clean"],
        "intent": result["intent"],
        "sentiment": result["sentiment"],
        "retrieved_docs": result.get("retrieved_docs", []),
        "retrieval_confidence": result.get("retrieval_confidence", "low"),
        "retrieval_top_source": result.get("retrieval_top_source"),
        "retrieval_error": result.get("retrieval_error", False),
    })
    result.update(response_result)
    result["response_ran"] = True

    escalation_result = _escalation_graph.invoke({
        "ticket_id": result.get("ticket_id"),
        "ticket_text_clean": result["ticket_text_clean"],
        "response_text": result.get("response_text"),
        "response_needs_review": result.get("response_needs_review"),
        "sentiment": result["sentiment"],
        "complexity": result.get("complexity"),
        "team": result["team"],
        "intent": result["intent"],
        "entity_amounts": result.get("entity_amounts"),
        "entity_prior_contact_unresolved": result.get("entity_prior_contact_unresolved", False),

    })
    result.update(escalation_result)

    _log_ticket(result)

    return result


if __name__ == "__main__":
    # Terminal demo: python -m src.orchestrator
    # Same interrupt/resume loop as intake_agent.py's own demo, but going
    # through start_ticket()/resume_ticket() rather than the graph directly
    # — exercises the actual boundary a FastAPI endpoint would use.
    import uuid
    ticket_id = f"DEMO-{uuid.uuid4().hex[:8]}"    
    text = input("Customer ticket: ")
    result = start_ticket(ticket_id, text)

    while result["status"] == "awaiting_customer":
        reply = input(f"\nAgent: {result['message']}\nCustomer: ")
        result = resume_ticket(result["thread_id"], reply)

    print(f"\nStatus: {result['status']} | intent: {result.get('intent')} | "
          f"sentiment: {result.get('sentiment')}")
    if result["triage_ran"]:
        print(f"Category: {result['category']} | Priority: {result['priority']} | "
              f"Team: {result['team']}")
        if result["retrieval_ran"]:
            print(f"Retrieval: confidence={result['retrieval_confidence']} | "
                  f"attempts={result['retrieval_attempts']} | top source={result['retrieval_top_source']}")
            for doc in result["retrieved_docs"][:3]:
                print(f"  dist={doc['distance']:.4f} {doc['metadata']['source_type']} | {doc['text'][:90]}")
            print(f"\nResponse ({result['response_mode']}, needs_review={result['response_needs_review']}):")
            print(result["response_text"])
        else:
            print("Retrieval skipped: ticket escalated by priority.")
    else:
        print("Routed to human review: Triage skipped (no valid intent).")
    if result["response_ran"]:
        print(f"\nEscalation: {result['escalation_decision']} "
              f"(reasons: {result['escalation_reasons'] or 'none'})")
        if result.get("jira_issue_key"):
            print(f"Jira issue filed: {result['jira_issue_key']}")
        elif result["escalation_decision"] == "escalate_to_team":
            print("Jira issue: not filed (Jira not configured)")
    else:
        print(f"\nEscalation: {result['escalation_decision']} "
              f"(reasons: {result['escalation_reasons'] or 'none'})")
        if result.get("jira_issue_key"):
            print(f"Jira issue filed: {result['jira_issue_key']}")
        elif result["escalation_decision"] == "escalate_to_team":
            print("Jira issue: not filed (Jira not configured)")