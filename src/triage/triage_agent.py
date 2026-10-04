"""
Phase 5 — Triage Agent: Category, Priority, Team (LangGraph)
Multi-Agent Customer Support Intelligence Platform

Category and Priority run in PARALLEL — neither reads the other's output,
both only need what Intake already produced (ticket_text_clean, sentiment).
Team runs AFTER both, because assign_team needs Priority (for the
Escalation override) and Intent (already available from Intake, but the
override check specifically needs Priority's result) — see assign_team.py.

Flow:
    START -> {category_node, priority_node} (parallel)
          -> team_node (waits for both)
          -> END

NOT included here: routing the intent_failed / escalate_to_human cases away
from this graph entirely. That check belongs in the orchestrator that calls
intake_graph.invoke(...) and decides whether to call this graph at all — see
the design decision logged for this. This graph assumes it is only ever
invoked with a ticket that already has a valid intent and sentiment.
"""

from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from src.triage.category_predict import predict_category
from src.triage.priority_assign import assign_priority
from src.triage.assign_team import assign_team
from src.triage.complexity_assign import assign_complexity

class TriageState(TypedDict, total=False):
    # inputs (already computed by Intake)
    ticket_text_clean: str
    sentiment: str
    intent: str
    follow_up_rounds: int              
    entity_days_elapsed: list         
    entity_order_ids: list            
    entity_amounts: list               
    # outputs
    category: str
    category_confidence: float        
    priority: str
    team: str
    complexity: str                  


def category_node(state):
    category, confidence = predict_category(state["ticket_text_clean"])
    return {"category": category, "category_confidence": confidence}

def complexity_node(state):
    complexity = assign_complexity(
        state["intent"],
        state.get("follow_up_rounds", 0),
        state.get("entity_days_elapsed"),
        state.get("entity_order_ids"),
        state.get("entity_amounts"),
    )
    return {"complexity": complexity}

def priority_node(state):
    priority = assign_priority(state["sentiment"], state["ticket_text_clean"])
    return {"priority": priority}


def team_node(state):
    # Requires state["priority"] to already exist — only correct to run
    # after the fan-in below, never in parallel with priority_node.
    team = assign_team(state["intent"], state["priority"])
    return {"team": team}

def build_triage_graph():
    graph = StateGraph(TriageState)
    graph.add_node("category_node", category_node)
    graph.add_node("priority_node", priority_node)
    graph.add_node("complexity_node", complexity_node)   
    graph.add_node("team_node", team_node)

    graph.add_edge(START, "category_node")
    graph.add_edge(START, "priority_node")
    graph.add_edge(START, "complexity_node")              

    graph.add_edge("category_node", "team_node")
    graph.add_edge("priority_node", "team_node")
    graph.add_edge("complexity_node", END)                 

    graph.add_edge("team_node", END)
    return graph.compile()

if __name__ == "__main__":
    # Full pipeline demo: python -m src.triage.triage_agent
    # Types a real ticket into Intake (same interrupt/resume loop as
    # intake_agent.py's own demo), then feeds Intake's finished state into
    # this Triage graph — the same handoff shape a real orchestrator needs.
    # Requires all three trained models (intent needs GROQ_API_KEY;
    # category_classifier.joblib and priority_neutral_classifier.joblib
    # must already exist on disk).
    from langgraph.types import Command

    from src.intake.intake_agent import build_intake_graph

    intake_graph = build_intake_graph()
    config = {"configurable": {"thread_id": "triage-demo-1"}}

    text = input("Customer ticket: ")
    intake_result = intake_graph.invoke({"ticket_id": "DEMO-1", "ticket_text": text}, config)
    while "__interrupt__" in intake_result:
        prompt = intake_result["__interrupt__"][0].value["message"]
        reply = input(f"\nAgent: {prompt}\nCustomer: ")
        intake_result = intake_graph.invoke(Command(resume=reply), config)

    print(f"\nIntake finished — status: {intake_result['status']}")
    print(f"Sentiment: {intake_result.get('sentiment')}")
    print(f"Intent: {intake_result.get('intent')}")

    # This is the check the real orchestrator must also make (see the logged
    # design decision) — assign_team has no valid intent to look up for
    # these two statuses, so Triage must never run on them.
    if intake_result["status"] in ("intent_failed", "escalate_to_human","flagged_injection" ):
        print("Routing to human review — skipping Triage "
              "(no valid intent to classify against).")
    else:
        triage_graph = build_triage_graph()
        triage_result = triage_graph.invoke({
            "ticket_text_clean": intake_result["ticket_text_clean"],
            "sentiment": intake_result["sentiment"],
            "intent": intake_result["intent"],
        })
        print(f"Category: {triage_result['category']}")
        print(f"Priority: {triage_result['priority']}")
        print(f"Team: {triage_result['team']}")