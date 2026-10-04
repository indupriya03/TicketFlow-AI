"""
Phase 6 — Retrieval Agent (LangGraph)
Multi-Agent Customer Support Intelligence Platform

Its own graph, separate from Intake and Triage — the orchestrator calls it
after Triage with Triage's outputs (see orchestrator.py).

Flow:
    START -> strategy -> retrieve -> (high confidence)        -> END
                                  -> (low, attempts < MAX)    -> strategy  (loop)
                                  -> (low, attempts >= MAX)   -> END  (best effort, flagged low)

Each pass through the loop is one attempt, and the query changes per attempt:
    attempt 1  "initial"    query = ticket_text_clean
    attempt 2  "rewritten"  "<intent phrase>: <ticket text>" — the intent phrase adds
                            topical signal, and the ticket text stays in the query so
                            off-topic tickets don't turn into false "high" results

MAX_ATTEMPTS = 2 means the initial search plus 1 retry.

NO category or intent pre-filtering: the index is ~212 documents, Chroma
compares the query against all of them in milliseconds, so narrowing first
saves nothing — and a wrong category filter would exclude the right document
before the search even runs. Both channels are always searched with ONE
query embedding. Revisit filtering if the corpus grows into the thousands
(e.g. if the Learning agent starts adding resolved tickets over time).

Confidence comes from retrieval_confidence.classify_confidence, per source
type. When both channels are confident, FAQ is listed first — its
correct-vs-wrong distance margin is cleaner than the template channel's
(see retrieval_confidence.py). A resolution_template "high" is weaker
evidence than a FAQ "high"; retrieval_top_source is returned so Escalation
can weigh them differently.

Inputs (from Intake/Triage):  ticket_text_clean, intent, category
                              (intent and category are used only to build the
                              attempt-2 rewrite)
Outputs:                      retrieved_docs, retrieval_confidence, retrieval_top_source,
                              retrieval_attempts, retrieval_strategy

Demo: python -m src.retrieval.retrieval_agent
"""

from typing import Optional, TypedDict

from langgraph.graph import END, START, StateGraph

from src.retrieval.retrieval_confidence import classify_confidence
from src.retrieval.retrieval_query import DEFAULT_K, Retriever
import logging

MAX_ATTEMPTS = 2          # initial search + 1 retry

MIN_TEMPLATE_DOCS = 2     # guarantee Response sees at least this many outcome-style examples
class RetrievalState(TypedDict, total=False):
    # inputs
    ticket_text_clean: str
    intent: str
    category: str
    # per-attempt plan (written by strategy_node)
    search_query: str
    retrieval_strategy: str
    # outputs
    retrieved_docs: list
    retrieval_confidence: str     # "high" | "low"
    retrieval_top_source: Optional[str]
    retrieval_attempts: int
    retrieval_error: bool         # True if the embedding call failed (docs empty, confidence "low")

_retriever = None


def get_retriever():
    """Built once per process — holds the Chroma client and Gemini client."""
    global _retriever
    if _retriever is None:
        _retriever = Retriever()
    return _retriever


def _rewrite_query(intent, ticket_text):
    phrase = (intent or "").replace("_", " ").strip()
    return f"{phrase}: {ticket_text}" if phrase else ticket_text

def _top_confidence(hits):
    if not hits:
        return None
    top = hits[0]
    return classify_confidence(top["distance"], top["metadata"]["source_type"])
def _fit_to_k(ordered, k):
    """Take the top k, but guarantee at least MIN_TEMPLATE_DOCS resolution
    templates when any were retrieved: append the best missing templates and
    drop the lowest-ranked non-template docs to make room."""
    top = ordered[:k]
    have = sum(1 for d in top if d["metadata"]["source_type"] == "resolution_template")
    missing = MIN_TEMPLATE_DOCS - have
    if missing <= 0:
        return top

    extras = [d for d in ordered[k:]
              if d["metadata"]["source_type"] == "resolution_template"][:missing]
    keep = list(top)
    for _ in extras:
        for i in range(len(keep) - 1, -1, -1):
            if keep[i]["metadata"]["source_type"] != "resolution_template":
                del keep[i]
                break
    return keep + extras

def _rank_and_grade(faq_hits, template_hits, k):
    """Each channel is graded against its own threshold (their distance scales
    differ). FAQ goes first when confident."""
    if _top_confidence(faq_hits) == "high":
        ordered, confidence = faq_hits + template_hits, "high"
    elif _top_confidence(template_hits) == "high":
        ordered, confidence = template_hits + faq_hits, "high"
    else:
        ordered = sorted(faq_hits + template_hits, key=lambda r: r["distance"])
        confidence = "low"
    return _fit_to_k(ordered, k), confidence


def strategy_node(state):
    if state.get("retrieval_attempts", 0) == 0:
        return {"search_query": state["ticket_text_clean"], "retrieval_strategy": "initial"}
    rewritten = _rewrite_query(state.get("intent"), state["ticket_text_clean"])
    return {
        "search_query": rewritten or state["ticket_text_clean"],
        "retrieval_strategy": "rewritten",
    }


def retrieve_node(state):
    """Search both channels with ONE query embedding, then grade and rank."""
    retriever = get_retriever()
    try:
        embedding = retriever.embedder.embed_query([state["search_query"]])[0]
    except Exception as e:
        logging.getLogger(__name__).error(f"Retrieval embedding failed: {e}")
        return {
            "retrieved_docs": [],
            "retrieval_confidence": "low",
            "retrieval_top_source": None,
            "retrieval_attempts": MAX_ATTEMPTS,
            "retrieval_error": True,
        }
    faq_hits = retriever._search(embedding, DEFAULT_K, {"source_type": "faq"})
    template_hits = retriever._search(embedding, DEFAULT_K, {"source_type": "resolution_template"})

    docs, confidence = _rank_and_grade(faq_hits, template_hits, DEFAULT_K)
    return {
        "retrieved_docs": docs,
        "retrieval_confidence": confidence,
        "retrieval_top_source": docs[0]["metadata"]["source_type"] if docs else None,
        "retrieval_attempts": state.get("retrieval_attempts", 0) + 1,
        "retrieval_error": False,
    }


def route_after_retrieve(state):
    if state["retrieval_confidence"] == "high":
        return "done"
    if state["retrieval_attempts"] >= MAX_ATTEMPTS:
        return "done"
    return "retry"


def build_retrieval_graph():
    graph = StateGraph(RetrievalState)
    graph.add_node("strategy", strategy_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_edge(START, "strategy")
    graph.add_edge("strategy", "retrieve")
    graph.add_conditional_edges(
        "retrieve", route_after_retrieve, {"done": END, "retry": "strategy"}
    )
    return graph.compile()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--query", required=True, help="ticket text to search with")
    parser.add_argument("--intent", default="", help="optional, used only in the attempt-2 rewrite")
    parser.add_argument("--category", default="", help="optional, used only in the attempt-2 rewrite")
    args = parser.parse_args()

    graph = build_retrieval_graph()
    result = graph.invoke({
        "ticket_text_clean": args.query,
        "intent": args.intent,
        "category": args.category,
    })

    print(f"\nQUERY: {args.query}")
    print(f"confidence={result['retrieval_confidence']} | attempts={result['retrieval_attempts']} "
          f"| last strategy={result['retrieval_strategy']} | top source={result['retrieval_top_source']}")
    for i, doc in enumerate(result["retrieved_docs"][:5], 1):
        print(f"  [{i}] dist={doc['distance']:.4f} {doc['metadata']['source_type']} | {doc['text'][:90]}")