"""
Phase 6 — Retrieval Agent: Evaluation
Multi-Agent Customer Support Intelligence Platform

Two gold-query sets, evaluated separately (they search different
source_type slices of the collection, matching the diagram's separate
FAQ Search / Past Ticket Search paths):

1. RESOLUTION-TEMPLATE gold set — built for free from
   data/processed/support_tickets_test.csv. Every test-set ticket has a
   real (ticket_text_clean, resolution_text) pair that was never used to
   train anything. Normalizing resolution_text the same way
   preprocessing_rag.py did (lowercase + strip digits) tells us exactly
   which res_<ticket_id> document in the collection is the "correct"
   answer for that ticket — no manual labeling needed.

2. FAQ gold set — no ticket->FAQ ground truth exists anywhere in the
   data, so this is a small hand-curated list below (seeded with a few
   obvious pairs from manual query testing). EXPAND THIS YOURSELF — the
   handful here are a starting point, not a real evaluation set.

Metrics: Recall@K (was the correct doc in the top K?) and MRR (mean
reciprocal rank), plus the distance the correct doc was actually found
at — that distance distribution is what an evidence-based "good enough"
threshold for agentic Retrieval's retry logic should be set from, rather
than a guessed number.

COST WARNING: each gold query costs one real embedding request against
the same free-tier quota retrieval_document.py was fighting all session.
Default --sample-size is deliberately small. Scale up once this is
trusted to be worth spending more quota on.

Usage:
    uv run python -m src.retrieval.retrieval_eval --sample-size 20
"""

import argparse
import random
from pathlib import Path

import pandas as pd

from src.retrieval.retrieval_query import Retriever
from src.retrieval.retrieval_agent import build_retrieval_graph
from src.retrieval.preprocessing_rag import _mask_product_name, TICKETS_PATH as TRAIN_PATH

ROOT_DIR = Path(__file__).parent.parent.parent
TEST_PATH = ROOT_DIR / "data/processed/support_tickets_test.csv"
REPORT_PATH = ROOT_DIR / "reports/retrieval_eval_report.txt"
K_VALUES = [1, 3, 5, 10]

# Starter FAQ gold set — seeded from manual query testing earlier in this
# project. Nowhere near comprehensive; add to this list as you find more
# (query, expected_faq_id) pairs you're confident about.
FAQ_GOLD_SET = [
    ("how do I return a defective product", "faq_FAQ048"),
    ("what if my package shows delivered but I didn't receive it", "faq_FAQ006"),
    ("where will my refund be credited", "faq_FAQ029"),
    ("how do I track my order", "faq_FAQ001"),
    ("what if the seller isn't responding", "faq_FAQ077"),
]
# Intents Intake would plausibly assign to the FAQ gold queries above, so the
# agent's attempt-2 rewrite has something to work with. The resolution-template
# gold queries have no intent column, so they run with intent="" (their retry
# is then the same query as attempt 1 — see note in evaluate_agent).
FAQ_INTENTS = {
    "how do I return a defective product": "initiate_return",
    "what if my package shows delivered but I didn't receive it": "report_delivery_issue",
    "where will my refund be credited": "request_information",
    "how do I track my order": "check_return_or_order_status",
    "what if the seller isn't responding": "escalate_unresponsive_seller",
}

# Queries no document should answer. The agent should end each one "low".
OFF_TOPIC_QUERIES = [
    "best pizza topping for a birthday party",
    "what's the weather in Berlin tomorrow",
    "how do I learn to play the guitar",
    "who won the football match last night",
    "recommend a good sci-fi movie",
    "how far is the moon from earth",
    "what is the capital of Australia",
    "tips for growing tomatoes at home",
    "how do I fix a leaking kitchen tap",
    "translate good morning into French",
]

report_lines = []


def log(msg):
    print(msg)
    report_lines.append(msg)


def build_resolution_gold_set(sample_size: int, seed: int = 42):
    """Returns list of (query, expected_doc_id) using test-set tickets,
    matched to the collection's actual indexed resolution-template ids
    via the same normalization preprocessing_rag.py used to dedupe."""
    test_df = pd.read_csv(TEST_PATH)

    from src.retrieval.retrieval_document import DOCUMENTS_PATH
    import json
    with open(DOCUMENTS_PATH) as f:
        records = json.load(f)
    resolution_records = [r for r in records if r["metadata"]["source_type"] == "resolution_template"]

    # Same normalization as preprocessing_rag.py's dedup key, applied to
    # each indexed resolution document's own stored text, so we can match
    # a test ticket's resolution_text back to the representative id that
    # survived dedup for its template group.
    def normalize(text,product_name=None):
        import re
        masked = _mask_product_name(text, product_name)
        return re.sub(r"\d+", "", masked.lower())
    train_df = pd.read_csv(TRAIN_PATH)
    indexed_ids = {r["id"] for r in resolution_records}
    doc_lookup = {}
    for row in train_df.itertuples():
        doc_id = f"res_{row.ticket_id}"
        key = normalize(row.resolution_text, row.product_name)
        if key not in doc_lookup and doc_id in indexed_ids:
            doc_lookup[key] = doc_id
    id_to_text = {r["id"]: r["document"] for r in resolution_records}

    rng = random.Random(seed)
    sample_size = min(sample_size, len(test_df))
    sampled = test_df.sample(n=sample_size, random_state=seed)

    gold_set = []
    unmatched = 0
    for row in sampled.itertuples():
        key = normalize(row.resolution_text, row.product_name)
        expected_id = doc_lookup.get(key)
        if expected_id is None:
            unmatched += 1
            continue
        query = row.ticket_text_clean if hasattr(row, "ticket_text_clean") else row.ticket_text
        gold_set.append((query, expected_id))

    if unmatched:
        log(f"Test tickets whose resolution template never appears in the train split cannot be matched; this is expected to be small but not necessarily 0.")

    return gold_set, id_to_text


def evaluate(retriever, gold_set, source_type_filter, label, id_to_text=None, show_misses=False):
    log(f"\n--- {label} ({len(gold_set)} gold queries) ---")

    recall_hits = {k: 0 for k in K_VALUES}
    reciprocal_ranks = []
    correct_distances = []
    top1_distances_when_wrong = []
    miss_details = []

    max_k = max(K_VALUES)
    for query, expected_id in gold_set:
        results = retriever.retrieve(query, k=max_k, where={"source_type": source_type_filter})
        result_ids = [r["id"] for r in results]

        if expected_id in result_ids:
            rank = result_ids.index(expected_id) + 1  # 1-indexed
            reciprocal_ranks.append(1.0 / rank)
            correct_distances.append(results[rank - 1]["distance"])
            for k in K_VALUES:
                if rank <= k:
                    recall_hits[k] += 1
            # A "correct" hit buried near rank 10 is still worth seeing —
            # it's a weak win, not a strong one, and MRR already tells you
            # this happens (see the query, expected_id, rank).
            if show_misses and rank > 3:
                miss_details.append((query, expected_id, rank, results))
        else:
            reciprocal_ranks.append(0.0)
            if results:
                top1_distances_when_wrong.append(results[0]["distance"])
            if show_misses:
                miss_details.append((query, expected_id, None, results))

    n = len(gold_set)
    if n == 0:
        log("No gold queries to evaluate.")
        return

    mrr = sum(reciprocal_ranks) / n
    log(f"MRR: {mrr:.4f}")
    for k in K_VALUES:
        log(f"Recall@{k}: {recall_hits[k] / n:.4f} ({recall_hits[k]}/{n})")

    if correct_distances:
        correct_distances.sort()
        mid = len(correct_distances) // 2
        median = correct_distances[mid]
        log(f"Distance of correct hit — min: {min(correct_distances):.4f}, "
            f"median: {median:.4f}, max: {max(correct_distances):.4f}")
    if top1_distances_when_wrong:
        log(f"Top-1 distance on MISSES — min: {min(top1_distances_when_wrong):.4f}, "
            f"max: {max(top1_distances_when_wrong):.4f} "
            f"(n={len(top1_distances_when_wrong)})")

    if correct_distances and top1_distances_when_wrong:
        log("\nA threshold between the max correct-hit distance and the min "
            "wrong-top-1 distance (if they don't overlap) would separate "
            "genuine matches from noise on this sample. Overlapping ranges "
            "mean no single cutoff works cleanly and either the threshold "
            "needs a bigger sample to locate, or agentic retry logic should "
            "lean on more than distance alone.")

    if show_misses and miss_details:
        log(f"\n--- {label}: {len(miss_details)} weak/missing hits (rank > 3 or not found), "
            f"detail ---")
        for query, expected_id, rank, results in miss_details:
            rr = f"1/{rank} = {1/rank:.3f}" if rank else "0 (not in top {})".format(max_k)
            log(f"\nQUERY: {query}")
            log(f"EXPECTED: {expected_id}  (reciprocal rank: {rr})")
            if id_to_text and expected_id in id_to_text:
                log(f"  expected text: {id_to_text[expected_id]}")
            log("ACTUALLY RETRIEVED (top 3):")
            for i, r in enumerate(results[:3], 1):
                marker = " <-- correct" if r["id"] == expected_id else ""
                log(f"  [{i}] dist={r['distance']:.4f} id={r['id']}{marker}")
                log(f"      {r['text']}")

def evaluate_agent(graph, gold_set, label, intents=None):
    """Runs each gold query through the full Retrieval agent (strategy + retry)
    and scores what the agent actually RETURNS. retrieved_docs is capped at
    DEFAULT_K, so ranks beyond that count as misses, and a confident FAQ channel
    lists FAQ hits first, which can push template documents out of the list."""
    intents = intents or {}
    n = len(gold_set)
    log(f"\n--- {label}: full agent ({n} gold queries) ---")
    if n == 0:
        log("No gold queries to evaluate.")
        return

    top1 = top3 = high = rescued = 0
    in_docs = 0
    reciprocal_ranks = []

    for query, expected_id in gold_set:
        result = graph.invoke({
            "ticket_text_clean": query,
            "intent": intents.get(query, ""),
            "category": "",
        })
        ids = [d["id"] for d in result["retrieved_docs"]]
        if expected_id in ids:
            in_docs += 1
            rank = ids.index(expected_id) + 1
            reciprocal_ranks.append(1.0 / rank)
            if rank == 1:
                top1 += 1
            if rank <= 3:
                top3 += 1
        else:
            reciprocal_ranks.append(0.0)

        if result["retrieval_confidence"] == "high":
            high += 1
            if result["retrieval_attempts"] > 1:
                rescued += 1   # low on attempt 1, high after the retry

    log(f"MRR (returned docs only): {sum(reciprocal_ranks) / n:.4f}")
    log(f"Recall@1: {top1 / n:.4f} ({top1}/{n})")
    log(f"Recall@3: {top3 / n:.4f} ({top3}/{n})")
    log(f"Confidence high: {high}/{n}")
    log(f"Rescued by retry (low on attempt 1, high after): {rescued}")
    log(f"Correct doc anywhere in returned docs: {in_docs}/{n}")

def evaluate_off_topic(graph, queries):
    """Every query here should end confidence=low. A 'high' is a false
    positive: an irrelevant ticket that would be grounded on wrong documents."""
    log(f"\n--- Off-topic negatives: full agent ({len(queries)} queries) ---")
    false_high = 0
    for query in queries:
        result = graph.invoke({
            "ticket_text_clean": query,
            "intent": "other",
            "category": "",
        })
        top = result["retrieved_docs"][0] if result["retrieved_docs"] else None
        dist = f"{top['distance']:.4f}" if top else "n/a"
        flag = "  <-- FALSE HIGH" if result["retrieval_confidence"] == "high" else ""
        if flag:
            false_high += 1
        log(f"  {result['retrieval_confidence']:<4} attempts={result['retrieval_attempts']} "
            f"top dist={dist} | {query}{flag}")
    log(f"False-high rate: {false_high}/{len(queries)}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-size", type=int, default=20,
                         help="Number of test-set tickets to sample for the resolution-template gold set")
    parser.add_argument("--show-misses", action="store_true",
                         help="Print query/expected/actual text for every weak hit (rank>3) or miss")
    parser.add_argument("--agent", action="store_true",
                         help="Also evaluate the full Retrieval agent (strategy + retry) and off-topic "
                              "negatives. Costs up to 2 embedding requests per query.")
    args = parser.parse_args()

    log("PHASE 6 — RETRIEVAL AGENT: EVALUATION")
    log("=" * 60)

    retriever = Retriever()

    resolution_gold, id_to_text = build_resolution_gold_set(args.sample_size)
    evaluate(retriever, resolution_gold, "resolution_template", "Resolution-template retrieval",
             id_to_text=id_to_text, show_misses=args.show_misses)

    evaluate(retriever, FAQ_GOLD_SET, "faq", "FAQ retrieval (starter set — expand this)",
             show_misses=args.show_misses)
    if args.agent:
        graph = build_retrieval_graph()
        evaluate_agent(graph, resolution_gold, "Resolution-template gold set")
        evaluate_agent(graph, FAQ_GOLD_SET, "FAQ gold set", intents=FAQ_INTENTS)
        evaluate_off_topic(graph, OFF_TOPIC_QUERIES)
    Path(ROOT_DIR / "reports").mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w") as f:
        f.write("\n".join(report_lines))
    log(f"\nSaved report to {REPORT_PATH}")


if __name__ == "__main__":
    main()