"""
Phase 7 — Response Agent: Evaluation (BLEU/ROUGE + Human Eval)
Multi-Agent Customer Support Intelligence Platform

Two independent evaluations, reported together:

1. BLEU/ROUGE: samples tickets that have BOTH a known intent (from
   intake_intent.py's 200-ticket validation sample) and a historical
   resolution_text (from the original dataset), runs each through the
   REAL Retrieval + Response pipeline, and scores the generated reply
   against resolution_text as the "gold" reference.

   CAVEAT, stated explicitly rather than hidden: resolution_text is a
   short, templated INTERNAL resolution note (e.g. "Full refund
   processed."), not customer-facing prose — structurally different
   from what Response generates even when both are substantively
   correct. Expect low n-gram overlap by design; a low score here is
   not evidence Response is wrong, only that word-overlap is a weak
   proxy for "is this a good customer-facing reply" in this dataset.
   Read scores as a rough signal, not a pass/fail bar.

2. Human eval: pulled from tickets_log.human_action — every real
   Approve/Reject decision made via Streamlit's review screen during
   actual use IS a human-eval judgment already. Approve = reviewer
   judged the draft adequate as-is; rejected_edited = reviewer judged
   it needed correction. Reported as a human review approval rate
   (n = reviewed/escalated tickets only), not overall response quality.

Usage:
    python -m src.response.response_eval --sample-size 15
    python -m src.response.response_eval --sample-size 15 --human-eval-only
"""

import argparse
from pathlib import Path
import csv
from dotenv import parser
import nltk
import pandas as pd
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from rouge_score import rouge_scorer
import time
from src.learning.learning_agent import get_connection
from src.response.response_agent import build_response_graph
from src.retrieval.retrieval_agent import build_retrieval_graph
from src.intake.intake_intent import classify_intent, get_structured_llm


ROOT_DIR = Path(__file__).parent.parent.parent
TICKETS_PATH = ROOT_DIR / "data/processed/support_tickets_test.csv"
INTENT_CACHE_PATH = ROOT_DIR / "data/processed/response_eval_intents.csv"
REPORT_PATH = ROOT_DIR / "reports/response_eval_report.txt"
RESULTS_CSV_PATH = ROOT_DIR / "data/processed/response_eval_results.csv"

report_lines = []


def log(msg):
    print(msg)
    report_lines.append(msg)


def _ensure_nltk_data():
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", quiet=True)


def build_gold_sample(sample_size, seed=42):
    """Held-out test tickets only. sample_size=0 means every ticket."""
    tickets = pd.read_csv(TICKETS_PATH)
    tickets = tickets[tickets["resolution_text"].notna() & (tickets["resolution_text"] != "")]
    tickets = tickets[tickets["ticket_text_clean"].notna()]
    if sample_size and sample_size < len(tickets):
        tickets = tickets.sample(n=sample_size, random_state=seed)
    return tickets


def load_intent_cache():
    if INTENT_CACHE_PATH.exists():
        df = pd.read_csv(INTENT_CACHE_PATH)
        return dict(zip(df["ticket_id"], df["intent"]))
    return {}


def get_intent(ticket_id, text, cache, structured_llm):
    """Classify once and save to disk, so a stopped run never repeats a call."""
    if ticket_id in cache:
        return cache[ticket_id]
    result = classify_intent(structured_llm, text)
    if not result.get("complete") or not result.get("intent"):
        return None
    cache[ticket_id] = result["intent"]
    new = not INTENT_CACHE_PATH.exists()
    with open(INTENT_CACHE_PATH, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["ticket_id", "intent"])
        w.writerow([ticket_id, result["intent"]])
    return result["intent"]


def score_pair(generated, reference, scorer, smoothing):
    ref_tokens = reference.split()
    gen_tokens = generated.split()
    bleu = sentence_bleu([ref_tokens], gen_tokens, smoothing_function=smoothing)
    rouge = scorer.score(reference, generated)
    return {
        "bleu": bleu,
        "rouge1_f1": rouge["rouge1"].fmeasure,
        "rougeL_f1": rouge["rougeL"].fmeasure,
    }


RESULT_FIELDS = ["ticket_id", "intent", "response_mode", "exact_dup", "near_dup",
                 "retrieval_confidence", "bleu", "rouge1_f1", "rougeL_f1",
                 "generated", "reference", "evidence"]

def _append_result(rec):
    new = not RESULTS_CSV_PATH.exists()
    with open(RESULTS_CSV_PATH, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=RESULT_FIELDS)
        if new:
            w.writeheader()
        w.writerow(rec)


def summarise(df, label):
    if len(df) == 0:
        log(f"\n{label}: no tickets.")
        return
    log(f"\n{label} (n={len(df)})")
    log(f"  Mean BLEU:       {df['bleu'].mean():.4f}")
    log(f"  Mean ROUGE-1 F1: {df['rouge1_f1'].mean():.4f}")
    log(f"  Mean ROUGE-L F1: {df['rougeL_f1'].mean():.4f}")


def run_bleu_rouge_eval(sample_size, sleep, resume):
    log("\n--- BLEU/ROUGE eval (held-out test tickets) ---")
    _ensure_nltk_data()

    if RESULTS_CSV_PATH.exists() and not resume:
        raise SystemExit(f"{RESULTS_CSV_PATH} exists. Use --resume to continue it, "
                         "or move it aside to start over.")

    sample = build_gold_sample(sample_size)
    done_ids = set(pd.read_csv(RESULTS_CSV_PATH)["ticket_id"]) if RESULTS_CSV_PATH.exists() else set()
    todo = sample[~sample["ticket_id"].isin(done_ids)]
    log(f"{len(sample)} sampled, {len(done_ids)} already done, {len(todo)} to run.")

    retrieval_graph = build_retrieval_graph()
    response_graph = build_response_graph()
    structured_llm = get_structured_llm()
    intent_cache = load_intent_cache()
    scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
    smoothing = SmoothingFunction().method1

    for i, row in enumerate(todo.itertuples(), 1):
        intent = get_intent(row.ticket_id, row.ticket_text_clean, intent_cache, structured_llm)
        if intent is None:
            print(f"[{i}/{len(todo)}] {row.ticket_id}: intent failed, skipped")
            continue
        retrieval = retrieval_graph.invoke({
            "ticket_text_clean": row.ticket_text_clean,
            "intent": intent,
            "category": "",
        })
        response = response_graph.invoke({
            "ticket_text_clean": row.ticket_text_clean,
            "intent": intent,
            "sentiment": getattr(row, "sentiment", "Neutral"),
            "retrieved_docs": retrieval.get("retrieved_docs", []),
            "retrieval_confidence": retrieval.get("retrieval_confidence", "low"),
            "retrieval_top_source": retrieval.get("retrieval_top_source"),
            "retrieval_error": retrieval.get("retrieval_error", False),
        })
        generated = response.get("response_text", "") or ""
        reference = str(row.resolution_text)
        scores = score_pair(generated, reference, scorer, smoothing)
        _append_result({
            "ticket_id": row.ticket_id, "intent": intent,
            "response_mode": response.get("response_mode"),
            "exact_dup": row.resolution_text_is_exact_dup,
            "near_dup": row.resolution_text_is_near_dup,
            "generated": generated, "reference": reference, **scores,
            "retrieval_confidence": retrieval.get("retrieval_confidence"),
            "evidence": " ||| ".join(d.get("text", "") for d in retrieval.get("retrieved_docs", [])),
        })
        print(f"[{i}/{len(todo)}] {row.ticket_id} -> {response.get('response_mode')}")
        if sleep:
            time.sleep(sleep)

    results_df = pd.read_csv(RESULTS_CSV_PATH)
    summarise(results_df, "Full set")
    summarise(results_df[~results_df["exact_dup"]],
              "Excluding exact-duplicate resolution_text")
    summarise(results_df[~results_df["near_dup"]],
              "Excluding near-duplicate resolution_text (tiny subset, read with care)")
    log(f"\nBy response_mode:\n"
        f"{results_df.groupby('response_mode')[['bleu', 'rouge1_f1', 'rougeL_f1']].mean()}")
    log("\nCAVEAT: resolution_text is a short internal note, not customer-facing text, "
        "so BLEU/ROUGE are weak proxies here. They are secondary to the LLM-judge results.")
    log(f"\nPer-ticket results: {RESULTS_CSV_PATH}")
    return results_df


def run_human_eval_summary():
    log("\n--- Human review approval rate (reviewed/escalated tickets only) ---")
    conn = get_connection()
    rows = conn.execute(
        "SELECT human_action, COUNT(*) FROM tickets_log "
        "WHERE human_action IS NOT NULL GROUP BY human_action"
    ).fetchall()
    conn.close()

    if not rows:
        log("No human review decisions logged yet — review some escalated "
            "tickets in Streamlit first.")
        return

    counts = dict(rows)
    total = sum(counts.values())
    approved = counts.get("approved", 0)
    rejected = counts.get("rejected_edited", 0)

    log(f"Human review approval rate (n = {total} reviewed/escalated tickets)")
    log(f"Approved as-is: {approved} ({approved/total:.1%})")
    log(f"Rejected/corrected: {rejected} ({rejected/total:.1%})")
    log(
        "\nCAVEAT: these decisions come from the development log (22 reviews, some of "
        "them repeated test inputs, all on tickets that had already escalated, reviewed "
        "by the developer while exercising the learning loop). This describes the "
        "review workflow, not the quality of auto-resolved replies. It is not a random "
        "sample of all responses, so it is not overall response quality. See the "
        "LLM-judge results for quality."
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-size", type=int, default=50,
                    help="0 = every test ticket")
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--human-eval-only", action="store_true",
                        help="Skip BLEU/ROUGE, only summarize logged human decisions")
    args = parser.parse_args()

    log("PHASE 7 — RESPONSE AGENT: EVALUATION (BLEU/ROUGE + Human Eval)")
    log("=" * 60)

    Path(ROOT_DIR / "reports").mkdir(parents=True, exist_ok=True)

    if not args.human_eval_only:
        run_bleu_rouge_eval(args.sample_size, args.sleep, args.resume)
    run_human_eval_summary()

    with open(REPORT_PATH, "w") as f:
        f.write("\n".join(report_lines))
    log(f"\nSaved report to {REPORT_PATH}")


if __name__ == "__main__":
    main()