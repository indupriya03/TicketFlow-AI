"""
Phase 4 — Intake Agent: Intent Classification (zero-shot LLM via LangChain)
Multi-Agent Customer Support Intelligence Platform

Zero-shot classification via LangChain's ChatGroq wrapper — no training, no
ground-truth labels exist for intent, per the process doc. Uses LangChain's
.with_structured_output() with a Pydantic schema, so the LLM's response is
validated into a real Python object rather than manually parsed JSON. This
also means classify_intent() is close to a drop-in LangGraph node once the
graph is wired up, since LangChain objects integrate directly with it.

Requires: GROQ_API_KEY environment variable (get one at console.groq.com).
Never hardcode the key — ChatGroq reads it from the environment automatically.

Intent taxonomy (action-oriented, not topic-like): each label captures what
the customer wants DONE, which is finer-grained than ticket_category (which
only captures WHICH TEAM should handle it). Two tickets in the same category
can have different intents (e.g. "initiate_return" vs. "check_return_or_
order_status" are both plausibly "Refund & Return" category tickets).

Taxonomy includes two labels added after reviewing an initial spot-check:
- report_item_not_as_described: listing/description mismatches (wrong brand,
  wrong color, misleading photos) — distinct from genuine physical defects
- dispute_return_rejection: an existing return was rejected and the customer
  is contesting that decision — distinct from initiating a fresh return

complete: True means the LLM call succeeded and returned output matching the
IntentResult schema — a technical parse-success flag, not a semantic
"ticket fully understood" judgment.

Input:  data/processed/support_tickets_clean.csv
Output: reports/intent_spot_check.txt (manual review sample, since no
        automated metric is possible without ground-truth labels)
        data/processed/intent_sample_results.csv (raw results for the
        sampled tickets, for your own reference/reuse)
"""

import os
import time
import pandas as pd
from pathlib import Path
from enum import Enum
from pydantic import BaseModel, Field
from langchain.chat_models import init_chat_model
from src.guardrails import looks_like_prompt_injection

Root_DIR = Path(__file__).parent.parent.parent
INPUT_PATH = Root_DIR/"data/processed/support_tickets_clean.csv"
OUTPUT_CSV_PATH = Root_DIR/"data/processed/intent_sample_results.csv"
SPOT_CHECK_PATH = Root_DIR/"reports/intent_spot_check.txt"

# Set the GROQ_MODEL env var to switch: 20b = cheap iteration only, 120b = the run that counts.
GROQ_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")

SAMPLE_SIZE = 200  # matches the process doc's evaluation method (~200 tickets, manual review)

INTENT_LABELS = [
    "request_refund",
    "initiate_return",
    "request_exchange",
    "check_return_or_order_status",
    "dispute_charge_or_refund",
    "dispute_return_rejection",
    "report_defect",
    "report_item_not_as_described",
    "request_cancellation",
    "report_delivery_issue",
    "report_account_or_login_issue",
    "report_technical_issue",
    "escalate_unresponsive_seller",
    "request_information",
    "request_order_modification",
    "other",
]

# Enum built from the taxonomy list, so Pydantic/LangChain enforce it as a closed
# set at the schema level — the model literally cannot return a label outside this list.
IntentLabel = Enum("IntentLabel", {label: label for label in INTENT_LABELS})


class IntentResult(BaseModel):
    intent: IntentLabel = Field(# type: ignore[valid-type] 
        description="The single best-matching customer intent label.")
    reasoning: str = Field(description="One short sentence explaining the choice.")


SYSTEM_PROMPT = """You are classifying the intent of customer support tickets for an
e-commerce platform. Choose exactly ONE label that best captures what the customer
wants DONE (not just the general topic). Note the distinctions between similar labels:
- report_defect (physical damage/malfunction) vs. report_item_not_as_described
  (correct/undamaged item, but doesn't match the listing — wrong brand, color, photos)
- initiate_return (starting a fresh return) vs. dispute_return_rejection (a return
  already happened and was rejected; customer is contesting that decision)
- request_information is ONLY for genuine questions needing no action (e.g. policy
  questions, or general how-to questions that don't refer to a specific order,
  like "How do I track my order?" or "How do I start a return?"). Do not use it
  for requests to change or fix something.
- request_order_modification: customer wants existing order details changed or
  corrected (delivery address, invoice/GSTIN details) — not a new item, refund,
  or return; just a correction to information already on the order.
- request_cancellation also covers cancelling an unwanted AUTOMATED action (e.g.
  an auto-triggered return), not just a fresh order.
- check_return_or_order_status (customer just asks where an order or return is, or
  what its status is, with no problem stated) vs. report_delivery_issue (customer
  says delivery has gone wrong: overdue or "hasn't arrived in N days", stuck in
  transit, marked delivered but not received, lost package). If the ticket
  complains that something is late or missing, use report_delivery_issue.  A general how-to question with no specific order ("How do I track my order?")
  is NOT a status check; use request_information.
"""


report_lines = []


def log(msg):
    print(msg)
    report_lines.append(msg)


def classify_intent(structured_llm, ticket_text, retries=2):
    """Zero-shot intent classification for a single ticket. This function is
    the reusable piece that becomes the actual Intake Agent's Intent node
    once wired into LangGraph — everything else in this file is mainly used to 
    test the classifier and save its results."""
    if looks_like_prompt_injection(ticket_text):
        return {
            "intent": "other",
            "reasoning": "Ticket contains an instruction directed at the classifier rather than a genuine support request.",
            "complete": True,
            "flagged_injection": True,
        }

    for attempt in range(retries + 1):
        try:
            result: IntentResult = structured_llm.invoke([
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Ticket: {ticket_text}"},
            ])
            return {
                "intent": result.intent.value,
                "reasoning": result.reasoning,
                "complete": True,
                "flagged_injection": False,
            }
        except Exception as e:
            if attempt < retries:
                time.sleep(1)
                continue
            return {
                "intent": None,
                "reasoning": f"FAILED after {retries+1} attempts: {e}",
                "complete": False,
                "flagged_injection": False,
            }
_structured_llm = None

def get_structured_llm():
    global _structured_llm
    if _structured_llm is None:
        from langchain.chat_models import init_chat_model
        llm = init_chat_model(GROQ_MODEL, model_provider="groq", temperature=0)
        _structured_llm = llm.with_structured_output(IntentResult)
    return _structured_llm

def build_intent_sample(df, target_total=SAMPLE_SIZE, per_label_boost=10, seed=1):
    """
    Single-pass sample builder: keyword-boosts every intent label with a known
    pattern (so rare-but-real intents get guaranteed coverage), then fills the
    remaining slots with a plain random sample. Also probes request_cancellation
    and request_information with their own patterns, even though prior rounds
    found ~0 hits — keeping the probe in this function makes that "confirmed
    absent" finding reproducible in a single run, not just a one-off check.
    """
    patterns = {
        "request_exchange": r"\bexchange\b",
        "dispute_return_rejection": r"reject|refuse.*return",
        "dispute_charge_or_refund": r"charged|overcharged|wrong amount|dispute|refund.*wrong|cashback",
        "request_refund": r"\bwant.*refund\b|please refund|refund me",
        "report_defect": r"defect|broken|damaged|not working|dead pixels",
        "report_delivery_issue": r"delivery|delivered|courier|tracking",
        "report_account_or_login_issue": r"login|password|account.*access",
        "escalate_unresponsive_seller": r"seller.*not respond|unresponsive seller",
        "request_cancellation": r"\bcancel my order\b|want to cancel",   # probe: expect ~0
        "request_information": r"\bhow do i\b|\bwhat is\b|\bcan you tell me\b",  # probe: expect mostly non-info
    }

    used_ids = set()
    boosted_parts = []

    for label, pattern in patterns.items():
        hits = df[
            df["ticket_text"].str.contains(pattern, case=False, na=False, regex=True)
            & ~df["ticket_id"].isin(used_ids)
        ]
        n = min(per_label_boost, len(hits))
        if n > 0:
            picked = hits.sample(n=n, random_state=seed)
            boosted_parts.append(picked)
            used_ids.update(picked["ticket_id"])

    boosted = pd.concat(boosted_parts).drop_duplicates(subset="ticket_id") if boosted_parts else pd.DataFrame(columns=df.columns)

    remaining_n = max(0, target_total - len(boosted))
    remaining_pool = df[~df["ticket_id"].isin(boosted["ticket_id"])]
    rest = remaining_pool.sample(n=remaining_n, random_state=seed)

    sample = pd.concat([boosted, rest]).sample(frac=1, random_state=seed)  # shuffle
    return sample
def main():
    log("PHASE 4 — INTAKE AGENT: INTENT CLASSIFICATION (zero-shot, LangChain)")
    log("=" * 60)
    log(f"Model in use: {GROQ_MODEL}")
    if not os.environ.get("GROQ_API_KEY"):
        raise RuntimeError(
            "GROQ_API_KEY environment variable not set. Get a key at console.groq.com "
            "and set it: export GROQ_API_KEY=your_key_here"
        )
    Path(Root_DIR/"reports").mkdir(parents=True, exist_ok=True)
    structured_llm = get_structured_llm()

    df = pd.read_csv(INPUT_PATH)
    log(f"Loaded: {INPUT_PATH} ({df.shape})")

    sample = build_intent_sample(df, target_total=SAMPLE_SIZE, per_label_boost=10, seed=1).copy()    
    log(f"Sampling {SAMPLE_SIZE} tickets for evaluation (per process doc: no ground truth "
        f"exists for intent, so this is a manual-review sample, not a train/test split).")

    results = []
    for i, row in enumerate(sample.itertuples(), 1):
        result = classify_intent(structured_llm, row.ticket_text_clean)
        results.append(result)
        if i % 10 == 0:
            log(f"  ...classified {i}/{SAMPLE_SIZE}")

    sample["intent"] = [r["intent"] for r in results]
    sample["intent_reasoning"] = [r["reasoning"] for r in results]
    sample["complete"] = [r["complete"] for r in results]
    sample["intent_model"] = GROQ_MODEL

    n_failed = (~sample["complete"]).sum()
    log(f"\nCompleted: {SAMPLE_SIZE - n_failed}/{SAMPLE_SIZE} succeeded, {n_failed} failed to parse")

    log("\nIntent distribution in this sample:")    
    log(str(sample["intent"].value_counts()))

    sample[["ticket_id", "ticket_text", "ticket_category", "intent", "intent_reasoning", "complete", "intent_model"]].to_csv(
        OUTPUT_CSV_PATH, index=False
    )
    log(f"\nSaved sample results to {OUTPUT_CSV_PATH}")

    lines = []
    for row in sample.itertuples():
        lines.append(f"[{row.ticket_id}] (category: {row.ticket_category}) {row.ticket_text}")
        lines.append(f"  -> intent={row.intent} | reasoning: {row.intent_reasoning}")
        lines.append("")
    with open(SPOT_CHECK_PATH, "w") as f:
        f.write("\n".join(lines))
    log(f"Saved spot-check file to {SPOT_CHECK_PATH} — manually review before treating "
        f"Intent as done, per the process doc (no automated metric exists without labels).")


if __name__ == "__main__":
    main()