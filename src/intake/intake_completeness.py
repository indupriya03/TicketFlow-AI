"""
Phase 4 — Intake Agent: Completeness Check (rule-based, no model)
Multi-Agent Customer Support Intelligence Platform

check_completeness(intent, entity_order_ids) is the real production function —
called per-ticket as a LangGraph node using that ticket's already-computed
intent and entity_order_ids. No CSV involved at inference time.

This file's main() is a TEST HARNESS only: it merges Intent's 200-ticket
validation sample (intent_sample_results.csv) with entity_order_ids from
Entities' full-dataset output (support_tickets_with_entities.csv), on
ticket_id, purely to validate the logic against real classified data.
"""

import ast
import pandas as pd
from pathlib import Path

Root_DIR = Path(__file__).parent.parent.parent
ENTITIES_PATH = Root_DIR/"data/processed/support_tickets_with_entities.csv"
INTENT_SAMPLE_PATH = Root_DIR/"data/processed/intent_sample_results.csv"
OUTPUT_PATH = Root_DIR/"data/processed/completeness_test_results.csv"
REPORT_PATH = Root_DIR/"reports/completeness_check_report.txt"

# Intents where we can't act without finding the order, so Intake asks for the ID.
# Complaint-style intents (report_defect, report_delivery_issue,
# report_item_not_as_described, escalate_unresponsive_seller) are deliberately
# NOT here: the customer is describing a problem, not requesting a lookup,
# so the pipeline proceeds without an order ID.
NEEDS_ORDER_ID = {
    "request_refund", "initiate_return", "request_exchange",
    "check_return_or_order_status", "dispute_charge_or_refund",
    "dispute_return_rejection", "request_cancellation",
    "request_order_modification",
}

report_lines = []


def log(msg):
    print(msg)
    report_lines.append(msg)


def check_completeness(intent, entity_order_ids):
    if isinstance(entity_order_ids, str):
       try:
           entity_order_ids = ast.literal_eval(entity_order_ids)
       except (ValueError, SyntaxError):
           entity_order_ids = []
    elif not isinstance(entity_order_ids, (list, tuple)):
       entity_order_ids = []

    missing = []
    if intent in NEEDS_ORDER_ID and not entity_order_ids:
        missing.append("order_id")

    return {"is_complete": len(missing) == 0, "missing_fields": missing}


def run_completeness_check(df):
    log("=== Intake — Completeness Check ===")

    if len(df) == 0:
        log("Input dataframe is empty — nothing to check.")
        df["is_complete"] = pd.Series(dtype=bool)
        df["missing_fields"] = pd.Series(dtype=object)
        return df

    results = df.apply(
        lambda row: check_completeness(row["intent"], row["entity_order_ids"]),
        axis=1,
    )
    df["is_complete"] = results.apply(lambda r: r["is_complete"])
    df["missing_fields"] = results.apply(lambda r: r["missing_fields"])

    n_incomplete = (~df["is_complete"]).sum()
    log(f"Tickets flagged incomplete: {n_incomplete}/{len(df)} ({n_incomplete/len(df):.1%})")
    log("\nIncomplete breakdown by intent:")
    log(str(df[~df["is_complete"]]["intent"].value_counts()))

    return df


def main():
    log("PHASE 4 — INTAKE AGENT: COMPLETENESS CHECK (test harness)")
    log("=" * 60)

    Path(Root_DIR/"reports").mkdir(parents=True, exist_ok=True)

    entities_df = pd.read_csv(ENTITIES_PATH)
    intent_sample = pd.read_csv(INTENT_SAMPLE_PATH)
    log(f"Loaded entities: {entities_df.shape}, intent sample: {intent_sample.shape}")

    df = intent_sample.merge(entities_df[["ticket_id", "entity_order_ids"]], on="ticket_id", how="left")
    log(f"Merged: {df.shape}")

    df = run_completeness_check(df)

    df.to_csv(OUTPUT_PATH, index=False)
    log(f"\nSaved: {OUTPUT_PATH}")

    with open(REPORT_PATH, "w") as f:
        f.write("\n".join(report_lines))
    log(f"Saved report to {REPORT_PATH}")


if __name__ == "__main__":
    main()