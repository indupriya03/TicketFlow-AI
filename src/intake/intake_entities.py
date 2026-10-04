"""
Phase 4 — Intake Agent: Entity Extraction (rule-based, no model)
Multi-Agent Customer Support Intelligence Platform

Extracts structured entities from raw ticket_text using regex only —
per the process doc, this sub-agent is explicitly rule-based, no ML model.

Entities extracted (chosen based on what actually appears in the data,
verified during EDA — not a generic NER schema):
- order_ids: #ORD\\d+ pattern, present in ~78% of tickets
- amounts: $/₹ currency amounts, present in ~2.3% of tickets
- product_mentioned: whether the customer's own wording names the product
  (vs. it only being known from the separate product_name column)

Reads from data/processed/support_tickets_clean.csv — the ORIGINAL
ticket_text column survives Phase 3 cleaning untouched (only the new
ticket_text_clean column has order IDs replaced with #ORDER_ID), so
this can safely be downstream of preprocessing.py rather than an
independent branch off data/raw. This also means the output file
contains both Phase 3 and Phase 4 columns together, no merge step needed.

Input:  data/processed/support_tickets_clean.csv
Output: data/processed/support_tickets_with_entities.csv
        data/processed/entity_spot_check.txt (manual review sample,
        per the process doc's evaluation method: spot-check ~50 tickets)
"""

import re
import pandas as pd
from pathlib import Path

Root_DIR = Path(__file__).parent.parent.parent
INPUT_PATH = Root_DIR/"data/processed/support_tickets_clean.csv"
OUTPUT_DIR = Root_DIR/"reports"
OUTPUT_PATH = Root_DIR/"data/processed/support_tickets_with_entities.csv"
SPOT_CHECK_PATH = OUTPUT_DIR/"entity_spot_check.txt"

ORDER_ID_RE = re.compile(r"(?<![A-Za-z0-9])#?(ORD\d+)", re.IGNORECASE)
MONEY_RE = re.compile(
    r"[\$₹€£]\s?\d+(?:,\d{3})*(?:\.\d{1,2})?"
    r"|\d+(?:,\d{3})*(?:\.\d{1,2})?\s?[\$₹€£]"
)
DAYS_ELAPSED_RE = re.compile(
    r'(\d+)\s*days?\s*(?:ago|passed)|been\s+(\d+)\s*days?|over a week ago',
    re.IGNORECASE,
)
PRIOR_CONTACT_RE = re.compile(
    r"\bignor(?:ed|ing)\s+(?:me|my|us|our)\b"
    r"|\b(?:no|zero)\s+(?:response|reply|answer)s?\b(?!\s+(?:needed|necessary|required))"
    r"|\b(?:nobody|no\s?one|no-one)\s+(?:has\s+|have\s+)?(?:replied|responded|answered|got\s+back|helped|is\s+helping)\b"
    r"|\b(?:haven'?t|have\s+not|hasn'?t|has\s+not|never)\s+(?:heard\s+back|(?:received|got)\s+(?:a|any)\s+(?:reply|response|answer))"
    r"|\bstill\s+waiting\b"
    r"|\b(?:contacted|emailed|messaged|called|reached\s+out)\b[^.!?]{0,40}\b(?:multiple|several|many)\s+times\b",
    re.IGNORECASE,
)
report_lines = []


def log(msg):
    print(msg)
    report_lines.append(msg)


def extract_days_elapsed(text):
    values = []
    for match in DAYS_ELAPSED_RE.finditer(text):
        number = match.group(1) or match.group(2)
        values.append(int(number) if number else 7)  # "over a week ago" is approximate
    return values

def _parse_amount(raw_match):
    """Strips currency symbols/commas from a MONEY_RE match and returns a
    float. Raw matches look like '$30', '€25', '25€', '₹1,250.50'."""
    cleaned = re.sub(r"[^\d.]", "", raw_match)
    return float(cleaned) if cleaned else None


def extract_entities(raw_text, product_name):
    if pd.isna(raw_text):
        return {"order_ids": [], "amounts": [], "product_mentioned": False,
                "days_elapsed": [], "prior_contact_unresolved": False}
    order_ids = ORDER_ID_RE.findall(raw_text)
    amounts = [a for a in (_parse_amount(m) for m in MONEY_RE.findall(raw_text)) if a is not None]
    days_elapsed = extract_days_elapsed(raw_text)
    product_mentioned = bool(product_name) and str(product_name).lower() in raw_text.lower()
    return {
        "order_ids": order_ids,
        "amounts": amounts,
        "product_mentioned": product_mentioned,
        "days_elapsed": days_elapsed,
        "prior_contact_unresolved": bool(PRIOR_CONTACT_RE.search(raw_text)),
    }


def run_entity_extraction(df):
    log("=== Intake — Entity Extraction ===")

    entities = df.apply(
        lambda row: extract_entities(row["ticket_text"], row.get("product_name")), axis=1
    )
    df["entity_order_ids"] = entities.apply(lambda e: e["order_ids"])
    df["entity_amounts"] = entities.apply(lambda e: e["amounts"])
    df["entity_product_mentioned"] = entities.apply(lambda e: e["product_mentioned"])
    df["entity_days_elapsed"] = entities.apply(lambda e: e["days_elapsed"])
    df["entity_prior_contact_unresolved"] = entities.apply(lambda e: e["prior_contact_unresolved"])
    n_prior = df["entity_prior_contact_unresolved"].sum()
    log(f"Tickets flagged prior_contact_unresolved: {n_prior}/{len(df)} ({n_prior/len(df):.1%})")
    n_order = (df["entity_order_ids"].str.len() > 0).sum()
    n_amount = (df["entity_amounts"].str.len() > 0).sum()
    n_product = df["entity_product_mentioned"].sum()
    n_days = (df["entity_days_elapsed"].str.len() > 0).sum()

    log(f"Tickets with at least one order_id extracted: {n_order}/{len(df)} ({n_order/len(df):.1%})")
    log(f"Tickets with at least one amount extracted: {n_amount}/{len(df)} ({n_amount/len(df):.1%})")
    log(f"Tickets where product_name is explicitly mentioned in text: {n_product}/{len(df)} ({n_product/len(df):.1%})")
    log(f"Tickets with at least one days-elapsed mention extracted: {n_days}/{len(df)} ({n_days/len(df):.1%})")

    return df


def write_spot_check(df, n=50, seed=1):
    log(f"\nWriting spot-check sample (n={n}) for manual review, per process doc's "
        f"evaluation method for the Entities sub-agent.")

    # Stratified, not pure random: guarantee rare entity types (amounts, ~2.3% base
    # rate) are represented rather than hoping a random seed happens to include them.
    has_amount = df[df["entity_amounts"].str.len() > 0]
    no_amount = df[df["entity_amounts"].str.len() == 0]

    n_amount_examples = min(10, len(has_amount))
    amount_sample = has_amount.sample(n=n_amount_examples, random_state=seed)
    remaining = n - n_amount_examples
    random_sample = no_amount.sample(n=remaining, random_state=seed)

    sample = pd.concat([amount_sample, random_sample]).sample(frac=1, random_state=seed)
    sample = sample[
        ["ticket_id", "ticket_text", "product_name", "entity_order_ids", "entity_amounts",
        "entity_product_mentioned", "entity_days_elapsed"]
    ]
    log(f"Sample composition: {n_amount_examples} guaranteed amount-bearing tickets + "
        f"{remaining} randomly sampled tickets, shuffled together.")

    lines = []
    for _, row in sample.iterrows():
        lines.append(f"[{row['ticket_id']}] {row['ticket_text']}")
        lines.append(
            f"  -> order_ids={row['entity_order_ids']} | amounts={row['entity_amounts']} "
            f"| product_mentioned={row['entity_product_mentioned']} | days_elapsed={row['entity_days_elapsed']}"
        )
        lines.append("")

    lines.append("=" * 60)
    lines.append("KNOWN DATA ARTIFACTS (identified during manual spot-check review)")
    lines.append("=" * 60)
    lines.append(
        "- 'Screen ... dead pixels' template mismatch: 107/10,000 tickets (1.07%)\n "
        "apply a screen/display defect template to non-electronic products (e.g. "
        "lipstick, board game, saree).\n Verified this does NOT propagate into "
        "resolution_text (resolutions are generic/defect-type-based,\n not verbatim "
        "reflections of ticket_text), so Retrieval/Response grounding is unaffected.\n "
        "Does not affect Category/Priority/Sentiment/Escalation, which use "
        "statistical text patterns rather than semantic coherence.\n Decision: "
        "documented as a known synthetic-data generation artifact, \n not corrected "
        "in the raw data, to preserve reproducibility of Phase 1-3 profiling.\n"
        "- Same mismatch pattern also seen with other defect templates ('noise during operation', "
        "'stopped working') on non-electronic products (e.g. cap, hair oil). Observed in the "
        "50-ticket sample only; not counted across the full dataset. Same decision as above."
    )

    with open(SPOT_CHECK_PATH, "w") as f:
        f.write("\n".join(lines))
    log(f"Saved spot-check sample + known-issues note to {SPOT_CHECK_PATH} — manually "
        f"review before treating Entities as done, per the process doc.")


def main():
    log("PHASE 4 — INTAKE AGENT: ENTITY EXTRACTION")
    log("=" * 60)

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_PATH)
    log(f"Loaded: {INPUT_PATH} ({df.shape})")

    df = run_entity_extraction(df)
    write_spot_check(df, n=50)

    df.to_csv(OUTPUT_PATH, index=False)
    log(f"\nSaved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()