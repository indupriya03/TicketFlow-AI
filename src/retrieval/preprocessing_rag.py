"""
Phase 6 — Retrieval Agent: Document Preprocessing
Multi-Agent Customer Support Intelligence Platform

Builds the FAQ + resolution-template documents that go into the vector
store, and saves them as plain JSON — no embedding, no API calls, no
quota cost. Kept separate from retrieval_document.py (which reads this
file's output and embeds it) so the two concerns don't share a run:
tweaking or re-checking how documents are built (dedup logic, formatting,
metadata) never needs to touch Gemini's API, and re-embedding never needs
to re-derive the documents.

Run this whenever the FAQ or resolved-ticket source data changes.
retrieval_document.py should be re-run afterward to pick up the new file.

Input:  data/processed/faq_knowledge_base_clean.csv
        data/processed/support_tickets_train.csv
Output: data/processed/rag_documents.json
        reports/rag_preprocessing_report.txt
"""

from pathlib import Path

import pandas as pd
import re

ROOT_DIR = Path(__file__).parent.parent.parent
FAQ_PATH = ROOT_DIR / "data/processed/faq_knowledge_base_clean.csv"
TICKETS_PATH = ROOT_DIR / "data/processed/support_tickets_train.csv"
OUTPUT_PATH = ROOT_DIR / "data/processed/rag_documents.json"
REPORT_PATH = ROOT_DIR / "reports/rag_preprocessing_report.txt"

report_lines = []


def log(msg):
    print(msg)
    report_lines.append(msg)


def build_faq_documents(faq_df):
    """One record per FAQ entry — question+answer already confirmed short
    enough (max 12/19 words respectively) to need no chunking."""
    records = []
    for row in faq_df.itertuples():
        records.append({
            "id": f"faq_{row.faq_id}",
            "document": f"{row.question_clean} {row.answer_clean}",
            "metadata": {
                "source_type": "faq",
                "faq_id": row.faq_id,
                "category": row.category,
            },
        })
    return records


def _mask_product_name(resolution_text, product_name):
    """Mask the ticket's own product name within its resolution text
    before computing the dedup key. Without this, "Replacement protein
    powder dispatched after quality team verification." and "Replacement
    vitamin supplements dispatched after quality team verification." are
    treated as two different templates purely because they name different
    products — even though they're the same template pattern. Confirmed as
    a real, concrete problem via retrieval_eval.py's --show-misses output:
    queries were matching whichever "Replacement <product>..." document
    happened to share a product-name word with the query, regardless of
    whether that was actually the correct resolution for that ticket."""
    text = str(resolution_text)
    if pd.notna(product_name) and str(product_name).strip():
        text = re.sub(re.escape(str(product_name).strip()), "<PRODUCT>", text, flags=re.IGNORECASE)
    return text


def build_resolution_documents(tickets_df):
    """De-duplicate via Phase 3's templating-flag normalization (lowercase
    + strip digits), PLUS masking each row's own product_name first — see
    _mask_product_name. This is a deliberate refinement beyond Phase 3's
    own near-dup flag, driven by retrieval_eval.py's miss-inspection
    output rather than guessed: it does not change or contradict Phase 3's
    original EDA numbers, which measured a different, narrower question
    (near-dup after digit-stripping only) for a different purpose (BLEU/
    ROUGE evaluation subsetting), not retrieval-document deduplication.

    TRADE-OFF, worth remembering: collapsing these templates means the ONE
    surviving document for a "Replacement <product>..." pattern shows
    whichever specific product happened to survive dedup (e.g. "protein
    powder"), even when a matching query is about a different product
    (e.g. "beard trimmer"). Retrieval finds the right KIND of resolution
    more reliably; the literal wording shown may reference an unrelated
    product. Matters if this document text is ever shown to a customer
    directly rather than summarized/rewritten by a Response agent."""
    tickets_df = tickets_df.copy()
    masked = [
        _mask_product_name(text, pname)
        for text, pname in zip(tickets_df["resolution_text"], tickets_df["product_name"])
    ]
    tickets_df["resolution_stripped"] = (
        pd.Series(masked, index=tickets_df.index)
        .str.lower()
        .str.replace(r"\d+", "", regex=True)
    )
    deduped = tickets_df.drop_duplicates(subset="resolution_stripped")

    records = []
    for row in deduped.itertuples():
        records.append({
            "id": f"res_{row.ticket_id}",
            "document": row.resolution_text,
            "metadata": {
                "source_type": "resolution_template",
                "ticket_id": row.ticket_id,
                "category": getattr(row, "ticket_category", None) or "",
            },
        })
    return records


def main():
    log("PHASE 6 — RETRIEVAL AGENT: DOCUMENT PREPROCESSING")
    log("=" * 60)

    Path(ROOT_DIR / "reports").mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    faq_df = pd.read_csv(FAQ_PATH)
    tickets_df = pd.read_csv(TICKETS_PATH)
    log(f"Loaded FAQ: {FAQ_PATH} ({faq_df.shape})")
    log(f"Loaded tickets: {TICKETS_PATH} ({tickets_df.shape})")

    faq_records = build_faq_documents(faq_df)
    resolution_records = build_resolution_documents(tickets_df)
    log(f"FAQ documents: {len(faq_records)}")
    log(f"Resolution-template documents: {len(resolution_records)} "
        f"(deduplicated from {len(tickets_df)} raw tickets)")

    all_records = faq_records + resolution_records
    log(f"Total documents: {len(all_records)}")

    import json
    with open(OUTPUT_PATH, "w") as f:
        json.dump(all_records, f, indent=2, ensure_ascii=False)
    log(f"\nSaved {OUTPUT_PATH}")

    with open(REPORT_PATH, "w") as f:
        f.write("\n".join(report_lines))
    log(f"Saved report to {REPORT_PATH}")


if __name__ == "__main__":
    main()