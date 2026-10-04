"""
Phase 3 — End-to-End Data Cleaning and Preprocessing Pipeline
Multi-Agent Customer Support Intelligence Platform
 
1. Light text cleaning (whitespace/HTML strip; order-ID normalization added
   per the Phase 2 EDA finding — the raw #ORD\\d+ pattern was polluting
   word-frequency/TF-IDF vocabulary as a fake "ord" token)
2. Deduplicate exact duplicate rows
3. Normalize categorical labels (casing/whitespace)
4. PII scrub pass (regex-based: emails, phone numbers, card numbers)
5. Resolution-text templating flag — tagged, not removed, so BLEU/ROUGE can
   be computed on the full set and on the de-duplicated subset. Counting note:
   Phase 1 reported extra copies only (75.8% exact, 98.41% near-dup); this
   step flags every row in a duplicated group (77.0% exact, 99.6% near-dup).
   Same data, different counting method.
6. Build leakage-safe text_features column
"""
import re
import pandas as pd
import os
from pathlib import Path
from sklearn.model_selection import StratifiedGroupKFold

Root_DIR = Path(__file__).parent.parent
print(f"Root directory: {Root_DIR}")
print(f"Current working directory: {os.getcwd()}")

INPUT_TICKETS = f"{Root_DIR}/data/raw/support_tickets_10k.csv"
INPUT_FAQ = f"{Root_DIR}/data/raw/faq_knowledge_base_150.csv"
OUTPUT_DIR = f"{Root_DIR}/data/processed"

LEAKAGE_COLS = [
    "confidence_score", "escalated", "resolution_text",
    "resolved_date", "resolution_days", "auto_resolved",
    "customer_satisfaction_score",
]
FEATURE_SOURCE_COLS = ["ticket_text_clean", "product_name", "product_segment"]
ORD_RE = r"(?<![A-Za-z0-9])#?ORD\d+" 

report_lines = []
 
def log(msg):
    print(msg)
    report_lines.append(msg)

# ---------------------------------------------------------------------------
# STEP 1 — Light text cleaning (+ order-ID normalization)
# ---------------------------------------------------------------------------
def light_clean_text(text):
    if pd.isna(text):
        return text
    t = text
    t = re.sub(r"<[^>]+>", "", t)                 # strip HTML tags
    t = re.sub(r"&[a-zA-Z]+;", " ", t)             # strip HTML entities
    t = re.sub(ORD_RE, "#ORDER_ID", t, flags=re.IGNORECASE)  # normalize order IDs (Phase 2 finding)
    t = re.sub(r"\s+", " ", t)                     # collapse whitespace
    return t.strip()
 
 
def step1_light_cleaning(df):
    log("\n=== STEP 1: Light Text Cleaning ===")
    df["ticket_text_clean"] = df["ticket_text"].apply(light_clean_text)
    changed = (df["ticket_text"] != df["ticket_text_clean"]).sum()
    order_id_hits = df["ticket_text"].str.contains(ORD_RE, case=False, regex=True, na=False).sum()    
    log(f"Rows with whitespace/HTML changes: {changed}/{len(df)} ({changed/len(df):.1%})")
    log(f"Rows with order-ID references normalized to #ORDER_ID: {order_id_hits}/{len(df)} ({order_id_hits/len(df):.1%})")  
    log(f"Order-ID pattern: {ORD_RE}")
    return df

# ---------------------------------------------------------------------------
# STEP 2 — Deduplicate exact duplicate rows
# ---------------------------------------------------------------------------
def step2_deduplicate(df):
    log("\n=== STEP 2: Deduplicate Exact Duplicate Rows ===")
    before = len(df)
    df = df.drop_duplicates(subset=["ticket_id"], keep="first")
    full_row_dupes = df.duplicated().sum()
    df = df.drop_duplicates(keep="first")
    after = len(df)
    log(f"Rows before: {before}, after: {after}, removed: {before - after}")
    log("(Phase 1 profiling found 0 duplicate ticket_id and 0 full-row duplicates — "
        "this step is expected to be a no-op, confirmed rather than assumed.)")
    return df


# ---------------------------------------------------------------------------
# STEP 3 — Normalize categorical labels
# ---------------------------------------------------------------------------
def normalize_category(value):
    """Scalar normalization for ONE categorical value — used by step3 below
    AND at inference (e.g. predict_sentiment), so both paths match exactly."""
    return re.sub(r"'S\b", "'s", str(value).strip().title())

def step3_normalize_categoricals(df):
    log("\n=== STEP 3: Normalize Categorical Labels ===")
    cat_cols = ["ticket_category", "priority", "sentiment", "escalated", "auto_resolved", "product_segment"]
    for col in cat_cols:
        before_vals = set(df[col].astype(str).unique())
        df[col] = df[col].apply(normalize_category)
        after_vals = set(df[col].unique())
        if before_vals != after_vals:
            log(
                f"{col}: "
                f"{len(before_vals)} raw values -> "
                f"{len(after_vals)} normalized values"
            )
            log(f"  Raw values: {sorted(before_vals)}")
            log(f"  Normalized values: {sorted(after_vals)}")
        else:
            log(
                f"{col}: already clean, "
                f"{len(after_vals)} unique values, no changes needed"
            )
    return df
# ---------------------------------------------------------------------------
# STEP 4 — PII scrub pass (regex-based)
# ---------------------------------------------------------------------------
EMAIL_RE = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")
PHONE_RE = re.compile(
    r"(?<!\d)(?:\+?91[-.\s]?|0)?[6-9]\d{4}[-.\s]?\d{5}(?!\d)"   # Indian mobile
    r"|"
    r"(?<!\d)(\+?\d{1,3}[-.\s]?)?\(?\d{3,4}\)?[-.\s]?\d{3,4}[-.\s]?\d{3,4}(?!\d)"  # your original
)
CARD_RE = re.compile(r"\b(?:\d[ -]*?){13,19}\b")
 
def pii_scrub(text):
    if pd.isna(text):
        return text, 0
    hits = 0
    t = text
    t, n = EMAIL_RE.subn("<EMAIL>", t)
    hits += n
    t, n = CARD_RE.subn("<CARD_NUMBER>", t)
    hits += n
    t, n = PHONE_RE.subn("<PHONE>", t)
    hits += n
    return t, hits
 
 
def step4_pii_scrub(df, text_col):
    log(f"\n=== STEP 4: PII Scrub Pass ({text_col}) ===")
    results = df[text_col].apply(pii_scrub)
    df[text_col] = results.apply(lambda x: x[0])
    total_hits = results.apply(lambda x: x[1]).sum()
    log(f"Total PII pattern matches redacted: {total_hits}")
    log("NOTE: this is a regex-only pass (email/phone/card patterns). It is not a substitute "
        "for a full spaCy NER pass — flag this explicitly in your report as a scoped MVP scrub, "
        "not a comprehensive PII removal guarantee.")
    return df


# ---------------------------------------------------------------------------
# STEP 5 — resolution_text templating flag
# Decision: retain resolution_text + add duplicate/template flags
# ---------------------------------------------------------------------------

def step5_templating_flag(df):
    log("\n=== STEP 5: resolution_text Templating Flag ===")

    # Flag resolution texts that are exactly repeated
    df["resolution_text_is_exact_dup"] = (
        df["resolution_text"].str.strip().duplicated(keep=False)
    )

    # Normalize resolution text for near-duplicate detection:
    # - lowercase
    # - remove numbers such as order/reference IDs
    # - remove leading/trailing whitespace
    normalized = (
        df["resolution_text"]
        .str.lower()
        .str.replace(r"\d+", "", regex=True)
        .str.strip()
    )

    # Flag resolution texts that become identical after normalization
    df["resolution_text_is_near_dup"] = (
        normalized.duplicated(keep=False)
    )

    exact_pct = df["resolution_text_is_exact_dup"].mean()
    near_pct = df["resolution_text_is_near_dup"].mean()

    log(
        f"Exact-duplicate resolution_text flagged: {exact_pct:.1%}"
    )

    log(
        f"Near-duplicate resolution_text flagged after "
        f"lowercasing and removing digits: {near_pct:.1%}"
    )

    log(
        "Interpretation: resolution_text is highly templated. "
        f"{exact_pct:.1%} of ticket rows use an exactly repeated "
        "resolution message, while "
        f"{near_pct:.1%} belong to resolution-text groups that "
        "become duplicates after numbers are removed."
    )

    log(
        "Decision: resolution_text is RETAINED rather than dropped "
        "because it is required as the reference text for downstream "
        "BLEU/ROUGE evaluation."
    )

    log(
        "Duplicate/template flags are retained so BLEU/ROUGE can be "
        "compared on the full evaluation set and on a non-duplicate "
        "subset, allowing the impact of templated resolutions to be assessed."
    )

    return df

# ---------------------------------------------------------------------------
# STEP 6 — Build leakage-safe text_features column
# ---------------------------------------------------------------------------
def build_text_features(clean_text, product_name, product_segment):
    """Scalar version of text_features — used by step6 below AND at inference
    (predict_sentiment), so both paths match exactly."""
    return re.sub(
        r"\s+", " ",
        f"{clean_text or ''} {product_name or ''} {product_segment or ''}"
    ).strip()

def step6_build_features(df):
    log("\n=== STEP 6: Build Leakage-Safe text_features ===")
    overlap = set(FEATURE_SOURCE_COLS) & set(LEAKAGE_COLS)
    assert not overlap, f"Leakage columns found in feature source: {overlap}"
    df["text_features"] = df.apply(
        lambda row: build_text_features(
                row["ticket_text_clean"], row["product_name"], row["product_segment"]
        ),
        axis=1,
    )
    log("Leakage assertion passed — text_features built only from ticket_text_clean, "
        "product_name, product_segment.")
    log(f"Excluded leakage/target columns confirmed absent from features: {LEAKAGE_COLS}")
    return df

# ---------------------------------------------------------------------------
# STEP 7 — Stratified train/test split
# ---------------------------------------------------------------------------
def step7_stratified_split(df):
    log("\n=== STEP 7: Stratified, Group-Safe Train/Test Split (80/20) ===")
    strat_key = df["ticket_category"].astype(str) + "_" + df["escalated"].astype(str)
    groups = df["ticket_text_clean"]

    # StratifiedGroupKFold keeps every group (rows sharing identical
    # ticket_text_clean) entirely on one side of the split, so no
    # duplicate/near-duplicate ticket can leak between train and test —
    # while still approximating stratification by ticket_category + escalated.
    # n_splits=5 gives an ~80/20 split; we take the first fold.
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    train_idx, test_idx = next(sgkf.split(df, strat_key, groups))
    train_df = df.iloc[train_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)

    log(f"Train: {len(train_df)} rows, Test: {len(test_df)} rows")
    log("Split via StratifiedGroupKFold on ticket_text_clean groups (prevents "
        "duplicate ticket text from appearing on both sides), stratified "
        "jointly on ticket_category + escalated.")

    overlap = set(train_df["ticket_text_clean"]) & set(test_df["ticket_text_clean"])
    log(f"Sanity check — ticket_text_clean values in BOTH train and test: "
        f"{len(overlap)} (should be 0)")

    for col in ["ticket_category", "escalated"]:
        train_prop = train_df[col].value_counts(normalize=True).round(3)
        test_prop = test_df[col].value_counts(normalize=True).round(3)
        max_diff = (train_prop - test_prop).abs().max()
        log(f"Max proportion difference (train vs test) for {col}: {max_diff:.3f}")
    return train_df, test_df

# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    log("PHASE 3 — END-TO-END DATA CLEANING PIPELINE")
    log("=" * 60)

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    tickets = pd.read_csv(INPUT_TICKETS)
    faq = pd.read_csv(INPUT_FAQ)
    log(f"Loaded support_tickets: {tickets.shape}")
    log(f"Loaded faq_knowledge_base: {faq.shape}")

    tickets = step1_light_cleaning(tickets)
    tickets = step2_deduplicate(tickets)
    tickets = step3_normalize_categoricals(tickets)
    tickets = step4_pii_scrub(tickets, "ticket_text_clean")
    tickets = step4_pii_scrub(tickets, "resolution_text")
    tickets = step5_templating_flag(tickets)
    tickets = step6_build_features(tickets)
    train_df, test_df = step7_stratified_split(tickets)

    log("\n=== FAQ Cleaning ===")
    faq["question_clean"] = faq["question"].apply(light_clean_text)
    faq["answer_clean"] = faq["answer"].apply(light_clean_text)
    # FAQ text is company-authored (no customer PII) and holds the support
    # email/phone customers need, so it is deliberately NOT PII-scrubbed.
    log(f"FAQ rows cleaned: {len(faq)}")

    tickets.to_csv(f"{OUTPUT_DIR}/support_tickets_clean.csv", index=False)
    train_df.to_csv(f"{OUTPUT_DIR}/support_tickets_train.csv", index=False)
    test_df.to_csv(f"{OUTPUT_DIR}/support_tickets_test.csv", index=False)
    faq.to_csv(f"{OUTPUT_DIR}/faq_knowledge_base_clean.csv", index=False)

    with open(f"{OUTPUT_DIR}/cleaning_report.txt", "w") as f:
        f.write("\n".join(report_lines))

    log("\n" + "=" * 60)
    log("Saved: support_tickets_clean.csv, support_tickets_train.csv, "
        "support_tickets_test.csv, faq_knowledge_base_clean.csv, cleaning_report.txt")

if __name__ == "__main__":
    main()
