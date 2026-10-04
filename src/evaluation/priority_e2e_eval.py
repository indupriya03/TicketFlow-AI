"""
Phase 5 — End-to-end Priority evaluation (rule + Neutral classifier)
Multi-Agent Customer Support Intelligence Platform

Runs the real assign_priority() on every row of the held-out test set, so the
number reflects the full pipeline (deterministic rule for Positive / Slightly
Negative / Negative / Very Negative + the trained classifier for Neutral),
not just the Neutral subset in priority_eval_report.txt.

Also re-checks the "sentiment determines priority" claim from
priority_assign.py's docstring against the TEST set: any non-Neutral row whose
true priority differs from the rule is listed as an exception.

Usage (from project root):
    python -m src.evaluation.priority_e2e_eval

Writes reports/priority_e2e_eval_report.txt and prints the same.
"""

from pathlib import Path

import pandas as pd
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score)

from src.triage.priority_assign import SENTIMENT_PRIORITY_RULE, assign_priority

ROOT_DIR = Path(__file__).parent.parent.parent

# --- ASSUMPTIONS: adjust these if your test file differs --------------------
TEST_PATH = ROOT_DIR / "data/processed/support_tickets_test.csv"
TEXT_COL = "ticket_text_clean"
SENTIMENT_COL = "sentiment"
LABEL_COL = "priority"   # same column name the retrainer merges into the train set
# -----------------------------------------------------------------------------

REPORT_PATH = ROOT_DIR / "reports/priority_e2e_eval_report.txt"
LABELS = ["Low", "Medium", "High"]


def main():
    df = pd.read_csv(TEST_PATH)
    missing = {TEXT_COL, SENTIMENT_COL, LABEL_COL} - set(df.columns)
    if missing:
        raise SystemExit(f"{TEST_PATH.name} is missing columns {sorted(missing)}. "
                         f"Found: {list(df.columns)}. Edit the constants at the top.")
    df = df.dropna(subset=[SENTIMENT_COL, LABEL_COL]).copy()
    df[TEXT_COL] = df[TEXT_COL].fillna("")

    df["pred"] = [assign_priority(s, t) for s, t in zip(df[SENTIMENT_COL], df[TEXT_COL])]
    df["via"] = df[SENTIMENT_COL].map(
        lambda s: "rule" if s in SENTIMENT_PRIORITY_RULE else "classifier")

    y, p = df[LABEL_COL], df["pred"]
    out = []
    out.append("PRIORITY — END-TO-END EVALUATION (rule + Neutral classifier)")
    out.append(f"Test file: {TEST_PATH.name} | rows: {len(df)}\n")
    out.append(f"Overall accuracy: {accuracy_score(y, p):.4f}")
    out.append(f"Macro F1: {f1_score(y, p, average='macro', labels=LABELS, zero_division=0):.4f}\n")

    out.append("Accuracy by component:")
    for via, g in df.groupby("via"):
        out.append(f"  {via:<11} n={len(g):<6} accuracy={accuracy_score(g[LABEL_COL], g['pred']):.4f}")

    out.append("\nPer-class report:")
    out.append(classification_report(y, p, labels=LABELS, digits=4, zero_division=0))

    out.append("Confusion matrix (rows = true, cols = predicted; order Low/Medium/High):")
    out.append(str(pd.DataFrame(confusion_matrix(y, p, labels=LABELS),
                                index=LABELS, columns=LABELS)))

    # Re-verify the determinism claim on the test set
    out.append("\nRule check on test set (non-Neutral sentiments):")
    exceptions = 0
    for sent, expected in SENTIMENT_PRIORITY_RULE.items():
        g = df[df[SENTIMENT_COL] == sent]
        bad = g[g[LABEL_COL] != expected]
        exceptions += len(bad)
        out.append(f"  {sent:<18} n={len(g):<6} expected={expected:<7} exceptions={len(bad)}")
    out.append("  -> rule holds with zero exceptions on the test set."
               if exceptions == 0 else
               f"  -> {exceptions} exception(s); the 'deterministic' claim does not fully hold.")

    unseen = sorted(set(df[SENTIMENT_COL]) - set(SENTIMENT_PRIORITY_RULE) - {"Neutral"})
    if unseen:
        out.append(f"  Unrecognized sentiment values in test set: {unseen}")

    text = "\n".join(out)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(text)
    print(text)
    print(f"\nSaved -> {REPORT_PATH}")


if __name__ == "__main__":
    main()