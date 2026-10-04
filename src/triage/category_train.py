"""
Phase 5 — Classification Agent: Category Sub-Agent (Training)
Multi-Agent Customer Support Intelligence Platform

Trains a frozen-embedding + LogisticRegression classifier for ticket_category,
using the same architecture as the Sentiment sub-agent (Intake) for
consistency across the pipeline.

Input:  data/processed/support_tickets_train.csv, support_tickets_test.csv
Output: models/category_classifier.joblib
        reports/category_eval_report.txt
"""

from pathlib import Path
import joblib
import pandas as pd
from src.embedding import get_embedder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score

ROOT_DIR = Path(__file__).parent.parent.parent
TRAIN_PATH = ROOT_DIR / "data/processed/support_tickets_train.csv"
TEST_PATH = ROOT_DIR / "data/processed/support_tickets_test.csv"
MODEL_PATH = ROOT_DIR / "models/category_classifier.joblib"
REPORT_PATH = ROOT_DIR / "reports/category_eval_report.txt"
EMBEDDER_NAME = "all-MiniLM-L6-v2"
RANDOM_STATE = 42

report_lines = []


def log(msg):
    print(msg)
    report_lines.append(msg)


def main(train_df=None):
    """train_df: optional DataFrame to train on (used by retrain_from_feedback.py
    to pass base data + verified feedback merged in memory). If None, reads
    TRAIN_PATH as before."""
    report_lines.clear()
    log("PHASE 5 — CLASSIFICATION AGENT: CATEGORY (train)")
    log("=" * 60)

    Path(ROOT_DIR / "reports").mkdir(parents=True, exist_ok=True)
    Path(ROOT_DIR / "models").mkdir(parents=True, exist_ok=True)

    if train_df is None:
        train_df = pd.read_csv(TRAIN_PATH)
        log(f"Loaded train: {TRAIN_PATH} ({train_df.shape})")
    else:
        log(f"Using in-memory train set ({train_df.shape}) — base data plus verified feedback")
    test_df = pd.read_csv(TEST_PATH)
    log(f"Loaded test: {TEST_PATH} ({test_df.shape})")
    log(f"\nCategory distribution (train):\n{train_df['ticket_category'].value_counts()}")

    log(f"\nEmbedding with frozen {EMBEDDER_NAME} (this may take a minute)...")
    embedder = get_embedder(EMBEDDER_NAME)    
    X_train = embedder.encode(train_df["ticket_text_clean"].tolist(), show_progress_bar=True)
    X_test = embedder.encode(test_df["ticket_text_clean"].tolist(), show_progress_bar=True)
    y_train = train_df["ticket_category"].values
    y_test = test_df["ticket_category"].values
    log(f"Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")

    # class_weight='balanced' protects the smallest class (App & Website Issue,
    # 717 rows) from being swallowed by the largest (Refund & Return, 2051 rows).
    clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    log(f"\nMacro F1: {macro_f1:.4f}")
    log(f"\nClassification report:\n{classification_report(y_test, y_pred)}")
    log(f"\nConfusion matrix (rows=true, cols=pred, labels={sorted(set(clf.classes_))}):\n"
        f"{confusion_matrix(y_test, y_pred, labels=sorted(set(clf.classes_)))}")

    joblib.dump(
        {"classifier": clf, "embedder_name": EMBEDDER_NAME, "labels": sorted(set(clf.classes_))},
        MODEL_PATH,
    )
    log(f"\nSaved model to {MODEL_PATH}")

    with open(REPORT_PATH, "w") as f:
        f.write("\n".join(report_lines))
    log(f"Saved report to {REPORT_PATH}")


if __name__ == "__main__":
    main()