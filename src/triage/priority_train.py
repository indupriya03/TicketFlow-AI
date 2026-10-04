"""
Phase 5 — Triage: Priority Sub-Agent (Training — Neutral subset only)
Multi-Agent Customer Support Intelligence Platform

Trains a binary classifier (Low vs Medium) on frozen all-MiniLM-L6-v2
embeddings, using ONLY the rows where sentiment == "Neutral" — the one
sentiment value that doesn't deterministically map to a priority (see
assign_priority.py for the full rationale and the rule handling the other
3 sentiment values).

Both LogisticRegression and XGBoost are trained and evaluated on the
IDENTICAL train/test split and embeddings, so their scores are directly
comparable. XGBoost is the production choice:
  - CV macro F1 (train, 5-fold): LR 0.8853 vs XGB 0.8869 -- essentially tied
  - Test macro F1: LR 0.8195 vs XGB 0.8536 -- XGB ahead by 0.034
  - CV -> test drop: LR -0.0658 vs XGB -0.0333 -- XGB generalized more
    consistently on this split
  - Error profile: LR mislabels 56/355 true-Medium tickets as Low (the
    costly mistake -- a genuinely urgent ticket gets silently downgraded);
    XGBoost mislabels only 8/355 that way, at the cost of over-flagging
    51/159 true-Low tickets as Medium (the cheap mistake -- extra handling
    time, nothing falls through the cracks)
  - For a support-triage system, under-triaging is the expensive failure
    mode, so XGBoost's error pattern is the safer one even though CV alone
    doesn't show it as a stronger model in general

Class imbalance (Medium=1502, Low=716 in the Neutral train subset) is
handled per-model: XGBoost via scale_pos_weight (count(Medium)/count(Low),
with Low explicitly mapped to label 1 since scale_pos_weight always
upweights class 1); LogisticRegression via its built-in class_weight=
"balanced".

Only the winning model (XGBoost) and its report are saved to disk. The
LogisticRegression run exists purely as the in-memory baseline comparison
recorded in the combined report below -- see the header comment for the
rationale this project used, in case that choice needs revisiting later.

Uses preprocessing.py's train/test split output directly, filtered down to
the Neutral subset -- consistent with how category_train.py reuses the same
canonical split rather than re-splitting.

Input:  data/processed/support_tickets_train.csv, support_tickets_test.csv
Output: models/priority_neutral_classifier.joblib   (winning model only)
        reports/priority_eval_report.txt            (both models' results)
"""

from pathlib import Path

import joblib
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score

from src.embedding import get_embedder

ROOT_DIR = Path(__file__).parent.parent.parent
TRAIN_PATH = ROOT_DIR / "data/processed/support_tickets_train.csv"
TEST_PATH = ROOT_DIR / "data/processed/support_tickets_test.csv"
MODEL_PATH = ROOT_DIR / "models/priority_neutral_classifier.joblib"
REPORT_PATH = ROOT_DIR / "reports/priority_eval_report.txt"
EMBEDDER_NAME = "all-MiniLM-L6-v2"
RANDOM_STATE = 42

report_lines = []


def log(msg):
    print(msg)
    report_lines.append(msg)


def evaluate(name, clf, X_test, y_test, inv_label_map, label_map):
    """Predict on the held-out test set and log F1 / report / confusion matrix."""
    y_pred_enc = clf.predict(X_test)
    y_pred = pd.Series(y_pred_enc).map(inv_label_map).values
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    log(f"\n--- {name}: test set ---")
    log(f"Macro F1 (Low vs Medium, Neutral subset, test set): {macro_f1:.4f}")
    log(f"\nClassification report:\n{classification_report(y_test, y_pred)}")
    log(f"\nConfusion matrix (rows=true, cols=pred, labels={sorted(label_map.keys())}):\n"
        f"{confusion_matrix(y_test, y_pred, labels=sorted(label_map.keys()))}")
    return macro_f1


def main(train_df=None):
    """train_df: optional DataFrame to train on (used by retrain_from_feedback.py
    to pass base data + verified feedback merged in memory). If None, reads
    TRAIN_PATH as before."""
    report_lines.clear()
    log("PHASE 5 — TRIAGE: PRIORITY (train, Neutral subset only)")
    log("Compares LogisticRegression vs XGBoost on the identical split; "
        "only the winner is saved.")
    log("=" * 60)

    Path(ROOT_DIR / "reports").mkdir(parents=True, exist_ok=True)
    Path(ROOT_DIR / "models").mkdir(parents=True, exist_ok=True)

    if train_df is None:
        train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    log(f"Loaded train: {train_df.shape}, test: {test_df.shape}")

    train_neutral = train_df[train_df["sentiment"] == "Neutral"].copy()
    test_neutral = test_df[test_df["sentiment"] == "Neutral"].copy()
    log(f"Neutral subset — train: {len(train_neutral)}, test: {len(test_neutral)}")
    log(f"\nPriority distribution within Neutral (train):\n"
        f"{train_neutral['priority'].value_counts()}")

    if set(train_neutral["priority"].unique()) - {"Low", "Medium"}:
        raise ValueError(
            "Neutral subset contains a priority value other than Low/Medium — "
            "the determinism assumption behind this design may not hold on "
            "this data. Re-check before proceeding."
        )

    embedder = get_embedder(EMBEDDER_NAME)
    log(f"\nEmbedding with frozen {EMBEDDER_NAME}...")
    X_train = embedder.encode(train_neutral["ticket_text_clean"].tolist(), show_progress_bar=True)
    X_test = embedder.encode(test_neutral["ticket_text_clean"].tolist(), show_progress_bar=True)

    # Force Low -> 1, Medium -> 0 so XGBoost's scale_pos_weight (which always
    # upweights class 1) boosts the minority class, not the majority.
    label_map = {"Low": 1, "Medium": 0}
    inv_label_map = {v: k for k, v in label_map.items()}
    y_train_enc = train_neutral["priority"].map(label_map).values
    y_test = test_neutral["priority"].values

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    # ---------------------------------------------------------------
    # Baseline: LogisticRegression (in-memory comparison only — not saved)
    # ---------------------------------------------------------------
    lr_clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE)
    lr_clf.fit(X_train, y_train_enc)

    lr_cv_scores = cross_val_score(
        LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE),
        X_train, y_train_enc, cv=cv, scoring="f1_macro",
    )
    log(f"\nLogisticRegression 5-fold CV macro F1 (train set): "
        f"mean={lr_cv_scores.mean():.4f}, std={lr_cv_scores.std():.4f}")
    log(f"CV fold scores: {lr_cv_scores}")

    lr_macro_f1 = evaluate("LogisticRegression", lr_clf, X_test, y_test, inv_label_map, label_map)

    # ---------------------------------------------------------------
    # Candidate: XGBoost (this is the one that gets saved if it wins)
    # ---------------------------------------------------------------
    n_medium = (y_train_enc == 0).sum()
    n_low = (y_train_enc == 1).sum()
    scale_pos_weight = n_medium / n_low
    log(f"\nscale_pos_weight = {scale_pos_weight:.4f} (Medium={n_medium}, Low={n_low})")

    xgb_clf = xgb.XGBClassifier(
        random_state=RANDOM_STATE,
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight,
    )
    xgb_clf.fit(X_train, y_train_enc)

    xgb_cv_scores = cross_val_score(
        xgb.XGBClassifier(
            random_state=RANDOM_STATE, eval_metric="logloss", scale_pos_weight=scale_pos_weight,
        ),
        X_train, y_train_enc, cv=cv, scoring="f1_macro",
    )
    log(f"\nXGBoost 5-fold CV macro F1 (train set): "
        f"mean={xgb_cv_scores.mean():.4f}, std={xgb_cv_scores.std():.4f}")
    log(f"CV fold scores: {xgb_cv_scores}")

    xgb_macro_f1 = evaluate("XGBoost", xgb_clf, X_test, y_test, inv_label_map, label_map)

    # ---------------------------------------------------------------
    # Decide winner and save ONLY that model
    # ---------------------------------------------------------------
    log(f"\n{'=' * 60}")
    log(f"COMPARISON: LogisticRegression={lr_macro_f1:.4f}  vs  XGBoost={xgb_macro_f1:.4f}")

    if xgb_macro_f1 >= lr_macro_f1:
        winner_name, winner_clf = "XGBoost", xgb_clf
    else:
        winner_name, winner_clf = "LogisticRegression", lr_clf

    log(f"Winner: {winner_name} (only this model is saved to {MODEL_PATH})")

    joblib.dump(
        {
            "classifier": winner_clf,
            "model_type": winner_name,
            "embedder_name": EMBEDDER_NAME,
            "label_map": label_map,
            "labels": sorted(label_map.keys()),
        },
        MODEL_PATH,
    )
    log(f"\nSaved model to {MODEL_PATH}")

    with open(REPORT_PATH, "w") as f:
        f.write("\n".join(report_lines))
    log(f"Saved report to {REPORT_PATH}")


if __name__ == "__main__":
    main()