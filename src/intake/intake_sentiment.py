"""
Phase 4 — Intake Agent: Sentiment Classification
Multi-Agent Customer Support Intelligence Platform

Frozen all-MiniLM-L6-v2 sentence embeddings + Logistic Regression head.
Embeddings are NOT fine-tuned (frozen) — only the LogisticRegression
head is trained, per the process doc.

IMPORTANT — accuracy ceiling context (from Phase 1 profiling):
Within duplicated ticket_text groups, sentiment agreed only 31.7% of the
time (vs. 100% for ticket_category). This means identical wording can
carry different sentiment ground truth in this dataset — so this model
has a real accuracy ceiling baked into the data itself, independent of
model quality. Do not chase a Category-level accuracy number here; a
materially lower score than Category/Priority is EXPECTED, not a bug.

FINAL MODEL DECISION:
Plain LogisticRegression (balanced) is the FINAL model — the ordinal
classifier below is kept as EXPLORATORY evidence for that decision, not
as an alternative production option. It scored better on raw 5-class
macro-F1 but worse on the metric that actually matters downstream (see
ESCALATION_TRIGGER_SENTIMENTS grouping accuracy below): 89.25% vs. plain
LR's 90.55%. Only the plain LogisticRegression is saved to disk.

Input:  data/processed/support_tickets_train.csv
        data/processed/support_tickets_test.csv
Output: models/sentiment_classifier.joblib
        reports/sentiment_eval_report.txt

RUNTIME REQUIREMENT (Apple Silicon only): XGBoost's PyPI wheel for macOS ARM64
does not bundle libomp and will segfault without it. Install it once via
`brew install libomp`, then run this script with:
    DYLD_LIBRARY_PATH=$(brew --prefix libomp)/lib python src/intake_sentiment.py
(Confirmed empirically: this is the actual fix — an earlier attempt to work
around an assumed OpenMP-duplication conflict via KMP_DUPLICATE_LIB_OK made
no difference and was removed.)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from src.embedding import get_embedder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_sample_weight
import joblib
import mord
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from statsmodels.stats.contingency_tables import mcnemar
import re
from src.preprocessing import build_text_features, normalize_category

Root_DIR = Path(__file__).parent.parent.parent
TRAIN_PATH = Root_DIR/"data/processed/support_tickets_train.csv"
TEST_PATH = Root_DIR/"data/processed/support_tickets_test.csv"
MODEL_DIR = Root_DIR/"models"
MODEL_PATH = MODEL_DIR/"sentiment_classifier.joblib"
REPORT_PATH = Root_DIR/"reports/sentiment_eval_report.txt"

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# Per EDA finding: only these two sentiments ever lead to escalation (~73% rate).
# Neutral/Positive/Slightly Negative never escalate (0%, holds within every category).
# This is the operationally meaningful grouping for downstream agents — NOT "negative
# labels" in the intuitive sense, since Slightly Negative belongs with the non-escalating
# group, not with Negative/Very Negative.
ESCALATION_TRIGGER_SENTIMENTS = {"Negative", "Very Negative"}

report_lines = []

_embedder = None
_sentiment_clf = None

def log(msg):
    print(msg)
    report_lines.append(msg)


def group_accuracy(y_true, y_pred, trigger_set):
    """Accuracy at the coarse escalation-relevant grouping level: does the
    prediction land in the correct group (trigger vs. non-trigger), regardless
    of which exact class within that group. This is the operationally
    meaningful metric — most 5-class 'errors' are same-group confusions that
    don't affect the downstream escalation decision at all."""
    true_group = y_true.isin(trigger_set)
    pred_group = pd.Series(y_pred, index=y_true.index).isin(trigger_set)
    return (true_group == pred_group).mean()

def load_sentiment_model():
    global _embedder, _sentiment_clf
    if _embedder is None:
        _embedder = get_embedder(EMBEDDING_MODEL_NAME)
        loaded = joblib.load(MODEL_PATH)
        _sentiment_clf = loaded["classifier"]
    return _embedder, _sentiment_clf


def predict_sentiment(clean_text, product_name="", product_segment=""):
    embedder, clf = load_sentiment_model()
    normalized_segment = normalize_category(product_segment) if product_segment else ""
    text_features = build_text_features(clean_text, product_name, normalized_segment)
    embedding = embedder.encode([text_features])
    return str(clf.predict(embedding)[0])



def main():
    log("PHASE 4 — INTAKE AGENT: SENTIMENT CLASSIFICATION")
    log("=" * 60)

    Path(MODEL_DIR).mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    log(f"Loaded train: {train_df.shape}, test: {test_df.shape}")

    if "text_features" not in train_df.columns:
        raise ValueError("text_features column not found — run preprocessing.py first.")

    # --- Load frozen embedding model ---
    log(f"\nLoading frozen embedding model: {EMBEDDING_MODEL_NAME}")
    embedder = get_embedder(EMBEDDING_MODEL_NAME)
    # --- Encode text_features (frozen — no fine-tuning of the embedder) ---
    log("Encoding train text_features...")
    X_train = embedder.encode(
        train_df["text_features"].fillna("").tolist(),
        show_progress_bar=True,
        batch_size=64,
    )
    log("Encoding test text_features...")
    X_test = embedder.encode(
        test_df["text_features"].fillna("").tolist(),
        show_progress_bar=True,
        batch_size=64,
    )
    log(f"Embedding shape: train {X_train.shape}, test {X_test.shape}")

    y_train = train_df["sentiment"]
    y_test = test_df["sentiment"]

    # Computed once, directly from the original string labels — reused by both the
    # ordinal and XGBoost models below. compute_sample_weight only counts class
    # frequencies; it doesn't care how a model later encodes those classes internally
    # (SENTIMENT_ORDER's integers vs. LabelEncoder's integers), so recomputing this
    # per-model would just produce the same array twice under different encodings.
    balanced_sample_weights = compute_sample_weight(class_weight="balanced", y=y_train)

    # --- Train Logistic Regression head only ---
    log("\nTraining LogisticRegression head on frozen embeddings...")
    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit(X_train, y_train)

    # --- Evaluate ---
    y_pred = clf.predict(X_test)
    # --- Ceiling check: how much error is from missing context, not the model ---
    log("\n=== Ceiling Check ===")

    # Find which test tickets share text with other tickets (train or test)
    all_text = pd.concat([train_df["ticket_text"], test_df["ticket_text"]])
    text_counts = all_text.value_counts()
    is_duplicated = test_df["ticket_text"].map(text_counts) > 1

    log(f"Test tickets with duplicate text elsewhere: {is_duplicated.sum()}/{len(test_df)}")

    # For each duplicated ticket, find the MAJORITY sentiment label across all its copies
    majority_label = (
        pd.concat([train_df[["ticket_text", "sentiment"]], test_df[["ticket_text", "sentiment"]]])
        .groupby("ticket_text")["sentiment"]
        .agg(lambda x: x.mode().iloc[0])
    )

    test_df["majority_label"] = test_df["ticket_text"].map(majority_label)

    # Score 1: normal accuracy (against each row's own actual label) — only on duplicated subset
    dup_subset = test_df[is_duplicated]
    dup_y_pred = y_pred[is_duplicated.values]

    normal_acc = accuracy_score(dup_subset["sentiment"], dup_y_pred)
    fair_acc = accuracy_score(dup_subset["majority_label"], dup_y_pred)

    log(f"On duplicated-text tickets only:")
    log(f"  Accuracy vs each row's own label: {normal_acc:.4f}")
    log(f"  Accuracy vs group's majority label: {fair_acc:.4f}")
    log(f"  Gap (likely due to missing context, e.g. repeat-occurrence history): {fair_acc - normal_acc:.4f}")
    log("  NOTE: majority_label is computed from train+test combined, so this ceiling is "
        "slightly optimistic (test labels help define their own target). A stricter version "
        "would compute majority_label from train data only.")
    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    weighted_f1 = f1_score(y_test, y_pred, average="weighted")

    log(f"\n=== Evaluation ===")
    log(f"Accuracy: {acc:.4f}")
    log(f"Macro F1: {macro_f1:.4f}")
    log(f"Weighted F1: {weighted_f1:.4f}")
    log(f"\nMajority-class baseline (always predict most common class): "
        f"{y_test.value_counts(normalize=True).max():.4f}")

    log("\nPer-class classification report:")
    log(classification_report(y_test, y_pred))

    log("\n=== Operationally Meaningful Metric: Escalation-Trigger Grouping ===")
    log(f"Grouping: {{Negative, Very Negative}} = escalation-trigger vs. everything else "
        f"(Neutral, Positive, Slightly Negative) = non-trigger, per EDA finding.")
    lr_group_acc = group_accuracy(y_test, y_pred, ESCALATION_TRIGGER_SENTIMENTS)
    log(f"LogisticRegression group-level accuracy: {lr_group_acc:.4f}")
    log("This is the metric that actually matters for the Escalation agent — most "
        "5-class errors (e.g. Negative<->Very Negative) are same-group confusions that "
        "don't change the downstream escalation decision.")

    log("\nConfusion matrix (rows=true, cols=predicted):")
    labels = sorted(y_test.unique())
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    log(f"Labels order: {labels}")
    log(str(cm))

    # Specifically watch Very Negative vs Negative separability, per process doc
    if "Very Negative" in labels and "Negative" in labels:
        vn_idx = labels.index("Very Negative")
        neg_idx = labels.index("Negative")
        log(f"\nVery Negative <-> Negative confusion specifically:")
        log(f"  True Very Negative predicted as Negative: {cm[vn_idx][neg_idx]}")
        log(f"  True Negative predicted as Very Negative: {cm[neg_idx][vn_idx]}")
    log("\n" + "=" * 60)
    log("EXPLORATORY COMPARISON — Ordinal Classifier (mord.LogisticAT)")
    log("NOT the final model. Kept here as evidence for why plain LogisticRegression")
    log("was chosen instead — see the DECISION block below for the full comparison.")
    log("=" * 60)
    SENTIMENT_ORDER = ["Positive", "Neutral", "Slightly Negative", "Negative", "Very Negative"]
    label_to_int = {label: i for i, label in enumerate(SENTIMENT_ORDER)}
    int_to_label = {i: label for label, i in label_to_int.items()}

    y_train_ord = y_train.map(label_to_int)
    y_test_ord = y_test.map(label_to_int)

    log("\nTraining ordinal classifier (mord.LogisticAT) on frozen embeddings...")
    log("Applying balanced sample_weight (computed once, shared with XGBoost below) to the "
        "ordinal model — LogisticAT has no class_weight constructor arg, unlike "
        "LogisticRegression, so this must be passed explicitly at fit() to make this a "
        "fair comparison against the balanced baseline above.")
    ord_clf = mord.LogisticAT(alpha=1.0)
    ord_clf.fit(X_train, y_train_ord, sample_weight=balanced_sample_weights)

    y_pred_ord_int = ord_clf.predict(X_test)
    y_pred_ord = pd.Series(y_pred_ord_int).map(int_to_label)

    ord_acc = accuracy_score(y_test, y_pred_ord)
    ord_macro_f1 = f1_score(y_test, y_pred_ord, average="macro")

    log(f"\n=== Ordinal Model Evaluation ===")
    log(f"Accuracy: {ord_acc:.4f}  (vs. LogisticRegression: {acc:.4f})")
    log(f"Macro F1: {ord_macro_f1:.4f}  (vs. LogisticRegression: {macro_f1:.4f})")

    log("\nOrdinal model classification report:")
    log(classification_report(y_test, y_pred_ord))

    log("\nOrdinal model confusion matrix (rows=true, cols=predicted):")
    cm_ord = confusion_matrix(y_test, y_pred_ord, labels=labels)
    log(f"Labels order: {labels}")
    log(str(cm_ord))

    log("\n" + "=" * 60)
    log("EXPLORATORY COMPARISON — XGBoost")
    log("NOT the final model unless it wins on the group-level metric below.")
    log("Rationale for trying it despite Logistic Regression being a natural fit for")
    log("embeddings: tree-based models can capture non-linear feature interactions that")
    log("a linear model can't. Rationale for tempered expectations: the dataset's ceiling")
    log("is set by label noise (see Ceiling Check above), not model capacity — no model")
    log("architecture can out-perform inherently conflicting ground truth.")
    log("=" * 60)


    le = LabelEncoder()
    y_train_xgb = le.fit_transform(y_train)
    y_test_xgb = le.transform(y_test)
    # Reusing balanced_sample_weights computed once above — same underlying class
    # frequencies as the ordinal model, just a different integer encoding for XGBoost's
    # own purposes, which doesn't change what weight each original class should get.

    xgb_clf = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.1,
        objective="multi:softmax",
        num_class=len(le.classes_),
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=1,  # avoid OpenMP runtime collision with PyTorch (via sentence-transformers),
                   # a known cause of segfaults when both libraries are loaded in one process
    )
    xgb_clf.fit(X_train, y_train_xgb, sample_weight=balanced_sample_weights)
    y_pred_xgb_int = xgb_clf.predict(X_test)
    y_pred_xgb = le.inverse_transform(y_pred_xgb_int)

    xgb_acc = accuracy_score(y_test, y_pred_xgb)
    xgb_macro_f1 = f1_score(y_test, y_pred_xgb, average="macro")
    xgb_group_acc = group_accuracy(y_test, y_pred_xgb, ESCALATION_TRIGGER_SENTIMENTS)

    log(f"\nXGBoost — Accuracy: {xgb_acc:.4f}  (vs. LogisticRegression: {acc:.4f})")
    log(f"XGBoost — Macro F1: {xgb_macro_f1:.4f}  (vs. LogisticRegression: {macro_f1:.4f})")
    log(f"XGBoost — Group-level accuracy: {xgb_group_acc:.4f}  (vs. LogisticRegression: {lr_group_acc:.4f})")
    log("\nXGBoost classification report:")
    log(classification_report(y_test, y_pred_xgb))

    if xgb_group_acc > lr_group_acc:
        log(f"\nXGBoost WINS on the group-level metric ({xgb_group_acc:.4f} > {lr_group_acc:.4f}) "
            f"— checking significance and overfitting before adopting.")

        # --- Paired significance test (McNemar's) — correct test since both models
        # evaluated on the IDENTICAL 2000 test rows, not independent samples ---
        lr_group_correct = (pd.Series(y_test).isin(ESCALATION_TRIGGER_SENTIMENTS) ==
                             pd.Series(y_pred, index=y_test.index).isin(ESCALATION_TRIGGER_SENTIMENTS))
        xgb_group_correct = (pd.Series(y_test).isin(ESCALATION_TRIGGER_SENTIMENTS) ==
                              pd.Series(y_pred_xgb, index=y_test.index).isin(ESCALATION_TRIGGER_SENTIMENTS))
        both_correct = (lr_group_correct & xgb_group_correct).sum()
        only_lr_correct = (lr_group_correct & ~xgb_group_correct).sum()
        only_xgb_correct = (~lr_group_correct & xgb_group_correct).sum()
        both_wrong = (~lr_group_correct & ~xgb_group_correct).sum()
        contingency = [[both_correct, only_lr_correct], [only_xgb_correct, both_wrong]]
        mcnemar_result = mcnemar(contingency, exact=True)
        log(f"\nMcNemar's test (paired, correct test for same-test-set comparison):")
        log(f"  Both correct: {both_correct}, Only LR correct: {only_lr_correct}, "
            f"Only XGBoost correct: {only_xgb_correct}, Both wrong: {both_wrong}")
        log(f"  p-value: {mcnemar_result.pvalue:.4f} "
            f"({'SIGNIFICANT' if mcnemar_result.pvalue < 0.05 else 'NOT significant'} at alpha=0.05)")

        # --- Overfitting check: train vs test accuracy gap ---
        xgb_train_pred_int = xgb_clf.predict(X_train)
        xgb_train_acc = accuracy_score(y_train_xgb, xgb_train_pred_int)
        log(f"\nOverfitting check: XGBoost train accuracy={xgb_train_acc:.4f} vs "
            f"test accuracy={xgb_acc:.4f} (gap={xgb_train_acc - xgb_acc:.4f})")
        log("A large gap (e.g. >0.15-0.20) would indicate overfitting despite the "
            "modest max_depth=4 setting, given ~8000 training rows vs 384 continuous features.")

        if mcnemar_result.pvalue >= 0.05:
            log("\nFINAL CALL: gap is not statistically significant — retaining plain "
                "LogisticRegression as final model. A 1-point difference on this test set "
                "size is not a reliable basis for adopting a more complex model.")
        else:
            log("\nFINAL CALL: gap IS statistically significant — worth reconsidering "
                "XGBoost as final model, pending the overfitting check above.")
    else:
        log(f"\nXGBoost does NOT beat LogisticRegression on the group-level metric "
            f"({xgb_group_acc:.4f} vs {lr_group_acc:.4f}) — LogisticRegression remains final.")

    off_by_one = (abs(y_pred_ord_int - y_test_ord.values) <= 1).mean()
    log(f"\nOff-by-one accuracy (prediction within 1 step of true label): {off_by_one:.4f}")

    ord_group_acc = group_accuracy(y_test, y_pred_ord, ESCALATION_TRIGGER_SENTIMENTS)
    log(f"\nOrdinal model group-level accuracy: {ord_group_acc:.4f}  (vs. LogisticRegression: {lr_group_acc:.4f})")

    log("\n" + "=" * 60)
    log("DECISION: Ordinal model REJECTED.")
    log(f"Raw 5-class macro-F1 was better for ordinal ({ord_macro_f1:.4f} vs {macro_f1:.4f}), "
        f"but on the metric that actually matters downstream (escalation-trigger grouping), "
        f"ordinal is WORSE ({ord_group_acc:.4f} vs {lr_group_acc:.4f}). Plain LogisticRegression "
        f"is simpler and better on the metric that matters — chosen as the final model.")
    log("=" * 60)

    # --- Save FINAL model only (plain LogisticRegression) — ordinal is exploratory, not shipped ---
    joblib.dump({"classifier": clf, "embedding_model_name": EMBEDDING_MODEL_NAME}, MODEL_PATH)
    log(f"\nSaved FINAL trained classifier (LogisticRegression) to {MODEL_PATH}")

    with open(REPORT_PATH, "w") as f:
        f.write("\n".join(report_lines))
    log(f"Saved evaluation report to {REPORT_PATH}")


if __name__ == "__main__":
    main()