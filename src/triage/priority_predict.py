"""
Phase 5 — Triage: Priority Sub-Agent (Inference — Neutral subset only)
Multi-Agent Customer Support Intelligence Platform

Called ONLY when sentiment == "Neutral" — see assign_priority.py, which
handles the other 4 sentiment values via a rule learned from the training data and never
calls this function for them.
"""

from pathlib import Path

import joblib

from src.embedding import get_embedder

ROOT_DIR = Path(__file__).parent.parent.parent
MODEL_PATH = ROOT_DIR / "models/priority_neutral_classifier.joblib"

_embedder = None
_priority_clf = None
_priority_inv_label_map = None


def load_priority_model():
    """Loads model, embedder, and the inverse of priority_train.py's label_map
    (e.g. {"Low": 1, "Medium": 0} -> {1: "Low", 0: "Medium"}), so XGBoost's
    integer predictions can be decoded back to the strings the rest of the
    pipeline expects (assign_team.py, assign_priority.py compare against
    "Low"/"Medium" directly)."""
    global _embedder, _priority_clf, _priority_inv_label_map
    if _priority_clf is None:
        loaded = joblib.load(MODEL_PATH)
        _priority_clf = loaded["classifier"]
        _embedder = get_embedder(loaded["embedder_name"])
        label_map = loaded["label_map"]
        _priority_inv_label_map = {v: k for k, v in label_map.items()}
    return _embedder, _priority_clf, _priority_inv_label_map


def predict_priority_for_neutral(clean_text):
    """clean_text should be state["ticket_text_clean"] from Intake."""
    embedder, clf, inv_label_map = load_priority_model()
    embedding = embedder.encode([clean_text])
    pred_encoded = int(clf.predict(embedding)[0])
    return str(inv_label_map[pred_encoded])