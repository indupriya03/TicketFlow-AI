"""
Phase 5 — Classification Agent: Category Sub-Agent (Inference)
Multi-Agent Customer Support Intelligence Platform

Frozen all-MiniLM-L6-v2 embedding + trained LogisticRegression head, loaded
once and reused across calls. Category was trained on ticket_text_clean
ONLY (not the full text_features used by Sentiment) — see category_train.py.

Input to predict_category(): the same ticket_text_clean already computed
once by Intake (state["ticket_text_clean"]) — do not re-clean it here.

Expects models/category_classifier.joblib to contain a dict with keys
"classifier", "embedder_name", "labels"
"""

from pathlib import Path
import joblib
from src.embedding import get_embedder

ROOT_DIR = Path(__file__).parent.parent.parent
MODEL_PATH = ROOT_DIR / "models/category_classifier.joblib"

_embedder = None
_category_clf = None


def load_category_model():
    global _embedder, _category_clf
    if _category_clf is None:
        loaded = joblib.load(MODEL_PATH)
        _category_clf = loaded["classifier"]
        # Reuse the SAME embedder instance across sub-agents if one is already
        # loaded elsewhere in the process (e.g. Sentiment's), rather than
        # loading all-MiniLM-L6-v2 into memory twice — swap this for a shared
        # singleton once Category and Sentiment run in the same process.
        _embedder = get_embedder(loaded["embedder_name"])    
    return _embedder, _category_clf

def predict_category(clean_text):
    """Frozen embedding + trained LogisticRegression head. clean_text should
    be state["ticket_text_clean"] from Intake — not raw ticket_text, and not
    the text_features string (Category doesn't use product_name/segment).

    Returns (category, confidence) — confidence is predict_proba's max
    probability across classes, i.e. how sure the model is about
    WHICHEVER category it picked, not the probability of any fixed class.
    Feeds Escalation's "classification confidence" input and Triage's
    Complexity score."""
    embedder, clf = load_category_model()
    embedding = embedder.encode([clean_text])
    probabilities = clf.predict_proba(embedding)[0]
    predicted_idx = probabilities.argmax()
    category = str(clf.classes_[predicted_idx])
    confidence = float(probabilities[predicted_idx])
    return category, confidence