"""
Phase 5 — Triage: Priority Sub-Agent (Rule + Model Hybrid)
Multi-Agent Customer Support Intelligence Platform

DESIGN RATIONALE (verified against support_tickets_clean.csv + entities):
In the training data, sentiment determined priority deterministically for 4 of 5
sentiment values, with zero exceptions (a property of this dataset, not assumed
to hold for other customer support data):
    Positive                    -> Low     (100%)
    Slightly Negative           -> Medium  (100%)
    Negative / Very Negative    -> High    (100%)
Only Neutral sentiment is genuinely ambiguous (~50/50 Low vs Medium in the
data), and nothing checked (ticket_category, entity_days_elapsed, escalated)
breaks that tie. So: a rule handles the 4 deterministic cases, and a small
binary classifier (see priority_predict.py), trained ONLY on the Neutral
subset, handles the one case that actually needs it.

Do not extend this rule table without re-verifying determinism first — it
is only correct because we checked it against the real label distribution,
not because it's an intuitive-sounding mapping.
"""

from src.triage.priority_predict import predict_priority_for_neutral

SENTIMENT_PRIORITY_RULE = {
    "Positive": "Low",
    "Slightly Negative": "Medium",
    "Negative": "High",
    "Very Negative": "High",
}


def assign_priority(sentiment, clean_text):
    """sentiment: state["sentiment"] from Intake.
    clean_text: state["ticket_text_clean"], only actually used when
    sentiment == "Neutral" (passed through to the trained classifier)."""
    if sentiment in SENTIMENT_PRIORITY_RULE:
        return SENTIMENT_PRIORITY_RULE[sentiment]
    if sentiment == "Neutral":
        return predict_priority_for_neutral(clean_text)
    raise ValueError(f"Unrecognized sentiment value: {sentiment!r}")