from src.escalation.escalation_agent import _check_refund_amount
from src.retrieval.retrieval_confidence import classify_confidence
from src.triage.priority_assign import SENTIMENT_PRIORITY_RULE


def test_priority_rule_maps_four_sentiments():
    assert SENTIMENT_PRIORITY_RULE == {
        "Positive": "Low",
        "Slightly Negative": "Medium",
        "Negative": "High",
        "Very Negative": "High",
    }


def test_refund_under_limit_is_not_flagged():
    assert _check_refund_amount("request_refund", [25]) is None


def test_refund_over_limit_uses_highest_amount():
    assert _check_refund_amount("request_refund", [10, 75, 20]) == "refund_amount=75>50"


def test_refund_without_amount_is_flagged():
    assert _check_refund_amount("request_refund", []) == "refund_amount_unstated"


def test_non_refund_intent_ignores_amount():
    assert _check_refund_amount("report_defect", [500]) is None


def test_confidence_labels():
    assert classify_confidence(0.20, "faq") == "high"
    assert classify_confidence(0.45, "faq") == "low"
    assert classify_confidence(None, "faq") == "low"
