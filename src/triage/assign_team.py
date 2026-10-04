"""
Phase 5 — Triage: Team Sub-Agent (Rule-Based Lookup)
Multi-Agent Customer Support Intelligence Platform

Team is assigned from INTENT, not Category — Category's 7 coarse buckets
couldn't distinguish tickets needing different teams within one bucket
(e.g. "Payment Issue" could be a billing dispute or a fraud report).
Intent's 16 action-oriented labels map far more precisely to a real team.

report_account_or_login_issue and report_technical_issue are merged into one
Technical Support team: in this e-commerce context (not banking), login
issues are mostly password/access problems, not high-stakes identity fraud —
the underlying skill needed (debug why the system isn't working for this
user) is the same as general technical issues, so a separate Account
Security team isn't justified here. Any genuinely serious case is still
caught by the priority override below, not by a dedicated team.

No ground-truth "team" column exists in the dataset — this mapping is a
domain design decision, not something verified against real labels the way
Category/Priority were. Re-check with real support-team names before this
goes into production.
"""

INTENT_TEAM_MAP = {
    "request_refund": "Returns & Refunds",
    "initiate_return": "Returns & Refunds",
    "dispute_return_rejection": "Returns & Refunds",
    "request_exchange": "Returns & Refunds",
    "dispute_charge_or_refund": "Billing",
    "report_delivery_issue": "Logistics",
    "check_return_or_order_status": "Logistics",
    "request_cancellation": "Logistics",
    "request_order_modification": "Logistics",
    "report_defect": "Product Quality",
    "report_item_not_as_described": "Marketplace Ops",
    "escalate_unresponsive_seller": "Marketplace Ops",
    "report_account_or_login_issue": "Technical Support",
    "report_technical_issue": "Technical Support",
    "request_information": "General Support",
    "other": "General Support",
}

ESCALATION_TEAM = "Escalation"


def assign_team(intent, priority):
    """intent: state["intent"] from Intake. priority: state["priority"] from
    the Priority sub-agent (must run before this — see triage_agent.py's
    node ordering). A High-priority ticket always routes to Escalation,
    regardless of what intent would otherwise map to."""
    if priority == "High":
        return ESCALATION_TEAM
    if intent not in INTENT_TEAM_MAP:
        raise ValueError(f"Unrecognized intent value: {intent!r}")
    return INTENT_TEAM_MAP[intent]