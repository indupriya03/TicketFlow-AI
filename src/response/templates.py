"""Customer-facing fallback/template strings — text shown to the customer
that bypasses the LLM, so it never gets checked by whatever validates LLM
output. These must independently satisfy response_agent.py's SYSTEM_PROMPT
rule: never claim anything was done, started, flagged, passed on, approved,
or arranged for the customer, and never promise anyone will review,
investigate, prioritize, follow up, or get back to them — state the
system's own limitation instead, never a claim about backend action.
Functions taking `sentiment` prepend a short apology when it's in
UPSET_SENTIMENTS, per the system prompt's "if the customer sounds upset,
start with a short, sincere apology" rule. UPSET_SENTIMENTS reuses the same
{Negative, Very Negative} escalation-trigger grouping established in the
Sentiment sub-agent's evaluation, for consistency across the project.
greeting_text() takes no sentiment: it's shown before any ticket exists
(orchestrator.py's trivial-message path), so it isn't a response_agent
output and isn't bound by that prompt, and there's nothing to apologize
for yet."""

UPSET_SENTIMENTS = {"Negative", "Very Negative"}
APOLOGY_PREFIX = "I'm sorry for the trouble this has caused. "


def _with_apology(text, sentiment):
    return APOLOGY_PREFIX + text if sentiment in UPSET_SENTIMENTS else text


def ack_text(sentiment=None):
    return _with_apology(
        "This specific situation isn't something I can fully resolve with the "
        "information available to me right now.",
        sentiment,
    )


def greeting_text():
    return (
        "Hi! I'm the support assistant. Could you tell me a bit more about what you "
        "need help with — an order, a return, a technical issue?"
    )


def escalated_text(sentiment=None):
    return _with_apology(
        "This needs more detailed handling than I'm able to provide automatically.",
        sentiment,
    )


def high_priority_text(sentiment=None):
    return _with_apology(
        "This appears to need urgent, hands-on attention that I'm not able to "
        "provide directly.",
        sentiment,
    )