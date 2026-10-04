COMPLEX_INTENTS = {
    "dispute_return_rejection",
    "dispute_charge_or_refund",
    "escalate_unresponsive_seller",
    "report_item_not_as_described",
}

def assign_complexity(intent, follow_up_rounds, entity_days_elapsed, entity_order_ids, entity_amounts):
    score = 0
    if intent in COMPLEX_INTENTS:
        score += 1
    if follow_up_rounds and follow_up_rounds > 0:
        score += 1
    if entity_days_elapsed and max(entity_days_elapsed) > 7:
        score += 1
    if len(entity_order_ids or []) > 1 or len(entity_amounts or []) > 1:
        score += 1
    return "Low" if score == 0 else "Medium" if score == 1 else "High"