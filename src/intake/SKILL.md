---
name: intake
description: First agent in Ticket-Flow-AI. Extracts entities, classifies intent and sentiment, checks completeness, and asks the customer for missing details. Runs once per ticket before Category and Priority.
---

# Intake Agent

## Role
Turn a raw customer ticket into structured signals that every later agent uses.
All four outputs are computed once, together, here. Category and Priority read
them and never recompute them.

## Inputs (state fields read)
`ticket_id`, `ticket_text`, `product_name`, `product_segment`, `customer_replies`

## Outputs (state fields written)
| Field | Meaning |
|---|---|
| `ticket_text_clean` | Cleaned, PII-scrubbed text (the only text sent to an LLM) |
| `intent` | One of 16 action-oriented labels (closed set) |
| `intent_reasoning` | One-sentence justification from the LLM |
| `sentiment` | Very Negative / Negative / Slightly Negative / Neutral / Positive |
| `entity_order_ids`, `entity_amounts`, `entity_product_mentioned` | Regex-extracted entities |
| `is_complete`, `missing_fields` | Whether required details are present |
| `status` | complete / awaiting_customer / escalate_to_human / intent_failed |

## Sub-agents and methods
| Sub-agent | Method | File |
|---|---|---|
| Entities | Regex, no model | `intake_entities.py` |
| Intent | Groq LLM, zero-shot, structured Pydantic output | `intake_intent.py` |
| Sentiment | Frozen `all-MiniLM-L6-v2` + LogisticRegression head | `intake_sentiment.py` |
| Completeness | Rule-based: intents in `NEEDS_ORDER_ID` require an order ID | `intake_completeness.py` |
| Orchestration | LangGraph node + follow-up loop | `intake_agent.py` |

## Follow-up loop
If a required field is missing, the graph pauses (`interrupt`), asks the customer
with a fixed template, and resumes on their reply. Entities and completeness are
re-run on original text plus replies. Intent and sentiment are not re-run.
Maximum 2 follow-ups, then `status = escalate_to_human`.

## Guardrails applied
See `docs/guardrails.md`. Summary: PII scrub before LLM calls, prompt-injection
pre-filter, closed intent label set, retry handling on LLM failure.

## Evaluation
- **Entities:** manual spot-check of ~50 tickets.
- **Intent:** manual review of a 200-ticket sample (no ground-truth labels exist).
- **Sentiment:** 5-class accuracy 0.439 (majority baseline 0.2985), macro F1 0.438.
  Group accuracy for the escalation-trigger grouping (Negative + Very Negative vs rest): 0.9055.
  Ordinal model and XGBoost tried and rejected (see `sentiment_eval_report.txt`).
- **Completeness:** rule-based; validated on the 200-ticket sample.

## Known limitations
- Sentiment labels are noisy (identical texts carry different labels),  
  so 5-class accuracy has a low ceiling. Use the group-level metric for downstream decisions.
- Very Negative vs Negative is the weakest boundary; it rarely changes the escalation decision.
- Regex PII scrub and pattern-based injection filter are scoped MVP defenses, not guarantees.
- Order-ID regex assumes the `ORD<digits>` format.