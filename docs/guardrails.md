# Guardrails — Ticket-Flow-AI

## Principle
All customer-controlled text (`ticket_text`) is untrusted data. Guardrails run
BEFORE any external LLM call and are shared via `src/guardrails.py`.

## Guardrail inventory

| # | Guardrail | Where applied | Implementation | Known limits |
|---|-----------|---------------|----------------|--------------|
| 1 | PII scrub | Phase 3 preprocessing; Intent uses `ticket_text_clean`, never raw text | Regex for email/phone/card (`preprocessing.py`) | Regex only; names/addresses not caught. Scoped MVP, not a guarantee |
| 2 | Prompt-injection pre-filter | Start of `classify_intent()`; reuse in Response agent | Word-boundary pattern match; hit returns intent `other` with no LLM call | Misses rephrasings, typos, non-English. First layer only |
| 3 | Data-not-instructions prompting | Intent system prompt | Ticket passed as user content only | Prompt-level defense is not reliable alone (hence #2) |
| 4 | Closed-label enforcement | Intent output schema | Enum + `with_structured_output` | Guarantees valid label, not a correct one |
| 5 | Retry / failure handling | `classify_intent()` | 2 retries, then `complete=False` | Does not handle daily quota exhaustion (see below) |
| 6 | Completeness gate | After Intent + Entities | Rule-based; missing order_id triggers customer follow-up | Regex must match the customer's ID format |

## Model tiers
- `gpt-oss-20b`: cheap iteration and debugging only.
- `gpt-oss-120b`: the only model whose output counts for validation and reporting.

## Rate limits
Per-minute limits: retry. Daily quota exhausted: retries won't help, so
fail fast and mark the ticket `complete=False` for re-run later.

## Limitations
Guardrails 1 and 2 are scoped MVP pre-filters, not comprehensive defenses.