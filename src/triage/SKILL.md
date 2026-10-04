---
name: triage
description: Second agent in Ticket-Flow-AI. Classifies category, assigns priority, and routes to a team. Runs once per ticket, only after Intake completes successfully.
---

# Triage Agent

## Role
Turn Intake's structured signals into a routing decision. Category and Priority
run in parallel — neither reads the other's output, both only need what Intake
already produced. Team runs last, since it needs Priority's result (for the
High-priority override) and Intent.

```
START -> {category_node, priority_node}  (parallel)
      -> team_node                        (fan-in, needs priority)
      -> END
```

This agent must never be invoked on a ticket with `status in
("intent_failed", "flagged_injection")` from Intake — there is no valid
intent to classify against. `triage_agent.py`'s demo enforces this at the
orchestration boundary; a real orchestrator must make the same check before
calling this graph.

## Inputs (state fields read)
`ticket_text_clean`, `sentiment`, `intent` — all produced once by Intake, never recomputed here.

## Outputs (state fields written)
| Field | Meaning |
|---|---|
| `category` | One of 7 coarse topic buckets (e.g. Delivery Issue, Refund & Return) |
| `priority` | Low / Medium / High |
| `team` | The support team this ticket routes to |

## Sub-agents and methods
| Sub-agent | Method | File |
|---|---|---|
| Category | Frozen `all-MiniLM-L6-v2` + LogisticRegression head, trained on `ticket_text_clean` only | `category_train.py` / `category_predict.py` |
| Priority | Hybrid rule + model: In the training data, Sentiment mapped deterministically to Priority for 4 of 5 values (100%), so those four use a hard-coded rule; only `Neutral` (~50/50 Low/Medium) is routed to a trained binary classifier | `priority_assign.py` / `priority_train.py` / `priority_predict.py` |
| Team | Rule-based lookup from Intent (not Category — Category's 7 buckets can't distinguish tickets needing different teams), with a High-priority override to Escalation | `assign_team.py` |
| Orchestration | LangGraph node, parallel fan-out + fan-in | `triage_agent.py` |

## Priority sub-agent detail
No ground-truth ambiguity exists for 3 of the 4 sentiment values — a plain
rule handles them (`SENTIMENT_PRIORITY_RULE`). Only the Neutral subset needs
a model, so `priority_train.py` trains on that subset only.

LogisticRegression and XGBoost were both trained and 5-fold cross-validated
on the identical split; XGBoost was chosen as the production model. Class
imbalance (Medium=1502, Low=716 in the Neutral train subset) is handled via
`compute_sample_weight(class_weight="balanced")`, not `scale_pos_weight` —
`scale_pos_weight` only rescales whichever class encodes to 1, which under
naive `LabelEncoder` ordering was "Medium" (the majority class), so it
doubled the correction in the wrong direction rather than helping the
minority "Low" class. Only the winning model is saved to disk; the loser is
an in-memory comparison logged to the eval report.

The saved joblib stores a `label_map` dict (`{"Low": 1, "Medium": 0}`), not a
`LabelEncoder` — `priority_predict.py` inverts this dict itself to decode
XGBoost's integer output back to `"Low"`/`"Medium"` strings, since every
downstream consumer (`assign_team.py`, `assign_priority.py`) compares
against those literal strings.

## Team sub-agent detail
`INTENT_TEAM_MAP` is a documented domain design decision, not a
metric-validated mapping — no ground-truth "team" column exists in the
dataset. `report_account_or_login_issue` and `report_technical_issue` are
merged into one Technical Support team (an e-commerce-context call: login
issues here are mostly low-stakes access problems, not identity fraud).
`assign_team()` raises `ValueError` on an intent not in the map, with no
fallback path in `team_node` — an unmapped intent will crash the graph
invocation rather than degrade gracefully.

## Guardrails applied
None — this agent makes zero LLM calls. Category and Priority are local
embedding + classifier inference; Team is a plain dict lookup. The
PII-scrub and prompt-injection guardrails in `docs/guardrails.md` are scoped
entirely to Intake's Intent sub-agent, the only place in the pipeline that
sends customer-controlled text to an external LLM. A `flagged_injection`
ticket never reaches this agent at all — Intake short-circuits to human
review before Triage is invoked.

## Evaluation
- **Category:** macro F1 reported in `reports/category_eval_report.txt`;
  `class_weight="balanced"` protects the smallest class (App & Website
  Issue) from the largest (Refund & Return).
- **Priority (Neutral subset):** LogisticRegression macro F1 0.8195 (test) vs
  XGBoost 0.8536–0.8663 (test, across notebook iterations) with 5-fold CV
  mean 0.8874 (std 0.0117) confirming stability. XGBoost's error profile
  favors under-triage safety: it mislabels far fewer true-Medium tickets as
  Low than LogisticRegression does, at the cost of over-flagging some
  true-Low tickets as Medium (the cheaper mistake for a support system).
- **Team:** not independently evaluated — no ground-truth labels exist;
  correctness depends entirely on `INTENT_TEAM_MAP` being reviewed against
  real support-team names before production use.

## Known limitations
- `assign_team`'s `ValueError` on an unrecognized intent has no fallback or
  human-review path — decide deliberately if that's the desired failure mode
  before Intent's label set is ever extended.
- Team mapping is a design decision, not a metric-backed choice; re-verify
  before production.
- Priority's Neutral-subset classifier depends on the sentiment→priority
  rule's determinism holding on new data — `priority_train.py` re-checks this
  assumption at train time and raises if it doesn't hold, but it is not
  re-verified at inference time.
- This mapping is a property of the training dataset and is not assumed to hold for other customer support data.