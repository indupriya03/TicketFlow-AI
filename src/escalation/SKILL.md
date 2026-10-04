---
name: escalation
description: Fifth agent in Ticket-Flow-AI. Decides whether a Response-drafted reply auto-sends or must go to a human on the assigned team, based on response_needs_review, sentiment, complexity, refund amount, and prior-contact-unresolved — and files a Jira issue when escalating.
---

# Escalation Agent

## Role
The final gate before a reply leaves the system. Reads signals Response,
Intake, and Triage already computed, decides auto-resolve vs. hand-off,
and — when escalating — files a real Jira issue so a human on the right
team actually sees it. Does not re-derive which team; `state["team"]`
already came from Triage's `assign_team`.

START -> decide -> END


Runs after Response for the full pipeline, but the SAME Jira dispatch
(`create_issue()`) is also invoked from two earlier exit points in
`orchestrator.py` — Intake's `SKIP_TRIAGE_STATUSES` branch and Triage's
own High-priority `ESCALATION_TEAM` routing — so that a ticket which
never reaches Response still gets a Jira issue filed. 

This agent's own "escalate_to_team" decision is distinct from
`assign_team.ESCALATION_TEAM`, despite the shared word. That earlier
check routes a High-priority ticket to a human team BEFORE
Retrieval/Response run, purely from priority — and when it fires, THIS
agent's own trigger checks never run at all (Retrieval/Response are
skipped), but Jira dispatch still happens for that path.

## Inputs (state fields read)
`ticket_id`, `ticket_text_clean`, `response_text`, `response_needs_review`,
`sentiment`, `complexity`, `team`, `intent`, `entity_amounts`,
`entity_prior_contact_unresolved`.

## Outputs (state fields written)
| Field | Meaning |
|---|---|
| `escalation_decision` | `auto_resolve` or `escalate_to_team` |
| `escalation_reasons` | List of which condition(s) fired; empty when auto-resolved |
| `jira_issue_key` | The filed issue's key (e.g. `KAN-29`), or `None` if auto-resolved, Jira unconfigured, or the Jira call failed |

## Triggers — escalates on ANY of
- `response_needs_review` is `True` (Response's own guardrails/weak-evidence flag)
- `sentiment` in `ESCALATION_TRIGGER_SENTIMENTS` (`"Negative"`, `"Very Negative"`
  — imported from `intake_sentiment.py`, never redefined here, so the two
  sentiment-escalation checks in the project never drift out of sync)
- `complexity == "High"` (`assign_complexity`'s own top bucket, read directly)
- **Refund amount rule:** `intent` is `request_refund` or
  `dispute_charge_or_refund`, AND either no amount was stated
  (`refund_amount_unstated`) or the highest stated amount exceeds
  `REFUND_AUTO_APPROVE_LIMIT` (currently $50). Uses the highest of
  multiple stated amounts, the safer of the two options.
- `entity_prior_contact_unresolved` is `True` (the customer says they were
  already contacted and got no reply or resolution; a human should see the
  draft before it's sent)

A single trigger is enough — this mirrors `assign_team`'s own "High
priority always overrides" rule, applied one stage later, rather than a
weighted score.

## Jira dispatch (`jira_tools.py`)
Called only when at least one trigger fires. Uses Agno's `JiraTools`,
called as a plain function — no LLM agent loop, since the decision is
already fully deterministic and routing it through an LLM would only add
cost, latency, and a new failure mode for no benefit. Team routes into
Jira's native Team field (`customfield_10001`) via `TEAM_TO_JIRA_TEAM`'s
real team UUIDs, so issues are filterable by team in the Jira UI, not
just described in text.

If `JIRA_URL`/`JIRA_EMAIL`/`JIRA_API_TOKEN`/`JIRA_PROJECT_KEY` are
missing, or the API call raises for any reason, `create_issue()` returns
`None` and logs a warning — Escalation still records `escalate_to_team`
correctly, just without a Jira key. A missing or misconfigured Jira
integration never blocks a ticket's escalation decision from being made.

## Guardrails applied
- **Decision logic fails safe toward escalation, not auto-resolve.**
  `decide_node`'s trigger-checking is wrapped in try/except. If anything
  raises (malformed `entity_amounts`, unexpected types), the error is
  logged and the ticket is escalated anyway
  (`escalation_reasons=["escalation_logic_error"]`) rather than silently
  defaulting to auto-resolve — the one failure direction that matters,
  since this agent's whole purpose is catching risky tickets before they
  auto-send.
- **Jira failures never block the decision.** `create_issue()`'s own
  try/except (in `jira_tools.py`) ensures a Jira outage or misconfiguration
  degrades to `jira_issue_key=None`, never an unhandled exception.
- **Reuses existing labels rather than inventing new ones.** Sentiment
  and complexity checks read directly from constants/labels other agents
  already own, so there's exactly one source of truth for each.

## Known limitations
- No retry loop: unlike the project spec's diagram, there is no
  Response-regenerate step on failure. Every escalation is a one-shot
  Jira hand-off.
- The refund limit ($50) and the other triggers are independent checks;
  no weighting or severity ranking exists between them — any one is
  sufficient, and `escalation_reasons` lists all that fired.
- No automatic notification back to the customer when escalated — a
  Streamlit-side holding message is a separate, UI-layer concern.
- `escalation_reasons` and `jira_issue_key` are logged to SQLite (via
  Learning) but not yet surfaced in any dashboard beyond raw queries.
