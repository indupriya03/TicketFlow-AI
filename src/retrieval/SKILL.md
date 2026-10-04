---
name: retrieval
description: Third agent in Ticket-Flow-AI. Finds the FAQ entries and deduplicated historical resolution templates most relevant to a ticket, grades how confident the match is, and retries once with a rewritten query if it is not. Runs once per ticket, after Triage.
---

# Retrieval Agent

## Role
Give the Response agent grounded evidence instead of letting it answer from
nothing. It searches two sources with one query embedding, grades the best
match, and returns the documents plus a confidence label that Escalation and
Response can use.

```
START -> strategy -> retrieve -> (high)                    -> END
                              -> (low, attempts < MAX)     -> strategy  (loop)
                              -> (low, attempts >= MAX)    -> END  (best effort, flagged low)
```

This is its own graph, not a node in Triage. The orchestrator calls it after
Triage. It must never run on `status in ("intent_failed", "flagged_injection",
"escalate_to_human")`; the orchestrator's skip check covers this because
Retrieval sits behind Triage.

## Inputs (state fields read)
`ticket_text_clean`, `intent`. `category` is accepted but not used: the
category pre-filter was removed (see Design decisions).

## Outputs (state fields written)
| Field | Meaning |
|---|---|
| `retrieved_docs` | Up to 5 documents, each `{id, text, metadata, distance}` |
| `retrieval_confidence` | `high` or `low`, from the top hit's distance vs its source's threshold |
| `retrieval_top_source` | `faq` or `resolution_template`: which source the top hit came from |
| `retrieval_attempts` | Searches performed (1 or 2) |
| `retrieval_strategy` | `initial` or `rewritten`: the plan used on the last attempt |

## Sub-agents and methods
| Component | Method | File |
|---|---|---|
| Document preprocessing | Builds FAQ + resolution-template documents; dedups templates with the product name masked; no API calls | `preprocessing_rag.py` |
| Indexing | Gemini embeddings (`RETRIEVAL_DOCUMENT`) into a persistent Chroma collection `support_kb`; upserts by deterministic id | `retrieval_document.py` |
| Query | Gemini embeddings (`RETRIEVAL_QUERY`), cosine search filtered by `source_type` | `retrieval_query.py` |
| Embedding client | `google.genai` directly (batching, retry, fail-fast on daily quota) | `gemini_embeddings.py` |
| Confidence | Distance thresholds per source: FAQ 0.33, resolution_template 0.35 | `retrieval_confidence.py` |
| Orchestration | LangGraph: strategy node + retrieve node, bounded retry | `retrieval_agent.py` |
| Evaluation | Gold queries, off-topic negatives, full-agent scoring | `retrieval_eval.py` |

Corpus: 150 FAQ entries + 62 deduplicated resolution templates = 212 documents.
Both sources are short (FAQ question 4-12 words, answer up to 19, template up
to 12), so nothing is chunked.

The initial corpus does not contain raw ticket-text/resolution pairs. Human-reviewed
responses enter the index later through `index_from_feedback.py`.

## Design decisions
- **No category or intent pre-filtering.** At 212 documents Chroma scans everything in
  milliseconds, so narrowing first saves nothing, and a wrong category
  filter would exclude the right document before the search runs. Revisit if the
  corpus grows into the thousands.
- **One embedding, both sources.** Each attempt embeds the query once and
  searches FAQ and templates with it.
- **FAQ first when confident.** Its correct-vs-wrong distance margin is cleaner than the
  template channel's. The returned list always includes at least 2 templates
  (`MIN_TEMPLATE_DOCS`) so Response sees outcome-style examples.
- **Retry (max 2 attempts).** Attempt 2 searches with `"<intent phrase>: <ticket text>"`. An
  earlier rewrite built from intent and category alone dropped the ticket
  content and turned off-topic queries into false "high" results; the ticket text
  must stay in the query.

## Guardrails applied
The only external call is embedding `ticket_text_clean` (already PII-scrubbed in Intake) with Gemini. An embedding model does not follow instructions, so prompt injection is not a risk at query time, and flagged tickets never reach this agent. The base corpus comes from a curated FAQ and template set, not from customer text. A second, manual path exists: `index_from_feedback.py` adds human-written replies (`rejected_edited` by default) to the index as `resolution_template` documents with `origin="human_reviewed"`. Those replies can still carry customer-influenced text, so the script scrubs PII, drops sentences in the agent's voice or containing questions, amounts, or apologies, and runs as a dry run by default. A person reads every document before `--apply`. Regex cannot catch everything (for example a customer's name
typed into a reply), so that manual review is the main safeguard.

## Evaluation
- **Independence:** the evaluation queries come from the test split and the template corpus is built from the train split only. `resolution_text` is heavily templated, so template Recall is easier than it would be on varied resolutions. The thresholds are still a snapshot of one small sample (20 template queries, 5 FAQ queries).
- **Off-topic negatives:** 0 of 10 returned high; top distances 0.38-0.44.
- **FAQ gold set (5 hand-picked queries):** 5/5 at rank 1, correct hit distances 0.20-0.26.
- **Resolution-template gold set (20 test tickets):** template-only Recall@3 0.15,
  Recall@10 0.50; through the agent the correct template is in the returned
  docs for 3/20. Confidence was high on 18/20, mostly because the FAQ channel
  found something under its threshold, not because the right template came back.
- **Retry:** rescued 0 queries, but the test could not show a rescue (only
  2 template queries were low on attempt 1, and they ran without intents).
  Benefit is unmeasured; off-topic queries correctly stay low after the retry.

## Known limitations
- The template channel is weak. Correct and wrong distances overlap, there
  are only 62 generic templates, and a template "high" is weaker evidence than a
  FAQ "high". Use `retrieval_top_source` before auto-resolving on it.
- Deduplication masks product names, so a returned template may name an
  unrelated product. Response should rewrite it, not quote it to a customer.
- The FAQ gold set is 5 queries. Expand it before trusting the FAQ numbers
  as a benchmark.
- Thresholds are a snapshot of one evaluation run. Re-derive them if the
  corpus or sample changes. Rewritten (attempt 2) queries may land at different
  distances than the ones the thresholds were set from; this has not been checked.
- `retrieve_node` has no error handling. An embedding failure (for example
  Gemini's free-tier daily quota) raises and fails the graph invocation.
- Free-tier embedding quota is shared with indexing and evaluation runs.
- `retrieval_document.py --reset` rebuilds the index from `rag_documents.json` only and drops the human-reviewed documents. Re-run `index_from_feedback.py --apply` afterwards. The confidence thresholds were set before these documents existed and have not been re-checked against them.