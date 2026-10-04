---
name: learning
description: Sixth and final agent in Ticket-Flow-AI. Logs every ticket's full trace to SQLite (data/logs/tickets.db) — the storage/feedback foundation, not yet an active learning loop.
---

# Learning Agent (Logging, Phase 1)

## Role
Persist the outcome of every ticket, regardless of which path it took
through the pipeline, to a real SQLite table. This is the storage layer
the project spec's "Database Layer" section calls for, and the
prerequisite for the spec's Agent Logs UI panel, resolution-time /
automation-rate / escalation-rate metrics, and any future feedback loop.