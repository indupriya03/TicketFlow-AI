"""
Phase 8 — Escalation Agent: Jira Dispatch
Multi-Agent Customer Support Intelligence Platform

Wraps Agno's JiraTools to file a real issue when Escalation decides a
ticket needs a human. Called as a PLAIN FUNCTION from escalation_agent.py's
decide_node — no LLM agent loop involved. Escalation's decision is already
fully deterministic (rule-based triggers), so routing "should I call
create_issue" through an LLM would only add latency, cost, and a new
failure mode for zero benefit.

Requires JIRA_URL, JIRA_EMAIL, JIRA_API_TOKEN, JIRA_PROJECT_KEY env vars
(get a token at id.atlassian.com/manage-profile/security/api-tokens).
If any are missing, or the Agno tool call raises, create_issue() returns
None rather than raising — Escalation still records "escalate_to_team",
just without a Jira key, so a missing/misconfigured integration never
blocks the pipeline.

Team -> Jira Team field mapping lives in TEAM_TO_JIRA_TEAM below, using
real team UUIDs from the Jira project (not a placeholder). Each team maps
to a UUID written into the issue's customfield_10001 (Jira's native Team
field) in create_issue(), so filed issues are filterable by team in the
Jira UI directly, not just described in the issue text. An unrecognized
team falls back to "General Support"'s UUID rather than failing the call.
"""

import logging
import os
import json

from agno import team

log = logging.getLogger(__name__)
from dotenv import load_dotenv
load_dotenv()

JIRA_URL = os.environ.get("JIRA_URL")
JIRA_EMAIL = os.environ.get("JIRA_EMAIL")
JIRA_API_TOKEN = os.environ.get("JIRA_API_TOKEN")
JIRA_PROJECT_KEY = os.environ.get("JIRA_PROJECT_KEY")


def _load_team_map():
    raw = os.environ.get("JIRA_TEAM_MAP", "")
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        log.error("JIRA_TEAM_MAP is not valid JSON; Team field will be skipped.")
        return {}

TEAM_TO_JIRA_TEAM = _load_team_map()

_jira_tools = None


def _configured():
    return all([JIRA_URL, JIRA_EMAIL, JIRA_API_TOKEN, JIRA_PROJECT_KEY])


def _get_jira_tools():
    """Built once per process, only if Jira is actually configured."""
    global _jira_tools
    if _jira_tools is None and _configured():
        from agno.tools.jira import JiraTools
        _jira_tools = JiraTools(
            server_url=JIRA_URL,
            username=JIRA_EMAIL,
            token=JIRA_API_TOKEN,
        )
    return _jira_tools


def create_issue(ticket_id, team, reasons, ticket_text, draft_reply):
    """Files a Jira issue for one escalated ticket. Returns the created
    issue's key (e.g. "SUP-142") on success, or None if Jira isn't
    configured or the call fails — callers must handle None."""
    if not _configured():
        log.warning("Jira not configured (missing env vars) — skipping create_issue "
                    f"for ticket {ticket_id}.")
        return None

    tools = _get_jira_tools()
    if tools is None:
        log.error(f"Jira tool initialization failed for ticket {ticket_id}.")
        return None

    jira_team_id = TEAM_TO_JIRA_TEAM.get(team) or TEAM_TO_JIRA_TEAM.get("General Support")   
    summary = f"[{team}] Ticket {ticket_id} needs review ({', '.join(reasons) or 'flagged'})"
    description = (
        f"*Ticket ID:* {ticket_id}\n"
        f"*Team:* {team}\n"
        f"*Escalation reasons:* {', '.join(reasons) or 'none listed'}\n\n"
        f"*Customer ticket:*\n{ticket_text}\n\n"
        f"*Drafted reply (for review, not yet sent):*\n{draft_reply or '(none generated)'}"
    )

    try:
        issue_fields = {
            "project": {"key": JIRA_PROJECT_KEY},
            "summary": summary,
            "description": description,
            "issuetype": {"name": "Request"},
        }
        if jira_team_id:
            issue_fields["customfield_10001"] = jira_team_id
        result = tools.jira.create_issue(fields=issue_fields)

        issue_key = result.get("key") if isinstance(result, dict) else str(result)
        log.info(f"Filed Jira issue {issue_key} for ticket {ticket_id}")
        return issue_key
    except Exception as e:
        log.error(f"Jira create_issue failed for ticket {ticket_id}: {e}")
        return None