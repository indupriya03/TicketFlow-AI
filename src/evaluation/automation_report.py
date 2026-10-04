"""
Phase 5 — Automation rate vs. escalation rate (from the SQLite tickets_log)
Multi-Agent Customer Support Intelligence Platform

Reads tickets_log via learning_agent.get_connection() and reports:
  - total rows, excluded rows (demo + greetings), real support tickets
  - escalation_decision breakdown on real tickets
  - automation rate (auto_resolve / real tickets) vs. the brief's 50%+ target
  - escalation rate by category

Usage (from project root):
    python -m src.evaluation.automation_report

Writes reports/automation_report.txt and prints the same.
"""

import re
from pathlib import Path

import pandas as pd

from src.learning.learning_agent import get_connection

ROOT_DIR = Path(__file__).parent.parent.parent
REPORT_PATH = ROOT_DIR / "reports/automation_report.txt"

TARGET_AUTOMATION = 0.50
EXCLUDE_GREETINGS = True   # False = count greetings in the denominator
AUTO_VALUE = "auto_resolve"

# --- ASSUMPTION: how greetings are identified. I haven't seen your Intake
# agent, so this uses the intent label plus a short-text pattern. The report
# prints how many rows it excluded so you can verify the count.
GREETING_INTENT_RE = re.compile(r"greet|small.?talk|trivial", re.I)
GREETING_TEXT_RE = re.compile(
    r"^\s*(hi|hello|hey|hallo|good (morning|afternoon|evening)|thanks?|thank you)\b[\s!.,?]*$",
    re.I,
)
# ---------------------------------------------------------------------------


def _is_greeting(row):
    return bool(GREETING_INTENT_RE.search(str(row["intent"] or ""))
                or GREETING_TEXT_RE.match(str(row["ticket_text_clean"] or "")))


def main():
    conn = get_connection()
    df = pd.read_sql_query(
        "SELECT ticket_id, intent, category, escalation_decision, ticket_text_clean "
        "FROM tickets_log", conn)
    conn.close()

    total = len(df)
    demo = df["ticket_id"].astype(str).str.startswith("DEMO")
    greet = df.apply(_is_greeting, axis=1) if EXCLUDE_GREETINGS else pd.Series(False, index=df.index)
    real = df[~demo & ~greet & df["escalation_decision"].notna()].copy()

    out = ["AUTOMATION vs. ESCALATION REPORT", ""]
    out.append(f"Total rows logged:              {total}")
    out.append(f"Excluded demo rows (DEMO-*):    {int(demo.sum())}")
    out.append(f"Excluded greetings/trivial:     {int((greet & ~demo).sum())}"
               f"{'' if EXCLUDE_GREETINGS else ' (EXCLUDE_GREETINGS=False)'}")
    out.append(f"Real support tickets (denominator): {len(real)}\n")

    if real.empty:
        out.append("No real tickets in the log yet.")
    else:
        out.append("escalation_decision breakdown:")
        for k, v in real["escalation_decision"].value_counts().items():
            out.append(f"  {k:<20} {v:>5}  ({v / len(real):.1%})")

        rate = (real["escalation_decision"] == AUTO_VALUE).mean()
        verdict = "PASS" if rate >= TARGET_AUTOMATION else "BELOW TARGET"
        out.append(f"\nAutomation rate: {rate:.1%} vs target {TARGET_AUTOMATION:.0%}+  -> {verdict}")
        out.append(f"Escalation rate: {1 - rate:.1%}")

        out.append("\nEscalation rate by category (non-auto_resolve share):")
        g = real.assign(escalated=real["escalation_decision"] != AUTO_VALUE).groupby(
            real["category"].fillna("(none)"))["escalated"].agg(["count", "mean"])
        for cat, r in g.sort_values("mean", ascending=False).iterrows():
            out.append(f"  {cat:<28} n={int(r['count']):<5} escalated={r['mean']:.1%}")

    text = "\n".join(out)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(text)
    print(text)
    print(f"\nSaved -> {REPORT_PATH}")


if __name__ == "__main__":
    main()