"""
Phase 5 — Batch evaluation through the full pipeline
Multi-Agent Customer Support Intelligence Platform

Samples DISTINCT tickets from the held-out test set, runs each through
orchestrator.start_ticket(), and reports automation rate (with a 95% Wilson
interval), escalation reasons, latency, and end-to-end Category/Priority
accuracy. Replaces the dev-log numbers in tickets.db, which are dominated by
repeated test inputs.

Isolation (so nothing in your real data is touched):
  - learning_agent.DB_PATH / CSV_PATH are redirected to data/eval/
  - create_issue is stubbed out so no Jira issues are filed

Results are appended to data/eval/batch_results.csv after every ticket, so a
crash or a Groq quota limit loses nothing: re-run with --resume.

Usage (from project root):
    python -m src.evaluation.batch_eval --n 100
    python -m src.evaluation.batch_eval --n 100 --resume --sleep 2
    python -m src.evaluation.batch_eval --report-only

Notes:
  - Tickets where Intake asks a follow-up question come back as
    "awaiting_customer"; there is no customer to answer, so they are counted
    separately and are NOT in the automation denominator. (The orchestrator
    does not log them to the DB either.)
  - LLM output varies run to run; report the interval, not just the point.
"""

import argparse
import csv
import json
import math
import time
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).parent.parent.parent
TEST_PATH = ROOT_DIR / "data/processed/support_tickets_test.csv"
EVAL_DIR = ROOT_DIR / "data/eval"
EVAL_DB = EVAL_DIR / "eval_tickets.db"
EVAL_RETRAIN_CSV = EVAL_DIR / "eval_retraining.csv"
RESULTS_PATH = EVAL_DIR / "batch_results.csv"
REPORT_PATH = ROOT_DIR / "reports/batch_eval_report.txt"

TARGET = 0.50
FIELDS = ["ticket_id", "outcome", "true_category", "true_priority", "intent",
          "sentiment", "category", "priority", "complexity", "team",
          "retrieval_confidence", "response_mode", "response_needs_review",
          "escalation_decision", "escalation_reasons", "seconds", "error",
          "ticket_text", "response_text"]


def _none(v):
    return None if v is None or (isinstance(v, float) and math.isnan(v)) else v


def _wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    a = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - a) / d, (c + a) / d


def _isolate():
    """Redirect logging and disable Jira BEFORE any ticket runs."""
    import src.learning.learning_agent as la
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    la.DB_PATH = EVAL_DB
    la.CSV_PATH = EVAL_RETRAIN_CSV

    import src.orchestrator as orch

    def _no_jira(*args, **kwargs):
        return None

    orch.create_issue = _no_jira
    try:  # the Escalation agent may import create_issue itself
        import src.escalation.escalation_agent as ea
        if hasattr(ea, "create_issue"):
            ea.create_issue = _no_jira
    except Exception:
        pass
    return orch


def _sample(n, seed):
    df = pd.read_csv(TEST_PATH)
    if "ticket_text_clean" not in df.columns:
        raise SystemExit(f"{TEST_PATH.name} has no ticket_text_clean column.")
    df = df.dropna(subset=["ticket_text_clean"]).drop_duplicates(subset="ticket_text_clean")
    df = df.sample(n=min(n, len(df)), random_state=seed)
    id_col = "ticket_id" if "ticket_id" in df.columns else None
    df["_eval_id"] = [f"EVAL-{r[id_col]}" if id_col else f"EVAL-{i:05d}"
                      for i, (_, r) in enumerate(df.iterrows())]
    return df


def _run_one(orch, eval_id, row):
    text = _none(row.get("ticket_text")) or row["ticket_text_clean"]
    rec = {"ticket_id": eval_id, "ticket_text": text,
           "true_category": _none(row.get("ticket_category")),
           "true_priority": _none(row.get("priority"))}
    t0 = time.time()
    try:
        res = orch.start_ticket(eval_id, text,
                                product_name=_none(row.get("product_name")),
                                product_segment=_none(row.get("product_segment")))
        rec["seconds"] = round(time.time() - t0, 2)
        if res.get("status") == "awaiting_customer":
            rec["outcome"] = "awaiting_customer"
            return rec
        rec["outcome"] = "done"
        for k in ("intent", "sentiment", "category", "priority", "complexity", "team",
                  "retrieval_confidence", "response_mode", "response_needs_review",
                  "escalation_decision", "response_text"):
            rec[k] = res.get(k)
        rec["escalation_reasons"] = json.dumps(res.get("escalation_reasons") or [])
    except Exception as e:
        rec["seconds"] = round(time.time() - t0, 2)
        rec["outcome"] = "error"
        rec["error"] = f"{type(e).__name__}: {e}"[:300]
    return rec


def _append(rec):
    new = not RESULTS_PATH.exists()
    with open(RESULTS_PATH, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow(rec)


def _report():
    df = pd.read_csv(RESULTS_PATH)
    out = ["BATCH EVALUATION — FULL PIPELINE (distinct test-set tickets)", ""]
    out.append(f"Tickets attempted: {len(df)}")
    for k, v in df["outcome"].value_counts().items():
        out.append(f"  {k:<18} {v}")

    done = df[df["outcome"] == "done"].copy()
    real = done[done["escalation_decision"] != "no_action_needed"]
    n = len(real)
    if n == 0:
        out.append("\nNo completed tickets to analyse.")
        return "\n".join(out)

    k = int((real["escalation_decision"] == "auto_resolve").sum())
    lo, hi = _wilson(k, n)
    out.append(f"\nDenominator: {n} completed tickets "
               f"({len(done) - n} greeting/no-action excluded; awaiting_customer and errors excluded)")
    out.append("\nescalation_decision:")
    for key, v in real["escalation_decision"].value_counts().items():
        out.append(f"  {key:<20} {v:>4}  ({v / n:.1%})")
    out.append(f"\nAutomation rate: {k / n:.1%}  (95% CI {lo:.1%} to {hi:.1%}) vs target {TARGET:.0%}+")
    if lo >= TARGET:
        out.append("  -> target met (whole interval above 50%).")
    elif hi < TARGET:
        out.append("  -> target missed (whole interval below 50%).")
    else:
        out.append("  -> inconclusive: the interval includes 50%.")

    out.append("\nEscalation reasons (a ticket can have several):")
    reasons = real[real["escalation_decision"] == "escalate_to_team"]["escalation_reasons"] \
        .dropna().map(json.loads).explode().dropna()
    for key, v in reasons.value_counts().items():
        out.append(f"  {key:<32} {v:>4}")

    out.append("\nAutomation by category (predicted):")
    g = real.assign(auto=real["escalation_decision"] == "auto_resolve") \
        .groupby(real["category"].fillna("(none)"))["auto"].agg(["count", "mean"])
    for cat, r in g.sort_values("count", ascending=False).iterrows():
        out.append(f"  {cat:<28} n={int(r['count']):<4} auto_resolved={r['mean']:.1%}")

    out.append("\nEnd-to-end label accuracy (pipeline output vs test-set label):")
    for pred, true in (("category", "true_category"), ("priority", "true_priority")):
        sub = done.dropna(subset=[pred, true])
        if len(sub):
            acc = (sub[pred].astype(str) == sub[true].astype(str)).mean()
            out.append(f"  {pred:<9} {acc:.1%}  (n={len(sub)})")
    out.append("\nEscalation vs the historical `escalated` label (test set):")
    truth = pd.read_csv(TEST_PATH, usecols=["ticket_id", "escalated"])
    truth["ticket_id"] = "EVAL-" + truth["ticket_id"].astype(str)
    ev = real.merge(truth, on="ticket_id", how="left").dropna(subset=["escalated"])
    if len(ev) == 0:
        out.append("  no tickets could be matched to the test-set label.")
    else:
        y_true = ev["escalated"].astype(str).str.lower().isin(["true", "1", "yes"])
        y_pred = ev["escalation_decision"] == "escalate_to_team"
        tp = int((y_true & y_pred).sum())
        fp = int((~y_true & y_pred).sum())
        fn = int((y_true & ~y_pred).sum())
        tn = int((~y_true & ~y_pred).sum())
        m = len(ev)
        prec = tp / (tp + fp) if (tp + fp) else float("nan")
        rec = tp / (tp + fn) if (tp + fn) else float("nan")
        f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else float("nan")
        pct = lambda v: "n/a" if v != v else f"{v:.1%}"
        p_lo, p_hi = _wilson(tp, tp + fp)
        r_lo, r_hi = _wilson(tp, tp + fn)
        out.append(f"  Tickets compared: {m}")
        out.append(f"  Historical escalated rate (these tickets): {y_true.mean():.1%}")
        out.append(f"  TicketFlow escalation rate (these tickets): {y_pred.mean():.1%}")
        out.append("  Confusion matrix (rows = historical label, columns = TicketFlow):")
        out.append(f"{'':<24}{'auto_resolve':>14}{'escalate_to_team':>18}")
        out.append(f"    {'not escalated':<20}{tn:>14}{fp:>18}")
        out.append(f"    {'escalated':<20}{fn:>14}{tp:>18}")
        out.append(f"  Precision: {pct(prec)}  (95% CI {p_lo:.1%} to {p_hi:.1%})")
        out.append(f"  Recall:    {pct(rec)}  (95% CI {r_lo:.1%} to {r_hi:.1%})")
        out.append(f"  F1:        {pct(f1)}")
        out.append(f"  Chance reference: a random escalator with TicketFlow's rate would "
                   f"score precision near {y_true.mean():.1%} and recall near {y_pred.mean():.1%}.")
        out.append("  Note: TicketFlow decides from runtime signals (priority, sentiment, "
                   "complexity, refund amount, review flag) and never sees the historical "
                   "`escalated` field; differences are measured, not errors by definition.")
    out.append("\nLatency per ticket (seconds, includes LLM calls):")
    out.append(f"  all:          mean {real['seconds'].mean():.1f} | median {real['seconds'].median():.1f}")
    auto = real[real["escalation_decision"] == "auto_resolve"]
    if len(auto):
        out.append(f"  auto_resolve: mean {auto['seconds'].mean():.1f} | median {auto['seconds'].median():.1f}")
    out.append("\nFor the response-quality spot-check, read response_text in "
               f"{RESULTS_PATH.relative_to(ROOT_DIR)}.")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100, help="distinct tickets to sample")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--sleep", type=float, default=0.0, help="seconds between tickets (rate limits)")
    ap.add_argument("--resume", action="store_true", help="skip tickets already in the results file")
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()

    if not args.report_only:
        if RESULTS_PATH.exists() and not args.resume:
            raise SystemExit(f"{RESULTS_PATH} exists. Use --resume to continue it, "
                             f"or delete it (and {EVAL_DB.name}) to start over.")
        orch = _isolate()
        sample = _sample(args.n, args.seed)
        done_ids = set(pd.read_csv(RESULTS_PATH)["ticket_id"]) if RESULTS_PATH.exists() else set()
        todo = sample[~sample["_eval_id"].isin(done_ids)]
        print(f"{len(sample)} sampled, {len(done_ids)} already done, {len(todo)} to run.")
        for i, (_, row) in enumerate(todo.iterrows(), 1):
            rec = _run_one(orch, row["_eval_id"], row)
            _append(rec)
            print(f"[{i}/{len(todo)}] {rec['ticket_id']} -> {rec['outcome']} "
                  f"{rec.get('escalation_decision') or rec.get('error') or ''}")
            if args.sleep:
                time.sleep(args.sleep)

    text = _report()
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(text)
    print("\n" + text + f"\n\nSaved -> {REPORT_PATH}")


if __name__ == "__main__":
    main()