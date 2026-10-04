"""
Phase 11 — Batch Retraining from Verified Feedback
Multi-Agent Customer Support Intelligence Platform

Manual, periodic script — NOT run automatically per ticket.

How it works (the "learning-from-feedback loop"):
  1. Read human-VERIFIED rows from tickets_for_retraining.csv (never
     unverified auto_resolve predictions — see learning_agent.py).
  2. Merge them with the base training set IN MEMORY. The shared
     support_tickets_train.csv is never modified, so no other script ever
     sees null-filled rows, and re-running can never add a row twice.
  3. Call the trainer with that merged DataFrame.
  4. Keep the new model only if its metric on the fixed test set is >= the
     previous one. Otherwise restore the old model AND its report.
  5. On a keep, write a small manifest next to the model recording exactly
     which feedback tickets it was trained on.

Priority-specific constraint: priority_train.py trains ONLY on the
sentiment == "Neutral" subset (the other sentiment values map to priority
deterministically via assign_priority.py). Verified feedback from a
non-Neutral ticket is therefore excluded from priority retraining.

REQUIRED trainer change (2 lines each, in category_train.py and
priority_train.py): main() must accept an optional DataFrame, e.g.

    def main(train_df=None):
        if train_df is None:
            train_df = pd.read_csv(TRAIN_PATH)
        ...

Usage:
    python -m src.learning.retrain_from_feedback --category
    python -m src.learning.retrain_from_feedback --priority
    python -m src.learning.retrain_from_feedback --both
    python -m src.learning.retrain_from_feedback --both --dry-run
"""

import argparse
import json
import re
import shutil
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).parent.parent.parent
FEEDBACK_PATH = ROOT_DIR / "data/processed/tickets_for_retraining.csv"
TRAIN_PATH = ROOT_DIR / "data/processed/support_tickets_train.csv"

CATEGORY_MODEL_PATH = ROOT_DIR / "models/category_classifier.joblib"
CATEGORY_REPORT_PATH = ROOT_DIR / "reports/category_eval_report.txt"
PRIORITY_MODEL_PATH = ROOT_DIR / "models/priority_neutral_classifier.joblib"
PRIORITY_REPORT_PATH = ROOT_DIR / "reports/priority_eval_report.txt"

# Column the trainers read as the ticket text. Change this if your trainers
# read "ticket_text" instead of "ticket_text_clean".
TEXT_COL = "ticket_text_clean"

MIN_NEW_ROWS = 5  # below this, warn but don't block


# ---------------------------------------------------------------- helpers

def _timestamped_backup(path):
    if not path.exists():
        return None
    backup = path.with_suffix(path.suffix + f".bak_{datetime.now():%Y%m%d%H%M%S}")
    shutil.copy2(path, backup)
    return backup


def _restore(backup, target):
    if backup and backup.exists():
        shutil.copy2(backup, target)
        print(f"Restored {target.name} from {backup.name}")


def _parse_category_f1(report_path):
    if not report_path.exists():
        return None
    matches = re.findall(r"Macro F1: (\d\.\d+)", report_path.read_text())
    return float(matches[-1]) if matches else None


def _parse_priority_best_f1(report_path):
    if not report_path.exists():
        return None
    matches = re.findall(
        r"COMPARISON: LogisticRegression=(\d\.\d+)\s+vs\s+XGBoost=(\d\.\d+)",
        report_path.read_text(),
    )
    return max(map(float, matches[-1])) if matches else None


def _load_feedback():
    if not FEEDBACK_PATH.exists():
        print(f"No feedback file at {FEEDBACK_PATH} yet — nothing to retrain from.")
        return pd.DataFrame()
    return pd.read_csv(FEEDBACK_PATH).fillna("")


def _verified_rows(feedback, verified_col):
    """Rows where a human confirmed the label. De-duplicated by ticket_id
    (latest row wins), then by text + label so a re-submitted ticket counts once."""
    if verified_col not in feedback.columns:
        return feedback.iloc[0:0].copy()
    rows = feedback[(feedback[verified_col] != "") & (feedback[TEXT_COL] != "")].copy()
    rows = rows.drop_duplicates(subset="ticket_id", keep="last")
    return rows.drop_duplicates(subset=[TEXT_COL, verified_col], keep="last")

def _write_manifest(model_path, verified, base_rows, old_metric, new_metric):
    manifest = {
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "base_rows": base_rows,
        "feedback_rows": len(verified),
        "feedback_ticket_ids": sorted(verified["ticket_id"].astype(str).tolist()),
        "previous_metric": old_metric,
        "new_metric": new_metric,
    }
    path = model_path.with_suffix(".manifest.json")
    path.write_text(json.dumps(manifest, indent=2))
    print(f"Wrote manifest -> {path.name}")


def _retrain(label, verified, needed_cols, train_fn, model_path, report_path,
             metric_fn, dry_run):
    """Shared flow for category and priority retraining."""
    if len(verified) < MIN_NEW_ROWS:
        print(f"Only {len(verified)} verified rows (< {MIN_NEW_ROWS}) — unlikely "
              f"to move metrics, but proceeding since it was requested.")
    print(f"{len(verified)} verified {label} rows available.")

    if dry_run:
        print(f"[dry-run] Would merge these rows in memory and retrain {label}. "
              f"{TRAIN_PATH.name} would not be modified.")
        return

    base_df = pd.read_csv(TRAIN_PATH)
    merged = pd.concat([base_df, verified[needed_cols]], ignore_index=True)
    print(f"In-memory training set: {len(base_df)} base + {len(verified)} feedback "
          f"= {len(merged)} rows (on disk: {len(base_df)}).")

    old_metric = metric_fn(report_path)
    print(f"Previous {label} metric: {old_metric}")

    model_backup = _timestamped_backup(model_path)
    report_backup = _timestamped_backup(report_path)

    try:
        train_fn(train_df=merged)
    except Exception as e:
        print(f"Training failed ({e}). Rolling back.")
        _restore(model_backup, model_path)
        _restore(report_backup, report_path)
        return

    new_metric = metric_fn(report_path)
    print(f"New {label} metric: {new_metric}")

    if new_metric is None or (old_metric is not None and new_metric < old_metric):
        reason = ("could not read the new metric from the report" if new_metric is None
                  else f"new model ({new_metric:.4f}) worse than previous ({old_metric:.4f})")
        print(f"REJECTED: {reason}. Rolling back.")
        _restore(model_backup, model_path)
        _restore(report_backup, report_path)
    else:
        print(f"KEPT: new {label} model retained ({old_metric} -> {new_metric}).")
        _write_manifest(model_path, verified, len(base_df), old_metric, new_metric)

# ------------------------------------------------------------- retrainers

def retrain_category(dry_run=False):
    print("\n" + "=" * 60)
    print("CATEGORY RETRAIN")
    feedback = _load_feedback()
    if feedback.empty:
        return

    verified = _verified_rows(feedback, "human_verified_category")
    if verified.empty:
        print("No verified category feedback rows yet.")
        return
    verified["ticket_category"] = verified["human_verified_category"]

    train_fn = None
    if not dry_run:
        from src.triage.category_train import main as train_fn

    _retrain("category", verified, [TEXT_COL, "ticket_category"], train_fn,
             CATEGORY_MODEL_PATH, CATEGORY_REPORT_PATH, _parse_category_f1, dry_run)


def retrain_priority(dry_run=False):
    print("\n" + "=" * 60)
    print("PRIORITY RETRAIN")
    feedback = _load_feedback()
    if feedback.empty:
        return

    verified = _verified_rows(feedback, "human_verified_priority")
    excluded = verified[verified["sentiment"] != "Neutral"]
    verified = verified[verified["sentiment"] == "Neutral"].copy()
    # priority_train.py raises if the Neutral subset holds anything but Low/Medium
    verified = verified[verified["human_verified_priority"].isin(["Low", "Medium"])].copy()

    if len(excluded) > 0:
        print(f"Excluded {len(excluded)} verified rows with non-Neutral sentiment — "
              f"priority_train.py only trains on the Neutral subset.")
    if verified.empty:
        print("No verified Neutral-sentiment priority feedback rows yet.")
        return
    verified["priority"] = verified["human_verified_priority"]

    train_fn = None
    if not dry_run:
        from src.triage.priority_train import main as train_fn

    _retrain("priority", verified, [TEXT_COL, "sentiment", "priority"], train_fn,
             PRIORITY_MODEL_PATH, PRIORITY_REPORT_PATH, _parse_priority_best_f1, dry_run)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", action="store_true")
    parser.add_argument("--priority", action="store_true")
    parser.add_argument("--both", action="store_true")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would happen without retraining")
    args = parser.parse_args()

    if not (args.category or args.priority or args.both):
        parser.error("Specify --category, --priority, or --both")

    if args.category or args.both:
        retrain_category(dry_run=args.dry_run)
    if args.priority or args.both:
        retrain_priority(dry_run=args.dry_run)