"""
Phase 6 — Retrieval Agent: Confidence
Multi-Agent Customer Support Intelligence Platform

Turns a retrieved result's distance into a "high"/"low" confidence label —
the signal Escalation's "Retrieval confidence" input needs, and what the
Retrieval Agent's own "confidence? LOW -> retry / GOOD -> return evidence"
branch in the project spec is checking.

Thresholds below come from retrieval_eval.py's actual runs (see
reports/retrieval_eval_report.txt, 20-query resolution / 5-query FAQ
sample, post product-name-mask fix), not guessed numbers — but the two
source types are NOT equally trustworthy, and using the same cutoff for
both would misrepresent what the evaluation actually found:

- FAQ: correct-hit distances 0.2028-0.2579 (5/5 correct in the gold set),
  with off-topic queries during manual testing scoring 0.41-0.48. Wide,
  clean margin — threshold 0.33 (params.yaml; raised from 0.30 after a
  login-problem ticket scored 0.3283) sits in that gap. A FAQ
  confidence label is genuinely meaningful.

- resolution_template: correct-hit distances 0.2713-0.4008 (median 0.326)
  heavily OVERLAP wrong-top-1 distances 0.2480-0.3594. No cutoff cleanly
  separates right from wrong here. Threshold 0.35 is a deliberate,
  explicit trade-off: it sits above the correct-hit MEDIAN (so most
  genuinely correct hits register as "high"), but it also sits above most
  of the wrong-top-1 range — meaning a real, non-trivial share of WRONG
  resolution-template matches will also score "high" confidence. That is
  a known, accepted false-positive risk, not an oversight. Downstream
  logic (Escalation) should treat a resolution-template "high" as weaker
  evidence than a FAQ "high" — e.g. don't auto-resolve on resolution-
  template confidence alone without a corroborating signal.

- Evaluation independence: the 20 resolution queries come from
support_tickets_test.csv, and the resolution-template corpus is built
from support_tickets_train.csv only (preprocessing_rag.py), so no query
ticket's own row is in the index. Resolution texts are heavily templated
(about 77% exact duplicates), so a test ticket can still match a train
template with the same wording. That is generalization across templated
text, not leakage, but it makes the template Recall numbers easier than
they would be on varied resolutions. The 5 FAQ queries are hand-picked
against a separate FAQ file.

Re-derive these thresholds if retrieval_eval.py is ever re-run on a larger
sample or after further pipeline changes — they're a snapshot of one
evaluation run, not fixed constants. Override via params.yaml's
rag.confidence_thresholds if you want to tune without editing code.
"""

from pathlib import Path
import yaml


def _load_params() -> dict:
    params_path = Path(__file__).resolve().parents[2] / "params.yaml"
    with open(params_path) as f:
        return yaml.safe_load(f)


_params = _load_params()

CONFIDENCE_THRESHOLDS = _params["rag"].get("confidence_thresholds", {
    "faq": 0.33,
    "resolution_template": 0.35,  # weaker signal — see module docstring
})
DEFAULT_THRESHOLD = 0.30


def classify_confidence(distance: float, source_type: str) -> str:
    """distance: the result's cosine distance (lower = more similar).
    source_type: "faq" or "resolution_template" (from the result's metadata).
    Returns "high" or "low"."""
    if distance is None:
        return "low"
    threshold = CONFIDENCE_THRESHOLDS.get(source_type, DEFAULT_THRESHOLD)
    return "high" if distance <= threshold else "low"