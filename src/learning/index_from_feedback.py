"""
Phase 11b — Retrieval learning from human-reviewed replies
Multi-Agent Customer Support Intelligence Platform

Manual, periodic script — NOT run per ticket. The retrieval half of the
learning loop (retrain_from_feedback.py is the classifier half).

What it does:
  1. Reads tickets_log rows where a human acted on the draft reply.
  2. By default keeps only replies a human WROTE (human_action ==
     "rejected_edited"). "approved" rows are the bot's own draft, so adding
     them would just feed the model's phrasing back to itself
     (--include-approved overrides this).
  3. Cleans each reply the same way the rest of the knowledge base is
     cleaned (order IDs, emails, phones, cards), but keeps the company's own
     support address (see COMPANY_EMAILS).
  4. Keeps only sentences that read as general knowledge. A sentence is
     dropped if it is written in the agent's voice (I / we / us / our, which
     is where every "I've passed your case...", "we're confirming...",
     "we'll be in touch" lives, whatever the verb), asks the customer a
     question, names a specific amount, apologises, or held a redacted
     customer detail. What survives is third-person policy and instructions
     ("Refunds are credited within...", "go to My Orders > Report..."),
     which is safe for the Response model to reuse.
  5. Skips what is left if it is too short, a duplicate, or already indexed.
  6. Upserts the survivors into the existing Chroma collection as
     source_type="resolution_template" (so Response treats them as PAST
     RESOLUTIONS) with origin="human_reviewed" in the metadata.

DRY RUN BY DEFAULT: prints every document it would add and why anything was
skipped. Read them before using --apply — regex cannot catch a customer's
name typed into a reply.

Notes:
  - retrieval_document.py --reset rebuilds the collection from
    rag_documents.json only, which DROPS these documents. tickets_log is the
    source of truth, so just re-run this script with --apply afterwards.
  - Stop the API (uvicorn) before --apply and restart it afterwards, so the
    running process reloads the updated index.

Usage:
    python -m src.learning.index_from_feedback
    python -m src.learning.index_from_feedback --apply
    python -m src.learning.index_from_feedback --apply --include-approved
    python -m src.learning.index_from_feedback --exclude TICKET-aaaa1111 TICKET-bbbb2222 --apply
"""

import argparse
import json
import re
from pathlib import Path

from src.preprocessing import light_clean_text, pii_scrub
from src.response.response_agent import claims_unsupported_action

ROOT_DIR = Path(__file__).parent.parent.parent
PERSIST_DIR = ROOT_DIR / "chroma_db"
COLLECTION_NAME = "support_kb"
AUDIT_PATH = ROOT_DIR / "data/processed/learned_documents.json"

# Real company addresses that must NOT be redacted to <EMAIL>.
COMPANY_EMAILS = ("support@shop.com",)
MIN_WORDS = 12  # after removing the sentences described above

# Sentences that are about this one customer or spoken in the agent's voice,
# not reusable knowledge. (?i:...) keeps "I" case-sensitive so "i.e." is safe.
NOT_REUSABLE_RE = re.compile(
    r"\bI\b|\b(?i:we|us|our)\b"                        # agent voice, any verb
    r"|\?"                                               # asks the customer something
    r"|[€$₹£]\s?\d"                                       # a specific amount
    r"|\b(?i:sorry|apolog\w*)\b"                         # ticket-specific tone
    r"|<(?:EMAIL|PHONE|CARD_NUMBER)>|#ORDER_ID",          # redacted customer detail
)

# ---------------------------------------------------------------- helpers

def _scrub(text):
    """light_clean_text + pii_scrub, but keep the company's own addresses
    (pii_scrub alone turns them into <EMAIL>, as it once did for the FAQ)."""
    # Same plain characters the Response prompt asks for (FAQ menu paths use ">").
    text = text.replace("\u202f", " ").replace("\u00a0", " ").replace("→", ">")
    protected = {addr: f"COMPANYMAIL{chr(65 + i)}X" for i, addr in enumerate(COMPANY_EMAILS)}
    for addr, token in protected.items():
        text = re.sub(re.escape(addr), token, text, flags=re.IGNORECASE)
    text, _ = pii_scrub(light_clean_text(text))
    for addr, token in protected.items():
        text = text.replace(token, addr)
    return text


def _keep_informational(text):
    """Keep only reusable, informational sentences (see NOT_REUSABLE_RE), with the
    Response agent's action-claim regex as a second net.
    Returns (text, n_dropped)."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    kept = [s for s in sentences
            if not claims_unsupported_action(s) and not NOT_REUSABLE_RE.search(s)]
    return " ".join(kept).strip(), len(sentences) - len(kept)


def select_documents(rows, include_approved=False, exclude=()):
    """rows: list of dicts from tickets_log. Returns (records, skipped), where
    records are Chroma-ready {"id","document","metadata"} and skipped is a
    list of (ticket_id, reason)."""
    allowed = {"rejected_edited"} | ({"approved"} if include_approved else set())
    records, skipped, seen = [], [], set()

    for r in rows:
        tid = r["ticket_id"]
        if tid in exclude:
            skipped.append((tid, "excluded with --exclude"))
            continue
        if r["human_action"] not in allowed:
            skipped.append((tid, f"human_action={r['human_action']} (the bot's own draft)"))
            continue

        text, dropped = _keep_informational(_scrub(r["final_sent_text"]))
        words = len(text.split())
        if words < MIN_WORDS:
            skipped.append((tid, f"only {words} words left after removing "
                                 f"{dropped} non-reusable sentence(s)"))
            continue

        key = re.sub(r"\W+", " ", text.lower()).strip()
        if key in seen:
            skipped.append((tid, "duplicate of an earlier reply"))
            continue
        seen.add(key)

        records.append({
            "id": f"human_{tid}",
            "document": text,
            "metadata": {
                "source_type": "resolution_template",
                "ticket_id": tid,
                "category": r.get("human_verified_category") or r.get("category") or "",
                "origin": "human_reviewed",
                "human_action": r["human_action"],
            },
        })
    return records, skipped


def load_rows():
    from src.learning.learning_agent import get_connection
    conn = get_connection()
    cur = conn.execute(
        "SELECT ticket_id, ticket_text_clean, category, human_verified_category, "
        "human_action, final_sent_text FROM tickets_log "
        "WHERE human_action IS NOT NULL AND final_sent_text IS NOT NULL "
        "AND final_sent_text != '' ORDER BY logged_at"
    )
    cols = [c[0] for c in cur.description]
    rows = [dict(zip(cols, r)) for r in cur.fetchall()]
    conn.close()
    return rows


def apply_to_chroma(records):
    import chromadb
    from src.retrieval.gemini_embeddings import get_document_embedder, BATCH_SIZE

    client = chromadb.PersistentClient(path=str(PERSIST_DIR))
    try:
        collection = client.get_collection(COLLECTION_NAME,
                                           embedding_function=get_document_embedder())
    except Exception as e:
        print(f"Collection '{COLLECTION_NAME}' not found ({e}). "
              f"Run src.retrieval.retrieval_document first.")
        return

    present = set(collection.get(ids=[r["id"] for r in records])["ids"])
    new = [r for r in records if r["id"] not in present]
    print(f"{len(records)} selected, {len(present)} already indexed, {len(new)} to embed.")

    for i in range(0, len(new), BATCH_SIZE):
        batch = new[i:i + BATCH_SIZE]
        collection.upsert(
            ids=[r["id"] for r in batch],
            documents=[r["document"] for r in batch],
            metadatas=[r["metadata"] for r in batch],
        )
        print(f"Upserted batch {i // BATCH_SIZE + 1} ({len(batch)} docs)")

    AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    AUDIT_PATH.write_text(json.dumps(records, indent=2, ensure_ascii=False))
    print(f"Collection now has {collection.count()} documents. Audit copy: {AUDIT_PATH.name}")


# ------------------------------------------------------------------- main

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true",
                        help="Actually embed and upsert (default is a dry run)")
    parser.add_argument("--include-approved", action="store_true",
                        help="Also learn from replies the human approved unchanged")
    parser.add_argument("--exclude", nargs="*", default=[], metavar="TICKET_ID",
                        help="Ticket IDs to leave out after reading the dry run")
    args = parser.parse_args()

    rows = load_rows()
    records, skipped = select_documents(rows, include_approved=args.include_approved,
                                        exclude=set(args.exclude))
    print(f"{len(rows)} human-reviewed rows -> {len(records)} usable, {len(skipped)} skipped.\n")

    for r in records:
        print(f"[ADD] {r['id']} (category: {r['metadata']['category'] or '-'})")
        print(f"      {r['document']}\n")
    for tid, reason in skipped:
        print(f"[SKIP] {tid}: {reason}")

    if not records:
        print("\nNothing to add.")
        return
    if not args.apply:
        print("\nDry run only. Read the [ADD] documents above, then re-run with --apply.")
        return
    apply_to_chroma(records)


if __name__ == "__main__":
    main()