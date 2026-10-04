"""
Phase 6 — Retrieval Agent: Document Indexing
Multi-Agent Customer Support Intelligence Platform

Embeds the pre-built documents from data/processed/rag_documents.json
(see preprocessing_rag.py — document construction is deliberately kept out
of this file, so re-checking dedup/formatting logic never needs a Gemini
call, and re-embedding never needs to re-derive the documents) into a
persistent Chroma collection.

This does NOT run per-ticket — retrieval_query.py is what runs at inference
time, per ticket, embedding ONE query at a time via
gemini_embeddings.get_query_embedder(). That per-query call pattern never
approaches Gemini's free-tier rate limit on its own; this script is the
only place in the project that submits a bulk burst of embedding calls in
one go, which is why batching matters here specifically.

Talks to google.genai directly via gemini_embeddings.GeminiEmbeddingFunction,
not through langchain-google-genai — an earlier LangChain-based version of
this script kept hitting RESOURCE_EXHAUSTED regardless of the chunk size
passed to it (same rate-limit hit rate at chunk sizes 20 AND 100), which
only makes sense if the wrapper was splitting requests beneath whatever
batch it was handed — meaning the chunk size was never actually controlling
how many real HTTP requests reached Google. Calling
client.models.embed_content(contents=...) directly (inside
gemini_embeddings.py) removes that opacity: the batch passed IS the request.

Re-running upserts by ID rather than wiping — ids are deterministic
("faq_<faq_id>", "res_<ticket_id>"), so re-running with unchanged source
data is a no-op past the first time, and changed rows just overwrite in
place. Pass --reset to delete and rebuild the collection from scratch.

Requires: GEMINI_API_KEY or GOOGLE_API_KEY environment variable, and a
params.yaml at the project root with:
    rag:
      embedding_model: models/gemini-embedding-001
      embedding_output_dim: 768
      embedding_batch_size: 100

Input:  data/processed/rag_documents.json  (from preprocessing_rag.py)
Output: chroma_db/  (persistent Chroma collection "support_kb")
        reports/retrieval_index_report.txt
"""

import argparse
import json
from pathlib import Path
import time
import chromadb

from src.retrieval.gemini_embeddings import get_document_embedder, BATCH_SIZE

ROOT_DIR = Path(__file__).parent.parent.parent
DOCUMENTS_PATH = ROOT_DIR / "data/processed/rag_documents.json"
PERSIST_DIR = ROOT_DIR / "chroma_db"
COLLECTION_NAME = "support_kb"
REPORT_PATH = ROOT_DIR / "reports/retrieval_index_report.txt"

report_lines = []


def log(msg):
    print(msg)
    report_lines.append(msg)


def load_documents():
    if not DOCUMENTS_PATH.exists():
        raise FileNotFoundError(
            f"{DOCUMENTS_PATH} not found — run preprocessing_rag.py first "
            "to build it from the FAQ and ticket source data."
        )
    with open(DOCUMENTS_PATH) as f:
        records = json.load(f)
    ids = [r["id"] for r in records]
    documents = [r["document"] for r in records]
    metadatas = [r["metadata"] for r in records]
    return ids, documents, metadatas


def build_vector_store(reset: bool = False):
    log("PHASE 6 — RETRIEVAL AGENT: DOCUMENT INDEXING")
    log("=" * 60)

    Path(ROOT_DIR / "reports").mkdir(parents=True, exist_ok=True)
    PERSIST_DIR.mkdir(parents=True, exist_ok=True)

    all_ids, all_docs, all_metas = load_documents()
    log(f"Loaded {len(all_ids)} pre-built documents from {DOCUMENTS_PATH}")

    client = chromadb.PersistentClient(path=str(PERSIST_DIR))

    if reset:
        try:
            client.delete_collection(COLLECTION_NAME)
            log(f"Deleted existing collection '{COLLECTION_NAME}' (--reset)")
        except Exception:
            pass

    embedder = get_document_embedder()
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedder,
        metadata={"hnsw:space": "cosine"},
    )

    existing_count = collection.count()
    if existing_count > 0 and not reset:
        log(f"Collection already has {existing_count} documents. "
            f"Upserting on top (pass --reset to rebuild from scratch instead).")

    log(f"\nEmbedding with {embedder.task_type} via gemini_embeddings.py, "
        f"in chunks of {BATCH_SIZE} (Gemini's documented max batch size)...")
    embedded_any = False
    for i in range(0, len(all_ids), BATCH_SIZE):
        batch_ids = all_ids[i:i + BATCH_SIZE]
        batch_docs = all_docs[i:i + BATCH_SIZE]
        batch_metas = all_metas[i:i + BATCH_SIZE]

        # Skip ids already in the collection (e.g. from a prior run that
        # got partway through before hitting a rate limit) — re-upserting
        # them would re-embed identical text, wasting scarce free-tier
        # quota on documents that already succeeded. Irrelevant right
        # after --reset (collection is empty), but real savings on a
        # resumed run.
        if not reset:
            already_present = set(collection.get(ids=batch_ids)["ids"])
            if already_present:
                keep = [idx for idx, doc_id in enumerate(batch_ids) if doc_id not in already_present]
                if not keep:
                    log(f"Batch {i // BATCH_SIZE + 1}: all {len(batch_ids)} docs already indexed, skipping")
                    continue
                skipped = len(batch_ids) - len(keep)
                log(f"Batch {i // BATCH_SIZE + 1}: {skipped} of {len(batch_ids)} docs already "
                    f"indexed, embedding only the remaining {len(keep)}")
                batch_ids = [batch_ids[idx] for idx in keep]
                batch_docs = [batch_docs[idx] for idx in keep]
                batch_metas = [batch_metas[idx] for idx in keep]
        if embedded_any:
            log("Waiting 65s so the per-minute embedding quota resets...")
            time.sleep(65)

        collection.upsert(ids=batch_ids, documents=batch_docs, metadatas=batch_metas)
        embedded_any = True
        log(f"Upserted batch {i // BATCH_SIZE + 1} ({len(batch_ids)} docs)")
        
    final_count = collection.count()
    log(f"\nIndexed: collection '{COLLECTION_NAME}' now has {final_count} documents "
        f"at {PERSIST_DIR}")

    # Sanity check via chromadb's own public .get(), not a private attribute.
    faq_stored = collection.get(where={"source_type": "faq"})
    res_stored = collection.get(where={"source_type": "resolution_template"})
    log(f"\nSanity check — stored FAQ docs: {len(faq_stored['ids'])}, "
        f"stored resolution-template docs: {len(res_stored['ids'])}")

    with open(REPORT_PATH, "w") as f:
        f.write("\n".join(report_lines))
    log(f"\nSaved report to {REPORT_PATH}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true",
                         help="Delete and rebuild the collection from scratch instead of upserting")
    args = parser.parse_args()
    build_vector_store(reset=args.reset)


if __name__ == "__main__":
    main()