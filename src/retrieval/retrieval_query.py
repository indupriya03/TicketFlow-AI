"""
Phase 6 — Retrieval Agent: Query
Multi-Agent Customer Support Intelligence Platform

Queries the persistent Chroma collection built by retrieval_document.py
(via preprocessing_rag.py). This is the per-ticket, inference-time half of
Retrieval — one query embedded at a time, via
gemini_embeddings.get_query_embedder() (task_type=RETRIEVAL_QUERY, NOT
RETRIEVAL_DOCUMENT — must stay asymmetric with how the documents were
indexed, or similarity scores are meaningless). This call pattern never
approaches the free-tier rate limit on its own; retrieval_document.py is
the only place in this project that submits a bulk burst of embedding
calls, which is why all the batching/backoff work lives there, not here.

Ported from BankBot-RAG's src/rag/retriever.py, with one deliberate
omission: BankBot's retrieve() enforces a min_policy_results floor,
force-including some low-scoring policy-document chunks because plain
cosine similarity structurally favors short, conversational QA-pair
chunks over long, formally-worded policy chunks. That asymmetry doesn't
exist here — FAQ entries and resolution templates are both short,
single-topic, similarly-phrased documents (see preprocessing_rag.py) — so
that floor is left out rather than carried over as unnecessary complexity.
If a future evaluation ever shows one source_type is being structurally
under-ranked here too, the same floor technique is the fix; nothing
observed so far suggests it's needed.

Usage:
    from src.retrieval.retrieval_query import Retriever

    r = Retriever()
    results = r.retrieve("my order hasn't arrived in 10 days")
    results = r.retrieve("refund policy", k=3, where={"source_type": "faq"})
"""

import logging
import os
from pathlib import Path
from typing import Optional

import chromadb
import yaml

from src.retrieval.gemini_embeddings import get_query_embedder
from src.retrieval.retrieval_confidence import classify_confidence

log = logging.getLogger(__name__)


def _load_params() -> dict:
    params_path = Path(__file__).resolve().parents[2] / "params.yaml"
    with open(params_path) as f:
        return yaml.safe_load(f)


_params = _load_params()

ROOT_DIR = Path(__file__).parent.parent.parent
DEFAULT_PERSIST_DIR = ROOT_DIR / "chroma_db"
COLLECTION_NAME = "support_kb"
DEFAULT_K = _params["rag"].get("top_k", 5)


class Retriever:
    def __init__(self, persist_dir: str = None):
        persist_dir = persist_dir or os.getenv("CHROMA_PERSIST_DIR", str(DEFAULT_PERSIST_DIR))
        persist_path = Path(persist_dir)
        if not persist_path.exists():
            raise FileNotFoundError(
                f"No vector store found at {persist_dir}. Run "
                "preprocessing_rag.py then retrieval_document.py first."
            )

        self.client = chromadb.PersistentClient(path=str(persist_path))
        self.embedder = get_query_embedder()
        self.collection = self.client.get_collection(
            name=COLLECTION_NAME,
            embedding_function=self.embedder,
        )
        log.info(f"Retriever ready — {self.collection.count()} documents in collection")

    def retrieve(self, query: str, k: int = DEFAULT_K, where: Optional[dict] = None) -> list:
        """
        Args:
            query: the ticket text (or a derived search string) to search with
            k: number of results to return
            where: optional Chroma metadata filter, e.g. {"source_type": "faq"}
                or {"category": "Refund & Return"} — combine with $and/$or
                per Chroma's filter syntax for multiple conditions

        Returns:
            List of {"id", "text", "metadata", "distance"} dicts, ordered
            by relevance (lowest distance first).
        """
        embedding = self.embedder.embed_query([query])[0]
        return self._search(embedding, k, where)

    def _search(self, embedding, k: int, where: Optional[dict]) -> list:
        """Runs one Chroma query from an already-computed query embedding.
        Split out from retrieve() so retrieve_merged() can embed the query
        ONCE and reuse the vector across both channel searches, instead of
        paying for a second identical embedding request."""
        raw = self.collection.query(
            query_embeddings=[embedding],
            n_results=k,
            where=where,
        )
        ids = raw.get("ids", [[]])[0]
        docs = raw.get("documents", [[]])[0]
        metas = raw.get("metadatas", [[]])[0]
        distances = raw.get("distances", [[]])[0]

        # Chroma returns results already sorted by distance (ascending).
        return [
            {"id": doc_id, "text": doc, "metadata": meta, "distance": dist}
            for doc_id, doc, meta, dist in zip(ids, docs, metas, distances)
        ]

    def retrieve_merged(self, query: str, k: int = DEFAULT_K):
        """Runs FAQ and resolution-template search as two separate
        channels (matching the project spec's parallel FAQ Search / Past
        Ticket Search), merges by distance, and evaluates confidence on
        the single best result.

        Confidence is based on the TOP result's distance and its own
        source_type's threshold — not a blended/averaged confidence across
        both channels — since FAQ and resolution_template distances live
        on different scales (see retrieval_confidence.py) and averaging
        them would produce a number that doesn't correspond to either
        scale's evidence.

        Returns (results, confidence) where confidence is "high" or "low".
        An empty result set returns ([], "low").
        """
        # Embed once, reuse for both channels — one API request per ticket
        # instead of two.
        embedding = self.embedder.embed_query([query])[0]
        faq_results = self._search(embedding, k, {"source_type": "faq"})
        resolution_results = self._search(embedding, k, {"source_type": "resolution_template"})

        merged = sorted(faq_results + resolution_results, key=lambda r: r["distance"])[:k]
        if not merged:
            return merged, "low"

        best = merged[0]
        confidence = classify_confidence(best["distance"], best["metadata"]["source_type"])
        return merged, confidence


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    parser = argparse.ArgumentParser()
    parser.add_argument("--query", required=True)
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    parser.add_argument("--persist-dir", default=None)
    args = parser.parse_args()

    r = Retriever(persist_dir=args.persist_dir)
    results, confidence = r.retrieve_merged(args.query, k=args.k)

    print(f"\nTop {len(results)} results for: \"{args.query}\"  (confidence: {confidence})\n")
    for i, res in enumerate(results, 1):
        print(f"[{i}] dist={res['distance']:.4f} source={res['metadata']['source_type']} id={res['id']}")
        print(f"    {res['text'][:150]}...")
        print(f"    metadata: {res['metadata']}\n")