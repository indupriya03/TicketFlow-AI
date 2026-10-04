"""
gemini_embeddings.py
─────────────────────
ChromaDB-compatible embedding function using Google's gemini-embedding-001.

Ported from BankBot-RAG's src/rag/gemini_embeddings.py — same pattern,
same reasoning: talking to google.genai directly (not through
langchain-google-genai) means the batch size we set here is the batch
size that actually reaches Google's API, with no wrapper layer free to
re-split it underneath us. That opacity is what caused
retrieval_document.py's earlier LangChain-based version to keep hitting
free-tier rate limits regardless of the chunk size we passed it.

Requires GEMINI_API_KEY (or GOOGLE_API_KEY) env var — get one free, no
credit card, at https://aistudio.google.com/apikey

Uses task_type=RETRIEVAL_DOCUMENT when embedding documents going INTO the
store, and RETRIEVAL_QUERY when embedding a user's search query — these
are asymmetric on purpose (Gemini's embedding space is tuned differently
for "this is a searchable passage" vs "this is what someone's looking
for"), which improves retrieval quality over using one embedding mode
for both sides. retrieval_document.py uses get_document_embedder();
retrieval_query.py should use get_query_embedder().
"""
import logging
import os
import re
import time
import yaml
from typing import List
from pathlib import Path
from dotenv import load_dotenv

from google import genai
from google.genai import types

load_dotenv()  # reads .env in the current/parent directory into os.environ

log = logging.getLogger(__name__)


def _load_params() -> dict:
    params_path = Path(__file__).resolve().parents[2] / "params.yaml"
    with open(params_path) as f:
        return yaml.safe_load(f)


_params = _load_params()

MAX_RETRIES = 6
MODEL = _params["rag"]["embedding_model"]
OUTPUT_DIM = _params["rag"]["embedding_output_dim"]  # good quality/storage tradeoff per Google's guidance (768/1536/3072)
BATCH_SIZE = _params["rag"]["embedding_batch_size"]  # Gemini API's documented max batch size per call


class GeminiEmbeddingFunction:
    """Conforms to chromadb's EmbeddingFunction protocol: callable that
    takes a list of strings and returns a list of embedding vectors.
    Implements name()/get_config()/build_from_config() since chromadb
    1.x requires these for its embedding-function registry, even when
    the function instance is passed explicitly rather than reloaded
    from stored collection config."""

    def __init__(self, task_type: str = "RETRIEVAL_DOCUMENT"):
        assert task_type in ("RETRIEVAL_DOCUMENT", "RETRIEVAL_QUERY")
        self.task_type = task_type
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError(
                "Set GEMINI_API_KEY (or GOOGLE_API_KEY) env var — get one free at "
                "https://aistudio.google.com/apikey"
            )
        self.client = genai.Client(api_key=api_key)

    def __call__(self, input: List[str]) -> List[List[float]]:
        all_embeddings = []
        for i in range(0, len(input), BATCH_SIZE):
            batch = input[i:i + BATCH_SIZE]
            all_embeddings.extend(self._embed_batch(batch))
        return all_embeddings

    def embed_query(self, input: List[str]) -> List[List[float]]:
        """chromadb calls this explicitly for query-side embedding rather
        than always falling back to __call__ — not defined by default
        unless subclassing chromadb.EmbeddingFunction, so provide it directly."""
        return self.__call__(input)

    @staticmethod
    def name() -> str:
        return "gemini"

    def default_space(self) -> str:
        return "cosine"

    def supported_spaces(self) -> List[str]:
        return ["cosine", "l2", "ip"]

    def get_config(self) -> dict:
        return {"task_type": self.task_type}

    @staticmethod
    def build_from_config(config: dict) -> "GeminiEmbeddingFunction":
        return GeminiEmbeddingFunction(task_type=config.get("task_type", "RETRIEVAL_DOCUMENT"))

    def _embed_batch(self, batch: List[str]) -> List[List[float]]:
        for attempt in range(MAX_RETRIES):
            try:
                result = self.client.models.embed_content(
                    model=MODEL,
                    contents=batch,
                    config=types.EmbedContentConfig(
                        task_type=self.task_type,
                        output_dimensionality=OUTPUT_DIM,
                    ),
                )
                return [e.values for e in result.embeddings]
            except Exception as e:
                text = str(e)
                # A PerDay quota failure can NEVER succeed by retrying within
                # the same day — Google's daily quota only resets at midnight
                # Pacific. Retrying anyway wastes time and, worse, can trip
                # the separate PER-MINUTE limiter with the retry attempts
                # themselves (observed in practice: 6 blind retries against
                # an exhausted daily cap ended up ALSO hitting the per-minute
                # limit). Fail fast and say so plainly instead.
                if "PerDay" in text:
                    raise RuntimeError(
                        "Gemini's free-tier DAILY request quota is exhausted. "
                        "This will not resolve by retrying — it only resets "
                        "at midnight Pacific Time. Check usage at "
                        "https://ai.dev/rate-limit and try again after the "
                        "reset (or use a different Google Cloud project's key)."
                    ) from e

                # Google's 429 response includes a suggested wait time
                # (e.g. "'retryDelay': '38s'") — use it when present. A
                # blind exponential schedule (2**attempt = 1,2,4,8s) gives
                # up in under 15s total, but rate-limit responses seen
                # against this project's free-tier key have asked for
                # 25-65s — retrying faster than that just burns through
                # MAX_RETRIES without ever actually waiting long enough.
                match = re.search(r"'retryDelay':\s*'(\d+(?:\.\d+)?)s'", text)
                wait = float(match.group(1)) if match else 2 ** attempt
                log.warning(f"Embedding call failed (attempt {attempt+1}/{MAX_RETRIES}): {e}. "
                            f"Retrying in {wait:.0f}s...")
                if attempt == MAX_RETRIES - 1:
                    raise
                time.sleep(wait)


def get_document_embedder() -> GeminiEmbeddingFunction:
    """Use this when embedding documents to STORE in the vector database."""
    return GeminiEmbeddingFunction(task_type="RETRIEVAL_DOCUMENT")


def get_query_embedder() -> GeminiEmbeddingFunction:
    """Use this when embedding a user's QUERY to search the vector database."""
    return GeminiEmbeddingFunction(task_type="RETRIEVAL_QUERY")