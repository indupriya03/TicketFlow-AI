"""
Shared Embedder Cache
Multi-Agent Customer Support Intelligence Platform

A single process-wide cache for SentenceTransformer instances, keyed by
model name. Every sub-agent that embeds text (Category, Sentiment, and any
future one using the same frozen embedding model) should call get_embedder()
instead of instantiating SentenceTransformer directly.

Why this exists: Category and Sentiment both use "all-MiniLM-L6-v2". Without
this cache, each sub-agent's own load_*_model() function loads its own
~90MB copy into memory the first time it's called — two copies of an
identical, frozen (never fine-tuned) model sitting in the same process for
no reason. get_embedder("all-MiniLM-L6-v2") called from either sub-agent
returns the SAME object after the first load.
"""

from sentence_transformers import SentenceTransformer

_embedder_cache = {}


def get_embedder(model_name):
    if model_name not in _embedder_cache:
        _embedder_cache[model_name] = SentenceTransformer(model_name)
    return _embedder_cache[model_name]