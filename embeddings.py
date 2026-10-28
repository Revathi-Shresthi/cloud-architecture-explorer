"""
Text embeddings for the document search and the grounding check.

EMBED_PROVIDER=ollama (default): nomic-embed-text on the local Ollama server. No key needed.
EMBED_PROVIDER=fastembed: bge-small-en-v1.5 (ONNX, about 130 MB) inside the app process. No server,
no key and no daily limit, so the app fits in a small server (Render free: 512 MB).
EMBED_PROVIDER=gemini: gemini-embedding-001 on the Gemini API. The free tier allows only 1000
embeddings per day, and an ingest of 200 pages uses about 360 of them.

The two providers make different vectors, so each has its own Chroma collection. Run ingest.py again
after you change the provider.
"""
import os
import time

import numpy as np
from google.genai import errors, types

import agents

EMBED_PROVIDER = os.environ.get("EMBED_PROVIDER", "ollama")
OLLAMA_MODEL = "nomic-embed-text"
FASTEMBED_MODEL = "BAAI/bge-small-en-v1.5"
GEMINI_EMBED_MODEL = os.environ.get("GEMINI_EMBED_MODEL", "gemini-embedding-001")
GEMINI_DIMENSIONS = 768   # same size as nomic-embed-text; also keeps the in-memory index small
GEMINI_BATCH = 100        # the API maximum per request
COLLECTION = "cloud" if EMBED_PROVIDER == "ollama" else f"cloud-{EMBED_PROVIDER}"

# Cosine similarity at which the grounding check counts a sentence as supported. Measured on the
# AWS Well-Architected PDF. nomic-embed-text: off-topic and generic sentences 0.52-0.59, supported
# sentences 0.64-0.72. gemini-embedding-001: off-topic and generic 0.46-0.66, supported 0.69-0.82.
# bge-small-en-v1.5: off-topic and generic 0.37-0.59, supported 0.66-0.91.
SUPPORT_THRESHOLDS = {"ollama": 0.62, "gemini": 0.68, "fastembed": 0.62}

OLLAMA_PREFIX = {"document": "search_document: ", "query": "search_query: "}
GEMINI_TASK = {"document": "RETRIEVAL_DOCUMENT", "query": "RETRIEVAL_QUERY"}


def _normalize(m: np.ndarray) -> np.ndarray:
    return m / np.maximum(np.linalg.norm(m, axis=-1, keepdims=True), 1e-12)


def _ollama(texts: list, task: str) -> list:
    import ollama
    res = ollama.embed(model=OLLAMA_MODEL, input=[OLLAMA_PREFIX[task] + t for t in texts])
    return res["embeddings"]


_fastembed = None


def _fastembed_model():
    global _fastembed
    if _fastembed is None:
        from fastembed import TextEmbedding
        _fastembed = TextEmbedding(FASTEMBED_MODEL, cache_dir=os.environ.get("FASTEMBED_CACHE_PATH"))
    return _fastembed


def _local(texts: list, task: str) -> list:
    model = _fastembed_model()
    # query_embed adds the instruction prefix that bge models expect for search queries
    return list(model.query_embed(texts) if task == "query" else model.passage_embed(texts))


def _gemini(texts: list, task: str) -> list:
    config = types.EmbedContentConfig(task_type=GEMINI_TASK[task], output_dimensionality=GEMINI_DIMENSIONS)
    rows = []
    for start in range(0, len(texts), GEMINI_BATCH):
        batch = texts[start:start + GEMINI_BATCH]
        for attempt in range(agents.MAX_RETRIES + 1):
            try:
                res = agents.get_client().models.embed_content(model=GEMINI_EMBED_MODEL, contents=batch, config=config)
                break
            except errors.APIError as e:
                if e.code not in (429, 500, 503, 504) or attempt == agents.MAX_RETRIES:
                    raise
                time.sleep(agents._retry_delay(e, attempt))
        rows.extend(e.values for e in res.embeddings)
    return rows


def embed(texts: list, task: str) -> np.ndarray:
    """Unit-length embeddings, one row per text. task: "document" (stored chunks) or "query"."""
    fn = {"gemini": _gemini, "fastembed": _local}.get(EMBED_PROVIDER, _ollama)
    return _normalize(np.asarray(fn(list(texts), task), dtype=np.float32))
