"""
Hybrid retrieval: vector search + keyword search (BM25), merged with Reciprocal Rank Fusion.

Vector search finds text with the same meaning ("make it highly available" ~ "multi-AZ failover").
BM25 finds exact terms that embeddings can miss ("RDS", "ALB", "Route 53").
RRF merges the two ranked lists without having to compare their different score scales.

Chroma is the persistent store. The chunks, their embeddings and their kind tags are loaded into
memory once (and again after ingest.py makes a new collection), so filtering by kind and cosine
similarity are plain NumPy operations. Kind tags are computed from the current keywords in
components.py, so a keyword change does not need a new ingest.
"""
import math
import re
import threading
from collections import Counter
from typing import Optional

import chromadb
import numpy as np
import ollama

from components import KINDS, kinds_in

EMBED_MODEL = "nomic-embed-text"
DB_PATH = "db"
COLLECTION = "cloud"
CANDIDATES = 12   # results taken from each search before fusion
RRF_K = 60        # standard RRF constant: lower values favour the top ranks more
MODES = ("hybrid", "vector", "bm25")

TOKEN_RE = re.compile(r"[a-z0-9]+")
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "do", "does", "for", "from", "how", "i",
    "if", "in", "is", "it", "its", "my", "of", "on", "or", "should", "that", "the", "this", "to",
    "we", "what", "when", "which", "why", "with", "you", "your",
}


def tokenize(text: str) -> list:
    return [t for t in TOKEN_RE.findall(text.lower()) if t not in STOPWORDS]


class BM25:
    """Okapi BM25 over an in-memory list of documents."""

    def __init__(self, docs: list, k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.tfs = [Counter(tokenize(d)) for d in docs]
        self.lens = [sum(tf.values()) for tf in self.tfs]
        self.avg_len = sum(self.lens) / len(self.lens) if self.lens else 0.0
        n = len(docs)
        df = Counter(t for tf in self.tfs for t in tf)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def search(self, query: str, allowed: Optional[set] = None, k: int = CANDIDATES) -> list:
        """Return the indexes of the k best documents, best first."""
        terms = tokenize(query)
        scored = []
        for i, tf in enumerate(self.tfs):
            if allowed is not None and i not in allowed:
                continue
            score = 0.0
            for t in terms:
                f = tf.get(t)
                if f:
                    norm = self.k1 * (1 - self.b + self.b * self.lens[i] / self.avg_len)
                    score += self.idf[t] * f * (self.k1 + 1) / (f + norm)
            if score > 0:
                scored.append((score, i))
        scored.sort(reverse=True)
        return [i for _, i in scored[:k]]


def _normalize(m: np.ndarray) -> np.ndarray:
    return m / np.maximum(np.linalg.norm(m, axis=-1, keepdims=True), 1e-12)


def embed_queries(texts: list) -> np.ndarray:
    """Unit-length query embeddings, one row per text."""
    res = ollama.embed(model=EMBED_MODEL, input=[f"search_query: {t}" for t in texts])
    return _normalize(np.asarray(res["embeddings"], dtype=np.float32))


class Retriever:
    def __init__(self, path: str = DB_PATH):
        self.client = chromadb.PersistentClient(path=path)
        self._lock = threading.Lock()
        self._version = None
        self.docs, self.metas, self.kinds = [], [], []
        self.vectors = np.zeros((0, 0), dtype=np.float32)

    def _collection(self):
        # Look up the collection on each call, so a fresh ingest.py run is picked up.
        return self.client.get_or_create_collection(COLLECTION)

    def refresh(self) -> int:
        """Load the collection into memory when ingest.py has made a new one. Returns the chunk count."""
        col = self._collection()
        version = (col.id, col.count())
        with self._lock:
            if version != self._version:
                data = col.get(include=["documents", "metadatas", "embeddings"])
                self.docs, self.metas = data["documents"], data["metadatas"]
                emb = data["embeddings"]
                self.vectors = _normalize(np.asarray(emb, dtype=np.float32)) if len(emb) else np.zeros((0, 0), np.float32)
                self.kinds = [kinds_in(d) for d in self.docs]
                self.bm25 = BM25(self.docs)
                self._version = version
        return len(self.docs)

    def count(self) -> int:
        return self.refresh()

    def _vector(self, question: str, allowed: Optional[set]) -> list:
        sims = self.vectors @ embed_queries([question])[0]
        order = np.argsort(-sims)
        if allowed is not None:
            order = [i for i in order if i in allowed]
        return [int(i) for i in order[:CANDIDATES]]

    def search(self, question: str, kind: Optional[str] = None, mode: str = "hybrid", k: int = 4) -> list:
        """Return up to k hits as dicts: {"index", "text", "source", "page"}."""
        if self.refresh() == 0:
            return []

        allowed = None
        if kind in KINDS:
            allowed = {i for i, found in enumerate(self.kinds) if kind in found} or None
        # No kind, or no chunks tagged with it: search everything

        rankings = []
        if mode in ("hybrid", "vector"):
            rankings.append(self._vector(question, allowed))
        if mode in ("hybrid", "bm25"):
            rankings.append(self.bm25.search(question, allowed))

        fused = Counter()
        for ranking in rankings:
            for rank, i in enumerate(ranking):
                fused[i] += 1 / (RRF_K + rank + 1)
        return [
            {"index": i, "text": self.docs[i], "source": self.metas[i]["source"], "page": self.metas[i]["page"]}
            for i, _ in fused.most_common(k)
        ]
