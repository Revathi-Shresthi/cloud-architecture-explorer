"""
Grounding check: is each sentence of the answer supported by the excerpts that it cites?

Each sentence is embedded and compared (cosine similarity) with the chunks it cites, or with all
retrieved chunks when it cites none. A low similarity means that the sentence may come from the
model's own knowledge, not from the documents. This needs one batch of embeddings, no extra LLM call.
"""
import re

import numpy as np

from retrieval import embed_queries

# Cosine similarity with nomic-embed-text. Measured on the AWS Well-Architected PDF: off-topic and
# generic sentences scored 0.52-0.59, sentences supported by a chunk scored 0.64-0.72.
SUPPORT_THRESHOLD = 0.62
MIN_WORDS = 6             # shorter fragments ("Here is why:") are not checked
CITE_RE = re.compile(r"\[(\d+)\]")
SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")


def split_sentences(answer: str) -> list:
    parts = (p.strip(" \t-*•#") for p in SPLIT_RE.split(answer))
    return [p for p in parts if len(p.split()) >= MIN_WORDS]


def check(answer: str, hits: list, doc_vectors: np.ndarray) -> dict:
    """
    doc_vectors: unit-length embeddings of the retrieved chunks, in the order of `hits`.
    Returns {"score", "supported", "total", "sentences": [...], "invalid_citations": [...]} or None.
    """
    sentences = split_sentences(answer)
    if not sentences or not hits:
        return None
    sims = embed_queries(sentences) @ doc_vectors.T

    results, invalid = [], set()
    for sentence, row in zip(sentences, sims):
        cited = [int(n) for n in CITE_RE.findall(sentence)]
        valid = [n for n in cited if 1 <= n <= len(hits)]
        invalid.update(n for n in cited if n not in valid)
        columns = [n - 1 for n in valid] or range(len(hits))
        best = max(float(row[c]) for c in columns)
        results.append({
            "text": sentence,
            "cited": valid,
            "similarity": round(best, 3),
            "supported": best >= SUPPORT_THRESHOLD,
        })

    supported = sum(r["supported"] for r in results)
    return {
        "score": round(supported / len(results), 3),
        "supported": supported,
        "total": len(results),
        "sentences": results,
        "invalid_citations": sorted(invalid),
    }
