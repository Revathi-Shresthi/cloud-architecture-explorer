import chromadb
import pytest

import retrieval
from retrieval import BM25, Retriever

DOCS = [
    "The database uses a read replica and automatic failover. Database backups run every night.",
    "Redis is an in-memory cache. A cache with a short TTL lowers the load on the database.",
    "The load balancer runs health checks and sends traffic only to healthy targets.",
    "Paris is the capital of France.",
]


def test_bm25_ranks_exact_terms_first():
    bm25 = BM25(DOCS)
    assert bm25.search("health checks")[0] == 2
    assert bm25.search("failover replica")[0] == 0
    assert bm25.search("health checks", allowed={0, 1}) == []


@pytest.fixture
def retriever(tmp_path, monkeypatch, fake_embed):
    monkeypatch.setattr(retrieval, "embed_queries", fake_embed)
    client = chromadb.PersistentClient(path=str(tmp_path / "db"))
    col = client.create_collection(retrieval.COLLECTION)
    col.add(
        ids=[f"c{i}" for i in range(len(DOCS))],
        documents=DOCS,
        metadatas=[{"source": "test.pdf", "page": i + 1} for i in range(len(DOCS))],
        embeddings=fake_embed(DOCS).tolist(),
    )
    return Retriever(path=str(tmp_path / "db"))


@pytest.mark.parametrize("mode", ["vector", "bm25", "hybrid"])
def test_each_mode_finds_the_failover_chunk(retriever, mode):
    hits = retriever.search("database failover replica", mode=mode, k=2)
    assert hits[0]["page"] == 1


def test_kind_filter_only_returns_tagged_chunks(retriever):
    hits = retriever.search("load on the database", kind="cache", mode="hybrid", k=4)
    assert [h["page"] for h in hits] == [2]


def test_kind_without_tagged_chunks_searches_everything(retriever):
    hits = retriever.search("France", kind="dns", mode="bm25", k=4)
    assert hits[0]["page"] == 4


def test_empty_collection(tmp_path):
    assert Retriever(path=str(tmp_path / "empty")).search("anything") == []
