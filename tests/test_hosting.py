"""Hosted-deployment features: Gemini as the RAG chat model, and the public demo limits."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

import agents
import app as app_module
import grounding

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"
HIT = {"index": 0, "text": "The database uses a replica and automatic failover.", "source": "test.pdf", "page": 7}


class FakeRetriever:
    def __init__(self, vectors):
        self.vectors = vectors

    def search(self, question, kind=None, mode="hybrid", k=4):
        return [HIT]


class FakeGeminiStream:
    def __init__(self):
        self.models = self
        self.kwargs = None

    def generate_content_stream(self, model, contents, config):
        self.kwargs = {"model": model, "contents": contents, "config": config}
        usage = SimpleNamespace(candidates_token_count=12)
        yield SimpleNamespace(text="The database fails over to a replica [1]. ", usage_metadata=None)
        yield SimpleNamespace(text="Users see a short pause during the failover.", usage_metadata=usage)


@pytest.fixture
def client(tmp_path, monkeypatch, fake_embed):
    monkeypatch.setattr(app_module, "retriever", FakeRetriever(fake_embed([HIT["text"]])))
    monkeypatch.setattr(grounding, "embed_queries", fake_embed)
    monkeypatch.setattr(app_module, "LOG_PATH", tmp_path / "requests.jsonl")
    return TestClient(app_module.app)


def test_rag_answers_can_come_from_gemini(client, monkeypatch):
    fake = FakeGeminiStream()
    monkeypatch.setattr(app_module, "CHAT_PROVIDER", "gemini")
    monkeypatch.setattr(agents, "get_client", lambda: fake)
    history = [{"question": "What is failover?", "answer": "Switching to a standby."}]
    r = client.post("/api/ask", json={"question": "What breaks?", "component": "database", "history": history}).json()
    assert r["answer"].startswith("The database fails over")
    assert r["metrics"]["model"] == agents.GEMINI_MODEL
    assert r["metrics"]["tokens"] == 12
    assert [c.role for c in fake.kwargs["contents"]] == ["user", "model", "user"]
    assert "cloud architecture tutor" in fake.kwargs["config"].system_instruction


def test_rate_limiter_per_visitor_and_per_day():
    limiter = app_module.RateLimiter(per_minute=2, per_day=3)
    limiter.check("a")
    limiter.check("a")
    with pytest.raises(HTTPException) as e:
        limiter.check("a")
    assert e.value.status_code == 429 and "per minute" in e.value.detail
    limiter.check("b")
    with pytest.raises(HTTPException) as e:
        limiter.check("c")
    assert "daily limit" in e.value.detail


def test_demo_mode_limits_requests_and_turns_off_delete(client, monkeypatch):
    monkeypatch.setattr(app_module, "DEMO_MODE", True)
    monkeypatch.setattr(app_module, "limiter", app_module.RateLimiter(per_minute=1, per_day=100))
    content = (EXAMPLES / "kubernetes" / "shop.yaml").read_text()
    first = client.post("/api/architectures/import", json={"filename": "a.yaml", "content": content},
                        headers={"X-Forwarded-For": "1.2.3.4, 10.0.0.1"})
    assert first.status_code == 200
    again = client.post("/api/architectures/import", json={"filename": "b.yaml", "content": content},
                        headers={"X-Forwarded-For": "1.2.3.4"})
    assert again.status_code == 429
    other = client.post("/api/architectures/import", json={"filename": "c.yaml", "content": content},
                        headers={"X-Forwarded-For": "5.6.7.8"})
    assert other.status_code == 200
    assert client.delete(f"/api/architectures/{first.json()['id']}").status_code == 403


def test_demo_mode_keeps_only_the_newest_imports(client, monkeypatch):
    monkeypatch.setattr(app_module, "DEMO_MODE", True)
    monkeypatch.setattr(app_module, "MAX_DEMO_IMPORTS", 2)
    monkeypatch.setattr(app_module, "limiter", app_module.RateLimiter(per_minute=100, per_day=100))
    content = (EXAMPLES / "kubernetes" / "shop.yaml").read_text()
    for name in ("one", "two", "three"):
        assert client.post("/api/architectures/import", json={"filename": f"{name}.yaml", "content": content}).status_code == 200
    ids = [a["id"] for a in client.get("/api/architectures").json()]
    assert len(ids) == 3  # default + the 2 newest


def test_status_reports_hosting_settings(client):
    status = client.get("/api/agent/status").json()
    assert set(status) == {"available", "model", "chat_provider", "demo"}
