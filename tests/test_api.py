import json
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

import app as app_module
import grounding

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"
HIT = {"index": 0, "text": "The database uses a replica and automatic failover.", "source": "test.pdf", "page": 7}


class FakeRetriever:
    def __init__(self, vectors):
        self.vectors = vectors
        self.calls = []

    def search(self, question, kind=None, mode="hybrid", k=4):
        self.calls.append({"question": question, "kind": kind, "mode": mode})
        return [HIT]


@pytest.fixture
def client(tmp_path, monkeypatch, fake_embed):
    fake = FakeRetriever(fake_embed([HIT["text"]]))
    captured = {}

    def fake_chat(model, messages, options, stream):
        captured["messages"] = messages
        yield {"message": {"content": "The database fails over to a replica [1]. "}, "done": False}
        yield {"message": {"content": "Users see a short pause during failover of the database."}, "done": False}
        yield {"message": {"content": ""}, "done": True, "eval_count": 20, "eval_duration": 2_000_000_000}

    monkeypatch.setattr(app_module, "retriever", fake)
    monkeypatch.setattr(app_module.ollama, "chat", fake_chat)
    monkeypatch.setattr(grounding, "embed_queries", fake_embed)
    monkeypatch.setattr(app_module, "LOG_PATH", tmp_path / "logs" / "requests.jsonl")
    c = TestClient(app_module.app)
    c.fake, c.captured = fake, captured
    return c


def events(response):
    return [json.loads(line) for line in response.iter_lines() if line]


def test_stream_event_order_metrics_and_grounding(client):
    with client.stream("POST", "/api/ask/stream", json={"question": "What breaks?", "component": "database"}) as r:
        evs = events(r)
    assert [e["type"] for e in evs] == ["sources", "token", "token", "done"]  # the empty last piece is not sent
    done = evs[-1]
    assert done["metrics"]["tokens_per_s"] == 10.0
    assert done["grounding"]["total"] == 2
    assert client.fake.calls[0]["kind"] == "database"


def test_failure_simulation_reaches_the_prompt(client):
    client.post("/api/ask", json={"question": "What happens?", "component": "database", "failure": True})
    prompt = client.captured["messages"][-1]["content"]
    assert "FAILURE SIMULATION: the Database is down." in prompt


def test_history_is_sent_and_short_follow_ups_keep_the_topic(client):
    history = [{"question": "How does database failover work?", "answer": "It promotes a replica."}]
    client.post("/api/ask", json={"question": "How do I test it?", "history": history})
    roles = [m["role"] for m in client.captured["messages"]]
    assert roles == ["system", "user", "assistant", "user"]
    assert client.fake.calls[0]["question"] == "How does database failover work? How do I test it?"


def test_stats_summarise_the_log(client):
    client.post("/api/ask", json={"question": "What breaks?", "mode": "bm25"})
    stats = client.get("/api/stats").json()
    assert stats["requests"] == 1
    assert stats["by_mode"]["local/bm25"]["avg_tokens_per_s"] == 10.0


def test_import_select_and_delete_an_architecture(client):
    content = (EXAMPLES / "kubernetes" / "shop.yaml").read_text()
    r = client.post("/api/architectures/import", json={"filename": "shop.yaml", "content": content})
    assert r.status_code == 200
    arch_id = r.json()["id"]
    arch = client.get(f"/api/architectures/{arch_id}").json()
    assert arch["boundary"] is not None
    impact = client.get(f"/api/architectures/{arch_id}/impact/statefulset-postgres").json()
    assert "deployment-api" in impact["upstream"]

    client.post("/api/ask", json={"question": "What is it?", "architecture": arch_id, "component": "statefulset-postgres"})
    assert client.fake.calls[-1]["kind"] == "database"

    assert client.delete(f"/api/architectures/{arch_id}").status_code == 200
    assert client.get(f"/api/architectures/{arch_id}").status_code == 404


def test_bad_import_and_unknown_ids(client):
    assert client.post("/api/architectures/import", json={"filename": "x.txt", "content": "x"}).status_code == 400
    assert client.get("/api/architectures/nope").status_code == 404
    assert client.get("/api/architectures/default/impact/nope").status_code == 404
    assert client.delete("/api/architectures/default").status_code == 404


def test_unknown_architecture_in_a_question_gives_an_error_event(client):
    with client.stream("POST", "/api/ask/stream", json={"question": "x", "architecture": "nope"}) as r:
        evs = events(r)
    assert [e["type"] for e in evs] == ["error", "done"]
