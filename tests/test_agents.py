import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from google.genai import errors, types

import agents
import app as app_module
import grounding
from architecture import DEFAULT
from iac import parse_iac

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"
HITS = [
    {"index": 0, "text": "Deploy the database across Availability Zones with failover.", "source": "a.pdf", "page": 3},
    {"index": 1, "text": "A cache lowers the load on the database.", "source": "a.pdf", "page": 9},
]


class FakeRetriever:
    def __init__(self, vectors=None):
        self.vectors = vectors
        self.calls = 0

    def search(self, question, kind=None, mode="hybrid", k=4):
        self.calls += 1
        return HITS


def call(name, **args):
    return SimpleNamespace(name=name, args=args)


def reply(calls=None, text=""):
    content = types.Content(role="model", parts=[types.Part.from_text(text=text or "thinking")])
    return SimpleNamespace(function_calls=calls, text=text, candidates=[SimpleNamespace(content=content)])


class FakeGemini:
    """Plays back scripted responses; an Exception in the script is raised instead."""

    def __init__(self, script):
        self.script, self.configs = list(script), []
        self.models = self

    def generate_content(self, model, contents, config):
        self.configs.append(config)
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def run(gen):
    """Drive an agent generator: return (events, result)."""
    events = []
    while True:
        try:
            events.append(next(gen))
        except StopIteration as stop:
            return events, stop.value


@pytest.fixture
def gemini(monkeypatch):
    def install(script):
        fake = FakeGemini(script)
        monkeypatch.setattr(agents, "get_client", lambda: fake)
        return fake
    return install


def test_tutor_uses_tools_then_answers(gemini):
    fake = gemini([
        reply([call("search_documents", query="database failover", component_kind="database"),
               call("simulate_failure", key="database")]),
        reply(text="Use Multi-AZ failover [1]. The Kubernetes Cluster loses its data."),
    ])
    toolbox = agents.Toolbox(DEFAULT, FakeRetriever())
    events, (answer, calls) = run(agents.tutor("What if it fails?", DEFAULT, "database", True, [], toolbox))
    assert answer.startswith("Use Multi-AZ failover [1]")
    assert calls == 2
    assert [e["tool"] for e in events] == ["search_documents", "simulate_failure"]
    assert "2 excerpts" in events[0]["text"]
    assert len(toolbox.hits) == 2
    assert fake.script == []


def test_citation_numbers_are_stable_across_searches(gemini):
    toolbox = agents.Toolbox(DEFAULT, FakeRetriever())
    first = toolbox.call("search_documents", {"query": "a"})
    second = toolbox.call("search_documents", {"query": "b"})
    assert [e["citation"] for e in first["excerpts"]] == ["[1]", "[2]"]
    assert [e["citation"] for e in second["excerpts"]] == ["[1]", "[2]"]
    assert len(toolbox.hits) == 2


def test_tool_errors_go_back_to_the_model():
    toolbox = agents.Toolbox(DEFAULT, FakeRetriever())
    assert "no component" in toolbox.call("get_component", {"key": "nope"})["error"]
    assert "Unknown tool" in toolbox.call("delete_everything", {})["error"]


def test_last_step_forces_a_text_answer(gemini):
    fake = gemini([reply([call("get_architecture")]), reply(text="Done.")])
    toolbox = agents.Toolbox(DEFAULT, FakeRetriever())
    _, (text, _, calls) = run(agents.run_agent("sys", ["q"], toolbox, "T", agents.TOOL_DECLARATIONS, max_steps=2))
    assert text == "Done."
    assert fake.configs[0].tool_config.function_calling_config.mode == "AUTO"
    assert fake.configs[1].tool_config.function_calling_config.mode == "NONE"


def test_rate_limit_is_retried_after_the_given_delay(gemini, monkeypatch):
    waits = []
    monkeypatch.setattr(agents.time, "sleep", waits.append)
    error = errors.ClientError(429, {"error": {"code": 429, "message": "Quota exceeded", "status": "RESOURCE_EXHAUSTED",
                                               "details": [{"retryDelay": "7s"}]}})
    gemini([error, reply(text="OK")])
    events, result = run(agents.generate(["q"], types.GenerateContentConfig(), "Tutor"))
    assert result.text == "OK"
    assert waits == [8.0]
    assert "rate limit" in events[0]["text"]


def test_critical_components_in_the_terraform_example():
    arch = parse_iac("shop.tf", (EXAMPLES / "terraform" / "shop.tf").read_text())
    ranked = agents.Toolbox(arch, FakeRetriever()).tool_rank_critical_components()["components"]
    by_key = {r["key"]: r for r in ranked}
    assert by_key["aws_db_instance.orders"]["redundancy_settings"] == {"multi_az": True}
    assert by_key["aws_autoscaling_group.app"]["redundancy_settings"] == {"min_size": 2, "max_size": 6}
    assert ranked[0]["components_affected"] >= ranked[-1]["components_affected"]


def test_review_runs_four_specialists_and_cleans_their_findings(gemini):
    script = []
    for pillar in agents.PILLARS:
        script.append(reply([call("get_architecture"), call("search_documents", query=pillar)]))
        script.append(reply([call("submit_findings", findings=[{
            "title": f"{pillar} risk", "severity": "critical",           # unknown severity -> medium
            "components": ["database", "not-a-key"], "recommendation": "Fix it.",
            "citations": [1, 99],                                        # 99 was never returned
        }])]))
    script.append(reply(text=json.dumps({"summary": "Mostly fine.", "priorities": ["Add replicas"]})))
    gemini(script)

    toolbox = agents.Toolbox(DEFAULT, FakeRetriever())
    events, report = run(agents.review(DEFAULT, toolbox))
    assert [e["agent"] for e in events if e["type"] == "findings"] == ["Reliability", "Security", "Performance", "Cost"]
    assert len(report["findings"]) == 4
    f = report["findings"][0]
    assert f["severity"] == "medium" and f["components"] == ["database"] and f["citations"] == [1]
    assert report["summary"] == "Mostly fine."
    assert report["model_calls"] == 9


def test_agent_endpoint_streams_steps_sources_and_grounding(gemini, monkeypatch, tmp_path, fake_embed):
    gemini([
        reply([call("search_documents", query="database failover")]),
        reply(text="Deploy the database across zones with failover for the database [1]."),
    ])
    monkeypatch.setattr(app_module, "retriever", FakeRetriever(fake_embed([h["text"] for h in HITS])))
    monkeypatch.setattr(grounding, "embed_queries", fake_embed)
    monkeypatch.setattr(app_module, "LOG_PATH", tmp_path / "requests.jsonl")
    client = TestClient(app_module.app)
    with client.stream("POST", "/api/agent/ask/stream", json={"question": "How?", "component": "database"}) as r:
        evs = [json.loads(line) for line in r.iter_lines() if line]
    assert [e["type"] for e in evs] == ["step", "sources", "token", "done"]
    assert evs[-1]["metrics"]["engine"] == "agent"
    assert evs[-1]["metrics"]["model_calls"] == 2
    assert evs[-1]["grounding"]["total"] == 1


def test_missing_key_gives_a_clear_error(monkeypatch):
    monkeypatch.setattr(agents, "_client", None)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
        agents.get_client()
