"""
Local backend: answers cloud architecture questions with hybrid retrieval + a local LLM.

Run:  uvicorn app:app --reload
Then open http://localhost:8000

Use a smaller chat model on slow computers:
    set CHAT_MODEL=llama3.2:1b        (Windows cmd)
    $env:CHAT_MODEL="llama3.2:1b"     (PowerShell)

Hosted deployments (see Dockerfile) set:
    CHAT_PROVIDER=gemini   RAG answers come from Gemini instead of Ollama (embeddings stay on Ollama)
    DEMO_MODE=1            request limits per visitor and per day, no deletes, capped imports
"""
import json
import os
import threading
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import ollama
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from google.genai import errors, types
from pydantic import BaseModel

import agents
import architecture as architecture_store
import grounding
from architecture import (delete_architecture, describe_failure, describe_flow, failure_impact,
                          get_architecture, kind_of, list_architectures, neighbours, private_boundary,
                          save_architecture)
from components import KINDS
from iac import parse_iac
from retrieval import MODES, Retriever

CHAT_MODEL = os.environ.get("CHAT_MODEL", "llama3.2")
TOP_K = 4
MAX_ANSWER_TOKENS = 350   # caps the answer time on slow computers
MAX_HISTORY_TURNS = 2     # earlier question/answer pairs sent to the model
MAX_IMPORT_BYTES = 1_000_000
LOG_PATH = Path("logs/requests.jsonl")
CHAT_PROVIDER = os.environ.get("CHAT_PROVIDER", "ollama")
DEMO_MODE = os.environ.get("DEMO_MODE") == "1"
MAX_DEMO_IMPORTS = 25

app = FastAPI(title="Cloud Architecture Explorer")
retriever = Retriever()


class RateLimiter:
    """Sliding-window limits for the public demo: requests per visitor per minute, and per day for everyone."""

    def __init__(self, per_minute: int, per_day: int):
        self.per_minute, self.per_day = per_minute, per_day
        self.visitors = {}   # visitor -> deque of request times
        self.day = deque()
        self._lock = threading.Lock()

    def check(self, visitor: str, cost: int = 1) -> None:
        now = time.time()
        with self._lock:
            if len(self.visitors) > 5000:  # forget visitors with no request in the last minute
                self.visitors = {v: q for v, q in self.visitors.items() if q and now - q[-1] < 60}
            recent = self.visitors.setdefault(visitor, deque())
            while recent and now - recent[0] > 60:
                recent.popleft()
            while self.day and now - self.day[0] > 86400:
                self.day.popleft()
            if len(recent) + cost > self.per_minute:
                raise HTTPException(429, f"Too many requests. The demo allows {self.per_minute} per minute. "
                                         "Wait a minute, then try again.")
            if len(self.day) + cost > self.per_day:
                raise HTTPException(429, "The demo has reached its daily limit. Try again tomorrow, "
                                         "or run the project on your own computer.")
            recent.extend([now] * cost)
            self.day.extend([now] * cost)


limiter = RateLimiter(int(os.environ.get("RATE_LIMIT_PER_MINUTE", "6")), int(os.environ.get("DAILY_REQUEST_LIMIT", "300")))


def limit(request: Request, cost: int = 1) -> None:
    """Apply the demo limits (DEMO_MODE only). Behind a proxy the visitor is the first X-Forwarded-For address."""
    if not DEMO_MODE:
        return
    forwarded = request.headers.get("x-forwarded-for", "").split(",")[0].strip()
    limiter.check(forwarded or (request.client.host if request.client else "unknown"), cost)

SYSTEM_PROMPT = (
    "You are a cloud architecture tutor for students and junior engineers. "
    "Answer using the numbered documentation excerpts provided, and cite them like [1] or [2]. "
    "You may also use the architecture diagram connections given to explain dependencies "
    "and what is affected when a component fails. "
    "If the excerpts do not contain the answer, say so instead of guessing. "
    "Do not copy or list the excerpts in your answer; the user already sees them. "
    "Keep answers clear, practical and under 170 words."
)

NO_DOCUMENTS = "No documents are loaded. Run ingest.py first."


class Turn(BaseModel):
    question: str
    answer: str


class AskRequest(BaseModel):
    question: str
    architecture: str = "default"
    component: Optional[str] = None
    mode: str = "hybrid"
    failure: bool = False      # True when the user simulates a failure of `component`
    history: list[Turn] = []   # earlier turns of this conversation, oldest first


class ImportRequest(BaseModel):
    filename: str
    content: str


# ---------- Architectures ----------

def require_architecture(arch_id: str) -> dict:
    arch = get_architecture(arch_id)
    if arch is None:
        raise HTTPException(404, f"Unknown architecture: {arch_id}")
    return arch


@app.get("/api/architectures")
def architectures():
    return list_architectures()


@app.get("/api/architectures/{arch_id}")
def architecture(arch_id: str):
    arch = require_architecture(arch_id)
    return {**arch, "boundary": private_boundary(arch)}


@app.post("/api/architectures/import")
def import_architecture(req: ImportRequest, request: Request):
    limit(request)
    if len(req.content.encode("utf-8")) > MAX_IMPORT_BYTES:
        raise HTTPException(413, "The file is larger than 1 MB.")
    try:
        arch = save_architecture(parse_iac(req.filename, req.content))
    except ValueError as e:
        raise HTTPException(400, str(e))
    if DEMO_MODE:  # keep only the newest imports
        stored = sorted(architecture_store.STORE.glob("*.json"), key=lambda p: p.stat().st_mtime)
        for old in stored[:-MAX_DEMO_IMPORTS]:
            old.unlink()
    return {"id": arch["id"], "name": arch["name"], "source": arch["source"],
            "components": len(arch["components"]), "links": len(arch["links"])}


@app.delete("/api/architectures/{arch_id}")
def remove_architecture(arch_id: str):
    if DEMO_MODE:
        raise HTTPException(403, "Deleting is turned off in the public demo.")
    if not delete_architecture(arch_id):
        raise HTTPException(404, f"Cannot delete architecture: {arch_id}")
    return {"deleted": arch_id}


@app.get("/api/architectures/{arch_id}/impact/{key}")
def impact(arch_id: str, key: str):
    arch = require_architecture(arch_id)
    if key not in arch["components"]:
        raise HTTPException(404, f"Unknown component: {key}")
    return failure_impact(arch, key)


# ---------- Questions ----------

def retrieval_query(req: AskRequest) -> str:
    # A short follow-up ("how do I prevent it?") has too few words to search with;
    # add the previous question so the search keeps the topic.
    if req.history and len(req.question.split()) < 8:
        return f"{req.history[-1].question} {req.question}"
    return req.question


def build_messages(req: AskRequest, arch: dict, hits: list) -> list:
    context = "\n\n".join(
        f"[{i + 1}] ({h['source']}, page {h['page']}) {h['text']}" for i, h in enumerate(hits)
    )
    diagram_note = ""
    if req.component in arch["components"]:
        c = arch["components"][req.component]
        linked = ", ".join(neighbours(arch, req.component)) or "nothing"
        diagram_note = (
            f"The user selected the {c['label']} (a {KINDS[c['kind']]['label']} component) in the architecture "
            f"'{arch['name']}'. In this diagram it connects to: {linked}.\n"
            f"Request flow: {describe_flow(arch)}.\n"
        )
        if req.failure:
            diagram_note += describe_failure(arch, req.component) + (
                "\nExplain the effect on users, how the team detects the failure, "
                "and how to recover from it or prevent it.\n"
            )
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for turn in req.history[-MAX_HISTORY_TURNS:]:
        messages.append({"role": "user", "content": turn.question})
        messages.append({"role": "assistant", "content": turn.answer[:600]})
    messages.append({"role": "user", "content": f"{diagram_note}Question: {req.question}\n\nDocumentation excerpts:\n{context}"})
    return messages


def source_list(hits: list) -> list:
    return [
        {"n": i + 1, "source": h["source"], "page": h["page"], "snippet": h["text"][:200] + "…"}
        for i, h in enumerate(hits)
    ]


def log_request(entry: dict) -> None:
    LOG_PATH.parent.mkdir(exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def chat_stream(messages: list):
    """Yield (text, stats) pairs from the chat model. stats is a dict on the last pair, else None."""
    if CHAT_PROVIDER == "gemini":
        contents = [types.Content(role="model" if m["role"] == "assistant" else "user",
                                  parts=[types.Part.from_text(text=m["content"])]) for m in messages[1:]]
        config = types.GenerateContentConfig(system_instruction=messages[0]["content"], temperature=0.3)
        start, usage, sent = time.perf_counter(), None, False
        models = agents.models_to_try()
        for attempt, model in enumerate(models):
            try:
                for chunk in agents.get_client().models.generate_content_stream(
                        model=model, contents=contents, config=config):
                    usage = chunk.usage_metadata or usage
                    sent = sent or bool(chunk.text)
                    yield chunk.text or "", None
                break
            except errors.APIError as e:
                # Retry only before the first token: the page cannot take back text it has shown.
                wait = None if sent else agents.retry_wait(
                    e, attempt, model, models[attempt + 1] if attempt + 1 < len(models) else None)
                if wait is None:
                    raise
                time.sleep(wait[0])
        tokens = usage.candidates_token_count if usage else None
        if tokens:
            yield "", {"tokens": tokens, "tokens_per_s": round(tokens / (time.perf_counter() - start), 1)}
        return
    stream = ollama.chat(model=CHAT_MODEL, messages=messages, options={"num_predict": MAX_ANSWER_TOKENS}, stream=True)
    for part in stream:
        stats = None
        if part.get("done") and part.get("eval_duration"):
            stats = {"tokens": part["eval_count"],
                     "tokens_per_s": round(part["eval_count"] / (part["eval_duration"] / 1e9), 1)}
        yield part["message"]["content"], stats


def answer_events(req: AskRequest):
    """
    Generate the answer as events:
      {"type": "sources", ...}, {"type": "token", "text"} ..., {"type": "error", "text"}?, {"type": "done", ...}
    The done event carries the metrics and the grounding check.
    """
    start = time.perf_counter()
    ms = lambda: round((time.perf_counter() - start) * 1000)
    model = agents.GEMINI_MODEL if CHAT_PROVIDER == "gemini" else CHAT_MODEL
    metrics = {"engine": "local", "model": model, "mode": req.mode if req.mode in MODES else "hybrid"}
    check = None
    try:
        arch = get_architecture(req.architecture)
        if arch is None:
            raise ValueError(f"Unknown architecture: {req.architecture}")
        hits = retriever.search(retrieval_query(req), kind_of(arch, req.component), mode=metrics["mode"], k=TOP_K)
        metrics["retrieval_ms"] = ms()
        if not hits:
            yield {"type": "token", "text": NO_DOCUMENTS}
            return
        yield {"type": "sources", "sources": source_list(hits)}

        answer = ""
        for piece, stats in chat_stream(build_messages(req, arch, hits)):
            if piece:
                if not answer:
                    metrics["first_token_ms"] = ms()
                answer += piece
                yield {"type": "token", "text": piece}
            if stats:
                metrics.update(stats)

        check = grounding.check(answer, hits, retriever.vectors[[h["index"] for h in hits]])
        metrics["total_ms"] = ms()
        log_request({
            "time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "architecture": req.architecture, "component": req.component, "failure": req.failure,
            "question": req.question, **metrics, "grounding": check and check["score"],
        })
    except Exception as e:  # Ollama stopped, model missing, quota used up, ...
        yield {"type": "error", "text": f"Error: {e}"}
    finally:
        yield {"type": "done", "metrics": metrics, "grounding": check}


@app.post("/api/ask")
def ask(req: AskRequest, request: Request):
    """Complete answer in one response (used by tests and scripts)."""
    limit(request)
    answer, result = "", {"sources": []}
    for ev in answer_events(req):
        if ev["type"] in ("token", "error"):
            answer += ev["text"]
        elif ev["type"] == "sources":
            result["sources"] = ev["sources"]
        elif ev["type"] == "done":
            result.update(metrics=ev["metrics"], grounding=ev["grounding"])
    return {"answer": answer, **result}


@app.post("/api/ask/stream")
def ask_stream(req: AskRequest, request: Request):
    """The same events as newline-delimited JSON, so the page can show each word as the model writes it."""
    limit(request)
    return ndjson(answer_events(req))


# ---------- Agents (Gemini) ----------

def ndjson(events):
    return StreamingResponse((json.dumps(ev) + "\n" for ev in events), media_type="application/x-ndjson")


@app.get("/api/agent/status")
def agent_status():
    return {"available": bool(agents.api_key()), "model": agents.GEMINI_MODEL,
            "chat_provider": CHAT_PROVIDER, "demo": DEMO_MODE}


def agent_answer_events(req: AskRequest):
    """
    Tutor agent. Events: {"type": "step", "agent", "text"} for each tool call, then the same
    "sources", "token" and "done" events as answer_events().
    """
    start = time.perf_counter()
    metrics = {"engine": "agent", "model": agents.GEMINI_MODEL, "mode": req.mode if req.mode in MODES else "hybrid"}
    check = None
    try:
        arch = get_architecture(req.architecture)
        if arch is None:
            raise ValueError(f"Unknown architecture: {req.architecture}")
        toolbox = agents.Toolbox(arch, retriever, metrics["mode"])
        history = [t.model_dump() for t in req.history]
        answer, calls = yield from agents.tutor(req.question, arch, req.component, req.failure, history, toolbox)
        metrics.update(model_calls=calls, tool_calls=toolbox.tool_calls)
        if toolbox.hits:
            yield {"type": "sources", "sources": source_list(toolbox.hits)}
        yield {"type": "token", "text": answer or "The agent did not give an answer. Try again."}
        if toolbox.hits and answer:
            check = grounding.check(answer, toolbox.hits, retriever.vectors[[h["index"] for h in toolbox.hits]])
        metrics["total_ms"] = round((time.perf_counter() - start) * 1000)
        log_request({
            "time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "architecture": req.architecture, "component": req.component, "failure": req.failure,
            "question": req.question, **metrics, "grounding": check and check["score"],
        })
    except Exception as e:  # missing key, quota used up, Ollama stopped (search), ...
        yield {"type": "error", "text": f"Error: {e}"}
    finally:
        yield {"type": "done", "metrics": metrics, "grounding": check}


@app.post("/api/agent/ask/stream")
def agent_ask_stream(req: AskRequest, request: Request):
    limit(request)
    return ndjson(agent_answer_events(req))


@app.post("/api/architectures/{arch_id}/review/stream")
def review_stream(arch_id: str, request: Request):
    """
    Multi-agent architecture review. Events: "step" (progress), "findings" (after each specialist),
    "report" (summary, priorities, all findings, sources), "error"?, "done".
    """
    arch = require_architecture(arch_id)
    limit(request, cost=5)  # a review makes about 20 model calls

    def events():
        start = time.perf_counter()
        try:
            toolbox = agents.Toolbox(arch, retriever)
            report = yield from agents.review(arch, toolbox)
            yield {"type": "report", **report, "sources": source_list(toolbox.hits), "metrics": {
                "model": agents.GEMINI_MODEL, "tool_calls": toolbox.tool_calls,
                "total_ms": round((time.perf_counter() - start) * 1000),
            }}
        except Exception as e:
            yield {"type": "error", "text": f"Error: {e}"}
        finally:
            yield {"type": "done"}

    return ndjson(events())


@app.get("/api/stats")
def stats():
    """Summary of logs/requests.jsonl, per engine and search mode."""
    if not LOG_PATH.exists():
        return {"requests": 0, "by_mode": {}}
    rows = [json.loads(line) for line in LOG_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
    for r in rows:
        r["group"] = f"{r.get('engine', 'local')}/{r['mode']}"
    avg = lambda xs: round(sum(xs) / len(xs), 2) if xs else None
    by_mode = {}
    for group in sorted({r["group"] for r in rows}):
        rs = [r for r in rows if r["group"] == group]
        by_mode[group] = {
            "requests": len(rs),
            "avg_retrieval_ms": avg([r["retrieval_ms"] for r in rs if "retrieval_ms" in r]),
            "avg_first_token_ms": avg([r["first_token_ms"] for r in rs if "first_token_ms" in r]),
            "avg_total_ms": avg([r["total_ms"] for r in rs if "total_ms" in r]),
            "avg_tokens_per_s": avg([r["tokens_per_s"] for r in rs if "tokens_per_s" in r]),
            "avg_grounding": avg([r["grounding"] for r in rs if r.get("grounding") is not None]),
        }
    return {"requests": len(rows), "by_mode": by_mode}


# Serve the frontend (must come after the API routes)
app.mount("/", StaticFiles(directory="static", html=True), name="static")
