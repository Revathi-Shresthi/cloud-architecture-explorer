"""
AI agents on the Gemini API (free key: https://aistudio.google.com/apikey).

Tutor agent: answers one question. It decides which tools to call (document search, architecture
details, failure simulation, critical-component ranking), possibly several times, then answers
with citations to the excerpts it found.

Architecture review: four specialist agents (reliability, security, performance, cost) examine the
architecture with the same tools and return structured findings through a submit_findings tool.
A lead agent then writes the summary. The specialists run one after another to stay inside the
free-tier requests-per-minute limit.

Only the reasoning runs on Gemini. Document embeddings and search still run locally (Ollama + Chroma).

The agent functions are generators: they yield progress events for the page and return their
result, so callers use `result = yield from ...`.
"""
import json
import os
import re
import time
from pathlib import Path

from google import genai
from google.genai import errors, types

from architecture import describe_failure, failure_impact, flow_links, is_observer, label, neighbours
from components import KINDS

MAX_STEPS = 6          # model calls per agent run
MAX_RETRIES = 4        # retries on rate limits (429) and server errors
MAX_WAIT_S = 60
SEVERITIES = ("high", "medium", "low")


def load_env_file(path: str = ".env") -> None:
    """Read KEY=VALUE lines from .env into the environment (existing variables win)."""
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


load_env_file()
# Flash-Lite: answered in about 1 s in tests, with higher free-tier limits. The larger Flash models
# were often overloaded (503) or timed out (504) for free-tier keys.
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.1-flash-lite")
# Used in turn when a model is busy (503) or out of free-tier quota (429); quotas are per model.
GEMINI_FALLBACK_MODELS = [m.strip() for m in os.environ.get(
    "GEMINI_FALLBACK_MODELS", "gemini-3.5-flash-lite,gemini-3-flash-preview").split(",") if m.strip()]
RETRY_CODES = (429, 500, 503, 504)
REQUEST_TIMEOUT_MS = 60_000
_client = None


def api_key():
    return os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")


def get_client():
    global _client
    if _client is None:
        if not api_key():
            raise RuntimeError("GEMINI_API_KEY is not set. Put it in the .env file and restart uvicorn")
        # One attempt per request: generate() does the retries, so the page can show each wait.
        _client = genai.Client(api_key=api_key(), http_options=types.HttpOptions(
            timeout=REQUEST_TIMEOUT_MS, retry_options=types.HttpRetryOptions(attempts=1)))
    return _client


def _retry_delay(error: errors.APIError, attempt: int) -> float:
    # A 429 response usually says how long to wait ("retryDelay": "23s").
    m = re.search(r"retry(?:Delay)?['\"]?\s*[:=]?\s*['\"]?(?:in\s+)?(\d+(?:\.\d+)?)s", str(error), re.IGNORECASE)
    return min(float(m[1]) + 1 if m else 5 * 2 ** attempt, MAX_WAIT_S)


def models_to_try() -> list:
    """The main model, then each fallback, then the main model again: one entry per attempt."""
    models = [GEMINI_MODEL] + [m for m in GEMINI_FALLBACK_MODELS if m != GEMINI_MODEL]
    return [models[i % len(models)] for i in range(max(MAX_RETRIES + 1, len(models)))]


def retry_wait(error: errors.APIError, attempt: int, model: str, next_model: str):
    """None if the error is final; else (seconds to wait, message for the page)."""
    if error.code not in RETRY_CODES or next_model is None:
        return None
    what = "Free-tier rate limit reached" if error.code == 429 else "Gemini is busy"
    if next_model != model:
        return 1.0, f"{what} on {model}. Trying {next_model}."
    delay = _retry_delay(error, attempt)
    return delay, f"{what}. Waiting {delay:.0f} s, then trying again."


def generate(contents: list, config: types.GenerateContentConfig, agent: str):
    """One model call, with retries and fallback models. Yields a progress event before each retry."""
    models = models_to_try()
    for attempt, model in enumerate(models):
        try:
            return get_client().models.generate_content(model=model, contents=contents, config=config)
        except errors.APIError as e:
            wait = retry_wait(e, attempt, model, models[attempt + 1] if attempt + 1 < len(models) else None)
            if wait is None:
                raise
            yield {"type": "step", "agent": agent, "text": wait[1]}
            time.sleep(wait[0])


# ---------- Tools ----------

def _schema(properties: dict, required: list = ()) -> dict:
    return {"type": "object", "properties": properties, "required": list(required)}


KIND_NAMES = list(KINDS)

TOOL_DECLARATIONS = [
    types.FunctionDeclaration(
        name="search_documents",
        description="Search the cloud documentation. Returns numbered excerpts; cite them in the answer "
                    "with their citation label, for example [3]. Search again with other words if the "
                    "excerpts do not answer the question.",
        parameters_json_schema=_schema({
            "query": {"type": "string", "description": "What to look for, in plain words."},
            "component_kind": {"type": "string", "enum": KIND_NAMES,
                               "description": "Optional: only search text about this kind of component."},
        }, ["query"]),
    ),
    types.FunctionDeclaration(
        name="get_architecture",
        description="List every component (key, label, kind, known settings) and every link of the "
                    "current architecture. Links go in the direction of the requests.",
        parameters_json_schema=_schema({}),
    ),
    types.FunctionDeclaration(
        name="get_component",
        description="Details of one component: kind, settings, connected components, and whether "
                    "monitoring observes it.",
        parameters_json_schema=_schema({"key": {"type": "string", "description": "Component key."}}, ["key"]),
    ),
    types.FunctionDeclaration(
        name="simulate_failure",
        description="Find what breaks when a component fails: components whose requests fail, "
                    "components that get no traffic, and components that are no longer observed.",
        parameters_json_schema=_schema({"key": {"type": "string", "description": "Component key."}}, ["key"]),
    ),
    types.FunctionDeclaration(
        name="rank_critical_components",
        description="Rank components by how many others a failure affects, with redundancy hints "
                    "(replicas, multi-AZ, minimum size, parallel components of the same kind).",
        parameters_json_schema=_schema({}),
    ),
]

SUBMIT_FINDINGS = types.FunctionDeclaration(
    name="submit_findings",
    description="Submit the final review findings. Call this exactly once, at the end.",
    parameters_json_schema=_schema({
        "findings": {
            "type": "array",
            "items": _schema({
                "title": {"type": "string"},
                "severity": {"type": "string", "enum": list(SEVERITIES)},
                "components": {"type": "array", "items": {"type": "string"},
                               "description": "Exact component keys."},
                "evidence": {"type": "string", "description": "What in the architecture shows the risk."},
                "recommendation": {"type": "string"},
                "citations": {"type": "array", "items": {"type": "integer"},
                              "description": "Numbers of the excerpts that support the recommendation."},
            }, ["title", "severity", "components", "recommendation"]),
        },
    }, ["findings"]),
)

# Settings that show redundancy, from the Terraform and Kubernetes importers
REDUNDANCY_PROPS = ("multi_az", "replicas", "min_size", "max_size", "num_cache_clusters", "desired_capacity")


class Toolbox:
    """The tools the agents can call, bound to one architecture. Keeps the excerpts it has shown."""

    def __init__(self, arch: dict, retriever, mode: str = "hybrid"):
        self.arch, self.retriever, self.mode = arch, retriever, mode
        self.hits = []        # every chunk shown to an agent; citation number = position + 1
        self._numbers = {}    # chunk index -> citation number
        self.tool_calls = 0

    def call(self, name: str, args: dict) -> dict:
        self.tool_calls += 1
        fn = getattr(self, f"tool_{name}", None)
        if fn is None:
            return {"error": f"Unknown tool: {name}"}
        try:
            return fn(**args)
        except (KeyError, TypeError) as e:
            return {"error": f"Bad arguments for {name}: {e}"}

    def _component(self, key: str) -> dict:
        if key not in self.arch["components"]:
            raise KeyError(f"no component with key '{key}'. Call get_architecture for the keys")
        return self.arch["components"][key]

    def tool_search_documents(self, query: str, component_kind: str = None) -> dict:
        excerpts = []
        for h in self.retriever.search(query, component_kind, mode=self.mode, k=4):
            n = self._numbers.get(h["index"])
            if n is None:
                self.hits.append(h)
                n = self._numbers[h["index"]] = len(self.hits)
            excerpts.append({"citation": f"[{n}]", "source": h["source"], "page": h["page"], "text": h["text"]})
        return {"excerpts": excerpts} if excerpts else {"excerpts": [], "note": "No documents matched."}

    def tool_get_architecture(self) -> dict:
        return {
            "name": self.arch["name"],
            "source": self.arch["source"],
            "components": [
                {"key": k, "label": c["label"], "kind": c["kind"], "settings": c.get("props", {})}
                for k, c in self.arch["components"].items()
            ],
            "links": [{"from": a, "to": b} for a, b in self.arch["links"]],
        }

    def tool_get_component(self, key: str) -> dict:
        c = self._component(key)
        return {
            "key": key, "label": c["label"], "kind": KINDS[c["kind"]]["label"], "settings": c.get("props", {}),
            "sends_requests_to": [label(self.arch, b) for a, b in flow_links(self.arch) if a == key],
            "receives_requests_from": [label(self.arch, a) for a, b in flow_links(self.arch) if b == key],
            "all_connections": neighbours(self.arch, key),
            "observed_by_monitoring": any(b == key and is_observer(self.arch, a) for a, b in self.arch["links"]),
        }

    def tool_simulate_failure(self, key: str) -> dict:
        self._component(key)
        impact = failure_impact(self.arch, key)
        return {
            "requests_fail_at": [label(self.arch, k) for k in impact["upstream"]],
            "no_traffic": [label(self.arch, k) for k in impact["downstream"]],
            "not_observed": [label(self.arch, k) for k in impact["blind"]],
            "summary": describe_failure(self.arch, key),
        }

    def tool_rank_critical_components(self) -> dict:
        flow = flow_links(self.arch)
        senders = {k: frozenset(a for a, b in flow if b == k) for k in self.arch["components"]}
        ranked = []
        for key, c in self.arch["components"].items():
            if c["kind"] == "monitoring":
                continue
            impact = failure_impact(self.arch, key)
            parallel = [k for k, o in self.arch["components"].items()
                        if k != key and o["kind"] == c["kind"] and senders[k] and senders[k] == senders[key]]
            ranked.append({
                "key": key, "label": c["label"],
                "components_affected": len(impact["upstream"]) + len(impact["downstream"]),
                "redundancy_settings": {p: v for p, v in c.get("props", {}).items() if p in REDUNDANCY_PROPS},
                "parallel_components": [label(self.arch, k) for k in parallel],
            })
        ranked.sort(key=lambda r: -r["components_affected"])
        return {"components": ranked[:12]}


def describe_call(name: str, args: dict, result: dict) -> str:
    """One line for the page: what the agent did and what it got."""
    if "error" in result:
        return f"{name} failed: {result['error']}"
    if name == "search_documents":
        where = f" in {args['component_kind']} text" if args.get("component_kind") else ""
        return f"Searched the documents{where} for \"{args.get('query', '')}\": {len(result['excerpts'])} excerpts"
    if name == "get_architecture":
        return f"Read the architecture: {len(result['components'])} components, {len(result['links'])} links"
    if name == "get_component":
        return f"Examined {result['label']}"
    if name == "simulate_failure":
        return (f"Simulated a failure of {args.get('key')}: requests fail at {len(result['requests_fail_at'])} "
                f"components, {len(result['no_traffic'])} get no traffic")
    if name == "rank_critical_components":
        top = result["components"][0]["label"] if result["components"] else "none"
        return f"Ranked the critical components (most critical: {top})"
    return name


# ---------- Agent loop ----------

def run_agent(system: str, contents: list, toolbox: Toolbox, agent: str, declarations: list,
              stop_tool: str = None, max_steps: int = MAX_STEPS):
    """
    Function-calling loop. The model calls tools until it answers in text, or calls `stop_tool`.
    Returns (text, stop_tool_args, model_calls). On the last step, the model must answer
    (or must call `stop_tool`), so the loop always ends.
    """
    contents = list(contents)
    for step in range(max_steps):
        last = step == max_steps - 1
        if last and stop_tool:
            calling = types.FunctionCallingConfig(mode="ANY", allowed_function_names=[stop_tool])
        else:
            calling = types.FunctionCallingConfig(mode="NONE" if last else "AUTO")
        config = types.GenerateContentConfig(
            system_instruction=system,
            tools=[types.Tool(function_declarations=declarations)],
            tool_config=types.ToolConfig(function_calling_config=calling),
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
            temperature=0.3,
        )
        response = yield from generate(contents, config, agent)
        calls = response.function_calls or []
        if not calls:
            return response.text or "", None, step + 1

        contents.append(response.candidates[0].content)
        replies = []
        for call in calls:
            args = dict(call.args or {})
            if call.name == stop_tool:
                return "", args, step + 1
            result = toolbox.call(call.name, args)
            yield {"type": "step", "agent": agent, "tool": call.name, "text": describe_call(call.name, args, result)}
            replies.append(types.Part.from_function_response(name=call.name, response=result))
        contents.append(types.Content(role="user", parts=replies))
    return "", None, max_steps


# ---------- Tutor agent ----------

TUTOR_PROMPT = (
    "You are a cloud architecture tutor for students and junior engineers. You can see the user's "
    "architecture through tools, and you can search the cloud documentation.\n"
    "Work like this: search the documentation for the facts you need (search again with other words when "
    "the excerpts are not useful); use the architecture tools when the question is about this system, its "
    "dependencies, or failures. Then answer.\n"
    "Rules: cite documentation facts with the excerpt labels, like [2]. Use only labels that a search returned. "
    "Facts that come from the architecture tools need no citation. If the documents do not contain the answer, "
    "say so instead of guessing. Do not list the excerpts; the user sees them. Keep the answer under 200 words."
)


def tutor(question: str, arch: dict, component: str, failure: bool, history: list, toolbox: Toolbox):
    """Returns (answer, model_calls)."""
    note = f"Architecture: {arch['name']} ({arch['source']})."
    if component in arch["components"]:
        c = arch["components"][component]
        note += f" The user selected component '{component}': {c['label']}, a {KINDS[c['kind']]['label']}."
        if failure:
            note += " The user is simulating a failure of this component: explain the effect on users, " \
                    "detection, and recovery or prevention."
    contents = []
    for turn in history[-2:]:
        contents.append(types.Content(role="user", parts=[types.Part.from_text(text=turn["question"])]))
        contents.append(types.Content(role="model", parts=[types.Part.from_text(text=turn["answer"][:800])]))
    contents.append(types.Content(role="user", parts=[types.Part.from_text(text=f"{note}\nQuestion: {question}")]))

    text, _, calls = yield from run_agent(TUTOR_PROMPT, contents, toolbox, "Tutor", TOOL_DECLARATIONS)
    return text, calls


# ---------- Architecture review (multi-agent) ----------

PILLARS = {
    "reliability": "single points of failure, redundancy (multi-AZ, replicas, minimum instance counts), failover, "
                   "backups, health checks, monitoring coverage, and the blast radius of failures",
    "security": "edge protection (WAF, DDoS), firewalls or security groups in front of each tier, network "
                "isolation of data stores, components reachable without a firewall, and least privilege",
    "performance": "caching, CDN use, load balancing, automatic scaling, queue-based decoupling, and bottlenecks "
                   "where many components depend on one",
    "cost": "over-provisioning, idle redundancy, storage classes and lifecycle rules, managed versus self-managed "
            "services, and data transfer that a CDN or cache can reduce",
}

REVIEWER_PROMPT = (
    "You are the {pillar} reviewer in a cloud architecture review team. Focus only on {focus}.\n"
    "Steps: 1) call get_architecture; 2) use rank_critical_components, get_component and simulate_failure "
    "where useful; 3) call search_documents to find guidance that supports each recommendation; "
    "4) call submit_findings once with 2 to 4 findings, most important first.\n"
    "Rules: report only risks that the architecture data shows, and give that data as evidence. If a setting is "
    "not in the data, say that it is not visible in the files instead of assuming it is missing. Use exact "
    "component keys. Cite only excerpt numbers that a search returned."
)

LEAD_PROMPT = (
    "You lead a cloud architecture review. Specialist reviewers sent the findings below. Write a summary for a "
    "junior engineer: the overall state of the architecture in 2 to 3 sentences, and the 3 most important actions "
    "in priority order. Use only the findings given."
)

LEAD_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "priorities": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["summary", "priorities"],
}


def clean_findings(raw, arch: dict, toolbox: Toolbox, pillar: str) -> list:
    """Keep well-formed findings; drop unknown component keys and citations that no search returned."""
    out = []
    for f in (raw or {}).get("findings", []) if isinstance(raw, dict) else []:
        if not isinstance(f, dict) or not f.get("title"):
            continue
        out.append({
            "pillar": pillar,
            "title": str(f["title"]),
            "severity": f.get("severity") if f.get("severity") in SEVERITIES else "medium",
            "components": [k for k in f.get("components", []) if k in arch["components"]],
            "evidence": str(f.get("evidence", "")),
            "recommendation": str(f.get("recommendation", "")),
            "citations": sorted({int(n) for n in f.get("citations", [])
                                 if isinstance(n, (int, float)) and 1 <= int(n) <= len(toolbox.hits)}),
        })
    return out


def review(arch: dict, toolbox: Toolbox):
    """Returns {"summary", "priorities", "findings", "model_calls"}."""
    findings, model_calls = [], 0
    for pillar, focus in PILLARS.items():
        agent = pillar.capitalize()
        yield {"type": "step", "agent": agent, "text": f"{agent} reviewer started"}
        prompt = REVIEWER_PROMPT.format(pillar=pillar, focus=focus)
        contents = [types.Content(role="user", parts=[types.Part.from_text(
            text=f"Review the architecture '{arch['name']}' ({arch['source']}).")])]
        _, submitted, calls = yield from run_agent(
            prompt, contents, toolbox, agent, TOOL_DECLARATIONS + [SUBMIT_FINDINGS], stop_tool="submit_findings")
        model_calls += calls
        found = clean_findings(submitted, arch, toolbox, pillar)
        findings += found
        yield {"type": "findings", "agent": agent, "findings": found}

    findings.sort(key=lambda f: SEVERITIES.index(f["severity"]))
    yield {"type": "step", "agent": "Lead", "text": "Lead reviewer is writing the summary"}
    config = types.GenerateContentConfig(
        system_instruction=LEAD_PROMPT, response_mime_type="application/json",
        response_json_schema=LEAD_SCHEMA, temperature=0.3,
    )
    brief = [{k: f[k] for k in ("pillar", "severity", "title", "evidence", "recommendation")} for f in findings]
    response = yield from generate([json.dumps(brief)], config, "Lead")
    model_calls += 1
    try:
        lead = json.loads(response.text)
    except (TypeError, ValueError):
        lead = {"summary": response.text or "", "priorities": []}
    return {"summary": lead.get("summary", ""), "priorities": lead.get("priorities", []),
            "findings": findings, "model_calls": model_calls}
