"""
Architectures as graphs.

An architecture is a dict:
    {"id", "name", "source",
     "components": {key: {"kind", "label", "color", "pos": [x, z], "size"?: [w, h, d]}},
     "links": [[a, b], ...]}            # a -> b: a sends requests to b
The built-in architecture is DEFAULT. Imported ones (see iac.py) are saved as JSON in architectures/.
"""
import json
import re
from collections import deque
from pathlib import Path

from components import KINDS, OBSERVER_KIND

STORE = Path("architectures")
ID_RE = re.compile(r"[a-z0-9-]{1,60}")


def component(kind: str, pos=None, label: str = None, size=None) -> dict:
    c = {"kind": kind, "label": label or KINDS[kind]["label"], "color": KINDS[kind]["color"], "pos": pos}
    if size:
        c["size"] = size
    return c


DEFAULT = {
    "id": "default",
    "name": "Reference web architecture",
    "source": "built-in",
    "components": {
        "dns": component("dns", [-6, 0]),
        "cdn": component("cdn", [-4, 0]),
        "firewall": component("firewall", [-2, 0]),
        "load_balancer": component("load_balancer", [0, 0]),
        "kubernetes": component("kubernetes", [2.4, 0], size=[1.6, 1.2, 1.6]),
        "cache": component("cache", [5, -2]),
        "database": component("database", [5, 0]),
        "queue": component("queue", [5, 2]),
        "storage": component("storage", [2.4, 2.8]),
        "monitoring": component("monitoring", [0, -2.8]),
    },
    "links": [
        ["dns", "cdn"],
        ["cdn", "firewall"],
        ["firewall", "load_balancer"],
        ["load_balancer", "kubernetes"],
        ["kubernetes", "cache"],
        ["kubernetes", "database"],
        ["kubernetes", "queue"],
        ["kubernetes", "storage"],
        ["monitoring", "load_balancer"],
        ["monitoring", "kubernetes"],
        ["monitoring", "database"],
    ],
}


# ---------- Storage ----------

def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:50] or "architecture"


def list_architectures() -> list:
    items = [{"id": DEFAULT["id"], "name": DEFAULT["name"], "source": DEFAULT["source"]}]
    for path in sorted(STORE.glob("*.json"), key=lambda p: p.stem):
        data = json.loads(path.read_text(encoding="utf-8"))
        items.append({"id": data["id"], "name": data["name"], "source": data["source"]})
    return items


def get_architecture(arch_id: str):
    if arch_id in (None, "", DEFAULT["id"]):
        return DEFAULT
    if not ID_RE.fullmatch(arch_id):  # also blocks path traversal
        return None
    path = STORE / f"{arch_id}.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def save_architecture(arch: dict) -> dict:
    STORE.mkdir(exist_ok=True)
    base = slug(arch["name"])
    arch_id, n = base, 2
    while arch_id == DEFAULT["id"] or (STORE / f"{arch_id}.json").exists():
        arch_id, n = f"{base}-{n}", n + 1
    arch["id"] = arch_id
    (STORE / f"{arch_id}.json").write_text(json.dumps(arch, indent=2), encoding="utf-8")
    return arch


def delete_architecture(arch_id: str) -> bool:
    if arch_id == DEFAULT["id"] or not ID_RE.fullmatch(arch_id):
        return False
    path = STORE / f"{arch_id}.json"
    if not path.exists():
        return False
    path.unlink()
    return True


# ---------- Layout ----------

def auto_layout(components: dict) -> None:
    """Place components in columns by tier (left to right); monitoring goes in a row at the front."""
    columns, observers = {}, []
    for key, c in components.items():
        if c["kind"] == OBSERVER_KIND:
            observers.append(key)
        else:
            columns.setdefault(KINDS[c["kind"]]["tier"], []).append(key)
    tiers = sorted(columns)
    min_z = 0.0
    for x_i, tier in enumerate(tiers):
        keys = sorted(columns[tier])
        for i, key in enumerate(keys):
            x = (x_i - (len(tiers) - 1) / 2) * 2.8
            z = (i - (len(keys) - 1) / 2) * 1.8
            components[key]["pos"] = [round(x, 2), round(z, 2)]
            min_z = min(min_z, z)
    for i, key in enumerate(sorted(observers)):
        x = (i - (len(observers) - 1) / 2) * 2.4
        components[key]["pos"] = [round(x, 2), round(min_z - 2.6, 2)]


def private_boundary(arch: dict):
    """[x_min, x_max, z_min, z_max] around the components behind the edge (firewall tier and later)."""
    points = [c["pos"] for c in arch["components"].values() if KINDS[c["kind"]]["tier"] >= KINDS["firewall"]["tier"]]
    if not points:
        return None
    pad = 0.9
    xs, zs = [p[0] for p in points], [p[1] for p in points]
    return [min(xs) - pad, max(xs) + pad, min(zs) - pad, max(zs) + pad]


# ---------- Graph queries ----------

def label(arch: dict, key: str) -> str:
    return arch["components"][key]["label"]


def kind_of(arch: dict, key: str):
    c = arch["components"].get(key) if key else None
    return c["kind"] if c else None


def is_observer(arch: dict, key: str) -> bool:
    return arch["components"][key]["kind"] == OBSERVER_KIND


def neighbours(arch: dict, key: str) -> list:
    """Labels of the components directly connected to `key`."""
    out = []
    for a, b in arch["links"]:
        if a == key:
            out.append(label(arch, b))
        elif b == key:
            out.append(label(arch, a))
    return out


def flow_links(arch: dict) -> list:
    return [(a, b) for a, b in arch["links"] if not (is_observer(arch, a) or is_observer(arch, b))]


def describe_flow(arch: dict) -> str:
    return "; ".join(f"{label(arch, a)} -> {label(arch, b)}" for a, b in flow_links(arch)) or "no request flow links"


def _reach(start: str, edges: list) -> list:
    """Breadth-first search: every component reachable from `start` along directed edges."""
    seen, queue = [], deque([start])
    while queue:
        node = queue.popleft()
        for a, b in edges:
            if a == node and b != start and b not in seen:
                seen.append(b)
                queue.append(b)
    return seen


def failure_impact(arch: dict, key: str) -> dict:
    """
    Effects of a failed component, from the request flow:
      upstream:   components whose requests pass through `key`, so those requests now fail.
      downstream: components that get no new requests, because traffic cannot get past `key`.
      blind:      components that are no longer observed (only when a monitoring component fails).
    """
    flow = flow_links(arch)
    return {
        "upstream": _reach(key, [(b, a) for a, b in flow]),
        "downstream": _reach(key, flow),
        "blind": [b for a, b in arch["links"] if a == key] if is_observer(arch, key) else [],
    }


def describe_failure(arch: dict, key: str) -> str:
    """Plain-text summary of failure_impact(), for the LLM prompt."""
    impact = failure_impact(arch, key)
    names = lambda keys: ", ".join(label(arch, k) for k in keys) or "none"
    observed = any(b == key and is_observer(arch, a) for a, b in arch["links"])
    lines = [
        f"FAILURE SIMULATION: the {label(arch, key)} is down.",
        f"These components still run, but user requests that pass through them now fail with errors: "
        f"{names(impact['upstream'])}.",
        f"These components still run, but they get no new requests: {names(impact['downstream'])}.",
    ]
    if impact["blind"]:
        lines.append(f"These components are no longer observed, so failures there are not detected: {names(impact['blind'])}.")
    elif observed:
        lines.append("Monitoring observes this component, so it can raise an alarm.")
    return "\n".join(lines)
