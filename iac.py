"""
Import an architecture from infrastructure as code:
  - Terraform (.tf): resource blocks become components; references between resources become links.
  - Kubernetes manifests (.yaml/.yml): workloads, Ingresses, LoadBalancer Services and NetworkPolicies
    become components; selectors, Ingress backends and service names in env vars become links.

The result is an architecture dict (see architecture.py) with an automatic 3D layout.
"""
import json
import re
from collections import deque
from pathlib import Path

import yaml

from architecture import auto_layout, component
from components import KINDS, OBSERVER_KIND

MAX_COMPONENTS = 80

# ---------- Terraform ----------

# Resource types that are shown as components. All other resources are "helpers": they are not shown,
# but links pass through them (for example an aws_lb_listener connects an aws_lb to a target group).
TERRAFORM_KINDS = {
    "dns": ["aws_route53_zone", "google_dns_managed_zone", "azurerm_dns_zone"],
    "cdn": ["aws_cloudfront_distribution", "azurerm_cdn_profile", "azurerm_cdn_frontdoor_profile",
            "google_compute_backend_bucket"],
    "firewall": ["aws_wafv2_web_acl", "aws_waf_web_acl", "aws_security_group", "aws_network_acl",
                 "google_compute_firewall", "google_compute_security_policy", "azurerm_network_security_group",
                 "azurerm_web_application_firewall_policy"],
    "load_balancer": ["aws_lb", "aws_alb", "aws_elb", "google_compute_url_map", "google_compute_forwarding_rule",
                      "azurerm_lb", "azurerm_application_gateway"],
    "kubernetes": ["aws_eks_cluster", "google_container_cluster", "azurerm_kubernetes_cluster"],
    "compute": ["aws_instance", "aws_autoscaling_group", "aws_ecs_service", "aws_lambda_function",
                "google_compute_instance", "google_compute_instance_group_manager", "google_cloud_run_service",
                "google_cloudfunctions_function", "azurerm_linux_virtual_machine", "azurerm_windows_virtual_machine",
                "azurerm_linux_virtual_machine_scale_set", "azurerm_function_app"],
    "cache": ["aws_elasticache_cluster", "aws_elasticache_replication_group", "google_redis_instance",
              "azurerm_redis_cache"],
    "database": ["aws_db_instance", "aws_rds_cluster", "aws_dynamodb_table", "google_sql_database_instance",
                 "google_spanner_instance", "azurerm_postgresql_flexible_server", "azurerm_mssql_server",
                 "azurerm_cosmosdb_account"],
    "queue": ["aws_sqs_queue", "aws_sns_topic", "aws_kinesis_stream", "aws_msk_cluster", "google_pubsub_topic",
              "azurerm_servicebus_queue", "azurerm_eventhub"],
    "storage": ["aws_s3_bucket", "google_storage_bucket", "azurerm_storage_account"],
    "monitoring": ["aws_cloudwatch_metric_alarm", "aws_cloudwatch_dashboard", "google_monitoring_alert_policy",
                   "azurerm_monitor_metric_alert"],
}
TYPE_TO_KIND = {t: kind for kind, types in TERRAFORM_KINDS.items() for t in types}

# Shared plumbing that almost everything references (network, IAM, keys, certificates).
# Links do not pass through these, or every component would connect to every other one.
PLUMBING_RE = re.compile(
    r"^(aws_(vpc|subnet|internet_gateway|nat_gateway|route|route_table|route_table_association|eip|iam_\w+|"
    r"kms_\w+|acm_\w+|db_subnet_group|elasticache_subnet_group|db_parameter_group|elasticache_parameter_group|"
    r"cloudwatch_log_group|ssm_\w+|secretsmanager_\w+)|"
    r"google_(compute_network|compute_subnetwork|project_\w+|service_account\w*)|"
    r"azurerm_(resource_group|virtual_network|subnet|key_vault\w*)|random_\w+|tls_\w+|null_resource)$"
)

RESOURCE_RE = re.compile(r'^\s*resource\s+"([\w-]+)"\s+"([\w-]+)"\s*\{', re.MULTILINE)
REFERENCE_RE = re.compile(r"\b([a-z][a-z0-9]*_[a-z0-9_]+)\.([A-Za-z_][\w-]*)")
LITERAL_RE = re.compile(r'^\s*([A-Za-z_]\w*)\s*=\s*("[^"$]*"|-?\d+(?:\.\d+)?|true|false)\s*$')
MAX_PROPS = 12


def _block_body(text: str, open_brace: int) -> str:
    """Text between `{` at open_brace and its matching `}`. Braces inside strings and comments are skipped."""
    depth, i, n = 0, open_brace, len(text)
    while i < n:
        ch = text[i]
        if ch == '"':
            i += 1
            while i < n and text[i] != '"':
                i += 2 if text[i] == "\\" else 1
        elif ch == "#" or text.startswith("//", i):
            i = text.find("\n", i)
            if i < 0:
                break
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[open_brace + 1:i]
        i += 1
    return text[open_brace + 1:]


def _literal_props(body: str) -> dict:
    """Top-level `key = literal` settings of a resource block (multi_az = true, min_size = 2, ...)."""
    props, depth = {}, 0
    for line in body.splitlines():
        m = LITERAL_RE.match(line)
        if depth == 0 and m and len(props) < MAX_PROPS:
            props[m[1]] = json.loads(m[2]) if not m[2].startswith('"') else m[2][1:-1]
        depth += line.count("{") - line.count("}")
    return props


def _orient(a: str, b: str, kinds: dict):
    """Direction of the request flow between two components, or None (same tier, or both monitoring)."""
    ka, kb = kinds[a], kinds[b]
    if ka == OBSERVER_KIND or kb == OBSERVER_KIND:
        if ka == kb:
            return None
        return (a, b) if ka == OBSERVER_KIND else (b, a)
    ta, tb = KINDS[ka]["tier"], KINDS[kb]["tier"]
    if ta == tb:
        return None
    return (a, b) if ta < tb else (b, a)


def _connect(kinds: dict, refs: dict) -> list:
    """
    Link components that reference each other directly, or only through helper resources.

    refs[x] is the set of resources that x references. A helper that components reference is "owned"
    by them: an aws_launch_template referenced by an aws_autoscaling_group belongs to that group.
    The search only goes through a helper from one of its owners (or when it has none, like an
    aws_route53_record). From any other component it stops at the owners, so a security group used
    in a launch template links to the Auto Scaling group, not to every database the template mentions.
    """
    adjacency = {x: set() for x in refs}
    for x, targets in refs.items():
        for y in targets:
            adjacency[x].add(y)
            adjacency[y].add(x)
    owners = {h: {p for p in kinds if h in refs[p]} for h in refs if h not in kinds}

    links = set()

    def add(a, b):
        link = _orient(a, b, kinds)
        if link:
            links.add(link)

    for start in kinds:
        seen, queue = {start}, deque([start])
        while queue:
            node = queue.popleft()
            for nxt in adjacency[node]:
                if nxt in seen:
                    continue
                seen.add(nxt)
                if nxt in kinds:
                    add(start, nxt)
                elif PLUMBING_RE.match(nxt.split(".")[0]):
                    continue
                elif owners[nxt] and start not in owners[nxt]:
                    for owner in owners[nxt]:
                        if owner != start:
                            add(start, owner)
                else:
                    queue.append(nxt)
    return [list(l) for l in sorted(links)]


def parse_terraform(text: str) -> tuple:
    bodies = {}
    for m in RESOURCE_RE.finditer(text):
        bodies[f"{m[1]}.{m[2]}"] = _block_body(text, m.end() - 1)
    if not bodies:
        raise ValueError("No Terraform resource blocks were found.")

    refs = {addr: set() for addr in bodies}
    for addr, body in bodies.items():
        for t, name in REFERENCE_RE.findall(body):
            ref = f"{t}.{name}"
            if ref in bodies and ref != addr:
                refs[addr].add(ref)

    kinds = {addr: TYPE_TO_KIND[addr.split(".")[0]] for addr in bodies if addr.split(".")[0] in TYPE_TO_KIND}
    if not kinds:
        raise ValueError("The Terraform file has no supported cloud components (load balancers, databases, ...).")
    components = {}
    for addr, kind in kinds.items():
        components[addr] = component(kind, label=f"{addr.split('.')[1]} ({KINDS[kind]['label']})")
        components[addr]["props"] = {"type": addr.split(".")[0], **_literal_props(bodies[addr])}
    return components, _connect(kinds, refs)


# ---------- Kubernetes ----------

IMAGE_KINDS = [
    ("redis", "cache"), ("memcached", "cache"), ("valkey", "cache"),
    ("postgres", "database"), ("mysql", "database"), ("mariadb", "database"), ("mongo", "database"),
    ("cassandra", "database"), ("cockroach", "database"),
    ("rabbitmq", "queue"), ("kafka", "queue"), ("nats", "queue"),
    ("minio", "storage"),
    ("prometheus", "monitoring"), ("grafana", "monitoring"), ("loki", "monitoring"), ("jaeger", "monitoring"),
    ("otel", "monitoring"), ("opentelemetry", "monitoring"),
]
WORKLOAD_KINDS = {"Deployment", "StatefulSet", "DaemonSet", "Job", "CronJob"}


def _pod_template(doc: dict) -> dict:
    spec = doc.get("spec") or {}
    if doc["kind"] == "CronJob":
        spec = ((spec.get("jobTemplate") or {}).get("spec")) or {}
    return spec.get("template") or {}


def _selects(selector: dict, labels: dict) -> bool:
    return bool(selector) and all(labels.get(k) == v for k, v in selector.items())


def _ingress_services(doc: dict) -> set:
    spec = doc.get("spec") or {}
    backends = [spec.get("defaultBackend") or spec.get("backend") or {}]
    for rule in spec.get("rules") or []:
        for path in ((rule.get("http") or {}).get("paths")) or []:
            backends.append(path.get("backend") or {})
    names = set()
    for b in backends:
        name = (b.get("service") or {}).get("name") or b.get("serviceName")
        if name:
            names.add(name)
    return names


def parse_kubernetes(text: str) -> tuple:
    try:
        docs = []
        for doc in yaml.safe_load_all(text):
            if isinstance(doc, dict) and doc.get("kind") == "List":
                docs.extend(doc.get("items") or [])
            elif isinstance(doc, dict):
                docs.append(doc)
    except yaml.YAMLError as e:
        raise ValueError(f"The YAML file is not valid: {e}") from None
    docs = [d for d in docs if d.get("kind")]

    components, kinds, labels, env_text = {}, {}, {}, {}
    services = {}  # service name -> pod selector
    edges = set()

    for d in docs:
        name = (d.get("metadata") or {}).get("name", "unnamed")
        if d["kind"] in WORKLOAD_KINDS:
            pod = _pod_template(d)
            containers = ((pod.get("spec") or {}).get("containers")) or []
            images = " ".join(str(c.get("image", "")) for c in containers).lower()
            kind = next((k for word, k in IMAGE_KINDS if word in images), "compute")
            key = f"{d['kind'].lower()}-{name}"
            kinds[key] = kind
            labels[key] = ((pod.get("metadata") or {}).get("labels")) or {}
            env_text[key] = json.dumps([{k: c.get(k) for k in ("env", "args", "command")} for c in containers])
            components[key] = component(kind, label=f"{name} ({KINDS[kind]['label']})")
            props = {"type": d["kind"], "image": images}
            if d["kind"] in ("Deployment", "StatefulSet"):
                props["replicas"] = (d.get("spec") or {}).get("replicas", 1)
            components[key]["props"] = props
        elif d["kind"] == "Service":
            services[name] = (d.get("spec") or {}).get("selector") or {}

    def workloads_of(service: str) -> list:
        selector = services.get(service, {})
        return [k for k in labels if _selects(selector, labels[k])]

    for d in docs:
        name = (d.get("metadata") or {}).get("name", "unnamed")
        spec = d.get("spec") or {}
        if d["kind"] == "Ingress":
            key = f"ingress-{name}"
            components[key], kinds[key] = component("load_balancer", label=f"{name} (Ingress)"), "load_balancer"
            for svc in _ingress_services(d):
                edges.update((key, w) for w in workloads_of(svc))
        elif d["kind"] == "Service" and spec.get("type") == "LoadBalancer":
            key = f"service-{name}"
            components[key], kinds[key] = component("load_balancer", label=f"{name} (LoadBalancer)"), "load_balancer"
            edges.update((key, w) for w in workloads_of(name))
        elif d["kind"] == "NetworkPolicy":
            key = f"networkpolicy-{name}"
            components[key], kinds[key] = component("firewall", label=f"{name} (NetworkPolicy)"), "firewall"
            selector = ((spec.get("podSelector") or {}).get("matchLabels")) or {}
            # An empty podSelector selects every pod in the namespace
            edges.update((key, w) for w in labels if not selector or _selects(selector, labels[w]))

    # A workload depends on a service when its env, args or command mention the service name
    for key, text in env_text.items():
        for svc in services:
            if re.search(rf"(?<![\w-]){re.escape(svc)}(?![\w-])", text):
                edges.update((key, w) for w in workloads_of(svc) if w != key)

    # Monitoring workloads observe all application workloads
    for m in (k for k, kind in kinds.items() if kind == OBSERVER_KIND):
        edges.update((m, w) for w, kind in kinds.items() if kind == "compute")

    if not components:
        raise ValueError("No Kubernetes workloads, Ingresses or Services were found.")
    return components, [list(e) for e in sorted(edges)]


# ---------- Entry point ----------

def parse_iac(filename: str, content: str) -> dict:
    """Return a new architecture dict (without an id) from a Terraform or Kubernetes file."""
    suffix = Path(filename).suffix.lower()
    if suffix == ".tf":
        components, links, source = *parse_terraform(content), "terraform"
    elif suffix in (".yaml", ".yml"):
        components, links, source = *parse_kubernetes(content), "kubernetes"
    else:
        raise ValueError("Use a Terraform file (.tf) or a Kubernetes manifest (.yaml or .yml).")
    if len(components) > MAX_COMPONENTS:
        raise ValueError(f"The file has {len(components)} components. The maximum is {MAX_COMPONENTS}.")
    auto_layout(components)
    return {"name": Path(filename).stem, "source": source, "components": components, "links": links}
