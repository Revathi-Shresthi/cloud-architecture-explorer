"""
Component kinds: label, color, request-flow tier, and keywords for tagging document chunks.
Every architecture (the built-in one, and those imported from Terraform or Kubernetes files)
is made of components of these kinds. See architecture.py for the graphs.
"""
import re

# tier: position in the request flow, from the user (0) to the data (5).
# Imported links are oriented from the lower tier to the higher tier.
#
# Keywords match at the start of a word. A keyword that ends with "*" is a stem
# ("load balanc*" matches "load balancer" and "load balancing"); all others must
# match the whole word, so "rds" does not match "words".
KINDS = {
    "dns": {
        "label": "DNS", "color": "#7aa2f7", "tier": 0,
        "keywords": ["dns", "route 53", "domain name*", "hosted zone*", "name server*", "record set*"],
    },
    "cdn": {
        "label": "CDN", "color": "#9ece6a", "tier": 1,
        "keywords": ["cdn", "cloudfront", "content delivery", "edge location*", "edge cach*"],
    },
    "firewall": {
        "label": "Firewall / WAF", "color": "#f7768e", "tier": 2,
        "keywords": ["firewall*", "waf", "security group*", "network acl*", "ddos", "shield", "ingress rule*"],
    },
    "load_balancer": {
        "label": "Load Balancer", "color": "#e0af68", "tier": 3,
        "keywords": ["load balanc*", "elb", "alb", "nlb", "target group*", "health check*", "listener*"],
    },
    "kubernetes": {
        "label": "Kubernetes Cluster", "color": "#2ac3de", "tier": 4,
        "keywords": ["kubernetes", "k8s", "pod", "pods", "node", "nodes", "deployment*", "replicaset*",
                     "kubelet", "container*", "eks"],
    },
    "compute": {
        "label": "Compute", "color": "#a9b1d6", "tier": 4,
        "keywords": ["ec2", "virtual machine*", "lambda", "ecs", "fargate", "auto scaling", "autoscaling",
                     "compute", "server", "servers", "instance type*"],
    },
    "cache": {
        "label": "Cache (Redis)", "color": "#ff9e64", "tier": 5,
        "keywords": ["cache*", "caching", "redis", "memcached", "elasticache", "ttl"],
    },
    "database": {
        "label": "Database", "color": "#bb9af7", "tier": 5,
        "keywords": ["database*", "rds", "aurora", "dynamodb", "replica*", "sql", "backup*", "failover*"],
    },
    "queue": {
        "label": "Message Queue", "color": "#73daca", "tier": 5,
        "keywords": ["queue*", "sqs", "kafka", "messag*", "event*", "pub/sub", "sns", "decoupl*"],
    },
    "storage": {
        "label": "Object Storage (S3)", "color": "#c0caf5", "tier": 5,
        "keywords": ["s3", "object storage", "bucket*", "storage class*", "glacier", "lifecycle*"],
    },
    "monitoring": {
        "label": "Monitoring", "color": "#db4b4b", "tier": 6,
        "keywords": ["monitor*", "cloudwatch", "metric*", "alarm*", "logging", "observab*", "prometheus",
                     "tracing", "trace", "traces"],
    },
}

# Monitoring observes other components; its links are not part of the request flow.
OBSERVER_KIND = "monitoring"

# A chunk is tagged with a kind when its keywords appear at least this many times.
MIN_KEYWORD_HITS = 2


def _pattern(keywords: list) -> re.Pattern:
    parts = []
    for kw in keywords:
        if kw.endswith("*"):
            parts.append(r"\b" + re.escape(kw[:-1]))
        else:
            parts.append(r"\b" + re.escape(kw) + r"\b")
    return re.compile("|".join(parts))


PATTERNS = {kind: _pattern(info["keywords"]) for kind, info in KINDS.items()}


def kinds_in(text: str) -> set:
    """Kinds whose keywords appear at least MIN_KEYWORD_HITS times in `text`."""
    lower = text.lower()
    return {kind for kind, pattern in PATTERNS.items() if len(pattern.findall(lower)) >= MIN_KEYWORD_HITS}


def tag_components(text: str) -> dict:
    """Return metadata like {"comp_database": True, "comp_cache": False, ...}."""
    found = kinds_in(text)
    return {f"comp_{kind}": kind in found for kind in KINDS}
