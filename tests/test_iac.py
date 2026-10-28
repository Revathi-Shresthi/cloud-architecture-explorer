from pathlib import Path

import pytest

from architecture import failure_impact
from iac import parse_iac

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


def links_of(arch):
    return {tuple(l) for l in arch["links"]}


def test_terraform_example():
    arch = parse_iac("shop.tf", (EXAMPLES / "terraform" / "shop.tf").read_text())
    assert arch["source"] == "terraform"
    assert len(arch["components"]) == 15
    links = links_of(arch)
    # through helpers: aws_route53_record, aws_lb_listener + aws_lb_target_group, aws_lambda_event_source_mapping
    assert ("aws_route53_zone.shop", "aws_cloudfront_distribution.cdn") in links
    assert ("aws_lb.web", "aws_autoscaling_group.app") in links
    assert ("aws_lambda_function.fulfilment", "aws_sqs_queue.orders") in links
    # through the launch template that the Auto Scaling group owns
    assert ("aws_autoscaling_group.app", "aws_db_instance.orders") in links
    # the app security group protects the instances, not the database the launch template mentions
    assert ("aws_security_group.app", "aws_autoscaling_group.app") in links
    assert ("aws_security_group.app", "aws_db_instance.orders") not in links
    # plumbing (VPC, subnets) never creates links
    assert not any("aws_vpc" in a or "aws_vpc" in b for a, b in links)


def test_terraform_failure_impact():
    arch = parse_iac("shop.tf", (EXAMPLES / "terraform" / "shop.tf").read_text())
    impact = failure_impact(arch, "aws_db_instance.orders")
    assert "aws_autoscaling_group.app" in impact["upstream"]
    assert "aws_lambda_function.fulfilment" in impact["upstream"]


def test_kubernetes_example():
    arch = parse_iac("shop.yaml", (EXAMPLES / "kubernetes" / "shop.yaml").read_text())
    assert arch["source"] == "kubernetes"
    kinds = {k: c["kind"] for k, c in arch["components"].items()}
    assert kinds["statefulset-postgres"] == "database"
    assert kinds["deployment-redis"] == "cache"
    assert kinds["ingress-shop"] == "load_balancer"
    assert kinds["networkpolicy-data-isolation"] == "firewall"
    links = links_of(arch)
    assert ("ingress-shop", "deployment-web") in links
    assert ("deployment-web", "deployment-api") in links          # API_URL env var
    assert ("deployment-api", "statefulset-postgres") in links    # DATABASE_URL env var
    assert ("deployment-worker", "statefulset-minio") in links
    assert ("networkpolicy-data-isolation", "deployment-redis") in links
    assert ("networkpolicy-data-isolation", "deployment-api") not in links


def test_every_component_gets_a_position():
    arch = parse_iac("shop.yaml", (EXAMPLES / "kubernetes" / "shop.yaml").read_text())
    assert all(len(c["pos"]) == 2 for c in arch["components"].values())


@pytest.mark.parametrize("filename, content, message", [
    ("notes.txt", "hello", "Use a Terraform file"),
    ("empty.tf", 'variable "x" {}', "No Terraform resource blocks"),
    ("vpc.tf", 'resource "aws_vpc" "main" { cidr_block = "10.0.0.0/16" }', "no supported cloud components"),
    ("bad.yaml", "a: [1, 2", "not valid"),
    ("config.yaml", "kind: ConfigMap\nmetadata: {name: x}\n", "No Kubernetes workloads"),
])
def test_errors(filename, content, message):
    with pytest.raises(ValueError, match=message):
        parse_iac(filename, content)


def test_braces_inside_strings_do_not_end_a_block():
    tf = '''
    resource "aws_lb" "web" { name = "a } b" }
    resource "aws_db_instance" "db" { tags = { Note = "uses ${aws_lb.web.arn}" } }
    '''
    arch = parse_iac("x.tf", tf)
    assert ("aws_lb.web", "aws_db_instance.db") in links_of(arch)
