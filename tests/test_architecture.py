import copy

import architecture as A


def test_database_failure_breaks_the_request_path_but_nothing_downstream():
    impact = A.failure_impact(A.DEFAULT, "database")
    assert impact["upstream"] == ["kubernetes", "load_balancer", "firewall", "cdn", "dns"]
    assert impact["downstream"] == []
    assert impact["blind"] == []


def test_kubernetes_failure_stops_traffic_to_the_data_tier():
    impact = A.failure_impact(A.DEFAULT, "kubernetes")
    assert set(impact["downstream"]) == {"cache", "database", "queue", "storage"}
    assert "monitoring" not in impact["upstream"]


def test_monitoring_failure_only_makes_components_blind():
    impact = A.failure_impact(A.DEFAULT, "monitoring")
    assert impact["upstream"] == [] and impact["downstream"] == []
    assert set(impact["blind"]) == {"load_balancer", "kubernetes", "database"}


def test_describe_failure_mentions_alarm_for_observed_component():
    text = A.describe_failure(A.DEFAULT, "database")
    assert "Database is down" in text
    assert "raise an alarm" in text


def test_save_get_delete_round_trip():
    arch = copy.deepcopy(A.DEFAULT)
    arch["name"] = "My Shop"
    first = A.save_architecture(copy.deepcopy(arch))
    second = A.save_architecture(copy.deepcopy(arch))
    assert first["id"] == "my-shop"
    assert second["id"] == "my-shop-2"
    assert A.get_architecture("my-shop")["name"] == "My Shop"
    assert [a["id"] for a in A.list_architectures()] == ["default", "my-shop", "my-shop-2"]
    assert A.delete_architecture("my-shop")
    assert A.get_architecture("my-shop") is None


def test_ids_cannot_escape_the_store():
    assert A.get_architecture("../app") is None
    assert A.get_architecture("..%2Fapp") is None
    assert not A.delete_architecture("default")


def test_auto_layout_puts_tiers_left_to_right():
    comps = {
        "db": A.component("database"),
        "lb": A.component("load_balancer"),
        "mon": A.component("monitoring"),
    }
    A.auto_layout(comps)
    assert comps["lb"]["pos"][0] < comps["db"]["pos"][0]
    assert comps["mon"]["pos"][1] < min(comps["lb"]["pos"][1], comps["db"]["pos"][1])
