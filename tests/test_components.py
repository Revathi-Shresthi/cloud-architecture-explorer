from components import KINDS, kinds_in, tag_components


def test_whole_word_keywords_do_not_match_inside_other_words():
    text = "These words go towards standards. A little contract to extract. Weeks of episodes."
    assert kinds_in(text) == set()


def test_stems_and_whole_words_match():
    text = "The RDS database uses a read replica. Load balancers run health checks."
    assert {"database", "load_balancer"} <= kinds_in(text)


def test_one_keyword_hit_is_not_enough():
    assert "cache" not in kinds_in("We use Redis.")


def test_tag_components_has_a_flag_for_every_kind():
    tags = tag_components("redis cache")
    assert set(tags) == {f"comp_{k}" for k in KINDS}
    assert tags["comp_cache"] is True
