import grounding

HITS = [
    {"text": "The database uses a replica and automatic failover."},
    {"text": "Redis cache lowers the load."},
]


def test_split_sentences_skips_short_fragments():
    text = "Here is why:\n- The database fails over to a replica in another zone.\nOK. It works well in most cases."
    assert grounding.split_sentences(text) == [
        "The database fails over to a replica in another zone.",
        "It works well in most cases.",
    ]


def test_supported_and_unsupported_sentences(monkeypatch, fake_embed):
    monkeypatch.setattr(grounding, "embed_queries", fake_embed)
    answer = ("The database has a replica and failover for the database [1]. "
              "Paris is the capital city of France and it is famous [2].")
    result = grounding.check(answer, HITS, fake_embed([h["text"] for h in HITS]))
    assert result["total"] == 2
    assert [s["supported"] for s in result["sentences"]] == [True, False]
    assert result["score"] == 0.5


def test_citations_that_do_not_exist_are_reported(monkeypatch, fake_embed):
    monkeypatch.setattr(grounding, "embed_queries", fake_embed)
    answer = "The database uses failover with a replica for safety [7]."
    result = grounding.check(answer, HITS, fake_embed([h["text"] for h in HITS]))
    assert result["invalid_citations"] == [7]


def test_no_sentences_means_no_check(fake_embed):
    assert grounding.check("Yes.", HITS, fake_embed(["x", "y"])) is None
