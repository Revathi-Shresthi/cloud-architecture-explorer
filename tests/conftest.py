import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import architecture  # noqa: E402


@pytest.fixture(autouse=True)
def isolated_store(tmp_path, monkeypatch):
    """Imported architectures go to a temporary folder, never to the real architectures/."""
    monkeypatch.setattr(architecture, "STORE", tmp_path / "architectures")


@pytest.fixture
def fake_embed():
    """Deterministic stand-in for nomic-embed-text: a bag-of-words vector over a small vocabulary."""
    vocab = ["database", "failover", "replica", "cache", "redis", "paris", "france", "load", "balancer", "health"]

    def embed(texts):
        rows = []
        for t in texts:
            words = t.lower().replace(".", " ").split()
            v = np.array([words.count(w) for w in vocab], dtype=np.float32) + 0.01
            rows.append(v / np.linalg.norm(v))
        return np.vstack(rows)

    return embed
