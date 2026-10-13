import hashlib

import numpy as np

from qtrix_engine.memory import ProjectMemory


def fake_embed(texts):
    """Bag-of-words hashing embedder: deterministic, no model download."""
    out = np.zeros((len(texts), 64), dtype=np.float32)
    for i, text in enumerate(texts):
        for word in text.lower().split():
            out[i, int(hashlib.md5(word.encode()).hexdigest(), 16) % 64] += 1
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    return out / np.where(norms == 0, 1, norms)


def test_add_search_and_persist(tmp_path):
    mem = ProjectMemory(tmp_path, fake_embed, 64)
    mem.add("user", "we want the returns to education with clustered errors")
    mem.add("tool", "panel fixed effects regression of wages on union membership")
    mem.add("dataset", "imported gdp quarterly series for spain")
    hits = mem.search("returns to education", k=1, min_score=0)
    assert hits[0][0].kind == "user"
    assert mem.search("spain gdp", k=3, kinds=["dataset"], min_score=0)[0][0].kind == "dataset"

    reopened = ProjectMemory(tmp_path, fake_embed, 64)
    assert len(reopened) == 3
    assert "union" in reopened.recall("union membership wages")
