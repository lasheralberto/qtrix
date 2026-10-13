"""memory_* tools: semantic recall over the project history."""

from __future__ import annotations

from typing import Any

from ..context import get_context
from .registry import Qx, tool


@tool("memory", requires=("turbovec",), qx=Qx("qx", "recall", ("query",)))
def memory_search(query: str, k: int = 5, kinds: list[str] | None = None) -> list[dict[str, Any]]:
    """Search the project memory (earlier messages, decisions, model results, imported datasets).

    Args:
        query: What to look for, in natural language.
        k: Number of hits.
        kinds: Optional filter among user, assistant, tool, dataset, note.
    """
    memory = get_context().memory
    if memory is None:
        return []
    return [
        {"kind": item.kind, "text": item.text[:800], "when": item.created_at, "score": round(score, 3)}
        for item, score in memory.search(query, k, kinds=kinds, min_score=0.0)
    ]
