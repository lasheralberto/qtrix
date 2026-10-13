"""Project memory: every chat turn, tool outcome and dataset event is embedded locally and indexed
with turbovec, so the lead agent and the specialists can recall earlier decisions semantically.

Embeddings are computed on-device with fastembed (ONNX). The model is downloaded once from the
Hugging Face hub on first use; no user data is ever sent anywhere.
"""

from __future__ import annotations

import json
import logging
import threading
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
EmbedFn = Callable[[Sequence[str]], np.ndarray]


@dataclass
class MemoryItem:
    id: int
    kind: str  # user | assistant | tool | dataset | note | preference (kept across notebooks)
    text: str
    created_at: str


def fastembed_embedder(model_name: str = DEFAULT_EMBEDDING_MODEL) -> tuple[EmbedFn, int]:
    from fastembed import TextEmbedding

    model = TextEmbedding(model_name)
    dim = int(next(iter(model.embed(["probe"]))).shape[0])

    def embed(texts: Sequence[str]) -> np.ndarray:
        vecs = np.asarray(list(model.embed(list(texts))), dtype=np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs / np.where(norms == 0, 1, norms)

    return embed, dim


class ProjectMemory:
    def __init__(self, directory: Path, embed: EmbedFn, dim: int, *, bit_width: int = 4) -> None:
        from turbovec import IdMapIndex

        self.directory = directory
        directory.mkdir(parents=True, exist_ok=True)
        self._embed = embed
        self._index_path = directory / "memory.tvim"
        self._items_path = directory / "items.jsonl"
        self._lock = threading.Lock()
        self.items: dict[int, MemoryItem] = {}
        if self._items_path.is_file():
            for line in self._items_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    item = MemoryItem(**json.loads(line))
                    self.items[item.id] = item
        if self._index_path.is_file():
            self._index = IdMapIndex.load(str(self._index_path))
        else:
            self._index = IdMapIndex(dim, bit_width)
        self._next_id = max(self.items, default=0) + 1

    def __len__(self) -> int:
        return len(self.items)

    def add(self, kind: str, text: str) -> MemoryItem | None:
        text = text.strip()
        if not text:
            return None
        vec = self._embed([text[:2000]])
        with self._lock:
            item = MemoryItem(self._next_id, kind, text[:4000], datetime.now(UTC).isoformat())
            self._next_id += 1
            self._index.add_with_ids(vec, np.array([item.id], dtype=np.uint64))
            self.items[item.id] = item
            with self._items_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(asdict(item), ensure_ascii=False) + "\n")
            self._index.write(str(self._index_path), durable=False)
        return item

    def search(self, query: str, k: int = 5, *, min_score: float = 0.35,
               kinds: Sequence[str] | None = None) -> list[tuple[MemoryItem, float]]:
        if not self.items or not query.strip():
            return []
        allow = None
        if kinds:
            ids = [i for i, it in self.items.items() if it.kind in kinds]
            if not ids:
                return []
            allow = np.array(ids, dtype=np.uint64)
        vec = self._embed([query])
        with self._lock:
            scores, ids = self._index.search(vec, min(k, len(self.items)), allowlist=allow)
        hits = []
        for score, item_id in zip(scores[0], ids[0], strict=True):
            item = self.items.get(int(item_id))
            if item is not None and float(score) >= min_score:
                hits.append((item, float(score)))
        return hits

    def recall(self, query: str, k: int = 5) -> str:
        """Format the most relevant memories for prompt injection."""
        return "\n".join(f"- ({it.kind}, {it.created_at[:16]}) {it.text[:500]}" for it, _ in self.search(query, k))


class NotebookMemory:
    """What the agent remembers inside one notebook, plus what it knows about the user across all of them.

    Everything a notebook's conversation produces goes to the notebook's own index, so a notebook never recalls
    another one's analysis. Only ``preference`` items are written to the user-wide index. A search looks in the
    notebook first and fills up with the user-wide hits.
    """

    def __init__(self, notebook: ProjectMemory, user: ProjectMemory | None = None) -> None:
        self.notebook, self.user = notebook, user

    def __len__(self) -> int:
        return len(self.notebook) + (len(self.user) if self.user is not None else 0)

    def add(self, kind: str, text: str) -> MemoryItem | None:
        target = self.user if kind == "preference" and self.user is not None else self.notebook
        return target.add(kind, text)

    def search(self, query: str, k: int = 5, *, min_score: float = 0.35,
               kinds: Sequence[str] | None = None) -> list[tuple[MemoryItem, float]]:
        hits = self.notebook.search(query, k, min_score=min_score, kinds=kinds)
        if self.user is not None and len(hits) < k:
            hits += self.user.search(query, k - len(hits), min_score=min_score, kinds=kinds)
        return hits

    def recall(self, query: str, k: int = 5) -> str:
        return "\n".join(f"- ({it.kind}, {it.created_at[:16]}) {it.text[:500]}" for it, _ in self.search(query, k))


def open_memory(directory: Path) -> ProjectMemory | None:
    """Open the project memory, or return None when the optional dependencies are unavailable."""
    try:
        embed, dim = fastembed_embedder()
        return ProjectMemory(directory, embed, dim)
    except Exception as exc:  # noqa: BLE001 - memory is an enhancement; the app must run without it
        log.warning("Project memory disabled: %s", exc)
        return None
