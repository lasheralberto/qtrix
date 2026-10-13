"""Dynamic analysis panel: the report (ADR 0002 §5).

The panel is a list of typed blocks (regression tables, Plotly figures, dataframes, ...).
Agents and the user mutate it through operations (add/update/remove/pin/reorder); every
operation bumps the version, is appended to an undo history and is broadcast to subscribers
(the UI listens on an SSE stream).

A block with a ``ref`` ({object, view, params}) is a *live view* of a named object: when the object gets a new
version the engine recomputes the view and updates the block in place (``objects.views.refresh``). A block without
``ref`` is a *frozen* snapshot; ``freeze`` turns a live block into one.
"""

from __future__ import annotations

import asyncio
import copy
import threading
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

BlockKind = Literal["regression_table", "plotly_figure", "dataframe", "diagnostics", "markdown", "equation",
                    "python_cell"]


class Block(BaseModel):
    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:8])
    kind: BlockKind
    title: str
    data: dict[str, Any]
    source: str = "agent"  # "user": a note or Python cell the user wrote in the panel (E7·08d); agents never touch it
    pinned: bool = False
    result_id: str | None = None  # name of the object the block shows
    stale: bool = False  # the object was re-estimated or its inputs changed after this block was made
    code: str | None = None
    ref: dict[str, Any] | None = None  # live view: {"object": name, "view": view, "params": {...}}
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())


class PanelOp(BaseModel):
    op: Literal["add", "update", "remove", "pin", "reorder", "reset"]
    version: int
    block: Block | None = None
    block_id: str | None = None
    order: list[str] | None = None


def _guard(block: Block, action: str) -> None:
    """Agents may not change what the user pinned or wrote themselves."""
    if block.pinned:
        raise PinnedBlockError(f"Block '{block.id}' is pinned by the user and cannot be {action}")
    if block.source == "user":
        raise PinnedBlockError(f"Block '{block.id}' was written by the user and cannot be {action}")


class PinnedBlockError(PermissionError):
    pass


class PanelState:
    def __init__(self, max_history: int = 200) -> None:
        self._blocks: list[Block] = []
        self._history: list[list[Block]] = []
        self._redo: list[list[Block]] = []
        self._max_history = max_history
        self.version = 0
        self._lock = threading.RLock()
        self._subscribers: list[tuple[asyncio.AbstractEventLoop, asyncio.Queue[PanelOp]]] = []
        #: Called (without arguments) after every change; the notebook's autosave hangs here (state.py).
        self.on_change: Callable[[], None] | None = None

    # ---- persistence (E8·02) -----------------------------------------------------
    def export_state(self, max_undo: int) -> dict[str, Any]:
        """The report and its last ``max_undo`` undo/redo steps as plain data."""
        def dump(steps: list[list[Block]]) -> list[list[dict[str, Any]]]:
            return [[b.model_dump() for b in step] for step in steps[-max_undo:]]

        with self._lock:
            return {"version": self.version, "blocks": [b.model_dump() for b in self._blocks],
                    "history": dump(self._history), "redo": dump(self._redo)}

    def restore(self, state: dict[str, Any]) -> None:
        """Load what ``export_state`` produced. A cell that was running when the app closed is not running now."""
        def blocks(items: list[dict[str, Any]]) -> list[Block]:
            out = [Block.model_validate(b) for b in items]
            return [b.model_copy(update={"data": {k: v for k, v in b.data.items() if k != "running"}})
                    if b.kind == "python_cell" and b.data.get("running") else b for b in out]

        with self._lock:
            self._blocks = blocks(state["blocks"])
            self._history = [blocks(step) for step in state["history"]]
            self._redo = [blocks(step) for step in state["redo"]]
            self.version = state["version"]

    def has_subscribers(self) -> bool:
        with self._lock:
            return bool(self._subscribers)

    # ---- queries -----------------------------------------------------------------
    def blocks(self) -> list[Block]:
        with self._lock:
            return list(self._blocks)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {"version": self.version, "blocks": [b.model_dump() for b in self._blocks]}

    def get(self, block_id: str) -> Block:
        with self._lock:
            for b in self._blocks:
                if b.id == block_id:
                    return b
        raise KeyError(f"Unknown block '{block_id}'")

    # ---- operations --------------------------------------------------------------
    def add(self, block: Block) -> Block:
        with self._lock:
            self._checkpoint()
            self._blocks.append(block)
            self._emit(PanelOp(op="add", version=self._bump(), block=block))
        return block

    def live(self, ref: dict[str, Any]) -> Block | None:
        """The live block showing exactly this view of this object, if the report has one."""
        with self._lock:
            return next((b for b in self._blocks if b.ref == ref), None)

    def update(self, block_id: str, *, title: str | None = None, data: dict[str, Any] | None = None,
               code: str | None = None, by_agent: bool = True, stale: bool | None = None,
               checkpoint: bool = True) -> Block:
        """``checkpoint=False`` folds this change into the previous undo step (a cell's output after its run)."""
        with self._lock:
            current = self.get(block_id)
            if by_agent:
                _guard(current, "modified")
            if checkpoint:
                self._checkpoint()
            updated = current.model_copy(update={
                "title": title or current.title,
                "data": data if data is not None else current.data,
                "code": code if code is not None else current.code,
                "stale": stale if stale is not None else current.stale,
                "updated_at": datetime.now(UTC).isoformat(),
            })
            self._blocks = [updated if b.id == block_id else b for b in self._blocks]
            self._emit(PanelOp(op="update", version=self._bump(), block=updated))
        return updated

    def remove(self, block_id: str, *, by_agent: bool = True) -> None:
        with self._lock:
            block = self.get(block_id)
            if by_agent:
                _guard(block, "removed")
            self._checkpoint()
            self._blocks = [b for b in self._blocks if b.id != block_id]
            self._emit(PanelOp(op="remove", version=self._bump(), block_id=block_id))

    def freeze(self, block_id: str, *, by_agent: bool = True) -> Block:
        """Turn a live block into a frozen snapshot: it keeps its content and no longer follows its object."""
        with self._lock:
            current = self.get(block_id)
            if by_agent:
                _guard(current, "modified")
            if current.ref is None:
                return current
            self._checkpoint()
            frozen = current.model_copy(update={"ref": None, "updated_at": datetime.now(UTC).isoformat()})
            self._blocks = [frozen if b.id == block_id else b for b in self._blocks]
            self._emit(PanelOp(op="update", version=self._bump(), block=frozen))
        return frozen

    def mark_stale(self, result_id: str, stale: bool = True) -> None:
        """Flag (or unflag) every block of an object; announced as ordinary ``update`` ops."""
        self._retarget(result_id, {"stale": stale})

    def rename_result(self, old: str, new: str) -> None:
        """An object was renamed: its blocks, and the live views that follow it, take the new name."""
        with self._lock:
            for block in [b for b in self._blocks if b.result_id == old or (b.ref or {}).get("object") == old]:
                changes: dict[str, Any] = {"result_id": new if block.result_id == old else block.result_id}
                if block.ref and block.ref.get("object") == old:
                    changes["ref"] = {**block.ref, "object": new}
                updated = block.model_copy(update=changes)
                self._blocks = [updated if b.id == block.id else b for b in self._blocks]
                self._emit(PanelOp(op="update", version=self._bump(), block=updated))

    def _retarget(self, result_id: str, changes: dict[str, Any]) -> None:
        with self._lock:
            for block in [b for b in self._blocks if b.result_id == result_id]:
                if all(getattr(block, k) == v for k, v in changes.items()):
                    continue
                updated = block.model_copy(update=changes)
                self._blocks = [updated if b.id == block.id else b for b in self._blocks]
                self._emit(PanelOp(op="update", version=self._bump(), block=updated))

    def pin(self, block_id: str, pinned: bool = True) -> Block:
        with self._lock:
            block = self.get(block_id).model_copy(update={"pinned": pinned})
            self._blocks = [block if b.id == block_id else b for b in self._blocks]
            self._emit(PanelOp(op="pin", version=self._bump(), block=block))
        return block

    def reorder(self, order: list[str]) -> None:
        with self._lock:
            by_id = {b.id: b for b in self._blocks}
            if set(order) != set(by_id) or len(order) != len(by_id):
                raise ValueError("Reorder must contain exactly the current block ids")
            self._checkpoint()
            self._blocks = [by_id[i] for i in order]
            self._emit(PanelOp(op="reorder", version=self._bump(), order=order))

    def undo(self) -> bool:
        with self._lock:
            if not self._history:
                return False
            self._redo.append(copy.deepcopy(self._blocks))
            self._blocks = self._history.pop()
            self._emit_reset()
            return True

    def redo(self) -> bool:
        with self._lock:
            if not self._redo:
                return False
            self._history.append(copy.deepcopy(self._blocks))
            self._blocks = self._redo.pop()
            self._emit_reset()
            return True

    def clear(self) -> None:
        with self._lock:
            self._checkpoint()
            self._blocks = [b for b in self._blocks if b.pinned]
            self._emit_reset()

    # ---- pub/sub -----------------------------------------------------------------
    def subscribe(self) -> asyncio.Queue[PanelOp]:
        queue: asyncio.Queue[PanelOp] = asyncio.Queue()
        with self._lock:
            self._subscribers.append((asyncio.get_running_loop(), queue))
        return queue

    def unsubscribe(self, queue: asyncio.Queue[PanelOp]) -> None:
        with self._lock:
            self._subscribers = [(lp, q) for lp, q in self._subscribers if q is not queue]

    # ---- internals ---------------------------------------------------------------
    def _checkpoint(self) -> None:
        self._history.append(copy.deepcopy(self._blocks))
        self._history = self._history[-self._max_history:]
        self._redo.clear()

    def _bump(self) -> int:
        self.version += 1
        return self.version

    def _emit_reset(self) -> None:
        self._emit(PanelOp(op="reset", version=self._bump()))

    def _emit(self, op: PanelOp) -> None:
        if self.on_change is not None:
            self.on_change()
        # Tools run in worker threads, so hand the op to each subscriber's own event loop.
        for loop, queue in list(self._subscribers):
            if not loop.is_closed():
                loop.call_soon_threadsafe(queue.put_nowait, op)
