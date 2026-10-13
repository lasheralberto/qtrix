"""Call-scoped context set by the dispatcher (E7·06): who is acting, what is being recorded, replay mode.

Kept in its own tiny module so the object store (lowest layer) and the dispatcher can share it without cycles.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any

USER = "user"
DEFAULT_ACTOR = "agent"  # a tool function called directly (tests, scripts) counts as an agent: locks apply

_actor: ContextVar[str] = ContextVar("qtrix_actor", default=DEFAULT_ACTOR)


def current_actor() -> str:
    return _actor.get()


@contextmanager
def acting_as(actor: str) -> Iterator[None]:
    token = _actor.set(actor)
    try:
        yield
    finally:
        _actor.reset(token)


_cell: ContextVar[str | None] = ContextVar("qtrix_cell", default=None)


def current_cell() -> str | None:
    """The chat or notebook cell the running dispatched call belongs to (a Python cell links its calls by it)."""
    return _cell.get()


@contextmanager
def in_cell(cell_id: str | None) -> Iterator[None]:
    token = _cell.set(cell_id)
    try:
        yield
    finally:
        _cell.reset(token)


@dataclass
class Recording:
    """What one dispatched call touched: before-images of objects and ids of pages, for affected/undo."""

    seq: int | None = None
    images: dict[str, dict[str, Any]] = field(default_factory=dict)  # object name -> before image
    pages: set[str] = field(default_factory=set)  # page ids the call targets
    page_ids: list[str] = field(default_factory=list)  # replay: ids the pages created by this call must get


_recording: ContextVar[Recording | None] = ContextVar("qtrix_recording", default=None)
_replaying: ContextVar[bool] = ContextVar("qtrix_replaying", default=False)


def current_recording() -> Recording | None:
    return _recording.get()


def replaying() -> bool:
    return _replaying.get()


@contextmanager
def recording(rec: Recording, *, replay: bool = False) -> Iterator[Recording]:
    t1, t2 = _recording.set(rec), _replaying.set(replay)
    try:
        yield rec
    finally:
        _recording.reset(t1)
        _replaying.reset(t2)


@dataclass
class Turn:
    """A chat turn: the dispatcher stamps its calls with ``turn_id`` and a per-turn cell number."""

    turn_id: str
    cells: int = 0

    def next_cell(self) -> str:
        self.cells += 1
        return f"{self.turn_id}:{self.cells}"


_turn: ContextVar[Turn | None] = ContextVar("qtrix_turn", default=None)


def start_turn(turn_id: str) -> Turn:
    turn = Turn(turn_id)
    _turn.set(turn)
    return turn


def current_turn() -> Turn | None:
    return _turn.get()
