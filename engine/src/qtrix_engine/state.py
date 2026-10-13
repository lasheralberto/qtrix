"""A notebook's conversation and report survive restarts (E8·02).

The chat (model history, UI transcript, journal cursor) and the report (panel blocks with their undo history) live in
memory while the notebook is open; here they are written to the notebook's own DuckDB file, next to its datasets,
objects and journal. ``Autosave`` coalesces the writes: the panel changes on every operation and holds figures, so
saving each time would cost more than it protects.

``thread_id`` is part of the chat key so a notebook can hold several conversations later; today there is one, ``main``.
"""

from __future__ import annotations

import json
import logging
import threading
from typing import TYPE_CHECKING, Any

from pydantic_ai.messages import ModelMessagesTypeAdapter

if TYPE_CHECKING:
    from .agents.lead import ChatSession
    from .panel import PanelState
    from .workspace import Workspace

log = logging.getLogger(__name__)

MAIN = "main"
#: Undo/redo steps that are kept across a restart (each step is a full copy of the blocks, figures included).
PERSISTED_UNDO = 20

CHAT_DDL = """
CREATE TABLE IF NOT EXISTS _qtrix_chat (
    thread_id  VARCHAR PRIMARY KEY,
    history    VARCHAR,
    transcript VARCHAR,
    seen_seq   BIGINT,
    updated_at TIMESTAMP DEFAULT now()
)"""

PANEL_DDL = """
CREATE TABLE IF NOT EXISTS _qtrix_panel (
    id         INTEGER PRIMARY KEY,
    version    BIGINT,
    blocks     VARCHAR,
    history    VARCHAR,
    redo       VARCHAR,
    updated_at TIMESTAMP DEFAULT now()
)"""


def _dumps(value: Any) -> str:
    return json.dumps(value, default=str, ensure_ascii=False)


class StateStore:
    def __init__(self, workspace: Workspace) -> None:
        self._ws = workspace
        with workspace._lock:
            workspace._con.execute(CHAT_DDL)
            workspace._con.execute(PANEL_DDL)

    def save_chat(self, chat: ChatSession, thread_id: str = MAIN) -> None:
        history = ModelMessagesTypeAdapter.dump_json(chat.history, fallback=str).decode()
        with self._ws._lock:
            self._ws._con.execute(
                "INSERT OR REPLACE INTO _qtrix_chat (thread_id, history, transcript, seen_seq) VALUES (?, ?, ?, ?)",
                [thread_id, history, _dumps(chat.transcript), chat.seen_seq])

    def load_chat(self, chat: ChatSession, thread_id: str = MAIN) -> bool:
        with self._ws._lock:
            row = self._ws._con.execute(
                "SELECT history, transcript, seen_seq FROM _qtrix_chat WHERE thread_id = ?", [thread_id]).fetchone()
        if row is None:
            return False
        chat.restore(ModelMessagesTypeAdapter.validate_json(row[0]), json.loads(row[1]), row[2])
        return True

    def save_panel(self, panel: PanelState) -> None:
        state = panel.export_state(PERSISTED_UNDO)
        with self._ws._lock:
            self._ws._con.execute(
                "INSERT OR REPLACE INTO _qtrix_panel (id, version, blocks, history, redo) VALUES (1, ?, ?, ?, ?)",
                [state["version"], _dumps(state["blocks"]), _dumps(state["history"]), _dumps(state["redo"])])

    def load_panel(self, panel: PanelState) -> bool:
        with self._ws._lock:
            row = self._ws._con.execute(
                "SELECT version, blocks, history, redo FROM _qtrix_panel WHERE id = 1").fetchone()
        if row is None:
            return False
        panel.restore({"version": row[0], "blocks": json.loads(row[1]), "history": json.loads(row[2]),
                       "redo": json.loads(row[3])})
        return True


class Autosave:
    """Loads a notebook's saved state when it opens and writes it back, a moment after each change."""

    def __init__(self, store: StateStore, panel: PanelState, chat: ChatSession, *, delay: float = 0.5) -> None:
        self._store, self._panel, self._chat, self._delay = store, panel, chat, delay
        self._lock = threading.Lock()
        self._dirty: set[str] = set()
        self._timer: threading.Timer | None = None
        self._closed = False

    def load(self) -> None:
        try:
            self._store.load_panel(self._panel)
            self._store.load_chat(self._chat)
        except Exception:  # noqa: BLE001 - an unreadable saved state must not keep the notebook from opening
            log.exception("Could not restore the saved chat/panel of a notebook; starting empty")
        # Hooks go on after loading, so restoring does not trigger a save.
        self._panel.on_change = lambda: self.schedule("panel")
        self._chat.on_change = lambda: self.schedule("chat")

    def schedule(self, what: str) -> None:
        with self._lock:
            if self._closed:
                return
            self._dirty.add(what)
            if self._timer is None:
                self._timer = threading.Timer(self._delay, self.flush)
                self._timer.daemon = True
                self._timer.start()

    def flush(self) -> None:
        """Write what changed, now."""
        with self._lock:
            dirty, self._dirty = self._dirty, set()
            timer, self._timer = self._timer, None
        if timer is not None:
            timer.cancel()
        try:
            if "panel" in dirty:
                self._store.save_panel(self._panel)
            if "chat" in dirty:
                self._store.save_chat(self._chat)
        except Exception:  # noqa: BLE001 - persistence must never break the chat or the panel
            log.exception("Could not save the chat/panel of a notebook")
            with self._lock:
                self._dirty |= dirty

    def close(self, *, save: bool = True) -> None:
        """Stop autosaving (the notebook is closing or being deleted); ``save`` writes the pending changes first."""
        if save:
            self.flush()
        with self._lock:
            self._closed = True
            timer, self._timer = self._timer, None
        if timer is not None:
            timer.cancel()
