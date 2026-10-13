"""A notebook's conversations and their reports survive restarts (E8·02).

A notebook holds several conversations (threads) that share its datasets, objects, journal and kernel. Each one has
its chat (model history, UI transcript, journal cursor) and its report (panel blocks with their undo history); they
live in memory while the notebook is open and are written to the notebook's own DuckDB file, next to its datasets,
objects and journal. ``Autosave`` coalesces the writes: the panel changes on every operation and holds figures, so
saving each time would cost more than it protects.

Notebooks from before threads have one chat row (``main``) and one panel row (``_qtrix_panel``): they become the
notebook's first conversation.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from pydantic_ai.messages import ModelMessagesTypeAdapter

from .context import DEFAULT_THREAD_TITLE, MAIN_THREAD, Thread

if TYPE_CHECKING:
    from .agents.lead import ChatSession
    from .context import EngineContext
    from .panel import PanelState
    from .workspace import Workspace

log = logging.getLogger(__name__)

MAIN = MAIN_THREAD
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

#: Before threads: the single report of the notebook (read once, to adopt it as the first thread's panel).
LEGACY_PANEL_DDL = """
CREATE TABLE IF NOT EXISTS _qtrix_panel (
    id         INTEGER PRIMARY KEY,
    version    BIGINT,
    blocks     VARCHAR,
    history    VARCHAR,
    redo       VARCHAR,
    updated_at TIMESTAMP DEFAULT now()
)"""

PANELS_DDL = """
CREATE TABLE IF NOT EXISTS _qtrix_panels (
    thread_id  VARCHAR PRIMARY KEY,
    version    BIGINT,
    blocks     VARCHAR,
    history    VARCHAR,
    redo       VARCHAR,
    updated_at TIMESTAMP DEFAULT now()
)"""

THREADS_DDL = """
CREATE TABLE IF NOT EXISTS _qtrix_threads (
    thread_id  VARCHAR PRIMARY KEY,
    title      VARCHAR,
    created_at DOUBLE,
    updated_at DOUBLE,
    deleted_at DOUBLE
)"""


def _dumps(value: Any) -> str:
    return json.dumps(value, default=str, ensure_ascii=False)


def title_from(text: str) -> str:
    """A conversation nobody named takes its name from its first question."""
    return " ".join(text.split())[:48] or DEFAULT_THREAD_TITLE


class StateStore:
    def __init__(self, workspace: Workspace) -> None:
        self._ws = workspace
        with workspace._lock:
            for ddl in (CHAT_DDL, LEGACY_PANEL_DDL, PANELS_DDL, THREADS_DDL):
                workspace._con.execute(ddl)

    # ---- threads --------------------------------------------------------------------------------------------
    def threads(self) -> list[dict[str, Any]]:
        """The notebook's conversations (not deleted), oldest first."""
        with self._ws._lock:
            rows = self._ws._con.execute(
                "SELECT thread_id, title, created_at, updated_at FROM _qtrix_threads WHERE deleted_at IS NULL "
                "ORDER BY created_at, thread_id").fetchall()
        return [{"id": r[0], "title": r[1], "created_at": r[2], "updated_at": r[3]} for r in rows]

    def save_thread(self, thread: Thread) -> None:
        with self._ws._lock:
            self._ws._con.execute(
                "INSERT OR REPLACE INTO _qtrix_threads (thread_id, title, created_at, updated_at, deleted_at) "
                "VALUES (?, ?, ?, ?, NULL)", [thread.id, thread.title, thread.created_at, thread.updated_at])

    def delete_thread(self, thread_id: str) -> None:
        """Soft delete: the conversation leaves the list; its chat and panel rows stay in the file."""
        with self._ws._lock:
            self._ws._con.execute("UPDATE _qtrix_threads SET deleted_at = ? WHERE thread_id = ?",
                                  [time.time(), thread_id])

    # ---- chat and panel -------------------------------------------------------------------------------------
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

    def save_panel(self, panel: PanelState, thread_id: str = MAIN) -> None:
        state = panel.export_state(PERSISTED_UNDO)
        with self._ws._lock:
            self._ws._con.execute(
                "INSERT OR REPLACE INTO _qtrix_panels (thread_id, version, blocks, history, redo) "
                "VALUES (?, ?, ?, ?, ?)",
                [thread_id, state["version"], _dumps(state["blocks"]), _dumps(state["history"]),
                 _dumps(state["redo"])])

    def load_panel(self, panel: PanelState, thread_id: str = MAIN) -> bool:
        with self._ws._lock:
            row = self._ws._con.execute(
                "SELECT version, blocks, history, redo FROM _qtrix_panels WHERE thread_id = ?", [thread_id]).fetchone()
            if row is None and thread_id == MAIN:
                row = self._ws._con.execute(
                    "SELECT version, blocks, history, redo FROM _qtrix_panel WHERE id = 1").fetchone()
        if row is None:
            return False
        panel.restore({"version": row[0], "blocks": json.loads(row[1]), "history": json.loads(row[2]),
                       "redo": json.loads(row[3])})
        return True


class Autosave:
    """Loads a notebook's saved conversations when it opens and writes each back, a moment after it changes."""

    def __init__(self, store: StateStore, ctx: EngineContext, new_chat: Callable[[], ChatSession], *,
                 delay: float = 0.5) -> None:
        self._store, self._ctx, self._new_chat, self._delay = store, ctx, new_chat, delay
        self._lock = threading.Lock()
        self._dirty: set[tuple[str, str]] = set()  # (what, thread_id)
        self._timer: threading.Timer | None = None
        self._closed = False

    def load(self) -> None:
        try:
            metas = self._store.threads()
        except Exception:  # noqa: BLE001 - an unreadable saved state must not keep the notebook from opening
            log.exception("Could not read the conversations of a notebook; starting with an empty one")
            metas = []
        self._ctx.threads.clear()
        for meta in metas:
            self._restore(Thread(meta["id"], meta["title"], created_at=meta["created_at"],
                                 updated_at=meta["updated_at"]))
        if not self._ctx.threads:  # a new notebook, or one saved before threads: its conversation is the first
            thread = self._restore(Thread(MAIN))
            first = next((m["content"] for m in thread.chat.transcript if m.get("role") == "user"), "")
            if first:
                thread.title = title_from(first)
            self.save_meta(thread)
        if self._ctx.thread_id not in self._ctx.threads:
            self._ctx.thread_id = max(self._ctx.threads.values(), key=lambda t: t.updated_at).id

    def add(self, thread: Thread) -> Thread:
        """A new conversation: give it a chat, start saving it and record it."""
        thread.chat = self._new_chat()
        self._ctx.threads[thread.id] = thread
        self.watch(thread)
        self.save_meta(thread)
        return thread

    def _restore(self, thread: Thread) -> Thread:
        thread.chat = self._new_chat()
        try:
            self._store.load_panel(thread.panel, thread.id)
            self._store.load_chat(thread.chat, thread.id)
        except Exception:  # noqa: BLE001 - one unreadable conversation must not keep the notebook from opening
            log.exception("Could not restore the conversation %s of a notebook; starting it empty", thread.id)
        self._ctx.threads[thread.id] = thread
        self.watch(thread)
        return thread

    def watch(self, thread: Thread) -> None:
        """Save the thread a moment after its panel or its chat changes (hooked after loading, so restoring does
        not trigger a save)."""
        thread.panel.on_change = lambda: self.schedule("panel", thread.id)

        def chat_changed() -> None:
            thread.updated_at = time.time()
            self.schedule("chat", thread.id)

        thread.chat.on_change = chat_changed

    def save_meta(self, thread: Thread) -> None:
        try:
            self._store.save_thread(thread)
        except Exception:  # noqa: BLE001 - persistence must never break the chat or the panel
            log.exception("Could not save the conversation %s of a notebook", thread.id)

    def remove(self, thread_id: str) -> None:
        """The conversation was deleted: drop its pending writes and take it off the list."""
        with self._lock:
            self._dirty = {d for d in self._dirty if d[1] != thread_id}
        self._store.delete_thread(thread_id)

    def schedule(self, what: str, thread_id: str = MAIN) -> None:
        with self._lock:
            if self._closed:
                return
            self._dirty.add((what, thread_id))
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
        failed: set[tuple[str, str]] = set()
        for what, thread_id in sorted(dirty):
            thread = self._ctx.threads.get(thread_id)
            if thread is None:
                continue
            try:
                if what == "panel":
                    self._store.save_panel(thread.panel, thread_id)
                else:
                    self._store.save_chat(thread.chat, thread_id)
                    self._store.save_thread(thread)
            except Exception:  # noqa: BLE001 - persistence must never break the chat or the panel
                log.exception("Could not save the %s of conversation %s", what, thread_id)
                failed.add((what, thread_id))
        if failed:
            with self._lock:
                self._dirty |= failed

    def close(self, *, save: bool = True) -> None:
        """Stop autosaving (the notebook is closing or being deleted); ``save`` writes the pending changes first."""
        if save:
            self.flush()
        with self._lock:
            self._closed = True
            timer, self._timer = self._timer, None
        if timer is not None:
            timer.cancel()
