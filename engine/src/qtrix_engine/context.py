"""Process-wide engine context shared by the API, the agents and the MCP tools."""

from __future__ import annotations

import time
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from typing import Any

from .config import Settings
from .journal import Journal
from .notebooks import Notebook
from .objects.store import ObjectStore
from .panel import PanelState
from .workspace import Workspace

MAIN_THREAD = "main"
DEFAULT_THREAD_TITLE = "Nueva conversación"


@dataclass
class Thread:
    """One conversation of a notebook: its chat and the report it builds. The datasets, objects, journal, kernel and
    memory belong to the notebook, so every conversation of it works on the same data."""

    id: str
    title: str = DEFAULT_THREAD_TITLE
    panel: PanelState = field(default_factory=PanelState)
    chat: Any = None  # agents.lead.ChatSession, set by the engine
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    reporting: bool = False  # reporting mode: the ReportingAgent rewrites ``report`` after every turn
    report: dict[str, Any] | None = None  # agents.reporting.ConversationReport as a dict
    report_version: int = 0

    def to_dict(self) -> dict[str, Any]:
        turns = sum(1 for m in self.chat.transcript if m.get("role") == "user") if self.chat is not None else 0
        return {"id": self.id, "title": self.title, "created_at": self.created_at, "updated_at": self.updated_at,
                "n_turns": turns, "n_blocks": len(self.panel.blocks()), "reporting": self.reporting,
                "report_version": self.report_version}


@dataclass
class EngineContext:
    settings: Settings
    workspace: Workspace
    objects: ObjectStore = field(init=False)
    journal: Journal = field(init=False)
    memory: Any = None  # ProjectMemory | None
    kernels: Any = None  # kernel.KernelPool | None: the project's Python kernel, when the HTTP API runs (E7·07b)
    allow_agent_code: bool = False  # user setting: agents may run Python cells without asking (E7·07b, E7·18)
    notebook: Notebook | None = None  # the notebook this context is the live state of (None: tests, bare MCP)
    #: The notebook's conversations; a request is bound to one (``activate_thread``), else ``thread_id`` serves.
    threads: dict[str, Thread] = field(default_factory=dict)
    thread_id: str = MAIN_THREAD  # the conversation open in the UI: unbound requests (kernel, MCP) act on it
    chat_lock: Any = None  # asyncio.Lock serialising the chat runs of the notebook (one turn at a time, any thread)
    autosave: Any = None  # state.Autosave: writes the chats and the panels into the notebook's database
    last_used: float = 0.0  # time.monotonic() of the last request that opened it (least recently used goes first)

    @property
    def uploads_dir(self):
        return self.notebook.uploads_dir if self.notebook is not None else self.settings.uploads_dir

    def __post_init__(self) -> None:
        self.objects = ObjectStore(self.workspace)
        self.journal = Journal(self.workspace)
        self.objects.subscribe(self._on_object_event)
        if not self.threads:
            self.threads[MAIN_THREAD] = Thread(MAIN_THREAD)

    @property
    def thread(self) -> Thread:
        """The conversation the running request is bound to, else the one open in the UI."""
        bound = _thread.get()
        if bound is not None and bound in self.threads:
            return self.threads[bound]
        return self.threads.get(self.thread_id) or next(iter(self.threads.values()))

    @property
    def panel(self) -> PanelState:
        return self.thread.panel

    @panel.setter
    def panel(self, value: PanelState) -> None:
        self.thread.panel = value

    @property
    def chat(self) -> Any:
        return self.thread.chat

    @chat.setter
    def chat(self, value: Any) -> None:
        self.thread.chat = value

    @property
    def results(self) -> ObjectStore:
        """Compatibility alias of the former ResultStore: ``results.get(id)`` takes names and old ``r_`` ids."""
        return self.objects

    def _on_object_event(self, ev: dict[str, Any]) -> None:
        """Keep the report in step with the objects (ADR 0002 §5): live views of an object that got a new version
        (or was restored by undo) are recomputed in place; frozen blocks of it, and blocks of stale or deleted
        objects, are flagged stale."""
        from .objects.views import refresh

        name = ev["name"]
        for panel in [t.panel for t in self.threads.values()]:  # the objects are shared by every conversation
            if ev["event"] == "version" and ev.get("previous"):
                refresh(panel, name, new_version=True)
            elif ev["event"] == "undo":
                if self.objects.exists(name):
                    refresh(panel, name, new_version=True)
                else:
                    panel.mark_stale(name, True)
            elif ev["event"] in ("stale", "delete"):
                panel.mark_stale(name, True)
            elif ev["event"] == "rename":
                panel.rename_result(ev["old_name"], name)


_current: EngineContext | None = None
_active: ContextVar[EngineContext | None] = ContextVar("qtrix_active_context", default=None)
_thread: ContextVar[str | None] = ContextVar("qtrix_active_thread", default=None)


def set_context(ctx: EngineContext) -> EngineContext:
    """The fallback context: what ``get_context()`` returns outside a request bound to a notebook."""
    global _current
    _current = ctx
    return ctx


def activate(ctx: EngineContext) -> Token:
    """Bind ``ctx`` to the running request/turn (a ContextVar, so threads and tasks started from it inherit it)."""
    return _active.set(ctx)


def activate_thread(thread_id: str | None) -> Token:
    """Bind a conversation of the notebook to the running request/turn (its chat and its panel)."""
    return _thread.set(thread_id)


def active_context() -> EngineContext | None:
    """The context bound to the running request/turn, if any."""
    return _active.get()


def get_context() -> EngineContext:
    ctx = _active.get() or _current
    if ctx is None:
        raise RuntimeError("Engine context not initialised")
    return ctx
