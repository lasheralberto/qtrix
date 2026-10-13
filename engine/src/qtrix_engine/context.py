"""Process-wide engine context shared by the API, the agents and the MCP tools."""

from __future__ import annotations

from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from typing import Any

from .config import Settings
from .journal import Journal
from .notebooks import Notebook
from .objects.store import ObjectStore
from .panel import PanelState
from .workspace import Workspace


@dataclass
class EngineContext:
    settings: Settings
    workspace: Workspace
    panel: PanelState = field(default_factory=PanelState)
    objects: ObjectStore = field(init=False)
    journal: Journal = field(init=False)
    memory: Any = None  # ProjectMemory | None
    kernels: Any = None  # kernel.KernelPool | None: the project's Python kernel, when the HTTP API runs (E7·07b)
    allow_agent_code: bool = False  # user setting: agents may run Python cells without asking (E7·07b, E7·18)
    notebook: Notebook | None = None  # the notebook this context is the live state of (None: tests, bare MCP)
    chat: Any = None  # agents.lead.ChatSession of the notebook, set by the engine
    chat_lock: Any = None  # asyncio.Lock serialising the chat runs of the notebook
    autosave: Any = None  # state.Autosave: writes the chat and the panel into the notebook's database
    last_used: float = 0.0  # time.monotonic() of the last request that opened it (least recently used goes first)

    @property
    def uploads_dir(self):
        return self.notebook.uploads_dir if self.notebook is not None else self.settings.uploads_dir

    def __post_init__(self) -> None:
        self.objects = ObjectStore(self.workspace)
        self.journal = Journal(self.workspace)
        self.objects.subscribe(self._on_object_event)

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
        if ev["event"] == "version" and ev.get("previous"):
            refresh(self, name, new_version=True)
        elif ev["event"] == "undo":
            if self.objects.exists(name):
                refresh(self, name, new_version=True)
            else:
                self.panel.mark_stale(name, True)
        elif ev["event"] in ("stale", "delete"):
            self.panel.mark_stale(name, True)
        elif ev["event"] == "rename":
            self.panel.rename_result(ev["old_name"], name)


_current: EngineContext | None = None
_active: ContextVar[EngineContext | None] = ContextVar("qtrix_active_context", default=None)


def set_context(ctx: EngineContext) -> EngineContext:
    """The fallback context: what ``get_context()`` returns outside a request bound to a notebook."""
    global _current
    _current = ctx
    return ctx


def activate(ctx: EngineContext) -> Token:
    """Bind ``ctx`` to the running request/turn (a ContextVar, so threads and tasks started from it inherit it)."""
    return _active.set(ctx)


def active_context() -> EngineContext | None:
    """The context bound to the running request/turn, if any."""
    return _active.get()


def get_context() -> EngineContext:
    ctx = _active.get() or _current
    if ctx is None:
        raise RuntimeError("Engine context not initialised")
    return ctx
