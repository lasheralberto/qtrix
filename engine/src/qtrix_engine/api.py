"""HTTP API of the qtrix engine (bound to 127.0.0.1 only, protected by a per-session token)."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import secrets
import shutil
import threading
import time
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse, Response, StreamingResponse
from pydantic import BaseModel, Field
from pydantic_ai import Agent

from . import __version__
from .agents.autocompact import context_window
from .agents.lead import ChatSession, TeamMember, build_lead_agent, build_team_agent, reporter_of
from .agents.planner import roster_text
from .agents.reporting import ConversationReport, turns_of, update_report
from .agents.skills import list_skills
from .config import LAYOUT, Settings, bootstrap_dir, resolve_home
from .context import (
    DEFAULT_THREAD_TITLE,
    EngineContext,
    Thread,
    activate,
    activate_thread,
    active_context,
    set_context,
)
from .dispatch import Call, PolicyDenied, dispatch
from .journal import JournalEntryNotFound
from .kernel import CellFailed, KernelError, KernelPool
from .kernel.bridge import dispatch_from_kernel
from .llm.gateway import GatewayError, LLMGateway, model_id
from .llm.providers import (
    ModelProfile,
    ProviderBlocked,
    SettingsStore,
    TeamAgentConfig,
    UserSettings,
    build_model,
    detect_local_models,
    get_api_key,
    has_api_key,
    list_cloud_models,
    set_api_key,
)
from .localai import hub
from .localai.jobs import JobManager
from .localai.llamacpp import VARIANTS, LlamaRuntime, platform_key
from .localai.models import ModelStore
from .memory import NotebookMemory, ProjectMemory, fastembed_embedder
from .methods import methods_catalog
from .notebooks import NotebookManager, NotebookNotFound
from .objects.store import LockedObjectError, ObjectError, ObjectNotFound, ObjectStore
from .objects.views import compute as compute_view
from .panel import Block, PinnedBlockError
from .qx import NotExpressible, QxError, parse_cell, print_call, print_object
from .qx.check import check_call
from .qx.code import run_cell
from .qx.forms import names as qx_names
from .qx.forms import signatures as qx_signatures
from .qx.forms import tool_help as qx_tool_help
from .replay import ReplayError, UndoError, replay
from .state import Autosave, StateStore, title_from
from .tools._format import clean, df_block_data
from .tools.registry import NAMESPACES, all_tools, catalog
from .workspace import DatasetNotFound, Workspace

log = logging.getLogger(__name__)

#: Notebooks kept open at once; the others are saved and closed (see Engine._evict).
MAX_OPEN = 8
#: A notebook's Python kernel is stopped after this many idle seconds (it restarts on the next cell).
KERNEL_IDLE = 1800.0


def _in_use(ctx: EngineContext) -> bool:
    return (ctx.chat_lock.locked() or ctx.kernels.project.busy
            or any(t.panel.has_subscribers() for t in ctx.threads.values()))


_NOTEBOOK_ROUTE = re.compile(r"^/notebooks/([^/]+)(/.*)$")
_NOTEBOOK_OWN = ("/open", "/duplicate")  # routes of the notebook entity itself, not of its content
_THREAD_ROUTE = re.compile(r"^/threads/([^/]+)(/.*)$")
_THREAD_OWN = ("/open",)


class ThreadNotFound(KeyError):
    def __str__(self) -> str:
        return f"Conversación '{self.args[0]}' no encontrada"


class ImportRequest(BaseModel):
    path: str
    name: str | None = None
    sheet: str | None = None


class NotebookRequest(BaseModel):
    title: str | None = None
    description: str | None = None


class ThreadRequest(BaseModel):
    title: str | None = None
    reporting: bool | None = None


class PanelStructureRequest(BaseModel):
    entity_col: str | None = None
    time_col: str | None = None


class SampleRequest(BaseModel):
    sample: str | None = None


class StructureRequest(BaseModel):
    kind: str
    freq: str | None = None
    date_col: str | None = None
    id_col: str | None = None
    t_col: str | None = None
    start: str | None = None


class RenameRequest(BaseModel):
    new_name: str


class CopyRequest(BaseModel):
    new_name: str | None = None


class ReestimateRequest(BaseModel):
    changes: dict[str, Any]


class CodeRequest(BaseModel):
    code: str
    cell_id: str | None = None
    edited_from: int | None = None  # journal entry this cell re-runs after the user edited it
    dry_run: bool = False  # classify and bind only: return the calls and their canonical code
    timeout: int = 120  # Python cells: seconds before the kernel interrupts the cell


class QxCallRequest(BaseModel):
    tool: str
    args: dict[str, Any] = {}


class NoteRequest(BaseModel):
    title: str = "Nota"
    markdown: str


class CellRequest(BaseModel):
    code: str
    timeout: int = 120


class PinRequest(BaseModel):
    pinned: bool = True


class ReorderRequest(BaseModel):
    order: list[str]


class ChatRequest(BaseModel):
    message: str
    #: A plan the user approved in plan mode (Markdown): the turn follows it.
    plan: str | None = None


class PlanRequest(BaseModel):
    message: str


class NextStepRequest(BaseModel):
    text: str | None = None
    done: bool | None = None


class ApiKeyRequest(BaseModel):
    provider: str
    key: str | None = None


class DownloadRequest(BaseModel):
    repo: str
    file: str


class UseModelRequest(BaseModel):
    id: str


class RuntimeInstallRequest(BaseModel):
    variant: str = "cpu"


class RuntimeConfigRequest(BaseModel):
    ctx_size: int = Field(ge=2048, le=262144)


class Engine:
    """Everything the API needs, created once per process."""

    def __init__(self, settings: Settings, *, model_override: Any = None, with_memory: bool = True,
                 llama_server_cmd: list[str] | None = None) -> None:
        self.settings = settings
        self.store = SettingsStore(settings.settings_file)
        self.user = self.store.load()
        if settings.local_only:
            self.user.local_only = True
        #: http://127.0.0.1:<port> once the server listens (set by __main__ or a test): the kernel calls back to it.
        self.url: str | None = None
        #: The kernel's credential: accepted on /qx/* only, and only together with the run of a cell in flight.
        self.kernel_token = secrets.token_urlsafe(24)
        #: Semantic memory: the user-wide index (preferences); each notebook has its own next to its data.
        self.memory: ProjectMemory | None = None
        self._embedder: tuple[Any, int] | None = None
        self.notebooks = NotebookManager(settings.notebooks_dir, legacy_db=settings.db_path,
                                         legacy_memory=settings.memory_dir)
        self._contexts: dict[str, EngineContext] = {}
        self._contexts_lock = threading.RLock()
        #: The notebook that requests without a /notebooks/<id> prefix act on: the one opened last.
        self.default_id = self.notebooks.current().id
        set_context(self.open(self.default_id))
        self.model_override = model_override
        self.jobs = JobManager()
        self.models = ModelStore(settings.models_dir, self.jobs, token_getter=lambda: get_api_key("huggingface"))
        self.runtime = LlamaRuntime(settings.runtime_dir, settings.logs_dir, self.jobs, server_cmd=llama_server_cmd)
        self.runtime_url: str | None = None
        self.gateway = LLMGateway(settings.logs_dir / "llm_usage.jsonl", settings=lambda: self.user,
                                  runtime_url=lambda: self.runtime_url)
        if model_override is None:
            self.gateway.warm_up()
        #: The team members' agents by id, with the warning of a member that fell back to the active model.
        self._team_agents: dict[str, tuple[Agent, str | None]] = {}
        self._agent = None
        self._stopping = threading.Event()
        threading.Thread(target=self._sweep_loop, name="qtrix-kernel-sweeper", daemon=True).start()
        self.memory_status = "disabled"
        if with_memory:
            # The embedding model may need a one-off download: never block startup on it.
            self.memory_status = "loading"
            threading.Thread(target=self._load_memory, name="qtrix-memory", daemon=True).start()

    def _kernel_env(self, notebook_id: str) -> dict[str, str]:
        if not self.url:
            raise KernelError("The Python kernel calls the engine over HTTP, and the engine's URL is not known yet")
        env = {k: v for k, v in os.environ.items() if k not in ("QTRIX_TOKEN", "QTRIX_URL")}
        # The kernel's URL carries its notebook, so its calls land in that notebook's context.
        return {**env, "QTRIX_URL": f"{self.url}/notebooks/{notebook_id}", "QTRIX_TOKEN": self.kernel_token,
                "PYTHONIOENCODING": "utf-8"}

    def _load_memory(self) -> None:
        try:
            self._embedder = fastembed_embedder()
            self.memory = ProjectMemory(self.settings.memory_dir, *self._embedder)
        except Exception as exc:  # noqa: BLE001 - memory is an enhancement; the app must run without it
            log.warning("Project memory disabled: %s", exc)
            self.memory_status = "unavailable"
            return
        with self._contexts_lock:
            for ctx in self._contexts.values():
                self._attach_memory(ctx)
        self.memory_status = "ready"

    def _attach_memory(self, ctx: EngineContext) -> None:
        """Give a notebook its own memory index (and read access to the user-wide one) once the embedder is loaded."""
        if self._embedder is None or ctx.memory is not None or ctx.notebook is None:
            return
        try:
            ctx.memory = NotebookMemory(ProjectMemory(ctx.notebook.dir / "memory", *self._embedder), self.memory)
        except Exception:  # noqa: BLE001 - a broken index must not keep the notebook from opening
            log.exception("Could not open the memory of notebook %s", ctx.notebook.id)

    # ---- notebooks ------------------------------------------------------------------
    def open(self, notebook_id: str) -> EngineContext:
        """The live context of a notebook (workspace, panel, chat, kernel), built on first use."""
        with self._contexts_lock:
            ctx = self._contexts.get(notebook_id)
            if ctx is None:
                nb = self.notebooks.get(notebook_id)
                ctx = EngineContext(settings=self.settings, workspace=Workspace(nb.db_path), notebook=nb,
                                    allow_agent_code=self.user.agent.allow_code)
                self._attach_memory(ctx)
                ctx.chat_lock = asyncio.Lock()
                ctx.kernels = KernelPool(lambda: self._kernel_env(nb.id), cwd=nb.dir)
                ctx.autosave = Autosave(StateStore(ctx.workspace), ctx, ChatSession)
                ctx.autosave.load()  # the conversations and their reports come back as they were
                ctx.last_used = time.monotonic()
                self._contexts[notebook_id] = ctx
                self._evict()
            ctx.last_used = time.monotonic()
            return ctx

    def _evict(self) -> None:
        """Close the least recently used notebooks beyond ``MAX_OPEN``: never the current one, nor one that is
        running a chat turn or a cell, nor one the UI is watching. Their state is saved and comes back on reopen."""
        idle = sorted((c for nid, c in self._contexts.items() if nid != self.default_id and not _in_use(c)),
                      key=lambda c: c.last_used)
        for ctx in idle:
            if len(self._contexts) <= MAX_OPEN:
                break
            self._close(ctx)

    def _close(self, ctx: EngineContext, *, save: bool = True) -> None:
        ctx.autosave.close(save=save)
        ctx.kernels.shutdown()
        ctx.workspace.close()
        self._contexts.pop(ctx.notebook.id, None)

    def _sweep_loop(self) -> None:
        while not self._stopping.wait(60):
            self.sweep_kernels()

    def sweep_kernels(self, max_idle: float | None = None) -> int:
        """Stop the Python kernels of notebooks nobody has run a cell in for ``KERNEL_IDLE`` seconds."""
        limit = KERNEL_IDLE if max_idle is None else max_idle
        with self._contexts_lock:
            contexts = list(self._contexts.values())
        return sum(ctx.kernels.shutdown_idle(limit) for ctx in contexts)

    def flush(self) -> None:
        """Write the pending chat and panel changes of every open notebook now."""
        with self._contexts_lock:
            for ctx in self._contexts.values():
                ctx.autosave.flush()

    @property
    def ctx(self) -> EngineContext:
        """The context of the notebook the running request is bound to (the default one outside a request)."""
        active = active_context()
        if active is not None and active.notebook is not None and self._contexts.get(active.notebook.id) is active:
            return active
        return self._contexts[self.default_id]

    @property
    def chat(self) -> ChatSession:
        return self.ctx.chat

    @property
    def _chat_lock(self) -> asyncio.Lock:
        return self.ctx.chat_lock

    def notebook_dict(self, notebook_id: str) -> dict[str, Any]:
        nb = self.notebooks.get(notebook_id)
        with self._contexts_lock:
            ctx = self._contexts.get(nb.id)
        if ctx is not None:
            n_datasets, n_objects = len(ctx.workspace.list()), len(ctx.objects.list())
        else:  # a closed notebook is counted with a short-lived connection, not opened
            ws = Workspace(nb.db_path)
            try:
                n_datasets, n_objects = len(ws.list()), len(ObjectStore(ws).list())
            finally:
                ws.close()
        return {**nb.to_dict(), "current": nb.id == self.default_id, "n_datasets": n_datasets, "n_objects": n_objects}

    def open_notebook(self, notebook_id: str) -> dict[str, Any]:
        """Mark a notebook as the one in use: unprefixed requests (MCP, scripts) act on it from now on."""
        ctx = self.open(notebook_id)
        ctx.notebook = self.notebooks.touch(notebook_id)
        self.default_id = notebook_id
        set_context(ctx)
        return self.notebook_dict(notebook_id)

    def update_notebook(self, notebook_id: str, req: NotebookRequest) -> dict[str, Any]:
        nb = self.notebooks.update(notebook_id, title=req.title, description=req.description)
        self.open(notebook_id).notebook = nb
        return self.notebook_dict(notebook_id)

    def duplicate_notebook(self, notebook_id: str, title: str | None = None) -> dict[str, Any]:
        src = self.notebooks.get(notebook_id)
        copy = self.notebooks.create(title or f"{src.title} (copia)", src.description)
        try:
            source = self.open(notebook_id)
            source.autosave.flush()  # the copy gets the conversation and the report as they are now
            source.workspace.copy_to(copy.db_path)
            if (source.notebook.dir / "memory").is_dir():
                shutil.copytree(source.notebook.dir / "memory", copy.dir / "memory")
        except Exception:
            shutil.rmtree(copy.dir, ignore_errors=True)
            raise
        return self.notebook_dict(copy.id)

    def delete_notebook(self, notebook_id: str) -> None:
        """Move the notebook to the trash after closing its kernel and database."""
        self.notebooks.get(notebook_id)  # 404 before anything is closed
        with self._contexts_lock:
            ctx = self._contexts.pop(notebook_id, None)
        if ctx is not None:
            self._close(ctx, save=False)
        self.notebooks.remove(notebook_id)
        if notebook_id == self.default_id:
            self.default_id = self.notebooks.current().id
            set_context(self.open(self.default_id))

    # ---- conversations (threads) of the notebook of the request -----------------------------------------------
    def thread(self, thread_id: str) -> Thread:
        thread = self.ctx.threads.get(thread_id)
        if thread is None:
            raise ThreadNotFound(thread_id)
        return thread

    def thread_dict(self, thread: Thread) -> dict[str, Any]:
        return {**thread.to_dict(), "current": thread.id == self.ctx.thread_id}

    def list_threads(self) -> list[dict[str, Any]]:
        return [self.thread_dict(t) for t in self.ctx.threads.values()]

    def create_thread(self, title: str | None = None) -> dict[str, Any]:
        """A new, empty conversation on the notebook's data; it becomes the one in use."""
        ctx = self.ctx
        thread = ctx.autosave.add(Thread(f"th-{secrets.token_hex(4)}", (title or "").strip() or DEFAULT_THREAD_TITLE))
        ctx.thread_id = thread.id
        return self.thread_dict(thread)

    def open_thread(self, thread_id: str) -> dict[str, Any]:
        """Mark a conversation as the one in use: unbound requests (kernel cells, MCP) act on its panel."""
        self.ctx.thread_id = self.thread(thread_id).id
        return self.thread_dict(self.ctx.threads[thread_id])

    def rename_thread(self, thread_id: str, title: str) -> dict[str, Any]:
        thread = self.thread(thread_id)
        thread.title = title.strip()[:80] or thread.title
        self.ctx.autosave.save_meta(thread)
        return self.thread_dict(thread)

    def set_reporting(self, thread_id: str, enabled: bool) -> dict[str, Any]:
        thread = self.thread(thread_id)
        thread.reporting = enabled
        self.ctx.autosave.save_reporting(thread)
        return self.thread_dict(thread)

    def reporting_dict(self, thread: Thread) -> dict[str, Any]:
        return {"enabled": thread.reporting, "version": thread.report_version, "report": thread.report}

    async def reporting_events(self, thread: Thread, *, full: bool) -> AsyncIterator[dict[str, Any]]:
        """Rewrite the conversation's report from its last turn (``full``: from every turn, starting over). The caller
        holds the chat lock. Yields ``reporting`` events; a failure is an ``error`` status, never an exception."""
        yield {"type": "reporting", "status": "running", "version": thread.report_version}
        try:
            reporter = reporter_of(self.agent())
            turns = turns_of(thread.chat.transcript)
            if reporter is None or not turns:
                yield {"type": "reporting", "status": "done", "version": thread.report_version}
                return
            previous = None if full or not thread.report else ConversationReport(**thread.report)
            known = {o.name for o in await asyncio.to_thread(self.ctx.objects.list)}
            report = await update_report(reporter, previous, turns if previous is None else turns[-1:], known=known)
            thread.report = report.model_dump()
            thread.report_version += 1
            await asyncio.to_thread(self.ctx.autosave.save_reporting, thread)
            yield {"type": "reporting", "status": "done", "version": thread.report_version}
        except Exception as exc:  # noqa: BLE001 - the report must never break the chat
            log.exception("Reporting failed")
            yield {"type": "reporting", "status": "error", "message": f"{type(exc).__name__}: {exc}"}

    def delete_thread(self, thread_id: str) -> None:
        """Take a conversation off the notebook (its rows stay in the file). The datasets and objects it made stay."""
        ctx = self.ctx
        self.thread(thread_id)
        if len(ctx.threads) == 1:
            raise HTTPException(409, "Un cuaderno necesita al menos una conversación")
        if ctx.chat_lock.locked() and ctx.thread_id == thread_id:
            raise HTTPException(409, "Espera a que termine el turno en curso")
        ctx.autosave.remove(thread_id)
        ctx.threads.pop(thread_id)
        if ctx.thread_id == thread_id:
            ctx.thread_id = max(ctx.threads.values(), key=lambda t: t.updated_at).id

    def shutdown(self) -> None:
        self._stopping.set()
        with self._contexts_lock:
            for ctx in self._contexts.values():
                ctx.autosave.flush()
                ctx.kernels.shutdown()

    # ---- model lifecycle ------------------------------------------------------------
    def validate_model(self) -> None:
        """Cheap checks that fail fast with a clear message before a chat run starts."""
        if self.model_override is not None:
            return
        profile = self.user.active()
        if profile.provider == "llamacpp":
            if self.runtime.installed() is None:
                raise RuntimeError("Instala el runtime llama.cpp desde Modelos para usar modelos descargados")
            if not self.models.list():
                raise ValueError("No hay modelos descargados: descarga uno desde Modelos")
        elif profile.provider not in ("ollama", "lmstudio"):
            build_model(profile, gateway=self.gateway, local_only=self.user.local_only)  # local-only / missing key

    def needs_loading(self) -> bool:
        profile = self.user.active()
        if self.model_override is not None or profile.provider != "llamacpp":
            return False
        try:
            path = self.models.resolve(profile.model)
        except FileNotFoundError:
            return True
        return not self.runtime.is_serving(path, self.user.llamacpp.ctx_size)

    async def ensure_local_model(self) -> None:
        """If the active Ollama/LM Studio profile points at a model that is not installed, use an installed one."""
        profile = self.user.active()
        if profile.provider not in ("ollama", "lmstudio") or self.model_override is not None:
            return
        installed = (await detect_local_models()).get(profile.provider, [])
        if not installed or profile.model in installed or f"{profile.model}:latest" in installed:
            return
        log.warning("Model '%s' is not installed in %s; using '%s'", profile.model, profile.provider, installed[0])
        profile.model = installed[0]
        self.user = self.store.save(self.user)
        self._agent = None

    async def prepare_model(self) -> None:
        if self.model_override is not None:
            return
        profile = self.user.active()
        if profile.provider in ("ollama", "lmstudio"):
            await self.ensure_local_model()
            return
        if profile.provider != "llamacpp":
            return
        try:
            path = self.models.resolve(profile.model)
        except FileNotFoundError:
            available = self.models.list()
            if not available:
                raise ValueError("No hay modelos descargados: descarga uno desde Modelos") from None
            profile.model = available[0]["id"]
            self.user = self.store.save(self.user)
            path = self.models.resolve(profile.model)
        url = await asyncio.to_thread(self.runtime.ensure, path, ctx_size=self.user.llamacpp.ctx_size)
        if url != self.runtime_url:
            self.runtime_url, self._agent = url, None

    def recall(self, query: str) -> str:
        memory = self.ctx.memory
        return memory.recall(query, k=6) if memory is not None else ""

    @property
    def _agent(self) -> Agent | None:
        return self._lead

    @_agent.setter
    def _agent(self, agent: Agent | None) -> None:
        # Whatever drops the lead (new settings, key, runtime) drops the team built with it.
        self._lead = agent
        if agent is None:
            self._team_agents.clear()

    def agent(self):
        if self._agent is None:
            profile = self.user.active()
            lead = self.user.team.lead()
            model = self.model_override or build_model(profile, gateway=self.gateway, local_only=self.user.local_only)
            self._agent = build_lead_agent(
                model,
                namespaces=lead.namespaces if lead.namespaces is not None else self.user.agent.namespaces,
                instructions_extra=lead.behavior,
                skills=lead.skills,
                memory_lookup=self.recall,
                # Small local models have a short context: send tool schemas on demand.
                compact=self.model_override is None and profile.provider in ("llamacpp", "ollama", "lmstudio"),
                roster=roster_text(self.user.team.active()),
            )
        return self._agent

    def _member_model(self, cfg: TeamAgentConfig) -> tuple[Any, ModelProfile, str | None]:
        """The model a member runs on: its own profile when it can be used, else the active one (with a warning)."""
        active = self.user.active()
        if self.model_override is not None:
            return self.model_override, active, None
        profile, warning = active, None
        if cfg.profile and cfg.profile != active.name:
            chosen = next((p for p in self.user.profiles if p.name == cfg.profile), None)
            if chosen is None:
                warning = f"{cfg.name}: el modelo «{cfg.profile}» ya no existe; usa el modelo activo"
            elif chosen.provider == "llamacpp":  # llama.cpp serves one model at a time: the active one
                warning = f"{cfg.name}: llama.cpp sirve un solo modelo a la vez; usa el modelo activo"
            else:
                profile = chosen
        try:
            return build_model(profile, gateway=self.gateway, local_only=self.user.local_only), profile, warning
        except (ProviderBlocked, ValueError) as exc:
            if profile is active:
                raise
            warning = f"{cfg.name}: {exc}; usa el modelo activo"
            return build_model(active, gateway=self.gateway, local_only=self.user.local_only), active, warning

    def team(self) -> tuple[list[TeamMember], list[str]]:
        """The members the lead may delegate to this turn (none while the team is off) and the warnings of those
        that cannot run on their own model."""
        members, warnings = [], []
        for cfg in self.user.team.active():
            if cfg.id not in self._team_agents:
                model, profile, warning = self._member_model(cfg)
                agent = build_team_agent(model, member_id=cfg.id, name=cfg.name, behavior=cfg.behavior,
                                         skills=cfg.skills, namespaces=cfg.namespaces, memory_lookup=self.recall,
                                         compact=self.model_override is None
                                         and profile.provider in ("llamacpp", "ollama", "lmstudio"))
                self._team_agents[cfg.id] = (agent, warning)
            agent, warning = self._team_agents[cfg.id]
            members.append(TeamMember(cfg.id, cfg.name, agent, cfg.emoji))
            if warning:
                warnings.append(warning)
        return members, warnings

    def context_window(self) -> int:
        return context_window(self.user.active(), self.user)

    def update_settings(self, new: UserSettings) -> UserSettings:
        if self.settings.local_only:
            new.local_only = True
        self.user = self.store.save(new)
        with self._contexts_lock:
            for ctx in self._contexts.values():
                ctx.allow_agent_code = self.user.agent.allow_code
        self._agent = None
        return self.user

    def use_local_model(self, model_id: str) -> UserSettings:
        self.models.resolve(model_id)
        existing = next((p for p in self.user.profiles if p.provider == "llamacpp" and p.model == model_id), None)
        if existing is None:
            name = f"qtrix · {Path(model_id).name.removesuffix('.gguf')}"
            existing = ModelProfile(name=name, provider="llamacpp", model=model_id)
            self.user.profiles.append(existing)
        self.user.active_profile = existing.name
        return self.update_settings(self.user)

    def remember(self, kind: str, text: str) -> None:
        memory = self.ctx.memory
        if memory is not None:
            try:
                memory.add(kind, text)
            except Exception:  # noqa: BLE001 - memory must never break the chat
                log.exception("Failed to index memory item")


def _sse(event: dict[str, Any]) -> str:
    return f"data: {json.dumps(event, default=str, ensure_ascii=False)}\n\n"


def _dir_size(path: Path) -> int:
    total = 0
    for p in path.rglob("*"):
        try:
            if p.is_file():
                total += p.stat().st_size
        except OSError:
            continue
    return total


def create_app(settings: Settings | None = None, *, model_override: Any = None, with_memory: bool = True,
               llama_server_cmd: list[str] | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    engine = Engine(settings, model_override=model_override, with_memory=with_memory,
                    llama_server_cmd=llama_server_cmd)
    app = FastAPI(title="qtrix engine", version=__version__)
    app.state.engine = engine
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

    @app.middleware("http")
    async def require_token(request: Request, call_next):
        # /notebooks/<id>/<route> is <route> acting on that notebook: strip the prefix and bind the notebook's
        # context to the request (a ContextVar, inherited by the threads and tasks the route starts).
        path, notebook_id = request.url.path, None
        prefixed = _NOTEBOOK_ROUTE.match(path)
        if prefixed and prefixed.group(2) not in _NOTEBOOK_OWN:
            notebook_id, path = prefixed.group(1), prefixed.group(2)
        if settings.token and path != "/health" and request.method != "OPTIONS":
            supplied = request.headers.get("x-qtrix-token") or request.query_params.get("token")
            bearer = request.headers.get("authorization", "")
            if supplied is None and path.startswith("/llm/") and bearer.startswith("Bearer "):
                supplied = bearer.removeprefix("Bearer ")  # OpenAI clients pointed at the gateway
            kernel = supplied == engine.kernel_token and path.startswith("/qx/")
            if supplied != settings.token and not kernel:
                return JSONResponse({"detail": "invalid token"}, status_code=401)
        if notebook_id is not None:
            try:
                ctx = engine.open(notebook_id)
                activate(ctx)
            except NotebookNotFound as exc:
                return JSONResponse({"detail": str(exc)}, status_code=404)
            # /notebooks/<id>/threads/<tid>/<route> is <route> on that conversation of the notebook (its chat and
            # its panel; the data are the notebook's).
            in_thread = _THREAD_ROUTE.match(path)
            if in_thread and in_thread.group(2) not in _THREAD_OWN:
                if in_thread.group(1) not in ctx.threads:
                    return JSONResponse({"detail": str(ThreadNotFound(in_thread.group(1)))}, status_code=404)
                activate_thread(in_thread.group(1))
                path = in_thread.group(2)
            else:
                activate_thread(None)  # the conversation open in the UI
            request.scope["path"], request.scope["raw_path"] = path, path.encode()
        return await call_next(request)

    @app.exception_handler(NotebookNotFound)
    async def _notebook_not_found(_: Request, exc: NotebookNotFound):
        return JSONResponse({"detail": str(exc)}, status_code=404)

    @app.exception_handler(ThreadNotFound)
    async def _thread_not_found(_: Request, exc: ThreadNotFound):
        return JSONResponse({"detail": str(exc)}, status_code=404)

    @app.exception_handler(DatasetNotFound)
    async def _not_found(_: Request, exc: DatasetNotFound):
        return JSONResponse({"detail": str(exc).strip("'\"")}, status_code=404)

    @app.exception_handler(PinnedBlockError)
    async def _pinned(_: Request, exc: PinnedBlockError):
        return JSONResponse({"detail": str(exc)}, status_code=409)

    @app.exception_handler(ObjectNotFound)
    async def _object_not_found(_: Request, exc: ObjectNotFound):
        return JSONResponse({"detail": str(exc)}, status_code=404)

    @app.exception_handler(LockedObjectError)
    async def _object_locked(_: Request, exc: LockedObjectError):
        return JSONResponse({"detail": str(exc)}, status_code=409)

    @app.exception_handler(JournalEntryNotFound)
    async def _entry_not_found(_: Request, exc: JournalEntryNotFound):
        return JSONResponse({"detail": str(exc)}, status_code=404)

    @app.exception_handler(UndoError)
    async def _cannot_undo(_: Request, exc: UndoError):
        return JSONResponse({"detail": str(exc)}, status_code=409)

    @app.exception_handler(ReplayError)
    async def _cannot_replay(_: Request, exc: ReplayError):
        return JSONResponse({"detail": str(exc)}, status_code=409)

    @app.exception_handler(PolicyDenied)
    async def _policy_denied(_: Request, exc: PolicyDenied):
        return JSONResponse({"detail": str(exc)}, status_code=403)

    @app.exception_handler(KernelError)
    async def _kernel_unavailable(_: Request, exc: KernelError):
        return JSONResponse({"detail": str(exc)}, status_code=503)

    @app.on_event("shutdown")
    def _stop_kernels() -> None:
        engine.shutdown()

    @app.exception_handler(ObjectError)
    async def _object_error(_: Request, exc: ObjectError):
        return JSONResponse({"detail": str(exc)}, status_code=400)

    # ---- notebooks ----------------------------------------------------------------
    # Everything else is reachable per notebook as /notebooks/<id>/<route> (see the middleware above).
    @app.get("/notebooks")
    async def list_notebooks() -> list[dict[str, Any]]:
        return [await asyncio.to_thread(engine.notebook_dict, nb.id) for nb in engine.notebooks.list()]

    @app.post("/notebooks")
    async def create_notebook(req: NotebookRequest) -> dict[str, Any]:
        nb = engine.notebooks.create(req.title or "Cuaderno sin título", req.description or "")
        return await asyncio.to_thread(engine.open_notebook, nb.id)  # a new notebook is the one in use

    @app.get("/notebooks/{notebook_id}")
    async def get_notebook(notebook_id: str) -> dict[str, Any]:
        return await asyncio.to_thread(engine.notebook_dict, notebook_id)

    @app.patch("/notebooks/{notebook_id}")
    async def update_notebook(notebook_id: str, req: NotebookRequest) -> dict[str, Any]:
        return await asyncio.to_thread(engine.update_notebook, notebook_id, req)

    @app.post("/notebooks/{notebook_id}/open")
    async def open_notebook(notebook_id: str) -> dict[str, Any]:
        return await asyncio.to_thread(engine.open_notebook, notebook_id)

    @app.post("/notebooks/{notebook_id}/duplicate")
    async def duplicate_notebook(notebook_id: str, req: NotebookRequest | None = None) -> dict[str, Any]:
        return await asyncio.to_thread(engine.duplicate_notebook, notebook_id, req.title if req else None)

    @app.delete("/notebooks/{notebook_id}")
    async def delete_notebook(notebook_id: str) -> dict[str, bool]:
        await asyncio.to_thread(engine.delete_notebook, notebook_id)
        return {"ok": True}

    # ---- conversations ------------------------------------------------------------
    # A notebook holds several conversations on the same data; /threads/<tid>/<route> is <route> (chat, panel,
    # code) on one of them (see the middleware above). Unprefixed routes act on the one open in the UI.
    @app.get("/threads")
    async def list_threads() -> list[dict[str, Any]]:
        return engine.list_threads()

    @app.post("/threads")
    async def create_thread(req: ThreadRequest | None = None) -> dict[str, Any]:
        return engine.create_thread(req.title if req else None)

    @app.post("/threads/{thread_id}/open")
    async def open_thread(thread_id: str) -> dict[str, Any]:
        return engine.open_thread(thread_id)

    @app.patch("/threads/{thread_id}")
    async def rename_thread(thread_id: str, req: ThreadRequest) -> dict[str, Any]:
        out = engine.thread_dict(engine.thread(thread_id))
        if req.title is not None:
            out = await asyncio.to_thread(engine.rename_thread, thread_id, req.title)
        if req.reporting is not None:
            out = await asyncio.to_thread(engine.set_reporting, thread_id, req.reporting)
        return out

    @app.get("/reporting")
    async def get_reporting() -> dict[str, Any]:
        return engine.reporting_dict(engine.ctx.thread)

    @app.post("/reporting/rebuild")
    async def rebuild_reporting() -> StreamingResponse:
        """Write the report again from every turn of the conversation (streams ``reporting`` events)."""
        try:
            engine.validate_model()
        except (ProviderBlocked, ValueError, RuntimeError) as exc:
            raise HTTPException(400, str(exc)) from exc
        thread = engine.ctx.thread

        async def events() -> AsyncIterator[str]:
            async with engine._chat_lock:
                await engine.prepare_model()
                async for ev in engine.reporting_events(thread, full=True):
                    yield _sse(ev)
                yield _sse({"type": "done"})

        return StreamingResponse(events(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})

    @app.delete("/threads/{thread_id}")
    async def delete_thread(thread_id: str) -> dict[str, bool]:
        await asyncio.to_thread(engine.delete_thread, thread_id)
        return {"ok": True}

    # ---- system -------------------------------------------------------------------
    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {"status": "ok", "version": __version__, "memory": engine.memory_status}

    @app.get("/system/info")
    async def system_info() -> dict[str, Any]:
        def collect() -> list[dict[str, Any]]:
            return [{"name": name, "description": desc, "path": str(settings.home / name),
                     "size": _dir_size(settings.home / name)} for name, desc in LAYOUT.items()]

        _, origin = resolve_home()
        return {"home": str(settings.home), "origin": origin, "version": __version__,
                "pointer": str(bootstrap_dir() / "location.json"), "folders": await asyncio.to_thread(collect)}

    @app.get("/tools")
    async def tools() -> dict[str, Any]:
        return {"namespaces": NAMESPACES, "tools": catalog()}

    @app.get("/tools/{name}/help")
    async def tool_help_(name: str) -> dict[str, Any]:
        """A tool's summary and parameters, for the help popup of the composer's autocomplete."""
        info = qx_tool_help(name)
        if info is None:
            raise HTTPException(404, f"Unknown tool '{name}'")
        return clean(info)

    @app.get("/methods")
    async def methods() -> dict[str, Any]:
        """The analysis methods by area, for the catalog the user browses before asking the agent."""
        return {"areas": methods_catalog()}

    def run(tool: str, args: dict[str, Any], *, actor: str, channel: str, **meta: Any) -> Any:
        """Every mutation goes through the dispatcher (ADR 0002 §6): journal, policy, locks."""
        return dispatch(Call(tool, {k: v for k, v in args.items() if v is not None}), actor, channel, **meta)

    @app.post("/tools/{name}")
    async def call_tool(name: str, args: dict[str, Any], edited_from: int | None = None,
                        cell_id: str | None = None) -> Any:
        """Run a registry tool directly (UI quick actions, scripting) without going through the LLM.

        ``edited_from`` links the entry to the journal entry this call re-runs after the user edited it.
        """
        if not any(t.name == name for t in all_tools()):
            raise HTTPException(404, f"Unknown or unavailable tool '{name}'")
        try:
            return clean(await asyncio.to_thread(
                dispatch, Call(name, args), "api", "api", edited_from=edited_from, cell_id=cell_id))
        except LockedObjectError as exc:
            raise HTTPException(409, str(exc)) from exc
        except (JournalEntryNotFound, PolicyDenied):
            raise
        except (TypeError, ValueError, KeyError, FileNotFoundError) as exc:
            raise HTTPException(400, f"{type(exc).__name__}: {exc}") from exc

    @app.post("/code")
    async def submit_code(req: CodeRequest) -> Any:
        """Submit a cell of Python (ADR 0002 §7.2) as the user.

        An API cell (only ``qx`` calls with literal arguments) is parsed with ``ast``, never executed, and its calls
        run in order through the dispatcher, stopping at the first error. Any other cell is a Python cell: it runs in
        the project kernel (E7·07b), journaled as one ``python`` entry plus an entry per ``qx`` call it makes.
        ``dry_run`` classifies the cell (and binds an API cell's calls) without running anything.
        """
        try:
            cell = parse_cell(req.code)
        except QxError as exc:
            if req.dry_run:  # the editor checks as the user types: a cell that does not parse yet is not an error
                return {"kind": "invalid", "error": str(exc)}
            raise HTTPException(400, str(exc)) from exc
        if req.dry_run:
            if cell.kind == "python" and (cell.reason or "").startswith("syntax error"):
                return {"kind": "invalid", "error": cell.reason}
            if cell.kind == "python":
                return {"kind": "python", "reason": cell.reason}
            return {"kind": "api", "calls": await asyncio.to_thread(lambda: [
                {"line": p.line, "tool": p.call.tool, "args": p.call.args,
                 "code": print_call(p.call.tool, p.call.args) if p.handle_from is None else None,
                 "issues": check_call(p.call) if p.handle_from is None else []}
                for p in cell.calls])}
        return await run_code(cell, req.code, cell_id=req.cell_id, edited_from=req.edited_from, timeout=req.timeout)

    async def run_code(cell: Any, code: str, *, cell_id: str | None, edited_from: int | None,
                       timeout: int) -> dict[str, Any]:
        """Run a parsed cell as the user: API cells through the dispatcher, anything else in the project kernel."""
        if edited_from is not None:
            engine.ctx.journal.get(edited_from)  # 404 before anything runs
        since = engine.ctx.journal.last_seq()
        if cell.kind == "python":
            call = Call("py_run", {"code": code, "timeout": timeout})
            try:
                result, seq = await asyncio.to_thread(dispatch, call, "user", "code", cell_id=cell_id,
                                                      edited_from=edited_from, with_seq=True)
            except CellFailed as exc:
                return clean({"kind": "python", "reason": cell.reason, **exc.result, "ok": False,
                              "seq": getattr(exc, "journal_seq", None), **_effects(since)})
            return clean({"kind": "python", "reason": cell.reason, **result, "seq": seq, **_effects(since)})

        def run_one(call: Call) -> tuple[Any, int | None]:
            return dispatch(call, "user", "code", cell_id=cell_id, edited_from=edited_from, with_seq=True)

        entries = await asyncio.to_thread(run_cell, cell, run_one)
        for entry in entries:
            entry.pop("exception", None)
        return clean({"kind": "api", "ok": all(e["ok"] for e in entries), "calls": entries, **_effects(since)})

    def _effects(since: int) -> dict[str, Any]:
        """What a cell changed: the objects it versioned and the ones that went stale because of it."""
        versions: dict[str, int] = {}
        touched: set[str] = set()
        for e in engine.ctx.journal.list(since=since, mutating=True, ok=True):
            for a in e["affected"]:
                if a.get("kind") != "object":
                    continue
                if a.get("version_after") and a.get("version_before") != a.get("version_after"):
                    versions[a["name"]] = a["version_after"]
                else:
                    touched.add(a["name"])
        stale = []
        for name in sorted(touched - set(versions)):
            try:
                if engine.ctx.objects.get(name).status == "stale":
                    stale.append(name)
            except ObjectNotFound:
                continue
        return {"versions": [{"name": n, "version": v} for n, v in versions.items()], "stale": stale}

    @app.get("/code/completions")
    async def code_completions() -> dict[str, Any]:
        """Names the cell editor completes: qx functions per target (with signatures), objects, pages and series."""
        return {"functions": {on: qx_names(on) for on in ("qx", "page", "obj", "report")},
                "signatures": qx_signatures(),
                "objects": [{"name": o.name, "type": o.type} for o in engine.ctx.objects.list()],
                "pages": [{"name": d.name, "columns": [c["name"] for c in d.columns or []]}
                          for d in engine.ctx.workspace.list()]}

    # ---- client-mode qx and the kernel (E7·07b) -----------------------------------
    @app.post("/qx/call")
    async def qx_call(req: QxCallRequest, request: Request) -> Any:
        """One ``qx`` call from the project kernel or an external client (the user's Jupyter, a script)."""
        run_id = request.headers.get("x-qtrix-run") or ""
        run = engine.ctx.kernels.runs.get(run_id)
        from_kernel = request.headers.get("x-qtrix-token") == engine.kernel_token
        if (run_id or from_kernel) and run is None:
            return JSONResponse({"detail": "The cell that made this call has finished: qx calls must happen while "
                                 "their cell runs", "type": "PolicyDenied"}, status_code=409)
        try:
            result = await asyncio.to_thread(dispatch_from_kernel, run, Call(req.tool, req.args))
        except Exception as exc:  # noqa: BLE001 - every error goes back to the client with its type
            status = (404 if isinstance(exc, ObjectNotFound | DatasetNotFound) else
                      409 if isinstance(exc, LockedObjectError) else 403 if isinstance(exc, PolicyDenied) else
                      503 if isinstance(exc, KernelError) else 400)
            return JSONResponse({"detail": str(exc).strip("'\""), "type": type(exc).__name__}, status_code=status)
        return {"result": clean(result)}

    @app.get("/qx/frame")
    async def qx_frame(page: str, columns: str | None = None, sample: str | None = None) -> Response:
        """A page as an Arrow stream, indexed by its structure and restricted to the sample (``page.frame()``)."""
        import pyarrow as pa

        cols = [c for c in columns.split(",") if c] if columns is not None else None
        try:
            df = await asyncio.to_thread(engine.ctx.workspace.frame, page, cols, sample)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        table = pa.Table.from_pandas(df, preserve_index=True)
        sink = pa.BufferOutputStream()
        with pa.ipc.new_stream(sink, table.schema) as writer:
            writer.write_table(table)
        return Response(sink.getvalue().to_pybytes(), media_type="application/vnd.apache.arrow.stream")

    @app.get("/kernel")
    async def kernel_status() -> dict[str, Any]:
        return engine.ctx.kernels.status()

    @app.post("/kernel/start")
    async def kernel_start() -> dict[str, Any]:
        await asyncio.to_thread(engine.ctx.kernels.project.start)
        return engine.ctx.kernels.status()

    @app.post("/kernel/interrupt")
    async def kernel_interrupt() -> dict[str, Any]:
        """Stop the running cell (from the chat or the notebook); the kernel and its variables stay."""
        return {"interrupted": engine.ctx.kernels.project.interrupt(), **engine.ctx.kernels.status()}

    @app.post("/kernel/restart")
    async def kernel_restart() -> dict[str, Any]:
        await asyncio.to_thread(engine.ctx.kernels.project.restart)
        return engine.ctx.kernels.status()

    @app.post("/kernel/shutdown")
    async def kernel_shutdown() -> dict[str, Any]:
        await asyncio.to_thread(engine.ctx.kernels.project.shutdown)
        return engine.ctx.kernels.status()

    # ---- journal ------------------------------------------------------------------
    @app.get("/journal")
    async def get_journal(actor: str | None = None, mutating: bool | None = None, since: int = 0,
                          ok: bool | None = None, tool: str | None = None, limit: int | None = None
                          ) -> list[dict[str, Any]]:
        """Journal entries in order. ``actor=agent`` matches every agent; ``since`` is an exclusive seq."""
        return await asyncio.to_thread(engine.ctx.journal.list, actor=actor, mutating=mutating, since=since,
                                       ok=ok, tool=tool, limit=limit)

    @app.get("/journal/script")
    async def journal_script() -> PlainTextResponse:
        """The mutating journal, in order, as a Python script."""
        return PlainTextResponse(await asyncio.to_thread(engine.ctx.journal.script), media_type="text/x-python")

    @app.post("/journal/replay")
    async def journal_replay() -> dict[str, Any]:
        """Rebuild pages and objects from source snapshots + the mutating journal and verify the summaries."""
        return await asyncio.to_thread(replay)

    @app.get("/journal/{seq}")
    async def journal_entry(seq: int) -> dict[str, Any]:
        return engine.ctx.journal.get(seq)

    @app.post("/journal/{seq}/undo")
    async def journal_undo(seq: int) -> dict[str, Any]:
        """Undo one call: restore the object versions and page state it changed (the user's action)."""
        return clean(await asyncio.to_thread(run, "journal_undo", {"seq": seq}, actor="user", channel="ui"))

    # ---- objects ------------------------------------------------------------------
    # The user acts through these endpoints (actor "user"), so locks do not stop them: they stop agents, MCP, /tools.
    def user_call(tool: str, **args: Any) -> Any:
        return run(tool, args, actor="user", channel="ui")

    @app.get("/objects")
    async def list_objects(type: str | None = None) -> list[dict[str, Any]]:
        return [o.to_dict(detail=False) for o in await asyncio.to_thread(engine.ctx.objects.list, type)]

    @app.get("/objects/{name}")
    async def get_object(name: str) -> dict[str, Any]:
        obj = await asyncio.to_thread(engine.ctx.objects.get, name)
        return clean({**obj.to_dict(), "dependants": engine.ctx.objects.dependants(obj.name)})

    @app.get("/objects/{name}/views")
    async def object_views(name: str) -> dict[str, Any]:
        """The views (with their parameters) and procs of an object: what the UI's View and Proc menus show."""
        return await asyncio.to_thread(dispatch_read, "obj_views", name=name)

    @app.get("/objects/{name}/view")
    async def object_default_view(name: str) -> dict[str, Any]:
        """The object's default view, computed without touching the panel (what a report embeds)."""
        def go() -> dict[str, Any]:
            result, live_ref = compute_view(name)
            return {**result.to_dict(), "ref": live_ref, "version": engine.ctx.objects.get(name).version}

        try:
            return clean(await asyncio.to_thread(go))
        except (ValueError, TypeError) as exc:
            raise HTTPException(400, f"{type(exc).__name__}: {exc}") from exc

    @app.get("/objects/{name}/views/{view}")
    async def object_view(name: str, view: str, params: str | None = None) -> dict[str, Any]:
        """One view of the object's current version (cached per version). ``params`` is a JSON object."""
        try:
            values = json.loads(params) if params else None
        except json.JSONDecodeError as exc:
            raise HTTPException(400, f"params must be a JSON object: {exc}") from exc

        def go() -> dict[str, Any]:
            result, live_ref = compute_view(name, view, values)
            return {**result.to_dict(), "ref": live_ref, "version": engine.ctx.objects.get(name).version}

        try:
            return clean(await asyncio.to_thread(go))
        except (ObjectError, ObjectNotFound):
            raise
        except (ValueError, TypeError) as exc:
            raise HTTPException(400, f"{type(exc).__name__}: {exc}") from exc

    def dispatch_read(tool: str, **args: Any) -> Any:
        return dispatch(Call(tool, args), "user", "ui")

    @app.get("/objects/{name}/versions")
    async def object_versions(name: str) -> dict[str, Any]:
        versions = engine.ctx.objects.versions(name)
        out: dict[str, Any] = {"name": engine.ctx.objects.resolve(name), "versions": versions}
        if len(versions) > 1:
            out["diff"] = engine.ctx.objects.diff(name)
        return clean(out)

    @app.get("/objects/{name}/code")
    async def object_code(name: str, version: int | None = None) -> dict[str, Any]:
        """The ``qx`` code that re-estimates the object from its spec (current version unless ``version``).

        Editing it (the formula, the sample…) and submitting it to ``POST /code`` creates a new version.
        """
        obj = await asyncio.to_thread(engine.ctx.objects.get, name)
        spec, current = obj.spec, obj.version
        if version is not None:
            match = [v for v in engine.ctx.objects.versions(obj.name) if v["version"] == version]
            if not match:
                raise HTTPException(404, f"'{obj.name}' has no version {version}")
            spec, current = match[0]["spec"], version
        if spec.get("method") == "python":  # registered from a Python cell: the cell is the spec
            return {"name": obj.name, "version": current, "kind": "python", "code": spec["code"], "reason": None}
        try:
            return {"name": obj.name, "version": current, "kind": "api", "code": print_object(spec, obj.name),
                    "reason": None}
        except NotExpressible as exc:
            return {"name": obj.name, "version": current, "kind": "api", "code": None, "reason": str(exc)}

    @app.post("/objects/{name}/rename")
    async def rename_object(name: str, req: RenameRequest) -> dict[str, Any]:
        out = await asyncio.to_thread(user_call, "obj_rename", name=name, new_name=req.new_name)
        return engine.ctx.objects.get(out["name"]).to_dict()

    @app.post("/objects/{name}/copy")
    async def copy_object(name: str, req: CopyRequest) -> dict[str, Any]:
        out = await asyncio.to_thread(user_call, "obj_copy", name=name, new_name=req.new_name)
        return engine.ctx.objects.get(out["name"]).to_dict()

    @app.post("/objects/{name}/delete")
    async def delete_object(name: str) -> dict[str, str]:
        return await asyncio.to_thread(user_call, "obj_delete", name=name)

    @app.post("/objects/{name}/lock")
    async def lock_object(name: str) -> dict[str, Any]:
        return await asyncio.to_thread(user_call, "obj_lock", name=name)

    @app.post("/objects/{name}/unlock")
    async def unlock_object(name: str) -> dict[str, Any]:
        return await asyncio.to_thread(user_call, "obj_unlock", name=name)

    @app.post("/objects/{name}/reestimate")
    async def reestimate_object(name: str, req: ReestimateRequest) -> dict[str, Any]:
        """New version of the object with some spec fields changed, e.g. {"sample": "1990q1 2000q4"}."""
        out = await asyncio.to_thread(user_call, "obj_reestimate", name=name, changes=req.changes)
        return clean(engine.ctx.objects.get(out["name"]).to_dict())

    # ---- datasets -----------------------------------------------------------------
    @app.get("/datasets")
    async def list_datasets() -> list[dict[str, Any]]:
        return [d.to_dict() for d in engine.ctx.workspace.list()]

    @app.post("/datasets/import")
    async def import_dataset(req: ImportRequest) -> dict[str, Any]:
        try:
            out = await asyncio.to_thread(user_call, "data_import_file", path=req.path, name=req.name,
                                          sheet=req.sheet)
        except (FileNotFoundError, ValueError) as exc:
            raise HTTPException(400, str(exc)) from exc
        info = engine.ctx.workspace.get(out["id"])
        cols = ", ".join(c["name"] for c in info.columns or [])
        await asyncio.to_thread(engine.remember, "dataset",
                                f"Imported dataset '{info.name}' ({info.n_rows} rows) with columns: {cols}")
        return info.to_dict()

    @app.post("/datasets/upload")
    async def upload_dataset(file: UploadFile) -> dict[str, Any]:
        target = engine.ctx.uploads_dir / Path(file.filename or "upload.csv").name
        with target.open("wb") as fh:
            shutil.copyfileobj(file.file, fh)
        return await import_dataset(ImportRequest(path=str(target)))

    @app.get("/datasets/{dataset_id}")
    async def get_dataset(dataset_id: str) -> dict[str, Any]:
        return engine.ctx.workspace.get(dataset_id).to_dict()

    @app.get("/datasets/{dataset_id}/preview")
    async def preview(dataset_id: str, limit: int = 100, offset: int = 0) -> dict[str, Any]:
        df = engine.ctx.workspace.preview(dataset_id, limit=min(limit, 1000), offset=offset)
        return df_block_data(df, max_rows=1000)

    @app.get("/datasets/{dataset_id}/profile")
    async def profile(dataset_id: str) -> list[dict[str, Any]]:
        df = await asyncio.to_thread(engine.ctx.workspace.df, dataset_id, None, "@all")  # profile ignores the sample
        out = []
        for col in df.columns:
            s = df[col]
            row: dict[str, Any] = {"name": str(col), "dtype": str(s.dtype), "missing": int(s.isna().sum()),
                                   "unique": int(s.nunique())}
            if s.dtype.kind in "iufb":
                row.update(mean=s.mean(), sd=s.std(), min=s.min(), max=s.max())
            out.append(clean(row))
        return out

    @app.put("/datasets/{dataset_id}/panel")
    async def set_panel(dataset_id: str, req: PanelStructureRequest) -> dict[str, Any]:
        try:
            if req.entity_col and req.time_col:
                await asyncio.to_thread(user_call, "data_set_panel_structure", dataset_id=dataset_id,
                                        entity_col=req.entity_col, time_col=req.time_col)
            else:  # an incomplete panel declaration resets the page to a cross-section
                await asyncio.to_thread(user_call, "data_set_structure", dataset_id=dataset_id,
                                        kind="cross_section")
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        return engine.ctx.workspace.get(dataset_id).to_dict()

    @app.get("/datasets/{dataset_id}/structure")
    async def get_structure(dataset_id: str) -> dict[str, Any]:
        return engine.ctx.workspace.describe_structure(dataset_id)

    @app.put("/datasets/{dataset_id}/structure")
    async def set_structure(dataset_id: str, req: StructureRequest) -> dict[str, Any]:
        try:
            await asyncio.to_thread(user_call, "data_set_structure", dataset_id=dataset_id, kind=req.kind,
                                    freq=req.freq, date_col=req.date_col, id_col=req.id_col, t_col=req.t_col,
                                    start=req.start)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        info = engine.ctx.workspace.get(dataset_id)
        await asyncio.to_thread(engine.remember, "dataset", f"Dataset '{info.name}' declared as {info.structure_text}")
        return info.to_dict()

    @app.get("/datasets/{dataset_id}/sample")
    async def get_sample(dataset_id: str) -> dict[str, Any]:
        try:
            return engine.ctx.workspace.resolve_sample(dataset_id).to_dict()
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc

    @app.put("/datasets/{dataset_id}/sample")
    async def set_sample(dataset_id: str, req: SampleRequest) -> dict[str, Any]:
        try:
            await asyncio.to_thread(user_call, "data_set_sample", dataset_id=dataset_id,
                                    sample=req.sample or "@all")
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        res = engine.ctx.workspace.resolve_sample(dataset_id)
        await asyncio.to_thread(engine.remember, "dataset",
                                f"Active sample of '{dataset_id}' set to {res.spec.text}: {res.text}")
        return res.to_dict()

    @app.delete("/datasets/{dataset_id}")
    async def delete_dataset(dataset_id: str) -> dict[str, str]:
        await asyncio.to_thread(user_call, "data_delete_dataset", dataset_id=dataset_id)
        return {"deleted": dataset_id}

    # ---- panel --------------------------------------------------------------------
    @app.get("/panel")
    async def get_panel() -> dict[str, Any]:
        return engine.ctx.panel.snapshot()

    @app.get("/panel/stream")
    async def panel_stream(request: Request) -> StreamingResponse:
        panel = engine.ctx.panel

        async def events() -> AsyncIterator[str]:
            queue = panel.subscribe()
            try:
                yield _sse({"op": "snapshot", **panel.snapshot()})
                while not await request.is_disconnected():
                    try:
                        op = await asyncio.wait_for(queue.get(), timeout=15)
                    except TimeoutError:
                        yield ": keep-alive\n\n"
                        continue
                    if op.op == "reset":
                        yield _sse({"op": "snapshot", **panel.snapshot()})
                    else:
                        yield _sse(op.model_dump())
            finally:
                panel.unsubscribe(queue)

        return StreamingResponse(events(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @app.post("/panel/blocks/{block_id}/pin")
    async def pin(block_id: str, req: PinRequest) -> dict[str, Any]:
        return engine.ctx.panel.pin(block_id, req.pinned).model_dump()

    @app.post("/panel/blocks/{block_id}/freeze")
    async def freeze_block(block_id: str) -> dict[str, Any]:
        """Freeze a live block (the user's action, so pinned blocks too): it stops following its object."""
        try:
            return engine.ctx.panel.freeze(block_id, by_agent=False).model_dump()
        except KeyError as exc:
            raise HTTPException(404, str(exc).strip("'\"")) from exc

    @app.delete("/panel/blocks/{block_id}")
    async def remove_block(block_id: str) -> dict[str, str]:
        engine.ctx.panel.remove(block_id, by_agent=False)
        return {"removed": block_id}

    # The user's own blocks (E7·08d): notes and Python cells written in the panel; agents never modify them.
    def _user_block(block_id: str, kind: str) -> Block:
        try:
            block = engine.ctx.panel.get(block_id)
        except KeyError as exc:
            raise HTTPException(404, str(exc).strip("'\"")) from exc
        if block.kind != kind or block.source != "user":
            raise HTTPException(400, f"Block '{block_id}' is not a {kind} block written by the user")
        return block

    @app.post("/panel/notes")
    async def add_note(req: NoteRequest) -> dict[str, Any]:
        block = Block(kind="markdown", title=req.title.strip() or "Nota", data={"markdown": req.markdown},
                      source="user")
        return engine.ctx.panel.add(block).model_dump()

    @app.put("/panel/blocks/{block_id}/note")
    async def edit_note(block_id: str, req: NoteRequest) -> dict[str, Any]:
        _user_block(block_id, "markdown")
        return engine.ctx.panel.update(block_id, title=req.title.strip() or "Nota", data={"markdown": req.markdown},
                                       by_agent=False).model_dump()

    async def _run_into(block: Block, code: str, timeout: int, edited_from: int | None) -> dict[str, Any]:
        """Run a panel cell and store its outcome in the block (one undo step with the edit that preceded it)."""
        try:
            cell = parse_cell(code)
            run = await run_code(cell, code, cell_id=block.id, edited_from=edited_from, timeout=timeout)
        except (QxError, HTTPException, JournalEntryNotFound) as exc:
            run = {"kind": "invalid", "ok": False, "error": getattr(exc, "detail", None) or str(exc)}
        except Exception as exc:  # noqa: BLE001 - e.g. the kernel cannot start: the cell shows why
            log.exception("Panel cell failed")
            run = {"kind": "invalid", "ok": False, "error": f"{type(exc).__name__}: {exc}"}
        return engine.ctx.panel.update(block.id, data={"run": run}, by_agent=False, checkpoint=False).model_dump()

    def _last_seq(block: Block) -> int | None:
        run = block.data.get("run") or {}
        if run.get("seq") is not None:
            return run["seq"]
        return next((c["seq"] for c in run.get("calls", []) if c.get("seq") is not None), None)

    @app.post("/panel/cells")
    async def add_cell(req: CellRequest) -> dict[str, Any]:
        """A Python cell written by the user in the panel: added first (so what it produces follows it), then run."""
        block = engine.ctx.panel.add(Block(kind="python_cell", title="Python", data={}, code=req.code, source="user"))
        return await _run_into(block, req.code, req.timeout, None)

    @app.put("/panel/blocks/{block_id}/cell")
    async def rerun_cell(block_id: str, req: CellRequest) -> dict[str, Any]:
        """Re-run a user cell, edited or not: the journal links the new run to the previous one."""
        block = _user_block(block_id, "python_cell")
        edited_from = _last_seq(block)
        block = engine.ctx.panel.update(block_id, code=req.code, data={**block.data, "running": True}, by_agent=False)
        return await _run_into(block, req.code, req.timeout, edited_from)

    @app.post("/panel/reorder")
    async def reorder(req: ReorderRequest) -> dict[str, Any]:
        engine.ctx.panel.reorder(req.order)
        return engine.ctx.panel.snapshot()

    @app.post("/panel/undo")
    async def undo() -> dict[str, Any]:
        return {"changed": engine.ctx.panel.undo(), **engine.ctx.panel.snapshot()}

    @app.post("/panel/redo")
    async def redo() -> dict[str, Any]:
        return {"changed": engine.ctx.panel.redo(), **engine.ctx.panel.snapshot()}

    @app.post("/panel/clear")
    async def clear() -> dict[str, Any]:
        engine.ctx.panel.clear()
        return engine.ctx.panel.snapshot()

    # ---- settings & models --------------------------------------------------------
    @app.get("/settings")
    async def get_settings() -> dict[str, Any]:
        providers = {p.provider for p in engine.user.profiles} | {"anthropic", "openai", "google", "mistral"}
        providers -= {"llamacpp", "ollama", "lmstudio"}
        return {**engine.user.model_dump(), "api_keys": {p: has_api_key(p) for p in sorted(providers)},
                "active_is_local": engine.user.active().is_local, "forced_local_only": settings.local_only}

    @app.put("/settings")
    async def put_settings(new: UserSettings) -> dict[str, Any]:
        engine.update_settings(new)
        return await get_settings()

    @app.get("/skills")
    async def skills() -> list[dict[str, str]]:
        """The skills a team member can be given (name and description)."""
        return list_skills()

    @app.post("/settings/api-key")
    async def put_api_key(req: ApiKeyRequest) -> dict[str, bool]:
        set_api_key(req.provider, req.key)
        engine._agent = None
        return {"stored": bool(req.key)}

    @app.get("/models/local")
    async def local_models() -> dict[str, list[str]]:
        found = await detect_local_models()
        cloud = {} if engine.user.local_only else await list_cloud_models()
        return {"llamacpp": [m["id"] for m in engine.models.list()], **found, **cloud}

    @app.post("/models/test")
    async def test_model(profile: ModelProfile) -> dict[str, Any]:
        from pydantic_ai import Agent

        try:
            if profile.provider == "llamacpp":
                path = engine.models.resolve(profile.model)
                url = await asyncio.to_thread(engine.runtime.ensure, path, ctx_size=engine.user.llamacpp.ctx_size)
                if url != engine.runtime_url:
                    engine.runtime_url, engine._agent = url, None
            model = build_model(profile, gateway=engine.gateway, local_only=engine.user.local_only)
            result = await asyncio.wait_for(Agent(model).run("Reply with the single word: ok"), timeout=180)
            return {"ok": True, "reply": str(result.output)[:200]}
        except Exception as exc:  # noqa: BLE001 - report any provider failure to the UI
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    # ---- LLM gateway (LiteLLM): OpenAI-compatible, for the agent and external tools -----
    @app.post("/llm/v1/chat/completions")
    async def llm_chat(request: Request):
        try:
            out = await engine.gateway.chat(await request.json())
        except GatewayError as exc:
            return JSONResponse({"error": {"message": str(exc), "code": exc.status}}, status_code=exc.status)
        if isinstance(out, dict):
            return out
        return StreamingResponse(out, media_type="text/event-stream")

    @app.get("/llm/v1/models")
    async def llm_models() -> dict[str, Any]:
        ids = {model_id(p) for p in engine.user.profiles}
        found = await local_models()
        ids |= {f"{provider}/{m}" for provider, models in found.items() for m in models}
        return {"object": "list", "data": [{"id": i, "object": "model", "owned_by": i.split("/", 1)[0]}
                                           for i in sorted(ids)]}

    @app.get("/llm/usage")
    async def llm_usage() -> dict[str, Any]:
        return engine.gateway.usage()

    # ---- local AI: Hugging Face models + embedded llama.cpp ---------------------------
    @app.get("/localai/status")
    async def localai_status() -> dict[str, Any]:
        return {
            "runtime": engine.runtime.status(),
            "variant": engine.user.llamacpp.variant,
            "variants": list(VARIANTS),
            "ctx_size": engine.user.llamacpp.ctx_size,
            "platform": platform_key(),
            "models_dir": str(settings.models_dir),
            "hf_token": has_api_key("huggingface"),
            "jobs": [j.to_dict() for j in engine.jobs.list()[:30]],
        }

    @app.get("/localai/recommended")
    async def recommended() -> list[dict[str, Any]]:
        local = {Path(m["id"]).name for m in engine.models.list()}
        return [{**r, "installed": r["file"] in local} for r in hub.RECOMMENDED]

    @app.get("/localai/search")
    async def search_models(q: str = "", limit: int = 25) -> list[dict[str, Any]]:
        try:
            return await asyncio.to_thread(hub.search, q, limit=min(limit, 50), token=get_api_key("huggingface"))
        except Exception as exc:  # noqa: BLE001 - network/Hub errors go to the UI
            raise HTTPException(502, f"Hugging Face no respondió: {exc}") from exc

    @app.get("/localai/repo")
    async def repo_listing(repo: str) -> dict[str, Any]:
        try:
            return await asyncio.to_thread(hub.repo_files, repo, token=get_api_key("huggingface"))
        except Exception as exc:  # noqa: BLE001 - not found / gated / network
            raise HTTPException(404, f"No se pudo leer {repo}: {type(exc).__name__}: {exc}") from exc

    @app.post("/localai/download")
    async def start_download(req: DownloadRequest) -> dict[str, Any]:
        return engine.models.start_download(req.repo, req.file).to_dict()

    @app.get("/localai/jobs")
    async def list_jobs() -> list[dict[str, Any]]:
        return [j.to_dict() for j in engine.jobs.list()]

    @app.delete("/localai/jobs/{job_id}")
    async def cancel_job(job_id: str) -> dict[str, Any]:
        try:
            return engine.jobs.cancel(job_id).to_dict()
        except KeyError as exc:
            raise HTTPException(404, f"Unknown job '{job_id}'") from exc

    @app.get("/localai/models")
    async def downloaded_models() -> list[dict[str, Any]]:
        active = engine.user.active()
        serving = engine.runtime.status().get("model")
        return [{**m, "active": active.provider == "llamacpp" and active.model == m["id"],
                 "serving": serving is not None and Path(serving) == Path(m["path"])} for m in engine.models.list()]

    @app.delete("/localai/models")
    async def delete_model(id: str) -> dict[str, str]:
        try:
            path = engine.models.resolve(id)
        except FileNotFoundError as exc:
            raise HTTPException(404, str(exc)) from exc
        serving = engine.runtime.status().get("model")
        if serving is not None and Path(serving) == path:
            await asyncio.to_thread(engine.runtime.stop)
            engine.runtime_url, engine._agent = None, None
        await asyncio.to_thread(engine.models.delete, id)
        return {"deleted": id}

    @app.post("/localai/use")
    async def use_model(req: UseModelRequest) -> dict[str, Any]:
        try:
            engine.use_local_model(req.id)
        except FileNotFoundError as exc:
            raise HTTPException(404, str(exc)) from exc
        return await get_settings()

    @app.post("/localai/runtime/install")
    async def install_runtime(req: RuntimeInstallRequest) -> dict[str, Any]:
        try:
            job = engine.runtime.start_install(req.variant)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        engine.user.llamacpp.variant = req.variant  # type: ignore[assignment]
        engine.update_settings(engine.user)
        return job.to_dict()

    @app.post("/localai/runtime/config")
    async def configure_runtime(req: RuntimeConfigRequest) -> dict[str, Any]:
        """Change the context window; a model being served is reloaded with it on the next chat turn."""
        engine.user.llamacpp.ctx_size = req.ctx_size
        engine.update_settings(engine.user)
        return await localai_status()

    @app.post("/localai/runtime/stop")
    async def stop_runtime() -> dict[str, Any]:
        await asyncio.to_thread(engine.runtime.stop)
        engine.runtime_url, engine._agent = None, None
        return engine.runtime.status()

    # ---- chat ---------------------------------------------------------------------
    @app.get("/chat/history")
    async def chat_history() -> list[dict[str, Any]]:
        return engine.chat.transcript

    @app.post("/chat/reset")
    async def chat_reset() -> dict[str, bool]:
        engine.chat.reset()
        return {"ok": True}

    # ---- next steps: what is left to do in the conversation, shared by the user and the agent -----------------------
    @app.get("/chat/next-steps")
    async def next_steps_list() -> list[dict[str, Any]]:
        return engine.chat.next_steps.dump()

    @app.post("/chat/next-steps")
    async def next_steps_add(req: NextStepRequest) -> list[dict[str, Any]]:
        try:
            return engine.chat.edit_next_steps(add=req.text or "")
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc

    @app.patch("/chat/next-steps/{step_id}")
    async def next_steps_edit(step_id: str, req: NextStepRequest) -> list[dict[str, Any]]:
        try:
            return engine.chat.edit_next_steps(ref=step_id, done=req.done, text=req.text)
        except KeyError as exc:
            raise HTTPException(404, f"No next step '{step_id}'.") from exc

    @app.delete("/chat/next-steps/{step_id}")
    async def next_steps_delete(step_id: str) -> list[dict[str, Any]]:
        try:
            return engine.chat.edit_next_steps(ref=step_id, remove=True)
        except KeyError as exc:
            raise HTTPException(404, f"No next step '{step_id}'.") from exc

    @app.get("/chat/context")
    async def chat_context() -> dict[str, Any]:
        """The context meter: tokens the conversation fills, the model's window and where it auto-compacts."""
        return {**engine.chat.context(engine.context_window()), "compact_at": engine.user.agent.compact_at,
                "auto_compact": engine.user.agent.auto_compact}

    @app.post("/chat/compact")
    async def chat_compact() -> dict[str, Any]:
        """Fold the conversation into a summary now (the meter's button)."""
        if not engine.chat.history:
            return await chat_context()
        try:
            engine.validate_model()
        except (ProviderBlocked, ValueError, RuntimeError) as exc:
            raise HTTPException(400, str(exc)) from exc
        async with engine._chat_lock:
            await engine.prepare_model()
            await engine.chat.compact(engine.agent(), engine.context_window())
        return await chat_context()

    @app.post("/chat/plan")
    async def chat_plan(req: PlanRequest) -> dict[str, str]:
        """Plan mode: draft the detailed plan of a request for the user to review. Nothing runs and the conversation
        is not touched; the approved plan comes back with the request through ``POST /chat``."""
        try:
            engine.validate_model()
        except (ProviderBlocked, ValueError, RuntimeError) as exc:
            raise HTTPException(400, str(exc)) from exc
        async with engine._chat_lock:
            try:
                await engine.prepare_model()
                plan = await engine.chat.draft_plan(engine.agent(), req.message)
            except Exception as exc:  # noqa: BLE001 - tell the UI why there is no plan
                log.exception("Plan drafting failed")
                raise HTTPException(502, f"{type(exc).__name__}: {exc}") from exc
        if not plan:
            raise HTTPException(502, "The model returned an empty plan.")
        return {"plan": plan}

    @app.post("/chat")
    async def chat(req: ChatRequest) -> StreamingResponse:
        try:
            engine.validate_model()
        except (ProviderBlocked, ValueError, RuntimeError) as exc:
            raise HTTPException(400, str(exc)) from exc
        thread = engine.ctx.thread
        if thread.title == DEFAULT_THREAD_TITLE and not thread.chat.transcript:
            thread.title = title_from(req.message)
            engine.ctx.autosave.save_meta(thread)

        async def events() -> AsyncIterator[str]:
            async with engine._chat_lock:
                tool_log: list[str] = []
                answer: list[str] = []
                try:
                    if engine.needs_loading():
                        name = Path(engine.user.active().model).name.removesuffix(".gguf")
                        yield _sse({"type": "status", "message": f"Cargando {name} en llama.cpp…"})
                    await engine.prepare_model()
                    agent = engine.agent()
                    team, warnings = engine.team()
                    for warning in warnings:
                        yield _sse({"type": "status", "message": warning})
                    team_worked = False
                    async for event in engine.chat.stream(agent, req.message,
                                                          max_tool_calls=engine.user.agent.max_tool_calls,
                                                          plan=engine.user.agent.planner,
                                                          context_window=engine.context_window()
                                                          if engine.user.agent.auto_compact else None,
                                                          compact_at=engine.user.agent.compact_at,
                                                          approved_plan=req.plan or "", team=team):
                        if event["type"] == "agent_spawn":
                            team_worked = True
                        elif "agent" in event:  # a member's work: the turn's answer and memory are the lead's
                            pass
                        elif event["type"] == "text":
                            answer.append(event["delta"])
                        elif event["type"] == "tool_call":
                            tool_log.append(f"{event['name']}({json.dumps(event['args'], default=str)[:300]})")
                        yield _sse(event)
                    # The team's conclusion is the conversation's report: a team turn always brings it up to date.
                    if thread.reporting or team_worked:
                        async for ev in engine.reporting_events(thread, full=thread.report is None):
                            yield _sse(ev)
                except Exception as exc:  # noqa: BLE001 - stream the failure to the UI instead of dropping
                    log.exception("Chat run failed")
                    yield _sse({"type": "error", "message": f"{type(exc).__name__}: {exc}"})
                finally:
                    await asyncio.to_thread(engine.remember, "user", req.message)
                    if tool_log:
                        await asyncio.to_thread(engine.remember, "tool", "Tools run: " + "; ".join(tool_log))
                    if answer:
                        await asyncio.to_thread(engine.remember, "assistant", "".join(answer))

        return StreamingResponse(events(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})

    return app
