"""The project's IPython kernel, managed with ``jupyter_client`` (ADR 0002 §7.4, E7·07b).

The kernel is a separate process started on demand. It runs with the user's privileges and is **not a sandbox**:
a cell can read files, open sockets and use the network like any script the user runs. It never opens the
workfile: DuckDB allows one writing process, the engine, so ``qx`` in the kernel runs in client mode and calls the
engine over loopback HTTP with the session token (``qtrix_engine.qx.client``).

Every cell runs under a *run* the engine registers first (``KernelPool.runs``): the kernel's ``qx`` sends the run
id with each call, so the engine knows who wrote the cell (the user or an agent), which chat or notebook cell the
calls belong to, and which journal entries the cell produced. A *rebuild* run (re-running a registered result's
cell to rebuild it) uses a second, fresh kernel; there the engine skips every call that would change the workfile and
captures the summary that ``qx.register`` hands over.
"""

from __future__ import annotations

import logging
import queue
import re
import secrets
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

KEEP_MIME = ("text/plain", "text/html", "text/markdown", "image/png", "image/svg+xml", "application/json",
             "application/vnd.plotly.v1+json")
_ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
INTERRUPT_GRACE = 5.0  # seconds a timed-out cell gets to honour the interrupt before the kernel is restarted
BOOTSTRAP = """
import qtrix as qx
qx.connect()
try:
    import plotly.io as _qx_pio
    _qx_pio.renderers.default = "plotly_mimetype"
    del _qx_pio
except Exception:
    pass
try:
    get_ipython().run_line_magic("matplotlib", "inline")
except Exception:
    pass
"""


class KernelError(RuntimeError):
    """The kernel is unavailable, died, or a cell could not be run."""


class CellFailed(KernelError):
    """The cell raised (or timed out). ``result`` holds its outputs and the error, for the caller to show."""

    def __init__(self, message: str, result: dict[str, Any]) -> None:
        super().__init__(message)
        self.result = result


@dataclass
class KernelRun:
    """One cell being run: who wrote it, where it belongs, and what it did through the dispatcher."""

    actor: str
    cell_id: str | None
    code: str
    rebuild: str | None = None  # name whose qx.register summary a rebuild run captures
    id: str = field(default_factory=lambda: secrets.token_hex(8))
    calls: list[int] = field(default_factory=list)  # journal seqs of the calls the cell made
    mutated: bool = False
    captured: dict[str, Any] | None = None  # rebuild: the summary qx.register handed over


@dataclass
class CellResult:
    ok: bool
    outputs: list[dict[str, Any]]
    error: dict[str, Any] | None = None
    execution_count: int | None = None
    duration_ms: int = 0
    timed_out: bool = False

    def error_text(self) -> str:
        if self.error is None:
            return ""
        return f"{self.error['ename']}: {self.error['evalue']}"

    def text(self, limit: int = 4000) -> str:
        """Compact plain-text rendering for an agent: streams and text results; rich outputs are only named."""
        parts: list[str] = []
        for out in self.outputs:
            if out["type"] == "stream":
                parts.append(out["text"])
            elif "text/plain" in out.get("data", {}):
                rich = [m for m in out["data"] if m != "text/plain"]
                parts.append(out["data"]["text/plain"] + (f"  [{', '.join(rich)}]" if rich else ""))
            elif out.get("data"):
                parts.append(f"[{', '.join(out['data'])}]")
        if self.error:
            parts.append("\n".join(self.error["traceback"][-6:]) or self.error_text())
        text = "\n".join(p.rstrip("\n") for p in parts)
        return text if len(text) <= limit else text[:limit] + f"\n… ({len(text) - limit} more characters)"


class ProjectKernel:
    """One IPython kernel process: start on demand, run cells one at a time, interrupt, restart, shut down."""

    def __init__(self, env: Callable[[], dict[str, str]], cwd: Path | None = None, *, name: str = "project",
                 bootstrap: str = BOOTSTRAP) -> None:
        self._env, self._cwd, self.name, self._bootstrap = env, cwd, name, bootstrap
        self._km: Any = None
        self._kc: Any = None
        self._start_lock = threading.RLock()
        self._run_lock = threading.Lock()
        self.started_at: float | None = None
        self.execution_count = 0
        self._last_activity = time.monotonic()

    # ---- lifecycle -------------------------------------------------------------------------------------------
    @property
    def alive(self) -> bool:
        return self._km is not None and self._km.is_alive()

    @property
    def busy(self) -> bool:
        return self._run_lock.locked()

    def start(self) -> None:
        with self._start_lock:
            if self.alive:
                return
            try:
                from jupyter_client.manager import KernelManager
            except ImportError as exc:
                raise KernelError("The Python kernel needs jupyter_client and ipykernel installed") from exc
            km = KernelManager(kernel_name="python3")
            km.kernel_spec.argv[0] = sys.executable  # the engine's interpreter: same packages, same qtrix
            km.start_kernel(env=self._env(), cwd=str(self._cwd) if self._cwd else None)
            kc = km.blocking_client()
            kc.start_channels()
            try:
                kc.wait_for_ready(timeout=60)
            except RuntimeError as exc:
                kc.stop_channels()
                km.shutdown_kernel(now=True)
                raise KernelError(f"The Python kernel did not start: {exc}") from exc
            self._km, self._kc, self.started_at, self.execution_count = km, kc, time.time(), 0
            self._prepare()
            log.info("Python kernel '%s' started", self.name)

    def _prepare(self) -> None:
        boot = self._execute(self._bootstrap, silent=True, timeout=60)
        if not boot.ok:
            raise KernelError(f"The Python kernel could not load qtrix: {boot.error_text()}")

    def interrupt(self) -> bool:
        """Interrupt the running cell (KeyboardInterrupt in the kernel). False when nothing was running."""
        if not self.alive or not self.busy:
            return False
        self._km.interrupt_kernel()
        return True

    def restart(self) -> None:
        """Fresh namespace: variables are lost, the workfile is untouched."""
        with self._start_lock:
            if not self.alive:
                self.start()
                return
            self._km.restart_kernel(now=True)
            self._kc.wait_for_ready(timeout=60)
            self.started_at, self.execution_count = time.time(), 0
            self._prepare()

    def _hard_restart(self) -> None:
        """Restart a kernel stuck in a cell (called while that cell holds the run lock)."""
        with self._start_lock:
            self._km.restart_kernel(now=True)
            self._kc.wait_for_ready(timeout=60)
            self.started_at, self.execution_count = time.time(), 0
            self._prepare()

    def shutdown(self) -> None:
        with self._start_lock:
            if self._kc is not None:
                self._kc.stop_channels()
            if self._km is not None:
                try:
                    self._km.shutdown_kernel(now=True)
                except Exception:  # noqa: BLE001 - shutting down a dead kernel must not fail the caller
                    log.exception("Kernel shutdown failed")
            self._km = self._kc = self.started_at = None

    def status(self) -> dict[str, Any]:
        return {"name": self.name, "alive": self.alive, "busy": self.busy, "started_at": self.started_at,
                "execution_count": self.execution_count}

    # ---- running cells ---------------------------------------------------------------------------------------
    def idle_for(self) -> float:
        """Seconds since the kernel last started or finished running a cell."""
        return time.monotonic() - self._last_activity

    def run(self, code: str, *, run_id: str | None = None, timeout: float = 120) -> CellResult:
        """Run a cell (one at a time), tagging the qx calls it makes with ``run_id``."""
        self.start()
        self._last_activity = time.monotonic()
        with self._run_lock:
            if run_id:
                self._execute(f"__import__('qtrix').set_run({run_id!r})", silent=True, timeout=30)
            try:
                result = self._execute(code, timeout=timeout)
            finally:
                if run_id and self.alive:
                    self._execute("__import__('qtrix').set_run(None)", silent=True, timeout=30)
        if result.execution_count:
            self.execution_count = result.execution_count
        self._last_activity = time.monotonic()
        return result

    def _execute(self, code: str, *, silent: bool = False, timeout: float | None = None) -> CellResult:
        kc = self._kc
        started = time.monotonic()
        msg_id = kc.execute(code, silent=silent, store_history=not silent, allow_stdin=False)
        outputs: list[dict[str, Any]] = []
        error: dict[str, Any] | None = None
        count: int | None = None
        timed_out = False
        deadline = started + timeout if timeout else None
        while True:
            try:
                msg = kc.get_iopub_msg(timeout=0.5)
            except queue.Empty:
                now = time.monotonic()
                if not self.alive:
                    raise KernelError("The Python kernel died while running the cell") from None
                if deadline is not None and now > deadline:
                    if timed_out:  # the interrupt was not honoured (native code, e.g. time.sleep on Windows)
                        self._hard_restart()
                        return CellResult(ok=False, outputs=outputs, error={
                            "ename": "TimeoutError", "evalue": f"the cell ran for more than {timeout:.0f} s and did "
                            "not stop when interrupted, so the kernel was restarted: its variables are lost",
                            "traceback": []}, execution_count=count,
                            duration_ms=round((time.monotonic() - started) * 1000), timed_out=True)
                    timed_out = True
                    self._km.interrupt_kernel()
                    deadline = now + INTERRUPT_GRACE
                continue
            if msg["parent_header"].get("msg_id") != msg_id:
                continue
            kind, content = msg["msg_type"], msg["content"]
            if kind == "status" and content.get("execution_state") == "idle":
                break
            if kind == "execute_input":
                count = content.get("execution_count")
            elif kind == "stream":
                if outputs and outputs[-1]["type"] == "stream" and outputs[-1]["name"] == content["name"]:
                    outputs[-1]["text"] += content["text"]
                else:
                    outputs.append({"type": "stream", "name": content["name"], "text": content["text"]})
            elif kind in ("execute_result", "display_data"):
                data = {m: v for m, v in content.get("data", {}).items() if m in KEEP_MIME}
                if data:
                    outputs.append({"type": "result" if kind == "execute_result" else "display", "data": data})
            elif kind == "error":
                error = {"ename": content["ename"], "evalue": content["evalue"],
                         "traceback": [_ANSI.sub("", line) for line in content.get("traceback", [])]}
        status = self._reply_status(msg_id)
        if timed_out and error is None:
            error = {"ename": "TimeoutError", "evalue": f"the cell ran for more than {timeout:.0f} s and was "
                     "interrupted", "traceback": []}
        elif timed_out:
            error["evalue"] = f"{error['evalue']} (interrupted after {timeout:.0f} s)"
        return CellResult(ok=status == "ok" and error is None, outputs=outputs, error=error, execution_count=count,
                          duration_ms=round((time.monotonic() - started) * 1000), timed_out=timed_out)

    def _reply_status(self, msg_id: str) -> str:
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            try:
                reply = self._kc.get_shell_msg(timeout=1)
            except queue.Empty:
                continue
            if reply["parent_header"].get("msg_id") == msg_id:
                return reply["content"].get("status", "error")
        return "error"


class KernelPool:
    """The project kernel plus a scratch kernel for rebuilds, and the runs in flight."""

    def __init__(self, env: Callable[[], dict[str, str]], cwd: Path | None = None) -> None:
        self._env, self._cwd = env, cwd
        self.project = ProjectKernel(env, cwd, name="project")
        self._rebuild: ProjectKernel | None = None
        self._rebuild_lock = threading.Lock()
        self.runs: dict[str, KernelRun] = {}

    def run_cell(self, code: str, *, actor: str, cell_id: str | None,
                 timeout: float = 120) -> tuple[CellResult, KernelRun]:
        run = KernelRun(actor=actor, cell_id=cell_id, code=code)
        self.runs[run.id] = run
        try:
            return self.project.run(code, run_id=run.id, timeout=timeout), run
        finally:
            self.runs.pop(run.id, None)

    def rebuild(self, code: str, name: str, *, timeout: float = 300) -> dict[str, Any]:
        """Re-run a registered result's cell in a fresh namespace and return the summary it registers as ``name``."""
        with self._rebuild_lock:
            if self._rebuild is None:
                self._rebuild = ProjectKernel(self._env, self._cwd, name="rebuild")
            kernel = self._rebuild
            kernel.start()
            kernel.run("%reset -f\nimport qtrix as qx", timeout=60)
            run = KernelRun(actor="user", cell_id=None, code=code, rebuild=name.lower())
            self.runs[run.id] = run
            try:
                result = kernel.run(code, run_id=run.id, timeout=timeout)
            finally:
                self.runs.pop(run.id, None)
        if not result.ok:
            raise KernelError(f"Re-running the cell of '{name}' failed: {result.error_text()}. A registered result is "
                              "rebuilt from its cell alone, in a fresh kernel: the cell must import what it uses and "
                              "read its data with qx.page(...).frame()")
        if run.captured is None:
            raise KernelError(f"Re-running the cell did not call qx.register({name!r}, ...)")
        return run.captured

    def shutdown_idle(self, max_idle: float) -> bool:
        """Stop the kernels that have been idle for ``max_idle`` seconds (their variables are lost; the notebook,
        its objects and its journal are not). True if any was stopped."""
        stopped = False
        for kernel in (self.project, self._rebuild):
            if kernel is not None and kernel.alive and not kernel.busy and kernel.idle_for() >= max_idle:
                kernel.shutdown()
                stopped = True
        return stopped

    def status(self) -> dict[str, Any]:
        return {**self.project.status(), "running": [{"actor": r.actor, "cell_id": r.cell_id}
                                                     for r in self.runs.values() if r.rebuild is None]}

    def shutdown(self) -> None:
        self.project.shutdown()
        if self._rebuild is not None:
            self._rebuild.shutdown()
