"""Embedded llama.cpp runtime: install the official llama-server build and serve GGUF models with it.

llama-server exposes an OpenAI-compatible API (tool calling via ``--jinja``) on 127.0.0.1, so the
agents talk to it like to any other OpenAI-compatible model.
"""

from __future__ import annotations

import atexit
import json
import logging
import platform
import re
import shutil
import socket
import subprocess
import sys
import tarfile
import threading
import time
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from .jobs import Job, JobManager, download

log = logging.getLogger(__name__)

RELEASES_URL = "https://api.github.com/repos/ggml-org/llama.cpp/releases?per_page=30"
VARIANTS = ("cpu", "vulkan", "cuda")
_BUILD_TAG = re.compile(r"^b\d+$")


def platform_key(system: str | None = None, machine: str | None = None) -> str:
    system = (system or platform.system()).lower()
    machine = (machine or platform.machine()).lower()
    arch = "arm64" if machine in ("arm64", "aarch64") else "x64"
    return {"windows": "win", "darwin": "macos"}.get(system, "linux") + "-" + arch


def asset_names(tag: str, variant: str, key: str | None = None) -> list[str]:
    """Release assets needed for this platform/variant (the first one contains llama-server)."""
    key = key or platform_key()
    arch = key.split("-")[1]
    if key.startswith("macos"):
        return [f"llama-{tag}-bin-{key}.tar.gz"]  # Metal is built in on macOS
    if key.startswith("win"):
        if variant == "cuda":
            cuda = "12.4" if arch == "x64" else "13.4"
            return [f"llama-{tag}-bin-win-cuda-{cuda}-{arch}.zip", f"cudart-llama-bin-win-cuda-{cuda}-{arch}.zip"]
        return [f"llama-{tag}-bin-win-{'vulkan' if variant == 'vulkan' else 'cpu'}-{arch}.zip"]
    if variant == "cuda":
        cuda = "12.8" if arch == "x64" else "13.4"
        return [f"llama-{tag}-bin-ubuntu-cuda-{cuda}-{arch}.tar.gz",
                f"cudart-llama-{tag}-bin-ubuntu-cuda-{cuda}-{arch}.tar.gz"]
    return [f"llama-{tag}-bin-ubuntu-{'vulkan-' if variant == 'vulkan' else ''}{arch}.tar.gz"]


def pick_release(releases: list[dict[str, Any]], variant: str, key: str | None = None) -> tuple[str, list[dict]]:
    """Newest bNNNN build that ships every asset we need."""
    for rel in releases:
        tag = rel.get("tag_name", "")
        if not _BUILD_TAG.match(tag):
            continue
        by_name = {a["name"]: a for a in rel.get("assets", [])}
        wanted = asset_names(tag, variant, key)
        if all(n in by_name for n in wanted):
            return tag, [by_name[n] for n in wanted]
    raise LookupError(f"No hay ningún build reciente de llama.cpp para {key or platform_key()} ({variant})")


def _extract(archive: Path, dest: Path) -> None:
    if archive.name.endswith(".zip"):
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(dest)
    else:
        with tarfile.open(archive) as tf:
            tf.extractall(dest, filter="data")


def _server_binary(root: Path) -> Path | None:
    name = "llama-server.exe" if sys.platform == "win32" else "llama-server"
    hits = sorted(root.rglob(name), key=lambda p: len(p.parts))
    return hits[0] if hits else None


_KILL_ON_CLOSE_JOB = None


def _tie_to_this_process(proc: subprocess.Popen) -> None:
    """Make the OS kill ``proc`` whenever this process dies, even on a hard kill (no orphan servers).

    Windows: put it in a Job Object with KILL_ON_JOB_CLOSE (the job handle closes when we die).
    Linux uses PR_SET_PDEATHSIG at spawn time instead (see ``_linux_pdeathsig``).
    """
    global _KILL_ON_CLOSE_JOB
    if sys.platform != "win32":
        return
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateJobObjectW.restype = wintypes.HANDLE
    kernel32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    if _KILL_ON_CLOSE_JOB is None:

        class BasicLimits(ctypes.Structure):
            _fields_ = [("PerProcessUserTimeLimit", ctypes.c_int64), ("PerJobUserTimeLimit", ctypes.c_int64),
                        ("LimitFlags", wintypes.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
                        ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wintypes.DWORD),
                        ("Affinity", ctypes.c_size_t), ("PriorityClass", wintypes.DWORD),
                        ("SchedulingClass", wintypes.DWORD)]

        class ExtendedLimits(ctypes.Structure):
            _fields_ = [("BasicLimitInformation", BasicLimits), ("IoInfo", ctypes.c_ulonglong * 6),
                        ("ProcessMemoryLimit", ctypes.c_size_t), ("JobMemoryLimit", ctypes.c_size_t),
                        ("PeakProcessMemoryUsed", ctypes.c_size_t), ("PeakJobMemoryUsed", ctypes.c_size_t)]

        job = kernel32.CreateJobObjectW(None, None)
        limits = ExtendedLimits()
        limits.BasicLimitInformation.LimitFlags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not job or not kernel32.SetInformationJobObject(job, 9, ctypes.byref(limits), ctypes.sizeof(limits)):
            log.warning("Could not create kill-on-close job object (error %s)", ctypes.get_last_error())
            return
        _KILL_ON_CLOSE_JOB = job  # keep the handle open for the whole engine lifetime
    if not kernel32.AssignProcessToJobObject(_KILL_ON_CLOSE_JOB, int(proc._handle)):  # type: ignore[attr-defined]
        log.warning("Could not tie llama-server to the engine (error %s)", ctypes.get_last_error())


def _linux_pdeathsig() -> None:  # runs in the child between fork and exec
    import ctypes
    import signal

    ctypes.CDLL("libc.so.6", use_errno=True).prctl(1, signal.SIGTERM)  # PR_SET_PDEATHSIG


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class LlamaRuntime:
    """Installs and runs llama-server from <qtrix>/runtime/llama.cpp."""

    def __init__(self, runtime_dir: Path, logs_dir: Path, jobs: JobManager, *, server_cmd: list[str] | None = None):
        self.root = runtime_dir / "llama.cpp"
        self.logs_dir = logs_dir
        self.jobs = jobs
        self._server_cmd_override = server_cmd  # tests inject a fake server
        self._proc: subprocess.Popen | None = None
        self._model: Path | None = None
        self._port: int | None = None
        self._ctx_size: int | None = None
        self._lock = threading.Lock()
        _RUNTIMES.append(self)

    # ---- install -------------------------------------------------------------------
    def installed(self) -> dict[str, Any] | None:
        if self._server_cmd_override:
            return {"tag": "test", "variant": "cpu", "server": self._server_cmd_override[0]}
        try:
            info = json.loads((self.root / "installed.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        return info if Path(info.get("server", "")).is_file() else None

    def start_install(self, variant: str) -> Job:
        if variant not in VARIANTS:
            raise ValueError(f"Variante desconocida '{variant}' (usa {', '.join(VARIANTS)})")
        running = self.jobs.active("runtime")
        if running:
            return running[0]
        return self.jobs.start("runtime", f"llama.cpp ({variant})", lambda job: self._install(job, variant))

    def _install(self, job: Job, variant: str) -> dict[str, Any]:
        job.message = "Buscando el último build de llama.cpp…"
        resp = httpx.get(RELEASES_URL, timeout=30, follow_redirects=True,
                         headers={"Accept": "application/vnd.github+json"})
        resp.raise_for_status()
        tag, assets = pick_release(resp.json(), variant)
        target = self.root / f"{tag}-{variant}"
        downloads = self.root / "downloads"
        job.total = sum(a["size"] for a in assets)
        for asset in assets:
            job.message = f"Descargando {asset['name']}"
            archive = download(asset["browser_download_url"], downloads / asset["name"], job)
            job.message = f"Descomprimiendo {asset['name']}"
            _extract(archive, target)
            archive.unlink(missing_ok=True)
        server = _server_binary(target)
        if server is None:
            raise FileNotFoundError("El paquete de llama.cpp no contiene llama-server")
        if sys.platform != "win32":
            server.chmod(0o755)
        if variant == "cuda":
            for lib in [*target.rglob("*.dll"), *target.rglob("*.so*")]:  # CUDA runtime next to the server
                if lib.parent != server.parent and not (server.parent / lib.name).exists():
                    shutil.copy2(lib, server.parent / lib.name)
        info = {"tag": tag, "variant": variant, "server": str(server), "installed_at": datetime.now(UTC).isoformat()}
        self.stop()
        (self.root / "installed.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
        for old in self.root.iterdir():  # keep only the active build
            if old.is_dir() and old not in (target, downloads):
                shutil.rmtree(old, ignore_errors=True)
        job.message = f"llama.cpp {tag} ({variant}) instalado"
        return info

    # ---- server --------------------------------------------------------------------
    @property
    def url(self) -> str | None:
        return f"http://127.0.0.1:{self._port}/v1" if self._port else None

    def running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def status(self) -> dict[str, Any]:
        running = self.running()
        return {"installed": self.installed(), "running": running,
                "model": str(self._model) if running and self._model else None,
                "url": self.url if running else None}

    def is_serving(self, model: Path, ctx_size: int) -> bool:
        return self.running() and self._model == model and self._ctx_size == ctx_size

    def _command(self, model: Path, port: int, ctx_size: int) -> list[str]:
        info = self.installed()
        if info is None:
            raise RuntimeError("El runtime llama.cpp no está instalado: instálalo desde Modelos")
        base = list(self._server_cmd_override) if self._server_cmd_override else [info["server"]]
        cmd = base + ["-m", str(model), "--host", "127.0.0.1", "--port", str(port), "--jinja",
                      "-c", str(ctx_size), "--alias", model.stem]
        if info.get("variant") in ("vulkan", "cuda"):
            cmd += ["-ngl", "999"]  # offload every layer to the GPU
        return cmd

    def ensure(self, model: Path, *, ctx_size: int = 8192, timeout: float = 300) -> str:
        """Serve ``model`` (blocking until healthy) and return the OpenAI-compatible base URL."""
        with self._lock:
            if self.is_serving(model, ctx_size):
                return self.url  # type: ignore[return-value]
            self.stop()
            port = _free_port()
            cmd = self._command(model, port, ctx_size)
            self.logs_dir.mkdir(parents=True, exist_ok=True)
            with (self.logs_dir / "llama-server.log").open("a", encoding="utf-8") as fh:
                fh.write(f"\n=== {datetime.now(UTC).isoformat()} {' '.join(cmd)}\n")
            log_file = (self.logs_dir / "llama-server.log").open("ab")
            flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if sys.platform == "win32" else 0
            self._proc = subprocess.Popen(
                cmd, stdout=log_file, stderr=subprocess.STDOUT, creationflags=flags,
                preexec_fn=_linux_pdeathsig if sys.platform.startswith("linux") else None,
            )
            log_file.close()  # the child keeps its own handle
            _tie_to_this_process(self._proc)
            self._model, self._port, self._ctx_size = model, port, ctx_size
            deadline = time.time() + timeout
            while time.time() < deadline:
                if self._proc.poll() is not None:
                    code = self._proc.returncode
                    self._proc = None
                    raise RuntimeError(f"llama-server terminó con código {code}. Últimas líneas:\n{self._log_tail()}")
                try:
                    if httpx.get(f"http://127.0.0.1:{port}/health", timeout=2).status_code == 200:
                        log.info("llama-server ready on port %s with %s", port, model.name)
                        return self.url  # type: ignore[return-value]
                except httpx.HTTPError:
                    pass
                time.sleep(0.5)
            self.stop()
            raise TimeoutError(f"llama-server no estuvo listo en {timeout:.0f} s")

    def stop(self) -> None:
        proc, self._proc = self._proc, None
        if proc is None or proc.poll() is not None:
            return
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()

    def _log_tail(self, n: int = 1200) -> str:
        try:
            return (self.logs_dir / "llama-server.log").read_text(encoding="utf-8", errors="replace")[-n:]
        except OSError:
            return ""


_RUNTIMES: list[LlamaRuntime] = []


def shutdown_all() -> None:
    """Stop every llama-server this process started (normal exit and parent-death exit)."""
    for rt in _RUNTIMES:
        try:
            rt.stop()
        except Exception:  # noqa: BLE001 - best effort during shutdown
            pass


atexit.register(shutdown_all)
