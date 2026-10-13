"""Entry point used by the desktop shell: ``python -m qtrix_engine --port 0``.

Prints one JSON line ``{"event": "ready", "port": N}`` on stdout once the server is listening, so the
Electron main process knows where to connect.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import socket
import sys
import threading
import time
from logging.handlers import RotatingFileHandler

import uvicorn

from .api import create_app
from .config import Settings
from .localai.llamacpp import shutdown_all


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def pid_alive(pid: int) -> bool:
    """True while a process with this pid is running (cross-platform, never signals the process)."""
    if sys.platform == "win32":
        import ctypes

        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            return False
        try:
            code = ctypes.c_ulong()
            return bool(kernel32.GetExitCodeProcess(handle, ctypes.byref(code))) and code.value == 259  # STILL_ACTIVE
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _exit_with_parent(pid: int) -> None:
    """Stop the engine when the desktop shell disappears, so a crashed/killed shell never leaves an
    orphan holding the workspace database open."""
    while pid_alive(pid):
        time.sleep(2)
    logging.getLogger(__name__).info("Desktop shell (pid %s) is gone; shutting down", pid)
    shutdown_all()  # os._exit skips atexit: stop llama-server explicitly
    os._exit(0)


def main() -> None:
    parser = argparse.ArgumentParser(prog="qtrix-engine")
    parser.add_argument("--port", type=int, default=0, help="0 picks a free port")
    parser.add_argument("--log-level", default="info")
    args = parser.parse_args()

    settings = Settings.from_env()
    fmt = "%(asctime)s %(levelname)s %(name)s: %(message)s"
    file_log = RotatingFileHandler(settings.logs_dir / "engine.log", maxBytes=5_000_000, backupCount=3,
                                   encoding="utf-8")
    file_log.setFormatter(logging.Formatter(fmt))
    logging.basicConfig(level=args.log_level.upper(), format=fmt,
                        handlers=[logging.StreamHandler(sys.stderr), file_log])
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger(__name__).info("qtrix engine starting; data folder: %s", settings.home)
    port = args.port or _free_port()
    parent = os.environ.get("QTRIX_PARENT_PID")
    if parent and parent.isdigit():
        threading.Thread(target=_exit_with_parent, args=(int(parent),), name="qtrix-parent-watch", daemon=True).start()
    try:
        app = create_app(settings)
    except Exception:
        logging.getLogger(__name__).exception("Engine failed to start")
        raise

    app.state.engine.url = f"http://127.0.0.1:{port}"  # the Python kernel calls the engine back here
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level=args.log_level, access_log=False,
                            log_config=None)
    _AnnouncingServer(config).run()


class _AnnouncingServer(uvicorn.Server):
    async def startup(self, sockets=None) -> None:
        await super().startup(sockets=sockets)
        if self.started:
            home = str(self.config.app.state.engine.settings.home)
            print(json.dumps({"event": "ready", "port": self.config.port, "home": home}), flush=True)


if __name__ == "__main__":
    main()
