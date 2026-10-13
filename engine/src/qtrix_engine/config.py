"""The qtrix folder: one user-visible directory that holds everything qtrix stores.

    <qtrix>/
      settings.json         model profiles and preferences (API keys live in the OS keychain, not here)
      notebooks/<id>/       one folder per notebook: DuckDB (datasets, objects, journal), meta.json, uploads/
      data/                 the single workspace of earlier versions, migrated into a notebook on first start
      memory/               turbovec semantic memory of the project
      models/               GGUF models downloaded from Hugging Face
      runtime/              local inference runtimes (llama.cpp)
      logs/                 engine.log, llama-server.log
      uploads/  exports/

Where the folder is: ``QTRIX_HOME`` if set, otherwise the path stored in the bootstrap pointer
``<config dir>/qtrix/location.json`` (the same directory Electron uses as userData), otherwise
``~/qtrix``. The pointer lets the user move the folder anywhere from the app.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)

APP_NAME = "qtrix"
LAYOUT: dict[str, str] = {
    "data": "Base de datos de versiones anteriores (se migra a un cuaderno)",
    "notebooks": "Cuadernos: datasets, objetos, diario y ficheros de cada uno",
    "memory": "Memoria semántica del proyecto (turbovec)",
    "models": "Modelos GGUF descargados de Hugging Face",
    "runtime": "Runtimes de inferencia local (llama.cpp)",
    "logs": "Registros del motor y de llama-server",
    "uploads": "Ficheros subidos a la app",
    "exports": "Scripts, notebooks e informes exportados",
}
README = """qtrix — carpeta de datos local

Todo lo que qtrix guarda está aquí y nunca sale de esta máquina:

{entries}
  settings.json  perfiles de modelos y preferencias (las API keys están en el llavero del sistema)

Puedes cambiar esta carpeta desde la app: Ajustes > Carpeta de qtrix.
"""


def bootstrap_dir() -> Path:
    """Fixed per-user config directory holding location.json (matches Electron's userData)."""
    if sys.platform == "win32":
        base = Path(os.environ.get("APPDATA") or Path.home() / "AppData" / "Roaming")
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        base = Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config")
    return base / APP_NAME


def default_home() -> Path:
    return Path.home() / APP_NAME


def legacy_home() -> Path:
    return Path.home() / ".qtrix"


def read_location() -> Path | None:
    pointer = bootstrap_dir() / "location.json"
    try:
        home = json.loads(pointer.read_text(encoding="utf-8")).get("home")
    except (OSError, ValueError):
        return None
    return Path(home) if home else None


def write_location(home: Path) -> None:
    pointer = bootstrap_dir() / "location.json"
    pointer.parent.mkdir(parents=True, exist_ok=True)
    pointer.write_text(json.dumps({"home": str(home)}, indent=2), encoding="utf-8")


def resolve_home() -> tuple[Path, str]:
    """Return the qtrix folder and where the choice came from (env | location | default)."""
    if os.environ.get("QTRIX_HOME"):
        return Path(os.environ["QTRIX_HOME"]), "env"
    stored = read_location()
    if stored is not None:
        return stored, "location"
    return default_home(), "default"


def ensure_layout(home: Path) -> None:
    home.mkdir(parents=True, exist_ok=True)
    for sub in LAYOUT:
        (home / sub).mkdir(exist_ok=True)
    readme = home / "LEEME.txt"
    if not readme.exists():
        entries = "\n".join(f"  {name + '/':<14} {desc}" for name, desc in LAYOUT.items())
        readme.write_text(README.format(entries=entries), encoding="utf-8")


def migrate_legacy(home: Path, legacy: Path | None = None) -> list[str]:
    """Move data from the old hidden ~/.qtrix folder into the new layout (only into an empty workspace)."""
    legacy = legacy or legacy_home()
    if not legacy.is_dir() or legacy.resolve() == home.resolve():
        return []
    if (home / "data" / "workspace.duckdb").exists():
        return []
    moves = {
        "workspace.duckdb": home / "data" / "workspace.duckdb",
        "workspace.duckdb.wal": home / "data" / "workspace.duckdb.wal",
        "settings.json": home / "settings.json",
        "memory": home / "memory",
        "uploads": home / "uploads",
    }
    moved = []
    for name, dest in moves.items():
        src = legacy / name
        if not src.exists():
            continue
        if dest.is_dir() and not any(dest.iterdir()):
            dest.rmdir()  # empty placeholder created by ensure_layout
        if dest.exists():
            continue
        try:
            shutil.move(str(src), str(dest))
        except OSError as exc:
            # shutil.move falls back to copy + delete: a locked source leaves a copy behind. Undo it, or the
            # next start would see a database in the new folder and never finish the migration.
            if src.exists() and dest.exists():
                if dest.is_dir():
                    shutil.rmtree(dest, ignore_errors=True)
                else:
                    dest.unlink(missing_ok=True)
            # The workspace goes first: if it is locked nothing has moved yet, so failing here keeps the old
            # folder intact and the migration is retried on the next start (no empty database is created).
            raise RuntimeError(
                f"No se pudo mover {src} a {dest} ({exc}). Cierra cualquier otra ventana o proceso de qtrix "
                "y vuelve a abrir la app."
            ) from exc
        moved.append(name)
    if moved:
        log.info("Migrated %s from %s to %s", ", ".join(moved), legacy, home)
        try:
            legacy.rmdir()  # only succeeds when nothing else is left
        except OSError:
            pass
    return moved


@dataclass(frozen=True)
class Settings:
    home: Path
    token: str | None
    local_only: bool

    def __post_init__(self) -> None:
        ensure_layout(self.home)

    @property
    def db_path(self) -> Path:
        return self.home / "data" / "workspace.duckdb"

    @property
    def notebooks_dir(self) -> Path:
        return self.home / "notebooks"

    @property
    def settings_file(self) -> Path:
        return self.home / "settings.json"

    @property
    def memory_dir(self) -> Path:
        return self.home / "memory"

    @property
    def models_dir(self) -> Path:
        return self.home / "models"

    @property
    def runtime_dir(self) -> Path:
        return self.home / "runtime"

    @property
    def logs_dir(self) -> Path:
        return self.home / "logs"

    @property
    def uploads_dir(self) -> Path:
        return self.home / "uploads"

    @property
    def exports_dir(self) -> Path:
        return self.home / "exports"

    @classmethod
    def from_env(cls) -> Settings:
        home, origin = resolve_home()
        if origin != "env":  # never move real user data when a test or script points QTRIX_HOME elsewhere
            ensure_layout(home)
            migrate_legacy(home)
        return cls(
            home=home,
            token=os.environ.get("QTRIX_TOKEN") or None,
            local_only=os.environ.get("QTRIX_LOCAL_ONLY", "0") == "1",
        )
