"""Notebooks (cuadernos): the container of a working session, like a chat session that keeps everything.

    <qtrix>/notebooks/<id>/
      meta.json          title, description, timestamps (the id is the folder name)
      notebook.duckdb    datasets, named objects and the journal (one connection, see workspace.py)
      uploads/           files uploaded into this notebook
    <qtrix>/notebooks/.trash/<id>-<timestamp>/   deleted notebooks (recoverable by moving them back)

The manager only deals with folders and metadata. The live state of an open notebook (workspace, panel, chat,
kernel) is an ``EngineContext`` built by the engine (api.py).
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

DB_NAME = "notebook.duckdb"
META_NAME = "meta.json"
TRASH = ".trash"
DEFAULT_TITLE = "Cuaderno sin título"
FIRST_TITLE = "Mi primer cuaderno"
_ID = re.compile(r"^[a-z0-9][a-z0-9_-]{0,39}$")


class NotebookNotFound(KeyError):
    def __str__(self) -> str:
        return f"Cuaderno '{self.args[0]}' no encontrado"


@dataclass
class Notebook:
    id: str
    title: str
    description: str
    created_at: float
    updated_at: float
    opened_at: float
    dir: Path

    @property
    def db_path(self) -> Path:
        return self.dir / DB_NAME

    @property
    def uploads_dir(self) -> Path:
        self.dir.joinpath("uploads").mkdir(exist_ok=True)
        return self.dir / "uploads"

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d.pop("dir")
        return d


def _write_json(path: Path, data: dict[str, Any]) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


class NotebookManager:
    def __init__(self, root: Path, *, legacy_db: Path | None = None, legacy_memory: Path | None = None) -> None:
        self.root = root
        root.mkdir(parents=True, exist_ok=True)
        if not self.list():
            self._adopt_legacy(legacy_db, legacy_memory)

    # ---- queries --------------------------------------------------------------------------------------------
    def _load(self, folder: Path) -> Notebook | None:
        try:
            meta = json.loads((folder / META_NAME).read_text(encoding="utf-8"))
            return Notebook(id=folder.name, title=meta["title"], description=meta.get("description", ""),
                            created_at=meta["created_at"], updated_at=meta["updated_at"],
                            opened_at=meta.get("opened_at", meta["updated_at"]), dir=folder)
        except (OSError, KeyError, ValueError):
            return None

    def list(self) -> list[Notebook]:
        found = [nb for p in self.root.iterdir() if p.is_dir() and not p.name.startswith(".")
                 and (nb := self._load(p)) is not None]
        return sorted(found, key=lambda nb: nb.opened_at, reverse=True)

    def get(self, notebook_id: str) -> Notebook:
        if not _ID.match(notebook_id or ""):
            raise NotebookNotFound(notebook_id)
        nb = self._load(self.root / notebook_id)
        if nb is None:
            raise NotebookNotFound(notebook_id)
        return nb

    def current(self) -> Notebook:
        """The most recently opened notebook; the first one is created when there is none."""
        found = self.list()
        return found[0] if found else self.create(FIRST_TITLE)

    # ---- changes --------------------------------------------------------------------------------------------
    def _save(self, nb: Notebook) -> Notebook:
        _write_json(nb.dir / META_NAME, {k: v for k, v in asdict(nb).items() if k not in ("id", "dir")})
        return nb

    def create(self, title: str = DEFAULT_TITLE, description: str = "") -> Notebook:
        now = time.time()
        notebook_id = f"nb-{uuid.uuid4().hex[:8]}"
        folder = self.root / notebook_id
        (folder / "uploads").mkdir(parents=True)
        return self._save(Notebook(notebook_id, title.strip() or DEFAULT_TITLE, description.strip(), now, now, now,
                                   folder))

    def update(self, notebook_id: str, *, title: str | None = None, description: str | None = None) -> Notebook:
        nb = self.get(notebook_id)
        if title is not None:
            nb.title = title.strip() or nb.title
        if description is not None:
            nb.description = description.strip()
        nb.updated_at = time.time()
        return self._save(nb)

    def touch(self, notebook_id: str) -> Notebook:
        nb = self.get(notebook_id)
        nb.opened_at = time.time()
        return self._save(nb)

    def remove(self, notebook_id: str) -> Path:
        """Move the folder to the trash (never deletes data). The caller closes the notebook's connections first."""
        nb = self.get(notebook_id)
        trash = self.root / TRASH
        trash.mkdir(exist_ok=True)
        dest = trash / f"{nb.id}-{time.strftime('%Y%m%d-%H%M%S')}"
        shutil.move(str(nb.dir), str(dest))
        return dest

    # ---- migration ------------------------------------------------------------------------------------------
    def _adopt_legacy(self, legacy_db: Path | None, legacy_memory: Path | None = None) -> None:
        """The single global workspace of earlier versions, and the conversation memory that grew with it, become
        the first notebook (the user-wide memory folder starts empty again)."""
        if legacy_db is None or not legacy_db.exists():
            return
        nb = self.create(FIRST_TITLE)
        try:
            for suffix in ("", ".wal"):
                src = legacy_db.with_name(legacy_db.name + suffix)
                if src.exists():
                    shutil.move(str(src), str(nb.dir / (DB_NAME + suffix)))
        except OSError as exc:
            shutil.rmtree(nb.dir, ignore_errors=True)  # retried on the next start; the old file stays where it was
            raise RuntimeError(f"No se pudo migrar {legacy_db} a un cuaderno ({exc}). Cierra otros procesos de qtrix "
                               "y vuelve a abrir la app.") from exc
        if legacy_memory is not None and legacy_memory.is_dir() and any(legacy_memory.iterdir()):
            try:
                shutil.move(str(legacy_memory), str(nb.dir / "memory"))
                legacy_memory.mkdir()
            except OSError:
                log.warning("Could not move the memory %s into notebook %s", legacy_memory, nb.id)
        log.info("Migrated the workspace %s to notebook %s", legacy_db, nb.id)
