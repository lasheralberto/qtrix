"""Inventory of GGUF models stored in <qtrix>/models and their download from Hugging Face."""

from __future__ import annotations

import json
import shutil
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from . import hub
from .jobs import Job, JobManager, download

META_SUFFIX = ".qtrix.json"


def repo_dir(models_dir: Path, repo: str) -> Path:
    return models_dir / repo.replace("/", "__")


class ModelStore:
    def __init__(self, models_dir: Path, jobs: JobManager, token_getter: Callable[[], str | None] = lambda: None):
        self.dir = models_dir
        self.jobs = jobs
        self._token = token_getter

    # ---- inventory -----------------------------------------------------------------
    def list(self) -> list[dict[str, Any]]:
        """Downloaded models. The id is the path of the first .gguf relative to the models folder."""
        out: list[dict[str, Any]] = []
        for meta_path in sorted(self.dir.rglob(f"*{META_SUFFIX}")):
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            files = [meta_path.parent / name for name in meta["files"]]
            if not all(f.exists() for f in files):
                continue
            out.append(self._entry(files[0], files, meta))
        known = {m["id"] for m in out}
        for gguf in sorted(self.dir.rglob("*.gguf")):  # GGUFs copied in by hand are usable too
            rel = gguf.relative_to(self.dir).as_posix()
            split = hub.SPLIT.match(gguf.name)
            if rel in known or "mmproj" in gguf.name.lower() or (split and split.group("part") != "00001"):
                continue
            parts = sorted(gguf.parent.glob(f"{split.group('stem')}-*-of-*.gguf")) if split else [gguf]
            out.append(self._entry(gguf, parts, {"quant": hub.quant_of(gguf.name)}))
        return out

    def _entry(self, first: Path, files: list[Path], meta: dict[str, Any]) -> dict[str, Any]:
        return {
            "id": first.relative_to(self.dir).as_posix(),
            "name": hub.SPLIT.sub(r"\g<stem>", first.name).removesuffix(".gguf"),
            "repo": meta.get("repo"),
            "quant": meta.get("quant"),
            "size": sum(f.stat().st_size for f in files),
            "path": str(first),
            "downloaded_at": meta.get("downloaded_at"),
        }

    def resolve(self, model_id: str) -> Path:
        path = (self.dir / model_id).resolve()
        if self.dir.resolve() not in path.parents or not path.is_file():
            raise FileNotFoundError(f"Modelo '{model_id}' no encontrado en {self.dir}")
        return path

    def delete(self, model_id: str) -> None:
        path = self.resolve(model_id)
        meta = path.with_name(path.name + META_SUFFIX)
        files = [path]
        if meta.exists():
            files = [path.parent / name for name in json.loads(meta.read_text(encoding="utf-8"))["files"]]
            meta.unlink()
        for f in files:
            f.unlink(missing_ok=True)
        if path.parent != self.dir and not any(path.parent.iterdir()):
            shutil.rmtree(path.parent, ignore_errors=True)

    # ---- download ------------------------------------------------------------------
    def start_download(self, repo: str, filename: str) -> Job:
        title = f"{repo}/{filename}"
        for job in self.jobs.active("model"):
            if job.title == title:
                return job
        return self.jobs.start("model", title, lambda job: self._download(job, repo, filename))

    def _download(self, job: Job, repo: str, filename: str) -> dict[str, Any]:
        token = self._token()
        job.message = "Consultando Hugging Face…"
        listing = hub.repo_files(repo, token=token)
        group = next((g for g in listing["files"] if filename in g["files"] or g["name"] == filename), None)
        if group is None:
            raise FileNotFoundError(f"{filename} no existe en {repo}")
        target = repo_dir(self.dir, repo)
        job.total = group["size"]
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        names = [Path(p).name for p in group["files"]]
        for i, path in enumerate(group["files"]):
            job.message = f"Descargando {names[i]}" + (f" ({i + 1}/{len(names)})" if len(names) > 1 else "")
            download(hub.resolve_url(repo, path), target / names[i], job, headers=headers,
                     sha256=group["sha256"].get(path))
        first = target / names[0]
        meta = {
            "repo": repo, "files": names, "quant": group["quant"], "size": group["size"],
            "sha256": group["sha256"], "source": hub.resolve_url(repo, group["files"][0]),
            "downloaded_at": datetime.now(UTC).isoformat(),
        }
        first.with_name(first.name + META_SUFFIX).write_text(json.dumps(meta, indent=2), encoding="utf-8")
        job.message = "Descargado"
        return {"id": first.relative_to(self.dir).as_posix(), "path": str(first)}
