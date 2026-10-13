"""Background jobs with progress (model downloads, runtime installs) and a resumable HTTP downloader."""

from __future__ import annotations

import hashlib
import threading
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field, fields
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

CHUNK = 1 << 20


class Cancelled(Exception):
    pass


@dataclass
class Job:
    id: str
    kind: str  # model | runtime
    title: str
    status: str = "queued"  # queued | running | verifying | done | error | cancelled
    total: int = 0
    done: int = 0
    message: str = ""
    error: str | None = None
    result: dict[str, Any] | None = None
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    _cancel: threading.Event = field(default_factory=threading.Event, repr=False)

    def to_dict(self) -> dict[str, Any]:
        # asdict() would deep-copy the cancel Event (holds a lock); copy public fields by hand.
        data = {f.name: getattr(self, f.name) for f in fields(self) if not f.name.startswith("_")}
        data["progress"] = round(min(self.done / self.total, 1.0), 4) if self.total else None
        return data

    def check_cancel(self) -> None:
        if self._cancel.is_set():
            raise Cancelled()


class JobManager:
    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def start(self, kind: str, title: str, work: Callable[[Job], dict[str, Any] | None]) -> Job:
        job = Job(id=uuid.uuid4().hex[:10], kind=kind, title=title)
        with self._lock:
            self._jobs[job.id] = job

        def run() -> None:
            job.status = "running"
            try:
                job.result = work(job)
                job.status = "done"
            except Cancelled:
                job.status = "cancelled"
            except Exception as exc:  # noqa: BLE001 - surfaced to the UI through the job
                job.status, job.error = "error", f"{type(exc).__name__}: {exc}"

        threading.Thread(target=run, name=f"qtrix-job-{job.id}", daemon=True).start()
        return job

    def list(self) -> list[Job]:
        with self._lock:
            return sorted(self._jobs.values(), key=lambda j: j.created_at, reverse=True)

    def get(self, job_id: str) -> Job:
        with self._lock:
            return self._jobs[job_id]

    def cancel(self, job_id: str) -> Job:
        job = self.get(job_id)
        job._cancel.set()
        return job

    def active(self, kind: str | None = None) -> list[Job]:
        return [j for j in self.list()
                if j.status in ("queued", "running", "verifying") and (kind is None or j.kind == kind)]


def download(url: str, dest: Path, job: Job, *, headers: dict[str, str] | None = None,
             sha256: str | None = None, client: httpx.Client | None = None) -> Path:
    """Download ``url`` to ``dest`` through ``dest.part``, resuming a previous partial download.

    Progress is added to ``job.done``; the caller sets ``job.total`` beforehand when known. When a
    SHA-256 is given the whole file is verified and a corrupt download is discarded.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    hasher = hashlib.sha256() if sha256 else None
    offset = part.stat().st_size if part.exists() else 0
    if hasher and offset:
        with part.open("rb") as fh:  # rehash what we already have so the digest covers the whole file
            while chunk := fh.read(CHUNK):
                job.check_cancel()
                hasher.update(chunk)
    job.done += offset
    own_client = client is None
    client = client or httpx.Client(follow_redirects=True, timeout=httpx.Timeout(30.0, read=120.0))
    try:
        req_headers = dict(headers or {})
        if offset:
            req_headers["Range"] = f"bytes={offset}-"
        with client.stream("GET", url, headers=req_headers) as resp:
            if resp.status_code != 416:  # 416: the partial file is already complete
                if resp.status_code in (401, 403):
                    raise PermissionError(
                        "Acceso denegado: este modelo exige aceptar su licencia en Hugging Face y un token")
                resp.raise_for_status()
                if offset and resp.status_code == 200:  # server ignored Range: start over
                    job.done -= offset
                    offset = 0
                    hasher = hashlib.sha256() if sha256 else None
                with part.open("ab" if offset else "wb") as fh:
                    for chunk in resp.iter_bytes(CHUNK):
                        job.check_cancel()
                        fh.write(chunk)
                        if hasher:
                            hasher.update(chunk)
                        job.done += len(chunk)
    finally:
        if own_client:
            client.close()
    if hasher and sha256:
        if hasher.hexdigest() != sha256:
            part.unlink(missing_ok=True)
            raise ValueError(f"Checksum SHA-256 incorrecto para {dest.name}; se ha descartado la descarga")
    part.replace(dest)
    return dest
