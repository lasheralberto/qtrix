"""Client mode of ``qx`` (ADR 0002 §7.4): the kernel, the user's Jupyter or a script drive a running engine over HTTP.

DuckDB has a single writing process, the engine, so outside it ``qx`` never opens the workfile: calls go to
``POST /qx/call`` (the dispatcher, so they are journaled and respect locks and policy) and data comes back from
``GET /qx/frame`` as Arrow. Inside a kernel cell the engine-assigned run id travels with every call.
"""

from __future__ import annotations

import os
from typing import Any

import httpx

from ..dispatch import Call


class EngineError(RuntimeError):
    """The engine refused or failed a call; the message is the engine's."""

    def __init__(self, message: str, *, status: int, type_: str | None = None) -> None:
        super().__init__(message)
        self.status, self.type = status, type_


class Client:
    """A ``qx`` backend that talks to the engine at ``url`` with the session token."""

    def __init__(self, url: str | None = None, token: str | None = None, *, timeout: float | None = None) -> None:
        url = url or os.environ.get("QTRIX_URL")
        if not url:
            raise EngineError("No engine URL: pass qx.connect(url, token) or set QTRIX_URL", status=0)
        self.url = url.rstrip("/")
        self.token = token if token is not None else os.environ.get("QTRIX_TOKEN")
        self.run: str | None = None  # set by the engine around each kernel cell
        self._http = httpx.Client(timeout=timeout)

    def _headers(self) -> dict[str, str]:
        headers = {}
        if self.token:
            headers["x-qtrix-token"] = self.token
        if self.run:
            headers["x-qtrix-run"] = self.run
        return headers

    @staticmethod
    def _raise(resp: httpx.Response) -> None:
        if resp.status_code < 400:
            return
        try:
            body = resp.json()
        except ValueError:
            body = {"detail": resp.text}
        raise EngineError(str(body.get("detail", body)), status=resp.status_code, type_=body.get("type"))

    def __call__(self, call: Call) -> Any:
        resp = self._http.post(f"{self.url}/qx/call", json={"tool": call.tool, "args": call.args},
                               headers=self._headers())
        self._raise(resp)
        return resp.json()["result"]

    def frame(self, page: str, columns: list[str] | None = None, sample: str | None = None) -> Any:
        import pyarrow as pa

        params: dict[str, Any] = {"page": page}
        if columns is not None:
            params["columns"] = ",".join(columns)
        if sample is not None:
            params["sample"] = sample
        resp = self._http.get(f"{self.url}/qx/frame", params=params, headers=self._headers())
        self._raise(resp)
        return pa.ipc.open_stream(resp.content).read_all().to_pandas()
