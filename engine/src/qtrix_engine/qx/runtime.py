"""The callable ``qx`` API (``import qtrix as qx``), generated from the tool registry (ADR 0002 §7.1).

Every function is the ``Qx`` form of a registry tool: calling it binds the arguments to a dispatcher call and hands
it to the backend. In the engine process the backend is the dispatcher itself (actor ``user``, channel ``code``);
the kernel, the user's Jupyter and scripts use the HTTP client (``qx.connect``, E7·07b). Docstrings are the tools'.

Next to the generated functions sits the bridge from free Python (E7·07b): ``page.frame()`` reads data,
``qx.register`` turns a statsmodels/linearmodels result into a named object, ``qx.series_from`` adds a series.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

from ..dispatch import Call
from . import forms

Backend = Callable[[Call], Any]  # optionally with .frame(page, columns, sample) and a settable .run


class InProcess:
    """The engine's own process: calls go straight to the dispatcher as the user."""

    def __call__(self, call: Call) -> Any:
        from ..dispatch import dispatch

        return dispatch(call, "user", "code")

    def frame(self, page: str, columns: list[str] | None = None, sample: str | None = None) -> Any:
        from ..context import get_context

        return get_context().workspace.frame(page, columns, sample)


_backend: Backend = InProcess()


def set_backend(backend: Backend | None) -> Backend:
    """Route ``qx`` calls through ``backend`` (``None`` restores the in-process dispatcher); returns the previous."""
    global _backend
    previous, _backend = _backend, backend or InProcess()
    return previous


def connect(url: str | None = None, token: str | None = None) -> Any:
    """Drive a running engine over HTTP (defaults: ``QTRIX_URL`` and ``QTRIX_TOKEN``). The kernel does this itself."""
    from .client import Client

    client = Client(url, token)
    set_backend(client)
    return client


def set_run(run_id: str | None) -> None:
    """Tag the next calls with the engine's run id (the engine sets it around each kernel cell)."""
    if hasattr(_backend, "run"):
        _backend.run = run_id


# ---- bridge from free Python ------------------------------------------------------------------------------------
def frame(page: str, columns: list[str] | None = None, sample: str | None = None) -> Any:
    reader = getattr(_backend, "frame", None)
    if reader is None:
        raise RuntimeError("This qx backend cannot read data")
    return reader(page, columns, sample)


def register(name: str, result: Any, *, code: str | None = None, page: str | None = None, title: str = "") -> Obj:
    """Make a statsmodels or linearmodels result a named object (method "python"), e.g. a quantile regression.

    In a kernel cell the cell itself becomes the object's spec (re-estimating or replaying re-runs it); elsewhere
    pass ``code=``. Only the summary (coefficients, fit statistics) travels to the engine.
    """
    from ..objects.estimators import summarize_result

    summary = summarize_result(result, page=page or "", title=title)
    args: dict[str, Any] = {"name": name, "summary": summary}
    args.update({k: v for k, v in (("code", code), ("page", page), ("title", title)) if v})
    out = _backend(Call("py_register", args))
    return Obj(out["name"] if isinstance(out, dict) and out.get("name") else name, out)


def series_from(page: str, name: str, values: Any, *, overwrite: bool = False) -> Any:
    """Add a series computed in Python to a page.

    A pandas Series is aligned on the page's index (as ``page.frame()`` returns it; rows it lacks become missing);
    anything else must hold one value per row of the page, in order.
    """
    index = getattr(values, "index", None)
    if index is not None and hasattr(values, "reindex"):
        import pandas as pd

        if not index.equals(pd.RangeIndex(len(values))):
            values = values.reindex(frame(page, sample="@all").index)
    out = []
    for v in list(values):
        try:
            number = float(v)
        except (TypeError, ValueError):
            number = math.nan
        out.append(None if math.isnan(number) else number)
    return _backend(Call("data_set_series", {"dataset_id": page, "name": name, "values": out,
                                             "overwrite": overwrite}))


def function(on: str, name: str, handle: str | None = None) -> Callable[..., Any]:
    """The ``qx`` function ``name`` on a target, bound to a page or object handle."""
    specs = forms.lookup(on, name)
    creates = forms.creates_object(specs[0])

    def qx_function(*args: Any, **kwargs: Any) -> Any:
        result = _backend(forms.bind(on, name, handle, args, kwargs))
        if creates and isinstance(result, dict) and isinstance(result.get("name"), str):
            return Obj(result["name"], result)
        return result

    qx_function.__name__ = qx_function.__qualname__ = name
    if specs[0].qx.family:
        qx_function.__doc__ = "\n\n".join(f"method={m!r} ({tool}): {forms.get_tool(tool).description.splitlines()[0]}"
                                          for m, (tool, _) in forms.FAMILIES[specs[0].qx.family][0].items())
    else:
        qx_function.__doc__ = specs[0].description
    return qx_function


class _Handle:
    _on = "qx"

    @property
    def _handle(self) -> str | None:
        return None

    def __getattr__(self, attr: str) -> Callable[..., Any]:
        if attr.startswith("_") or (self._on, attr) not in forms.index():
            raise AttributeError(f"{forms.WHERE[self._on]} has no function '{attr}'")
        return function(self._on, attr, self._handle)

    def __dir__(self) -> list[str]:
        return sorted(set(super().__dir__()) | set(forms.names(self._on)))


class Page(_Handle):
    """A page of the workfile: ``qx.page("wages").generate("lwage", "log(wage)")``."""

    _on = "page"

    def __init__(self, name: str) -> None:
        self.name = name

    def frame(self, columns: list[str] | None = None, sample: str | None = None) -> Any:
        """The page as a pandas DataFrame indexed by its structure, restricted to the active sample (or ``sample``)."""
        return frame(self.name, columns, sample)

    def series_from(self, name: str, values: Any, *, overwrite: bool = False) -> Any:
        return series_from(self.name, name, values, overwrite=overwrite)

    @property
    def _handle(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return f"qx.page({self.name!r})"


class Obj(_Handle):
    """A named object: ``qx.obj("eq01").estimate(sample="1990q1 2007q4")``. ``result`` is what created it, if known."""

    _on = "obj"

    def __init__(self, name: str, result: Any = None) -> None:
        self.name, self.result = name, result

    @property
    def _handle(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return f"qx.obj({self.name!r})"


class Report(_Handle):
    """The analysis report (today's panel): ``qx.report.add_note("Findings", "...")``."""

    _on = "report"

    def __repr__(self) -> str:
        return "qx.report"


report = Report()


def page(name: str) -> Page:
    return Page(name)


def obj(name: str) -> Obj:
    return Obj(name)


def tool(name: str, /, **args: Any) -> Any:
    """Call a registry tool by name (the form the printer falls back to for calls without a ``qx`` spelling)."""
    return _backend(Call(name, args))


def module_getattr(attr: str) -> Callable[..., Any]:
    """``qx.<function>`` for the functions that live on ``qx`` itself (``qx.equation``, ``qx.sql``…)."""
    if attr.startswith("_") or ("qx", attr) not in forms.index():
        raise AttributeError(f"module 'qtrix' has no attribute '{attr}'")
    return function("qx", attr)


def module_dir() -> list[str]:
    return sorted({"page", "obj", "report", "tool", "set_backend", "connect", "register", "series_from", "Page", "Obj",
                   *forms.names("qx")})
