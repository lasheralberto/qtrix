"""Registry behind the qtrix mega-MCP.

Every econometric capability is a plain Python function registered with ``@tool(namespace, requires=...)``.
The same registry feeds the in-process PydanticAI toolset of the super-agent and the standalone MCP
server, so external MCP clients see exactly the tools the agents use. Tools whose libraries are not
installed are skipped automatically.
"""

from __future__ import annotations

import importlib
import importlib.util
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

NAMESPACES = {
    "data": "Dataset catalog, descriptives, SQL and transformations",
    "ols": "Cross-section regression (OLS/WLS/GLM, logit/probit, count, ordered, multinomial, quantile) and "
           "specification tests",
    "panel": "Panel data estimators (fixed/random effects, first differences)",
    "iv": "Instrumental variables (2SLS, LIML, GMM)",
    "ts": "Time series: unit roots, ARIMA, ARDL, Markov switching, VAR/VECM, Granger, cointegration, structural "
          "breaks, forecasting",
    "vol": "Volatility models: GARCH, EGARCH, GJR",
    "obj": "Named objects of the workfile: list, inspect versions, rename, copy, delete, groups",
    "viz": "Charts rendered in the analysis panel",
    "ui": "Direct manipulation of the dynamic analysis panel",
    "memory": "Semantic recall over the project conversation and results (turbovec)",
    "journal": "The journal of workfile actions (internal: undo)",
    "py": "Free Python cells in the project kernel, and results registered from them",
}

TOOL_MODULES = ("data", "ols", "panelreg", "ts", "vol", "obj", "viz", "ui", "recall", "journal", "py")

QxTarget = Literal["qx", "page", "obj", "report"]


@dataclass(frozen=True)
class Qx:
    """How a tool is exposed in the ``qx`` Python API (ADR 0002 §7.1); ``qtrix_engine.qx`` reads it.

    ``on`` is where the function lives: ``qx.<name>()``, a page handle (``qx.page("wages").<name>()``, which carries
    ``dataset_id``), an object handle (``qx.obj("eq01").<name>()``, which carries ``name`` or ``result_id``) or
    ``qx.report``. ``positional`` lists the tool parameters printed positionally, in that order; ``rename`` maps a
    tool parameter to its ``qx`` keyword (``dataset_id`` is ``page`` by default when no handle carries it).
    ``family`` names a group of tools behind one function whose ``method=`` picks the tool (``qx.equation``,
    ``qx.var``); ``varkw`` is the tool parameter (a dict) that collects ``**kwargs``.
    """

    on: QxTarget
    name: str
    positional: tuple[str, ...] = ()
    aliases: tuple[str, ...] = ()
    rename: Mapping[str, str] = field(default_factory=dict)
    family: Literal["equation", "var"] | None = None
    varkw: str | None = None

    def __str__(self) -> str:
        return {"qx": "qx", "page": "qx.page(…)", "obj": "qx.obj(…)", "report": "qx.report"}[self.on] + "." + self.name


@dataclass(frozen=True)
class ToolSpec:
    name: str
    namespace: str
    fn: Callable[..., Any]
    requires: tuple[str, ...] = field(default_factory=tuple)
    kinds: frozenset[str] | None = None  # page structures the tool applies to; None = any (ADR 0002 §1)
    # Does a call change workfile state (pages, series, sample, objects)? A predicate over the call's arguments
    # covers tools that only sometimes do (data_sql with save_as). Panel layout is not workfile state.
    mutating: bool | Callable[[dict[str, Any]], bool] = False
    internal: bool = False  # dispatchable by the UI/API only: never offered to agents, MCP clients or /tools
    qx: Qx | None = None  # its form in the qx Python API (ADR 0002 §7.1)
    cell: bool = False  # runs a Python cell: journaled as kind "python" with the cell source as its code

    def is_mutating(self, args: dict[str, Any] | None = None) -> bool:
        return bool(self.mutating(args or {})) if callable(self.mutating) else bool(self.mutating)

    @property
    def description(self) -> str:
        return (self.fn.__doc__ or "").strip()

    @property
    def available(self) -> bool:
        return all(importlib.util.find_spec(mod) is not None for mod in self.requires)

    def applies_to(self, kinds: Iterable[str]) -> bool:
        """Whether the tool is useful for a workspace holding pages of these structures."""
        return self.kinds is None or bool(self.kinds & set(kinds))


_REGISTRY: dict[str, ToolSpec] = {}


def tool(namespace: str, *, requires: Iterable[str] = (), kinds: Iterable[str] | None = None,
         mutating: bool | Callable[[dict[str, Any]], bool] = False,
         internal: bool = False, qx: Qx | None = None,
         cell: bool = False) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    if namespace not in NAMESPACES:
        raise ValueError(f"Unknown namespace '{namespace}'")

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        if not fn.__name__.startswith(f"{namespace}_"):
            raise ValueError(f"Tool '{fn.__name__}' must be prefixed with '{namespace}_'")
        _REGISTRY[fn.__name__] = ToolSpec(fn.__name__, namespace, fn, tuple(requires),
                                          frozenset(kinds) if kinds is not None else None, mutating, internal, qx,
                                          cell)
        return fn

    return decorator


def load_all() -> None:
    for module in TOOL_MODULES:
        importlib.import_module(f"qtrix_engine.tools.{module}")


def all_tools(*, namespaces: Iterable[str] | None = None, only_available: bool = True,
              include_internal: bool = False) -> list[ToolSpec]:
    load_all()
    wanted = set(namespaces) if namespaces is not None else None
    return [
        spec for spec in _REGISTRY.values()
        if (wanted is None or spec.namespace in wanted) and (spec.available or not only_available)
        and (include_internal or not spec.internal)
    ]


def get_tool(name: str) -> ToolSpec | None:
    """A registered tool by name, internal ones included (the dispatcher serves the UI and the journal too)."""
    load_all()
    return _REGISTRY.get(name)


def catalog() -> list[dict[str, Any]]:
    """Machine-readable listing used by the UI settings and the docs."""
    return [
        {"name": s.name, "namespace": s.namespace, "available": s.available, "requires": list(s.requires),
         "kinds": sorted(s.kinds) if s.kinds is not None else None, "mutating": bool(s.mutating),
         "qx": str(s.qx) if s.qx else None, "description": s.description.split("\n")[0]}
        for s in all_tools(only_available=False)
    ]
