"""How registry tools are spelled in ``qx``, and the translation between ``qx`` arguments and dispatcher calls.

Every tool declares a ``Qx`` form in its ``@tool`` decorator. ``bind`` turns a ``qx`` function call (target, name,
handle, positional and keyword arguments) into a dispatcher ``Call``; ``encode`` does the reverse for the printer.
Two families group several tools behind one function whose ``method=`` picks the tool, so that ``qx`` speaks in
object specs (ADR 0002 §3) rather than in tool names:

    qx.equation("eq01", page="wages", method="ls", formula="lwage ~ educ")   → ols_fit
    qx.equation("eq_iv", page="wages", method="tsls", formula="...")          → iv_fit(method="2sls")
    qx.var("var01", page="macro", method="vecm", columns=["gdp", "rate"])    → ts_vecm

The keywords of ``qx.equation`` and ``qx.var`` are the fields of the object's spec, so a stored spec prints as the
call that re-creates it (``call_for_spec``) and a call reads back as a spec (``spec_for_call``).
"""

from __future__ import annotations

import inspect
import keyword
import typing
from collections.abc import Sequence
from dataclasses import dataclass, field
from functools import cache
from typing import Any

from ..dispatch import Call
from ..objects.specs import _Spec, parse_spec
from ..tools.registry import ToolSpec, all_tools, get_tool


class QxError(ValueError):
    """A ``qx`` call that does not bind to a tool: unknown function or method, misplaced arguments."""


class NotExpressible(QxError):
    """A call or object that has no ``qx`` form yet."""


WHERE = {"qx": "qx", "page": "qx.page(…)", "obj": "qx.obj(…)", "report": "qx.report"}
HANDLE_PARAMS = {"page": ("dataset_id",), "obj": ("name", "result_id")}  # tool parameter a handle carries
_EMPTY = inspect.Parameter.empty

# ---- families: method → (tool, tool arguments it fixes) ---------------------------------------------------------
EQUATION_METHODS: dict[str, tuple[str, dict[str, Any]]] = {
    "ls": ("ols_fit", {}), "wls": ("ols_fit", {}), "binary": ("ols_binary", {}),
    "tsls": ("iv_fit", {"method": "2sls"}), "liml": ("iv_fit", {"method": "liml"}),
    "gmm": ("iv_fit", {"method": "gmm"}),
    "panel": ("panel_fit", {}), "arima": ("ts_arima", {}), "garch": ("vol_garch", {}),
}
VAR_METHODS: dict[str, tuple[str, dict[str, Any]]] = {"var": ("ts_var", {}), "vecm": ("ts_vecm", {})}
FAMILIES: dict[str, tuple[dict[str, tuple[str, dict[str, Any]]], str]] = {
    "equation": (EQUATION_METHODS, "ls"), "var": (VAR_METHODS, "var")}


# ---- the index --------------------------------------------------------------------------------------------------
@cache
def index() -> dict[tuple[str, str], tuple[ToolSpec, ...]]:
    """(target, function name or alias) → the tools behind it (several only for a family)."""
    out: dict[tuple[str, str], list[ToolSpec]] = {}
    for spec in all_tools(only_available=False, include_internal=True):
        if spec.qx is not None:
            for name in (spec.qx.name, *spec.qx.aliases):
                out.setdefault((spec.qx.on, name), []).append(spec)
    return {k: tuple(v) for k, v in out.items()}


def names(on: str) -> list[str]:
    """Function names (aliases included) available on a target, for ``dir()`` and autocompletion."""
    return sorted(n for (target, n) in index() if target == on)


def lookup(on: str, name: str) -> tuple[ToolSpec, ...]:
    specs = index().get((on, name))
    if not specs:
        raise QxError(f"{WHERE[on]} has no function '{name}'")
    return specs


@cache
def params(tool: str) -> dict[str, inspect.Parameter]:
    spec = get_tool(tool)
    return dict(inspect.signature(spec.fn).parameters) if spec is not None else {}


def handle_param(spec: ToolSpec) -> str | None:
    """The tool parameter carried by the page or object handle the function is called on."""
    if spec.qx is None or spec.qx.on not in HANDLE_PARAMS:
        return None
    return next((p for p in HANDLE_PARAMS[spec.qx.on] if p in params(spec.name)), None)


def renames(spec: ToolSpec) -> dict[str, str]:
    """Tool parameter → ``qx`` keyword."""
    out = dict(spec.qx.rename) if spec.qx else {}
    if "dataset_id" in params(spec.name) and handle_param(spec) != "dataset_id":
        out.setdefault("dataset_id", "page")
    return out


def creates_object(spec: ToolSpec) -> bool:
    """Functions whose first argument names the object they create (``eq01 = qx.equation("eq01", …)``)."""
    return spec.qx is not None and spec.qx.on == "qx" and spec.qx.positional[:1] == ("name",)


def label(spec: ToolSpec) -> str:
    return f"{WHERE[spec.qx.on]}.{spec.qx.name}" if spec.qx else spec.name


def is_default(tool: str, param: str, value: Any) -> bool:
    default = params(tool)[param].default if param in params(tool) else _EMPTY
    return default is not _EMPTY and type(value) is type(default) and value == default


# ---- qx arguments → call ------------------------------------------------------------------------------------------
def bind(on: str, name: str, handle: str | None, args: Sequence[Any], kwargs: dict[str, Any]) -> Call:
    """The dispatcher call for ``<target>.<name>(*args, **kwargs)`` on a handle (page or object name)."""
    specs = lookup(on, name)
    form = specs[0].qx
    if len(args) > len(form.positional):
        accepted = ", ".join(renames(specs[0]).get(p, p) for p in form.positional) or "none"
        raise QxError(f"{label(specs[0])}() takes {len(form.positional)} positional argument(s) ({accepted}), "
                      f"got {len(args)}")
    if form.family:
        merged = dict(zip(form.positional, args, strict=False))
        for key, value in kwargs.items():
            if key in merged:
                raise QxError(f"{label(specs[0])}() got multiple values for '{key}'")
            merged[key] = value
        return decode_family(form.family, merged)

    spec = specs[0]
    known, h, inverse = params(spec.name), handle_param(spec), {v: k for k, v in renames(spec).items()}
    out: dict[str, Any] = {h: handle} if h else {}
    out.update(zip(form.positional, args, strict=False))
    extra: dict[str, Any] = {}
    for key, value in kwargs.items():
        param = inverse.get(key, key)
        if param == h:
            raise QxError(f"{label(spec)}() takes '{key}' from the handle it is called on")
        if param in out:
            raise QxError(f"{label(spec)}() got multiple values for '{key}'")
        if param in known:
            out[param] = value
        elif form.varkw:
            extra[key] = value
        else:
            raise QxError(f"{label(spec)}() got an unexpected keyword '{key}'")
    # A required catch-all is always passed (``obj.estimate()`` with no changes re-estimates as is).
    if form.varkw and (extra or known[form.varkw].default is _EMPTY):
        out[form.varkw] = {**(out.get(form.varkw) or {}), **extra}
    return Call(spec.name, out)


def decode_family(family: str, kwargs: dict[str, Any]) -> Call:
    """``qx.equation(**kwargs)`` / ``qx.var(**kwargs)`` → the call of the tool its ``method`` names."""
    methods, default = FAMILIES[family]
    kw = dict(kwargs)
    method = kw.pop("method", default)
    if method not in methods:
        raise NotExpressible(f"qx.{family}() has no method '{method}' yet; methods: {', '.join(methods)}")
    tool, fixed = methods[method]
    if "page" in kw:
        kw["dataset_id"] = kw.pop("page")
    if method == "wls" and not kw.get("weights"):
        raise QxError("qx.equation(method='wls') needs weights=")
    if method == "ls" and kw.get("weights"):
        raise QxError("qx.equation(method='ls') takes no weights: use method='wls'")
    for key in kw:
        if key not in params(tool) or key in fixed:
            raise QxError(f"qx.{family}(method={method!r}) got an unexpected keyword '{key}'")
    return Call(tool, {**kw, **fixed})


# ---- call → qx arguments ------------------------------------------------------------------------------------------
@dataclass
class Form:
    """A call spelled in ``qx``: ``[assign =] <target>[(handle)].<name>(*positional, **keywords)``."""

    on: str
    name: str
    handle: Any = None
    positional: list[Any] = field(default_factory=list)
    keywords: list[tuple[str, Any]] = field(default_factory=list)
    assign: str | None = None


def encode_family(family: str, call: Call) -> dict[str, Any]:
    """Tool arguments → ``qx.equation`` / ``qx.var`` keywords: name, page, method, then the tool's own order."""
    methods = FAMILIES[family][0]
    args = dict(call.args)
    if call.tool == "ols_fit":
        method = "wls" if args.get("weights") else "ls"
    else:
        def matches(fixed: dict[str, Any]) -> bool:
            return all(args.get(k, params(call.tool)[k].default) == v for k, v in fixed.items())

        method = next((m for m, (tool, fixed) in methods.items() if tool == call.tool and matches(fixed)), None)
        if method is None:
            raise NotExpressible(f"{call.tool}({args}) has no qx.{family} method")
        for key in methods[method][1]:
            args.pop(key, None)
    out: dict[str, Any] = {}
    if "name" in args:
        out["name"] = args.pop("name")
    if "dataset_id" in args:
        out["page"] = args.pop("dataset_id")
    out["method"] = method
    order = list(params(call.tool))
    out.update((k, args[k]) for k in order if k in args)
    out.update((k, args[k]) for k in sorted(args) if k not in order)
    return out


def encode(call: Call) -> Form:
    """How a call prints in ``qx``. Arguments equal to the tool's default are left out."""
    spec = get_tool(call.tool)
    if spec is None or spec.qx is None:
        raise NotExpressible(f"Tool '{call.tool}' has no qx form")
    form, args = spec.qx, dict(call.args)
    if form.family:
        kw = encode_family(form.family, call)
        name = kw.pop("name", None)
        keywords = [(k, v) for k, v in kw.items() if k in ("page", "method") or not is_default(call.tool, k, v)]
        return Form("qx", form.name, positional=[name] if name is not None else [], keywords=keywords,
                    assign=_assignable(name))

    h = handle_param(spec)
    handle = args.pop(h, None) if h else None
    if h and not isinstance(handle, str):
        raise NotExpressible(f"{label(spec)}() needs '{h}'")
    positional: list[Any] = []
    for param in form.positional:
        if param not in args or (args[param] is None and is_default(call.tool, param, None)):
            break
        positional.append(args.pop(param))
    expanded: list[tuple[str, Any]] = []
    required_varkw = form.varkw is not None and params(call.tool)[form.varkw].default is _EMPTY
    if form.varkw and isinstance(args.get(form.varkw), dict) and (args[form.varkw] or required_varkw) and all(
            isinstance(k, str) and k.isidentifier() and not keyword.iskeyword(k) and k not in params(call.tool)
            for k in args[form.varkw]):
        expanded = list(args.pop(form.varkw).items())
    ren, order = renames(spec), list(params(call.tool))
    keys = [k for k in order if k in args] + sorted(k for k in args if k not in order)
    keywords = [(ren.get(k, k), args[k]) for k in keys if not is_default(call.tool, k, args[k])] + expanded
    name = call.args.get("name") if creates_object(spec) else None
    return Form(form.on, form.name, handle, positional, keywords, _assignable(name))


def _assignable(name: Any) -> str | None:
    """The variable an object-creating call is assigned to (its object name, when that is a usable identifier)."""
    if isinstance(name, str) and name.isidentifier() and not keyword.iskeyword(name) and name != "qx":
        return name
    return None


# ---- object specs ---------------------------------------------------------------------------------------------------
def call_for_spec(spec: _Spec | dict[str, Any], name: str) -> Call:
    """The call that (re-)estimates ``name`` with this spec: ``eq01 = qx.equation("eq01", page=…, method="ls", …)``."""
    s = spec if isinstance(spec, _Spec) else parse_spec(spec)
    fields = s.model_dump()
    method = fields.pop("method")
    if s.type in FAMILIES and method in FAMILIES[s.type][0]:
        return decode_family(s.type, {"name": name, "method": method, **fields})
    if s.type == "group":
        return Call("obj_group", {"name": name, "members": fields["members"],
                                  **({"dataset_id": fields["page"]} if fields["page"] else {})})
    if s.type == "table":
        return Call("ols_compare", {"result_ids": fields["sources"], "title": fields["title"], "name": name})
    raise NotExpressible(f"{s.type} objects (method '{method}') have no qx form yet")


def spec_for_call(call: Call) -> _Spec:
    """The spec an object-creating call describes (arguments it omits take the spec's defaults)."""
    spec = get_tool(call.tool)
    args = {k: v for k, v in call.args.items() if v is not None}
    if spec is not None and spec.qx is not None and spec.qx.family:
        kw = encode_family(spec.qx.family, Call(call.tool, args))
        kw.pop("name", None)
        return parse_spec(kw)
    if call.tool == "obj_group":
        return parse_spec({"method": "group", "page": args.get("dataset_id", ""), "members": args["members"]})
    if call.tool == "ols_compare":
        return parse_spec({"method": "table", "sources": args["result_ids"],
                           **({"title": args["title"]} if "title" in args else {})})
    raise NotExpressible(f"{call.tool} does not create an object with a spec")


# ---- signature help for the cell editor -----------------------------------------------------------------------------
def _doc_parts(doc: str) -> tuple[str, dict[str, str]]:
    """A Google-style docstring → (summary before ``Args:``, parameter → description)."""
    text = inspect.cleandoc(doc or "")
    head, _, rest = text.partition("\nArgs:")
    described: dict[str, str] = {}
    current, entry_indent = None, None
    for line in rest.splitlines():
        if line and not line[0].isspace():  # Returns:, Raises:… end the Args section
            break
        stripped = line.strip()
        if not stripped:
            continue
        indent = len(line) - len(line.lstrip())
        name, sep, desc = stripped.partition(":")
        if sep and name.isidentifier() and (entry_indent is None or indent <= entry_indent):
            current, entry_indent, described[name] = name, indent, desc.strip()
        elif current:  # continuation line of the current entry
            described[current] += " " + stripped
    return " ".join(head.split()), described


def _annotation(tool: str, param: inspect.Parameter) -> str | None:
    """A readable type: resolved hints (Literal values instead of an alias name), else the annotation text."""
    spec = get_tool(tool)
    try:
        hint = typing.get_type_hints(spec.fn).get(param.name, _EMPTY) if spec else _EMPTY
    except Exception:  # noqa: BLE001 - an unresolvable hint falls back to its source text
        hint = _EMPTY
    if hint is _EMPTY:
        return None if param.annotation is _EMPTY else str(param.annotation)
    text = inspect.formatannotation(hint)
    return text.replace("typing.", "").replace("NoneType", "None")


def _param(tool: str, p: inspect.Parameter, keyword: str, docs: dict[str, str],
           default: Any = _EMPTY) -> dict[str, Any]:
    value = p.default if default is _EMPTY else default
    return {"name": keyword, "type": _annotation(tool, p), "required": value is _EMPTY,
            "default": None if value is _EMPTY else repr(value), "doc": docs.get(p.name, "")}


def _family_signatures(family: str) -> list[dict[str, Any]]:
    """One signature per ``method=``: name, page, method, then the parameters of the tool behind it."""
    out = []
    for method, (tool, fixed) in FAMILIES[family][0].items():
        spec = get_tool(tool)
        if spec is None:
            continue
        summary, docs = _doc_parts(spec.description)
        ps = params(tool)
        head = [{"name": "name", "type": "str", "required": False, "default": "None",
                 "doc": docs.get("name", "Object name; an existing name is re-estimated as a new version.")}]
        if "dataset_id" in ps:
            head.append(_param(tool, ps["dataset_id"], "page", docs))
        head.append({"name": "method", "type": "str", "required": False, "default": repr(method),
                     "doc": f"Estimator: {', '.join(FAMILIES[family][0])}."})
        rest = [_param(tool, p, k, docs) for k, p in ps.items()
                if k not in ("name", "dataset_id", *fixed) and not (k == "weights" and method == "ls")]
        if method == "wls":
            rest = [{**r, "required": True, "default": None} if r["name"] == "weights" else r for r in rest]
        out.append({"method": method, "doc": summary, "positional": ["name"], "params": head + rest})
    return out


@cache
def signatures() -> dict[str, dict[str, Any]]:
    """``"<target>.<function>"`` → its ``qx`` signatures (several for a family), for the editor's signature help."""
    out: dict[str, dict[str, Any]] = {}
    for (on, name), specs in index().items():
        form = specs[0].qx
        if form.family:
            sigs = _family_signatures(form.family)
        else:
            spec = specs[0]
            summary, docs = _doc_parts(spec.description)
            h, ren, ps = handle_param(spec), renames(spec), params(spec.name)
            order = [*form.positional, *(k for k in ps if k not in form.positional)]
            items = [_param(spec.name, ps[k], ren.get(k, k), docs) for k in order
                     if k in ps and k != h and k != form.varkw]
            if form.varkw:
                items.append({"name": f"**{form.varkw}", "type": None, "required": False, "default": None,
                              "doc": docs.get(form.varkw, "")})
            sigs = [{"method": None, "doc": summary, "positional": [ren.get(p, p) for p in form.positional],
                     "params": items}]
        out[f"{on}.{name}"] = {"on": on, "name": name, "label": f"{WHERE[on]}.{name}", "signatures": sigs}
    return out
