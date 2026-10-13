"""Canonical printer (call → Python) and API-cell parser (Python → calls) of the ``qx`` API (ADR 0002 §7.2-7.3).

The printer is deterministic: one statement per call, arguments in a fixed order, defaults left out, wrapped one
argument per line past ``WIDTH`` characters. The parser reads a cell with ``ast`` and **never executes it**: a cell
made only of ``qx`` calls with literal arguments, assignments of their results and references to object names is an
*API cell* and becomes a list of dispatcher calls; anything else is a *Python cell* for the project kernel (E7·07b).
Together they round-trip: ``parse_cell(print_call(c)).calls[0].call == c`` up to arguments left at their default.
"""

from __future__ import annotations

import ast
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

from ..dispatch import Call
from ..objects.specs import _Spec
from . import forms
from .forms import Form, QxError

WIDTH = 88
IMPORT = "import qtrix as qx"


# ---- printer ----------------------------------------------------------------------------------------------------
def literal(value: Any) -> str:
    """Python literal of a JSON-like value; strings use double quotes unless they contain one."""
    if isinstance(value, str):
        text = repr(value)
        return f'"{text[1:-1]}"' if text[0] == "'" and '"' not in value else text
    if isinstance(value, dict):
        return "{" + ", ".join(f"{literal(k)}: {literal(v)}" for k, v in value.items()) + "}"
    if isinstance(value, list | tuple):
        return "[" + ", ".join(literal(v) for v in value) + "]"
    if value is None or isinstance(value, bool | int):
        return repr(value)
    if isinstance(value, float) and value == value and value not in (float("inf"), float("-inf")):
        return repr(value)
    return literal(str(value))


def render(form: Form) -> str:
    receiver = {"qx": "qx", "page": f"qx.page({literal(form.handle)})", "obj": f"qx.obj({literal(form.handle)})",
                "report": "qx.report"}[form.on]
    parts = [literal(v) for v in form.positional] + [f"{k}={literal(v)}" for k, v in form.keywords]
    head = (f"{form.assign} = " if form.assign else "") + f"{receiver}.{form.name}("
    line = head + ", ".join(parts) + ")"
    if len(line) <= WIDTH or not parts:
        return line
    return head + "\n" + "".join(f"    {p},\n" for p in parts) + ")"


def print_call(tool: str, args: dict[str, Any]) -> str:
    """Canonical ``qx`` code of a call. Calls with no ``qx`` form fall back to ``qx.tool("<name>", **args)``."""
    try:
        return render(forms.encode(Call(tool, dict(args))))
    except QxError:
        return render(Form("qx", "tool", positional=[tool], keywords=sorted(args.items())))


def print_object(spec: _Spec | dict[str, Any], name: str) -> str:
    """The code that (re-)estimates an object from its spec; raises ``NotExpressible`` for types without a form."""
    call = forms.call_for_spec(spec, name)
    return print_call(call.tool, call.args)


# ---- parser -----------------------------------------------------------------------------------------------------
@dataclass
class ParsedCall:
    call: Call
    line: int
    handle_from: int | None = None  # index of an earlier call whose result names the object this one acts on

    def resolve(self, results: list[Any]) -> Call:
        """The call with its handle filled from an earlier result (``e = qx.equation(...); e.rename("x")``)."""
        if self.handle_from is None:
            return self.call
        result = results[self.handle_from]
        name = result.get("name") if isinstance(result, dict) else None
        if not isinstance(name, str):
            raise QxError(f"line {self.line}: the object created on an earlier line has no name")
        spec = forms.get_tool(self.call.tool)
        return Call(self.call.tool, {**self.call.args, forms.handle_param(spec): name})


@dataclass
class Cell:
    kind: Literal["api", "python"]
    calls: list[ParsedCall] = field(default_factory=list)
    reason: str | None = None  # for a Python cell: the first construct outside the API


class _NotApi(Exception):
    """The cell steps outside the API: it is a Python cell."""


@dataclass(frozen=True)
class _Handle:
    kind: Literal["qx", "report", "page", "obj", "pending", "value"]
    name: str | None = None
    origin: int | None = None  # for "pending": the call whose result names the object


_QX = _Handle("qx")


def parse_cell(code: str) -> Cell:
    """Classify a cell and, for an API cell, return its calls in order. Nothing is executed.

    Raises ``QxError`` (with the line) when an API call has arguments its function does not accept.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return Cell("python", reason=f"syntax error on line {exc.lineno}: {exc.msg}")
    parser = _Parser()
    for stmt in tree.body:
        try:
            parser.statement(stmt)
        except _NotApi as exc:
            return Cell("python", reason=f"line {stmt.lineno}: {exc}")
        except QxError as exc:
            raise QxError(f"line {stmt.lineno}: {exc}") from None
    if not parser.calls and tree.body:  # e.g. `x` to display a kernel variable: nothing for the API to do
        return Cell("python", reason="the cell makes no qx call")
    return Cell("api", parser.calls)


class _Parser:
    def __init__(self) -> None:
        self.env: dict[str, _Handle] = {}
        self.calls: list[ParsedCall] = []

    # ---- statements ----------------------------------------------------------------------------------------------
    def statement(self, stmt: ast.stmt) -> None:
        if isinstance(stmt, ast.Import):
            if [(a.name, a.asname) for a in stmt.names] != [("qtrix", "qx")]:
                raise _NotApi(f"`{ast.unparse(stmt)}` imports a module; only `{IMPORT}` belongs to an API cell")
        elif isinstance(stmt, ast.Pass):
            pass
        elif isinstance(stmt, ast.Expr):
            if isinstance(stmt.value, ast.Constant | ast.Name):
                return  # a docstring or a bare name (displayed in a notebook): nothing to run
            if not isinstance(stmt.value, ast.Call):
                raise _NotApi(f"`{ast.unparse(stmt.value)}` is an expression, not a qx call")
            self.call(stmt.value)
        elif isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name):
            target = stmt.targets[0].id
            if target == "qx":
                raise _NotApi("assigning to `qx` replaces the API")
            self.env[target] = self.value(stmt.value)
        else:
            raise _NotApi(f"a `{type(stmt).__name__}` statement is Python, not a qx call")

    # ---- expressions -----------------------------------------------------------------------------------------------
    def value(self, node: ast.expr) -> _Handle:
        """What an expression stands for: the API, the report, a page or object handle, or an opaque result."""
        if isinstance(node, ast.Name):
            if node.id == "qx":
                return _QX
            return self.env.get(node.id) or _Handle("obj", node.id)  # a bare name refers to an object
        if isinstance(node, ast.Attribute) and node.attr == "report" and self.value(node.value) == _QX:
            return _Handle("report")
        if isinstance(node, ast.Call):
            return self.call(node)
        raise _NotApi(f"`{ast.unparse(node)}` is not a qx page, object or call")

    def call(self, node: ast.Call) -> _Handle:
        if not isinstance(node.func, ast.Attribute):
            raise _NotApi(f"`{ast.unparse(node.func)}()` is not a qx function")
        receiver, name = self.value(node.func.value), node.func.attr
        if receiver.kind == "value":
            raise _NotApi(f"`.{name}()` is called on a result, not on a qx page or object")
        if any(isinstance(a, ast.Starred) for a in node.args) or any(k.arg is None for k in node.keywords):
            raise _NotApi("*args / **kwargs unpacking is Python")
        args = [self.literal(a) for a in node.args]
        kwargs = {k.arg: self.literal(k.value) for k in node.keywords}

        if receiver == _QX and name in ("page", "obj"):
            target = args[0] if len(args) == 1 and not kwargs else kwargs.get("name") if not args else None
            if not isinstance(target, str):
                raise QxError(f"qx.{name}() takes one name, e.g. qx.{name}(\"wages\")")
            return _Handle(name, target)
        if receiver == _QX and name == "tool":
            if len(args) != 1 or not isinstance(args[0], str):
                raise QxError('qx.tool() takes the tool name, e.g. qx.tool("ols_fit", dataset_id=..., ...)')
            self.calls.append(ParsedCall(Call(args[0], kwargs), node.lineno))
            return _Handle("value")

        on = "obj" if receiver.kind == "pending" else receiver.kind
        if (on, name) not in forms.index():
            raise _NotApi(f"{forms.WHERE[on]} has no function '{name}'")
        call = forms.bind(on, name, receiver.name, args, kwargs)
        self.calls.append(ParsedCall(call, node.lineno, receiver.origin if receiver.kind == "pending" else None))
        if forms.creates_object(forms.lookup(on, name)[0]):
            created = call.args.get("name")
            if isinstance(created, str):
                return _Handle("obj", created)
            return _Handle("pending", origin=len(self.calls) - 1)
        return _Handle("value")

    def literal(self, node: ast.expr) -> Any:
        """A literal argument; names bound to a page or object stand for its name."""
        if isinstance(node, ast.Name):
            bound = self.env.get(node.id)
            if bound is not None and bound.kind in ("page", "obj") and bound.name:
                return bound.name
            raise _NotApi(f"`{node.id}` is a Python variable, not a literal" if bound is None
                          else f"`{node.id}` is only known after the cell runs")
        if isinstance(node, ast.List | ast.Tuple):
            return [self.literal(e) for e in node.elts]
        if isinstance(node, ast.Dict):
            if any(k is None for k in node.keys):
                raise _NotApi("**dict unpacking is Python")
            return {self.literal(k): self.literal(v) for k, v in zip(node.keys, node.values, strict=True)}
        try:
            return ast.literal_eval(node)
        except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
            raise _NotApi(f"`{ast.unparse(node)}` is not a literal") from None


# ---- running an API cell ----------------------------------------------------------------------------------------
def run_cell(cell: Cell, run: Callable[[Call], tuple[Any, int | None]]) -> list[dict[str, Any]]:
    """Run the calls of an API cell in order, stopping at the first error.

    ``run`` dispatches one call and returns ``(result, journal seq)``; an exception it raises may carry
    ``journal_seq``. Returns one entry per call: ``{line, tool, code, seq, ok, result | error, exception}``; calls after
    a failure are ``skipped``.
    """
    if cell.kind != "api":
        raise QxError("Python cells run in the project kernel, not as API calls")
    results: list[Any] = []
    out: list[dict[str, Any]] = []
    failed = False
    for parsed in cell.calls:
        entry: dict[str, Any] = {"line": parsed.line, "tool": parsed.call.tool}
        if failed:
            out.append({**entry, "ok": False, "skipped": True})
            results.append(None)
            continue
        try:
            call = parsed.resolve(results)
            entry["code"] = print_call(call.tool, call.args)
            result, entry["seq"] = run(call)
        except Exception as exc:  # noqa: BLE001 - reported per call; the caller decides how to surface it
            failed = True
            out.append({**entry, "seq": getattr(exc, "journal_seq", None), "ok": False,
                        "error": f"{type(exc).__name__}: {exc}", "exception": exc})
            results.append(None)
            continue
        results.append(result)
        out.append({**entry, "ok": True, "result": result})
    return out
