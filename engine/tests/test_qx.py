"""The qx Python API: forms, canonical printer, API-cell parser, runtime and /code (ADR 0002 §7, E7·07)."""

from __future__ import annotations

import keyword
import random
import types
import typing
from typing import Any, Literal

import pytest
from fastapi.testclient import TestClient

import qtrix as qx
from qtrix_engine.api import create_app
from qtrix_engine.config import Settings
from qtrix_engine.dispatch import Call
from qtrix_engine.objects import specs as S
from qtrix_engine.qx import (
    NotExpressible,
    QxError,
    call_for_spec,
    forms,
    parse_cell,
    print_call,
    print_object,
    spec_for_call,
)
from qtrix_engine.tools.registry import all_tools

TOOLS = all_tools(only_available=False, include_internal=True)


# ---- every tool has a form ------------------------------------------------------------------------------------------
def test_every_tool_has_a_qx_form():
    assert [t.name for t in TOOLS if t.qx is None] == []
    seen: dict[tuple[str, str], str] = {}
    for t in TOOLS:
        params = forms.params(t.name)
        assert set(t.qx.positional) <= set(params), t.name
        assert set(t.qx.rename) <= set(params), t.name
        if t.qx.on in ("page", "obj"):
            assert forms.handle_param(t) is not None, t.name
        for name in (t.qx.name, *t.qx.aliases):
            owner = seen.setdefault((t.qx.on, name), t.qx.family or t.name)
            assert owner == (t.qx.family or t.name), f"{t.qx.on}.{name} is used by {owner} and {t.name}"
    family_tools = {tool for methods, _ in forms.FAMILIES.values() for tool, _ in methods.values()}
    assert family_tools == {t.name for t in TOOLS if t.qx.family}


# ---- property tests: call → code → call -----------------------------------------------------------------------------
STRINGS = ["y ~ x", "lwage ~ educ + I(exper**2)", "it's", 'say "hi"', "both ' and \"", "line\nbreak", "ñandú €",
           "back\\slash", "1990q1 2008q4 if age >= 25", "eq01", "wages"]


def _value(rng: random.Random, tp: Any, depth: int = 0) -> Any:
    origin, args = typing.get_origin(tp), typing.get_args(tp)
    if origin is Literal:
        return rng.choice(args)
    if origin in (typing.Union, types.UnionType):
        options = [a for a in args if a is not type(None)]
        return None if type(None) in args and rng.random() < 0.3 else _value(rng, rng.choice(options), depth)
    if origin is list:
        return [_value(rng, args[0], depth + 1) for _ in range(rng.randint(0, 3))]
    if origin is dict:
        keys = ["sample", "cov_type", "a b", "name", "if", "x1"]
        return {rng.choice(keys): _value(rng, args[1], depth + 1) for _ in range(rng.randint(0, 3))}
    if tp is bool:
        return rng.random() < 0.5
    if tp is int:
        return rng.randint(-5, 50)
    if tp is float:
        return rng.choice([0.5, -1.25, 1e-8, 3.0])
    if tp is str:
        return rng.choice(STRINGS)
    if tp is Any:
        return rng.choice([1, "x", True, None, [1, 2], {"k": "v"}] if depth < 2 else [1, "x"])
    raise AssertionError(f"no generator for {tp}")


def _random_args(rng: random.Random, tool: Any) -> dict[str, Any]:
    hints = typing.get_type_hints(tool.fn)
    out = {}
    for name, p in forms.params(tool.name).items():
        required = p.default is p.empty or name == forms.handle_param(tool)
        if required or rng.random() < 0.6:
            out[name] = _value(rng, hints[name])
            if name == forms.handle_param(tool) and not isinstance(out[name], str):
                out[name] = rng.choice(STRINGS)
    return out


def _normal(tool: str, args: dict[str, Any]) -> dict[str, Any]:
    """Arguments up to the ones left at their default (the printer omits them)."""
    return {k: v for k, v in args.items() if not forms.is_default(tool, k, v)}


@pytest.mark.parametrize("tool", TOOLS, ids=lambda t: t.name)
def test_calls_round_trip_through_code(tool):
    rng = random.Random(tool.name)
    for _ in range(60):
        args = _random_args(rng, tool)
        if tool.name == "ols_fit" and args.get("weights") == "":
            args["weights"] = None
        code = print_call(tool.name, args)
        compile(code, "<qx>", "exec")
        cell = parse_cell(code)
        assert cell.kind == "api", (code, cell.reason)
        assert len(cell.calls) == 1
        back = cell.calls[0].call
        assert back.tool == tool.name and _normal(tool.name, back.args) == _normal(tool.name, args), code
        assert print_call(back.tool, back.args) == code  # canonical: printing is idempotent


EXPRESSIBLE = [S.LsSpec, S.WlsSpec, S.BinarySpec, S.TslsSpec, S.LimlSpec, S.GmmSpec, S.PanelSpec, S.ArimaSpec,
               S.GarchSpec, S.VarSpec, S.VecmSpec, S.GroupSpec, S.TableSpec]


def _random_spec(rng: random.Random, cls: type) -> Any:
    hints = typing.get_type_hints(cls)
    data = {}
    for name, info in cls.model_fields.items():
        if name == "method":
            continue
        if info.is_required() or rng.random() < 0.6:
            data[name] = _value(rng, hints[name])
            if name == "weights" and not data[name]:
                data[name] = "w"
    return cls(**data)


@pytest.mark.parametrize("cls", EXPRESSIBLE, ids=lambda c: c.__name__)
def test_specs_round_trip_through_code(cls):
    rng = random.Random(cls.__name__)
    for _ in range(60):
        spec = _random_spec(rng, cls)
        code = print_object(spec, "eq_x")
        (parsed,) = parse_cell(code).calls
        back = spec_for_call(parsed.call)
        assert type(back) is cls and back.model_dump() == spec.model_dump(), code
        direct = call_for_spec(spec, "eq_x")
        assert parsed.call.tool == direct.tool and _normal(direct.tool, parsed.call.args) == _normal(direct.tool,
                                                                                                     direct.args)


def test_types_without_a_form_say_so():
    for spec in (S.GlmSpec(page="w", formula="y ~ x"), S.TextSpec(markdown="hi"), S.GraphSpec(title="t", figure={})):
        with pytest.raises(NotExpressible):
            print_object(spec, "x01")


def test_printer_is_readable():
    assert print_call("data_genr", {"dataset_id": "macro", "name": "infl", "expression": "400 * dlog(cpi)"}) == (
        'qx.page("macro").generate("infl", "400 * dlog(cpi)")')
    assert print_call("obj_reestimate", {"name": "eq01", "changes": {"sample": "1990q1 2000q4"}}) == (
        'qx.obj("eq01").estimate(sample="1990q1 2000q4")')
    assert print_call("iv_fit", {"dataset_id": "w", "formula": "y ~ 1 + [x ~ z]", "method": "2sls"}) == (
        'qx.equation(page="w", method="tsls", formula="y ~ 1 + [x ~ z]")')
    assert print_call("ols_compare", {"result_ids": ["eq01", "eq02"], "name": "tab01"}) == (
        'tab01 = qx.table("tab01", ["eq01", "eq02"])')
    assert print_call("no_such_tool", {"a": 1}) == 'qx.tool("no_such_tool", a=1)'  # the fallback still parses
    assert parse_cell(print_call("no_such_tool", {"a": 1})).calls[0].call == Call("no_such_tool", {"a": 1})
    long = print_call("ts_vecm", {"dataset_id": "macro", "columns": ["gdp", "rate"], "coint_rank": 1,
                                  "sample": "1990q1 2010q4", "name": "vec_a"})
    assert long.splitlines()[0] == 'vec_a = qx.var(' and long.splitlines()[-1] == ")"


# ---- the parser never executes, and says why a cell is Python ------------------------------------------------------
@pytest.mark.parametrize("code, reason", [
    ("import os\nos.system('echo hi')", "imports a module"),
    ("qx.sql('select 1')\nimport os", "imports a module"),
    ("for c in ['a', 'b']:\n    qx.page('w').generate(c, 'x')", "For"),
    ("def f():\n    pass", "FunctionDef"),
    ("qx.page('w').generate('y2', name)", "Python variable"),
    ("qx.page('w').generate('y2', f'{1}')", "not a literal"),
    ("qx.page('w').head()", "has no function 'head'"),
    ("qx.sql(open('x').read())", "not a literal"),
    ("__import__('os').system('x')", "not a qx function"),
    ("df = qx.page('w').preview()\ndf.describe()", "called on a result"),
    ("qx = 1", "replaces the API"),
    ("x = (", "syntax error"),
])
def test_python_cells(code, reason):
    cell = parse_cell(code)
    assert cell.kind == "python" and cell.calls == [] and reason in cell.reason


def test_api_cells_with_assignments_aliases_and_pending_names():
    cell = parse_cell(
        "import qtrix as qx\n"
        "m = qx.page('macro')\n"
        "m.smpl('1990q1 2008q4')\n"
        "e = qx.equation(page='macro', formula='gdp ~ gdp(-1)')\n"  # name chosen by the engine
        "e.rename('eq_ar')\n"
        "eq_b = qx.equation('eq_b', page='macro', formula='gdp ~ rate')\n"
        "qx.table('tab', [eq_b, 'eq_ar'])\n"
        "eq_b.estimate(cov_type='HC3')\n"
        "eq_b\n")
    assert cell.kind == "api"
    tools = [(p.call.tool, p.handle_from) for p in cell.calls]
    assert tools == [("data_set_sample", None), ("ols_fit", None), ("obj_rename", 1), ("ols_fit", None),
                     ("ols_compare", None), ("obj_reestimate", None)]
    assert cell.calls[2].resolve([None, {"name": "eq01"}]).args == {"name": "eq01", "new_name": "eq_ar"}
    assert cell.calls[4].call.args["result_ids"] == ["eq_b", "eq_ar"]
    assert cell.calls[5].call.args == {"name": "eq_b", "changes": {"cov_type": "HC3"}}


def test_bad_arguments_are_errors_with_the_line():
    with pytest.raises(QxError, match="line 2: .*unexpected keyword 'formla'"):
        parse_cell("import qtrix as qx\nqx.equation('eq1', page='w', formla='y ~ x')")
    with pytest.raises(QxError, match="no method 'glm'"):
        parse_cell("qx.equation('eq1', page='w', method='glm', formula='y ~ x')")
    with pytest.raises(QxError, match="from the handle"):
        parse_cell("qx.page('w').generate('a', 'b', dataset_id='z')")
    with pytest.raises(QxError, match="positional"):
        parse_cell("qx.obj('eq1').rename('a', 'b')")


# ---- runtime: import qtrix as qx ------------------------------------------------------------------------------------
def test_runtime_dispatches_as_the_user(ctx, wages):
    eq = qx.equation("eq_a", page="wages", formula="lwage ~ educ")
    assert isinstance(eq, qx.Obj) and eq.name == "eq_a" and eq.result["version"] == 1
    assert eq.estimate(cov_type="HC3")["version"] == 2
    qx.page("wages").generate("educ2", "educ * educ")
    auto = qx.equation(page="wages", method="binary", formula="high ~ educ")
    assert auto.name == "eq01"
    assert [d["name"] for d in qx.pages()] == ["wages"]
    assert qx.tool("obj_get", name="eq_a")["version"] == 2
    entries = ctx.journal.list()
    assert {(e["actor"], e["channel"]) for e in entries} == {("user", "code")}
    assert [e["code"] for e in entries[:2]] == [
        'eq_a = qx.equation("eq_a", page="wages", method="ls", formula="lwage ~ educ")',
        'qx.obj("eq_a").estimate(cov_type="HC3")']
    assert "generate" in dir(qx.page("wages")) and "equation" in dir(qx) and "genr" in dir(qx.page("wages"))
    assert qx.equation.__doc__.startswith("method='ls' (ols_fit)")
    with pytest.raises(AttributeError):
        qx.no_such_function  # noqa: B018
    seen = []
    previous = qx.set_backend(lambda call: seen.append(call) or {"name": "z"})
    try:
        assert qx.obj("eq_a").lock() == {"name": "z"} and seen == [Call("obj_lock", {"name": "eq_a"})]
    finally:
        qx.set_backend(previous)


# ---- HTTP: /objects/{name}/code and POST /code ---------------------------------------------------------------------
@pytest.fixture
def api(tmp_path, wages_csv):
    c = TestClient(create_app(Settings(home=tmp_path, token=None, local_only=False), with_memory=False))
    c.post("/datasets/import", json={"path": str(wages_csv)})
    return c


def test_edit_printed_code_creates_a_new_version(api):
    fit = api.post("/tools/ols_fit", json={"dataset_id": "wages", "formula": "lwage ~ educ", "name": "eq_a"})
    seq = fit.status_code == 200 and api.get("/journal").json()[-1]["seq"]
    code = api.get("/objects/eq_a/code").json()
    assert code["version"] == 1 and code["code"] == (
        'eq_a = qx.equation(\n    "eq_a",\n    page="wages",\n    method="ls",\n    formula="lwage ~ educ",\n'
        '    sample="@all",\n)')
    edited = code["code"].replace("lwage ~ educ", "lwage ~ educ + exper")
    out = api.post("/code", json={"code": edited, "cell_id": "cell-9", "edited_from": seq}).json()
    assert out["ok"] and out["calls"][0]["result"]["version"] == 2
    entry = api.get(f"/journal/{out['calls'][0]['seq']}").json()
    assert (entry["actor"], entry["channel"], entry["edited_from"], entry["cell_id"]) == ("user", "code", seq, "cell-9")
    obj = api.get("/objects/eq_a").json()
    assert obj["version"] == 2 and obj["spec"]["formula"] == "lwage ~ educ + exper"
    assert "educ + exper" not in api.get("/objects/eq_a/code?version=1").json()["code"]
    assert api.get("/objects/eq_a/code?version=7").status_code == 404


def test_python_cells_never_run_on_the_api_path(api, tmp_path):
    """A Python cell only ever runs in the kernel (E7·07b); without one (no HTTP server here) nothing runs at all."""
    marker = tmp_path / "pwned.txt"
    for code in (f"import os\nopen({str(marker)!r}, 'w').write('x')",
                 f"qx.sql('select 1')\nopen({str(marker)!r}, 'w')"):
        assert api.post("/code", json={"code": code, "dry_run": True}).json()["kind"] == "python"
        resp = api.post("/code", json={"code": code})
        assert resp.status_code == 503 and "kernel" in resp.json()["detail"]
    assert not marker.exists()
    cells = [e for e in api.get("/journal").json() if e["kind"] == "python"]
    assert len(cells) == 2 and not any(e["ok"] or e["mutating"] for e in cells)  # journaled as failed, nothing ran


def test_code_cells_stop_at_the_first_error(api):
    out = api.post("/code", json={"code": (
        "qx.page('wages').generate('educ2', 'educ * educ')\n"
        "qx.equation('eq_b', page='wages', formula='lwage ~ nope')\n"
        "qx.obj('eq_b').lock()")}).json()
    assert not out["ok"] and [c["ok"] for c in out["calls"]] == [True, False, False]
    assert out["calls"][1]["seq"] is not None and "nope" in out["calls"][1]["error"]
    assert out["calls"][2]["skipped"]
    dry = api.post("/code", json={"code": "qx.page('wages').smpl('if educ > 10')", "dry_run": True}).json()
    assert dry["calls"] == [{"line": 1, "tool": "data_set_sample", "args": {"dataset_id": "wages",
                             "sample": "if educ > 10"}, "code": 'qx.page("wages").set_sample("if educ > 10")',
                             "issues": []}]
    assert api.post("/code", json={"code": "qx.obj('x').rename()"}).status_code == 200  # a dispatcher error, per call
    assert api.post("/code", json={"code": "qx.obj('x').rename(nam='y')"}).status_code == 400  # does not bind
    assert api.post("/code", json={"code": "qx.sql('select 1')", "edited_from": 999}).status_code == 404


def test_table_and_group_print_as_qx(api):
    for name, formula in (("eq_a", "lwage ~ educ"), ("eq_b", "lwage ~ educ + exper")):
        api.post("/tools/ols_fit", json={"dataset_id": "wages", "formula": formula, "name": name})
    api.post("/code", json={"code": "qx.table('tab01', ['eq_a', 'eq_b'], title='Mincer')\n"
                                    "qx.group('gr_x', ['educ', 'exper'], page='wages')"})
    assert api.get("/objects/tab01/code").json()["code"] == (
        'tab01 = qx.table("tab01", ["eq_a", "eq_b"], title="Mincer")')
    assert api.get("/objects/gr_x/code").json()["code"] == 'gr_x = qx.group("gr_x", ["educ", "exper"], page="wages")'


def test_no_keyword_object_names_are_assigned():
    assert forms._assignable("if") is None and forms._assignable("qx") is None and forms._assignable("eq1") == "eq1"
    assert all(not keyword.iskeyword(n) for n in forms.names("qx"))
