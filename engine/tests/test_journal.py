"""Dispatcher, journal, undo and replay (ADR 0002 §6, E7·06)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from pydantic_ai import ModelRetry

from qtrix_engine.agents.lead import _wrap
from qtrix_engine.api import create_app
from qtrix_engine.config import Settings
from qtrix_engine.dispatch import (
    Call,
    PolicyDenied,
    PolicyRequest,
    UnknownTool,
    dispatch,
    register_policy,
)
from qtrix_engine.journal import print_call
from qtrix_engine.mcp_server import build_server
from qtrix_engine.objects.estimators import summaries_match
from qtrix_engine.objects.store import LockedObjectError
from qtrix_engine.qx import parse_cell
from qtrix_engine.replay import ReplayError, UndoError, replay, undo
from qtrix_engine.tools.registry import all_tools, get_tool

from .test_api import client as chat_client
from .test_api import events


def run(tool: str, actor: str = "user", channel: str = "code", **args):
    return dispatch(Call(tool, args), actor, channel)


@pytest.fixture
def imported(ctx, wages_csv):
    run("data_import_file", path=str(wages_csv), name="wages")
    return ctx


def coefs(ctx, name: str) -> dict[str, tuple[float, float]]:
    return {c["name"]: (c["coef"], c["se"]) for c in ctx.objects.get(name).summary["coefficients"]}


# ---- registry flags and the canonical printer -----------------------------------------------------------------
def test_every_tool_declares_mutating_correctly():
    mutating = {t.name for t in all_tools(include_internal=True) if t.mutating}
    assert {"data_import_file", "data_genr", "data_set_structure", "data_set_sample", "data_transform", "ols_fit",
            "ols_binary", "ols_compare", "panel_fit", "iv_fit", "ts_arima", "ts_var", "obj_rename", "obj_copy",
            "obj_delete", "obj_group", "obj_lock", "obj_unlock", "obj_reestimate", "data_delete_dataset",
            "journal_undo"} <= mutating
    read_only = {t.name for t in all_tools()} - mutating
    assert {"data_list_datasets", "data_summary_stats", "ols_diagnostics", "obj_list", "obj_get", "viz_scatter",
            "ui_add_note", "memory_search", "ts_unit_root", "panel_hausman"} <= read_only
    sql = get_tool("data_sql")
    assert not sql.is_mutating({"query": "select 1"}) and sql.is_mutating({"query": "select 1", "save_as": "x"})
    assert {t.name for t in all_tools(include_internal=True)} - {t.name for t in all_tools()} == {
        "obj_unlock", "data_delete_dataset", "journal_undo",
        "data_set_series", "py_register"}  # the kernel bridge (qx.series_from, qx.register)


def test_print_call_is_deterministic_python():
    code = print_call("ols_fit", {"name": "eq01", "formula": "y ~ x", "dataset_id": "wages", "cluster": None})
    assert code == 'eq01 = qx.equation("eq01", page="wages", method="ls", formula="y ~ x")'
    compile(code, "<journal>", "exec")  # it is valid Python
    assert print_call("data_sql", {"query": "select 1", "zzz": [1, {"a": True}]}) == (
        'qx.sql("select 1", zzz=[1, {"a": True}])')


# ---- the single write path ---------------------------------------------------------------------------------------
def test_dispatch_journals_reads_and_writes(imported):
    ctx = imported
    out = run("ols_fit", dataset_id="wages", formula="lwage ~ educ", name="eq_a")
    run("data_list_datasets")
    with pytest.raises(ValueError):
        run("ols_fit", dataset_id="wages", formula="lwage ~ educ", cov_type="bogus")  # fails the tool's schema
    with pytest.raises(UnknownTool):
        run("nope")
    imp, fit, read, bad = ctx.journal.list()
    assert [e["seq"] for e in (imp, fit, read, bad)] == [1, 2, 3, 4]
    assert (imp["mutating"], fit["mutating"], read["mutating"]) == (True, True, False)
    assert fit["tool"] == "ols_fit" and fit["ok"] and fit["kind"] == "api" and fit["actor"] == "user"
    assert fit["code"] == 'eq_a = qx.equation("eq_a", page="wages", method="ls", formula="lwage ~ educ")'
    assert fit["affected"] == [{"kind": "object", "name": "eq_a", "version_before": None, "version_after": 1}]
    assert imp["affected"][0]["kind"] == "page" and imp["affected"][0]["change"] == "created"
    assert imp["affected"][0]["snapshot"].startswith("src_")  # the immutable source snapshot of the import
    assert not bad["ok"] and "cov_type" in bad["error"] and bad["mutating"]
    assert out["version"] == 1 and fit["duration_ms"] >= 0
    run("ols_fit", dataset_id="wages", formula="lwage ~ educ + exper", name="eq_a")
    assert ctx.journal.get(5)["affected"][0] | {} == {
        "kind": "object", "name": "eq_a", "version_before": 1, "version_after": 2}
    assert [v["journal_seq"] for v in _version_rows(ctx, "eq_a")] == [2, 5]


def _version_rows(ctx, name):
    rows = ctx.workspace._con.execute(
        "SELECT version, journal_seq FROM _qtrix_object_versions WHERE name = ? ORDER BY version", [name]).fetchall()
    return [{"version": v, "journal_seq": s} for v, s in rows]


def test_dataset_ids_are_journaled_as_names(imported):
    ctx = imported
    page_id = ctx.workspace.get("wages").id
    run("data_summary_stats", dataset_id=page_id, show_in_panel=False)
    assert ctx.journal.list()[-1]["args"]["dataset_id"] == "wages"


def test_policy_hook_and_locks(imported):
    ctx = imported
    run("ols_fit", dataset_id="wages", formula="lwage ~ educ", name="eq01")
    run("obj_lock", name="eq01")
    with pytest.raises(LockedObjectError):
        run("ols_fit", actor="agent:lead", channel="chat", dataset_id="wages", formula="lwage ~ educ", name="eq01")
    assert not ctx.journal.list()[-1]["ok"] and "locked" in ctx.journal.list()[-1]["error"]
    assert ctx.objects.get("eq01").version == 1
    run("obj_copy", actor="agent:lead", name="eq01", new_name="eq_copy")  # copying a locked object is allowed
    assert run("obj_reestimate", name="eq01", changes={"cov_type": "HC3"})["version"] == 2  # the user may
    seen: list[str] = []

    def autonomy(req: PolicyRequest) -> None:  # the hook E7·18 plugs into
        seen.append(req.call.tool)
        if req.actor == "mcp" and req.mutating:
            raise PolicyDenied("supervised mode: mcp writes need approval")

    register_policy(autonomy)
    try:
        with pytest.raises(PolicyDenied, match="supervised"):
            run("obj_delete", actor="mcp", channel="mcp", name="eq_copy")
        assert ctx.objects.exists("eq_copy") and "obj_delete" in seen
    finally:
        from qtrix_engine import dispatch as d

        d._POLICIES.remove(autonomy)


# ---- every mutation path produces an entry ------------------------------------------------------------------
def test_agent_wrapper_dispatches_as_agent(imported):
    ctx = imported
    spec = next(s for s in all_tools() if s.name == "ols_fit")
    out = _wrap(spec)("wages", "lwage ~ educ", name="eq_agent")  # positional, like a direct call
    assert out["name"] == "eq_agent"
    e = ctx.journal.list()[-1]
    assert (e["actor"], e["channel"], e["tool"], e["mutating"]) == ("agent:lead_econometrician", "chat", "ols_fit",
                                                                    True)
    ctx.objects.lock("eq_agent")
    with pytest.raises(ModelRetry, match="locked"):
        _wrap(spec)("wages", "lwage ~ educ", name="eq_agent")
    assert ctx.journal.list()[-1]["ok"] is False


async def test_mcp_dispatches_as_mcp(imported):
    ctx = imported
    server = build_server()
    await server.call_tool("ols_fit", {"dataset_id": "wages", "formula": "lwage ~ educ", "name": "eq_mcp"})
    await server.call_tool("data_list_datasets", {})
    fit, read = ctx.journal.list()[-2:]
    assert (fit["actor"], fit["channel"], fit["mutating"]) == ("mcp", "mcp", True)
    assert (read["actor"], read["mutating"]) == ("mcp", False)
    assert ctx.objects.exists("eq_mcp")


def test_api_paths_all_leave_entries(tmp_path, wages_csv):
    c = TestClient(create_app(Settings(home=tmp_path, token=None, local_only=False), with_memory=False))
    j = c.app.state.engine.ctx.journal
    assert c.post("/datasets/import", json={"path": str(wages_csv)}).json()["name"] == "wages"
    assert c.put("/datasets/wages/structure", json={"kind": "panel", "freq": "A", "date_col": "year",
                                                    "id_col": "id"}).status_code == 200
    assert c.put("/datasets/wages/sample", json={"sample": "if year >= 2002"}).json()["n_selected"] > 0
    c.post("/tools/ols_fit", json={"dataset_id": "wages", "formula": "lwage ~ educ", "name": "eq_a"})
    assert c.post("/objects/eq_a/lock").json()["locked"]
    assert c.post("/objects/eq_a/reestimate", json={"changes": {"cov_type": "HC3"}}).json()["version"] == 2
    assert c.post("/objects/eq_a/unlock").status_code == 200
    assert c.post("/objects/eq_a/copy", json={"new_name": "eq_b"}).json()["parent"] == "eq_a"
    assert c.post("/objects/eq_b/rename", json={"new_name": "eq_c"}).json()["name"] == "eq_c"
    assert c.post("/objects/eq_c/delete").json() == {"deleted": "eq_c"}
    assert c.put("/datasets/wages/panel", json={}).status_code == 200  # resets to a cross-section
    assert c.delete("/datasets/wages").status_code == 200
    rows = c.get("/journal").json()
    assert [(e["tool"], e["actor"], e["channel"]) for e in rows] == [
        ("data_import_file", "user", "ui"), ("data_set_structure", "user", "ui"), ("data_set_sample", "user", "ui"),
        ("ols_fit", "api", "api"), ("obj_lock", "user", "ui"), ("obj_reestimate", "user", "ui"),
        ("obj_unlock", "user", "ui"), ("obj_copy", "user", "ui"), ("obj_rename", "user", "ui"),
        ("obj_delete", "user", "ui"), ("data_set_structure", "user", "ui"), ("data_delete_dataset", "user", "ui")]
    assert all(e["mutating"] and e["ok"] for e in rows)
    assert len(j.list()) == 12
    # /tools refuses locked objects (actor "api") and the failure is journaled
    c.post("/objects/eq_a/lock")
    r = c.post("/tools/obj_delete", json={"name": "eq_a"})
    assert r.status_code == 409 and c.get("/journal", params={"ok": False}).json()[0]["tool"] == "obj_delete"
    assert c.post("/tools/journal_undo", json={"seq": 1}).status_code == 404  # internal tools are not exposed


def test_chat_turn_is_journaled_as_the_agent(tmp_path, wages_csv):
    c = chat_client(tmp_path)
    c.post("/datasets/import", json={"path": str(wages_csv)})
    assert any(e["type"] == "tool_result" and e["ok"] for e in events(c.post("/chat", json={"message": "fit"})))
    c.post("/tools/ols_fit", json={"dataset_id": "wages", "formula": "lwage ~ educ", "name": "eq_api"})
    agent = c.get("/journal", params={"actor": "agent"}).json()
    assert [(e["tool"], e["channel"], e["actor"]) for e in agent] == [
        ("ols_fit", "chat", "agent:lead_econometrician")]
    assert agent[0]["turn_id"] and agent[0]["cell_id"].startswith(agent[0]["turn_id"])
    # history tells the agent from the user and the API apart
    by_actor = {a: [e["tool"] for e in c.get("/journal", params={"actor": a}).json()] for a in ("user", "api")}
    assert by_actor == {"user": ["data_import_file"], "api": ["ols_fit"]}
    assert [e["tool"] for e in c.get("/journal", params={"mutating": True, "since": 1}).json()] == ["ols_fit"] * 2
    assert c.get("/journal", params={"mutating": False}).json() == []


def test_edited_from_links_a_rerun_to_its_origin(tmp_path, wages_csv):
    c = TestClient(create_app(Settings(home=tmp_path, token=None, local_only=False), with_memory=False))
    c.post("/datasets/import", json={"path": str(wages_csv)})
    c.post("/tools/ols_fit", json={"dataset_id": "wages", "formula": "lwage ~ educ", "name": "eq_a"})
    seq = c.get("/journal", params={"tool": "ols_fit"}).json()[0]["seq"]
    out = c.post("/tools/ols_fit", params={"edited_from": seq, "cell_id": "cell-1"},
                 json={"dataset_id": "wages", "formula": "lwage ~ educ + exper", "name": "eq_a"}).json()
    assert out["version"] == 2
    edit = c.get("/journal", params={"since": seq}).json()[-1]
    assert edit["edited_from"] == seq and edit["cell_id"] == "cell-1" and "exper" in edit["code"]
    assert c.get(f"/journal/{edit['seq']}").json()["edited_from"] == seq
    assert c.post("/tools/ols_fit", params={"edited_from": 999},
                  json={"dataset_id": "wages", "formula": "lwage ~ educ"}).status_code == 404
    assert f"edited from [{seq}]" in c.get("/journal/script").text


# ---- undo -------------------------------------------------------------------------------------------------------
def test_undo_restores_object_versions(imported):
    ctx = imported
    run("ols_fit", dataset_id="wages", formula="lwage ~ educ", name="eq_a")  # 2
    run("ols_fit", dataset_id="wages", formula="lwage ~ educ", name="eq_b")  # 3
    run("ols_compare", result_ids=["eq_a", "eq_b"], name="tab01")  # 4
    v1 = coefs(ctx, "eq_a")
    run("ols_fit", dataset_id="wages", formula="lwage ~ educ + exper", sample="if year > 2002", name="eq_a")  # 5
    assert ctx.objects.get("tab01").status == "stale" and ctx.objects.get("eq_a").version == 2
    with pytest.raises(UndoError, match="changed the same"):
        run("journal_undo", seq=2)  # a later call (5) changed eq_a
    res = run("journal_undo", seq=5)
    assert res["restored"] == ["eq_a v1", "tab01 v1"]  # the stale mark it put on tab01 is undone with it
    assert ctx.objects.get("eq_a").version == 1 and coefs(ctx, "eq_a") == v1
    assert ctx.objects.get("tab01").status == "unchecked"  # the stale mark of the undone call is undone too
    assert [v["version"] for v in ctx.objects.versions("eq_a")] == [1]
    assert ctx.journal.list(since=4)[0]["undone_by"] == ctx.journal.list()[-1]["seq"] == 7  # [6] was the refusal
    with pytest.raises(UndoError, match="already undone"):
        undo(5)
    # the next estimate numbers its version as if the undone one never happened, and journals it normally
    assert run("ols_fit", dataset_id="wages", formula="lwage ~ exper", name="eq_a")["version"] == 2
    # undoing a creation removes the object; the compare depended on it so it goes stale
    run("journal_undo", seq=3)
    assert not ctx.objects.exists("eq_b") and ctx.objects.get("tab01").status == "stale"
    # rename / delete / lock round-trip
    run("obj_rename", name="eq_a", new_name="eq_z")
    assert ctx.objects.exists("eq_z")
    run("journal_undo", seq=ctx.journal.list()[-1]["seq"])
    assert ctx.objects.exists("eq_a") and not ctx.objects.exists("eq_z")
    assert [v["version"] for v in ctx.objects.versions("eq_a")] == [1, 2]
    run("obj_delete", name="eq_a")
    run("journal_undo", seq=ctx.journal.list()[-1]["seq"])
    assert ctx.objects.get("eq_a").version == 2
    run("obj_lock", name="eq_a")
    run("journal_undo", seq=ctx.journal.list()[-1]["seq"])
    assert not ctx.objects.get("eq_a").locked
    run("data_list_datasets")
    with pytest.raises(UndoError, match="nothing to undo"):
        undo(ctx.journal.list()[-1]["seq"])


def test_undo_restores_page_state_and_data(imported):
    ctx = imported
    ws = ctx.workspace
    run("data_genr", dataset_id="wages", name="exper2", expression="exper * exper")  # 2
    run("ols_fit", dataset_id="wages", formula="lwage ~ educ", name="eq_a")  # 3
    before = ws.df("wages").copy()
    cols = list(before.columns)
    run("data_set_sample", dataset_id="wages", sample="if year >= 2004")  # 4
    assert ws.get("wages").sample == "if year >= 2004"
    run("journal_undo", seq=4)
    assert ws.get("wages").sample is None
    run("data_genr", dataset_id="wages", name="exper2", expression="exper + 1", overwrite=True)  # 6: data changes
    assert ctx.objects.get("eq_a").status in {"unchecked", "stale"}  # lazily marked on read
    run("data_set_structure", dataset_id="wages", kind="panel", freq="A", date_col="year", id_col="id")  # 7
    assert ws.get("wages").kind == "panel" and ctx.objects.get("eq_a").status == "stale"
    run("journal_undo", seq=7)
    assert ws.get("wages").kind == "cross_section" and list(ws.df("wages").columns) == cols
    run("journal_undo", seq=6)
    after = ws.df("wages")
    assert (after["exper2"] == before["exper2"]).all() and ws.generated_series("wages") == {"exper2": "exper * exper"}
    assert ctx.objects.get("eq_a").status == "unchecked"  # it only went stale because of the data changes
    run("journal_undo", seq=2)  # the genr that created the column
    assert "exper2" not in ws.df("wages").columns and ws.generated_series("wages") == {}
    # undo of an import drops the page (its source snapshot stays for replay); a page deletion is not undoable
    run("data_transform", dataset_id="wages", new_columns={"x": "educ * 2"}, save_as="w2")
    run("journal_undo", seq=ctx.journal.list()[-1]["seq"])
    assert [p.name for p in ws.list()] == ["wages"]
    run("data_delete_dataset", dataset_id="wages")
    with pytest.raises(UndoError, match="not undoable"):
        undo(ctx.journal.list()[-1]["seq"])


# ---- replay --------------------------------------------------------------------------------------------------------
def test_replay_after_deleting_all_objects_reproduces_identical_coefficients(imported, tmp_path):
    ctx = imported
    run("data_set_structure", dataset_id="wages", kind="panel", freq="A", date_col="year", id_col="id")
    run("data_genr", dataset_id="wages", name="exper2", expression="exper * exper")
    run("data_set_sample", dataset_id="wages", sample="2000 2006")
    run("ols_fit", actor="agent:lead", channel="chat", dataset_id="wages", formula="lwage ~ educ + exper2",
        cov_type="cluster", cluster="id", name="eq_ols")
    run("ols_fit", actor="agent:lead", channel="chat", dataset_id="wages", formula="lwage ~ educ + exper2",
        sample="2002 2006", cov_type="cluster", cluster="id", name="eq_ols")  # version 2
    run("panel_fit", dataset_id="wages", formula="lwage ~ 1 + exper + union", estimator="fe", name="eq_fe")
    run("ols_binary", dataset_id="wages", formula="high ~ educ", name="eq_logit")
    run("ols_compare", result_ids=["eq_ols", "eq_fe"], name="tab01")
    run("obj_group", name="gr_x", members=["lwage", "educ"], dataset_id="wages")
    run("obj_rename", name="eq_logit", new_name="eq_binary")
    run("obj_copy", name="eq_fe", new_name="eq_fe_copy")
    run("obj_delete", name="eq_fe_copy")
    run("ols_fit", dataset_id="wages", formula="lwage ~ union", name="eq_undone")
    run("journal_undo", seq=ctx.journal.list()[-1]["seq"])
    run("data_transform", dataset_id="wages", new_columns={"x": "educ * 2"}, save_as="wages_t")
    run("ols_fit", dataset_id="wages_t", formula="lwage ~ x", name="eq_t")
    run("data_summary_stats", dataset_id="wages", show_in_panel=False)  # read-only: skipped by replay

    names = [o.name for o in ctx.objects.list()]
    recorded = {n: ctx.objects.get(n).summary for n in names}
    coefs_before = {n: coefs(ctx, n) for n in ("eq_ols", "eq_fe")}
    history = {n: [v["summary"] for v in ctx.objects.versions(n)] for n in names}
    n_entries = len(ctx.journal.list())
    pages = [(p.name, p.kind, p.freq, p.sample, p.data_version, p.n_rows) for p in ctx.workspace.list()]
    panel_blocks = len(ctx.panel.blocks())
    for n in names:  # delete every object outside the journal (soft delete: their versions stay as the record)
        ctx.objects.delete(n)
    assert ctx.objects.list() == []
    n_entries = len(ctx.journal.list())

    report = replay()
    assert report["ok"], report
    assert report["versions_checked"] >= len(names) + 1  # every version verified against the recorded summary
    assert {o.name for o in ctx.objects.list()} == set(names) == {
        "eq_ols", "eq_fe", "eq_binary", "tab01", "gr_x", "eq_t"}
    for n in names:
        assert summaries_match(ctx.objects.get(n).summary, recorded[n], 1e-10), n
        assert all(summaries_match(a, b, 1e-10) for a, b in zip(
            [v["summary"] for v in ctx.objects.versions(n)], history[n], strict=True))
    ctx.objects.reset()  # and with objects and versions wiped altogether, coefficients still come back identical
    assert replay()["ok"]
    assert coefs(ctx, "eq_ols") == coefs_before["eq_ols"] and coefs(ctx, "eq_fe") == coefs_before["eq_fe"]  # exact
    for n in names:
        assert summaries_match(ctx.objects.get(n).summary, recorded[n], 1e-10), n
    assert ctx.objects.get("eq_ols").version == 2 and ctx.objects.get("eq_ols").spec["sample"] == "2002 2006"
    assert [(p.name, p.kind, p.freq, p.sample, p.data_version, p.n_rows) for p in ctx.workspace.list()] == pages
    assert len(ctx.journal.list()) == n_entries  # replay writes no new entries
    assert len(ctx.panel.blocks()) == panel_blocks  # nor does it touch the report
    assert ctx.objects.get("eq_ols").author == "agent:lead"  # original actors are kept
    # it also survives a second replay and a restart (snapshots are tables of the workfile)
    assert replay()["ok"]


def test_replay_reports_a_diverging_summary_and_refuses_foreign_pages(imported):
    ctx = imported
    run("ols_fit", dataset_id="wages", formula="lwage ~ educ", name="eq_a")
    ctx.workspace._con.execute("UPDATE _qtrix_object_versions SET summary = replace(summary, '\"n_obs\": 480', "
                               "'\"n_obs\": 481') WHERE name = 'eq_a'")
    report = replay()
    assert not report["ok"] and report["mismatches"] == [{"name": "eq_a", "version": 1}]
    ctx.workspace.add_dataframe(ctx.workspace.df("wages").head(10), "outsider", "test")  # not from the journal
    with pytest.raises(ReplayError, match="outsider"):
        replay()
    assert ctx.objects.exists("eq_a")  # nothing was wiped


def test_script_is_the_mutating_journal_in_order(tmp_path, wages_csv):
    c = TestClient(create_app(Settings(home=tmp_path, token=None, local_only=False), with_memory=False))
    c.post("/datasets/import", json={"path": str(wages_csv)})
    c.post("/tools/ols_fit", json={"dataset_id": "wages", "formula": "lwage ~ educ", "name": "eq_a"})
    c.post("/tools/data_list_datasets", json={})
    c.post("/tools/ols_fit", json={"dataset_id": "wages", "formula": "lwage ~ educ + exper", "name": "eq_b"})
    assert c.post("/journal/2/undo").json()["restored"] == ["eq_a removed"]
    assert c.post("/journal/2/undo").status_code == 409  # already undone
    assert c.post("/journal/99/undo").status_code == 404
    script = c.get("/journal/script").text
    lines = [ln for ln in script.splitlines() if ln and not ln.startswith("# [")]
    assert lines[0].startswith("# qtrix session script") and lines[1] == "import qtrix as qx"
    assert lines[2].startswith("qx.import_file(")  # wrapped one argument per line: the path is long
    undone = lines.index("# undone by [5]:")
    assert lines[undone + 1].startswith('#   eq_a = qx.equation("eq_a", page="wages"')
    assert lines[-1].startswith('eq_b = qx.equation("eq_b", page="wages"')
    assert "pages()" not in script and "undo(" not in script
    compile(script, "<script>", "exec")
    cell = parse_cell(script)  # the session script is itself an API cell: it reads back as the same calls
    assert cell.kind == "api" and [p.call.tool for p in cell.calls] == ["data_import_file", "ols_fit"]
    assert c.post("/journal/replay").json()["ok"]
