"""Named, persistent, re-estimable objects (ADR 0002 §3, E7·05)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from qtrix_engine.api import create_app
from qtrix_engine.config import Settings
from qtrix_engine.context import EngineContext, set_context
from qtrix_engine.objects.estimators import summaries_match
from qtrix_engine.objects.specs import GlmSpec, LsSpec, parse_spec
from qtrix_engine.objects.store import (
    LockedObjectError,
    ObjectError,
    ObjectNotFound,
    StaleObjectError,
    acting_as,
)
from qtrix_engine.tools.registry import all_tools
from qtrix_engine.workspace import Workspace


def tools() -> dict:
    return {t.name: t.fn for t in all_tools()}


def restart(ctx, tmp_path) -> EngineContext:
    """A fresh engine on the same DuckDB file (what a process restart does)."""
    ctx.workspace._con.close()
    return set_context(EngineContext(settings=ctx.settings, workspace=Workspace(tmp_path / "ws.duckdb")))


def test_names_and_defaults(ctx, wages):
    t = tools()
    assert t["ols_fit"]("wages", "lwage ~ educ")["result_id"] == "eq01"
    assert t["ols_fit"]("wages", "lwage ~ educ + exper")["name"] == "eq02"
    assert t["ols_fit"]("wages", "lwage ~ educ", name="EQ_Wage")["name"] == "eq_wage"  # case-insensitive
    for bad in ("1abc", "a-b", "x" * 33, ""):
        with pytest.raises(ObjectError, match="Invalid object name"):
            ctx.objects.valid_name(bad)
    assert ctx.objects.get("EQ_WAGE").name == "eq_wage"
    with pytest.raises(ObjectNotFound):
        ctx.objects.get("nope")


def test_object_survives_restart_and_rebuild_matches(ctx, wages, tmp_path):
    t = tools()
    ctx.workspace.set_structure("wages", "panel", date_col="year", id_col="id")
    t["ols_fit"]("wages", "lwage ~ educ + exper", cov_type="cluster", cluster="id", name="eq01")
    t["panel_fit"]("wages", "lwage ~ 1 + exper + union", estimator="fe", name="eq_fe")
    t["iv_fit"]("wages", "lwage ~ 1 + exper + [educ ~ union + high]", name="eq_iv")
    before = {o.name: o.summary for o in ctx.objects.list()}
    ctx2 = restart(ctx, tmp_path)
    assert {o.name for o in ctx2.objects.list()} == {"eq01", "eq_fe", "eq_iv"}
    for name, summary in before.items():
        obj = ctx2.objects.get(name)
        assert obj.summary == summary and obj.code and obj.spec["method"]
        assert summaries_match(ctx2.objects.rebuild(name).summary, summary, 1e-10)
        assert ctx2.objects.fitted(name) is not None  # lazily rebuilt after the restart
    assert tools()["viz_coefplot"]("eq01")["block_id"]  # views regenerate from the stored object


def test_rebuild_all_methods(ctx, wages, macro):
    t = tools()
    t["ols_fit"]("wages", "lwage ~ educ", weights="educ", name="e_wls")
    t["ols_binary"]("wages", "high ~ educ", name="e_logit")
    t["ts_arima"]("macro", "gdp", order=[1, 0, 0], forecast_steps=2, name="e_arima")
    t["ts_var"]("macro", ["gdp", "rate"], maxlags=2, name="var_m")
    ctx.objects.estimate(GlmSpec(page="wages", formula="lwage ~ educ", family="gaussian"), "e_glm")
    for name in ("e_wls", "e_logit", "e_arima", "var_m", "e_glm"):
        assert summaries_match(ctx.objects.rebuild(name).summary, ctx.objects.get(name).summary), name
    assert ctx.objects.get("var_m").type == "var" and ctx.objects.get("e_arima").type == "equation"


def test_spec_roundtrip():
    spec = LsSpec(page="p", formula="y ~ x", sample="if a > 1", cov_type="HAC")
    assert parse_spec(spec.dump()) == spec
    with pytest.raises(ValueError):
        LsSpec(page="p", formula="y ~ x", bogus=1)


def test_reestimate_creates_version_and_diff(ctx, wages):
    t = tools()
    t["ols_fit"]("wages", "lwage ~ educ + exper", name="eq_w")
    out = t["ols_fit"]("wages", "lwage ~ educ + exper", sample="if year <= 2003", name="eq_w")
    assert out["version"] == 2 and out["name"] == "eq_w"
    assert set(out["diff_vs_previous"]) == {"Intercept", "educ", "exper"}
    obj = ctx.objects.get("eq_w")
    assert obj.version == 2 and obj.spec["sample"] == "if year <= 2003"
    diff = obj.summary["diff"]
    assert diff["from_version"] == 1 and diff["stats"]["n_obs"]["before"] == 480 and obj.summary["n_obs"] == 240
    assert [v["version"] for v in ctx.objects.versions("eq_w")] == [1, 2]
    assert ctx.objects.diff("eq_w", 1, 2)["coefficients"]["educ"]["delta"] == pytest.approx(
        diff["coefficients"]["educ"]["delta"])
    assert summaries_match(ctx.objects.rebuild("eq_w").summary, obj.summary)  # the diff does not break rebuilds
    v3 = ctx.objects.reestimate("eq_w", cov_type="nonrobust")  # option change, same sample
    assert v3.version == 3 and v3.spec["sample"] == "if year <= 2003"
    with pytest.raises(ObjectError, match="cannot be replaced"):
        ctx.objects.estimate(parse_spec({"method": "text", "markdown": "x"}), "eq_w")


def test_active_sample_is_frozen_into_the_spec(ctx, wages):
    ctx.workspace.set_sample("wages", "if year <= 2003")
    tools()["ols_fit"]("wages", "lwage ~ educ", name="eq01")
    ctx.workspace.set_sample("wages", None)  # the active sample changes later...
    assert ctx.objects.get("eq01").spec["sample"] == "if year <= 2003"
    assert ctx.objects.rebuild("eq01").summary["n_obs"] == 240  # ...the rebuild still uses the original one


def test_dependants_go_stale_and_refresh(ctx, wages):
    t = tools()
    t["ols_fit"]("wages", "lwage ~ educ", name="eq_a")
    t["ols_fit"]("wages", "lwage ~ educ + exper", name="eq_b")
    cmp = t["ols_compare"](["eq_a", "eq_b"], name="tab_cmp")
    assert ctx.objects.dependants("eq_a") == ["tab_cmp"] and ctx.objects.get("tab_cmp").status == "unchecked"
    assert not ctx.panel.get(cmp["block_id"]).stale
    t["ols_fit"]("wages", "lwage ~ educ", sample="if year <= 2003", name="eq_a")
    stale = ctx.objects.get("tab_cmp")
    assert stale.status == "stale" and "eq_a" in stale.status_detail[0]["message"]
    assert ctx.panel.get(cmp["block_id"]).stale  # the report flags the block of the stale table
    new = t["ols_compare"](["eq_a", "eq_b"], name="tab_cmp")  # refresh = new version
    assert new["version"] == 2 and ctx.objects.get("tab_cmp").status == "unchecked"
    assert ctx.objects.get("tab_cmp").summary["models"][0]["n_obs"] == 240


def test_data_change_makes_object_stale_and_blocks_silent_rebuild(ctx, wages, tmp_path):
    tools()["ols_fit"]("wages", "lwage ~ educ", name="eq01")
    assert ctx.objects.get("eq01").status == "unchecked"
    ctx.workspace.genr("wages", "educ", "educ + 0", overwrite=True)  # rewrites a column: bumps the page's data_version
    assert ctx.objects.get("eq01").status == "stale"
    ctx2 = restart(ctx, tmp_path)  # cache is empty after a restart: stale objects are not rebuilt silently
    with pytest.raises(StaleObjectError, match="re-estimate"):
        ctx2.objects.fitted("eq01")
    assert ctx2.objects.get("eq01").summary["n_obs"] == 480  # the stored results stay readable
    assert tools()["ols_fit"]("wages", "lwage ~ educ", name="eq01")["version"] == 2
    assert ctx2.objects.get("eq01").status == "unchecked"


def test_lock_blocks_agents_not_the_user(ctx, wages):
    t = tools()
    t["ols_fit"]("wages", "lwage ~ educ", name="eq01")
    ctx.objects.lock("eq01")
    for call in (lambda: t["ols_fit"]("wages", "lwage ~ educ + exper", name="eq01"),
                 lambda: ctx.objects.reestimate("eq01", cov_type="HC3"),
                 lambda: t["obj_delete"]("eq01"), lambda: t["obj_rename"]("eq01", "eq99"),
                 lambda: ctx.objects.unlock("eq01")):
        with pytest.raises(LockedObjectError):
            call()
    assert ctx.objects.get("eq01").version == 1
    t["obj_copy"]("eq01", "eq_copy")  # copying is allowed and the copy is not locked
    assert not ctx.objects.get("eq_copy").locked and ctx.objects.get("eq_copy").parent == "eq01"
    with acting_as("user"):
        assert ctx.objects.reestimate("eq01", cov_type="HC3").version == 2
        assert not ctx.objects.unlock("eq01").locked
    assert t["ols_fit"]("wages", "lwage ~ educ", name="eq01")["version"] == 3


def test_agent_wrapper_maps_lock_to_model_retry(ctx, wages):
    from pydantic_ai import ModelRetry

    from qtrix_engine.agents.lead import _wrap

    spec = next(s for s in all_tools() if s.name == "ols_fit")
    tools()["ols_fit"]("wages", "lwage ~ educ", name="eq01")
    ctx.objects.lock("eq01")
    with pytest.raises(ModelRetry, match="locked"):
        _wrap(spec)("wages", "lwage ~ educ", name="eq01")


def test_rename_copy_delete(ctx, wages):
    t = tools()
    t["ols_fit"]("wages", "lwage ~ educ", name="eq_a")
    t["ols_fit"]("wages", "lwage ~ educ + exper", name="eq_b")
    t["ols_compare"](["eq_a", "eq_b"], name="tab01")
    block = ctx.panel.blocks()[0]
    assert block.result_id == "eq_a"
    ctx.objects.rename("eq_a", "eq_base")
    assert ctx.panel.get(block.id).result_id == "eq_base"  # the report follows the rename
    assert ctx.objects.get("tab01").spec["sources"] == ["eq_base", "eq_b"]
    assert [v["version"] for v in ctx.objects.versions("eq_base")] == [1]
    with pytest.raises(ObjectNotFound):
        ctx.objects.get("eq_a")
    with pytest.raises(ObjectError, match="already exists"):
        ctx.objects.rename("eq_base", "eq_b")
    twin = ctx.objects.copy("eq_base")
    assert twin.name == "eq01" and twin.parent == "eq_base" and twin.version == 1
    assert twin.summary == ctx.objects.get("eq_base").summary
    assert ctx.objects.fitted("eq01") is not None
    ctx.objects.delete("eq_base")
    assert not ctx.objects.exists("eq_base") and ctx.objects.get("tab01").status == "stale"
    assert [v["version"] for v in ctx.objects.versions("eq_base")] == [1]  # soft delete keeps the history
    assert ctx.objects.copy("eq_b").name == "eq02"  # default names skip used ones
    with pytest.raises(ObjectError, match="used before"):
        ctx.objects.copy("eq_b", "eq_base")


def test_legacy_result_ids_alias(ctx, wages):
    tools()["ols_fit"]("wages", "lwage ~ educ", name="eq01")
    ctx.objects.add_alias("r_1a2b3c4d", "eq01")
    assert ctx.results.get("r_1a2b3c4d").summary["n_obs"] == 480
    diag = tools()["ols_diagnostics"]("r_1a2b3c4d")  # old ids keep working in tools too
    assert diag["result_id"] == "eq01"
    ctx.objects.rename("eq01", "eq_x")
    assert ctx.results.get("r_1a2b3c4d").name == "eq_x"


def test_groups_events_and_obj_tools(ctx, wages):
    events: list[str] = []
    ctx.objects.subscribe(lambda ev: events.append(ev["event"]))
    t = tools()
    t["obj_group"]("gr_wage", ["lwage", "educ"], dataset_id="wages")
    with pytest.raises(ValueError, match="Series not in page"):
        t["obj_group"]("gr_bad", ["nope"], dataset_id="wages")
    t["ols_fit"]("wages", "lwage ~ educ", name="eq01")
    t["ols_fit"]("wages", "lwage ~ educ", name="eq01")
    assert events == ["version", "version", "version"]
    assert ctx.objects.get("gr_wage").summary["members"] == ["lwage", "educ"]
    assert [o["name"] for o in t["obj_list"]()] == ["gr_wage", "eq01"]
    assert t["obj_versions"]("eq01")["last_diff"]["to_version"] == 2
    assert t["obj_get"]("eq01")["version"] == 2


def test_api_objects(tmp_path, wages_csv):
    c = TestClient(create_app(Settings(home=tmp_path, token=None, local_only=False), with_memory=False))
    c.post("/datasets/import", json={"path": str(wages_csv)})
    c.post("/tools/ols_fit", json={"dataset_id": "wages", "formula": "lwage ~ educ", "name": "eq_a"})
    c.post("/tools/ols_fit", json={"dataset_id": "wages", "formula": "lwage ~ educ", "name": "eq_a",
                                   "sample": "if year >= 2002"})
    c.post("/tools/ols_compare", json={"result_ids": ["eq_a"], "name": "tab01"})
    assert [o["name"] for o in c.get("/objects").json()] == ["eq_a", "tab01"]
    assert [o["name"] for o in c.get("/objects", params={"type": "table"}).json()] == ["tab01"]
    one = c.get("/objects/eq_a").json()
    assert one["version"] == 2 and one["dependants"] == ["tab01"] and one["spec"]["sample"] == "if year >= 2002"
    hist = c.get("/objects/eq_a/versions").json()
    assert [v["version"] for v in hist["versions"]] == [1, 2] and "educ" in hist["diff"]["coefficients"]
    assert c.get("/objects/nope").status_code == 404
    assert c.post("/objects/eq_a/lock").json()["locked"] is True
    # agents and /tools cannot touch a locked object; the user (these endpoints) can
    r = c.post("/tools/ols_fit", json={"dataset_id": "wages", "formula": "lwage ~ educ", "name": "eq_a"})
    assert r.status_code == 409 and "locked" in r.json()["detail"]
    assert c.post("/objects/eq_a/reestimate", json={"changes": {"cov_type": "HC3"}}).json()["version"] == 3
    assert c.post("/objects/eq_a/copy", json={"new_name": "eq_b"}).json()["parent"] == "eq_a"
    assert c.post("/objects/eq_a/rename", json={"new_name": "eq_c"}).json()["name"] == "eq_c"
    assert c.post("/objects/eq_c/rename", json={"new_name": "Bad Name"}).status_code == 400
    assert c.post("/objects/eq_c/unlock").json()["locked"] is False
    assert c.post("/objects/eq_c/delete").json() == {"deleted": "eq_c"}
    assert c.get("/objects/eq_c").status_code == 404
