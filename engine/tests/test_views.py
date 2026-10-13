"""Views and procs of objects and the report's live blocks (ADR 0002 §4-5, E7·08)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from qtrix_engine.api import create_app
from qtrix_engine.config import Settings
from qtrix_engine.dispatch import Call, dispatch
from qtrix_engine.objects.store import ObjectError, StaleObjectError
from qtrix_engine.objects.views import compute


def run(tool: str, **args):
    return dispatch(Call(tool, args), "user", "ui")


def blocks_of(ctx, name):
    return [b for b in ctx.panel.blocks() if b.result_id == name or (b.ref or {}).get("object") == name]


def test_reestimating_refreshes_live_views_and_leaves_frozen_ones(ctx, wages):
    run("ols_fit", dataset_id="wages", formula="lwage ~ educ", name="eq01")
    resid = run("obj_view", name="eq01", view="residuals", params={"kind": "qq"})
    diag = run("ols_diagnostics", result_id="eq01")
    frozen_block = run("obj_view", name="eq01", view="coefplot")["block_id"]
    run("ui_freeze_block", block_id=frozen_block)
    snapshot = run("obj_freeze", name="eq01", view="estimation", new_name="tab_v1")
    before = {b.id: b.data for b in ctx.panel.blocks()}
    assert len(blocks_of(ctx, "eq01")) == 4

    run("ols_fit", dataset_id="wages", formula="lwage ~ educ + exper", name="eq01")  # version 2
    after = {b.id: b for b in ctx.panel.blocks()}
    assert len(blocks_of(ctx, "eq01")) == 4  # refreshed in place, nothing added
    estimation = next(b for b in blocks_of(ctx, "eq01") if (b.ref or {}).get("view") == "estimation")
    assert "exper" in str(estimation.data) and not estimation.stale
    assert after[resid["block_id"]].data != before[resid["block_id"]] and not after[resid["block_id"]].stale
    assert after[diag["block_id"]].data != before[diag["block_id"]]
    assert after[frozen_block].data == before[frozen_block] and after[frozen_block].stale  # frozen: same, flagged
    tab = ctx.objects.get("tab_v1")
    assert tab.status != "stale" and "exper" not in str(tab.summary)  # a frozen object never follows its source
    assert after[snapshot["block_id"]].data == before[snapshot["block_id"]] and not after[snapshot["block_id"]].stale


def test_views_are_cached_per_version_and_validated(ctx, wages):
    run("ols_fit", dataset_id="wages", formula="lwage ~ educ", name="eq_a")
    first, ref = compute("eq_a", "residuals")
    assert compute("eq_a", "residuals", {"kind": "fitted"})[0] is first  # defaults are part of the key
    assert ref == {"object": "eq_a", "view": "residuals", "params": {"kind": "fitted"}}
    run("obj_reestimate", name="eq_a", changes={"cov_type": "HC3"})
    assert compute("eq_a", "residuals")[0] is not first
    with pytest.raises(ObjectError, match="no view 'irf'.*Views: estimation, coefplot, residuals, diagnostics"):
        compute("eq_a", "irf")
    with pytest.raises(ObjectError, match="kind"):
        compute("eq_a", "residuals", {"kind": "boxplot"})
    with pytest.raises(ObjectError, match="lags"):
        compute("eq_a", "residuals", {"lags": 3})


def test_views_and_procs_follow_type_and_method(ctx, wages, macro):
    run("ols_fit", dataset_id="wages", formula="lwage ~ educ", name="eq_ls")
    run("ols_binary", dataset_id="wages", formula="high ~ educ", name="eq_logit")
    run("ts_var", dataset_id="macro", columns=["gdp", "rate"], maxlags=2, name="var_m")
    run("ts_arima", dataset_id="macro", column="gdp", order=[1, 0, 0], forecast_steps=4, name="eq_ar")
    run("ols_compare", result_ids=["eq_ls", "eq_logit"], name="tab_cmp")
    names = {n: {v["name"] for v in run("obj_views", name=n)["views"]} for n in
             ("eq_ls", "eq_logit", "var_m", "eq_ar", "tab_cmp")}
    assert names == {"eq_ls": {"estimation", "coefplot", "residuals", "diagnostics", "margins", "sensitivity"},
                     "eq_logit": {"estimation", "coefplot", "residuals", "margins"},
                     "var_m": {"estimation", "irf"}, "eq_ar": {"estimation", "coefplot", "residuals", "forecast"},
                     "tab_cmp": {"table"}}
    listing = run("obj_views", name="eq_ls")
    resid = next(v for v in listing["views"] if v["name"] == "residuals")
    assert resid["params"]["kind"]["enum"] == ["fitted", "qq", "histogram"] and resid["group"] == "Residual Diagnostics"
    assert {p["name"] for p in listing["procs"]} >= {"estimate", "freeze", "copy", "rename", "lock", "delete"}
    # Estimation tools put live views in the report: a VAR shows its table and its impulse responses.
    assert {(b.ref or {}).get("view") for b in blocks_of(ctx, "var_m")} == {"estimation", "irf"}
    assert {(b.ref or {}).get("view") for b in blocks_of(ctx, "eq_ar")} == {"estimation", "forecast"}
    assert run("obj_view", name="eq_ls", view="diagnostics", show_in_panel=False)["outcome"] in ("ok", "warn")


def test_stale_objects_keep_cached_views_and_refuse_new_fitted_ones(ctx, wages):
    run("ols_fit", dataset_id="wages", formula="lwage ~ educ", name="eq_s")
    block = run("obj_view", name="eq_s", view="residuals")["block_id"]
    run("data_genr", dataset_id="wages", name="educ", expression="educ + 1", overwrite=True)  # data changed
    assert ctx.objects.get("eq_s").status == "stale"
    assert compute("eq_s", "residuals")[0] is not None  # computed before: still readable
    assert compute("eq_s", "coefplot")[0].kind == "plotly_figure"  # from the summary: always available
    ctx.objects._cache.clear()  # as after a restart: the fitted model is gone and is not rebuilt silently
    with pytest.raises(StaleObjectError):
        compute("eq_s", "residuals", {"kind": "qq"})
    run("ols_fit", dataset_id="wages", formula="lwage ~ educ", name="eq_s")  # re-estimate: fresh again
    assert not ctx.panel.get(block).stale


def test_rename_and_undo_keep_live_views_attached(ctx, wages):
    run("ols_fit", dataset_id="wages", formula="lwage ~ educ", name="eq_r")
    v1 = ctx.panel.get(run("obj_view", name="eq_r")["block_id"]).data
    run("obj_rename", name="eq_r", new_name="eq_mincer")
    (live,) = [b for b in ctx.panel.blocks() if b.ref]
    assert live.ref["object"] == "eq_mincer" and live.result_id == "eq_mincer"
    run("ols_fit", dataset_id="wages", formula="lwage ~ educ + exper", name="eq_mincer")
    assert ctx.panel.get(live.id).data != v1
    seq = ctx.journal.list(tool="ols_fit")[-1]["seq"]
    run("journal_undo", seq=seq)
    assert ctx.panel.get(live.id).data == v1  # undo brings the view back to the restored version


def test_python_and_qx_spelling_of_views(ctx, wages):
    from qtrix_engine.qx import parse_cell, print_call

    code = print_call("obj_view", {"name": "eq01", "view": "residuals", "params": {"kind": "qq"}})
    assert code == 'qx.obj("eq01").view("residuals", kind="qq")'
    assert parse_cell(code).calls[0].call.args == {"name": "eq01", "view": "residuals", "params": {"kind": "qq"}}
    assert print_call("ui_add_view", {"name": "eq01", "view": "coefplot"}) == 'qx.report.add("eq01", "coefplot")'
    assert print_call("obj_freeze", {"name": "eq01", "view": "residuals", "new_name": "gph_r"}) == (
        'qx.obj("eq01").freeze("residuals", "gph_r")')


def test_http_views_and_freeze(tmp_path, wages_csv):
    c = TestClient(create_app(Settings(home=tmp_path, token=None, local_only=False), with_memory=False))
    c.post("/datasets/import", json={"path": str(wages_csv)})
    c.post("/tools/ols_fit", json={"dataset_id": "wages", "formula": "lwage ~ educ", "name": "eq01"})
    listing = c.get("/objects/eq01/views").json()
    assert listing["default_view"] == "estimation" and len(listing["views"]) == 6  # + margins, sensitivity
    qq = c.get("/objects/eq01/views/residuals", params={"params": '{"kind": "qq"}'}).json()
    assert qq["kind"] == "plotly_figure" and qq["ref"]["params"] == {"kind": "qq"} and qq["version"] == 1
    assert c.get("/objects/eq01/views/nope").status_code == 400
    assert c.get("/objects/eq01/views/residuals", params={"params": "{bad"}).status_code == 400
    block = c.get("/panel").json()["blocks"][0]
    assert block["ref"] == {"object": "eq01", "view": "estimation", "params": {}}
    c.post(f"/panel/blocks/{block['id']}/pin", json={"pinned": True})
    frozen = c.post(f"/panel/blocks/{block['id']}/freeze").json()  # the user may freeze a pinned block
    assert frozen["ref"] is None and frozen["pinned"]
    assert c.post("/panel/blocks/nope/freeze").status_code == 404
