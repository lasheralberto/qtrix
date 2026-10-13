import statsmodels.formula.api as smf
from linearmodels.panel import PanelOLS

from qtrix_engine.tools import all_tools, catalog

from .conftest import make_wages


def tools():
    return {t.name: t.fn for t in all_tools()}


def test_registry_lists_namespaced_tools():
    names = {t["name"] for t in catalog()}
    assert {"ols_fit", "panel_fit", "iv_fit", "ts_arima", "viz_scatter", "ui_add_note", "memory_search"} <= names
    assert all(t["name"].startswith(t["namespace"] + "_") for t in catalog())


def test_ols_matches_statsmodels_and_renders_table(ctx, wages):
    out = tools()["ols_fit"]("wages", "lwage ~ educ + exper + union", cov_type="HC1")
    ref = smf.ols("lwage ~ educ + exper + union", data=make_wages()).fit(cov_type="HC1")
    coef = {c["name"]: c for c in ctx.results.get(out["result_id"]).summary["coefficients"]}
    assert abs(coef["educ"]["coef"] - ref.params["educ"]) < 1e-10
    assert abs(coef["educ"]["se"] - ref.bse["educ"]) < 1e-10
    block = ctx.panel.blocks()[-1]
    assert block.kind == "regression_table" and block.result_id == out["result_id"] and "smf.ols" in block.code


def test_clustered_errors(ctx, wages):
    out = tools()["ols_fit"]("wages", "lwage ~ educ", cov_type="cluster", cluster="id")
    df = make_wages()
    ref = smf.ols("lwage ~ educ", data=df).fit(cov_type="cluster", cov_kwds={"groups": df["id"]})
    se = {c["name"]: c["se"] for c in ctx.results.get(out["result_id"]).summary["coefficients"]}
    assert abs(se["educ"] - ref.bse["educ"]) < 1e-10


def test_panel_fe_matches_linearmodels(ctx, wages):
    t = tools()
    t["data_set_panel_structure"]("wages", "id", "year")
    out = t["panel_fit"]("wages", "lwage ~ 1 + exper + union", estimator="fe")
    df = make_wages().set_index(["id", "year"])
    ref = PanelOLS.from_formula("lwage ~ 1 + exper + union + EntityEffects", data=df).fit(
        cov_type="clustered", cluster_entity=True)
    coef = {c["name"]: c for c in ctx.results.get(out["result_id"]).summary["coefficients"]}
    assert abs(coef["exper"]["coef"] - ref.params["exper"]) < 1e-10
    assert abs(coef["exper"]["se"] - ref.std_errors["exper"]) < 1e-10


def test_diagnostics_compare_and_charts(ctx, wages):
    t = tools()
    a = t["ols_fit"]("wages", "lwage ~ educ")
    b = t["ols_fit"]("wages", "lwage ~ educ + exper")
    diag = t["ols_diagnostics"](b["result_id"])
    assert {x["test"] for x in diag["tests"]} >= {"Breusch-Pagan", "Jarque-Bera", "Durbin-Watson"}
    t["ols_compare"]([a["result_id"], b["result_id"]])
    assert len(ctx.panel.blocks()[-1].data["models"]) == 2
    chart = t["viz_scatter"]("wages", "educ", "lwage")
    t["viz_scatter"]("wages", "exper", "lwage", replace_block_id=chart["block_id"])
    assert ctx.panel.get(chart["block_id"]).title == "lwage vs exper"


def test_time_series_tools_use_dates(ctx, macro):
    t = tools()
    assert t["ts_unit_root"]("macro", "gdp", test="adf")["test"] == "ADF"
    out = t["ts_arima"]("macro", "gdp", order=[1, 0, 0], forecast_steps=3)
    assert list(out["forecast"]) == ["2015-01-01", "2015-04-01", "2015-07-01"]
    forecast_fig = ctx.panel.blocks()[-1].data
    assert forecast_fig["data"][0]["x"][0].startswith("1990-01-01")
    var = t["ts_var"]("macro", ["gdp", "rate"], maxlags=2)
    assert var["sample"].startswith("1990-01-01")


def test_time_series_tools_refuse_cross_sections(ctx, wages):
    import pytest

    with pytest.raises(ValueError, match="needs a time series.*cross-section"):
        tools()["ts_unit_root"]("wages", "lwage")
