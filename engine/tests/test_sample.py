"""Active sample (ADR 0002 §2): grammar, per-structure semantics, tools, persistence and API."""

import duckdb
import numpy as np
import pandas as pd
import pytest
import statsmodels.formula.api as smf
from fastapi.testclient import TestClient

from qtrix_engine.tools import all_tools
from qtrix_engine.workfile import sample as sp
from qtrix_engine.workspace import Workspace

from .conftest import make_macro, make_wages


def tools():
    return {t.name: t.fn for t in all_tools()}


# ---- grammar -----------------------------------------------------------------------

@pytest.mark.parametrize(("text", "ranges", "where"), [
    (None, (), None),
    ("", (), None),
    ("@ALL", (), None),
    ("1990q1 2008q4", (("1990q1", "2008q4"),), None),
    ("1960q1 1979q4, 1990q1 2007q4", (("1960q1", "1979q4"), ("1990q1", "2007q4")), None),
    ("@first 2000q4 if unemp > 5", (("@first", "2000q4"),), "unemp > 5"),
    ("if age >= 25 and female = 1", (), "age >= 25 and female = 1"),
    ("1 100 IF(educ > 10)", (("1", "100"),), "(educ > 10)"),
])
def test_parse(text, ranges, where):
    spec = sp.parse(text)
    assert (spec.ranges, spec.where) == (ranges, where)


@pytest.mark.parametrize("bad", ["1990q1", "@all 1990q1 2000q1", "1990q1 2000q1 if", "if x > 1; drop table t",
                                 "if x > 1 -- comment", "if x /* c */ > 1"])
def test_parse_rejects(bad):
    with pytest.raises(sp.SampleError):
        sp.parse(bad)


def test_text_round_trips():
    spec = sp.parse("  1990q1   2008q4 if  unemp>5 ")
    assert spec.text == "1990q1 2008q4 if unemp>5" and sp.parse(spec.text) == spec
    assert sp.parse("@all").text == "@all"


# ---- time series -----------------------------------------------------------------------

def test_time_series_sample_cuts_the_range(ctx, macro):
    ws = ctx.workspace
    res = ws.set_sample("macro", "1990q1 2008q4")
    assert (res.n_selected, res.n_total, res.first, res.last) == (76, 100, "1990Q1", "2008Q4")
    frame = ws.frame("macro")
    assert len(frame) == 76 and frame.index[-1] == pd.Timestamp("2008-10-01")
    assert ws.frame("macro", sample="@all").shape[0] == 100  # per-call override
    assert ws.get("macro").sample == "1990q1 2008q4"


def test_time_series_keeps_a_regular_index_with_holes(ctx, macro):
    ws = ctx.workspace
    ws.set_sample("macro", "1990q1 1999q4 2010q1 2014q4")
    frame = ws.frame("macro")
    assert frame.shape[0] == 100 and frame["gdp"].notna().sum() == 60  # 40 + 20 quarters
    assert frame.loc["2000-01-01":"2009-10-01", "gdp"].isna().all()
    assert frame.index.freqstr.startswith("QS")


def test_open_ended_and_condition_samples(ctx, macro):
    ws = ctx.workspace
    assert ws.set_sample("macro", "@first 1994q4").n_selected == 20
    assert ws.set_sample("macro", "2010q1 @last").n_selected == 20
    positive = int((make_macro()["gdp"] > 0).sum())
    assert ws.set_sample("macro", "if gdp > 0").n_selected == positive
    assert ws.set_sample("macro", "if rate == rate").n_selected == 100  # '==' is accepted too


def test_conditions_accept_reserved_words_as_column_names(ctx, wages):
    # 'union' is a SQL keyword and a real econometric variable; quoted literals must stay untouched.
    n = int(make_wages()["union"].sum())
    assert ctx.workspace.set_sample("wages", "if union = 1 and 'union' = 'union'").n_selected == n
    assert ctx.workspace.set_sample("wages", 'if "union" = 1').n_selected == n


def test_tools_respect_the_sample_and_forecast_after_it(ctx, macro):
    t = tools()
    t["data_set_sample"]("macro", "1990q1 2004q4")
    out = t["ts_arima"]("macro", "gdp", order=[1, 0, 0], forecast_steps=2)
    assert list(out["forecast"]) == ["2005-01-01", "2005-04-01"]
    assert ctx.results.get(out["result_id"]).summary["n_obs"] == 60
    full = t["ts_arima"]("macro", "gdp", order=[1, 0, 0], sample="@all")
    assert ctx.results.get(full["result_id"]).summary["n_obs"] == 100


# ---- cross-section ------------------------------------------------------------------------

def test_cross_section_condition_matches_pandas(ctx, wages):
    t = tools()
    t["data_set_sample"]("wages", "if educ >= 12 and union = 1")
    out = t["ols_fit"]("wages", "lwage ~ educ + exper", cov_type="HC1")
    sub = make_wages().query("educ >= 12 and union == 1")
    ref = smf.ols("lwage ~ educ + exper", data=sub).fit(cov_type="HC1")
    summary = ctx.results.get(out["result_id"]).summary
    coef = {c["name"]: c for c in summary["coefficients"]}
    assert summary["n_obs"] == len(sub)
    assert abs(coef["educ"]["coef"] - ref.params["educ"]) < 1e-10
    assert abs(coef["educ"]["se"] - ref.bse["educ"]) < 1e-10
    assert "if educ >= 12 and union = 1" in summary["sample"] and f"{len(sub)} of 480" in summary["sample"]


def test_same_equation_two_samples_gives_different_reproducible_results(ctx, wages):
    t = tools()
    full = t["ols_fit"]("wages", "lwage ~ educ")
    t["data_set_sample"]("wages", "1 240")
    first_half = t["ols_fit"]("wages", "lwage ~ educ")
    again = t["ols_fit"]("wages", "lwage ~ educ")
    override = t["ols_fit"]("wages", "lwage ~ educ", sample="@all")

    def educ(out):
        return ctx.results.get(out["result_id"]).summary["coefficients"][1]["coef"]

    assert educ(first_half) != educ(full) and educ(first_half) == educ(again) and educ(override) == educ(full)
    assert ctx.results.get(first_half["result_id"]).summary["n_obs"] == 240
    assert ctx.results.get(full["result_id"]).summary["sample"] == "all 480 observations"


def test_observation_number_ranges(ctx, wages):
    ws = ctx.workspace
    assert ws.set_sample("wages", "1 100 401 @last").n_selected == 180
    assert len(ws.df("wages")) == 180
    assert len(ws.preview("wages", limit=1000)) == 480  # previews and profiles always show everything
    with pytest.raises(sp.SampleError, match="observation number"):
        ws.set_sample("wages", "1990q1 2000q1")


def test_descriptives_and_binary_models_follow_the_sample(ctx, wages):
    t = tools()
    t["data_set_sample"]("wages", "if union = 1")
    n_union = int(make_wages()["union"].sum())
    assert t["data_summary_stats"]("wages", ["lwage"])["lwage"]["count"] == n_union
    assert ctx.results.get(t["ols_binary"]("wages", "high ~ educ")["result_id"]).summary["n_obs"] == n_union


# ---- panels --------------------------------------------------------------------------------

def test_dated_panel_sample_by_year(ctx, wages):
    t = tools()
    t["data_set_structure"]("wages", "panel", date_col="year", id_col="id")
    t["data_set_sample"]("wages", "2002 2005")
    out = t["panel_fit"]("wages", "lwage ~ 1 + exper + union", estimator="fe")
    summary = ctx.results.get(out["result_id"]).summary
    assert summary["n_obs"] == 60 * 4 and "2002–2005" in summary["sample"]


def test_undated_panel_sample_uses_time_index_values(ctx):
    df = pd.DataFrame({"student": np.repeat(range(1, 6), 4), "wave": np.tile([1, 2, 3, 4], 5),
                       "score": np.arange(20.0)})
    info = ctx.workspace.add_dataframe(df, "waves", "t")
    ctx.workspace.set_structure(info.id, "panel_undated", id_col="student", t_col="wave")
    res = ctx.workspace.set_sample("waves", "2 3")
    assert (res.n_selected, res.first, res.last) == (10, "2", "3")
    assert set(ctx.workspace.frame("waves").index.get_level_values("wave")) == {2, 3}


# ---- validation, safety and persistence -----------------------------------------------------

def test_empty_and_invalid_samples_are_rejected_with_guidance(ctx, macro, wages):
    ws = ctx.workspace
    with pytest.raises(sp.SampleError, match="selects no observations.*1990Q1–2014Q4"):
        ws.set_sample("macro", "2030q1 2031q4")
    with pytest.raises(sp.SampleError, match="starts after it ends"):
        ws.set_sample("macro", "2000q1 1990q1")
    with pytest.raises(sp.SampleError, match="Cannot evaluate the sample condition"):
        ws.set_sample("wages", "if nonexistent_column > 1")
    assert ws.get("wages").sample is None and ws.get("macro").sample is None  # nothing was stored


def test_condition_cannot_modify_data(ctx, wages):
    for attack in ("if 1=1) ; drop table wages --", "if (select count(*) from wages) > 0; drop table wages"):
        with pytest.raises(sp.SampleError):
            ctx.workspace.set_sample("wages", attack)
    assert ctx.workspace.get("wages").n_rows == 480


def test_changing_structure_resets_the_sample(ctx, macro):
    ctx.workspace.set_sample("macro", "1990q1 2000q4")
    ctx.workspace.set_structure("macro", "time_series", date_col="quarter")
    assert ctx.workspace.get("macro").sample is None


def test_derived_pages_start_with_every_row(ctx, macro):
    t = tools()
    t["data_set_sample"]("macro", "1990q1 1999q4")
    out = t["data_transform"]("macro", {"gdp2": "gdp * 2"}, save_as="macro2")  # transforms see all rows
    derived = ctx.workspace.get(out["dataset_id"])
    assert derived.n_rows == 100 and derived.sample is None


def test_sample_survives_reopen_and_old_databases_gain_the_column(tmp_path):
    path = tmp_path / "ws.duckdb"
    ws = Workspace(path)
    info = ws.add_dataframe(make_macro(), "macro", "t")
    ws.set_structure(info.id, "time_series", date_col="quarter")
    ws.set_sample("macro", "1990q1 2004q4")
    del ws
    again = Workspace(path)
    assert again.get("macro").sample == "1990q1 2004q4" and again.frame("macro").shape[0] == 60
    del again
    con = duckdb.connect(str(path))
    con.execute("ALTER TABLE _qtrix_pages DROP COLUMN sample_spec")  # a schema-2 database
    con.close()
    assert Workspace(path).get("macro").sample is None


def test_structure_description_includes_the_sample(ctx, macro):
    t = tools()
    t["data_set_sample"]("macro", "2000q1 @last")
    desc = t["data_describe_structure"]("macro")
    assert desc["active_sample"]["n_selected"] == 60 and desc["active_sample"]["first"] == "2000Q1"
    assert t["data_set_sample"]("macro")["description"] == "1990Q1–2014Q4"  # default resets to @all


def test_agent_context_shows_the_active_sample(ctx, macro):
    from qtrix_engine.agents.lead import workspace_context

    assert "ACTIVE SAMPLE" not in workspace_context()
    ctx.workspace.set_sample("macro", "1990q1 2004q4")
    assert "ACTIVE SAMPLE: 1990q1 2004q4" in workspace_context()


def test_sample_endpoints(tmp_path):
    from qtrix_engine.api import create_app
    from qtrix_engine.config import Settings

    c = TestClient(create_app(Settings(home=tmp_path, token=None, local_only=False), with_memory=False))
    path = tmp_path / "wages.csv"
    make_wages().to_csv(path, index=False)
    info = c.post("/datasets/import", json={"path": str(path)}).json()
    assert info["sample"] is None
    r = c.put(f"/datasets/{info['id']}/sample", json={"sample": "if educ > 15"})
    assert r.status_code == 200 and r.json()["n_selected"] == int((make_wages()["educ"] > 15).sum())
    assert c.get(f"/datasets/{info['id']}").json()["sample"] == "if educ > 15"
    assert c.get(f"/datasets/{info['id']}/sample").json()["sample"] == "if educ > 15"
    assert c.get(f"/datasets/{info['id']}/profile").json()[0]["name"] == "id"  # profile covers all rows
    bad = c.put(f"/datasets/{info['id']}/sample", json={"sample": "if nope > 1"})
    assert bad.status_code == 400 and "Cannot evaluate" in bad.json()["detail"]
    reset = c.put(f"/datasets/{info['id']}/sample", json={"sample": None}).json()
    assert reset["description"] == "all 480 observations"
