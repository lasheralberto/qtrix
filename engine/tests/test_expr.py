"""Expression language (ADR 0002 §3): EViews time operators in formulas and genr, checked against pandas."""

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from linearmodels.panel import PanelOLS

from qtrix_engine import expr
from qtrix_engine.tools import all_tools
from qtrix_engine.workspace import Workspace

from .conftest import make_wages


def tools():
    return {t.name: t.fn for t in all_tools()}


def make_levels(n: int = 100, seed: int = 3) -> pd.DataFrame:
    """Quarterly positive series (log random walks) so dlog() is defined everywhere."""
    rng = np.random.default_rng(seed)
    gdp = np.exp(np.cumsum(0.01 + rng.normal(scale=0.02, size=n)) + 5)
    rate = 2 + np.cumsum(rng.normal(scale=0.3, size=n))
    return pd.DataFrame({"quarter": [f"{1990 + i // 4}Q{i % 4 + 1}" for i in range(n)], "gdp": gdp, "r": rate})


@pytest.fixture
def levels(ctx):
    info = ctx.workspace.add_dataframe(make_levels(), "levels", "synthetic")
    return ctx.workspace.set_structure(info.id, "time_series", date_col="quarter")


@pytest.fixture
def panel(ctx, wages):
    return ctx.workspace.set_structure("wages", "panel", date_col="year", id_col="id")


def summary_of(ctx, out):
    return ctx.results.get(out["result_id"]).summary


def coefs(ctx, out):
    return {c["name"]: c["coef"] for c in summary_of(ctx, out)["coefficients"]}


# ---- translation ---------------------------------------------------------------------------

@pytest.mark.parametrize(("source", "code"), [
    ("y(-1)", "_qx_L(y, 1)"),
    ("y(-4)", "_qx_L(y, 4)"),
    ("y(+2)", "_qx_L(y, -2)"),
    ("y(1)", "_qx_L(y, -1)"),
    ("dlog(y(-1))", "_qx_dlog(_qx_L(y, 1))"),
    ("d(y, 1, 4)", "_qx_d(y, 1, 4)"),
    ("@trend", "_qx_trend()"),
    ("@seas(2)", "_qx_seas(2)"),
    ("@MOVAV(y, 4)", "_qx_movav(y, 4)"),
    ("log(y)", "_qx_log(y)"),
    ("np.log(y)", "np.log(y)"),  # numpy stays numpy
    ("foo(-1)", "foo(-1)"),  # only columns can be lagged with the x(-k) syntax
    ("I(y**2)", "I(y**2)"),
])
def test_translate(source, code):
    assert expr.translate(source, ["y"]) == code


def test_a_column_named_like_an_operator_still_works_as_a_column():
    assert expr.translate("d + d(-1) + d(d)", ["d"]) == "d + _qx_L(d, 1) + _qx_d(d)"


def test_unknown_at_function_and_forbidden_code_are_rejected():
    with pytest.raises(expr.ExprError, match="Unknown function @nope"):
        expr.translate("@nope(1)", [])
    for bad in ("__import__('os').system('x')", "open('f')", "x + eval('1')", "(lambda: 1)()"):
        with pytest.raises(expr.ExprError, match="not allowed"):
            expr.translate(bad, ["x"])


# ---- quarterly equation: the acceptance example ------------------------------------------------

def test_dlog_ols_matches_manual_pandas(ctx, levels):
    out = tools()["ols_fit"]("levels", "dlog(gdp) ~ dlog(gdp(-1)) + d(r)", cov_type="HC1")
    d = make_levels()
    y, x1, x2 = np.log(d.gdp).diff(), np.log(d.gdp.shift(1)).diff(), d.r.diff()
    X = sm.add_constant(pd.DataFrame({"a": x1, "b": x2}))
    ok = y.notna() & X.notna().all(axis=1)
    ref = sm.OLS(y[ok], X[ok]).fit(cov_type="HC1")
    summary = summary_of(ctx, out)
    got = {c["name"]: c for c in summary["coefficients"]}
    assert list(got) == ["Intercept", "dlog(gdp(-1))", "d(r)"]  # terms print as the user wrote them
    assert summary["n_obs"] == int(ok.sum()) == 98  # 1990Q1 and 1990Q2 lose their lagged difference
    for name, col in (("Intercept", "const"), ("dlog(gdp(-1))", "a"), ("d(r)", "b")):
        assert abs(got[name]["coef"] - ref.params[col]) < 1e-10
        assert abs(got[name]["se"] - ref.bse[col]) < 1e-10
    assert summary["formula"] == "dlog(gdp) ~ dlog(gdp(-1)) + d(r)"  # the user's text, not the rewrite


def test_seasonal_trend_dummy_and_recode_terms(ctx, levels):
    formula = ("dlog(gdp) ~ @trend + @seas(1) + dummy('2008q3', '2009q2') + recode(r > 2, 1, 0) + "
               "movav(r, 4) + pch(gdp)")
    out = tools()["ols_fit"]("levels", formula)
    d = make_levels()
    q = np.arange(100)
    frame = pd.DataFrame({
        "y": np.log(d.gdp).diff(), "trend": q, "seas": (q % 4 == 0).astype(int),
        "dum": ((q >= 74) & (q <= 77)).astype(int), "rec": (d.r > 2).astype(int),
        "mav": d.r.rolling(4).mean(), "pch": d.gdp / d.gdp.shift(1) - 1,
    }).dropna()
    ref = sm.OLS(frame["y"], sm.add_constant(frame.drop(columns="y"))).fit(cov_type="HC1")
    got = coefs(ctx, out)
    assert list(got) == ["Intercept", "@trend", "@seas(1)", "dummy('2008q3', '2009q2')", "recode(r > 2, 1, 0)",
                         "movav(r, 4)", "pch(gdp)"]
    for name, col in zip(list(got)[1:], frame.columns[1:], strict=True):
        assert abs(got[name] - ref.params[col]) < 1e-9


def test_operator_values_equal_pandas(ctx, levels):
    frame = ctx.workspace.frame("levels")
    where = expr.Context("time_series", "Q", frame.index, tuple(frame.columns))
    d = make_levels()

    def ev(text):
        return expr.evaluate(text, frame, where).to_numpy()

    np.testing.assert_allclose(ev("d(gdp, 2)"), d.gdp.diff().diff(), equal_nan=True)
    np.testing.assert_allclose(ev("d(r, 1, 4)"), d.r.diff().diff(4), equal_nan=True)
    np.testing.assert_allclose(ev("dlog(gdp, 1, 4)"), np.log(d.gdp).diff().diff(4), equal_nan=True)
    np.testing.assert_allclose(ev("r(+1)"), d.r.shift(-1), equal_nan=True)
    np.testing.assert_allclose(ev("L(r, 3)"), d.r.shift(3), equal_nan=True)
    np.testing.assert_allclose(ev("movsum(r, 3)"), d.r.rolling(3).sum(), equal_nan=True)
    np.testing.assert_allclose(ev("@trend"), np.arange(100))
    np.testing.assert_allclose(ev("trend('1991q1')"), np.arange(100) - 4)
    np.testing.assert_allclose(ev("sqrt(gdp) + abs(-r) + exp(0) + ln(gdp)"),
                               np.sqrt(d.gdp) + d.r.abs() + 1 + np.log(d.gdp))


def test_gaps_stay_gaps_and_lags_follow_the_calendar(ctx):
    df = make_levels(20).drop(index=[5]).reset_index(drop=True)  # 1991Q2 is missing
    info = ctx.workspace.add_dataframe(df, "gappy", "t")
    ctx.workspace.set_structure(info.id, "time_series", date_col="quarter")
    ctx.workspace.genr("gappy", "r1", "r(-1)")
    got = ctx.workspace.df("gappy")["r1"].to_numpy()
    assert not np.isnan(got[5]) and np.isnan(got[6])  # the lag of 1991Q3 is the missing 1991Q2


# ---- the sample and lags ---------------------------------------------------------------------

def test_lags_look_before_the_sample_starts(ctx, levels):
    t = tools()
    inside = t["ols_fit"]("levels", "gdp ~ gdp(-1)", sample="1995q1 2008q4")
    assert summary_of(ctx, inside)["n_obs"] == 56  # 1995Q1 uses 1994Q4, which is outside the sample
    d = make_levels()
    ref = sm.OLS(d["gdp"].iloc[20:76], sm.add_constant(d["gdp"].shift(1).iloc[20:76])).fit(cov_type="HC1")
    assert abs(coefs(ctx, inside)["gdp(-1)"] - ref.params["gdp"]) < 1e-10
    first = t["ols_fit"]("levels", "gdp ~ gdp(-1)", sample="@first 2000q4")
    assert summary_of(ctx, first)["n_obs"] == 43  # 1990Q1 has no lag at all


def test_active_sample_and_lags_together_in_a_panel(ctx, panel):
    t = tools()
    t["data_set_sample"]("wages", "2003 2006")
    out = t["ols_fit"]("wages", "d(lwage) ~ lwage(-1) + d(exper)", cov_type="nonrobust")
    df = make_wages()
    g = df.groupby("id")
    frame = pd.DataFrame({"y": g.lwage.diff(), "a": g.lwage.shift(1), "b": g.exper.diff(), "year": df.year})
    frame = frame[(frame.year >= 2003) & (frame.year <= 2006)].dropna()
    ref = sm.OLS(frame["y"], sm.add_constant(frame[["a", "b"]])).fit()
    got = coefs(ctx, out)
    assert summary_of(ctx, out)["n_obs"] == len(frame) == 240
    assert abs(got["lwage(-1)"] - ref.params["a"]) < 1e-10 and abs(got["d(exper)"] - ref.params["b"]) < 1e-10


# ---- panels ---------------------------------------------------------------------------------

def test_panel_lags_stay_inside_each_entity(ctx, panel):
    ws = ctx.workspace
    ws.genr("wages", "lag_wage", "lwage(-1)")
    expected = make_wages().groupby("id").lwage.shift(1).to_numpy()
    frame = ws.df("wages")
    np.testing.assert_allclose(frame["lag_wage"].to_numpy(), expected, equal_nan=True)
    assert frame.loc[frame["year"] == 2000, "lag_wage"].isna().all()  # no leakage from the previous entity


def test_panel_fixed_effects_with_a_lagged_regressor(ctx, panel):
    out = tools()["panel_fit"]("wages", "lwage ~ 1 + exper + union(-1)", estimator="fe", cov_type="robust")
    df = make_wages()
    df["union_l"] = df.groupby("id").union.shift(1)
    d = df.dropna().set_index(["id", "year"])
    ref = PanelOLS.from_formula("lwage ~ 1 + exper + union_l + EntityEffects", data=d).fit(cov_type="robust")
    got = coefs(ctx, out)
    assert {"exper", "union(-1)"} <= set(got)
    assert abs(got["union(-1)"] - ref.params["union_l"]) < 1e-10
    assert summary_of(ctx, out)["n_obs"] == len(d)


def test_panel_pooled_difference_with_a_sample(ctx, panel):
    out = tools()["panel_fit"]("wages", "d(lwage) ~ 1 + d(exper)", estimator="pooled", sample="2002 2005")
    assert summary_of(ctx, out)["n_obs"] == 60 * 4  # 2002's difference uses 2001, outside the sample


def test_undated_panel_lags_follow_the_time_index_not_the_row(ctx):
    df = pd.DataFrame({"s": [1, 1, 1, 2, 2, 2], "wave": [1, 2, 4, 1, 2, 3], "score": [10., 20, 40, 1, 2, 3]})
    info = ctx.workspace.add_dataframe(df, "waves", "t")
    ctx.workspace.set_structure(info.id, "panel_undated", id_col="s", t_col="wave")
    ctx.workspace.genr("waves", "prev", "score(-1)")
    prev = ctx.workspace.df("waves")["prev"].to_numpy()
    np.testing.assert_allclose(prev, [np.nan, 10, np.nan, np.nan, 1, 2], equal_nan=True)  # wave 3 is missing


def test_iv_with_a_lagged_instrument_sample_and_clusters(ctx, panel):
    out = tools()["iv_fit"]("wages", "lwage ~ 1 + exper + [educ ~ union + union(-1)]", cov_type="clustered",
                            cluster="id", sample="2002 2006")
    summary = summary_of(ctx, out)
    names = {c["name"] for c in summary["coefficients"]}
    assert summary["n_obs"] == 300 and {"exper", "educ"} <= names and "union(-1)" not in names
    assert "2002–2006" in summary["sample"]


def test_logit_with_time_operators(ctx, panel):
    out = tools()["ols_binary"]("wages", "high ~ educ + union(-1)", sample="2001 2007", marginal_effects=True)
    summary = summary_of(ctx, out)
    assert summary["n_obs"] == 60 * 7 and "union(-1)" in summary["marginal_effects"]


def test_clusters_must_be_present_for_a_row_to_count(ctx, panel):
    out = tools()["ols_fit"]("wages", "d(lwage) ~ d(exper)", cov_type="cluster", cluster="id")
    assert summary_of(ctx, out)["n_obs"] == 60 * 7


# ---- errors an agent can act on ------------------------------------------------------------------

def test_lags_on_a_cross_section_explain_how_to_declare_time(ctx, wages):
    with pytest.raises(ValueError, match=r"L\(\) needs time order.*cross-section.*data_set_structure"):
        tools()["ols_fit"]("wages", "lwage ~ educ(-1)")
    with pytest.raises(ValueError, match="needs time order"):
        tools()["data_genr"]("wages", "x", "d(lwage)")
    tools()["data_genr"]("wages", "ln_wage_sq", "log(exp(lwage)) ** 2")  # element-wise operators are fine


def test_unknown_variables_and_bad_syntax(ctx, levels):
    with pytest.raises(ValueError, match="Unknown variable 'gdpp'.*Columns: quarter, gdp, r"):
        tools()["ols_fit"]("levels", "dlog(gdpp) ~ d(r)")
    with pytest.raises(ValueError, match="Unknown variable 'rr'"):
        tools()["ols_fit"]("levels", "dlog(gdp) ~ d(rr)")
    with pytest.raises(ValueError, match="Unknown variable 'zzz'"):  # a plain term the engine evaluates itself
        tools()["ols_fit"]("levels", "dlog(gdp) ~ d(r) + zzz")
    with pytest.raises(ValueError, match="not a formula"):
        tools()["ols_fit"]("levels", "dlog(gdp)")
    with pytest.raises(ValueError, match=r"seas\(5\).*seasons 1 to 4"):
        tools()["ols_fit"]("levels", "gdp ~ @seas(5)")
    with pytest.raises(ValueError, match="outside the data"):
        tools()["ols_fit"]("levels", "gdp ~ trend('1980q1')")


def test_diagnostics_use_the_users_term_names(ctx, levels):
    t = tools()
    out = t["ols_fit"]("levels", "dlog(gdp) ~ dlog(gdp(-1)) + d(r)")
    diag = t["ols_diagnostics"](out["result_id"])
    assert {v["variable"] for v in diag["vif"]} == {"dlog(gdp(-1))", "d(r)"}


# ---- genr -----------------------------------------------------------------------------------------

def test_genr_persists_and_guards_overwrites(tmp_path):
    ws = Workspace(tmp_path / "ws.duckdb")
    info = ws.add_dataframe(make_levels(), "levels", "t")
    ws.set_structure(info.id, "time_series", date_col="quarter")
    ws.set_sample("levels", "2000q1 2004q4")  # genr ignores the sample: every row is computed
    out = ws.genr("levels", "g", "100 * dlog(gdp)")
    assert (out["n_rows"], out["n_missing"], out["replaced"]) == (100, 1, False)
    np.testing.assert_allclose(ws.df("levels", sample="@all")["g"], 100 * np.log(make_levels().gdp).diff(),
                               equal_nan=True)
    with pytest.raises(ValueError, match="already exists"):
        ws.genr("levels", "g", "d(gdp)")
    assert ws.genr("levels", "g", "d(gdp)", overwrite=True)["replaced"] is True
    for bad in ("1x", "__obs", "a b"):
        with pytest.raises(ValueError, match="not a valid series name"):
            ws.genr("levels", bad, "gdp")
    assert ws.describe_structure("levels")["generated_series"] == {"g": "d(gdp)"}
    del ws
    again = Workspace(tmp_path / "ws.duckdb")
    assert "g" in {c["name"] for c in again.get("levels").columns}
    assert again.generated_series("levels") == {"g": "d(gdp)"}


def test_generated_series_are_recomputed_when_the_structure_changes(ctx):
    df = make_levels(12)
    info = ctx.workspace.add_dataframe(df, "lv", "t")
    ctx.workspace.set_structure(info.id, "time_series", date_col="quarter")
    ctx.workspace.genr("lv", "g", "dlog(gdp)")
    ctx.workspace.genr("lv", "g2", "g * 2")
    ctx.workspace.set_structure(info.id, "time_series", date_col="quarter")  # rebuilt from the source rows
    np.testing.assert_allclose(ctx.workspace.df("lv")["g2"], 2 * np.log(df.gdp).diff(), equal_nan=True)
    ctx.workspace.set_structure(info.id, "cross_section")  # dlog needs time, so the series cannot survive
    assert "g" not in {c["name"] for c in ctx.workspace.get("lv").columns}
    assert ctx.workspace.generated_series("lv") == {}


def test_agent_tools_document_the_operators():
    specs = {t.name: t for t in all_tools()}
    assert "dlog(x)" in specs["data_genr"].description
    assert "x(-1) lag" in specs["ols_fit"].description
