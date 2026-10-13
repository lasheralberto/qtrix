"""ts_* (cointegration, VECM, correlogram, breaks, forecast) and vol_* tools on synthetic series."""

import numpy as np
import pandas as pd
import pytest

from qtrix_engine.tools import all_tools


def tools():
    return {t.name: t.fn for t in all_tools()}


@pytest.fixture
def coint(ctx):
    rng = np.random.default_rng(3)
    n = 200
    x = np.cumsum(rng.normal(size=n))
    y = 2.0 * x + rng.normal(scale=0.5, size=n)
    walk = np.cumsum(rng.normal(size=n))
    shift = np.where(np.arange(n) < 100, 0.0, 6.0) + rng.normal(scale=0.5, size=n)
    df = pd.DataFrame({"month": pd.date_range("2000-01-01", periods=n, freq="MS").strftime("%Y-%m-%d"),
                       "x": x, "y": y, "walk": walk, "shift": shift})
    info = ctx.workspace.add_dataframe(df, "coint", "synthetic")
    return ctx.workspace.set_structure(info.id, "time_series", date_col="month")


@pytest.fixture
def returns(ctx):
    rng = np.random.default_rng(3)
    n = 600
    r, h = np.zeros(n), np.full(n, 0.2)
    for i in range(1, n):
        h[i] = 0.05 + 0.15 * r[i - 1] ** 2 + 0.75 * h[i - 1]
        r[i] = np.sqrt(h[i]) * rng.normal()
    df = pd.DataFrame({"day": pd.date_range("2020-01-01", periods=n, freq="B").strftime("%Y-%m-%d"), "ret": r})
    info = ctx.workspace.add_dataframe(df, "returns", "synthetic")
    return ctx.workspace.set_structure(info.id, "time_series", date_col="day")


def test_cointegration_johansen_and_engle_granger(ctx, coint):
    t = tools()
    jo = t["ts_cointegration"]("coint", ["y", "x"], test="johansen")
    assert jo["rank_trace_5pct"] == 1 and len(jo["relations"]) == 2
    eg = t["ts_cointegration"]("coint", ["y", "x"], test="engle_granger")
    assert eg["p_value"] < 0.05
    independent = t["ts_cointegration"]("coint", ["x", "walk"], test="engle_granger")
    assert independent["p_value"] > eg["p_value"]
    assert ctx.panel.blocks()[-1].kind == "diagnostics"


def test_vecm_is_a_versioned_var_object(ctx, coint):
    t = tools()
    out = t["ts_vecm"]("coint", ["y", "x"], k_ar_diff=1, name="vecm1")
    assert out["coint_rank"] == 1 and ctx.objects.get("vecm1").type == "var"
    assert set(out["beta"]["ect1"]) == {"y", "x"}
    again = t["ts_vecm"]("coint", ["y", "x"], k_ar_diff=2, name="vecm1")
    assert again["version"] == 2
    with pytest.raises(ValueError, match="no cointegration"):
        t["ts_vecm"]("coint", ["x", "walk"], k_ar_diff=1)


def test_correlogram_and_breaks(ctx, coint):
    t = tools()
    out = t["ts_acf_pacf"]("coint", "x", lags=12, diff=1)
    assert out["series"] == "d1(x)" and out["ljung_box"]["p_value"] is not None
    assert ctx.panel.blocks()[-1].kind == "plotly_figure"
    res = t["ts_breaks"]("coint", "shift", max_breaks=1)
    assert abs(pd.Timestamp(res["break_dates"][0]) - pd.Timestamp("2008-05-01")).days < 120


def test_forecast_is_dated(ctx, coint):
    out = tools()["ts_forecast"]("coint", "x", horizon=3, model="naive")
    assert list(out["forecast"]) == ["2016-09-01", "2016-10-01", "2016-11-01"]
    assert len(out["lower_95"]) == 3


@pytest.mark.parametrize("model", ["garch", "egarch", "gjr"])
def test_garch_family(ctx, returns, model):
    out = tools()["vol_garch"]("returns", "ret", model=model, forecast_steps=3, name=f"g_{model}")
    assert len(out["volatility_forecast"]) == 3 and all(v > 0 for v in out["volatility_forecast"].values())
    assert ctx.objects.get(f"g_{model}").type == "equation"
    if model == "garch":
        assert 0.5 < out["persistence"] < 1.05
    assert ctx.panel.blocks()[-1].kind == "plotly_figure"


def test_garch_rebuilds_deterministically(ctx, returns):
    tools()["vol_garch"]("returns", "ret", name="g1")
    ctx.objects.rebuild("g1")
