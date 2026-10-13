"""Count, ordered, multinomial, quantile; ARDL, Markov switching, Granger and the extra unit-root tests."""

import numpy as np
import pandas as pd
import pytest

from qtrix_engine.methods import methods_catalog
from qtrix_engine.objects.specs import parse_spec
from qtrix_engine.qx.forms import call_for_spec
from qtrix_engine.tools import all_tools


def tools():
    return {t.name: t.fn for t in all_tools()}


def coefs(ctx, name: str) -> dict[str, float]:
    return {c["name"]: c["coef"] for c in ctx.objects.get(name).summary["coefficients"]}


@pytest.fixture
def micro(ctx):
    rng = np.random.default_rng(1)
    n = 800
    x, z = rng.normal(size=n), rng.normal(size=n)
    lat = 0.9 * x - 0.4 * z + rng.logistic(size=n)
    utility = np.column_stack([np.zeros(n), 1.0 * x, -1.0 * x]) + rng.gumbel(size=(n, 3))
    df = pd.DataFrame({
        "x": x, "z": z,
        # overdispersed, with excess zeros
        "visits": rng.poisson(np.exp(0.3 + 0.5 * x) * rng.gamma(2.0, 0.5, size=n)) * (rng.random(n) > 0.3),
        "rating": pd.cut(lat, [-np.inf, -1, 1, np.inf], labels=False),
        "mode": np.argmax(utility, axis=1),
        "wage": 2 + 1.5 * x + np.exp(0.4 * x) * rng.normal(size=n),  # spread grows with x
        "grp": rng.integers(0, 40, size=n),
    })
    ctx.workspace.add_dataframe(df, "micro", "synthetic")
    return ctx


@pytest.fixture
def macro(ctx):
    rng = np.random.default_rng(2)
    n = 240
    regime = np.repeat([0, 1, 0], [90, 70, 80])
    growth = np.where(regime == 1, -1.0, 1.5) + rng.normal(scale=0.5, size=n)
    x = np.cumsum(rng.normal(size=n))
    y = 0.8 * x + rng.normal(scale=0.5, size=n)
    shock = rng.normal(size=n)
    follower = np.r_[0.0, shock[:-1]] + rng.normal(scale=0.3, size=n)
    df = pd.DataFrame({"month": pd.date_range("2000-01-01", periods=n, freq="MS").strftime("%Y-%m-%d"),
                       "growth": growth, "x": x, "y": y, "shock": shock, "follower": follower})
    info = ctx.workspace.add_dataframe(df, "macro", "synthetic")
    ctx.workspace.set_structure(info.id, "time_series", date_col="month")
    return ctx


# ---- cross-section ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize("model", ["poisson", "negbin", "zip", "zinb"])
def test_count_models_recover_the_coefficient(micro, model):
    out = tools()["ols_count"]("micro", "visits ~ x + z", model=model, name=f"c_{model}")
    assert coefs(micro, f"c_{model}")["x"] == pytest.approx(0.5, abs=0.12)
    assert ("incidence_rate_ratios" in out) == (model in ("poisson", "negbin"))
    assert micro.panel.blocks()[-1].kind == "regression_table"


def test_count_cluster_and_validation(micro):
    t = tools()
    assert t["ols_count"]("micro", "visits ~ x", cov_type="cluster", cluster="grp")["cov_type"] == "cluster"
    with pytest.raises(ValueError, match="requires the 'cluster'"):
        t["ols_count"]("micro", "visits ~ x", cov_type="cluster")
    with pytest.raises(ValueError, match="non-negative integer"):
        t["ols_count"]("micro", "wage ~ x")


def test_ordered_logit_has_thresholds_and_no_constant(micro):
    tools()["ols_ordered"]("micro", "rating ~ x + z", name="ord1")
    c = coefs(micro, "ord1")
    assert "Intercept" not in c and sum(n.startswith("cut ") for n in c) == 2
    assert c["x"] > 0.5 > 0 > c["z"]
    with pytest.raises(ValueError, match="ols_binary"):
        tools()["ols_ordered"]("micro", "I(x > 0) ~ z")


def test_multinomial_logit_reports_each_category_against_the_base(micro):
    out = tools()["ols_multinomial"]("micro", "mode ~ x", name="mn1")
    c = coefs(micro, "mn1")
    assert set(c) == {"x [1]", "x [2]", "Intercept [1]", "Intercept [2]"}
    assert out["base_category"] == 0 and c["x [1]"] > 0 > c["x [2]"]


def test_quantile_regression_slope_grows_with_the_quantile(micro):
    t = tools()
    t["ols_quantile"]("micro", "wage ~ x", quantile=0.1, name="q10")
    t["ols_quantile"]("micro", "wage ~ x", quantile=0.9, name="q90")
    assert coefs(micro, "q10")["x"] < coefs(micro, "q90")["x"]
    assert micro.objects.get("q10").summary["quantile"] == 0.1
    with pytest.raises(ValueError, match="strictly between"):
        t["ols_quantile"]("micro", "wage ~ x", quantile=1.0)
    t["ols_compare"](["q10", "q90"])
    micro.objects.rebuild("q10")


def test_new_equations_round_trip_through_qx_and_rebuild(micro):
    t = tools()
    t["ols_count"]("micro", "visits ~ x", name="c1")
    t["ols_multinomial"]("micro", "mode ~ x", name="m1")
    for name, tool in (("c1", "ols_count"), ("m1", "ols_multinomial")):
        call = call_for_spec(parse_spec(micro.objects.get(name).spec), name)
        assert call.tool == tool and call.args["dataset_id"] == "micro"
        micro.objects.rebuild(name)


# ---- time series -----------------------------------------------------------------------------------------------
def test_markov_switching_finds_two_regimes(macro):
    out = tools()["ts_markov"]("macro", "growth", k_regimes=2, name="mk1")
    means = sorted(v for k, v in coefs(macro, "mk1").items() if k.startswith("const"))
    assert means[0] == pytest.approx(-1.0, abs=0.4) and means[1] == pytest.approx(1.5, abs=0.3)
    assert set(out["transition_matrix"]) == {"from 0", "from 1"}
    assert all(d > 5 for d in out["expected_duration"].values())
    assert any(b.kind == "plotly_figure" for b in macro.panel.blocks())
    macro.objects.rebuild("mk1")


def test_markov_autoregression_and_validation(macro):
    out = tools()["ts_markov"]("macro", "growth", order=1, switching_variance=False)
    assert "AR(1)" in out["model"]
    with pytest.raises(ValueError, match="at least two regimes"):
        tools()["ts_markov"]("macro", "growth", k_regimes=1)


def test_ardl_selects_lags_and_runs_the_bounds_test(macro):
    out = tools()["ts_ardl"]("macro", "y", exog=["x"], name="ard1")
    s = macro.objects.get("ard1").summary
    assert s["bounds_test"]["reject_no_cointegration_5pct"] is True  # y is cointegrated with x by construction
    assert len(s["ardl_order"]) == 2 and out["model"].startswith("ARDL(")
    assert tools()["ts_ardl"]("macro", "y", exog=["x"], lags=1, exog_lags=1)["model"] == "ARDL(1, 1)"
    macro.objects.rebuild("ard1")


def test_ardl_without_regressors_is_an_autoregression(macro):
    out = tools()["ts_ardl"]("macro", "growth", lags=2, name="ard2")
    assert out["model"] == "ARDL(2)" and macro.objects.get("ard2").summary["bounds_test"] is None


def test_granger_finds_the_lead_lag_relation(macro):
    out = tools()["ts_granger"]("macro", "follower", ["shock"], maxlag=2)
    assert out["results"]["shock"]["lag 1"]["p_value"] < 0.01
    reverse = tools()["ts_granger"]("macro", "shock", ["follower"], maxlag=2)
    assert reverse["results"]["follower"]["lag 1"]["p_value"] > 0.01
    assert macro.panel.blocks()[-1].kind == "diagnostics"


@pytest.mark.parametrize("test", ["dfgls", "vr"])
def test_extra_unit_root_tests(macro, test):
    walk = tools()["ts_unit_root"]("macro", "x", test=test)
    noise = tools()["ts_unit_root"]("macro", "shock", test=test)
    assert walk["p_value"] > 0.05 > noise["p_value"]
    assert walk["test"] == test.upper()


def test_catalog_lists_the_new_methods_as_available():
    ids = {m["id"]: m for area in methods_catalog() for m in area["methods"]}
    for key in ("count", "ordered", "multinomial", "quantile", "ardl", "markov", "granger"):
        assert ids[key]["available"], key
