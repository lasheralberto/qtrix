"""Staggered event studies (pyfixest), RDD (rdrobust), double ML (DoubleML) and synthetic control (pysyncon)."""

import numpy as np
import pandas as pd
import pytest

from qtrix_engine.methods import methods_catalog
from qtrix_engine.objects.specs import parse_spec
from qtrix_engine.qx.forms import call_for_spec
from qtrix_engine.tools import all_tools

TRUE_EFFECT = {0: 2.0, 1: 2.5, 2: 3.0, 3: 3.5}


def tools():
    return {t.name: t.fn for t in all_tools()}


def coefs(ctx, name: str) -> dict[str, float]:
    return {c["name"]: c["coef"] for c in ctx.objects.get(name).summary["coefficients"]}


@pytest.fixture
def staggered(ctx):
    """90 firms, 2001-2012; cohorts adopt in 2006 and 2009 (a third never); dynamic effects 2 + 0.5 e."""
    rng = np.random.default_rng(0)
    rows = []
    for i in range(90):
        g = 0 if i < 30 else (2006 if i < 60 else 2009)
        a = rng.normal()
        for year in range(2001, 2013):
            e = year - g
            eff = TRUE_EFFECT.get(e, 4.0) if g and e >= 0 else 0.0
            rows.append({"firm": f"f{i}", "year": year, "first_year": g, "never": 0,
                         "sales": a + 0.1 * (year - 2000) + eff + rng.normal()})
    info = ctx.workspace.add_dataframe(pd.DataFrame(rows), "firms", "synthetic")
    ctx.workspace.set_structure(info.id, "panel", date_col="year", id_col="firm")
    return ctx


@pytest.fixture
def micro(ctx):
    rng = np.random.default_rng(4)
    n = 2000
    score = rng.uniform(-1, 1, n)
    x = rng.normal(size=(n, 3))
    dose = x[:, 0] + rng.normal(size=n)
    aid = np.where(score >= 0, rng.random(n) < 0.8, rng.random(n) < 0.2).astype(int)
    df = pd.DataFrame({
        "score": score, "aid": aid, "x0": x[:, 0], "x1": x[:, 1], "x2": x[:, 2], "dose": dose,
        "grade": 0.5 * score + 1.0 * (score >= 0) + rng.normal(scale=0.3, size=n),
        "fuzzy_grade": 0.5 * score + 1.5 * aid + rng.normal(scale=0.3, size=n),
        "y": 0.7 * dose + np.sin(x[:, 0]) + x[:, 1] + rng.normal(size=n),
        "region": rng.choice(["n", "s", "e"], size=n),
    })
    df["coin"] = rng.integers(0, 2, n)  # a randomised treatment whose effect grows with x0
    df["het_y"] = (1 + df["x0"]) * df["coin"] + df["x1"] + rng.normal(scale=0.5, size=n)
    controls = rng.normal(size=(n, 30))  # many candidate controls, two of them confounders
    for j in range(30):
        df[f"c{j}"] = controls[:, j]
    df["policy"] = 0.8 * controls[:, 0] + rng.normal(size=n)
    df["pds_y"] = 0.7 * df["policy"] + 1.0 * controls[:, 0] + 0.5 * controls[:, 1] + rng.normal(size=n)
    female = rng.integers(0, 2, n)
    index = -0.5 + 1.0 * x[:, 0] + 0.8 * female + 0.7 * x[:, 0] * female
    df["female"] = np.where(female == 1, "yes", "no")
    df["hired"] = (rng.random(n) < 1 / (1 + np.exp(-index))).astype(int)
    ctx.workspace.add_dataframe(df, "students", "synthetic")
    return ctx


@pytest.fixture
def regions(ctx):
    """12 regions with a two-factor structure; r0 is a mix of r1-r3 and loses 2 units from 2003."""
    rng = np.random.default_rng(5)
    years = np.arange(1990, 2011)
    f1, f2 = np.cumsum(rng.normal(size=len(years))), np.linspace(0, 3, len(years))
    load = {f"r{u}": (rng.uniform(0.2, 1.5), rng.uniform(-1, 1)) for u in range(1, 12)}
    load["r0"] = tuple(np.mean([load["r1"], load["r2"], load["r3"]], axis=0))
    rows = []
    for u, (l1, l2) in load.items():
        for k, year in enumerate(years):
            effect = -2.0 if u == "r0" and year >= 2003 else 0.0
            rows.append({"region": u, "year": year,
                         "gdp": 10 + l1 * f1[k] + l2 * f2[k] + effect + rng.normal(scale=0.05)})
    info = ctx.workspace.add_dataframe(pd.DataFrame(rows), "regions", "synthetic")
    ctx.workspace.set_structure(info.id, "panel", date_col="year", id_col="region")
    return ctx


# ---- event study -------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("estimator", ["did2s", "cs", "saturated", "lpdid"])
def test_heterogeneity_robust_event_studies_recover_the_dynamics(staggered, estimator):
    out = tools()["panel_event_study"]("firms", "sales", "first_year", estimator=estimator, pre=3, post=3,
                                       name=f"es_{estimator}")
    rows = {r["name"]: r for r in staggered.objects.get(f"es_{estimator}").summary["coefficients"]}
    for e, effect in TRUE_EFFECT.items():  # within three standard errors of the true dynamic effect
        assert abs(rows[f"t{e:+d}"]["coef"] - effect) < 3 * rows[f"t{e:+d}"]["se"], (estimator, e)
    for e in (-3, -2):
        assert abs(rows[f"t{e:+d}"]["coef"]) < 3 * rows[f"t{e:+d}"]["se"], (estimator, e)
    assert "t-1" not in rows
    assert out["treated_entities"] == 60 and out["cohorts"] == 2
    assert staggered.panel.blocks()[-1].kind == "plotly_figure"


def test_did2s_reports_att_and_pretrend_test(staggered):
    out = tools()["panel_event_study"]("firms", "sales", "first_year", name="es1")
    assert out["pretrend_test"]["p_value"] > 0.05 and out["pretrend_test"]["df"] == 3
    assert 2.0 < coefs(staggered, "es1")["ATT"] < 4.0
    call = call_for_spec(parse_spec(staggered.objects.get("es1").spec), "es1")
    assert call.tool == "panel_event_study" and call.args["cohort"] == "first_year"
    staggered.objects.rebuild("es1")


def test_callaway_santanna_with_not_yet_treated_controls(staggered):
    out = tools()["panel_event_study"]("firms", "sales", "first_year", estimator="cs", control_group="notyet",
                                       name="es_cs")
    c = coefs(staggered, "es_cs")
    assert 2.0 < c["ATT"] < 4.5 and "t-1" not in c
    assert "Callaway" in out["model"]
    staggered.objects.rebuild("es_cs")


def test_event_study_validation(staggered):
    t = tools()
    with pytest.raises(ValueError, match="pre must be at least 2"):
        t["panel_event_study"]("firms", "sales", "first_year", pre=1)
    with pytest.raises(ValueError, match="No entity is ever treated"):
        t["panel_event_study"]("firms", "sales", "never")


# ---- regression discontinuity ------------------------------------------------------------------------------------
def test_sharp_rdd_finds_the_jump(micro):
    out = tools()["causal_rdd"]("students", "grade", "score", cutoff=0.0, name="rd1")
    c = coefs(micro, "rd1")
    assert set(c) == {"Conventional", "Bias-corrected", "Robust"}
    assert c["Robust"] == pytest.approx(1.0, abs=0.2)
    assert out["bandwidth_h"]["left"] > 0 and out["n_effective"]["left"] > 50
    assert micro.panel.blocks()[-1].kind == "plotly_figure"
    narrow = tools()["causal_rdd"]("students", "grade", "score", h=0.15)
    assert narrow["bandwidth_h"]["left"] == pytest.approx(0.15)
    micro.objects.rebuild("rd1")


def test_fuzzy_rdd_scales_by_the_first_stage(micro):
    tools()["causal_rdd"]("students", "fuzzy_grade", "score", treatment="aid", name="rd_fuzzy")
    assert coefs(micro, "rd_fuzzy")["Robust"] == pytest.approx(1.5, abs=0.5)
    assert "fuzzy" in micro.objects.get("rd_fuzzy").summary["model"]


def test_rd_density_detects_bunching(micro, ctx):
    clean_test = tools()["causal_rd_density"]("students", "score", cutoff=0.0)
    assert clean_test["p_value"] > 0.05 and micro.panel.blocks()[-1].kind == "diagnostics"
    rng = np.random.default_rng(9)
    bunched = np.r_[rng.uniform(-1, 1, 1500), rng.uniform(0, 0.1, 400)]  # sorting just above the cutoff
    ctx.workspace.add_dataframe(pd.DataFrame({"score": bunched}), "bunched", "synthetic")
    assert tools()["causal_rd_density"]("bunched", "score")["p_value"] < 0.01


def test_rdd_needs_both_sides(micro):
    with pytest.raises(ValueError, match="one side of the cutoff"):
        tools()["causal_rdd"]("students", "grade", "score", cutoff=5.0)


# ---- double ML ---------------------------------------------------------------------------------------------------
def test_dml_plr_recovers_the_effect_and_is_reproducible(micro):
    out = tools()["causal_dml"]("students", "y", "dose", ["x0", "x1", "x2", "region"], name="dml1")
    assert coefs(micro, "dml1")["dose"] == pytest.approx(0.7, abs=0.1)
    assert out["nuisance_rmse"]
    micro.objects.rebuild("dml1")  # seeded: same estimate on rebuild


def test_dml_irm_needs_a_binary_treatment(micro):
    with pytest.raises(ValueError, match="0/1 treatment"):
        tools()["causal_dml"]("students", "y", "dose", ["x0"], model="irm")
    out = tools()["causal_dml"]("students", "grade", "aid", ["score"], model="irm", learner="lasso")
    assert out["coefficients"]["aid"]


def test_causal_forest_finds_the_effect_modifier(micro):
    out = tools()["causal_forest"]("students", "het_y", "coin", ["x0", "x2"], controls=["x1"], n_estimators=200,
                                   name="cf1")
    assert coefs(micro, "cf1")["ATE (coin)"] == pytest.approx(1.0, abs=0.2)
    assert next(iter(out["feature_importance"])) == "x0"
    terciles = out["cate_by_top_feature"]["terciles"]
    assert terciles["low"] < terciles["middle"] < terciles["high"]
    assert micro.panel.blocks()[-1].kind == "plotly_figure"
    micro.objects.rebuild("cf1")


def test_post_double_selection_keeps_the_confounder(micro):
    out = tools()["causal_pds"]("students", "pds_y", "policy", [f"c{j}" for j in range(30)], name="pds1")
    assert coefs(micro, "pds1")["policy"] == pytest.approx(0.7, abs=0.1)
    assert {"c0", "c1"} <= set(out["selected_controls"]) and out["n_candidate_controls"] == 30
    assert len(out["selected_controls"]) < 30
    micro.objects.rebuild("pds1")


def test_sensitivity_robustness_values_and_bounds(micro):
    t = tools()
    t["ols_fit"]("students", "y ~ dose + x0 + x1", name="eq_y")
    out = t["causal_sensitivity"]("eq_y", "dose", benchmarks=["x1"])
    f = abs(out["t"]) / np.sqrt(out["dof"])
    assert out["rv_q"] == pytest.approx(0.5 * (np.sqrt(f ** 4 + 4 * f ** 2) - f ** 2))
    assert 0 < out["rv_qa"] < out["rv_q"] < 1
    assert [b["k"] for b in out["bounds"]] == [1.0, 2.0, 3.0]
    adjusted = [b["adjusted_estimate"] for b in out["bounds"]]
    assert adjusted[0] < out["estimate"] and adjusted == sorted(adjusted, reverse=True)
    assert micro.panel.blocks()[-1].kind == "markdown"
    strong = t["causal_sensitivity"]("eq_y", "dose", benchmarks=["x0"])  # x0 explains half of dose
    assert {b["k"] for b in strong["bounds_undefined"]} == {1.0, 2.0, 3.0}
    with pytest.raises(Exception, match="not a term"):
        t["causal_sensitivity"]("eq_y", "nothing")


def test_margins_of_a_logit_with_an_interaction(micro):
    t = tools()
    t["ols_binary"]("students", "hired ~ x0 * C(female)", marginal_effects=False, name="eq_hired")
    slopes = t["ols_margins"]("eq_hired", by="female")["margins"]
    x0 = {r["female"]: r["estimate"] for r in slopes if r["term"] == "x0"}
    assert 0 < x0["no"] < x0["yes"]  # the interaction makes x0 matter more for women
    preds = t["ols_margins"]("eq_hired", kind="predictions", by="female")["margins"]
    assert all(0 < r["estimate"] < 1 for r in preds) and len(preds) == 2
    assert micro.panel.blocks()[-1].kind == "dataframe"


# ---- synthetic control -------------------------------------------------------------------------------------------
def test_synthetic_control_recovers_the_gap_and_weights(regions):
    out = tools()["causal_synth"]("regions", "gdp", "r0", "2003", name="sc1")
    assert out["att"] == pytest.approx(-2.0, abs=0.4)
    assert out["pre_rmspe"] < 0.5
    assert sum(out["weights"].values()) == pytest.approx(1.0, abs=0.01)
    assert out["placebo"]["rank"] == 1 and out["placebo"]["p_value"] == pytest.approx(1 / 12)
    assert list(out["gaps"])[0] == "2003-01-01"
    assert regions.panel.blocks()[-1].kind == "plotly_figure"


def test_synthetic_control_validation(regions):
    t = tools()
    with pytest.raises(ValueError, match="not an entity"):
        t["causal_synth"]("regions", "gdp", "nowhere", "2003", placebo=False)
    with pytest.raises(ValueError, match="two periods before"):
        t["causal_synth"]("regions", "gdp", "r0", "1990", placebo=False)


def test_catalog_lists_the_causal_methods():
    ids = {m["id"]: m for area in methods_catalog() for m in area["methods"]}
    for key in ("event_study", "rdd", "dml", "synth", "cate", "pds", "sensitivity", "margins"):
        assert ids[key]["available"], key
    assert ids["event_study"]["tools"] == ["panel_event_study"]
