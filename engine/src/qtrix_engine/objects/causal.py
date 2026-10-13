"""Estimators of the causal-inference objects (ADR 0002 §3): staggered event studies (pyfixest), regression
discontinuity (rdrobust), double/debiased machine learning (DoubleML) and synthetic control (pysyncon).

Same contract as ``estimators``: ``spec -> Estimate`` from the page data, deterministic (learners and sample splits
are seeded) so a stored object rebuilds to the same summary.
"""

from __future__ import annotations

import re
import warnings
from typing import Any

import numpy as np
import pandas as pd

from ..context import get_context
from ..tools._format import clean, figure_data
from . import specs as sp
from .estimators import PANEL_KINDS, Estimate


# ---- panels: entity, ordinal period and treatment cohort ---------------------------------------------------------
def panel_long(spec: sp._Fitted, columns: list[str], what: str) -> tuple[Any, pd.DataFrame, str, str]:
    """The page as a long table with its entity and time columns (the declared panel structure)."""
    ws = get_context().workspace
    info = ws.require(spec.page, PANEL_KINDS, what)
    frame = ws.frame(spec.page, list(dict.fromkeys(columns)), sample=spec.sample)
    keys = frame.index.to_frame(index=False)
    entity, time = (str(n) for n in keys.columns)
    body = frame.reset_index(drop=True).drop(columns=[c for c in keys.columns if c in frame.columns])
    return info, pd.concat([keys, body], axis=1), entity, time


def _as_period_values(values: pd.Series, levels: pd.Index) -> pd.Series:
    """Cohort / period values in the units of the time index (a year 2006 on a dated panel is 2006-01-01)."""
    if isinstance(levels, pd.DatetimeIndex):
        numeric = pd.to_numeric(values, errors="coerce")
        years = numeric[numeric.notna() & (numeric != 0)]  # 0 marks the never treated
        if len(years) and years.between(1000, 9999).all():
            out = pd.to_datetime(numeric.where(numeric != 0).astype("Int64").astype("string") + "-01-01",
                                 errors="coerce")
        else:
            out = pd.to_datetime(values.astype("string"), errors="coerce")
        return out.astype(levels.dtype)
    if pd.api.types.is_numeric_dtype(levels):
        return pd.to_numeric(values, errors="coerce")
    return values.astype("string")


def period_codes(times: pd.Series) -> tuple[np.ndarray, pd.Index]:
    """Ordinal period of each row (1..T) and the sorted distinct periods."""
    levels = pd.Index(sorted(times.unique()))
    return levels.get_indexer(times) + 1, levels


def cohort_codes(values: pd.Series, levels: pd.Index) -> np.ndarray:
    """Ordinal of the first treated period (the first period on or after the cohort value); 0 = never treated."""
    conv = _as_period_values(values.reset_index(drop=True), levels)
    raw = pd.to_numeric(values.reset_index(drop=True), errors="coerce")
    treated = (conv.notna() & ~(raw == 0).fillna(False)).to_numpy()
    pos = np.zeros(len(values), dtype=int)
    pos[treated] = levels.searchsorted(pd.Index(conv[treated]), side="left") + 1
    pos[pos > len(levels)] = 0
    return pos


def _x_values(levels: pd.Index) -> list[str]:
    return [t.isoformat() for t in levels] if isinstance(levels, pd.DatetimeIndex) else [str(t) for t in levels]


# ---- staggered event study (pyfixest) ----------------------------------------------------------------------------
_ES_LABEL = {"did2s": "Gardner two-stage DiD (did2s)", "cs": "Callaway-Sant'Anna (doubly robust)",
             "saturated": "Sun-Abraham saturated event study", "lpdid": "Local-projections DiD",
             "twfe": "Dynamic TWFE"}


def _normal_table(index: list[Any], est: np.ndarray, se: np.ndarray) -> pd.DataFrame:
    """A pyfixest-style coefficient table from estimates and standard errors (normal inference)."""
    from scipy import stats

    est, se = np.asarray(est, dtype=float), np.asarray(se, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        p = 2 * stats.norm.sf(np.abs(est / se))
    return pd.DataFrame({"Estimate": est, "Std. Error": se, "Pr(>|t|)": p, "2.5%": est - 1.959964 * se,
                         "97.5%": est + 1.959964 * se}, index=index)


def _callaway_santanna(d: pd.DataFrame, spec: sp.EventStudySpec) -> tuple[Any, pd.DataFrame, pd.Series]:
    """Group-time ATTs (doubly robust, universal base period t-1) aggregated by event time and overall."""
    import contextlib
    import io

    from csdid.att_gt import ATTgt

    control = ["nevertreated" if spec.control_group == "never" else "notyettreated"]
    with contextlib.redirect_stdout(io.StringIO()):  # csdid prints its summaries
        model = ATTgt(yname="y", tname="t", idname="unit_id", gname="g", data=d[["y", "t", "unit_id", "g", "cl_id"]],
                      control_group=control, clustervar="cl_id" if spec.cluster else None)
        model.fit(est_method="dr", base_period="universal", bstrap=False)
        model.aggte(typec="dynamic", bstrap=False)
        dynamic = dict(model.atte)
        model.aggte(typec="simple", bstrap=False)
        simple = dict(model.atte)
    events = [int(e) for e in np.ravel(dynamic["egt"])]
    table = _normal_table(events, np.ravel(dynamic["att_egt"]), np.ravel(dynamic["se_egt"]))
    table = table[table.index != -1]  # the universal base period: zero by construction
    att = _normal_table(["ATT"], np.ravel([simple["overall_att"]]), np.ravel(simple["overall_se"])).iloc[0]
    return model, table, att


def _event_time(name: Any) -> int | None:
    found = re.findall(r"-?\d+(?:\.\d+)?|inf", str(name))
    if not found or found[-1] == "inf" or "first_treated" in str(name):
        return None
    return int(round(float(found[-1])))


def _event_rows(table: pd.DataFrame, pre: int, post: int) -> list[dict[str, Any]]:
    rows = []
    for name, r in table.iterrows():
        k = _event_time(name)
        if k is None or not -pre <= k <= post:
            continue
        coef, se = float(r["Estimate"]), float(r["Std. Error"])
        rows.append(clean({"name": f"t{k:+d}", "k": k, "coef": coef, "se": se, "stat": coef / se if se else None,
                           "p": float(r["Pr(>|t|)"]), "ci_low": float(r["2.5%"]), "ci_high": float(r["97.5%"])}))
    return sorted(rows, key=lambda r: r["k"])


def _pretrend_test(model: Any, table: pd.DataFrame, pre: int) -> dict[str, Any] | None:
    """Joint Wald test that every pre-treatment coefficient is zero (no anticipation / parallel pre-trends)."""
    names = list(table.index)
    idx = [i for i, n in enumerate(names) if (k := _event_time(n)) is not None and -pre <= k < -1]
    if not idx or not hasattr(model, "wald_test"):
        return None
    r = np.zeros((len(idx), len(names)))
    for row, col in enumerate(idx):
        r[row, col] = 1.0
    try:
        out = model.wald_test(R=r, distribution="chi2")
        stat, p = float(out["statistic"]), float(out["pvalue"])
    except Exception:  # noqa: BLE001 - the event study stands without its pre-trend test
        return None
    return clean({"test": "Joint pre-trends (all leads = 0)", "statistic": stat, "df": len(idx), "p_value": p,
                  "reject_5pct": bool(p < 0.05)})


def _att_row(att: pd.Series) -> dict[str, Any]:
    coef, se = float(att["Estimate"]), float(att["Std. Error"])
    return clean({"name": "ATT", "coef": coef, "se": se, "stat": coef / se if se else None,
                  "p": float(att["Pr(>|t|)"]), "ci_low": float(att["2.5%"]), "ci_high": float(att["97.5%"])})


def _event_study(spec: sp.EventStudySpec) -> Estimate:
    import plotly.graph_objects as go
    import pyfixest as pf

    if spec.pre < 2 or spec.post < 0:
        raise ValueError("pre must be at least 2 (t-1 is the reference period) and post at least 0")
    cols = [spec.outcome, spec.cohort, *([spec.cluster] if spec.cluster else [])]
    info, df, entity, time = panel_long(spec, cols, "An event study")
    t, levels = period_codes(df[time])
    d = pd.DataFrame({"y": pd.to_numeric(df[spec.outcome], errors="coerce"), "unit": df[entity].astype(str),
                      "t": t, "g": cohort_codes(df[spec.cohort], levels)})
    d["cl"] = df[spec.cluster].astype(str) if spec.cluster else d["unit"]
    d = d.dropna(subset=["y"]).reset_index(drop=True)
    if (d.groupby("unit")["g"].nunique() > 1).any():
        raise ValueError(f"'{spec.cohort}' must be the same in every period of an entity (its first treated period)")
    treated_units = d.loc[d["g"] > 0, "unit"].nunique()
    if treated_units == 0:
        raise ValueError(f"No entity is ever treated according to '{spec.cohort}'")
    d["unit_id"] = pd.factorize(d["unit"])[0]
    d["cl_id"] = pd.factorize(d["cl"])[0]
    rel = (d["t"] - d["g"]).clip(-spec.pre, spec.post).astype(float)  # endpoints bin the periods outside
    d["treat"] = ((d["g"] > 0) & (d["t"] >= d["g"])).astype(int)
    vcov = {"CRV1": "cl_id"}
    att = None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if spec.estimator == "twfe":
            d["rel"] = np.where(d["g"] > 0, rel, -1.0)  # never treated: in the reference period throughout
            model = pf.feols("y ~ i(rel, ref=-1.0) | unit_id + t", d, vcov=vcov)
            table = model.tidy()
            att = pf.feols("y ~ treat | unit_id + t", d, vcov=vcov).tidy().loc["treat"]
        elif spec.estimator == "did2s":
            d["rel"] = np.where(d["g"] > 0, rel, np.inf)
            model = pf.did2s(d, yname="y", first_stage="~ 0 | unit_id + t", second_stage="~ i(rel, ref=-1.0)",
                             treatment="treat", cluster="cl_id")
            table = model.tidy()
            att = pf.did2s(d, yname="y", first_stage="~ 0 | unit_id + t", second_stage="~ treat",
                           treatment="treat", cluster="cl_id").tidy().loc["treat"]
        elif spec.estimator == "cs":
            model, table, att = _callaway_santanna(d, spec)
        elif spec.estimator == "saturated":
            model = pf.event_study(d, yname="y", idname="unit_id", tname="t", gname="g", estimator="saturated",
                                   cluster="cl_id")
            table = model.aggregate()
        else:
            model = pf.lpdid(d, yname="y", idname="unit_id", tname="t", gname="g", vcov=vcov,
                             pre_window=spec.pre, post_window=spec.post, att=False)
            table = model.tidy()
            att = pf.lpdid(d, yname="y", idname="unit_id", tname="t", gname="g", vcov=vcov,
                           pre_window=spec.pre, post_window=spec.post, att=True).tidy().iloc[0]
    rows = _event_rows(table, spec.pre, spec.post)
    if not rows:
        raise ValueError("The event study estimated no coefficient inside the event window")
    post_rows = [r for r in rows if r["k"] >= 0]
    pretrend = _pretrend_test(model, table, spec.pre) if spec.estimator in ("twfe", "did2s") else None
    coefficients = [{k: v for k, v in r.items() if k != "k"} for r in rows]
    if att is not None:
        coefficients.append(_att_row(att))
    labels = _x_values(levels)
    summary = clean({
        "model": f"Event study · {_ES_LABEL[spec.estimator]}",
        "formula": f"{spec.outcome} ~ event time ({spec.cohort})",
        "dataset": info.name, "dep_var": spec.outcome, "n_obs": int(len(d)),
        "n_entities": int(d["unit"].nunique()), "treated_entities": int(treated_units),
        "cohorts": int(d.loc[d["g"] > 0, "g"].nunique()), "cov_type": f"cluster ({spec.cluster or entity})",
        "window": [-spec.pre, spec.post], "reference_period": "t-1",
        "post_average": float(np.mean([r["coef"] for r in post_rows])) if post_rows else None,
        "pretrend_test": pretrend, "coefficients": coefficients, "sample": f"{labels[0][:10]} – {labels[-1][:10]}",
    })
    points = sorted([(-1, 0.0, 0.0, 0.0), *[(r["k"], r["coef"], r["ci_low"], r["ci_high"]) for r in rows]])
    ks, est, lo, hi = (np.array(v) for v in zip(*points, strict=True))
    fig = go.Figure(go.Scatter(x=ks, y=est, mode="markers+lines", name="Effect",
                               error_y={"type": "data", "symmetric": False, "array": hi - est,
                                        "arrayminus": est - lo}))
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.add_vline(x=-0.5, line_dash="dot", line_color="gray")
    fig.update_layout(xaxis_title="Periods relative to treatment (t-1 = reference)", yaxis_title=spec.outcome,
                      margin={"l": 40, "r": 20, "t": 30, "b": 40})
    call = {"did2s": "pf.did2s(df, yname='y', first_stage='~ 0 | unit + t', second_stage='~ i(rel, ref=-1.0)', "
                     "treatment='treat', cluster='unit')",
            "twfe": "pf.feols('y ~ i(rel, ref=-1.0) | unit + t', df, vcov={'CRV1': 'unit'})",
            "cs": "ATTgt(yname='y', tname='t', idname='unit', gname='g', data=df).fit(base_period='universal')"
                  ".aggte(typec='dynamic')  # from csdid.att_gt import ATTgt",
            "saturated": "pf.event_study(df, yname='y', idname='unit', tname='t', gname='g', estimator='saturated')",
            "lpdid": f"pf.lpdid(df, yname='y', idname='unit', tname='t', gname='g', pre_window={spec.pre}, "
                     f"post_window={spec.post}, att=False)"}[spec.estimator]
    code = (f"import pyfixest as pf\ndf = qtrix.frame({info.name!r}).reset_index()\n"
            f"# t = period number, g = first treated period (0 never), rel = t - g binned to [-{spec.pre}, "
            f"{spec.post}], treat = 1 from g on\nres = {call}\n"
            + {"saturated": "res.aggregate()", "cs": "res.atte"}.get(spec.estimator, "res.tidy()"))
    return Estimate(summary, model, code, {"figure": figure_data(fig), "title": f"Event study · {spec.outcome}"})


# ---- regression discontinuity (rdrobust) -------------------------------------------------------------------------
def _rdd(spec: sp.RddSpec) -> Estimate:
    import plotly.graph_objects as go
    from rdrobust import rdrobust

    extra = [c for c in (spec.treatment, spec.cluster) if c]
    cols = list(dict.fromkeys([spec.outcome, spec.running, *spec.covariates, *extra]))
    ws = get_context().workspace
    info = ws.get(spec.page)
    d = ws.df(spec.page, cols, sample=spec.sample).dropna()
    y, x = d[spec.outcome].astype(float), d[spec.running].astype(float)
    if not ((x < spec.cutoff).any() and (x >= spec.cutoff).any()):
        raise ValueError(f"'{spec.running}' has observations on one side of the cutoff {spec.cutoff:g} only")
    kwargs: dict[str, Any] = {"c": spec.cutoff, "p": spec.p, "kernel": spec.kernel[:3], "bwselect": spec.bwselect}
    if spec.treatment:
        kwargs["fuzzy"] = d[spec.treatment].astype(float)
    if spec.covariates:
        kwargs["covs"] = d[spec.covariates].astype(float)
    if spec.h is not None:
        kwargs["h"] = spec.h
    if spec.cluster:
        kwargs["cluster"] = d[spec.cluster]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = rdrobust(y, x, **kwargs)
    rows = []
    for label, key in (("Conventional", "Conventional"), ("Bias-corrected", "Bias-Corrected"), ("Robust", "Robust")):
        coef, se = float(r.coef.loc[key].iloc[0]), float(r.se.loc[key].iloc[0])
        rows.append(clean({"name": label, "coef": coef, "se": se, "stat": coef / se if se else None,
                           "p": float(r.pv.loc[key].iloc[0]), "ci_low": float(r.ci.loc[key].iloc[0]),
                           "ci_high": float(r.ci.loc[key].iloc[1])}))
    h_l, h_r = (float(v) for v in r.bws.loc["h"])
    design = "fuzzy" if spec.treatment else "sharp"
    jump = f"1[{spec.running} >= {spec.cutoff:g}]"
    summary = clean({
        "model": f"RDD ({design}, local polynomial p={spec.p}, {spec.kernel})",
        "formula": f"{spec.outcome} ~ {jump}" + (f" -> {spec.treatment}" if spec.treatment else ""),
        "dataset": info.name, "dep_var": spec.outcome, "n_obs": int(sum(r.N)), "cov_type": "robust bias-corrected",
        "cutoff": spec.cutoff, "bandwidth_h": {"left": h_l, "right": h_r},
        "bandwidth_b": {"left": float(r.bws.loc["b"].iloc[0]), "right": float(r.bws.loc["b"].iloc[1])},
        "n_effective": {"left": int(r.N_h[0]), "right": int(r.N_h[1])}, "bwselect": spec.bwselect,
        "report": "Robust", "coefficients": rows, "sample": ws.resolve_sample(info, spec.sample).text,
    })
    # The RD plot: binned means on each side and the local polynomial fitted within the bandwidth.
    fig = go.Figure()
    sides = (("left", x < spec.cutoff, h_l, "#2f62d6"), ("right", x >= spec.cutoff, h_r, "#d6612f"))
    for side, mask, h, color in sides:
        xs, ys = x[mask], y[mask]
        bins = pd.qcut(xs, q=min(20, max(2, xs.nunique() // 5)), duplicates="drop")
        means = pd.DataFrame({"x": xs, "y": ys}).groupby(bins, observed=True).mean()
        fig.add_scatter(x=means["x"], y=means["y"], mode="markers", name=f"Bin means ({side})",
                        marker={"color": color})
        near = (xs - spec.cutoff).abs() <= h
        if near.sum() > spec.p + 1:
            coefs = np.polyfit(xs[near] - spec.cutoff, ys[near], spec.p)
            grid = np.linspace(xs[near].min(), xs[near].max(), 50)
            fig.add_scatter(x=grid, y=np.polyval(coefs, grid - spec.cutoff), mode="lines", name=f"Fit ({side})",
                            line={"color": color})
    fig.add_vline(x=spec.cutoff, line_dash="dash", line_color="gray")
    fig.update_layout(xaxis_title=spec.running, yaxis_title=spec.outcome, margin={"l": 40, "r": 20, "t": 30, "b": 40})
    fuzzy = f", fuzzy=df[{spec.treatment!r}]" if spec.treatment else ""
    code = (f"from rdrobust import rdrobust\ndf = qtrix.load({info.name!r}).dropna()\n"
            f"res = rdrobust(df[{spec.outcome!r}], df[{spec.running!r}], c={spec.cutoff}, p={spec.p}, "
            f"kernel={spec.kernel[:3]!r}, bwselect={spec.bwselect!r}{fuzzy})\nprint(res)")
    return Estimate(summary, r, code, {"figure": figure_data(fig), "title": f"RD plot · {spec.outcome}"})


# ---- double / debiased machine learning (DoubleML) ---------------------------------------------------------------
LEARNER_LABEL = {"forest": "random forest", "lasso": "lasso", "boosting": "gradient boosting"}


def _learners(kind: str, seed: int) -> tuple[Any, Any]:
    """(regressor, classifier) of a learner family, seeded."""
    if kind == "forest":
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

        opts = {"n_estimators": 200, "min_samples_leaf": 5, "max_features": "sqrt", "random_state": seed}
        return RandomForestRegressor(**opts), RandomForestClassifier(**opts)
    if kind == "boosting":
        from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

        return HistGradientBoostingRegressor(random_state=seed), HistGradientBoostingClassifier(random_state=seed)
    from sklearn.linear_model import LassoCV, LogisticRegressionCV
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return (make_pipeline(StandardScaler(), LassoCV(cv=5, random_state=seed)),
            make_pipeline(StandardScaler(), LogisticRegressionCV(cv=5, max_iter=2000)))


def _dml(spec: sp.DmlSpec) -> Estimate:
    import doubleml as dml

    if not spec.controls:
        raise ValueError("Double ML needs the control variables that make the treatment as good as random")
    ws = get_context().workspace
    info = ws.get(spec.page)
    d = ws.df(spec.page, list(dict.fromkeys([spec.outcome, spec.treatment, *spec.controls])),
              sample=spec.sample).dropna()
    x = pd.get_dummies(d[spec.controls], drop_first=True, dtype=float)
    x.columns = [str(c) for c in x.columns]
    data = pd.concat([d[[spec.outcome, spec.treatment]].astype(float), x], axis=1)
    binary = set(data[spec.treatment].unique()) <= {0.0, 1.0}
    if spec.model == "irm" and not binary:
        raise ValueError(f"model='irm' needs a 0/1 treatment; '{spec.treatment}' is continuous: use model='plr'")
    reg, clf = _learners(spec.learner, spec.seed)
    obj = dml.DoubleMLData(data, spec.outcome, spec.treatment, list(x.columns))
    np.random.seed(spec.seed)  # DoubleML draws its sample splits from numpy's global generator
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if spec.model == "irm":
            model = dml.DoubleMLIRM(obj, ml_g=reg, ml_m=clf, n_folds=spec.n_folds)
        else:
            ml_m = clf if binary else _learners(spec.learner, spec.seed)[0]
            model = dml.DoubleMLPLR(obj, ml_l=reg, ml_m=ml_m, n_folds=spec.n_folds)
        model.fit()
    ci = model.confint()
    coef, se = float(model.coef[0]), float(model.se[0])
    row = clean({"name": spec.treatment, "coef": coef, "se": se, "stat": coef / se, "p": float(model.pval[0]),
                 "ci_low": float(ci.iloc[0, 0]), "ci_high": float(ci.iloc[0, 1])})
    try:
        losses = {k: float(np.ravel(v)[0]) for k, v in model.evaluate_learners().items()}
    except Exception:  # noqa: BLE001 - the nuisance losses are informative only
        losses = {}
    label = {"plr": "partially linear (PLR)", "irm": "interactive (IRM, ATE)"}[spec.model]
    summary = clean({
        "model": f"Double ML · {label} · {LEARNER_LABEL[spec.learner]}",
        "formula": f"{spec.outcome} ~ {spec.treatment} | {' + '.join(spec.controls)}", "dataset": info.name,
        "dep_var": spec.outcome, "n_obs": int(len(data)), "cov_type": f"cross-fitted, {spec.n_folds} folds",
        "n_folds": spec.n_folds, "seed": spec.seed, "nuisance_rmse": losses, "coefficients": [row],
        "sample": ws.resolve_sample(info, spec.sample).text,
    })
    cls = "DoubleMLIRM" if spec.model == "irm" else "DoubleMLPLR"
    code = (f"import doubleml as dml\ndf = qtrix.load({info.name!r}).dropna()\n"
            f"data = dml.DoubleMLData(df, {spec.outcome!r}, {spec.treatment!r}, {spec.controls!r})\n"
            f"# {LEARNER_LABEL[spec.learner]} learners for the nuisance functions; np.random.seed({spec.seed})\n"
            f"res = dml.{cls}(data, <learners>, n_folds={spec.n_folds}).fit()\nprint(res.summary)")
    return Estimate(summary, model, code)


# ---- heterogeneous effects: causal forest (EconML) ---------------------------------------------------------------
def _cate(spec: sp.CateSpec) -> Estimate:
    import plotly.graph_objects as go
    from econml.dml import CausalForestDML

    if not spec.features:
        raise ValueError("A causal forest needs the features along which the effect may vary")
    ws = get_context().workspace
    info = ws.get(spec.page)
    cols = list(dict.fromkeys([spec.outcome, spec.treatment, *spec.features, *spec.controls]))
    d = ws.df(spec.page, cols, sample=spec.sample).dropna()
    x = pd.get_dummies(d[spec.features], drop_first=True, dtype=float)
    x.columns = [str(c) for c in x.columns]
    w = pd.get_dummies(d[spec.controls], drop_first=True, dtype=float) if spec.controls else None
    y, t = d[spec.outcome].astype(float), d[spec.treatment].astype(float)
    binary = set(t.unique()) <= {0.0, 1.0}
    reg, clf = _learners("forest", spec.seed)
    model_t = clf if binary else _learners("forest", spec.seed)[0]
    n_trees = max(4, spec.n_estimators - spec.n_estimators % 4)  # a multiple of the subforest size
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        forest = CausalForestDML(model_y=reg, model_t=model_t, discrete_treatment=binary, n_estimators=n_trees,
                                 random_state=spec.seed, n_jobs=1)
        forest.fit(y.to_numpy(), t.to_numpy(), X=x.to_numpy(), W=None if w is None else w.to_numpy())
        inf = forest.ate_inference(x.to_numpy())
        cate = np.ravel(forest.effect(x.to_numpy()))
    ate, se = float(np.ravel(inf.mean_point)[0]), float(np.ravel(inf.stderr_mean)[0])
    lo, hi = (float(np.ravel(v)[0]) for v in inf.conf_int_mean())
    row = clean({"name": f"ATE ({spec.treatment})", "coef": ate, "se": se, "stat": ate / se if se else None,
                 "p": float(np.ravel(inf.pvalue())[0]), "ci_low": lo, "ci_high": hi})
    importance = dict(sorted(zip(x.columns, map(float, forest.feature_importances_), strict=True),
                             key=lambda kv: -kv[1]))
    top = next(iter(importance))
    terciles = pd.qcut(x[top].rank(method="first"), 3, labels=["low", "middle", "high"])
    by_top = pd.Series(cate, index=x.index).groupby(terciles, observed=True).mean()
    by_top = {str(k): float(v) for k, v in by_top.items()}
    quantiles = {f"p{int(q * 100)}": float(np.quantile(cate, q)) for q in (0.1, 0.25, 0.5, 0.75, 0.9)}
    summary = clean({
        "model": "Causal forest (DML)", "dataset": info.name, "dep_var": spec.outcome,
        "formula": f"{spec.outcome} ~ {spec.treatment} × f({', '.join(spec.features)})"
                   + (f" | {' + '.join(spec.controls)}" if spec.controls else ""),
        "n_obs": int(len(d)), "cov_type": "honest forest, cross-fitted", "n_estimators": n_trees, "seed": spec.seed,
        "coefficients": [row], "cate_quantiles": quantiles, "feature_importance": importance,
        "cate_by_top_feature": {"feature": top, "terciles": by_top},
        "share_positive": float(np.mean(cate > 0)), "sample": ws.resolve_sample(info, spec.sample).text,
    })
    fig = go.Figure(go.Histogram(x=cate, nbinsx=40, name="CATE"))
    fig.add_vline(x=ate, line_dash="dash", line_color="gray", annotation_text="ATE")
    fig.update_layout(xaxis_title=f"Effect of {spec.treatment} on {spec.outcome}", yaxis_title="Observations",
                      margin={"l": 40, "r": 20, "t": 30, "b": 40})
    code = (f"from econml.dml import CausalForestDML\ndf = qtrix.load({info.name!r}).dropna()\n"
            f"cf = CausalForestDML(discrete_treatment={binary}, n_estimators={n_trees}, random_state={spec.seed})\n"
            f"cf.fit(df[{spec.outcome!r}], df[{spec.treatment!r}], X=df[{spec.features!r}]"
            + (f", W=df[{spec.controls!r}]" if spec.controls else "") + ")\ncf.ate_inference(X).summary()")
    title = f"Effect heterogeneity · {spec.outcome}"
    return Estimate(summary, forest, code, {"figure": figure_data(fig), "title": title})


# ---- post-double-selection lasso (Belloni-Chernozhukov-Hansen) ----------------------------------------------------
def _pds(spec: sp.PdsSpec) -> Estimate:
    import statsmodels.api as sm
    from sklearn.linear_model import LassoCV
    from sklearn.preprocessing import StandardScaler

    from .estimators import model_summary

    if spec.cov_type == "cluster" and not spec.cluster:
        raise ValueError("cov_type='cluster' requires the 'cluster' column")
    ws = get_context().workspace
    info = ws.get(spec.page)
    extra = [spec.cluster] if spec.cluster else []
    cols = list(dict.fromkeys([spec.outcome, spec.treatment, *spec.controls, *extra]))
    d = ws.df(spec.page, cols, sample=spec.sample).dropna().reset_index(drop=True)
    x = pd.get_dummies(d[spec.controls], drop_first=True, dtype=float)
    x.columns = [str(c) for c in x.columns]
    x = x.loc[:, x.std() > 0]
    if x.shape[1] == 0:
        raise ValueError("None of the controls varies in the sample")
    y, t = d[spec.outcome].astype(float), d[spec.treatment].astype(float)
    xs = StandardScaler().fit_transform(x)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for_y = x.columns[LassoCV(cv=5, random_state=spec.seed).fit(xs, y).coef_ != 0]
        for_t = x.columns[LassoCV(cv=5, random_state=spec.seed).fit(xs, t).coef_ != 0]
    kept = [c for c in x.columns if c in set(for_y) | set(for_t)]
    exog = sm.add_constant(pd.concat([t.rename(spec.treatment), x[kept]], axis=1), has_constant="add")
    kwargs: dict[str, Any] = {"cov_type": spec.cov_type}
    if spec.cov_type == "cluster":
        kwargs["cov_kwds"] = {"groups": d[spec.cluster].to_numpy()}
    res = sm.OLS(y, exog).fit(**kwargs)
    formula = f"{spec.outcome} ~ {spec.treatment}" + (f" + {' + '.join(kept)}" if kept else "")
    summary = model_summary(res, model="Post-double-selection lasso", formula=formula, dataset=info.name,
                            cov_type=spec.cov_type)
    summary.update(clean({
        "selected_controls": kept, "n_candidate_controls": int(x.shape[1]),
        "selected_by": {"outcome": list(for_y), "treatment": list(for_t)},
        "sample": ws.resolve_sample(info, spec.sample).text,
    }))
    code = ("import statsmodels.api as sm\nfrom sklearn.linear_model import LassoCV\n"
            f"df = qtrix.load({info.name!r}).dropna()\n"
            "# keep the controls a cross-validated lasso selects for the outcome OR for the treatment\n"
            f"res = sm.OLS(df[{spec.outcome!r}], sm.add_constant(df[[{spec.treatment!r}, *{kept!r}]]))"
            f".fit(cov_type={spec.cov_type!r})\nprint(res.summary())")
    return Estimate(summary, res, code)


# ---- sensitivity to omitted variables (Cinelli & Hazlett 2020) -----------------------------------------------------
def _rv(f: float) -> float:
    return 0.5 * (np.sqrt(f ** 4 + 4 * f ** 2) - f ** 2)


def sensitivity(res: Any, treatment: str, benchmarks: list[str], kd: list[float], q: float = 1.0,
                alpha: float = 0.05) -> dict[str, Any]:
    """Robustness values and benchmark bounds of an OLS coefficient (the formulas of R/Python ``sensemakr``).

    How strong would an omitted confounder have to be, in partial R² with the treatment and with the outcome, to
    reduce the estimate by 100·q% (RV_q) or to make it insignificant at ``alpha`` (RV_q,α); and what the estimate
    would be with a confounder ``kd`` times as strong as each observed benchmark covariate.
    """
    import statsmodels.api as sm
    from scipy import stats

    fit = res.model.fit()  # classical OLS/WLS standard errors, as the derivation assumes
    names = list(fit.model.exog_names)
    if treatment not in names:
        raise ValueError(f"'{treatment}' is not a term of the model; terms: {', '.join(names)}")
    est, se, t = float(fit.params[treatment]), float(fit.bse[treatment]), float(fit.tvalues[treatment])
    dof = float(fit.df_resid)
    f = abs(t) / np.sqrt(dof)
    fq = q * f
    t_crit = float(stats.t.ppf(1 - alpha / 2, dof - 1))
    f_crit = t_crit / np.sqrt(dof - 1)
    fqa = fq - f_crit
    rv_qa = 0.0 if fqa <= 0 else _rv(fqa)
    if fqa > 0 and rv_qa > 1 - 1 / fq ** 2:
        rv_qa = (fq ** 2 - f_crit ** 2) / (1 + fq ** 2)
    out: dict[str, Any] = {
        "treatment": treatment, "estimate": est, "se": se, "t": t, "dof": dof, "q": q, "alpha": alpha,
        "partial_r2_treatment": t ** 2 / (t ** 2 + dof), "rv_q": _rv(fq), "rv_qa": rv_qa, "bounds": [],
        "bounds_undefined": [],
    }
    exog = pd.DataFrame(fit.model.exog, columns=names)
    others = exog.drop(columns=[treatment])
    aux = sm.OLS(exog[treatment], others).fit()
    for b in benchmarks:
        if b not in others.columns:
            raise ValueError(f"Benchmark '{b}' is not a covariate of the model; terms: {', '.join(others.columns)}")
        r2yxj = float(fit.tvalues[b]) ** 2 / (float(fit.tvalues[b]) ** 2 + dof)
        r2dxj = float(aux.tvalues[b]) ** 2 / (float(aux.tvalues[b]) ** 2 + float(aux.df_resid))
        for k in kd:
            r2dz = k * r2dxj / (1 - r2dxj)
            if r2dz >= 1:  # a confounder that strong would explain all the treatment's residual variance
                out["bounds_undefined"].append({"benchmark": b, "k": k})
                continue
            r2zxj = k * r2dxj ** 2 / ((1 - k * r2dxj) * (1 - r2dxj))
            r2yz = min(1.0, ((np.sqrt(k) + np.sqrt(r2zxj)) / np.sqrt(1 - r2zxj)) ** 2 * r2yxj / (1 - r2yxj))
            bias = np.sqrt(r2yz * r2dz / (1 - r2dz)) * se * np.sqrt(dof)
            adj = float(np.sign(est) * (abs(est) - bias))
            adj_se = float(se * np.sqrt((1 - r2yz) / (1 - r2dz) * dof / (dof - 1)))
            out["bounds"].append({"benchmark": b, "k": k, "r2dz_x": r2dz, "r2yz_dx": r2yz, "adjusted_estimate": adj,
                                  "adjusted_se": adj_se, "adjusted_t": adj / adj_se if adj_se else None,
                                  "ci_low": adj - t_crit * adj_se, "ci_high": adj + t_crit * adj_se})
    return clean(out)


# ---- synthetic control (pysyncon) --------------------------------------------------------------------------------
def _fit_synth(long: pd.DataFrame, outcome: str, treated: str, donors: list[str], pre: list[int],
               predictors: list[str]) -> pd.Series:
    """Donor weights of the synthetic control of ``treated``: it matches the predictors' pre-period means and the
    outcome in (up to ten) pre-treatment periods."""
    from pysyncon import Dataprep, Synth

    picks = sorted({pre[int(i)] for i in np.linspace(0, len(pre) - 1, min(10, len(pre)))})
    dp = Dataprep(foo=long, predictors=predictors or [outcome], predictors_op="mean", dependent=outcome,
                  unit_variable="unit", time_variable="period", treatment_identifier=treated,
                  controls_identifier=donors, time_predictors_prior=pre, time_optimize_ssr=pre,
                  special_predictors=[(outcome, [p], "mean") for p in picks])
    synth = Synth()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        synth.fit(dataprep=dp)
    return synth.weights(round=6).reindex(donors).fillna(0.0)


def _gaps(wide: pd.DataFrame, treated: str, weights: pd.Series) -> pd.Series:
    return wide[treated] - wide[list(weights.index)].to_numpy() @ weights.to_numpy()


def _rmspe(gap: pd.Series) -> float:
    return float(np.sqrt(np.mean(np.square(gap.to_numpy())))) if len(gap) else float("nan")


def _synth(spec: sp.SynthSpec) -> Estimate:
    import plotly.graph_objects as go

    info, df, entity, time = panel_long(spec, [spec.outcome, *spec.predictors], "A synthetic control")
    periods, levels = period_codes(df[time])
    long = pd.DataFrame({"unit": df[entity].astype(str), "period": periods,
                         spec.outcome: pd.to_numeric(df[spec.outcome], errors="coerce")})
    for p in spec.predictors:
        long[p] = pd.to_numeric(df[p], errors="coerce")
    start = _as_period_values(pd.Series([spec.treatment_period]), levels).iloc[0]
    if pd.isna(start):
        raise ValueError(f"treatment_period '{spec.treatment_period}' is not a period of '{info.name}'")
    t0 = int(levels.searchsorted(start, side="left")) + 1
    pre = list(range(1, t0))
    if len(pre) < 2 or t0 > len(levels):
        raise ValueError("A synthetic control needs at least two periods before and one after the treatment")
    wide = long.pivot_table(index="period", columns="unit", values=spec.outcome)
    if spec.treated not in wide.columns:
        raise ValueError(f"'{spec.treated}' is not an entity of '{info.name}'")
    candidates = spec.donors or [u for u in wide.columns if u != spec.treated]
    donors = [u for u in candidates if u in wide.columns and u != spec.treated and wide[u].notna().all()]
    if len(donors) < 2:
        raise ValueError("A synthetic control needs at least two donor entities observed in every period")
    if wide.loc[pre, spec.treated].isna().any():
        raise ValueError(f"'{spec.treated}' has missing values before the treatment")
    weights = _fit_synth(long, spec.outcome, spec.treated, donors, pre, spec.predictors)
    gap = _gaps(wide, spec.treated, weights)
    post = gap.index >= t0
    pre_rmspe, post_rmspe = _rmspe(gap[~post]), _rmspe(gap[post].dropna())
    att = float(gap[post].mean())
    ratio = post_rmspe / max(pre_rmspe, 1e-12)
    placebo: dict[str, Any] | None = None
    if spec.placebo:  # in-space placebos: each donor in turn treated as if it were the treated entity
        ratios = []
        for j in donors:
            g = _gaps(wide, j, _fit_synth(long, spec.outcome, j, [u for u in donors if u != j], pre, spec.predictors))
            ratios.append(_rmspe(g[g.index >= t0]) / max(_rmspe(g[g.index < t0]), 1e-12))
        rank = 1 + sum(r >= ratio for r in ratios)
        placebo = {"p_value": rank / (len(ratios) + 1), "rank": rank, "n_placebos": len(ratios),
                   "post_pre_rmspe_ratio": ratio}
    labels = _x_values(levels)
    row = clean({"name": "ATT (post average gap)", "coef": att, "se": None, "stat": None,
                 "p": placebo["p_value"] if placebo else None, "ci_low": None, "ci_high": None})
    summary = clean({
        "model": "Synthetic control", "formula": f"{spec.outcome}: {spec.treated} vs synthetic ({len(donors)} donors)",
        "dataset": info.name, "dep_var": spec.outcome, "n_obs": int(len(long)), "cov_type": "placebo (in-space)",
        "treated": spec.treated, "treatment_period": labels[t0 - 1][:10],
        "weights": {u: float(w) for u, w in weights.sort_values(ascending=False).items() if w >= 0.001},
        "att": att, "pre_rmspe": pre_rmspe, "post_rmspe": post_rmspe, "placebo": placebo,
        "gaps": {labels[int(i) - 1][:10]: float(v) for i, v in gap[post].items()},
        "coefficients": [row], "sample": f"{labels[0][:10]} – {labels[-1][:10]}",
    })
    xs = [labels[int(i) - 1] for i in wide.index]
    fig = go.Figure()
    fig.add_scatter(x=xs, y=wide[spec.treated].to_numpy(), mode="lines", name=spec.treated)
    fig.add_scatter(x=xs, y=(wide[spec.treated] - gap).to_numpy(), mode="lines", name=f"Synthetic {spec.treated}",
                    line={"dash": "dash"})
    fig.add_vline(x=labels[t0 - 1], line_dash="dot", line_color="gray")
    fig.update_layout(yaxis_title=spec.outcome, margin={"l": 40, "r": 20, "t": 30, "b": 40})
    code = (f"from pysyncon import Dataprep, Synth\ndf = qtrix.frame({info.name!r}).reset_index()\n"
            f"dp = Dataprep(foo=df, predictors={spec.predictors or [spec.outcome]!r}, predictors_op='mean', "
            f"dependent={spec.outcome!r}, unit_variable={entity!r}, time_variable={time!r}, "
            f"treatment_identifier={spec.treated!r}, controls_identifier={donors!r}, ...)\n"
            "synth = Synth(); synth.fit(dataprep=dp); synth.path_plot()")
    return Estimate(summary, None, code, {"figure": figure_data(fig), "title": f"Synthetic control · {spec.treated}"})


ESTIMATORS = {"event_study": _event_study, "rdd": _rdd, "dml": _dml, "cate": _cate, "pds": _pds, "synth": _synth}
