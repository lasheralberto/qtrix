"""Pure estimation per method (ADR 0002 §3): ``estimate(spec) -> Estimate`` re-runs a spec on the page data.

These are the bodies of the former ``ols_*``/``panel_*``/``iv_*``/``ts_*`` tools. They read the page through
the workspace of the current context and touch no store, so a stored spec can always be re-estimated
(``ObjectStore.rebuild``) and must reproduce the stored summary while the page's ``data_version`` is unchanged.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from .. import expr
from ..context import get_context
from ..tools._format import clean, coef_rows, figure_data
from ..tools._model import prepare
from . import specs as sp

PANEL_KINDS = ("panel", "panel_undated")
TS_KINDS = ("time_series",)


@dataclass
class Estimate:
    summary: dict[str, Any]
    fitted: Any = None  # library result object (rebuilt on demand, never persisted)
    code: str = ""  # equivalent Python shown to the user
    extras: dict[str, Any] = field(default_factory=dict)  # transient by-products, e.g. a Plotly figure


# ---- summaries ----------------------------------------------------------------------------------------------
def model_summary(res: Any, *, model: str, formula: str, dataset: str, cov_type: str,
                  rename: Callable[[str], str] | None = None) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "model": model, "formula": formula, "dataset": dataset, "dep_var": formula.split("~")[0].strip(),
        "n_obs": int(res.nobs), "cov_type": cov_type,
        "coefficients": coef_rows(res.params, res.bse, res.tvalues, res.pvalues, res.conf_int(), rename),
    }
    for attr, key in (("rsquared", "r2"), ("rsquared_adj", "adj_r2"), ("prsquared", "pseudo_r2"),
                      ("fvalue", "f_stat"), ("f_pvalue", "f_pvalue"), ("llf", "log_likelihood"),
                      ("aic", "aic"), ("bic", "bic")):
        value = getattr(res, attr, None)
        if value is not None:
            try:
                summary[key] = float(value)
            except (TypeError, ValueError):
                pass
    return clean(summary)


def lm_summary(res: Any, *, model: str, formula: str, dataset: str, cov_type: str,
               rename: Callable[[str], str] | None = None) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "model": model, "formula": formula, "dataset": dataset,
        "dep_var": formula.split("~")[0].strip(), "n_obs": int(res.nobs), "cov_type": cov_type,
        "coefficients": coef_rows(res.params, res.std_errors, res.tstats, res.pvalues, res.conf_int(), rename),
    }
    for attr, key in (("rsquared", "r2"), ("rsquared_within", "r2_within"), ("rsquared_between", "r2_between"),
                      ("rsquared_overall", "r2_overall")):
        value = getattr(res, attr, None)
        if value is not None:
            summary[key] = float(value)
    f_stat = getattr(res, "f_statistic", None)
    if f_stat is not None:
        summary["f_stat"], summary["f_pvalue"] = float(f_stat.stat), float(f_stat.pval)
    entity_info = getattr(res, "entity_info", None)
    if entity_info is not None:
        summary["n_entities"] = int(entity_info["total"])
    return clean(summary)


def _finish(summary: dict[str, Any], prep: Any) -> dict[str, Any]:
    summary["sample"] = prep.sample
    if prep.names:
        summary["term_names"] = prep.names
    return summary


@contextmanager
def _rows_outside_sample_are_expected():
    """The dependent variable is blank outside the sample and where lags are missing, so linearmodels drops
    those rows on purpose; its warning about it is noise here."""
    from linearmodels.shared.exceptions import MissingValueWarning

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", MissingValueWarning)
        yield


# ---- equations: statsmodels ---------------------------------------------------------------------------------
def _ls(spec: sp.LsSpec) -> Estimate:
    import statsmodels.formula.api as smf

    weights = getattr(spec, "weights", None)
    cov_type, cluster = spec.cov_type, spec.cluster
    if cov_type == "cluster" and not cluster:
        raise ValueError("cov_type='cluster' requires the 'cluster' column")
    prep = prepare(spec.page, spec.formula, spec.sample, notna=[c for c in (cluster, weights) if c])
    info, df = prep.info, prep.frame
    label = "WLS" if weights else "OLS"
    with expr.readable_errors(prep.columns):
        mod = (smf.wls(prep.formula, data=df, weights=df[weights]) if weights
               else smf.ols(prep.formula, data=df))
    fit_kwargs: dict[str, Any] = {"cov_type": cov_type}
    cov_code = ""
    if cov_type == "cluster":
        # Cluster ids must be aligned with the rows the formula kept after dropping missing values.
        rows = mod.data.row_labels
        groups = df[cluster].reindex(rows) if rows is not None else df[cluster]
        fit_kwargs["cov_kwds"] = {"groups": groups.to_numpy()}
        cov_code = f", cov_kwds={{'groups': df[{cluster!r}]}}"
    elif cov_type == "HAC":
        fit_kwargs["cov_kwds"] = {"maxlags": spec.hac_lags}
        cov_code = f", cov_kwds={{'maxlags': {spec.hac_lags}}}"
    res = mod.fit(**fit_kwargs)
    weights_code = f", weights=df[{weights!r}]" if weights else ""
    code = ("import statsmodels.formula.api as smf\n"
            f"df = qtrix.load({info.name!r})\n"
            f"res = smf.{label.lower()}({spec.formula!r}, data=df{weights_code})"
            f".fit(cov_type={cov_type!r}{cov_code})\nprint(res.summary())")
    summary = model_summary(res, model=label, formula=spec.formula, dataset=info.name, cov_type=cov_type,
                            rename=prep.pretty)
    return Estimate(_finish(summary, prep), res, code)


def _glm(spec: sp.GlmSpec) -> Estimate:
    import statsmodels.api as sm
    import statsmodels.formula.api as smf

    prep = prepare(spec.page, spec.formula, spec.sample)
    family = {"gaussian": sm.families.Gaussian, "binomial": sm.families.Binomial,
              "poisson": sm.families.Poisson, "gamma": sm.families.Gamma}[spec.family]()
    with expr.readable_errors(prep.columns):
        res = smf.glm(prep.formula, data=prep.frame, family=family).fit()
    summary = model_summary(res, model=f"GLM ({spec.family})", formula=spec.formula, dataset=prep.info.name,
                            cov_type="nonrobust", rename=prep.pretty)
    code = (f"import statsmodels.api as sm, statsmodels.formula.api as smf\ndf = qtrix.load({prep.info.name!r})\n"
            f"res = smf.glm({spec.formula!r}, data=df, family=sm.families.{family.__class__.__name__}()).fit()")
    return Estimate(_finish(summary, prep), res, code)


def _binary(spec: sp.BinarySpec) -> Estimate:
    import statsmodels.formula.api as smf

    prep = prepare(spec.page, spec.formula, spec.sample)
    info, df = prep.info, prep.frame
    with expr.readable_errors(prep.columns):
        fit = getattr(smf, spec.link)(prep.formula, data=df).fit(disp=0)
    summary = model_summary(fit, model=spec.link.title(), formula=spec.formula, dataset=info.name,
                            cov_type="nonrobust", rename=prep.pretty)
    _finish(summary, prep)
    code = (f"import statsmodels.formula.api as smf\ndf = qtrix.load({info.name!r})\n"
            f"res = smf.{spec.link}({spec.formula!r}, data=df).fit()\nprint(res.summary())")
    if spec.marginal_effects:
        frame = fit.get_margeff().summary_frame()
        summary["marginal_effects"] = clean({
            prep.pretty(str(k)): {"dy/dx": float(r.iloc[0]), "se": float(r.iloc[1]), "p": float(r.iloc[3])}
            for k, r in frame.iterrows()
        })
        code += "\nprint(res.get_margeff().summary())"
    return Estimate(summary, fit, code)


# ---- equations: linearmodels --------------------------------------------------------------------------------
_PANEL_COV: dict[str, dict[str, Any]] = {
    "unadjusted": {"cov_type": "unadjusted"},
    "robust": {"cov_type": "robust"},
    "cluster_entity": {"cov_type": "clustered", "cluster_entity": True},
    "cluster_both": {"cov_type": "clustered", "cluster_entity": True, "cluster_time": True},
    "kernel": {"cov_type": "kernel"},
}
_PANEL_LABEL = {"fe": "Panel FE", "twfe": "Panel TWFE", "re": "Random effects", "between": "Between",
                "fd": "First differences", "pooled": "Pooled OLS"}


def _panel(spec: sp.PanelSpec) -> Estimate:
    from linearmodels import panel as lm

    ctx = get_context()
    est = spec.estimator
    full_formula = spec.formula
    if est == "fe":
        full_formula += " + EntityEffects"
    elif est == "twfe":
        full_formula += " + EntityEffects + TimeEffects"
    entity, time = spec.entity, spec.time
    if entity and time:  # explicit columns override the declared structure (lags then ignore the sample's past)
        df = ctx.workspace.df(spec.page, sample=spec.sample).set_index([entity, time]).sort_index()
        prep = prepare(spec.page, full_formula, spec.sample, frame=df, kind="panel_undated")
    else:
        ctx.workspace.require(spec.page, PANEL_KINDS, "A panel estimator")
        prep = prepare(spec.page, full_formula, spec.sample)
        entity, time = prep.info.id_col, prep.frame.index.names[1]
    info, df = prep.info, prep.frame
    cls = {"fe": lm.PanelOLS, "twfe": lm.PanelOLS, "re": lm.RandomEffects, "between": lm.BetweenOLS,
           "fd": lm.FirstDifferenceOLS, "pooled": lm.PooledOLS}[est]
    kwargs = _PANEL_COV[spec.cov_type]
    if est == "between" and kwargs.get("cov_type") == "clustered":
        kwargs = {"cov_type": "robust"}
    with expr.readable_errors(prep.columns), _rows_outside_sample_are_expected():
        res = cls.from_formula(prep.formula, data=df).fit(**kwargs)
    code = (f"from linearmodels.panel import {cls.__name__}\n"
            f"df = qtrix.load({info.name!r}).set_index([{entity!r}, {time!r}])\n"
            f"res = {cls.__name__}.from_formula({full_formula!r}, data=df).fit(**{kwargs!r})\nprint(res)")
    summary = lm_summary(res, model=_PANEL_LABEL[est], formula=spec.formula, dataset=info.name,
                         cov_type=spec.cov_type, rename=prep.pretty)
    return Estimate(_finish(summary, prep), res, code)


def _iv(spec: sp._Iv) -> Estimate:
    from linearmodels import iv as lmiv

    if spec.cov_type == "clustered" and not spec.cluster:
        raise ValueError("cov_type='clustered' requires the 'cluster' column")
    prep = prepare(spec.page, spec.formula, spec.sample, notna=[spec.cluster] if spec.cluster else [])
    info, df = prep.info, prep.complete_cases()  # the IV formula parser does not accept missing values
    cls = {"tsls": lmiv.IV2SLS, "liml": lmiv.IVLIML, "gmm": lmiv.IVGMM}[spec.method]
    kwargs: dict[str, Any] = {"cov_type": spec.cov_type}
    with expr.readable_errors(prep.columns), _rows_outside_sample_are_expected():
        model = cls.from_formula(prep.formula, data=df)
        if spec.cov_type == "clustered":  # cluster ids for the rows the model kept
            kwargs["clusters"] = df[spec.cluster].reindex(model.dependent.pandas.index)
        res = model.fit(**kwargs)
    label = {"tsls": "2SLS", "liml": "LIML", "gmm": "GMM"}[spec.method]
    summary = lm_summary(res, model=f"IV ({label})", formula=spec.formula, dataset=info.name,
                         cov_type=spec.cov_type, rename=prep.pretty)
    _finish(summary, prep)
    diagnostics: dict[str, Any] = {}
    try:
        fs = res.first_stage.diagnostics
        diagnostics["first_stage"] = clean({
            str(k): {"partial_r2": float(r["partial.rsquared"]), "f_stat": float(r["f.stat"]),
                     "f_pvalue": float(r["f.pval"])}
            for k, r in fs.iterrows()
        })
    except (AttributeError, KeyError, ValueError):
        pass
    for name in ("sargan", "wooldridge_overid", "wu_hausman"):
        try:
            t = getattr(res, name)
            t = t() if callable(t) else t
            diagnostics[name] = clean({"stat": float(t.stat), "p_value": float(t.pval)})
        except (AttributeError, ValueError, TypeError):
            continue
    summary["iv_diagnostics"] = diagnostics
    code = (f"from linearmodels.iv import {cls.__name__}\ndf = qtrix.load({info.name!r})\n"
            f"res = {cls.__name__}.from_formula({spec.formula!r}, data=df).fit(cov_type={spec.cov_type!r})\n"
            "print(res); print(res.first_stage)")
    return Estimate(summary, res, code)


# ---- time series --------------------------------------------------------------------------------------------
def _ts_frame(spec: sp._Fitted, columns: list[str], what: str) -> tuple[Any, pd.DataFrame]:
    ws = get_context().workspace
    info = ws.require(spec.page, TS_KINDS, what)
    return info, ws.frame(spec.page, columns, sample=spec.sample)


def trim(df: pd.DataFrame | pd.Series):
    """Drop leading/trailing missing periods; interior gaps are left to the estimator."""
    valid = df.notna() if isinstance(df, pd.Series) else df.notna().all(axis=1)
    if not valid.any():
        raise ValueError("The series has no observations")
    first, last = valid.idxmax(), valid[::-1].idxmax()
    return df.loc[first:last]


def _x(index: pd.Index) -> list[str]:
    return [t.isoformat() for t in pd.DatetimeIndex(index)]


def _arima(spec: sp.ArimaSpec) -> Estimate:
    import plotly.graph_objects as go
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    exog = spec.exog or []
    info, df = _ts_frame(spec, [spec.column, *exog], "ARIMA")
    data = trim(df[[spec.column, *exog]])
    y, x = data[spec.column], (data[exog] if exog else None)
    order, seasonal = tuple(spec.order), tuple(spec.seasonal_order) if spec.seasonal_order else (0, 0, 0, 0)
    res = SARIMAX(y, exog=x, order=order, seasonal_order=seasonal).fit(disp=False)
    label = f"ARIMA{order}" + (f"x{tuple(spec.seasonal_order)}" if spec.seasonal_order else "")
    summary = clean({
        "model": label, "formula": f"{spec.column} ~ {label}", "dataset": info.name, "dep_var": spec.column,
        "n_obs": int(res.nobs), "cov_type": "opg", "aic": float(res.aic), "bic": float(res.bic),
        "log_likelihood": float(res.llf),
        "sample": f"{y.index[0].date()} – {y.index[-1].date()}",
        "coefficients": coef_rows(res.params, res.bse, res.tvalues, res.pvalues, res.conf_int()),
    })
    code = (f"from statsmodels.tsa.statespace.sarimax import SARIMAX\ny = qtrix.frame({info.name!r})[{spec.column!r}]\n"
            f"res = SARIMAX(y, order={order!r}, seasonal_order={seasonal!r}).fit(disp=False)\nprint(res.summary())")
    out = Estimate(summary, res, code, {"title": f"{label} · {spec.column}"})
    if spec.forecast_steps and x is None:
        fc = res.get_forecast(spec.forecast_steps)
        mean, ci = fc.predicted_mean, fc.conf_int(alpha=0.05)
        fig = go.Figure()
        fig.add_scatter(x=_x(y.index), y=y.to_numpy(), name=spec.column, mode="lines")
        fut = _x(mean.index)
        fig.add_scatter(x=fut + fut[::-1], y=list(ci.iloc[:, 1]) + list(ci.iloc[:, 0])[::-1], fill="toself",
                        name="95% interval", line={"width": 0}, opacity=0.3)
        fig.add_scatter(x=fut, y=mean.to_numpy(), name="Forecast", mode="lines")
        summary["forecast"] = clean({str(t.date()): float(v) for t, v in mean.items()})
        out.extras["forecast_figure"] = figure_data(fig)
        out.extras["forecast_title"] = f"Forecast {spec.column} · {label}"
    return out


def _var(spec: sp.VarSpec) -> Estimate:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from statsmodels.tsa.api import VAR

    columns = spec.columns
    info, df = _ts_frame(spec, columns, "A VAR")
    data = trim(df[columns])
    if data.isna().any().any():
        raise ValueError(f"The VAR sample {data.index[0].date()}–{data.index[-1].date()} has missing periods; "
                         "fill them or use a shorter sample")
    res = VAR(data).fit(maxlags=spec.maxlags, ic="aic")
    if res.k_ar == 0:
        res = VAR(data).fit(1)
    granger = {}
    for caused in columns:
        for causing in columns:
            if caused != causing:
                t = res.test_causality(caused, [causing], kind="f")
                granger[f"{causing} -> {caused}"] = {"f_stat": float(t.test_statistic), "p_value": float(t.pvalue)}
    irf = res.irf(spec.irf_periods)
    k = len(columns)
    fig = make_subplots(rows=k, cols=k, subplot_titles=[f"{imp} → {resp}" for resp in columns for imp in columns])
    for r in range(k):
        for c in range(k):
            fig.add_trace(go.Scatter(y=irf.irfs[:, r, c], mode="lines", showlegend=False), row=r + 1, col=c + 1)
    fig.update_layout(height=220 * k)
    code = (f"from statsmodels.tsa.api import VAR\nres = VAR(qtrix.frame({info.name!r})[{columns!r}].dropna())"
            f".fit(maxlags={spec.maxlags}, ic='aic')\nres.irf({spec.irf_periods}).plot()")
    summary = clean({
        "model": "VAR", "formula": f"VAR({res.k_ar}): {', '.join(columns)}", "dataset": info.name,
        "columns": columns, "lags": res.k_ar, "n_obs": int(res.nobs), "aic": float(res.aic), "granger": granger,
        "sample": f"{data.index[0].date()} – {data.index[-1].date()}",
    })
    return Estimate(summary, res, code, {"figure": figure_data(fig), "title": f"Impulse responses · VAR({res.k_ar})"})


def _vecm(spec: sp.VecmSpec) -> Estimate:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from statsmodels.tsa.vector_ar.vecm import VECM, select_coint_rank

    columns = spec.columns
    if len(columns) < 2:
        raise ValueError("A VECM needs at least two series")
    info, df = _ts_frame(spec, columns, "A VECM")
    data = trim(df[columns])
    if data.isna().any().any():
        raise ValueError(f"The VECM sample {data.index[0].date()}–{data.index[-1].date()} has missing periods; "
                         "fill them or use a shorter sample")
    rank = spec.coint_rank
    if rank is None:
        rank = int(select_coint_rank(data, det_order=0 if spec.deterministic in ("n", "co", "ci") else 1,
                                     k_ar_diff=spec.k_ar_diff, method="trace", signif=0.05).rank)
    if not 0 <= rank < len(columns):
        raise ValueError(f"coint_rank must be between 0 and {len(columns) - 1}")
    if rank == 0:
        raise ValueError("The Johansen trace test finds no cointegration (rank 0): use ts_var on differences instead")
    res = VECM(data, k_ar_diff=spec.k_ar_diff, coint_rank=rank, deterministic=spec.deterministic).fit()
    names = [f"ect{i + 1}" for i in range(rank)]
    beta = pd.DataFrame(res.beta, index=columns, columns=names)
    alpha = pd.DataFrame(res.alpha, index=columns, columns=names)
    irf = res.irf(spec.irf_periods)
    k = len(columns)
    fig = make_subplots(rows=k, cols=k, subplot_titles=[f"{imp} → {resp}" for resp in columns for imp in columns])
    for r in range(k):
        for c in range(k):
            fig.add_trace(go.Scatter(y=irf.irfs[:, r, c], mode="lines", showlegend=False), row=r + 1, col=c + 1)
    fig.update_layout(height=220 * k)
    code = (f"from statsmodels.tsa.vector_ar.vecm import VECM\n"
            f"res = VECM(qtrix.frame({info.name!r})[{columns!r}].dropna(), k_ar_diff={spec.k_ar_diff}, "
            f"coint_rank={rank}, deterministic={spec.deterministic!r}).fit()\nprint(res.summary())")
    summary = clean({
        "model": "VECM", "formula": f"VECM(rank={rank}, k_ar_diff={spec.k_ar_diff}): {', '.join(columns)}",
        "dataset": info.name, "columns": columns, "lags": spec.k_ar_diff, "coint_rank": rank,
        "n_obs": int(res.nobs), "log_likelihood": float(res.llf.real),
        "beta": {c: beta[c].to_dict() for c in names}, "alpha": {c: alpha[c].to_dict() for c in names},
        "sample": f"{data.index[0].date()} – {data.index[-1].date()}",
    })
    return Estimate(summary, res, code, {"figure": figure_data(fig), "title": f"Impulse responses · VECM(rank {rank})"})


def _garch(spec: sp.GarchSpec) -> Estimate:
    import plotly.graph_objects as go
    from arch import arch_model

    info, df = _ts_frame(spec, [spec.column], "A GARCH model")
    y = trim(df[spec.column]).dropna()
    o = spec.o or (1 if spec.model == "gjr" else 0)
    vol = "EGARCH" if spec.model == "egarch" else "GARCH"
    mean = {"constant": "Constant", "zero": "Zero", "ar": "AR"}[spec.mean]
    kwargs = {"lags": 1} if spec.mean == "ar" else {}
    res = arch_model(y, mean=mean, vol=vol, p=spec.p, o=o, q=spec.q, dist=spec.dist, **kwargs).fit(disp="off")
    label = {"garch": "GARCH", "egarch": "EGARCH", "gjr": "GJR-GARCH"}[spec.model] + f"({spec.p},{spec.q})"
    ci = res.conf_int()
    coefs = [clean({"name": str(n), "coef": float(res.params[n]), "se": float(res.std_err[n]),
                    "stat": float(res.tvalues[n]), "p": float(res.pvalues[n]),
                    "ci_low": float(ci.loc[n].iloc[0]), "ci_high": float(ci.loc[n].iloc[1])})
             for n in res.params.index]
    persistence = None
    if spec.model != "egarch":
        persistence = float(sum(v for n, v in res.params.items() if str(n).startswith(("alpha[", "beta[")))
                            + 0.5 * sum(v for n, v in res.params.items() if str(n).startswith("gamma[")))
    summary = {
        "model": label, "formula": f"{spec.column} ~ {label}, {spec.dist}", "dataset": info.name,
        "dep_var": spec.column, "n_obs": int(res.nobs), "cov_type": str(res.cov_type).lower(),
        "aic": float(res.aic), "bic": float(res.bic), "log_likelihood": float(res.loglikelihood),
        "persistence": persistence, "sample": f"{y.index[0].date()} – {y.index[-1].date()}", "coefficients": coefs,
    }
    code = (f"from arch import arch_model\ny = qtrix.frame({info.name!r})[{spec.column!r}].dropna()\n"
            f"res = arch_model(y, mean={mean!r}, vol={vol!r}, p={spec.p}, o={o}, q={spec.q}, "
            f"dist={spec.dist!r}).fit(disp='off')\nprint(res.summary())")
    fig = go.Figure()
    fig.add_scatter(x=_x(y.index), y=y.abs().to_numpy(), name=f"|{spec.column}|", mode="lines", opacity=0.3)
    fig.add_scatter(x=_x(y.index), y=res.conditional_volatility.to_numpy(), name="Conditional volatility",
                    mode="lines")
    out = Estimate(clean(summary), res, code,
                   {"figure": figure_data(fig), "title": f"Conditional volatility · {label}"})
    if spec.forecast_steps:
        var = res.forecast(horizon=spec.forecast_steps, method="simulation" if spec.model == "egarch" else "analytic",
                           reindex=False).variance.iloc[-1].to_numpy()
        freq = pd.infer_freq(y.index)
        keys = ([str(t.date()) for t in pd.date_range(y.index[-1], periods=spec.forecast_steps + 1, freq=freq)[1:]]
                if freq else [f"h{i + 1}" for i in range(len(var))])
        out.summary["volatility_forecast"] = clean(dict(zip(keys, (var ** 0.5).tolist(), strict=True)))
    return out


# ---- objects built from other objects or from plain content -------------------------------------------------
def _group(spec: sp.GroupSpec) -> Estimate:
    if spec.page:
        names = {c["name"] for c in get_context().workspace.get(spec.page).columns or []}
        missing = [m for m in spec.members if m not in names]
        if missing:
            raise ValueError(f"Series not in page '{spec.page}': {', '.join(missing)}")
    return Estimate({"members": list(spec.members), "page": spec.page})


def _table(spec: sp.TableSpec) -> Estimate:
    objects = get_context().objects
    models = [{k: v for k, v in objects.get(ref).summary.items() if k != "diff"} for ref in spec.sources]
    return Estimate({"title": spec.title, "sources": [objects.get(r).name for r in spec.sources], "models": models})


def _graph(spec: sp.GraphSpec) -> Estimate:
    return Estimate({"title": spec.title, "figure": spec.figure, "source": spec.source})


def _text(spec: sp.TextSpec) -> Estimate:
    return Estimate({"title": spec.title, "markdown": spec.markdown})


def _frozen_table(spec: sp.FrozenTableSpec) -> Estimate:
    return Estimate({"title": spec.title, "kind": spec.kind, "data": spec.data, "source": spec.source,
                     "view": spec.view})


def _frozen_graph(spec: sp.FrozenGraphSpec) -> Estimate:
    return Estimate({"title": spec.title, "figure": spec.figure, "source": spec.source, "view": spec.view})


# ---- results computed in the kernel (E7·07b) ----------------------------------------------------------------
def summarize_result(res: Any, *, page: str = "", title: str = "") -> dict[str, Any]:
    """Summary of a statsmodels or linearmodels result registered from a Python cell (``qx.register``).

    Runs in the kernel, next to the result: only this JSON travels to the engine.
    """
    model = getattr(res, "model", None)
    formula = getattr(model, "formula", None) or ""
    if not isinstance(formula, str):
        formula = str(formula)
    label = title or (type(model).__name__ if model is not None else type(res).__name__)
    cov_type = str(getattr(res, "cov_type", "") or "")
    if hasattr(res, "std_errors") and hasattr(res, "tstats"):  # linearmodels
        summary = lm_summary(res, model=label, formula=formula, dataset=page, cov_type=cov_type)
    elif all(hasattr(res, a) for a in ("params", "bse", "tvalues", "pvalues", "conf_int", "nobs")):  # statsmodels
        summary = model_summary(res, model=label, formula=formula, dataset=page, cov_type=cov_type)
    else:
        raise TypeError(f"qx.register supports statsmodels and linearmodels results, not {type(res).__name__}. "
                        "Put other output in the report (qx.report.add_note) or as series (qx.series_from)")
    if not formula:
        summary["dep_var"] = str(getattr(model, "endog_names", "") or "")
    return summary


def _python(spec: sp.PythonSpec) -> Estimate:
    kernels = get_context().kernels
    if kernels is None:
        raise ValueError("Re-running a Python cell needs the project kernel, which this engine process does not have")
    return Estimate(kernels.rebuild(spec.code, spec.registered_as), None, spec.code)


_ESTIMATORS: dict[str, Callable[[Any], Estimate]] = {
    "ls": _ls, "wls": _ls, "glm": _glm, "binary": _binary, "tsls": _iv, "liml": _iv, "gmm": _iv,
    "panel": _panel, "arima": _arima, "garch": _garch, "var": _var,
    "vecm": _vecm, "group": _group, "table": _table, "graph": _graph,
    "text": _text, "python": _python, "frozen_table": _frozen_table, "frozen_graph": _frozen_graph,
}


def estimate(spec: sp._Spec) -> Estimate:
    """Run the estimator of ``spec.method`` on the current page data (no side effects on any store)."""
    return _ESTIMATORS[spec.method](spec)  # type: ignore[attr-defined]


def summaries_match(a: Any, b: Any, tol: float = 1e-10) -> bool:
    """Structural equality of two summaries with a numeric tolerance; the ``diff`` view is ignored."""
    if isinstance(a, dict) and isinstance(b, dict):
        ka, kb = {k for k in a if k != "diff"}, {k for k in b if k != "diff"}
        return ka == kb and all(summaries_match(a[k], b[k], tol) for k in ka)
    if isinstance(a, list | tuple) and isinstance(b, list | tuple):
        return len(a) == len(b) and all(summaries_match(x, y, tol) for x, y in zip(a, b, strict=True))
    if isinstance(a, int | float) and isinstance(b, int | float) and not isinstance(a, bool):
        return abs(a - b) <= tol * max(1.0, abs(a), abs(b))
    return a == b
