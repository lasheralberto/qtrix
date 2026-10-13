"""ts_* tools: time-series econometrics (unit roots, ARIMA, VAR).

They only apply to pages declared as time series (ADR 0002 §1): the series come indexed by real dates, so
lags respect the calendar, gaps stay visible as missing values and forecasts are dated.
"""

from __future__ import annotations

from typing import Any, Literal

from ..context import get_context
from ..objects.specs import ArimaSpec, VarSpec, VecmSpec
from ..objects.views import show
from ..panel import Block
from ._format import clean, figure_data
from .ols import publish, resolved_sample
from .registry import Qx, tool

TS = ("time_series",)


def _frame(dataset_id: str, columns: list[str], what: str, sample: str | None = None):
    ctx = get_context()
    info = ctx.workspace.require(dataset_id, TS, what)
    return info, ctx.workspace.frame(dataset_id, columns, sample=sample)


@tool("ts", requires=("statsmodels",), kinds=TS, qx=Qx("page", "unit_root", ("column",), aliases=("uroot",)))
def ts_unit_root(dataset_id: str, column: str, test: Literal["adf", "kpss", "pp", "za"] = "adf",
                 trend: Literal["n", "c", "ct"] = "c", sample: str | None = None) -> dict[str, Any]:
    """Unit-root / stationarity test on a series: ADF, KPSS, Phillips-Perron or Zivot-Andrews (structural break).

    Args:
        dataset_id: Dataset id or name (must be a time series).
        column: Series to test.
        test: adf (H0 unit root), kpss (H0 stationary), pp (H0 unit root, needs arch), za (needs arch).
        trend: n (none), c (constant) or ct (constant + trend).
        sample: Sample for this call only, e.g. "1990q1 2008q4"; omit for the dataset's active sample.
    """
    info, df = _frame(dataset_id, [column], "A unit-root test", sample)
    y = df[column].dropna()
    if test in {"pp", "za"}:
        from arch import unitroot

        if test == "pp":
            res = unitroot.PhillipsPerron(y.to_numpy(), trend=trend)
        else:
            res = unitroot.ZivotAndrews(y.to_numpy(), trend="c" if trend == "n" else trend)
        out = {"statistic": float(res.stat), "p_value": float(res.pvalue), "lags": int(res.lags),
               "critical_values": {k: float(v) for k, v in res.critical_values.items()},
               "null_hypothesis": res.null_hypothesis}
    elif test == "kpss":
        from statsmodels.tsa.stattools import kpss

        stat, p, lags, crit = kpss(y.to_numpy(), regression="ct" if trend == "ct" else "c", nlags="auto")
        out = {"statistic": stat, "p_value": p, "lags": lags, "critical_values": crit,
               "null_hypothesis": "The series is stationary"}
    else:
        from statsmodels.tsa.stattools import adfuller

        stat, p, lags, nobs, crit, _ = adfuller(y.to_numpy(), regression=trend, autolag="AIC")
        out = {"statistic": stat, "p_value": p, "lags": lags, "n_obs": nobs, "critical_values": crit,
               "null_hypothesis": "The series has a unit root"}
    out.update({"test": test.upper(), "series": column, "dataset": info.name, "trend": trend,
                "sample": f"{y.index[0].date()} – {y.index[-1].date()}"})
    out = clean(out)
    get_context().panel.add(Block(
        kind="diagnostics", title=f"{test.upper()} test · {column}",
        data={"tests": [{"test": out["test"], "statistic": out["statistic"], "p_value": out["p_value"],
                         "null_hypothesis": out["null_hypothesis"],
                         "reject_5pct": out["p_value"] is not None and out["p_value"] < 0.05}]},
    ))
    return out


@tool("ts", requires=("statsmodels",), kinds=TS, mutating=True, qx=Qx("qx", "equation", ("name",), family="equation"))
def ts_arima(dataset_id: str, column: str, order: list[int], seasonal_order: list[int] | None = None,
             exog: list[str] | None = None, forecast_steps: int = 0, sample: str | None = None,
             name: str | None = None) -> dict[str, Any]:
    """Fit a (S)ARIMA(X) model and optionally forecast; table and dated forecast chart go to the panel.
    Stored as a named equation object; `name` re-estimates an existing one as a new version.

    Args:
        dataset_id: Dataset id or name (must be a time series).
        column: Dependent series.
        order: [p, d, q].
        seasonal_order: Optional [P, D, Q, s].
        exog: Optional exogenous regressors (no forecast when given).
        forecast_steps: Periods ahead to forecast with 95% intervals, starting after the sample's last period.
        sample: Sample for this call only, e.g. "1990q1 2008q4"; omit for the dataset's active sample.
        name: Object name (e.g. eq_gdp_ar1). An existing name is re-estimated as a new version; omit for eq01...
    """
    ctx = get_context()
    spec = ArimaSpec(page=ctx.workspace.get(dataset_id).name, sample=resolved_sample(dataset_id, sample),
                     column=column, order=order, seasonal_order=seasonal_order, exog=exog,
                     forecast_steps=forecast_steps)
    obj, out = publish(spec, name, title=f"ARIMA · {column}")
    if obj.extras.get("forecast_figure"):
        show(obj.name, "forecast")
        out["forecast"] = obj.summary["forecast"]
    return out


@tool("ts", requires=("statsmodels",), kinds=TS, mutating=True, qx=Qx("qx", "var", ("name",), family="var"))
def ts_var(dataset_id: str, columns: list[str], maxlags: int = 8, irf_periods: int = 10,
           sample: str | None = None, name: str | None = None) -> dict[str, Any]:
    """Vector autoregression with lag selection by AIC, Granger causality tests and impulse-response chart.
    Stored as a named var object; `name` re-estimates an existing one as a new version.

    Args:
        dataset_id: Dataset id or name (must be a time series).
        columns: Endogenous series.
        maxlags: Largest lag considered by AIC.
        irf_periods: Horizon of the impulse responses.
        sample: Sample for this call only, e.g. "1990q1 2008q4"; omit for the dataset's active sample.
        name: Object name (e.g. var_macro). An existing name is re-estimated as a new version; omit for var01...
    """
    ctx = get_context()
    spec = VarSpec(page=ctx.workspace.get(dataset_id).name, sample=resolved_sample(dataset_id, sample),
                   columns=columns, maxlags=maxlags, irf_periods=irf_periods)
    obj = ctx.objects.estimate(spec, name)
    show(obj.name, "estimation")
    show(obj.name, "irf")
    s = obj.summary
    out = {"result_id": obj.name, "name": obj.name, "version": obj.version, "selected_lags": s["lags"],
           "aic": s["aic"], "granger": s["granger"], "sample": s["sample"]}
    if "diff" in s:
        out["previous_version"] = s["diff"]["from_version"]
    return clean(out)


@tool("ts", requires=("statsmodels",), kinds=TS, mutating=True, qx=Qx("qx", "var", ("name",), family="var"))
def ts_vecm(dataset_id: str, columns: list[str], coint_rank: int | None = None, k_ar_diff: int = 2,
            deterministic: Literal["n", "co", "ci", "lo", "li"] = "co", irf_periods: int = 10,
            sample: str | None = None, name: str | None = None) -> dict[str, Any]:
    """Vector error-correction model for cointegrated I(1) series, with impulse responses.
    The cointegration rank is chosen with the Johansen trace test (5%) unless given.
    Stored as a named var object; `name` re-estimates an existing one as a new version.

    Args:
        dataset_id: Dataset id or name (must be a time series).
        columns: Endogenous I(1) series (at least two).
        coint_rank: Number of cointegrating relations; omit to select it with the Johansen trace test.
        k_ar_diff: Lagged differences in the VECM.
        deterministic: n none, co/ci constant outside/inside the relation, lo/li linear trend outside/inside.
        irf_periods: Horizon of the impulse responses.
        sample: Sample for this call only, e.g. "1990q1 2008q4"; omit for the dataset's active sample.
        name: Object name (e.g. vecm_rates). An existing name is re-estimated as a new version.
    """
    ctx = get_context()
    spec = VecmSpec(page=ctx.workspace.get(dataset_id).name, sample=resolved_sample(dataset_id, sample),
                    columns=columns, coint_rank=coint_rank, k_ar_diff=k_ar_diff, deterministic=deterministic,
                    irf_periods=irf_periods)
    obj = ctx.objects.estimate(spec, name)
    show(obj.name, "estimation")
    show(obj.name, "irf")
    s = obj.summary
    out = {"result_id": obj.name, "name": obj.name, "version": obj.version, "coint_rank": s["coint_rank"],
           "beta": s["beta"], "alpha": s["alpha"], "sample": s["sample"]}
    if "diff" in s:
        out["previous_version"] = s["diff"]["from_version"]
    return clean(out)


@tool("ts", requires=("statsmodels",), kinds=TS, qx=Qx("page", "cointegration", ("columns",), aliases=("coint",)))
def ts_cointegration(dataset_id: str, columns: list[str], test: Literal["johansen", "engle_granger"] = "johansen",
                     trend: Literal["n", "c", "ct"] = "c", k_ar_diff: int = 1,
                     sample: str | None = None) -> dict[str, Any]:
    """Cointegration test among I(1) series: Johansen trace/max-eigenvalue (any number of series) or
    Engle-Granger (first series against the others).

    Args:
        dataset_id: Dataset id or name (must be a time series).
        columns: Series to test (at least two). For Engle-Granger the first is the dependent one.
        test: johansen or engle_granger.
        trend: n none, c constant, ct constant + trend.
        k_ar_diff: Lagged differences (Johansen only).
        sample: Sample for this call only; omit for the dataset's active sample.
    """
    if len(columns) < 2:
        raise ValueError("A cointegration test needs at least two series")
    info, df = _frame(dataset_id, columns, "A cointegration test", sample)
    data = df[columns].dropna()
    span = f"{data.index[0].date()} – {data.index[-1].date()}"
    if test == "engle_granger":
        from statsmodels.tsa.stattools import coint

        stat, p, crit = coint(data[columns[0]], data[columns[1:]], trend=trend)
        out = {"test": "Engle-Granger", "dependent": columns[0], "statistic": stat, "p_value": p,
               "critical_values": dict(zip(["1%", "5%", "10%"], crit, strict=True)),
               "null_hypothesis": "No cointegration"}
        rows = [{"test": "Engle-Granger", "statistic": stat, "p_value": p,
                 "null_hypothesis": out["null_hypothesis"], "reject_5pct": bool(p < 0.05)}]
    else:
        from statsmodels.tsa.vector_ar.vecm import coint_johansen

        res = coint_johansen(data, det_order={"n": -1, "c": 0, "ct": 1}[trend], k_ar_diff=k_ar_diff)
        relations = [{"null": f"rank <= {r}", "trace": float(res.lr1[r]), "trace_crit_5pct": float(res.cvt[r, 1]),
                      "max_eig": float(res.lr2[r]), "max_eig_crit_5pct": float(res.cvm[r, 1]),
                      "reject_trace_5pct": bool(res.lr1[r] > res.cvt[r, 1])} for r in range(len(columns))]
        rank = 0
        for rel in relations:
            if not rel["reject_trace_5pct"]:
                break
            rank += 1
        out = {"test": "Johansen", "relations": relations, "rank_trace_5pct": rank,
               "eigenvalues": [float(v.real) for v in res.eig]}
        rows = [{"test": f"Johansen trace, H0 {r['null']}", "statistic": r["trace"], "p_value": None,
                 "null_hypothesis": f"{r['null']} (5% critical value {r['trace_crit_5pct']:.2f})",
                 "reject_5pct": r["reject_trace_5pct"]} for r in relations]
    out.update({"series": columns, "dataset": info.name, "trend": trend, "sample": span})
    out = clean(out)
    get_context().panel.add(Block(kind="diagnostics", title=f"Cointegration ({out['test']}) · {', '.join(columns)}",
                                  data={"tests": clean(rows)}))
    return out


@tool("ts", requires=("statsmodels",), kinds=TS, qx=Qx("page", "correlogram", ("column",)))
def ts_acf_pacf(dataset_id: str, column: str, lags: int = 24, diff: int = 0,
                sample: str | None = None) -> dict[str, Any]:
    """Autocorrelation and partial-autocorrelation correlogram with 95% bands, plus the Ljung-Box test.

    Args:
        dataset_id: Dataset id or name (must be a time series).
        column: Series to analyse.
        lags: Number of lags shown.
        diff: Times the series is differenced first (0, 1 or 2).
        sample: Sample for this call only; omit for the dataset's active sample.
    """
    import numpy as np
    from plotly.subplots import make_subplots
    from statsmodels.stats.diagnostic import acorr_ljungbox
    from statsmodels.tsa.stattools import acf, pacf

    info, df = _frame(dataset_id, [column], "A correlogram", sample)
    y = df[column].dropna()
    for _ in range(diff):
        y = y.diff().dropna()
    lags = max(1, min(lags, len(y) // 2 - 1))
    a, a_ci = acf(y, nlags=lags, alpha=0.05)
    p, p_ci = pacf(y, nlags=lags, alpha=0.05)
    k = list(range(1, lags + 1))
    fig = make_subplots(rows=1, cols=2, subplot_titles=["ACF", "PACF"])
    for col, vals, ci in ((1, a, a_ci), (2, p, p_ci)):
        fig.add_bar(x=k, y=vals[1:], row=1, col=col, showlegend=False)
        band = float(np.mean(ci[1:, 1] - vals[1:]))
        fig.add_hline(y=band, line_dash="dot", row=1, col=col)
        fig.add_hline(y=-band, line_dash="dot", row=1, col=col)
    label = column if not diff else f"d{diff}({column})"
    get_context().panel.add(Block(kind="plotly_figure", title=f"Correlogram · {label}", data=figure_data(fig)))
    lb = acorr_ljungbox(y, lags=[min(lags, 10)])
    return clean({"series": label, "dataset": info.name, "n_obs": len(y),
                  "acf": {str(i): float(a[i]) for i in k[:12]}, "pacf": {str(i): float(p[i]) for i in k[:12]},
                  "ljung_box": {"lags": min(lags, 10), "statistic": float(lb["lb_stat"].iloc[0]),
                                "p_value": float(lb["lb_pvalue"].iloc[0])}})


@tool("ts", requires=("ruptures",), kinds=TS, qx=Qx("page", "breaks", ("column",)))
def ts_breaks(dataset_id: str, column: str, max_breaks: int = 3, min_size: int = 8,
              sample: str | None = None) -> dict[str, Any]:
    """Detect structural breaks in the mean of a series (binary segmentation, ruptures) and plot the segments.

    Args:
        dataset_id: Dataset id or name (must be a time series).
        column: Series to scan.
        max_breaks: Maximum number of breakpoints to report.
        min_size: Minimum observations between breaks.
        sample: Sample for this call only; omit for the dataset's active sample.
    """
    import plotly.graph_objects as go
    import ruptures as rpt

    info, df = _frame(dataset_id, [column], "A structural-break search", sample)
    y = df[column].dropna()
    ends = sorted(rpt.Binseg(model="l2", min_size=min_size, jump=1).fit(y.to_numpy()).predict(n_bkps=max_breaks))
    starts = [0, *ends[:-1]]
    segments = [{"from": str(y.index[s].date()), "to": str(y.index[e - 1].date()), "mean": float(y.iloc[s:e].mean())}
                for s, e in zip(starts, ends, strict=True)]
    fig = go.Figure()
    fig.add_scatter(x=[t.isoformat() for t in y.index], y=y.to_numpy(), name=column, mode="lines")
    for seg, s, e in zip(segments, starts, ends, strict=True):
        fig.add_scatter(x=[y.index[s].isoformat(), y.index[e - 1].isoformat()], y=[seg["mean"]] * 2,
                        name=f"mean {seg['mean']:.3g}", mode="lines", line={"dash": "dash"})
    get_context().panel.add(Block(kind="plotly_figure", title=f"Structural breaks · {column}", data=figure_data(fig)))
    return clean({"series": column, "dataset": info.name, "break_dates": [str(y.index[e].date()) for e in ends[:-1]],
                  "segments": segments})


@tool("ts", requires=("statsforecast",), kinds=TS, qx=Qx("page", "forecast", ("column",)))
def ts_forecast(dataset_id: str, column: str, horizon: int = 8,
                model: Literal["auto_arima", "auto_ets", "naive", "seasonal_naive"] = "auto_arima",
                season_length: int | None = None, sample: str | None = None) -> dict[str, Any]:
    """Out-of-the-box forecast with statsforecast (automatic ARIMA/ETS or naive benchmarks), with 95% fan chart.
    Use ts_arima to forecast with a model you specify yourself.

    Args:
        dataset_id: Dataset id or name (must be a time series).
        column: Series to forecast.
        horizon: Periods ahead, dated after the sample's last period.
        model: auto_arima, auto_ets, naive or seasonal_naive.
        season_length: Seasonal period (e.g. 4 quarterly, 12 monthly); inferred from the frequency when omitted.
        sample: Sample for this call only; omit for the dataset's active sample.
    """
    import pandas as pd
    import plotly.graph_objects as go
    from statsforecast import StatsForecast
    from statsforecast.models import AutoARIMA, AutoETS, Naive, SeasonalNaive

    info, df = _frame(dataset_id, [column], "A forecast", sample)
    y = df[column].dropna()
    freq = pd.infer_freq(y.index)
    if freq is None:
        raise ValueError("The series has no regular frequency (gaps or irregular dates): forecasting needs one")
    if season_length is None:
        season_length = {"Q": 4, "M": 12, "W": 52, "D": 7, "B": 5}.get(freq[0].upper(), 1)
    algo = {"auto_arima": AutoARIMA, "auto_ets": AutoETS, "naive": Naive, "seasonal_naive": SeasonalNaive}[model]
    kwargs = {} if model == "naive" else {"season_length": season_length}
    frame = pd.DataFrame({"unique_id": column, "ds": y.index, "y": y.to_numpy()})
    fc = StatsForecast(models=[algo(**kwargs)], freq=freq).forecast(df=frame, h=horizon, level=[95])
    label = algo.__name__
    mean, lo, hi = fc[label].to_numpy(), fc[f"{label}-lo-95"].to_numpy(), fc[f"{label}-hi-95"].to_numpy()
    fut = [t.isoformat() for t in pd.DatetimeIndex(fc["ds"])]
    fig = go.Figure()
    fig.add_scatter(x=[t.isoformat() for t in y.index], y=y.to_numpy(), name=column, mode="lines")
    fig.add_scatter(x=fut + fut[::-1], y=list(hi) + list(lo)[::-1], fill="toself", name="95% interval",
                    line={"width": 0}, opacity=0.3)
    fig.add_scatter(x=fut, y=mean, name="Forecast", mode="lines")
    get_context().panel.add(Block(kind="plotly_figure", title=f"Forecast {column} · {model}", data=figure_data(fig)))
    return clean({"series": column, "dataset": info.name, "model": model, "season_length": season_length,
                  "forecast": {t[:10]: float(v) for t, v in zip(fut, mean, strict=True)},
                  "lower_95": [float(v) for v in lo], "upper_95": [float(v) for v in hi]})
