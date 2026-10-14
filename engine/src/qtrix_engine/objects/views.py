"""Views and procs of named objects, and the report's live blocks (ADR 0002 §4-5, E7·08).

A **view** is a pure function of an object version and some parameters: ``@view("equation", "residuals")``. It
returns a ``ViewResult`` (a block payload of the existing kinds, a compact ``llm`` dict for agents and, for checks,
an outcome) and is cached by ``(name, version, view, params)``. Views that need the fitted model rebuild it from the
spec when it is not in memory, and fail with an actionable message when the object is stale; views computed from the
stored summary always work. A **proc** is an action on an object that goes through the dispatcher (a registry tool);
the registry only describes it, so menus, the ``qx`` API and the agent tools all come from one place.

The report: ``show`` adds a *live* block (``Block.ref``) and ``refresh`` recomputes the live blocks of an object when it
gets a new version, so re-estimating ``eq01`` updates its table and its charts in place. Blocks without ``ref`` are
frozen snapshots and are only flagged stale.
"""

from __future__ import annotations

import importlib.util
import inspect
import json
import logging
import typing
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import ConfigDict, ValidationError, create_model

from ..context import get_context
from ..panel import Block, PanelState
from ..tools._format import clean, figure_data, llm_view
from .store import ObjectError, StoredObject

log = logging.getLogger(__name__)

DEFAULT_VIEW = {"equation": "estimation", "var": "estimation", "table": "table", "graph": "graph", "group": "members",
                "text": "text"}


@dataclass
class ViewResult:
    kind: str  # block kind: regression_table, plotly_figure, dataframe, diagnostics, markdown, equation
    title: str
    data: dict[str, Any]
    llm: dict[str, Any] | None = None  # compact view handed to agents
    outcome: Literal["ok", "warn", "fail"] | None = None  # for views that are checks
    code: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "title": self.title, "data": self.data, "llm": self.llm, "outcome": self.outcome,
                "code": self.code}


@dataclass(frozen=True)
class ViewSpec:
    type: str
    name: str
    fn: Callable[..., ViewResult]
    title: str
    group: str = "View"
    methods: frozenset[str] | None = None  # object methods it applies to (None: any)
    kinds: frozenset[str] | None = None  # page structures it applies to (None: any)
    check: bool = False  # a diagnostic check: its result carries an outcome
    cost: Literal["cheap", "expensive"] = "cheap"
    requires: tuple[str, ...] = ()
    when: Callable[[StoredObject], bool] | None = None
    _model: Any = field(default=None, compare=False, repr=False)

    def applies(self, obj: StoredObject, page_kind: str | None) -> bool:
        return (obj.type == self.type and (self.methods is None or obj.method in self.methods)
                and (self.kinds is None or page_kind in self.kinds)
                and all(importlib.util.find_spec(m) is not None for m in self.requires)
                and (self.when is None or bool(self.when(obj))))

    def params(self, given: dict[str, Any] | None) -> dict[str, Any]:
        """Validated parameters with defaults filled in (so equal views share one cache entry and one ref)."""
        try:
            return self._model.model_validate(given or {}).model_dump()
        except ValidationError as exc:
            err = exc.errors()[0]
            raise ObjectError(f"View '{self.name}': {'.'.join(map(str, err['loc']))}: {err['msg']}") from None

    def describe(self) -> dict[str, Any]:
        schema = self._model.model_json_schema().get("properties", {})
        return {"name": self.name, "title": self.title, "group": self.group, "check": self.check, "cost": self.cost,
                "params": {k: {key: v[key] for key in ("type", "default", "enum") if key in v}
                           for k, v in schema.items()}}


@dataclass(frozen=True)
class ProcSpec:
    """An action on an object, run through the dispatcher as the registry tool ``tool``."""

    type: str
    name: str
    tool: str
    title: str
    group: str = "Proc"
    methods: frozenset[str] | None = None

    def describe(self) -> dict[str, Any]:
        return {"name": self.name, "tool": self.tool, "title": self.title, "group": self.group}


_VIEWS: dict[tuple[str, str], ViewSpec] = {}
_PROCS: dict[tuple[str, str], ProcSpec] = {}


def view(type_: str, name: str, *, title: str, group: str = "View", methods: typing.Iterable[str] | None = None,
         kinds: typing.Iterable[str] | None = None, check: bool = False, cost: Literal["cheap", "expensive"] = "cheap",
         requires: typing.Iterable[str] = (),
         when: Callable[[StoredObject], bool] | None = None) -> Callable[[Callable[..., ViewResult]], Any]:
    def decorator(fn: Callable[..., ViewResult]) -> Callable[..., ViewResult]:
        hints = typing.get_type_hints(fn)
        fields = {p.name: (hints.get(p.name, Any), ... if p.default is inspect.Parameter.empty else p.default)
                  for p in list(inspect.signature(fn).parameters.values())[1:]}
        model = create_model(f"view_{type_}_{name}", __config__=ConfigDict(extra="forbid"), **fields)
        _VIEWS[(type_, name)] = ViewSpec(type_, name, fn, title, group, frozenset(methods) if methods else None,
                                         frozenset(kinds) if kinds else None, check, cost, tuple(requires), when,
                                         model)
        return fn

    return decorator


def proc(type_: str, name: str, *, tool: str, title: str, group: str = "Proc",
         methods: typing.Iterable[str] | None = None) -> None:
    _PROCS[(type_, name)] = ProcSpec(type_, name, tool, title, group, frozenset(methods) if methods else None)


# ---- lookup ---------------------------------------------------------------------------------------------------
def _page_kind(obj: StoredObject) -> str | None:
    if not obj.page:
        return None
    try:
        return get_context().workspace.get(obj.page).kind
    except Exception:  # noqa: BLE001 - a page deleted under the object: views that need no page still apply
        return None


def views_of(obj: StoredObject) -> list[ViewSpec]:
    kind = _page_kind(obj)
    return [v for v in _VIEWS.values() if v.applies(obj, kind)]


def procs_of(obj: StoredObject) -> list[ProcSpec]:
    return [p for p in _PROCS.values() if p.type in (obj.type, "*") and (p.methods is None or obj.method in p.methods)]


def lookup(obj: StoredObject, name: str | None) -> ViewSpec:
    name = name or DEFAULT_VIEW.get(obj.type, "")
    spec = _VIEWS.get((obj.type, name))
    if spec is None or not spec.applies(obj, _page_kind(obj)):
        available = ", ".join(v.name for v in views_of(obj)) or "none"
        raise ObjectError(f"'{obj.name}' ({obj.type}, {obj.method}) has no view '{name}'. Views: {available}")
    return spec


def ref(obj_name: str, view_name: str, params: dict[str, Any]) -> dict[str, Any]:
    return {"object": obj_name, "view": view_name, "params": params}


# ---- computing ------------------------------------------------------------------------------------------------
def compute(name: str, view_name: str | None = None,
            params: dict[str, Any] | None = None) -> tuple[ViewResult, dict[str, Any]]:
    """A view of the object's current version (cached), and the canonical ``ref`` that names it."""
    objects = get_context().objects
    obj = objects.get(name)
    spec = lookup(obj, view_name)
    values = spec.params(params)
    key = (obj.name, obj.version, spec.name, json.dumps(values, sort_keys=True, default=str))
    result = objects.view_cached(key)
    if result is None:
        result = spec.fn(obj, **values)
        result.data = clean(result.data)
        objects.view_put(key, result)
    return result, ref(obj.name, spec.name, values)


def show(name: str, view_name: str | None = None, params: dict[str, Any] | None = None) -> Block:
    """Put a live view of an object in the report (the existing block, if that exact view is already there)."""
    ctx = get_context()
    result, live_ref = compute(name, view_name, params)
    existing = ctx.panel.live(live_ref)
    if existing is not None:
        return existing
    return ctx.panel.add(Block(kind=result.kind, title=result.title, data=result.data, result_id=live_ref["object"],
                               code=result.code, ref=live_ref))


def add_artifact(kind: str, title: str, data: dict[str, Any], code: str | None = None, *,
                 replace_block_id: str | None = None) -> Block:
    """Put a table or chart the assistant just produced in the report *as an object*: a graph object (gph01...) for
    a Plotly figure, a table object (tab01...) for anything else. The block is a live view of that object, so every
    artifact can be named, locked, renamed, copied and listed like any other object. With ``replace_block_id`` the
    block's object gets a new version (the block follows it); a block that has no object yet just takes the new content.
    """
    from .specs import FrozenGraphSpec, FrozenTableSpec

    ctx = get_context()
    data = clean(data)
    spec: Any = (FrozenGraphSpec(title=title, figure=data) if kind == "plotly_figure"
                 else FrozenTableSpec(title=title, kind=kind, data=data))
    name = None
    if replace_block_id:
        block = ctx.panel.get(replace_block_id)
        name = block.result_id if block.result_id and ctx.objects.exists(block.result_id) else None
        if name is None:
            return ctx.panel.update(replace_block_id, title=title, data=data, code=code)
    obj = ctx.objects.estimate(spec, name)
    block = show(obj.name)
    if code and block.code != code:
        block = ctx.panel.update(block.id, code=code, by_agent=False, checkpoint=False)
    return block


def refresh(panel: PanelState, name: str, *, new_version: bool) -> None:
    """After an object changed: recompute its live blocks; flag its frozen blocks stale when it has a new version."""
    for block in panel.blocks():
        live = block.ref is not None and block.ref.get("object") == name
        if not live:
            if new_version and block.result_id == name and not block.stale:
                panel.update(block.id, by_agent=False, stale=True)
            continue
        try:
            result, _ = compute(name, block.ref["view"], block.ref.get("params"))
        except Exception as exc:  # noqa: BLE001 - the block keeps its last content and says it is out of date
            log.info("Live block %s of %s not refreshed: %s", block.id, name, exc)
            if not block.stale:
                panel.update(block.id, by_agent=False, stale=True)
            continue
        panel.update(block.id, title=result.title, data=result.data, code=result.code, by_agent=False,
                         stale=False)


# ---- views: equations ---------------------------------------------------------------------------------------------
def _has_coefficients(obj: StoredObject) -> bool:
    return bool(obj.summary.get("coefficients"))


def _summary(obj: StoredObject) -> dict[str, Any]:
    return {k: v for k, v in obj.summary.items() if k != "diff"}


@view("equation", "estimation", title="Estimation output", when=_has_coefficients)
def eq_estimation(obj: StoredObject) -> ViewResult:
    s = _summary(obj)
    label = ": ".join(x for x in (s.get("model"), s.get("formula")) if x)
    return ViewResult("regression_table", f"{obj.name} · {label}", {"models": [s]}, llm_view(s), code=obj.code)


@view("equation", "coefplot", title="Coefficient plot", group="Coefficient Diagnostics", requires=("plotly",),
      when=_has_coefficients)
def eq_coefplot(obj: StoredObject, exclude_intercept: bool = True) -> ViewResult:
    import plotly.graph_objects as go

    coefs = [c for c in obj.summary.get("coefficients", [])
             if not (exclude_intercept and c["name"].lower() in {"intercept", "const"})
             and None not in (c.get("coef"), c.get("ci_low"), c.get("ci_high"))]
    fig = go.Figure(go.Scatter(
        x=[c["coef"] for c in coefs], y=[c["name"] for c in coefs], mode="markers",
        error_x={"type": "data", "symmetric": False, "array": [c["ci_high"] - c["coef"] for c in coefs],
                 "arrayminus": [c["coef"] - c["ci_low"] for c in coefs]}))
    fig.add_vline(x=0, line_dash="dash", line_color="gray")
    fig.update_layout(margin={"l": 40, "r": 20, "t": 30, "b": 40})
    return ViewResult("plotly_figure", f"{obj.name} · coefficients (95% CI)", figure_data(fig),
                      {"chart": "coefficients with 95% confidence intervals"}, code=obj.code)


FITTED = ("ls", "wls", "glm", "binary", "quantile", "tsls", "liml", "gmm", "panel", "arima", "ardl")


@view("equation", "residuals", title="Residual graph", group="Residual Diagnostics", methods=FITTED,
      requires=("plotly", "scipy"))
def eq_residuals(obj: StoredObject, kind: Literal["fitted", "qq", "histogram"] = "fitted") -> ViewResult:
    import numpy as np
    import plotly.graph_objects as go
    from scipy import stats

    res = get_context().objects.fitted(obj.name)
    raw = getattr(res, "resid", None)
    if raw is None:
        raw = getattr(res, "resids", None)
    if raw is None:
        raise ObjectError(f"'{obj.name}' has no residuals")
    resid = np.asarray(raw, dtype=float).ravel()
    if kind == "fitted":
        fitted = getattr(res, "fittedvalues", None)
        fitted = np.asarray(fitted if fitted is not None else np.arange(len(resid)), dtype=float).ravel()
        fig = go.Figure(go.Scatter(x=fitted, y=resid, mode="markers", opacity=0.6))
        fig.add_hline(y=0, line_dash="dash")
        fig.update_layout(xaxis_title="Fitted", yaxis_title="Residual")
    elif kind == "qq":
        (osm, osr), (slope, intercept, _) = stats.probplot(resid, dist="norm")
        fig = go.Figure([go.Scatter(x=osm, y=osr, mode="markers", name="Residuals"),
                         go.Scatter(x=osm, y=slope * np.asarray(osm) + intercept, mode="lines", name="Normal")])
        fig.update_layout(xaxis_title="Theoretical quantiles", yaxis_title="Sample quantiles")
    else:
        fig = go.Figure(go.Histogram(x=resid, nbinsx=40))
    fig.update_layout(margin={"l": 40, "r": 20, "t": 30, "b": 40})
    llm = {"residuals": {"n": int(resid.size), "mean": float(resid.mean()), "sd": float(resid.std(ddof=1)),
                         "skew": float(stats.skew(resid)), "kurtosis": float(stats.kurtosis(resid, fisher=False))}}
    return ViewResult("plotly_figure", f"{obj.name} · residuals ({kind})", figure_data(fig), clean(llm),
                      code=obj.code)


@view("equation", "diagnostics", title="Specification tests", group="Residual Diagnostics", methods=("ls", "wls"),
      check=True, requires=("statsmodels",))
def eq_diagnostics(obj: StoredObject) -> ViewResult:
    """Breusch-Pagan, White, Ramsey RESET, Jarque-Bera, Durbin-Watson and variance inflation factors."""
    import numpy as np
    from statsmodels.stats.diagnostic import het_breuschpagan, het_white, linear_reset
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    from statsmodels.stats.stattools import durbin_watson, jarque_bera

    from .. import expr

    res = get_context().objects.fitted(obj.name)
    exog = res.model.exog
    tests: list[dict[str, Any]] = []

    def add(name: str, stat: float, p: float | None, h0: str) -> None:
        tests.append(clean({"test": name, "statistic": float(stat), "p_value": None if p is None else float(p),
                            "null_hypothesis": h0, "reject_5pct": None if p is None else bool(p < 0.05)}))

    bp = het_breuschpagan(res.resid, exog)
    add("Breusch-Pagan", bp[0], bp[1], "Homoskedasticity")
    try:
        wh = het_white(res.resid, exog)
        add("White", wh[0], wh[1], "Homoskedasticity")
    except (ValueError, np.linalg.LinAlgError, AssertionError):
        pass
    try:
        reset = linear_reset(res, power=3, use_f=True)
        add("Ramsey RESET", reset.fvalue, reset.pvalue, "Correct functional form")
    except (ValueError, np.linalg.LinAlgError):
        pass
    jb = jarque_bera(res.resid)
    add("Jarque-Bera", jb[0], jb[1], "Normally distributed residuals")
    add("Durbin-Watson", durbin_watson(res.resid), None, "No first-order autocorrelation (value near 2)")
    term_names = obj.summary.get("term_names") or {}
    names = [expr.pretty(n, term_names) for n in res.model.exog_names]
    vif = [{"variable": names[i], "vif": float(variance_inflation_factor(exog, i))}
           for i in range(exog.shape[1]) if names[i] != "Intercept"]
    data = clean({"tests": tests, "vif": vif, "result_id": obj.name})
    outcome = "warn" if any(t.get("reject_5pct") for t in tests) else "ok"
    code = ("from statsmodels.stats.diagnostic import het_breuschpagan, het_white, linear_reset\n"
            "from statsmodels.stats.stattools import durbin_watson, jarque_bera\n"
            "het_breuschpagan(res.resid, res.model.exog); linear_reset(res, power=3, use_f=True)")
    llm = {"tests": {t["test"]: {"stat": t["statistic"], "p": t["p_value"]} for t in tests},
           "max_vif": max((v["vif"] for v in vif), default=None)}
    return ViewResult("diagnostics", f"{obj.name} · diagnostics · {obj.summary.get('formula')}", data, clean(llm),
                      outcome, code)


def _figure_view(obj: StoredObject, figure_key: str, title_key: str) -> ViewResult:
    extras = get_context().objects.extras(obj.name)
    if figure_key not in extras:
        raise ObjectError(f"'{obj.name}' has no {figure_key.replace('_', ' ')}")
    return ViewResult("plotly_figure", f"{obj.name} · {extras.get(title_key, '')}", extras[figure_key],
                      {"chart": extras.get(title_key, "")}, code=obj.code)


@view("equation", "forecast", title="Forecast graph", group="Forecast", methods=("arima",),
      when=lambda o: bool(o.spec.get("forecast_steps")) and not o.spec.get("exog"))
def arima_forecast(obj: StoredObject) -> ViewResult:
    out = _figure_view(obj, "forecast_figure", "forecast_title")
    out.llm = {"forecast": obj.summary.get("forecast")}
    return out


@view("equation", "volatility", title="Conditional volatility", methods=("garch",), requires=("plotly",))
def garch_volatility(obj: StoredObject) -> ViewResult:
    out = _figure_view(obj, "figure", "title")
    out.llm = {"persistence": obj.summary.get("persistence"),
               "volatility_forecast": obj.summary.get("volatility_forecast")}
    return out


@view("equation", "effects", title="Event-study graph", methods=("event_study",), requires=("plotly",))
def event_study_effects(obj: StoredObject) -> ViewResult:
    out = _figure_view(obj, "figure", "title")
    out.llm = {k: obj.summary.get(k) for k in ("post_average", "pretrend_test")}
    return out


@view("equation", "rd_plot", title="RD plot", methods=("rdd",), requires=("plotly",))
def rdd_plot(obj: StoredObject) -> ViewResult:
    out = _figure_view(obj, "figure", "title")
    out.llm = {k: obj.summary.get(k) for k in ("cutoff", "bandwidth_h", "n_effective")}
    return out


@view("equation", "synth_path", title="Treated vs synthetic", methods=("synth",), requires=("plotly",))
def synth_path(obj: StoredObject) -> ViewResult:
    out = _figure_view(obj, "figure", "title")
    out.llm = {k: obj.summary.get(k) for k in ("att", "pre_rmspe", "placebo")}
    return out


@view("equation", "heterogeneity", title="Effect heterogeneity", methods=("cate",), requires=("plotly",))
def cate_heterogeneity(obj: StoredObject) -> ViewResult:
    out = _figure_view(obj, "figure", "title")
    out.llm = {k: obj.summary.get(k) for k in ("cate_quantiles", "feature_importance", "cate_by_top_feature")}
    return out


@view("equation", "sensitivity", title="Sensitivity to omitted variables", group="Coefficient Diagnostics",
      methods=("ls", "wls"), check=True, requires=("statsmodels",))
def eq_sensitivity(obj: StoredObject, treatment: str | None = None, benchmarks: list[str] | None = None,
                   kd: list[float] | None = None, q: float = 1.0, alpha: float = 0.05) -> ViewResult:
    """Cinelli-Hazlett robustness values and benchmark bounds for the coefficient of ``treatment`` (default: the
    first regressor)."""
    from .causal import sensitivity

    res = get_context().objects.fitted(obj.name)
    if treatment is None:
        treatment = next((n for n in res.model.exog_names if n.lower() not in {"intercept", "const"}), "")
    s = sensitivity(res, treatment, benchmarks or [], kd or [1.0, 2.0, 3.0], q, alpha)
    lines = [f"**{treatment}**: estimate {s['estimate']:.4g} (t = {s['t']:.2f}, df = {s['dof']:.0f})", "",
             "| Statistic | Value |", "|---|---|",
             f"| Partial R² of {treatment} with the outcome | {s['partial_r2_treatment']:.2%} |",
             f"| Robustness value RV (q = {q:g}) | {s['rv_q']:.2%} |",
             f"| Robustness value RV (q = {q:g}, α = {alpha:g}) | {s['rv_qa']:.2%} |",
             "", f"A confounder explaining {s['rv_q']:.1%} of the residual variance of both {treatment} and the "
             f"outcome would remove {q:.0%} of the estimate; one explaining {s['rv_qa']:.1%} would make it "
             f"insignificant at {alpha:.0%}."]
    if s["bounds"]:
        lines += ["", "| Confounder as strong as | R²(D~Z|X) | R²(Y~Z|D,X) | Adjusted estimate | Adjusted t | CI |",
                  "|---|---|---|---|---|---|"]
        def num(v: Any, spec: str) -> str:  # a bound with R²(Y~Z|D,X) = 1 has no standard error
            return "—" if v is None else format(v, spec)

        lines += [f"| {b['k']:g}× {b['benchmark']} | {b['r2dz_x']:.2%} | {b['r2yz_dx']:.2%} | "
                  f"{num(b['adjusted_estimate'], '.4g')} | {num(b['adjusted_t'], '.2f')} | "
                  f"[{num(b['ci_low'], '.3g')}, {num(b['ci_high'], '.3g')}] |" for b in s["bounds"]]
    if s["bounds_undefined"]:
        lines += ["", "Not defined (such a confounder would explain all the residual variance of the treatment): "
                  + ", ".join(f"{b['k']:g}× {b['benchmark']}" for b in s["bounds_undefined"]) + "."]
    fragile = s["rv_qa"] < 0.05 or any(b["ci_low"] is not None and b["ci_low"] <= 0 <= b["ci_high"]
                                       for b in s["bounds"])
    code = (f"# Cinelli & Hazlett (2020), as sensemakr: robustness values of {treatment}\n"
            f"from qtrix_engine.objects.causal import sensitivity\nsensitivity(res, {treatment!r}, {benchmarks!r}, "
            f"{kd or [1.0, 2.0, 3.0]!r})")
    return ViewResult("markdown", f"{obj.name} · sensitivity of {treatment}", {"markdown": "\n".join(lines)}, s,
                      "warn" if fragile else "ok", code)


def _margins_frame(res: Any, kind: str, by: str | None, variables: list[str] | None) -> Any:
    import copy

    import marginaleffects as me
    import pandas as pd

    # marginaleffects refuses string columns: hand it a copy of the model whose data has them as categories.
    frame = res.model.data.frame.copy()
    for c in frame.columns:
        if frame[c].dtype == object or pd.api.types.is_string_dtype(frame[c]):
            frame[c] = frame[c].astype("category")
    model = copy.copy(res.model)
    model.data = copy.copy(res.model.data)
    model.data.frame = frame
    fitted = copy.copy(res)
    fitted.model = model
    fn = {"slopes": me.avg_slopes, "comparisons": me.avg_comparisons, "predictions": me.avg_predictions}[kind]
    kwargs: dict[str, Any] = {}
    if by:
        kwargs["by"] = by
    if variables and kind != "predictions":
        kwargs["variables"] = variables
    out = fn(fitted, **kwargs)
    table = out.to_pandas() if hasattr(out, "to_pandas") else pd.DataFrame(out)
    keep = [c for c in (by, "term", "contrast", "estimate", "std_error", "statistic", "p_value", "conf_low",
                        "conf_high") if c and c in table.columns]
    return table[keep]


@view("equation", "margins", title="Marginal effects", group="Coefficient Diagnostics",
      methods=("ls", "wls", "binary", "glm"), requires=("marginaleffects",))
def eq_margins(obj: StoredObject, kind: Literal["slopes", "comparisons", "predictions"] = "slopes",
               by: str | None = None, variables: list[str] | None = None) -> ViewResult:
    """Average marginal effects (slopes), contrasts or predictions, optionally by group (marginaleffects)."""
    from ..tools._format import df_block_data

    table = _margins_frame(get_context().objects.fitted(obj.name), kind, by, variables)
    term_names = obj.summary.get("term_names") or {}
    if "term" in table.columns and term_names:
        from .. import expr

        table["term"] = [expr.pretty(str(t), term_names) for t in table["term"]]
    label = {"slopes": "average marginal effects", "comparisons": "average contrasts",
             "predictions": "average predictions"}[kind] + (f" by {by}" if by else "")
    rows = clean(table.to_dict(orient="records"))
    code = (f"import marginaleffects as me\nme.avg_{kind}(res" + (f", by={by!r}" if by else "")
            + (f", variables={variables!r}" if variables and kind != "predictions" else "") + ")")
    return ViewResult("dataframe", f"{obj.name} · {label}", df_block_data(table.round(6)), {"margins": rows[:40]},
                      code=code)


@view("equation", "regimes", title="Regime probabilities", methods=("markov",), requires=("plotly",))
def markov_regimes(obj: StoredObject) -> ViewResult:
    out = _figure_view(obj, "figure", "title")
    out.llm = {k: obj.summary.get(k) for k in ("transition_matrix", "expected_duration", "regime_share")}
    return out


# ---- views: var, table, graph, group, text ---------------------------------------------------------------------------
@view("var", "estimation", title="Estimation output")
def var_estimation(obj: StoredObject) -> ViewResult:
    s = _summary(obj)
    lines = [f"**{s.get('formula', obj.name)}** · {s.get('sample', '')} · n = {s.get('n_obs')}"]
    if s.get("aic") is not None:
        lines.append(f"AIC {s['aic']:.4g}")
    if s.get("granger"):
        lines += ["", "| Granger causality | F | p |", "|---|---|---|"]
        lines += [f"| {k} | {v['f_stat']:.3f} | {v['p_value']:.3g} |" for k, v in s["granger"].items()]
    for label in ("beta", "alpha"):
        if s.get(label):
            cols = list(s[label])
            lines += ["", f"| {label} | " + " | ".join(cols) + " |", "|---" * (len(cols) + 1) + "|"]
            rows = list(next(iter(s[label].values())))
            lines += [f"| {r} | " + " | ".join(f"{s[label][c][r]:.4g}" for c in cols) + " |" for r in rows]
    return ViewResult("markdown", f"{obj.name} · {s.get('model', 'VAR')}", {"markdown": "\n".join(lines)},
                      {k: v for k, v in s.items() if k not in ("dataset", "columns")}, code=obj.code)


@view("var", "irf", title="Impulse responses", group="Impulse")
def var_irf(obj: StoredObject) -> ViewResult:
    return _figure_view(obj, "figure", "title")


@view("table", "table", title="Table")
def table_view(obj: StoredObject) -> ViewResult:
    s = obj.summary
    if obj.method == "frozen_table":
        return ViewResult(s["kind"], s["title"], s["data"], {"frozen_from": s.get("source"), "view": s.get("view")})
    return ViewResult("regression_table", s.get("title") or obj.name, {"models": s.get("models", [])},
                      {"models": [m.get("formula") for m in s.get("models", [])], "sources": s.get("sources")})


@view("graph", "graph", title="Graph")
def graph_view(obj: StoredObject) -> ViewResult:
    s = obj.summary
    return ViewResult("plotly_figure", s.get("title") or obj.name, s.get("figure", {}),
                      {"chart": s.get("title"), "source": s.get("source")})


@view("group", "members", title="Members")
def group_view(obj: StoredObject) -> ViewResult:
    members = obj.summary.get("members", [])
    where = f" of page {obj.page}" if obj.page else ""
    return ViewResult("markdown", f"{obj.name} · group",
                      {"markdown": f"Group **{obj.name}**{where}: " + ", ".join(members)}, {"members": members})


@view("text", "text", title="Text")
def text_view(obj: StoredObject) -> ViewResult:
    s = obj.summary
    return ViewResult("markdown", s.get("title") or obj.name, {"markdown": s.get("markdown", "")},
                      {"markdown": s.get("markdown", "")[:2000]})


# ---- procs (registry tools acting on the object) ---------------------------------------------------------------------
for _type in ("equation", "var", "table", "group"):
    proc(_type, "estimate", tool="obj_reestimate", title="Estimate again (new version)",
         methods={"table"} if _type == "table" else None)
proc("*", "freeze", tool="obj_freeze", title="Freeze a view", group="Object")
proc("*", "copy", tool="obj_copy", title="Copy", group="Object")
proc("*", "rename", tool="obj_rename", title="Rename", group="Object")
proc("*", "lock", tool="obj_lock", title="Lock", group="Object")
proc("*", "delete", tool="obj_delete", title="Delete", group="Object")
