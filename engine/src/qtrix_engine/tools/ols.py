"""ols_* tools: cross-section regression with statsmodels and specification tests."""

from __future__ import annotations

from typing import Any, Literal

from ..context import get_context
from ..objects.estimators import model_summary  # noqa: F401 - re-exported for older callers
from ..objects.specs import (
    BinarySpec,
    CountSpec,
    LsSpec,
    MultinomialSpec,
    OrderedSpec,
    QuantileSpec,
    TableSpec,
    WlsSpec,
)
from ..objects.views import show
from ._format import llm_view
from .registry import Qx, tool

CovType = Literal["nonrobust", "HC0", "HC1", "HC2", "HC3", "cluster", "HAC"]


def sample_text(dataset_id: str, sample: str | None) -> str:
    """Readable description of the sample a tool used (active sample unless overridden)."""
    return get_context().workspace.resolve_sample(dataset_id, sample).text


def resolved_sample(dataset_id: str, sample: str | None) -> str:
    """Sample spec to store in an object: the override, else the page's active sample (so rebuilds are exact)."""
    return sample if sample is not None else (get_context().workspace.get(dataset_id).sample or "@all")


def publish_object(spec: Any, name: str | None, *, title: str) -> dict[str, Any]:
    """Estimate and store an object, add its table to the panel and return the compact LLM view."""
    return publish(spec, name, title=title)[1]


def publish(spec: Any, name: str | None, *, title: str) -> tuple[Any, dict[str, Any]]:
    """Like publish_object but also returns the stored object (with its transient ``extras``).

    The report gets a live view of the estimation output; on a re-estimate the live blocks already showing the
    object were refreshed in place, so no duplicate is added.
    """
    ctx = get_context()
    obj = ctx.objects.estimate(spec, name)
    block = show(obj.name, "estimation")
    out = {"result_id": obj.name, "name": obj.name, "version": obj.version, "block_id": block.id,
           **llm_view(obj.summary)}
    if "diff" in obj.summary:
        out["diff_vs_previous"] = {k: v["delta"] for k, v in obj.summary["diff"]["coefficients"].items()
                                   if "delta" in v}
    return obj, out


@tool("ols", requires=("statsmodels",), mutating=True, qx=Qx("qx", "equation", ("name",), family="equation"))
def ols_fit(dataset_id: str, formula: str, cov_type: CovType = "HC1", cluster: str | None = None,
            weights: str | None = None, hac_lags: int = 4, sample: str | None = None,
            name: str | None = None) -> dict[str, Any]:
    """Estimate a linear regression (OLS, or WLS when weights are given) with an R-style formula.

    The result is a named equation object (result_id = its name) that persists across restarts; the full
    regression table is added to the analysis panel and a compact summary is returned. It uses the
    dataset's active sample (see data_set_sample) unless `sample` overrides it. Passing the name of an
    existing equation re-estimates it as a new version and returns the coefficient changes.

    Time-series and panel datasets also accept time operators in the formula: x(-1) lag, d(x) difference,
    dlog(x), pch(x), L(x, 2), movav(x, 4), @trend, @seas(q), dummy("2008q3"), recode(x > 3, 1, 0). Lags work
    within each entity of a panel and may look before the sample starts.

    Args:
        dataset_id: Dataset id or name.
        formula: R/patsy formula, e.g. "lwage ~ educ + exper + I(exper**2) + C(region)" or, with dates,
            "dlog(gdp) ~ dlog(gdp(-1)) + d(r) + @seas(1)".
        cov_type: Standard errors: nonrobust, HC0-HC3 (heteroskedasticity-robust), cluster or HAC (Newey-West).
        cluster: Column with cluster ids when cov_type="cluster".
        weights: Column with analytic weights (WLS).
        hac_lags: Max lags for HAC standard errors.
        sample: Sample for this call only, e.g. "1990q1 2008q4", "if age >= 25", "@all"; omit for the active one.
        name: Object name (e.g. eq_wage). An existing name is re-estimated as a new version; omit for eq01, eq02...
    """
    if cov_type == "cluster" and not cluster:
        raise ValueError("cov_type='cluster' requires the 'cluster' column")
    page = get_context().workspace.get(dataset_id).name
    common = {"page": page, "formula": formula, "sample": resolved_sample(dataset_id, sample),
              "cov_type": cov_type, "cluster": cluster, "hac_lags": hac_lags}
    spec = WlsSpec(**common, weights=weights) if weights else LsSpec(**common)
    return publish_object(spec, name, title=f"{'WLS' if weights else 'OLS'}: {formula}")


@tool("ols", requires=("statsmodels",), mutating=True, qx=Qx("qx", "equation", ("name",), family="equation"))
def ols_binary(dataset_id: str, formula: str, link: Literal["logit", "probit"] = "logit",
               marginal_effects: bool = True, sample: str | None = None,
               name: str | None = None) -> dict[str, Any]:
    """Estimate a binary-choice model (logit or probit), optionally with average marginal effects.
    Uses the active sample unless `sample` (e.g. "if age >= 25", "@all") overrides it. Stored as a named
    equation object like ols_fit; `name` re-estimates an existing one as a new version."""
    page = get_context().workspace.get(dataset_id).name
    spec = BinarySpec(page=page, formula=formula, sample=resolved_sample(dataset_id, sample), link=link,
                      marginal_effects=marginal_effects)
    return publish_object(spec, name, title=f"{link.title()}: {formula}")


@tool("ols", requires=("statsmodels",), mutating=True, qx=Qx("qx", "equation", ("name",), family="equation"))
def ols_count(dataset_id: str, formula: str, model: Literal["poisson", "negbin", "zip", "zinb"] = "poisson",
              cov_type: Literal["nonrobust", "HC1", "cluster"] = "HC1", cluster: str | None = None,
              sample: str | None = None, name: str | None = None) -> dict[str, Any]:
    """Estimate a count-data model for a non-negative integer outcome (visits, patents, trade flows).

    Poisson with robust standard errors is the pseudo-Poisson (PPML) estimator, which is also the right choice for
    a positive continuous outcome in a gravity model. Choose negbin when the variance clearly exceeds the mean,
    and zip/zinb when there are more zeros than the count distribution allows (the zero-inflation part is a
    constant). Poisson and negbin results report incidence-rate ratios exp(b). Stored as a named equation object
    like ols_fit; `name` re-estimates an existing one as a new version.

    Args:
        dataset_id: Dataset id or name.
        formula: R/patsy formula, e.g. "visits ~ age + income + C(region)".
        model: poisson, negbin (negative binomial), zip (zero-inflated Poisson) or zinb.
        cov_type: nonrobust, HC1 (robust) or cluster.
        cluster: Column with cluster ids when cov_type="cluster".
        sample: Sample for this call only; omit for the dataset's active sample.
        name: Object name; an existing name is re-estimated as a new version.
    """
    spec = CountSpec(page=get_context().workspace.get(dataset_id).name, formula=formula,
                     sample=resolved_sample(dataset_id, sample), model=model, cov_type=cov_type, cluster=cluster)
    return publish_object(spec, name, title=f"{model.upper()}: {formula}")


@tool("ols", requires=("statsmodels",), mutating=True, qx=Qx("qx", "equation", ("name",), family="equation"))
def ols_ordered(dataset_id: str, formula: str, link: Literal["logit", "probit"] = "logit",
                sample: str | None = None, name: str | None = None) -> dict[str, Any]:
    """Estimate an ordered logit/probit for an outcome with ordered categories (ratings, education level).

    The outcome must be numeric with at least three distinct ordered values. There is no constant: the estimated
    thresholds ("cut a/b") separate neighbouring categories. Stored as a named equation object like ols_fit;
    `name` re-estimates an existing one as a new version.

    Args:
        dataset_id: Dataset id or name.
        formula: R/patsy formula with the ordered outcome on the left, e.g. "rating ~ price + C(brand)".
        link: logit or probit.
        sample: Sample for this call only; omit for the dataset's active sample.
        name: Object name; an existing name is re-estimated as a new version.
    """
    spec = OrderedSpec(page=get_context().workspace.get(dataset_id).name, formula=formula,
                       sample=resolved_sample(dataset_id, sample), link=link)
    return publish_object(spec, name, title=f"Ordered {link}: {formula}")


@tool("ols", requires=("statsmodels",), mutating=True, qx=Qx("qx", "equation", ("name",), family="equation"))
def ols_multinomial(dataset_id: str, formula: str, sample: str | None = None,
                    name: str | None = None) -> dict[str, Any]:
    """Estimate a multinomial logit for an unordered categorical outcome coded with numbers (choice of
    transport mode, occupation). Coefficients are relative to the lowest category, reported as "x [category]".
    Stored as a named equation object like ols_fit; `name` re-estimates an existing one as a new version.

    Args:
        dataset_id: Dataset id or name.
        formula: R/patsy formula with the numerically coded outcome on the left, e.g. "mode ~ cost + time".
        sample: Sample for this call only; omit for the dataset's active sample.
        name: Object name; an existing name is re-estimated as a new version.
    """
    spec = MultinomialSpec(page=get_context().workspace.get(dataset_id).name, formula=formula,
                           sample=resolved_sample(dataset_id, sample))
    return publish_object(spec, name, title=f"Multinomial logit: {formula}")


@tool("ols", requires=("statsmodels",), mutating=True, qx=Qx("qx", "equation", ("name",), family="equation"))
def ols_quantile(dataset_id: str, formula: str, quantile: float = 0.5, vcov: Literal["robust", "iid"] = "robust",
                 sample: str | None = None, name: str | None = None) -> dict[str, Any]:
    """Estimate a quantile regression: the effect of the regressors on one quantile of the outcome (0.5 is the
    median; 0.1 and 0.9 show the tails, e.g. a wage gap that differs along the distribution). Fit it for several
    quantiles under different names and put them side by side with ols_compare. Stored as a named equation object
    like ols_fit; `name` re-estimates an existing one as a new version.

    Args:
        dataset_id: Dataset id or name.
        formula: R/patsy formula, e.g. "lwage ~ educ + exper".
        quantile: Quantile strictly between 0 and 1.
        vcov: robust (heteroskedasticity-robust) or iid standard errors.
        sample: Sample for this call only; omit for the dataset's active sample.
        name: Object name; an existing name is re-estimated as a new version.
    """
    spec = QuantileSpec(page=get_context().workspace.get(dataset_id).name, formula=formula,
                        sample=resolved_sample(dataset_id, sample), quantile=quantile, vcov=vcov)
    return publish_object(spec, name, title=f"Quantile {quantile:g}: {formula}")


@tool("ols", requires=("statsmodels",), qx=Qx("obj", "diagnostics"))
def ols_diagnostics(result_id: str) -> dict[str, Any]:
    """Specification tests for a fitted OLS/WLS model: Breusch-Pagan, White, Ramsey RESET,
    Jarque-Bera, Durbin-Watson and variance inflation factors. Results are shown in the panel
    (the equation's "diagnostics" view: it refreshes when the equation is re-estimated)."""
    block = show(result_id, "diagnostics")
    return {"block_id": block.id, **block.data}


@tool("ols", requires=("statsmodels",), mutating=True,
      qx=Qx("qx", "table", ("name", "result_ids"), rename={"result_ids": "sources"}))
def ols_compare(result_ids: list[str], title: str = "Model comparison", name: str | None = None) -> dict[str, Any]:
    """Put several fitted models (any estimator) side by side in one regression table, stargazer-style.
    The comparison is a named table object that goes stale when any of the models is re-estimated; passing
    its name again refreshes it as a new version."""
    ctx = get_context()
    obj = ctx.objects.estimate(TableSpec(sources=[ctx.objects.get(r).name for r in result_ids], title=title), name)
    block = show(obj.name, "table")
    return {"result_id": obj.name, "name": obj.name, "version": obj.version, "block_id": block.id,
            "models": [m.get("formula") for m in obj.summary["models"]]}
