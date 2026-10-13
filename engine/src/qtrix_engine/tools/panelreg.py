"""panel_* and iv_* tools: panel-data and instrumental-variable estimators (linearmodels)."""

from __future__ import annotations

from typing import Any, Literal

from ..context import get_context
from ..objects.specs import EventStudySpec, GmmSpec, LimlSpec, PanelSpec, TslsSpec
from ..objects.views import show
from ._format import clean
from .ols import publish, publish_object, resolved_sample
from .registry import Qx, tool

PanelEstimator = Literal["fe", "twfe", "re", "between", "fd", "pooled"]
PanelCov = Literal["unadjusted", "robust", "cluster_entity", "cluster_both", "kernel"]


PANEL = ("panel", "panel_undated")


@tool("panel", requires=("linearmodels",), kinds=PANEL, mutating=True,
      qx=Qx("qx", "equation", ("name",), family="equation"))
def panel_fit(dataset_id: str, formula: str, estimator: PanelEstimator = "fe",
              cov_type: PanelCov = "cluster_entity", entity: str | None = None,
              time: str | None = None, sample: str | None = None, name: str | None = None) -> dict[str, Any]:
    """Estimate a linear panel-data model on the dataset's active sample (or `sample`).
    Stored as a named equation object; `name` re-estimates an existing one as a new version.

    Args:
        dataset_id: Dataset id or name.
        formula: Formula WITHOUT effect terms, e.g. "lwage ~ 1 + exper + union + married".
        estimator: fe (entity fixed effects), twfe (entity + time effects), re (random effects),
            between, fd (first differences) or pooled OLS.
        cov_type: unadjusted, robust, cluster_entity, cluster_both or kernel (Driscoll-Kraay).
        entity: Entity column (defaults to the declared panel structure).
        time: Time column (defaults to the declared panel structure).
        sample: Sample for this call only, e.g. "2002 2005 if educ > 10", "@all"; omit for the active one.
        name: Object name (e.g. eq_fe). An existing name is re-estimated as a new version; omit for eq01...
    """
    page = get_context().workspace.get(dataset_id).name
    spec = PanelSpec(page=page, formula=formula, sample=resolved_sample(dataset_id, sample), estimator=estimator,
                     cov_type=cov_type, entity=entity, time=time)
    label = {"fe": "Panel FE", "twfe": "Panel TWFE", "re": "Random effects", "between": "Between",
             "fd": "First differences", "pooled": "Pooled OLS"}[estimator]
    return publish_object(spec, name, title=f"{label}: {formula}")


@tool("panel", requires=("pyfixest",), kinds=PANEL, mutating=True,
      qx=Qx("qx", "equation", ("name",), family="equation"))
def panel_event_study(dataset_id: str, outcome: str, cohort: str,
                      estimator: Literal["did2s", "cs", "saturated", "lpdid", "twfe"] = "did2s", pre: int = 4,
                      post: int = 4, cluster: str | None = None,
                      control_group: Literal["never", "notyet"] = "never", sample: str | None = None,
                      name: str | None = None) -> dict[str, Any]:
    """Event study / difference-in-differences with staggered adoption: the effect in each period before and
    after treatment, relative to the period just before (t-1), plus the overall ATT and a joint pre-trends test.

    With staggered timing and effects that change over time, plain TWFE compares late adopters with early ones
    already treated and is biased (Goodman-Bacon 2021): the default did2s (Gardner two-stage imputation),
    Callaway-Sant'Anna (cs: doubly robust group-time ATTs, never- or not-yet-treated controls), the Sun-Abraham
    saturated estimator and local-projections DiD are robust to it; twfe is kept as a benchmark. Comparing two of
    them is a good robustness check.
    Needs a declared panel. Periods outside [-pre, post] are binned into the endpoints. The event-study chart
    goes to the panel. Stored as a named equation object; `name` re-estimates an existing one as a new version.

    Args:
        dataset_id: Dataset id or name (a panel).
        outcome: Outcome column.
        cohort: Column with each entity's first treated period, in the units of the time column (e.g. 2006 on
            a yearly panel); 0, missing or after the last period = never treated.
        estimator: did2s, cs (Callaway-Sant'Anna, needs the causal extra), saturated (Sun-Abraham), lpdid or twfe.
        pre: Leads shown (at least 2; t-1 is the reference).
        post: Lags shown (0 = the treatment period).
        cluster: Cluster column for the standard errors; defaults to the panel entity.
        control_group: cs only: never (never-treated entities) or notyet (also those not yet treated).
        sample: Sample for this call only; omit for the dataset's active sample.
        name: Object name (e.g. eq_es). An existing name is re-estimated as a new version.
    """
    spec = EventStudySpec(page=get_context().workspace.get(dataset_id).name, sample=resolved_sample(dataset_id, sample),
                          outcome=outcome, cohort=cohort, estimator=estimator, pre=pre, post=post, cluster=cluster,
                          control_group=control_group)
    obj, out = publish(spec, name, title=f"Event study · {outcome}")
    show(obj.name, "effects")
    for key in ("post_average", "pretrend_test", "treated_entities", "cohorts"):
        out[key] = obj.summary.get(key)
    return clean(out)


@tool("panel", requires=("linearmodels",), kinds=PANEL,
      qx=Qx("qx", "hausman", ("fe_result_id", "re_result_id"), rename={"fe_result_id": "fe", "re_result_id": "re"}))
def panel_hausman(fe_result_id: str, re_result_id: str) -> dict[str, Any]:
    """Hausman test comparing a fixed-effects and a random-effects fit (H0: RE is consistent)."""
    import numpy as np
    from scipy import stats

    ctx = get_context()
    fe, re = ctx.results.get(fe_result_id).obj, ctx.results.get(re_result_id).obj
    common = [c for c in fe.params.index if c in re.params.index and c.lower() not in {"const", "intercept"}]
    diff = (fe.params[common] - re.params[common]).to_numpy()
    cov = (fe.cov.loc[common, common] - re.cov.loc[common, common]).to_numpy()
    stat = float(diff @ np.linalg.pinv(cov) @ diff)
    p = float(stats.chi2.sf(stat, len(common)))
    return clean({"statistic": stat, "df": len(common), "p_value": p,
                  "conclusion": "Reject RE, prefer FE" if p < 0.05 else "Cannot reject RE"})


@tool("iv", requires=("linearmodels",), mutating=True, qx=Qx("qx", "equation", ("name",), family="equation"))
def iv_fit(dataset_id: str, formula: str, method: Literal["2sls", "liml", "gmm"] = "2sls",
           cov_type: Literal["unadjusted", "robust", "clustered", "kernel"] = "robust",
           cluster: str | None = None, sample: str | None = None, name: str | None = None) -> dict[str, Any]:
    """Instrumental-variables regression with first-stage and over-identification diagnostics,
    on the dataset's active sample (or `sample`). Stored as a named equation object (method tsls, liml or gmm).

    Args:
        dataset_id: Dataset id or name.
        formula: linearmodels IV formula with the endogenous block in brackets,
            e.g. "lwage ~ 1 + exper + expersq + [educ ~ motheduc + fatheduc]".
        method: 2sls, liml or gmm.
        cov_type: unadjusted, robust, clustered (requires cluster) or kernel.
        cluster: Cluster column when cov_type="clustered".
        sample: Sample for this call only, e.g. "if age >= 25", "@all"; omit for the active one.
        name: Object name (e.g. eq_iv). An existing name is re-estimated as a new version; omit for eq01...
    """
    if cov_type == "clustered" and not cluster:
        raise ValueError("cov_type='clustered' requires the 'cluster' column")
    page = get_context().workspace.get(dataset_id).name
    cls = {"2sls": TslsSpec, "liml": LimlSpec, "gmm": GmmSpec}[method]
    spec = cls(page=page, formula=formula, sample=resolved_sample(dataset_id, sample), cov_type=cov_type,
               cluster=cluster)
    return publish_object(spec, name, title=f"IV {method.upper()}: {formula}")
