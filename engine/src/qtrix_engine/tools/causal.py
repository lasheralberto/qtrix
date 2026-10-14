"""causal_* tools: regression discontinuity (rdrobust), double machine learning (DoubleML) and synthetic control
(pysyncon). Extra ``causal``; the staggered event study is ``panel_event_study`` (pyfixest)."""

from __future__ import annotations

from typing import Any, Literal

from ..context import get_context
from ..objects.specs import DmlSpec, RddSpec, SynthSpec
from ..objects.views import add_artifact, compute, show
from ._format import clean
from .ols import publish, resolved_sample
from .registry import Qx, tool

PANEL = ("panel", "panel_undated")


@tool("causal", requires=("rdrobust",), mutating=True, qx=Qx("qx", "equation", ("name",), family="equation"))
def causal_rdd(dataset_id: str, outcome: str, running: str, cutoff: float = 0.0, treatment: str | None = None,
               covariates: list[str] | None = None,
               kernel: Literal["triangular", "epanechnikov", "uniform"] = "triangular", p: int = 1,
               bwselect: Literal["mserd", "msetwo", "cerrd", "certwo"] = "mserd", h: float | None = None,
               cluster: str | None = None, sample: str | None = None, name: str | None = None) -> dict[str, Any]:
    """Regression discontinuity with rdrobust (Calonico-Cattaneo-Titiunik): local polynomial at the cutoff of a
    running variable, data-driven bandwidth and robust bias-corrected inference. Report the "Robust" row; the
    "Conventional" point estimate with robust confidence interval is the usual presentation. The RD plot (binned
    means and the local fits) goes to the panel. Check robustness with half and double the bandwidth (`h`).
    Stored as a named equation object; `name` re-estimates an existing one as a new version.

    Args:
        dataset_id: Dataset id or name.
        outcome: Outcome column.
        running: Running (assignment) variable.
        cutoff: Threshold of the running variable; treatment starts at running >= cutoff.
        treatment: Fuzzy design only: the treatment actually received (0/1); the estimate is then a local Wald
            (LATE) ratio.
        covariates: Pre-determined covariates to add for precision (they must not jump at the cutoff).
        kernel: triangular, epanechnikov or uniform.
        p: Order of the local polynomial (1 = local linear, the default to keep).
        bwselect: mserd (one MSE-optimal bandwidth), msetwo (one per side), cerrd/certwo (coverage-error optimal).
        h: Fixed bandwidth instead of the data-driven one (for robustness checks).
        cluster: Cluster column for the standard errors.
        sample: Sample for this call only; omit for the dataset's active sample.
        name: Object name (e.g. eq_rdd). An existing name is re-estimated as a new version.
    """
    spec = RddSpec(page=get_context().workspace.get(dataset_id).name, sample=resolved_sample(dataset_id, sample),
                   outcome=outcome, running=running, cutoff=cutoff, treatment=treatment, covariates=covariates or [],
                   kernel=kernel, p=p, bwselect=bwselect, h=h, cluster=cluster)
    obj, out = publish(spec, name, title=f"RDD · {outcome}")
    show(obj.name, "rd_plot")
    for key in ("bandwidth_h", "n_effective"):
        out[key] = obj.summary.get(key)
    return clean(out)


@tool("causal", requires=("doubleml", "sklearn"), mutating=True,
      qx=Qx("qx", "equation", ("name",), family="equation"))
def causal_dml(dataset_id: str, outcome: str, treatment: str, controls: list[str],
               model: Literal["plr", "irm"] = "plr", learner: Literal["forest", "lasso", "boosting"] = "forest",
               n_folds: int = 5, seed: int = 42, sample: str | None = None, name: str | None = None) -> dict[str, Any]:
    """Double/debiased machine learning (Chernozhukov et al. 2018, DoubleML): the effect of a treatment when
    selection is on many observed controls whose influence may be nonlinear. Machine-learning models of the
    outcome and the treatment are fitted with cross-fitting and combined in an orthogonal score, so the estimate
    has valid standard errors. Identification is still selection on observables: no unobserved confounders.

    Args:
        dataset_id: Dataset id or name.
        outcome: Outcome column.
        treatment: Treatment column (continuous or 0/1).
        controls: Confounders to adjust for (categorical columns are dummy-encoded).
        model: plr (partially linear: one constant effect, any treatment) or irm (interactive: ATE of a 0/1
            treatment with fully heterogeneous effects).
        learner: forest (random forest), lasso or boosting (gradient boosting) for the nuisance functions.
        n_folds: Cross-fitting folds.
        seed: Seed of the sample splits and learners (the same seed reproduces the estimate).
        sample: Sample for this call only; omit for the dataset's active sample.
        name: Object name (e.g. eq_dml). An existing name is re-estimated as a new version.
    """
    spec = DmlSpec(page=get_context().workspace.get(dataset_id).name, sample=resolved_sample(dataset_id, sample),
                   outcome=outcome, treatment=treatment, controls=controls, model=model, learner=learner,
                   n_folds=n_folds, seed=seed)
    obj, out = publish(spec, name, title=f"Double ML · {outcome}")
    out["nuisance_rmse"] = obj.summary.get("nuisance_rmse")
    return clean(out)


@tool("causal", requires=("rddensity",), qx=Qx("page", "rd_density", ("running",)))
def causal_rd_density(dataset_id: str, running: str, cutoff: float = 0.0,
                      sample: str | None = None) -> dict[str, Any]:
    """Manipulation test of a regression discontinuity (Cattaneo-Jansson-Ma, rddensity): is the density of the
    running variable continuous at the cutoff? A significant jump means units may sort around the threshold
    (e.g. bunching just above an eligibility score), which undermines the RD design. Run it before trusting
    causal_rdd.

    Args:
        dataset_id: Dataset id or name.
        running: Running (assignment) variable.
        cutoff: The RD threshold.
        sample: Sample for this call only; omit for the dataset's active sample.
    """
    import warnings

    from rddensity import rddensity


    ws = get_context().workspace
    x = ws.df(dataset_id, [running], sample=sample)[running].dropna().astype(float)
    if not ((x < cutoff).any() and (x >= cutoff).any()):
        raise ValueError(f"'{running}' has observations on one side of the cutoff {cutoff:g} only")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = rddensity(X=x.to_numpy(), c=cutoff)
    stat, p = float(res.test["t_jk"]), float(res.test["p_jk"])
    out = clean({"running": running, "cutoff": cutoff, "statistic": stat, "p_value": p,
                 "bandwidth": {"left": float(res.h["left"]), "right": float(res.h["right"])},
                 "n_effective": {"left": int(res.n["eff_left"]), "right": int(res.n["eff_right"])},
                 "density_left": float(res.hat["left"]), "density_right": float(res.hat["right"]),
                 "null_hypothesis": "The density of the running variable is continuous at the cutoff",
                 "conclusion": "Evidence of manipulation at the cutoff" if p < 0.05
                 else "No evidence of manipulation at the cutoff"})
    add_artifact("diagnostics", f"RD manipulation test · {running} at {cutoff:g}",
                 {"tests": [{"test": "Cattaneo-Jansson-Ma density test", "statistic": stat, "p_value": p,
                             "null_hypothesis": out["null_hypothesis"], "reject_5pct": p < 0.05}]})
    return out


@tool("causal", requires=("econml",), mutating=True, qx=Qx("qx", "equation", ("name",), family="equation"))
def causal_forest(dataset_id: str, outcome: str, treatment: str, features: list[str],
                  controls: list[str] | None = None, n_estimators: int = 400, seed: int = 42,
                  sample: str | None = None, name: str | None = None) -> dict[str, Any]:
    """Heterogeneous treatment effects with a causal forest (EconML CausalForestDML): who benefits more? Estimates
    the conditional effect for every observation (CATE) as a function of `features`, the average effect with
    valid inference, which features drive the heterogeneity and the CATE distribution (chart in the panel).
    Identification is selection on observables: `features` and `controls` must contain the confounders.

    Args:
        dataset_id: Dataset id or name.
        outcome: Outcome column.
        treatment: Treatment column (0/1 or continuous).
        features: Variables along which the effect may vary (also used as confounders).
        controls: Further confounders that are not effect modifiers.
        n_estimators: Trees in the forest.
        seed: Seed of the forest and the cross-fitting (the same seed reproduces the estimate).
        sample: Sample for this call only; omit for the dataset's active sample.
        name: Object name (e.g. eq_cate). An existing name is re-estimated as a new version.
    """
    from ..objects.specs import CateSpec

    spec = CateSpec(page=get_context().workspace.get(dataset_id).name, sample=resolved_sample(dataset_id, sample),
                    outcome=outcome, treatment=treatment, features=features, controls=controls or [],
                    n_estimators=n_estimators, seed=seed)
    obj, out = publish(spec, name, title=f"Causal forest · {outcome}")
    show(obj.name, "heterogeneity")
    for key in ("cate_quantiles", "feature_importance", "cate_by_top_feature", "share_positive"):
        out[key] = obj.summary.get(key)
    return clean(out)


@tool("causal", requires=("sklearn", "statsmodels"), mutating=True,
      qx=Qx("qx", "equation", ("name",), family="equation"))
def causal_pds(dataset_id: str, outcome: str, treatment: str, controls: list[str],
               cov_type: Literal["nonrobust", "HC1", "cluster"] = "HC1", cluster: str | None = None, seed: int = 42,
               sample: str | None = None, name: str | None = None) -> dict[str, Any]:
    """Post-double-selection lasso (Belloni, Chernozhukov & Hansen 2014): the effect of a treatment when there are
    many candidate controls. A cross-validated lasso picks the controls that predict the outcome and those that
    predict the treatment; OLS of the outcome on the treatment and the union of both sets gives an estimate with
    valid standard errors (selecting on the outcome alone would drop confounders). Stored as a named equation.

    Args:
        dataset_id: Dataset id or name.
        outcome: Outcome column.
        treatment: Treatment column.
        controls: Candidate controls (categorical columns are dummy-encoded).
        cov_type: nonrobust, HC1 or cluster.
        cluster: Cluster column when cov_type="cluster".
        seed: Seed of the cross-validation folds.
        sample: Sample for this call only; omit for the dataset's active sample.
        name: Object name (e.g. eq_pds). An existing name is re-estimated as a new version.
    """
    from ..objects.specs import PdsSpec

    spec = PdsSpec(page=get_context().workspace.get(dataset_id).name, sample=resolved_sample(dataset_id, sample),
                   outcome=outcome, treatment=treatment, controls=controls, cov_type=cov_type, cluster=cluster,
                   seed=seed)
    obj, out = publish(spec, name, title=f"PDS lasso · {outcome}")
    out["selected_controls"] = obj.summary.get("selected_controls")
    out["n_candidate_controls"] = obj.summary.get("n_candidate_controls")
    return clean(out)


@tool("causal", requires=("statsmodels",), qx=Qx("obj", "sensitivity", ("treatment",)))
def causal_sensitivity(result_id: str, treatment: str, benchmarks: list[str] | None = None,
                       kd: list[float] | None = None, q: float = 1.0, alpha: float = 0.05) -> dict[str, Any]:
    """Sensitivity of an OLS estimate to omitted-variable bias (Cinelli & Hazlett 2020, the sensemakr method).
    Reports the robustness value RV (how much residual variance of both treatment and outcome a confounder must
    explain to remove the estimate) and RV_α (to make it insignificant), and the estimate adjusted for a
    confounder 1, 2, 3 times as strong as each observed benchmark covariate. The table goes to the panel and
    refreshes when the equation is re-estimated.

    Args:
        result_id: A fitted OLS/WLS equation (ols_fit).
        treatment: The term whose robustness is assessed, as named in the model.
        benchmarks: Observed covariates of the model to calibrate plausible confounders (e.g. the strongest
            control); omit for the robustness values only.
        kd: Multiples of the benchmarks' strength (default 1, 2, 3).
        q: Share of the estimate the confounder must remove (1 = all of it).
        alpha: Significance level of RV_α.
    """
    params = {"treatment": treatment, "benchmarks": benchmarks, "kd": kd, "q": q, "alpha": alpha}
    result, _ = compute(result_id, "sensitivity", params)
    return clean({"block_id": show(result_id, "sensitivity", params).id, **(result.llm or {})})


@tool("causal", requires=("pysyncon",), kinds=PANEL, mutating=True,
      qx=Qx("qx", "equation", ("name",), family="equation"))
def causal_synth(dataset_id: str, outcome: str, treated: str, treatment_period: str, donors: list[str] | None = None,
                 predictors: list[str] | None = None, placebo: bool = True, sample: str | None = None,
                 name: str | None = None) -> dict[str, Any]:
    """Synthetic control (Abadie et al.): one treated entity (a region, country, firm) compared with a weighted
    combination of untreated donor entities that reproduces its pre-treatment path. Reports the donor weights,
    the post-treatment gap (ATT), the pre-treatment fit (RMSPE) and an in-space placebo p-value (the rank of the
    treated entity's post/pre RMSPE ratio among placebo runs on every donor). The treated-vs-synthetic chart goes
    to the panel. Needs a declared panel. Stored as a named equation object.

    Args:
        dataset_id: Dataset id or name (a panel).
        outcome: Outcome column.
        treated: The treated entity (its value in the entity column).
        treatment_period: First treated period, in the units of the time column (e.g. "2003").
        donors: Donor entities; omit for every other entity observed in all periods (exclude any that were also
            treated or affected by spillovers).
        predictors: Extra covariates matched on their pre-treatment means (the outcome path is always matched).
        placebo: Run the placebo inference (one extra fit per donor).
        sample: Sample for this call only; omit for the dataset's active sample.
        name: Object name (e.g. eq_synth). An existing name is re-estimated as a new version.
    """
    spec = SynthSpec(page=get_context().workspace.get(dataset_id).name, sample=resolved_sample(dataset_id, sample),
                     outcome=outcome, treated=str(treated), treatment_period=str(treatment_period), donors=donors,
                     predictors=predictors or [], placebo=placebo)
    obj, out = publish(spec, name, title=f"Synthetic control · {treated}")
    show(obj.name, "synth_path")
    for key in ("att", "weights", "pre_rmspe", "post_rmspe", "placebo", "gaps"):
        out[key] = obj.summary.get(key)
    return clean(out)
