"""The catalog of analysis methods the user sees before talking to the agent.

The tool registry groups capabilities by code namespace; an analyst thinks in areas and methods (fixed effects, unit
roots, propensity score matching). This module is that second view: area -> method -> variants, each method pointing
at the registry tools (or the causal library run in a Python cell) that implement it. Availability and the data
structures a method needs are derived from the registry, so the catalog cannot claim what the engine cannot run.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib.util import find_spec
from typing import Any

from qtrix_engine.tools.registry import get_tool


@dataclass(frozen=True)
class Method:
    id: str
    label: str
    summary: str
    tools: tuple[str, ...]
    prompt: str  # an example request; «…» marks what the user fills in
    variants: tuple[str, ...] = ()
    library: tuple[str, ...] = ()  # functions of contrib.econometrics.econometric_algorithm, run via py_run
    structures: tuple[str, ...] | None = None  # overrides the structures derived from the tools' kinds


@dataclass(frozen=True)
class Area:
    id: str
    label: str
    summary: str
    methods: tuple[Method, ...]


PANEL = ("panel", "panel_undated")

AREAS: tuple[Area, ...] = (
    Area("data", "Data and descriptives", "Import, explore, transform and structure the data", (
        Method("import", "Import data", "CSV, Excel or Parquet as a notebook page", ("data_import_file",),
               "Import the file «path» and describe its structure"),
        Method("structure", "Data structure", "Declare cross-section, time series or panel",
               ("data_set_structure", "data_set_panel_structure", "data_describe_structure"),
               "Declare «dataset» as a panel with «id» as the individual and «year» as the period",
               ("Cross-section", "Time series", "Panel")),
        Method("descriptives", "Descriptive statistics", "Mean, dispersion, quantiles and missing values",
               ("data_summary_stats", "data_preview"), "Describe the variables in «dataset»"),
        Method("correlation", "Correlations", "Correlation matrix between variables", ("data_correlation",),
               "Show the correlation matrix of «variables»"),
        Method("transform", "Transformations and new series", "Logs, differences, lags, dummies",
               ("data_genr", "data_transform"), "Generate log(«variable») and its first difference"),
        Method("sample", "Active sample", "Restrict the analysis to a subset of observations",
               ("data_set_sample",), "Restrict the sample to «condition»"),
        Method("sql", "SQL queries", "Filter, aggregate and join datasets with DuckDB", ("data_sql",),
               "Aggregate «dataset» by «group» with the mean of «variable»"),
    )),
    Area("cross_section", "Cross-section", "Linear regression, binary response and specification tests", (
        Method("ols", "Least squares (OLS)", "Linear regression with robust, cluster or HAC errors", ("ols_fit",),
               "Estimate «y» on «x1, x2» by OLS with robust errors",
               ("Classical errors", "Robust HC0–HC3", "Cluster", "HAC (Newey-West)")),
        Method("binary", "Binary response", "Probability models for a 0/1 variable", ("ols_binary",),
               "Estimate a logit of «y» on «x1, x2» and give the marginal effects", ("Logit", "Probit")),
        Method("count", "Count data", "Number of events: Poisson (PPML), negative binomial, zero-inflated",
               ("ols_count",), "Estimate a Poisson model of «y» on «x1, x2» with robust errors",
               ("Poisson / PPML", "Negative binomial", "Zero-inflated Poisson", "Zero-inflated negative binomial")),
        Method("ordered", "Ordered response", "Categories with a natural order (ratings, levels)",
               ("ols_ordered",), "Estimate an ordered logit of «y» on «x1, x2»",
               ("Ordered logit", "Ordered probit")),
        Method("multinomial", "Multinomial response", "Choice between unordered categories", ("ols_multinomial",),
               "Estimate a multinomial logit of «y» on «x1, x2»"),
        Method("quantile", "Quantile regression", "Effect on a quantile of the variable, not on the mean",
               ("ols_quantile",), "Estimate the quantile regression of «y» on «x» at quantiles 0.1, 0.5 and 0.9",
               ("Median", "Extreme quantiles")),
        Method("margins", "Marginal effects", "Average effects and predictions of logit, probit and interactions",
               ("ols_margins",), "Compute the average marginal effects of «model» by «group»",
               ("Slopes", "Contrasts", "Predictions")),
        Method("diagnostics", "Specification tests", "Heteroskedasticity, autocorrelation, normality, RESET",
               ("ols_diagnostics",), "Is there heteroskedasticity in «model»? Check the assumptions"),
        Method("compare", "Compare models", "Side-by-side regression table", ("ols_compare",),
               "Compare «model1» and «model2» in a table"),
    )),
    Area("panel", "Panel data", "Estimators for individuals observed over time", (
        Method("panel_fit", "Panel estimators", "Unobserved heterogeneity by individual and period",
               ("panel_fit",), "Estimate fixed effects of «y» on «x1, x2» with errors clustered by individual",
               ("Fixed effects", "Two-way fixed effects", "Random effects", "First differences",
                "Between", "Pooled")),
        Method("hausman", "Hausman test", "Choose between fixed and random effects", ("panel_hausman",),
               "Compare fixed and random effects with the Hausman test"),
    )),
    Area("causal", "Causal inference", "Identification strategies for causal effects", (
        Method("iv", "Instrumental variables", "Endogenous regressor with external instruments", ("iv_fit",),
               "Estimate the effect of «x» on «y» instrumenting with «z»", ("2SLS", "LIML", "GMM")),
        Method("propensity", "Selection on observables", "Propensity score: matching and weighting",
               ("py_run",), "Estimate the effect of «treatment» on «y» with propensity score matching",
               ("Matching", "IPW", "Doubly robust"),
               library=("propensity_score_construction", "propensity_score_matching",
                        "propensity_score_inverse_probability_weighting",
                        "propensity_score_double_robust_estimator_augmented_IPW")),
        Method("did", "Difference in differences", "Policy with a common adoption date", ("panel_fit",),
               "Estimate a difference in differences of «y» with «treated» and «post»", ("TWFE",),
               library=("Static_Diff_in_Diff_regression",)),
        Method("event_study", "Event study", "Staggered adoption and effect dynamics",
               ("panel_event_study",),
               "Estimate an event study of «y» with adoption date «cohort» and show the pre-trends",
               ("did2s (Gardner)", "Callaway-Sant'Anna", "Sun-Abraham", "LP-DiD", "Dynamic TWFE")),
        Method("rdd", "Regression discontinuity", "Threshold in an assignment variable",
               ("causal_rdd", "causal_rd_density"),
               "Estimate a regression discontinuity of «y» at the threshold «c» of «variable» and check manipulation",
               ("Sharp", "Fuzzy", "Manipulation test")),
        Method("synth", "Synthetic control", "One treated unit against a combination of control units",
               ("causal_synth",), "Build a synthetic control for treated «region» in «year» for «y»",
               ("Donor weights", "Placebos")),
        Method("dml", "Double machine learning", "Effect with many controls and nonlinear relationships",
               ("causal_dml",), "Estimate the effect of «treatment» on «y» with double ML controlling for «x1, x2»",
               ("Partially linear (PLR)", "Interactive (IRM)")),
        Method("cate", "Heterogeneous effects", "Who benefits most from the treatment? Causal forest",
               ("causal_forest",), "Estimate with a causal forest how the effect of «treatment» on «y» varies "
               "with «x1, x2»"),
        Method("pds", "Control selection with lasso", "Post-double-selection when there are many controls",
               ("causal_pds",), "Estimate the effect of «treatment» on «y» choosing controls with lasso among "
               "«x1…xn»"),
        Method("sensitivity", "Sensitivity to confounders", "How strong would an omitted variable have to be?",
               ("causal_sensitivity",), "Is the effect of «treatment» in «model» robust to omitted variables?",
               ("Robustness value", "Bounds with benchmark covariates")),
    )),
    Area("time_series", "Time series", "Dynamics, stationarity, systems and forecasting", (
        Method("correlogram", "Correlogram", "Simple and partial autocorrelation functions", ("ts_acf_pacf",),
               "Show the correlogram of «series»", ("ACF", "PACF")),
        Method("unit_root", "Unit roots", "Is the series stationary?", ("ts_unit_root",),
               "Test whether «series» has a unit root",
               ("ADF", "KPSS", "Phillips-Perron", "Zivot-Andrews", "DF-GLS", "Variance ratio")),
        Method("arima", "ARIMA", "Univariate models, with seasonality", ("ts_arima",),
               "Estimate an ARIMA(1,1,1) for «series»", ("ARIMA", "SARIMA")),
        Method("ardl", "ARDL and bounds test", "Dynamic effects and long-run relationship, without requiring I(1)",
               ("ts_ardl",), "Estimate an ARDL of «y» on «x» and test for a long-run relationship",
               ("ARDL", "Bounds test")),
        Method("markov", "Regime switching", "Hidden regimes (expansion/recession, calm/turbulence)",
               ("ts_markov",), "Estimate a Markov regime-switching model of «series» with two regimes",
               ("Switching mean", "Markov autoregressive")),
        Method("granger", "Granger causality", "Does the past of one series help predict another?",
               ("ts_granger",), "Test whether «x» Granger-causes «y»"),
        Method("var", "VAR", "Vector autoregressions and impulse responses", ("ts_var",),
               "Estimate a VAR with «series» and show the impulse response functions"),
        Method("cointegration", "Cointegration and VECM", "Long-run equilibrium relationships",
               ("ts_cointegration", "ts_vecm"), "Are «series» cointegrated? If so, estimate a VECM",
               ("Johansen", "Engle-Granger", "VECM")),
        Method("breaks", "Structural breaks", "Detect breaks in the series", ("ts_breaks",),
               "Look for structural breaks in «series»"),
        Method("forecast", "Forecasting", "Automatic forecasts with intervals", ("ts_forecast",),
               "Forecast «series» 8 periods ahead", ("Auto-ARIMA", "Auto-ETS", "Naive", "Seasonal naive")),
    )),
    Area("volatility", "Volatility", "Conditional variance of returns", (
        Method("garch", "GARCH models", "Volatility clustering and asymmetries", ("vol_garch",),
               "Estimate a GARCH(1,1) with t errors for «returns»", ("GARCH", "EGARCH", "GJR-GARCH")),
    )),
    Area("charts", "Charts", "Visualizations in the analysis panel", (
        Method("line", "Lines", "Evolution over time of one or more series", ("viz_line",),
               "Plot «series» over time"),
        Method("scatter", "Scatter", "Relationship between two variables with a fitted line", ("viz_scatter",),
               "Plot «y» against «x»"),
        Method("histogram", "Histogram", "Distribution of a variable", ("viz_histogram",),
               "Show the histogram of «variable»"),
        Method("coefplot", "Coefficients", "Estimates with confidence intervals", ("viz_coefplot",),
               "Plot the coefficients of «model»"),
        Method("residuals", "Residuals", "Residuals vs fitted, QQ and histogram", ("viz_residuals",),
               "Plot the residuals of «model»", ("Fitted", "QQ", "Histogram")),
    )),
)


def _method_entry(m: Method) -> dict[str, Any]:
    specs = [get_tool(name) for name in m.tools]
    missing_tools = [name for name, spec in zip(m.tools, specs, strict=True) if spec is None]
    present = [s for s in specs if s is not None]
    missing_modules = sorted({mod for s in present for mod in s.requires if find_spec(mod) is None})
    if m.structures is not None:
        structures: list[str] | None = list(m.structures)
    elif not present or any(s.kinds is None for s in present):
        structures = None
    else:
        structures = sorted(set().union(*(s.kinds or () for s in present)))
    return {
        "id": m.id, "label": m.label, "summary": m.summary, "tools": list(m.tools), "variants": list(m.variants),
        "library": list(m.library), "prompt": m.prompt, "structures": structures,
        "available": not missing_tools and not missing_modules, "missing": missing_tools + missing_modules,
    }


def methods_catalog() -> list[dict[str, Any]]:
    """The catalog as the UI reads it: areas with their methods, availability resolved against the registry."""
    return [{"id": a.id, "label": a.label, "summary": a.summary, "methods": [_method_entry(m) for m in a.methods]}
            for a in AREAS]
