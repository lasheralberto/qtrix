---
name: qtrix-econometrics
description: "Causal inference and applied econometric design in qtrix: pinning down the estimand and the identification strategy, picking the estimator, running it with the qtrix tools or, for what they lack (propensity scores, PSM/IPW, DID event studies, RDD), with the bundled causal library in a Python cell, then diagnostics, robustness and interpretation. Read it for treatment effects (ATE/ATT/LATE/ITT), policy evaluation, IV design, DID, RDD, falsification tests, heterogeneity or a research-design memo."
---

# Econometrics in qtrix

This guide adapts the bundled econometrics skill (`eco-skills/econometrics-skill`) to qtrix: the qtrix tools come
first, because their results are named, versioned objects with live views; the skill's causal library fills the
gaps in a Python cell.

Read the reference files this guide names with `read_file` (paths as written here); for a long one, find the
heading's line with `search_files` and read from there with `offset`/`limit`.

## 1. Identification before estimation

Settle these four points before any estimate, from the data and the conversation; ask the user when they are not
stated (in one short question, not a questionnaire):

1. Outcome Y (continuous, binary, count) and its unit.
2. Treatment T (binary, continuous, policy dummy) and the estimand: ATE, ATT, LATE or ITT, for which population.
3. Why a naive `Y ~ T` is biased: selection, reverse causality, omitted variables.
4. The source of identifying variation. It picks the method family (section 2).

When the choice is not obvious read `eco-skills/econometrics-skill/references/method_selection.md`. Inspect the data first with
data_describe_structure and data_summary_stats: the structure (cross-section, time series, panel) rules methods in
or out.

## 2. Method map: qtrix tool first, library cell when there is none

| Identifying variation | Use | Notes |
|---|---|---|
| Randomised / as-if random | `ols_fit` (HC1, or `cov_type="cluster"` + `cluster`) | Binary Y: also `ols_binary` (logit/probit) as a check |
| Selection on observables | Library: `propensity_score_construction`, overlap plot, `propensity_score_matching`, `propensity_score_inverse_probability_weighting`; `ols_fit` with the controls as baseline | Always plot the overlap first |
| Instrument Z | `iv_fit` (2SLS/LIML/GMM, correct SEs, first-stage diagnostics) | Never the library's `IV_2SLS_regression`: its SEs are biased down. `IV_2SLS_IV_setting_test` is fine as an extra reduced-form/falsification check |
| Policy timing in a panel, one adoption date | `panel_fit` with `estimator="twfe"` and a `treated:post` (or `D`) regressor, `cov_type="cluster_entity"` | Library `Static_Diff_in_Diff_regression` is equivalent |
| Staggered adoption / dynamics | Library `Staggered_Diff_in_Diff_Event_Study_regression` + `..._visualization` for pre-trends | TWFE with staggered timing and heterogeneous effects is biased (Goodman-Bacon 2021): say so, report the event study |
| Cutoff in a running variable | Library `Sharp_Regression_Discontinuity_Design_regression` / `Fuzzy_Regression_Discontinuity_Design_regression` | At least two bandwidths; fuzzy: keep `target_type="summary"` |
| Dynamics of macro series | `ts_*`, `vol_garch` | Outside causal identification: no treatment effect claims |

Keep the specification in the qtrix tools whenever they can express it: a formula with `C(...)`, interactions, a
sample (`data_set_sample`) or a generated series (`data_genr`) is better than a cell.

## 3. Running the causal library in a Python cell

Cells run with `py_run` in the project kernel. The user must have allowed agents to run code; if `py_run` is
refused, show the cell in your answer and ask the user to run it (or to enable it). Never retry a refused cell.

A cell must stand alone: a registered result is rebuilt by re-running its cell in a fresh kernel, so import what
you use and read the data inside the cell.

```python
import qtrix as qx
from qtrix_engine.contrib.econometrics.econometric_algorithm import (
    propensity_score_construction, propensity_score_visualize_propensity_score_distribution,
    propensity_score_matching, propensity_score_inverse_probability_weighting)

df = qx.page("survey").frame(["earn", "training", "age", "educ", "married"]).dropna()
X = df[["age", "educ", "married"]]
ps = propensity_score_construction(df["training"], X)
att = propensity_score_matching(df["earn"], df["training"], ps, target_type="ATT")
ate = propensity_score_inverse_probability_weighting(df["earn"], df["training"], ps)
print(f"PSM ATT = {att:.3f}  IPW ATE = {ate:.3f}  PS range = [{ps.min():.3f}, {ps.max():.3f}]")
propensity_score_visualize_propensity_score_distribution(df["training"], ps)  # last line: shown in the cell
```

Data:
- `qx.page(name).frame(columns)` honours the active sample. Panels come indexed by (entity, time), which is what
  the DID functions need; time series by a DatetimeIndex; cross-sections by row number.
- Drop or handle missing values explicitly; the functions call `.astype(float)`.
- Treatment must be a 0/1 integer dummy: `(df["g"] == "treated").astype(int)`.
- Name computed Series (`s.name = "treat_post"`): OLS and RDD label coefficients with it.
- Dummy-encode categorical covariates: `pd.get_dummies(df[["region"]], drop_first=True, dtype=float)`.
- Never print raw rows of data.

What comes back, and what to do with it:

| Returns | Functions | Do |
|---|---|---|
| statsmodels / linearmodels result | OLS, PS regression, IPW-RA, Static/Staggered DID, event study, Sharp RDD, Fuzzy RDD global polynomial | `qx.register("eq_did", res, page="survey")`: a named object with its table in the panel |
| float | PSM, IPW, AIPW | Print it; report it in your answer (and `qx.report.add_note` if the user wants it kept) |
| dict | Fuzzy RDD (`summary`), `IV_2SLS_IV_setting_test` | Print the key entries |
| matplotlib Figure | overlap plot, event-study plot | Make it the last expression of the cell so it is displayed |

Standard errors in the library differ from the qtrix tools:
- statsmodels-based functions (OLS, RDD, PS regression, IPW-RA) take `cov_info` / `cov_type`: `"HC1"`,
  `"nonrobust"`, `{"cluster": df["firm"]}`, `{"HAC": 4}`.
- DID functions (linearmodels) take `cov_type`: `"unadjusted"`, `"robust"`, `"cluster_entity"`, `"cluster_time"`,
  `"cluster_both"`. Default for panels: `"cluster_entity"`.

Exact signatures: `eco-skills/econometrics-skill/references/method_details.md` (read only the section you need, e.g. "## 14. Sharp RDD"). Its
snippets import `econometric_algorithm` directly; in qtrix import it from `qtrix_engine.contrib.econometrics`.

## 4. Known library limitations: disclose them when relevant

- IV 2SLS SEs are too small: use `iv_fit`.
- The AIPW function is not doubly robust (it only imputes one direction): prefer PSM and IPW as primary estimates.
- IPW-RA takes the square root of the IPW weights: non-standard, label it as such.
- The event study assigns leads and lags from the row position: the panel must be balanced, sorted by time and
  without gaps. Check with data_describe_structure; restrict the sample to a balanced panel if needed.
- No synthetic control, triple differences, quantile treatment effects or DML / causal forests. Say so and, if
  the user wants it, write it from statsmodels / linearmodels / scikit-learn in a cell.

## 5. Diagnostics, robustness, interpretation

Before presenting an estimate as causal run the checks of `eco-skills/econometrics-skill/references/diagnostic_checklist.md` for its method:
overlap and balance (PS), first-stage strength (IV: `iv_fit` reports it), pre-trends (DID), bandwidth sensitivity
and covariate continuity (RDD). Compare specifications of the qtrix objects with `ols_compare`.

Interpret, do not dump: sign and magnitude in the outcome's units, uncertainty (SE / CI), practical relevance and
the most fragile assumption. Templates per method: `eco-skills/econometrics-skill/references/interpretation.md`.

For publication-grade work, referee-style robustness or a research-design memo, follow
`eco-skills/econometrics-skill/references/advanced_applied_workflow.md` (estimand statement, identification memo, specification grid, robustness
suite, heterogeneity plan); for a whole project, `eco-skills/econometrics-skill/references/applied_project_workflow.md`.

## 6. Reference files

- `eco-skills/econometrics-skill/references/method_selection.md`: decision guide for the estimator
- `eco-skills/econometrics-skill/references/method_details.md`: library signatures and snippets, by section
- `eco-skills/econometrics-skill/references/diagnostic_checklist.md`: checks before calling an estimate causal
- `eco-skills/econometrics-skill/references/interpretation.md`: how to report each method family
- `eco-skills/econometrics-skill/references/advanced_applied_workflow.md`, `eco-skills/econometrics-skill/references/applied_project_workflow.md`: research-grade workflows
- `eco-skills/econometrics-skill/references/data_preprocessing_advice.md`: cleaning advice; in qtrix start from data_summary_stats and
  data_describe_structure, and `analyze_dataset(df)` from `qtrix_engine.contrib.econometrics.data_preprocess` on a
  frame for a fuller report
- `eco-skills/econometrics-skill/references/result_tables.md`: comparison tables; in qtrix prefer `ols_compare` over registered objects
- `eco-skills/econometrics-skill/examples/*.py`: runnable end-to-end examples per method (simulated data; adapt the imports as in section 3)
