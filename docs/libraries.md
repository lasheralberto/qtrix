# Python econometrics libraries — catalogue for qtrix-mcp

Status: **core** = installed by default and exposed now · **extra** = optional dependency group, tools auto-enable
when installed · **planned** = tool namespace designed, not implemented yet. Versions are the ones resolved in
`engine/uv.lock` (Oct 2026) or the latest release checked at the time.

| Library | Domain | Licence | Status | Namespace | What qtrix exposes / will expose |
|---|---|---|---|---|---|
| **statsmodels** 0.15 | OLS/WLS/GLM, discrete choice, time series, tests | BSD-3 | core | `ols`, `ts` | `ols_fit` (HC0–3, cluster, HAC), `ols_binary` (+AME), `ols_count` (Poisson/PPML, NegBin, ZIP, ZINB), `ols_ordered`, `ols_multinomial`, `ols_quantile`, `ols_diagnostics` (BP, White, RESET, JB, DW, VIF), `ts_unit_root` (ADF/KPSS), `ts_arima` (SARIMAX + forecast), `ts_ardl` (ARDL + bounds test), `ts_markov` (regime switching), `ts_granger`, `ts_var` (VAR, Granger, IRF) |
| **scipy** | distributions, optimisation | BSD-3 | core | — | Hausman χ², Q-Q plots |
| **linearmodels** 7.0 | panel FE/RE/BE/FD, IV-2SLS/LIML/GMM, system | NCSA | extra `panel` | `panel`, `iv` | `panel_fit`, `panel_hausman`, `iv_fit` with first-stage F and over-id tests |
| **pyfixest** 0.60 | HDFE OLS/Poisson, wild bootstrap, event study, `etable` | MIT | extra `panel` | `panel` | planned: `panel_feols` (multi-way FE), `panel_fepois`, wild cluster bootstrap, DiD `did2s` |
| **arch** 8.0 | GARCH family, unit roots, cointegration, bootstrap | NCSA | extra `ts` | `ts`, `vol` | `ts_unit_root` PP / Zivot-Andrews / DF-GLS / variance ratio; planned `vol_garch`, Engle-Granger / Phillips-Ouliaris |
| **pmdarima** | auto-ARIMA | MIT | planned | `ts` | order selection |
| **statsforecast** (Nixtla) | fast ETS/ARIMA/Theta, many series | Apache-2 | planned | `ts` | panel forecasting |
| **ruptures** | structural break detection | BSD-2 | planned | `ts` | `ts_breaks` (PELT, binary segmentation) |
| **DoubleML** | double/debiased ML (PLR, IRM, IIVM) | BSD-3 | planned | `causal` | `causal_dml` |
| **EconML** (Microsoft) | heterogeneous effects, causal forests, DR learners | MIT | planned | `causal` | `causal_cate` |
| **DoWhy** | causal graphs, identification, refuters | MIT | planned | `causal` | `causal_refute` |
| **differences / csdid** | staggered DiD (Callaway–Sant'Anna) | MIT / GPL | planned | `did` | `did_staggered`, event-study plots |
| **rdrobust / rddensity** | regression discontinuity, manipulation test | BSD | planned | `rd` | `rd_estimate`, `rd_density`, RD plots |
| **pysyncon** | synthetic control (ADH, robust, augmented) | MIT | planned | `sc` | `sc_fit`, placebo inference |
| **PyMC / bambi / ArviZ** | Bayesian models with formulas, diagnostics | Apache-2 | planned | `bayes` | `bayes_fit` (background job), R-hat/ESS, PPC |
| **pyblp** | BLP demand estimation | MIT | planned | `io` | random-coefficients logit |
| **quantecon** | dynamic programming, Markov chains | BSD-3 | planned | `macro` | calibration helpers |
| **scikit-learn** | ML, regularisation, CV | BSD-3 | planned | `ml` | lasso/post-lasso selection, CV |
| **pandas / polars / DuckDB / pyarrow** | data wrangling | BSD/MIT | core | `data` | workspace, `data_sql`, `data_transform`, Parquet/CSV import |
| **openpyxl** | Excel | MIT | core | `data` | `.xlsx/.xls` import |
| **pyreadstat** | Stata/SPSS/SAS with labels | Apache-2 | planned | `data` | `.dta/.sav/.sas7bdat/.xpt` import keeping labels |
| **SQLAlchemy + psycopg / DuckDB postgres ext.** | databases | MIT/LGPL | planned | `data` | Postgres, MySQL, SQL Server, SQLite connectors |
| **fredapi / wbgapi / sdmx1** | public data (FRED, World Bank, OECD/Eurostat) | Apache/MIT | planned | `data` | explicit, download-only fetchers |
| **plotly** 7 | interactive charts | MIT | core | `viz` | scatter (+fit), histogram, line, coefplot, residuals, correlation heatmap, IRFs, forecast fans |
| **great_tables / stargazer** | publication tables | MIT | planned | `report` | LaTeX/HTML export of the panel |
| **turbovec** 1.1 + **fastembed** 0.9 | vector memory, local embeddings | MIT / Apache-2 | extra `memory` | `memory` | `memory_search`; automatic recall injected into every agent run |

## Priorities

1. Breadth for the most common applied work first: cross-section, panel, IV, time series (done), then
   modern DiD / RD / synthetic control (most requested causal designs), then DML and Bayesian.
2. Every tool must: take datasets by id, return a compact summary to the LLM, render a typed panel block,
   and emit a reproducible Python snippet.
3. Heavy stacks (PyMC, EconML) ship as optional extras so the base installer stays small.
