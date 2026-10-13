# ADR 0002 — Workfile and named objects (an agentic EViews)

Status: accepted · 2026-10-11 · supersedes parts of ADR 0001 ("Dynamic panel", "Reproducibility")
Epic: E7 (issues #216–#247)

Implementation notes, E7·05 (objects): `objects/{specs,store,estimators}.py` follow §3. Deviations: `estimate(spec)` reads the page
through the current context instead of taking a frame; the resolved sample is frozen into the spec at estimate time;
version diffs are stored in `summary["diff"]` (ignored by the rebuild check); `_qtrix_objects` has a `code` column and
spec/summary are JSON; the actor (`user` vs agent/api/mcp) is a context variable until the dispatcher (E7·06) exists;
stale data is detected by comparing `data_version` on read; `r_xxxx` aliases are in-memory (`ObjectStore.add_alias`).
Not yet objects: `viz_*` graphs and `ui_add_note` texts (specs exist).

Implementation notes, E7·06 (journal): `journal.py`, `dispatch.py`, `replay.py`, `callctx.py` follow §6. Deviations:
`affected` entries carry a `kind` (`object` | `page`) and page entries add the created page's id, source and snapshot;
before-images for undo live in a side table `_qtrix_journal_before` (plus `undo_<seq>_<page>` data copies for
`data_set_structure` / `data_genr`); `ToolSpec` gets `mutating` (bool or predicate) and `internal` (UI-only tools:
`obj_unlock`, `data_delete_dataset`, `journal_undo`); `ui_*` writers are not mutating (report layout, not state) until
`ui_add_note` becomes a `text` object; every import keeps an immutable `src_<id>` snapshot (`Workspace.delete` of a page
through the dispatcher keeps it); replay re-executes derived pages (`data_transform`, `data_sql save_as`) with the original
page ids and refuses pages that did not come from the journal; `cell_id` in chat is `<turn_id>:<n>`; undoing a page
deletion is not supported; `kind = python` and `proposal_id` are written only by later issues.

Implementation notes, E7·07 (`qx`): `qtrix_engine/qx/{forms,code,runtime}.py` plus the public package `qtrix`
(`src/qtrix`). Each `@tool` declares `qx=Qx(on, name, positional, aliases, rename, family, varkw)`; `on` is `qx`,
`page` (handle carries `dataset_id`), `obj` (carries `name`/`result_id`) or `report`. Estimators are grouped in two
families whose keywords are the spec fields: `qx.equation(name, page=, method=, …)` (`ls`/`wls` → `ols_fit`, `binary`,
`tsls`/`liml`/`gmm` → `iv_fit`, `panel`, `arima`, `garch`) and `qx.var(…, method="var"|"vecm")`; `ols_compare` is
`qx.table`, `obj_group` is `qx.group`, `obj_reestimate` is `obj.estimate(**changes)`. Open question 5 is resolved as
proposed: Pythonic names (`set_sample`, `generate`, `summary_stats`) with EViews aliases (`smpl`, `genr`, `stats`,
`uroot`, `coint`) accepted by the parser and the runtime; the printer always writes the Pythonic name. The printer
omits arguments equal to the tool's default, writes object-creating calls as an assignment (`eq01 = qx.equation("eq01",
…)`), wraps past 88 characters and falls back to `qx.tool("<tool>", **args)` for calls without a form. Stored specs
always carry their resolved sample, so `GET /objects/{name}/code` prints `sample=` explicitly. The parser binds bare
names to objects, variables to the page/object they were assigned, and an object created without a name to the name
the engine gives it when the cell runs (`handle_from`). `POST /code` runs API cells as `user` via channel `code`,
stops at the first error (later calls `skipped`), and refuses Python cells with 422 until the kernel exists (E7·07b).
`glm`, `graph` and `text` objects have no form yet (`NotExpressible`): no tool creates them.

Implementation notes, E7·07b (kernel): `kernel/{manager,bridge}.py`, `qx/client.py`, `tools/py.py`. One IPython kernel
per engine (`jupyter_client`, the engine's own interpreter), started on demand; `GET /kernel`, `POST
/kernel/{start,interrupt,restart,shutdown}`. A Python cell is the tool `py_run` (`ToolSpec.cell`), so `POST /code`, agents
and MCP all reach it through the dispatcher; it is journaled as one `kind = python` entry whose `code` is the source and
whose `mutating` says whether the cell changed the workfile, and every `qx` call the cell makes is an entry of its own
(channel `kernel`, the cell's `cell_id`, the cell author as actor). The kernel's `qx` runs in client mode (`POST /qx/call`,
`GET /qx/frame` as an Arrow stream). It authenticates with a kernel token that only opens `/qx/*` and only while a cell
of its own is running (the engine-issued run id travels in `x-qtrix-run`), so a thread an agent's cell leaves behind
cannot act later as the user; external clients (`qx.connect(url, token)`) use the session token and act as `user` on
channel `client`. Bridge: `page.frame()`, `qx.series_from()` (internal tool `data_set_series`, one value per stored row;
a pandas Series is aligned on the page index) and `qx.register()` (internal tool `py_register`: the summary is computed
in the kernel with `summarize_result`, the cell becomes a `PythonSpec(code, registered_as, page)`). Rebuilding a Python
object (re-estimating it, possibly with `code=` changed, and replay) re-runs its cell in a second, fresh kernel where
mutating calls are skipped and the `qx.register` of `registered_as` hands back the summary, so replay checks that the
result is reproducible; the cell must therefore be self-contained. Its fitted result stays in the kernel
(`fitted()` refuses) until views exist (E7·08). Replay skips `python` entries (their calls are replayed instead); the
session script prints the cells and leaves out their calls; undoing a cell is refused (undo its calls). Policy: only the
user runs cells unless `agent.allow_code` is set (proposals come with E7·18). Timeouts interrupt the cell; a cell that
ignores the interrupt (native code such as `time.sleep` on Windows) gets the kernel restarted after 5 s. **The kernel is
not a sandbox**: it runs with the user's privileges and can read files and use the network (privacy audit, #210).

Implementation notes, E7·08 (views): `objects/views.py` holds both registries. `@view(type, name, title=, group=,
methods=, kinds=, check=, cost=, requires=, when=)`: the parameters are the function's keyword arguments, validated (and
filled with defaults) by a Pydantic model, so `ref = {object, view, params}` is canonical. Results are cached in the
object store by `(name, version, view, params)` next to the fitted LRU and the estimate's by-products (`extras`: the
forecast, IRF and volatility figures, rebuilt by re-estimating when evicted); everything cached for a name is dropped on
rename, delete, undo and when a version number is reused. Procs are metadata pointing at registry tools (`estimate` →
`obj_reestimate`, `freeze` → `obj_freeze`, `copy`, `rename`, `lock`, `delete`). Views today: equation `estimation` and
`coefplot` (from the summary, so they also serve stale and Python objects), `residuals(kind)` and `diagnostics` (check;
`ls`/`wls`) from the fitted model, `forecast` (ARIMA with a forecast), `volatility` (GARCH); var `estimation`, `irf`;
table `table`; graph `graph`; group `members`; text `text`. Estimation tools add live blocks through `show`, which
never duplicates a view already in the report, and on a new version (or an undo) `refresh` recomputes the object's live
blocks in place; frozen blocks of it are flagged stale. `freeze` exists twice: `ui_freeze_block` / `POST
/panel/blocks/{id}/freeze` drop a block's `ref`, and `obj_freeze` stores the view as a `frozen_table` / `frozen_graph`
object, whose `source` is provenance only (it is not a dependency, so it never goes stale). Agent and `qx` access:
`obj_views`, `obj_view(name, view, **params)` (`qx.obj("eq01").view("residuals", kind="qq")`), `ui_add_view`
(`qx.report.add`). HTTP: `GET /objects/{name}/views`, `GET /objects/{name}/views/{view}?params=<json>`. A view of a stale
object reuses the fitted model while it is still in memory and refuses to rebuild it otherwise. Check outcomes are
returned but do not set the object's `status` yet (playbooks, E7·17).

Implementation notes, E7·08b (chat as cells): every chat turn starts with a `turn` event; `tool_call` carries the call's
canonical `qx` code and `tool_result` links it to its journal entry (`seq`, `cell_id`, the journaled code) and to what
it produced (`object`, `version`, `block_id`). The journal entry is found by turn id and tool (journal `list(turn_id=)`);
`/chat/history` keeps the cells, so they survive a reload. The desktop chat renders each one as a CodeMirror 6 cell
(`apps/desktop/src/cells.tsx`): Python highlighting, the `formula=` argument highlighted, completion of `qx` functions per
target, objects, pages and series (`GET /code/completions`), and live checks while editing through `POST /code` with
`dry_run` — each API call now returns `issues` (`qx/check.py`: unknown variables, time operators the page cannot take,
unknown pages/columns; formulas are compiled against the page as estimation would) and a syntax error or a call that does
not bind is `kind: "invalid"`. Running an edited cell sends `edited_from` (the last seq of that cell, so re-editing chains)
and the response reports `versions` (objects that got a new version) and `stale` (dependants that went stale). A cell that
turns into free Python is warned about before running (kernel, and loss of the structured spec if it registers its own
model). Turn code: a turn with several cells shows them as one copyable script. The agent learns about edits on its next
turn: `ChatSession.user_edits()` lists the mutating user entries since its last answer — with a diff against the agent's
original call (`formula: + gap(-1)`; tool defaults filled in on both sides so omitted defaults do not show as changes)
and the new versions — prepended to the prompt in brackets (the transcript keeps the user's text); the instructions tell
it to build on them and never revert them. Report blocks now show live / frozen / stale badges and a freeze button.

## Context

qtrix today is "datasets + a panel of blocks". A tool call such as `ols_fit` reads a DuckDB table, fits a
model, keeps the fitted object in an in-memory `ResultStore` under an opaque id (`r_1a2b3c4d`) and appends a
block to the panel. This is enough for a chat demo but not for econometric work:

- Data have no **frequency or dates**: `ts_*` tools sort by a column and forecasts are drawn on `0..n`.
- There is no **estimation sample**; every tool works on the whole table.
- Formulas have no **time-series operators** (`d()`, `dlog()`, `y(-1)`), so the agent creates helper columns
  with SQL.
- Fitted models are **lost on restart** and cannot be re-estimated, renamed, compared over versions or
  locked. The panel is the only trace, and it is append-only output, not state.
- What the agent did is visible only in the chat transcript; there is **no reproducible program**.

EViews has served applied macro and finance for decades because of one mental model: a **workfile** with a
declared structure and an active sample, holding **named objects** (series, groups, equations, VARs,
models…) that each offer **views** and **procs**, all driven equally by menus and by a command language.
That model is also what an agent needs: stable names to refer to, declarative state to inspect, and a
small set of verbs whose effects are explicit.

qtrix is **chat-first**. The researcher mostly talks to the agent; every action the agent takes is
**translated into Python** that the researcher can read, edit and re-run. Formulas and generated code
belong to the user, not to the agent. qtrix therefore takes EViews' object model, but **not its command
language**: the internal language is Python, through a small `qtrix` API (imported as `qx`), plus free
Python with pandas and statsmodels for anything the API does not cover.

This ADR adopts that model and defines how agents, the user, Python code, the HTTP API and MCP clients all
act on it through a single path.

## Decision summary

| Concern | Decision |
|---|---|
| Unit of work | A **workfile** (the project's DuckDB file) with one or more **pages**; each page has a **structure** (cross-section by default; time series or panel only when declared) and an **active sample** |
| Adaptivity | Tools, views, operators and agent playbooks declare the **page kinds** they support; nothing assumes dates |
| State | **Named objects** in an object store (DuckDB tables), one namespace per workfile |
| Persistence | Objects persist as **declarative specs + JSON summaries**, never pickles; fitted library objects are rebuilt by re-estimation and cached in memory |
| Versioning | Every estimation creates a new **object version**; history is kept; dependants become **stale** |
| Behaviour | Each object type exposes **views** (pure, cached) and **procs** (create or modify objects), declared in a registry |
| Report | The panel becomes a **report**: blocks reference `object.view(params)` and refresh when the object changes; `freeze` snapshots |
| Single write path | Every mutation goes through one **dispatcher**, which writes a **journal** entry |
| Internal language | **Python.** A `qx` API whose functions map 1:1 to registry tools; the journal stores canonical Python. **Free Python** (pandas, statsmodels…) runs in a per-project kernel |
| Code round-trip | Every agent action appears in the chat as an **editable Python cell**. API-only cells are parsed back into specs **without executing them**; editing a formula and re-running creates a new object version |
| Agents | Agents see a compact **workfile snapshot**, act only through `qx` tools (or free Python under policy), cannot touch **locked** objects, and are gated by an **autonomy policy** at the dispatcher |

## 1. Workfile, pages and structure

**Workfile** = the DuckDB database that holds a project. Until project files exist (E7·30, `.qtx`), the
workfile is the single `<qtrix>/data/workspace.duckdb`.

**Page** = what is now a dataset. A page is one table and one catalog row. Pages are in the schema from day
one so that monthly and quarterly data can live side by side (E7·21); the UI shows one page at a time
until then.

**Time is optional.** A workfile is not a time-series container. Each page declares the structure its data
actually have, and nothing assumes dates unless the page has them:

| Kind | Example | Identified by | `freq` |
|---|---|---|---|
| `cross_section` | survey of 5,000 workers, firms in 2019 | row number, or an optional `id_col` (worker id) | `U` |
| `time_series` | quarterly macro aggregates | a date | `A` … `D7` |
| `panel` | countries × years, firms × quarters | `id_col` + a date | `A` … `D7` |
| `panel_undated` | students × test attempt, pseudo-panels | `id_col` + an integer `t_col` | `U` |

Pages start as **`cross_section` by default**. Import only *suggests* a structure when the evidence is
strong (a column parses as dates, or an id × date pair is unique), and the user or the agent confirms it.
A page can be re-structured at any time (`page.structure(...)`), for example a pooled cross-section turned into a
panel.

**Structure** of a page (`_qtrix_pages`, which extends today's `_qtrix_datasets`):

| Field | Values |
|---|---|
| `kind` | `cross_section` · `time_series` · `panel` · `panel_undated` |
| `freq` | `U` undated · `A` · `S` · `Q` · `M` · `W` · `D5` · `D7` · (`H`/intraday deferred) |
| `start`, `end` | first and last period (`1990Q1`, `2024Q4`); for undated pages, first and last observation number |
| `id_col` | entity column (required for panels, optional row id for cross-sections) |
| `date_col` / `t_col` | the user's original date column (kept for provenance) or integer time index |
| `data_version` | counter, bumped on every write to the page's data |

**Regularisation (dated pages only).** When a page gets a dated structure, its table is rewritten so that
rows cover the full range: one row per period, or per entity × period, with gaps filled with NULL. Two
hidden canonical columns are added: `__obs` (period start as `DATE`) and, for panels, `__id`. Every tool
gets a pandas `PeriodIndex` (or an `(id, period)` MultiIndex) built from them, so frequency-aware lags,
conversions and date axes never depend on how the user's file spelled its dates. Cross-section pages are
never reshaped: their index is the observation number (`1..n`), or `id_col` when one is declared.

**Capabilities follow the structure.** Every tool, view, proc and expression operator declares the page
kinds it supports, for example `@view("equation", "resid_correlogram", kinds={"time_series", "panel"})`:

- The UI menus show only what applies. A cross-section equation offers heteroskedasticity tests, RESET,
  influence and leverage, clustered errors and marginal effects; it shows no correlogram, Chow-by-date or
  forecast-by-date.
- The agent's tool list and the playbooks (E7·17) are filtered the same way, so a model never proposes a
  unit-root test on survey data.
- Time operators (`x(-1)`, `d()`, `@trend`, `@seas`) fail on cross-section pages with an actionable message.
  In panels they work within each entity.
- If a page has no dates but a requested operation needs them, the error says how to add a structure.
  It never guesses one.

**Series** are the columns of a page. Each also has an object row (section 3) holding its label, units,
source and, for generated series, the expression that defines it.

**Source snapshots.** An import keeps the imported table as an immutable `src_<id>` table. The page table
is derived from it, so the journal can be replayed (section 6) without the original file. This roughly
doubles the storage of imported data (DuckDB compresses it); `qx.compact()` can drop a snapshot, and
replay is then marked as needing the original file (checked against its recorded SHA-256).

## 2. Sample

The **active sample** is per-page state, stored as text in `_qtrix_pages.sample_spec` (NULL = `@all`). It is
reset when the page is re-structured, and a derived page (SQL, transform) starts with every row.

- Spec grammar (as in EViews): `<start> <end> [if <expr>]`, with `@first`, `@last`, `@all`, and
  several ranges (`1960q1 1979q4 1990q1 2007q4`). Bounds are periods on dated pages, observation numbers
  on cross-sections and values of the time index on undated panels, so on surveys the condition does most
  of the work: `if female = 1 and age >= 25`.
- The condition is a DuckDB SQL boolean over the page's columns. It is validated before it is stored, may
  not contain `;` or comments, and runs as a read-only `SELECT`. Column names are quoted automatically, so
  variables called `union` or `order` work.
- Resolution: spec → boolean row mask over the stored table (`Workspace.resolve_sample`), which also gives
  the number of rows kept and a readable description. An invalid sample, or one that selects no rows, is an
  error that names the data's span.
- Every estimation, descriptive and chart tool uses the active sample by default. An optional `sample=`
  argument overrides it for one call without changing the state. Transformations, SQL, previews and
  profiles always see every row.
- `Workspace.df` and `frame` apply the sample. On dated panels and cross-sections, rows outside it are
  dropped. On time series the index stays regular: it is cut to the sample's first and last period, and
  excluded periods inside it become missing, so ARIMA forecasts start right after the sample.
- **Lags look outside the sample** (as in EViews: the sample selects which observations are *used*, not
  which values exist). Formulas with lags arrive in E7·04, which will build variables on the full page
  first and apply the mask afterwards.
- Each estimated object records its **sample spec, resolved first and last periods, and effective nobs** (after
  lags and missing values). Tables print `Sample (adjusted): 1990Q3 2008Q4 · n = 74`.

## 3. Objects

### Types (closed list for v1)

| Type | What it holds | Created by |
|---|---|---|
| `series` | metadata of a page column; optional generating expression | import, `genr`, procs (forecast, make resid) |
| `group` | ordered list of series names | `group`, selecting several series |
| `equation` | one single-equation estimate; `method` ∈ `ls`, `wls`, `glm`, `binary`, `tsls`, `liml`, `gmm`, `panel`, `arima`, `arch` | `equation`, today's `ols_*`, `panel_fit`, `iv_fit`, `ts_arima` |
| `var` | VAR / VECM / SVAR (`method` ∈ `var`, `vec`, `svar`) | `var`, today's `ts_var` |
| `system` | multi-equation estimate (SUR, 3SLS, system GMM) | E7·27 |
| `model` | equations + identities for simulation and scenarios | E7·28 |
| `sspace` | state-space specification | E7·29 |
| `table` | frozen tabular output or a comparison (`ols_compare`) | `freeze`, `table` |
| `graph` | frozen or composed chart (Plotly JSON + style) | `freeze`, `graph` |
| `text` | markdown note (interpretation, conclusions) | `text`, today's `ui_add_note` |

`scalar`, `vector` and `matrix` objects are deferred until a feature needs them. A **forecast is a proc,
not a type**: it produces series (`yf`, `yf_se`) and optionally a graph, linked to the source object.

### Storage

```
_qtrix_objects(
  name VARCHAR PRIMARY KEY,      -- unique in the workfile, case-insensitive, [a-z][a-z0-9_]{0,31}
  type VARCHAR, page VARCHAR,
  version INTEGER,               -- current version
  spec JSON,                     -- declarative definition (below)
  summary JSON,                  -- compact, renderable results of the current version
  status VARCHAR,                -- unchecked | ok | warn | fail | stale
  status_detail JSON,            -- [{check, outcome, view, message}]
  author VARCHAR,                -- user | agent:<name> | mcp | api
  locked BOOLEAN,
  parent VARCHAR,                -- lineage: object this one was derived from (spec search, copies)
  data_version INTEGER,          -- page data_version used by the current version
  created_at TIMESTAMP, updated_at TIMESTAMP
)
_qtrix_object_versions(name, version, spec, summary, status, author, journal_seq, created_at)
```

### Spec

A spec is a Pydantic model per type and method (a discriminated union), serialised as JSON. It holds
everything needed to re-estimate deterministically, and nothing else:

```json
{
  "type": "equation", "method": "ls", "page": "macro_q",
  "formula": "dlog(gdp) ~ dlog(gdp(-1)) + d(r) + @seas(1)",
  "sample": "1990q1 2008q4",
  "cov": {"type": "HAC", "lags": 4},
  "options": {}
}
```

Formulas are written in the **qtrix expression language** (`qtrix_engine/expr.py`): R/patsy syntax
extended with EViews operators (`x(-k)` lag and `x(+k)` lead of a column, `L(x, k)`, `d(x, n, s)`,
`dlog(x)`, `pch(x)`, `movav(x, n)`, `@trend`, `@seas(q)`, `dummy("2008q3")`, `recode(c, a, b)`), which the
page's structure resolves: lags follow the calendar, stay inside each entity of a panel and move along the
time index of an undated panel. They fail with an actionable message on a cross-section.

Estimation libraries never see the operators. Before a fit, each formula term that uses one is **evaluated by
qtrix over the whole page** and handed over as a plain generated column (`qx_t1`), and the dependent
variable is blanked outside the active sample (`qx_lhs`). Doing it ourselves gives EViews' semantics (lags
may look outside the sample) and works unchanged with statsmodels (patsy) and linearmodels (formulaic),
which evaluate formulas differently. Results print the terms as the user wrote them (`dlog(gdp(-1))`, not
`qx_t1`); the mapping is kept in the model summary as `term_names`.

`genr` (tool `data_genr`) uses the same expressions to add a persistent series to a page. It always
computes every row (unlike EViews, which restricts it to the sample, a frequent surprise) and its
definition is stored in `_qtrix_series`, so the series is recomputed if the page is re-structured. The formula is a plain string inside Python code (`formula="dlog(gdp) ~ d(r)"`). It is
the part users edit most, so the UI offers it as an editable field with validation and autocompletion of
series names.

Estimation is a **pure function** `estimate(spec, page_frame) -> (fitted, summary)`, one per method,
wrapping statsmodels / linearmodels / arch. Today's tool bodies become these functions.

### Fitted cache and rebuild

- Fitted library objects are kept in an in-process LRU keyed by `(name, version)`.
- On a cache miss (after a restart, or on eviction), the object is **rebuilt**: re-estimated from its spec
  on the page data. If `page.data_version` still equals the recorded `data_version`, the result must equal
  the stored summary (a test enforces this to 1e-10).
- If the data changed since the object was estimated, the object is **stale**. It keeps its summary, and
  views already cached stay readable. A view that needs the fitted object fails with an actionable message
  ("eq01 is stale; re-estimate it"). Nothing is re-estimated silently: as in EViews, results change only
  when someone asks.
- **Why not pickle?** Pickles break across library versions, are unsafe to load from a shared project
  file, and are large. Specs are small, readable, diffable and portable.

### Lifecycle and naming

- `estimate` on an existing name creates version *n+1*. The previous version stays in
  `_qtrix_object_versions` and can be restored. Version diffs (coefficients, fit statistics, sample) are a
  view.
- Default names follow EViews (`eq01`, `var01`, `gr01`, `tab01`, `gph01`, `txt01`). Agents are told to choose
  meaningful names (`eq_taylor`).
- `rename`, `copy` (sets `parent`), `delete` (soft: kept in the version table until the history is compacted),
  `lock` and `unlock`.
- **Locked** objects cannot be modified, re-estimated or deleted by anyone but the user, through the UI or the
  user's own code. This replaces today's "pinned block" rule and extends it to state.
- **Dependants.** Objects referencing another object (a table comparing eq01 and eq02, a forecast of eq01,
  a model using eq01) record those references. A new version of the referenced object marks them stale,
  and the report refreshes them (section 5).

### Diagnostic status

`status` gives the workfile browser and the agent a traffic light:

- `unchecked`: estimated, no checks have run yet.
- `ok`, `warn`, `fail`: set from diagnostic views tagged as checks. The *playbooks* of E7·17 define which
  checks are mandatory per method and their thresholds.
- `stale`: inputs changed since estimation.

`status_detail` lists each check, its outcome and the view that produced it, so the UI and the agent can
cite the evidence.

## 4. Views and procs

Two registries sit beside today's tool registry:

```python
@view("equation", "resid_correlogram", group="Residual Diagnostics", check=True, cost="cheap")
def eq_resid_correlogram(obj: Equation, lags: int = 24) -> ViewResult: ...

@proc("equation", "forecast", group="Forecast")
def eq_forecast(obj: Equation, name: str, start: str, end: str, dynamic: bool = True) -> ProcResult: ...
```

- A **view** is a pure function of `(object version, params)`. It returns a `ViewResult`: a block payload
  in the existing kinds (`regression_table`, `plotly_figure`, `dataframe`, `diagnostics`, `markdown`,
  `equation`), an `llm` dict (the compact view handed to agents, as `llm_view` does today), and for checks
  an outcome (`ok`, `warn`, `fail`). Results are cached by `(name, version, view, params)`.
- A **proc** goes through the dispatcher and creates or modifies objects (forecast series, residual series,
  a frozen table, a new equation).
- The UI's View and Proc menus (E7·10), the Python API (`eq01.view("wald", "c(2) = 0")`) and agent tools
  (`obj_view(name, view, params)`) are all generated from these registries. Adding a test means adding one
  function.

## 5. The report (today's panel)

The panel stays (`PanelState`: ops, versions, undo, SSE). Its meaning changes:

- `Block` gets an optional `ref: {object, view, params}`. A block with a `ref` is a **live view**: when the
  object gets a new version, the engine recomputes the view and emits `update`. Blocks keep their last
  payload, so the report renders immediately after a restart.
- **Frozen** blocks (no `ref`) are snapshots. `freeze` turns a live block into a frozen one, or into a
  `table` or `graph` object if a name is given.
- Free-standing notes become `text` objects shown through a `ref`.
- Pinning a block is a report-layout concern only. Protection of state is the job of object locking.
- Existing blocks without `ref` load as frozen, so no migration is needed.

## 6. Dispatcher and journal

**Invariant: nothing mutates the workfile except `dispatch(call, actor)`.** Agent tool calls, UI actions,
edited code cells, `POST /tools/{name}`, `POST /code`, the `qx` client in the kernel and MCP clients all go
through the dispatcher. A *call* is one `qx` operation: a tool name plus validated arguments. It:

1. validates the arguments and resolves names;
2. applies the **policy**: lock checks, local-only mode, and the autonomy mode for agent actors (E7·18);
   in supervised mode an agent call becomes a *proposal* instead of running;
3. executes the tool or proc in the worker thread;
4. writes a journal entry and emits SSE events (objects, report).

```
_qtrix_journal(
  seq BIGINT PRIMARY KEY, ts TIMESTAMP,
  actor VARCHAR,          -- user | agent:<name> | mcp | api
  channel VARCHAR,        -- chat | ui | code | api | mcp | replay
  kind VARCHAR,           -- api (one qx call) | python (free cell run in the kernel)
  code VARCHAR,           -- canonical Python: the qx call, or the cell source
  tool VARCHAR, args JSON,-- for kind = api
  mutating BOOLEAN,
  affected JSON,          -- [{name, version_before, version_after}]
  ok BOOLEAN, error VARCHAR, duration_ms INTEGER,
  turn_id VARCHAR,        -- chat turn that produced it
  cell_id VARCHAR,        -- chat or notebook cell it belongs to
  edited_from BIGINT,     -- seq of the entry this one re-runs after a user edit
  proposal_id VARCHAR     -- set when it came from an approved proposal
)
```

- Read-only calls (views, `show`) are journaled with `mutating = false`. They are useful history but are
  skipped on replay.
- **Replay** rebuilds a workfile from its source snapshots and the mutating entries in order, then checks
  every object summary against the recorded one. It powers reproducibility tests, `.qtx` portability and
  the exports (E7·31).
- **Undo** per call restores the previous object versions and page data listed in `affected`. It sits beside
  the report's own undo and the two do not conflict: one is state, the other is layout.
- The **session as code**: the mutating journal, printed in order, *is* a runnable Python script (E7·31).

## 7. Python as the internal language

### 7.1 The `qx` API

`qtrix` is an installable Python package. Its public API is generated from the tool registry, one function
or method per tool, so it is identical for the agent, the UI, the kernel and external scripts:

```python
import qtrix as qx

macro = qx.page("macro_q")
macro.structure(kind="time_series", freq="Q", date="obs")
macro.smpl("1990q1", "2008q4")
macro.genr("infl", "400 * dlog(cpi)")

eq_taylor = qx.equation(
    "eq_taylor", page="macro_q", method="ls",
    formula="r ~ r(-1) + infl + gap",
    cov=qx.HAC(lags=4),
)
eq_taylor.view("resid_correlogram", lags=12)
eq_taylor.view("wald", "infl = 1.5")
eq_taylor.forecast("rf", start="2009q1", end="2010q4", dynamic=True)

survey = qx.page("wages")                         # cross-section: no dates, no lags
eq_mincer = qx.equation("eq_mincer", page="wages", formula="np.log(wage) ~ educ + exper + I(exper**2)",
                        cov="HC1")
```

| Area | API (v1) |
|---|---|
| Workfile | `qx.import_file()` · `qx.page()` · `page.structure()` · `page.smpl()` · `page.frame()` |
| Series | `page.genr()` · `qx.group()` |
| Objects | `qx.obj()` · `.rename()` · `.copy()` · `.delete()` · `.lock()` · `.unlock()` · `.versions()` |
| Estimation | `qx.equation(name, formula, method=…)` · `obj.estimate(**changes)` (re-estimate) · `qx.var()` · `qx.system()` |
| Views and procs | `obj.view(name, *args, **params)` · `obj.<proc>()` (`forecast`, `makeresid`, `hpf`…) · `obj.freeze()` |
| Report | `qx.text()` · `qx.report.add(obj, view)` · `qx.report.remove(block)` |

Formulas stay **strings** in the qtrix expression language (section 3), so a formula reads the same in
code, in the object window and in the chat.

### 7.2 Two kinds of cells

Code shown to and edited by the user is organised in **cells**: chat cells (one per agent action), the
object window's "Code" tab, and a session notebook. A cell is classified when it is submitted:

- **API cell**: only `qx` calls with literal arguments, assignments of their results, and references to
  object names. qtrix **parses it with `ast` and never executes it**. Each call becomes a dispatcher call
  with a spec, so structured state is preserved. This is the path for nearly everything the agent does,
  and it is safe to run without a kernel.
- **Python cell**: anything else (pandas, numpy, statsmodels, loops, functions). It runs in the
  **project kernel** (7.4). It reads data through `page.frame()` and brings results back through the API:
  `qx.series_from(...)`, or `qx.register("eq_custom", result, code=cell)` for a statsmodels/linearmodels
  result. Objects registered this way get `method = "python"` and the cell source as their spec. They are
  rebuilt by re-running the cell, and they get the standard views that fit the result's type.

### 7.3 Round-trip: generated code is editable

- Each agent action appears in the chat as a **collapsible Python cell** with the code that ran, its result
  (table, chart, diagnostics) and buttons for Edit, Run again and Open object.
- **Spec → code** is a canonical printer: deterministic, formatted, one call per action. **Code → spec** is
  the API-cell parser. Together they round-trip, so an edited cell is not text detached from the object: it
  is the object's new spec.
- Editing a formula in a cell (or in the object window) and running it creates a **new version** of the same
  object (journal `edited_from`). Dependants become stale, and live report blocks refresh.
- If an edit turns an API cell into a Python cell (for example the user adds a pandas step), it runs in the
  kernel and the object is re-registered as `method = "python"`. qtrix says so in the cell, because the
  object loses its structured spec.
- The agent sees user edits in its next turn ("the user changed `eq_taylor` from … to …") and must build on
  them, not revert them.

### 7.4 Project kernel

- One **IPython kernel per project**, started on demand through `jupyter_client` (a separate process, with
  interrupt and restart). Using a standard kernel gives rich output, and `.ipynb` export comes for free.
- DuckDB allows one writing process, held by the engine, so the kernel does not open the workfile. Its
  `qx` runs in **client mode**: calls go to the engine over loopback HTTP with the session token, and frames
  travel as Arrow. The same client mode lets a user drive qtrix from their own Jupyter or scripts.
- Agent-generated **Python cells** are gated by policy. They need approval unless the user enabled "allow
  the agent to run code" in autonomous mode. They have a timeout, can be interrupted from the chat, and
  their source is always shown before or alongside the result.
- The kernel runs with the user's privileges and is not a security sandbox. This is documented, and the
  privacy audit (#210) covers it. qtrix never sends kernel data to the LLM except the outputs the agent
  explicitly reads.

## 8. Agent contract

- **Act through the `qx` API.** Each tool the agent sees is one `qx` call, and its result shows in the chat
  as the equivalent Python. For something the API cannot express, the agent may submit a Python cell
  (`py_run`), under policy (7.4). No tool writes to the workfile directly.
- **Write code the user can own.** Generated code is idiomatic, minimal and uses meaningful names. The
  agent prefers the API over free Python, so that objects keep structured specs.
- **See the workfile, not the data.** Each turn the agent gets a compact snapshot: pages (structure, active
  sample), objects (name, type, method or short formula, version, status, locked, author), and the object or
  view selected in the UI. When objects are many, it is a summary by type, and `obj_list` / `obj_get` /
  `obj_view` give detail on demand. Raw rows never enter a prompt, which keeps the privacy guarantee of
  ADR 0001.
- **Refer to things by name.** `@eq01`, `@eq01.resid_correlogram` and `@gdp` mentions in chat resolve to
  typed references.
- **Prefer versions over duplicates.** To change a specification, re-estimate the same name or `copy` it
  with a `parent`. Do not create `eq07` through `eq19` for small tweaks.
- **Respect locks and policy.** Lock and policy violations come back as `ModelRetry` with the reason. The
  agent must not work around them by creating copies under new names.
- **Evidence before conclusions.** A `text` conclusion that cites an object in `fail` or `stale` must say so.
  Playbooks enforce this (E7·17).

## 9. Impact on the current code

| Module | Change |
|---|---|
| `workspace.py` | `_qtrix_datasets` → `_qtrix_pages` (structure columns, `data_version`); source snapshots; regularisation; frame access returns an indexed frame restricted to the sample; series metadata |
| new `workfile/` | `structure.py` (frequency detection, regularisation, `PeriodIndex`), `sample.py` (smpl grammar, mask) |
| new `expr.py` | expression language: EViews operators → patsy functions aware of frequency and panel |
| `context.py` | `ResultStore` → `ObjectStore` (DuckDB tables + fitted LRU). `results.get("r_…")` stays as a compatibility alias that maps old ids to names for one release |
| new `objects/` | `specs.py` (Pydantic specs), `store.py`, `estimators.py` (pure `estimate()` per method, from today's tool bodies), `views.py` (view and proc registries) |
| new `journal.py`, `dispatch.py` | dispatcher, policy hook, journal |
| new `qtrix` package (`qx/`) | public API generated from the registry; canonical code printer (spec → Python); API-cell parser (`ast` → calls, no exec); client mode (HTTP + Arrow) for the kernel and external scripts |
| new `kernel/` | per-project IPython kernel manager (`jupyter_client`): start, interrupt, restart, run a cell, capture rich output; `qx.register` / `qx.series_from` bridge |
| `tools/*.py` | tools build a spec, call `dispatch`, return the `llm` view. `publish_regression` becomes "save object + add live block". `ols_diagnostics` is split into views. `ui_add_note` creates `text` objects |
| `tools/registry.py` | `ToolSpec` gains `mutating`, `kinds` (page kinds supported), `object_type` and `qx` (the API name and method it is exposed as); `NAMESPACES` gains `obj` and `py` |
| `panel.py` | `Block.ref`; refresh live blocks on `object_version` events; `freeze` |
| `agents/lead.py` | `workspace_context()` → workfile snapshot; instructions rewritten around objects, names, versions and editable code; user edits since the last turn are injected; the `_wrap` error mapping adds `LockedObjectError` and `PolicyDenied` |
| `api.py` | `/workfile`, `/workfile/sample`, `/objects`, `/objects/{name}`, `/objects/{name}/views/{view}`, `/objects/{name}/versions`, `/objects/{name}/code`, `POST /code` (submit a cell), `/kernel/*`, `/journal`, SSE `objects` stream; `POST /tools/{name}` routed through the dispatcher with `actor=api` |
| `mcp_server.py` | tools registered through the dispatcher with `actor=mcp`; instructions mention object names |
| UI | chat cells with editable Python (Monaco/CodeMirror) and formula fields; workfile browser, object window with a "Code" tab, Python console and session notebook (E7·09–11); `blocks.tsx` renderers unchanged |

Schema changes are versioned in a new `_qtrix_meta(schema_version)` table, with forward-only migrations run
at startup. Existing datasets become `cross_section` pages, or `panel` pages when they already declare entity and time
columns. Existing panel blocks become frozen blocks. There are no
stored results to migrate, because `ResultStore` was memory-only.

## 10. Alternatives considered

- **Keep the result store and pickle fitted objects.** Rejected: pickles break across library versions,
  are unsafe to load from shared project files, and are large. Specs are small and auditable.
- **Make the panel the store of results.** Rejected: it mixes layout with state, so blocks cannot be
  re-estimated, and the agent loses its stable handles.
- **An EViews-like command language** (`equation eq01.ls y c x(-1)`). Rejected: it is one more language
  for users and LLMs to learn, cannot express what the API does not anticipate, and still needs a translation
  to Python for export. Python is what applied economists increasingly know, what LLMs write best, and what
  runs everywhere. From EViews we keep the object model and the time operators inside formula strings.
- **Free Python only, like a notebook.** Rejected: if every action were opaque code, objects would lose
  their specs, so no re-estimation by spec, diagnostic status, version diffs or menus generated from
  structure. API cells keep all of that and are still plain Python.
- **Executing API cells too.** Rejected: parsing them with `ast` is safer (no code execution for the
  common path), works without a kernel, and gives exact specs instead of side effects.
- **One table per series (EViews-like storage).** Rejected: DuckDB's columnar tables already store each
  series separately, and wide tables keep SQL, imports and exports simple.
- **Silent auto re-estimation when data change.** Rejected: results that change by themselves break trust
  and reproducibility. Objects go `stale` and someone (user or agent) re-estimates explicitly, which is
  journaled.

## 11. Consequences

- Every new econometric feature is a pure estimator, a spec, and some views or procs. The UI menus, the
  `qx` API, the code printer and the agent tools come for free.
- The user is never locked in: everything the agent did is Python they can edit, re-run, or take away and
  run without qtrix's UI.
- The journal makes agent work auditable and replayable. "What did the agent do?" has a precise answer, and
  exports to `.py` or `.ipynb` derive from it.
- Restart-safe state is a prerequisite for `.qtx` projects (E7·30), and this ADR provides it.
- Cost: a dispatcher indirection on every call, a second copy of imported data (source snapshots), and a
  migration of every existing tool. Tools are migrated one namespace at a time, with the compatibility
  alias for `r_` ids in between.
- Agents need fewer tokens of context per turn (names and statuses instead of block lists), and they can
  be evaluated on behaviour: did they check, re-specify and cite? E7·32 measures this.

## 12. Order of implementation

E7·02 structure → E7·03 sample → E7·04 expressions → E7·05 objects → E7·06 journal → E7·07 `qx` API,
printer and parser → E7·07b kernel → E7·08 views and report → E7·08b chat with editable code cells.
These complete the model and the chat-first experience. The EViews-style UI (E7·09–11), the agent contract
(E7·12) and the agentic loop (E7·17–20) build on them.

## Open questions (resolved in the issue that needs them)

1. Intraday frequencies and business calendars with holidays (E7·02 or later).
2. Whether `genr` series are **materialised** (stored values) or **formula-linked** (recomputed when inputs
   change, like EViews `frml`). Proposal: materialised by default, plus an `frml` variant (E7·04).
3. Retention of the version history in large projects: keep all, or compact after N (E7·30).
4. How a proposal that the user approves and then edits is shown in the journal (E7·18).
5. Whether `qx` names follow EViews (`genr`, `smpl`) or Pythonic names (`generate`, `set_sample`) with
   aliases (E7·07). Proposal: Pythonic names, with EViews aliases for discoverability.
6. Which Python packages the kernel environment ships with beyond the engine's (E7·07b, #208).
