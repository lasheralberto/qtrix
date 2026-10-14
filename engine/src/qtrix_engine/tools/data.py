"""data_* tools: catalog, descriptives, SQL, transformations."""

from __future__ import annotations

from typing import Any, Literal

from ..context import get_context
from ..objects.views import add_artifact
from ._format import clean, df_block_data
from .registry import Qx, tool


@tool("data", qx=Qx("qx", "pages"))
def data_list_datasets() -> list[dict[str, Any]]:
    """List the datasets in the local workspace with their columns, types and structure
    (cross-section, time series or panel) plus any structure suggested at import."""
    return [
        {"id": d.id, "name": d.name, "n_rows": d.n_rows, "columns": d.columns, "structure": d.structure_text,
         "kind": d.kind, "freq": d.freq, "id_col": d.id_col, "date_col": d.date_col,
         "suggestion": d.suggestion, "source": d.source}
        for d in get_context().workspace.list()
    ]


@tool("data", mutating=True, qx=Qx("page", "structure", ("kind",)))
def data_set_structure(dataset_id: str, kind: Literal["cross_section", "time_series", "panel", "panel_undated"],
                       freq: Literal["A", "S", "Q", "M", "W", "D5", "D7"] | None = None,
                       date_col: str | None = None, id_col: str | None = None, t_col: str | None = None,
                       start: str | None = None) -> dict[str, Any]:
    """Declare what a dataset is. Only declare time when the data really are dated: surveys and other
    cross-sections stay 'cross_section' (the default).

    Dated structures get one row per period (or entity x period) over the full range; missing periods become
    empty rows, and time-series tools then use real dates. Re-declaring is always possible (the original
    rows are kept).

    Args:
        dataset_id: Dataset id or name.
        kind: cross_section (no time), time_series (one row per date), panel (entity x date) or
            panel_undated (entity x integer index, e.g. survey wave).
        freq: A annual, S semiannual, Q quarterly, M monthly, W weekly, D5/D7 daily. Inferred from date_col
            when omitted.
        date_col: Column with dates or period labels (2020-03-31, 2020Q1, 2020M03, 2020).
        id_col: Entity column for panels (optional row id for cross-sections).
        t_col: Integer time index for panel_undated.
        start: First period (e.g. '1990Q1') for a time series whose rows are in time order without a date column.
    """
    ws = get_context().workspace
    info = ws.set_structure(dataset_id, kind, freq=freq, date_col=date_col, id_col=id_col, t_col=t_col, start=start)
    return ws.describe_structure(info.id)


@tool("data", mutating=True, qx=Qx("page", "set_sample", ("sample",), aliases=("smpl",)))
def data_set_sample(dataset_id: str, sample: str = "@all") -> dict[str, Any]:
    """Set the dataset's active estimation sample (EViews `smpl`). Every estimation, descriptive and chart
    tool then uses only those observations; change it again to re-estimate on another sample without
    rewriting anything. Transformations (data_transform) and SQL always see all rows.

    Syntax: `@all` | ranges | `if <condition>` | ranges + condition.
    - Ranges are pairs `start end`. On time series and dated panels they are periods ("1990q1 2008q4",
      "2000", "2001m3"); on cross-sections they are observation numbers ("1 500"); on undated panels they are
      values of the time index. `@first` / `@last` are the ends of the data; several pairs are allowed.
    - The condition is a SQL boolean over the columns: "age >= 25 and female = 1", "unemp > 5".
    Examples: "1990q1 2008q4", "@first 2000q4 if inflation < 10", "if region = 'North'", "@all".

    Args:
        dataset_id: Dataset id or name.
        sample: The sample (default "@all" resets it).
    """
    res = get_context().workspace.set_sample(dataset_id, sample)
    return {"dataset": dataset_id, **res.to_dict()}


@tool("data", qx=Qx("page", "describe"))
def data_describe_structure(dataset_id: str) -> dict[str, Any]:
    """Structure of a dataset: kind, frequency, range, entity/date columns, filled gaps and the structure
    suggested at import (if any) so it can be confirmed with data_set_structure."""
    return get_context().workspace.describe_structure(dataset_id)


@tool("data", mutating=True, qx=Qx("qx", "import_file", ("path",)))
def data_import_file(path: str, name: str | None = None, sheet: str | None = None) -> dict[str, Any]:
    """Import a local CSV, TSV, Parquet or Excel file into the workspace.

    Args:
        path: Absolute path of the file on the user's machine.
        name: Optional dataset name (defaults to the file name).
        sheet: Excel sheet name, defaults to the first sheet.
    """
    info = get_context().workspace.import_file(path, name=name, sheet=sheet)
    return {"id": info.id, "name": info.name, "n_rows": info.n_rows, "columns": info.columns}


@tool("data", mutating=True, qx=Qx("page", "set_panel", ("entity_col", "time_col")))
def data_set_panel_structure(dataset_id: str, entity_col: str, time_col: str) -> dict[str, Any]:
    """Declare the entity (individual/firm/country) and time columns of a panel dataset. Shortcut for
    data_set_structure: a dated panel when time_col holds dates or years, an undated panel otherwise."""
    info = get_context().workspace.set_panel_structure(dataset_id, entity_col, time_col)
    return {"id": info.id, "entity_col": info.entity_col, "time_col": info.time_col, "structure": info.structure_text}


@tool("data", qx=Qx("page", "summary_stats", aliases=("stats",)))
def data_summary_stats(dataset_id: str, columns: list[str] | None = None, show_in_panel: bool = True,
                       sample: str | None = None) -> dict[str, Any]:
    """Descriptive statistics (count, mean, sd, min, quartiles, max, missing) for numeric columns,
    over the dataset's active sample (or `sample`).

    Args:
        dataset_id: Dataset id or name.
        columns: Subset of columns; all numeric columns when omitted.
        show_in_panel: Add the table to the analysis panel.
        sample: Sample for this call only, e.g. "if female = 1", "@all"; omit for the active one.
    """
    ctx = get_context()
    info = ctx.workspace.get(dataset_id)
    df = ctx.workspace.df(dataset_id, columns, sample=sample)
    numeric = df.select_dtypes("number")
    desc = numeric.describe().T
    desc["missing"] = numeric.isna().sum()
    desc = desc.round(4)
    if show_in_panel:
        cols = f", columns={columns!r}" if columns else ""
        add_artifact("dataframe", f"Descriptive statistics · {info.name}",
                     df_block_data(desc.reset_index(names="variable")),
                     f"df = qtrix.load({info.name!r}{cols})\ndf.describe().T")
    return clean(desc.to_dict(orient="index"))


@tool("data", qx=Qx("page", "correlation", aliases=("cor",)))
def data_correlation(dataset_id: str, columns: list[str] | None = None,
                     method: Literal["pearson", "spearman", "kendall"] = "pearson",
                     sample: str | None = None) -> dict[str, Any]:
    """Correlation matrix of numeric columns, shown as a heatmap in the panel. Uses the active sample
    unless `sample` (e.g. "if female = 1", "@all") overrides it."""
    import plotly.express as px

    from ._format import figure_data

    ctx = get_context()
    info = ctx.workspace.get(dataset_id)
    corr = (ctx.workspace.df(dataset_id, columns, sample=sample).select_dtypes("number")
            .corr(method=method).round(3))
    fig = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r", zmin=-1, zmax=1)
    add_artifact("plotly_figure", f"{method.title()} correlations · {info.name}", figure_data(fig),
                 f"df.corr(method={method!r})")
    return clean(corr.to_dict())


@tool("data", qx=Qx("page", "preview"))
def data_preview(dataset_id: str, limit: int = 20, show_in_panel: bool = False) -> dict[str, Any]:
    """First rows of a dataset."""
    ctx = get_context()
    info = ctx.workspace.get(dataset_id)
    df = ctx.workspace.preview(dataset_id, limit=min(limit, 500))
    data = df_block_data(df)
    if show_in_panel:
        add_artifact("dataframe", f"{info.name} (first {len(df)} rows)", data)
    return {"columns": data["columns"], "rows": data["rows"][:20]}


@tool("data", mutating=lambda a: bool(a.get("save_as")), qx=Qx("qx", "sql", ("query",)))
def data_sql(query: str, save_as: str | None = None, show_in_panel: bool = True) -> dict[str, Any]:
    """Run a read-only SQL SELECT over the workspace. Each dataset is a view named after the dataset.

    Args:
        query: A single SELECT or WITH statement (DuckDB dialect).
        save_as: Store the result as a new dataset with this name.
        show_in_panel: Display the result table in the analysis panel.
    """
    ctx = get_context()
    df = ctx.workspace.sql(query)
    out: dict[str, Any] = {"n_rows": len(df), "columns": list(map(str, df.columns))}
    if save_as:
        info = ctx.workspace.add_dataframe(df, save_as, source="sql")
        out["dataset_id"] = info.id
    if show_in_panel:
        add_artifact("dataframe", save_as or "SQL result", df_block_data(df), f"qtrix.sql({query!r})")
    out["head"] = clean(df.head(10).to_dict(orient="records"))
    return out


@tool("data", mutating=True, qx=Qx("page", "generate", ("name", "expression"), aliases=("genr",)))
def data_genr(dataset_id: str, name: str, expression: str, overwrite: bool = False) -> dict[str, Any]:
    """Add a series computed from an expression (EViews `genr`), over every row of the dataset.

    Prefer this to data_transform for anything with time: lags, differences, growth rates, trends, dummies.
    The definition is remembered, so the series is recomputed if the dataset is re-structured.

    Expression: Python over the column names plus
    x(-1) / L(x, k) lag (x(+1) lead), d(x) / d(x, n, s) difference, dlog(x), pch(x), log/exp/sqrt/abs,
    movav(x, n), movsum(x, n), @trend, @seas(q), dummy("2008q3") or dummy("2008q3", "2009q2"),
    recode(cond, a, b). Lags work within each entity of a panel and need a time series, panel or undated
    panel; on a cross-section use the element-wise functions only.
    Examples: "dlog(gdp)", "100 * dlog(cpi, 1, 4)", "recode(unemp > 8, 1, 0)", "movav(rate, 4)".

    Args:
        dataset_id: Dataset id or name.
        name: New series name (letters, digits, underscore).
        expression: The expression.
        overwrite: Replace the series if it already exists.
    """
    ws = get_context().workspace
    out = ws.genr(dataset_id, name, expression, overwrite)
    info = ws.get(dataset_id)
    return {**out, "structure": info.structure_text}


@tool("data", mutating=True, internal=True, qx=Qx("page", "set_series", ("name", "values")))
def data_set_series(dataset_id: str, name: str, values: list[float | None], overwrite: bool = False) -> dict[str, Any]:
    """Add a series with values computed in a Python cell (``qx.series_from``): one value per row of the page,
    in stored order. Used by the kernel bridge; agents create series with data_genr."""
    ws = get_context().workspace
    return {**ws.set_series(dataset_id, name, values, overwrite), "structure": ws.get(dataset_id).structure_text}


@tool("data", mutating=True, qx=Qx("page", "transform", ("new_columns",)))
def data_transform(dataset_id: str, new_columns: dict[str, str], save_as: str | None = None,
                   filter: str | None = None) -> dict[str, Any]:
    """Create a NEW dataset with derived variables from SQL expressions (logs, dummies, interactions, filters).
    For lags, differences or growth rates of a time series use data_genr instead.

    Args:
        dataset_id: Source dataset id or name.
        new_columns: Mapping new_name -> DuckDB SQL expression, e.g. {"lwage": "ln(wage)",
            "wage_lag": "lag(wage) OVER (PARTITION BY id ORDER BY year)", "female": "CAST(sex = 'F' AS INT)"}.
        save_as: Name of the resulting dataset (defaults to '<name>_t').
        filter: Optional SQL WHERE condition applied to the rows.
    """
    ctx = get_context()
    info = ctx.workspace.get(dataset_id)
    exprs = ", ".join(f'{expr} AS "{name}"' for name, expr in new_columns.items())
    query = f'SELECT *, {exprs} FROM "{info.table_name}"' + (f" WHERE {filter}" if filter else "")
    df = ctx.workspace.sql(query, limit=10_000_000)
    new = ctx.workspace.add_derived(df, save_as or f"{info.name}_t", f"transform:{info.name}", like=info)
    return {"dataset_id": new.id, "name": new.name, "n_rows": new.n_rows, "added": list(new_columns),
            "structure": new.structure_text}


@tool("data", mutating=True, internal=True, qx=Qx("page", "delete"))
def data_delete_dataset(dataset_id: str) -> dict[str, Any]:
    """Delete a dataset (page). Its source snapshot is kept so that a journal replay can still rebuild it."""
    ws = get_context().workspace
    info = ws.get(dataset_id)
    ws.delete(info.id, keep_source=True)
    return {"deleted": info.name}
