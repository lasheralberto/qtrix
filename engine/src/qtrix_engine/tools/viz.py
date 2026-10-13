"""viz_* tools: Plotly charts rendered as blocks of the dynamic panel."""

from __future__ import annotations

from typing import Any, Literal

from ..context import get_context
from ..objects.views import compute, show
from ..panel import Block
from ._format import figure_data
from .registry import Qx, tool


def _add_figure(fig: Any, title: str, code: str, result_id: str | None = None,
                replace_block_id: str | None = None) -> dict[str, Any]:
    ctx = get_context()
    fig.update_layout(margin={"l": 40, "r": 20, "t": 30, "b": 40})
    data = figure_data(fig)
    if replace_block_id:
        block = ctx.panel.update(replace_block_id, title=title, data=data, code=code)
    else:
        block = ctx.panel.add(Block(kind="plotly_figure", title=title, data=data, code=code, result_id=result_id))
    return {"block_id": block.id, "title": title}


@tool("viz", requires=("plotly",), qx=Qx("page", "scatter", ("x", "y")))
def viz_scatter(dataset_id: str, x: str, y: str, color: str | None = None, trendline: bool = True,
                log_x: bool = False, log_y: bool = False, replace_block_id: str | None = None) -> dict[str, Any]:
    """Scatter plot of y against x with an optional least-squares fit line.

    Args:
        dataset_id: Dataset id or name.
        x: Column on the horizontal axis.
        y: Column on the vertical axis.
        color: Optional grouping column.
        trendline: Overlay a least-squares line per group.
        log_x: Log scale on x.
        log_y: Log scale on y.
        replace_block_id: Update this existing panel block instead of adding a new one.
    """
    import numpy as np
    import plotly.express as px

    ctx = get_context()
    info = ctx.workspace.get(dataset_id)
    cols = list(dict.fromkeys(c for c in (x, y, color) if c))
    df = ctx.workspace.df(dataset_id, cols).dropna()
    if color and df[color].nunique() <= 12:
        df[color] = df[color].astype(str)  # few distinct values: discrete legend, one fit line per group
    fig = px.scatter(df, x=x, y=y, color=color, log_x=log_x, log_y=log_y, opacity=0.6)
    if trendline:
        groups = df.groupby(color) if color else [(None, df)]
        for name, g in groups:
            if len(g) < 2:
                continue
            gx = np.log(g[x]) if log_x else g[x]
            gy = np.log(g[y]) if log_y else g[y]
            slope, intercept = np.polyfit(gx, gy, 1)
            xs = np.linspace(gx.min(), gx.max(), 50)
            ys = intercept + slope * xs
            fig.add_scatter(x=np.exp(xs) if log_x else xs, y=np.exp(ys) if log_y else ys, mode="lines",
                            name=f"fit {name}" if name is not None else f"fit (slope={slope:.3g})")
    code = f"import plotly.express as px\npx.scatter(qtrix.load({info.name!r}), x={x!r}, y={y!r}, color={color!r})"
    return _add_figure(fig, f"{y} vs {x}", code, replace_block_id=replace_block_id)


@tool("viz", requires=("plotly",), qx=Qx("page", "histogram", ("column",)))
def viz_histogram(dataset_id: str, column: str, by: str | None = None, bins: int = 40,
                  log_x: bool = False, replace_block_id: str | None = None) -> dict[str, Any]:
    """Histogram (distribution) of a column, optionally overlaid by group."""
    import plotly.express as px

    ctx = get_context()
    info = ctx.workspace.get(dataset_id)
    df = ctx.workspace.df(dataset_id, [c for c in (column, by) if c])
    fig = px.histogram(df, x=column, color=by, nbins=bins, log_x=log_x, barmode="overlay", opacity=0.7)
    code = f"px.histogram(qtrix.load({info.name!r}), x={column!r}, color={by!r}, nbins={bins})"
    return _add_figure(fig, f"Distribution of {column}", code, replace_block_id=replace_block_id)


@tool("viz", requires=("plotly",), qx=Qx("page", "line", ("y",)))
def viz_line(dataset_id: str, y: list[str], x: str | None = None, group: str | None = None,
             replace_block_id: str | None = None) -> dict[str, Any]:
    """Line chart of one or more series. On time series and dated panels x defaults to the dates (and group
    to the entity in panels); on other datasets pass x explicitly.

    Args:
        dataset_id: Dataset id or name.
        y: Series to plot.
        x: Horizontal axis column; omit to use the dataset's dates.
        group: One line per value of this column (one series only).
        replace_block_id: Update this existing panel block instead of adding a new one.
    """
    import plotly.express as px

    ctx = get_context()
    info = ctx.workspace.get(dataset_id)
    if x is None:
        if not info.dated:
            raise ValueError(f"'{info.name}' is a {info.structure_text}: pass x (the column for the horizontal "
                             "axis) or declare dates with data_set_structure")
        if info.kind == "panel" and group is None and len(y) == 1:
            group = info.id_col
        cols = list(dict.fromkeys([*y, *([group] if group else [])]))
        df = ctx.workspace.frame(dataset_id, cols).reset_index(level="obs")
        x = "obs"
    else:
        df = ctx.workspace.df(dataset_id, list(dict.fromkeys([x, *y, *([group] if group else [])]))).sort_values(x)
    fig = px.line(df, x=x, y=y[0], color=group) if group else px.line(df, x=x, y=y)
    code = f"px.line(qtrix.frame({info.name!r}).reset_index(), x={x!r}, y={y!r})"
    return _add_figure(fig, f"{', '.join(y)} over {'time' if x == 'obs' else x}", code,
                       replace_block_id=replace_block_id)


@tool("viz", requires=("plotly",), qx=Qx("obj", "coefplot"))
def viz_coefplot(result_id: str, exclude_intercept: bool = True) -> dict[str, Any]:
    """Coefficient plot with 95% confidence intervals for a fitted model (a live view: it follows re-estimates)."""
    block = show(result_id, "coefplot", {"exclude_intercept": exclude_intercept})
    return {"block_id": block.id, "title": block.title}


@tool("viz", requires=("plotly", "scipy"), qx=Qx("obj", "residual_plot", ("kind",)))
def viz_residuals(result_id: str, kind: Literal["fitted", "qq", "histogram"] = "fitted") -> dict[str, Any]:
    """Residual diagnostics chart for a fitted regression: residuals vs fitted, normal Q-Q or histogram
    (a live view: it follows re-estimates)."""
    result, _ = compute(result_id, "residuals", {"kind": kind})
    block = show(result_id, "residuals", {"kind": kind})
    return {"block_id": block.id, "title": block.title, **(result.llm or {})}
