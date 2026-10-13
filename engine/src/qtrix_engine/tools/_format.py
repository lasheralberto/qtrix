"""Helpers to turn library objects into JSON-friendly summaries and panel blocks."""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd


def clean(value: Any) -> Any:
    """Recursively convert numpy/pandas scalars and NaN/inf into JSON-safe Python values."""
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [clean(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def df_block_data(df: pd.DataFrame, max_rows: int = 200) -> dict[str, Any]:
    shown = df.head(max_rows)
    return {
        "columns": [str(c) for c in shown.columns],
        "rows": clean(json.loads(shown.to_json(orient="values", date_format="iso"))),
        "total_rows": int(len(df)),
        "index": [str(i) for i in shown.index] if not isinstance(shown.index, pd.RangeIndex) else None,
    }


def figure_data(fig: Any) -> dict[str, Any]:
    return json.loads(fig.to_json())


def coef_rows(params: pd.Series, bse: pd.Series, tvalues: pd.Series, pvalues: pd.Series,
              conf_int: pd.DataFrame, rename: Callable[[str], str] | None = None) -> list[dict[str, Any]]:
    """Coefficient rows; ``rename`` shows generated term names as the user wrote them."""
    rows = []
    for name in params.index:
        rows.append(clean({
            "name": rename(str(name)) if rename else str(name),
            "coef": float(params[name]),
            "se": float(bse[name]),
            "stat": float(tvalues[name]),
            "p": float(pvalues[name]),
            "ci_low": float(conf_int.loc[name].iloc[0]),
            "ci_high": float(conf_int.loc[name].iloc[1]),
        }))
    return rows


def stars(p: float | None) -> str:
    if p is None:
        return ""
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.1 else ""


def llm_view(model: dict[str, Any], max_coefs: int = 25) -> dict[str, Any]:
    """Compact summary handed back to the LLM (the full table goes to the panel)."""
    coefs = model.get("coefficients", [])[:max_coefs]
    return {
        **{k: v for k, v in model.items() if k != "coefficients"},
        "coefficients": {
            c["name"]: f"{c['coef']:.4g} (se {c['se']:.3g}, p={c['p']:.3g}){stars(c['p'])}"
            for c in coefs if c.get("coef") is not None and c.get("se") is not None and c.get("p") is not None
        },
    }
