"""vol_* tools: volatility models (GARCH family) on time-series pages."""

from __future__ import annotations

from typing import Any, Literal

from ..context import get_context
from ..objects.specs import GarchSpec
from ..objects.views import show
from ._format import clean
from .ols import publish, resolved_sample
from .registry import Qx, tool

LABELS = {"garch": "GARCH", "egarch": "EGARCH", "gjr": "GJR-GARCH"}


@tool("vol", requires=("arch",), kinds=("time_series",), mutating=True,
      qx=Qx("qx", "equation", ("name",), family="equation"))
def vol_garch(dataset_id: str, column: str, model: Literal["garch", "egarch", "gjr"] = "garch", p: int = 1,
              q: int = 1, o: int = 0, mean: Literal["constant", "zero", "ar"] = "constant",
              dist: Literal["normal", "t", "skewt", "ged"] = "normal", forecast_steps: int = 0,
              sample: str | None = None, name: str | None = None) -> dict[str, Any]:
    """Fit a GARCH, EGARCH or GJR-GARCH volatility model (arch); table and conditional-volatility chart go to
    the panel. Typically applied to returns (e.g. dlog of a price). Stored as a named equation object;
    `name` re-estimates an existing one as a new version.

    Args:
        dataset_id: Dataset id or name (must be a time series).
        column: Return series (not the price level).
        model: garch, egarch (log-variance, asymmetric) or gjr (threshold asymmetry).
        p: ARCH order (lagged squared shocks).
        q: GARCH order (lagged variances).
        o: Asymmetry order; gjr uses 1 when left at 0.
        mean: Mean model: constant, zero or ar (AR(1)).
        dist: Innovation distribution: normal, t, skewt or ged.
        forecast_steps: Periods of volatility forecast (standard deviation) after the sample.
        sample: Sample for this call only, e.g. "1990q1 2008q4"; omit for the dataset's active sample.
        name: Object name (e.g. eq_garch). An existing name is re-estimated as a new version.
    """
    ctx = get_context()
    spec = GarchSpec(page=ctx.workspace.get(dataset_id).name, sample=resolved_sample(dataset_id, sample),
                     column=column, model=model, p=p, q=q, o=o, mean=mean, dist=dist,
                     forecast_steps=forecast_steps)
    obj, out = publish(spec, name, title=f"{LABELS[model]} · {column}")
    show(obj.name, "volatility")
    if "volatility_forecast" in obj.summary:
        out["volatility_forecast"] = obj.summary["volatility_forecast"]
    out["persistence"] = obj.summary.get("persistence")
    return clean(out)
