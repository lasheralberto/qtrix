"""py_* tools: Python cells in the project kernel and the results they register (ADR 0002 §7.2, §7.4; E7·07b)."""

from __future__ import annotations

from typing import Any

from ..callctx import current_actor, current_cell, replaying
from ..context import get_context
from ..kernel import CellFailed, KernelError
from ..objects.specs import PythonSpec
from ..objects.views import show
from ._format import clean, llm_view
from .registry import Qx, tool


def _kernels() -> Any:
    kernels = get_context().kernels
    if kernels is None:
        raise KernelError("This engine process has no Python kernel (it needs the HTTP API to run)")
    return kernels


@tool("py", cell=True, qx=Qx("qx", "run", ("code",)))
def py_run(code: str, timeout: int = 120) -> dict[str, Any]:
    """Run a cell of free Python in the project kernel, for what the qx tools cannot express (custom estimators,
    pandas reshaping, simulations). The kernel keeps its variables between cells and has `qx` imported.

    Read data with `qx.page("name").frame()` (respects the active sample); bring results back with
    `qx.register("eq_name", result)` for a statsmodels/linearmodels result (it becomes a named object) or
    `qx.series_from("page", "name", series)` for a new series. Prefer the qx tools whenever they can do the job:
    their objects keep structured specs. Never print raw rows of data.

    Agents may run cells only when the user allowed it; otherwise show the code to the user and ask them to run it.

    Args:
        code: The Python source of the cell.
        timeout: Seconds before the cell is interrupted.
    """
    kernels = _kernels()
    result, run = kernels.run_cell(code, actor=current_actor(), cell_id=current_cell(), timeout=timeout)
    out = {"ok": result.ok, "outputs": result.outputs, "error": result.error, "text": result.text(),
           "execution_count": result.execution_count, "duration_ms": result.duration_ms,
           "timed_out": result.timed_out, "calls": run.calls, "mutated": run.mutated, "cell_id": run.cell_id}
    if not result.ok:
        raise CellFailed(f"The cell failed: {result.error_text()}\n{result.text(1500)}", out)
    return out


@tool("py", mutating=True, internal=True, qx=Qx("qx", "register_summary", ("name", "summary")))
def py_register(name: str, summary: dict[str, Any], code: str | None = None, page: str | None = None,
                title: str = "") -> dict[str, Any]:
    """Store a result computed in a Python cell as a named object (method "python"); `qx.register` calls this.

    The cell is the object's spec: re-estimating or replaying the object re-runs it in a fresh kernel.
    """
    ctx = get_context()
    if not code:
        raise ValueError("qx.register needs the cell that computes the result: pass code=... when calling it from "
                         "outside a kernel cell")
    spec = PythonSpec(page=ctx.workspace.get(page).name if page else "", code=code,
                      registered_as=name.strip().lower(), title=title)
    if replaying():  # prove the result is reproducible: re-run its cell instead of trusting the stored summary
        summary = _kernels().rebuild(code, spec.registered_as)
    obj = ctx.objects.register(spec, clean(summary), name)
    out = {"name": obj.name, "version": obj.version}
    if obj.summary.get("coefficients"):
        out["block_id"] = show(obj.name, "estimation").id
    return {**out, **llm_view(obj.summary)} if obj.summary.get("coefficients") else out
