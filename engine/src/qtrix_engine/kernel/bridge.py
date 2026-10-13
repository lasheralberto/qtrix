"""Engine side of client-mode ``qx`` (E7·07b): a call from the kernel or an external client, to the dispatcher.

A call made inside a kernel cell runs as the cell's author (the user or an agent), on channel ``kernel``, linked to
the cell by ``cell_id``; the run records its journal entries. A call from an external client (the user's Jupyter, a
script) runs as the user on channel ``client``. During a rebuild nothing is changed: mutating calls are skipped and
the summary ``qx.register`` hands over for the rebuilt name is captured.
"""

from __future__ import annotations

from typing import Any

from ..dispatch import Call, PolicyDenied, dispatch
from ..journal import KERNEL_CHANNEL
from ..tools.registry import get_tool
from .manager import KernelRun


def dispatch_from_kernel(run: KernelRun | None, call: Call) -> Any:
    spec = get_tool(call.tool)
    if run is None:
        return dispatch(call, "user", "client")
    if spec is not None and spec.cell:
        raise PolicyDenied("A Python cell cannot run another cell: put the code in this one")
    if run.rebuild is not None:
        return _during_rebuild(run, call)
    args = dict(call.args)
    if call.tool == "py_register" and not args.get("code"):
        args["code"] = run.code  # the cell that computed the result is its spec
    try:
        result, seq = dispatch(Call(call.tool, args), run.actor, KERNEL_CHANNEL, cell_id=run.cell_id, with_seq=True)
    except Exception as exc:
        if getattr(exc, "journal_seq", None) is not None:
            run.calls.append(exc.journal_seq)
        raise
    run.calls.append(seq)
    if spec is not None and spec.is_mutating(args):
        run.mutated = True
    return result


def _during_rebuild(run: KernelRun, call: Call) -> Any:
    if call.tool == "py_register" and str(call.args.get("name", "")).strip().lower() == run.rebuild:
        run.captured = call.args.get("summary")
        return {"name": run.rebuild, "rebuilt": True}
    spec = get_tool(call.tool)
    if spec is None or spec.is_mutating(call.args) or spec.namespace in ("viz", "ui"):
        return None  # a rebuild leaves the workfile and the report as they are
    return dispatch(call, "user", "replay", journal=False)
