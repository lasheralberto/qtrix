"""obj_* tools: inspect and manage the named objects of the workfile (ADR 0002 §3).

Locks are enforced by the store: agents get a LockedObjectError (a ModelRetry) when they try to modify,
re-estimate, rename or delete a locked object, and cannot unlock it.
"""

from __future__ import annotations

from typing import Any

from ..context import get_context
from ..objects.specs import FrozenGraphSpec, FrozenTableSpec, GroupSpec
from ..objects.views import DEFAULT_VIEW, compute, procs_of, show, views_of
from ._format import clean, llm_view
from .registry import Qx, tool


@tool("obj", qx=Qx("qx", "objects", ("type",)))
def obj_list(type: str | None = None) -> list[dict[str, Any]]:
    """List the named objects (equations, vars, groups, tables...) with version, status and lock.

    Args:
        type: Optional filter: equation, var, group, table, graph or text.
    """
    return [o.to_dict(detail=False) for o in get_context().objects.list(type)]


@tool("obj", qx=Qx("obj", "show"))
def obj_get(name: str) -> dict[str, Any]:
    """Show one object: its spec (how to re-estimate it), compact results, status and version.

    Args:
        name: Object name (a former r_xxxx result id also works).
    """
    o = get_context().objects.get(name)
    out = o.to_dict(detail=False)
    out.update(spec=o.spec, status_detail=o.status_detail, summary=llm_view(o.summary) if o.summary.get("coefficients")
               else o.summary)
    return clean(out)


@tool("obj", qx=Qx("obj", "versions"))
def obj_versions(name: str) -> dict[str, Any]:
    """Version history of an object and the coefficient changes between its last two versions.

    Args:
        name: Object name.
    """
    store = get_context().objects
    versions = [{k: v for k, v in x.items() if k in ("version", "author", "status", "created_at")}
                | {"sample": x["spec"].get("sample"), "formula": x["spec"].get("formula")}
                for x in store.versions(name)]
    out: dict[str, Any] = {"name": store.get(name).name, "versions": versions}
    if len(versions) > 1:
        out["last_diff"] = store.diff(name)
    return clean(out)


@tool("obj", qx=Qx("obj", "views"))
def obj_views(name: str) -> dict[str, Any]:
    """What can be done with an object: its views (estimation output, residual graphs, diagnostics, impulse
    responses...) with their parameters, and its procs (actions such as estimate again, freeze, copy).

    Args:
        name: Object name.
    """
    obj = get_context().objects.get(name)
    return {"name": obj.name, "type": obj.type, "method": obj.method, "default_view": DEFAULT_VIEW.get(obj.type),
            "views": [v.describe() for v in views_of(obj)], "procs": [p.describe() for p in procs_of(obj)]}


@tool("obj", qx=Qx("obj", "view", ("view",), varkw="params"))
def obj_view(name: str, view: str | None = None, params: dict[str, Any] | None = None,
             show_in_panel: bool = True) -> dict[str, Any]:
    """Compute a view of an object (see obj_views), e.g. "residuals" with {"kind": "qq"} or "diagnostics".
    With show_in_panel the report gets a live block that refreshes when the object is re-estimated.

    Args:
        name: Object name.
        view: View name; omit for the object's main view.
        params: View parameters.
        show_in_panel: Add the view to the report (once: the same view is never duplicated).
    """
    result, live_ref = compute(name, view, params)
    out: dict[str, Any] = {"name": live_ref["object"], "view": live_ref["view"], "title": result.title}
    if show_in_panel:
        out["block_id"] = show(name, view, params).id
    if result.outcome:
        out["outcome"] = result.outcome
    return clean({**out, **(result.llm or {})})


@tool("obj", mutating=True, qx=Qx("obj", "freeze", ("view", "new_name")))
def obj_freeze(name: str, view: str | None = None, new_name: str | None = None,
               params: dict[str, Any] | None = None) -> dict[str, Any]:
    """Freeze a view of an object as a new table or graph object (a snapshot: it never changes, whatever
    happens to the source later). Charts become graph objects (gph01...), everything else table objects (tab01...).

    Args:
        name: Object whose view is frozen.
        view: View name; omit for the object's main view.
        new_name: Name of the frozen object; omit for a default name.
        params: View parameters.
    """
    ctx = get_context()
    result, live_ref = compute(name, view, params)
    source, view_name = live_ref["object"], live_ref["view"]
    if result.kind == "plotly_figure":
        spec: Any = FrozenGraphSpec(title=result.title, figure=result.data, source=source, view=view_name)
    else:
        spec = FrozenTableSpec(title=result.title, kind=result.kind, data=result.data, source=source, view=view_name)
    obj = ctx.objects.estimate(spec, new_name)
    return {"name": obj.name, "type": obj.type, "version": obj.version, "frozen_from": f"{source}.{view_name}",
            "block_id": show(obj.name).id}


@tool("obj", mutating=True, qx=Qx("obj", "rename", ("new_name",)))
def obj_rename(name: str, new_name: str) -> dict[str, Any]:
    """Rename an object (not allowed when it is locked). Tables and graphs built from it follow the new name.

    Args:
        name: Current name.
        new_name: New name: a letter then letters, digits or '_', at most 32 characters.
    """
    return get_context().objects.rename(name, new_name).to_dict(detail=False)


@tool("obj", mutating=True, qx=Qx("obj", "copy", ("new_name",)))
def obj_copy(name: str, new_name: str | None = None) -> dict[str, Any]:
    """Copy an object (the copy records the original as its parent). To change a specification prefer
    re-estimating the same name (a new version) over making copies.

    Args:
        name: Object to copy.
        new_name: Name of the copy; omit for a default like eq02.
    """
    return get_context().objects.copy(name, new_name).to_dict(detail=False)


@tool("obj", mutating=True, qx=Qx("obj", "delete"))
def obj_delete(name: str) -> dict[str, Any]:
    """Delete an object (its history is kept; objects built from it become stale). Locked objects cannot be deleted.

    Args:
        name: Object to delete.
    """
    return {"deleted": get_context().objects.delete(name)}


@tool("obj", mutating=True, qx=Qx("qx", "group", ("name", "members")))
def obj_group(name: str, members: list[str], dataset_id: str | None = None) -> dict[str, Any]:
    """Create or update a named group: an ordered list of series of a dataset.

    Args:
        name: Group name (e.g. gr_macro).
        members: Series names, in order.
        dataset_id: Dataset the series belong to.
    """
    ctx = get_context()
    page = ctx.workspace.get(dataset_id).name if dataset_id else ""
    return ctx.objects.estimate(GroupSpec(page=page, members=members), name).to_dict(detail=False)


@tool("obj", mutating=True, qx=Qx("obj", "lock"))
def obj_lock(name: str) -> dict[str, Any]:
    """Lock an object so that only the user can modify, re-estimate, rename or delete it.

    Args:
        name: Object to lock.
    """
    return get_context().objects.lock(name).to_dict(detail=False)


@tool("obj", mutating=True, internal=True, qx=Qx("obj", "unlock"))
def obj_unlock(name: str) -> dict[str, Any]:
    """Unlock an object (user only: agents and API clients are refused by the store)."""
    return get_context().objects.unlock(name).to_dict(detail=False)


@tool("obj", mutating=True, qx=Qx("obj", "estimate", aliases=("reestimate",), varkw="changes"))
def obj_reestimate(name: str, changes: dict[str, Any]) -> dict[str, Any]:
    """Re-estimate an object as a new version with some spec fields changed (sample, cov_type, formula...).

    Args:
        name: Object to re-estimate.
        changes: Spec fields to replace, e.g. {"sample": "1990q1 2000q4"} or {"cov_type": "HC3"}.
    """
    return clean(get_context().objects.reestimate(name, **changes).to_dict(detail=False))
