"""ui_* tools: let agents curate the dynamic analysis panel (notes, removal, ordering)."""

from __future__ import annotations

from typing import Any

from ..context import get_context
from ..objects.views import show
from ..panel import Block
from .registry import Qx, tool


@tool("ui", qx=Qx("report", "blocks"))
def ui_list_blocks() -> list[dict[str, Any]]:
    """List the blocks currently shown in the analysis panel (id, kind, title, pinned, result_id), in order."""
    return [
        {"id": b.id, "kind": b.kind, "title": b.title, "pinned": b.pinned, "result_id": b.result_id,
         "live": b.ref is not None, "view": (b.ref or {}).get("view"), "stale": b.stale}
        for b in get_context().panel.blocks()
    ]


@tool("ui", qx=Qx("report", "add", ("name", "view")))
def ui_add_view(name: str, view: str | None = None, params: dict[str, Any] | None = None) -> dict[str, Any]:
    """Add a live view of an object to the report (e.g. eq01's "residuals"): it refreshes whenever the object is
    re-estimated. Omit `view` for the object's main view; if that view is already in the report, nothing is added.

    Args:
        name: Object name.
        view: View name (obj_views lists them), e.g. estimation, coefplot, residuals, diagnostics, irf.
        params: View parameters, e.g. {"kind": "qq"} for residuals.
    """
    block = show(name, view, params)
    return {"block_id": block.id, "title": block.title}


@tool("ui", qx=Qx("report", "freeze", ("block_id",)))
def ui_freeze_block(block_id: str) -> dict[str, Any]:
    """Freeze a live block: it keeps its current content and stops following its object's new versions.
    To keep a snapshot as a named table or graph object, use obj_freeze instead."""
    block = get_context().panel.freeze(block_id)
    return {"block_id": block.id, "frozen": True}


@tool("ui", qx=Qx("report", "add_note", ("title", "markdown")))
def ui_add_note(title: str, markdown: str, replace_block_id: str | None = None) -> dict[str, Any]:
    """Add (or rewrite) a markdown block in the panel: interpretation of results, conclusions, next steps.

    Args:
        title: Short heading of the note.
        markdown: Body in Markdown; LaTeX math between $...$ is supported.
        replace_block_id: Rewrite this existing note instead of adding a new one.
    """
    panel = get_context().panel
    if replace_block_id:
        block = panel.update(replace_block_id, title=title, data={"markdown": markdown})
    else:
        block = panel.add(Block(kind="markdown", title=title, data={"markdown": markdown}))
    return {"block_id": block.id}


@tool("ui", qx=Qx("report", "add_latex", ("title", "latex")))
def ui_add_equation(title: str, latex: str, description: str | None = None) -> dict[str, Any]:
    """Show an estimating equation in LaTeX, e.g. "\\ln(wage_i) = \\beta_0 + \\beta_1 educ_i + u_i"."""
    block = get_context().panel.add(Block(kind="equation", title=title,
                                          data={"latex": latex, "description": description}))
    return {"block_id": block.id}


@tool("ui", qx=Qx("report", "remove", ("block_id",)))
def ui_remove_block(block_id: str) -> dict[str, Any]:
    """Remove a block that is no longer relevant. Pinned blocks belong to the user and cannot be removed."""
    get_context().panel.remove(block_id)
    return {"removed": block_id}


@tool("ui", qx=Qx("report", "reorder", ("order",)))
def ui_reorder(order: list[str]) -> dict[str, Any]:
    """Reorder the panel; pass every current block id in the desired order."""
    get_context().panel.reorder(order)
    return {"order": order}
