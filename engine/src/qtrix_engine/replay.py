"""Undo per call and replay of the journal (ADR 0002 §6, E7·06).

Undo restores what a journal entry changed from the before-images saved when it ran: object rows with their
versions, and page state (structure, sample, series definitions, and the data of pages rewritten by
``data_set_structure`` / ``data_genr``). Replay rebuilds the workfile from the immutable ``src_<id>`` snapshots
and the mutating entries, in order, then checks every object version against the summary recorded before.

Limits: undoing a page deletion is not supported (replay rebuilds it); imported pages need their snapshot
(``src_<id>``) to replay.
"""

from __future__ import annotations

import logging
from typing import Any

from .callctx import current_recording, replaying
from .context import EngineContext, get_context
from .dispatch import Call, dispatch
from .journal import UNDO_TOOL
from .objects.estimators import summaries_match
from .panel import PanelState

log = logging.getLogger(__name__)


class UndoError(ValueError):
    """The call cannot be undone (already undone, later calls depend on it, or no before-image)."""


class ReplayError(RuntimeError):
    """The journal cannot rebuild this workfile."""


# ---- undo ------------------------------------------------------------------------------------------------------
def _touched(entry: dict[str, Any]) -> set[str]:
    return {f"{a['kind']}:{a['id'] if a['kind'] == 'page' else a['name']}" for a in entry["affected"]}


def _check_undoable(ctx: EngineContext, entry: dict[str, Any]) -> None:
    seq = entry["seq"]
    if entry["tool"] == UNDO_TOOL:
        raise UndoError(f"[{seq}] is itself an undo; redo is not supported: repeat the call instead")
    if entry["kind"] == "python":
        raise UndoError(f"[{seq}] is a Python cell: undo the calls it made one by one (cell {entry['cell_id']})")
    if not entry["mutating"] or not entry["ok"]:
        raise UndoError(f"[{seq}] {entry['tool']} did not change the workfile, so there is nothing to undo")
    undone = ctx.journal.undone()
    if seq in undone:
        raise UndoError(f"[{seq}] was already undone by [{undone[seq]}]")
    mine = _touched(entry)
    for later in ctx.journal.list(mutating=True, ok=True, since=seq):
        if later["tool"] == UNDO_TOOL or later["seq"] in undone:
            continue
        if mine & _touched(later):
            raise UndoError(f"[{seq}] cannot be undone yet: the later call [{later['seq']}] {later['tool']} changed "
                            "the same objects or pages. Undo it first")
    if any(a["kind"] == "page" and a.get("undoable") is False for a in entry["affected"]):
        raise UndoError(f"[{seq}] deleted a dataset: that is not undoable (replay the journal to rebuild it)")


def undo(seq: int) -> dict[str, Any]:
    """Undo journal entry ``seq`` (runs inside the dispatcher as the internal tool ``journal_undo``)."""
    ctx = get_context()
    entry = ctx.journal.get(seq)
    if not replaying():
        _check_undoable(ctx, entry)
    objects = [a for a in entry["affected"] if a["kind"] == "object"]
    pages = [a for a in entry["affected"] if a["kind"] == "page"]
    images = {a["name"]: ctx.journal.load_before(seq, "object", a["name"]) for a in objects}
    states = {a["id"]: ctx.journal.load_before(seq, "page", a["id"]) for a in pages if a["change"] == "changed"}
    missing = [k for k, v in {**images, **states}.items() if v is None]
    if missing:
        raise UndoError(f"[{seq}] has no before-image for {missing}")
    rec = current_recording()
    restored: list[str] = []
    for a in pages:  # pages first: objects that depended on the data become valid again below
        if a["change"] == "created":
            if rec:
                rec.pages.add(a["id"])
            try:
                ctx.workspace.delete(a["id"], keep_source=True)
            except KeyError:
                continue
            restored.append(f"page {a['name']} removed")
        else:
            if rec:
                rec.pages.add(a["id"])
            ctx.workspace.restore_page_state(states[a["id"]])
            ctx.objects.clear_stale_data(a["name"], states[a["id"]]["catalog"]["data_version"])
            restored.append(f"page {a['name']} restored")
    changed = []
    for a in objects:
        ctx.objects.restore_image(a["name"], images[a["name"]])
        changed.append(a["name"])
        restored.append(f"{a['name']} v{a['version_before']}" if a["version_before"] else f"{a['name']} removed")
        ctx.panel.mark_stale(a["name"], True)
    for name in changed:  # objects built from the ones that moved no longer match them
        for dep in ctx.objects.dependants(name):
            if dep not in changed:
                ctx.objects.mark_stale(dep, f"{name} was restored by the undo of [{seq}]; re-estimate {dep}")
    return {"undone": seq, "tool": entry["tool"], "code": entry["code"], "restored": restored}


# ---- replay ----------------------------------------------------------------------------------------------------
def _created_pages(entries: list[dict[str, Any]]) -> set[str]:
    return {a["id"] for e in entries for a in e["affected"] if a["kind"] == "page" and a["change"] == "created"}


def replay(*, verify: bool = True) -> dict[str, Any]:
    """Rebuild the workfile's pages and objects from the journal and check the objects against what was recorded.

    Everything the journal can reproduce is wiped first (objects, versions, pages), then each mutating entry is
    re-executed in order with its original actor, without writing new entries and without touching the report.
    Returns ``{ok, replayed, errors, mismatches, missing, lost, extra}`` (``extra``: objects that exist now but did
    not before, e.g. after the user deleted them outside the journal); raises ReplayError before wiping anything if the
    journal cannot rebuild the workfile (a page that did not come from it, or a missing source snapshot).
    """
    ctx = get_context()
    # A Python cell is not re-run: the calls it made are entries of their own, and a registered result re-runs its
    # cell in a fresh kernel (py_register) to check that it is reproducible.
    entries = [e for e in ctx.journal.list(mutating=True, ok=True) if e["kind"] != "python"]
    created = _created_pages(entries)
    legacy = [p.name for p in ctx.workspace.list() if p.id not in created]
    if legacy:
        raise ReplayError(f"Pages {legacy} were not created through the journal (no source snapshot is tied to an "
                          "entry), so replay cannot rebuild them")
    for e in entries:
        if e["tool"] == "data_import_file":
            for a in (x for x in e["affected"] if x["kind"] == "page" and x["change"] == "created"):
                if not a.get("snapshot") or not ctx.workspace._con.execute(
                        "SELECT 1 FROM information_schema.tables WHERE table_name = ?", [a["snapshot"]]).fetchall():
                    raise ReplayError(f"Source snapshot of '{a['name']}' is gone: replay needs the original file")

    expected: dict[tuple[str, int], Any] = {}
    for obj in {name for (name,) in ctx.workspace._con.execute("SELECT DISTINCT name FROM _qtrix_object_versions")
                .fetchall()}:
        for v in ctx.objects.versions(obj):
            expected[(obj, v["version"])] = v["summary"]
    live_before = {o.name for o in ctx.objects.list()}

    ctx.objects.reset()
    ctx.workspace.wipe_pages()
    live_panel, ctx.panel = ctx.panel, PanelState()  # tools add blocks: replay must not duplicate the report
    errors: list[dict[str, Any]] = []
    try:
        for e in entries:
            try:
                if e["tool"] == "data_import_file":
                    for a in (x for x in e["affected"] if x["kind"] == "page" and x["change"] == "created"):
                        ctx.workspace.restore_page(a["id"], a["name"], a["source"], a["table_name"], a["snapshot"],
                                                   a["created_at"])
                else:
                    ids = [a["id"] for a in e["affected"] if a["kind"] == "page" and a["change"] == "created"]
                    dispatch(Call(e["tool"], e["args"]), e["actor"], "replay", journal=False, seq=e["seq"],
                             page_ids=ids)
            except Exception as exc:  # noqa: BLE001 - reported, and the rest still runs
                errors.append({"seq": e["seq"], "code": e["code"], "error": f"{type(exc).__name__}: {exc}"})
    finally:
        ctx.panel = live_panel

    mismatches: list[dict[str, Any]] = []
    missing: list[str] = []
    if verify:
        for (name, version), summary in sorted(expected.items()):
            try:
                got = next((v for v in ctx.objects.versions(name) if v["version"] == version), None)
            except KeyError:
                got = None
            if got is None:
                missing.append(f"{name} v{version}")
            elif not summaries_match(got["summary"], summary, 1e-10):
                mismatches.append({"name": name, "version": version})
    live_after = {o.name for o in ctx.objects.list()}
    extra = sorted(live_after - live_before)
    lost = sorted(live_before - live_after)
    return {"ok": not (errors or mismatches or missing or lost), "replayed": len(entries),
            "errors": errors, "mismatches": mismatches, "missing": missing, "extra": extra, "lost": lost,
            "objects": len(live_after), "versions_checked": len(expected)}
