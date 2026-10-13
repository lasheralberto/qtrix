"""The dispatcher: the single write path of the workfile (ADR 0002 §6, E7·06).

Nothing mutates the workfile except ``dispatch(call, actor, channel)``. Agent tool calls, ``POST /tools/{name}``,
MCP clients and the UI's object/page actions all arrive here. A call is validated against the tool's signature,
goes through the policy hooks, runs in the caller's worker thread and leaves a journal entry (code, arguments,
what changed, outcome, timing) plus an event.
"""

from __future__ import annotations

import inspect
import logging
import secrets
import time
import typing
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from pydantic import ConfigDict, create_model

from .callctx import Recording, acting_as, current_turn, in_cell, recording
from .context import EngineContext, get_context
from .objects.store import LockedObjectError
from .tools.registry import ToolSpec, get_tool
from .workspace import DatasetNotFound

log = logging.getLogger(__name__)


class PolicyDenied(ValueError):
    """A policy hook refused the call; agents get the reason back as a ModelRetry."""


class UnknownTool(KeyError):
    def __str__(self) -> str:
        return str(self.args[0]) if self.args else ""


@dataclass(frozen=True)
class Call:
    """One ``qx`` operation: a tool name plus its arguments."""

    tool: str
    args: dict[str, Any] = field(default_factory=dict)


@dataclass
class PolicyRequest:
    call: Call  # arguments already canonical (dataset names, object names)
    spec: ToolSpec
    actor: str
    channel: str
    mutating: bool
    ctx: EngineContext


# ---- policy ----------------------------------------------------------------------------------------------------
# Calls whose ``name`` argument is not the object they write to.
_NAME_IS_NOT_A_TARGET = {"obj_copy", "obj_lock"}


def lock_policy(req: PolicyRequest) -> None:
    """Locked objects only accept the user: refuse early, before any work is done (the store enforces it too)."""
    name = req.call.args.get("name")
    if (not req.mutating or req.actor == "user" or not isinstance(name, str)
            or req.call.tool in _NAME_IS_NOT_A_TARGET or not req.ctx.objects.exists(name)):
        return
    if req.ctx.objects.get(name).locked:
        raise LockedObjectError(f"'{name}' is locked by the user and cannot be modified by {req.actor}. Ask the user "
                                "to unlock it, or work on a different name; do not copy it to get around the lock")


def python_policy(req: PolicyRequest) -> None:
    """Python cells run arbitrary code with the user's privileges: only the user runs them, unless the user allowed
    agents to (setting ``agent.allow_code``). Until proposals exist (E7·18), the agent shows the cell and asks."""
    if not req.spec.cell or req.actor == "user" or req.ctx.allow_agent_code:
        return
    raise PolicyDenied(f"{req.actor} may not run Python cells: the user has not allowed agents to run code. Show the "
                       "user the cell and ask them to run it (or to enable 'allow the agent to run code'), or use the "
                       "qx tools instead")


# Autonomy modes (E7·18) plug in here: a hook may raise PolicyDenied, and later turn an agent call into a proposal.
_POLICIES: list[Callable[[PolicyRequest], None]] = [lock_policy, python_policy]


def register_policy(hook: Callable[[PolicyRequest], None]) -> None:
    _POLICIES.append(hook)


# ---- arguments -------------------------------------------------------------------------------------------------
_MODELS: dict[str, Any] = {}


def _args_model(spec: ToolSpec) -> Any:
    """Pydantic model of the tool's signature, the schema its arguments are validated against."""
    if spec.name not in _MODELS:
        try:
            hints = typing.get_type_hints(spec.fn)
            fields = {p.name: (hints.get(p.name, Any), ... if p.default is inspect.Parameter.empty else p.default)
                      for p in inspect.signature(spec.fn).parameters.values()}
            _MODELS[spec.name] = create_model(f"{spec.name}_args", __config__=ConfigDict(extra="forbid"), **fields)
        except Exception:  # noqa: BLE001 - an unresolvable annotation must not make the tool unusable
            log.warning("Cannot build an argument schema for %s; calls are not validated", spec.name)
            _MODELS[spec.name] = None
    return _MODELS[spec.name]


def validate_args(spec: ToolSpec, args: dict[str, Any]) -> dict[str, Any]:
    """Check the arguments against the tool's schema (unknown names, types, Literal choices)."""
    model = _args_model(spec)
    if model is None:
        return dict(args)
    validated = model.model_validate(args)
    return {k: getattr(validated, k) for k in args}


def canonical_args(ctx: EngineContext, args: dict[str, Any]) -> dict[str, Any]:
    """Refer to pages and objects by name, so a journal entry means the same after a replay (page ids are random)."""
    out = dict(args)
    ds = out.get("dataset_id")
    if isinstance(ds, str):
        try:
            out["dataset_id"] = ctx.workspace.get(ds).name
        except DatasetNotFound:
            pass
    for key in ("result_id", "fe_result_id", "re_result_id"):
        if isinstance(out.get(key), str):
            out[key] = ctx.objects.resolve(out[key])
    if isinstance(out.get("result_ids"), list):
        out["result_ids"] = [ctx.objects.resolve(r) if isinstance(r, str) else r for r in out["result_ids"]]
    return out


# ---- what a call touched ---------------------------------------------------------------------------------------
# Tools that rewrite an existing page; True when its data (not just its catalog row) must be kept for undo.
_PAGE_TOOLS = {"data_set_structure": True, "data_set_panel_structure": True, "data_genr": True,
               "data_set_series": True, "data_set_sample": False, "data_delete_dataset": False}


def _capture_pages(ctx: EngineContext, spec: ToolSpec, args: dict[str, Any], rec: Recording) -> None:
    if spec.name not in _PAGE_TOOLS or not isinstance(args.get("dataset_id"), str):
        return
    try:
        info = ctx.workspace.get(args["dataset_id"])
    except DatasetNotFound:
        return
    rec.pages.add(info.id)
    if spec.name == "data_delete_dataset":
        return  # not undoable: replay rebuilds it
    undo_table = f"undo_{rec.seq}_{info.id}" if _PAGE_TOOLS[spec.name] else None
    ctx.journal.save_before(rec.seq, "page", info.id, ctx.workspace.capture_page_state(info.id, undo_table))


def _affected(ctx: EngineContext, rec: Recording, pages_before: dict[str, list[Any]],
              keep: bool) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for name, before in rec.images.items():
        after = ctx.objects.image(name)
        if before["row"] == after["row"] and len(before["versions"]) == len(after["versions"]):
            continue
        v_before = before["row"][3] if before["row"] else None
        v_after = after["row"][3] if after["row"] else None
        out.append({"kind": "object", "name": name, "version_before": v_before, "version_after": v_after})
        if keep:
            ctx.journal.save_before(rec.seq, "object", name, before)
    pages_after = ctx.workspace.page_rows()
    for pid in sorted(set(pages_before) | set(pages_after) | rec.pages):
        before, after = pages_before.get(pid), pages_after.get(pid)
        if before is None and after is not None:
            out.append({"kind": "page", "change": "created", "id": pid, "name": after[1], "source": after[2],
                        "table_name": after[3], "created_at": after[4], "snapshot": after[13]})
        elif before is not None and after is None:
            out.append({"kind": "page", "change": "deleted", "id": pid, "name": before[1], "undoable": False})
        elif before is not None and (before != after or pid in rec.pages):
            out.append({"kind": "page", "change": "changed", "id": pid, "name": after[1],
                        "data_version_before": before[15], "data_version_after": after[15]})
    return out


# ---- dispatch --------------------------------------------------------------------------------------------------
def dispatch(call: Call, actor: str, channel: str, *, turn_id: str | None = None, cell_id: str | None = None,
             edited_from: int | None = None, proposal_id: str | None = None, journal: bool = True,
             seq: int | None = None, page_ids: list[str] | None = None, with_seq: bool = False) -> Any:
    """Validate, apply policy, execute and journal one call. Errors are journaled and re-raised unchanged.

    ``journal=False`` (with the original ``seq`` and the ids of the pages it created) is used by replay: the call
    runs but writes no entry. ``with_seq=True`` returns ``(result, seq of the journal entry)``; a failing call's
    exception then carries the seq as ``journal_seq``.
    """
    ctx = get_context()
    spec = get_tool(call.tool)
    if spec is None or not spec.available:
        raise UnknownTool(f"Unknown or unavailable tool '{call.tool}'")
    turn = current_turn()
    if turn is not None and turn_id is None and channel == "chat":
        turn_id, cell_id = turn.turn_id, cell_id or turn.next_cell()
    if edited_from is not None:
        ctx.journal.get(edited_from)  # must exist
    if spec.cell and cell_id is None:
        cell_id = f"py-{secrets.token_hex(6)}"  # the calls the cell makes are linked to it by this id

    started = time.perf_counter()
    args = canonical_args(ctx, call.args)
    mutating = spec.is_mutating(args)
    with ctx.journal.mutation_lock if mutating else _NoLock():
        rec = Recording(seq=ctx.journal.reserve() if journal and mutating else seq, page_ids=list(page_ids or []))
        pages_before = ctx.workspace.page_rows() if mutating else {}
        ok, error, result, failure = True, None, None, None
        try:
            args = validate_args(spec, args)
            mutating = spec.is_mutating(args)
            for hook in _POLICIES:
                hook(PolicyRequest(Call(call.tool, args), spec, actor, channel, mutating, ctx))
            if mutating and journal:
                _capture_pages(ctx, spec, args, rec)
            with acting_as(actor), in_cell(cell_id), recording(rec, replay=not journal):
                result = spec.fn(**args)
        except Exception as exc:  # noqa: BLE001 - journaled, then re-raised as it was
            ok, error, failure = False, f"{type(exc).__name__}: {exc}", exc
        if not journal:
            if failure:
                raise failure
            return result
        affected = _affected(ctx, rec, pages_before, keep=ok) if mutating else []
        if mutating and not ok:
            ctx.journal.discard_before(rec.seq)
        entry_seq = rec.seq if mutating else ctx.journal.reserve()
        kind, code, journaled_mutating = "api", None, mutating
        if spec.cell:  # a Python cell: its own calls are journaled one by one; this entry keeps the source
            outcome = result if ok else getattr(failure, "result", None)
            kind, code = "python", args.get("code", "")
            journaled_mutating = bool(isinstance(outcome, dict) and outcome.get("mutated"))
        ctx.journal.append(
            seq=entry_seq, actor=actor, channel=channel, tool=call.tool, args=args, mutating=journaled_mutating,
            affected=affected, ok=ok, error=error, duration_ms=round((time.perf_counter() - started) * 1000),
            turn_id=turn_id, cell_id=cell_id, edited_from=edited_from, proposal_id=proposal_id, kind=kind,
            code=code)
    if failure:
        if with_seq:
            try:
                failure.journal_seq = entry_seq  # type: ignore[attr-defined]
            except AttributeError:
                pass
        raise failure
    return (result, entry_seq) if with_seq else result


class _NoLock:
    def __enter__(self) -> None:
        return None

    def __exit__(self, *exc: object) -> None:
        return None
