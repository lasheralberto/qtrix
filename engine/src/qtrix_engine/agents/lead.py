"""The configurable super-agent ("Lead Econometrician") and its chat session."""

from __future__ import annotations

import functools
import inspect
import json
import logging
import uuid
import weakref
from collections.abc import AsyncIterator, Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic_ai import (
    Agent,
    AgentRunResultEvent,
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    FunctionToolset,
    ModelRetry,
    PartDeltaEvent,
    PartStartEvent,
    RunContext,
    TextPart,
    TextPartDelta,
    Tool,
    ToolDefinition,
    UsageLimits,
    capture_run_messages,
)
from pydantic_ai.exceptions import UnexpectedModelBehavior, UsageLimitExceeded
from pydantic_ai.messages import ModelMessage, RetryPromptPart
from pydantic_ai.tools import ToolPrepareFunc

from ..callctx import start_turn
from ..context import get_context
from ..dispatch import Call, PolicyDenied, dispatch
from ..objects.store import LockedObjectError, StaleObjectError
from ..panel import PinnedBlockError
from ..tools.registry import ToolSpec, all_tools
from ..workspace import DatasetNotFound
from . import findings as fx
from . import planner as pl
from . import skills as sk

LEAD_INSTRUCTIONS = """\
You are the Lead Econometrician of qtrix, a local-first research workbench. You work with an applied
economist on their own data, which never leaves their machine.

How you work:
- Act through tools. Inspect data before modelling (data_list_datasets, data_summary_stats); never invent numbers.
- Respect each dataset's structure (listed below). Cross-sections (surveys, firms in one year…) have no
  time: do not lag them or run time-series tests on them. When a dataset shows a suggested structure, ask
  the user or confirm it with data_set_structure before using time-series or panel tools; never invent dates.
- Formulas accept EViews time operators on time series and panels, so do not build lag columns by hand:
  y(-1), d(y), dlog(y), pch(y), L(y, 2), movav(y, 4), @trend, @seas(q), dummy("2008q3"), recode(c, a, b).
  data_genr creates a persistent series from the same expressions. Lags work within entities of a panel and
  can look before the sample starts.
- The sample is state: data_set_sample restricts which rows (periods, observation numbers, a condition) every
  estimate and descriptive uses. Change it instead of filtering data by hand, mention the sample in your
  interpretation, and pass `sample` to a single call only for a one-off comparison.
- Choose estimators deliberately and say why: OLS with robust/clustered errors, panel FE/RE (+ Hausman),
  IV when endogeneity is plausible (report first-stage strength), time-series tools for dynamics.
- Every estimate is a named, persistent object (result_id = its name: eq01, var01...). Choose meaningful names
  with `name` (eq_taylor). To change a specification re-estimate the SAME name with another formula, sample or
  option: it becomes a new version and you get the coefficient changes. Do not pile up copies; obj_* tools
  list, inspect, rename and delete objects. Tables comparing objects go stale when those change: refresh them
  by repeating ols_compare with the table's name. Locked objects belong to the user: never modify, re-estimate or
  delete them, and do not work around a lock by copying under a new name.
- Every estimate you run is rendered in the analysis panel automatically, as a live view: when you re-estimate
  an object, its table and the views of it in the panel (residuals, diagnostics, coefficient plots...) refresh
  by themselves, so never add them again. obj_views lists what each object offers; obj_view computes one (and
  shows it once); obj_freeze keeps a snapshot as a table/graph object that never changes. The conversation lives
  in the same panel: the user's question opens each turn and your final answer is shown right after the blocks
  the turn produced, so do not repeat your answer in a ui_add_note. Use ui_add_note only for a lasting note the
  user asked for or that stands apart from your answer. Curate the panel: compare specifications with
  ols_compare and remove blocks that are obsolete. Never touch pinned blocks, nor blocks marked BY USER (the
  user's own notes and Python cells): read them as context, build on them, never modify or remove them.
- The user sees each of your tool calls as an editable Python cell and may change and re-run it (a formula, an
  option, the sample). Their edits reach you in brackets before their message: they are the user's decisions, so
  comment on the new versions and never undo or re-estimate them back.
- Check assumptions (the "diagnostics" and "residuals" views, unit-root tests) before drawing conclusions.
- Work in a loop until the goal is reached. A tool error is not the end: read the message, fix the cause
  (a wrong table or column name, a bad formula, an unsupported option) and call the tool again. Error messages
  list the valid names; use them exactly. Never repeat the same failing call unchanged: change something, or
  inspect first (data_list_datasets, data_describe_structure, data_preview). Only after several different
  attempts fail, stop and explain to the user what you tried and what is blocking you.
- When a turn comes with a plan, follow it: settle each step with plan_update as soon as it is done (or skipped),
  and plan_revise it when the results call for a different route.
- You have a workspace the user watches while you work (plan, tasks, progress and findings fill it by themselves).
  Add to it with scratch_note only what the cells do not show: an alternative you weigh before choosing, a
  provisional conclusion once a key result settles a question, a deliverable at the end. A few notes per turn,
  one or two sentences each; never your internal deliberation, and never instead of the final answer.
- Explain every result as you go: right after a tool returns, before your next call, write its finding in 1-3
  sentences (the key numbers and what they mean, or why it failed and what you will change). It is shown inside
  that call's cell, so the user can follow the analysis cell by cell.
- Always end the turn with a final answer in text: what you did, the key results and what they mean. Never end on
  a tool call or with only a promise of what you will do next.
- Answer in the user's language. Be concise; report coefficients with standard errors and interpret
  magnitudes economically, flagging identification threats.
"""


log = logging.getLogger(__name__)

AGENT_NAME = "lead_econometrician"
TOOL_RETRIES = 6  # failed attempts the model gets per tool before the run gives up
MAX_ROUNDS = 3  # executor runs per turn: the first one plus continuations while plan steps are pending


@dataclass
class _Helpers:
    """The agents that work around a lead agent, with its model: one drafts each turn's plan, one explains cells."""

    planner: Agent[None, pl.PlanDraft]
    explainer: Agent[None, fx.Findings]


#: Helpers by id of the lead agent they were built with.
_helpers: dict[int, _Helpers] = {}


def _wrap(spec: ToolSpec, agent_name: str = AGENT_NAME) -> Callable[..., Any]:
    """Run the tool through the dispatcher as ``agent:<name>`` and turn domain and library errors into
    ModelRetry so the model can fix its arguments."""
    signature = inspect.signature(spec.fn)

    @functools.wraps(spec.fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        pl.check_tools_allowed()
        try:
            call_args = dict(signature.bind(*args, **kwargs).arguments)
            return dispatch(Call(spec.name, call_args), f"agent:{agent_name}", "chat")
        except (ValueError, KeyError, DatasetNotFound, PinnedBlockError, LockedObjectError, StaleObjectError,
                PolicyDenied, FileNotFoundError) as exc:
            raise ModelRetry(f"{type(exc).__name__}: {exc}") from exc
        except Exception as exc:  # noqa: BLE001 - patsy/linalg/library errors are actionable for the model
            raise ModelRetry(f"{spec.name} failed: {type(exc).__name__}: {exc}") from exc

    return wrapper


def _only_for_structures(spec: ToolSpec) -> ToolPrepareFunc[None] | None:
    """Hide tools that cannot apply to any page in the workspace (no ts_* when there is no time series)."""
    if spec.kinds is None:
        return None

    async def prepare(ctx: RunContext[None], tool_def: ToolDefinition) -> ToolDefinition | None:
        return tool_def if spec.applies_to(get_context().workspace.kinds()) else None

    return prepare


#: Tools a compact agent always sees; the rest are deferred until the model finds them with ``search_tools``.
COMPACT_CORE = frozenset({"data_list_datasets", "data_describe_structure", "data_preview", "data_summary_stats",
                          "data_set_structure", "ols_fit", "obj_list", "obj_view"})

#: Calls that manage the agent itself (skills, tool discovery): never shown as Python cells.
BOOKKEEPING_TOOLS = sk.SKILL_TOOLS | {"search_tools"}

COMPACT_NOTE = """\
- To keep the prompt small only a few tools are loaded. The others exist but are hidden: when you need one, call
  search_tools with its name or a few keywords (e.g. ["panel_fit"], ["arima forecast"]) and it becomes available.
  Hidden tools: {hidden}
"""


def econometrics_toolset(namespaces: Sequence[str] | None = None, *,
                         core: frozenset[str] | None = None) -> FunctionToolset:
    """``core`` (compact mode) lists the tools sent with every request; the others are deferred to tool search."""
    tools = [
        Tool(_wrap(spec), name=spec.name, takes_ctx=False, docstring_format="google",
             prepare=_only_for_structures(spec), defer_loading=core is not None and spec.name not in core)
        for spec in all_tools(namespaces=namespaces)
    ]
    # Tool errors come back to the model as retries; the run's tool_calls_limit is the real bound on the loop.
    return FunctionToolset(tools, max_retries=TOOL_RETRIES)


def workspace_context() -> str:
    """Live snapshot of datasets and panel injected into the instructions of every run."""
    ctx = get_context()
    lines = ["## Datasets in the workspace"]
    datasets = ctx.workspace.list()
    if not datasets:
        lines.append("(none yet: ask the user to import data, or use data_import_file with a local path)")
    for d in datasets:
        cols = ", ".join(f"{c['name']}:{c['type']}" for c in (d.columns or [])[:60])
        lines.append(f"- {d.name} [id={d.id}] {d.n_rows} rows · {d.structure_text} :: {cols}")
        if d.sample:
            lines.append(f"  ACTIVE SAMPLE: {d.sample} (estimates and descriptives use only these rows)")
        if d.suggestion:
            s = d.suggestion
            lines.append(f"  suggested structure (not applied): {s['kind']} freq={s['freq']} date_col={s['date_col']}"
                         + (f" id_col={s['id_col']}" if s.get("id_col") else "") + f" ({s['reason']})")
    lines.append("\n## Analysis panel (in order)")
    blocks = ctx.panel.blocks()
    if not blocks:
        lines.append("(empty)")
    for i, b in enumerate(blocks, 1):
        pin = " PINNED" if b.pinned else ""
        res = f" result_id={b.result_id}" if b.result_id else ""
        mine = " BY USER" if b.source == "user" else ""
        lines.append(f"{i}. [{b.id}] {b.kind}: {b.title}{res}{pin}{mine}")
        if b.source == "user":  # the user's own notes and cells: their content is part of the conversation
            body = b.data.get("markdown") if b.kind == "markdown" else b.code
            if body:
                body = body if len(body) <= 600 else body[:600] + "…"
                lines.append("   " + body.replace("\n", "\n   "))
    return "\n".join(lines)


def build_lead_agent(model: Any, *, namespaces: Sequence[str] | None = None, instructions_extra: str = "",
                     extra_toolsets: Sequence[Any] = (),
                     memory_lookup: Callable[[str], str] | None = None,
                     skills_library: Path | None = sk.SKILLS_DIR, compact: bool = False) -> Agent:
    """``skills_library`` holds the method guides (skills) the agent may load; ``None`` gives it none.

    ``compact`` is for small local models with a short context: only ``COMPACT_CORE`` goes with every request and
    the other tool schemas are loaded on demand through ``search_tools`` (skills are always loaded on demand)."""
    instructions = LEAD_INSTRUCTIONS
    if compact:
        hidden = [s.name for s in all_tools(namespaces=namespaces) if s.name not in COMPACT_CORE]
        instructions += COMPACT_NOTE.format(hidden=", ".join(hidden))
    agent = Agent(
        model,
        name="lead_econometrician",
        instructions=instructions + ("\n" + instructions_extra if instructions_extra else ""),
        toolsets=[econometrics_toolset(namespaces, core=COMPACT_CORE if compact else None), pl.plan_toolset(),
                  *extra_toolsets],
        capabilities=sk.skill_capabilities(skills_library),
        retries=TOOL_RETRIES,
    )
    _helpers[id(agent)] = _Helpers(pl.build_planner(model, namespaces=namespaces), fx.build_explainer(model))
    weakref.finalize(agent, _helpers.pop, id(agent), None)

    @agent.instructions
    def _live_state(ctx: RunContext[None]) -> str:
        parts = [workspace_context()]
        if memory_lookup is not None and isinstance(ctx.prompt, str):
            recalled = memory_lookup(ctx.prompt)
            if recalled:
                parts.append("## Relevant memory from earlier in this project\n" + recalled)
        return "\n\n".join(parts)

    return agent


def _short(value: Any, limit: int = 1500) -> str:
    text = value if isinstance(value, str) else json.dumps(value, default=str)
    return text if len(text) <= limit else text[:limit] + "…"


def _code(tool: str, args: dict[str, Any]) -> str | None:
    """The canonical qx code of a call, shown as the chat cell (None when the arguments do not bind)."""
    from ..qx import print_call

    try:
        return print_call(tool, args)
    except Exception:  # noqa: BLE001 - invalid arguments from the model: the cell shows the raw call instead
        return None


def _terms(formula: str) -> tuple[str, list[str]]:
    from ..expr import _first_tilde, _segments

    cut = _first_tilde(formula)
    lhs, rhs = (formula[:cut].strip(), formula[cut + 1:]) if cut >= 0 else ("", formula)
    terms, sign = [], "+"
    for is_piece, text in _segments(rhs):
        if not is_piece:
            sign = text if text in "+-" else sign
        elif text.strip():
            terms.append(("-" if sign == "-" else "") + text.strip())
    return lhs, terms


def describe_edit(before: dict[str, Any], after: dict[str, Any]) -> str:
    """What the user changed in a call, e.g. ``formula: + gap(-1), - infl; cov_type: 'HC1' → 'HC3'``."""
    parts: list[str] = []
    for key in [*before, *(k for k in after if k not in before)]:
        old, new = before.get(key), after.get(key)
        if old == new:
            continue
        if key == "formula" and isinstance(old, str) and isinstance(new, str):
            (lhs_old, t_old), (lhs_new, t_new) = _terms(old), _terms(new)
            change = [f"+ {t}" for t in t_new if t not in t_old] + [f"- {t}" for t in t_old if t not in t_new]
            if lhs_old != lhs_new:
                change.insert(0, f"dependent {lhs_old} → {lhs_new}")
            parts.append("formula: " + (", ".join(change) or f"{old!r} → {new!r}"))
        else:
            parts.append(f"{key}: {old!r} → {new!r}")
    return "; ".join(parts)


def _with_defaults(tool: str, args: dict[str, Any]) -> dict[str, Any]:
    """Arguments with the tool's defaults filled in: the printed code leaves defaults out, the agent's call may not."""
    from ..qx.forms import params

    out = {name: p.default for name, p in params(tool).items()
           if p.default is not inspect.Parameter.empty and p.default is not None}
    out.update({k: v for k, v in args.items() if v is not None})
    return out


def _result_fields(content: Any) -> dict[str, Any]:
    """The object and panel block a tool result points at, so a cell can open them."""
    if not isinstance(content, dict):
        return {}
    obj = content.get("result_id") or content.get("name")
    return {"object": obj if isinstance(obj, str) else None,
            "version": content.get("version") if isinstance(content.get("version"), int) else None,
            "block_id": content.get("block_id") if isinstance(content.get("block_id"), str) else None}


class ChatSession:
    """Conversation state with the super-agent; yields UI events while the agent works.

    Every tool call is a cell (E7·08b): its canonical ``qx`` code, its journal entry and what it produced. The user
    can edit and re-run a cell (``POST /code`` with ``edited_from``); the next turn tells the agent what changed.
    """

    def __init__(self) -> None:
        self.history: list[ModelMessage] = []
        self.transcript: list[dict[str, Any]] = []
        self.seen_seq: int | None = None  # journal position at the end of the agent's last turn
        #: Called (without arguments) when a turn ends or the chat is reset; the notebook's autosave hangs here.
        self.on_change: Callable[[], None] | None = None

    def restore(self, history: list[ModelMessage], transcript: list[dict[str, Any]], seen_seq: int | None) -> None:
        self.history, self.transcript, self.seen_seq = history, transcript, seen_seq

    def reset(self) -> None:
        self.history.clear()
        self.transcript.clear()
        self.seen_seq = None
        if self.on_change is not None:
            self.on_change()

    def user_edits(self) -> str:
        """The changes the user made to the workfile since the agent last answered, for its next prompt."""
        if self.seen_seq is None:
            return ""
        journal = get_context().journal
        lines: list[str] = []
        for e in journal.list(since=self.seen_seq, mutating=True, ok=True):
            if e["actor"].startswith("agent") or e["channel"] == "kernel":
                continue
            code = e["code"] if len(e["code"]) <= 400 else e["code"][:400] + "…"
            if e["edited_from"]:
                try:
                    before = journal.get(e["edited_from"])
                except Exception:  # noqa: BLE001 - the original entry is gone: report the call alone
                    before = None
                change = (describe_edit(_with_defaults(e["tool"], before["args"]), _with_defaults(e["tool"], e["args"]))
                          if before and before["tool"] == e["tool"] else "")
                lines.append(f"- [{e['seq']}] the user edited your call [{e['edited_from']}] and re-ran it"
                             + (f" ({change})" if change else "") + f":\n  {code}")
            else:
                lines.append(f"- [{e['seq']}] the user ran:\n  {code}")
            for a in e["affected"]:
                if (a.get("kind") == "object" and a.get("version_after")
                        and a.get("version_before") != a.get("version_after")):
                    lines.append(f"  → {a['name']} is now version {a['version_after']}")
        if not lines:
            return ""
        return ("[Workfile changes the user made since your last answer. They are deliberate: build on them, comment "
                "on the new versions when relevant, and do not revert them.]\n" + "\n".join(lines) + "\n\n")

    async def stream(self, agent: Agent, message: str, *, max_tool_calls: int = 40,
                     plan: bool = True) -> AsyncIterator[dict[str, Any]]:
        """One turn, plan-and-execute: the planner drafts the steps, the agent's ReAct loop runs them (and is sent
        back to work while steps are pending), and the turn always closes with a final answer in text."""
        ctx = get_context()
        # The panel and the chat are one surface (E7·08c): a turn sits after the last block present when it started.
        blocks = ctx.panel.blocks()
        after_block = blocks[-1].id if blocks else None
        earlier = list(self.transcript)
        self.transcript.append({"role": "user", "content": message, "after_block": after_block})
        request = self.user_edits() + message
        turn = start_turn(uuid.uuid4().hex[:12])  # the dispatcher stamps this turn's calls in the journal
        yield {"type": "turn", "turn_id": turn.turn_id, "after_block": after_block}
        state = _TurnState(turn_id=turn.turn_id, budget=max_tool_calls)
        try:
            helpers = _helpers.get(id(agent))
            if plan and helpers is not None:
                yield {"type": "status", "message": "Planificando…"}
                state.plan, state.scratch = await self._draft_plan(helpers.planner, request, earlier)
                if state.plan.steps:
                    yield {"type": "plan", "steps": state.plan.dump()}
                if state.scratch:
                    yield {"type": "scratch", **state.scratch.dump()}
            prompt = pl.executor_prompt(request, state.plan)
            with pl.active_plan(state.plan if state.plan.steps else None), pl.active_scratch(state.scratch):
                for round_no in range(MAX_ROUNDS):
                    if round_no:
                        yield {"type": "status", "message": "Continuando con el plan…"}
                    async for ev in self._round(agent, prompt, state):
                        yield ev
                    if state.stopped or state.budget <= 0:
                        break
                    if not state.plan.pending():
                        break
                    prompt = pl.continue_prompt(state.plan)
                if not state.final_text():
                    yield {"type": "status", "message": "Redactando la respuesta…"}
                    with pl.finalizing():
                        async for ev in self._round(agent, pl.final_prompt(message), state, final=True):
                            yield ev
                if not state.final_text():  # the model would not write one: summarise the turn ourselves
                    for ev in state.say(self._fallback_answer(state)):
                        yield ev
            unexplained = [c for c in state.cells.values() if c.get("ok") and not c.get("finding")]
            if unexplained and helpers is not None:
                yield {"type": "status", "message": "Explicando los resultados…"}
                for ev in await self._explain(helpers.explainer, message, unexplained):
                    yield ev
        finally:
            self.seen_seq = ctx.journal.last_seq()
            entry = {"role": "assistant", "content": "".join(state.answer), "turn_id": turn.turn_id,
                     "after_block": after_block,
                     "tools": [c["name"] for c in state.cells.values()], "cells": list(state.cells.values())}
            if state.plan.steps:
                entry["plan"] = state.plan.dump()
            if state.scratch:
                entry["scratch"] = state.scratch.dump()
            if state.intro:
                entry["intro"] = state.intro
            self.transcript.append(entry)
            if self.on_change is not None:
                self.on_change()
        yield {"type": "done"}

    @staticmethod
    async def _draft_plan(planner: Agent[None, pl.PlanDraft], request: str,
                          earlier: list[dict[str, Any]]) -> tuple[pl.Plan, pl.Scratch]:
        """The plan of the turn, and the workspace it opens with (the goal and the alternatives weighed)."""
        try:
            result = await planner.run(pl.planner_prompt(request, workspace_context(), earlier),
                                       usage_limits=UsageLimits(request_limit=3))
            return pl.Plan.of(result.output.steps), pl.Scratch.of(result.output)
        except Exception as exc:  # noqa: BLE001 - planning is an aid: without a plan the agent still works
            log.warning("Planner failed, running without a plan: %s", exc)
            return pl.Plan(), pl.Scratch()

    @staticmethod
    async def _explain(explainer: Agent[None, fx.Findings], request: str,
                       cells: list[dict[str, Any]]) -> list[dict[str, Any]]:
        try:
            found = await fx.explain(explainer, request, cells)
        except Exception as exc:  # noqa: BLE001 - the cells and the answer stand without it
            log.warning("Explaining cells failed: %s", exc)
            return []
        for c in cells:
            if c["id"] in found:
                c["finding"] = found[c["id"]]
        return [{"type": "finding", "ids": [cid], "text": text, "strip": 0} for cid, text in found.items()]

    async def _round(self, agent: Agent, prompt: str, state: _TurnState, *,
                     final: bool = False) -> AsyncIterator[dict[str, Any]]:
        """One run of the agent's ReAct loop. Errors that end the loop (budget, retries) are absorbed: the history
        keeps what was done so the next round or the final answer can build on it."""
        journal = get_context().journal
        limits = UsageLimits(tool_calls_limit=max(state.budget, 1) if not final else 3)
        settings = {"tool_choice": "none"} if final else None
        used = 0
        with capture_run_messages() as captured:
            try:
                async with agent.run_stream_events(prompt, message_history=self.history, usage_limits=limits,
                                                   model_settings=settings) as run:
                    async for event in run:
                        if isinstance(event, PartStartEvent) and isinstance(event.part, TextPart):
                            for ev in state.say(event.part.content):
                                yield ev
                        elif isinstance(event, PartDeltaEvent) and isinstance(event.delta, TextPartDelta):
                            for ev in state.say(event.delta.content_delta):
                                yield ev
                        elif isinstance(event, FunctionToolCallEvent):
                            used += 1
                            # Text before a call is not the answer: it explains the results the model just saw.
                            moved = state.take_finding()
                            if moved is not None:
                                yield moved
                            if final or event.part.tool_name in pl.WORK_TOOLS | BOOKKEEPING_TOOLS:
                                continue
                            args = event.part.args_as_dict()
                            cell = {"id": event.part.tool_call_id, "name": event.part.tool_name, "args": args,
                                    "code": _code(event.part.tool_name, args)}
                            state.cells[cell["id"]] = cell
                            yield {"type": "tool_call", **cell}
                        elif isinstance(event, FunctionToolResultEvent):
                            if final:
                                continue
                            if event.part.tool_name in pl.PLAN_TOOLS:
                                yield {"type": "plan", "steps": state.plan.dump()}
                                continue
                            if event.part.tool_name in pl.SCRATCH_TOOLS:
                                yield {"type": "scratch", **state.scratch.dump()}
                                continue
                            if event.part.tool_name in BOOKKEEPING_TOOLS:
                                continue
                            yield self._tool_result(event.part, state, journal)
                            state.unexplained.append(event.part.tool_call_id)
                        elif isinstance(event, AgentRunResultEvent):
                            self.history = list(event.result.all_messages())
            except (UsageLimitExceeded, UnexpectedModelBehavior) as exc:
                log.info("Agent loop stopped (%s): %s", type(exc).__name__, exc)
                state.stopped = True
                self.history = pl.resumable(captured)
        state.budget -= used

    @staticmethod
    def _tool_result(part: Any, state: _TurnState, journal: Any) -> dict[str, Any]:
        ok = not isinstance(part, RetryPromptPart)
        content = part.content if ok else part.model_response()
        # The journal entry of this call: the first of this turn with the tool not yet linked.
        entry = next((e for e in journal.list(turn_id=state.turn_id, tool=part.tool_name)
                      if e["seq"] not in state.linked), None)
        if entry is not None:
            state.linked.add(entry["seq"])
        out = {"id": part.tool_call_id, "name": part.tool_name, "ok": ok, "content": _short(content),
               "seq": entry["seq"] if entry else None, "cell_id": entry["cell_id"] if entry else None,
               "code": entry["code"] if entry else None, **_result_fields(content if ok else None)}
        cell = state.cells.setdefault(part.tool_call_id, {"id": part.tool_call_id, "name": part.tool_name,
                                                          "args": {}, "code": None})
        cell.update({k: v for k, v in out.items() if v is not None or k not in cell})
        return {"type": "tool_result", **out}

    @staticmethod
    def _fallback_answer(state: _TurnState) -> str:
        lines = ["No he podido redactar la respuesta final. Esto es lo que se hizo en este turno:"]
        if state.plan.steps:
            lines.append(state.plan.render())
        for c in state.cells.values():
            lines.append(f"- `{c['name']}` {'✓' if c.get('ok', True) else '✗'}"
                         + (f" → {c['object']}" if c.get("object") else ""))
        if len(lines) == 1:
            lines.append("- (ninguna acción)")
        lines.append("\nPídeme que continúe o dime qué cambiar.")
        return "\n".join(lines)


@dataclass
class _TurnState:
    """What one chat turn has produced so far, across the executor rounds."""

    turn_id: str
    budget: int  # tool calls left in this turn
    plan: pl.Plan = field(default_factory=pl.Plan)
    scratch: pl.Scratch = field(default_factory=pl.Scratch)  # the agent's workspace notes in this turn
    answer: list[str] = field(default_factory=list)  # every text delta streamed to the UI
    tail: list[str] = field(default_factory=list)  # text since the last tool call (the end of ``answer``)
    cells: dict[str, dict[str, Any]] = field(default_factory=dict)
    linked: set[int] = field(default_factory=set)
    unexplained: list[str] = field(default_factory=list)  # cells whose results the model saw, not yet explained
    intro: str = ""  # what the model said before its first call
    stopped: bool = False  # the loop hit its tool budget or ran out of retries

    def final_text(self) -> str:
        return "".join(self.tail).strip()

    def take_finding(self) -> dict[str, Any] | None:
        """Move the text written since the last call out of the answer: it is the finding of the cells the model
        has just seen (or the turn's intro before the first call). Returns the UI event, None without text."""
        if not self.tail:
            return None
        strip = sum(map(len, self.tail))
        del self.answer[-len(self.tail):]
        text, self.tail = "".join(self.tail).strip(), []
        ids, self.unexplained = self.unexplained, []
        if not text:
            return {"type": "finding", "ids": [], "text": "", "strip": strip}
        for cid in ids:
            cell = self.cells.get(cid)
            if cell is not None:
                cell["finding"] = (cell.get("finding", "") + "\n\n" + text).strip()
        if not ids:
            self.intro = (self.intro + "\n\n" + text).strip()
        return {"type": "finding", "ids": ids, "text": text, "strip": strip}

    def say(self, delta: str) -> list[dict[str, Any]]:
        if not delta:
            return []
        if not self.tail and self.answer and not "".join(self.answer).endswith("\n\n"):
            delta = "\n\n" + delta.lstrip("\n")  # a new stretch of text (another round): separate it
        self.answer.append(delta)
        self.tail.append(delta)
        return [{"type": "text", "delta": delta}]
