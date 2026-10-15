"""The configurable super-agent ("Lead Econometrician") and its chat session."""

from __future__ import annotations

import asyncio
import contextlib
import functools
import inspect
import json
import logging
import re
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
from . import autocompact as ac
from . import findings as fx
from . import planner as pl
from . import reporting as rp
from . import skills as sk

LEAD_INSTRUCTIONS = """\
You are the Lead Econometrician of qtrix, a local-first research workbench. You work with an applied
economist on their own data, which never leaves their machine.

How you work:
- Do only what the user asks. A greeting, thanks, small talk or a question about you or about what was already
  done gets a short text reply and no tool calls: never inspect, describe or model data on your own initiative.
- When the request needs data, act through tools. Inspect data before modelling (data_list_datasets,
  data_summary_stats); never invent numbers.
- Respect each dataset's structure (listed below). Cross-sections (surveys, firms in one year…) have no
  time: do not lag them or run time-series tests on them. When a dataset shows a suggested structure, ask
  the user (ask_user) or confirm it with data_set_structure before using time-series or panel tools; never
  invent dates.
- When you need a decision only the user can make (outcome, treatment, estimand, which reading of the request),
  call ask_user with your 1-3 questions BEFORE running anything else, then end the turn. Never write a question
  and carry on in the same turn: the user answers first, you act afterwards. Decide small things yourself.
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
  provisional conclusion once a key result settles a question, an "issue" for each caveat the analyst must keep
  in mind (failed assumption, weak identification, data limits; never tool errors you recovered from), a
  deliverable at the end. A few notes per turn, one or two sentences each; never your internal deliberation, and
  never instead of the final answer.
- The workspace also has "Next steps": what is still open in the whole conversation, shared with the user, who can
  add, tick and delete them. Before your final answer, leave with next_step the 1 to 3 follow-ups the analysis
  leaves open (a robustness check not run, a specification to compare...; refer to objects as [[name]], they become
  links the user can open), and close with next_step_done the ones this turn settled. When the prompt lists next
  steps, pick them up if the user asks to continue or refers to one ("do step 2").
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
    """The agents that work around a lead agent, with its model: one drafts each turn's plan, one explains cells,
    one folds the conversation into a summary when the context fills up."""

    planner: Agent[None, pl.PlanDraft]
    explainer: Agent[None, fx.Findings]
    compactor: Agent[None, str]
    reporter: Agent[None, rp.ConversationReport]
    drafter: Agent[None, str]


#: Helpers by id of the lead agent they were built with.
_helpers: dict[int, _Helpers] = {}


def reporter_of(agent: Agent) -> Agent[None, rp.ConversationReport] | None:
    helpers = _helpers.get(id(agent))
    return helpers.reporter if helpers else None


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


def econometrics_toolset(namespaces: Sequence[str] | None = None, *, core: frozenset[str] | None = None,
                         agent_name: str = AGENT_NAME) -> FunctionToolset:
    """``core`` (compact mode) lists the tools sent with every request; the others are deferred to tool search."""
    tools = [
        Tool(_wrap(spec, agent_name), name=spec.name, takes_ctx=False, docstring_format="google",
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
    if len(ctx.threads) > 1:  # other conversations of the notebook may have built objects this one has not seen
        objects = ctx.objects.list()
        if objects:
            names = ", ".join(f"{o.name} ({o.type} v{o.version})" for o in objects[:40])
            more = f", … {len(objects) - 40} more (obj_list)" if len(objects) > 40 else ""
            lines.append(f"\n## Objects in the workfile (shared by every conversation of the notebook)\n{names}{more}")
    steps = getattr(ctx.chat, "next_steps", None)  # a context may have no chat yet
    if steps:
        lines.append("\n## Next steps of this conversation (the user sees this list; [[name]] is a workfile object)\n"
                     + steps.render()
                     + "\nPick them up when the user asks to continue or refers to one (\"do step 2\"). Close one "
                       "with next_step_done when you settle it; leave a follow-up with next_step.")
    lines.append("\n## Analysis panel of this conversation (in order)")
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
                     skills_library: Path | None = sk.SKILLS_DIR, skills: Sequence[str] | None = None,
                     compact: bool = False, agent_name: str = AGENT_NAME, roster: str = "") -> Agent:
    """``skills_library`` holds the method guides (skills) the agent may load; ``None`` gives it none, and ``skills``
    limits them to those names. ``agent_name`` signs its calls in the journal (``agent:<name>``); ``roster``
    (``planner.roster_text``) is the team its planner may delegate to.

    ``compact`` is for small local models with a short context: only ``COMPACT_CORE`` goes with every request and
    the other tool schemas are loaded on demand through ``search_tools`` (skills are always loaded on demand)."""
    instructions = LEAD_INSTRUCTIONS
    if compact:
        hidden = [s.name for s in all_tools(namespaces=namespaces) if s.name not in COMPACT_CORE]
        instructions += COMPACT_NOTE.format(hidden=", ".join(hidden))
    agent = Agent(
        model,
        name=agent_name,
        instructions=instructions + ("\n" + instructions_extra if instructions_extra else ""),
        toolsets=[econometrics_toolset(namespaces, core=COMPACT_CORE if compact else None, agent_name=agent_name),
                  pl.plan_toolset(), *extra_toolsets],
        capabilities=sk.skill_capabilities(skills_library, include=skills),
        retries=TOOL_RETRIES,
    )
    _helpers[id(agent)] = _Helpers(pl.build_planner(model, namespaces=namespaces, roster=roster),
                                   fx.build_explainer(model),
                                   ac.build_compactor(model), rp.build_reporter(model),
                                   pl.build_plan_drafter(model, namespaces=namespaces))
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


TEAM_MEMBER_NOTE = """\

## You are a member of a team
You are {name}, a specialist on the Lead Econometrician's team. The lead hands you one part of the user's request: do
that task and only that, with the tools, then end with your conclusions (the key numbers, what they mean and the
caveats); the lead builds the final answer from them. Never ask the user anything: when a choice is open, take the
sensible default and say so. Name your objects so the lead can tell them apart. Leave next steps to the lead.
"""


@dataclass
class TeamMember:
    """A member of the agent team, ready to work: its configuration and the agent built on its model."""

    id: str
    name: str
    agent: Agent
    emoji: str = ""


def build_team_agent(model: Any, *, member_id: str, name: str, behavior: str = "", skills: Sequence[str] = (),
                     namespaces: Sequence[str] | None = None, memory_lookup: Callable[[str], str] | None = None,
                     compact: bool = False) -> Agent:
    """A specialist of the team: the lead's tools and workspace, its own behaviour and skills, signing its calls in
    the journal as ``agent:<member_id>``. It never delegates further."""
    extra = TEAM_MEMBER_NOTE.format(name=name) + ("\n## How you work\n" + behavior.strip() if behavior.strip() else "")
    return build_lead_agent(model, namespaces=namespaces, instructions_extra=extra, memory_lookup=memory_lookup,
                            skills_library=sk.SKILLS_DIR if skills else None, skills=list(skills),
                            compact=compact, agent_name=member_id)


def delegated_prompt(task: str, request: str, earlier: Sequence[str] = (), limit: int = 2500) -> str:
    """What a team member is asked: its task, with the user's request for context and its own conclusions from the
    earlier turns of the conversation (``earlier``, oldest first) so it builds on them."""
    parts = ["[Task from the lead, part of the user's request. Do this task only, then give your conclusions.]\n"
             + task, f"[The user's request, for context]\n{request}"]
    shown = [c if len(c) <= limit else c[:limit] + "…" for c in earlier if c.strip()]
    if shown:
        parts.append("[Your conclusions from earlier turns of this conversation, oldest first. Build on them: do not "
                     "redo what they settle, refer to the objects you made, and revise them if the new task shows "
                     "they were wrong.]\n" + "\n\n---\n\n".join(shown))
    return "\n\n".join(parts)


def earlier_conclusions(transcript: Sequence[dict[str, Any]], member_id: str) -> list[str]:
    """A member's conclusions in the earlier turns of the conversation, oldest first."""
    return [str(a.get("content") or "") for m in transcript if m.get("role") == "assistant"
            for a in m.get("agents") or [] if a.get("id") == member_id and a.get("content")]


def team_prompt(agents: Sequence[dict[str, Any]], limit: int = 3000) -> str:
    """The members' conclusions as they reach the lead, which answers from them."""
    parts = ["\n\n[Your team has already worked on this request. Their conclusions are below: build your final answer "
             "from them — the conclusion of the whole analysis. Attribute results to each member, reconcile or "
             "flag disagreements, and do not redo their work; run tools only for what is still missing.]"]
    for a in agents:
        text = (a.get("content") or "").strip() or "(no conclusions: the member failed)"
        parts.append(f"### {a['name']} — task: {a['task']}\n" + (text if len(text) <= limit else text[:limit] + "…"))
    return "\n".join(parts)


#: Events of a member's own turn that never reach the UI: the turn is the lead's.
_MEMBER_HIDDEN = frozenset({"turn", "done", "context", "compacted", "next_steps"})


async def _in_own_task(events: AsyncIterator[dict[str, Any]]) -> AsyncIterator[dict[str, Any]]:
    """Iterate ``events`` inside a task of its own: it runs in a copy of the current context, so the turn, plan and
    workspace a member sets (context variables) never leak into the lead's. Cancelled when the consumer stops."""
    queue: asyncio.Queue[Any] = asyncio.Queue()
    end = object()

    async def pump() -> None:
        try:
            async for ev in events:
                await queue.put(ev)
        except Exception as exc:  # noqa: BLE001 - a failed member is reported, the lead goes on
            log.exception("Team member failed")
            await queue.put({"type": "error", "message": f"{type(exc).__name__}: {exc}"})
        finally:
            await queue.put(end)

    task = asyncio.create_task(pump())
    try:
        while (ev := await queue.get()) is not end:
            yield ev
    finally:
        if not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task


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
            "block_id": content.get("block_id") if isinstance(content.get("block_id"), str) else None,
            "objects": _cell_objects(content) or None}


def _cell_objects(content: dict[str, Any]) -> list[dict[str, Any]]:
    """The objects a Python cell registered through its own calls (py_run reports their journal entries)."""
    journal = get_context().journal
    seen: dict[str, int | None] = {}
    for seq in content.get("calls") or []:
        try:
            entry = journal.get(seq)
        except Exception:  # noqa: BLE001 - an entry that is gone just shows no object
            continue
        for a in entry["affected"] if entry["ok"] else []:
            if a.get("kind") == "object" and a.get("version_after"):
                seen[a["name"]] = a["version_after"]
    return [{"name": n, "version": v} for n, v in seen.items()]


class ChatSession:
    """Conversation state with the super-agent; yields UI events while the agent works.

    Every tool call is a cell (E7·08b): its canonical ``qx`` code, its journal entry and what it produced. The user
    can edit and re-run a cell (``POST /code`` with ``edited_from``); the next turn tells the agent what changed.
    """

    def __init__(self) -> None:
        self.history: list[ModelMessage] = []
        self.transcript: list[dict[str, Any]] = []
        self.seen_seq: int | None = None  # journal position at the end of the agent's last turn
        self.overhead = 0  # tokens of instructions and tool schemas, measured before the last compaction
        #: What is left to do in this conversation: written by the agent and the user, read by the agent every turn.
        self.next_steps = pl.NextSteps()
        #: Called (without arguments) when a turn ends or the chat is reset; the notebook's autosave hangs here.
        self.on_change: Callable[[], None] | None = None

    def restore(self, history: list[ModelMessage], transcript: list[dict[str, Any]], seen_seq: int | None,
                next_steps: list[dict[str, Any]] | None = None) -> None:
        self.history, self.transcript, self.seen_seq = history, transcript, seen_seq
        self.next_steps.load(next_steps)

    def reset(self) -> None:
        self.history.clear()
        self.transcript.clear()
        self.next_steps.load(None)
        self.seen_seq = None
        if self.on_change is not None:
            self.on_change()

    def edit_next_steps(self, *, add: str | None = None, ref: str | None = None, done: bool | None = None,
                        text: str | None = None, remove: bool = False) -> list[dict[str, Any]]:
        """The user's changes to the next steps: add one, tick or untick one, rewrite one, delete one."""
        steps = self.next_steps
        if add is not None:
            if not add.strip():
                raise ValueError("The step is empty.")
            steps.add(add, author="user")
        elif ref is not None:
            item = steps.get(ref)
            if item is None:
                raise KeyError(ref)
            if remove:
                steps.remove(ref)
            else:
                if done is not None:
                    item["done"] = done
                if text is not None and text.strip():
                    item["text"] = " ".join(text.split())
        if self.on_change is not None:
            self.on_change()
        return steps.dump()

    def context(self, window: int) -> dict[str, Any]:
        """How much of the model's context window the conversation fills (the meter next to the model picker)."""
        used, estimated = ac.history_tokens(self.history, overhead=self.overhead)
        return {"used": used, "limit": window, "estimated": estimated}

    async def compact(self, agent: Agent, window: int) -> dict[str, Any]:
        """Fold the model's history into a summary. The transcript the user sees stays whole."""
        before = self.context(window)["used"]
        self.overhead = ac.overhead(self.history) or self.overhead
        helpers = _helpers.get(id(agent))
        self.history = await ac.compact(helpers.compactor if helpers else None, self.history, window=window)
        if self.on_change is not None:
            self.on_change()
        return {"before": before, **self.context(window)}

    def _full(self, window: int | None, compact_at: float) -> bool:
        # A history that is only the last summary has nothing left to fold (the prompt itself may be what is full).
        return (window is not None and bool(self.history) and not ac.only_summary(self.history)
                and self.context(window)["used"] >= window * compact_at)

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

    async def draft_plan(self, agent: Agent, message: str) -> str:
        """Plan mode: the detailed plan of ``message`` as Markdown for the user to edit and approve. Nothing runs and
        the conversation is left untouched."""
        helpers = _helpers.get(id(agent))
        if helpers is None:
            raise RuntimeError("This agent cannot draft plans.")
        prompt = pl.planner_prompt(self.user_edits() + message, workspace_context(), list(self.transcript))
        result = await helpers.drafter.run(prompt, usage_limits=UsageLimits(request_limit=3))
        return result.output.strip()

    async def stream(self, agent: Agent, message: str, *, max_tool_calls: int = 40, plan: bool = True,
                     context_window: int | None = None, compact_at: float = 0.8,
                     approved_plan: str = "", team: Sequence[TeamMember] = ()) -> AsyncIterator[dict[str, Any]]:
        """One turn, plan-and-execute: the planner drafts the steps, the agent's ReAct loop runs them (and is sent
        back to work while steps are pending), and the turn always closes with a final answer in text.

        With a ``context_window``, the history is compacted into a summary whenever it fills ``compact_at`` of it
        (before the turn and between rounds). ``approved_plan`` is a plan the user approved in plan mode: it goes
        with the request so the planner and the executor follow it; the transcript keeps the bare message.

        With a ``team``, the planner may hand parts of the request to its members: they work first, one after the
        other, and the lead's answer builds on their conclusions."""
        ctx = get_context()
        # The panel and the chat are one surface (E7·08c): a turn sits after the last block present when it started.
        blocks = ctx.panel.blocks()
        after_block = blocks[-1].id if blocks else None
        earlier = list(self.transcript)
        self.transcript.append({"role": "user", "content": message, "after_block": after_block})
        request = self.user_edits() + message
        if approved_plan.strip():
            request += pl.approved_plan_prompt(approved_plan)
        turn = start_turn(uuid.uuid4().hex[:12])  # the dispatcher stamps this turn's calls in the journal
        yield {"type": "turn", "turn_id": turn.turn_id, "after_block": after_block}
        state = _TurnState(turn_id=turn.turn_id, budget=max_tool_calls)
        try:
            async for ev in self._compact_if_full(agent, context_window, compact_at):
                yield ev
            helpers = _helpers.get(id(agent))
            talk_only = False
            delegations: list[pl.AgentTask] = []
            if plan and helpers is not None:
                yield {"type": "status", "message": "Planificando…"}
                state.plan, state.scratch, drafted, questions, delegations = await self._draft_plan(
                    helpers.planner, request, earlier)
                if questions:  # the plan is blocked on the user: nothing runs until they answer
                    for ev in state.say(pl.questions_text(questions)):
                        yield ev
                    state.asked = questions
                # The planner found nothing to do (a greeting, a question about the conversation): answer in text,
                # with the tools off, so the agent cannot start an analysis nobody asked for.
                talk_only = drafted and not state.plan.steps
                if state.plan.steps:
                    yield {"type": "plan", "steps": state.plan.dump()}
                if state.scratch:
                    yield {"type": "scratch", **state.scratch.dump()}
            ids = {m.id for m in team}
            tasks = [d for d in delegations if d.agent_id in ids] if not state.asked else []
            # With a team the work is the members': a plan the planner did not split goes whole to every member.
            if team and not tasks and not state.asked and state.plan.steps:
                tasks = [pl.AgentTask(agent_id=m.id, task=message) for m in team]
            if tasks:
                async for ev in self._run_team(team, tasks, request, state, plan=plan,
                                               budget=max(8, max_tool_calls // (len(tasks) + 1))):
                    yield ev
                request += team_prompt(state.agents)
            prompt = pl.executor_prompt(request, state.plan)
            with (pl.active_plan(state.plan if state.plan.steps else None), pl.active_scratch(state.scratch),
                  pl.active_next(self.next_steps), pl.asking(state.asked)):
                if talk_only and not state.asked:
                    with pl.finalizing():
                        async for ev in self._round(agent, prompt, state, final=True):
                            yield ev
                for round_no in range(0 if talk_only else MAX_ROUNDS):
                    if round_no:
                        async for ev in self._compact_if_full(agent, context_window, compact_at):
                            yield ev
                        yield {"type": "status", "message": "Continuando con el plan…"}
                    async for ev in self._round(agent, prompt, state):
                        yield ev
                    if state.stopped or state.budget <= 0 or state.asked:
                        break
                    if not state.plan.pending():
                        break
                    prompt = pl.continue_prompt(state.plan)
                for ev in self._drop_leaked_call(state):
                    yield ev
                if state.asked and not state.final_text():  # the agent asked through the tool but wrote nothing
                    for ev in state.say(pl.questions_text(state.asked)):
                        yield ev
                if not state.final_text():
                    yield {"type": "status", "message": "Redactando la respuesta…"}
                    with pl.finalizing():
                        async for ev in self._round(agent, pl.final_prompt(message), state, final=True):
                            yield ev
                    for ev in self._drop_leaked_call(state):
                        yield ev
                if not state.final_text():  # the model would not write one: summarise the turn ourselves
                    for ev in state.say(self._fallback_answer(state)):
                        yield ev
            unexplained = [c for c in state.cells.values() if c.get("ok") and not c.get("finding")]
            # A model that never wrote its conclusions, caveats or follow-ups in the workspace gets them drawn from
            # its answer (small models seldom call the bookkeeping tools on their own).
            has = {n["section"] for n in state.scratch.notes}
            summarise = bool(state.cells) and not ({"conclusion", "issue"} <= has and state.noted_next)
            answer = (state.final_text() or "".join(state.answer).strip()) if summarise else ""
            if (unexplained or answer) and helpers is not None:
                yield {"type": "status", "message": "Explicando los resultados…"}
                events, conclusions, issues, follow_ups = await self._explain(
                    helpers.explainer, message, unexplained, answer,
                    objects=state.objects(), open_steps=[i["text"] for i in self.next_steps.open()])
                for ev in events:
                    yield ev
                for text in conclusions if "conclusion" not in has else []:
                    state.scratch.add("conclusion", text)
                for text in issues if "issue" not in has else []:
                    state.scratch.add("issue", text)
                if not state.noted_next and self._add_follow_ups(follow_ups):
                    yield {"type": "next_steps", "items": self.next_steps.dump()}
            # The objects the turn produced, as a deliverable of the workspace.
            if not any(n["section"] == "result" for n in state.scratch.notes):
                names = state.objects()
                if names:
                    state.scratch.add("result", "Objetos creados o actualizados: " + ", ".join(names) + ".")
            if state.scratch:
                yield {"type": "scratch", **state.scratch.dump()}
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
            if state.agents:
                entry["agents"] = state.agents
            self.transcript.append(entry)
            if self.on_change is not None:
                self.on_change()
        if context_window is not None:
            yield {"type": "context", **self.context(context_window)}
        yield {"type": "done"}

    async def _compact_if_full(self, agent: Agent, window: int | None,
                               compact_at: float) -> AsyncIterator[dict[str, Any]]:
        if window is None or not self._full(window, compact_at):
            return
        yield {"type": "status", "message": "Compactando la conversación…"}
        info = await self.compact(agent, window)
        # Marked on the turn's question so the panel shows where the agent's memory was folded.
        question = next((t for t in reversed(self.transcript) if t["role"] == "user"), None)
        if question is not None:
            question["compacted"] = True
        yield {"type": "compacted", **info}

    @staticmethod
    async def _draft_plan(planner: Agent[None, pl.PlanDraft], request: str, earlier: list[dict[str, Any]],
                          ) -> tuple[pl.Plan, pl.Scratch, bool, list[str], list[pl.AgentTask]]:
        """The plan of the turn, the workspace it opens with (the goal and the alternatives weighed), whether the
        planner answered at all (an empty plan from a failed planner does not mean there is nothing to do), the
        questions the user must answer before anything runs (then there are no steps) and the tasks it hands to the
        team."""
        try:
            result = await planner.run(pl.planner_prompt(request, workspace_context(), earlier),
                                       usage_limits=UsageLimits(request_limit=3))
            draft = result.output
            questions = [q.strip() for q in draft.questions if q and q.strip()]
            if questions:
                return pl.Plan(), pl.Scratch(), True, questions, []
            scratch = pl.Scratch.of(draft)
            if draft.delegations and not scratch.goal:  # a turn the team does whole still has a goal to show
                scratch.goal = draft.goal.strip()
            return pl.Plan.of(draft.steps), scratch, True, [], list(draft.delegations)
        except Exception as exc:  # noqa: BLE001 - planning is an aid: without a plan the agent still works
            log.warning("Planner failed, running without a plan: %s", exc)
            return pl.Plan(), pl.Scratch(), False, [], []

    async def _run_team(self, team: Sequence[TeamMember], tasks: Sequence[pl.AgentTask], request: str,
                        state: _TurnState, *, budget: int, plan: bool) -> AsyncIterator[dict[str, Any]]:
        """The members' turns, one after the other (each sees the objects the earlier ones built). A member works in
        a conversation of its own and its events reach the UI tagged with ``agent``; what it did is kept in
        ``state.agents`` for the lead, the transcript and the report."""
        members = {m.id: m for m in team}
        for task in tasks:
            member = members.get(task.agent_id)
            if member is None or not task.task.strip():
                continue
            label = f"{member.emoji} {member.name}".strip()
            yield {"type": "agent_spawn", "agent": member.id, "name": member.name, "emoji": member.emoji,
                   "task": task.task}
            yield {"type": "status", "message": f"{label} trabajando…"}
            session = ChatSession()
            ok = True
            prompt = delegated_prompt(task.task, request, earlier_conclusions(self.transcript, member.id))
            async for ev in _in_own_task(session.stream(member.agent, prompt, max_tool_calls=budget, plan=plan)):
                if ev["type"] == "error":  # the member's failure, not the turn's: shown on its tab only
                    ok = False
                    continue
                if ev["type"] not in _MEMBER_HIDDEN:
                    yield {**ev, "agent": member.id}
            entry = next((m for m in reversed(session.transcript) if m["role"] == "assistant"), {})
            work = {"id": member.id, "name": member.name, "emoji": member.emoji, "task": task.task, "ok": ok,
                    **{k: entry[k] for k in ("content", "turn_id", "cells", "plan", "scratch", "intro") if k in entry}}
            state.agents.append(work)
            yield {"type": "agent_done", "agent": member.id, "ok": ok, "content": work.get("content", "")}

    def _add_follow_ups(self, texts: list[str]) -> bool:
        """The explainer's follow-ups as next steps: a reference to an object that does not exist loses its brackets,
        and a step already open is not added twice. Whether any was added."""
        added = False
        for text in texts:
            for name in pl.unknown_objects(text):
                text = re.sub(r"\[\[\s*" + re.escape(name) + r"\s*\]\]", name, text, flags=re.I)
            text = " ".join(text.split())
            if text and not any(i["text"] == text for i in self.next_steps.open()):
                self.next_steps.add(text)
                added = True
        return added

    @staticmethod
    async def _explain(explainer: Agent[None, fx.Findings], request: str, cells: list[dict[str, Any]],
                       answer: str = "", *, objects: Sequence[str] = (), open_steps: Sequence[str] = (),
                       ) -> tuple[list[dict[str, Any]], list[str], list[str], list[str]]:
        try:
            found, conclusions, issues, follow_ups = await fx.explain(explainer, request, cells, answer=answer,
                                                                      objects=objects, open_steps=open_steps)
        except Exception as exc:  # noqa: BLE001 - the cells and the answer stand without it
            log.warning("Explaining cells failed: %s", exc)
            return [], [], [], []
        for c in cells:
            if c["id"] in found:
                c["finding"] = found[c["id"]]
        steps = [{"type": "finding", "ids": [cid], "text": text, "strip": 0} for cid, text in found.items()]
        return steps, conclusions, issues, follow_ups

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
                            if final or state.asked or event.part.tool_name in pl.WORK_TOOLS | BOOKKEEPING_TOOLS:
                                continue
                            args = event.part.args_as_dict()
                            cell = {"id": event.part.tool_call_id, "name": event.part.tool_name, "args": args,
                                    "code": _code(event.part.tool_name, args)}
                            state.cells[cell["id"]] = cell
                            yield {"type": "tool_call", **cell}
                        elif isinstance(event, FunctionToolResultEvent):
                            if final or state.asked:  # once the user is asked, later calls are refused: no cells
                                continue
                            if event.part.tool_name in pl.PLAN_TOOLS:
                                yield {"type": "plan", "steps": state.plan.dump()}
                                continue
                            if event.part.tool_name in pl.SCRATCH_TOOLS:
                                yield {"type": "scratch", **state.scratch.dump()}
                                continue
                            if event.part.tool_name in pl.NEXT_TOOLS:
                                state.noted_next = True
                                yield {"type": "next_steps", "items": self.next_steps.dump()}
                                continue
                            if event.part.tool_name in pl.ASK_TOOLS:
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
    def _drop_leaked_call(state: _TurnState) -> list[dict[str, Any]]:
        """A final text that is a tool call written as text (a model that cannot call tools says
        ``scratch_note{...}`` instead of answering) leaves the answer; a note's text is what the model meant to say,
        so it becomes the answer and the turn's conclusion."""
        leaked = pl.leaked_call(state.final_text())
        if leaked is None:
            return []
        name, body = leaked
        strip = sum(map(len, state.tail))
        del state.answer[-len(state.tail):]
        state.tail = []
        events: list[dict[str, Any]] = [{"type": "finding", "ids": [], "text": "", "strip": strip}]
        note = pl.leaked_note(body) if name in pl.SCRATCH_TOOLS else ""
        if note:
            state.scratch.add("conclusion", note)
            events += state.say(note)
        return events

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
    asked: list[str] = field(default_factory=list)  # questions the turn stops on, until the user answers
    noted_next: bool = False  # the agent wrote or closed a next step itself
    agents: list[dict[str, Any]] = field(default_factory=list)  # what each team member did in this turn, in order

    def objects(self) -> list[str]:
        """Names of the objects the turn's cells produced, once each, in order."""
        names: dict[str, None] = {}
        for c in self.cells.values():
            if not c.get("ok"):
                continue
            for name in [c.get("object"), *(o["name"] for o in c.get("objects") or [])]:
                if name:
                    names[name] = None
        return list(names)

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
