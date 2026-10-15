"""Plan-and-execute around the lead agent: a planner drafts the steps of a turn, the executor (the ReAct loop of the
lead agent) works through them and keeps the plan up to date with the ``plan_*`` tools, and the session keeps the
loop going until every step is settled and there is a final answer."""

from __future__ import annotations

import re
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, Field
from pydantic_ai import Agent, FunctionToolset, ModelRetry, RunContext, Tool, ToolDefinition
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, ToolCallPart

from ..tools.registry import all_tools

PLANNER_INSTRUCTIONS = """\
You are the planner of qtrix, a research workbench for applied econometrics. You do not run anything: you
decide the steps that an executor agent with the tools listed below will follow to fulfil the user's LATEST
message, and it adapts them as results come in.

Rules:
- Return no steps when the message needs no tools: greetings, thanks, or a question answerable from the
  conversation so far. The executor then simply answers.
- Otherwise 2 to 6 short, concrete steps in the user's language, each one verifiable action: name the dataset,
  variables, estimator or test when you know them ("Estimar OLS de lwage sobre educ y exper con errores HC1").
- When the request cannot be carried out without a decision only the user can make (the outcome, the treatment or
  the estimand, which of several readings of the request they mean, a missing identification choice), return NO
  steps and put 1 to 3 short, concrete questions in `questions`, in the user's language. Nothing runs until they
  answer, so ask only what blocks the work and cannot be learnt by inspecting the data shown below; whatever has a
  sensible default, decide it and say so in the plan instead of asking.
- Start by inspecting data only when the request depends on columns or structure not shown below.
- When the message carries a plan approved by the user, your steps are that plan's steps in the same order (condensed
  to at most 8 if it has more), and you never ask questions: the user already settled them.
- Do not add a step for writing the final answer: the executor always ends with it.
- When you plan steps, also state the goal in one sentence and, when there is a real choice of method, up to 3
  alternatives you weighed, each as "approach: why chosen or discarded" in a few words. The user sees them in the
  agent's workspace: keep them operational, no long deliberation.
"""

PLAN_MODE_INSTRUCTIONS = """\
You are the planning mode of qtrix, a research workbench for applied econometrics. The user wants to review a plan
BEFORE anything runs, so you only write the plan: you run nothing and you do not answer the request itself.
The plan is a Markdown document the user will edit and approve; an executor agent with the tools listed below will
then follow it step by step.

Write everything in the user's language, section headings included (translate the ones below). Use these sections,
in this order, as `##` headings:

## Goal
One or two sentences: what the work must establish and how the result will be used.

## Data and assumptions
Bullets: the datasets and variables you rely on, and every assumption you make. Decide whatever has a sensible
default and say so here.

## Steps
One `###` heading per step, numbered and titled with the action: `### 1. Check the data`. 3 to 10 steps. Under each
heading, exactly these bullets, each starting with a bold label (translated): **Action** (what to do: dataset,
variables, estimator or test), **Tool** (the registry tool(s) that do it, by exact name), **Check** (the diagnostic
that tells whether it worked), **Output** (the object, table or decision it produces, named when you can).

## Alternatives considered
Up to 3 bullets "approach: why chosen or discarded". Omit the section when there is no real choice.

## Risks
Bullets: a threat to validity and what you will do if it materialises. Omit the section when there are none.

## Open questions
Only decisions that block the work and the data cannot settle (the outcome, the treatment, the estimand), each with
the default you would take if the user does not answer. Omit the section when there are none.

Formatting rules:
- Put every tool name, dataset and variable in `backticks`, and nothing else: the interface shows each tool's help
  when the user hovers its name. Use only tool names from the list below, spelled exactly.
- Plain Markdown only: no LaTeX or `$` maths (write R², not a formula), no tables, no bold outside the labels above.
- Be concrete and operational, not generic. Output only the document, no preface and no closing remarks.
"""

#: Steps the executor may settle; "pending" ones keep the loop going.
Status = Literal["pending", "done", "skipped"]


class PlanDraft(BaseModel):
    """The planner's output."""

    steps: list[str] = Field(default_factory=list, max_length=8,
                             description="Ordered steps; empty when the message needs no tools.")
    goal: str = Field("", description="What the turn must achieve, in one sentence (empty without steps).")
    alternatives: list[str] = Field(default_factory=list, max_length=3,
                                    description='Approaches weighed, each "approach: why chosen or discarded".')
    questions: list[str] = Field(default_factory=list, max_length=3,
                                 description="Questions the user must answer before anything runs; then no steps.")
    delegations: list[AgentTask] = Field(default_factory=list, max_length=10,
                                         description="Parts of the request handed to team members; empty without a "
                                                     "team or when you handle it alone.")


class AgentTask(BaseModel):
    """A part of the request the lead hands to one member of its team."""

    agent_id: str = Field(description="Id of the team member, exactly as in the roster.")
    task: str = Field(description="The self-contained task for that member, in the user's language: what to do, on "
                                  "which data, and what to report back.")


PlanDraft.model_rebuild()

TEAM_PLANNER_INSTRUCTIONS = """\

## Your team
The lead executor works with the specialists below. Whenever the request needs tools, the work is theirs: hand
EVERY part of it over in `delegations` (agent_id and a self-contained task), split by the members' roles, so that
each one works on what fits it; a request with one part goes to the one member whose role fits best. Only a
message that needs no tools (a greeting, a question answerable from the conversation) has no delegations. Each
member works alone and in order, sees the objects the earlier ones built and remembers its own earlier conclusions
in this conversation, so a follow-up goes to the member that did the related work. The lead then gives the final
answer from their conclusions; your `steps` are only what the lead itself still has to do (usually none).
"""


def roster_text(members: Sequence[Any]) -> str:
    """The team as the planner sees it: one line per member (``id``, ``name``, ``behavior``, ``skills``)."""
    lines = []
    for m in members:
        role = " ".join(m.behavior.split())
        role = role if len(role) <= 300 else role[:300] + "…"
        skills = f" (skills: {', '.join(m.skills)})" if m.skills else ""
        lines.append(f"- {m.id}: {m.name}" + (f" — {role}" if role else "") + skills)
    return "\n".join(lines)


@dataclass
class Step:
    text: str
    status: Status = "pending"
    note: str = ""


@dataclass
class Plan:
    steps: list[Step] = field(default_factory=list)

    @classmethod
    def of(cls, texts: Sequence[str]) -> Plan:
        return cls([Step(t.strip()) for t in texts if t and t.strip()])

    def pending(self) -> list[int]:
        return [i for i, s in enumerate(self.steps, 1) if s.status == "pending"]

    def render(self) -> str:
        mark = {"pending": "[ ]", "done": "[x]", "skipped": "[-]"}
        return "\n".join(f"{i}. {mark[s.status]} {s.text}" + (f" ({s.note})" if s.note else "")
                         for i, s in enumerate(self.steps, 1))

    def dump(self) -> list[dict[str, str]]:
        return [{"text": s.text, "status": s.status, "note": s.note} for s in self.steps]


#: Sections of the agent's workspace it can write to; plan, tasks and progress come from the plan and the cells.
NoteSection = Literal["idea", "conclusion", "issue", "result"]


@dataclass
class Scratch:
    """The agent's workspace in one turn (E8): the goal and alternatives of the plan, and the short notes the agent
    writes as it works. Operational summaries only, shown to the user beside the analysis."""

    goal: str = ""
    notes: list[dict[str, str]] = field(default_factory=list)

    @classmethod
    def of(cls, draft: PlanDraft) -> Scratch:
        if not draft.steps:
            return cls()
        ideas = [{"section": "idea", "text": a.strip()} for a in draft.alternatives if a.strip()]
        return cls(draft.goal.strip(), ideas)

    def add(self, section: str, text: str) -> None:
        self.notes.append({"section": section, "text": text})

    def dump(self) -> dict[str, Any]:
        return {"goal": self.goal, "notes": list(self.notes)}

    def __bool__(self) -> bool:
        return bool(self.goal or self.notes)


#: A reference to a workfile object inside a next step's text: ``[[eq_salary]]``.
OBJECT_REF = re.compile(r"\[\[\s*([A-Za-z][A-Za-z0-9_]{0,31})\s*\]\]")


def object_refs(text: str) -> list[str]:
    """Names of the objects a text refers to with ``[[name]]``, once each, lowercased, in order."""
    return list(dict.fromkeys(m.lower() for m in OBJECT_REF.findall(text)))


@dataclass
class NextSteps:
    """What is left to do in the conversation (not in one turn): the agent and the user both write it, it survives
    the turns, and the agent reads it at the start of each one. A step may refer to workfile objects as ``[[name]]``."""

    items: list[dict[str, Any]] = field(default_factory=list)

    def load(self, items: Sequence[dict[str, Any]] | None) -> None:
        self.items = [dict(i) for i in items or []]

    def _new_id(self) -> str:
        n = 1 + max((int(i["id"][1:]) for i in self.items if i["id"][1:].isdigit()), default=0)
        return f"n{n}"

    def add(self, text: str, author: str = "agent") -> dict[str, Any]:
        item = {"id": self._new_id(), "text": " ".join(text.split()), "done": False, "author": author}
        self.items.append(item)
        return item

    def get(self, ref: str | int) -> dict[str, Any] | None:
        """A step by id (``n3``) or by its 1-based position among all steps."""
        if isinstance(ref, int) or str(ref).isdigit():
            pos = int(ref)
            return self.items[pos - 1] if 1 <= pos <= len(self.items) else None
        return next((i for i in self.items if i["id"] == ref), None)

    def remove(self, ref: str) -> bool:
        item = self.get(ref)
        if item is None:
            return False
        self.items.remove(item)
        return True

    def open(self) -> list[dict[str, Any]]:
        return [i for i in self.items if not i["done"]]

    def dump(self) -> list[dict[str, Any]]:
        return [{**i, "objects": object_refs(i["text"])} for i in self.items]

    def render(self) -> str:
        """The list for the agent's prompt: numbered, with the done ones marked."""
        return "\n".join(f"{n}. [{'x' if i['done'] else ' '}] {i['text']}"
                         + (" (added by the user)" if i.get("author") == "user" else "")
                         for n, i in enumerate(self.items, 1))

    def __bool__(self) -> bool:
        return bool(self.items)


def unknown_objects(text: str) -> list[str]:
    """The objects ``text`` refers to that are not in the workfile."""
    from ..context import get_context  # imported late: the context imports the agents' tools

    store = get_context().objects
    return [n for n in object_refs(text) if not store.exists(n)]


_plan: ContextVar[Plan | None] = ContextVar("qtrix_plan", default=None)
_scratch: ContextVar[Scratch | None] = ContextVar("qtrix_scratch", default=None)
_next: ContextVar[NextSteps | None] = ContextVar("qtrix_next_steps", default=None)
_finalizing: ContextVar[bool] = ContextVar("qtrix_finalizing", default=False)
#: Questions the executor has put to the user in this turn (a mutable holder: tools run in a copied context).
_asked: ContextVar[list[str] | None] = ContextVar("qtrix_asked", default=None)

PLAN_TOOLS = frozenset({"plan_update", "plan_revise"})
SCRATCH_TOOLS = frozenset({"scratch_note"})
ASK_TOOLS = frozenset({"ask_user"})
NEXT_TOOLS = frozenset({"next_step", "next_step_done"})
#: Bookkeeping tools of the agent: never shown as Python cells.
WORK_TOOLS = PLAN_TOOLS | SCRATCH_TOOLS | ASK_TOOLS | NEXT_TOOLS
NOTE_LIMIT = 400  # characters of one workspace note


@contextmanager
def active_plan(plan: Plan | None) -> Iterator[None]:
    token = _plan.set(plan)
    try:
        yield
    finally:
        _plan.reset(token)


@contextmanager
def active_scratch(scratch: Scratch | None) -> Iterator[None]:
    token = _scratch.set(scratch)
    try:
        yield
    finally:
        _scratch.reset(token)


@contextmanager
def active_next(steps: NextSteps | None) -> Iterator[None]:
    token = _next.set(steps)
    try:
        yield
    finally:
        _next.reset(token)


@contextmanager
def finalizing() -> Iterator[None]:
    """While active every tool refuses to run: the model must answer with what it already has."""
    token = _finalizing.set(True)
    try:
        yield
    finally:
        _finalizing.reset(token)


@contextmanager
def asking(holder: list[str]) -> Iterator[None]:
    """Collects the questions ``ask_user`` receives; once there are any, every other tool refuses to run."""
    token = _asked.set(holder)
    try:
        yield
    finally:
        _asked.reset(token)


def check_tools_allowed() -> None:
    if _finalizing.get():
        raise ModelRetry("Tools are disabled now. Write your final answer to the user as text, from the results above.")
    if _asked.get():
        raise ModelRetry("You asked the user a question: nothing runs until they answer. Stop calling tools and end "
                         "your turn with a short text that asks it.")


def ask_user(questions: list[str]) -> str:
    """Ask the user what you need before going on, and stop. Use it when you cannot proceed without a decision only
    they can make (outcome, treatment, estimand, which reading of the request they mean) and the data cannot tell
    you. The turn ends after this call and resumes with their answer: do not run anything else meanwhile.

    Args:
        questions: One to three short, concrete questions, in the user's language.
    """
    check_tools_allowed()
    holder = _asked.get()
    asked = [" ".join(q.split()) for q in questions if q and q.strip()][:3]
    if holder is None or not asked:
        raise ModelRetry("Give at least one question.")
    holder.extend(asked)
    return "Questions sent to the user. End your turn now: write them as your reply and wait for the answer."


def questions_text(questions: Sequence[str]) -> str:
    """The reply of a turn that stops to ask: the questions, numbered when there are several."""
    items = [q.strip() for q in questions if q and q.strip()]
    if len(items) == 1:
        return items[0]
    return "Antes de empezar necesito saber:\n\n" + "\n".join(f"{i}. {q}" for i, q in enumerate(items, 1))


def plan_update(step: int, status: Literal["done", "skipped"] = "done", note: str = "") -> str:
    """Mark a step of the current plan as done or skipped, as soon as it is settled.

    Args:
        step: Number of the step (1-based).
        status: "done" when finished, "skipped" when it turned out unnecessary or impossible.
        note: Short outcome or reason, e.g. "eq01 estimated" or "no time series in the data".
    """
    check_tools_allowed()
    plan = _plan.get()
    if plan is None or not 1 <= step <= len(plan.steps):
        raise ModelRetry(f"There is no step {step}. Current plan:\n{plan.render() if plan else '(none)'}")
    plan.steps[step - 1].status, plan.steps[step - 1].note = status, note.strip()
    left = plan.pending()
    return plan.render() + ("\nNext: step " + str(left[0]) if left else "\nAll steps settled: give your final answer.")


def plan_revise(steps: list[str]) -> str:
    """Replace the pending steps of the plan when results show it must change (settled steps are kept).

    Args:
        steps: The new remaining steps, in order.
    """
    check_tools_allowed()
    plan = _plan.get()
    if plan is None:
        raise ModelRetry("There is no plan in this turn.")
    plan.steps = [s for s in plan.steps if s.status != "pending"] + Plan.of(steps).steps
    return plan.render()


def scratch_note(section: NoteSection, text: str) -> str:
    """Write a short note in your workspace, the panel where the user follows your work beside the analysis.

    Use it sparingly, only when it adds something the cells do not show: an "idea" when you weigh alternatives
    before choosing (approach and why), a "conclusion" when a key result settles a question (provisional,
    with the number behind it), an "issue" for a caveat the analyst must keep in mind (a failed assumption, weak
    identification, a data problem that limits the estimate; never a tool error you fixed), a "result" for a
    deliverable the turn produced (an object, a table, a decision).
    One or two sentences, in the user's language. Operational summaries only: never your internal deliberation.

    Args:
        section: "idea", "conclusion", "issue" or "result".
        text: The note, one or two sentences.
    """
    check_tools_allowed()
    scratch = _scratch.get()
    text = " ".join(text.split())
    if scratch is None:
        raise ModelRetry("There is no workspace in this turn.")
    if not text:
        raise ModelRetry("The note is empty.")
    if len(text) > NOTE_LIMIT:
        raise ModelRetry(f"Too long ({len(text)} characters): summarise it in at most {NOTE_LIMIT}.")
    scratch.add(section, text)
    return "Noted."


def next_step(text: str) -> str:
    """Leave a step for later in the conversation's "Next steps": work that is still open and that you or the user
    should pick up in a later turn (a robustness check you did not run, a model to compare, a question to settle).

    Refer to workfile objects as [[name]], e.g. "Compare [[eq_salary_ext]] with a fixed-effects model": the user can
    open them from the list and you will see the list again at the start of every turn. One to three per turn,
    before the final answer: real follow-ups, not the steps of the current plan.

    Args:
        text: The step, one or two sentences, in the user's language. Objects as [[name]].
    """
    check_tools_allowed()
    steps = _next.get()
    text = " ".join(text.split())
    if steps is None:
        raise ModelRetry("There is no next-steps list in this turn.")
    if not text:
        raise ModelRetry("The step is empty.")
    if len(text) > NOTE_LIMIT:
        raise ModelRetry(f"Too long ({len(text)} characters): summarise it in at most {NOTE_LIMIT}.")
    if (missing := unknown_objects(text)):
        raise ModelRetry("These objects do not exist: " + ", ".join(f"[[{n}]]" for n in missing)
                         + ". Refer only to objects of the workfile (obj_list), or drop the brackets.")
    if any(i["text"] == text and not i["done"] for i in steps.items):
        return "Already in the list."
    steps.add(text)
    return "Added.\n" + steps.render()


def next_step_done(step: int) -> str:
    """Mark a step of the "Next steps" list as done when this turn settled it.

    Args:
        step: Number of the step in the list (1-based).
    """
    check_tools_allowed()
    steps = _next.get()
    item = steps.get(step) if steps is not None else None
    if item is None:
        raise ModelRetry(f"There is no next step {step}. Current list:\n{steps.render() if steps else '(empty)'}")
    item["done"] = True
    return steps.render()


async def _when_scratching(ctx: RunContext[None], tool_def: ToolDefinition) -> ToolDefinition | None:
    return tool_def if _scratch.get() is not None else None


async def _when_next_steps(ctx: RunContext[None], tool_def: ToolDefinition) -> ToolDefinition | None:
    return tool_def if _next.get() is not None else None


async def _when_asking(ctx: RunContext[None], tool_def: ToolDefinition) -> ToolDefinition | None:
    return tool_def if _asked.get() is not None else None


async def _when_planned(ctx: RunContext[None], tool_def: ToolDefinition) -> ToolDefinition | None:
    return tool_def if _plan.get() is not None else None


def plan_toolset() -> FunctionToolset:
    tools = [Tool(fn, takes_ctx=False, docstring_format="google", prepare=_when_planned)
             for fn in (plan_update, plan_revise)]
    tools.append(Tool(scratch_note, takes_ctx=False, docstring_format="google", prepare=_when_scratching))
    tools.append(Tool(ask_user, takes_ctx=False, docstring_format="google", prepare=_when_asking))
    tools += [Tool(fn, takes_ctx=False, docstring_format="google", prepare=_when_next_steps)
              for fn in (next_step, next_step_done)]
    return FunctionToolset(tools, max_retries=3)


def tool_catalog(namespaces: Sequence[str] | None = None) -> str:
    lines = []
    for spec in all_tools(namespaces=namespaces):
        first = spec.description.splitlines()[0] if spec.description else ""
        lines.append(f"- {spec.name}: {first}")
    return "\n".join(lines)


def build_planner(model: Any, *, namespaces: Sequence[str] | None = None, roster: str = "") -> Agent[None, PlanDraft]:
    """``roster`` (``roster_text``) lists the team the planner may delegate to; without one it never delegates."""
    team = TEAM_PLANNER_INSTRUCTIONS + roster + "\n" if roster else "\nThere is no team: leave `delegations` empty.\n"
    return Agent(model, name="planner", output_type=PlanDraft, retries=2,
                 instructions=PLANNER_INSTRUCTIONS + team + "\n## Executor tools\n" + tool_catalog(namespaces))


def build_plan_drafter(model: Any, *, namespaces: Sequence[str] | None = None) -> Agent[None, str]:
    """Plan mode: writes the detailed, user-editable plan of a request as Markdown, without running anything."""
    return Agent(model, name="plan_drafter", retries=2,
                 instructions=PLAN_MODE_INSTRUCTIONS + "\n## Executor tools\n" + tool_catalog(namespaces))


def approved_plan_prompt(plan: str) -> str:
    """The plan the user approved in plan mode, as it goes with their request into the normal flow."""
    return ("\n\n[Plan approved by the user for this request. Follow it: its steps are your plan, in order, and "
            "what it says overrides your own preferences (the user may have edited it). Change it only if results "
            "show it cannot be carried out, and say so.]\n" + plan.strip())


def planner_prompt(message: str, workspace: str, transcript: Sequence[dict[str, Any]], *, turns: int = 3,
                   limit: int = 600) -> str:
    """What the planner sees: the workspace, the end of the conversation and the new message (not the raw model
    history, which is long and full of tool calls the planner cannot make)."""
    recent = []
    for m in list(transcript)[-2 * turns:]:
        text = (m.get("content") or "").strip()
        if text:
            recent.append(f"{m['role']}: " + (text if len(text) <= limit else text[:limit] + "…"))
    return (workspace + "\n\n## Conversation so far\n" + ("\n".join(recent) or "(new conversation)")
            + "\n\n## Latest message from the user\n" + message)


def executor_prompt(message: str, plan: Plan) -> str:
    if not plan.steps:
        return message
    return (message + "\n\n[Plan for this request. Work through it step by step with the tools; after each step call "
            "plan_update (done or skipped, with a short note); if results show the plan is wrong, plan_revise it. "
            "When every step is settled, write the final answer.]\n" + plan.render())


def continue_prompt(plan: Plan) -> str:
    return ("[Not finished: these plan steps are still pending. Continue with them now; mark a step skipped if it "
            "is no longer needed. Then write the final answer.]\n" + plan.render())


def final_prompt(message: str) -> str:
    return ("[Write now your final answer to the user's request, in their language, from the results above. Do not "
            "call tools. Report what was done, the key numbers and their interpretation, and anything left undone "
            "and why.]\nRequest: " + message)


_LEAKED_CALL = re.compile(r"\s*(?P<name>[a-z][a-z0-9_]*)\s*\{(?P<body>.*)\}\s*", re.S)
_LEAKED_TEXT = re.compile(r"""["']?text["']?\s*[:=]\s*["']?(?P<text>.*?)["']?\s*$""", re.S)


def leaked_call(text: str) -> tuple[str, str] | None:
    """A model with its tools off for the final answer sometimes writes a call as text instead
    (``scratch_note{section:conclusion,text:...}``): the tool's name and the raw arguments, when the whole text is
    one such call."""
    m = _LEAKED_CALL.fullmatch(text)
    if m is None:
        return None
    known = {spec.name for spec in all_tools()} | PLAN_TOOLS | SCRATCH_TOOLS | ASK_TOOLS
    return (m["name"], m["body"]) if m["name"] in known else None


def leaked_note(body: str) -> str:
    """The ``text`` argument of a leaked ``scratch_note`` call, on one line (empty when there is none)."""
    m = _LEAKED_TEXT.search(body)
    return " ".join(m["text"].split()) if m else ""


def resumable(messages: list[ModelMessage]) -> list[ModelMessage]:
    """A run cut short ends with tool calls nobody answered (or an empty request): drop them so the history can be
    sent to the model again."""
    out = list(messages)
    while out and ((isinstance(out[-1], ModelRequest) and not out[-1].parts)
                   or (isinstance(out[-1], ModelResponse) and any(isinstance(p, ToolCallPart) for p in out[-1].parts))):
        out.pop()
    return out
