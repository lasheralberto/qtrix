"""Reporting: the econometric report of a conversation, rewritten after every turn.

Each turn the reporter gets the report so far, the new turn(s) (the question, the answer, the findings and the
workspace notes) and a compact reading of the objects they produced, and returns the improved report. The report
points at objects by name; the UI renders them from the object store.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Sequence
from typing import Any

from pydantic import BaseModel, Field
from pydantic_ai import Agent, UsageLimits

log = logging.getLogger(__name__)

REPORTER_INSTRUCTIONS = """\
You write and keep up to date the econometric report of one conversation, for an applied economist who will read
it as a paper-style document. After each turn you receive the report so far and the new turns; return the whole
report, improved: rewrite and reorganise, do not just append.
- Write in the language of the user. Be precise and concise, in an academic register.
- Structure: Datos, Especificación, Estimación, Diagnóstico, Robustez (only the sections the work supports), then
  conclusions, limitations and next steps. Merge turns that treat the same question into one section.
- Put the objects that support a section in its `objects` (name exactly as given, a short caption). Use only
  objects listed under "Objects"; never invent names. Pick the view only if it was given.
- Use only numbers that appear in the turns or in the objects' readings; never invent figures. Say plainly what a
  failed or missing step means for the conclusions.
- `body_md` is Markdown (it may use **bold**, lists and LaTeX between $ ... $).
- A turn may come with the work of a team of specialists (under "Team"). The report is the conclusion of the whole
  analysis: integrate their results where they belong, credit each specialist by name, and state plainly where they
  disagree and which reading the evidence favours.
"""


class ObjectRef(BaseModel):
    name: str = Field(description="Object name, exactly as listed (eq01, gph02, tab01...).")
    view: str | None = Field(default=None, description="View of the object, if one was listed.")
    caption: str = Field(default="", description="Caption of the figure or table, one sentence.")


class ReportSection(BaseModel):
    heading: str
    body_md: str
    objects: list[ObjectRef] = Field(default_factory=list)


class ConversationReport(BaseModel):
    title: str
    abstract: str = ""
    sections: list[ReportSection] = Field(default_factory=list)
    conclusions_md: str = ""
    limitations_md: str = ""
    next_steps: list[str] = Field(default_factory=list)


def build_reporter(model: Any) -> Agent[None, ConversationReport]:
    return Agent(model, name="reporter", output_type=ConversationReport, retries=2,
                 instructions=REPORTER_INSTRUCTIONS)


def _clip(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit] + "…"


def turns_of(transcript: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """The transcript as turns: each user question with the assistant entry that answered it."""
    turns: list[dict[str, Any]] = []
    question = ""
    for m in transcript:
        if m.get("role") == "user":
            question = str(m.get("content") or "")
        elif m.get("role") == "assistant" and m.get("turn_id"):
            turns.append({"question": question, **m})
    return turns


def object_names(turns: Sequence[dict[str, Any]]) -> list[str]:
    names: list[str] = []
    for t in turns:
        for c in [*(t.get("cells") or []), *(c for a in t.get("agents") or [] for c in a.get("cells") or [])]:
            n = c.get("object")
            if n and n not in names:
                names.append(n)
    return names


def _work(parts: list[str], work: dict[str, Any]) -> None:
    for note in (work.get("scratch") or {}).get("notes", []):
        parts.append(f"- [{note.get('section')}] {note.get('text')}")
    for c in work.get("cells") or []:
        if c.get("finding"):
            parts.append(f"- {c.get('name')}{' → ' + c['object'] if c.get('object') else ''}: {c['finding']}")


def reading(name: str) -> dict[str, Any] | None:
    """Compact reading of an object (its default view's ``llm`` dict), None when it cannot be computed."""
    from ..objects.views import compute

    try:
        result, ref = compute(name)
    except Exception:  # noqa: BLE001 - an object that cannot be read just goes without reading
        return None
    return {"type": result.kind, "view": ref["view"], "title": result.title, "reading": result.llm}


def reporter_prompt(previous: ConversationReport | None, turns: Sequence[dict[str, Any]],
                    objects: dict[str, dict[str, Any] | None]) -> str:
    parts = ["## Report so far", previous.model_dump_json(indent=1) if previous else "(none yet)", "", "## New turns"]
    for t in turns:
        parts += [f"### Question: {t.get('question', '')}", f"Answer: {_clip(str(t.get('content') or ''), 2500)}"]
        _work(parts, t)
        for a in t.get("agents") or []:
            parts += [f"#### Team: {a.get('name', a.get('id'))} — task: {a.get('task', '')}",
                      f"Conclusions: {_clip(str(a.get('content') or '(failed)'), 2000)}"]
            _work(parts, a)
        parts.append("")
    parts.append("## Objects")
    for name, info in objects.items():
        parts.append(f"- {name}: " + (_clip(json.dumps(info, default=str, ensure_ascii=False), 1500) if info
                                      else "(no reading)"))
    return "\n".join(parts)


def _valid(report: ConversationReport, known: set[str]) -> ConversationReport:
    """Drop references to objects the notebook does not have."""
    for s in report.sections:
        s.objects = [o for o in s.objects if o.name in known]
    return report


async def update_report(reporter: Agent[None, ConversationReport], previous: ConversationReport | None,
                        turns: Sequence[dict[str, Any]], *, known: set[str] | None = None) -> ConversationReport:
    """The report after ``turns``; the previous one stands when the model fails."""
    if not turns:
        if previous is not None:
            return previous
        return ConversationReport(title="Reporte")
    names = object_names(turns)
    names += [o.name for s in (previous.sections if previous else []) for o in s.objects if o.name not in names]
    objects = await asyncio.to_thread(lambda: {n: reading(n) for n in names})
    try:
        result = await reporter.run(reporter_prompt(previous, turns, objects),
                                    usage_limits=UsageLimits(request_limit=3))
    except Exception:  # noqa: BLE001 - a failed update must not cost the user the report they have
        log.exception("The reporting agent failed; keeping the previous report")
        if previous is not None:
            return previous
        raise
    return _valid(result.output, known if known is not None else {n for n, i in objects.items() if i is not None})
