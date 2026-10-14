"""Findings: what each cell of a turn showed, in the model's words.

The agent explains a result in the text it writes right after it, before its next call: the session moves that text
from the answer to the cells it explains. Cells still unexplained when the turn ends (usually the last ones, whose
reading went into the final answer) get a short finding from one extra call to the same model.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from pydantic import BaseModel, Field
from pydantic_ai import Agent, UsageLimits

EXPLAINER_INSTRUCTIONS = """\
You explain results of an econometric analysis to the applied economist who ran it. For each cell (a call and
the result it returned) write its finding: 1 to 3 sentences in the user's language with the key numbers and what
they mean (sign, magnitude, significance, a failed assumption...). Use only what the result shows; never invent
numbers. Return one finding per cell id given.

When the prompt also gives the answer written for the turn, return in "conclusions" the 1 to 3 key conclusions
the analysis settled (one sentence each, with the number behind it); otherwise leave "conclusions" empty. In that case also return in "issues" the econometric caveats the analyst must
keep in mind (a failed assumption or diagnostic, weak identification or instruments, small or unbalanced samples,
influential data, a specification that limits the interpretation), one sentence each. Never list tool or syntax
errors the agent fixed along the way. Leave "issues" empty when there are none; do not invent any.
"""


class CellFinding(BaseModel):
    id: str = Field(description="The cell id, exactly as given.")
    finding: str


class Findings(BaseModel):
    findings: list[CellFinding] = Field(default_factory=list)
    conclusions: list[str] = Field(default_factory=list)
    issues: list[str] = Field(default_factory=list)


def build_explainer(model: Any) -> Agent[None, Findings]:
    return Agent(model, name="explainer", output_type=Findings, retries=2, instructions=EXPLAINER_INSTRUCTIONS)


def explainer_prompt(request: str, cells: Sequence[dict[str, Any]], *, limit: int = 1200,
                     answer: str = "") -> str:
    parts = [f"User request: {request}", ""]
    if answer:
        parts += ["Answer written for the turn (give its key conclusions):", answer[:3000], ""]
    for c in cells:
        content = str(c.get("content") or "")
        parts += [f"### Cell {c['id']}", c.get("code") or c["name"],
                  "Result:", content if len(content) <= limit else content[:limit] + "…", ""]
    return "\n".join(parts)


async def explain(explainer: Agent[None, Findings], request: str, cells: Sequence[dict[str, Any]], *,
                  answer: str = "") -> tuple[dict[str, str], list[str], list[str]]:
    """Findings by cell id for ``cells`` and, given the turn's ``answer``, its key conclusions and the econometric
    issues to keep in mind (empty when the model cannot produce them)."""
    if not cells and not answer:
        return {}, [], []
    result = await explainer.run(explainer_prompt(request, cells, answer=answer),
                                 usage_limits=UsageLimits(request_limit=3))
    wanted = {c["id"] for c in cells}
    found = {f.id: f.finding.strip() for f in result.output.findings if f.id in wanted and f.finding.strip()}
    conclusions = [c.strip() for c in result.output.conclusions if c.strip()] if answer else []
    issues = [i.strip() for i in result.output.issues if i.strip()] if answer else []
    return found, conclusions[:3], issues[:5]
