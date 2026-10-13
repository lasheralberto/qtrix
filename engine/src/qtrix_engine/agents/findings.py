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
"""


class CellFinding(BaseModel):
    id: str = Field(description="The cell id, exactly as given.")
    finding: str


class Findings(BaseModel):
    findings: list[CellFinding] = Field(default_factory=list)


def build_explainer(model: Any) -> Agent[None, Findings]:
    return Agent(model, name="explainer", output_type=Findings, retries=2, instructions=EXPLAINER_INSTRUCTIONS)


def explainer_prompt(request: str, cells: Sequence[dict[str, Any]], *, limit: int = 1200) -> str:
    parts = [f"User request: {request}", ""]
    for c in cells:
        content = str(c.get("content") or "")
        parts += [f"### Cell {c['id']}", c.get("code") or c["name"],
                  "Result:", content if len(content) <= limit else content[:limit] + "…", ""]
    return "\n".join(parts)


async def explain(explainer: Agent[None, Findings], request: str,
                  cells: Sequence[dict[str, Any]]) -> dict[str, str]:
    """Findings by cell id for ``cells`` (empty when the model cannot produce them)."""
    if not cells:
        return {}
    result = await explainer.run(explainer_prompt(request, cells), usage_limits=UsageLimits(request_limit=3))
    wanted = {c["id"] for c in cells}
    return {f.id: f.finding.strip() for f in result.output.findings if f.id in wanted and f.finding.strip()}
