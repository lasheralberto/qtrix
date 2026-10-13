"""Live checks of a cell's calls before they run (E7·08b): unknown variables, time operators the page cannot take.

Nothing here estimates anything: formulas are compiled against the page (time operators evaluated, as estimation
would) and the names left over are looked up among its columns.
"""

from __future__ import annotations

import keyword
import re
from typing import TYPE_CHECKING, Any

from .. import expr
from ..context import get_context
from ..workspace import DatasetNotFound

if TYPE_CHECKING:
    from ..dispatch import Call

# Names a formula may use besides the page's columns: patsy/formulaic helpers and linearmodels' effects.
_FORMULA_NAMES = {"C", "I", "Q", "np", "EntityEffects", "TimeEffects", "FixedEffects", "Treatment", "Sum", "Poly",
                  "center", "standardize", "scale", "bs", "cr", "cc", "te", "poly"}
_OPERATORS = {f.lower() for f in expr.FUNCTIONS}
_IDENT = re.compile(r"(?<![\w.@'\"])([A-Za-z_]\w*)(?!\s*[(.\w])")
_QUOTED = re.compile(r"'[^']*'|\"[^\"]*\"")


def _unknown_names(text: str, known: set[str]) -> list[str]:
    out: list[str] = []
    for m in _IDENT.finditer(_QUOTED.sub("''", text)):
        name = m.group(1)
        if (name in known or name in _FORMULA_NAMES or name.lower() in _OPERATORS or keyword.iskeyword(name)
                or name in out):
            continue
        out.append(name)
    return out


def _issue(param: str, message: str, level: str = "error") -> dict[str, Any]:
    return {"param": param, "level": level, "message": message}


def check_call(call: Call) -> list[dict[str, Any]]:
    """Problems with a call's formula, expression or columns, worded for the user editing the cell."""
    from ..tools import _model

    args = call.args
    page = args.get("dataset_id")
    if not isinstance(page, str):
        return []
    ws = get_context().workspace
    try:
        info = ws.get(page)
    except DatasetNotFound:
        return [_issue("dataset_id", f"No page named '{page}'")]
    columns = {str(c["name"]) for c in info.columns or []}
    issues: list[dict[str, Any]] = []
    for param in ("formula", "expression"):
        text = args.get(param)
        if not isinstance(text, str) or not text.strip():
            continue
        try:
            if param == "formula":
                prepared = _model.prepare(page, text)
                known = columns | set(prepared.names) | {str(c) for c in prepared.frame.columns}
                rewritten = prepared.formula
            else:
                frame = ws.frame(info.id, sample="@all")
                expr.evaluate(text, frame, _model.context_for(info, frame))
                known, rewritten = columns, ""
        except (ValueError, KeyError) as exc:  # ExprError is a ValueError
            issues.append(_issue(param, str(exc)))
            continue
        for name in _unknown_names(rewritten, known):
            issues.append(_issue(param, f"Unknown variable '{name}' in page '{info.name}'"))
    for param in ("column", "columns"):
        value = args.get(param)
        for name in [value] if isinstance(value, str) else value if isinstance(value, list) else []:
            if isinstance(name, str) and name not in columns:
                issues.append(_issue(param, f"Unknown variable '{name}' in page '{info.name}'"))
    return issues
