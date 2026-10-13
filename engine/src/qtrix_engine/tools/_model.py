"""Shared preparation for formula-based tools: full-page frame, sample mask and time operators."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field

import pandas as pd

from .. import expr
from ..context import get_context
from ..workspace import DatasetInfo


@dataclass
class Prepared:
    info: DatasetInfo
    frame: pd.DataFrame  # every row of the page plus the columns generated for time-operator terms
    formula: str  # rewritten formula to hand to the estimation library
    names: dict[str, str] = field(default_factory=dict)  # generated column -> term as the user wrote it
    sample: str = ""  # readable description of the sample used
    columns: tuple[str, ...] = ()

    def pretty(self, name: str) -> str:
        return expr.pretty(name, self.names)

    def complete_cases(self) -> pd.DataFrame:
        """Rows with no missing value in any column the formula uses (for libraries that cannot drop them)."""
        used = [c for c in self.frame.columns
                if isinstance(c, str) and re.search(rf"(?<![\w.]){re.escape(c)}(?!\w)", self.formula)]
        return self.frame.dropna(subset=used)


def context_for(info: DatasetInfo, frame: pd.DataFrame, kind: str | None = None) -> expr.Context:
    return expr.Context(kind or info.kind, info.freq, frame.index, tuple(map(str, frame.columns)))


def prepare(dataset_id: str, formula: str, sample: str | None = None, *, notna: Sequence[str] = (),
            frame: pd.DataFrame | None = None, kind: str | None = None) -> Prepared:
    """Evaluate ``formula`` against the dataset.

    Time operators run over the whole page (lags may look outside the sample); the sample is applied by
    blanking the dependent variable outside it. ``notna`` columns (clusters, weights) must also be present
    for a row to count. ``frame``/``kind`` replace the declared structure (explicit entity/time columns).
    """
    ws = get_context().workspace
    info = ws.get(dataset_id)
    res = ws.resolve_sample(info, sample)
    if frame is None:
        frame = ws.frame(dataset_id, sample="@all")
        mask = res.mask
    else:  # the caller already restricted the rows
        mask = None
    for col in notna:
        if col not in frame.columns:
            raise ValueError(f"Column '{col}' not in dataset '{info.name}'")
        present = frame[col].notna().to_numpy()
        mask = present if mask is None else mask & present
    with expr.readable_errors(frame.columns):
        comp = expr.compile_formula(formula, frame, context_for(info, frame, kind), mask=mask)
    return Prepared(info, comp.frame, comp.formula, comp.names, res.text, tuple(map(str, frame.columns)))
