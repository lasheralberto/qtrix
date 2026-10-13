"""Estimation sample (ADR 0002 §2): grammar, parsing and range masks.

A sample is workfile state, not a filter each tool reimplements. Its text form follows EViews' ``smpl``::

    @all
    1990q1 2008q4                      one range (dated pages: periods; cross-sections: observation numbers)
    1960q1 1979q4 1990q1 2007q4        several ranges
    @first 2000q4                      @first / @last are the ends of the data
    1990q1 2008q4 if unemp > 5         plus a condition (a DuckDB SQL boolean expression over the columns)
    if age >= 25 and female = 1        condition only

Range bounds mean different things per page kind: periods on dated pages, observation numbers (1-based, in
stored order) on cross-sections and values of the integer time index on undated panels.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import structure as st

ALL, FIRST, LAST = "@all", "@first", "@last"
_IF = re.compile(r"(?:^|\s)if(?=[\s(])", re.I)
_FORBIDDEN = re.compile(r";|--|/\*|\*/")


class SampleError(ValueError):
    """Invalid or empty sample; the message says what to change."""


@dataclass(frozen=True)
class SampleSpec:
    ranges: tuple[tuple[str, str], ...] = ()
    where: str | None = None

    @property
    def is_all(self) -> bool:
        return not self.ranges and not self.where

    @property
    def text(self) -> str:
        if self.is_all:
            return ALL
        parts = [f"{a} {b}" for a, b in self.ranges]
        if self.where:
            parts.append(f"if {self.where}")
        return " ".join(parts)


def parse(text: str | None) -> SampleSpec:
    """Parse the text form. ``None``, empty and ``@all`` mean every observation."""
    raw = " ".join(str(text or "").split())
    if not raw or raw.lower() == ALL:
        return SampleSpec()
    where: str | None = None
    match = _IF.search(raw)
    if match:
        left, where = raw[:match.start()].strip(), raw[match.end():].strip()
        if not where:
            raise SampleError("The condition after 'if' is empty (example: 1990q1 2008q4 if unemp > 5)")
        if _FORBIDDEN.search(where):
            raise SampleError("The condition must be a single boolean expression "
                              "(no ';' or SQL comments), e.g. age >= 25 and female = 1")
    else:
        left = raw
    tokens = [t for t in re.split(r"[\s,]+", left) if t]
    if any(t.lower() == ALL for t in tokens):
        if len(tokens) > 1:
            raise SampleError("'@all' cannot be combined with ranges; write the ranges or just '@all'")
        tokens = []
    if len(tokens) % 2:
        raise SampleError(f"Sample ranges come in pairs (start end), got {len(tokens)} bounds: {tokens}. "
                          "Use @first / @last for the ends of the data")
    ranges = tuple((tokens[i], tokens[i + 1]) for i in range(0, len(tokens), 2))
    return SampleSpec(ranges, where)


def _bound(token: str, keys: pd.Series, kind: str, freq: str):
    t = token.lower()
    if t == FIRST:
        return keys.min()
    if t == LAST:
        return keys.max()
    if kind in st.DATED_KINDS:
        try:
            return st.parse_period(token, freq)
        except st.StructureError as exc:
            raise SampleError(str(exc)) from exc
    try:
        return int(token)
    except ValueError:
        what = "an observation number" if kind == "cross_section" else "a value of the time index"
        raise SampleError(f"'{token}' is not {what}; use a whole number, @first or @last") from None


def range_mask(spec: SampleSpec, keys: pd.Series, *, kind: str, freq: str) -> np.ndarray:
    """Rows inside any of the ranges. ``keys`` are period starts (dated), time-index values (undated panels)
    or observation numbers (cross-sections)."""
    if not spec.ranges:
        return np.ones(len(keys), dtype=bool)
    mask = np.zeros(len(keys), dtype=bool)
    for a, b in spec.ranges:
        lo, hi = _bound(a, keys, kind, freq), _bound(b, keys, kind, freq)
        if lo > hi:
            raise SampleError(f"The range '{a} {b}' starts after it ends")
        mask |= ((keys >= lo) & (keys <= hi)).to_numpy()
    return mask


@dataclass
class ResolvedSample:
    spec: SampleSpec
    mask: np.ndarray | None  # None = every row
    n_selected: int
    n_total: int
    first: str | None
    last: str | None
    text: str

    def to_dict(self) -> dict:
        return {"sample": self.spec.text, "description": self.text, "n_selected": self.n_selected,
                "n_total": self.n_total, "first": self.first, "last": self.last}
